import torch
import torchaudio # type: ignore
import threading
from fastapi import Depends
from asyncio import get_event_loop
from pyannote.audio import Pipeline  # type: ignore
from datetime import datetime, timezone
from settings.config import SettingsDep
from faster_whisper import WhisperModel # type: ignore
from app.file.service import FileService
from typing import Any, Annotated, Optional, Tuple
from faster_whisper.transcribe import TranscriptionInfo # type: ignore
from concurrent.futures import ThreadPoolExecutor
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification  # type: ignore
from .schemas.process_audio_schema import ProcessAudioSchema
from .schemas.process_audio_response_schema import ProcessAudioResponseSchema, SpeakerTurn

thread_pool_executor = ThreadPoolExecutor(max_workers=4)

_whisper_model: Optional[WhisperModel] = None
_diarization_pipeline: Optional[Pipeline] = None
_ast_model = None
_ast_feature_extractor = None

whisper_model_lock = threading.Lock()
diarization_pipeline_lock = threading.Lock()
_ast_lock = threading.Lock()

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_CONFIDENCE_THRESHOLD = 0.1
AST_TARGET_EVENTS = {
    "Music", "Laughter", "Cough", "Silence", "Singing", "Speech",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Background music", "Vocal music", "Sneeze", "Breathing", "Crying, sobbing",
}
AST_EMBEDDABLE_EVENTS = {"Cough", "Laughter", "Sneeze", "Breathing", "Crying, sobbing"}

AST_WINDOW_SIZE = 1.5
AST_STEP_SIZE = 0.5
AST_EMBED_THRESHOLD = 0.07

# Whisper sometimes transcribes non-verbal sounds inline as "(laughing)", "(cough)", etc.
# This maps those annotation keywords to our standardized event label strings.
WHISPER_ANNOTATION_MAP = {
    "laughing":  "Laughter",
    "laughter":  "Laughter",
    "laugh":     "Laughter",
    "coughing":  "Cough",
    "cough":     "Cough",
    "sneezing":  "Sneeze",
    "sneeze":    "Sneeze",
    "crying":    "Crying, sobbing",
    "sobbing":   "Crying, sobbing",
    "breathing": "Breathing",
    "sighing":   "Breathing",
    "sigh":      "Breathing",
    "music":     "Music",
    "singing":   "Singing",
    "applause":  "Applause",
}

def get_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    global _whisper_model
    with whisper_model_lock:
        if _whisper_model is None:
            print("Loading Whisper model...")
            _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _whisper_model

def get_ast_model():
    global _ast_model, _ast_feature_extractor
    with _ast_lock:
        if _ast_model is None:
            print("Loading AST audio classification model...")
            _ast_feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_ID)
            _ast_model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_ID)
            _ast_model.eval()
            if torch.cuda.is_available():
                _ast_model = _ast_model.to(torch.device("cuda"))
    return _ast_model, _ast_feature_extractor


def extract_whisper_annotations(words: list) -> list[tuple[float, str]]:
    """
    Scan Whisper word tokens for inline non-verbal annotations like (laughing).

    Whisper marks sounds it hears but can't transcribe as speech using parenthetical
    tokens, e.g. "(laughing)" or "(cough)". These come with word-level timestamps
    so we know exactly when they occurred.

    Returns a list of (timestamp, event_label) tuples sorted by timestamp.
    """
    import re
    detected = []

    for word in words:
        # Whisper annotation tokens look exactly like "(word)" — parentheses wrapping one keyword.
        # We strip surrounding whitespace then check for that pattern.
        match = re.match(r'^\s*\((\w+)\)\s*$', word["word"])
        if match:
            keyword = match.group(1).lower()
            if keyword in WHISPER_ANNOTATION_MAP:
                detected.append((word["start"], WHISPER_ANNOTATION_MAP[keyword]))

    return detected


def classify_audio_segment(audio_file_path: str, start: float, end: float, words: list = []) -> list[str]:
    import numpy as np
    TARGET_SR = 16000

    # --- Source 1: Whisper inline annotations ---
    # Highly reliable — Whisper only adds these when it's confident.
    # We collect just the unique label strings for the audio_events list.
    whisper_labels = {label for _, label in extract_whisper_annotations(words)}

    # --- Source 2: AST model classifying the full turn audio ---
    # Catches events Whisper didn't transcribe (e.g. background music, subtle breathing).
    waveform, sr = safe_load_audio(audio_file_path)
    start_frame = max(0, int(start * sr))
    end_frame = min(waveform.shape[-1], int(end * sr))

    if end_frame <= start_frame:
        # No audio slice to analyse — return whatever Whisper found
        return list(whisper_labels)

    waveform = waveform[:, start_frame:end_frame]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    if len(audio_np) < TARGET_SR * 0.1:
        return list(whisper_labels)

    model, feature_extractor = get_ast_model()
    inputs = feature_extractor(audio_np, sampling_rate=TARGET_SR, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(torch.device("cuda")) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).squeeze(0)
    id2label = model.config.id2label

    ast_labels = {
        id2label[idx]
        for idx, score in enumerate(probs.tolist())
        if score >= AST_CONFIDENCE_THRESHOLD and id2label[idx] in AST_TARGET_EVENTS
    }

    # Merge both sources — union of all detected labels across Whisper and AST
    return list(whisper_labels | ast_labels)


def embed_events_in_text(audio_file_path: str, start: float, end: float, words: list) -> str:
    import re
    import numpy as np
    TARGET_SR = 16000

    if not words:
        return ""

    # --- Source 1: Whisper inline annotations ---
    # Pull events that Whisper already flagged with precise word-level timestamps.
    # These are our most reliable position markers, so we process them first.
    whisper_detections = extract_whisper_annotations(words)

    # --- Source 2: AST sliding window on the raw audio ---
    # Runs a small classification window across the turn to catch events Whisper
    # didn't transcribe (e.g. quiet breathing, background sounds between words).
    waveform, sr = safe_load_audio(audio_file_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    start_frame = max(0, int(start * TARGET_SR))
    end_frame = min(waveform.shape[-1], int(end * TARGET_SR))
    audio_np = waveform.squeeze(0)[start_frame:end_frame].numpy().astype(np.float32)

    ast_detections: list[tuple[float, str]] = []

    if len(audio_np) >= TARGET_SR * 0.1:
        model, feature_extractor = get_ast_model()
        window_samples = int(AST_WINDOW_SIZE * TARGET_SR)
        step_samples = int(AST_STEP_SIZE * TARGET_SR)
        total_samples = len(audio_np)
        window_start = 0

        while window_start < total_samples:
            window_end = min(window_start + window_samples, total_samples)
            window_audio = audio_np[window_start:window_end]

            if len(window_audio) < TARGET_SR * 0.1:
                break

            inputs = feature_extractor(window_audio, sampling_rate=TARGET_SR, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(torch.device("cuda")) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.sigmoid(outputs.logits).squeeze(0)
            id2label = model.config.id2label
            # Absolute timestamp of the centre of this window
            window_mid_abs = start + (window_start + (window_end - window_start) / 2) / TARGET_SR

            for idx, score in enumerate(probs.tolist()):
                label = id2label[idx]
                if score >= AST_EMBED_THRESHOLD and label in AST_EMBEDDABLE_EVENTS:
                    ast_detections.append((window_mid_abs, label))

            window_start += step_samples

    # --- Merge and deduplicate ---
    # Whisper detections go first so their timestamps win when both sources
    # detect the same event within 1 second of each other.
    all_detections = whisper_detections + ast_detections
    deduped: list[tuple[float, str]] = []
    for ts, label in all_detections:
        already_present = any(
            label == prev_label and abs(ts - prev_ts) < 1.0
            for prev_ts, prev_label in deduped
        )
        if not already_present:
            deduped.append((ts, label))

    # Sort by timestamp so we can walk left-to-right through the word stream
    deduped.sort(key=lambda x: x[0])

    # --- Build the output text ---
    # Walk words in order, inserting [Event] markers at the right positions.
    # Whisper annotation tokens like "(laughing)" are SKIPPED — they're replaced
    # by the standardized [Laughter] marker we already inserted from whisper_detections.
    annotation_re = re.compile(r'^\s*\((\w+)\)\s*$')

    event_idx = 0
    tokens = []

    for word in words:
        # Insert any [Event] markers whose timestamp falls before this word starts
        while event_idx < len(deduped) and deduped[event_idx][0] <= word["start"]:
            tokens.append(f" [{deduped[event_idx][1]}]")
            event_idx += 1

        # Drop Whisper annotation tokens — already represented as [Event] markers above
        word_match = annotation_re.match(word["word"])
        if word_match and word_match.group(1).lower() in WHISPER_ANNOTATION_MAP:
            continue

        tokens.append(word["word"])

    # Flush any events that fall after the last word
    while event_idx < len(deduped):
        tokens.append(f" [{deduped[event_idx][1]}]")
        event_idx += 1

    return "".join(tokens).strip()


def get_diarization_pipeline(hf_token: str, clustering_threshold: float = 0.7045, min_duration_off: float = 0.0, min_cluster_size: int = 12) -> Pipeline:
    global _diarization_pipeline
    with diarization_pipeline_lock:
        if _diarization_pipeline is None:
            print("Loading diarization pipeline...")
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            _diarization_pipeline.instantiate({
                "segmentation": {
                    "min_duration_off": min_duration_off,
                },
                "clustering": {
                    "threshold": clustering_threshold,
                    "method": "centroid",
                    "min_cluster_size": min_cluster_size,
                }
            })
            if torch.cuda.is_available():
                _diarization_pipeline.to(torch.device("cuda"))
    return _diarization_pipeline

class WhisperService:
    def __init__(self, settings: SettingsDep, file_service: Annotated[FileService, Depends(FileService)]):
        self.settings = settings
        self.file_service = file_service
        self.thread_pool_executor = thread_pool_executor

    async def process_audio(self, process_audio_schema: ProcessAudioSchema) -> ProcessAudioResponseSchema:
        file = await self.file_service.download_file(str(process_audio_schema.audio_file_url))

        loop = get_event_loop()

        processing_time_start = datetime.now(timezone.utc)

        results, transcription_info = await loop.run_in_executor(
            self.thread_pool_executor,
            transcribe_audio,
            file.path,
            self.settings.WHISPER_MODEL_SIZE,
            self.settings.WHISPER_MODEL_DEVICE,
            self.settings.WHISPER_COMPUTE_TYPE,
            self.settings.HF_TOKEN,
            process_audio_schema.num_of_speakers,
            process_audio_schema.language,
            self.settings.DIARIZATION_CLUSTERING_THRESHOLD,
            self.settings.DIARIZATION_MIN_DURATION_OFF,
            self.settings.DIARIZATION_MIN_CLUSTER_SIZE,
            process_audio_schema.beam_size,
            process_audio_schema.no_speech_threshold,
            process_audio_schema.initial_prompt,
            process_audio_schema.vad_filter,
            process_audio_schema.hallucination_silence_threshold,
            process_audio_schema.classify_events,
        )

        speaker_set = set()

        for turn in results:
            speaker_set.add(turn["speaker"])


        processing_time_end = datetime.now(timezone.utc)

        processed_audio_response_schema = ProcessAudioResponseSchema(
            num_of_speakers=len(speaker_set),
            detected_language=transcription_info.language,
            speaker_set=list(speaker_set),
            language_probability=transcription_info.language_probability,
            audio_duration_seconds=transcription_info.duration,
            turns= [
                SpeakerTurn(
                    start=turn["start"], 
                    end=turn["end"], 
                    text=turn["text"],
                    processed_speaker=int(str(turn["speaker"]).split("_")[1]) + 1,
                    speaker=turn["speaker"],
                    processed_start= round(turn["start"], 2),
                    processed_end=round(turn["end"], 2),
                    audio_events=turn["audio_events"],
                    text_with_events=turn["text_with_events"],
                ) for turn in results
            ],
            processing_time_start=processing_time_start,
            processing_time_end=processing_time_end,
            processing_duration_in_seconds= (processing_time_end - processing_time_start).total_seconds()
        )
    
        await self.file_service.delete_file(file)

        return processed_audio_response_schema
    

def transcribe_audio(file_path: str, model_size: str, device: str, compute_type: str, hf_token: str, num_of_speakers: Optional[int] = None, language: Optional[str] = None, clustering_threshold: float = 0.7045, min_duration_off: float = 0.0, min_cluster_size: int = 12, beam_size: Optional[int] = None, no_speech_threshold: Optional[float] = None, initial_prompt: Optional[str] = None, vad_filter: Optional[bool] = None, hallucination_silence_threshold: Optional[float] = None, classify_events: Optional[bool] = False) -> Tuple[list[Any], TranscriptionInfo]:

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


    whisper_model = get_whisper_model(model_size, device, compute_type)

    _beam_size = beam_size if beam_size is not None else 3
    _no_speech_threshold = no_speech_threshold if no_speech_threshold is not None else 0.3
    _initial_prompt = initial_prompt
    _vad_filter = vad_filter if vad_filter is not None else False
    _hallucination_silence_threshold = hallucination_silence_threshold

    print("=" * 50)
    print("TRANSCRIPTION CONFIG")
    print(f"  beam_size:                    {_beam_size}")
    print(f"  model_size:                   {model_size}")
    print(f"  compute_type:                 {compute_type}")
    print(f"  device:                       {device}")
    print(f"  language:                     {language or 'auto-detect'}")
    print(f"  num_of_speakers:              {num_of_speakers or 'auto'}")
    print(f"  no_speech_threshold:          {_no_speech_threshold}")
    print(f"  initial_prompt:               {_initial_prompt or 'none'}")
    print(f"  vad_filter:                   {_vad_filter}")
    print(f"  hallucination_silence_threshold: {_hallucination_silence_threshold or 'none'}")
    print(f"  clustering_threshold:         {clustering_threshold}")
    print(f"  min_duration_off:             {min_duration_off}")
    print(f"  min_cluster_size:             {min_cluster_size}")
    print(f"  classify_events:              {classify_events or False}")
    print("=" * 50)

    print("Transcribing...")
    segments, info = whisper_model.transcribe(
        file_path,
        beam_size=_beam_size,
        word_timestamps=True,
        language=language,
        no_speech_threshold=_no_speech_threshold,
        initial_prompt=_initial_prompt,
        vad_filter=_vad_filter,
        hallucination_silence_threshold=_hallucination_silence_threshold,
        suppress_tokens=[],
        condition_on_previous_text=False,
    )

    print("Loading diarization model...")


    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")
    else:
        print("CUDA NOT AVAILABLE, USING CPU for Diarization")
   
    diarization_pipeline = get_diarization_pipeline(hf_token, clustering_threshold, min_duration_off, min_cluster_size)


    result_segments = []
    for segment in segments:
        result_segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'words': segment.words
        })

    words_with_speakers = assign_word_speakers(file_path, result_segments, diarization_pipeline, num_of_speakers)
    speaker_turns = group_by_speaker_turns(words_with_speakers)

    if classify_events:
        print(f"Classifying audio events for {len(speaker_turns)} turns...")
        for turn in speaker_turns:
            turn_words = [w for w in words_with_speakers if turn["start"] <= w["start"] <= turn["end"]]
            # Pass words so both classify_ and embed_ can mine Whisper annotations
            turn["audio_events"] = classify_audio_segment(file_path, turn["start"], turn["end"], turn_words)
            turn["text_with_events"] = embed_events_in_text(file_path, turn["start"], turn["end"], turn_words)
    else:
        for turn in speaker_turns:
            turn["audio_events"] = []
            turn["text_with_events"] = None

    return speaker_turns, info


def safe_load_audio(audio_file_path: str):
    import os
    tmp_wav_path = None
    try:
        return torchaudio.load(str(audio_file_path))
    except RuntimeError:
        from pydub import AudioSegment
        tmp_wav_path = str(audio_file_path) + "_converted.wav"
        AudioSegment.from_file(audio_file_path).export(tmp_wav_path, format="wav")
        waveform, sample_rate = torchaudio.load(tmp_wav_path)
        return waveform, sample_rate
    finally:
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


def pad_audio(audio_file_path: str) -> dict:
    waveform, sample_rate = safe_load_audio(audio_file_path)

    chunk_size = 160000
    remainder = waveform.shape[-1] % chunk_size
    if remainder != 0:
        pad_size = chunk_size - remainder
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    return {"waveform": waveform, "sample_rate": sample_rate}


def assign_word_speakers(audio_file_path: str, transcription_segments, diarization_pipeline: Pipeline, num_of_speakers: Optional[int] = None):
    print("Diarizing audio...")
    diarization_kwargs = {}

    if num_of_speakers:
        diarization_kwargs["min_speakers"] = num_of_speakers
        diarization_kwargs["max_speakers"] = num_of_speakers

    
    # Pad audio file with empty audio after chunk split
    audio_input = pad_audio(audio_file_path)

    diarization = diarization_pipeline(audio_input, **diarization_kwargs)
    words_with_speakers = []
    
    for segment in transcription_segments:
        if segment['words'] is None:
            continue
            
        for word in segment['words']:
            assigned_speaker = None
            max_overlap = 0
            
            diarization_tracks = list(diarization.itertracks(yield_label=True))

            for turn, _, speaker in diarization_tracks:
                overlap_start = max(word.start, turn.start)
                overlap_end = min(word.end, turn.end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = speaker

            if assigned_speaker is None:
                word_mid = (word.start + word.end) / 2
                min_distance = float('inf')
                for turn, _, speaker in diarization_tracks:
                    turn_mid = (turn.start + turn.end) / 2
                    distance = abs(word_mid - turn_mid)
                    if distance < min_distance:
                        min_distance = distance
                        assigned_speaker = speaker

            if assigned_speaker is None:
                print(f"WARNING: No speaker found for word '{word.word}' at {word.start:.2f}s-{word.end:.2f}s")

            words_with_speakers.append({
                'word': word.word,
                'start': word.start,
                'end': word.end,
                'speaker': assigned_speaker
            })
    
    return words_with_speakers


def group_by_speaker_turns(words_with_speakers: list[Any]):
    if not words_with_speakers:
        return []
    
    valid_words = [w for w in words_with_speakers if w['speaker'] is not None]
    dropped = len(words_with_speakers) - len(valid_words)
    if dropped > 0:
        print(f"WARNING: {dropped} word(s) dropped due to no speaker assignment")

    if not valid_words:
        return []
    
    turns = []
    current_speaker = valid_words[0]['speaker']
    current_words = [valid_words[0]['word']]
    current_start = valid_words[0]['start']
    current_end = valid_words[0]['end']
    
    for word_info in valid_words[1:]:
        if word_info['speaker'] == current_speaker:
            current_words.append(word_info['word'])
            current_end = word_info['end']
        else:
            turns.append({
                'speaker': current_speaker,
                'start': current_start,
                'end': current_end,
                'text': ''.join(current_words).strip()
            })
            current_speaker = word_info['speaker']
            current_words = [word_info['word']]
            current_start = word_info['start']
            current_end = word_info['end']
    
    turns.append({
        'speaker': current_speaker,
        'start': current_start,
        'end': current_end,
        'text': ''.join(current_words).strip()
    })
    
    return turns
