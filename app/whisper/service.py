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
from .schemas.process_audio_schema import ProcessAudioSchema
from .schemas.process_audio_response_schema import ProcessAudioResponseSchema, SpeakerTurn

thread_pool_executor = ThreadPoolExecutor(max_workers=4)

_whisper_model: Optional[WhisperModel] = None
_diarization_pipeline: Optional[Pipeline] = None

whisper_model_lock = threading.Lock()
diarization_pipeline_lock = threading.Lock()

def get_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    global _whisper_model
    with whisper_model_lock:
        if _whisper_model is None:
            print("Loading Whisper model...")
            _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _whisper_model

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
                    processed_end=round(turn["end"], 2)
                ) for turn in results
            ],
            processing_time_start=processing_time_start,
            processing_time_end=processing_time_end,
            processing_duration_in_seconds= (processing_time_end - processing_time_start).total_seconds()
        )
    
        await self.file_service.delete_file(file)

        return processed_audio_response_schema
    

def transcribe_audio(file_path: str, model_size: str, device: str, compute_type: str, hf_token: str, num_of_speakers: Optional[int] = None, language: Optional[str] = None, clustering_threshold: float = 0.7045, min_duration_off: float = 0.0, min_cluster_size: int = 12, beam_size: Optional[int] = None, no_speech_threshold: Optional[float] = None, initial_prompt: Optional[str] = None, vad_filter: Optional[bool] = None, hallucination_silence_threshold: Optional[float] = None) -> Tuple[list[Any], TranscriptionInfo]:

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

    return speaker_turns, info


def pad_audio(audio_file_path: str) -> dict:
    waveform, sample_rate = torchaudio.load(str(audio_file_path))

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
