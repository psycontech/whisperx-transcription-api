from fastapi import Depends
from typing import Any, Annotated, Optional, Tuple
from asyncio import get_event_loop
from pyannote.audio import Pipeline  # type: ignore
from settings.config import SettingsDep
from faster_whisper import WhisperModel # type: ignore
from faster_whisper.transcribe import TranscriptionInfo # type: ignore
from app.file.service import FileService
from concurrent.futures import ProcessPoolExecutor
from .schemas.process_audio_schema import ProcessAudioSchema
from .schemas.process_audio_response_schema import ProcessAudioResponseSchema, SpeakerTurn


class WhisperService:
    def __init__(self, settings: SettingsDep, file_service: Annotated[FileService, Depends(FileService)]):
        self.settings = settings
        self.file_service = file_service
        self.process_pool_executor = ProcessPoolExecutor(max_workers=4)

    async def process_audio(self, process_audio_schema: ProcessAudioSchema) -> ProcessAudioResponseSchema:
        file = await self.file_service.upload_file(process_audio_schema.file)

        loop = get_event_loop()

        results, transcription_info = await loop.run_in_executor(
            self.process_pool_executor,
            transcribe_audio,
            file.path,
            self.settings.WHISPER_MODEL_SIZE,
            self.settings.WHISPER_MODEL_DEVICE,
            self.settings.WHISPER_COMPUTE_TYPE,
            self.settings.HF_TOKEN
        )

        speaker_set = set()

        for turn in results:
            print(turn["speaker"])
            speaker_set.add(turn["speaker"])

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
            ]
        )

        return processed_audio_response_schema
    

def transcribe_audio(file_path: str, model_size: str, device: str, compute_type: str, hf_token: str, num_of_speakers: Optional[int] = None) -> Tuple[list[Any], TranscriptionInfo]:

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    whisper_model = WhisperModel(
        model_size, 
        device=device, 
        compute_type=compute_type
    )

    print("Transcribing...")
    segments, info = whisper_model.transcribe(file_path, beam_size=1, word_timestamps=True)


    print("Loading diarization model...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

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


def assign_word_speakers(audio_file_path: str, transcription_segments, diarization_pipeline: Pipeline, num_of_speakers: Optional[int] = None):
    print("Diarizing audio...")
    diarization_kwargs = {}

    if num_of_speakers:
        diarization_kwargs["min_speakers"] = num_of_speakers
        diarization_kwargs["max_speakers"] = num_of_speakers

    diarization = diarization_pipeline(audio_file_path, **diarization_kwargs)
    words_with_speakers = []
    
    for segment in transcription_segments:
        if segment['words'] is None:
            continue
            
        for word in segment['words']:
            assigned_speaker = None
            max_overlap = 0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(word.start, turn.start)
                overlap_end = min(word.end, turn.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = speaker
            
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
