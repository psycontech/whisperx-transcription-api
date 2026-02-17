from typing import List, Any
from pydantic import BaseModel, PositiveInt, PositiveFloat

class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    processed_start: float
    processed_end: float
    processed_speaker: int

class ProcessAudioResponseSchema(BaseModel):
    num_of_speakers: int
    detected_language: str
    speaker_set: list[str]
    language_probability: PositiveFloat
    audio_duration_seconds: PositiveFloat
    turns: List[SpeakerTurn]
