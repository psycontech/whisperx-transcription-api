from typing import Optional
from pydantic import PositiveInt, BaseModel, HttpUrl, Field

class ProcessAudioSchema(BaseModel):
    num_of_speakers: Optional[PositiveInt] = None
    audio_file_url: HttpUrl
    language: Optional[str] = None
    beam_size: Optional[int] = Field(default=None, nullable=True)
    no_speech_threshold: Optional[float] = Field(default=None, nullable=True)
    initial_prompt: Optional[str] = Field(default=None, nullable=True)
    vad_filter: Optional[bool] = Field(default=None, nullable=True)
    hallucination_silence_threshold: Optional[float] = Field(default=None, nullable=True)