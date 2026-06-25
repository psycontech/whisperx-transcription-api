from typing import Optional
from pydantic import PositiveInt, BaseModel, HttpUrl

class ProcessAudioSchema(BaseModel):
    num_of_speakers: Optional[PositiveInt] = None
    audio_file_url: HttpUrl
    language: Optional[str] = None
    beam_size: Optional[int] = None
    no_speech_threshold: Optional[float] = None
    initial_prompt: Optional[str] = None
    vad_filter: Optional[bool] = None
    hallucination_silence_threshold: Optional[float] = None
    classify_events: Optional[bool] = False
    clustering_threshold: Optional[float] = None
    min_duration_off: Optional[float] = None
    min_cluster_size: Optional[int] = None