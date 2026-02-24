from typing import Optional
from pydantic import PositiveInt, BaseModel, HttpUrl

class ProcessAudioSchema(BaseModel):
    num_of_speakers: Optional[PositiveInt] = None
    audio_file_url: HttpUrl