from typing import Optional
from fastapi import UploadFile, Form
from dataclasses import dataclass
from pydantic import PositiveInt

@dataclass
class ProcessAudioSchema:
    file: UploadFile
    num_of_speakers: Optional[PositiveInt] = Form(None)