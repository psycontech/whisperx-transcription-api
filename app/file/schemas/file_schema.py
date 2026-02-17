from typing import Optional
from pydantic import BaseModel


class File(BaseModel):
    name: Optional[str] = None

    path: str

    content_type: Optional[str] = None

    size: Optional[int] = None