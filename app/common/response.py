"""HTTP response wrapper."""
from fastapi import status
from pydantic import BaseModel
from typing import TypeVar, Generic

T = TypeVar("T")


class HttpResponse(BaseModel, Generic[T]):
    message: str
    data: T
    status_code: int = status.HTTP_200_OK
