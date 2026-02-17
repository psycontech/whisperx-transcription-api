from pydantic import BaseModel
from fastapi import Request, status
from typing import Union, Any, Optional
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from slowapi.errors import RateLimitExceeded

ErrorDataType = Optional[Union[str, dict[Any, Any], list[Any]]]

class ErrorResponse(BaseModel):
    message: str
    status_code: int
    data: ErrorDataType = None

class BaseHTTPException(HTTPException):
    def __init__(self, message: str, data: ErrorDataType = None):

        error = ErrorResponse(message=message, data=data, status_code=self.status_code)

        super().__init__(status_code=self.status_code, detail=error)


class BadRequestException(HTTPException):
    status_code = status.HTTP_400_BAD_REQUEST
    def __init__(self, message: str, data: ErrorDataType = None):

        error = ErrorResponse(message=message, data=data, status_code=self.status_code)

        super().__init__(status_code=self.status_code, detail=error)

class UnauthorizedException(HTTPException):
    status_code = status.HTTP_401_UNAUTHORIZED
    def __init__(self, message: str, data: ErrorDataType = None):

        error = ErrorResponse(message=message, data=data, status_code=self.status_code)
        super().__init__(status_code=self.status_code, detail=error)


class NotFoundException(HTTPException):
    status_code = status.HTTP_404_NOT_FOUND
    def __init__(self, message: str, data: ErrorDataType = None):
        error = ErrorResponse(message=message, data=data, status_code=self.status_code)
        super().__init__(status_code=self.status_code, detail=error)


class UnsupportedMediaException(BaseHTTPException):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

class ForbiddenException(HTTPException):
    def __init__(self, message: str):

        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=message)


class InternalServerException(HTTPException):
    def __init__(self, message: str):

        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"status": status.HTTP_429_TOO_MANY_REQUESTS, "message": str(exc.detail), "success": False},
    )

