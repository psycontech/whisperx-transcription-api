from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException
from app.common.exceptions import ErrorResponse

def configure_error_middleware(app: FastAPI) -> None:

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        content = None

        if isinstance(exc.detail, ErrorResponse):
            content = exc.detail.model_dump()

        if isinstance(exc.detail, str):
            content = {"message": exc.detail, "data": None}

        return JSONResponse(content=content, status_code=exc.status_code)
        

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Use Sentry to capture exception
        return JSONResponse(content={"message": "Internal Server Error", "data": None}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)                               