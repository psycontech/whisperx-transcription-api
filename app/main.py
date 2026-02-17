from fastapi import FastAPI
from typing import AsyncGenerator
from settings.config import settings
from contextlib import asynccontextmanager
from app.health.router import router as health_router
from app.whisper.router import router as whisper_router
from app.common.handlers import configure_error_middleware

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Only should be used in development, most preferred to use alembic to track migrations
        # await create_db_and_tables()
        yield
        print("Shutting Down Server")
    finally:
        pass


def register_routers(app: FastAPI) -> None:
    """Register all application routers/controllers"""
    app.include_router(health_router)
    app.include_router(whisper_router)

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="Speech diarization with LLM analysis",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url=settings.API_DOCS.API_DOCS_URL,
        redoc_url=settings.API_DOCS.API_REDOC_URL,
        openapi_url=settings.API_DOCS.OPENAPI_URL,
        swagger_ui_parameters={"persistAuthorization": True},
    )

    configure_error_middleware(app)

    register_routers(app)

    return app
