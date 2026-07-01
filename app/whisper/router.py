from typing import Annotated
from app.common.router import VersionRouter
from app.common.response import HttpResponse
from app.whisper.service import WhisperService
from fastapi import status as HttpStatus, Depends
from .schemas.process_audio_schema import ProcessAudioSchema
from .schemas.process_audio_response_schema import ProcessAudioResponseSchema
from .schemas.thread_pool_status_schema import ThreadPoolStatusSchema


router = VersionRouter(version="1", path="whisper", tags=["Whisper"])

@router.post("/", status_code=HttpStatus.HTTP_200_OK, response_model=HttpResponse[ProcessAudioResponseSchema])
async def process_audio(
    process_audio_schema: ProcessAudioSchema,
    whisper_service: Annotated[WhisperService, Depends(WhisperService)],
) -> HttpResponse[ProcessAudioResponseSchema]:

    results = await whisper_service.process_audio(process_audio_schema)

    return HttpResponse(message="Processed Audio Successfully", data=results, status_code=HttpStatus.HTTP_200_OK)


@router.get("/thread-pool-status", status_code=HttpStatus.HTTP_200_OK, response_model=HttpResponse[ThreadPoolStatusSchema])
async def get_thread_pool_status(
    whisper_service: Annotated[WhisperService, Depends(WhisperService)],
) -> HttpResponse[ThreadPoolStatusSchema]:

    status = whisper_service.get_thread_pool_status()

    return HttpResponse(message="Thread Pool Status Fetched Successfully", data=status, status_code=HttpStatus.HTTP_200_OK)
