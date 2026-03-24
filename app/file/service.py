import os
import aiofiles
from uuid import uuid4
from fastapi import UploadFile
import httpx
from .schemas.file_schema import File
from settings.config import SettingsDep

class FileService:
    def __init__(self, settings: SettingsDep):
        self.settings = settings

    async def upload_file(self, file: UploadFile) -> File:

        key = str(uuid4())

        if not os.path.exists(self.settings.UPLOAD_DIR):
            os.mkdir(self.settings.UPLOAD_DIR)

        new_file_path = self.settings.UPLOAD_DIR / key

        async with aiofiles.open(new_file_path, mode="wb") as buffer:
            # read file to memory 1MB at a time
            while chunk := await file.read(1024 * 1024):  
                await buffer.write(chunk)
        
        new_file = File(name=file.filename, path=str(new_file_path), content_type=file.content_type, size=file.size)
        
        return new_file

    async def delete_file(self, file: File) -> None:        
        file_path = file.path
        
        # Delete physical file after database operation
        if os.path.exists(file_path):
            os.remove(file_path)

    _AUDIO_MIME_EXTENSIONS = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/x-flac": ".flac",
        "audio/mp4": ".mp4",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/webm": ".webm",
        "audio/aac": ".aac",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
    }

    async def download_file(self, url: str) -> File:
        from urllib.parse import urlparse

        key = str(uuid4())

        if not os.path.exists(self.settings.UPLOAD_DIR):
            os.mkdir(self.settings.UPLOAD_DIR)

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                content_type = response.headers.get("content-type", "application/octet-stream")
                content_length = response.headers.get("content-length")

                # Try URL extension first, then content-type map, then Content-Disposition
                url_path = urlparse(url).path
                ext = os.path.splitext(url_path)[1].lower()

                if not ext or ext == ".bin":
                    mime = content_type.split(";")[0].strip().lower()
                    ext = self._AUDIO_MIME_EXTENSIONS.get(mime, "")

                if not ext or ext == ".bin":
                    disposition = response.headers.get("content-disposition", "")
                    if "filename=" in disposition:
                        fname = disposition.split("filename=")[-1].strip().strip('"')
                        ext = os.path.splitext(fname)[1].lower()

                if not ext or ext == ".bin":
                    print(f"WARNING: Could not determine audio format for content-type '{content_type}', defaulting to .mp3")
                    ext = ".mp3"

                new_file_path = self.settings.UPLOAD_DIR / f"{key}{ext}"

                with open(new_file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

        new_file = File(name=f"{key}{ext}", path=str(new_file_path), content_type=content_type, size=content_length)

        return new_file