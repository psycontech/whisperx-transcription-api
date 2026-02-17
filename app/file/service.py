import os
import aiofiles
from uuid import uuid4
from fastapi import UploadFile
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

        