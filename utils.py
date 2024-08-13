import os

from fastapi import UploadFile

def load_image(path: str):
    with open(path, "rb") as buffer:
        return buffer.read()

async def save_image(path: str, file: UploadFile):
    if hasattr(file, 'read'):
        await file.seek(0)  # Move to the beginning of the file-like object
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as buffer:
            buffer.write(await file.read())
    else:
        raise TypeError("Expected a file-like object or bytes.")
