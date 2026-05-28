from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional

from app.services.image_service import ImageService

router = APIRouter()

image_service = ImageService()

@router.get("/{image_id}")
async def get_image(image_id: str):
    file_path = await image_service.get_image(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@router.get("/{image_id}/thumbnail")
async def get_image_thumbnail(image_id: str, width: int = 100, height: int = 100):
    file_path = await image_service.get_thumbnail(image_id, width, height)
    if not file_path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)