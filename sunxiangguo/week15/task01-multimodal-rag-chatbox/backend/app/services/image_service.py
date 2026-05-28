import os

class ImageService:
    def __init__(self):
        self.local_cache = "cache/images"
        os.makedirs(self.local_cache, exist_ok=True)
    
    async def get_image(self, image_id: str) -> str:
        cache_path = os.path.join(self.local_cache, f"{image_id}.png")
        
        if not os.path.exists(cache_path):
            return None
        
        return cache_path
    
    async def get_thumbnail(self, image_id: str, width: int, height: int) -> str:
        return None
