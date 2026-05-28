from typing import List

class ClipClient:
    async def encode_text(self, text: str) -> List[float]:
        return [0.0] * 100
    
    async def encode_image(self, image_path: str) -> List[float]:
        return [0.0] * 100
