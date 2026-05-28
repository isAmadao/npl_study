class QwenVLClient:
    async def generate_description(self, image_path: str) -> str:
        return "这是一张图片的描述"
    
    async def get_status(self):
        return {
            "status": "healthy",
            "version": "demo",
            "latency_ms": 10
        }
