import asyncio
from loguru import logger

from app.clients.kafka_client import KafkaClient
from app.services.vector_service import VectorService

class VectorizationWorker:
    def __init__(self):
        self.kafka_client = KafkaClient()
        self.vector_service = VectorService()
    
    async def process_chunk(self, message: dict):
        document_id = message.get("document_id")
        chunk_id = message.get("chunk_id")
        content_type = message.get("content_type")
        content = message.get("content")
        page_number = message.get("page_number")
        image_path = message.get("image_path")
        
        try:
            logger.info(f"Vectorizing chunk: {chunk_id}")
            
            await self.vector_service.process_chunk(
                document_id=document_id,
                content_type=content_type,
                content=content,
                page_number=page_number,
                image_path=image_path
            )
            
            logger.info(f"Chunk vectorized successfully: {chunk_id}")
            
        except Exception as e:
            logger.error(f"Error vectorizing chunk {chunk_id}: {e}")
    
    async def start(self):
        logger.info("Vectorization worker started")
        async for message in self.kafka_client.consume_messages():
            if message.get("type") == "vectorize":
                await self.process_chunk(message)