import asyncio
from loguru import logger

from app.clients.kafka_client import KafkaClient
from app.clients.minio_client import MinioClient
from app.utils.pdf_parser import PDFParser
from app.utils.text_utils import TextUtils
from app.services.vector_service import VectorService
from app.repositories.document_repo import DocumentRepo

class DocumentProcessor:
    def __init__(self):
        self.kafka_client = KafkaClient()
        self.minio_client = MinioClient()
        self.vector_service = VectorService()
        self.document_repo = DocumentRepo()
    
    async def process_document(self, message: dict):
        document_id = message.get("document_id")
        file_path = message.get("file_path")
        
        if not document_id or not file_path:
            logger.error("Missing document_id or file_path")
            return
        
        try:
            logger.info(f"Processing document: {document_id}")
            
            await self.document_repo.update(document_id, {"status": "processing"})
            
            file_data = await self.minio_client.download_file(file_path)
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            
            parsed = PDFParser.parse_pdf(temp_path)
            
            text_chunks = TextUtils.split_text(parsed["text"], chunk_size=512)
            for idx, chunk in enumerate(text_chunks):
                await self.vector_service.process_chunk(
                    document_id=document_id,
                    content_type="text",
                    content=chunk,
                    page_number=1
                )
            
            for table in parsed["tables"]:
                await self.vector_service.process_chunk(
                    document_id=document_id,
                    content_type="table",
                    content=table["content"],
                    page_number=table["page_number"]
                )
            
            for image in parsed["images"]:
                await self.vector_service.process_chunk(
                    document_id=document_id,
                    content_type="image",
                    content="",
                    page_number=image["page_number"],
                    image_path=image["path"]
                )
            
            await self.document_repo.update(document_id, {"status": "completed"})
            logger.info(f"Document processed successfully: {document_id}")
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            await self.document_repo.update(document_id, {"status": "failed"})
    
    async def start(self):
        logger.info("Document processor started")
        async for message in self.kafka_client.consume_messages():
            await self.process_document(message)