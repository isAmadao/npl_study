import re
from typing import List

class TextUtils:
    @staticmethod
    def split_text(text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."