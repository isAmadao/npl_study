import os
from typing import List, Dict

class PDFParser:
    @staticmethod
    def parse_pdf(file_path: str) -> Dict:
        try:
            from mineru import extract_text, extract_images, extract_tables
            
            result = {
                "text": "",
                "tables": [],
                "images": []
            }
            
            result["text"] = extract_text(file_path)
            
            tables = extract_tables(file_path)
            for i, table in enumerate(tables):
                result["tables"].append({
                    "id": f"table-{i}",
                    "content": str(table),
                    "page_number": 1
                })
            
            image_dir = f"tmp/images/{os.path.basename(file_path).replace('.pdf', '')}"
            os.makedirs(image_dir, exist_ok=True)
            
            images = extract_images(file_path, output_dir=image_dir)
            for i, image_path in enumerate(images):
                result["images"].append({
                    "id": f"image-{i}",
                    "path": image_path,
                    "page_number": 1
                })
            
            return result
        except ImportError:
            return {
                "text": "MinerU not available, using fallback parser",
                "tables": [],
                "images": []
            }
        except Exception as e:
            return {"text": "", "tables": [], "images": [], "error": str(e)}