import os
from PIL import Image

class ImageProcessor:
    @staticmethod
    def resize_image(input_path: str, output_path: str, max_size: int = 1024):
        with Image.open(input_path) as img:
            width, height = img.size
            
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            img.save(output_path)
    
    @staticmethod
    def convert_to_png(input_path: str, output_path: str):
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img.save(output_path, "PNG")
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }