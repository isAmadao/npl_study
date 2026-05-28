from loguru import logger
import sys
import os

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    
    logger.remove()
    
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days"
    )