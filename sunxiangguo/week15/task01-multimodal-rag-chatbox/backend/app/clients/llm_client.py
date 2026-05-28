import requests
import json
from loguru import logger

from app.core.config import settings

class LLMClient:
    def __init__(self):
        self.api_key = settings.llm_api_key
        self.api_base = settings.llm_api_base
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
        
    async def generate_answer(self, prompt: str) -> str:
        if not self.api_key or not self.api_base:
            return "LLM配置未完成，请在config.yml中配置API密钥和地址"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的多模态知识问答助手，请根据用户提供的参考资料回答问题。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        try:
            response = requests.post(self.api_base, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"LLM Response: {json.dumps(result, ensure_ascii=False)[:200]}...")
            
            if "output" in result:
                if "text" in result["output"]:
                    return result["output"]["text"]
                elif "choices" in result["output"] and len(result["output"]["choices"]) > 0:
                    if "message" in result["output"]["choices"][0]:
                        return result["output"]["choices"][0]["message"]["content"]
                    elif "text" in result["output"]["choices"][0]:
                        return result["output"]["choices"][0]["text"]
                else:
                    return str(result["output"])
            else:
                error_msg = result.get("message", result.get("error", "Unknown error"))
                logger.error(f"LLM API Error: {error_msg}")
                return f"LLM调用失败: {error_msg}"
                
        except Exception as e:
            logger.error(f"LLM API Request Error: {str(e)}")
            return f"LLM调用异常: {str(e)}"
    
    async def get_status(self):
        return {
            "status": "healthy" if self.api_key else "unconfigured",
            "version": self.model,
            "latency_ms": 0
        }
