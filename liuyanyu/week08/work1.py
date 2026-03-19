from pydantic import BaseModel, Field
from typing import Optional
import openai

# 初始化 OpenAI 客户端（请替换为有效的 API Key 和 base_url）
client = openai.OpenAI(
    api_key="your-api-key",  # 替换为实际密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [{"role": "user", "content": user_prompt}]
        # 根据 response_model 动态构建 tools 格式
        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get("title", response_model.__name__),
                    "description": schema.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema.get("required", []),
                    },
                }
            }
        ]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception:
            print("ERROR:", response.choices[0].message)
            return None

# 定义翻译任务的 Pydantic 模型
class TranslationRequest(BaseModel):
    """翻译任务参数抽取"""
    source_language: str = Field(description="原始语种，例如：英文、中文、法语等")
    target_language: str = Field(description="目标语种，例如：中文、英文、日语等")
    text: str = Field(description="需要翻译的文本内容")

# 测试翻译智能体
if __name__ == "__main__":
    agent = ExtractionAgent(model_name="qwen-plus")
    user_input = "帮我将good！翻译为中文"
    result = agent.call(user_input, TranslationRequest)
    print("提取结果：", result)
