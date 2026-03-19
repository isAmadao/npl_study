from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json
import os

# 从环境变量获取 API Key，如果没有设置则使用默认值
# 请确保设置 DASHSCOPE_API_KEY 环境变量或在下方填入你的 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-78cc4e9ac8f44efdb207b7232e1ae6d8")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    """信息抽取智能体"""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
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
        except:
            print('ERROR', response.choices[0].message)
            return None


class TranslationInfo(BaseModel):
    """从用户输入中抽取翻译相关信息"""
    source_language: str = Field(description="原始语种（如：英语、日语、法语等）")
    target_language: str = Field(description="目标语种（如：中文、英语、日语等）")
    text_to_translate: str = Field(description="待翻译的文本内容")


class TranslationAgent:
    """文本翻译智能体"""
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.extraction_agent = ExtractionAgent(model_name)

    def translate(self, user_input: str) -> str:
        # 第一步：提取翻译信息
        translation_info = self.extraction_agent.call(user_input, TranslationInfo)
        
        if translation_info is None:
            return "无法解析翻译请求"
        
        print(f"原始语种: {translation_info.source_language}")
        print(f"目标语种: {translation_info.target_language}")
        print(f"待翻译文本: {translation_info.text_to_translate}")
        
        # 第二步：调用大模型进行翻译
        messages = [
            {
                "role": "system",
                "content": f"你是一个专业的翻译助手。请将{translation_info.source_language}翻译成{translation_info.target_language}，只返回翻译结果，不要添加任何解释。"
            },
            {
                "role": "user",
                "content": translation_info.text_to_translate
            }
        ]
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        
        return response.choices[0].message.content


if __name__ == "__main__":
    # 创建翻译智能体
    agent = TranslationAgent()
    
    # 测试用例
    test_inputs = [
        "帮我将good！翻译为中文",
        "请把'Hello, how are you?'翻译成日语",
        "把'今天天气很好'翻译成英文",
        "Translate 'Bonjour' to English",
    ]
    
    for user_input in test_inputs:
        print("=" * 50)
        print(f"用户输入: {user_input}")
        print("-" * 50)
        result = agent.translate(user_input)
        print(f"翻译结果: {result}")
        print()
