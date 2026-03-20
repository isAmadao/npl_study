from pydantic import BaseModel, Field
from typing import Optional
import openai

# 初始化客户端（请替换为你的有效 API Key 和 Base URL）
client = openai.OpenAI(
    api_key="sk-f0abuabu58044adcb75b5a60974549b3",  # 替换为实际密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class ExtractionAgent:
    """复用之前的信息抽取智能体"""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [{"role": "user", "content": user_prompt}]
        # 将 Pydantic 模型转换为工具描述
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
            print('抽取失败，原始响应：', response.choices[0].message)
            return None

# 定义翻译任务所需的参数结构
class TranslationParams(BaseModel):
    """翻译任务参数抽取"""
    source_lang: str = Field(description="原始语种，例如'英文'、'中文'、'法语'等，如果未明确给出可根据文本内容推断")
    target_lang: str = Field(description="目标语种，例如'中文'、'英文'、'日语'等")
    text: str = Field(description="待翻译的文本内容")

# 创建翻译智能体实例
agent = ExtractionAgent(model_name="qwen-plus")  # 也可换用其他支持 function calling 的模型

# 测试几个不同风格的翻译请求
test_queries = [
    "帮我将good！翻译为中文",
    "把hello world翻译成法语",
    "请问'今天天气真好'用英语怎么说？",
    "翻译 'I love programming' 到简体中文",
    "将这段文字译成日文：明日は晴れるでしょう",
]

for query in test_queries:
    print(f"\n用户问题：{query}")
    result = agent.call(query, TranslationParams)
    if result:
        print(f"抽取结果：原始语种={result.source_lang}, 目标语种={result.target_lang}, 文本={result.text}")
    else:
        print("抽取失败")
