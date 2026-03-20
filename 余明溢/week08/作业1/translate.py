from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-abuabub209d34b1a89932c3ced430028", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 本质是function call
# 可以传入多个待选函数，让大模型选择其中一个
# 传的是我们的函数的描述，让大模型选择，生成调用这个函数的传入参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "根据用户提供的信息翻译成目标语种",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "description": "要翻译的内容",
                        "title": "Content",
                        "type": "string",
                    },
                    "source": {
                        "description": "原语种",
                        "title": "Source",
                        "type": "string",
                    },
                    "target": {
                        "description": "目标语种",
                        "title": "Target",
                        "type": "string",
                    },
                },
                "required": ["content", "source", "target"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "你能帮我把good翻译成中文吗？"
    }
]

# 大模型选择了一个函数，生成了函数的调用过程， 这也是agent 的核心功能
response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools, # 生成函数的调用方式，并不是所有的模型都支持（某些比较小的模型不支持）
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls[0].function)
