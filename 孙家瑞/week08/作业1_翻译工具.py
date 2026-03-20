import os
from typing import List

import openai
from pydantic import BaseModel, Field  # 定义传入的数据请求格式
from typing_extensions import Literal

client = openai.OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", "该环境变量不存在"),
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "Ticket",
            "description": "根据用户提供的信息查询火车时刻",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "description": "要查询的火车日期",
                        "title": "Date",
                        "type": "string",
                    },
                    "departure": {
                        "description": "出发城市或车站",
                        "title": "Departure",
                        "type": "string",
                    },
                    "destination": {
                        "description": "要查询的火车日期",
                        "title": "Destination",
                        "type": "string",
                    },
                },
                "required": ["date", "departure", "destination"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "你能帮我查一下2024年1月1日从北京南站到上海的火车票吗？"
    }
]

response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,  # 生成函数的调用方式，并不是所有的模型都支持（某些比较小的模型不支持）
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls[0].function)

"""
这个智能体（不是满足agent的功能），能自动生成tools的json，实现信息信息抽取
"""


class ExtractionAgent:
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
                    "name": response_model.model_json_schema()['title'],  # 工具名字
                    "description": response_model.model_json_schema()['description'],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],  # 参数说明
                        "required": response_model.model_json_schema()['required'],  # 必须要传的参数
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
            function = response.choices[0].message.tool_calls[0].function
            arguments = function.arguments
            print("tool_calls is :\n",{
                    "function": function,
                    "arguments": arguments
                })
            return response_model.model_validate_json(arguments
                )
        except:
            print('ERROR', response.choices[0].message)
            return None


class Translate(BaseModel):
    """根据用户输入进行翻译"""
    input: str = Field(description="需要翻译的文本")
    inputType: str = Field(description="需要翻译文本的语言类型")
    outputType: str = Field(description="目标语言类型")


result = ExtractionAgent(model_name="qwen-plus").call("今天天气很好，月亮很美翻译为日语", Translate)
print(result)
