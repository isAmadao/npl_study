from pydantic import BaseModel, Field

import openai

client = openai.OpenAI(
    api_key="sk-02e847ab13a543798c4860e15d459293", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


"""
这个智能体（不是满足agent所有的功能），能自动生成tools的json，实现信息信息抽取
指定写的tool的格式
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
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
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
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Translate(BaseModel):
    """识别客户提供的文本将其翻译成目标语言"""
    text: str = Field(description="待翻译的原始文本")
    src_lang: str = Field(description="原始文本的语言，需自动识别")
    tar_lang: str = Field(description="目标语言，用户提供")
result = ExtractionAgent(model_name = "qwen-plus").call('将good！翻译为大韩民国使用的官方语言', Translate)
print(result)
