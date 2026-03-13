### 作业2: 基于 02-joint-bert-training-only  中的数据集，希望你自己写一个提示词能完成任务（信息解析的智能对话系统）

import openai  # 导入 OpenAI 官方的 Python SDK
import json  # 导入 Python 内置的 JSON 处理模块，用于解析 / 生成 JSON 格式数据
# Field：用来给模型的字段添加描述、默认值等元信息
from pydantic import BaseModel, Field  # 定义传入的数据请求格式
# 定义数据类型的辅助工具，告诉程序 “某个字段是列表 / 可选值”
from typing import List, Optional, Dict
# Literal：表示字段只能取指定的固定值
from typing_extensions import Literal

# 配置阿里云百炼 API
client = openai.OpenAI(
    api_key="sk-fe0209453f0d48179de8bd53a6ce028c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 创建一个 “信息抽取机器人” 的模板
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    # user_prompt：用户输入的文本
    # response_model：指定抽取结果的格式
    def call(self, user_prompt, response_model):
        # 构建精准的提示词，引导模型按规则抽取信息
        prompt = f"""请严格按照指定格式抽取以下文本的领域、意图和槽位信息：
用户输入文本：{user_prompt}
要求：
1. 领域（domain）从 ['music', 'app', 'weather', 'bus', 'train', 'cinemas', 'cookbook', 'flight'] 中选择；
2. 意图（intent）从 ['OPEN', 'SEARCH', 'QUERY', 'PLAY', 'SEND'] 中选择；
3. 槽位（slots）为字典格式，键为实体类型（如Src/Des/name/dishName等），值为实体值，无槽位则返回空字典。
"""
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        # 告诉大模型 “请按照我指定的字段和格式返回结果”
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],  # 工具名字
                    "description": response_model.model_json_schema()['description'],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties']  # 参数说明
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
        # 解析大模型返回的结构化结果
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print('ERROR', response.choices[0].message)
            print('Exception:', e)
            return None


class IntentDomainSlotTask(BaseModel):
    """对文本抽取领域类别、意图类型、槽位实体标签"""
    domain: Literal['music', 'app', 'weather', 'bus', 'train', 'cinemas', 'cookbook', 'flight'] = Field(
        description="领域")
    intent: Literal['OPEN', 'SEARCH', 'QUERY', 'PLAY', 'SEND'] = Field(description="意图")
    slots: Dict[str, str] = Field(default_factory=dict, description="槽位信息，键为实体类型，值为实体原始值")


# 抽取函数封装
def extract_intent_domain_slots(text: str) -> dict:
    """抽取文本中的领域、意图和槽位信息"""
    agent = ExtractionAgent(model_name="qwen-plus")
    result = agent.call(text, IntentDomainSlotTask)
    if result:
        return result.model_dump()
    return None


# 测试函数（精简为5个测试用例）
def test_extraction():
    """测试信息抽取功能"""
    test_cases = [
        "查询武汉到香港的动车",
        "打开微信",
        "红烧肉怎么做",
        "明天去上海的航班",
        "播放周杰伦的稻香"
    ]

    print("=" * 60)
    print("信息抽取测试结果".center(50))
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        print(f"\n【测试 {i}/{len(test_cases)}】")
        print(f"输入：{text}")
        print("-" * 60)

        result = extract_intent_domain_slots(text)

        if result:
            print(f"领域 (domain): {result['domain']}")
            print(f"意图 (intent): {result['intent']}")
            print(f"槽位 (slots): {json.dumps(result['slots'], ensure_ascii=False, indent=2)}")
        else:
            print("抽取失败")

    print("-" * 60)

if __name__ == "__main__":
    # 运行测试
    test_extraction()
