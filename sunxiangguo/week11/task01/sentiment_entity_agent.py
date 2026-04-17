import os

os.environ["OPENAI_API_KEY"] = "sk-be4235589ac240b099ce67bc1af07581"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional
from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)



class SentimentResult(BaseModel):
    """情感分类结果"""
    sentiment: str  # positive, negative, neutral
    confidence: float  # 置信度 0-1
    explanation: str  # 解释原因


class EntityItem(BaseModel):
    """单个实体项"""
    type: str = Field(description="实体类型，如 PERSON、LOCATION、ORGANIZATION 等")
    value: str = Field(description="实体的具体值")


class EntityResult(BaseModel):
    """实体识别结果"""
    entities: List[EntityItem] = Field(description="识别出的实体列表")



# 子agent 1: 情感分类代理
sentiment_agent = Agent(
    name="Sentiment Analysis Agent",
    model="qwen-max",
    handoff_description="专门用于分析文本情感倾向的专家，可以判断文本是积极、消极还是中性。",
    instructions="""你是一个专业的情感分析专家。请分析用户输入文本的情感倾向：
1. 判断情感类别：positive（积极）、negative（消极）或 neutral（中性）
2. 给出置信度评分（0-1之间）
3. 简要解释判断原因

请以JSON格式返回结果，包含 sentiment、confidence 和 explanation 三个字段。""",
    output_type=SentimentResult,
)

# 子agent 2: 实体识别代理
entity_agent = Agent(
    name="Entity Recognition Agent",
    model="qwen-max",
    handoff_description="专门用于从文本中识别和提取命名实体的专家，如人名、地名、机构名等。",
    instructions="""你是一个专业的命名实体识别专家。请从用户输入的文本中识别并提取所有命名实体。

常见的实体类型包括：
- PERSON（人名）
- LOCATION（地点）
- ORGANIZATION（组织机构）
- DATE（日期）
- TIME（时间）
- MONEY（金额）
- PERCENT（百分比）

请以JSON格式返回结果，包含 entities 字段，该字段是一个列表，列表中每个元素包含 type（实体类型）和 value（实体值）。""",
    output_type=EntityResult,
)

# 主agent: 路由代理
router_agent = Agent(
    name="Router Agent",
    model="qwen-max",
    instructions="""你是一个智能路由助手。你的任务是根据用户的请求内容，判断应该将请求分派给哪个专业代理：

- 如果用户想要分析文本的情感倾向（如判断是积极、消极还是中性），请将请求分派给 'Sentiment Analysis Agent'
- 如果用户想要从文本中提取命名实体（如人名、地名、机构名等），请将请求分派给 'Entity Recognition Agent'

请仔细理解用户的需求，选择合适的代理进行处理。""",
    handoffs=[sentiment_agent, entity_agent],
)


async def main():
    print("=" * 60)
    print("启动智能路由Agent系统")
    print("=" * 60)

    # 测试1: 情感分类
    print("\n" + "=" * 60)
    print("测试1: 情感分类")
    print("=" * 60)
    query1 = "这部电影太精彩了，演员表演非常出色，剧情也很吸引人！"
    print(f"\n**用户输入:** {query1}")
    result1 = await Runner.run(router_agent, query1)
    print("\n**处理结果:**")
    print(result1.final_output)

    # 测试2: 实体识别
    print("\n" + "=" * 60)
    print("测试2: 实体识别")
    print("=" * 60)
    query2 = "马云于1999年在杭州创立了阿里巴巴集团，现在该公司在全球拥有超过10万名员工。"
    print(f"\n**用户输入:** {query2}")
    result2 = await Runner.run(router_agent, query2)
    print("\n**处理结果:**")
    print(result2.final_output)

    # 测试3: 另一个情感分类示例
    print("\n" + "=" * 60)
    print("测试3: 情感分类（负面情感）")
    print("=" * 60)
    query3 = "这个产品质量太差了，服务态度也很糟糕，非常失望。"
    print(f"\n**用户输入:** {query3}")
    result3 = await Runner.run(router_agent, query3)
    print("\n**处理结果:**")
    print(result3.final_output)

    # 测试4: 另一个实体识别示例
    print("\n" + "=" * 60)
    print("测试4: 实体识别（新闻文本）")
    print("=" * 60)
    query4 = "苹果公司CEO蒂姆·库克将于2024年3月15日在加州库比蒂诺发布新款iPhone。"
    print(f"\n**用户输入:** {query4}")
    result4 = await Runner.run(router_agent, query4)
    print("\n**处理结果:**")
    print(result4.final_output)

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
