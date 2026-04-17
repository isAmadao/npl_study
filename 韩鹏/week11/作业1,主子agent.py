import os

os.environ["OPENAI_API_KEY"] = "sk-9dac9bc999f246eca490105ea0fd5a30"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 情感分类智能体
sentiment_classification_agent = Agent(
    name="sentiment_classification_agent",
    model="qwen3.5-plus",
    handoff_description="一个情感分类助手",
    instructions="你是一个情感分类助手。对用户提供的文本进行感情分析，要求输出文本的情感倾向，且只能从以下三种情感倾向中选择：积极、消极、中立，不要输出其他任何信息。",
)

# 实体识别智能体
entity_recognition_agent = Agent(
    name="entity_recognition_agent",
    model="qwen3.5-plus",
    handoff_description="一个实体识别助手",
    instructions="""
        你是一个实体识别助手。对于用户输入的文本,你需要从中提取实体类别，并按照以下要求输出：
        1. 实体类别包括：
           - "PERSON"   ：人名
           - "LOC"      ：地名（国家、城市、山川、河流等）
           - "ORG"      ：组织机构名（公司、政府、学校、团体等）
           - "DATE"     ：日期或时间段
           - "TIME"     ：具体时间点或时长
           - "OTH"      ：不属于以上类别的其他专有名词
        2.输出格式为Json数组，每个元素包含实体名称、实体类别。
        示例输入：
        "马云于1999年在杭州创立了阿里巴巴集团。"
        示例输出：
        [
          {"text": "马云", "type": "PERSON"},
          {"text": "1999年", "type": "DATE"},
          {"text": "杭州", "type": "LOC"},
          {"text": "阿里巴巴集团", "type": "ORG"}
        ]
    """,
)

# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen3.5-plus",
    instructions="""你是一个路由助手。根据用户的请求内容，判断用户是需要进行情感分类还是实体识别，然后将请求分配给相应的智能体来处理。
                 如果不符合以上两种需求，请直接回答用户：无法回答。""",
    handoffs=[sentiment_classification_agent, entity_recognition_agent],
)

async def main():
    print("你好，我可以帮你做情感分类或实体识别。输入 'exit' 退出。")
    while True:
        user_msg = input("\n请输入: ")
        if user_msg.lower() == "exit":
            break
        agent = triage_agent
        inputs: list[TResponseInputItem] = [{"content": user_msg, "role": "user"}]
        with trace("Routing example"):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent): # 只处理原始响应事件，其他事件类型（如工具调用事件）暂时忽略
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):    # 每当有新的文本增量时，打印增量文本
                    print(data.delta, end="", flush=True)   # 打印增量文本，不换行，刷新输出缓冲区
                elif isinstance(data, ResponseContentPartDoneEvent):    # 每当一个内容部分完成时，打印换行符
                    print("\n")
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())