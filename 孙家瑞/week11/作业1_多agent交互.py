import os

from agents.extensions.visualization import draw_graph

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

sentiment_cls = Agent(
    name="sentiment_cls",
    model="qwen3.5-plus",
    instructions="""你是小王，擅长对用户的情感进行分类，目前主要类别如下：
积极：Positive
消极：Negative
中性：Neutral
愤怒：Anger
喜悦：Joy
悲伤：Sadness
恐惧：Fear
惊讶：Surprise
厌恶：Disgust

请在以上列举的范围内进行回答
如果不知道，那么就返回给 triage_agent
    """,
)

entity_recognition = Agent(
    name="entity_recognition",
    model="qwen3.5-plus",
    instructions="你擅长对用户输入文本进行实体识别，需要对用户输入的语句提取<5个主要的实体\n如果不知道，那么就返回给 triage_agent",
)

# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen3.5-plus",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[sentiment_cls, entity_recognition],
)

# 设置返回
sentiment_cls.handoffs=[triage_agent]
entity_recognition.handoffs=[triage_agent]


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    try:
        draw_graph(triage_agent, filename="路由Handoffs")
    except Exception as e:
        print("绘制agent失败，默认跳过。。。")
        print(e)

    msg = input("你好，我可以对你的输入进行情感分析和文本识别，你需要什么功能呢？\n")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())