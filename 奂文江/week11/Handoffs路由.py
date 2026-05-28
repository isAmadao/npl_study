import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-069ef3db280142b8a8f783d5e451d6b3"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 意图识别 -》 路由
# 用户提问 -》 类型1  类型2  类型3

math_agent = Agent(
    name="math_agent",
    model="qwen3.5-flash",
    instructions="你是小王，擅长数学计算，回答问题的时候先告诉我你是谁。",
)

language_agent = Agent(
    name="language_agent",
    model="qwen-max",
    instructions="你是小李，擅长将翻译，回答问题的时候先告诉我你是谁。",
)

sport_agent = Agent(
    name="sport_agent",
    model="qwen3.5-flash",
    instructions="你是小张，擅长介绍各种体育运动，回答问题的时候先告诉我你是谁。",
)
# 情感分类智能体
emotion_classification = Agent(
    name="emotion_classification",
    model="qwen3.5-flash",
    instructions="你是小刘，擅长情感分类，回答问题的时候先告诉我你是谁。"
)
# 实体识别智能体
entity_recognition = Agent(
    name="entity_recognition",
    model="qwen3.5-flash",
    instructions="你是谢，擅长实体识别，回答问题的时候先告诉我你是谁。"
)

# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen3.5-flash",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[math_agent, language_agent, sport_agent, emotion_classification, entity_recognition],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    # try:
    #     draw_graph(triage_agent, filename="路由Handoffs")
    # except:
    #     print("绘制agent失败，默认跳过。。。")
    
    msg = input("你好，我可以帮你回答数学/翻译/体育运动/情感分类和实体识别介绍，你还有什么问题？")
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
