from __future__ import annotations

import asyncio
import uuid

from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, Runner, TResponseInputItem, set_tracing_disabled, trace
from agents.model_settings import ModelSettings


DEEPSEEK_API_KEY = "sk-409ede8a980a4019a1a1e2baf0cea889"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def build_router_system(
    *,
    openai_client: AsyncOpenAI,
    model_name: str,
) -> Agent:
    model = OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client)

    sentiment_agent = Agent(
        name="sentiment_agent",
        instructions=(
            "你只负责情感分析。根据用户给出的文本，输出："
            "1）整体情感标签（如：正面、负面、中性或更细粒度）；"
            "2）一两句简要依据。"
            "不要进行命名实体识别。"
        ),
        model=model,
    )

    ner_agent = Agent(
        name="ner_agent",
        instructions=(
            "你只负责命名实体识别。从用户文本中列出重要实体，并标注类型"
            "（人名、地名、组织机构、时间、产品名等）。用条目或表格形式输出。"
            "不要做情感倾向判断作为主结论。"
        ),
        model=model,
    )

    main_agent = Agent(
        name="main_router",
        instructions=(
            "你是主调度 Agent。用户会粘贴一段文字或提出分析需求。"
            "你必须且只能通过下面两个工具之一完成任务，禁止在工具之外自己编造情感或实体结果。\n"
            "选择规则：\n"
            "- 用户要情感、情绪、褒贬、满意度、态度、心情、身体不适感受、评价好坏 → 调用 analyze_sentiment。\n"
            "- 用户要实体、人名、地名、公司机构、NER、抽取专名、列出实体 → 调用 extract_named_entities。\n"
            "- 用户未说明类型时：日常感受/抱怨/身体不舒服类 → analyze_sentiment；"
            "明显在列人名地名机构时间等事实要素 → extract_named_entities。\n"
            "每次用户发话只调用**一个**工具；把用户原文完整交给该工具对应的子任务。"
        ),
        tools=[
            sentiment_agent.as_tool(
                tool_name="analyze_sentiment",
                tool_description=(
                    "情感分析：判断文本整体情绪倾向（正面/负面/中性等）。"
                    "用于用户关心态度、情绪、满意度、评论褒贬、心情或身体不适表达时。"
                ),
            ),
            ner_agent.as_tool(
                tool_name="extract_named_entities",
                tool_description=(
                    "命名实体识别（NER）：从文本中抽取人名、地名、组织机构、时间、公司名等专名。"
                    "用于用户明确要求抽实体、人名地名、NER、公司名、机构名时。"
                ),
            ),
        ],
        model_settings=ModelSettings(
            tool_choice="required",
            parallel_tool_calls=False,
        ),
        model=model,
    )

    return main_agent


async def main() -> None:
    api_key = DEEPSEEK_API_KEY
    base_url = DEEPSEEK_BASE_URL
    model_name = DEEPSEEK_MODEL

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_tracing_disabled(True)

    main_agent = build_router_system(openai_client=client, model_name=model_name)

    conversation_id = uuid.uuid4().hex[:16]
    print("已连接 DeepSeek，模型：", model_name)
    print("（路由方式：主 Agent 通过工具调用子 Agent，适配 DeepSeek）")
    print("输入 quit / exit 结束。\n")

    first = input("请输入你的问题或待分析文本： ").strip()
    if first.lower() in {"quit", "exit"}:
        return

    inputs: list[TResponseInputItem] = [{"role": "user", "content": first}]

    while True:
        with trace("作业1-主Agent路由情感与NER", group_id=conversation_id):
            # 始终从主调度 Agent 开始；子 Agent 仅通过工具执行，避免 handoff 与 DeepSeek 不兼容
            result = await Runner.run(starting_agent=main_agent, input=inputs)

        print("\n--- 本轮模型输出 ---\n")
        print(result.final_output)
        print("\n--------------------\n")

        follow = input("继续输入（quit 退出）： ").strip()
        if follow.lower() in {"quit", "exit"}:
            break

        inputs = result.to_input_list()
        inputs.append({"role": "user", "content": follow})


if __name__ == "__main__":
    asyncio.run(main())
