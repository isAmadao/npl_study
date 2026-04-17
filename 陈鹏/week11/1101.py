"""
作业1: 安装openai-agents框架，实现如下的一个程序：
• 有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
• 子agent 1: 对文本进行情感分类
• 子agent 2: 对文本进行实体识别
"""
import os

os.environ["OPENAI_API_KEY"] = "sk-fe0209453f0d48179de8bd53a6ce028c"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

class HomeworkOutput(BaseModel):
    """用于判断用户请求是否属于情感分类或实体识别类问题的结构"""
    is_homework: bool

# 守卫检查代理 - 》 本质也是通过大模型调用完成的
guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions="判断用户的问题是否属于情感分类、实体识别相关问题。如果是，'is_homework'应为 True， json 返回",
    output_type=HomeworkOutput,  # openai 官方推荐的一个语法，推荐大模型输出的格式类型，国内模型支持的不太好；
)

# 文本情感分类代理
sentiment_analysis_agent = Agent(
    name="Sentiment Analysis",
    model="qwen-max",
    handoff_description="专门对文本进行情感分类，判断正面、负面、中性情绪。",
    instructions="您是专业的情感导师。你只需要输出情感结果：正面 / 负面 / 中性。不要多余解释，简洁输出。",
)

# 文本识别代理
entity_identify_agent = Agent(
    name="Entity Identify",
    model="qwen-max",
    handoff_description="专门从文本中提取人名、地名、机构名、时间、数字等实体。",
    instructions="你是实体识别专家。从用户输入文本里提取所有实体，分行列出。格式：实体类型：内容例如：人名：张三。",
)


async def homework_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为已知的。
    如果不是功课 ('is_homework' 为 False)，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(HomeworkOutput)

    tripwire_triggered = not final_output.is_homework

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )

# 先进行输入的校验 guardrail_agent
triage_agent = Agent(
    name="Triage Agent",
    model="qwen-max",
    instructions="您的任务是根据用户的输入内容，判断应该将请求分派给 'Sentiment Analysis' 还是 'Entity Identify'。",
    handoffs=[sentiment_analysis_agent, entity_identify_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)


async def main():
    print("--- 启动智多星 ---")

    print("\n" + "=" * 50)
    try:
        # 只改这里：从固定文字 → 用户输入
        query = input("请输入你的疑问：")
        print(f"**用户提问:** {query}")

        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)


if __name__ == "__main__":
    asyncio.run(main())
