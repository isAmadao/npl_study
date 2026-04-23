import os
import asyncio

os.environ["OPENAI_API_KEY"] = "sk-**"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


sentiment_agent = Agent(
    name="SentimentAgent",
    model="qwen3.5-plus",
    handoff_description="对文本进行情感分析，判断是积极、消极还是中性情感",
    instructions="你是一个情感分析专家。请分析用户输入的文本，判断其情感倾向。\
要求：\
1. 分析文本的情感（积极/消极/中性）\
2. 给出情感评分（-10到10，负数表示消极，正数表示积极）\
3. 简要解释判断理由\
请用JSON格式返回结果，包含以下字段：sentiment(积极/消极/中性)、score(评分)、reason(理由)"
)

entity_agent = Agent(
    name="EntityAgent",
    model="qwen3.5-plus",
    handoff_description="对文本进行命名实体识别，提取人名、地名、机构名等实体",
    instructions="你是一个命名实体识别专家。请分析用户输入的文本，识别其中的命名实体。\
要求：\
1. 识别人名（PER）\
2. 识别地名（LOC）\
3. 识别机构名（ORG）\
4. 识别其他重要实体\
请用JSON格式返回结果，包含实体列表，每项包含：text(实体文本)、type(实体类型)、description(描述)"
)

triage_agent = Agent(
    name="TriageAgent",
    model="qwen3.5-plus",
    instructions="你是一个智能助手，负责接收用户请求并分派给专业的子Agent。\
根据用户的需求，将请求路由到以下其中一个Agent：\
1. SentimentAgent - 当用户需要分析文本的情感倾向时使用\
2. EntityAgent - 当用户需要从文本中提取人名、地名、机构名等实体时使用\
如果用户没有明确指定，请根据文本内容判断：\
- 包含情感词（喜欢、讨厌、开心、悲伤等）-> 使用SentimentAgent\
- 包含具体名称（人名、地名、公司名等）-> 使用EntityAgent\
请直接使用handoff将请求发送给对应的Agent，不要自己回答。",
    handoffs=[sentiment_agent, entity_agent]
)


async def main():
    print("=" * 60)
    print("🤖 多Agent文本分析系统")
    print("=" * 60)
    print("\n功能说明：")
    print("  • 情感分类 - 分析文本的情感倾向（积极/消极/中性）")
    print("  • 实体识别 - 提取文本中的人名、地名、机构名等")
    print("\n输入 'quit' 退出程序")
    print("=" * 60)

    while True:
        print("\n" + "-" * 60)
        user_input = input("\n📝 请输入文本进行分析: ").strip()

        if user_input.lower() in ['quit', '退出', 'exit']:
            print("\n👋 再见！感谢使用多Agent文本分析系统！")
            break

        if not user_input:
            print("⚠️ 请输入有效的文本！")
            continue

        print("\n🔄 正在分析和路由请求...\n")

        try:
            result = await Runner.run(triage_agent, user_input)
            print("\n📤 分析结果:")
            print("-" * 40)
            print(result.final_output)
            print("-" * 40)
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
