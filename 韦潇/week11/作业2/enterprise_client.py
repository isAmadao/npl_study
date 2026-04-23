import os
os.environ["OPENAI_API_KEY"] = "sk-**"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.mcp.server import MCPServerSse
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

SYSTEM_PROMPT = """你是一个企业职能助手，可以帮助员工查询各类企业信息。

你可以使用以下工具：
1. search_employee - 根据员工姓名查询员工信息（部门、职位、邮箱、电话）
2. check_meeting_room - 查询会议室的预订情况和基本信息
3. query_policy - 查询公司政策和规章制度（年假、迟到、加班、报销、请假、入职、离职、晋升等）
4. list_departments - 获取公司所有部门列表

当用户提出查询请求时，请选择合适的工具进行调用，返回结构化的结果。"""


async def main():
    print("=" * 60)
    print("🏢 企业职能助手")
    print("=" * 60)
    print("\n功能说明：")
    print("  • 查询员工信息 - 输入员工姓名")
    print("  • 查询会议室 - 输入会议室编号")
    print("  • 查询公司政策 - 输入关键词（年假/迟到/加班/报销等）")
    print("  • 查看部门列表")
    print("\n输入 'quit' 退出程序")
    print("=" * 60)

    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    async with MCPServerSse(
        name="EnterpriseToolsServer",
        params={"url": "http://localhost:8902/sse"}
    ) as mcp_server:

        agent = Agent(
            name="企业职能助手",
            instructions=SYSTEM_PROMPT,
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model="qwen3.5-plus",
                openai_client=external_client,
            )
        )

        while True:
            print("\n" + "-" * 60)
            user_input = input("\n📝 请输入您的需求: ").strip()

            if user_input.lower() in ['quit', '退出', 'exit']:
                print("\n👋 再见！感谢使用企业职能助手！")
                break

            if not user_input:
                print("⚠️ 请输入有效的需求！")
                continue

            print("\n🔄 正在分析和执行...\n")

            try:
                result = await Runner.run(agent, user_input)
                print("\n📤 执行结果:")
                print("-" * 40)
                print(result.final_output)
                print("-" * 40)
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
