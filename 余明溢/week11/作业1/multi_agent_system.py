import asyncio
import os

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_default_openai_key,
    set_tracing_disabled
)


class MultiAgentSystem:
    """基于通义千问的多 Agent 系统"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = "qwen-plus",
    ):
        """
        初始化多 Agent 系统

        Args:
            api_key: DashScope API Key，默认从环境变量读取
            base_url: DashScope OpenAI 兼容 API 地址
            model_name: 通义千问模型名称（qwen-turbo / qwen-plus / qwen-max）
        """
        # 1. 禁用 tracing
        set_tracing_disabled(True)
        # 获取 API Key
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY","sk-8fb3abb209d34b1a89932c3ced430028")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")

        # DashScope OpenAI 兼容模式地址
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 创建 OpenAI 兼容客户端
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # 创建模型实例
        self.model = OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=self.client,
        )

        # 设置为默认客户端
        set_default_openai_client(self.client, use_for_tracing=False)

        # 创建三个 Agent
        self._create_agents()

    def _create_agents(self):
        """创建三个 Agent 实例"""

        # 情感分析子 Agent
        self.sentiment_agent = Agent(
            name="Sentiment Analyzer",
            instructions="""
            你是一个情感分析专家。请对用户输入的文本进行情感分类。
            分析要求：
            1. 判断情感类别：积极(Positive)、消极(Negative)或中性(Neutral)
            2. 给出置信度评分（0-100分）
            3. 简要说明判断理由

            输出格式示例：
            ---
            情感: 积极
            置信度: 95
            理由: 文本中包含"excellent"、"great"等正面词汇，整体语气积极。
            ---
            """,
            model=self.model,
        )

        # 实体识别子 Agent
        self.ner_agent = Agent(
            name="Named Entity Recognizer",
            instructions="""
            你是一个命名实体识别专家。请识别用户文本中的命名实体。

            需要识别的实体类型：
            - 人名 (PER)：个人姓名
            - 地名 (LOC)：地理位置、城市、国家等
            - 组织名 (ORG)：公司、机构、政府组织等
            - 日期时间 (DATE)：时间表达、日期
            - 其他 (MISC)：其他专有名词

            输出格式示例：
            ---
            识别到的实体：
            1. 实体: 张三, 类型: PER, 位置: [0:2]
            2. 实体: 北京, 类型: LOC, 位置: [5:7]
            3. 实体: 2024年1月1日, 类型: DATE, 位置: [10:16]

            未识别到实体时，输出：未识别到任何命名实体。
            ---
            """,
            model=self.model,
        )

        # 主控 Agent
        self.triage_agent = Agent(
            name="Triage Agent",
            instructions="""
            你是一个任务分发智能体。请根据用户请求的内容，选择合适的专家Agent来回答。

            路由规则：
            1. 如果用户请求包含以下关键词，请转交给情感分析专家(Sentiment Analyzer)：
               - "情感"、"心情"、"态度"、"情绪"
               - "评价"、"feedback"、"sentiment"
               - 明显包含情绪表达的文本（如"我很喜欢..."、"太糟糕了..."）

            2. 如果用户请求包含以下关键词，请转交给实体识别专家(Named Entity Recognizer)：
               - "实体"、"人名"、"地名"、"组织"、"日期"
               - "NER"、"extract entities"、"识别"
               - 询问"谁"、"哪里"、"什么时候"、"什么公司"

            3. 如果用户没有明确指示或请求模糊，默认转交给实体识别专家。

            判断完成后，使用对应的 transfer_to_* 工具转交给合适的专家Agent。
            """,
            handoffs=[self.sentiment_agent, self.ner_agent],
            model=self.model,
        )

    async def run(self, user_input: str) -> str:
        """运行多 Agent 系统"""
        print(f"\n📝 用户输入: {user_input}")
        print("🔄 正在分析并路由请求...")

        result = await Runner.run(self.triage_agent, input=user_input)

        if hasattr(result, "last_agent") and result.last_agent:
            print(f"✅ 最终由 {result.last_agent.name} 回答")

        return result.final_output


async def main():
    """主函数"""

    system = MultiAgentSystem(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        model_name="qwen-plus",  # 可选: qwen-turbo, qwen-plus, qwen-max
    )

    print("=" * 60)
    print("🤖 多 Agent 智能路由系统（基于通义千问）")
    print("=" * 60)
    print("\n使用说明：")
    print("- 输入文本进行情感分析或实体识别")
    print("- 系统会根据请求内容自动路由到合适的 Agent")
    print("- 输入 'exit' 或 'quit' 退出\n")

    while True:
        user_input = input("\n💬 请输入分析内容: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 再见！")
            break

        if not user_input:
            print("⚠️ 输入不能为空，请重新输入")
            continue

        try:
            response = await system.run(user_input)
            print("\n" + "=" * 50)
            print("📋 回答结果:")
            print(response)
            print("=" * 50)
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())