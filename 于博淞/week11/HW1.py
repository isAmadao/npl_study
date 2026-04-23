import os
import asyncio
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any

# --- 0. 配置环境变量 ---
os.environ["OPENAI_API_KEY"] = "sk-02e847ab13a543798c4860e15d459293"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner


# --- 1. 定义输出结构 ---

class SentimentOutput(BaseModel):

    sentiment: str = Field(..., alias="sentiment", description="情感类型：积极/消极/中立")
    confidence: float = Field(..., description="0到1之间的置信度")
    explanation: str = Field(..., alias="explanation", description="简短的解释")


class NEROutput(BaseModel):
    entities: List[Dict[str, str]] = Field(..., description="提取出的实体列表")


# --- 2. 初始化大模型配置 ---
try:
    from agents import set_default_openai_api, set_tracing_disabled

    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
except ImportError:
    pass

# --- 3. 创建 Agents ---

sentiment_agent = Agent(
    name="Sentiment Classifier",
    model="qwen3.5-plus",
    instructions="""你是一个精准的情感分析引擎。
    请分析用户文本的情感。
    **必须**输出严格的 JSON 格式，且字段名必须完全匹配以下定义：
    {
        "sentiment": "积极" (或者 "消极", "中立"),
        "confidence": 0.95,
        "explanation": "原因..."
    }
    不要使用 sentiment_type 或 brief_explanation 等其他字段名。""",
    output_type=SentimentOutput
)

ner_agent = Agent(
    name="NER Extractor",
    model="qwen3.5-plus",
    instructions="""你是一个命名实体识别系统。
    请提取文本中的人名、地名、组织名。
    **必须**输出严格的 JSON 格式：
    {
        "entities": [
            {"name": "实体名", "type": "类型"},
            ...
        ]
    }""",
    output_type=NEROutput
)

triage_agent = Agent(
    name="Triage Agent",
    model="qwen3.5-plus",
    instructions="你是路由助手。如果是分析情绪，转给 Sentiment Classifier。如果是提取实体，转给 NER Extractor。",
    handoffs=[sentiment_agent, ner_agent],
)


# --- 4. 主程序 ---
async def main():
    print("--- 启动文本分析代理系统 ---\n")

    test_cases = [
        "这部电影的特效太震撼了，剧情也非常感人，绝对是今年的佳作！",
        "苹果公司在2023年发布了新的iPhone 15，发布会地点在加利福尼亚州的库比蒂诺。",
        "请帮我预订明天下午从北京飞往上海的机票。"
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"--- 测试 {i} ---")
        print(f"👤 用户输入: {query}")

        try:
            result = await Runner.run(triage_agent, query)

            print(f"最终输出:")
            # 如果 final_output 是 Pydantic 对象，打印其字典形式
            if hasattr(result.final_output, 'model_dump'):
                print(result.final_output.model_dump())
            else:
                print(result.final_output)

        except Exception as e:
            print(f"❌ 运行异常: {e}")

        print("-" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())