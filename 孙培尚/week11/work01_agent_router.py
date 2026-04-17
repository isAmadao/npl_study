import os

# 阿里云百炼 API Key：https://bailian.console.aliyun.com/?tab=model#/api-key
# 请替换为你自己的有效 API Key
os.environ["OPENAI_API_KEY"] = "sk-1f8f970c557d41b9899269dc981366f9"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 启动方式：
# uvicorn 37_NLP任务路由_FastAPI:app --reload --port 8000
# 访问文档：http://127.0.0.1:8000/docs
# 测试接口：http://127.0.0.1:8000/analyze?text=苹果公司的CEO库克昨天在北京宣布了一项重大投资计划

from fastapi import FastAPI, Query
from pydantic import BaseModel

from agents import Agent, Runner, trace
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# -------------------------------------------------------
# 子 Agent 1：情感分类
# -------------------------------------------------------
sentiment_agent = Agent(
    name="sentiment_agent",
    model="qwen-max",
    instructions=(
        "你是一个专业的情感分析助手。"
        "请对用户提供的文本进行情感分类，判断其情感倾向。\n\n"
        "输出格式要求：\n"
        "1. **情感类别**：正面 / 负面 / 中性\n"
        "2. **置信度**：高 / 中 / 低\n"
        "3. **关键情感词**：列出触发该判断的关键词\n"
        "4. **分析说明**：简短说明判断依据"
    ),
)

# -------------------------------------------------------
# 子 Agent 2：实体识别
# -------------------------------------------------------
ner_agent = Agent(
    name="ner_agent",
    model="qwen-max",
    instructions=(
        "你是一个专业的命名实体识别（NER）助手。"
        "请对用户提供的文本进行实体识别，提取其中的命名实体。\n\n"
        "需要识别的实体类型包括：\n"
        "- **人名（PER）**：人物姓名\n"
        "- **地名（LOC）**：地点、地区、国家等\n"
        "- **组织机构（ORG）**：公司、学校、政府机构等\n"
        "- **时间（TIME）**：日期、时间表达式\n"
        "- **数值（NUM）**：数字、金额、比例等\n"
        "- **其他（MISC）**：产品名、事件名等其他重要实体\n\n"
        "输出格式要求：按实体类型分组列出，每个实体标注其在原文中的位置。"
    ),
)

# -------------------------------------------------------
# 主 Agent（Orchestrator）：路由调度
# -------------------------------------------------------
main_agent = Agent(
    name="main_agent",
    model="qwen-max",
    instructions=(
        "你是一个 NLP 任务调度助手。你的职责是理解用户的请求，"
        "然后调用合适的工具完成任务。\n\n"
        "可用工具说明：\n"
        "- analyze_sentiment：对文本进行情感分析，判断文本的情感倾向（正面/负面/中性）\n"
        "- recognize_entities：对文本进行命名实体识别，提取人名、地名、组织机构等实体\n\n"
        "判断规则：\n"
        "1. 如果用户想了解文本的情感、态度、倾向 → 调用 analyze_sentiment\n"
        "2. 如果用户想提取文本中的人名、地名、机构等信息 → 调用 recognize_entities\n"
        "3. 如果用户同时需要两者，按顺序依次调用两个工具\n"
        "4. 如果用户没有明确说明任务类型，根据文本内容自动判断最合适的任务\n\n"
        "注意：你只能通过工具完成分析，不要自己直接给出分析结果。"
    ),
    tools=[
        sentiment_agent.as_tool(
            tool_name="analyze_sentiment",
            tool_description="对文本进行情感分析，识别文本的情感倾向（正面/负面/中性），并给出置信度和关键情感词",
        ),
        ner_agent.as_tool(
            tool_name="recognize_entities",
            tool_description="对文本进行命名实体识别（NER），提取人名、地名、组织机构、时间、数值等命名实体",
        ),
    ],
)

# -------------------------------------------------------
# FastAPI 应用
# -------------------------------------------------------
app = FastAPI(
    title="NLP 智能分析系统",
    description="基于多 Agent 路由的 NLP 服务，支持情感分类和实体识别",
    version="1.0.0",
)


# 响应数据结构
class AnalyzeResponse(BaseModel):
    text: str        # 用户输入的原始文本
    result: str      # 分析结果


@app.get("/analyze", response_model=AnalyzeResponse, summary="NLP 智能分析")
async def analyze(
    text: str = Query(..., description="需要分析的文本，可以说明任务类型（情感分析/实体识别），也可直接输入文本由系统自动判断")
):

    with trace("NLP路由-FastAPI"):
        result = await Runner.run(main_agent, text)

    return AnalyzeResponse(text=text, result=result.final_output)

