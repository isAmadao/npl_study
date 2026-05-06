from openai import OpenAI, AsyncOpenAI
from database_tools import DatabaseTools
import os
import re


class NL2SQLAgent:
    def __init__(self, tools: DatabaseTools, api_key: str = None, model: str = "qwen-plus"):
        self.tools = tools
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-8fb3abb209d34b1a89932c3ced430028"),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建包含数据库 Schema 的系统提示"""
        tables_info = self.tools.get_tables_info()
        return f"""你是一个 SQL 专家。根据用户的问题，生成正确的 SQLite SQL 查询。
                数据库信息：
                {tables_info}
                外键关系：
                {self.tools.db_parser.get_data_relations().to_string()}

                要求：
                - 只生成 SELECT 查询（禁止 INSERT/UPDATE/DELETE）。
                - 不要添加任何解释，只输出 SQL 语句。
                - 如果问题无法用 SQL 回答，输出 "UNSUPPORTED"。
                """

    def ask(self, user_question: str) -> str:
        """处理用户问题：生成 SQL → 执行 → 返回结果"""
        # 调用 LLM 生成 SQL
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_question},
            ],
            temperature=0,
        )
        sql_candidate = response.choices[0].message.content.strip()
        # 清理可能的 markdown 标记
        sql_candidate = re.sub(r"^```sql\n|```$", "", sql_candidate, flags=re.MULTILINE).strip()

        if sql_candidate.upper() == "UNSUPPORTED":
            return "抱歉，我无法将此问题转换为 SQL 查询。"

        print(f"\n[Agent 生成的 SQL]:\n{sql_candidate}\n")
        # 执行 SQL
        result = self.tools.run_sql(sql_candidate)
        return f"查询结果：\n{result}"
