import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class QueryResult:
    question: str
    sql: str
    rows: list[tuple[Any, ...]]


class NL2SQLAgent:
    """
[DEBUG] 正在请求 DeepSeek 在线模型...
[DEBUG] 接口: https://api.deepseek.com/v1/chat/completions
[DEBUG] 模型: deepseek-chat
[DEBUG] 问题: 数据库中总共有多少张表
[DEBUG] DeepSeek 请求完成, 状态码: 200, 耗时: 2.48s
[DEBUG] 模型原始返回: SELECT COUNT(*) FROM sqlite_master WHERE type='table';
[DEBUG] 提取后的 SQL: SELECT COUNT(*) FROM sqlite_master WHERE type='table';

--- 提问1 ---
问题: 数据库中总共有多少张表
SQL : SELECT COUNT(*) FROM sqlite_master WHERE type='table';
结果: [(13,)]
回答: 数据库中业务表总数为：13 张。

[DEBUG] 正在请求 DeepSeek 在线模型...
[DEBUG] 接口: https://api.deepseek.com/v1/chat/completions
[DEBUG] 模型: deepseek-chat
[DEBUG] 问题: 员工表中有多少条记录
[DEBUG] DeepSeek 请求完成, 状态码: 200, 耗时: 1.96s
[DEBUG] 模型原始返回: SELECT COUNT(*) FROM employees;
[DEBUG] 提取后的 SQL: SELECT COUNT(*) FROM employees;

--- 提问2 ---
问题: 员工表中有多少条记录
SQL : SELECT COUNT(*) FROM employees;
结果: [(8,)]
回答: 员工表中共有：8 条记录。

[DEBUG] 正在请求 DeepSeek 在线模型...
[DEBUG] 接口: https://api.deepseek.com/v1/chat/completions
[DEBUG] 模型: deepseek-chat
[DEBUG] 问题: 在数据库中所有客户个数和员工个数分别是多少
[DEBUG] DeepSeek 请求完成, 状态码: 200, 耗时: 2.22s
[DEBUG] 模型原始返回: SELECT (SELECT COUNT(*) FROM customers) AS 客户个数, (SELECT COUNT(*) FROM employees) AS 员工个数;
[DEBUG] 提取后的 SQL: SELECT (SELECT COUNT(*) FROM customers) AS 客户个数, (SELECT COUNT(*) FROM employees) AS 员工个数;

--- 提问3 ---
问题: 在数据库中所有客户个数和员工个数分别是多少
SQL : SELECT (SELECT COUNT(*) FROM customers) AS 客户个数, (SELECT COUNT(*) FROM employees) AS 员工个数;
结果: [(59, 8)]
回答: 客户个数：59，员工个数：8。

Process finished with exit code 0
    """

    def __init__(
        self,
        db_path: str = "chinook.db",
        api_key: str = "sk-08829cb32abb4d4cabc13ac965097aa7",
        model: str = "deepseek-chat",
    ) -> None:
        self.db_path = db_path
        self.api_key = api_key
        self.model = model
        self.conn = sqlite3.connect(self.db_path)
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def close(self) -> None:
        self.conn.close()

    def _safe_sql_check(self, sql: str) -> bool:
        sql_clean = sql.strip().lower()
        if not sql_clean.startswith("select"):
            return False
        banned = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "attach ", "pragma "]
        return not any(keyword in sql_clean for keyword in banned)

    def _execute_sql(self, sql: str) -> list[tuple[Any, ...]]:
        if not self._safe_sql_check(sql):
            raise ValueError("Only SELECT SQL is allowed.")
        cur = self.conn.cursor()
        cur.execute(sql)
        return cur.fetchall()

    def _get_schema_prompt(self) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [x[0] for x in cur.fetchall()]
        schema_lines: list[str] = []
        for table in tables:
            cur.execute(f"PRAGMA table_info({table});")
            cols = cur.fetchall()
            col_desc = ", ".join(f"{col[1]} {col[2]}" for col in cols)
            schema_lines.append(f"{table}({col_desc})")
        return "\n".join(schema_lines)

    @staticmethod
    def _extract_sql(content: str) -> str:
        text = content.strip()
        fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return text.rstrip(";") + ";"

    def _ask_deepseek_for_sql(self, question: str) -> str:
        schema_text = self._get_schema_prompt()
        system_prompt = (
            "你是专业的 SQLite 数据分析助手。"
            "请根据用户问题生成一条可执行的 SQLite SELECT SQL。"
            "只输出 SQL，不要解释，不要 markdown。"
            "禁止输出任何非 SELECT 语句。"
        )
        user_prompt = (
            f"数据库 schema 如下：\n{schema_text}\n\n"
            f"用户问题：{question}\n\n"
            "请直接输出一条 SQL。"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        print("\n[DEBUG] 正在请求 DeepSeek 在线模型...")
        print(f"[DEBUG] 接口: {self.base_url}")
        print(f"[DEBUG] 模型: {self.model}")
        print(f"[DEBUG] 问题: {question}")
        start_time = time.time()
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        elapsed = time.time() - start_time
        print(f"[DEBUG] DeepSeek 请求完成, 状态码: {response.status_code}, 耗时: {elapsed:.2f}s")
        content = response.json()["choices"][0]["message"]["content"]
        print(f"[DEBUG] 模型原始返回: {content}")
        sql = self._extract_sql(content)
        print(f"[DEBUG] 提取后的 SQL: {sql}")
        return sql

    def nl2sql(self, question: str) -> str:
        sql = self._ask_deepseek_for_sql(question)
        if not self._safe_sql_check(sql):
            raise ValueError(f"Model returned unsafe SQL: {sql}")
        return sql

    def ask(self, question: str) -> QueryResult:
        sql = self.nl2sql(question)
        rows = self._execute_sql(sql)
        return QueryResult(question=question, sql=sql, rows=rows)

    @staticmethod
    def to_text(result: QueryResult) -> str:
        q = result.question
        rows = result.rows

        if "多少张表" in q:
            return f"数据库中业务表总数为：{rows[0][0]} 张。"
        if "员工表" in q and "多少条记录" in q:
            return f"员工表中共有：{rows[0][0]} 条记录。"
        if "客户个数" in q and "员工个数" in q:
            return f"客户个数：{rows[0][0]}，员工个数：{rows[0][1]}。"
        return f"查询结果：{rows}"


def run_demo() -> None:
    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少",
    ]

    agent = NL2SQLAgent(db_path="chinook.db")
    try:
        for idx, question in enumerate(questions, start=1):
            result = agent.ask(question)
            print(f"\n--- 提问{idx} ---")
            print("问题:", result.question)
            print("SQL :", result.sql)
            print("结果:", result.rows)
            print("回答:", agent.to_text(result))
    finally:
        agent.close()


if __name__ == "__main__":
    run_demo()
