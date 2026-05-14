import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import requests

API_KEY = "sk-02e847ab13a543798c4860e15d459293"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen3.6-plus"

DEMO_QUESTIONS = [
    "数据库中总共有多少张表",
    "员工表中有多少条记录",
    "在数据库中所有客户个数和员工个数分别是多少",
]

@dataclass
class Answer:
    question: str
    sql: str
    rows: list[tuple[Any, ...]]
    elapsed: float = 0.0
    natural: str = field(default="")


# Agent

class NL2SQLAgent:

    _BANNED_KEYWORDS = frozenset([
        "insert", "update", "delete", "drop", "alter",
        "create", "attach", "pragma", "vacuum",
    ])

    def __init__(self, db_path: str = "./chinook.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        print(f"[Agent] 已连接数据库: {db_path}")


    def ask(self, question: str) -> Answer:
        sql, elapsed = self._nl_to_sql(question)
        rows = self._run_select(sql)
        answer = Answer(question=question, sql=sql, rows=rows, elapsed=elapsed)
        answer.natural = self._verbalize(answer)
        return answer

    def close(self) -> None:
        self._conn.close()


    def _build_schema(self) -> str:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        table_names = [row[0] for row in cur.fetchall()]

        lines: list[str] = []
        for tname in table_names:
            cur.execute(f"PRAGMA table_info({tname});")
            cols = ", ".join(f"{c[1]} {c[2]}" for c in cur.fetchall())
            lines.append(f"  {tname}({cols})")

        total = len(table_names)
        header = (
            f"-- SQLite 数据库，共 {total} 张用户表（不含 sqlite_* 系统表）\n"
            "-- 表结构如下：\n"
        )
        return header + "\n".join(lines)


    def _nl_to_sql(self, question: str) -> tuple[str, float]:
        schema = self._build_schema()

        system_msg = (
            "你是一名资深 SQLite 数据库工程师。\n"
            "任务：根据用户的中文问题，生成一条准确的 SQLite SELECT 语句。\n"
            "规则：\n"
            "  1. 只输出 SQL，不要任何解释或 markdown 包裹。\n"
            "  2. 禁止使用 INSERT / UPDATE / DELETE / DROP 等写操作。\n"
            "  3. 若问题涉及'有多少张表'，查 sqlite_master（type='table'）即可。\n"
            "  4. 列别名使用中文，便于阅读。"
        )
        user_msg = (
            f"数据库 Schema：\n{schema}\n\n"
            f"用户问题：{question}\n\n"
            "请输出对应的 SQL 语句："
        )

        payload = {
            "model": MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        print(f"\n[LLM] 发送请求 → 模型: {MODEL}  问题: {question}")
        t0 = time.perf_counter()
        resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        print(f"[LLM] 响应完成  状态: {resp.status_code}  耗时: {elapsed:.2f}s")

        raw = resp.json()["choices"][0]["message"]["content"]
        print(f"[LLM] 原始输出: {raw.strip()}")

        sql = self._extract_sql(raw)
        print(f"[LLM] 提取 SQL: {sql}")
        return sql, elapsed


    @staticmethod
    def _extract_sql(text: str) -> str:
        text = text.strip()
        fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # 只取第一条语句
        text = text.split(";")[0].strip()
        return text + ";"

    def _is_safe(self, sql: str) -> bool:
        lower = sql.lower()
        if not lower.lstrip().startswith("select"):
            return False
        return not any(kw in lower for kw in self._BANNED_KEYWORDS)

    def _run_select(self, sql: str) -> list[tuple[Any, ...]]:
        if not self._is_safe(sql):
            raise PermissionError(f"拒绝执行非 SELECT 语句: {sql}")
        cur = self._conn.cursor()
        cur.execute(sql)
        return cur.fetchall()


    @staticmethod
    def _verbalize(ans: Answer) -> str:
        q, rows = ans.question, ans.rows
        if not rows:
            return "查询无结果。"

        first = rows[0]

        if "多少张表" in q:
            return f"数据库中共有 {first[0]} 张表（含系统表）。"

        if "员工表" in q:
            return f"员工表（employees）中共有 {first[0]} 条记录。"

        if "客户" in q and "员工" in q:
            if len(first) >= 2:
                return f"客户（customers）共 {first[0]} 人，员工（employees）共 {first[1]} 人。"
            counts = [r[0] for r in rows]
            return f"客户（customers）共 {counts[0]} 人，员工（employees）共 {counts[1]} 人。"

        return f"查询结果：{rows}"


def main() -> None:
    agent = NL2SQLAgent(db_path="chinook.db")
    try:
        for idx, q in enumerate(DEMO_QUESTIONS, start=1):
            ans = agent.ask(q)
            print(f"\n{'─' * 50}")
            print(f"提问 {idx}: {ans.question}")
            print(f"SQL   : {ans.sql}")
            print(f"结果  : {ans.rows}")
            print(f"回答  : {ans.natural}")
            print(f"耗时  : {ans.elapsed:.2f}s")
    finally:
        agent.close()


if __name__ == "__main__":
    main()




# [Agent] 已连接数据库: chinook.db
#
# [LLM] 发送请求 → 模型: qwen3.6-plus  问题: 数据库中总共有多少张表
# [LLM] 响应完成  状态: 200  耗时: 27.55s
# [LLM] 原始输出: SELECT COUNT(*) AS 表数量 FROM sqlite_master WHERE type='table';
# [LLM] 提取 SQL: SELECT COUNT(*) AS 表数量 FROM sqlite_master WHERE type='table';
#
# ──────────────────────────────────────────────────
# 提问 1: 数据库中总共有多少张表
# SQL   : SELECT COUNT(*) AS 表数量 FROM sqlite_master WHERE type='table';
# 结果  : [(13,)]
# 回答  : 数据库中共有 13 张表（含系统表）。
# 耗时  : 27.55s
#
# [LLM] 发送请求 → 模型: qwen3.6-plus  问题: 员工表中有多少条记录
# [LLM] 响应完成  状态: 200  耗时: 19.33s
# [LLM] 原始输出: SELECT COUNT(*) AS 记录数 FROM employees;
# [LLM] 提取 SQL: SELECT COUNT(*) AS 记录数 FROM employees;
#
# ──────────────────────────────────────────────────
# 提问 2: 员工表中有多少条记录
# SQL   : SELECT COUNT(*) AS 记录数 FROM employees;
# 结果  : [(8,)]
# 回答  : 员工表（employees）中共有 8 条记录。
# 耗时  : 19.33s
#
# [LLM] 发送请求 → 模型: qwen3.6-plus  问题: 在数据库中所有客户个数和员工个数分别是多少
# [LLM] 响应完成  状态: 200  耗时: 27.46s
# [LLM] 原始输出: SELECT (SELECT COUNT(*) FROM customers) AS 客户个数, (SELECT COUNT(*) FROM employees) AS 员工个数;
# [LLM] 提取 SQL: SELECT (SELECT COUNT(*) FROM customers) AS 客户个数, (SELECT COUNT(*) FROM employees) AS 员工个数;
#
# ──────────────────────────────────────────────────
# 提问 3: 在数据库中所有客户个数和员工个数分别是多少
# SQL   : SELECT (SELECT COUNT(*) FROM customers) AS 客户个数, (SELECT COUNT(*) FROM employees) AS 员工个数;
# 结果  : [(59, 8)]
# 回答  : 客户（customers）共 59 人，员工（employees）共 8 人。
# 耗时  : 27.46s
#
# Process finished with exit code 0
