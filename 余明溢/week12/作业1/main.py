#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from database_tools import DatabaseTools
from db_parser import DBParser
from nl_2_sql_agent import NL2SQLAgent

if __name__ == "__main__":
    # 初始化数据库解析器（请确保 chinook.db 在当前目录）
    parser = DBParser("sqlite:///chinook.db")
    tools = DatabaseTools(parser)

    # 初始化 Agent（需设置环境变量 OPENAI_API_KEY，或直接传入）
    agent = NL2SQLAgent(tools)  # 从环境变量读取 API Key

    print("数据库问答 Agent 已启动。输入问题（输入 'exit' 退出）：")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        answer = agent.ask(user_input)
        print(answer)