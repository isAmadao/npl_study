import sqlite3

from agents import OpenAIChatCompletionsModel, Agent, function_tool, Runner, set_default_openai_api, \
    set_tracing_disabled
from openai import AsyncOpenAI

import config
from config import DEFAULT_MODEL

conn = sqlite3.connect('../code/04_SQL-Code-Agent-Demo/chinook.db')

execResult = conn.execute("select name from sqlite_master where type='table'")

print(execResult.fetchall())

'''数据库解析'''
from typing import Any
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData
import pandas as pd


class DBParser:
    '''DBParser'''

    def __init__(self, db_url: str) -> None:
        '''初始化
        db_url: 数据库链接地址
        '''

        # 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'
        elif 'postgres' in db_url:
            self.db_type = 'postgresql'

        # 链接数据库
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()

        # 查看表明
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        self._table_fields = {}  # 数据表字段
        self.foreign_keys = []  # 数据库外键
        self._table_sample = {}  # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            # 累计外键
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']: x for x in table_columns}

            # 对当前字段进行统计
            for column_meta in table_columns:
                # 获取当前字段
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计unique
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                # 统计most frequency value
                field_type = self._table_fields[table_name][column_meta['name']]['type']
                field_type = str(field_type)
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计missing个数
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计max
                query = select(func.max(column_instance))
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                # 统计min
                query = select(func.min(column_instance))
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                # 任意取值
                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [x[0] for x in random_value]
                random_value = [str(x) for x in random_value if x is not None]
                random_value = list(set(random_value))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            # 获取表样例（第一行）
            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库链接信息（主键和外键）'''
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例'''
        return self._table_sample[table_name]

    def check_sql(self, sql) -> tuple[bool, str]:
        '''检查sql是否合理

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        '''
        try:
            self.engine.execute(sql)
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> list[Any]:
        '''运行SQL'''
        result = self.engine.execute(sql)
        return list(result)


dbParser = DBParser("sqlite:///../code/04_SQL-Code-Agent-Demo/chinook.db")


@function_tool
def get_table_names():
    """获取数据库中所有表的名称"""
    return dbParser.table_names


@function_tool
def execute_sql(sql: str):
    """执行SQL查询并返回结果"""
    flag, err = dbParser.check_sql(sql)
    if not flag:
        return f"SQL执行错误: {err}"
    result = dbParser.execute_sql(sql)
    return result


@function_tool
def get_table_info(table_name: str):
    """获取指定表的结构信息和样例数据"""
    fields = dbParser.get_table_fields(table_name)
    sample = dbParser.get_table_sample(table_name)
    return f"表名: {table_name}\n结构:\n{fields.to_markdown()}\n样例:\n{sample.to_markdown()}"


@function_tool
def count_records(table_name: str):
    """统计指定表的记录数"""
    result = dbParser.execute_sql(f"SELECT COUNT(*) FROM {table_name}")
    return result[0][0] if result else 0


# 构建工具列表
tools = [get_table_names,execute_sql,get_table_info, count_records]

import os
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = config.OPENAI_BASE_URL
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

main_agent = Agent(
    name="sql_resolver",
    model=config.DEFAULT_MODEL,
    instructions="""
    你是一个专业的数据库专家，请根据用户的提问，结合数据库工具回答问题。
    你有以下工具可用：
    - get_table_names: 获取数据库中所有表的名称
    - execute_sql: 执行SQL查询
    - get_table_info: 获取指定表的结构信息和样例数据
    - count_records: 统计指定表的记录数

    请根据用户的问题选择合适的工具来回答。
    """,
    tools=tools
)


async def main():
    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]

    for q in questions:
        print(f"==================================\n问题: {q}")
        result = await Runner.run(main_agent, q)
        print(f"回答: {result.final_output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
