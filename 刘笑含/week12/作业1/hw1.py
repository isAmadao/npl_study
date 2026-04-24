import os
import json
import anthropic
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text
import pandas as pd


class DBParser:
    '''DBParser'''
    def __init__(self, db_url:str) -> None:
        '''初始化
        db_url: 数据库链接地址
        '''

        # 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 链接数据库
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        # 查看表明
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        self._table_fields = {} # 数据表字段
        self.foreign_keys = [] # 数据库外键
        self._table_sample = {} # 数据表样例
        self._table_records = {} #每个表的记录个数

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
            self._table_fields[table_name] = {x['name']:x for x in table_columns}

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


    def check_sql(self, sql) -> Union[bool, str]:
        '''检查sql是否合理

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        '''
        try:
            with self.engine.connect() as conn:
                conn.execute(text(sql))
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            return list(result)



answer_prompt = """You are a SQL expert working with a SQLite database. Given the database information below, generate a SQLite-compatible SQL query to answer the question. Output only the SQL query with no explanation or markdown.

All tables in the database ({table_count} total): {all_tables}

Table sample data:
{data_sample_md}

Table schema:
{data_schema}

Question: {question}
"""


answer_rewrite_prompt = """You are a data analyst. Rewrite the SQL query result as a clear, concise natural language answer.

Question: {question}
SQL executed: {sql}
Raw result: {answer}

Provide only the final answer in plain English.
"""

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '04_SQL-Code-Agent-Demo', 'chinook.db')
parser = DBParser(f'sqlite:///{DB_PATH}')
client = anthropic.Anthropic()


def ask_llm(prompt:str) -> str:
  message = client.messages.create(
    model = 'claude-haiku-4-5-20251001',
    max_tokens = 1024,
    messages = [{
      "role":'user',
      "content": prompt
    }]
  )
  return message.content[0].text.strip()


def get_answer(question:str, table_names:list) -> str:
  samples = []
  schemas = []
  for table in table_names[:50]:
    data_sample = parser.get_table_sample(table)
    data_fields = parser.get_table_fields(table)
    samples.append(f"[{table}]\n{data_sample.to_markdown(index=False)}")
    schemas.append(f"[{table}]\n{data_fields[['name', 'type']].to_markdown(index=False)}")
    data_sample_md = '\n\n'.join(samples)
    data_schema = '\n\n'.join(schemas)

    prompt= answer_prompt.format(
      table_count=len(parser.table_names),
      all_tables = ','.join(parser.table_names),
      data_sample_md = data_sample_md,
      data_schema = data_schema,
      question= question
    )

    sql = ask_llm(prompt)
    print(f"Generated SQL: {sql}")

    ok, err = parser.check_sql(sql)
    if not ok:
      return f'validation failed: {err}'

    raw_output = parser.execute_sql(sql)
    print(f"Raw output: {raw_output}")

    rewrite_prompt = answer_rewrite_prompt.format(
      question = question,
      sql = sql,
      answer = raw_output
    )

    final_answer = ask_llm(rewrite_prompt)
    return final_answer


if __name__ == '__main__':
    table_map = {t.lower(): t for t in parser.table_names}
    employees_tbl = table_map.get('employees', 'employees')
    customers_tbl = table_map.get('customers', 'customers')


    cases = [
        ('How many tables are in the database?', parser.table_names),
        (f'How many records are in the {employees_tbl} table?', [employees_tbl]),
        (f'How many customers and employees are there respectively?', [customers_tbl, employees_tbl]),
    ]

    for i, (question, tables) in enumerate(cases):
        print(f"\n{'='*60}")
        print(f"Q{i}: {question}")
        answer = get_answer(question, tables)
        print(f"Answer: {answer}")
