'''数据库解析'''
import traceback
from typing import Union, Any

import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text, CursorResult


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

        # 查看表名
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        self._table_fields = {} # 数据表字段
        self.foreign_keys = [] # 数据库外键
        self._table_sample = {} # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            # print("Table ->", table_name)
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
            # self.engine.execute(sql)
            return True, 'ok'
        except Exception as e:
            print(e)
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> list:
        '''运行SQL'''
        with self.engine.connect() as conn:
            result =  conn.execute(text(sql))
            return list(result)
parser = DBParser('sqlite:///./chinook.db')
def ask_glm(question, nretry=5):
    if nretry == 0:
        return None
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key="sk-685d5da74c2047dbb7d80c0e80fcb05d",

        # 大模型厂商的地址
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        # 调用
        completion = client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model="qwen-plus",
            messages=[
                {"role": "user", "content": question},
            ]
        )
        return completion.choices[0].message.content
    except:
        return ask_glm(question, nretry-1)

def get_table_info() -> str:
    table_all_info = ""
    prompt = '''
            表名称：{table_name}
            数据表样例如下：
            {data_sample_mk}
            数据表schema如下：
            {data_schema}
            
            '''
    for table_name in parser.table_names[:50]:
        # 表样例
        data_sample = parser.get_table_sample(table_name)
        data_sample_mk = data_sample.to_markdown()

        # 表格式
        data_schema = parser.get_table_fields(table_name).to_markdown()
        # data_fields = list(data_sample.columns)
        table_all_info += prompt.format(table_name=table_name, data_sample_mk=data_sample_mk,data_schema=data_schema)

    return table_all_info
answer_prompt = '''你是一个专业的数据库专家，现在需要你结合数据库的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：
提问内容是{question}''' + get_table_info()

answer_rewrite_prompt = '''你是一个专业的数据库专家，将下面的问题回答组织为自然语言。：

原始问题：{question}

执行SQL：{sql}

原始结果：{answer}
'''

gt_qes_answer = []

def get_answer(quest : str) -> str:
    print("提问："+quest)
    # 重试次数
    for _ in range(5):
        # 生成提问
        try:
            # 生成答案SQL
            input_str = answer_prompt.format(question=quest)

            answer = ask_glm(input_str)

            # 判断SQL是否符合逻辑
            flag, _ = parser.check_sql(answer)
            if not flag:
                continue

            # 获取SQL答案
            sql_answer = parser.execute_sql(answer)
            if len(sql_answer) > 1:
                continue
            sql_answer = sql_answer[0]
            sql_answer = ' '.join([str(x) for x in sql_answer])

            # 将SQL和结果改为为自然语言
            input_str = answer_rewrite_prompt.format(question=quest, sql=answer, answer=sql_answer)
            nl_answer = ask_glm(input_str)
            return nl_answer
        except Exception as e:
            print(e)
            continue
print(get_answer("数据库中总共有多少张表"))
print(get_answer("员工表中有多少条记录"))
print(get_answer("在数据库中所有客户个数和员工个数分别是多少"))