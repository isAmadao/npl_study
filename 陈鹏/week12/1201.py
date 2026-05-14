import time
import requests
from sqlalchemy import create_engine, inspect, text

"""
参考sql agent，实现一下基于 chinook.db 数据集进行问答agent（nl2sql），需要能回答如下提问：
• 提问1: 数据库中总共有多少张表；
• 提问2: 员工表中有多少条记录
• 提问3: 在数据库中所有客户个数和员工个数分别是多少
"""


# ===================== 1. 通义千问 GLM-5.1 调用 =====================
def ask_glm5(question, api_key, nretry=5):
    if nretry == 0:
        return None

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }

    data = {
        "model": "glm-5.1",
        "p": 0.5,
        "messages": [{"role": "user", "content": question}]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.json()
    except:
        time.sleep(0.5)
        return ask_glm5(question, api_key, nretry - 1)


# ===================== 2. 数据库解析器 =====================
class DBParser:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.conn = self.engine.connect()
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

    def get_table_info(self, table_name):
        return self.inspector.get_columns(table_name)

    def execute_sql(self, sql):
        try:
            result = self.conn.execute(text(sql))
            data = result.fetchall()
            return data
        except:
            return None


# ===================== 3. NL2SQL 提示词 =====================
SQL_PROMPT = '''
你是专业数据库专家，根据用户问题生成正确SQL。
数据库类型：SQLite
所有表：{tables}

根据下面表结构生成SQL，只输出SQL，不要任何其他文字。
表结构：
{schema}

用户问题：{question}
'''

ANSWER_PROMPT = '''
根据问题、SQL、结果，用自然语言回答。
问题：{question}
结果：{result}
回答：
'''


# ===================== 4. 智能问答 Agent =====================
class NL2SQLAgent:
    def __init__(self, db_path, api_key):
        self.db = DBParser(f"sqlite:///{db_path}")
        self.api_key = api_key

    def get_all_schema(self):
        schemas = []
        for t in self.db.table_names:
            fields = self.db.get_table_info(t)
            sch_str = f"表:{t}\n" + "\n".join([f"{f['name']} {str(f['type'])}" for f in fields])
            schemas.append(sch_str)
        return "\n\n".join(schemas)

    def ask(self, question):
        print(f"🤖 问题：{question}")

        # 1. 获取表结构
        all_tables = self.db.table_names
        all_schema = self.get_all_schema()

        # 2. 生成 SQL
        prompt = SQL_PROMPT.format(
            tables=all_tables,
            schema=all_schema,
            question=question
        )
        resp = ask_glm5(prompt, self.api_key)
        sql = resp['choices'][0]['message']['content'].strip()
        sql = sql.strip('`').strip('\n').replace('sql\n', '')
        print(f"📝 生成SQL：{sql}")

        # 3. 执行 SQL
        result = self.db.execute_sql(sql)
        print(f"📊 查询结果：{result}")

        # 4. 生成自然语言回答
        resp2 = ask_glm5(
            ANSWER_PROMPT.format(question=question, result=result),
            self.api_key
        )
        answer = resp2['choices'][0]['message']['content']
        print(f"✅ 最终回答：{answer}\n")
        return answer


# ===================== 5. 测试：回答你指定的 3 个问题 =====================
if __name__ == "__main__":
    # 填入你的通义千问 API KEY
    API_KEY = "sk-399b434c3f5b4329a4600ec76ce4f7cc"
    DB_PATH = "chinook.db"

    # 创建 Agent
    agent = NL2SQLAgent(DB_PATH, API_KEY)

    # 测试 3 个问题
    agent.ask("数据库中总共有多少张表?")
    agent.ask("员工表中有多少条记录?")
    agent.ask("在数据库中所有客户个数和员工个数分别是多少?")
