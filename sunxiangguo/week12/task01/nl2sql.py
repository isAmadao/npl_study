import sqlite3
from typing import Union, List, Dict, Any
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text
import pandas as pd
from openai import OpenAI


class DBParser:
    """数据库解析器 - 用于获取数据库结构和元数据"""

    def __init__(self, db_url: str) -> None:
        """初始化
        db_url: 数据库链接地址
        """
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

        self._table_fields = {}  # 数据表字段
        self.foreign_keys = []  # 数据库外键
        self._table_sample = {}  # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print(f"正在解析表: {table_name}")
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
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计unique
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                # 统计most frequency value
                field_type = str(self._table_fields[table_name][column_meta['name']]['type'])
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
        """获取表字段信息"""
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        """获取数据库链接信息（主键和外键）"""
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        """获取数据表样例"""
        return self._table_sample[table_name]

    def check_sql(self, sql) -> Union[bool, str]:
        """检查sql是否合理

        参数:
            sql: 待执行句子

        返回:
            (是否可以运行, 报错信息)
        """
        try:
            self.conn.execute(text(sql))
            return True, 'ok'
        except Exception as e:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> List:
        """运行SQL"""
        result = self.conn.execute(text(sql))
        return list(result)

    def get_database_schema(self) -> str:
        """获取完整的数据库schema信息"""
        schema_info = []
        schema_info.append(f"数据库中共有 {len(self.table_names)} 张表\n")
        schema_info.append(f"表名列表: {', '.join(self.table_names)}\n")

        for table_name in self.table_names:
            schema_info.append(f"\n=== 表: {table_name} ===")
            fields_df = self.get_table_fields(table_name)
            schema_info.append(fields_df[['name', 'type']].to_string(index=False))

            sample_df = self.get_table_sample(table_name)
            if not sample_df.empty:
                schema_info.append(f"\n样例数据:")
                schema_info.append(sample_df.to_string(index=False))

        return '\n'.join(schema_info)


class NL2SQLAgent:
    """自然语言到SQL的问答Agent"""

    def __init__(self, db_parser: DBParser, api_key: str = "sk-Q1eBhuOPH5WuV5u5SsrTnjdKHpg7skKVNN5ZoexvAkNiAV5a",
                 base_url: str = "https://clawapi.vip/v1", model: str = "qwen-turbo"):
        """
        初始化Agent

        参数:
            db_parser: 数据库解析器实例
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型名称
        """
        self.db_parser = db_parser
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        schema_info = self.db_parser.get_database_schema()

        prompt = f"""你是一个专业的SQL数据库专家助手。你的任务是将用户的自然语言问题转换为准确的SQL查询语句，并执行查询返回结果。

数据库Schema信息：
{schema_info}

请遵循以下规则：
1. 只使用上述提到的表和字段
2. SQLite语法，注意表名和字段名的大小写
3. 如果用户问的是计数问题，使用COUNT(*)或COUNT(字段名)
4. 如果涉及多表查询，注意使用正确的JOIN条件
5. 直接输出SQL语句，不要包含其他解释
6. 确保SQL语句可以被正确执行

当收到用户问题时：
1. 首先分析用户意图
2. 生成对应的SQL查询
3. 执行SQL并返回结果
4. 将结果用自然语言回答用户"""

        return prompt

    def ask_llm(self, question: str, max_retries: int = 3) -> str:
        """调用大语言模型

        参数:
            question: 用户问题
            max_retries: 最大重试次数

        返回:
            LLM的回答
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.3,
                    timeout=30
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"第{attempt + 1}次尝试失败: {e}")
                if attempt == max_retries - 1:
                    raise
        return ""

    def extract_sql(self, llm_response: str) -> str:
        """从LLM响应中提取SQL语句"""
        # 清理可能的markdown格式
        sql = llm_response.strip()
        sql = sql.replace('```sql', '').replace('```', '')
        sql = sql.strip()

        # 如果包含解释，尝试提取SQL部分
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_stripped = line.strip()
            # 检测SQL开始
            if not in_sql and line_stripped.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
                in_sql = True
                sql_lines.append(line_stripped)
            elif in_sql:
                # SQL continuation - 收集直到遇到空行或非SQL内容
                if line_stripped == '':
                    break
                sql_lines.append(line_stripped)
        
        if sql_lines:
            return ' '.join(sql_lines)
        
        # 如果没有找到SQL关键字，返回清理后的全部内容
        return sql

    def answer_question(self, question: str) -> str:
        """
        回答用户问题

        参数:
            question: 用户的自然语言问题

        返回:
            回答结果
        """
        print(f"\n{'='*60}")
        print(f"用户问题: {question}")
        print(f"{'='*60}")

        # 第一步：让LLM生成SQL
        print("\n[步骤1] 正在生成SQL查询...")
        sql_prompt = f"请将以下问题转换为SQL查询语句：\n\n问题: {question}\n\nSQL:"
        llm_response = self.ask_llm(sql_prompt)
        print(f"LLM原始响应:\n{llm_response}")

        # 第二步：提取SQL
        sql_query = self.extract_sql(llm_response)
        print(f"\n[步骤2] 提取的SQL:\n{sql_query}")

        # 第三步：验证SQL
        print("\n[步骤3] 验证SQL语法...")
        is_valid, error_msg = self.db_parser.check_sql(sql_query)
        if not is_valid:
            print(f"SQL验证失败: {error_msg}")
            return f"抱歉，生成的SQL查询有误：{error_msg}"

        print("SQL验证通过 ✓")

        # 第四步：执行SQL
        print("\n[步骤4] 执行SQL查询...")
        try:
            result = self.db_parser.execute_sql(sql_query)
            print(f"查询结果: {result}")
        except Exception as e:
            print(f"SQL执行失败: {e}")
            return f"抱歉，执行查询时出错：{str(e)}"

        # 第五步：将结果转换为自然语言
        print("\n[步骤5] 生成自然语言回答...")
        answer_prompt = f"""问题: {question}
SQL查询: {sql_query}
查询结果: {result}

请用简洁的自然语言回答用户的问题，直接给出答案。"""

        final_answer = self.ask_llm(answer_prompt)
        print(f"\n最终回答:\n{final_answer}")

        return final_answer


def main():
    """主函数 - 演示问答Agent"""
    print("=" * 60)
    print("Chinook 数据库 NL2SQL 问答 Agent")
    print("=" * 60)

    # 初始化数据库解析器
    print("\n正在初始化数据库解析器...")
    parser = DBParser('sqlite:///./chinook.db')
    print(f"\n数据库加载完成！共发现 {len(parser.table_names)} 张表")
    print(f"表名: {parser.table_names}")

    # 初始化问答Agent
    agent = NL2SQLAgent(
        db_parser=parser,
        api_key="sk-Q1eBhuOPH5WuV5u5SsrTnjdKHpg7skKVNN5ZoexvAkNiAV5a",  # 替换为你的API key
        base_url="https://clawapi.vip/v1",
        model="qwen-turbo"
    )

    # 测试问题列表
    test_questions = [
        "数据库中存在多少张表？",
        "员工表中有多少记录？",
        "数据库中所有客户个数和员工个数分别是多少？"
    ]

    # 逐个回答问题
    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"问题 {i}/{len(test_questions)}")
        print(f"{'#'*60}")

        try:
            answer = agent.answer_question(question)
            print(f"\n✅ 回答完成")
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")

    print(f"\n\n{'='*60}")
    print("所有问题处理完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
