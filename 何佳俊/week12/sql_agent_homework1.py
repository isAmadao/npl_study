"""
基于Chinook数据库的SQL问答Agent
参考sql-agent.ipynb实现
"""
import os
import traceback
from typing import Union

import pandas as pd
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData


class DBParser:
    """数据库解析器"""
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

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        """
        try:
            with self.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text(sql))
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql):
        """运行SQL"""
        from sqlalchemy import text
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            return result.fetchall()


class SQLAgent:
    """SQL问答Agent"""
    
    def __init__(self, db_path: str):
        """初始化SQL Agent
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_parser = DBParser(f'sqlite:///{db_path}')
        
    def answer_question_1(self) -> dict:
        """问题1: 数据库中总共多少张表"""
        table_count = len(self.db_parser.table_names)
        table_list = self.db_parser.table_names
        
        return {
            "question": "数据库中总共多少张表",
            "answer": f"数据库中共有 {table_count} 张表",
            "details": {
                "table_count": table_count,
                "table_names": table_list
            }
        }
    
    def answer_question_2(self) -> dict:
        """问题2: 员工表中总共多少条记录"""
        sql = "SELECT COUNT(*) FROM employees"
        result = self.db_parser.execute_sql(sql)
        employee_count = result[0][0]
        
        return {
            "question": "员工表中总共多少条记录",
            "answer": f"员工表中共有 {employee_count} 条记录",
            "details": {
                "sql": sql,
                "employee_count": employee_count
            }
        }
    
    def answer_question_3(self) -> dict:
        """问题3: 在数据库中所有客户个数和员工个数是多少？"""
        # 查询客户数量
        customer_sql = "SELECT COUNT(*) FROM customers"
        customer_result = self.db_parser.execute_sql(customer_sql)
        customer_count = customer_result[0][0]
        
        # 查询员工数量
        employee_sql = "SELECT COUNT(*) FROM employees"
        employee_result = self.db_parser.execute_sql(employee_sql)
        employee_count = employee_result[0][0]
        
        # 计算总数
        total_count = customer_count + employee_count
        
        return {
            "question": "在数据库中所有客户个数和员工个数是多少？",
            "answer": f"数据库中共有 {total_count} 个客户和员工，其中客户 {customer_count} 个，员工 {employee_count} 个",
            "details": {
                "customer_sql": customer_sql,
                "employee_sql": employee_sql,
                "customer_count": customer_count,
                "employee_count": employee_count,
                "total_count": total_count
            }
        }
    
    def run_all_questions(self):
        """运行所有问题并输出结果"""
        print("=" * 80)
        print("Chinook数据库 SQL Agent 问答系统")
        print("=" * 80)
        print()
        
        # 问题1
        print("【问题1】数据库中总共多少张表")
        result1 = self.answer_question_1()
        print(f"答案: {result1['answer']}")
        print(f"表名列表: {', '.join(result1['details']['table_names'])}")
        print()
        
        # 问题2
        print("【问题2】员工表中总共多少条记录")
        result2 = self.answer_question_2()
        print(f"答案: {result2['answer']}")
        print(f"执行SQL: {result2['details']['sql']}")
        print()
        
        # 问题3
        print("【问题3】在数据库中所有客户个数和员工个数是多少？")
        result3 = self.answer_question_3()
        print(f"答案: {result3['answer']}")
        print(f"客户SQL: {result3['details']['customer_sql']}")
        print(f"员工SQL: {result3['details']['employee_sql']}")
        print()
        
        print("=" * 80)
        print("所有问题回答完成！")
        print("=" * 80)
        
        return [result1, result2, result3]


def main():
    """主函数"""
    # 获取数据库文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, '..', '04_SQL-Code-Agent-Demo', 'chinook.db')
    
    # 创建SQL Agent
    agent = SQLAgent(db_path)
    
    # 运行所有问题
    results = agent.run_all_questions()
    
    return results


if __name__ == "__main__":
    main()
