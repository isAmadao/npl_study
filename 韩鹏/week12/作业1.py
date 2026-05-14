"""
作业1: 参考sql agent，实现一下基于 chinook.db 数据集进行问答agent（nl2sql），需要能回答如下提问：
• 提问1: 数据库中总共有多少张表；
• 提问2: 员工表中有多少条记录
• 提问3: 在数据库中所有客户个数和员工个数分别是多少
"""
import sqlite3

# 连接到Chinook数据库
conn = sqlite3.connect('chinook.db')

# 创建一个游标对象
cursor = conn.cursor()

# 数据库中总共有多少张表
cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table';")
result = cursor.fetchone()[0]  # 获取第一行第一列的值
print(f"表数量: {result}")

# 查询员工记录条数/人数
cursor.execute("SELECT count(*) FROM employees ;")
result = cursor.fetchone()[0]  # 获取第一行第一列的值
print(f"员工记录条数/人数: {result}")

# 查询客户个数
cursor.execute("SELECT count(*) FROM customers ;")
result = cursor.fetchone()[0]  # 获取第一行第一列的值
print(f"客户个数: {result}")


