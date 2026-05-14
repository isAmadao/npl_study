# 第十二周作业1：基于chinook.db的NL2SQL问答Agent
import sqlite3

# 连接数据库
def get_connection():
    return sqlite3.connect("chinook.db")

# 执行SQL语句
def execute_sql(sql):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        conn.close()
        return f"SQL执行错误：{str(e)}"

# SQL问答Agent
def nl2sql_agent(question):
    question = question.lower()
    
    # 问题1：数据库中总共有多少张表
    if "多少张表" in question:
        sql = """
        SELECT COUNT(*) FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%';
        """
        table_num = execute_sql(sql)[0][0]
        return f"数据库中总共有 {table_num} 张表"
    
    # 问题2：员工表中有多少条记录
    elif "员工表" in question and "多少条记录" in question:
        sql = "SELECT COUNT(*) FROM employees;"
        emp_count = execute_sql(sql)[0][0]
        return f"员工表中有 {emp_count} 条记录"
    
    # 问题3：客户个数和员工个数分别是多少
    elif "客户个数" in question or "员工个数" in question:
        sql_customer = "SELECT COUNT(*) FROM customers;"
        sql_employee = "SELECT COUNT(*) FROM employees;"
        customer_num = execute_sql(sql_customer)[0][0]
        employee_num = execute_sql(sql_employee)[0][0]
        return f"客户总数：{customer_num}，员工总数：{employee_num}"
    
    else:
        return "暂不支持该问题，请重新提问"

# 测试运行
if __name__ == "__main__":
    print("测试问题1：数据库中总共有多少张表")
    print(nl2sql_agent("数据库中总共有多少张表"))
    
    print("\n测试问题2：员工表中有多少条记录")
    print(nl2sql_agent("员工表中有多少条记录"))
    
    print("\n测试问题3：所有客户个数和员工个数分别是多少")
    print(nl2sql_agent("客户个数和员工个数分别是多少"))
