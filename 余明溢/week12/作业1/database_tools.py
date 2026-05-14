import traceback

from db_parser import DBParser


class DatabaseTools:
    """将 DBParser 的方法包装为 Agent 可调用的工具"""

    def __init__(self, db_parser: DBParser):
        self.db_parser = db_parser

    def get_tables_info(self) -> str:
        """返回所有表名及字段概要（用于 NL2SQL 提示）"""
        info = "数据库表结构：\n"
        for table in self.db_parser.table_names:
            fields = self.db_parser.get_table_fields(table)
            info += f"表 {table}: 字段 {', '.join(fields.index)}\n"
        return info

    def run_sql(self, sql: str) -> str:
        """执行 SQL 并返回格式化的结果"""
        try:
            rows = self.db_parser.execute_sql(sql)
            if not rows:
                return "查询无结果。"
            # 取前 20 行展示
            output = "\n".join(str(row) for row in rows[:20])
            if len(rows) > 20:
                output += f"\n... 共 {len(rows)} 行，仅显示前20行。"
            return output
        except Exception as e:
            return f"SQL 执行错误：{e}\n{traceback.format_exc()}"
