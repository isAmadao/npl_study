from typing import Annotated
from fastmcp import FastMCP

mcp = FastMCP(
    name="Employee-MCP-Server",
    instructions="""This server contains employee information query tools for enterprise use.""",
)

# Mock employee database
EMPLOYEE_DATABASE = {
    "zhangsan": {
        "name": "张三",
        "department": "技术部",
        "position": "高级工程师",
        "email": "zhangsan@company.com",
        "phone": "13800138001",
        "status": "在职"
    },
    "lisi": {
        "name": "李四", 
        "department": "市场部",
        "position": "市场经理",
        "email": "lisi@company.com",
        "phone": "13800138002",
        "status": "在职"
    },
    "wangwu": {
        "name": "王五",
        "department": "人事部", 
        "position": "人事专员",
        "email": "wangwu@company.com",
        "phone": "13800138003",
        "status": "在职"
    },
    "zhaoliu": {
        "name": "赵六",
        "department": "财务部",
        "position": "财务主管",
        "email": "zhaoliu@company.com", 
        "phone": "13800138004",
        "status": "在职"
    }
}

@mcp.tool
def get_employee_info(employee_id: Annotated[str, "员工ID或姓名拼音"]):
    """根据员工ID或姓名拼音查询员工详细信息"""
    employee = EMPLOYEE_DATABASE.get(employee_id.lower())
    if employee:
        return {
            "success": True,
            "data": employee
        }
    else:
        return {
            "success": False,
            "message": f"未找到员工: {employee_id}",
            "available_employees": list(EMPLOYEE_DATABASE.keys())
        }

@mcp.tool  
def search_employees_by_department(department: Annotated[str, "部门名称"]):
    """根据部门名称查询该部门所有员工"""
    department_employees = []
    for emp_id, emp_info in EMPLOYEE_DATABASE.items():
        if department in emp_info["department"]:
            department_employees.append({
                "employee_id": emp_id,
                **emp_info
            })
    
    return {
        "department": department,
        "count": len(department_employees),
        "employees": department_employees
    }

@mcp.tool
def get_all_employees():
    """获取公司所有员工列表"""
    return {
        "total_count": len(EMPLOYEE_DATABASE),
        "employees": [
            {"employee_id": emp_id, **emp_info} 
            for emp_id, emp_info in EMPLOYEE_DATABASE.items()
        ]
    }