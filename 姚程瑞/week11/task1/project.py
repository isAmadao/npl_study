from typing import Annotated
from fastmcp import FastMCP

mcp = FastMCP(
    name="Project-MCP-Server", 
    instructions="""This server contains project management and tracking tools for enterprise use.""",
)

# Mock project database
PROJECT_DATABASE = {
    "project001": {
        "name": "企业ERP系统升级",
        "status": "进行中",
        "progress": 65,
        "start_date": "2024-01-15",
        "end_date": "2024-06-30",
        "manager": "zhangsan",
        "department": "技术部",
        "priority": "高",
        "description": "升级现有ERP系统，提升性能和功能"
    },
    "project002": {
        "name": "新产品市场推广",
        "status": "已完成", 
        "progress": 100,
        "start_date": "2024-02-01",
        "end_date": "2024-04-15",
        "manager": "lisi",
        "department": "市场部",
        "priority": "中",
        "description": "新产品上市推广活动"
    },
    "project003": {
        "name": "员工培训计划",
        "status": "规划中",
        "progress": 20,
        "start_date": "2024-05-01", 
        "end_date": "2024-12-31",
        "manager": "wangwu",
        "department": "人事部",
        "priority": "中",
        "description": "年度员工技能培训计划"
    },
    "project004": {
        "name": "财务系统优化",
        "status": "进行中",
        "progress": 80,
        "start_date": "2024-03-10",
        "end_date": "2024-05-20",
        "manager": "zhaoliu",
        "department": "财务部", 
        "priority": "高",
        "description": "财务报销流程优化"
    }
}

@mcp.tool
def get_project_info(project_id: Annotated[str, "项目ID"]):
    """根据项目ID查询项目详细信息"""
    project = PROJECT_DATABASE.get(project_id)
    if project:
        return {
            "success": True,
            "project_id": project_id,
            "data": project
        }
    else:
        return {
            "success": False,
            "message": f"未找到项目: {project_id}",
            "available_projects": list(PROJECT_DATABASE.keys())
        }

@mcp.tool
def get_projects_by_status(status: Annotated[str, "项目状态: 进行中/已完成/规划中"]):
    """根据状态筛选项目列表"""
    filtered_projects = []
    for project_id, project_info in PROJECT_DATABASE.items():
        if project_info["status"] == status:
            filtered_projects.append({
                "project_id": project_id,
                **project_info
            })
    
    return {
        "status": status,
        "count": len(filtered_projects),
        "projects": filtered_projects
    }

@mcp.tool
def get_all_projects():
    """获取所有项目列表"""
    return {
        "total_count": len(PROJECT_DATABASE),
        "projects": [
            {"project_id": project_id, **project_info}
            for project_id, project_info in PROJECT_DATABASE.items()
        ]
    }

@mcp.tool
def get_projects_by_department(department: Annotated[str, "部门名称"]):
    """根据部门查询相关项目"""
    department_projects = []
    for project_id, project_info in PROJECT_DATABASE.items():
        if department in project_info["department"]:
            department_projects.append({
                "project_id": project_id,
                **project_info
            })
    
    return {
        "department": department,
        "count": len(department_projects),
        "projects": department_projects
    }