from typing import Annotated
import random
from fastmcp import FastMCP

mcp = FastMCP(
    name="Enterprise-Tools-MCP-Server",
    instructions="This server contains enterprise internal tools for employee assistance.",
)

EMPLOYEE_DB = {
    "张三": {"dept": "技术部", "position": "高级工程师", "email": "zhangsan@company.com", "phone": "13800138001"},
    "李四": {"dept": "市场部", "position": "市场经理", "email": "lisi@company.com", "phone": "13800138002"},
    "王五": {"dept": "财务部", "position": "财务主管", "email": "wangwu@company.com", "phone": "13800138003"},
    "赵六": {"dept": "人力资源部", "position": "HR专员", "email": "zhaoliu@company.com", "phone": "13800138004"},
    "钱七": {"dept": "技术部", "position": "产品经理", "email": "qianqi@company.com", "phone": "13800138005"},
}

MEETING_ROOMS = {
    "A101": {"name": "星光会议室", "capacity": 10, "floor": 10, "available": True},
    "A102": {"name": "月亮会议室", "capacity": 6, "floor": 10, "available": True},
    "B201": {"name": "太阳会议室", "capacity": 20, "floor": 20, "available": False},
    "B202": {"name": "云端会议室", "capacity": 8, "floor": 20, "available": True},
    "C301": {"name": "创新会议室", "capacity": 15, "floor": 30, "available": True},
}

POLICY_DB = {
    "年假": "员工入职满1年享受5天年假，满2年享受10天年假，以此类推，最多15天。申请需提前3天在OA系统提交。",
    "迟到": "每月允许迟到3次，每次不超过10分钟。超过后每次扣款50元。",
    "加班": "工作日加班可申请调休或按1.5倍工资计算，周末加班按2倍工资计算。法定节假日加班按3倍工资。",
    "报销": "普通报销需在30天内提交，差旅报销需在返回后7天内提交。单笔超过5000元需部门负责人审批。",
    "请假": "病假需提供医院证明，事假需提前申请。婚假10天，产假98天，陪产假10天。",
    "入职": "新员工入职需准备身份证、学历证明、体检报告。试用期3个月，试用期工资为正式工资的80%。",
    "离职": "员工主动离职需提前30天提交申请。离职时需完成工作交接，归还公司财物。",
    "晋升": "每年两次晋升评估，分别为4月和10月。需满足绩效达标、本职工作满2年等条件。",
}


@mcp.tool
def search_employee(
    name: Annotated[str, "员工姓名，需要查询的员工名字"]
) -> dict:
    """根据员工姓名查询员工信息，包括部门、职位、邮箱、电话等。"""
    if name in EMPLOYEE_DB:
        emp = EMPLOYEE_DB[name]
        return {
            "status": "success",
            "name": name,
            "dept": emp["dept"],
            "position": emp["position"],
            "email": emp["email"],
            "phone": emp["phone"]
        }
    else:
        similar = [k for k in EMPLOYEE_DB.keys() if name in k or k in name]
        if similar:
            return {
                "status": "not_found",
                "message": f"未找到员工 '{name}'，您是否在找：{', '.join(similar)}"
            }
        all_names = list(EMPLOYEE_DB.keys())
        return {
            "status": "not_found",
            "message": f"未找到员工 '{name}'",
            "hint": f"可用的员工有：{', '.join(all_names)}"
        }


@mcp.tool
def check_meeting_room(
    room_id: Annotated[str, "会议室编号，如 A101, B202"],
    date: Annotated[str, "查询日期，格式 YYYY-MM-DD"] = None
) -> dict:
    """查询会议室的预订情况和基本信息。"""
    if room_id not in MEETING_ROOMS:
        available_rooms = [f"{k}: {v['name']}" for k, v in MEETING_ROOMS.items()]
        return {
            "status": "error",
            "message": f"会议室 '{room_id}' 不存在",
            "available_rooms": available_rooms
        }

    room = MEETING_ROOMS[room_id]
    bookings = []
    if room_id == "A101" and date:
        bookings.append({"time": "09:00-11:00", "booked_by": "张三", "purpose": "项目评审"})
    if room_id == "B201":
        room["available"] = False
        bookings.append({"time": "14:00-16:00", "booked_by": "李四", "purpose": "客户洽谈"})

    return {
        "status": "success",
        "room_id": room_id,
        "name": room["name"],
        "capacity": room["capacity"],
        "floor": room["floor"],
        "available": room["available"],
        "date": date or "今天",
        "bookings": bookings if bookings else "暂无预订"
    }


@mcp.tool
def query_policy(
    keyword: Annotated[str, "查询关键词，如 年假、迟到、加班、报销、请假、入职、离职、晋升"]
) -> dict:
    """查询公司政策和规章制度的相关信息。"""
    keyword = keyword.strip()

    if keyword in POLICY_DB:
        return {
            "status": "success",
            "keyword": keyword,
            "content": POLICY_DB[keyword]
        }

    related = {k: v for k, v in POLICY_DB.items() if keyword.lower() in k.lower() or keyword.lower() in v.lower()}
    if related:
        results = []
        for k, v in related.items():
            results.append(f"【{k}】{v}")
        return {
            "status": "found_related",
            "message": f"未找到 '{keyword}' 的直接匹配，以下是相关内容：",
            "results": results
        }

    all_keywords = list(POLICY_DB.keys())
    return {
        "status": "not_found",
        "message": f"未找到关于 '{keyword}' 的政策信息",
        "hint": f"可查询的关键词有：{', '.join(all_keywords)}"
    }


@mcp.tool
def list_departments() -> dict:
    """获取公司所有部门列表及部门员工数量。"""
    dept_count = {}
    for emp in EMPLOYEE_DB.values():
        dept = emp["dept"]
        dept_count[dept] = dept_count.get(dept, 0) + 1

    return {
        "status": "success",
        "total_employees": len(EMPLOYEE_DB),
        "departments": [{"name": dept, "count": count} for dept, count in dept_count.items()]
    }


if __name__ == "__main__":
    mcp.run(transport="sse", port=8902)
