from typing import Annotated
from datetime import datetime, timedelta
from fastmcp import FastMCP

mcp = FastMCP(
    name="Meeting-MCP-Server",
    instructions="""This server contains meeting room booking and management tools for enterprise use.""",
)

# Mock meeting room database
MEETING_ROOMS = {
    "room101": {
        "name": "101会议室",
        "capacity": 10,
        "equipment": ["投影仪", "白板", "视频会议系统"],
        "location": "A栋1楼"
    },
    "room201": {
        "name": "201会议室", 
        "capacity": 20,
        "equipment": ["投影仪", "音响系统", "视频会议系统", "电子白板"],
        "location": "A栋2楼"
    },
    "room301": {
        "name": "301会议室",
        "capacity": 8,
        "equipment": ["投影仪", "电话会议系统"],
        "location": "A栋3楼"
    },
    "room401": {
        "name": "401贵宾室",
        "capacity": 6,
        "equipment": ["电视", "沙发", "茶具"],
        "location": "A栋4楼"
    }
}

# Mock booking data (simulating today's bookings)
TODAY_BOOKINGS = {
    "room101": [
        {"start_time": "09:00", "end_time": "10:00", "booker": "zhangsan", "purpose": "技术部晨会"},
        {"start_time": "14:00", "end_time": "16:00", "booker": "lisi", "purpose": "产品评审会"}
    ],
    "room201": [
        {"start_time": "10:30", "end_time": "12:00", "booker": "wangwu", "purpose": "新员工培训"},
        {"start_time": "15:00", "end_time": "17:00", "booker": "zhaoliu", "purpose": "财务分析会"}
    ],
    "room301": [
        {"start_time": "13:00", "end_time": "14:00", "booker": "zhangsan", "purpose": "项目讨论"}
    ],
    "room401": []  # No bookings today
}

@mcp.tool
def get_meeting_room_info(room_id: Annotated[str, "会议室ID"]):
    """查询会议室基本信息"""
    room = MEETING_ROOMS.get(room_id)
    if room:
        return {
            "success": True,
            "room_id": room_id,
            "data": room
        }
    else:
        return {
            "success": False,
            "message": f"未找到会议室: {room_id}",
            "available_rooms": list(MEETING_ROOMS.keys())
        }

@mcp.tool
def get_today_room_bookings(room_id: Annotated[str, "会议室ID"]):
    """查询会议室今日预定情况"""
    if room_id not in MEETING_ROOMS:
        return {
            "success": False,
            "message": f"会议室不存在: {room_id}"
        }
    
    bookings = TODAY_BOOKINGS.get(room_id, [])
    return {
        "room_id": room_id,
        "room_name": MEETING_ROOMS[room_id]["name"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "bookings_count": len(bookings),
        "bookings": bookings
    }

@mcp.tool
def check_room_availability(room_id: Annotated[str, "会议室ID"], 
                           start_time: Annotated[str, "开始时间 (HH:MM格式)"],
                           duration: Annotated[int, "会议时长(分钟)"]):
    """检查会议室在指定时间段是否可用"""
    if room_id not in MEETING_ROOMS:
        return {
            "success": False,
            "message": f"会议室不存在: {room_id}"
        }
    
    # Parse time and calculate end time
    try:
        start_dt = datetime.strptime(start_time, "%H:%M")
        end_dt = start_dt + timedelta(minutes=duration)
        end_time = end_dt.strftime("%H:%M")
    except ValueError:
        return {
            "success": False,
            "message": "时间格式错误，请使用HH:MM格式"
        }
    
    # Check for conflicts
    bookings = TODAY_BOOKINGS.get(room_id, [])
    conflict = False
    conflicting_booking = None
    
    for booking in bookings:
        booking_start = datetime.strptime(booking["start_time"], "%H:%M")
        booking_end = datetime.strptime(booking["end_time"], "%H:%M")
        
        # Check if time ranges overlap
        if not (end_dt <= booking_start or start_dt >= booking_end):
            conflict = True
            conflicting_booking = booking
            break
    
    return {
        "room_id": room_id,
        "room_name": MEETING_ROOMS[room_id]["name"],
        "requested_time": f"{start_time} - {end_time}",
        "available": not conflict,
        "conflict_info": conflicting_booking if conflict else None,
        "suggested_times": get_available_time_slots(room_id, duration) if conflict else None
    }

@mcp.tool
def get_all_meeting_rooms():
    """获取所有会议室列表"""
    rooms_with_bookings = []
    for room_id, room_info in MEETING_ROOMS.items():
        booking_count = len(TODAY_BOOKINGS.get(room_id, []))
        rooms_with_bookings.append({
            "room_id": room_id,
            **room_info,
            "today_bookings": booking_count
        })
    
    return {
        "total_rooms": len(MEETING_ROOMS),
        "rooms": rooms_with_bookings
    }

def get_available_time_slots(room_id, duration):
    """获取会议室可用的时间段建议"""
    # Simple implementation - suggest times after existing bookings
    bookings = TODAY_BOOKINGS.get(room_id, [])
    if not bookings:
        return ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]
    
    # Sort bookings by start time
    sorted_bookings = sorted(bookings, key=lambda x: x["start_time"])
    
    # Suggest times after each booking ends
    suggestions = []
    for booking in sorted_bookings:
        end_time = datetime.strptime(booking["end_time"], "%H:%M")
        suggested_time = (end_time + timedelta(minutes=30)).strftime("%H:%M")
        if suggested_time <= "17:00":  # Only suggest during working hours
            suggestions.append(suggested_time)
    
    return suggestions