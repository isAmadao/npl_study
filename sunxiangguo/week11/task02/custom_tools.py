from datetime import datetime
import math
from typing import Annotated, Union
from fastmcp import FastMCP

mcp = FastMCP(
    name="Custom-Tools-MCP-Server",
    instructions="""This server contains custom utility tools for calculation, time, and unit conversion.""",
)

@mcp.tool
def calculate_expression(expression: Annotated[str, "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(30)')"]):
    """Evaluates a mathematical expression and returns the result. Supports basic arithmetic, sqrt, sin, cos, tan, log, etc."""
    try:
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sqrt': math.sqrt, 'pow': pow,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'log': math.log, 'log10': math.log10, 'log2': math.log2,
            'exp': math.exp, 'pi': math.pi, 'e': math.e,
            'factorial': math.factorial, 'ceil': math.ceil, 'floor': math.floor
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@mcp.tool
def get_current_datetime(format_type: Annotated[str, "Format type: 'date' (YYYY-MM-DD), 'time' (HH:MM:SS), 'datetime' (full), or 'weekday' (day of week)"] = "datetime"):
    """Retrieves the current date and/or time in various formats."""
    try:
        now = datetime.now()
        weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
        
        if format_type == "date":
            return f"当前日期: {now.strftime('%Y年%m月%d日')}"
        elif format_type == "time":
            return f"当前时间: {now.strftime('%H时%M分%S秒')}"
        elif format_type == "weekday":
            return f"今天是: {weekdays[now.weekday()]}"
        elif format_type == "datetime":
            return f"当前时间: {now.strftime('%Y年%m月%d日 %H时%M分%S秒')} {weekdays[now.weekday()]}"
        else:
            return f"当前完整时间: {now.isoformat()}"
    except Exception as e:
        return f"获取时间失败: {str(e)}"

@mcp.tool
def convert_units(value: Annotated[Union[int, float], "The numeric value to convert"], 
                  from_unit: Annotated[str, "Source unit (e.g., 'km', 'm', 'cm', 'kg', 'g', 'lb', 'mile', 'foot')"], 
                  to_unit: Annotated[str, "Target unit (e.g., 'km', 'm', 'cm', 'kg', 'g', 'lb', 'mile', 'foot')"]):
    """Converts a value from one unit to another. Supports length (km, m, cm, mm, mile, foot, inch) and weight (kg, g, lb, oz)."""
    try:
        length_to_meters = {
            'km': 1000, 'm': 1, 'cm': 0.01, 'mm': 0.001,
            'mile': 1609.344, 'foot': 0.3048, 'ft': 0.3048,
            'inch': 0.0254, 'in': 0.0254, 'yard': 0.9144, 'yd': 0.9144
        }
        
        weight_to_grams = {
            'kg': 1000, 'g': 1, 'mg': 0.001,
            'lb': 453.592, 'pound': 453.592,
            'oz': 28.3495, 'ounce': 28.3495
        }
        
        from_unit_lower = from_unit.lower().strip()
        to_unit_lower = to_unit.lower().strip()
        
        if from_unit_lower in length_to_meters and to_unit_lower in length_to_meters:
            base_value = value * length_to_meters[from_unit_lower]
            result = base_value / length_to_meters[to_unit_lower]
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        
        elif from_unit_lower in weight_to_grams and to_unit_lower in weight_to_grams:
            base_value = value * weight_to_grams[from_unit_lower]
            result = base_value / weight_to_grams[to_unit_lower]
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        
        else:
            supported_length = ', '.join(length_to_meters.keys())
            supported_weight = ', '.join(weight_to_grams.keys())
            return f"不支持的单位转换。支持的长度单位: {supported_length}; 支持的重量单位: {supported_weight}"
    
    except Exception as e:
        return f"单位转换错误: {str(e)}"
