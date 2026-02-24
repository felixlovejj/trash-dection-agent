"""
Detect Trash Tool (tools/detect_trash_tool.py)
===============================================
MCP Tool — 废品识别工具

输入：图片路径
输出：双模型检测结果（主模型 5 大类 + 辅助模型 18 细分类）
"""

import json
from typing import Any, Dict

from tools import YA_MCPServer_Tool


@YA_MCPServer_Tool(
    name="detect_waste_type",
    title="废品识别工具",
    description="使用 YOLO11 + YOLOv8 双模型对垃圾图片进行检测，返回分类结果与置信度",
)
async def detect_waste_type(image_path: str) -> Dict[str, Any]:
    """
    使用双模型检测图片中的垃圾类型。

    Args:
        image_path (str): 待检测图片的文件路径

    Returns:
        Dict[str, Any]: 标准化检测结果
            {
                "primary":    {"class_name": str, "confidence": float},
                "secondary":  {"class_name": str, "confidence": float},
                "algorithms": list[str],
                "error":      str
            }

    Example:
        {
            "primary":   {"class_name": "plastic", "confidence": 0.95},
            "secondary": {"class_name": "Bottle", "confidence": 0.82},
            "algorithms": ["YOLO11-trash", "YOLOv8-trash-18"],
            "error": ""
        }
    """
    try:
        from core.trash_detection import detect_trash

        result = await detect_trash(image_path)
        return result

    except Exception as e:
        return {
            "primary":    {"class_name": "unknown", "confidence": 0.0},
            "secondary":  {"class_name": "unknown", "confidence": 0.0},
            "algorithms": ["YOLO11-trash", "YOLOv8-trash-18"],
            "error":      f"检测工具异常: {e}",
        }
