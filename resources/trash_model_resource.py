"""
Trash Model Resource (resources/trash_model_resource.py)
========================================================
MCP Resource — YOLO 模型权重资源

返回模型权重路径、类别配置、模型元数据等信息。
"""

import os
from typing import Any, Dict

from resources import YA_MCPServer_Resource


@YA_MCPServer_Resource(
    "config://models/primary",
    name="primary_model_info",
    title="主模型信息 (YOLO11)",
    description="返回 YOLO11 主模型的权重路径、类别列表及元数据",
)
async def get_primary_model_info() -> Dict[str, Any]:
    """
    获取主模型（YOLO11）信息。

    Returns:
        Dict: 包含权重路径、类别数、类别列表
    """
    try:
        from core.trash_detection import (
            WASTE_CATEGORIES, WASTE_CN, _get_weight_path,
        )

        weight_path = _get_weight_path("models.primary_weight", "best_trash_detector.pt")

        return {
            "model_name": "YOLO11-trash",
            "weight_path": weight_path,
            "weight_exists": os.path.exists(weight_path),
            "num_classes": len(WASTE_CATEGORIES),
            "categories": WASTE_CATEGORIES,
            "categories_cn": {k: WASTE_CN[k] for k in WASTE_CATEGORIES},
        }
    except Exception as e:
        return {"error": str(e)}


@YA_MCPServer_Resource(
    "config://models/secondary",
    name="secondary_model_info",
    title="辅助模型信息 (YOLOv8)",
    description="返回 YOLOv8 辅助模型的权重路径、类别列表及元数据",
)
async def get_secondary_model_info() -> Dict[str, Any]:
    """
    获取辅助模型（YOLOv8 18类）信息。

    Returns:
        Dict: 包含权重路径、类别数、类别列表
    """
    try:
        from core.trash_detection import YOLOV8_CLASS_NAMES, _get_weight_path

        weight_path = _get_weight_path("models.secondary_weight", "best_yolov8_trash.pt")

        return {
            "model_name": "YOLOv8-trash-18",
            "weight_path": weight_path,
            "weight_exists": os.path.exists(weight_path),
            "num_classes": len(YOLOV8_CLASS_NAMES),
            "categories": YOLOV8_CLASS_NAMES,
        }
    except Exception as e:
        return {"error": str(e)}


@YA_MCPServer_Resource(
    "config://models/all",
    name="all_models_info",
    title="全部模型信息",
    description="返回所有 YOLO 检测模型的概要信息",
)
async def get_all_models_info() -> Dict[str, Any]:
    """
    获取全部模型的概要信息。

    Returns:
        Dict: 包含所有模型的元信息
    """
    try:
        primary = await get_primary_model_info()
        secondary = await get_secondary_model_info()
        return {
            "primary": primary,
            "secondary": secondary,
            "architecture": "dual-model fusion (YOLO11 + YOLOv8)",
        }
    except Exception as e:
        return {"error": str(e)}
