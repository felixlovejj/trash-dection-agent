"""
Trash Detection Core (core/trash_detection.py)
===============================================
双模型（YOLO11 + YOLOv8）垃圾检测核心逻辑。

- 主模型  : YOLO11 (best_trash_detector.pt)  — 5 大类
- 辅助模型: YOLOv8 (best_yolov8_trash.pt)    — 18 细分类

所有函数均支持 async/await 异步调用，并包含 try-except 异常处理。
"""

import os
import asyncio
from typing import Any, Dict, Optional

from modules.YA_Common.utils.logger import get_logger
from modules.YA_Common.utils.config import get_config

logger = get_logger("core.trash_detection")

# ── 主模型（YOLO11）5 大类 ────────────────────────────────
WASTE_CATEGORIES = ["glass", "metal", "paper", "plastic", "waste"]

# ── 辅助模型（YOLOv8）18 细分类 ──────────────────────────
YOLOV8_CLASS_NAMES = [
    "Aluminium foil", "Bottle cap", "Bottle", "Broken glass",
    "Can", "Carton", "Cigarette", "Cup", "Lid",
    "Other litter", "Other plastic", "Paper",
    "Plastic bag - wrapper", "Plastic container", "Pop tab",
    "Straw", "Styrofoam piece", "Unlabeled litter",
]

# 中英对照
WASTE_CN = {
    "glass": "玻璃", "metal": "金属", "paper": "纸张",
    "plastic": "塑料", "waste": "其他垃圾", "unknown": "未知",
}

# 主模型类名映射
_PRIMARY_CLASS_MAP = {
    "Glass": "glass", "Metal": "metal", "Paper": "paper",
    "Plastic": "plastic", "Waste": "waste",
}

CONFIDENCE_THRESHOLD = 0.25

# ── 模型权重路径（从 config.yaml 或默认路径获取） ─────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_weight_path(key: str, default_name: str) -> str:
    """获取模型权重文件的绝对路径。"""
    configured = get_config(key, None)
    if configured:
        path = os.path.join(_PROJECT_ROOT, configured)
    else:
        path = os.path.join(_PROJECT_ROOT, default_name)
    return os.path.abspath(path)


# ═══════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════

_primary_model = None
_secondary_model = None


async def load_primary_model():
    """
    异步加载主模型 YOLO11。

    Returns:
        YOLO: 主模型实例

    Raises:
        FileNotFoundError: 权重文件不存在
        RuntimeError: 模型加载失败
    """
    global _primary_model
    if _primary_model is not None:
        return _primary_model

    try:
        from ultralytics import YOLO

        weight_path = _get_weight_path("models.primary_weight", "best_trash_detector.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"YOLO11 主模型权重不存在: {weight_path}")

        logger.info(f"加载主模型: {weight_path}")
        _primary_model = await asyncio.to_thread(YOLO, weight_path)
        logger.info("主模型加载完成")
        return _primary_model

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"主模型加载失败: {e}")
        raise RuntimeError(f"主模型加载失败: {e}") from e


async def load_secondary_model():
    """
    异步加载辅助模型 YOLOv8 (18 类)。

    Returns:
        YOLO: 辅助模型实例

    Raises:
        FileNotFoundError: 权重文件不存在
        RuntimeError: 模型加载失败
    """
    global _secondary_model
    if _secondary_model is not None:
        return _secondary_model

    try:
        from ultralytics import YOLO

        weight_path = _get_weight_path("models.secondary_weight", "best_yolov8_trash.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"YOLOv8 辅助模型权重不存在: {weight_path}")

        logger.info(f"加载辅助模型: {weight_path}")
        _secondary_model = await asyncio.to_thread(YOLO, weight_path)
        logger.info("辅助模型加载完成")
        return _secondary_model

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"辅助模型加载失败: {e}")
        raise RuntimeError(f"辅助模型加载失败: {e}") from e


# ═══════════════════════════════════════════════════════════
#  核心检测接口
# ═══════════════════════════════════════════════════════════

async def detect_trash(image_path: str) -> Dict[str, Any]:
    """
    双模型检测图片中的垃圾类型。

    两个 YOLO 模型同时对图片进行推理，各自给出最佳预测结果。
    主模型给出 5 大类，辅助模型给出 18 细分类。

    Args:
        image_path (str): 图片文件路径

    Returns:
        Dict[str, Any]: 标准化检测结果
            {
                "primary":    {"class_name": str, "confidence": float},
                "secondary":  {"class_name": str, "confidence": float},
                "algorithms": ["YOLO11-trash", "YOLOv8-trash-18"],
                "error":      str
            }
    """
    output = {
        "primary":    {"class_name": "unknown", "confidence": 0.0},
        "secondary":  {"class_name": "unknown", "confidence": 0.0},
        "algorithms": ["YOLO11-trash", "YOLOv8-trash-18"],
        "error": "",
    }

    # ── 校验文件 ──
    if not os.path.exists(image_path):
        output["error"] = f"图片文件不存在: {image_path}"
        logger.warning(output["error"])
        return output

    # ── 加载模型 ──
    try:
        primary_model = await load_primary_model()
    except Exception as e:
        output["error"] = f"主模型加载失败: {e}"
        logger.error(output["error"])
        return output

    secondary_model = None
    try:
        secondary_model = await load_secondary_model()
    except Exception as e:
        output["error"] = f"辅助模型加载失败(仅使用主模型): {e}"
        logger.warning(output["error"])

    # ── 主模型推理 ──
    try:
        primary_results = await asyncio.to_thread(
            primary_model, image_path, verbose=False
        )
        pred = _extract_best_prediction(primary_results, normalize_primary=True)
        if pred["confidence"] >= CONFIDENCE_THRESHOLD:
            output["primary"] = pred
        logger.info(
            f"主模型检测: {pred['class_name']} ({pred['confidence']:.2%})"
        )
    except Exception as e:
        output["error"] = f"主模型推理失败: {e}"
        logger.error(output["error"])
        return output

    # ── 辅助模型推理 ──
    if secondary_model:
        try:
            secondary_results = await asyncio.to_thread(
                secondary_model, image_path, verbose=False
            )
            pred2 = _extract_best_prediction(secondary_results, normalize_primary=False)
            if pred2["confidence"] >= CONFIDENCE_THRESHOLD:
                output["secondary"] = pred2
            logger.info(
                f"辅助模型检测: {pred2['class_name']} ({pred2['confidence']:.2%})"
            )
        except Exception as e:
            err = f"辅助模型推理失败: {e}"
            output["error"] = (
                f"{output['error']} | {err}" if output["error"] else err
            )
            logger.warning(err)

    return output


# ═══════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════

def _extract_best_prediction(results, normalize_primary: bool = True) -> Dict[str, Any]:
    """
    从 YOLO 推理结果中提取置信度最高的预测。

    Args:
        results: YOLO 推理结果列表
        normalize_primary: True — 按主模型映射小写; False — 保留原始类名

    Returns:
        Dict: {"class_name": str, "confidence": float}
    """
    best_conf = 0.0
    best_class = "unknown"

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_idx = confidences.argmax()
            if confidences[max_idx] > best_conf:
                best_conf = float(confidences[max_idx])
                class_id = int(class_ids[max_idx])
                raw_name = result.names.get(class_id, "unknown")
                if normalize_primary:
                    best_class = _PRIMARY_CLASS_MAP.get(raw_name, raw_name.lower())
                else:
                    best_class = raw_name

    return {"class_name": best_class, "confidence": round(best_conf, 4)}


def get_waste_cn(class_name: str) -> str:
    """获取垃圾类别的中文名称。"""
    return WASTE_CN.get(class_name, class_name)
