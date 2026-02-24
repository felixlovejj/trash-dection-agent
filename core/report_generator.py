"""
Report Generator Core (core/report_generator.py)
=================================================
检测报告生成模块：

- Markdown 格式报告（含时间线表格、类别统计、分布柱状图）
- CSV 导出（UTF-8 BOM，兼容 Excel 中文）

纯数据处理层，不依赖 LLM/Streamlit，保持可测试性。
"""

import csv
import io
from collections import Counter
from typing import Any, Dict, List

from modules.YA_Common.utils.logger import get_logger

logger = get_logger("core.report_generator")

# CSV 表头
_CSV_HEADERS = [
    "序号", "时间", "主模型类别（英）", "主模型类别（中）",
    "主模型置信度", "辅助模型类别", "辅助模型置信度",
    "千问推理分类", "千问分类描述",
]


async def generate_csv(history: List[Dict[str, Any]]) -> str:
    """
    将检测历史导出为 CSV 字符串（UTF-8 with BOM）。

    Args:
        history (list): 检测记录列表

    Returns:
        str: CSV 文本
    """
    try:
        buf = io.StringIO()
        buf.write("\ufeff")  # BOM for Excel

        writer = csv.writer(buf)
        writer.writerow(_CSV_HEADERS)

        for idx, rec in enumerate(history, 1):
            p_conf = rec.get("primary_confidence", 0)
            s_conf = rec.get("secondary_confidence", 0)
            s_cls = rec.get("secondary_class", "unknown")

            writer.writerow([
                idx,
                rec.get("time", "-"),
                rec.get("primary_class", "unknown"),
                rec.get("primary_cn", "未知"),
                f"{p_conf:.2%}" if p_conf else "-",
                s_cls if s_cls != "unknown" else "未检出",
                f"{s_conf:.2%}" if s_cls != "unknown" and s_conf else "-",
                rec.get("price_category", "-"),
                rec.get("price_category_desc", "无"),
            ])

        return buf.getvalue()

    except Exception as e:
        logger.error(f"CSV 生成失败: {e}")
        return ""


async def generate_report(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    根据检测历史生成结构化报告。

    Args:
        history (list): 检测记录列表

    Returns:
        dict: {
            "total", "markdown", "csv_content",
            "primary_counter", "secondary_counter", "price_cat_counter",
            "recyclable_count", "non_recyclable_count"
        }
    """
    empty_result = {
        "total": 0,
        "markdown": "",
        "primary_counter": {},
        "secondary_counter": {},
        "price_cat_counter": {},
        "recyclable_count": 0,
        "non_recyclable_count": 0,
        "csv_content": "",
    }

    if not history:
        return empty_result

    try:
        # 统计
        primary_counter: Counter = Counter()
        secondary_counter: Counter = Counter()
        price_cat_counter: Counter = Counter()

        for rec in history:
            p_cls = rec.get("primary_class", "unknown")
            if p_cls != "unknown":
                p_cn = rec.get("primary_cn", p_cls)
                primary_counter[f"{p_cn} ({p_cls})"] += 1
            s_cls = rec.get("secondary_class", "unknown")
            if s_cls != "unknown":
                secondary_counter[s_cls] += 1
            pc = rec.get("price_category_desc", "")
            if pc and pc != "无":
                price_cat_counter[pc] += 1

        # Markdown
        parts: List[str] = []
        parts.append("## 📋 垃圾识别汇总报告\n")
        parts.append(f"**本次会话共识别 {len(history)} 张图片**\n")

        # 时间线表格
        parts.append("### 🕐 识别时间线\n")
        parts.append("| 序号 | 时间 | 主模型结果 | 置信度 | 辅助模型结果 | 置信度 | 回收分类 |")
        parts.append("|------|------|-----------|--------|-------------|--------|---------|")
        for i, rec in enumerate(history, 1):
            p_cn = rec.get("primary_cn", "未知")
            p_conf = rec.get("primary_confidence", 0)
            s_cls = rec.get("secondary_class", "未检出")
            s_conf = rec.get("secondary_confidence", 0)
            pc_desc = rec.get("price_category_desc", "无")
            s_display = s_cls if s_cls != "unknown" else "未检出"
            s_conf_display = f"{s_conf:.1%}" if s_cls != "unknown" else "-"
            parts.append(
                f"| {i} | {rec.get('time', '-')} | {p_cn} | {p_conf:.1%} | "
                f"{s_display} | {s_conf_display} | {pc_desc} |"
            )
        parts.append("")

        # 类别统计
        parts.append("### 📊 类别统计\n")
        if primary_counter:
            parts.append("**主模型（YOLO11 · 5大类）分布：**\n")
            for name, count in primary_counter.most_common():
                parts.append(f"- {name}: **{count}** 次 {'█' * count}")
            parts.append("")
        if secondary_counter:
            parts.append("**辅助模型（YOLOv8 · 18细类）分布：**\n")
            for name, count in secondary_counter.most_common():
                parts.append(f"- {name}: **{count}** 次 {'█' * count}")
            parts.append("")

        recyclable_count = sum(
            1 for r in history
            if r.get("primary_class") in ("glass", "metal", "paper", "plastic")
        )

        csv_content = await generate_csv(history)

        return {
            "total": len(history),
            "markdown": "\n".join(parts),
            "primary_counter": dict(primary_counter),
            "secondary_counter": dict(secondary_counter),
            "price_cat_counter": dict(price_cat_counter),
            "recyclable_count": recyclable_count,
            "non_recyclable_count": len(history) - recyclable_count,
            "csv_content": csv_content,
        }

    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        empty_result["error"] = str(e)
        return empty_result
