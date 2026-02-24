"""
Get Price Tool (tools/get_price_tool.py)
========================================
MCP Tool — 废品回收价格查询工具

输入：垃圾分类 key + 地区（可选）
输出：实时参考价格 + 数据来源
"""

from typing import Any, Dict

from tools import YA_MCPServer_Tool


@YA_MCPServer_Tool(
    name="get_recycling_price",
    title="废品价格查询工具",
    description="实时查询废品回收价格，支持废金属/废塑料/废纸等 20+ 细分品类",
)
async def get_recycling_price(
    trash_type: str,
    area: str = "全国",
) -> Dict[str, Any]:
    """
    查询指定垃圾类别的回收价格。

    Args:
        trash_type (str): 垃圾分类 key，如 "plastic"、"waste_copper"、"paper_carton"
        area (str): 查询地区（目前仅支持 "全国"）

    Returns:
        Dict[str, Any]: 价格查询结果
            {
                "trash_type": str,
                "price":      str,
                "area":       str,
                "date":       str,
                "source":     str,
                "error":      str
            }

    Example:
        {
            "trash_type": "waste_copper",
            "price": "52000元/吨",
            "area": "全国",
            "date": "2026-02-23",
            "source": "https://baojia.huishoushang.com/...",
            "error": ""
        }
    """
    try:
        from core.price_crawler import get_trash_price

        result = await get_trash_price(trash_type)
        return result

    except Exception as e:
        return {
            "trash_type": trash_type,
            "price": "N/A",
            "area": area,
            "date": "",
            "source": "",
            "error": f"价格查询工具异常: {e}",
        }
