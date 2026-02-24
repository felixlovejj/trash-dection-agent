"""
Price Cache Resource (resources/price_cache_resource.py)
========================================================
MCP Resource — 价格缓存与分类映射资源

返回支持的废品回收分类列表、分类描述映射等配置数据。
"""

from typing import Any, Dict, List

from resources import YA_MCPServer_Resource


@YA_MCPServer_Resource(
    "config://price/categories",
    name="price_categories",
    title="支持的回收分类列表",
    description="返回价格爬虫支持的所有废品分类 key 及其中文描述",
)
async def get_price_categories() -> Dict[str, Any]:
    """
    获取价格爬虫支持的所有分类。

    Returns:
        Dict: 分类 key → 中文描述映射
    """
    try:
        from core.price_crawler import PRICE_CATEGORY_DESC, WASTE_TO_URL_SUFFIX

        return {
            "categories": PRICE_CATEGORY_DESC,
            "total_categories": len(PRICE_CATEGORY_DESC),
            "url_suffixes": WASTE_TO_URL_SUFFIX,
            "source": "回收商网 (baojia.huishoushang.com)",
        }
    except Exception as e:
        return {"error": str(e)}


@YA_MCPServer_Resource(
    "config://price/category/{category_key}",
    name="price_category_detail",
    title="单个分类详情",
    description="返回指定分类 key 的详细信息",
)
async def get_price_category_detail(category_key: str) -> Dict[str, Any]:
    """
    获取某个分类的详细信息。

    Args:
        category_key (str): 分类 key，如 "waste_copper"

    Returns:
        Dict: 分类详情
    """
    try:
        from core.price_crawler import (
            PRICE_CATEGORY_DESC,
            WASTE_TO_URL_SUFFIX,
            DEFAULT_SOURCE_URL_PREFIX,
        )

        if category_key not in PRICE_CATEGORY_DESC:
            return {"error": f"不支持的分类: {category_key}"}

        url_suffix = WASTE_TO_URL_SUFFIX.get(category_key, "")
        url = f"{DEFAULT_SOURCE_URL_PREFIX}-{url_suffix}/" if url_suffix else ""

        return {
            "key": category_key,
            "name_cn": PRICE_CATEGORY_DESC[category_key],
            "url_suffix": url_suffix,
            "full_url": url,
        }
    except Exception as e:
        return {"error": str(e)}
