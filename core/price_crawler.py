"""
Price Crawler Core (core/price_crawler.py)
==========================================
废品回收价格实时爬取模块。

数据来源：回收商网 (baojia.huishoushang.com)
支持 20+ 细分品类的全国参考价格查询。

所有函数均支持 async/await 异步调用，并包含 try-except 异常处理。
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

import requests
from bs4 import BeautifulSoup

from modules.YA_Common.utils.logger import get_logger

logger = get_logger("core.price_crawler")

# ── 垃圾分类 → URL 后缀映射 ────────────────────────────
WASTE_TO_URL_SUFFIX = {
    # 大类
    "metal": "c1", "plastic": "c2", "paper": "c3", "rubber": "c5",
    # 废金属
    "waste_iron": "c10", "waste_steel": "c11",
    "waste_stainless_steel": "c12", "waste_copper": "c13",
    "waste_aluminum": "c14", "waste_zinc_lead": "c15",
    "waste_tin": "c16", "waste_nickel": "c17",
    # 废塑料
    "plastic_abs": "c19", "plastic_pa": "c22", "plastic_pc": "c23",
    "plastic_pet": "c24", "plastic_pp": "c27", "plastic_ps": "c28",
    "plastic_pvc": "c29", "plastic_pe": "c40",
    # 废纸
    "paper_japanese_waste": "c30", "paper_european_waste": "c31",
    "paper_american_waste": "c32", "paper_carton": "c42",
    # 废橡胶
    "rubber_domestic": "c36",
}

# 价格类别中文描述
PRICE_CATEGORY_DESC = {
    "metal": "废金属（大类）", "plastic": "废塑料（大类）",
    "paper": "废纸（大类）", "rubber": "废橡胶（大类）",
    "waste_iron": "废铁", "waste_steel": "废钢",
    "waste_stainless_steel": "废不锈钢", "waste_copper": "废铜",
    "waste_aluminum": "废铝", "waste_zinc_lead": "废锌铅",
    "waste_tin": "废锡", "waste_nickel": "废镍",
    "plastic_abs": "ABS塑料", "plastic_pa": "PA塑料（尼龙）",
    "plastic_pc": "PC塑料", "plastic_pet": "PET塑料",
    "plastic_pp": "PP塑料", "plastic_ps": "PS塑料",
    "plastic_pvc": "PVC塑料", "plastic_pe": "PE塑料",
    "paper_japanese_waste": "日废纸", "paper_european_waste": "欧废纸",
    "paper_american_waste": "美废纸", "paper_carton": "废纸箱/纸板",
    "rubber_domestic": "国产废橡胶",
}

DEFAULT_SOURCE_URL_PREFIX = "https://baojia.huishoushang.com/material/list"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ═══════════════════════════════════════════════════════════
#  内部函数
# ═══════════════════════════════════════════════════════════

def _fetch_page(url: str, retries: int = 3) -> str | None:
    """
    获取网页 HTML 内容（带重试）。

    Args:
        url (str): 目标 URL
        retries (int): 最大重试次数

    Returns:
        str | None: HTML 内容
    """
    import time

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            if response.encoding == "ISO-8859-1":
                response.encoding = "utf-8"
            return response.text
        except Exception as e:
            logger.warning(f"抓取失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None


def _parse_price_page(html_content: str) -> str:
    """
    解析回收商网价格页面，计算第一页条目的均价。

    Args:
        html_content (str): HTML 文本

    Returns:
        str: 价格文本（如 "1200元/吨"），未查到返回 "NOT_FOUND"
    """
    if not html_content:
        return "NOT_FOUND"

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        prices_sum = 0
        date_num = 0

        info_list_div = soup.find("div", class_="info-list")
        if not info_list_div:
            return "NOT_FOUND"

        list_items = info_list_div.find_all("li", class_="list-item")
        for item in list_items:
            try:
                p_tags = item.find_all("p")
                if len(p_tags) >= 9:
                    price_text = p_tags[3].get_text(strip=True)
                    avg_price = 0.0
                    if "-" in price_text:
                        parts = price_text.split("-")
                        if len(parts) == 2:
                            try:
                                price_min = float(parts[0])
                                price_max = float(parts[1])
                                avg_price = (price_min + price_max) / 2
                            except ValueError:
                                avg_price = 0.0
                    else:
                        try:
                            avg_price = float(price_text)
                        except ValueError:
                            avg_price = 0.0
                    prices_sum += avg_price
                    date_num += 1
            except Exception:
                continue

        if date_num == 0:
            return "NOT_FOUND"

        return f"{int(prices_sum / date_num)}元/吨"

    except Exception as e:
        logger.error(f"解析价格页面失败: {e}")
        return "NOT_FOUND"


# ═══════════════════════════════════════════════════════════
#  对外接口
# ═══════════════════════════════════════════════════════════

async def get_trash_price(trash_type: str) -> Dict[str, Any]:
    """
    异步查询某类垃圾的回收价格。

    Args:
        trash_type (str): 垃圾类别 key（如 "plastic"、"waste_copper"）

    Returns:
        Dict[str, Any]: 标准化结果
            {
                "trash_type": str,
                "price":      str,
                "area":       str,
                "date":       str,
                "source":     str,
                "error":      str
            }
    """
    today = datetime.now().strftime("%Y-%m-%d")
    not_found = {
        "trash_type": trash_type,
        "price": "NA (NOT FOUND)",
        "area": "全国",
        "date": today,
        "source": DEFAULT_SOURCE_URL_PREFIX + "/",
        "error": "",
    }

    try:
        if trash_type not in WASTE_TO_URL_SUFFIX:
            not_found["error"] = f"不支持的类别: {trash_type}"
            logger.warning(not_found["error"])
            return not_found

        url = f"{DEFAULT_SOURCE_URL_PREFIX}-{WASTE_TO_URL_SUFFIX[trash_type]}/"

        # 在线程中执行阻塞的网络请求
        html_content = await asyncio.to_thread(_fetch_page, url)
        if not html_content:
            logger.warning(f"价格页面获取失败: {url}")
            return not_found

        price = await asyncio.to_thread(_parse_price_page, html_content)

        logger.info(f"查询 [{trash_type}] 价格: {price}")
        return {
            "trash_type": trash_type,
            "price": price,
            "area": "全国",
            "date": today,
            "source": url,
            "error": "",
        }

    except Exception as e:
        logger.error(f"价格查询异常: {e}")
        not_found["error"] = str(e)
        return not_found
