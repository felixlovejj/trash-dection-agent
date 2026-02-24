"""
Intent Prompt (prompts/intent_prompt.py)
========================================
MCP Prompt — LLM 意图识别指令

所有 prompt 文本统一来自 prompts/prompt_templates.py，本文件只负责注册为 MCP Prompt。
"""

from typing import Any, Dict

from prompts import YA_MCPServer_Prompt
from prompts.prompt_templates import (
    build_intent_classify_prompt,
    PRICE_CLASSIFY_SYSTEM,
    build_price_classify_prompt,
)


@YA_MCPServer_Prompt(
    name="intent_classify",
    title="意图分类指令",
    description="生成用于意图识别的 system prompt，引导 LLM 对用户输入进行分类",
)
async def intent_classify_prompt(user_text: str) -> str:
    """
    生成意图分类的 LLM 指令。

    Args:
        user_text (str): 用户的输入文本

    Returns:
        str: 组装好的 prompt 文本
    """
    return build_intent_classify_prompt(user_text)


@YA_MCPServer_Prompt(
    name="price_classify",
    title="价格分类推理指令（主模型优先）",
    description="生成用于价格分类推理的 prompt，以主模型结果为准确定大类方向",
)
async def price_classify_prompt(
    primary_class: str,
    primary_confidence: float,
    secondary_class: str,
    secondary_confidence: float,
) -> str:
    """
    生成价格分类推理的 LLM 指令。

    Args:
        primary_class (str): 主模型检测类别
        primary_confidence (float): 主模型置信度
        secondary_class (str): 辅助模型检测类别
        secondary_confidence (float): 辅助模型置信度

    Returns:
        str: 组装好的 prompt 文本
    """
    try:
        from core.price_crawler import PRICE_CATEGORY_DESC
        cat_lines = "\n".join(f"  - {k}: {v}" for k, v in PRICE_CATEGORY_DESC.items())
    except Exception:
        cat_lines = "(类别列表加载失败)"

    return build_price_classify_prompt(
        primary_class=primary_class,
        primary_confidence=primary_confidence,
        secondary_class=secondary_class,
        secondary_confidence=secondary_confidence,
        category_desc_lines=cat_lines,
    )
