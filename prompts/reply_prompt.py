"""
Reply Prompt (prompts/reply_prompt.py)
======================================
MCP Prompt — 智能体回复生成指令

所有 prompt 文本统一来自 prompts/prompt_templates.py，本文件只负责注册为 MCP Prompt。
"""

from typing import Any

from prompts import YA_MCPServer_Prompt
from prompts.prompt_templates import (
    CHAT_REPLY_SYSTEM,
    build_report_advice_prompt,
)


@YA_MCPServer_Prompt(
    name="chat_reply",
    title="闲聊回复指令",
    description="生成用于智能闲聊的 system prompt，让 LLM 以垃圾分类助手身份回复",
)
async def chat_reply_prompt(user_text: str) -> str:
    """
    生成智能闲聊的 system prompt。

    Args:
        user_text (str): 用户输入

    Returns:
        str: 完整的 prompt 文本
    """
    return f"{CHAT_REPLY_SYSTEM}\n\n用户说: {user_text}"


@YA_MCPServer_Prompt(
    name="report_advice",
    title="报告建议指令",
    description="生成用于环保建议的 prompt，根据识别数据给出分析和投放指导",
)
async def report_advice_prompt(
    total_count: int,
    data_summary: str,
) -> str:
    """
    生成报告建议的 LLM 指令。

    Args:
        total_count (int): 识别总数
        data_summary (str): 识别数据摘要

    Returns:
        str: 完整的 prompt 文本
    """
    return build_report_advice_prompt(total_count, data_summary)
