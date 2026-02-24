"""
Intent Recognize Tool (tools/intent_recognize_tool.py)
======================================================
MCP Tool — LLM 意图识别工具

输入：用户文本
输出：意图类别（识别垃圾 / 查价格 / 生成报告 / 闲聊）
"""

from typing import Any, Dict

from tools import YA_MCPServer_Tool


@YA_MCPServer_Tool(
    name="recognize_intent",
    title="意图识别工具",
    description="基于通义千问 LLM 识别用户输入的意图类别（识别垃圾/查价格/生成报告/闲聊）",
)
async def recognize_intent(user_text: str) -> Dict[str, Any]:
    """
    识别用户输入的意图。

    使用通义千问进行语义级意图分类，LLM 不可用时降级为关键词匹配。

    Args:
        user_text (str): 用户输入的文本

    Returns:
        Dict[str, Any]: 意图识别结果
            {
                "intent": str,   # "识别垃圾" / "查价格" / "生成报告" / "闲聊"
                "method": str    # "llm" / "keyword"
            }

    Example:
        {
            "intent": "识别垃圾",
            "method": "llm"
        }
    """
    try:
        from core.llm_intent import recognize_intent as _recognize

        intent = await _recognize(user_text)
        return {"intent": intent, "method": "llm"}

    except Exception as e:
        # 最终降级
        from core.llm_intent import _keyword_intent
        intent = _keyword_intent(user_text)
        return {"intent": intent, "method": "keyword", "error": str(e)}
