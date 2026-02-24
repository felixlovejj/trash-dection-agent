"""
LLM Intent Recognition Core (core/llm_intent.py)
=================================================
基于通义千问（DashScope OpenAI 兼容接口）的智能模块：

1. 意图识别 — 将用户输入分为「识别垃圾 / 查价格 / 生成报告 / 闲聊」
2. 价格分类推理 — 根据双模型检测结果推理出回收价格细分类别
3. 智能对话 — 作为垃圾分类助手与用户自然对话
4. 报告建议 — 根据识别历史给出环保建议

所有函数均支持 async/await 异步调用，并包含 try-except 异常处理。
"""

import json
import asyncio
from typing import Any, Dict, List, Optional

from modules.YA_Common.utils.logger import get_logger
from modules.YA_Common.utils.config import get_config

# ── Prompt 模板（集中管理于 prompts/prompt_templates.py）──
from prompts.prompt_templates import (
    INTENT_CLASSIFY_SYSTEM,
    PRICE_CLASSIFY_SYSTEM,
    build_price_classify_prompt,
    CHAT_REPLY_SYSTEM,
    REPORT_ADVICE_SYSTEM,
    build_report_advice_prompt,
)

logger = get_logger("core.llm_intent")

# ── LLM 客户端缓存 ───────────────────────────────────────
_llm_client = None


async def _get_llm_client():
    """
    获取 DashScope OpenAI 兼容客户端（惰性初始化 + 缓存）。

    Returns:
        openai.OpenAI 实例，或 None
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    try:
        # 优先从 SOPS 密钥管理获取 API Key
        api_key = None
        try:
            from modules.YA_Secrets.secrets_parser import get_secret
            api_key = get_secret("dashscope_api_key")
        except Exception:
            pass

        # 降级：从 config.yaml 获取（开发环境）
        if not api_key:
            api_key = get_config("secrets.dashscope_api_key", None)

        if not api_key:
            logger.warning("未配置 DASHSCOPE_API_KEY")
            return None

        from openai import OpenAI
        _llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        logger.info("通义千问 LLM 客户端初始化成功")
        return _llm_client

    except Exception as e:
        logger.error(f"LLM 客户端初始化失败: {e}")
        return None


def _get_model_name() -> str:
    """获取配置中的模型名称。"""
    return get_config("llm.model_name", "qwen-turbo")


# ═══════════════════════════════════════════════════════════
#  1. 意图识别
# ═══════════════════════════════════════════════════════════

async def recognize_intent(user_text: str) -> str:
    """
    识别用户输入的意图。

    优先使用 LLM 进行语义级分类，失败时降级为关键词匹配。

    Args:
        user_text (str): 用户输入的文本

    Returns:
        str: "识别垃圾" / "查价格" / "生成报告" / "闲聊"
    """
    try:
        client = await _get_llm_client()
        if client:
            intent = await _llm_intent(client, user_text)
            if intent:
                logger.info(f"LLM 意图识别: '{user_text}' -> {intent}")
                return intent
    except Exception as e:
        logger.warning(f"LLM 意图识别失败: {e}，降级为关键词匹配")

    # 降级：关键词匹配
    intent = _keyword_intent(user_text)
    logger.info(f"关键词意图匹配: '{user_text}' -> {intent}")
    return intent


async def _llm_intent(client, user_text: str) -> Optional[str]:
    """
    调用通义千问进行意图分类。

    Args:
        client: OpenAI 客户端
        user_text (str): 用户输入

    Returns:
        str | None: 意图类别
    """
    try:
        model_name = _get_model_name()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": INTENT_CLASSIFY_SYSTEM,
                },
                {"role": "user", "content": user_text},
            ],
            max_tokens=10,
            temperature=0.05,
            timeout=10,
        )

        if response and response.choices:
            answer = response.choices[0].message.content.strip()
            if "识别" in answer:
                return "识别垃圾"
            if "价" in answer:
                return "查价格"
            if "报告" in answer or "汇总" in answer:
                return "生成报告"
            return "闲聊"

    except Exception:
        raise

    return None


def _keyword_intent(user_text: str) -> str:
    """
    关键词匹配降级方案。

    Args:
        user_text (str): 用户输入

    Returns:
        str: 意图类别
    """
    text = user_text.lower()

    detect_kw = ["识别", "检测", "这是什么", "什么垃圾", "分类", "detect", "recognize", "看看"]
    price_kw = ["价格", "多少钱", "值多少", "回收", "卖", "price", "cost", "worth"]
    report_kw = ["报告", "汇总", "总结", "统计", "摘要", "report", "summary"]

    for kw in detect_kw:
        if kw in text:
            return "识别垃圾"
    for kw in price_kw:
        if kw in text:
            return "查价格"
    for kw in report_kw:
        if kw in text:
            return "生成报告"
    return "闲聊"


# ═══════════════════════════════════════════════════════════
#  2. 价格分类推理
# ═══════════════════════════════════════════════════════════

async def classify_for_price(
    primary_result: Dict[str, Any],
    secondary_result: Dict[str, Any],
    category_desc: Dict[str, str],
) -> Dict[str, Any]:
    """
    调用千问根据双模型检测结果推理出价格爬虫支持的细分类别。

    Args:
        primary_result (dict): 主模型结果 {"class_name": str, "confidence": float}
        secondary_result (dict): 辅助模型结果
        category_desc (dict): 价格类别中文描述映射

    Returns:
        dict: {"price_category": str | None, "reasoning": str}
    """
    try:
        client = await _get_llm_client()
        if not client:
            return {"price_category": None, "reasoning": "LLM 不可用，无法推理细分类别"}

        cat_lines = "\n".join(f"  - {k}: {v}" for k, v in category_desc.items())
        model_name = _get_model_name()

        prompt = build_price_classify_prompt(
            primary_class=primary_result["class_name"],
            primary_confidence=primary_result["confidence"],
            secondary_class=secondary_result["class_name"],
            secondary_confidence=secondary_result["confidence"],
            category_desc_lines=cat_lines,
        )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": PRICE_CLASSIFY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.1,
            timeout=15,
        )

        if response and response.choices:
            answer = response.choices[0].message.content.strip()
            # 处理 markdown 代码块
            if answer.startswith("```"):
                answer = answer.strip("`").strip()
                if answer.startswith("json"):
                    answer = answer[4:].strip()
            try:
                parsed = json.loads(answer)
                cat = parsed.get("price_category")
                reasoning = parsed.get("reasoning", "")
                if cat and cat in category_desc:
                    return {"price_category": cat, "reasoning": reasoning}
                elif cat is None:
                    return {"price_category": None, "reasoning": reasoning or "该类型无对应回收价格"}
                else:
                    return {"price_category": cat, "reasoning": reasoning}
            except json.JSONDecodeError:
                for key in category_desc:
                    if key in answer:
                        return {"price_category": key, "reasoning": answer}
                return {"price_category": None, "reasoning": f"LLM 返回格式异常: {answer[:100]}"}

        return {"price_category": None, "reasoning": "LLM 未返回有效内容"}

    except Exception as e:
        logger.error(f"价格分类推理异常: {e}")
        return {"price_category": None, "reasoning": f"LLM 调用失败: {e}"}


# ═══════════════════════════════════════════════════════════
#  3. 智能对话
# ═══════════════════════════════════════════════════════════

async def chat(
    user_text: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    作为垃圾分类智能助手与用户对话。

    Args:
        user_text (str): 用户消息
        history (list): 最近的对话历史 [{"role": "user"/"assistant", "content": str}]

    Returns:
        str: 助手回复文本
    """
    try:
        client = await _get_llm_client()
        if not client:
            return _fallback_chat_reply()

        model_name = _get_model_name()
        messages = [
            {
                "role": "system",
                "content": CHAT_REPLY_SYSTEM,
            },
        ]

        if history:
            messages.extend(history[-10:])

        messages.append({"role": "user", "content": user_text})

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
            timeout=30,
        )

        if response and response.choices:
            return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"智能对话异常: {e}")

    return _fallback_chat_reply()


def _fallback_chat_reply() -> str:
    """LLM 不可用时的降级回复。"""
    return (
        "🤖 你好！我是垃圾分类智能体，我可以帮你：\n\n"
        "1. **识别垃圾** — 上传图片后说「识别一下」\n"
        "2. **查价格** — 说「查一下塑料回收价格」\n"
        "3. **生成报告** — 说「生成报告」\n"
        "4. **了解原理** — 问「你的原理是什么」\n\n"
        "试试看吧！😊"
    )


# ═══════════════════════════════════════════════════════════
#  4. 报告建议生成
# ═══════════════════════════════════════════════════════════

async def generate_report_advice(
    history: List[Dict[str, Any]],
    primary_counter: Dict[str, int],
) -> Optional[str]:
    """
    根据识别历史数据生成千问智能建议。

    Args:
        history (list): 检测历史记录
        primary_counter (dict): 主模型类别计数

    Returns:
        str | None: 建议文本
    """
    try:
        client = await _get_llm_client()
        if not client:
            return None

        summary_lines = []
        for rec in history:
            p = rec.get("primary_cn", "未知")
            s = rec.get("secondary_class", "无")
            pc = rec.get("price_category_desc", "无")
            summary_lines.append(f"- 主模型:{p}, 辅助模型:{s}, 回收分类:{pc}")

        model_name = _get_model_name()
        data_summary = "\n".join(summary_lines)
        prompt = build_report_advice_prompt(len(history), data_summary)

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": REPORT_ADVICE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7,
            timeout=25,
        )

        if response and response.choices:
            return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"报告建议生成失败: {e}")

    return None
