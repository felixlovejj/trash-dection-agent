"""
Streamlit 聊天式智能体界面 — MCP Client 版 (ui/chat_voice_llm_agent.py)
========================================================================
功能:
  1. 仿 ChatGPT 聊天界面（st.chat_message）
  2. 侧边栏图片上传 + 预览
  3. 底部文字输入 + 麦克风语音输入（audio-recorder-streamlit）
  4. 通过 MCP SSE 协议调用 Server 的 Tool（意图识别 / 垃圾检测 / 价格查询 / 语音 / 报告）
  5. 通义千问 LLM 智能对话 & 价格推理（客户端侧）
  6. 报告生成 + CSV 导出下载

MCP Server Tools（通过 SSE 调用）:
  - detect_waste_type          : 双模型（YOLO11 + YOLOv8）垃圾检测
  - get_recycling_price        : 实时废品回收价格爬取
  - recognize_intent           : LLM 意图识别
  - speech_to_text             : 百度语音识别（STT）
  - text_to_speech             : 语音合成（TTS）
  - generate_detection_report  : 检测报告生成与 CSV 导出

运行方式:
    1. 先启动 MCP Server: cd YA_MCPServer_Template && uv run server.py
    2. 再启动界面:        streamlit run ui/chat_voice_llm_agent.py
"""

# ── 标准库 ──────────────────────────────────────────────
import os
import sys
import json
import csv
import io
import tempfile
import asyncio
import base64
import wave
import struct
from datetime import datetime

# ── 第三方库 ────────────────────────────────────────────
import streamlit as st
import pandas as pd

# ── 将项目根目录（YA_MCPServer_Template）加入 sys.path ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Prompt 模板（集中管理于 prompts/prompt_templates.py）──
from prompts.prompt_templates import (
    INTENT_CLASSIFY_SYSTEM,
    PRICE_CLASSIFY_SYSTEM,
    build_price_classify_prompt,
    CHAT_REPLY_SYSTEM,
    REPORT_ADVICE_SYSTEM,
    build_report_advice_prompt,
)


# ═══════════════════════════════════════════════════════════
#  【新增】MCP Server 配置
# ═══════════════════════════════════════════════════════════
MCP_SERVER_URL = "http://127.0.0.1:12346"
"""MCP Server 的 SSE 端点地址（默认由 config.yaml 的 transport 配置决定）"""


# ═══════════════════════════════════════════════════════════
#  【新增】从 config.yaml / SOPS(env.yaml) 读取密钥 & 模型配置
# ═══════════════════════════════════════════════════════════
def _load_config():
    """从 config.yaml 读取配置。"""
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_secrets():
    """
    加载密钥：优先从 SOPS (env.yaml) 解密获取，降级到 config.yaml 中的 secrets 字段。
    """
    # 1. 优先尝试 SOPS 解密
    try:
        from modules.YA_Secrets.secrets_parser import load_secrets
        sops_secrets = load_secrets(
            path=os.path.join(PROJECT_ROOT, "env.yaml"),
            sops_config=os.path.join(PROJECT_ROOT, ".sops.yaml"),
        )
        if sops_secrets:
            return sops_secrets
    except Exception:
        pass

    # 2. 降级：从 config.yaml 的 secrets 字段读取（本地开发调试）
    return _CFG.get("secrets", {}) or {}


_CFG = _load_config()
_llm_cfg = _CFG.get("llm", {})
_secrets = _load_secrets()

DASHSCOPE_API_KEY = _secrets.get("dashscope_api_key", "")
QWEN_MODEL_NAME = _llm_cfg.get("model_name", "qwen-turbo")
BAIDU_APP_ID = _secrets.get("baidu_app_id", "")
BAIDU_API_KEY = _secrets.get("baidu_api_key", "")
BAIDU_SECRET_KEY = _secrets.get("baidu_secret_key", "")


# ═══════════════════════════════════════════════════════════
#  常量 & 路径（100% 保留原有定义）
# ═══════════════════════════════════════════════════════════
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

WASTE_CATEGORIES = [
    "glass", "metal", "paper", "plastic", "waste"
]

# 垃圾类别中英对照
WASTE_CN = {
    "glass": "玻璃", "metal": "金属",
    "paper": "纸张", "plastic": "塑料", "waste": "其他垃圾",
    "unknown": "未知",
}

# ── 价格爬虫支持的细分类别（来自 WASTE_TO_URL_SUFFIX） ──
PRICE_CATEGORY_DESC = {
    "metal": "废金属（大类）",
    "plastic": "废塑料（大类）",
    "paper": "废纸（大类）",
    "rubber": "废橡胶（大类）",
    "waste_iron": "废铁",
    "waste_steel": "废钢",
    "waste_stainless_steel": "废不锈钢",
    "waste_copper": "废铜",
    "waste_aluminum": "废铝",
    "waste_zinc_lead": "废锌铅",
    "waste_tin": "废锡",
    "waste_nickel": "废镍",
    "plastic_abs": "ABS塑料",
    "plastic_pa": "PA塑料（尼龙）",
    "plastic_pc": "PC塑料",
    "plastic_pet": "PET塑料",
    "plastic_pp": "PP塑料",
    "plastic_ps": "PS塑料",
    "plastic_pvc": "PVC塑料",
    "plastic_pe": "PE塑料",
    "paper_japanese_waste": "日废纸",
    "paper_european_waste": "欧废纸",
    "paper_american_waste": "美废纸",
    "paper_carton": "废纸箱/纸板",
    "rubber_domestic": "国产废橡胶",
}


# ═══════════════════════════════════════════════════════════
#  辅助：状态日志记录器（保留原有逻辑）
# ═══════════════════════════════════════════════════════════

def _log_service(service_name, success, detail=""):
    """
    记录服务调用状态到 session_state，用于 UI 展示。

    输入:
        service_name (str)  : 服务名称
        success      (bool) : 是否成功
        detail       (str)  : 详细信息
    """
    if "service_logs" not in st.session_state:
        st.session_state["service_logs"] = []
    st.session_state["service_logs"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "service": service_name,
        "status": "✅ 成功" if success else "❌ 失败",
        "detail": detail,
    })


# ═══════════════════════════════════════════════════════════
#  【新增】MCP SSE Client — 连接检测 & Tool 调用
# ═══════════════════════════════════════════════════════════

def _check_mcp_server_online() -> bool:
    """
    【新增】检测 MCP Server 是否在线。

    通过 TCP socket 检测端口是否开放（避免 SSE 流阻塞 HTTP 请求）。

    Returns:
        bool: True 表示 Server 在线
    """
    import socket
    try:
        # 从 MCP_SERVER_URL 解析 host 和 port
        from urllib.parse import urlparse
        parsed = urlparse(MCP_SERVER_URL)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 12345

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


async def _call_mcp_tool_sse(tool_name: str, arguments: dict):
    """
    【改造核心】通过 MCP SSE 协议调用 Server 的 Tool。

    替换原有的 stdio 方式，使用 SSE Transport 连接到 MCP Server。

    输入:
        tool_name  (str) : MCP Tool 名称（如 "detect_waste_type"）
        arguments  (dict): 传递给 Tool 的参数

    输出:
        str 或 None: Tool 返回的文本内容
    """
    from mcp.client.sse import sse_client
    from mcp import ClientSession

    server_url = MCP_SERVER_URL + "/"

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if result.content:
                # 尝试获取文本内容
                return result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            return None


def call_mcp_tool(tool_name, arguments, status_container=None):
    """
    【改造核心】同步包装：在 Streamlit 中调用异步 MCP Tool（SSE 方式），带状态显示。

    替换原有的 call_mcp_tool(service_name, tool_name, arguments) 三参数形式，
    改为直接调用 MCP Server 的 Tool（无需指定 service_name）。

    输入:
        tool_name        (str)  : MCP Tool 名称
        arguments        (dict) : Tool 参数
        status_container (st容器, 可选)
    输出:
        str 或 None
    """
    svc_label = f"MCP/{tool_name}"

    def _run_async():
        """安全地运行异步 MCP 调用，兼容已有事件循环的环境。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 已经有事件循环在运行（Streamlit 环境常见），
            # 在新线程中运行避免冲突
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _call_mcp_tool_sse(tool_name, arguments))
                return future.result(timeout=120)
        else:
            return asyncio.run(_call_mcp_tool_sse(tool_name, arguments))

    try:
        if status_container:
            with status_container.status(f"📡 调用 {svc_label} ...", expanded=True) as s:
                st.write(f"参数: `{json.dumps(arguments, ensure_ascii=False)[:200]}`")
                result = _run_async()
                if result:
                    st.write("✅ 服务返回成功")
                    s.update(label=f"📡 {svc_label} 调用成功", state="complete")
                else:
                    st.write("⚠️ 服务返回空结果")
                    s.update(label=f"📡 {svc_label} 返回空", state="error")
        else:
            result = _run_async()

        _log_service(svc_label, bool(result),
                     "返回成功" if result else "返回空")
        return result

    except Exception as e:
        _log_service(svc_label, False, str(e))
        if status_container:
            status_container.error(f"❌ {svc_label} 调用失败: {e}\n\n请确认 MCP Server 已启动（`uv run server.py`）")
        return None


# ═══════════════════════════════════════════════════════════
#  1. 意图识别 — 【改造】通过 MCP Tool 调用
# ═══════════════════════════════════════════════════════════

def _get_llm_client():
    """
    获取 DashScope OpenAI 兼容客户端（缓存到 session_state）。

    保留在客户端侧，用于智能对话和价格推理。

    输出: openai.OpenAI 实例，或 None
    """
    if not DASHSCOPE_API_KEY:
        return None

    if "llm_client" not in st.session_state:
        try:
            from openai import OpenAI
            st.session_state["llm_client"] = OpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            st.session_state["llm_available"] = True
        except Exception as e:
            st.session_state["llm_client"] = None
            st.session_state["llm_available"] = False
            st.session_state["llm_error"] = str(e)

    return st.session_state.get("llm_client")


def recognize_intent(user_text, status_container=None):
    """
    【改造】识别用户输入的意图 — 通过 MCP Tool "recognize_intent" 调用。

    原有逻辑：直接调用本地 LLM + 关键词降级
    改造后：通过 MCP SSE 调用 Server 的 recognize_intent Tool

    输入: user_text (str), status_container (st容器, 可选)
    输出: str — "识别垃圾" / "查价格" / "生成报告" / "闲聊"
    """
    try:
        if status_container:
            with status_container.status("🧠 MCP 意图识别中...", expanded=True) as s:
                st.write("正在通过 MCP Server 调用意图识别 Tool...")
                result_json = call_mcp_tool(
                    "recognize_intent",
                    {"user_text": user_text},
                )
                if result_json:
                    result = json.loads(result_json) if isinstance(result_json, str) else result_json
                    intent = result.get("intent", "闲聊")
                    method = result.get("method", "unknown")
                    st.write(f"识别结果: **{intent}** (方式: {method})")
                    s.update(label=f"🧠 意图: {intent}", state="complete")
                    _log_service("MCP/recognize_intent", True, f"意图={intent}, 方式={method}")
                    return intent
                else:
                    st.write("⚠️ MCP 意图识别返回空，降级为关键词匹配")
                    s.update(label="🧠 MCP 返回空 → 关键词匹配", state="error")
        else:
            result_json = call_mcp_tool("recognize_intent", {"user_text": user_text})
            if result_json:
                result = json.loads(result_json) if isinstance(result_json, str) else result_json
                intent = result.get("intent", "闲聊")
                _log_service("MCP/recognize_intent", True, f"意图={intent}")
                return intent

    except Exception as e:
        _log_service("MCP/recognize_intent", False, str(e))
        if status_container:
            status_container.warning(f"⚠️ MCP 意图识别异常: {e}，降级为关键词匹配")

    # 降级：本地关键词匹配
    intent = _keyword_intent(user_text)
    _log_service("关键词意图匹配(降级)", True, f"意图={intent}")
    return intent


def _keyword_intent(user_text):
    """
    关键词匹配降级方案（保留原有逻辑）。

    输入: user_text (str)
    输出: str — 意图类别
    """
    text = user_text.lower()

    detect_keywords = ["识别", "检测", "这是什么", "什么垃圾", "分类", "detect", "recognize", "看看"]
    price_keywords = ["价格", "多少钱", "值多少", "回收", "卖", "price", "cost", "worth"]
    report_keywords = ["报告", "汇总", "总结", "统计", "摘要", "report", "summary"]

    for kw in detect_keywords:
        if kw in text:
            return "识别垃圾"
    for kw in price_keywords:
        if kw in text:
            return "查价格"
    for kw in report_keywords:
        if kw in text:
            return "生成报告"
    return "闲聊"


def _llm_intent(client, user_text):
    """
    调用通义千问进行意图分类（客户端侧，仅作为本地降级备用）。

    输入: client (OpenAI), user_text (str)
    输出: str 或 None
    """
    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL_NAME,
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
        pass

    return None


def _llm_classify_for_price(primary_result, secondary_result):
    """
    调用千问推理出价格爬虫支持的细分类别（保留原有逻辑）。

    输入:
        primary_result   (dict): {"class_name": str, "confidence": float}
        secondary_result (dict): {"class_name": str, "confidence": float}
    输出:
        dict: {"price_category": str|None, "reasoning": str}
    """
    client = _get_llm_client()
    if not client:
        return {"price_category": None, "reasoning": "LLM 不可用"}

    cat_lines = "\n".join(f"  - {k}: {v}" for k, v in PRICE_CATEGORY_DESC.items())

    prompt = build_price_classify_prompt(
        primary_class=primary_result["class_name"],
        primary_confidence=primary_result["confidence"],
        secondary_class=secondary_result["class_name"],
        secondary_confidence=secondary_result["confidence"],
        category_desc_lines=cat_lines,
    )

    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL_NAME,
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
            if answer.startswith("```"):
                answer = answer.strip("`").strip()
                if answer.startswith("json"):
                    answer = answer[4:].strip()
            try:
                parsed = json.loads(answer)
                cat = parsed.get("price_category")
                reasoning = parsed.get("reasoning", "")
                if cat and cat in PRICE_CATEGORY_DESC:
                    return {"price_category": cat, "reasoning": reasoning}
                elif cat is None:
                    return {"price_category": None, "reasoning": reasoning or "该类型无对应回收价格"}
                else:
                    return {"price_category": cat, "reasoning": reasoning}
            except json.JSONDecodeError:
                for key in PRICE_CATEGORY_DESC:
                    if key in answer:
                        return {"price_category": key, "reasoning": answer}
                return {"price_category": None, "reasoning": f"LLM 返回格式异常: {answer[:100]}"}

    except Exception as e:
        return {"price_category": None, "reasoning": f"LLM 调用失败: {e}"}

    return {"price_category": None, "reasoning": "LLM 未返回有效结果"}


# ═══════════════════════════════════════════════════════════
#  2. 语音功能 — 【改造】通过 MCP Tool 调用 STT/TTS
# ═══════════════════════════════════════════════════════════

def _convert_audio_to_wav16k(audio_bytes):
    """
    将 audio-recorder-streamlit 返回的音频转为 16kHz 单声道 WAV（保留原有逻辑）。

    输入: audio_bytes (bytes)
    输出: bytes — 16kHz WAV 音频
    """
    try:
        with io.BytesIO(audio_bytes) as src_io:
            with wave.open(src_io, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

        if sampwidth == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = list(struct.unpack(fmt, raw_data))
        elif sampwidth == 1:
            samples = [((b - 128) << 8) for b in raw_data]
        else:
            fmt = f"<{n_frames * n_channels}i"
            int_samples = struct.unpack(fmt, raw_data)
            samples = [s >> 16 for s in int_samples]

        if n_channels > 1:
            samples = samples[::n_channels]

        target_rate = 16000
        if framerate != target_rate:
            ratio = framerate / target_rate
            new_length = int(len(samples) / ratio)
            new_samples = []
            for i in range(new_length):
                src_idx = i * ratio
                idx = int(src_idx)
                frac = src_idx - idx
                if idx + 1 < len(samples):
                    val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
                else:
                    val = samples[idx]
                new_samples.append(int(max(-32768, min(32767, val))))
            samples = new_samples

        out_io = io.BytesIO()
        with wave.open(out_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_rate)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        return out_io.getvalue()

    except Exception:
        return audio_bytes


def speech_to_text(audio_bytes, status_container=None):
    """
    【改造】语音识别（STT）— 通过 MCP Tool "speech_to_text" 调用。

    原有逻辑：直接调用百度 AipSpeech
    改造后：将音频 base64 编码后通过 MCP SSE 发送给 Server 的 speech_to_text Tool

    输入:
        audio_bytes      (bytes)  : WAV 格式音频
        status_container (st容器) : 可选
    输出:
        str — 识别出的文字，失败时返回 None
    """
    try:
        # 将音频字节编码为 base64 以便通过 MCP 传输
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        if status_container:
            with status_container.status("🎤 MCP 语音识别中...", expanded=True) as s:
                st.write("正在通过 MCP Server 调用语音识别 Tool...")
                result_json = call_mcp_tool(
                    "speech_to_text",
                    {"audio_base64": audio_base64},
                )
                if result_json:
                    result = json.loads(result_json) if isinstance(result_json, str) else result_json
                    text = result.get("text", "")
                    error = result.get("error", "")
                    if text:
                        st.write(f"识别结果: **{text}**")
                        s.update(label="🎤 语音识别完成", state="complete")
                        _log_service("MCP/speech_to_text", True, f"文字={text}")
                        return text
                    else:
                        st.write(f"识别失败: {error}")
                        s.update(label="🎤 语音识别失败", state="error")
                        _log_service("MCP/speech_to_text", False, error)
                        return None
                else:
                    s.update(label="🎤 MCP STT 返回空", state="error")
                    _log_service("MCP/speech_to_text", False, "返回空")
                    return None
        else:
            result_json = call_mcp_tool("speech_to_text", {"audio_base64": audio_base64})
            if result_json:
                result = json.loads(result_json) if isinstance(result_json, str) else result_json
                text = result.get("text", "")
                if text:
                    _log_service("MCP/speech_to_text", True, f"文字={text}")
                    return text
            _log_service("MCP/speech_to_text", False, "返回空或无文字")
            return None

    except Exception as e:
        _log_service("MCP/speech_to_text", False, str(e))
        return None


def text_to_speech(text, status_container=None):
    """
    【改造】语音合成（TTS）— 通过 MCP Tool "text_to_speech" 调用。

    原有逻辑：直接调用百度 TTS / DashScope CosyVoice
    改造后：通过 MCP SSE 发送文本给 Server 的 text_to_speech Tool

    输入:
        text             (str)   : 要播报的文字
        status_container (st容器): 可选
    输出:
        bytes 或 None : 音频数据
    """
    # ── 清理文字（去 Markdown、截断） ──
    clean_text = text.replace("*", "").replace("#", "").replace("|", "")
    clean_text = clean_text.replace("---", "").replace("```", "").replace("  ", " ")
    lines = [l for l in clean_text.split("\n") if not l.strip().startswith("|-")]
    clean_text = "\n".join(lines).strip()
    if len(clean_text) > 500:
        clean_text = clean_text[:500] + "..."
    if not clean_text:
        return None

    try:
        if status_container:
            with status_container.status("🔊 MCP TTS 合成中...", expanded=False) as s:
                result_json = call_mcp_tool(
                    "text_to_speech",
                    {"text": clean_text},
                )
                if result_json:
                    result = json.loads(result_json) if isinstance(result_json, str) else result_json
                    audio_b64 = result.get("audio_base64", "")
                    error = result.get("error", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        s.update(label="🔊 TTS 合成完成", state="complete")
                        _log_service("MCP/text_to_speech", True, f"生成 {len(audio_data)} 字节")
                        return audio_data
                    else:
                        s.update(label="🔊 TTS 失败", state="error")
                        _log_service("MCP/text_to_speech", False, error)
                        return None
                else:
                    s.update(label="🔊 MCP TTS 返回空", state="error")
                    return None
        else:
            result_json = call_mcp_tool("text_to_speech", {"text": clean_text})
            if result_json:
                result = json.loads(result_json) if isinstance(result_json, str) else result_json
                audio_b64 = result.get("audio_base64", "")
                if audio_b64:
                    audio_data = base64.b64decode(audio_b64)
                    _log_service("MCP/text_to_speech", True, f"{len(audio_data)} 字节")
                    return audio_data
            _log_service("MCP/text_to_speech", False, "返回空或无音频数据")
            return None

    except Exception as e:
        _log_service("MCP/text_to_speech", False, str(e))
        return None


# ═══════════════════════════════════════════════════════════
#  3. Streamlit 页面组件（100% 保留原有 UI）
# ═══════════════════════════════════════════════════════════

def init_page():
    """初始化页面配置与 session_state。"""
    st.set_page_config(
        page_title="♻️ 垃圾分类智能体",
        page_icon="♻️",
        layout="centered",
    )

    defaults = {
        "messages": [],
        "detection_result": None,
        "uploaded_image_path": None,
        "tmp_dirs": [],
        "service_logs": [],
        "last_audio_hash": None,
        "voice_enabled": True,       # 语音播报默认开启
        "pending_tts_text": None,    # 待合成的TTS文字（延迟到下轮rerun）
        "llm_price_category": None,  # 千问推理出的价格细分类别
        "detection_history": [],     # 本次会话的所有识别记录
        "last_report_csv": None,     # 最近一次报告的 CSV 内容（供下载）
        "mcp_server_online": None,   # 【新增】MCP Server 在线状态
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sidebar_upload():
    """
    侧边栏：图片上传 + 预览 + 语音设置 + API 状态 + 运行说明 + 服务日志。

    输出: str 或 None — 上传图片的临时路径
    """
    with st.sidebar:
        st.header("📷 上传垃圾图片")
        uploaded_file = st.file_uploader(
            "选择图片文件",
            type=["jpg", "jpeg", "png", "bmp"],
            key="img_uploader",
        )

        if uploaded_file is not None:
            tmp_dir = tempfile.mkdtemp(prefix="trash_detect_")
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["tmp_dirs"].append(tmp_dir)
            st.session_state["uploaded_image_path"] = tmp_path
            st.image(uploaded_file, caption="已上传图片", use_container_width=True)
        else:
            _cleanup_tmp_dirs()
            st.session_state["uploaded_image_path"] = None

        # ── 语音设置 ──
        st.divider()
        st.subheader("🔊 语音设置")
        voice_on = st.toggle(
            "启用语音播报 (TTS)",
            key="voice_enabled",
        )

        # ── 【改造】API 状态 — 显示 MCP Server 连接状态 ──
        st.divider()
        st.subheader("🔗 连接状态")
        server_online = st.session_state.get("mcp_server_online", False)
        if server_online:
            st.caption(f"✅ MCP Server 已连接 ({MCP_SERVER_URL})")
        else:
            st.caption(f"❌ MCP Server 未连接 ({MCP_SERVER_URL})")
            st.caption("→ 请先执行 `uv run server.py` 启动 Server")

        llm_client = _get_llm_client()
        if llm_client:
            st.caption(f"✅ 通义千问 ({QWEN_MODEL_NAME}) 已就绪")
        else:
            llm_err = st.session_state.get("llm_error", "未配置 API Key")
            st.caption(f"⚠️ 通义千问: {llm_err}")

        # ── 【新增】运行说明 ──
        st.divider()
        st.subheader("📖 运行说明")
        st.markdown(
            "1. **启动 MCP Server**\n"
            "   ```\n"
            "   cd YA_MCPServer_Template\n"
            "   uv run server.py\n"
            "   ```\n"
            "2. **启动聊天界面**\n"
            "   ```\n"
            "   streamlit run ui/chat_voice_llm_agent.py\n"
            "   ```\n"
            "3. **调试功能**\n"
            "   使用 MCP Inspector 连接\n"
            f"   `{MCP_SERVER_URL}` 验证 Tool"
        )

        # ── 服务调用日志 ──
        st.divider()
        _sidebar_service_log()

    return st.session_state.get("uploaded_image_path")


def _sidebar_service_log():
    """侧边栏：服务调用日志（实时查看各服务调用成功/失败）。"""
    st.subheader("📋 服务调用日志")
    logs = st.session_state.get("service_logs", [])
    if not logs:
        st.caption("暂无调用记录")
        return

    # 只显示最近 20 条
    recent = logs[-20:]
    for log in reversed(recent):
        st.caption(
            f"`{log['time']}` {log['status']} **{log['service']}** "
            f"{'| ' + log['detail'] if log['detail'] else ''}"
        )

    if st.button("清空日志", key="btn_clear_logs"):
        st.session_state["service_logs"] = []
        st.rerun()


def _cleanup_tmp_dirs():
    """清理临时文件夹。"""
    import shutil
    for d in st.session_state.get("tmp_dirs", []):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    st.session_state["tmp_dirs"] = []


# ═══════════════════════════════════════════════════════════
#  4. 聊天核心逻辑（100% 保留交互流程）
# ═══════════════════════════════════════════════════════════

def render_chat_history():
    """渲染聊天历史记录，含报告消息的 CSV 下载按钮。"""
    csv_data = st.session_state.get("last_report_csv")
    for idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and "⬇️ **点击下方按钮下载 CSV 报告文件**" in msg["content"]
                and csv_data
            ):
                st.download_button(
                    label="📥 下载 CSV 报告",
                    data=csv_data.encode("utf-8"),
                    file_name=f"trash_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{idx}",
                )


def add_message(role, content):
    """向聊天记录中添加一条消息。"""
    st.session_state["messages"].append({"role": role, "content": content})


def handle_user_input(user_text, image_path, status_container):
    """
    处理用户输入（保留原有流程）。

    步骤:
        1. 意图识别（通过 MCP Tool）
        2. 根据意图调用对应 MCP Tool
        3. 添加文字回复

    输入:
        user_text        (str)     : 用户输入文字
        image_path       (str/None): 上传的图片路径
        status_container (st容器)  : 用于展示各步骤状态
    输出:
        str : 文字回复
    """
    # —— 1. 意图识别 —— 【改造】通过 MCP Tool 调用
    intent = recognize_intent(user_text, status_container)

    # —— 2. 根据意图执行 ——
    if intent == "识别垃圾":
        reply = _handle_detect(image_path, status_container)
    elif intent == "查价格":
        reply = _handle_price(user_text, status_container)
    elif intent == "生成报告":
        reply = _handle_report(status_container)
    elif intent == "闲聊":
        reply = _handle_chat(user_text, status_container)
    else:
        reply = _handle_chat(user_text, status_container)

    # —— 3. 添加助手回复 ——
    add_message("assistant", reply)

    return reply


def _handle_detect(image_path, status_container=None):
    """
    【改造】处理「识别垃圾」意图 — 通过 MCP Tool "detect_waste_type" 调用。

    流程:
      1. 调用 MCP detect_waste_type Tool（双模型同时检测）
      2. 展示主模型 & 辅助模型各自的检测结果
      3. 调用千问推理出价格爬虫的细分类别（客户端侧 LLM）
      4. 自动调用 MCP get_recycling_price Tool 查询价格
      5. 将所有结果汇总展示
    """
    if not image_path:
        return "⚠️ 请先在左侧边栏上传一张垃圾图片，然后再说「识别」。"

    try:
        # ── Step 1: 【改造】通过 MCP Tool 进行双模型检测 ──
        result_json = call_mcp_tool(
            "detect_waste_type",
            {"image_path": image_path},
            status_container,
        )
        if not result_json:
            return "❌ 识别服务返回了空结果，请检查 MCP Server 是否启动。"

        result = json.loads(result_json) if isinstance(result_json, str) else result_json
        st.session_state["detection_result"] = result

        if result.get("error") and "推理失败" not in result.get("error", ""):
            if "主模型" in result.get("error", ""):
                return f"❌ 识别出错: {result['error']}"

        primary = result.get("primary", {"class_name": "unknown", "confidence": 0.0})
        secondary = result.get("secondary", {"class_name": "unknown", "confidence": 0.0})
        algorithms = result.get("algorithms", [])

        primary_cn = WASTE_CN.get(primary["class_name"], primary["class_name"])
        secondary_name = secondary["class_name"]

        # ── 记录到检测历史 ──
        st.session_state["detection_history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "primary_class": primary["class_name"],
            "primary_cn": primary_cn,
            "primary_confidence": primary["confidence"],
            "secondary_class": secondary_name,
            "secondary_confidence": secondary["confidence"],
        })

        # ── Step 2: 构建双模型结果展示（保留原有 Markdown 格式） ──
        reply_parts = []
        reply_parts.append("## 🔍 双模型垃圾检测结果\n")
        reply_parts.append("### 📊 主模型（YOLO11 · 5大类）\n")
        reply_parts.append(f"| 项目 | 内容 |")
        reply_parts.append(f"|------|------|")
        reply_parts.append(
            f"| 垃圾类型 | **{primary_cn}** ({primary['class_name']}) |"
        )
        reply_parts.append(f"| 置信度 | {primary['confidence']:.2%} |")
        reply_parts.append("")
        reply_parts.append("### 🔬 辅助模型（YOLOv8 · 18细类）\n")
        reply_parts.append(f"| 项目 | 内容 |")
        reply_parts.append(f"|------|------|")
        if secondary_name != "unknown" and secondary["confidence"] > 0:
            reply_parts.append(
                f"| 垃圾类型 | **{secondary_name}** |"
            )
            reply_parts.append(f"| 置信度 | {secondary['confidence']:.2%} |")
        else:
            reply_parts.append(f"| 垃圾类型 | 未检出 |")
            reply_parts.append(f"| 置信度 | - |")
        reply_parts.append("")

        # ── Step 3: 千问推理细分类别（客户端侧 LLM 调用，保留原有逻辑） ──
        if status_container:
            with status_container.status("🧠 通义千问推理回收类别中...", expanded=True) as s:
                st.write("正在综合两个模型结果进行智能分类...")
                llm_result = _llm_classify_for_price(primary, secondary)
                price_cat = llm_result.get("price_category")
                reasoning = llm_result.get("reasoning", "")
                if price_cat:
                    st.write(f"推理结果: **{PRICE_CATEGORY_DESC.get(price_cat, price_cat)}** ({price_cat})")
                    s.update(label=f"🧠 推理完成: {PRICE_CATEGORY_DESC.get(price_cat, price_cat)}", state="complete")
                    _log_service("千问分类推理", True, f"category={price_cat}")
                else:
                    st.write(f"⚠️ 无法确定回收类别: {reasoning}")
                    s.update(label="🧠 推理完成 (无匹配类别)", state="complete")
                    _log_service("千问分类推理", True, f"无匹配: {reasoning[:50]}")
        else:
            llm_result = _llm_classify_for_price(primary, secondary)
            price_cat = llm_result.get("price_category")
            reasoning = llm_result.get("reasoning", "")

        # 保存千问推理结果
        st.session_state["llm_price_category"] = price_cat

        # 将千问推理结果补充到最近一条检测历史
        if st.session_state["detection_history"]:
            st.session_state["detection_history"][-1]["price_category"] = price_cat
            st.session_state["detection_history"][-1]["price_category_desc"] = (
                PRICE_CATEGORY_DESC.get(price_cat, "无") if price_cat else "无"
            )

        reply_parts.append("### 🧠 通义千问智能推理\n")
        reply_parts.append(f"| 项目 | 内容 |")
        reply_parts.append(f"|------|------|")
        if price_cat:
            cat_desc = PRICE_CATEGORY_DESC.get(price_cat, price_cat)
            reply_parts.append(f"| 回收分类 | **{cat_desc}** (`{price_cat}`) |")
        else:
            reply_parts.append(f"| 回收分类 | 无对应回收类别 |")
        reply_parts.append(f"| 推理依据 | {reasoning} |")
        reply_parts.append(f"| 使用算法 | {', '.join(algorithms)} |")
        reply_parts.append("")

        # ── Step 4: 【改造】通过 MCP Tool 查询回收价格 ──
        if price_cat:
            try:
                price_json = call_mcp_tool(
                    "get_recycling_price",
                    {"trash_type": price_cat},
                    status_container,
                )
                if price_json:
                    price_result = json.loads(price_json) if isinstance(price_json, str) else price_json
                    price_val = price_result.get("price", "N/A")
                    reply_parts.append("### 💰 回收价格查询\n")
                    reply_parts.append(f"| 项目 | 内容 |")
                    reply_parts.append(f"|------|------|")
                    reply_parts.append(
                        f"| 查询类别 | **{PRICE_CATEGORY_DESC.get(price_cat, price_cat)}** |"
                    )
                    reply_parts.append(f"| 参考价格 | **{price_val}** |")
                    reply_parts.append(f"| 地区 | {price_result.get('area', '全国')} |")
                    reply_parts.append(f"| 数据来源 | {price_result.get('source', '-')} |")
                    reply_parts.append(f"| 查询日期 | {price_result.get('date', '-')} |")
                else:
                    reply_parts.append("### 💰 回收价格\n⚠️ 价格查询返回为空")
            except Exception as e:
                reply_parts.append(f"### 💰 回收价格\n❌ 价格查询失败: {str(e)}")
        else:
            reply_parts.append("### 💰 回收价格\n⚠️ 该类型暂无对应回收价格数据")

        if result.get("error"):
            reply_parts.append(f"\n> ⚠️ 注意: {result['error']}")

        return "\n".join(reply_parts)

    except Exception as e:
        return f"❌ 调用识别服务失败: {str(e)}\n\n请检查 MCP Server 是否已启动（`uv run server.py`）"


def _handle_price(user_text, status_container=None):
    """
    【改造】处理「查价格」意图 — 通过 MCP Tool "get_recycling_price" 调用。

    优先使用上次检测 + 千问推理出的 price_category，否则从文字中提取类别。
    """
    # 优先用千问推理出的 price_category
    price_cat = st.session_state.get("llm_price_category")

    # 如果没有，尝试从检测结果中重新推理
    if not price_cat:
        det = st.session_state.get("detection_result")
        if det:
            primary = det.get("primary", {"class_name": "unknown", "confidence": 0.0})
            secondary = det.get("secondary", {"class_name": "unknown", "confidence": 0.0})
            if primary["class_name"] != "unknown":
                llm_result = _llm_classify_for_price(primary, secondary)
                price_cat = llm_result.get("price_category")

    # 还是没有，从文字中提取
    if not price_cat:
        price_cat = _extract_trash_type_from_text(user_text)

    if not price_cat:
        return "⚠️ 请先识别垃圾类型，或在消息中说明垃圾种类（如「塑料回收多少钱」「废铜价格」）。"

    try:
        # 【改造】通过 MCP Tool 调用价格查询
        result_json = call_mcp_tool(
            "get_recycling_price",
            {"trash_type": price_cat},
            status_container,
        )
        if not result_json:
            return "❌ 价格查询服务返回了空结果，请检查 MCP Server 是否启动。"

        result = json.loads(result_json) if isinstance(result_json, str) else result_json

        if result.get("error"):
            return f"❌ 价格查询出错: {result['error']}"

        cat_desc = PRICE_CATEGORY_DESC.get(price_cat, price_cat)
        reply = (
            f"💰 **回收价格查询**\n\n"
            f"| 项目 | 内容 |\n"
            f"|------|------|\n"
            f"| 查询类别 | **{cat_desc}** (`{price_cat}`) |\n"
            f"| 参考价格 | **{result.get('price', 'N/A')}** |\n"
            f"| 地区 | {result.get('area', '全国')} |\n"
            f"| 数据来源 | {result.get('source', '-')} |\n"
            f"| 查询日期 | {result.get('date', '-')} |\n"
        )
        return reply

    except Exception as e:
        return f"❌ 调用价格查询服务失败: {str(e)}\n\n请检查 MCP Server 是否已启动。"


def _handle_report(status_container=None):
    """
    【改造】处理「生成报告」意图 — 通过 MCP Tool "generate_detection_report" 调用。

    流程:
      1. 调用 MCP report Tool 生成结构化报告（Markdown + CSV）
      2. 调用千问生成分析建议（客户端侧 LLM）
      3. 将 CSV 数据存入 session_state
    """
    history = st.session_state.get("detection_history", [])

    if not history:
        return (
            "📋 **暂无识别记录**\n\n"
            "本次会话还没有进行过垃圾识别。\n"
            "请先上传图片并说「识别一下」，识别完成后再来生成报告。"
        )

    # ── Step 1: 【改造】通过 MCP Tool 生成报告 ──
    try:
        report_json_str = call_mcp_tool(
            "generate_detection_report",
            {"history_json": json.dumps(history, ensure_ascii=False)},
            status_container,
        )
        if not report_json_str:
            # MCP 调用失败时尝试本地降级
            report_data = _local_generate_report(history)
        else:
            report_data = json.loads(report_json_str) if isinstance(report_json_str, str) else report_json_str
    except Exception as e:
        _log_service("MCP/generate_detection_report 降级", False, str(e))
        report_data = _local_generate_report(history)

    markdown_text = report_data.get("markdown", "")
    csv_content = report_data.get("csv_content", "")
    recyclable_count = report_data.get("recyclable_count", 0)
    non_recyclable_count = report_data.get("non_recyclable_count", 0)

    # 保存 CSV 到 session_state
    st.session_state["last_report_csv"] = csv_content

    # ── Step 2: 千问智能建议（客户端侧 LLM 调用，保留原有逻辑） ──
    client = _get_llm_client()
    advice = None
    if client:
        try:
            if status_container:
                with status_container.status("🧠 通义千问生成分析建议中...", expanded=True) as s:
                    st.write("正在基于识别数据生成智能建议...")
                    from collections import Counter
                    primary_counter = Counter(report_data.get("primary_counter", {}))
                    advice = _llm_report_advice(client, history, primary_counter)
                    if advice:
                        s.update(label="🧠 建议生成完成", state="complete")
                        _log_service("千问报告建议", True, f"生成{len(advice)}字")
                    else:
                        s.update(label="🧠 建议生成失败", state="error")
            else:
                from collections import Counter
                primary_counter = Counter(report_data.get("primary_counter", {}))
                advice = _llm_report_advice(client, history, primary_counter)
        except Exception as e:
            advice = None
            _log_service("千问报告建议", False, str(e))

    # ── Step 3: 组装最终报告文本 ──
    parts = [markdown_text]

    parts.append("### 💡 智能建议\n")
    if advice:
        parts.append(advice)
    else:
        parts.append(
            f"本次共识别 **{len(history)}** 件垃圾，其中可回收物 **{recyclable_count}** 件，"
            f"不可回收物 **{non_recyclable_count}** 件。\n\n"
            "♻️ **建议**: 可回收物请投入蓝色垃圾桶，不可回收物请投入灰色垃圾桶。"
        )

    parts.append("\n---\n⬇️ **点击下方按钮下载 CSV 报告文件**")

    return "\n".join(parts)


def _local_generate_report(history):
    """
    本地降级报告生成（当 MCP Server 不可用时使用）。

    输入: history (list)
    输出: dict
    """
    from collections import Counter

    total = len(history)
    primary_counter = Counter(r.get("primary_class", "unknown") for r in history)
    recyclable_types = {"glass", "metal", "paper", "plastic"}
    recyclable_count = sum(1 for r in history if r.get("primary_class") in recyclable_types)
    non_recyclable_count = total - recyclable_count

    # 生成 Markdown
    md_parts = [
        f"## 📋 垃圾识别汇总报告\n",
        f"- 总识别次数: **{total}**",
        f"- 可回收物: **{recyclable_count}** 件",
        f"- 不可回收物: **{non_recyclable_count}** 件\n",
        "### 📊 分类统计\n",
        "| 类别 | 数量 |",
        "|------|------|",
    ]
    waste_cn = {"glass": "玻璃", "metal": "金属", "paper": "纸张", "plastic": "塑料", "waste": "其他垃圾"}
    for cls, cnt in primary_counter.most_common():
        cn = waste_cn.get(cls, cls)
        md_parts.append(f"| {cn} ({cls}) | {cnt} |")

    # 生成 CSV
    csv_io = io.StringIO()
    writer = csv.writer(csv_io)
    writer.writerow(["时间", "主模型分类", "主模型置信度", "辅助模型分类", "辅助模型置信度", "回收分类"])
    for r in history:
        writer.writerow([
            r.get("time", ""),
            r.get("primary_cn", ""),
            f"{r.get('primary_confidence', 0):.2%}",
            r.get("secondary_class", ""),
            f"{r.get('secondary_confidence', 0):.2%}",
            r.get("price_category_desc", "无"),
        ])

    return {
        "markdown": "\n".join(md_parts),
        "csv_content": csv_io.getvalue(),
        "recyclable_count": recyclable_count,
        "non_recyclable_count": non_recyclable_count,
        "primary_counter": dict(primary_counter),
    }


def _llm_report_advice(client, history, primary_counter):
    """
    调用千问为识别报告生成智能建议（保留原有逻辑）。

    输入:
        client          : OpenAI 客户端
        history   (list): 检测历史记录
        primary_counter : 主模型类别计数

    输出: str 或 None
    """
    summary_lines = []
    for rec in history:
        p = rec.get("primary_cn", "未知")
        s = rec.get("secondary_class", "无")
        pc = rec.get("price_category_desc", "无")
        summary_lines.append(f"- 主模型:{p}, 辅助模型:{s}, 回收分类:{pc}")

    data_summary = "\n".join(summary_lines)

    prompt = build_report_advice_prompt(len(history), data_summary)

    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL_NAME,
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
    except Exception:
        pass
    return None


def _handle_chat(user_text, status_container=None):
    """
    处理「闲聊」意图 — 优先通过 LLM 智能回复（保留原有逻辑）。

    输入: user_text (str), status_container (st容器, 可选)
    输出: str
    """
    client = _get_llm_client()
    if client:
        try:
            if status_container:
                with status_container.status("💬 LLM 生成回复中...", expanded=False) as s:
                    reply = _llm_chat(client, user_text)
                    if reply:
                        s.update(label="💬 LLM 回复完成", state="complete")
                        _log_service("通义千问智能对话", True, f"回复{len(reply)}字")
                        return reply
                    s.update(label="💬 LLM 返回为空", state="error")
            else:
                reply = _llm_chat(client, user_text)
                if reply:
                    _log_service("通义千问智能对话", True, f"回复{len(reply)}字")
                    return reply
        except Exception as e:
            _log_service("通义千问智能对话", False, str(e))
            if status_container:
                status_container.warning(f"⚠️ LLM 对话失败: {e}，使用预设回复。")

    # 降级：固定引导文本
    _log_service("预设回复（LLM不可用）", True, "")
    return (
        "🤖 你好！我是垃圾分类智能体，我可以帮你：\n\n"
        "1. **识别垃圾** — 在左侧上传图片，然后说「识别一下」\n"
        "2. **查价格** — 说「这个值多少钱」或「查一下塑料回收价格」\n"
        "3. **生成报告** — 说「生成报告」汇总本次识别结果\n"
        "4. **了解原理** — 问我「你的代码原理是什么」或「介绍一下智能体流程」\n\n"
        "试试看吧！😊"
    )


def _llm_chat(client, user_text):
    """
    调用通义千问进行智能对话（保留原有逻辑）。

    输入: client (OpenAI), user_text (str)
    输出: str 或 None
    """
    history_messages = []
    recent_msgs = st.session_state.get("messages", [])[-10:]
    for msg in recent_msgs:
        history_messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    messages = [
        {
            "role": "system",
            "content": CHAT_REPLY_SYSTEM,
        },
        *history_messages,
        {"role": "user", "content": user_text},
    ]

    response = client.chat.completions.create(
        model=QWEN_MODEL_NAME,
        messages=messages,
        max_tokens=1500,
        temperature=0.7,
        timeout=30,
    )

    if response and response.choices:
        return response.choices[0].message.content.strip()
    return None


def _extract_trash_type_from_text(text):
    """
    从用户文字中提取垃圾类别关键词（保留原有逻辑）。

    输入: text (str)
    输出: str 或 None — PRICE_CATEGORY_DESC 中的 key
    """
    # 细分类别关键词（优先级更高）
    fine_cn_to_key = {
        "废铁": "waste_iron", "铁": "waste_iron",
        "废钢": "waste_steel", "钢": "waste_steel",
        "不锈钢": "waste_stainless_steel",
        "废铜": "waste_copper", "铜": "waste_copper",
        "废铝": "waste_aluminum", "铝": "waste_aluminum",
        "锌": "waste_zinc_lead", "铅": "waste_zinc_lead",
        "废锡": "waste_tin", "锡": "waste_tin",
        "废镍": "waste_nickel", "镍": "waste_nickel",
        "ABS": "plastic_abs", "abs": "plastic_abs",
        "尼龙": "plastic_pa", "PA": "plastic_pa",
        "PC": "plastic_pc", "pc塑料": "plastic_pc",
        "PET": "plastic_pet", "pet": "plastic_pet",
        "PP": "plastic_pp", "pp": "plastic_pp",
        "PS": "plastic_ps", "ps": "plastic_ps",
        "PVC": "plastic_pvc", "pvc": "plastic_pvc",
        "PE": "plastic_pe", "pe": "plastic_pe",
        "纸箱": "paper_carton", "纸板": "paper_carton", "瓦楞纸": "paper_carton",
        "橡胶": "rubber_domestic",
    }
    for cn, key in fine_cn_to_key.items():
        if cn in text:
            return key

    # 大类关键词
    broad_cn_to_key = {
        "金属": "metal", "易拉罐": "metal",
        "塑料": "plastic", "塑料瓶": "plastic", "瓶子": "plastic",
        "纸": "paper", "报纸": "paper",
        "玻璃": "glass",
        "垃圾": "waste", "废物": "waste",
    }
    for cn, key in broad_cn_to_key.items():
        if cn in text:
            return key

    for key in PRICE_CATEGORY_DESC:
        if key in text.lower():
            return key

    return None


# ═══════════════════════════════════════════════════════════
#  5. 主入口（保留原有交互流程）
# ═══════════════════════════════════════════════════════════

def main():
    """Streamlit 应用主入口。"""
    init_page()

    # ── 【新增】MCP Server 在线检测 ──
    if st.session_state.get("mcp_server_online") is None:
        st.session_state["mcp_server_online"] = _check_mcp_server_online()

    # ── 【新增】Server 未在线时顶部红色提示 ──
    if not st.session_state.get("mcp_server_online", False):
        st.error(
            "⚠️ **MCP Server 未启动！** 请先在终端执行以下命令启动 Server：\n\n"
            "```\n"
            "cd YA_MCPServer_Template\n"
            "uv run server.py\n"
            "```\n\n"
            "启动后刷新本页面即可。"
        )
        # 提供手动重新检测按钮
        if st.button("🔄 重新检测 MCP Server"):
            st.session_state["mcp_server_online"] = _check_mcp_server_online()
            st.rerun()

    st.title("♻️ 垃圾分类智能体")
    st.caption("上传垃圾图片 → 智能识别 → 查询回收价格 → 语音播报 | MCP Client ↔ Server (SSE)")

    # ── 侧边栏 ──
    image_path = sidebar_upload()

    # ── 延迟 TTS 播放（上一轮 rerun 设置的 pending_tts_text） ──
    pending_text = st.session_state.get("pending_tts_text")
    if pending_text:
        st.session_state["pending_tts_text"] = None
        if st.session_state.get("voice_enabled", True):
            audio_data = text_to_speech(pending_text)
            if audio_data:
                fmt = "audio/wav" if audio_data[:4] == b"RIFF" else "audio/mp3"
                st.audio(audio_data, format=fmt, autoplay=True)

    # ── 渲染聊天历史 ──
    render_chat_history()

    # ── 欢迎语 ──
    if len(st.session_state["messages"]) == 0:
        welcome = (
            "🤖 你好！我是垃圾分类智能体。\n\n"
            "📌 **使用方法：**\n"
            "1. 在左侧上传垃圾图片\n"
            "2. 在下方输入「识别一下」来检测垃圾类型\n"
            "3. 输入「查价格」来查询回收价格\n"
            "4. 输入「生成报告」来汇总识别结果并下载 CSV\n"
            "5. 问我「你的原理是什么」了解智能体架构\n\n"
            "🔊 语音播报默认开启（通过 MCP Server TTS），可在左侧设置中关闭。"
        )
        add_message("assistant", welcome)
        with st.chat_message("assistant"):
            st.markdown(welcome)

    # ── 语音输入（麦克风）— 保留原有逻辑 ──
    voice_text = None
    try:
        from audio_recorder_streamlit import audio_recorder
        col1, col2 = st.columns([1, 10])
        with col1:
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=2.0,
                sample_rate=16000,
                key="voice_recorder",
            )
        with col2:
            st.caption("🎤 点击麦克风录音，松开自动识别")

        if audio_bytes:
            audio_hash = hash(audio_bytes)
            if audio_hash != st.session_state.get("last_audio_hash"):
                st.session_state["last_audio_hash"] = audio_hash
                # 【改造】音频转换后通过 MCP Tool 进行语音识别
                wav_data = _convert_audio_to_wav16k(audio_bytes)
                recognized = speech_to_text(wav_data)
                if recognized:
                    voice_text = recognized
                    st.info(f"🎤 语音识别: {recognized}")
    except ImportError:
        pass  # audio_recorder 未安装，静默跳过
    except Exception as e:
        st.caption(f"⚠️ 语音录入不可用: {e}")

    # ── 文字输入 ──
    user_input = st.chat_input("输入消息，例如「识别一下」或「查价格」...")

    # 语音输入优先级：如果有语音识别结果，使用它
    effective_input = user_input or voice_text

    if effective_input:
        add_message("user", effective_input)
        with st.chat_message("user"):
            st.markdown(effective_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                status_area = st.container()
                reply = handle_user_input(effective_input, image_path, status_area)
            st.markdown(reply)
            # 如果是报告消息，附加 CSV 下载按钮
            csv_data = st.session_state.get("last_report_csv")
            if (
                "⬇️ **点击下方按钮下载 CSV 报告文件**" in reply
                and csv_data
            ):
                st.download_button(
                    label="📥 下载 CSV 报告",
                    data=csv_data.encode("utf-8"),
                    file_name=f"trash_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="dl_csv_current",
                )

        # 延迟 TTS：先让文字渲染，设置 pending 后 rerun，下一轮播放音频
        if st.session_state.get("voice_enabled", True):
            st.session_state["pending_tts_text"] = reply
            st.rerun()


if __name__ == "__main__":
    import traceback as _tb
    try:
        main()
    except Exception as _e:
        _err_msg = _tb.format_exc()
        st.error(f"应用启动异常: {_e}")
        st.code(_err_msg)
        # 同时写到文件方便排查
        with open(os.path.join(PROJECT_ROOT, "app_crash.log"), "w", encoding="utf-8") as _f:
            _f.write(_err_msg)
