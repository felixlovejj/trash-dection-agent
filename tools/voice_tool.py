"""
Voice Tool (tools/voice_tool.py)
================================
MCP Tool — 语音交互工具

- STT: 音频 → 文字（百度语音识别）
- TTS: 文字 → 音频（百度 TTS / DashScope CosyVoice）
"""

import base64
from typing import Any, Dict

from tools import YA_MCPServer_Tool


@YA_MCPServer_Tool(
    name="speech_to_text",
    title="语音识别工具 (STT)",
    description="将 WAV 音频转为文字，使用百度语音识别（16kHz 单声道 WAV）",
)
async def speech_to_text_tool(audio_base64: str) -> Dict[str, Any]:
    """
    将 base64 编码的 WAV 音频转为文字。

    Args:
        audio_base64 (str): base64 编码的 WAV 音频数据

    Returns:
        Dict[str, Any]: {"text": str, "error": str}

    Example:
        {"text": "识别一下这个垃圾", "error": ""}
    """
    try:
        from core.voice_interact import speech_to_text, convert_audio_to_wav16k

        audio_bytes = base64.b64decode(audio_base64)
        wav_bytes = await convert_audio_to_wav16k(audio_bytes)
        text = await speech_to_text(wav_bytes)

        if text:
            return {"text": text, "error": ""}
        else:
            return {"text": "", "error": "语音识别未返回结果"}

    except Exception as e:
        return {"text": "", "error": f"语音识别异常: {e}"}


@YA_MCPServer_Tool(
    name="text_to_speech",
    title="语音合成工具 (TTS)",
    description="将文字合成为语音音频，返回 base64 编码的音频数据",
)
async def text_to_speech_tool(text: str) -> Dict[str, Any]:
    """
    将文字合成为语音。

    Args:
        text (str): 要合成的文本

    Returns:
        Dict[str, Any]: {"audio_base64": str, "format": str, "error": str}

    Example:
        {"audio_base64": "UklGRi...", "format": "wav", "error": ""}
    """
    try:
        from core.voice_interact import text_to_speech

        audio_data = await text_to_speech(text)

        if audio_data:
            encoded = base64.b64encode(audio_data).decode("utf-8")
            fmt = "wav" if audio_data[:4] == b"RIFF" else "mp3"
            return {"audio_base64": encoded, "format": fmt, "error": ""}
        else:
            return {"audio_base64": "", "format": "", "error": "语音合成未返回数据"}

    except Exception as e:
        return {"audio_base64": "", "format": "", "error": f"语音合成异常: {e}"}


@YA_MCPServer_Tool(
    name="generate_detection_report",
    title="检测报告生成工具",
    description="根据检测历史生成 Markdown 报告 + CSV 导出数据",
)
async def generate_detection_report_tool(history_json: str) -> Dict[str, Any]:
    """
    根据检测历史 JSON 生成汇总报告。

    Args:
        history_json (str): JSON 格式的检测历史数组

    Returns:
        Dict[str, Any]: 报告数据
            {
                "total": int,
                "markdown": str,
                "csv_content": str,
                "recyclable_count": int,
                "non_recyclable_count": int,
                "primary_counter": dict,
                "error": str
            }
    """
    try:
        import json
        from core.report_generator import generate_report

        history = json.loads(history_json)
        result = await generate_report(history)
        result["error"] = ""
        return result

    except Exception as e:
        return {
            "total": 0,
            "markdown": "",
            "csv_content": "",
            "recyclable_count": 0,
            "non_recyclable_count": 0,
            "primary_counter": {},
            "error": f"报告生成异常: {e}",
        }
