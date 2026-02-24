"""
Voice Interaction Core (core/voice_interact.py)
================================================
语音交互模块：

- STT (语音→文字): 百度语音识别 (AipSpeech)
- TTS (文字→语音): 百度 TTS（主）+ DashScope CosyVoice（备用）

所有函数均支持 async/await 异步调用，并包含 try-except 异常处理。
"""

import io
import wave
import struct
import asyncio
from typing import Optional

from modules.YA_Common.utils.logger import get_logger
from modules.YA_Common.utils.config import get_config

logger = get_logger("core.voice_interact")

# ── 百度 AipSpeech 客户端缓存 ────────────────────────────
_aip_client = None


def _get_aip_client():
    """
    获取百度 AipSpeech 客户端（惰性初始化 + 缓存）。

    Returns:
        AipSpeech 实例，或 None
    """
    global _aip_client
    if _aip_client is not None:
        return _aip_client

    try:
        # 优先从 SOPS 获取密钥
        app_id = api_key = secret_key = None
        try:
            from modules.YA_Secrets.secrets_parser import get_secret
            app_id = get_secret("baidu_app_id")
            api_key = get_secret("baidu_api_key")
            secret_key = get_secret("baidu_secret_key")
        except Exception:
            pass

        # 降级：从 config.yaml 获取
        if not app_id:
            app_id = get_config("secrets.baidu_app_id", None)
        if not api_key:
            api_key = get_config("secrets.baidu_api_key", None)
        if not secret_key:
            secret_key = get_config("secrets.baidu_secret_key", None)

        if not (app_id and api_key and secret_key):
            logger.warning("未配置百度语音 API 密钥")
            return None

        from aip import AipSpeech
        _aip_client = AipSpeech(str(app_id), str(api_key), str(secret_key))
        logger.info("百度 AipSpeech 客户端初始化成功")
        return _aip_client

    except ImportError:
        logger.error("百度 AIP SDK 未安装: pip install baidu-aip")
        return None
    except Exception as e:
        logger.error(f"百度 AipSpeech 初始化失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  1. 语音识别（STT）
# ═══════════════════════════════════════════════════════════

async def speech_to_text(audio_bytes: bytes) -> Optional[str]:
    """
    将音频字节流转为文字（百度语音识别）。

    Args:
        audio_bytes (bytes): WAV 格式音频数据 (16kHz, 单声道)

    Returns:
        str | None: 识别出的文字
    """
    try:
        client = _get_aip_client()
        if not client:
            logger.warning("百度语音 API 不可用")
            return None

        result = await asyncio.to_thread(
            client.asr, audio_bytes, "wav", 16000, {"dev_pid": 1537}
        )

        if result and result.get("err_no") == 0:
            text = result["result"][0]
            logger.info(f"语音识别结果: {text}")
            return text
        else:
            err_msg = result.get("err_msg", "未知错误") if result else "返回为空"
            logger.warning(f"语音识别失败: {err_msg}")
            return None

    except Exception as e:
        logger.error(f"语音识别异常: {e}")
        return None


async def convert_audio_to_wav16k(audio_bytes: bytes) -> bytes:
    """
    将音频转为 16kHz 单声道 WAV（百度 ASR 要求的格式）。

    Args:
        audio_bytes (bytes): 原始音频数据

    Returns:
        bytes: 16kHz WAV 音频
    """
    try:
        return await asyncio.to_thread(_sync_convert_audio, audio_bytes)
    except Exception:
        return audio_bytes


def _sync_convert_audio(audio_bytes: bytes) -> bytes:
    """同步版本的音频转换。"""
    try:
        with io.BytesIO(audio_bytes) as src_io:
            with wave.open(src_io, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

        # 解码样本
        if sampwidth == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = list(struct.unpack(fmt, raw_data))
        elif sampwidth == 1:
            samples = [((b - 128) << 8) for b in raw_data]
        else:
            fmt = f"<{n_frames * n_channels}i"
            int_samples = struct.unpack(fmt, raw_data)
            samples = [s >> 16 for s in int_samples]

        # 多声道 → 单声道
        if n_channels > 1:
            samples = samples[::n_channels]

        # 重采样到 16kHz
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


# ═══════════════════════════════════════════════════════════
#  2. 语音合成（TTS）
# ═══════════════════════════════════════════════════════════

async def text_to_speech(text: str) -> Optional[bytes]:
    """
    将文字转为语音音频。

    策略：百度 TTS 优先 → DashScope CosyVoice 备用 → None。

    Args:
        text (str): 要合成的文字

    Returns:
        bytes | None: 音频数据 (WAV/MP3)
    """
    # 清理文字
    clean_text = text.replace("*", "").replace("#", "").replace("|", "")
    clean_text = clean_text.replace("---", "").replace("```", "").replace("  ", " ")
    lines = [l for l in clean_text.split("\n") if not l.strip().startswith("|-")]
    clean_text = "\n".join(lines).strip()
    if len(clean_text) > 500:
        clean_text = clean_text[:500] + "..."
    if not clean_text:
        return None

    # 方案1: 百度 TTS
    try:
        audio = await _tts_baidu(clean_text)
        if audio:
            return audio
    except Exception as e:
        logger.warning(f"百度 TTS 失败: {e}")

    # 方案2: DashScope CosyVoice
    try:
        audio = await _tts_dashscope(clean_text)
        if audio:
            return audio
    except Exception as e:
        logger.warning(f"DashScope TTS 失败: {e}")

    return None


async def _tts_baidu(text: str) -> Optional[bytes]:
    """百度 TTS 语音合成。"""
    try:
        client = _get_aip_client()
        if not client:
            return None

        result = await asyncio.to_thread(
            client.synthesis, text, "zh", 1,
            {"per": 4, "spd": 5, "vol": 9, "pit": 5, "aue": 6}
        )

        if isinstance(result, bytes):
            logger.info(f"百度 TTS 生成 {len(result)} 字节")
            return result
        else:
            logger.warning(f"百度 TTS 返回非字节: {result}")
            return None

    except Exception as e:
        logger.error(f"百度 TTS 异常: {e}")
        return None


async def _tts_dashscope(text: str) -> Optional[bytes]:
    """DashScope CosyVoice TTS（备用方案）。"""
    try:
        # 获取 API Key
        api_key = None
        try:
            from modules.YA_Secrets.secrets_parser import get_secret
            api_key = get_secret("dashscope_api_key")
        except Exception:
            pass
        if not api_key:
            api_key = get_config("secrets.dashscope_api_key", None)
        if not api_key:
            return None

        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer

        dashscope.api_key = api_key

        def _synthesize():
            synthesizer = SpeechSynthesizer(
                model="cosyvoice-v1", voice="longxiaochun"
            )
            return synthesizer.call(text)

        audio_data = await asyncio.to_thread(_synthesize)
        if audio_data and len(audio_data) > 0:
            logger.info(f"DashScope TTS 生成 {len(audio_data)} 字节")
            return audio_data

        return None

    except ImportError:
        logger.warning("dashscope 版本不支持 tts_v2")
        return None
    except Exception as e:
        logger.error(f"DashScope TTS 异常: {e}")
        return None
