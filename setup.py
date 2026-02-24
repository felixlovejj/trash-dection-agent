"""
Setup (setup.py)
================
项目初始化：检查 YOLO 权重文件、验证依赖、初始化数据目录。
在 server.py 启动时自动调用。
"""

import os
import sys

from modules.YA_Common.utils.logger import get_logger
from modules.YA_Common.utils.config import get_config

logger = get_logger("setup")

# YOLO 权重文件的下载地址（如有 GitHub Releases 可配置）
WEIGHT_DOWNLOAD_URLS = {
    "best_trash_detector.pt": None,   # YOLO11 主模型 — 需手动放置
    "best_yolov8_trash.pt": None,     # YOLOv8 辅助模型 — 需手动放置
}


def _check_weights():
    """
    检查 YOLO 模型权重文件是否存在。

    如果权重文件不存在且配置了下载 URL，尝试自动下载。
    否则打印警告提示用户手动放置。
    """
    project_root = os.path.dirname(os.path.abspath(__file__))

    primary_weight = get_config("models.primary_weight", "best_trash_detector.pt")
    secondary_weight = get_config("models.secondary_weight", "best_yolov8_trash.pt")

    for weight_name in [primary_weight, secondary_weight]:
        weight_path = os.path.join(project_root, weight_name)

        if os.path.exists(weight_path):
            size_mb = os.path.getsize(weight_path) / (1024 * 1024)
            logger.info(f"✅ 权重文件已就绪: {weight_name} ({size_mb:.1f} MB)")
        else:
            download_url = WEIGHT_DOWNLOAD_URLS.get(weight_name)
            if download_url:
                logger.info(f"⬇️ 正在下载权重: {weight_name} ...")
                try:
                    import requests
                    resp = requests.get(download_url, stream=True, timeout=120)
                    resp.raise_for_status()
                    with open(weight_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"✅ 权重下载完成: {weight_name}")
                except Exception as e:
                    logger.error(f"❌ 权重下载失败: {weight_name} — {e}")
                    logger.warning(f"   请手动将 {weight_name} 放置到: {project_root}")
            else:
                logger.warning(
                    f"⚠️ 权重文件缺失: {weight_name}\n"
                    f"   请将文件放置到: {weight_path}"
                )


def _check_dependencies():
    """检查核心依赖是否已安装。"""
    required = {
        "ultralytics": "YOLO 目标检测",
        "openai": "通义千问 LLM (OpenAI 兼容接口)",
        "requests": "网络请求",
        "beautifulsoup4": "HTML 解析",
        "pandas": "数据处理",
    }

    for pkg, desc in required.items():
        try:
            __import__(pkg if pkg != "beautifulsoup4" else "bs4")
            logger.debug(f"✅ {pkg} ({desc})")
        except ImportError:
            logger.warning(f"⚠️ 缺少依赖: {pkg} ({desc}) — pip install {pkg}")


def _init_data_dirs():
    """初始化数据目录。"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    dirs = ["logs", "data"]
    for d in dirs:
        path = os.path.join(project_root, d)
        os.makedirs(path, exist_ok=True)
        logger.debug(f"📁 目录就绪: {d}/")


def setup():
    """
    项目初始化入口。

    在 server.py 启动时自动调用，执行以下操作：
    1. 初始化数据目录
    2. 检查 YOLO 模型权重
    3. 验证核心依赖
    """
    try:
        logger.info("=" * 50)
        logger.info("🚀 垃圾分类智能体 — 初始化中...")
        logger.info("=" * 50)

        _init_data_dirs()
        _check_weights()
        _check_dependencies()

        logger.info("✅ Setup 完成 — 服务器即将启动")

    except Exception as e:
        logger.error(f"❌ Setup 失败: {e}")
        raise e
