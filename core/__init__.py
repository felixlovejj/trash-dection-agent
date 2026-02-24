"""
core — 核心业务逻辑模块
========================
包含垃圾分类智能体的所有业务逻辑实现：

- trash_detection : 双模型（YOLO11 + YOLOv8）垃圾检测
- price_crawler   : 废品回收价格实时爬取
- llm_intent      : 通义千问 LLM 意图识别 & 智能推理
- voice_interact  : 百度语音识别 / DashScope 语音合成
"""
