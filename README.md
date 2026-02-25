## TrashDetectionAgent — 基于 MCP 架构的垃圾分类智能体

一个基于 Model Context Protocol (MCP) 的 AI 垃圾分类与价格查询智能体，采用 YOLO11 + YOLOv8 双模型融合检测，通义千问 LLM 意图识别与智能推理，百度语音 STT/TTS 交互。

### 组员信息

| 姓名 | 学号 | 分工 |
| :--: | :--: | :--: |
| 娄卫健 | U202414787 | YOLO 模型训练及检测、LLM 集成、任务分配、MCP 服务 |
| 张哲 | U202414647 | 价格爬虫、语音识别tool、语音合成tool、重构、MCP服务 |
| 韩先涛 | U202414626 | 前端 UI、报告生成tool、调试程序、技术文档、MCP服务 |

### Tool 列表

| 工具名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| `detect_waste_type` | 双模型（YOLO11+YOLOv8）垃圾检测 | `image_path: str` 图片路径 | 主/辅模型分类+置信度 | 核心检测工具 |
| `get_recycling_price` | 废品回收价格实时查询 | `trash_type: str` 分类key | 价格+来源+日期 | 支持20+细分品类 |
| `recognize_intent` | LLM 意图识别 | `user_text: str` 用户输入 | 意图类别 | 通义千问+关键词降级 |
| `speech_to_text` | 语音识别 (STT) | `audio_base64: str` 音频 | 识别文字 | 百度语音 16kHz WAV |
| `text_to_speech` | 语音合成 (TTS) | `text: str` 文本 | base64 音频 | 百度+DashScope双方案 |
| `generate_detection_report` | 检测报告生成 | `history_json: str` 历史记录 | Markdown+CSV报告 | 含统计分析 |

### Resource 列表

| 资源名称 | 功能描述 | URI | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| `primary_model_info` | YOLO11 主模型信息 | `config://models/primary` | 权重路径+5大类列表 | 只读配置 |
| `secondary_model_info` | YOLOv8 辅助模型信息 | `config://models/secondary` | 权重路径+18细类列表 | 只读配置 |
| `all_models_info` | 全部模型概要 | `config://models/all` | 双模型元信息 | 聚合视图 |
| `price_categories` | 支持的回收分类 | `config://price/categories` | 分类key→中文映射 | 20+品类 |
| `price_category_detail` | 单个分类详情 | `config://price/category/{key}` | 分类URL+描述 | 模板资源 |

### Prompts 列表

| 指令名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| `intent_classify` | 意图分类指令 | `user_text` | 结构化 prompt | 引导LLM做4分类 |
| `price_classify` | 价格分类推理指令 | 双模型结果 | 结构化 prompt | 引导LLM推理回收类别 |
| `chat_reply` | 闲聊回复指令 | `user_text` | system prompt | 智能体身份设定 |
| `report_advice` | 报告建议指令 | 识别数据摘要 | 结构化 prompt | 环保建议生成 |

### 项目结构

```
TrashDetectionAgent/
├── server.py                  # MCP Server 启动入口（不修改）
├── setup.py                   # 初始化：检查权重、验证依赖、创建目录
├── config.yaml                # 元数据配置（模型路径、LLM参数、爬虫设置）
├── env.yaml                   # SOPS 加密密钥文件
├── pyproject.toml             # 项目依赖与元信息
│
├── core/                      # 核心业务逻辑（异步 + 异常处理）
│   ├── __init__.py
│   ├── trash_detection.py     # YOLO11+YOLOv8 双模型垃圾检测
│   ├── price_crawler.py       # 回收商网废品价格爬取
│   ├── llm_intent.py          # 通义千问意图识别 & 智能推理 & 对话
│   ├── voice_interact.py      # 百度STT/TTS + DashScope CosyVoice
│   └── report_generator.py    # 检测报告 Markdown+CSV 生成
│
├── tools/                     # MCP Tool 实现（@YA_MCPServer_Tool 注册）
│   ├── __init__.py            # 自动扫描注册
│   ├── detect_trash_tool.py   # 废品识别 Tool
│   ├── get_price_tool.py      # 价格查询 Tool
│   ├── intent_recognize_tool.py # 意图识别 Tool
│   └── voice_tool.py          # 语音交互 + 报告生成 Tool
│
├── resources/                 # MCP Resource 实现（@YA_MCPServer_Resource 注册）
│   ├── __init__.py            # 自动扫描注册
│   ├── trash_model_resource.py  # YOLO 模型权重资源
│   └── price_cache_resource.py  # 价格分类映射资源
│
├── prompts/                   # MCP Prompt 实现（@YA_MCPServer_Prompt 注册）
│   ├── __init__.py            # 自动扫描注册
│   ├── intent_prompt.py       # 意图识别 & 价格分类指令
│   └── reply_prompt.py        # 闲聊回复 & 报告建议指令
│
├── modules/                   # 框架模块（不修改）
│   ├── YA_Common/             # 通用工具（配置、日志、中间件）
│   └── YA_Secrets/            # SOPS 密钥管理
│
├── docs/                      # 教程文档
│   ├── prerequisites/         # 环境安装（uv、node.js、sops）
│   └── development-guides/    # 开发指南（MCP 调试、项目结构）
│
├── best_trash_detector.pt     # YOLO11 主模型权重（5大类，需手动放置）
└── best_yolov8_trash.pt       # YOLOv8 辅助模型权重（18细类，需手动放置）
```

### 快速开始

#### 0. 环境要求

在开始之前，请确保已安装以下工具（详见 `docs/prerequisites/` 目录下的教程文档）：

| 工具 | 最低版本 | 用途 | 安装参考 |
| :--: | :------: | :--: | :------: |
| Python | ≥ 3.10 | 运行环境 | [python.org](https://www.python.org/downloads/) |
| uv | 最新版 | Python 包与环境管理 | `docs/prerequisites/uv.mdx` |
| Node.js | LTS | 运行 MCP Inspector 调试工具 | `docs/prerequisites/node.js.mdx` |
| Git | 最新版 | 代码版本管理 | `docs/prerequisites/git.mdx` |
| SOPS + Age | 最新版 | 密钥加密管理（可选） | `docs/prerequisites/sops.mdx` |

#### 1. 克隆项目

```bash
git clone --recursive https://github.com/felixlovejj/trash-dection-agent.git
cd YA_MCPServer_Template
```

> **提示**：`--recursive` 会同时拉取 `modules/YA_Common` 和 `modules/YA_Secrets` 子模块。

#### 2. 安装 uv 包管理器（如未安装）

**Windows (PowerShell)：**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

或使用pip：

pip install uv
```

**macOS / Linux：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后验证：

```bash
uv --version
```

#### 3. 创建虚拟环境并安装依赖

```bash
uv sync
```

> `uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 自动创建 `.venv` 虚拟环境并安装全部依赖。
>
> 

#### 4. 放置 YOLO 模型权重

确保以下两个权重文件已放置在**项目根目录**下：

- `best_trash_detector.pt` — YOLO11 主模型（5 大类检测）
- `best_yolov8_trash.pt` — YOLOv8 辅助模型（18 细分类检测）

> 启动时 `setup.py` 会自动检测权重文件是否存在并给出提示。

#### 5. 启动 MCP Server（SSE 模式）

```bash
uv run server.py
```

启动成功后，服务将运行在 `http://127.0.0.1:12345`（可在 `config.yaml` 的 `transport` 部分修改主机和端口）。

#### 6. 使用 MCP Inspector 调试（新终端）

打开一个**新的终端窗口**，运行：

```bash
npx @modelcontextprotocol/inspector
```

在 MCP Inspector 界面中选择 **SSE** 模式，输入连接地址：

```
http://127.0.0.1:12346
```

即可在线调试所有 Tool、Resource 和 Prompt。

#### 7. 启动 Streamlit UI 界面（新终端）

打开一个**新的终端窗口**，运行：

```bash
uv run streamlit run ui/chat_voice_llm_agent.py
```

Streamlit 将自动在浏览器中打开聊天界面（默认地址 `http://localhost:8501`），支持：
- 文字对话（意图识别 → 垃圾检测 → 价格查询）
- 语音输入 / 语音播报
- 图片上传检测
- 检测报告生成与 CSV 导出

### 其他需要说明的情况



#### 深度学习框架与模型

- **PyTorch**: ✅ 使用 — 作为 YOLO 模型的推理后端
- **Ultralytics**: ✅ 使用 — YOLO 目标检测框架
  - **YOLO11** (`best_trash_detector.pt`): 使用训练的模型进行5 大类检测（Glass, Metal, Paper, Plastic, Waste）
  - **YOLOv8** (`best_yolov8_trash.pt`): 使用训练的模型进行18 细分类检测（Can, Carton, Bottle 等）
- **通义千问 (qwen-turbo)**: ✅ 使用 — 意图识别、价格分类推理、智能对话、报告建议生成
- **百度 AI**: ✅ 使用 — 语音识别 (ASR) + 语音合成 (TTS)
- **DashScope CosyVoice**: ✅ 使用 — TTS 备用方案

#### 技术亮点

| 特性 | 说明 |
| :--: | :--: |
| MCP 标准协议 | Tool / Resource / Prompt 三层分离 |
| 双模型融合 | YOLO11 粗分类 + YOLOv8 细分类 |
| LLM 推理链 | 意图识别 → 检测 → 分类推理 → 价格查询 |
| 全异步架构 | core/ 模块全部 async/await |
| 双 TTS 降级 | 百度 TTS → DashScope CosyVoice |
| SOPS 密钥管理 | Age 加密保护 API 密钥 |
