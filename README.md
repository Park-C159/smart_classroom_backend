# 智能学伴 · 后端服务（Smart Classroom Backend）

> 面向本科生数学教材的智能答疑系统后端，基于大模型与 RAG（检索增强生成）提供智能答疑、学情分析、智能组卷、讨论区与私信能力。

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

配套前端：[smart_classroom_app](https://github.com/Park-C159/smart_classroom_app)

## 简介

智能学伴（Smart Classroom）是一套面向高校数学课程的智能教学辅助系统。本仓库为**后端服务**，负责：

- 将教材 PDF 解析为结构化知识树与题库，构建「知识库 + 题库」双向量索引；
- 基于 RAG 提供流式智能答疑（支持深度思考与联网搜索）；
- 支撑学情分析、智能组卷（作业 / 测试 / 考试 / 自测）、讨论区与师生私信。

## 功能特性

| 模块 | 说明 |
|------|------|
| 智能答疑（RAG） | 知识库（KB）+ 题库（QB）双向量库，全局检索 → 分别重排 → LLM 流式生成（SSE），支持深度思考与联网搜索 |
| 教材解析 | PDF 上传 → MinerU 高精度解析 → 知识树 / 分块 / 题目入库 |
| 知识库管理 | 知识树 CRUD、分块管理、题目（题库）审核与编辑 |
| 组卷与练习 | 独立试题库 + 按题型组卷（作业 / 测试 / 考试 / 自测练习），逐题判分，简答题支持 LLM 自动评分 |
| 学情分析 | 按知识点逐题更新掌握度（EWMA），班级 / 个人学情统计 |
| 讨论区 & 私信 | 发帖 / 回帖 / 点赞 / 置顶，学生与教师一对一私信 |
| 语音识别 | Whisper 本地转写（按需加载 / 空闲卸载） |
| 用户与权限 | JWT 认证 + RBAC（student / teacher / admin），bcrypt 密码哈希 |

## 技术栈

| 层 | 技术 |
|----|------|
| Web 框架 | FastAPI（Python 3.11+） |
| ORM | SQLAlchemy 2.x（async）+ SQLite（生产可迁移 PostgreSQL） |
| 向量检索 | FAISS + BGE-M3（Embedding）+ BGE-Reranker-v2-m3（重排） |
| LLM | DeepSeek API（OpenAI 兼容，支持思考模式流式输出） |
| PDF 解析 | MinerU（magic-pdf，独立环境运行） |
| 语音 | faster-whisper（本地 CPU） |
| 异步任务 | Celery + Redis（可选） |

## 目录结构

```
backend/
├── app/
│   ├── main.py                  # FastAPI 入口，路由注册
│   ├── config.py                # 环境变量配置（Settings）
│   ├── dependencies.py          # 公共依赖
│   ├── api/                     # API 路由
│   │   ├── auth.py              # 登录 / 注册 / 刷新 JWT
│   │   ├── users.py             # 用户管理 CRUD
│   │   ├── rag.py               # 答疑 SSE 流式接口（KB + QB 检索）
│   │   ├── document.py          # 教材上传 / 解析 / PDF 查看
│   │   ├── knowledge.py         # 知识树 CRUD / 分块管理
│   │   ├── analytics.py         # 学情分析 / 系统概览
│   │   ├── exam.py              # 组卷 / 批改
│   │   ├── test_bank.py         # 独立试题库 CRUD
│   │   ├── papers.py            # 组卷（作业/测试/考试）/ 提交 / 批改
│   │   ├── messages.py          # 私信
│   │   ├── discussion.py        # 讨论区 CRUD
│   │   ├── feedback.py          # 反馈建议
│   │   ├── speech.py            # 语音识别
│   │   ├── subjects.py          # 学科管理
│   │   └── upload.py            # 文件上传 / Excel 导入
│   ├── models/__init__.py       # SQLAlchemy 模型（全部表）
│   ├── schemas/                 # Pydantic 请求 / 响应模型
│   ├── services/                # 业务服务
│   │   ├── rag_service.py       # 全局检索 + FAISS 索引 + 重排
│   │   ├── llm_service.py       # DeepSeek 流式调用 + 简答判分
│   │   ├── knowledge_tree_service.py
│   │   ├── document_processor.py
│   │   ├── file_processor.py
│   │   ├── chapter_service.py
│   │   ├── stt_service.py       # Whisper 封装
│   │   ├── vlm_service.py       # VLM 图片描述
│   │   ├── web_search.py        # 联网搜索（百度 AI 搜索 / Bing 回退）
│   │   ├── gpu_manager.py       # GPU 显存调度
│   │   └── mastery_service.py   # 掌握度更新 helper
│   ├── core/
│   │   ├── database.py          # 数据库引擎 & session factory
│   │   ├── security.py          # JWT 认证 & RBAC
│   │   └── redis_client.py      # Redis 连接
│   └── tasks/                   # Celery 异步任务 / 定时调度
├── alembic/                     # 数据库迁移骨架
├── requirements.txt
├── Dockerfile
├── .env.example                 # 环境变量模板（无真实密钥）
└── LICENSE
```

## 快速开始

> 以下步骤假设你已经**下载并解压了本仓库的源码压缩包**（或 `git clone` 了本仓库）。

### 0. 运行前准备

| 准备项 | 是否必需 | 说明 |
|--------|:---:|------|
| Python 3.11+ | ✅ | 命令行执行 `python --version` 检查 |
| DeepSeek API Key | ✅ | 在 [platform.deepseek.com](https://platform.deepseek.com) 注册并创建 API Key（智能答疑与组卷判分都依赖它） |
| CUDA GPU | 可选 | 用于本地 Embedding / Reranker / MinerU 加速；没有 GPU 可改用 CPU |
| Redis | 可选 | 仅 Celery 异步任务需要，不启动任务可不装 |

### 1. 安装依赖

```bash
# 进入后端目录
cd smart_classroom_backend

# 创建并激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> - 没有 NVIDIA GPU 时，把 `requirements.txt` 里的 `faiss-gpu` 改为 `faiss-cpu`，再执行安装。
> - `torch` 建议按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 指引，安装与你的 CUDA 版本匹配的 wheel。

### 2. 配置环境变量（密钥 / 隐私保护）

```bash
cp .env.example .env
```

编辑 `.env`，至少修改下面两项：

```ini
# JWT 签名密钥 —— 务必改成强随机值，不要用默认值
SECRET_KEY=用下面的命令生成一段

# 你的 DeepSeek API Key
DEEPSEEK_API_KEY=sk-你的真实key
```

生成强随机 `SECRET_KEY`：

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

其余变量（百度搜索 Key、内网穿透 token、MinerU 路径等）按需填写，不用的留空即可。详见下方「环境变量说明」。

> ⚠️ **`.env` 是你的私密文件**，包含真实密钥，请勿提交到 Git、勿上传到任何公开平台、勿发给他人。

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

- 接口文档（Swagger）：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/api/health`

> 首次启动会自动建表，并加载 RAG 模型（Embedding + Reranker），第一次启动会稍慢，属正常现象。

### 4. Docker 部署（可选）

```bash
docker build -t smart-classroom-backend .
docker run --rm -p 8000:8000 --env-file .env smart-classroom-backend
```

## 环境变量说明

| 变量 | 必填 | 说明 |
|------|:---:|------|
| `SECRET_KEY` | ✅ | JWT 签名密钥，**必须改为强随机值** |
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek 开放平台 API Key |
| `DEEPSEEK_BASE_URL` | - | DeepSeek API 地址，默认 `https://api.deepseek.com/v1` |
| `DEEPSEEK_CHAT_MODEL` | - | 对话模型，默认 `deepseek-v4-pro` |
| `DEEPSEEK_REASONER_MODEL` | - | 推理（深度思考）模型 |
| `DATABASE_URL` | - | 默认 `sqlite+aiosqlite:///./math_qa.db` |
| `REDIS_URL` | - | 默认 `redis://localhost:6379/0` |
| `BAIDU_SEARCH_API_KEY` | - | 百度 AI 搜索 Key，留空则回退 Bing 抓取 |
| `VLM_MODEL` / `VLM_BASE_URL` | - | 视觉模型（图片识别），留空回退 DeepSeek 配置 |
| `NATAPP_AUTHTOKEN` / `NATAPP_BIN` | - | 内网穿透（可选） |
| `CUDA_DEVICE` | - | `cuda:0` 或 `cpu` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | - | 模型离线加载开关 |
| `MINERU_MODE` / `MINERU_API_URL` / `MINERU_BIN` | - | MinerU 解析配置（可选） |
| `CORS_ORIGINS` | - | 允许的跨域来源列表 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` / `REFRESH_TOKEN_EXPIRE_DAYS` | - | JWT 有效期 |

## API 概览

所有接口前缀为 `/api`，认证方式为 `Authorization: Bearer <access_token>`。

| 前缀 | 说明 | 权限 |
|------|------|------|
| `/api/auth` | 注册 / 登录 / 刷新令牌 | 公开 |
| `/api/rag` | 智能答疑（SSE 流式） | 登录 |
| `/api/users` | 用户管理 | 管理员 / 教师（部分） |
| `/api/subjects` | 学科管理 | 管理员 / 教师 |
| `/api/documents` | 教材上传 / 解析 / 查看 | 管理员 / 教师 |
| `/api/knowledge` | 知识树 / 分块 / 题库 | 管理员 / 教师 |
| `/api/test-bank` | 独立试题库 CRUD | 管理员 / 教师 |
| `/api/papers` | 组卷 / 发布 / 提交 / 批改 | 教师 + 学生 |
| `/api/exam` | 自测练习 | 登录 |
| `/api/analytics` | 学情分析 / 系统概览 | 登录 / 管理员 |
| `/api/discussion` | 讨论区 | 登录 |
| `/api/messages` | 私信 | 登录 |
| `/api/speech` | 语音转写 | 登录 |
| `/api/feedback` | 反馈建议 | 登录 |
| `/api/upload` | 文件上传 / Excel 批量导入 | 管理员 / 教师 |

## 隐私与安全

1. **密钥只放在 `.env`**：所有 API Key（DeepSeek、百度搜索、natapp 等）都在 `.env` 中配置，源码里不含真实密钥；`.env` 已被 `.gitignore` 忽略，不会随代码上传。
2. **修改 `SECRET_KEY`**：`config.py` 里给的是占位默认值，部署前务必在 `.env` 里改成强随机值，否则 JWT 可被伪造。
3. **前端不存密钥**：密钥一律只配在后端，前端代码与仓库中不出现任何 Key。
4. **用户数据留在本地**：数据库（`*.db`）、上传的 PDF、解析结果、向量索引都存放在本机 `data/` 目录，不会进入仓库；备份或迁移时请自行妥善保管，不要外泄。
5. **生产部署建议**：使用 HTTPS；收紧 `CORS_ORIGINS` 只放自己的域名；数据库从 SQLite 迁移到 PostgreSQL。

## 贡献指南

欢迎提交 Issue 与 Pull Request。

1. Fork 本仓库，克隆到本地；
2. 新建分支：`git checkout -b feature/your-feature`；
3. 提交改动，遵循既有代码风格（中文注释、`app/api` 路由 + `app/services` 服务分层）；
4. 推送到你的 Fork，发起 Pull Request 到 `main` 分支。

## 开源许可

本项目采用 [MIT License](LICENSE)。

## 相关项目

- 前端：[smart_classroom_app](https://github.com/Park-C159/smart_classroom_app)
