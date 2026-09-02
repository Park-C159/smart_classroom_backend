# 智能学伴 · 后端服务

基于大模型的数学教材智能答疑系统后端，面向本科生提供智能答疑、学情分析、智能组卷、讨论区、私信与用户管理能力。

> 前端仓库见 [smart_classroom_app](https://github.com/Park-C159/smart_classroom_app)。

## 功能特性

| 模块 | 说明 |
|------|------|
| 智能答疑（RAG） | 知识库（KB）+ 题库（QB）双向量库，全局检索 → 分别重排 → LLM 流式生成（SSE），支持深度思考与联网搜索 |
| 教材解析 | PDF 上传 → MinerU 高精度解析 → 知识树 / 分块 / 题目入库 |
| 知识库管理 | 知识树 CRUD、分块管理、题目（题库）审核与编辑 |
| 组卷与练习 | 独立试题库 + 按题型组卷（作业 / 测试 / 考试 / 自测练习），逐题判分 |
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
└── .env.example                 # 环境变量模板（无真实密钥）
```

## 快速开始

### 1. 环境要求

- Python 3.11+
- （可选）CUDA GPU，用于本地 Embedding / Reranker / MinerU；无 GPU 可改 `CUDA_DEVICE=cpu`
- （可选）Redis，用于 Celery 异步任务

### 2. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> 依赖中的 `faiss-gpu` 需有 CUDA 环境；无 GPU 请改为 `faiss-cpu`。
> `torch` 请按官方指引安装与 CUDA 版本匹配的 wheel。

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入真实密钥（详见下方「环境变量说明」）。**`.env` 已被 `.gitignore` 忽略，切勿提交到仓库。**

### 4. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

- 接口文档（Swagger）：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/api/health`

> 首次启动会自动 `create_all` 建表，并加载 RAG 模型（Embedding + Reranker）。

### 5. Docker 部署

```bash
docker build -t smart-classroom-backend .
docker run --rm -p 8000:8000 --env-file .env smart-classroom-backend
```

## 环境变量说明

| 变量 | 必填 | 说明 |
|------|:---:|------|
| `SECRET_KEY` | ✅ | JWT 签名密钥，生产环境必须改为强随机值 |
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

1. **密钥不入库**：所有密钥（`DEEPSEEK_API_KEY`、`BAIDU_SEARCH_API_KEY`、`NATAPP_AUTHTOKEN` 等）一律放在 `.env`，已被 `.gitignore` 排除；仓库中仅保留 `.env.example` 模板，值为占位符。
2. **强密钥**：`config.py` 中 `SECRET_KEY` 的默认值仅用于本地开发，生产环境务必改为强随机字符串并妥善保管。
3. **用户数据不入库**：`data/`（上传 PDF / 解析结果 / 向量索引）、`output/`（解析产物）、`*.db`（含用户与答题数据）均被 `.gitignore` 排除，请勿强制提交。
4. **日志不入库**：`*.log` 可能包含请求与提示词内容，已被排除。
5. **密码安全**：用户密码使用 bcrypt 哈希存储，接口鉴权基于 JWT，接口按 `student / teacher / admin` 三级 RBAC 控制访问。
6. **提交前自查**：`git add` 后务必 `git status` 确认无 `.env`、`.db`、`data/`、`output/`、`*.log` 被暂存；也可用 `git check-ignore` 校验。

## 许可

本仓库为课程/毕业设计项目，未指定开源许可。引用或复用前请先联系作者。
