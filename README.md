# 🚀 Enterprise AI Agent API (智能企业知识库与自动化 Agent)

基于 FastAPI + LangChain 构建的企业级 AI 智能体微服务。项目采用**异构大模型架构**，实现了文档的本地私有化检索（RAG）与云端大模型的智能规划与工具调用（Agent）。

## ✨ 核心架构与亮点

* **算力分层解耦**：采用 `云端大模型 (DeepSeek) + 本地开源模型 (BAAI/bge-small-zh)` 的混合架构，兼顾了顶级推理能力与企业知识库的绝对隐私安全。
* **企业级 RAG 检索**：实现本地文档的长文本动态切片 (Chunking) 与向量化入库，通过 FAISS 向量检索消除大模型幻觉。
* **Tool Calling Agent**：基于最新版 LangChain 构建支持工具调用的智能体，能够自主思考并调用外部 API（如天气查询、薪资计算）完成复杂任务。
* **提示词工程 (Prompt Engineering)**：严格落实代码与提示词解耦，统一在 `core/prompts.py` 中管理，便于后续接入大模型运维 (LLMOps) 平台。

## 🛠️ 技术栈

* **后端框架**: FastAPI, Pydantic, Uvicorn
* **AI/LLM 框架**: LangChain (Core/Community/OpenAI)
* **模型接入**: DeepSeek (生成), HuggingFace bge-small-zh (本地向量化)
* **向量引擎**: FAISS

## 📦 快速启动

1. 克隆项目并安装依赖：
   ```bash
   pip install -r requirements.txt