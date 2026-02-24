from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

def get_llm():
    return ChatOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        model="deepseek-chat",  # 指定使用 DeepSeek 的对话模型
        temperature=0.1
    )

def get_embeddings():
    # 工业界标准做法：LLM 用云端，Embedding 用本地开源模型
    # 第一次运行时会自动下载 BAAI (智源研究院) 的中文小模型
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )