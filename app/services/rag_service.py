from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter # 新增：文本切分器
from app.services.llm_service import get_llm, get_embeddings
from app.core.prompts import rag_prompt

# 初始化的默认数据（系统启动时带的底库）
dummy_docs = [
    "公司规定：转正员工每年享有5天带薪年假。",
    "报销流程：餐饮费报销需提供发票，并在每月25日前提交给财务部。"
]

# 全局初始化向量库 (注意：实际企业中这里会连接 Milvus/Qdrant 等独立数据库)
vectorstore = FAISS.from_texts(dummy_docs, get_embeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ================= 新增：处理上传文件并存入向量库 =================
def ingest_text_to_vectorstore(text_content: str) -> int:
    """
    接收长文本，进行切块 (Chunking) 后转化为向量存入 FAISS 库。
    """
    # 1. 实例化切分器 (面试考点：为什么要切分？因为大模型上下文有限，且切分后检索更精准)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,    # 每个文本块最大 200 个字符
        chunk_overlap=20   # 块与块之间重叠 20 个字符，防止上下文断裂
    )
    
    # 2. 切分文本
    chunks = text_splitter.split_text(text_content)
    
    # 3. 将切分好的文本块灌入向量数据库 (Embedding 模型会自动将其转为向量)
    vectorstore.add_texts(chunks)
    
    # 返回切分了多少块，用于给前端提示
    return len(chunks)
# =================================================================

def process_rag_query(query: str) -> str:
    llm = get_llm()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(query)