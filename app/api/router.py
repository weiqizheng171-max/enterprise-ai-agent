from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas.models import ChatRequest, ChatResponse
from app.services.rag_service import process_rag_query, ingest_text_to_vectorstore
from app.services.agent_service import process_agent_query

api_router = APIRouter()

# ================= 新增：文件上传入库接口 =================
@api_router.post("/knowledge/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # 读取上传的文件内容
        content = await file.read()
        # 这里为了演示，我们假设上传的是 .txt 纯文本文件 (UTF-8编码)
        text_content = content.decode("utf-8")
        
        # 调用 RAG 服务进行切片和向量化
        chunk_count = ingest_text_to_vectorstore(text_content)
        
        return {
            "status": "success", 
            "filename": file.filename,
            "message": f"成功存入知识库，并将其切分为了 {chunk_count} 个向量片段。"
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="目前仅支持 UTF-8 编码的 TXT 文件")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ==========================================================

@api_router.post("/chat/rag", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    # ... (保持原有代码不变)
    try:
        answer = process_rag_query(request.query)
        return ChatResponse(answer=answer, source="RAG Knowledge Base")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/chat/agent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        # 注意这里：要把 request.session_id 也传进去！
        answer = process_agent_query(request.query, request.session_id)
        return ChatResponse(answer=answer, source="Agent Tool Calling")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))