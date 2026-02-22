from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="用户的查询问题", example="公司年假怎么休？")
    # 把这行加到这里来，合二为一
    session_id: str = "default_user_001" 

class ChatResponse(BaseModel):
    answer: str
    source: str = "Knowledge Base"