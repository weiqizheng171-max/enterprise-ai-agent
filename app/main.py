from fastapi import FastAPI
from app.api.router import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="企业级 AI Agent 后端接口服务",
    version="1.0.0"
)

# 注册路由
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    # 本地启动测试：python -m app.main
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)