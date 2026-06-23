"""
RAG 知识库问答系统 - FastAPI 入口

【学习要点】
1. FastAPI 是什么？
   - Python 的现代 Web 框架，专为构建 API 设计
   - 核心优势：
     a. 异步支持（async/await），性能高
     b. 自动生成 API 文档（Swagger UI），访问 /docs 即可查看
     c. 基于 Pydantic 的数据验证，类型安全
     d. 依赖注入系统，代码组织清晰

2. 这个文件的角色：
   - 它是整个后端的"入口"，所有请求都从这里进来
   - 它负责：创建 app、配置中间件、注册路由
   - 各个 service 里的 router 在这里被"挂载"到 app 上

3. CORS 中间件是什么？
   - CORS = Cross-Origin Resource Sharing（跨域资源共享）
   - 前端（Streamlit，端口 8501）和后端（FastAPI，端口 8000）端口不同
   - 浏览器默认禁止跨域请求，CORS 中间件就是允许跨域
   - allow_origins=["*"] 表示允许所有来源（开发阶段用，生产要限制）

4. 启动方式：
   - 开发：python main.py（带热重载）
   - 生产：uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# ============================================================
# 创建 FastAPI 应用实例
# ============================================================
# title, version 等参数会显示在自动生成的 API 文档中
# 访问 http://localhost:8000/docs 查看 Swagger UI
# 访问 http://localhost:8000/redoc 查看 ReDoc 格式文档

app = FastAPI(
    title="RAG 知识库问答系统",
    version="1.0",
    description="一个基于 RAG 技术的知识问答系统",
)
# ============================================================
# 创建 FastAPI 应用实例
# ============================================================
# title, version 等参数会显示在自动生成的 API 文档中
# 访问 http://localhost:8000/docs 查看 Swagger UI
# 访问 http://localhost:8000/redoc 查看 ReDoc 格式文档

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
# ============================================================
# 注册路由
# ============================================================
# 每个服务模块有自己的 router（APIRouter 实例）
# include_router 就是把子路由"挂载"到主 app 上
# 这样代码可以按模块拆分，不用全写在一个文件里
# ============================================================
from services.document_service import router as document_router
from services.rag_service import router as rag_router
from services.memory_service import router as memory_router

app.include_router(document_router)
app.include_router(rag_router)
app.include_router(memory_router)

# ============================================================
# 基础路由
# ============================================================

@app.get('/')
async def root():
    """
    根路径，返回系统信息
    可用于健康检查：如果这个接口能返回，说明服务在运行
    """
    return {
        "message": "RAG 知识库问答系统已启动",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get('/health')
async def health_check():
    """
    健康检查接口
    部署时监控系统会用这个接口判断服务是否正常
    """
    return {"status": "ok"}

# ============================================================
# 启动入口
# ============================================================
# python main.py 时执行
# uvicorn.run 参数说明：
#   "main:app" → main.py 文件中的 app 变量
#   host="0.0.0.0" → 监听所有网络接口（允许外部访问）
#   port=8000 → 端口号
#   reload=True → 代码修改后自动重启（仅开发环境用）
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
