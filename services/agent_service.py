"""
智能数据分析 Agent - Agent 调用封装

【学习要点】
1. 这个文件的作用：
   - 把 LangGraph 工作流包装成 HTTP API
   - 前端（Streamlit）通过 HTTP 请求调用这些 API
   - 这样前后端分离，可以独立开发和部署

2. 为什么需要 API 封装层？
   - LangGraph 的工作流是 Python 函数调用
   - 前端是 Streamlit（也是 Python），理论上可以直接调用
   - 但 API 封装的好处：
     a. 前后端解耦（可以换前端，如 React）
     b. 多个前端可以共享同一个后端
     c. API 文档自动生成（FastAPI 的优势）
     d. 方便做权限控制、日志记录等

3. 文件上传的处理流程：
   - 前端通过 multipart/form-data 上传文件
   - FastAPI 用 UploadFile 接收
   - 保存到本地文件系统
   - 返回文件路径给前端
   - 前端拿着文件路径调用分析接口

4. Pydantic 模型的作用：
   - 定义请求和响应的数据格式
   - FastAPI 自动做数据验证
   - 自动生成 API 文档
"""
import uuid
import os
from pathlib import Path

from fastapi import APIRouter,UploadFile,File,HTTPException
from pydantic import BaseModel,Field
from typing import Optional

from graph.workflow import run_analysis
from config import UPLLOAD_DIR,MAX_UPLOAD_SIZE


router = APIRouter(prefix='/agent',tags=['数据分析Agent'])
# ============================================================
# 请求/响应模型
# ============================================================
class AnalyzeRequest(BaseModel):
    """
    数据分析请求

    前端调用分析接口时需要传：
    - file_path: 上传后返回的文件路径
    - request: 自然语言描述的分析需求
    """
    file_path: str = Field(..., description="数据文件路径")
    request: str = Field(..., description="自然语言分析需求")

class AnalyzeResponse(BaseModel):
    """
    数据分析响应

    工作流执行完成后返回：
    - report: 分析报告（Markdown 格式）
    - chart_paths: 生成的图表文件路径列表
    - success: 是否分析成功
    - error: 错误信息（如果失败）
    """
    report: str = Field("", description="分析报告")
    chart_paths: list[str] = Field(default_factory=list, description="图表路径列表")
    success: bool = Field(True, description="是否成功")
    error: str = Field("", description="错误信息")

class UploadResponse(BaseModel):
    """
    文件上传响应

    上传成功后返回：
    - file_path: 文件保存路径（后续分析需要用）
    - filename: 原始文件名
    - rows: 数据行数
    - columns: 列名列表
    """
    file_path: str = Field(..., description="文件保存路径")
    filename: str = Field(..., description="原始文件名")
    rows: int = Field(0, description="数据行数")
    columns: list[str] = Field(default_factory=list, description="列名列表")


# ============================================================
# API 路由
# ============================================================
@router.post('/upload',response_model=UploadResponse)
async def upload_file(file:UploadFile= File(...)):
    """
    上传数据文件

    工作流程：
    1. 校验文件格式（只支持 CSV/Excel）
    2. 校验文件大小
    3. 保存文件到本地
    4. 读取文件获取基本信息（行数、列名）
    5. 返回文件路径和基本信息

    前端拿到 file_path 后，调用 /agent/analyze 进行分析
    """
    # 校验文件格式
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: .{ext}，支持: csv, xlsx, xls"
        )

    # 校验文件大小
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大: {len(content)} 字节，最大: {MAX_UPLOAD_SIZE} 字节"
        )

    # 保存文件
    # 用 UUID 生成唯一文件名，避免同名文件覆盖
    unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    # 读取文件基本信息
    try:
        from tools.file_reader import read_data_file
        df = read_data_file(file_path)
        rows = len(df)
        columns = list(df.columns)
    except Exception as e:
        # 读取失败时删除文件
        os.unlink(file_path)
        raise HTTPException(status_code=422, detail=f"读取文件失败: {str(e)}")

    return UploadResponse(
        file_path=file_path,
        filename=file.filename,
        rows=rows,
        columns=columns,
    )

@router.post('/analyze',response_model=AnalyzeResponse)
async def analyze_data(request:AnalyzeRequest):
    """
    执行数据分析

    工作流程：
    1. 校验文件是否存在
    2. 调用 LangGraph 工作流执行分析
    3. 返回分析报告和图表路径

    这个接口是整个系统的核心：
    - 接收文件路径和用户需求
    - 启动 LangGraph 工作流
    - 工作流内部：读取数据 → 规划 → 生成代码 → 执行 → 分析 → 生成报告
    - 返回最终结果
    """
     # 校验文件是否存在
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=404,
            detail=f"文件不存在: {request.file_path}"
        )

    try:
        # 调用工作流
        result = run_analysis(
            file_path=request.file_path,
            user_request=request.request,
        )

        return AnalyzeResponse(
            report=result.get("report", ""),
            chart_paths=result.get("chart_paths", []),
            success=True,
            error="",
        )

    except Exception as e:
        return AnalyzeResponse(
            report="",
            chart_paths=[],
            success=False,
            error=str(e),
        )

@router.get('/charts/{chart_filename}')
async def get_chart(chart_filename: str):
    """
    获取生成的图表图片

    前端通过这个接口获取图表图片
    例如: GET /agent/charts/chart_abc123.png

    使用 FastAPI 的 FileResponse 返回图片文件
    """
    from fastapi.responses import FileResponse
    from config import CHART_DIR

    chart_path = os.path.join(CHART_DIR, chart_filename)

    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="图表不存在")

    return FileResponse(chart_path, media_type="image/png")