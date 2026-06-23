"""
RAG 知识库问答系统 - Pydantic 数据模型

【学习要点】
1. 什么是 Pydantic？
   - Python 的数据验证库，FastAPI 的核心依赖
   - 你定义一个类继承 BaseModel，Pydantic 会自动：
     a. 验证传入数据的类型（字段类型不对就报错）
     b. 序列化/反序列化（Python 对象 ↔ JSON）
     c. 生成 JSON Schema（FastAPI 用来生成 API 文档）

2. 为什么需要数据模型？
   - 前端传来的数据可能格式不对（比如该传数字传了字符串）
   - Pydantic 模型在入口处就拦截了这些错误
   - 代码也更清晰：看模型就知道接口需要什么、返回什么

3. Optional[str] = None 的含义：
   - 这个字段可以不传，不传时默认值是 None
   - 比如 session_id，第一次对话时没有，后续对话才需要

4. FastAPI 如何使用这些模型？
   - 请求模型（如 ChatRequest）：用在函数参数中，FastAPI 自动解析请求体
   - 响应模型（如 ChatResponse）：用在 response_model 参数中，FastAPI 自动：
     a. 验证返回数据格式
     b. 在 API 文档中展示返回结构
"""

from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    """
    文档上传响应

    上传成功后返回给前端的信息：
    - filename: 上传的文件名
    - chunks_count: 文档被分成了多少块
    - message: 提示信息
    """
    filename: str = Field(..., description="上传的文件名")
    chunks_count: int = Field(..., description="分块数量")
    message: str = Field(..., description="处理结果提示")


class ChatRequest(BaseModel):
    """
    问答请求

    前端发来的提问：
    - question: 用户的问题（必填）
    - session_id: 会话 ID（可选，用于多轮对话）
      - 第一次提问可以不传，系统会生成一个新的
      - 后续提问带上同一个 session_id，系统就能记住之前的对话
    """
    question: str = Field(..., description="用户的问题")
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")


class SourceInfo(BaseModel):
    """
    引用来源信息

    RAG 回答必须标注来源，这是 RAG 和普通聊天的关键区别：
    - source: 来自哪个文档
    - content: 引用的原文片段
    - relevance_score: 相关度分数（0-1，越接近1越相关）
    """
    source: str = Field(..., description="来源文件名")
    content: str = Field(..., description="引用的原文片段")
    relevance_score: float = Field(0.0, description="相关度分数")


class ChatResponse(BaseModel):
    """
    问答响应

    返回给前端的回答：
    - answer: AI 生成的回答
    - sources: 引用的文档片段列表（RAG 的核心特征：有据可查）
    - session_id: 会话 ID（前端需要保存，下次提问时传回来）
    """
    answer: str = Field(..., description="AI 生成的回答")
    sources: list[SourceInfo] = Field(default_factory=list, description="引用来源列表")
    session_id: str = Field(..., description="会话ID")


class HistoryItem(BaseModel):
    """
    对话历史条目

    一条对话记录：
    - role: "user"（用户）或 "assistant"（AI）
    - content: 对话内容
    """
    role: str = Field(..., description="角色: user 或 assistant")
    content: str = Field(..., description="对话内容")


class HistoryResponse(BaseModel):
    """
    对话历史响应

    返回某个会话的所有对话记录：
    - session_id: 会话 ID
    - history: 对话记录列表
    """
    session_id: str = Field(..., description="会话ID")
    history: list[HistoryItem] = Field(default_factory=list, description="对话历史")
