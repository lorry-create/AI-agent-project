"""
RAG 知识库问答系统 - 对话记忆管理

【学习要点】
1. 为什么需要对话记忆？
   - 没有记忆：每次提问都是全新的，AI 不知道之前聊了什么
   - 有记忆：AI 能理解"它指的是什么"、"继续上面的分析"
   - 这就是"多轮对话"的基础

2. 这里用内存存储（dict）作为简单实现：
   - 优点：简单、快速、不需要额外服务
   - 缺点：程序重启后记忆丢失
   - 生产环境应该用 Redis / 数据库 持久化存储

3. 记忆的裁剪策略：
   - LLM 有 Token 限制，不能把所有历史都塞进去
   - 常见策略：
     a. 只保留最近 N 轮对话（本文件采用）
     b. 按 Token 数裁剪
     c. 用 LLM 总结历史，只保留摘要

4. session_id 的作用：
   - 不同用户的对话要隔离
   - 同一用户的不同话题也要隔离
   - session_id 就是区分不同对话的标识
"""
from fastapi import APIRouter,HTTPException

from models.schemas import HistoryResponse, HistoryItem

router = APIRouter(prefix='/history',tags=['对话历史'])

# ============================================================
# 内存存储：用字典保存对话历史
# ============================================================
# 结构：{ "session_id_1": [{"role": "user", "content": "..."}, ...], ... }
# 注意：这只是开发阶段的简单实现，生产环境要用 Redis/数据库
# ============================================================
_conversation_store:dict[str,list[dict]] = {}

#每个会话最多保留的历史条数
MAX_HISTORY_LENGTH = 20

def get_history(session_id:str) -> list[dict]:
    """
    获取指定会话的对话历史

    参数:
        session_id: 会话 ID

    返回:
        对话历史列表，如 [{"role": "user", "content": "什么是RAG?"}, ...]
    """
    return _conversation_store.get(session_id,[])


def add_to_history(session_id:str,role:str,content:str)->None:
    """
    向指定会话添加一条对话记录

    参数:
        session_id: 会话 ID
        role: 角色（"user" 或 "assistant"）
        content: 对话内容
    """
    #如果会话不存在，创建空列表
    if session_id not in _conversation_store:
        _conversation_store[session_id] = []
    
    #添加记录
    _conversation_store[session_id].append({
        'role':role,
        "content":content,
    })
    # 裁剪：只保留最近 MAX_HISTORY_LENGTH 条
    # 避免历史太长导致 Token 超限
    if len(_conversation_store[session_id]) > MAX_HISTORY_LENGTH:
        _conversation_store[session_id] = _conversation_store[session_id][-MAX_HISTORY_LENGTH:]
    

def clear_history(session_id:str) ->None:
    """
    清除指定会话的对话历史

    参数:
        session_id: 会话 ID
    """
    if session_id in _conversation_store:
        del _conversation_store[session_id]

# ============================================================
# API 路由
# ============================================================

@router.get('/{session_id}',response_model=HistoryResponse)
async def get_histroy_api(session_id:str):
    """
    获取指定会话的对话历史

    用法：GET /history/abc123
    返回：该会话的所有对话记录
    """
    history = get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        history=[HistoryItem(**item) for item in history],
    )

@router.delete('/{session_id}')
async def clear_history_api(session_id:str):
    """
    清除指定会话的对话历史

    用法：DELETE /history/abc123
    效果：该会话的所有记录被删除
    """
    clear_history(session_id)
    return {"message": f"会话 {session_id} 的历史已清除"}

@router.get("/")
async def list_sessions():
    """
    列出所有活跃的会话

    用法：GET /history/
    返回：所有有对话记录的会话 ID 列表
    """
    return {
        "sessions": list(_conversation_store.keys()),
        "total": len(_conversation_store),
    }