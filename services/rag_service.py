"""
RAG 知识库问答系统 - 检索、问答逻辑

【学习要点】
1. RAG 的核心流程（这个文件实现了后半段）：
   前半段（document_service）：上传 → 解析 → 分块 → 向量化 → 存入 Chroma
   后半段（rag_service）：  问题 → 向量化 → 检索 → 拼接 Prompt → LLM 生成回答

2. 什么是 Prompt 模板？
   - 给 LLM 的"指令模板"，里面有占位符（{context}、{question}）
   - 运行时替换占位符，生成完整的 Prompt
   - 好的 Prompt 是 RAG 效果的关键

3. 为什么要在回答中标注来源？
   - RAG 和普通聊天的核心区别：回答基于文档，不是瞎编
   - 标注来源让用户能验证答案的可靠性
   - 面试中这是加分项：说明你理解 RAG 的核心价值

4. 对话记忆如何工作？
   - 每次提问时，把之前的对话历史也传给 LLM
   - LLM 就能理解上下文（"它"指的是什么？）
   - 但历史不能太长，否则超出 Token 限制 → 需要裁剪
"""


import uuid
from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, SourceInfo
from services.document_service import get_chroma_collection
from services.memory_service import get_history, add_to_history

from config import (
    TOP_K,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
)

router = APIRouter(prefix="/chat", tags=["问答接口"])

# ============================================================
# RAG Prompt 模板
# ============================================================
# 这是给 LLM 的"指令"，告诉它：
# 1. 你是一个知识库问答助手
# 2. 只能根据提供的文档内容回答
# 3. 如果文档中没有相关信息，就说不知道
# 4. 要标注引用来源
# ============================================================

RAG_PROMPT_TEMPLATE = """你是一个专业的知识库问答助手。请根据以下参考文档来回答用户的问题。

要求：
1. 只根据参考文档中的内容回答，不要编造信息
2. 如果参考文档中没有相关信息，请明确说"根据现有文档，我无法回答这个问题"
3. 回答时请引用具体的文档来源
4. 用清晰、准确的语言回答

参考文档：
{context}

用户问题：{question}

请给出回答："""

def retrieve_documents(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    从向量数据库中检索与问题最相关的文档片段

    工作流程：
    1. 把用户的问题转成向量
    2. 在 Chroma 中查找最相似的 top_k 个文档
    3. 返回文档内容、元数据和相似度分数

    参数:
        query: 用户的提问
        top_k: 返回最相关的 K 个结果

    返回:
        检索结果列表，每项包含 text、source、distance
    """
    collection = get_chroma_collection()

    #检查向量库是否为空
    if collection.count() == 0:
        raise []
    
    if OPENAI_API_KEY:
        # 用 OpenAI Embedding 把问题转成向量
        from langchain_openai import OpenAIEmbeddings

        embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL or None,
        )
        query_embedding = embeddings_model.embed_query(query)

        # 在 Chroma 中查询最相似的文档
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
    else:
        # 让 Chroma 自动把问题转成向量再查询
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
        )
    
    # 整理结果
    retrieved = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0

            retrieved.append({
                "text": doc,
                "source": metadata.get("source", "未知来源"),
                "chunk_index": metadata.get("chunk_index", 0),
                # 余弦距离转相似度分数：1 - distance
                # distance 越小越相似，score 越大越相似
                "relevance_score": round(1 - distance, 4),
            })
        
    return retrieved


def build_prompt(context:str,question:str)->str:
    """
    构建 RAG Prompt

    把检索到的文档内容和用户问题填入模板

    参数:
        context: 检索到的文档片段（拼接后的文本）
        question: 用户的问题

    返回:
        完整的 Prompt 字符串
    """
    return RAG_PROMPT_TEMPLATE.format(context=context,question=question)


def generate_answer(prompt:str,history:list[dict] = None)->str:
    """
    用 LLM 生成回答

    参数:
        prompt: 构建好的 Prompt 字符串
        history: 之前的对话记录，用于上下文

    返回:
        生成的回答字符串
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage,SystemMessage

    #创建LLm实例
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key = OPENAI_API_KEY,
        openai_api_base = OPENAI_BASE_URL or None,
        temperature=0.3,# 低温度 = 更确定性的回答，减少编造
    )
    # 构建消息列表
    messages = [
        # System 消息：设定 AI 的角色和行为规则
        SystemMessage(content="你是一个专业的知识库问答助手，只根据提供的文档内容回答问题。"),
    ]
     # 加入对话历史（如果有）
    if history:
        for msg in history[-6:]:  # 只取最近 6 条，避免 Token 超限
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg["content"]))

    # 加入当前问题（包含检索到的文档）
    messages.append(HumanMessage(content=prompt))

    #调用llm
    response = llm.invoke(messages)
    return response.content