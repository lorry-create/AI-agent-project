"""
RAG 知识库问答系统 - 文档上传、解析、入库服务

【学习要点】
1. 这个服务是 RAG 的"入库"流程：上传 → 解析 → 分块 → 向量化 → 存入 Chroma
   这是 RAG 系统的前半段，没有这一步，后面就没东西可检索

2. 向量数据库（Chroma）的工作原理：
   - 文本 → Embedding 模型 → 向量（一组数字，如 [0.12, -0.34, 0.56, ...]）
   - 向量存入 Chroma，同时存储原始文本和元数据
   - 查询时：问题 → 向量 → 在 Chroma 中找最相似的向量 → 返回对应文本
   - "相似"是用余弦距离衡量的，方向越接近越相似

3. Chroma 的核心概念：
   - Collection: 类似数据库的"表"，一个 Collection 存一类文档
   - Document: 原始文本
   - Embedding: 文本对应的向量
   - Metadata: 元数据（如来源文件名），方便后续过滤

4. 为什么用 get_or_create_collection？
   - 第一次运行时创建 Collection
   - 后续运行时获取已有的 Collection（不会重复创建）
   - 这样已入库的文档不会丢失
"""
import uuid
from pathlib import Path

from fastapi import APIRouter,UploadFile,File,HTTPException
from langsmith.run_helpers import _METADATA
import chromadb

from models.schemas import UploadResponse
from utils.document_parser import parse_document,SUPPORTED_EXTENSIONS
from utils.text_splitter import split_text_with_metadata

from config import (
    UPLOAD_DIR,
    MAX_UPLOAD_SIZE,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
)

router = APIRouter(prefix='/upload',tags=['文档上传'])

# ============================================================
# 初始化 Chroma 客户端和 Collection
# ============================================================
# PersistentClient:w
# PersistentClient = 数据持久化到磁盘，程序重启数据不丢失
# HttpClient = 连接远程 Chroma 服务器（生产环境用）
# ============================================================

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def get_chroma_collection():
    """
    获取 Chroma 的 Collection（向量集合）

    工作流程：
    1. 创建 PersistentClient，数据保存在 CHROMA_PERSIST_DIR
    2. 获取或创建名为 CHROMA_COLLECTION_NAME 的 Collection
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )
    return collection


def get_embeddings(texts:list[str])->list[list[float]]:
    """
    将文本列表转换为向量列表

    两种模式：
    1. 如果配置了 OpenAI API Key → 用 OpenAI 的 Embedding 模型（效果好）
    2. 如果没配置 → 用 Chroma 自带的默认模型（效果一般，但能跑起来）

    Embedding 的作用：
    - 把文本变成数字向量，语义相近的文本 → 向量也相近
    - 这样就能用数学方法计算"两个文本有多相似"
    """
    if OPENAI_API_KEY:
        #使用openai的 embedding模型
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL or None,
        )
        return embeddings.embed_query(texts)
    
    else:
        # 使用 Chroma 默认的 Embedding（all-MiniLM-L6-v2）
        # 不需要 API Key，但效果不如 OpenAI
        collection = get_chroma_collection()
        # Chroma 的 add 方法会自动用默认模型生成向量
        # 这里我们用 Chroma 内置的方式
        return None  # 返回 None 表示让 Chroma 自动生成向量
    
@router.post('/',response_model=UploadResponse)
async def upload_document(file:UploadFile=File(...)):
    """
    上传文档并处理入库

    完整流程：
    1. 校验文件格式和大小
    2. 保存文件到本地
    3. 解析文档为纯文本
    4. 文本分块
    5. 向量化并存入 Chroma

    参数:
        file: 上传的文件对象（FastAPI 自动处理 multipart/form-data）

    返回:
        UploadResponse: 上传结果（文件名、分块数、提示信息）
    """
    # ---- 第1步：校验文件格式 ----
    # 从文件名提取扩展名
    filename = file.filename
    ext = Path(filename).suffix.lstrip('.').lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式：{ext}"
        )
    
    # ---- 第2步：校验文件大小 ----
    content = await file.read()

    if len(content) >  MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小不能超过 {MAX_UPLOAD_SIZE} 字节"
        )
    
    # ---- 第3步：保存文件到本地 ----
    # 用 UUID 生成唯一文件名，避免同名文件覆盖
    unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = Path(UPLOAD_DIR) / unique_filename
    with open(file_path, "wb") as f:
        f.write(content)
    
    # ---- 第4步：解析文档为纯文本 ----
    try:
        text = parse_document(str(file_path))
    except Exception as e:
        # 解析失败时删除已保存的文件
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"文档解析失败: {str(e)}")

    # ---- 第5步：文本分块 ----
    chunks_with_meta = split_text_with_metadata(
        text,
        source_filename=filename,
    )

    if not chunks_with_meta:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422,detail="文档解析为空")
    
    # ---- 第六步 向量化并存入chroma
    try:
        collection = get_chroma_collection()

        #准备数据
        texts = [chunk['text'] for chunk in chunks_with_meta]
        metadatas = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks_with_meta
        ]
        # 为每个块生成唯一的ID
        ids = [f"{uuid.uuid4().hex}" for _ in chunks_with_meta]

        # 获取向量（如果用 OpenAI）
        embeddings = get_embeddings(texts)

        # 存入 Chroma
        if embeddings:
            # 用 OpenAI 向量
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            # 让chroma 自动生成向量
            collection.add(
                documents = texts,
                metadatas = metadatas,
                ids=ids,
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")
    
    return UploadResponse(
        filename=filename,
        chunks_count=len(chunks_with_meta),
        message=f"文档处理完成，共分为 {len(chunks_with_meta)} 个文本块"
    )

@router.get('/status')
async def get_upload_status():
    """
    获取当前向量库的状态
    返回已存储的文档数量等信息
    """
    try:
        collection = get_chroma_collection()
        count = collection.count()
        return {
            "total_documents": count,
            'collection_name': CHROMA_COLLECTION_NAME,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")