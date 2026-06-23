"""
RAG 知识库问答系统 - 文本分块策略

【学习要点】
1. 为什么要分块（Chunking）？
   - LLM 有上下文长度限制（如 GPT-4o 最多 128K tokens）
   - 整篇文档太长，不能全部塞给 LLM
   - 向量检索也需要小块：块越小，检索越精准

2. 什么是 RecursiveCharacterTextSplitter？
   - LangChain 提供的文本分块工具
   - "Recursive" = 递归：先尝试按第一个分隔符切，切出来还是太长就换下一个分隔符
   - 分隔符优先级：段落(\n\n) > 换行(\n) > 句号(。) > 空格 > 字符
   - 这样能尽量在语义边界处切分，避免把一句话切成两半

3. chunk_size 和 chunk_overlap 怎么选？
   - chunk_size: 每块最大字符数
     - 英文：500-1000 字符较常见
     - 中文：300-600 字符较常见（中文字符信息密度更高）
   - chunk_overlap: 相邻块重叠的字符数
     - 通常设为 chunk_size 的 10%-20%
     - 作用：避免关键信息恰好在切分边界被截断

4. 分块策略的权衡：
   - 小块 → 检索精准，但上下文可能不完整
   - 大块 → 上下文完整，但检索可能混入无关内容
   - 实际项目中需要根据数据特点调优
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    将长文本分块

    参数:
        text: 要分块的纯文本
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数

    返回:
        文本块列表，例如 ["第一块...", "第二块...", ...]

    示例:
        >>> chunks = split_text("很长的文本...", chunk_size=500, chunk_overlap=50)
        >>> print(f"共分为 {len(chunks)} 块")
        >>> print(f"第一块: {chunks[0][:100]}...")
    """
    # 创建分块器
    # separators 定义了分隔符的优先级：
    # 1. 先尝试按双换行（段落边界）切
    # 2. 切出来还是太长，按单换行（行边界）切
    # 3. 还是太长，按中文句号切
    # 4. 依此类推...
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # 段落边界
            "\n",    # 行边界
            "。",    # 中文句号
            "！",    # 中文感叹号
            "？",    # 中文问号
            ".",     # 英文句号
            "!",     # 英文感叹号
            "?",     # 英文问号
            " ",     # 空格
            "",      # 最后手段：按字符切
        ],
    )

    #执行分块
    chunks = splitter.split_text(text)

    return chunks


def split_text_with_metadata(
    text: str,
    source_filename: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    分块并附带元数据（来源文件名、块序号等）

    元数据的作用：
    - 回答问题时可以标注"这个答案来自哪个文档"
    - 方便追溯和调试

    参数:
        text: 要分块的纯文本
        source_filename: 来源文件名
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数

    返回:
        包含文本和元数据的字典列表
        例如: [{"text": "...", "source": "report.pdf", "chunk_index": 0}, ...]
    """
    chunks = split_text(text, chunk_size, chunk_overlap)

    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "source": source_filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

    return result
