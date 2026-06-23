"""
RAG 知识库问答系统 - 配置文件

【学习要点】
1. 为什么要用配置文件？
   - 把所有"可变的参数"集中管理，改配置不用改代码
   - 敏感信息（API Key）不硬编码在代码里，而是从环境变量读取

2. python-dotenv 是什么？
   - 它会读取 .env 文件中的键值对，加载到 os.environ 中
   - 这样你在本地开发时，把 API Key 写在 .env 文件里（不提交到 Git）
   - 部署时通过系统环境变量传入，代码不用改

3. os.getenv("KEY", "default") 的含义：
   - 先从环境变量找 KEY，找不到就用 default 值
   - 这样即使没有 .env 文件，程序也能用默认值跑起来
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# load_dotenv() 会查找当前目录及上级目录的 .env 文件
load_dotenv()

# ============================================================
# API 配置（阿里云百炼）
# ============================================================
# OPENAI_API_KEY: 阿里云 API Key（请替换为你的真实 Key）
# OPENAI_BASE_URL: 阿里云百炼 API 地址
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ============================================================
# 模型配置（阿里云百炼）
# ============================================================
# LLM_MODEL: 对话模型（qwen-turbo 免费 / qwen-plus 效果好 / qwen-max 效果最佳）
# EMBEDDING_MODEL: 向量化模型
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# ============================================================
# 向量数据库配置
# ============================================================
# CHROMA_PERSIST_DIR: Chroma 数据库存数据的目录
#   持久化 = 程序关了数据还在，下次启动不用重新处理文档
# CHROMA_COLLECTION_NAME: 集合名，类似数据库的"表名"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# ============================================================
# 文档处理配置
# ============================================================
# CHUNK_SIZE: 每个文本块的最大字符数
#   太大 → 检索不精准（混入太多无关内容）
#   太小 → 语义不完整（一句话被切成两半）
#   500 是一个常用的起始值，中文约 250-300 字
# CHUNK_OVERLAP: 相邻块之间的重叠字符数
#   重叠是为了避免关键信息恰好在切分边界被截断
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ============================================================
# 上传文件配置
# ============================================================
# UPLOAD_DIR: 用户上传的文件保存目录
# MAX_UPLOAD_SIZE: 最大文件大小（字节），10MB = 10 * 1024 * 1024
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))

# ============================================================
# 检索配置
# ============================================================
# TOP_K: 检索时返回最相似的 K 个文档片段
#   K 太小 → 可能漏掉相关信息
#   K 太大 → 噪音多，Token 消耗高
#   3-5 是常用值
TOP_K = int(os.getenv("TOP_K", "5"))

# ============================================================
# 启动时自动创建必要目录
# ============================================================
# Path 是 Python 3 的路径处理库，比 os.path 更好用
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
