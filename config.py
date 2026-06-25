"""
智能数据分析 Agent - 配置文件

【学习要点】
1. 这个文件和第一周 RAG 项目的 config.py 结构一样
   - 用 python-dotenv 从 .env 文件读取环境变量
   - 用 os.getenv() 获取，支持默认值
   - 敏感信息（API Key）不硬编码，从环境变量读取

2. 新增的配置项：
   - CODE_EXECUTION_TIMEOUT: 代码执行超时时间
     为什么需要超时？AI 生成的代码可能死循环，必须设上限
   - MAX_RETRIES: 代码修复最大重试次数
     AI 生成的代码可能一次修不好，但不能无限重试

3. .env 文件的作用：
   - 存放敏感信息（API Key），不提交到 Git
   - 不同环境（开发/生产）可以用不同的 .env
   - .env.example 是模板，告诉别人需要哪些配置
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# .env 文件格式：KEY=VALUE，每行一个
# load_dotenv() 会把 .env 中的变量加载到 os.environ 中

load_dotenv()
# ============================================================
# API 配置
# ============================================================
# OPENAI_API_KEY: OpenAI 的 API 密钥
#   - 必填项，没有它无法调用 LLM
#   - 从 .env 文件读取，不要硬编码在代码里
#   - 也支持兼容 OpenAI 接口的第三方服务（如 DeepSeek、智谱等）

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OPENAI_BASE_URL: API 的基础地址
#   - 默认是 OpenAI 官方地址，不需要填
#   - 如果用第三方兼容服务，填他们的地址（如 https://api.deepseek.com/v1）
#   - 留空字符串表示使用默认地址
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
# ============================================================
# 模型配置
# ============================================================
# LLM_MODEL: 使用的大语言模型名称
#   - gpt-4o: OpenAI 最新多模态模型，代码生成能力强
#   - gpt-4o-mini: 更便宜，速度更快，适合简单任务
#   - 也可以用其他兼容模型的名称

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# ============================================================
# 文件上传配置
# ============================================================
# UPLOAD_DIR: 上传文件的保存目录
#   - 用户上传的 CSV/Excel 文件保存在这里
#   - 默认是 ./uploads，相对于项目根目录
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# MAX_UPLOAD_SIZE: 上传文件的最大字节数
#   - 默认 50MB (50 * 1024 * 1024 = 52428800)
#   - CSV/Excel 文件通常不会太大，50MB 足够
#   - int() 转换是因为环境变量读出来是字符串
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "52428800"))

# ============================================================
# 代码执行配置
# ============================================================
# CODE_EXECUTION_TIMEOUT: 代码执行超时时间（秒）
#   - AI 生成的代码可能死循环或运行太久
#   - 超过这个时间就强制终止
#   - 30 秒对大多数数据分析任务足够
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))

# MAX_RETRIES: 代码修复的最大重试次数
#   - 代码执行失败后，Agent 会尝试修复代码重新执行
#   - 不能无限重试，否则可能陷入死循环
#   - 3 次是合理的上限
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ============================================================
# 图表保存配置
# ============================================================
# CHART_DIR: 生成的图表保存目录
#   - matplotlib 生成的图表保存为图片文件
#   - 前端通过 URL 访问这些图片
CHART_DIR = os.getenv("CHART_DIR", "./charts")