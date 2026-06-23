"""
RAG 知识库问答系统 - 文档解析工具
支持 PDF / Word / Markdown / TXT 格式

【学习要点】
1. 为什么需要文档解析？
   - 用户上传的文档是"人读"的格式（PDF有排版、Word有样式）
   - RAG 系统需要的是"纯文本"，所以要把各种格式统一转成纯文本

2. 每种格式的解析思路：
   - TXT: 最简单，直接读文件内容
   - Markdown: 也是纯文本，直接读取（格式符号保留，后续分块时处理）
   - PDF: 用 pypdf 库，逐页提取文本
   - Word(.docx): 用 python-docx 库，逐段落提取文本

3. 设计模式：策略模式
   - parse_document() 是统一入口，根据文件扩展名自动选择解析器
   - 新增格式支持时，只需加一个 parse_xxx 函数和一行映射
"""


##下面是解析PDF文件的代码实现

from pathlib import Path

def parse_pdf(file_path:str) -> str:
    """
    解析 PDF 文件，返回纯文本

    pypdf 的工作原理：
    - PdfReader 打开文件，获取所有页面对象
    - 每个页面调用 extract_text() 提取文字
    - 注意：扫描版 PDF（图片型）无法提取文字，需要 OCR
    """
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text_parts = []

    for i,page in enumerate(reader.pages):
        #提取当前的页的文本
        page_text = page.extract_text()
        if page_text:
            #用分页标记区分不同页，方便后续溯源
            text_parts.append(f"Page {i+1}: {page_text}")

    return "\n".join(text_parts)


def parse_docx(file_path:str)-> str:
    """
    解析 Word (.docx) 文件，返回纯文本

    python-docx 的工作原理：
    - Document() 打开文件
    - document.paragraphs 获取所有段落
    - 每个段落有 .text 属性，就是文字内容
    - 注意：表格、图片中的文字需要额外处理（这里暂不处理）
    """
    from docx import Document

    doc = Document(file_path)
    text_parts = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
             text_parts.append(paragraph.text)
            
    return "\n\n".join(text_parts)


def parse_markdown(file_path:str) -> str:
    """
    解析 Markdown 文件，返回纯文本

    Markdown 本身就是纯文本格式，直接读取即可
    保留原始格式符号（#、** 等），因为它们也包含语义信息
    """
    with open(file_path,'r',encoding='utf-8') as f:
        return f.read()

def parse_txt(file_path:str)->str:
    """
    解析 TXT 文件，返回纯文本
    最简单的解析器，直接读取文件内容
    """
    with open(file_path,'r',encoding='utf-8') as f:
        return f.read()

    
# ============================================================
# 解析器映射表：文件扩展名 -> 对应的解析函数
# 这是策略模式的核心：用字典代替一堆 if-elif
# ============================================================
PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "md": parse_markdown,
    "txt": parse_txt,
}
# 支持的文件格式（从映射表的 key 生成，方便统一维护）
SUPPORTED_EXTENSIONS = set(PARSERS.keys())

def parse_document(file_path:str)->str:
    """
    统一入口：根据文件扩展名自动选择解析器

    工作流程：
    1. 从文件路径提取扩展名（如 "report.pdf" → "pdf"）
    2. 在映射表中查找对应的解析函数
    3. 调用解析函数，返回纯文本

    参数:
        file_path: 文件路径

    返回:
        解析出的纯文本

    异常:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    #检查文件是否存在
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    # 提取扩展名（去掉点号，转小写）
    # path.suffix 返回 ".pdf"，[1:] 去掉点号得到 "pdf"
    ext = path.suffix.lstrip(".").lower()

    #查找对应的解析器
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(
            f"不支持的文件格式: .{ext}\n"
            f"支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    #调用解析器
    text = parser(file_path)
    # 基本校验：解析结果不能为空
    if not text.strip():
        raise ValueError(f"文件解析结果为空，可能是扫描版 PDF 或空白文档: {file_path}")

    return text