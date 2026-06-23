"""
智能数据分析 Agent - 文件读取工具

【学习要点】
1. 为什么需要文件读取工具？
   - Agent 要分析数据，第一步就是读取数据文件
   - 不同格式（CSV/Excel）需要不同的读取方式
   - 读取后要生成"数据摘要"，让 LLM 了解数据长什么样

2. pandas 是什么？
   - Python 最流行的数据分析库
   - DataFrame 是它的核心数据结构，类似 Excel 表格
   - pd.read_csv() / pd.read_excel() 是最常用的读取函数

3. 数据摘要（get_data_summary）为什么重要？
   - LLM 不能直接"看"数据文件（太大了）
   - 数据摘要把关键信息浓缩成文字：多少行、多少列、列名、数据类型
   - LLM 根据摘要来规划分析步骤和生成代码
   - 这就像人做数据分析前先"看看数据长什么样"

4. 为什么只读取前 N 行？
   - 大文件全部读取可能很慢
   - LLM 的上下文有限，不可能把整个数据集塞进去
   - 前几行足以让 LLM 理解数据结构
"""
import pandas as pd
from pathlib import Path

def read_csv(file_path:str) -> pd.DataFrame:
    """
    读取 CSV 文件

    pd.read_csv() 的常用参数：
    - encoding: 编码格式，中文文件常用 'gbk' 或 'utf-8-sig'
    - nrows: 只读取前 N 行（大文件预览用）
    - parse_dates: 自动解析日期列

    参数:
        file_path: CSV 文件路径

    返回:
        DataFrame 对象（类似 Excel 表格的数据结构）
    """
    try:
        #先尝试UTF-8编码
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        # UTF-8 失败，尝试 GBK（中文 Windows 常见编码）
        return pd.read_csv(file_path, encoding="gbk")

def read_excel(file_path:str) -> pd.DataFrame:
    """
    读取 Excel 文件

    pd.read_excel() 的常用参数：
    - sheet_name: 指定工作表名或序号，默认读取第一个
    - engine: 解析引擎，openpyxl 支持 .xlsx

    参数:
        file_path: Excel 文件路径

    返回:
        DataFrame 对象
    """
    return pd.read_excel(file_path,engine="openpyxl")


def read_data_file(file_path:str) ->pd.DataFrame:
    """
    根据文件扩展名自动选择读取方式

    这是"统一入口"模式：
    - 调用者不需要知道具体用哪个函数
    - 只需要传文件路径，自动判断格式
    - 新增格式支持时，只需在这里加一个 elif

    参数:
        file_path: 数据文件路径

    返回:
        DataFrame 对象

    异常:
        ValueError: 不支持的文件格式
    """
    # 从文件路径提取扩展名
    # Path(file_path).suffix 得到 ".csv"，[1:] 去掉点号得到 "csv"
    ext = Path(file_path).suffix.lstrip(".").lower()

    if ext == "csv":
        return read_csv(file_path)
    elif ext in ("xlsx","xls"):
        return read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
def get_data_summary(df:pd.DataFrame,max_rows:int = 5) ->str:
    """
    获取数据摘要信息

    这个函数是 Agent 能理解数据的关键！
    它把 DataFrame 的结构信息转成文字，供 LLM 阅读。

    摘要包含：
    1. 数据形状：多少行多少列
    2. 列名和数据类型：有哪些字段，是数字还是文字
    3. 前几行数据：让 LLM 看到实际数据长什么样
    4. 统计摘要：数值列的均值、最大最小值等

    参数:
        df: pandas DataFrame
        max_rows: 预览的最大行数（默认5行）

    返回:
        数据摘要的文本描述
    """
    summary_parts = []

    #1.数据形状
    #df.shape 返回(行数，列数)
    summary_parts.append(f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")

    # 2. 列名和数据类型
    # df.dtypes 返回每列的名称和类型
    summary_parts.append(f"列名: {list(df.columns)}")
    summary_parts.append(f"数据类型:\n{df.dtypes.to_string()}")

    # 3. 前几行数据预览
    # df.head() 返回前 N 行
    # to_string() 转成纯文本格式
    summary_parts.append(f"前 {max_rows} 行数据:\n{df.head(max_rows).to_string()}")

    # 4. 数值列的统计摘要
    # df.describe() 返回均值、标准差、最小最大值等
    # 只对数值列有效
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        summary_parts.append(f"数值列统计:\n{df.describe().to_string()}")

    
    # 5. 缺失值信息
    # df.isnull().sum() 返回每列的缺失值数量
    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary_parts.append(f"缺失值:\n{missing[missing > 0].to_string()}")

    return "\n\n".join(summary_parts)