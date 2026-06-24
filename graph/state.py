"""
智能数据分析 Agent - LangGraph State 定义

【学习要点】
1. 什么是 State（状态）？
   - State 就是 Agent 工作流中的"共享数据"
   - 想象一条流水线：每个工人（节点）从传送带上取数据、处理、放回去
   - State 就是那条传送带
   - 所有节点都能读取和修改 State

2. 为什么用 TypedDict？
   - TypedDict 是 Python 的类型提示工具
   - 它让 State 的字段有明确的类型
   - LangGraph 要求 State 必须是 TypedDict 的子类
   - 好处：IDE 有代码补全，写错字段名会报错

3. MessagesState 是什么？
   - LangGraph 内置的状态类，包含 messages 字段
   - messages 用来存储对话历史（HumanMessage, AIMessage 等）
   - 我们继承它，再添加自己的字段

4. 每个字段的含义（对照工作流理解）：

   用户上传CSV → data_reader节点读取
       ↓
   data_file_path: CSV文件路径
   data_summary: 数据摘要（列名、行数等）
       ↓
   analysis_planner节点规划
       ↓
   user_request: 用户需求（"分析销售趋势"）
   analysis_plan: 规划的分析步骤
       ↓
   code_generator节点生成代码
       ↓
   generated_code: AI生成的pandas代码
       ↓
   code_executor节点执行代码
       ↓
   execution_result: 代码输出
   execution_success: 是否成功
   error_message: 错误信息（失败时）
       ↓
   code_fixer节点修复代码（失败时）
       ↓
   retry_count: 已重试次数
       ↓
   result_analyzer节点分析结果
       ↓
   report_generator节点生成报告
       ↓
   chart_paths: 生成的图表文件路径
   report: 最终分析报告

5. Optional[str] vs str 的区别：
   - str: 必须有值，不能是 None
   - Optional[str]: 可以是 None（字段可以不存在）
   - 初始状态下很多字段还没有值，所以用 Optional
"""

from typing import TypedDict,Optional,Annotated
from langgraph.graph  import MessagesState

class AgentState(MessagesState):
    """
    Agent 工作流的状态定义

    这个类定义了所有节点之间共享的数据。
    每个节点可以读取这些字段，也可以修改这些字段。
    LangGraph 会自动管理状态的传递。

    继承 MessagesState 的原因：
    - MessagesState 已经包含了 messages 字段（对话历史）
    - 我们只需要添加自己的业务字段
    """

    # ---- 输入数据 ----
    # 用户上传的数据文件路径
    # 例: "./uploads/sales_data.csv"
    data_file_path: Optional[str] = None

    # 数据摘要（由 data_reader 节点生成）
    # 包含：行数、列名、数据类型、前几行预览等
    # 例: "数据形状: 100 行 x 5 列\n列名: ['date', 'sales', 'region']..."
    data_summary: str = ""

    # ---- 用户需求 ----
    # 用户用自然语言描述的分析需求
    # 例: "分析各地区的销售趋势"
    user_request: str = ""

    # ---- 分析规划 ----
    # AI 规划的分析步骤（由 analysis_planner 节点生成）
    # 例: "1. 按地区分组\n2. 计算每月销售额\n3. 绘制趋势图"
    analysis_plan: str = ""

    # ---- 代码生成 ----
    # AI 生成的 Python 代码（由 code_generator 节点生成）
    # 例: "import pandas as pd\ndf = pd.read_csv('...')\n..."
    generated_code: str = ""

    # ---- 代码执行 ----
    # 代码执行后的输出（print 的内容）
    # 例: "   region  sales\n0  北京    10000\n1  上海    12000"
    execution_result: str = ""

    # 代码是否执行成功
    execution_success: bool = False

    # 执行失败时的错误信息
    # 例: "NameError: name 'df' is not defined"
    error_message: str = ""

    # ---- 重试控制 ----
    # 当前已重试的次数（防止无限重试）
    retry_count: int = 0

    # ---- 输出结果 ----
    # 生成的图表文件路径列表
    # 例: ["./charts/chart_abc123.png", "./charts/chart_def456.png"]
    chart_paths: list[str] = []

    # 最终的分析报告（由 report_generator 节点生成）
    report: str = ""
