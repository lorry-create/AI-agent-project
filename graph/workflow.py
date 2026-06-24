"""
智能数据分析 Agent - 工作流定义和编译

【学习要点】
1. 什么是工作流（Workflow）？
   - 工作流 = 节点 + 边 + 条件边
   - 节点：每个处理步骤（已在上一个文件定义）
   - 边：节点之间的连接（数据流向）
   - 条件边：根据条件选择下一个节点（如成功走A，失败走B）

2. StateGraph 是什么？
   - LangGraph 的核心类，用来定义工作流
   - 创建时传入 State 类型（AgentState）
   - 通过 add_node / add_edge 构建图
   - 最后 compile() 编译成可执行的工作流

3. 边的三种类型：

   a. 普通边（add_edge）：固定流向
      workflow.add_edge("A", "B")  → A 完成后一定去 B

   b. 条件边（add_conditional_edges）：根据条件选择
      workflow.add_conditional_edges("C", judge_func, {"yes": "D", "no": "E"})
      → C 完成后，调用 judge_func 判断：
         返回 "yes" → 去 D
         返回 "no"  → 去 E

   c. 入口点（set_entry_point）：工作流的起点
      workflow.set_entry_point("A")  → 从 A 开始

4. 本工作流的结构：

   START → data_reader → analysis_planner → code_generator
                                              ↓
                                         code_executor
                                          ↓        ↑
                                    (成功)↓   (失败)↑
                                  result_analyzer  code_fixer
                                        ↓              ↓
                                  report_generator → END

5. compile() 的作用：
   - 检查图的完整性（有没有孤立节点、有没有环等）
   - 生成可执行的工作流对象
   - 编译后可以用 workflow.invoke(state) 执行

6. 和 if-else 有什么区别？
   - if-else：逻辑写死在代码里，改流程要改代码
   - LangGraph：逻辑用图定义，改流程只需改边
   - LangGraph 还支持：可视化、断点续跑、人工审批等
"""

from langgraph.graph import StateGraph
from graph.state import AgentState
from graph.nodes import (
    data_reader_node,
    analysis_planner_node,
    code_generator_node,
    code_executor_node,
    code_fixer_node,
    result_analyzer_node,
    report_generator_node,
)
from config import MAX_RETRIES

def should_fix_code(state:AgentState)->str:
    """
    条件边判断函数：代码执行后，判断下一步走哪条路

    这是 LangGraph 条件边的核心：一个函数决定流程走向

    判断逻辑：
    1. 如果执行成功 → 去 result_analyzer（分析结果）
    2. 如果执行失败但重试次数未超限 → 去 code_fixer（修复代码）
    3. 如果重试次数已超限 → 去 report_generator（直接生成报告，说明失败）

    参数:
        state: 当前状态

    返回:
        下一个节点的名称（字符串）
    """
    if state.get("execution_success"):
        # 执行成功 → 去分析结果
        return "result_analyzer"
    elif state.get("retry_count", 0) < MAX_RETRIES:
        # 执行失败但还能重试 → 去修复代码
        return "code_fixer"
    else:
        # 重试次数用完 → 直接生成报告（包含错误信息）
        return "report_generator"

def build_workflow() ->StateGraph:
    """
    构建 LangGraph 工作流

    步骤：
    1. 创建 StateGraph，传入 State 类型
    2. 添加所有节点（每个节点是一个函数）
    3. 设置入口点（工作流从哪个节点开始）
    4. 添加边（定义节点之间的连接）
    5. 编译工作流

    返回:
        编译后的工作流对象，可以用 .invoke() 执行
    """
    # ---- 第1步：创建 StateGraph ----
    # AgentState 定义了状态的结构
    # LangGraph 会根据这个类型来管理状态
    workflow = StateGraph(AgentState)

    # ---- 第2步：添加节点 ----
    # add_node(节点名称, 节点函数)
    # 节点名称：字符串，用于在边中引用
    # 节点函数：接收 state，返回 dict 的函数
    workflow.add_node("data_reader", data_reader_node)
    workflow.add_node("analysis_planner", analysis_planner_node)
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("code_fixer", code_fixer_node)
    workflow.add_node("result_analyzer", result_analyzer_node)
    workflow.add_node("report_generator", report_generator_node)

    # ---- 第3步：设置入口点 ----
    # 工作流从 data_reader 节点开始
    # 等同于 workflow.add_edge(START, "data_reader")
    workflow.set_entry_point("data_reader")

    # ---- 第4步：添加边 ----

    # 普通边：A → B（固定流向）
    # 数据读取完成后，去分析规划
    workflow.add_edge("data_reader", "analysis_planner")

    # 分析规划完成后，去代码生成
    workflow.add_edge("analysis_planner", "code_generator")

    # 代码生成完成后，去代码执行
    workflow.add_edge("code_generator", "code_executor")

    # 代码修复完成后，回到代码执行（重试）
    workflow.add_edge("code_fixer", "code_executor")

    # 结果分析完成后，去报告生成
    workflow.add_edge("result_analyzer", "report_generator")

    # 报告生成完成后，结束
    workflow.add_edge("report_generator", END)

    # ---- 条件边：代码执行后的分支 ----
    # add_conditional_edges(
    #   源节点,           从哪个节点出发
    #   判断函数,          用哪个函数判断
    #   {返回值: 目标节点}  函数返回值 → 对应的目标节点
    # )
    workflow.add_conditional_edges(
        "code_executor",          # 从 code_executor 出发
        should_fix_code,          # 用 should_fix_code 判断
        {
            "result_analyzer": "result_analyzer",  # 成功 → 分析结果
            "code_fixer": "code_fixer",            # 失败可重试 → 修复代码
            "report_generator": "report_generator", # 重试耗尽 → 生成报告
        },
    )

    # ---- 第5步：编译工作流 ----
    # compile() 会检查图的完整性：
    # - 所有节点都有入边和出边（除了入口和出口）
    # - 条件边的所有可能返回值都有对应的目标节点
    # - 没有不可达的节点
    compiled = workflow.compile()

    return compiled

# ============================================================
# 便捷函数：直接运行工作流
# ============================================================

def run_analysis(file_path: str, user_request: str) -> dict:
    """
    运行完整的数据分析工作流

    这是对外暴露的便捷接口：
    - 不需要手动构建工作流
    - 传入文件路径和需求，直接获取结果

    参数:
        file_path: 数据文件路径
        user_request: 用户的分析需求

    返回:
        最终的状态字典，包含报告、图表路径等
    """
    # 构建初始状态
    initial_state = {
        "data_file_path": file_path,
        "user_request": user_request,
        "messages": [],
    }

    # 构建并执行工作流
    workflow = build_workflow()
    result = workflow.invoke(initial_state)

    return result