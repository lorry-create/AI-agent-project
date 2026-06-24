"""
智能数据分析 Agent - 各节点逻辑

【学习要点】
1. 什么是"节点"（Node）？
   - LangGraph 中，每个节点是一个函数
   - 函数接收当前 State，处理后返回要更新的字段
   - 节点之间通过 State 传递数据（不是直接调用）

2. 节点函数的写法规范：
   - 参数：state: AgentState（当前状态）
   - 返回：dict（要更新的字段）
   - 返回的字典会被 LangGraph 自动合并到 State 中

   例如：
   def my_node(state: AgentState) -> dict:
       # 读取 state 中的字段
       name = state["user_request"]
       # 处理逻辑...
       # 返回要更新的字段
       return {"analysis_plan": "步骤1..."}

3. 为什么要拆成多个节点？
   - 每个节点职责单一，容易测试和调试
   - 可以灵活组合（加节点、改顺序、加条件分支）
   - LangGraph 可以可视化工作流，一目了然

4. LLM 调用的模式：
   - 创建 ChatOpenAI 实例
   - 构建 Prompt（系统提示 + 用户输入）
   - 调用 llm.invoke() 获取回复
   - 从回复中提取需要的信息

5. 节点之间的数据流转：
   data_reader → 写入 data_summary
   analysis_planner → 读取 data_summary，写入 analysis_plan
   code_generator → 读取 analysis_plan，写入 generated_code
   code_executor → 读取 generated_code，写入 execution_result
   code_fixer → 读取 error_message，写入 generated_code
   result_analyzer → 读取 execution_result，写入分析结论
   report_generator → 读取所有结果，写入 report
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage

from graph.state import AgentState

from tools.file_reader import read_data_file,get_data_summary
from tools.python_repl import execute_python_code

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LLM_MODEL,
    MAX_RETRIES,
    UPLOAD_DIR,
)

def _get_llm() -> ChatOpenAI:
    """
    创建 LLM 实例的辅助函数

    为什么要单独写一个函数？
    - 多个节点都需要创建 LLM 实例
    - 统一配置，避免重复代码
    - 修改模型配置时只改一处
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL or None,
        temperature=0.1,  # 低温度：代码生成需要确定性，不要"创意"
    )

# ============================================================
# 节点1：数据读取
# ============================================================
def data_reader_node(state:AgentState)->dict:
    """
    数据读取节点：读取用户上传的数据文件，生成数据摘要

    工作流程：
    1. 从 state 中获取文件路径
    2. 用 file_reader 工具读取数据
    3. 生成数据摘要（列名、行数、数据类型等）
    4. 返回摘要信息，供后续节点使用
    为什么需要数据摘要？
    - LLM 不能直接读取文件（太大了）
    - 摘要把关键信息浓缩成文字
    - LLM 根据摘要来规划分析步骤
    """
    file_path = state.get('data_file_path','')
    if not file_path:
        return {'data_summary':'错误，未提供数据文件路径'}
    
    try:
        #读取数据文件
        df = read_data_file(file_path)
        #生成数据摘要
        summary = get_data_summary(df)

        return {'data_summary':summary}

    
    except Exception as e:
        return {"data_summary": f"读取数据文件失败: {str(e)}"}


# ============================================================
# 节点2：分析规划
# ============================================================
def analysis_planner_node(state:AgentState)->dict:
    """
    分析规划节点：根据用户需求和数据摘要，规划分析步骤

    工作流程：
    1. 读取用户需求和数据摘要
    2. 调用 LLM 规划分析步骤
    3. 返回分析计划

    这个节点的价值：
    - 不是直接让 AI 写代码，而是先"想清楚"步骤
    - 类似人做分析前先列大纲
    - 好的规划 = 好的代码
    """
    llm = _get_llm()

    # 构建 Prompt
    system_prompt = """你是一个数据分析专家。根据用户提供的数据摘要和分析需求，规划详细的分析步骤。

要求：
1. 步骤要具体、可执行
2. 每个步骤说明要做什么分析、用什么方法
3. 考虑数据的实际特点（列名、数据类型等）
4. 步骤之间要有逻辑顺序

输出格式：
1. [步骤1描述]
2. [步骤2描述]
..."""

    user_prompt = f"""数据摘要：
{state.get('data_summary', '')}

用户需求：{state.get('user_request', '')}

请规划分析步骤："""

    #调用llm
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {'analysis_plan':response.content}


# ============================================================
# 节点3：代码生成
# ============================================================
def code_generator_node(state:AgentState)->dict:
    """
    代码生成节点：根据分析计划生成 Python 数据分析代码

    工作流程：
    1. 读取分析计划和数据摘要
    2. 调用 LLM 生成 pandas + matplotlib 代码
    3. 返回生成的代码

    Prompt 设计要点：
    - 明确告诉 LLM 可以用 pd, plt, np（已预导入）
    - 告诉 LLM 数据文件路径（代码需要读取文件）
    - 要求代码健壮（处理缺失值、异常等）
    - 要求用 print() 输出关键结果
    - 要求用 plt.savefig() 保存图表
    """

    llm = _get_llm()
    system_prompt = """你是一个 Python 数据分析代码生成专家。

规则：
1. 可以直接使用 pd（pandas）、plt（matplotlib.pyplot）、np（numpy），不需要 import
2. 数据文件路径在代码中用变量 file_path 表示
3. 用 print() 输出关键分析结果
4. 用 plt.savefig('chart_描述.png', dpi=150, bbox_inches='tight') 保存图表
5. 每画一个图表后调用 plt.close() 关闭
6. 处理可能的缺失值和异常
7. 只输出 Python 代码，不要解释文字
8. 代码必须完整可执行"""

    user_prompt = f"""数据摘要：
{state.get('data_summary', '')}

分析计划：
{state.get('analysis_plan', '')}

数据文件路径：{state.get('data_file_path', '')}

请生成完整的 Python 分析代码："""

    #调用llm
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    #提取代码
    code = response.content
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    return {"generated_code": code.strip()}

# ============================================================
# 节点4：代码执行
# ============================================================
def code_executor_node(state:AgentState)->dict:
    """
    代码执行节点：在沙箱中执行生成的代码

    工作流程：
    1. 从 state 中获取生成的代码
    2. 调用 python_repl 工具执行代码
    3. 根据执行结果更新 state

    执行结果可能是：
    - 成功：返回输出内容和图表路径
    - 失败：返回错误信息（交给 code_fixer 修复）
    """
    code = state.get('generated_code','')

    if not code:
        return {
            "execution_success": False,
            "error_message": "没有可执行的代码",
            "execution_result": "",
        }
    
    #执行代码
    result = execute_python_code(code)

    #更新state
    update = {
        "execution_success": result["success"],
        "execution_result": result.get("output", ""),
        "chart_paths": result.get("figures", []),
    }

    if not result["success"]:
        update["error_message"] = result.get("error", "未知错误")
    else:
        update["error_message"] = ""

    return update

# ============================================================
# 节点5：代码修复
# ===========================================================
def code_fixer_node(state: AgentState) -> dict:
    """
    代码修复节点：当代码执行失败时，尝试修复代码

    工作流程：
    1. 读取原始代码和错误信息
    2. 调用 LLM 分析错误并修复
    3. 返回修复后的代码

    重试控制：
    - retry_count 记录已重试次数
    - 超过 MAX_RETRIES 就不再修复，直接报错
    - 防止无限循环（修不好 → 再修 → 还是修不好...）
    """
    retry_count = state.get("retry_count", 0)

    # 检查重试次数
    if retry_count >= MAX_RETRIES:
        return {
            "error_message": f"代码修复已重试 {MAX_RETRIES} 次仍失败，请检查需求描述。",
            "generated_code": state.get("generated_code", ""),
        }

    llm = _get_llm()

    system_prompt = """你是一个 Python 代码修复专家。
根据错误信息修复代码，只输出修复后的完整代码，不要解释。

常见错误修复方法：
- NameError: 变量名拼写错误或未定义
- KeyError: DataFrame 列名不存在，检查列名拼写
- TypeError: 数据类型不匹配，用 astype() 转换
- FileNotFoundError: 文件路径错误
- IndexError: 索引越界"""

    user_prompt = f"""原始代码：
```python
{state.get('generated_code', '')}
```

错误信息：
{state.get('error_message', '')}

数据摘要：
{state.get('data_summary', '')}

请修复代码："""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # 提取修复后的代码
    code = response.content
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    return {
        "generated_code": code.strip(),
        "retry_count": retry_count + 1,  # 增加重试计数
    }

# ============================================================
# 节点6：结果分析
# ============================================================
def result_analyzer_node(state:AgentState)->dict:
    """
    结果分析节点：分析代码执行结果，提取关键发现

    工作流程：
    1. 读取代码执行输出
    2. 调用 LLM 分析结果
    3. 返回分析结论

    这个节点的价值：
    - 原始输出可能是表格、数字，不够直观
    - LLM 能"读懂"数据，提炼出关键发现
    - 为最终报告提供分析素材
    """
    llm = _get_llm()

    system_prompt = """你是一个数据分析专家。根据代码执行结果，提炼关键发现和洞察。

要求：
1. 用清晰的语言总结分析结果
2. 突出最重要的发现
3. 如果有数字，用人类可读的方式表达（如"约1.2万"）
4. 指出可能的趋势或异常"""

    user_prompt = f"""用户需求：{state.get('user_request', '')}

分析计划：
{state.get('analysis_plan', '')}

代码执行结果：
{state.get('execution_result', '')}

请分析结果："""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {'report':response.content}

# ============================================================
# 节点7：报告生成
# ============================================================
def report_generator_node(state:AgentState)->dict:
    """
    报告生成节点：整合所有信息，生成最终分析报告

    工作流程：
    1. 读取分析结果、图表信息
    2. 调用 LLM 生成结构化报告
    3. 返回最终报告

    报告的结构：
    - 标题
    - 分析背景（用户需求）
    - 关键发现
    - 图表说明
    - 结论和建议
    """
    llm = _get_llm()

    system_prompt = """你是一个数据分析报告撰写专家。根据分析结果生成结构化的分析报告。

报告格式：
# 数据分析报告

## 分析背景
[用户需求描述]

## 关键发现
[主要分析结论]

## 图表说明
[每个图表的解读]

## 结论与建议
[总结和建议]"""

    chart_info = ""
    chart_paths = state.get("chart_paths", [])
    if chart_paths:
        chart_info = f"生成了 {len(chart_paths)} 个图表：\n"
        for i, path in enumerate(chart_paths, 1):
            chart_info += f"  图表{i}: {path}\n"

    user_prompt = f"""用户需求：{state.get('user_request', '')}

分析结论：
{state.get('report', '')}

{chart_info}

请生成完整的分析报告："""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {"report": response.content}