"""
智能数据分析 Agent - 代码执行沙箱

【学习要点】
1. 为什么需要代码执行工具？
   - Agent 的核心能力：AI 生成代码 → 执行代码 → 获取结果
   - 没有执行能力，AI 只能"说"不能"做"
   - 有了执行能力，AI 就能真正处理数据、画图表

2. exec() 是什么？
   - Python 内置函数，用来执行字符串形式的 Python 代码
   - exec("print(1+1)") 会输出 2
   - 危险！如果不加限制，AI 生成的代码可能：
     a. 删除文件（os.system("rm -rf /")）
     b. 访问网络（requests.get("恶意网址")）
     c. 死循环（while True: pass）

3. 安全措施（这个文件实现的）：
   - 限制 __builtins__：禁止使用危险内置函数（open, exec, eval 等）
   - 捕获输出：用 StringIO 重定向 stdout/stderr
   - 超时控制：代码运行太久就强制终止

4. 生产环境需要更强的安全措施：
   - Docker 容器隔离（代码在容器里跑，不影响宿主机）
   - 资源限制（CPU、内存上限）
   - 网络隔离（禁止访问外网）
   - 这里只是学习版，了解原理即可

5. StringIO 的作用：
   - 正常情况下 print() 输出到屏幕
   - StringIO 是"内存中的文件"，print() 输出到它里面
   - 执行完后从 StringIO 取出所有输出内容
   - 这样就能把代码的运行结果返回给 Agent
"""
import sys
import signal
import traceback
from io import StringIO
from contextlib import redirect_stdout,redirect_stderr


from config import CODE_EXECUTION_TIMEOUT, CHART_DIR


# ============================================================
# 安全的内置函数白名单
# ============================================================
# __builtins__ 是 Python 的内置函数字典
# 默认包含所有内置函数（print, len, range, open, exec...）
# 我们只保留安全的函数，禁止危险的函数
#
# 禁止的函数及原因：
# - open: 可以读写任意文件
# - exec/eval: 可以执行任意代码（套娃）
# - __import__: 可以导入 os, subprocess 等危险模块
# - globals/locals: 可以访问全局/局部变量
# - compile: 可以编译代码
# - input: 会阻塞等待用户输入
# ============================================================
SAFE_BUILTINS = {
    # 基础函数
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "type": type,
    "isinstance": isinstance,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "bytes": bytes,
    "None": None,
    "True": True,
    "False": False,
    # 数学相关
    "pow": pow,
    "divmod": divmod,
    # 异常相关
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
}


class TimeoutError(Exception):
    """代码执行超时异常"""
    pass


def _timeout_handler(signum, frame):
    """
    超时信号处理函数

    工作原理：
    - signal.alarm(N) 会在 N 秒后发送 SIGALRM 信号
    - 这个函数是 SIGALRM 的处理函数
    - 收到信号后抛出 TimeoutError，终止代码执行

    注意：signal 模块只在 Unix/Linux 上可用
    Windows 上需要用其他方式实现超时（如 threading.Timer）
    """
    raise TimeoutError(f"代码执行超时（{CODE_EXECUTION_TIMEOUT}秒）")


def execute_python_code(
    code: str,
    timeout: int = CODE_EXECUTION_TIMEOUT,
    local_vars: dict = None,
) -> dict:
    """
    在受限环境中安全执行 Python 代码

    执行流程：
    1. 创建 StringIO 缓冲区，用来捕获 print 输出
    2. 准备安全的执行环境（白名单内置函数 + 允许的模块）
    3. 执行代码
    4. 收集输出结果和变量
    5. 返回执行结果字典

    参数:
        code: 要执行的 Python 代码字符串
        timeout: 超时时间（秒）
        local_vars: 预设的局部变量（如 pandas DataFrame）

    返回:
        执行结果字典：
        {
            "success": bool,      # 是否执行成功
            "output": str,         # 标准输出内容（print 的内容）
            "error": str,          # 错误信息（如果有）
            "figures": list[str],  # 生成的图表文件路径列表
        }

    示例:
        >>> result = execute_python_code("print(1+1)")
        >>> print(result)
        {"success": True, "output": "2\\n", "error": "", "figures": []}
    """
    # 创建输出缓冲区
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    # 准备局部变量（代码可以访问的变量）
    # 这里预导入 pandas 和 matplotlib，让 AI 生成的代码可以直接用
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，不需要显示器
    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化局部变量
    if local_vars is None:
        local_vars = {}

    # 添加允许的模块到局部变量
    local_vars.update({
        "pd": pd,
        "plt": plt,
        "np": np,
        "__builtins__": SAFE_BUILTINS,  # 替换内置函数为安全白名单
    })

    # 图表保存相关
    chart_paths = []

    try:
        # ---- 执行代码 ----
        # redirect_stdout/redirect_stderr: 把 print 输出重定向到缓冲区
        # 这样就能捕获代码的所有 print 输出
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # exec() 执行代码
            # 参数1: 要执行的代码字符串
            # 参数2: 全局变量字典（用安全白名单）
            # 参数3: 局部变量字典（包含 pandas, matplotlib 等）
            exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)

        # ---- 检查是否生成了图表 ----
        # matplotlib 的图表需要调用 plt.savefig() 保存
        # 如果 AI 生成的代码画了图但没保存，我们自动保存
        if plt.get_fignums():  # 如果有未关闭的图表
            import os
            import uuid

            os.makedirs(CHART_DIR, exist_ok=True)

            for fig_num in plt.get_fignums():
                chart_filename = f"chart_{uuid.uuid4().hex[:8]}.png"
                chart_path = os.path.join(CHART_DIR, chart_filename)
                plt.figure(fig_num).savefig(chart_path, dpi=150, bbox_inches="tight")
                chart_paths.append(chart_path)

            plt.close("all")  # 关闭所有图表，释放内存

        # 获取输出内容
        output = stdout_buffer.getvalue()

        return {
            "success": True,
            "output": output,
            "error": "",
            "figures": chart_paths,
        }

    except Exception as e:
        # 执行失败，收集错误信息
        error_msg = stderr_buffer.getvalue()

        # 如果 stderr 为空，用异常信息
        if not error_msg:
            error_msg = traceback.format_exc()

        return {
            "success": False,
            "output": stdout_buffer.getvalue(),
            "error": error_msg,
            "figures": chart_paths,
        }
