import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.tools.shell import RunCommandTool


def make_python_command(code: str) -> str:
    return subprocess.list2cmdline([sys.executable, "-c", code])


def paths_match(first: str | Path, second: str | Path) -> bool:
    first_path = os.path.normcase(str(Path(first).resolve()))
    second_path = os.path.normcase(str(Path(second).resolve()))
    return first_path == second_path


def run_python(tool: RunCommandTool, title: str, code: str) -> dict[str, Any]:
    result = json.loads(tool.execute(command=make_python_command(code)))

    print(f"\n{title}")
    print(f"退出码：{result['exit_code']}")
    print(f"标准输出：{result['stdout'].strip()}")
    print(f"标准错误：{result['stderr'].strip()}")
    print(f"是否超时：{result['timed_out']}")

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parent_pid = os.getpid()
    tool = RunCommandTool(project_root)

    print(f"验证工作目录：{project_root}")

    text_result = run_python(
        tool,
        "1. 执行文本输出命令",
        'print("stage5-ok")',
    )
    workspace_result = run_python(
        tool,
        "2. 验证命令工作目录",
        "from pathlib import Path; print(Path.cwd())",
    )
    calculation_result = run_python(
        tool,
        "3. 执行 Python 计算",
        "print(2 + 3)",
    )
    child_result = run_python(
        tool,
        "4. 验证真实本地 Python 子进程",
        (
            "import json, os, sys; "
            "from pathlib import Path; "
            'print(json.dumps({"pid": os.getpid(), "python": sys.executable, '
            '"workspace": str(Path.cwd())}))'
        ),
    )

    try:
        child_info = json.loads(child_result["stdout"])
        child_pid = int(child_info["pid"])
        child_python = str(child_info["python"])
        child_workspace = str(child_info["workspace"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Failed to parse child process details.") from error

    print(f"验证脚本 PID：{parent_pid}")
    print(f"子进程 PID：{child_pid}")
    print(f"当前 Python 路径：{sys.executable}")
    print(f"子进程 Python 路径：{child_python}")
    print(f"子进程工作目录：{child_workspace}")

    text_ok = (
        text_result["exit_code"] == 0
        and text_result["stdout"].strip() == "stage5-ok"
        and text_result["timed_out"] is False
    )
    workspace_ok = (
        workspace_result["exit_code"] == 0
        and paths_match(workspace_result["stdout"].strip(), project_root)
        and workspace_result["timed_out"] is False
    )
    calculation_ok = (
        calculation_result["exit_code"] == 0
        and calculation_result["stdout"].strip() == "5"
        and calculation_result["timed_out"] is False
    )
    child_process_ok = (
        child_result["exit_code"] == 0
        and child_result["timed_out"] is False
        and child_pid != parent_pid
        and paths_match(child_python, sys.executable)
        and paths_match(child_workspace, project_root)
    )

    checks = {
        "text output": text_ok,
        "workspace": workspace_ok,
        "calculation": calculation_ok,
        "real Python subprocess": child_process_ok,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    if failed_checks:
        failed = ", ".join(failed_checks)
        raise RuntimeError(f"Stage 5 verification failed: {failed}.")

    print("\n5. 验证结果")
    print("Stage 5 本地命令执行验证成功")


if __name__ == "__main__":
    main()
