import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.tools.shell import RunCommandTool


def make_python_command(code: str) -> str:
    parts = [sys.executable, "-c", code]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def execute_python(
    tool: RunCommandTool,
    code: str,
    timeout: int = 30,
) -> dict[str, Any]:
    result = tool.execute(
        command=make_python_command(code),
        timeout=timeout,
    )
    return json.loads(result)


def test_run_command_schema(tmp_path: Path) -> None:
    tool = RunCommandTool(tmp_path)

    schema = tool.to_schema()
    parameters = schema["function"]["parameters"]

    assert tool.name == "run_command"
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "run_command"
    assert parameters["properties"]["command"]["type"] == "string"
    assert parameters["properties"]["timeout"]["type"] == "integer"
    assert parameters["properties"]["timeout"]["default"] == 30
    assert parameters["properties"]["timeout"]["minimum"] == 1
    assert parameters["required"] == ["command"]


def test_run_command_stdout(tmp_path: Path) -> None:
    result = execute_python(RunCommandTool(tmp_path), 'print("stage5-ok")')

    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "stage5-ok"
    assert result["stderr"] == ""
    assert result["timed_out"] is False


def test_run_command_uses_workspace(tmp_path: Path) -> None:
    result = execute_python(
        RunCommandTool(tmp_path),
        "from pathlib import Path; print(Path.cwd())",
    )

    assert result["exit_code"] == 0
    assert Path(result["stdout"].strip()).resolve() == tmp_path.resolve()
    assert result["timed_out"] is False


def test_run_command_nonzero_exit_code(tmp_path: Path) -> None:
    result = execute_python(
        RunCommandTool(tmp_path),
        "import sys; sys.exit(3)",
    )

    assert result["exit_code"] == 3
    assert result["timed_out"] is False


def test_run_command_stderr(tmp_path: Path) -> None:
    result = execute_python(
        RunCommandTool(tmp_path),
        'import sys; print("stage5-error", file=sys.stderr)',
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == ""
    assert result["stderr"].strip() == "stage5-error"
    assert result["timed_out"] is False


def test_run_command_timeout(tmp_path: Path) -> None:
    result = execute_python(
        RunCommandTool(tmp_path),
        "import time; time.sleep(2)",
        timeout=1,
    )

    assert result["exit_code"] is None
    assert result["timed_out"] is True
