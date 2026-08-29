import json
import subprocess
from pathlib import Path
from typing import Any

from src.tools.base import BaseTool


def _output_to_text(output: str | bytes | None) -> str:
    if output is None:
        return ""

    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")

    return output


class RunCommandTool(BaseTool):
    name = "run_command"
    description = (
        "Execute a shell command in the local workspace and return its exit status, "
        "standard output, standard error, and timeout status."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute in the workspace.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds.",
                "default": 30,
                "minimum": 1,
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve(strict=False)

    def execute(self, **kwargs: Any) -> str:
        command = str(kwargs["command"])
        timeout = int(kwargs.get("timeout", 30))

        if timeout < 1:
            raise ValueError("timeout must be at least 1 second.")

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = {
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            result = {
                "exit_code": None,
                "stdout": _output_to_text(error.stdout),
                "stderr": _output_to_text(error.stderr),
                "timed_out": True,
            }

        return json.dumps(result, ensure_ascii=False)
