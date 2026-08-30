import json
import subprocess
from pathlib import Path
from typing import Any

from src.tools.base import BaseTool


DEFAULT_SHELL_MAX_OUTPUT_CHARS = 20_000


def _output_to_text(output: str | bytes | None) -> str:
    if output is None:
        return ""

    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")

    return output


def _truncate_head_tail(
    output: str | bytes | None,
    max_chars: int,
    label: str,
) -> tuple[str, bool, int]:
    text = _output_to_text(output)
    original_chars = len(text)
    if original_chars <= max_chars:
        return text, False, original_chars

    marker = f"\n[... {label} truncated: original {original_chars} chars ...]\n"
    if len(marker) >= max_chars:
        return marker[:max_chars], True, original_chars

    remaining_chars = max_chars - len(marker)
    head_chars = (remaining_chars + 1) // 2
    tail_chars = remaining_chars - head_chars
    tail = text[-tail_chars:] if tail_chars else ""
    bounded = text[:head_chars] + marker + tail
    return bounded, True, original_chars


class RunCommandTool(BaseTool):
    name = "run_command"
    description = (
        "Execute a shell command in the local workspace and return its exit status, "
        "standard output, standard error, and timeout status."
    )
    mutates_workspace = True
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

    def __init__(
        self,
        workspace: Path,
        *,
        max_output_chars: int = DEFAULT_SHELL_MAX_OUTPUT_CHARS,
    ) -> None:
        if (
            isinstance(max_output_chars, bool)
            or not isinstance(max_output_chars, int)
            or max_output_chars < 1
        ):
            raise ValueError("max_output_chars must be a positive integer.")
        self.workspace = Path(workspace).resolve(strict=False)
        self.max_output_chars = max_output_chars

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
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            timed_out = False
        except subprocess.TimeoutExpired as error:
            exit_code = None
            stdout = error.stdout
            stderr = error.stderr
            timed_out = True

        bounded_stdout, stdout_truncated, stdout_original_chars = (
            _truncate_head_tail(
                stdout,
                self.max_output_chars,
                "stdout",
            )
        )
        bounded_stderr, stderr_truncated, stderr_original_chars = (
            _truncate_head_tail(
                stderr,
                self.max_output_chars,
                "stderr",
            )
        )
        result = {
            "exit_code": exit_code,
            "stdout": bounded_stdout,
            "stderr": bounded_stderr,
            "timed_out": timed_out,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout_original_chars": stdout_original_chars,
            "stderr_original_chars": stderr_original_chars,
        }

        return json.dumps(result, ensure_ascii=False)
