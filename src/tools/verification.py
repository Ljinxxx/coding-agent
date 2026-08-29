import json
from pathlib import Path
from typing import Any, Iterable

from src.tools.base import BaseTool
from src.tools.shell import RunCommandTool


class VerifyWorkspaceTool(BaseTool):
    name = "verify_workspace"
    description = "Run the host-configured workspace verification checks."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path,
        commands: Iterable[str],
        timeout: int = 30,
    ) -> None:
        if isinstance(commands, (str, bytes)):
            raise ValueError("commands must be an iterable of command strings.")
        try:
            configured_commands = tuple(commands)
        except TypeError as error:
            raise ValueError(
                "commands must be an iterable of command strings."
            ) from error
        if not configured_commands or any(
            not isinstance(command, str) or not command.strip()
            for command in configured_commands
        ):
            raise ValueError("commands must contain at least one non-empty command.")
        if timeout < 1:
            raise ValueError("timeout must be at least 1 second.")

        self.commands = configured_commands
        self.timeout = timeout
        self._runner = RunCommandTool(workspace)
        self.workspace = self._runner.workspace

    def execute(self, **kwargs: Any) -> str:
        if kwargs:
            raise TypeError("verify_workspace does not accept arguments.")

        checks: list[dict[str, Any]] = []
        ok = True

        for command in self.commands:
            command_result = json.loads(
                self._runner.execute(
                    command=command,
                    timeout=self.timeout,
                )
            )
            check = {
                "command": command,
                "exit_code": command_result["exit_code"],
                "stdout": command_result["stdout"],
                "stderr": command_result["stderr"],
                "timed_out": command_result["timed_out"],
            }
            checks.append(check)

            if check["exit_code"] != 0 or check["timed_out"] is not False:
                ok = False
                break

        return json.dumps(
            {
                "ok": ok,
                "checks": checks,
            },
            ensure_ascii=False,
        )
