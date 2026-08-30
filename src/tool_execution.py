import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    content: str
    execution_ok: bool
    error_type: str | None = None


def _format_execution_error(
    *,
    tool_name: str,
    error_type: str,
    message: str,
) -> str:
    return json.dumps(
        {
            "ok": False,
            "tool": tool_name,
            "error_type": error_type,
            "message": message or "Tool execution failed.",
        },
        ensure_ascii=False,
    )


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        before_execute: Callable[[BaseTool], None] | None = None,
    ) -> ToolExecutionResult:
        try:
            tool = self._registry.get(tool_name)
        except KeyError:
            error_type = "UnknownTool"
            return ToolExecutionResult(
                tool_name=tool_name,
                content=_format_execution_error(
                    tool_name=tool_name,
                    error_type=error_type,
                    message=f"Tool '{tool_name}' is not registered.",
                ),
                execution_ok=False,
                error_type=error_type,
            )

        try:
            if before_execute is not None:
                before_execute(tool)
            content = tool.execute(**arguments)
        except Exception as error:
            error_type = type(error).__name__
            return ToolExecutionResult(
                tool_name=tool_name,
                content=_format_execution_error(
                    tool_name=tool_name,
                    error_type=error_type,
                    message=str(error),
                ),
                execution_ok=False,
                error_type=error_type,
            )

        return ToolExecutionResult(
            tool_name=tool_name,
            content=content,
            execution_ok=True,
        )
