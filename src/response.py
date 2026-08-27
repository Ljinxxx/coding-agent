from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ParsedResponse:
    content: str
    tool_calls: list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
