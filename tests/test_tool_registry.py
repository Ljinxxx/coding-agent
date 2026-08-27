from typing import Any

import pytest

from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the provided text."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
            },
        },
        "required": ["text"],
    }

    def execute(self, **kwargs: Any) -> str:
        return str(kwargs["text"])


def test_register_and_get_tool() -> None:
    registry = ToolRegistry()
    tool = EchoTool()

    registry.register(tool)

    assert registry.get("echo") is tool
    assert registry.names() == ["echo"]


def test_tool_schema() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    schemas = registry.schemas()

    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Return the provided text.",
                "parameters": EchoTool.parameters,
            },
        }
    ]


def test_duplicate_tool_registration() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="Tool 'echo' is already registered\\."):
        registry.register(EchoTool())


def test_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="Tool 'unknown' is not registered\\."):
        registry.get("unknown")
