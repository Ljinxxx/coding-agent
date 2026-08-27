import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLMClient
from src.parser import ResponseParser
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


def main() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    messages = [
        {
            "role": "system",
            "content": (
                "You are testing tool calling. When requested, use the provided "
                "tool instead of answering directly."
            ),
        },
        {
            "role": "user",
            "content": (
                'Please call the echo tool with its text argument set to "stage3-ok". '
                "Do not answer directly."
            ),
        },
    ]

    message = LLMClient().chat(messages, tools=registry.schemas())
    parsed = ResponseParser().parse(message)

    if not parsed.has_tool_calls:
        raise RuntimeError("The model did not produce a tool call.")

    print("检测到工具调用")
    for tool_call in parsed.tool_calls:
        print(f"\n工具名称：{tool_call.name}")
        print(f"工具参数：{tool_call.arguments}")


if __name__ == "__main__":
    main()
