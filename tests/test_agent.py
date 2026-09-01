import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from src.agent import Agent
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


def _without_execution_budget(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    budget_messages = [
        message
        for message in messages
        if str(message.get("content") or "").startswith(
            "[Execution Budget]\n"
        )
    ]
    assert len(budget_messages) == 1
    return [message for message in messages if message not in budget_messages]


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )
        return self.responses.pop(0)


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the provided value."
    parameters = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
            },
        },
        "required": ["value"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return str(kwargs["value"])


class SecretTool(BaseTool):
    name = "read_secret"
    description = "Return a value only available at runtime."
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.execute_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.execute_count += 1
        return self.secret


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def tool_response(
    *calls: tuple[str, str, dict[str, Any]],
) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(arguments),
                ),
            )
            for call_id, name, arguments in calls
        ],
    )


def test_agent_returns_direct_text_response() -> None:
    llm = FakeLLM([text_response("stage6-done")])
    agent = Agent(llm, ToolRegistry())

    result = agent.run("test")

    assert result == "stage6-done"
    assert len(llm.calls) == 1


def test_agent_executes_tool_and_continues() -> None:
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    llm = FakeLLM(
        [
            tool_response(("call_1", "echo", {"value": "stage6"})),
            text_response("final-answer"),
        ]
    )

    result = Agent(llm, registry).run("test")

    assert result == "final-answer"
    assert tool.calls == [{"value": "stage6"}]
    assert len(llm.calls) == 2


def test_agent_sends_tool_result_back_to_model() -> None:
    tool = SecretTool("runtime-secret-123")
    registry = ToolRegistry()
    registry.register(tool)
    llm = FakeLLM(
        [
            tool_response(("call_secret", "read_secret", {})),
            text_response("done"),
        ]
    )

    Agent(llm, registry).run("test")

    tool_message = llm.calls[1]["messages"][-1]
    assert tool.execute_count == 1
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call_secret",
        "content": "runtime-secret-123",
    }


def test_agent_preserves_assistant_tool_call_message() -> None:
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    llm = FakeLLM(
        [
            tool_response(("call_1", "echo", {"value": "stage6"})),
            text_response("done"),
        ]
    )

    Agent(llm, registry).run("test")

    messages = _without_execution_budget(llm.calls[1]["messages"])
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"value": "stage6"}',
                },
            }
        ],
    }
    assert messages[1]["tool_calls"][0]["id"] == messages[2]["tool_call_id"]


def test_agent_executes_multiple_tool_calls() -> None:
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    llm = FakeLLM(
        [
            tool_response(
                ("call_a", "echo", {"value": "first"}),
                ("call_b", "echo", {"value": "second"}),
            ),
            text_response("done"),
        ]
    )

    result = Agent(llm, registry).run("test")

    assert result == "done"
    assert len(llm.calls) == 2
    assert tool.calls == [{"value": "first"}, {"value": "second"}]
    assistant_message = llm.calls[1]["messages"][-3]
    tool_messages = llm.calls[1]["messages"][-2:]
    assert [call["id"] for call in assistant_message["tool_calls"]] == [
        "call_a",
        "call_b",
    ]
    assert [message["role"] for message in tool_messages] == ["tool", "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call_a",
        "call_b",
    ]
    assert [message["content"] for message in tool_messages] == [
        "first",
        "second",
    ]


def test_agent_passes_registered_tool_schemas_to_model() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    llm = FakeLLM(
        [
            tool_response(("call_1", "echo", {"value": "stage6"})),
            text_response("done"),
        ]
    )

    Agent(llm, registry).run("test")

    expected_schemas = registry.schemas()
    assert len(llm.calls) == 2
    assert all(call["tools"] == expected_schemas for call in llm.calls)
