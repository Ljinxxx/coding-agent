import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import Agent, AgentMaxStepsError
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        self.call_count += 1
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )
        return self.responses.pop(0)


class LoopingFakeLLM:
    def __init__(self, allowed_calls: int) -> None:
        self.allowed_calls = allowed_calls
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        self.call_count += 1

        if self.call_count > self.allowed_calls:
            raise AssertionError("Agent must not make an extra LLM call.")

        return _tool_response(
            (
                f"call_{self.call_count}",
                "echo",
                {"value": f"step-{self.call_count}"},
            )
        )


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
        self.execute_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.execute_count += 1
        self.calls.append(dict(kwargs))
        return str(kwargs["value"])


def _text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def _tool_response(
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


def _registry_with_echo(tool: EchoTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def test_agent_has_default_max_steps() -> None:
    agent = Agent(FakeLLM([]), ToolRegistry())

    assert agent.max_steps == 20


def test_agent_rejects_invalid_max_steps() -> None:
    for invalid_max_steps in (0, -1):
        with pytest.raises(ValueError, match="max_steps"):
            Agent(
                FakeLLM([]),
                ToolRegistry(),
                max_steps=invalid_max_steps,
            )


def test_agent_finishes_within_max_steps() -> None:
    tool = EchoTool()
    llm = FakeLLM(
        [
            _tool_response(("call_1", "echo", {"value": "stage7"})),
            _text_response("finished"),
        ]
    )
    agent = Agent(llm, _registry_with_echo(tool), max_steps=2)

    result = agent.run("test")

    assert result == "finished"
    assert llm.call_count == 2
    assert tool.calls == [{"value": "stage7"}]


def test_agent_stops_at_max_steps() -> None:
    tool = EchoTool()
    llm = LoopingFakeLLM(allowed_calls=3)
    agent = Agent(llm, _registry_with_echo(tool), max_steps=3)

    with pytest.raises(AgentMaxStepsError, match="3"):
        agent.run("test")

    assert llm.call_count == 3
    assert tool.execute_count == 3
    assert tool.calls == [
        {"value": "step-1"},
        {"value": "step-2"},
        {"value": "step-3"},
    ]


def test_multiple_tool_calls_count_as_one_step() -> None:
    tool = EchoTool()
    llm = FakeLLM(
        [
            _tool_response(
                ("call_a", "echo", {"value": "first"}),
                ("call_b", "echo", {"value": "second"}),
            ),
            _text_response("finished"),
        ]
    )
    agent = Agent(llm, _registry_with_echo(tool), max_steps=2)

    result = agent.run("test")

    assert result == "finished"
    assert llm.call_count == 2
    assert tool.execute_count == 2
    assert tool.calls == [
        {"value": "first"},
        {"value": "second"},
    ]
