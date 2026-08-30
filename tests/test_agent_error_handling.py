import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import Agent, AgentLLMError, AgentResponseError
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


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
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


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


class FailingTool(BaseTool):
    name = "fail"
    description = "Always fail with a runtime error."
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self) -> None:
        self.execute_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.execute_count += 1
        raise RuntimeError("intentional failure")


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


def _registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_unknown_tool_error_is_sent_back_to_model() -> None:
    llm = FakeLLM(
        [
            _tool_response(("call_unknown", "unknown_tool", {})),
            _text_response("recovered"),
        ]
    )

    result = Agent(llm, ToolRegistry()).run("test")

    assert result == "recovered"
    assert len(llm.calls) == 2
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_unknown"
    error_result = json.loads(tool_message["content"])
    assert set(error_result) == {"ok", "tool", "error_type", "message"}
    assert error_result["ok"] is False
    assert error_result["tool"] == "unknown_tool"
    assert error_result["error_type"] == "UnknownTool"
    assert error_result["message"]


def test_agent_recovers_after_tool_execution_error() -> None:
    failing_tool = FailingTool()
    echo_tool = EchoTool()
    llm = FakeLLM(
        [
            _tool_response(("call_fail", "fail", {})),
            _tool_response(
                ("call_recovery", "echo", {"value": "recovered"})
            ),
            _text_response("recovery-complete"),
        ]
    )
    agent = Agent(llm, _registry_with(failing_tool, echo_tool))

    result = agent.run("test")

    assert result == "recovery-complete"
    assert len(llm.calls) == 3
    assert failing_tool.execute_count == 1
    assert echo_tool.calls == [{"value": "recovered"}]

    error_message = llm.calls[1]["messages"][-1]
    assert error_message["tool_call_id"] == "call_fail"
    assert json.loads(error_message["content"]) == {
        "ok": False,
        "tool": "fail",
        "error_type": "RuntimeError",
        "message": "intentional failure",
    }

    recovery_message = llm.calls[2]["messages"][-1]
    assert recovery_message == {
        "role": "tool",
        "tool_call_id": "call_recovery",
        "content": "recovered",
    }


def test_one_failed_tool_does_not_block_other_tool_calls() -> None:
    echo_tool = EchoTool()
    failing_tool = FailingTool()
    llm = FakeLLM(
        [
            _tool_response(
                ("call_first", "echo", {"value": "first"}),
                ("call_fail", "fail", {}),
                ("call_third", "echo", {"value": "third"}),
            ),
            _text_response("done"),
        ]
    )
    agent = Agent(llm, _registry_with(echo_tool, failing_tool))

    result = agent.run("test")

    assert result == "done"
    assert echo_tool.calls == [
        {"value": "first"},
        {"value": "third"},
    ]
    assert failing_tool.execute_count == 1

    tool_messages = llm.calls[1]["messages"][-3:]
    assert [message["role"] for message in tool_messages] == [
        "tool",
        "tool",
        "tool",
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call_first",
        "call_fail",
        "call_third",
    ]
    assert tool_messages[0]["content"] == "first"
    assert json.loads(tool_messages[1]["content"])["ok"] is False
    assert tool_messages[2]["content"] == "third"


def test_llm_error_is_wrapped_as_agent_llm_error() -> None:
    original_error = RuntimeError("api unavailable")
    llm = FakeLLM([original_error])

    with pytest.raises(AgentLLMError) as error_info:
        Agent(llm, ToolRegistry()).run("test")

    assert error_info.value.__cause__ is original_error
    assert len(llm.calls) == 1


def test_parser_error_is_wrapped_as_agent_response_error() -> None:
    malformed_response = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_bad_json",
                function=SimpleNamespace(
                    name="echo",
                    arguments="{not valid json",
                ),
            )
        ],
    )
    echo_tool = EchoTool()
    llm = FakeLLM([malformed_response])

    with pytest.raises(AgentResponseError) as error_info:
        Agent(llm, _registry_with(echo_tool)).run("test")

    assert isinstance(error_info.value.__cause__, ValueError)
    assert echo_tool.calls == []
