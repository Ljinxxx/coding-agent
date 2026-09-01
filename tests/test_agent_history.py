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


def test_agent_saves_user_and_final_assistant_messages() -> None:
    llm = FakeLLM([_text_response("final-answer")])
    agent = Agent(llm, ToolRegistry())

    result = agent.run("hello")

    assert result == "final-answer"
    assert agent.history == [
        {
            "role": "user",
            "content": "hello",
        },
        {
            "role": "assistant",
            "content": "final-answer",
        },
    ]


def test_second_run_receives_previous_conversation_history() -> None:
    llm = FakeLLM(
        [
            _text_response("first-answer"),
            _text_response("second-answer"),
        ]
    )
    agent = Agent(llm, ToolRegistry())

    first_result = agent.run("first-message")
    second_result = agent.run("second-message")

    assert first_result == "first-answer"
    assert second_result == "second-answer"
    assert _without_execution_budget(llm.calls[1]["messages"]) == [
        {
            "role": "user",
            "content": "first-message",
        },
        {
            "role": "assistant",
            "content": "first-answer",
        },
        {
            "role": "user",
            "content": "second-message",
        },
    ]


def test_tool_messages_are_preserved_across_runs() -> None:
    tool = EchoTool()
    llm = FakeLLM(
        [
            _tool_response(("call_echo", "echo", {"value": "abc"})),
            _text_response("done"),
            _text_response("continued"),
        ]
    )
    agent = Agent(llm, _registry_with_echo(tool))

    first_result = agent.run("use-tool")
    second_result = agent.run("continue")

    assert first_result == "done"
    assert second_result == "continued"
    assert tool.calls == [{"value": "abc"}]
    assert _without_execution_budget(llm.calls[2]["messages"]) == [
        {
            "role": "user",
            "content": "use-tool",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_echo",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"value": "abc"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_echo",
            "content": "abc",
        },
        {
            "role": "assistant",
            "content": "done",
        },
        {
            "role": "user",
            "content": "continue",
        },
    ]


def test_system_prompt_appears_only_once_across_runs() -> None:
    llm = FakeLLM(
        [
            _text_response("answer-one"),
            _text_response("answer-two"),
            _text_response("answer-three"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="test-system",
    )

    agent.run("one")
    agent.run("two")
    agent.run("three")

    history = agent.history
    system_messages = [
        message for message in history if message["role"] == "system"
    ]
    assert history[0] == {
        "role": "system",
        "content": "test-system",
    }
    assert system_messages == [history[0]]
    assert all(
        sum(
            message["role"] == "system"
            for message in _without_execution_budget(call["messages"])
        )
        == 1
        for call in llm.calls
    )


def test_reset_history_clears_conversation_but_keeps_system_prompt() -> None:
    llm = FakeLLM([_text_response("remembered")])
    registry = ToolRegistry()
    agent = Agent(
        llm,
        registry,
        system_prompt="test-system",
        verbose=True,
        max_steps=3,
    )
    parser = agent.parser

    agent.run("remember-me")
    assert any(
        message.get("content") == "remember-me"
        for message in agent.history
    )

    agent.reset_history()

    assert agent.history == [
        {
            "role": "system",
            "content": "test-system",
        }
    ]
    assert agent.llm_client is llm
    assert agent.tool_registry is registry
    assert agent.parser is parser
    assert agent.system_prompt == "test-system"
    assert agent.verbose is True
    assert agent.max_steps == 3

    no_system_agent = Agent(
        FakeLLM([_text_response("temporary-answer")]),
        ToolRegistry(),
    )
    no_system_agent.run("temporary-message")

    no_system_agent.reset_history()

    assert no_system_agent.history == []


def test_history_returns_deep_copy() -> None:
    tool = EchoTool()
    llm = FakeLLM(
        [
            _tool_response(("call_echo", "echo", {"value": "original"})),
            _text_response("done"),
        ]
    )
    agent = Agent(llm, _registry_with_echo(tool))
    agent.run("use-tool")
    expected_history = agent.history

    snapshot = agent.history
    assistant_tool_message = next(
        message for message in snapshot if "tool_calls" in message
    )
    function = assistant_tool_message["tool_calls"][0]["function"]
    function["name"] = "changed"
    function["arguments"] = '{"value": "changed"}'
    snapshot[0]["content"] = "changed-user"

    assert agent.history == expected_history
    actual_tool_message = next(
        message for message in agent.history if "tool_calls" in message
    )
    assert actual_tool_message["tool_calls"][0]["function"] == {
        "name": "echo",
        "arguments": '{"value": "original"}',
    }

    snapshot.clear()
    assert agent.history == expected_history
