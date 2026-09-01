import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import (
    Agent,
    AgentContextLimitError,
    _build_execution_budget_message,
)
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

    def execute(self, **kwargs: Any) -> str:
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


def _context_size(messages: list[dict[str, Any]]) -> int:
    return len(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _execution_budget_chars(
    *,
    current_step: int = 1,
    max_steps: int = 20,
) -> int:
    message = _build_execution_budget_message(current_step, max_steps)
    return _context_size([message]) - 1


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


def test_context_limit_disabled_sends_full_history() -> None:
    llm = FakeLLM(
        [
            _text_response("first-answer"),
            _text_response("second-answer"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="context-system",
        max_context_chars=None,
    )

    agent.run("first-question")
    history_before_second_run = agent.history
    agent.run("second-question")

    assert _without_execution_budget(llm.calls[1]["messages"]) == (
        history_before_second_run
        + [
            {
                "role": "user",
                "content": "second-question",
            }
        ]
    )


def test_context_trims_oldest_turns_and_keeps_recent_contiguous_window(
) -> None:
    llm = FakeLLM(
        [
            _text_response("oldest-small-answer"),
            _text_response("middle-blocker-" + "x" * 500),
            _text_response("recent-answer"),
            _text_response("current-answer"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="context-system",
    )
    agent.run("oldest-small-question")
    agent.run("middle-blocker-question")
    agent.run("recent-question")

    history = agent.history
    system_messages = history[:1]
    oldest_turn = history[1:3]
    middle_turn = history[3:5]
    recent_turn = history[5:7]
    current_user = {
        "role": "user",
        "content": "current-question",
    }
    expected_context = system_messages + recent_turn + [current_user]
    context_if_middle_were_added = (
        system_messages + middle_turn + recent_turn + [current_user]
    )
    context_if_middle_were_skipped = (
        system_messages + oldest_turn + recent_turn + [current_user]
    )
    budget = _context_size(
        context_if_middle_were_skipped
    ) + _execution_budget_chars()

    budget_chars = _execution_budget_chars()
    assert _context_size(expected_context) + budget_chars <= budget
    assert _context_size(context_if_middle_were_added) + budget_chars > budget
    agent.max_context_chars = budget

    agent.run("current-question")

    request_context = llm.calls[-1]["messages"]
    actual_context = _without_execution_budget(request_context)
    assert actual_context == expected_context
    assert actual_context[0]["role"] == "system"
    assert sum(
        message["role"] == "system" for message in actual_context
    ) == 1
    assert not any(
        "oldest-small" in str(message.get("content"))
        for message in request_context
    )
    assert not any(
        "middle-blocker" in str(message.get("content"))
        for message in request_context
    )


def test_context_trimming_does_not_modify_full_history() -> None:
    llm = FakeLLM(
        [
            _text_response("old-answer"),
            _text_response("recent-answer"),
            _text_response("current-answer"),
        ]
    )
    agent = Agent(llm, ToolRegistry())
    agent.run("old-question")
    agent.run("recent-question")

    history_before_current_run = agent.history
    recent_turn = history_before_current_run[-2:]
    current_user = {
        "role": "user",
        "content": "current-question",
    }
    expected_context = recent_turn + [current_user]
    full_context = history_before_current_run + [current_user]
    budget = _context_size(expected_context) + _execution_budget_chars()

    assert _context_size(full_context) + _execution_budget_chars() > budget
    agent.max_context_chars = budget

    agent.run("current-question")

    assert _without_execution_budget(llm.calls[-1]["messages"]) == (
        expected_context
    )
    assert agent.history == history_before_current_run + [
        current_user,
        {
            "role": "assistant",
            "content": "current-answer",
        },
    ]
    assert any(
        message.get("content") == "old-question"
        for message in agent.history
    )


def test_context_trimming_preserves_complete_tool_turn() -> None:
    echo = EchoTool()
    registry = ToolRegistry()
    registry.register(echo)
    llm = FakeLLM(
        [
            _text_response("old-answer"),
            _tool_response(
                ("call_context", "echo", {"value": "runtime-value"})
            ),
            _text_response("tool-turn-answer"),
            _text_response("current-answer"),
        ]
    )
    agent = Agent(
        llm,
        registry,
        system_prompt="context-system",
    )
    agent.run("old-question")
    agent.run("tool-turn-question")

    history = agent.history
    system_messages = history[:1]
    tool_turn = history[3:]
    current_user = {
        "role": "user",
        "content": "current-question",
    }
    expected_context = system_messages + tool_turn + [current_user]
    full_context = history + [current_user]
    budget = _context_size(expected_context) + _execution_budget_chars()

    assert _context_size(full_context) + _execution_budget_chars() > budget
    agent.max_context_chars = budget

    agent.run("current-question")

    actual_context = _without_execution_budget(
        llm.calls[-1]["messages"]
    )
    assert actual_context == expected_context
    assert [message["role"] for message in actual_context] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    tool_call_message = actual_context[2]
    tool_result_message = actual_context[3]
    assert tool_call_message["tool_calls"][0]["id"] == "call_context"
    assert tool_result_message == {
        "role": "tool",
        "tool_call_id": "call_context",
        "content": "runtime-value",
    }


def test_context_trimming_always_keeps_system_prompt() -> None:
    llm = FakeLLM(
        [
            _text_response("old-answer"),
            _text_response("current-answer"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="required-system-prompt",
    )
    agent.run("old-question")

    current_user = {
        "role": "user",
        "content": "current-question",
    }
    expected_context = [
        {
            "role": "system",
            "content": "required-system-prompt",
        },
        current_user,
    ]
    full_context = agent.history + [current_user]
    budget = _context_size(expected_context) + _execution_budget_chars()

    assert _context_size(full_context) + _execution_budget_chars() > budget
    agent.max_context_chars = budget

    agent.run("current-question")

    actual_context = _without_execution_budget(
        llm.calls[-1]["messages"]
    )
    assert actual_context == expected_context
    assert actual_context[0]["role"] == "system"
    assert sum(
        message["role"] == "system" for message in actual_context
    ) == 1


def test_current_turn_over_budget_raises_before_llm_call() -> None:
    mandatory_context = [
        {
            "role": "system",
            "content": "required-system-prompt",
        },
        {
            "role": "user",
            "content": "current-question",
        },
    ]
    budget = (
        _context_size(mandatory_context)
        + _execution_budget_chars()
        - 1
    )
    llm = FakeLLM([_text_response("must-not-be-used")])
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="required-system-prompt",
        max_context_chars=budget,
    )

    with pytest.raises(AgentContextLimitError):
        agent.run("current-question")

    assert issubclass(AgentContextLimitError, RuntimeError)
    assert llm.calls == []
    assert agent.history == mandatory_context


def test_max_context_chars_validation() -> None:
    with pytest.raises(ValueError):
        Agent(FakeLLM([]), ToolRegistry(), max_context_chars=0)

    with pytest.raises(ValueError):
        Agent(FakeLLM([]), ToolRegistry(), max_context_chars=-1)

    unlimited_agent = Agent(
        FakeLLM([]),
        ToolRegistry(),
        max_context_chars=None,
    )
    one_char_agent = Agent(
        FakeLLM([]),
        ToolRegistry(),
        max_context_chars=1,
    )
    limited_agent = Agent(
        FakeLLM([]),
        ToolRegistry(),
        system_prompt="context-system",
        max_context_chars=1000,
    )

    limited_agent.reset_history()

    assert unlimited_agent.max_context_chars is None
    assert one_char_agent.max_context_chars == 1
    assert limited_agent.max_context_chars == 1000
    assert limited_agent.history == [
        {
            "role": "system",
            "content": "context-system",
        }
    ]
