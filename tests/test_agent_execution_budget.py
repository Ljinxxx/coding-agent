import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import (
    Agent,
    AgentContextLimitError,
    AgentMaxStepsError,
    _build_execution_budget_message,
)
from src.context_compaction import (
    CURRENT_RUN_HEADER,
    PRIOR_CONTEXT_HEADER,
    estimate_messages_size,
)
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


EXECUTION_BUDGET_HEADER = "[Execution Budget]"


class RecordingLLM:
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
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, **kwargs: Any) -> str:
        value = str(kwargs["value"])
        self.calls.append(value)
        return value


def _text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def _tool_response(call_id: str, value: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name="echo",
                    arguments=json.dumps({"value": value}),
                ),
            )
        ],
    )


def _registry(tool: EchoTool | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    return registry


def _budget_messages(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in call["messages"]
        if str(message.get("content") or "").startswith(
            EXECUTION_BUDGET_HEADER
        )
    ]


@pytest.mark.parametrize(
    ("current_step", "max_steps", "remaining"),
    (
        (1, 20, 20),
        (19, 20, 2),
        (20, 20, 1),
        (35, 40, 6),
        (40, 40, 1),
    ),
)
def test_execution_budget_message_uses_including_current_step_math(
    current_step: int,
    max_steps: int,
    remaining: int,
) -> None:
    message = _build_execution_budget_message(current_step, max_steps)

    assert message["role"] == "system"
    assert EXECUTION_BUDGET_HEADER in message["content"]
    assert f"Current model step: {current_step} / {max_steps}" in message[
        "content"
    ]
    assert (
        "Remaining model responses including this one: "
        f"{remaining}"
    ) in message["content"]
    assert "finite" in message["content"].casefold()
    assert "concrete progress" in message["content"].casefold()
    assert "repeated inspection" in message["content"].casefold()


def test_each_request_has_one_current_budget_without_history_or_tool_pollution(
) -> None:
    tool = EchoTool()
    registry = _registry(tool)
    llm = RecordingLLM(
        [
            _tool_response("step-1", "one"),
            _tool_response("step-2", "two"),
            _text_response("finished"),
        ]
    )
    agent = Agent(llm, registry, max_steps=3)

    assert agent.run("original-user-message") == "finished"

    assert len(llm.calls) == 3
    for index, call in enumerate(llm.calls, start=1):
        budgets = _budget_messages(call)
        assert len(budgets) == 1
        assert f"Current model step: {index} / 3" in budgets[0]["content"]
        assert (
            "Remaining model responses including this one: "
            f"{4 - index}"
        ) in budgets[0]["content"]
        assert call["tools"] == registry.schemas()

    serialized_history = json.dumps(agent.history, ensure_ascii=False)
    assert EXECUTION_BUDGET_HEADER not in serialized_history
    assert {
        "role": "user",
        "content": "original-user-message",
    } in agent.history
    assert tool.calls == ["one", "two"]


def test_execution_budget_survives_compaction_and_resets_with_each_run() -> None:
    llm = RecordingLLM(
        [
            _text_response("old-answer-" + "a" * 900),
            _text_response("compacted-answer"),
            _text_response("after-reset"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="system",
        max_context_chars=1_300,
        compaction_trigger_chars=400,
        max_compaction_chars=450,
    )

    agent.run("old-question-" + "q" * 650)
    agent.run("current-question")

    compacted_request = llm.calls[1]["messages"]
    assert any(
        PRIOR_CONTEXT_HEADER in str(message.get("content") or "")
        for message in compacted_request
    )
    assert len(_budget_messages(llm.calls[1])) == 1
    assert "Current model step: 1 / 20" in _budget_messages(llm.calls[1])[0][
        "content"
    ]
    assert estimate_messages_size(compacted_request) <= 1_300
    assert all(
        EXECUTION_BUDGET_HEADER
        not in str(message.get("content") or "")
        for message in compacted_request
        if message not in _budget_messages(llm.calls[1])
    )
    assert EXECUTION_BUDGET_HEADER not in json.dumps(
        agent.history,
        ensure_ascii=False,
    )

    agent.reset_history()
    assert agent.run("after-reset-question") == "after-reset"
    assert len(_budget_messages(llm.calls[2])) == 1
    assert "Current model step: 1 / 20" in _budget_messages(llm.calls[2])[0][
        "content"
    ]
    assert EXECUTION_BUDGET_HEADER not in json.dumps(
        agent.history,
        ensure_ascii=False,
    )


def test_execution_budget_is_counted_inside_context_hard_limit() -> None:
    user_message = {"role": "user", "content": "hard-limit-user"}
    budget = _build_execution_budget_message(1, 20)
    exact_limit = estimate_messages_size([budget, user_message])

    accepted_llm = RecordingLLM([_text_response("accepted")])
    accepted_agent = Agent(
        accepted_llm,
        ToolRegistry(),
        max_context_chars=exact_limit,
    )
    assert accepted_agent.run(user_message["content"]) == "accepted"
    assert estimate_messages_size(accepted_llm.calls[0]["messages"]) == (
        exact_limit
    )

    rejected_llm = RecordingLLM([_text_response("must-not-run")])
    rejected_agent = Agent(
        rejected_llm,
        ToolRegistry(),
        max_context_chars=exact_limit - 1,
    )
    with pytest.raises(AgentContextLimitError):
        rejected_agent.run(user_message["content"])
    assert rejected_llm.calls == []


def test_execution_budget_can_trigger_compaction_before_request_overflow(
) -> None:
    tool = EchoTool()
    llm = RecordingLLM(
        [
            _tool_response("large-result", "x" * 220),
            _text_response("compacted-finish"),
        ]
    )
    agent = Agent(
        llm,
        _registry(tool),
        max_steps=2,
        max_context_chars=900,
        compaction_trigger_chars=850,
        max_compaction_chars=350,
    )

    assert agent.run("use the large tool result") == "compacted-finish"

    second_request = llm.calls[1]["messages"]
    assert any(
        CURRENT_RUN_HEADER in str(message.get("content") or "")
        for message in second_request
    )
    assert len(_budget_messages(llm.calls[1])) == 1
    assert estimate_messages_size(second_request) <= 900


def test_execution_budget_does_not_extend_max_step_termination() -> None:
    tool = EchoTool()

    class LoopingLLM(RecordingLLM):
        def __init__(self) -> None:
            super().__init__([])

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
            index = len(self.calls)
            return _tool_response(f"loop-{index}", f"value-{index}")

    llm = LoopingLLM()
    agent = Agent(llm, _registry(tool), max_steps=3)

    with pytest.raises(AgentMaxStepsError, match="3"):
        agent.run("loop")

    assert len(llm.calls) == 3
    assert tool.calls == ["value-1", "value-2", "value-3"]
    assert [
        _budget_messages(call)[0]["content"].splitlines()[1]
        for call in llm.calls
    ] == [
        "Current model step: 1 / 3",
        "Current model step: 2 / 3",
        "Current model step: 3 / 3",
    ]
