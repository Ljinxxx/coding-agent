import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import (
    Agent,
    AgentContextLimitError,
    _build_execution_budget_message,
)
from src.context_compaction import (
    CURRENT_RUN_HEADER,
    _build_compaction_messages,
    build_compacted_context,
    build_context_units,
)
from src.tools.base import BaseTool
from src.tools.files import WriteFileTool
from src.tools.registry import ToolRegistry
from src.tools.verification import VerifyWorkspaceTool


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
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
            for call_id, name, arguments in calls
        ],
    )


def _assistant_tool_message(
    *calls: tuple[str, str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                    ),
                },
            }
            for call_id, name, arguments in calls
        ],
    }


def _compaction_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if "[Compacted " in str(message.get("content") or "")
    ]


def _native_tool_ids(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    call_ids = [
        str(call["id"])
        for message in messages
        for call in message.get("tool_calls", [])
    ]
    result_ids = [
        str(message["tool_call_id"])
        for message in messages
        if message.get("role") == "tool"
    ]
    return call_ids, result_ids


def _make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.states: list[tuple[int, int]] = []
        self.agent: Agent | None = None

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
        if self.agent is not None:
            self.states.append(
                (
                    self.agent.workspace_revision,
                    self.agent.verified_revision,
                )
            )
        if not self.responses:
            raise AssertionError("Agent made an unexpected extra LLM call.")
        return self.responses.pop(0)


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the provided value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, **kwargs: Any) -> str:
        return str(kwargs["value"])


class FailingTool(BaseTool):
    name = "fail"
    description = "Always raise a deterministic error."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("deterministic tool failure")


def _registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_compaction_disabled_or_below_trigger_preserves_legacy_context(
) -> None:
    legacy_llm = FakeLLM(
        [
            _text_response("old-answer"),
            _text_response("blocker-" + "x" * 500),
            _text_response("recent-answer"),
            _text_response("current-answer"),
        ]
    )
    legacy_agent = Agent(
        legacy_llm,
        ToolRegistry(),
        system_prompt="system",
    )
    legacy_agent.run("old-question")
    legacy_agent.run("blocker-question")
    legacy_agent.run("recent-question")
    history = legacy_agent.history
    expected_context = [
        history[0],
        *history[-2:],
        {"role": "user", "content": "current-question"},
    ]
    legacy_agent.max_context_chars = (
        _context_size(expected_context) + _execution_budget_chars()
    )

    legacy_agent.run("current-question")

    assert _without_execution_budget(
        legacy_llm.calls[-1]["messages"]
    ) == expected_context
    assert _compaction_messages(legacy_llm.calls[-1]["messages"]) == []

    below_llm = FakeLLM([_text_response("short-answer")])
    below_agent = Agent(
        below_llm,
        ToolRegistry(),
        system_prompt="system",
        max_context_chars=5_000,
        compaction_trigger_chars=4_000,
        max_compaction_chars=1_000,
    )
    below_agent.run("short-question")
    assert _without_execution_budget(below_llm.calls[0]["messages"]) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "short-question"},
    ]
    assert _compaction_messages(below_llm.calls[0]["messages"]) == []

    invalid_options = [
        {
            "max_context_chars": 100,
            "compaction_trigger_chars": 0,
            "max_compaction_chars": 10,
        },
        {
            "max_context_chars": 100,
            "compaction_trigger_chars": 1,
            "max_compaction_chars": 0,
        },
        {
            "max_context_chars": None,
            "compaction_trigger_chars": 1,
            "max_compaction_chars": 1,
        },
        {
            "max_context_chars": 100,
            "compaction_trigger_chars": 101,
            "max_compaction_chars": 10,
        },
        {
            "max_context_chars": 100,
            "compaction_trigger_chars": 1,
            "max_compaction_chars": None,
        },
        {
            "max_context_chars": 100,
            "compaction_trigger_chars": None,
            "max_compaction_chars": 10,
        },
    ]
    for options in invalid_options:
        with pytest.raises(ValueError):
            Agent(FakeLLM([]), ToolRegistry(), **options)


def test_old_context_is_compacted_after_trigger() -> None:
    early_marker = "EARLY_CONSTRAINT=keep-tests-unchanged"
    old_noise = "OLD_RAW_NOISE_START" + "n" * 1_200 + "OLD_RAW_NOISE_END"
    old_user = f"{early_marker}\n{old_noise}"
    recent_user = "RECENT_MARKER=still-raw"
    current_user = "CURRENT_MARKER=raw-request"
    llm = FakeLLM(
        [
            _text_response("old-recorded"),
            _text_response("recent-recorded"),
            _text_response("compacted-answer"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="system",
        max_context_chars=1_800 + _execution_budget_chars(),
        compaction_trigger_chars=700,
        max_compaction_chars=600,
    )

    agent.run(old_user)
    agent.run(recent_user)
    history_before_current = agent.history
    agent.run(current_user)

    final_context = llm.calls[-1]["messages"]
    summaries = _compaction_messages(final_context)
    summary_text = "\n".join(str(item["content"]) for item in summaries)
    assert summaries
    assert "[Compacted Prior Context - Harness Generated]" in summary_text
    assert early_marker in summary_text
    assert old_noise not in summary_text
    assert {"role": "user", "content": old_user} not in final_context
    assert {"role": "user", "content": recent_user} in final_context
    assert {"role": "user", "content": current_user} in final_context
    assert all(set(message) == {"role", "content"} for message in summaries)
    assert _context_size(final_context) <= agent.max_context_chars
    assert agent.history[: len(history_before_current)] == history_before_current
    assert {"role": "user", "content": old_user} in agent.history
    assert _compaction_messages(agent.history) == []


def test_current_user_request_is_never_compacted() -> None:
    current_user = "CURRENT_REQUEST_MARKER=exact\n" + "c" * 800
    llm = FakeLLM(
        [
            _text_response("old-answer"),
            _text_response("current-answer"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="system",
        max_context_chars=2_000 + _execution_budget_chars(),
        compaction_trigger_chars=500,
        max_compaction_chars=500,
    )
    agent.run("OLD_MARKER\n" + "o" * 1_000)
    agent.run(current_user)

    final_context = llm.calls[-1]["messages"]
    current_message = {"role": "user", "content": current_user}
    assert final_context.count(current_message) == 1
    assert current_message in final_context

    mandatory = [
        {"role": "system", "content": "required-system"},
        current_message,
    ]
    budget = _context_size(mandatory) + _execution_budget_chars() - 1
    overflow_llm = FakeLLM([_text_response("must-not-run")])
    overflow_agent = Agent(
        overflow_llm,
        ToolRegistry(),
        system_prompt="required-system",
        max_context_chars=budget,
        compaction_trigger_chars=1,
        max_compaction_chars=100,
    )
    with pytest.raises(AgentContextLimitError):
        overflow_agent.run(current_user)
    assert overflow_llm.calls == []
    assert overflow_agent.history == mandatory


def test_current_run_old_progress_compacts_but_recent_units_stay_raw(
) -> None:
    values = [
        f"TOOL_PROGRESS_{index}=" + chr(96 + index) * 340
        for index in range(1, 5)
    ]
    llm = FakeLLM(
        [
            *[
                _tool_response(
                    (f"progress-{index}", "echo", {"value": value})
                )
                for index, value in enumerate(values, start=1)
            ],
            _text_response("current-run-complete"),
        ]
    )
    registry = _registry_with(EchoTool())
    agent = Agent(
        llm,
        registry,
        system_prompt="system",
        max_steps=5,
        max_context_chars=(
            2_200
            + _execution_budget_chars(current_step=5, max_steps=5)
        ),
        compaction_trigger_chars=500,
        max_compaction_chars=600,
    )
    current_user = "CURRENT_MARKER=long-running-task"

    assert agent.run(current_user) == "current-run-complete"

    final_context = llm.calls[-1]["messages"]
    summaries = _compaction_messages(final_context)
    summary_text = "\n".join(str(item["content"]) for item in summaries)
    call_ids, result_ids = _native_tool_ids(final_context)
    assert {"role": "user", "content": current_user} in final_context
    assert "[Compacted Current-Run Progress - Harness Generated]" in summary_text
    assert "TOOL_PROGRESS_1=" in summary_text
    assert "progress-1" not in call_ids
    assert "progress-1" not in result_ids
    assert "progress-4" in call_ids
    assert "progress-4" in result_ids
    assert call_ids == result_ids
    history_call_ids, history_result_ids = _native_tool_ids(agent.history)
    assert history_call_ids == [
        "progress-1",
        "progress-2",
        "progress-3",
        "progress-4",
    ]
    assert history_result_ids == history_call_ids
    assert _compaction_messages(agent.history) == []


def test_tool_call_and_results_are_never_split() -> None:
    raw_exchange = [
        _assistant_tool_message(
            ("multi-a", "echo", {"value": "MULTI_A"}),
            ("multi-b", "fail", {}),
        ),
        {
            "role": "tool",
            "tool_call_id": "multi-a",
            "content": "MULTI_A",
        },
        {
            "role": "tool",
            "tool_call_id": "multi-b",
            "content": json.dumps(
                {
                    "ok": False,
                    "tool": "fail",
                    "error_type": "RuntimeError",
                    "message": "deterministic tool failure",
                }
            ),
        },
    ]
    units = build_context_units(raw_exchange)
    assert len(units) == 1
    assert units[0].kind == "tool_error"
    assert list(units[0].messages) == raw_exchange

    multi_value = "MULTI_A=" + "a" * 320
    latest_value = "LATEST_RAW=" + "z" * 320
    llm = FakeLLM(
        [
            _tool_response(
                ("multi-a", "echo", {"value": multi_value}),
                ("multi-b", "fail", {}),
            ),
            _tool_response(
                ("latest", "echo", {"value": latest_value})
            ),
            _text_response("atomic-complete"),
        ]
    )
    agent = Agent(
        llm,
        _registry_with(EchoTool(), FailingTool()),
        max_steps=3,
        max_context_chars=(
            1_600
            + _execution_budget_chars(current_step=3, max_steps=3)
        ),
        compaction_trigger_chars=400,
        max_compaction_chars=600,
    )
    agent.run("Keep tool protocol atomic.")

    final_context = llm.calls[-1]["messages"]
    summary_text = "\n".join(
        str(item["content"]) for item in _compaction_messages(final_context)
    )
    call_ids, result_ids = _native_tool_ids(final_context)
    assert "multi-a" not in call_ids + result_ids
    assert "multi-b" not in call_ids + result_ids
    assert "MULTI_A=" in summary_text
    assert "RuntimeError" in summary_text
    assert call_ids == result_ids == ["latest"]


def test_compacted_digest_is_bounded_and_deterministic() -> None:
    early_marker = "EARLY_DETERMINISTIC=retained"
    anchor_index = 3
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": early_marker + "\n" + "x" * 2_000,
        },
        {"role": "assistant", "content": "acknowledged"},
        {"role": "user", "content": "CURRENT_DETERMINISTIC=raw"},
        _assistant_tool_message(
            ("old-progress", "echo", {"value": "OLD_PROGRESS=" + "o" * 500})
        ),
        {
            "role": "tool",
            "tool_call_id": "old-progress",
            "content": "OLD_PROGRESS=" + "o" * 500,
        },
        _assistant_tool_message(
            ("new-progress", "echo", {"value": "NEW_PROGRESS=" + "n" * 300})
        ),
        {
            "role": "tool",
            "tool_call_id": "new-progress",
            "content": "NEW_PROGRESS=" + "n" * 300,
        },
    ]

    first = build_compacted_context(
        messages,
        current_user_index=anchor_index,
        max_context_chars=1_500,
        max_compaction_chars=500,
    )
    second = build_compacted_context(
        messages,
        current_user_index=anchor_index,
        max_context_chars=1_500,
        max_compaction_chars=500,
    )

    summaries = _compaction_messages(first)
    summary_text = "\n".join(str(item["content"]) for item in summaries)
    assert first == second
    assert len(summaries) == 2
    assert sum(len(str(item["content"])) for item in summaries) <= 500
    assert _context_size(first) <= 1_500
    assert first[0] == {"role": "system", "content": "system"}
    assert sum(message.get("role") == "system" for message in first) == 1
    assert early_marker in summary_text
    assert "x" * 2_000 not in summary_text
    assert "timestamp" not in summary_text.casefold()
    assert "random" not in summary_text.casefold()
    call_ids, result_ids = _native_tool_ids(first)
    assert call_ids == result_ids == ["new-progress"]
    assert "old-progress" not in call_ids + result_ids
    assert {"role": "assistant", "content": "acknowledged"} not in first

    small_max_context_chars = 1_000
    small_max_compaction_chars = 250
    units = build_context_units(messages[1:], start_index=1)
    prior_units = [
        unit for unit in units if unit.end_index < anchor_index
    ]
    progress_units = [
        unit for unit in units if unit.start_index > anchor_index
    ]
    blocks = _build_compaction_messages(
        prior_units,
        progress_units,
        small_max_compaction_chars,
    )

    assert [block.kind for block in blocks] == ["prior", "current_run"]
    assert all(set(block.message) == {"role", "content"} for block in blocks)
    assert not str(blocks[1].message["content"]).startswith(
        CURRENT_RUN_HEADER
    )

    small_first = build_compacted_context(
        messages,
        current_user_index=anchor_index,
        max_context_chars=small_max_context_chars,
        max_compaction_chars=small_max_compaction_chars,
    )
    small_second = build_compacted_context(
        messages,
        current_user_index=anchor_index,
        max_context_chars=small_max_context_chars,
        max_compaction_chars=small_max_compaction_chars,
    )
    small_summaries = _compaction_messages(small_first)

    assert small_first == small_second
    assert small_first == [
        {"role": "system", "content": "system"},
        blocks[0].message,
        messages[anchor_index],
        blocks[1].message,
    ]
    assert len(small_summaries) == 2
    assert (
        sum(len(str(item["content"])) for item in small_summaries)
        <= small_max_compaction_chars
    )
    assert _context_size(small_first) <= small_max_context_chars
    assert all(
        set(message) == {"role", "content"}
        for message in small_summaries
    )


def test_compaction_never_mutates_full_history_and_reset_still_works(
) -> None:
    llm = FakeLLM(
        [
            _text_response("old-answer"),
            _text_response("current-answer"),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(),
        system_prompt="system",
        max_context_chars=1_600 + _execution_budget_chars(),
        compaction_trigger_chars=400,
        max_compaction_chars=500,
    )
    agent.run("FULL_HISTORY_MARKER\n" + "h" * 1_000)
    current_user_index = len(agent.history)
    agent.run("current-question")
    snapshot = agent.history

    first_view = agent._build_context_messages(
        current_user_index=current_user_index
    )
    second_view = agent._build_context_messages(
        current_user_index=current_user_index
    )
    assert first_view == second_view
    assert _compaction_messages(first_view)
    assert _compaction_messages(snapshot) == []
    first_view[0]["content"] = "changed-view"
    assert agent.history == snapshot

    history_copy = agent.history
    history_copy.clear()
    assert agent.history == snapshot
    agent.reset_history()
    assert agent.history == [{"role": "system", "content": "system"}]
    assert agent.compaction_trigger_chars == 400
    assert agent.max_compaction_chars == 500


def test_compaction_preserves_verification_gate_semantics(
    tmp_path: Path,
) -> None:
    command = _make_command(
        sys.executable,
        "-B",
        "-c",
        'print("verification passed")',
    )
    verifier = VerifyWorkspaceTool(tmp_path, [command])
    registry = _registry_with(WriteFileTool(tmp_path), verifier)
    llm = FakeLLM(
        [
            _tool_response(
                (
                    "write",
                    "write_file",
                    {"path": "changed.txt", "content": "w" * 1_000},
                )
            ),
            _text_response("premature-final"),
            _tool_response(("verify", "verify_workspace", {})),
            _text_response("verified-final"),
        ]
    )
    agent = Agent(
        llm,
        registry,
        max_steps=4,
        max_context_chars=(
            2_000
            + _execution_budget_chars(current_step=4, max_steps=4)
        ),
        compaction_trigger_chars=400,
        max_compaction_chars=600,
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent
    current_user = "CURRENT_REAL_USER=write-and-verify"

    assert agent.run(current_user) == "verified-final"

    assert len(llm.calls) == 4
    assert llm.states == [(0, 0), (1, 0), (1, 0), (1, 1)]
    assert agent.workspace_revision == agent.verified_revision == 1
    assert agent.verification_required is False
    feedback_context = llm.calls[2]["messages"]
    assert {"role": "user", "content": current_user} in feedback_context
    feedback = next(
        message
        for message in feedback_context
        if "[Verification Required]" in str(message.get("content") or "")
    )
    assert feedback["role"] == "user"
    final_context = llm.calls[3]["messages"]
    call_ids, result_ids = _native_tool_ids(final_context)
    assert "verify" in call_ids
    assert "verify" in result_ids
    verify_result = next(
        message
        for message in final_context
        if message.get("tool_call_id") == "verify"
    )
    assert json.loads(verify_result["content"])["ok"] is True
    assert any(
        _compaction_messages(call["messages"])
        for call in llm.calls
    )
    assert all(
        _context_size(call["messages"]) <= agent.max_context_chars
        for call in llm.calls
    )
    assert _compaction_messages(agent.history) == []
    assert {"role": "user", "content": current_user} in agent.history
