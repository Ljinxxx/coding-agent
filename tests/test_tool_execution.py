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

from src.agent import Agent
from src.tool_execution import ToolExecutionResult, ToolExecutor
from src.tools.base import BaseTool
from src.tools.files import WriteFileTool
from src.tools.registry import ToolRegistry
from src.tools.shell import RunCommandTool
from src.tools.verification import VerifyWorkspaceTool


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

    def __init__(self, execution_order: list[str] | None = None) -> None:
        self.execution_order = execution_order
        self.calls: list[str] = []

    def execute(self, **kwargs: Any) -> str:
        value = str(kwargs["value"])
        self.calls.append(value)
        if self.execution_order is not None:
            self.execution_order.append(value)
        return f"echo:{value}"


class FailingTool(BaseTool):
    name = "fail"
    description = "Raise a deterministic runtime error."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, execution_order: list[str] | None = None) -> None:
        self.execution_order = execution_order
        self.execute_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.execute_count += 1
        if self.execution_order is not None:
            self.execution_order.append("B")
        raise RuntimeError("intentional p1-4 failure")


class MutatingFailingTool(BaseTool):
    name = "mutating_fail"
    description = "Write a partial marker, then raise an error."
    mutates_workspace = True
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.execute_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.execute_count += 1
        (self.workspace / "partial.txt").write_text(
            "partial-side-effect",
            encoding="utf-8",
        )
        raise RuntimeError("mutating p1-4 failure")


class InterruptingTool(BaseTool):
    name = "interrupt"
    description = "Raise a process-control exception."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        raise KeyboardInterrupt


class RuntimeVerificationTool(BaseTool):
    name = "verify_workspace"
    description = "Raise a deterministic verification runtime error."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("verification runtime failure")


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


def _registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


def _tool_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "tool"]


def test_registered_tool_success_returns_unified_execution_result() -> None:
    tool = EchoTool()
    executor = ToolExecutor(_registry_with(tool))

    result = executor.execute("echo", {"value": "hello"})

    assert isinstance(result, ToolExecutionResult)
    assert result == ToolExecutionResult(
        tool_name="echo",
        content="echo:hello",
        execution_ok=True,
        error_type=None,
    )
    assert tool.calls == ["hello"]


def test_unknown_tool_becomes_structured_execution_error() -> None:
    result = ToolExecutor(ToolRegistry()).execute("ghost_tool", {})

    assert result.tool_name == "ghost_tool"
    assert result.execution_ok is False
    assert result.error_type == "UnknownTool"
    payload = json.loads(result.content)
    assert payload == {
        "ok": False,
        "tool": "ghost_tool",
        "error_type": "UnknownTool",
        "message": "Tool 'ghost_tool' is not registered.",
    }
    assert "Traceback" not in result.content
    assert str(Path.cwd()) not in result.content

    interrupting_executor = ToolExecutor(
        _registry_with(InterruptingTool())
    )
    with pytest.raises(KeyboardInterrupt):
        interrupting_executor.execute("interrupt", {})


def test_registered_tool_exception_becomes_structured_error_without_traceback(
) -> None:
    tool = FailingTool()

    result = ToolExecutor(_registry_with(tool)).execute("fail", {})

    assert tool.execute_count == 1
    assert result.tool_name == "fail"
    assert result.execution_ok is False
    assert result.error_type == "RuntimeError"
    assert json.loads(result.content) == {
        "ok": False,
        "tool": "fail",
        "error_type": "RuntimeError",
        "message": "intentional p1-4 failure",
    }
    assert "Traceback" not in result.content
    assert str(Path.cwd()) not in result.content


def test_agent_preserves_tool_call_id_and_emits_exactly_one_result_per_call(
) -> None:
    tool = EchoTool()
    llm = FakeLLM(
        [
            _tool_response(("call-123", "echo", {"value": "hello"})),
            _text_response("done"),
        ]
    )
    agent = Agent(llm, _registry_with(tool))

    assert agent.run("test") == "done"

    history = agent.history
    assistant_index = next(
        index
        for index, message in enumerate(history)
        if message.get("tool_calls")
    )
    tool_message = history[assistant_index + 1]
    assert history[assistant_index]["tool_calls"][0]["id"] == "call-123"
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call-123",
        "content": "echo:hello",
    }
    assert sum(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call-123"
        for message in history
    ) == 1
    assert llm.calls[1]["messages"][-1] == tool_message
    assert tool.calls == ["hello"]


def test_multiple_tool_calls_preserve_order_and_continue_after_error() -> None:
    execution_order: list[str] = []
    echo = EchoTool(execution_order)
    failing = FailingTool(execution_order)
    llm = FakeLLM(
        [
            _tool_response(
                ("call-A", "echo", {"value": "A"}),
                ("call-B", "fail", {}),
                ("call-C", "echo", {"value": "C"}),
            ),
            _text_response("done"),
        ]
    )
    agent = Agent(llm, _registry_with(echo, failing))

    assert agent.run("test") == "done"

    tool_messages = _tool_messages(agent.history)
    assert execution_order == ["A", "B", "C"]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-A",
        "call-B",
        "call-C",
    ]
    assert all(
        set(message) == {"role", "tool_call_id", "content"}
        for message in tool_messages
    )
    assert [message["tool_call_id"] for message in llm.calls[1]["messages"][-3:]] == [
        "call-A",
        "call-B",
        "call-C",
    ]
    assert tool_messages[0]["content"] == "echo:A"
    assert json.loads(tool_messages[1]["content"])["error_type"] == (
        "RuntimeError"
    )
    assert tool_messages[2]["content"] == "echo:C"
    assert all(
        sum(
            message.get("tool_call_id") == call_id
            for message in tool_messages
        )
        == 1
        for call_id in ("call-A", "call-B", "call-C")
    )


def test_domain_failures_remain_normal_tool_results(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tool = RunCommandTool(tmp_path)
    executor = ToolExecutor(_registry_with(tool))
    nonzero = executor.execute(
        "run_command",
        {
            "command": _make_command(
                sys.executable,
                "-B",
                "-c",
                "raise SystemExit(7)",
            )
        },
    )

    nonzero_payload = json.loads(nonzero.content)
    assert nonzero.execution_ok is True
    assert nonzero.error_type is None
    assert nonzero_payload["exit_code"] == 7
    assert nonzero_payload["timed_out"] is False
    assert "ok" not in nonzero_payload

    def raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(
            "timeout",
            1,
            output="partial-out",
            stderr="partial-err",
        )

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    timeout = executor.execute(
        "run_command",
        {"command": "timeout", "timeout": 1},
    )

    timeout_payload = json.loads(timeout.content)
    assert timeout.execution_ok is True
    assert timeout.error_type is None
    assert timeout_payload["exit_code"] is None
    assert timeout_payload["timed_out"] is True
    assert "ok" not in timeout_payload


def test_mutating_tool_exception_still_invalidates_verification(
    tmp_path: Path,
) -> None:
    unknown_llm = FakeLLM(
        [
            _tool_response(("unknown", "ghost_tool", {})),
            _text_response("unknown-observed"),
        ]
    )
    unknown_agent = Agent(unknown_llm, ToolRegistry())
    assert unknown_agent.run("unknown") == "unknown-observed"
    assert unknown_agent.workspace_revision == 0
    assert unknown_agent.verified_revision == 0
    assert unknown_agent.verification_required is False

    mutating_tool = MutatingFailingTool(tmp_path)
    mutating_llm = FakeLLM(
        [
            _tool_response(("mutating", "mutating_fail", {})),
            _text_response("failure-observed"),
        ]
    )
    mutating_agent = Agent(
        mutating_llm,
        _registry_with(mutating_tool),
    )
    mutating_llm.agent = mutating_agent

    assert mutating_agent.run("mutate") == "failure-observed"

    assert mutating_tool.execute_count == 1
    assert (tmp_path / "partial.txt").read_text(encoding="utf-8") == (
        "partial-side-effect"
    )
    assert mutating_llm.states == [(0, 0), (1, 0)]
    assert mutating_agent.workspace_revision == 1
    assert mutating_agent.verified_revision == 0
    assert mutating_agent.verification_required is True
    error_message = mutating_llm.calls[1]["messages"][-1]
    assert error_message["tool_call_id"] == "mutating"
    assert json.loads(error_message["content"])["error_type"] == (
        "RuntimeError"
    )


def test_unified_tool_execution_preserves_verification_gate_semantics(
    tmp_path: Path,
) -> None:
    runtime_outcome = ToolExecutor(
        _registry_with(RuntimeVerificationTool())
    ).execute("verify_workspace", {})
    assert runtime_outcome.execution_ok is False
    assert runtime_outcome.error_type == "RuntimeError"
    assert json.loads(runtime_outcome.content) == {
        "ok": False,
        "tool": "verify_workspace",
        "error_type": "RuntimeError",
        "message": "verification runtime failure",
    }

    failing_verifier = VerifyWorkspaceTool(
        tmp_path,
        [_make_command(sys.executable, "-B", "-c", "raise SystemExit(7)")],
    )
    failed_outcome = ToolExecutor(
        _registry_with(failing_verifier)
    ).execute("verify_workspace", {})
    failed_payload = json.loads(failed_outcome.content)
    assert failed_outcome.execution_ok is True
    assert failed_outcome.error_type is None
    assert failed_payload["ok"] is False
    assert failed_payload["checks"][0]["exit_code"] == 7

    passing_verifier = VerifyWorkspaceTool(
        tmp_path,
        [
            _make_command(
                sys.executable,
                "-B",
                "-c",
                "from pathlib import Path; "
                "assert Path('changed.txt').read_text(encoding='utf-8') "
                "== 'changed'",
            )
        ],
    )
    registry = _registry_with(WriteFileTool(tmp_path), passing_verifier)
    llm = FakeLLM(
        [
            _tool_response(
                (
                    "write",
                    "write_file",
                    {"path": "changed.txt", "content": "changed"},
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
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent

    assert agent.run("write and verify") == "verified-final"

    assert llm.states == [(0, 0), (1, 0), (1, 0), (1, 1)]
    feedback = llm.calls[2]["messages"][-1]
    assert feedback["role"] == "user"
    assert "[Verification Required]" in feedback["content"]
    verify_messages = [
        message
        for message in agent.history
        if message.get("role") == "tool"
        and message.get("tool_call_id") == "verify"
    ]
    assert len(verify_messages) == 1
    assert json.loads(verify_messages[0]["content"])["ok"] is True
    assert agent.workspace_revision == 1
    assert agent.verified_revision == 1
    assert agent.verification_required is False
