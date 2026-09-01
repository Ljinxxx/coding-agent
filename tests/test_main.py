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

import src.main as main_module
from src.agent import Agent, AgentMaxStepsError


BASE_TOOL_NAMES = [
    "list_directory",
    "read_file",
    "edit_file",
    "write_file",
    "run_command",
]


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
        if not self.responses:
            raise AssertionError("The formal Agent made an unexpected LLM call.")
        return self.responses.pop(0)


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def tool_response(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
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
        ],
    )


def make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


def tool_result(agent: Agent, call_id: str) -> dict[str, Any]:
    message = next(
        item
        for item in agent.history
        if item.get("role") == "tool" and item.get("tool_call_id") == call_id
    )
    return json.loads(message["content"])


def tool_content(agent: Agent, call_id: str) -> str:
    message = next(
        item
        for item in agent.history
        if item.get("role") == "tool" and item.get("tool_call_id") == call_id
    )
    return str(message["content"])


def read_payload(result: str) -> str:
    return result.split("\n\n", 1)[1]


def reject_real_client() -> None:
    raise AssertionError("The integration test must not create a real LLM client.")


def test_build_agent_preserves_default_and_accepts_max_steps_override(
    tmp_path: Path,
) -> None:
    default_agent = main_module.build_agent(
        tmp_path,
        llm_client=FakeLLM([]),
    )
    challenge_agent = main_module.build_agent(
        tmp_path,
        llm_client=FakeLLM([]),
        max_steps=40,
    )

    assert default_agent.max_steps == main_module.DEFAULT_MAX_STEPS == 20
    assert challenge_agent.max_steps == 40


def test_main_runs_read_only_agent_with_default_context_and_no_verification(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    token = "p0-formal-agent-local-content"
    (workspace / "hello.txt").write_text(token, encoding="utf-8")
    fake_llm = FakeLLM(
        [
            tool_response("read-call", "read_file", {"path": "hello.txt"}),
            text_response("formal-agent-complete"),
        ]
    )
    built_agents: list[Agent] = []
    original_build_agent = main_module.build_agent

    def record_build_agent(configured_workspace: Path, **options: Any) -> Agent:
        agent = original_build_agent(
            configured_workspace,
            llm_client=fake_llm,
            **options,
        )
        built_agents.append(agent)
        return agent

    monkeypatch.setattr(main_module, "build_agent", record_build_agent)
    monkeypatch.setattr(main_module, "LLMClient", reject_real_client)
    monkeypatch.setattr("builtins.input", lambda _prompt: "Read hello.txt.")

    exit_code = main_module.main(
        ["--workspace", str(workspace / "nested" / "..")]
    )

    assert exit_code == 0
    assert len(built_agents) == 1
    agent = built_agents[0]
    assert agent.max_context_chars == main_module.DEFAULT_MAX_CONTEXT_CHARS
    assert agent.compaction_trigger_chars is None
    assert agent.max_compaction_chars is None
    assert agent.tool_registry.names() == BASE_TOOL_NAMES
    assert agent.verification_tool_name is None
    assert "verify_workspace" not in str(agent.system_prompt)
    assert agent.tool_registry.get("read_file").workspace == workspace.resolve()
    assert agent.workspace_revision == agent.verified_revision == 0

    assert len(fake_llm.calls) == 2
    for call in fake_llm.calls:
        assert [schema["function"]["name"] for schema in call["tools"]] == (
            BASE_TOOL_NAMES
        )
    assert fake_llm.calls[1]["messages"][-2]["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": json.dumps({"path": "hello.txt"}),
    }
    read_message = fake_llm.calls[1]["messages"][-1]
    assert read_message["role"] == "tool"
    assert read_message["tool_call_id"] == "read-call"
    assert read_payload(read_message["content"]) == token

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "formal-agent-complete" in captured.out


def test_main_passes_custom_context_and_host_verification_commands(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    fake_llm = FakeLLM([text_response("configured-agent-complete")])
    built_agents: list[Agent] = []
    original_build_agent = main_module.build_agent

    def record_build_agent(configured_workspace: Path, **options: Any) -> Agent:
        agent = original_build_agent(
            configured_workspace,
            llm_client=fake_llm,
            **options,
        )
        built_agents.append(agent)
        return agent

    monkeypatch.setattr(main_module, "build_agent", record_build_agent)
    monkeypatch.setattr(main_module, "LLMClient", reject_real_client)
    monkeypatch.setattr("builtins.input", lambda _prompt: "Inspect the workspace.")

    exit_code = main_module.main(
        [
            "--workspace",
            str(tmp_path),
            "--max-context-chars",
            "12345",
            "--compact-context",
            "--verify",
            "host-check-one",
            "--verify",
            "host-check-two",
        ]
    )

    assert exit_code == 0
    agent = built_agents[0]
    assert agent.max_context_chars == 12345
    assert agent.compaction_trigger_chars == 12345 * 3 // 4
    assert agent.max_compaction_chars == 12345 // 4
    assert agent.tool_registry.names() == [*BASE_TOOL_NAMES, "verify_workspace"]
    assert agent.verification_tool_name == "verify_workspace"
    verifier = agent.tool_registry.get("verify_workspace")
    assert verifier.commands == ("host-check-one", "host-check-two")
    assert verifier.to_schema()["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert "verify_workspace" in str(agent.system_prompt)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("invalid_value", ["0", "-1"])
def test_cli_rejects_non_positive_context_budget(
    invalid_value: str,
    capsys: Any,
) -> None:
    with pytest.raises(SystemExit) as error:
        main_module._parse_args(["--max-context-chars", invalid_value])

    assert error.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_formal_agent_blocks_final_until_host_verification_succeeds(
    tmp_path: Path,
) -> None:
    calculator = tmp_path / "calculator.py"
    calculator.write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    command = make_command(
        sys.executable,
        "-B",
        "-c",
        "from calculator import add; assert add(2, 3) == 5",
    )
    fake_llm = FakeLLM(
        [
            tool_response("read", "read_file", {"path": "calculator.py"}),
            tool_response(
                "edit",
                "edit_file",
                {
                    "path": "calculator.py",
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                },
            ),
            text_response("premature-final"),
            tool_response("verify-pass", "verify_workspace", {}),
            text_response("verified-final"),
        ]
    )
    agent = main_module.build_agent(
        tmp_path,
        llm_client=fake_llm,
        verification_commands=[command],
    )

    result = agent.run("Change the file and verify it.")

    assert result == "verified-final"
    assert calculator.read_text(encoding="utf-8") == (
        "def add(a, b):\n    return a + b\n"
    )
    assert agent.workspace_revision == agent.verified_revision == 1
    assert read_payload(tool_content(agent, "read")) == (
        "def add(a, b):\n    return a - b\n"
    )
    assert "replaced exactly one occurrence" in tool_content(agent, "edit")
    assert tool_result(agent, "verify-pass")["ok"] is True
    assert len(fake_llm.calls) == 5
    feedback = fake_llm.calls[3]["messages"][-1]
    assert feedback["role"] == "user"
    assert "[Verification Required]" in feedback["content"]
    assert json.loads(fake_llm.calls[4]["messages"][-1]["content"])["ok"] is True


def test_formal_agent_keeps_final_blocked_after_verification_failure(
    tmp_path: Path,
) -> None:
    command = make_command(sys.executable, "-c", "raise SystemExit(7)")
    fake_llm = FakeLLM(
        [
            tool_response(
                "write",
                "write_file",
                {"path": "changed.txt", "content": "changed"},
            ),
            tool_response("verify-fail", "verify_workspace", {}),
            text_response("must-not-finish"),
        ]
    )
    agent = main_module.build_agent(
        tmp_path,
        llm_client=fake_llm,
        verification_commands=[command],
    )
    agent.max_steps = 3

    with pytest.raises(AgentMaxStepsError, match="3"):
        agent.run("Change the file but fail verification.")

    failed_result = tool_result(agent, "verify-fail")
    assert failed_result["ok"] is False
    assert failed_result["checks"][0]["exit_code"] == 7
    assert agent.workspace_revision == 1
    assert agent.verified_revision == 0
    assert agent.verification_required is True
    assert fake_llm.calls[2]["messages"][-1]["tool_call_id"] == "verify-fail"
    assert json.loads(fake_llm.calls[2]["messages"][-1]["content"])["ok"] is False
    assert "[Verification Required]" in agent.history[-1]["content"]


def test_formal_agent_does_not_run_hidden_pytest_when_verification_is_disabled(
    tmp_path: Path,
) -> None:
    target = tmp_path / "plain.txt"
    target.write_text("pytest required", encoding="utf-8")
    fake_llm = FakeLLM(
        [
            tool_response("read", "read_file", {"path": "plain.txt"}),
            tool_response(
                "edit",
                "edit_file",
                {
                    "path": "plain.txt",
                    "old_text": "pytest required",
                    "new_text": "no pytest required",
                },
            ),
            text_response("unverified-by-policy-final"),
        ]
    )
    agent = main_module.build_agent(tmp_path, llm_client=fake_llm)

    result = agent.run("Edit a file without configured verification.")

    assert result == "unverified-by-policy-final"
    assert target.read_text(encoding="utf-8") == "no pytest required"
    assert agent.tool_registry.names() == BASE_TOOL_NAMES
    assert agent.verification_tool_name is None
    assert agent.workspace_revision == 1
    assert agent.verified_revision == 0
    assert len(fake_llm.calls) == 3
    read_message = fake_llm.calls[1]["messages"][-1]
    assert read_message["role"] == "tool"
    assert read_message["tool_call_id"] == "read"
    assert read_payload(read_message["content"]) == "pytest required"
    assert "replaced exactly one occurrence" in (
        fake_llm.calls[2]["messages"][-1]["content"]
    )
    assert not any(
        "[Verification Required]" in str(message.get("content") or "")
        for message in agent.history
    )
