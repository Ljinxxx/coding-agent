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

from src.agent import Agent, AgentMaxStepsError
from src.tools.base import BaseTool
from src.tools.files import ListDirectoryTool, ReadFileTool, WriteFileTool
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
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
            for call_id, name, arguments in calls
        ],
    )


def make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


def registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def result_for_call(
    history: list[dict[str, Any]],
    call_id: str,
) -> dict[str, Any]:
    message = next(
        item
        for item in history
        if item.get("role") == "tool"
        and item.get("tool_call_id") == call_id
    )
    return json.loads(message["content"])


def tool_names(history: list[dict[str, Any]]) -> list[str]:
    return [
        call["function"]["name"]
        for message in history
        for call in message.get("tool_calls", [])
    ]


def test_verify_workspace_returns_success_when_all_checks_pass(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    first_script = workspace / "verify_ok_1.py"
    second_script = workspace / "verify_ok_2.py"
    first_script.write_text(
        "from pathlib import Path\n"
        'print("check-one")\n'
        "print(Path.cwd())\n",
        encoding="utf-8",
    )
    second_script.write_text('print("check-two")\n', encoding="utf-8")
    commands = [
        make_command(sys.executable, first_script.name),
        make_command(sys.executable, second_script.name),
    ]
    tool = VerifyWorkspaceTool(
        workspace / "nested" / "..",
        commands,
    )

    payload = json.loads(tool.execute())

    assert tool.name == "verify_workspace"
    assert tool.mutates_workspace is False
    assert tool.workspace == workspace.resolve()
    assert tool.commands == tuple(commands)
    assert tool.to_schema()["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert payload["ok"] is True
    assert len(payload["checks"]) == 2
    assert all(
        set(check)
        == {"command", "exit_code", "stdout", "stderr", "timed_out"}
        for check in payload["checks"]
    )
    assert [check["command"] for check in payload["checks"]] == commands
    assert all(check["exit_code"] == 0 for check in payload["checks"])
    assert all(check["stderr"] == "" for check in payload["checks"])
    assert all(check["timed_out"] is False for check in payload["checks"])
    first_output = payload["checks"][0]["stdout"].splitlines()
    assert first_output[0] == "check-one"
    assert Path(first_output[1]).resolve() == workspace.resolve()
    assert payload["checks"][1]["stdout"].strip() == "check-two"

    with pytest.raises(ValueError, match="commands"):
        VerifyWorkspaceTool(workspace, [])
    with pytest.raises(ValueError, match="commands"):
        VerifyWorkspaceTool(workspace, ["   "])
    with pytest.raises(ValueError, match="commands"):
        VerifyWorkspaceTool(workspace, "echo bypass")
    with pytest.raises(ValueError, match="commands"):
        VerifyWorkspaceTool(workspace, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout"):
        VerifyWorkspaceTool(workspace, commands, timeout=0)
    with pytest.raises(TypeError, match="does not accept arguments"):
        tool.execute(command="echo bypass")


def test_verify_workspace_returns_failure_details_on_nonzero_exit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pass.py").write_text(
        'print("first check passed")\n',
        encoding="utf-8",
    )
    (workspace / "fail.py").write_text(
        'print("verification failed")\nraise SystemExit(3)\n',
        encoding="utf-8",
    )
    marker = workspace / "should_not_run.txt"
    (workspace / "should_not_run.py").write_text(
        "from pathlib import Path\n"
        "Path('should_not_run.txt').write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )
    commands = [
        make_command(sys.executable, "pass.py"),
        make_command(sys.executable, "fail.py"),
        make_command(sys.executable, "should_not_run.py"),
    ]

    payload = json.loads(VerifyWorkspaceTool(workspace, commands).execute())

    assert payload["ok"] is False
    assert len(payload["checks"]) == 2
    assert payload["checks"][0]["exit_code"] == 0
    failed_check = payload["checks"][1]
    assert failed_check == {
        "command": commands[1],
        "exit_code": 3,
        "stdout": "verification failed\n",
        "stderr": "",
        "timed_out": False,
    }
    assert not marker.exists()

    timeout_command = make_command(
        sys.executable,
        "-c",
        "import time; time.sleep(2)",
    )
    timeout_payload = json.loads(
        VerifyWorkspaceTool(
            workspace,
            [timeout_command],
            timeout=1,
        ).execute()
    )
    assert timeout_payload["ok"] is False
    assert timeout_payload["checks"][0]["exit_code"] is None
    assert timeout_payload["checks"][0]["timed_out"] is True


def test_read_only_task_can_finish_without_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("read-only", encoding="utf-8")
    verifier = VerifyWorkspaceTool(
        workspace,
        [make_command(sys.executable, "-c", 'print("unused")')],
    )
    registry = registry_with(
        ListDirectoryTool(workspace),
        ReadFileTool(workspace),
        verifier,
    )
    llm = FakeLLM(
        [
            tool_response(
                ("list", "list_directory", {"path": "."}),
                ("read", "read_file", {"path": "safe.txt"}),
            ),
            text_response("read-only-complete"),
        ]
    )
    agent = Agent(
        llm,
        registry,
        max_steps=2,
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent

    result = agent.run("Inspect the workspace without changing it.")

    assert result == "read-only-complete"
    assert len(llm.calls) == 2
    assert llm.states == [(0, 0), (0, 0)]
    assert tool_names(agent.history) == ["list_directory", "read_file"]
    assert agent.workspace_revision == 0
    assert agent.verified_revision == 0
    assert agent.verification_required is False
    assert ListDirectoryTool(workspace).mutates_workspace is False
    assert ReadFileTool(workspace).mutates_workspace is False

    with pytest.raises(ValueError, match="not registered"):
        Agent(
            FakeLLM([]),
            ToolRegistry(),
            verification_tool_name="verify_workspace",
        )


def test_final_is_blocked_after_workspace_mutation_until_verification(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    verifier = VerifyWorkspaceTool(
        workspace,
        [make_command(sys.executable, "-c", 'print("verified")')],
    )
    registry = registry_with(WriteFileTool(workspace), verifier)
    llm = FakeLLM(
        [
            tool_response(
                ("write", "write_file", {"path": "changed.txt", "content": "x"})
            ),
            text_response("premature-complete"),
            tool_response(("verify", "verify_workspace", {})),
            text_response("verified-complete"),
        ]
    )
    agent = Agent(
        llm,
        registry,
        max_steps=4,
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent

    result = agent.run("Change the workspace and finish.")

    assert result == "verified-complete"
    assert len(llm.calls) == 4
    assert llm.states == [(0, 0), (1, 0), (1, 0), (1, 1)]
    feedback = llm.calls[2]["messages"][-1]
    assert feedback["role"] == "user"
    assert "[Verification Required]" in feedback["content"]
    assert "verify_workspace" in feedback["content"]
    assert "harness-generated" in feedback["content"]
    assert llm.calls[2]["messages"][-2] == {
        "role": "assistant",
        "content": "premature-complete",
    }
    assert feedback in agent.history
    assert result_for_call(agent.history, "verify")["ok"] is True
    assert agent.verification_required is False

    refusing_llm = FakeLLM(
        [
            tool_response(
                (
                    "write-again",
                    "write_file",
                    {"path": "still-dirty.txt", "content": "dirty"},
                )
            ),
            text_response("still-premature"),
        ]
    )
    refusing_agent = Agent(
        refusing_llm,
        registry,
        max_steps=2,
        verification_tool_name="verify_workspace",
    )
    refusing_llm.agent = refusing_agent
    with pytest.raises(AgentMaxStepsError, match="2"):
        refusing_agent.run("Refuse verification.")
    assert len(refusing_llm.calls) == 2
    assert refusing_agent.verification_required is True


@pytest.mark.parametrize(
    "explicitly_require_verification",
    [False, True],
    ids=["default", "explicit-true"],
)
def test_later_run_verifies_state_left_dirty_by_intermediate_run(
    tmp_path: Path,
    explicitly_require_verification: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    verifier = VerifyWorkspaceTool(
        workspace,
        [make_command(sys.executable, "-c", 'print("verified")')],
    )
    registry = registry_with(WriteFileTool(workspace), verifier)
    llm = FakeLLM(
        [
            tool_response(
                (
                    "draft-write",
                    "write_file",
                    {"path": "draft.txt", "content": "draft"},
                )
            ),
            text_response("intermediate-complete"),
            text_response("premature-release-final"),
            tool_response(("release-verify", "verify_workspace", {})),
            text_response("verified-release-final"),
        ]
    )
    agent = Agent(
        llm,
        registry,
        max_steps=3,
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent

    with pytest.raises(TypeError, match="positional"):
        agent.run("Positional policy is not supported.", False)  # type: ignore[misc]

    assert (
        agent.run(
            "Prepare an intermediate draft.",
            require_verified_completion=False,
        )
        == "intermediate-complete"
    )
    assert agent.workspace_revision == 1
    assert agent.verified_revision == 0
    assert agent.verification_required is True
    assert not any(
        str(message.get("content") or "").startswith(
            "[Verification Required]"
        )
        for message in agent.history
    )

    if explicitly_require_verification:
        result = agent.run(
            "Finish the release.",
            require_verified_completion=True,
        )
    else:
        result = agent.run("Finish the release.")

    assert result == "verified-release-final"
    assert llm.states == [(0, 0), (1, 0), (1, 0), (1, 0), (1, 1)]
    assert llm.calls[3]["messages"][-1]["role"] == "user"
    assert "[Verification Required]" in llm.calls[3]["messages"][-1][
        "content"
    ]
    assert result_for_call(agent.history, "release-verify")["ok"] is True
    assert agent.workspace_revision == agent.verified_revision == 1
    assert agent.verification_required is False
    assert agent.history[-1] == {
        "role": "assistant",
        "content": "verified-release-final",
    }


def test_failed_verification_keeps_workspace_dirty_and_agent_continues(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "check_calculator.py").write_text(
        "from pathlib import Path\n"
        "namespace = {}\n"
        "source = Path('calculator.py').read_text(encoding='utf-8')\n"
        "exec(compile(source, 'calculator.py', 'exec'), namespace)\n"
        "if namespace['add'](2, 3) != 5:\n"
        '    print("calculator verification failed")\n'
        "    raise SystemExit(3)\n"
        'print("calculator verification passed")\n',
        encoding="utf-8",
    )
    verifier = VerifyWorkspaceTool(
        workspace,
        [make_command(sys.executable, "check_calculator.py")],
    )
    registry = registry_with(WriteFileTool(workspace), verifier)
    llm = FakeLLM(
        [
            tool_response(
                (
                    "write-wrong",
                    "write_file",
                    {"path": "calculator.py", "content": "def add(a, b):\n    return a - b\n"},
                )
            ),
            tool_response(("verify-fail", "verify_workspace", {})),
            tool_response(
                (
                    "write-fix",
                    "write_file",
                    {"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n"},
                )
            ),
            tool_response(("verify-pass", "verify_workspace", {})),
            text_response("repair-complete"),
        ]
    )
    agent = Agent(
        llm,
        registry,
        max_steps=5,
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent

    result = agent.run("Repair calculator.py.")

    assert result == "repair-complete"
    assert llm.states == [(0, 0), (1, 0), (1, 0), (2, 0), (2, 2)]
    failed_payload = result_for_call(agent.history, "verify-fail")
    assert failed_payload["ok"] is False
    assert failed_payload["checks"][0]["exit_code"] == 3
    assert failed_payload["checks"][0]["timed_out"] is False
    assert "calculator verification failed" in failed_payload["checks"][0]["stdout"]
    assert llm.calls[2]["messages"][-1]["tool_call_id"] == "verify-fail"
    assert json.loads(llm.calls[2]["messages"][-1]["content"])["ok"] is False
    assert result_for_call(agent.history, "verify-pass")["ok"] is True
    assert agent.workspace_revision == 2
    assert agent.verified_revision == 2
    assert agent.verification_required is False


def test_successful_verification_allows_final_answer(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    verifier = VerifyWorkspaceTool(
        workspace,
        [make_command(sys.executable, "-c", 'print("pass")')],
    )
    registry = registry_with(WriteFileTool(workspace), verifier)
    llm = FakeLLM(
        [
            tool_response(
                ("write", "write_file", {"path": "done.txt", "content": "done"})
            ),
            tool_response(("verify", "verify_workspace", {})),
            text_response("successfully-verified"),
        ]
    )
    agent = Agent(
        llm,
        registry,
        max_steps=3,
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent

    assert agent.run("Write and verify.") == "successfully-verified"
    assert llm.states == [(0, 0), (1, 0), (1, 1)]
    assert agent.workspace_revision == agent.verified_revision == 1
    assert agent.verification_required is False
    assert not any(
        "[Verification Required]" in str(message.get("content") or "")
        for message in agent.history
    )

    final_step_llm = FakeLLM(
        [
            tool_response(
                (
                    "last-write",
                    "write_file",
                    {"path": "last-step.txt", "content": "value"},
                )
            ),
            tool_response(("last-verify", "verify_workspace", {})),
            text_response("next-run-final"),
        ]
    )
    final_step_agent = Agent(
        final_step_llm,
        registry,
        max_steps=2,
        verification_tool_name="verify_workspace",
    )
    final_step_llm.agent = final_step_agent
    with pytest.raises(AgentMaxStepsError, match="2"):
        final_step_agent.run("Verify on the final step.")
    assert len(final_step_llm.calls) == 2
    assert final_step_agent.verification_required is False
    assert final_step_agent.run("Now finish.") == "next-run-final"
    assert len(final_step_llm.calls) == 3


def test_mutation_after_successful_verification_invalidates_previous_verification(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    verification_command = make_command(
        sys.executable,
        "-c",
        'print("host verification passed")',
    )
    mutation_command = make_command(
        sys.executable,
        "-c",
        "from pathlib import Path; "
        "Path('after_verify.txt').write_text('changed', encoding='utf-8')",
    )
    failing_mutation_command = make_command(
        sys.executable,
        "-c",
        "from pathlib import Path; "
        "Path('partial.txt').write_text('partial', encoding='utf-8'); "
        "raise SystemExit(3)",
    )
    verifier = VerifyWorkspaceTool(workspace, [verification_command])
    runner = RunCommandTool(workspace)
    registry = registry_with(WriteFileTool(workspace), runner, verifier)
    llm = FakeLLM(
        [
            tool_response(
                ("write", "write_file", {"path": "initial.txt", "content": "one"})
            ),
            tool_response(("verify-one", "verify_workspace", {})),
            tool_response(
                ("ordinary-shell", "run_command", {"command": mutation_command})
            ),
            text_response("premature-after-shell"),
            tool_response(("verify-two", "verify_workspace", {})),
            text_response("second-verification-complete"),
            tool_response(
                (
                    "failing-shell",
                    "run_command",
                    {"command": failing_mutation_command},
                )
            ),
        ]
    )
    agent = Agent(
        llm,
        registry,
        max_steps=6,
        verification_tool_name="verify_workspace",
    )
    llm.agent = agent

    result = agent.run("Verify, mutate through the shell, and verify again.")

    assert result == "second-verification-complete"
    assert runner.mutates_workspace is True
    assert WriteFileTool(workspace).mutates_workspace is True
    assert llm.states == [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 1),
        (2, 1),
        (2, 2),
    ]
    assert (workspace / "after_verify.txt").read_text(encoding="utf-8") == "changed"
    ordinary_result = result_for_call(agent.history, "ordinary-shell")
    assert ordinary_result["exit_code"] == 0
    assert ordinary_result["timed_out"] is False
    assert "[Verification Required]" in llm.calls[4]["messages"][-1]["content"]
    assert result_for_call(agent.history, "verify-one")["ok"] is True
    assert result_for_call(agent.history, "verify-two")["ok"] is True
    assert agent.workspace_revision == agent.verified_revision == 2

    agent.max_steps = 1
    with pytest.raises(AgentMaxStepsError, match="1"):
        agent.run("Run a mutation-capable command that exits nonzero.")
    assert (workspace / "partial.txt").read_text(encoding="utf-8") == "partial"
    assert agent.workspace_revision == 3
    assert agent.verified_revision == 2
    assert agent.verification_required is True
    agent.reset_history()
    assert agent.history == []
    assert agent.workspace_revision == 3
    assert agent.verified_revision == 2
    assert agent.verification_required is True
