import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import Agent, AgentMaxStepsError
from src.context_compaction import (
    CURRENT_RUN_HEADER,
    build_compacted_context,
    estimate_messages_size,
)
from src.progress_guard import ProgressGuardConfig
from src.tools.base import BaseTool
from src.tools.files import ListDirectoryTool, ReadFileTool, WriteFileTool
from src.tools.registry import ToolRegistry


TRACKED_COMMAND = "python -B -m pytest -q --basetemp=.pytest_tmp"
OTHER_COMMAND = "python -B diagnose.py"
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
        if not self.responses:
            raise AssertionError("Agent made an unexpected extra LLM call.")
        return self.responses.pop(0)


@dataclass(frozen=True)
class CommandOutcome:
    exit_code: Any
    timed_out: bool = False


class ScriptedCommandTool(BaseTool):
    name = "run_command"
    description = "Return scripted command results without a subprocess."
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    mutates_workspace = True

    def __init__(self, outcomes: list[CommandOutcome | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    def execute(self, **kwargs: Any) -> str:
        command = str(kwargs["command"])
        self.calls.append(command)
        if not self.outcomes:
            raise AssertionError("Unexpected run_command execution.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return json.dumps(
            {
                "exit_code": outcome.exit_code,
                "stdout": "scripted command output",
                "stderr": "",
                "timed_out": outcome.timed_out,
            }
        )


class RecordingTool(BaseTool):
    description = "Record deterministic in-memory tool calls."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "additionalProperties": True,
    }

    def __init__(
        self,
        name: str,
        *,
        mutates_workspace: bool = False,
        outcomes: list[str | Exception] | None = None,
        default_result: str | None = None,
    ) -> None:
        self.name = name
        self.mutates_workspace = mutates_workspace
        self.outcomes = list(outcomes or [])
        self.default_result = default_result or f"{name}-result"
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> str:
        self.calls.append(deepcopy(kwargs))
        outcome: str | Exception
        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = self.default_result
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


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


def registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def guard_config(
    *,
    diagnosis_responses: int = 1,
    tracked_commands: tuple[str, ...] = (TRACKED_COMMAND,),
    mutation_tool_names: tuple[str, ...] = ("edit_file", "write_file"),
    initial_pending_command: str | None = None,
    initial_diagnosis_responses: int = 0,
) -> ProgressGuardConfig:
    return ProgressGuardConfig(
        tracked_commands=tracked_commands,
        mutation_tool_names=mutation_tool_names,
        diagnosis_responses=diagnosis_responses,
        initial_pending_command=initial_pending_command,
        initial_diagnosis_responses=initial_diagnosis_responses,
    )


def result_message(
    history: list[dict[str, Any]],
    call_id: str,
) -> dict[str, Any]:
    return next(
        message
        for message in history
        if message.get("role") == "tool"
        and message.get("tool_call_id") == call_id
    )


def result_payload(
    history: list[dict[str, Any]],
    call_id: str,
) -> dict[str, Any]:
    return json.loads(result_message(history, call_id)["content"])


def tool_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in history if message.get("role") == "tool"]


def budget_messages(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in call["messages"]
        if str(message.get("content") or "").startswith(
            EXECUTION_BUDGET_HEADER
        )
    ]


def test_progress_guard_is_opt_in_and_run_argument_is_keyword_only() -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("baseline", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("read-one", "read_file", {"path": "one.py"})),
            tool_response(("read-two", "read_file", {"path": "two.py"})),
            text_response("baseline-complete"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=4)

    with pytest.raises(TypeError):
        agent.run("invalid positional policy", guard_config())  # type: ignore[misc]

    assert agent.run("Run an unguarded baseline.") == "baseline-complete"
    assert command.calls == [TRACKED_COMMAND]
    assert read.calls == [{"path": "one.py"}, {"path": "two.py"}]
    assert all(
        "ProgressGuardBlocked" not in str(message.get("content") or "")
        for message in agent.history
    )


def test_progress_guard_config_rejects_simple_invalid_values() -> None:
    with pytest.raises(ValueError, match="tracked_commands"):
        guard_config(tracked_commands=())
    with pytest.raises(ValueError, match="tracked_commands"):
        guard_config(tracked_commands=("   ",))
    with pytest.raises(ValueError, match="mutation_tool_names"):
        guard_config(mutation_tool_names=())
    with pytest.raises(ValueError, match="mutation_tool_names"):
        guard_config(mutation_tool_names=("",))
    with pytest.raises(ValueError, match="diagnosis_responses"):
        guard_config(diagnosis_responses=-1)
    with pytest.raises(ValueError, match="initial_pending_command"):
        guard_config(initial_pending_command="   ")
    with pytest.raises(ValueError, match="initial_pending_command"):
        guard_config(initial_pending_command=OTHER_COMMAND)
    with pytest.raises(ValueError, match="initial_diagnosis_responses"):
        guard_config(initial_diagnosis_responses=-1)
    with pytest.raises(ValueError, match="initial_diagnosis_responses"):
        guard_config(initial_diagnosis_responses=1)


def test_one_diagnosis_response_allows_multiple_reads_then_blocks_broad_tools(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    listing = RecordingTool("list_directory")
    verify = RecordingTool("verify_workspace")
    llm = RecordingLLM(
        [
            tool_response(
                ("test-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                ("diagnose-a", "read_file", {"path": "a.py"}),
                ("diagnose-b", "read_file", {"path": "b.py"}),
            ),
            tool_response(
                ("blocked-read", "read_file", {"path": "c.py"}),
                ("blocked-list", "list_directory", {"path": "."}),
                ("blocked-command", "run_command", {"command": OTHER_COMMAND}),
                ("blocked-verify", "verify_workspace", {}),
            ),
            text_response("diagnosis-ended"),
        ]
    )
    agent = Agent(
        llm,
        registry_with(command, read, listing, verify),
        max_steps=4,
        verification_tool_name="verify_workspace",
    )

    assert agent.run(
        "Repair after a tracked failure.",
        require_verified_completion=False,
        progress_guard=guard_config(),
    ) == "diagnosis-ended"

    assert command.calls == [TRACKED_COMMAND]
    assert read.calls == [{"path": "a.py"}, {"path": "b.py"}]
    assert listing.calls == []
    assert verify.calls == []
    assert agent.workspace_revision == 1
    blocked_ids = (
        "blocked-read",
        "blocked-list",
        "blocked-command",
        "blocked-verify",
    )
    for call_id in blocked_ids:
        payload = result_payload(agent.history, call_id)
        assert payload["ok"] is False
        assert payload["error_type"] == "ProgressGuardBlocked"
        assert payload["policy_blocked"] is True
        assert payload["tool"]
        assert payload["message"]
        assert "Traceback" not in result_message(agent.history, call_id)[
            "content"
        ]
        assert sum(
            message.get("tool_call_id") == call_id
            for message in tool_messages(agent.history)
        ) == 1
    assert [
        message["tool_call_id"]
        for message in tool_messages(agent.history)[-4:]
    ] == list(blocked_ids)
    assert llm.calls[-1]["messages"][-4:] == tool_messages(agent.history)[-4:]


def test_two_diagnosis_responses_allow_two_rounds_then_block_the_third() -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("diagnose-one", "read_file", {"path": "one.py"})),
            tool_response(("diagnose-two", "read_file", {"path": "two.py"})),
            tool_response(
                ("blocked-third", "read_file", {"path": "three.py"})
            ),
            text_response("diagnosis-complete"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=5)

    assert agent.run(
        "Use two bounded diagnosis responses.",
        require_verified_completion=False,
        progress_guard=guard_config(diagnosis_responses=2),
    ) == "diagnosis-complete"

    assert read.calls == [{"path": "one.py"}, {"path": "two.py"}]
    assert result_payload(agent.history, "blocked-third")["error_type"] == (
        "ProgressGuardBlocked"
    )


def test_mutation_during_second_diagnosis_response_immediately_allows_retest(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1), CommandOutcome(0)])
    read = RecordingTool("read_file")
    edit = RecordingTool(
        "edit_file",
        mutates_workspace=True,
        default_result="File edited successfully: source.py",
    )
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                ("diagnose", "read_file", {"path": "source.py"})
            ),
            tool_response(
                (
                    "repair",
                    "edit_file",
                    {"path": "source.py", "content": "fixed"},
                )
            ),
            tool_response(
                ("blocked-other", "run_command", {"command": OTHER_COMMAND}),
                ("exact-retest", "run_command", {"command": TRACKED_COMMAND}),
            ),
            text_response("repair-complete"),
        ]
    )
    agent = Agent(
        llm,
        registry_with(command, read, edit),
        max_steps=5,
    )

    assert agent.run(
        "Mutate before the diagnosis budget is exhausted.",
        require_verified_completion=False,
        progress_guard=guard_config(diagnosis_responses=2),
    ) == "repair-complete"

    assert edit.calls == [{"path": "source.py", "content": "fixed"}]
    assert command.calls == [TRACKED_COMMAND, TRACKED_COMMAND]
    assert result_payload(agent.history, "blocked-other")["error_type"] == (
        "ProgressGuardBlocked"
    )
    assert result_payload(agent.history, "exact-retest")["exit_code"] == 0


def test_zero_diagnosis_responses_blocks_the_first_followup_read() -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("blocked", "read_file", {"path": "source.py"})),
            text_response("done"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=3)

    assert agent.run(
        "No diagnosis allowance.",
        progress_guard=guard_config(diagnosis_responses=0),
    ) == "done"
    assert read.calls == []
    assert result_payload(agent.history, "blocked")["error_type"] == (
        "ProgressGuardBlocked"
    )


def test_sibling_calls_keep_pairing_and_apply_state_transitions_immediately(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    edit = RecordingTool(
        "edit_file",
        mutates_workspace=True,
        default_result="File edited successfully: source.py",
    )
    listing = RecordingTool("list_directory")
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("diagnose", "read_file", {"path": "source.py"})),
            tool_response(
                ("blocked-before-edit", "read_file", {"path": "again.py"}),
                (
                    "repair",
                    "edit_file",
                    {"path": "source.py", "content": "fixed"},
                ),
                ("blocked-after-edit", "list_directory", {"path": "."}),
            ),
            text_response("intermediate-complete"),
        ]
    )
    agent = Agent(llm, registry_with(command, read, edit, listing), max_steps=4)

    assert agent.run(
        "Exercise sibling transitions.",
        progress_guard=guard_config(),
    ) == "intermediate-complete"

    assert read.calls == [{"path": "source.py"}]
    assert edit.calls == [{"path": "source.py", "content": "fixed"}]
    assert listing.calls == []
    assert agent.workspace_revision == 2
    sibling_results = tool_messages(agent.history)[-3:]
    assert [message["tool_call_id"] for message in sibling_results] == [
        "blocked-before-edit",
        "repair",
        "blocked-after-edit",
    ]
    assert result_payload(agent.history, "blocked-before-edit")[
        "error_type"
    ] == "ProgressGuardBlocked"
    assert result_message(agent.history, "repair")["content"].startswith(
        "File edited successfully"
    )
    assert result_payload(agent.history, "blocked-after-edit")[
        "error_type"
    ] == "ProgressGuardBlocked"


@pytest.mark.parametrize(
    "edit_outcome",
    [
        RuntimeError("scripted edit failure"),
        "No changes made: requested replacement is identical.",
    ],
    ids=["execution-error", "domain-no-op"],
)
def test_failed_or_noop_edit_does_not_release_mutation_requirement(
    edit_outcome: str | Exception,
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    edit = RecordingTool(
        "edit_file",
        mutates_workspace=True,
        outcomes=[edit_outcome],
    )
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("diagnose", "read_file", {"path": "source.py"})),
            tool_response(
                ("attempt", "edit_file", {"path": "source.py"}),
                ("must-stay-blocked", "read_file", {"path": "source.py"}),
            ),
            text_response("still-pending"),
        ]
    )
    agent = Agent(llm, registry_with(command, read, edit), max_steps=4)

    assert agent.run(
        "Do not mistake a failed mutation for progress.",
        progress_guard=guard_config(),
    ) == "still-pending"
    assert len(edit.calls) == 1
    assert read.calls == [{"path": "source.py"}]
    assert agent.workspace_revision == 2
    assert result_payload(agent.history, "must-stay-blocked")[
        "error_type"
    ] == "ProgressGuardBlocked"
    if isinstance(edit_outcome, Exception):
        assert result_payload(agent.history, "attempt")["error_type"] == (
            "RuntimeError"
        )
    else:
        assert result_message(agent.history, "attempt")["content"].startswith(
            "No changes made"
        )


def test_successful_mutations_enable_only_mutations_and_exact_retest_until_pass(
) -> None:
    command = ScriptedCommandTool(
        [CommandOutcome(1), CommandOutcome(0), CommandOutcome(1)]
    )
    read = RecordingTool("read_file")
    listing = RecordingTool("list_directory")
    verify = RecordingTool("verify_workspace")
    edit = RecordingTool(
        "edit_file",
        mutates_workspace=True,
        default_result="File edited successfully: source.py",
    )
    write = RecordingTool(
        "write_file",
        mutates_workspace=True,
        default_result="File written successfully: helper.py",
    )
    llm = RecordingLLM(
        [
            tool_response(
                ("initial-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("diagnose", "read_file", {"path": "source.py"})),
            tool_response(
                ("write", "write_file", {"path": "helper.py", "content": "x"})
            ),
            tool_response(
                ("strict-read", "read_file", {"path": "helper.py"}),
                ("strict-command", "run_command", {"command": OTHER_COMMAND}),
                ("edit", "edit_file", {"path": "source.py", "content": "y"}),
                ("retest-pass", "run_command", {"command": TRACKED_COMMAND}),
            ),
            tool_response(
                ("normal-read", "read_file", {"path": "helper.py"}),
                ("normal-list", "list_directory", {"path": "."}),
                ("normal-command", "run_command", {"command": OTHER_COMMAND}),
                ("normal-verify", "verify_workspace", {}),
            ),
            text_response("repair-complete"),
        ]
    )
    agent = Agent(
        llm,
        registry_with(command, read, listing, verify, edit, write),
        max_steps=6,
    )

    assert agent.run(
        "Repair and retest.",
        progress_guard=guard_config(),
    ) == "repair-complete"

    assert write.calls == [{"path": "helper.py", "content": "x"}]
    assert edit.calls == [{"path": "source.py", "content": "y"}]
    assert command.calls == [TRACKED_COMMAND, TRACKED_COMMAND, OTHER_COMMAND]
    assert read.calls == [
        {"path": "source.py"},
        {"path": "helper.py"},
    ]
    assert listing.calls == [{"path": "."}]
    assert verify.calls == [{}]
    assert result_payload(agent.history, "strict-read")["error_type"] == (
        "ProgressGuardBlocked"
    )
    assert result_payload(agent.history, "strict-command")["error_type"] == (
        "ProgressGuardBlocked"
    )
    assert result_message(agent.history, "retest-pass")["content"]
    assert result_message(agent.history, "normal-command")["content"]


def test_failed_exact_retest_opens_one_new_diagnosis_response() -> None:
    command = ScriptedCommandTool([CommandOutcome(1), CommandOutcome(2)])
    read = RecordingTool("read_file")
    edit = RecordingTool(
        "edit_file",
        mutates_workspace=True,
        default_result="File edited successfully: source.py",
    )
    llm = RecordingLLM(
        [
            tool_response(
                ("initial-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("diagnose-one", "read_file", {"path": "one.py"})),
            tool_response(("repair", "edit_file", {"path": "one.py"})),
            tool_response(
                ("retest-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                ("rediagnose-a", "read_file", {"path": "two.py"}),
                ("rediagnose-b", "read_file", {"path": "three.py"}),
            ),
            tool_response(
                ("blocked-after-rediagnosis", "read_file", {"path": "four.py"})
            ),
            text_response("pending-again"),
        ]
    )
    agent = Agent(llm, registry_with(command, read, edit), max_steps=7)

    assert agent.run(
        "Retry the repair cycle.",
        progress_guard=guard_config(),
    ) == "pending-again"
    assert command.calls == [TRACKED_COMMAND, TRACKED_COMMAND]
    assert read.calls == [
        {"path": "one.py"},
        {"path": "two.py"},
        {"path": "three.py"},
    ]
    assert result_payload(agent.history, "blocked-after-rediagnosis")[
        "error_type"
    ] == "ProgressGuardBlocked"


def test_diagnosis_response_cannot_retest_before_a_successful_mutation(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("initial-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                ("diagnose", "read_file", {"path": "source.py"}),
                (
                    "premature-retest",
                    "run_command",
                    {"command": f"  {TRACKED_COMMAND}  "},
                ),
            ),
            text_response("pending-mutation"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=3)

    assert agent.run(
        "Do not retest without a repair.",
        progress_guard=guard_config(),
    ) == "pending-mutation"
    assert command.calls == [TRACKED_COMMAND]
    assert read.calls == [{"path": "source.py"}]
    assert result_payload(agent.history, "premature-retest")[
        "error_type"
    ] == "ProgressGuardBlocked"


def test_another_tracked_command_cannot_clear_the_pending_failure() -> None:
    alternate_command = "python -B -m pytest tests/alternate -q"
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("initial-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                (
                    "alternate-pass-bypass",
                    "run_command",
                    {"command": alternate_command},
                ),
                ("diagnose", "read_file", {"path": "source.py"}),
            ),
            tool_response(("blocked", "read_file", {"path": "again.py"})),
            text_response("still-pending"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=4)

    assert agent.run(
        "Do not substitute a different tracked check.",
        progress_guard=guard_config(
            tracked_commands=(TRACKED_COMMAND, alternate_command),
        ),
    ) == "still-pending"
    assert command.calls == [TRACKED_COMMAND]
    assert read.calls == [{"path": "source.py"}]
    assert result_payload(agent.history, "alternate-pass-bypass")[
        "error_type"
    ] == "ProgressGuardBlocked"
    assert result_payload(agent.history, "blocked")["error_type"] == (
        "ProgressGuardBlocked"
    )


def test_configured_nonmutating_tool_does_not_fake_revision_progress() -> None:
    command = ScriptedCommandTool(
        [CommandOutcome(1), CommandOutcome(0)]
    )
    inspect = RecordingTool("inspect_only")
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("initial-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("diagnose-command", "run_command", {"command": OTHER_COMMAND})),
            tool_response(("configured-but-readonly", "inspect_only", {})),
            tool_response(("must-stay-blocked", "read_file", {"path": "source.py"})),
            text_response("still-pending"),
        ]
    )
    agent = Agent(llm, registry_with(command, inspect, read), max_steps=5)

    assert agent.run(
        "Require revision progress from the mutation call itself.",
        progress_guard=guard_config(
            mutation_tool_names=("inspect_only",),
        ),
    ) == "still-pending"
    assert command.calls == [TRACKED_COMMAND, OTHER_COMMAND]
    assert inspect.calls == [{}]
    assert read.calls == []
    assert result_payload(agent.history, "must-stay-blocked")[
        "error_type"
    ] == "ProgressGuardBlocked"


def test_new_failure_in_a_diagnosis_response_keeps_its_fresh_allowance(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1), CommandOutcome(2)])
    read = RecordingTool("read_file")
    edit = RecordingTool(
        "edit_file",
        mutates_workspace=True,
        default_result="File edited successfully: source.py",
    )
    llm = RecordingLLM(
        [
            tool_response(
                ("initial-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                ("diagnose", "read_file", {"path": "one.py"}),
                ("repair", "edit_file", {"path": "source.py"}),
                ("same-response-fail", "run_command", {"command": TRACKED_COMMAND}),
            ),
            tool_response(
                ("fresh-diagnosis-a", "read_file", {"path": "two.py"}),
                ("fresh-diagnosis-b", "read_file", {"path": "three.py"}),
            ),
            tool_response(
                ("blocked-after-fresh-cycle", "read_file", {"path": "four.py"})
            ),
            text_response("pending-again"),
        ]
    )
    agent = Agent(llm, registry_with(command, read, edit), max_steps=5)

    assert agent.run(
        "Keep a new failure cycle's diagnosis allowance.",
        progress_guard=guard_config(mutation_tool_names=("edit_file",)),
    ) == "pending-again"
    assert command.calls == [TRACKED_COMMAND, TRACKED_COMMAND]
    assert read.calls == [
        {"path": "one.py"},
        {"path": "two.py"},
        {"path": "three.py"},
    ]
    assert result_payload(agent.history, "blocked-after-fresh-cycle")[
        "error_type"
    ] == "ProgressGuardBlocked"


@pytest.mark.parametrize(
    "outcome",
    [
        CommandOutcome(0),
        CommandOutcome(1, timed_out=True),
        CommandOutcome(None),
        RuntimeError("command infrastructure failure"),
    ],
    ids=["pass", "timeout", "missing-exit-code", "execution-error"],
)
def test_only_completed_integer_nonzero_tracked_results_trigger_guard(
    outcome: CommandOutcome | Exception,
) -> None:
    command = ScriptedCommandTool([outcome])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("tracked", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("read-a", "read_file", {"path": "a.py"})),
            tool_response(("read-b", "read_file", {"path": "b.py"})),
            text_response("not-pending"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=4)

    assert agent.run(
        "Do not trigger on an incomplete or passing result.",
        progress_guard=guard_config(),
    ) == "not-pending"
    assert read.calls == [{"path": "a.py"}, {"path": "b.py"}]
    assert all(
        "ProgressGuardBlocked" not in str(message.get("content") or "")
        for message in agent.history
    )


def test_nontracked_failing_command_does_not_trigger_guard() -> None:
    command = ScriptedCommandTool([CommandOutcome(7)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("baseline-fail", "run_command", {"command": OTHER_COMMAND})
            ),
            tool_response(("read-a", "read_file", {"path": "a.py"})),
            tool_response(("read-b", "read_file", {"path": "b.py"})),
            text_response("baseline-audit-complete"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=4)

    assert agent.run(
        "Run an unrelated failing baseline.",
        progress_guard=guard_config(),
    ) == "baseline-audit-complete"
    assert command.calls == [OTHER_COMMAND]
    assert len(read.calls) == 2


def test_progress_guard_state_resets_per_run_without_resetting_full_history(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("run-a-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("run-a-read", "read_file", {"path": "a.py"})),
            text_response("run-a-pending"),
            tool_response(("run-b-read", "read_file", {"path": "b.py"})),
            text_response("run-b-complete"),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=3)
    config = guard_config()

    assert agent.run("run-a", progress_guard=config) == "run-a-pending"
    history_after_a = agent.history
    assert agent.run("run-b", progress_guard=config) == "run-b-complete"

    assert read.calls == [{"path": "a.py"}, {"path": "b.py"}]
    assert all(message in agent.history for message in history_after_a)
    second_run_first_request = llm.calls[3]["messages"]
    assert {"role": "user", "content": "run-a"} in second_run_first_request
    assert result_message(agent.history, "run-a-fail") in second_run_first_request
    assert result_message(agent.history, "run-a-read") in second_run_first_request


def test_initial_pending_failure_allows_three_responses_then_blocks_broad_tools(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    listing = RecordingTool("list_directory")
    verify = RecordingTool("verify_workspace")
    llm = RecordingLLM(
        [
            tool_response(
                ("run3-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            text_response("baseline-recorded"),
            tool_response(
                ("refresh-list", "list_directory", {"path": "."}),
                ("blocked-diagnostic", "run_command", {"command": OTHER_COMMAND}),
                ("blocked-verify", "verify_workspace", {}),
                ("refresh-read-a", "read_file", {"path": "a.py"}),
            ),
            tool_response(("refresh-read-b", "read_file", {"path": "b.py"})),
            tool_response(("refresh-list-two", "list_directory", {"path": "src"})),
            tool_response(
                ("blocked-read", "read_file", {"path": "c.py"}),
                ("blocked-list", "list_directory", {"path": "tests"}),
                ("blocked-retest", "run_command", {"command": TRACKED_COMMAND}),
            ),
            text_response("refresh-window-ended"),
        ]
    )
    agent = Agent(
        llm,
        registry_with(command, read, listing, verify),
        max_steps=5,
    )

    assert agent.run("Record the failing baseline.") == "baseline-recorded"
    revision_at_handoff = agent.workspace_revision
    assert agent.run(
        "Begin a bounded repair refresh.",
        progress_guard=guard_config(
            initial_pending_command=TRACKED_COMMAND,
            initial_diagnosis_responses=3,
        ),
    ) == "refresh-window-ended"

    assert revision_at_handoff == 1
    assert agent.workspace_revision == revision_at_handoff
    assert command.calls == [TRACKED_COMMAND]
    assert read.calls == [{"path": "a.py"}, {"path": "b.py"}]
    assert listing.calls == [{"path": "."}, {"path": "src"}]
    assert verify.calls == []
    for call_id in (
        "blocked-diagnostic",
        "blocked-verify",
        "blocked-read",
        "blocked-list",
        "blocked-retest",
    ):
        assert result_payload(agent.history, call_id)["error_type"] == (
            "ProgressGuardBlocked"
        )


def test_initial_pending_mutation_allows_only_exact_retest_then_clears(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    command = ScriptedCommandTool([CommandOutcome(1), CommandOutcome(0)])
    verify = RecordingTool("verify_workspace")
    llm = RecordingLLM(
        [
            tool_response(
                ("run3-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            text_response("baseline-recorded"),
            tool_response(("refresh-a", "read_file", {"path": "a.py"})),
            tool_response(("refresh-b", "read_file", {"path": "b.py"})),
            tool_response(("refresh-list", "list_directory", {"path": "."})),
            tool_response(
                (
                    "repair",
                    "write_file",
                    {"path": "fixed.py", "content": "fixed"},
                ),
                ("wrong-command", "run_command", {"command": OTHER_COMMAND}),
                ("exact-retest", "run_command", {"command": TRACKED_COMMAND}),
                ("normal-read", "read_file", {"path": "fixed.py"}),
                ("normal-verify", "verify_workspace", {}),
            ),
            text_response("repair-complete"),
        ]
    )
    agent = Agent(
        llm,
        registry_with(
            command,
            ReadFileTool(tmp_path),
            ListDirectoryTool(tmp_path),
            WriteFileTool(tmp_path),
            verify,
        ),
        max_steps=5,
    )

    assert agent.run("Record the failing baseline.") == "baseline-recorded"
    revision_at_handoff = agent.workspace_revision
    assert agent.run(
        "Repair and run the exact pending check.",
        progress_guard=guard_config(
            mutation_tool_names=("write_file",),
            initial_pending_command=TRACKED_COMMAND,
            initial_diagnosis_responses=3,
        ),
    ) == "repair-complete"

    assert command.calls == [TRACKED_COMMAND, TRACKED_COMMAND]
    assert agent.workspace_revision > revision_at_handoff
    assert (tmp_path / "fixed.py").read_text(encoding="utf-8") == "fixed"
    assert result_payload(agent.history, "wrong-command")["error_type"] == (
        "ProgressGuardBlocked"
    )
    assert result_payload(agent.history, "exact-retest")["exit_code"] == 0
    assert result_message(agent.history, "normal-read")["content"].startswith(
        "[read_file]"
    )
    assert verify.calls == [{}]


def test_initial_and_post_failure_diagnosis_budgets_are_independent(
    tmp_path: Path,
) -> None:
    (tmp_path / "source.py").write_text("old", encoding="utf-8")
    command = ScriptedCommandTool([CommandOutcome(1), CommandOutcome(2)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("run3-fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            text_response("baseline-recorded"),
            tool_response(
                (
                    "repair",
                    "write_file",
                    {"path": "source.py", "content": "changed"},
                ),
                ("retest-fail", "run_command", {"command": TRACKED_COMMAND}),
            ),
            tool_response(("diagnose-once", "read_file", {"path": "one.py"})),
            tool_response(("blocked-second", "read_file", {"path": "two.py"})),
            text_response("still-pending"),
        ]
    )
    agent = Agent(
        llm,
        registry_with(command, read, WriteFileTool(tmp_path)),
        max_steps=4,
    )

    assert agent.run("Record the failing baseline.") == "baseline-recorded"
    assert agent.run(
        "Repair with separate diagnosis budgets.",
        progress_guard=guard_config(
            mutation_tool_names=("write_file",),
            diagnosis_responses=1,
            initial_pending_command=TRACKED_COMMAND,
            initial_diagnosis_responses=3,
        ),
    ) == "still-pending"

    assert command.calls == [TRACKED_COMMAND, TRACKED_COMMAND]
    assert read.calls == [{"path": "one.py"}]
    assert result_payload(agent.history, "blocked-second")["error_type"] == (
        "ProgressGuardBlocked"
    )


def test_progress_guard_does_not_add_model_steps_or_pollute_execution_budget(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file")
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(("diagnose", "read_file", {"path": "one.py"})),
            tool_response(("blocked-a", "read_file", {"path": "two.py"})),
            tool_response(("blocked-b", "read_file", {"path": "three.py"})),
        ]
    )
    agent = Agent(llm, registry_with(command, read), max_steps=4)

    with pytest.raises(AgentMaxStepsError, match="4"):
        agent.run("Respect the existing budget.", progress_guard=guard_config())

    assert len(llm.calls) == 4
    assert read.calls == [{"path": "one.py"}]
    for index, call in enumerate(llm.calls, start=1):
        budgets = budget_messages(call)
        assert len(budgets) == 1
        assert f"Current model step: {index} / 4" in budgets[0]["content"]
        assert (
            "Remaining model responses including this one: "
            f"{5 - index}"
        ) in budgets[0]["content"]
    assert EXECUTION_BUDGET_HEADER not in json.dumps(agent.history)


def test_initial_pending_guard_stops_a_forty_response_read_only_loop() -> None:
    read = RecordingTool("read_file")
    responses = [
        tool_response(
            (
                f"read-loop-{index}",
                "read_file",
                {"path": f"module-{index}.py"},
            )
        )
        for index in range(1, 41)
    ]
    agent = Agent(
        RecordingLLM(responses),
        registry_with(read),
        max_steps=40,
    )

    with pytest.raises(AgentMaxStepsError, match="40"):
        agent.run(
            "Do not allow a seeded failure to degrade into a read-only loop.",
            progress_guard=guard_config(
                initial_pending_command=TRACKED_COMMAND,
                initial_diagnosis_responses=3,
            ),
        )

    assert read.calls == [
        {"path": "module-1.py"},
        {"path": "module-2.py"},
        {"path": "module-3.py"},
    ]
    for index in range(4, 41):
        assert result_payload(agent.history, f"read-loop-{index}")[
            "error_type"
        ] == "ProgressGuardBlocked"


def test_blocked_results_remain_bounded_and_compact_as_generic_tool_errors(
) -> None:
    command = ScriptedCommandTool([CommandOutcome(1)])
    read = RecordingTool("read_file", default_result="diagnostic-" + "x" * 650)
    responses: list[Any] = [
        tool_response(("fail", "run_command", {"command": TRACKED_COMMAND})),
        tool_response(("diagnose", "read_file", {"path": "large.py"})),
    ]
    responses.extend(
        tool_response(
            (f"blocked-{index}", "read_file", {"path": f"blocked-{index}.py"})
        )
        for index in range(1, 7)
    )
    responses.append(text_response("bounded-complete"))
    llm = RecordingLLM(responses)
    agent = Agent(
        llm,
        registry_with(command, read),
        max_steps=len(responses),
        max_context_chars=2_500,
        compaction_trigger_chars=1_000,
        max_compaction_chars=1_300,
    )

    assert agent.run(
        "Keep policy feedback bounded.",
        progress_guard=guard_config(),
    ) == "bounded-complete"

    assert all(
        estimate_messages_size(call["messages"]) <= 2_500
        for call in llm.calls
    )
    compacted_requests = [
        call["messages"]
        for call in llm.calls
        if any(
            CURRENT_RUN_HEADER in str(message.get("content") or "")
            for message in call["messages"]
        )
    ]
    assert compacted_requests
    assert any(
        "ProgressGuardBlocked"
        in json.dumps(messages, ensure_ascii=False)
        for messages in compacted_requests
    )

    compacted = build_compacted_context(
        agent.history,
        current_user_index=0,
        max_context_chars=1_500,
        max_compaction_chars=1_200,
    )
    compacted_text = json.dumps(compacted, ensure_ascii=False)
    assert "ProgressGuardBlocked" in compacted_text
    assert "execution_ok: false" in compacted_text
    assert estimate_messages_size(compacted) <= 1_500


def test_identical_real_write_keeps_mutation_required_after_revision_increment(
    tmp_path: Path,
) -> None:
    target = tmp_path / "workspace_note.txt"
    target.write_text("already repaired", encoding="utf-8")
    command = ScriptedCommandTool(
        [CommandOutcome(1), CommandOutcome(0)]
    )
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                (
                    "diagnose",
                    "read_file",
                    {"path": "workspace_note.txt"},
                )
            ),
            tool_response(
                (
                    "identical-write",
                    "write_file",
                    {
                        "path": "workspace_note.txt",
                        "content": "already repaired",
                    },
                ),
                (
                    "blocked-retest",
                    "run_command",
                    {"command": TRACKED_COMMAND},
                ),
                (
                    "blocked-read",
                    "read_file",
                    {"path": "workspace_note.txt"},
                ),
                ("blocked-list", "list_directory", {"path": "."}),
            ),
            text_response("repair-still-pending"),
        ]
    )
    agent = Agent(
        llm,
        registry_with(
            command,
            ReadFileTool(tmp_path),
            ListDirectoryTool(tmp_path),
            WriteFileTool(tmp_path),
        ),
        max_steps=4,
    )

    assert agent.run(
        "Require a real workspace change before retesting.",
        progress_guard=guard_config(
            mutation_tool_names=("write_file",),
        ),
    ) == "repair-still-pending"

    assert command.calls == [TRACKED_COMMAND]
    assert agent.workspace_revision == 2
    assert target.read_text(encoding="utf-8") == "already repaired"
    assert result_message(agent.history, "identical-write")[
        "content"
    ].startswith("No changes made:")
    for call_id in ("blocked-retest", "blocked-read", "blocked-list"):
        assert result_payload(agent.history, call_id)["error_type"] == (
            "ProgressGuardBlocked"
        )


@pytest.mark.parametrize(
    ("initial_content", "path", "new_content"),
    [
        ("old value", "existing.txt", "new value"),
        (None, "created.txt", "new file value"),
    ],
    ids=["changed-existing-file", "created-file"],
)
def test_real_write_progress_allows_exact_retest(
    tmp_path: Path,
    initial_content: str | None,
    path: str,
    new_content: str,
) -> None:
    target = tmp_path / path
    if initial_content is not None:
        target.write_text(initial_content, encoding="utf-8")
    else:
        assert not target.exists()
    command = ScriptedCommandTool(
        [CommandOutcome(1), CommandOutcome(0)]
    )
    llm = RecordingLLM(
        [
            tool_response(
                ("fail", "run_command", {"command": TRACKED_COMMAND})
            ),
            tool_response(
                ("diagnose", "read_file", {"path": "context.txt"})
            ),
            tool_response(
                (
                    "write-progress",
                    "write_file",
                    {"path": path, "content": new_content},
                ),
                (
                    "allowed-retest",
                    "run_command",
                    {"command": TRACKED_COMMAND},
                ),
            ),
            text_response("repair-complete"),
        ]
    )
    (tmp_path / "context.txt").write_text(
        "diagnostic context",
        encoding="utf-8",
    )
    agent = Agent(
        llm,
        registry_with(
            command,
            ReadFileTool(tmp_path),
            WriteFileTool(tmp_path),
        ),
        max_steps=4,
    )

    assert agent.run(
        "Make real progress and run the exact tracked retest.",
        progress_guard=guard_config(
            mutation_tool_names=("write_file",),
        ),
    ) == "repair-complete"

    assert command.calls == [TRACKED_COMMAND, TRACKED_COMMAND]
    assert target.read_text(encoding="utf-8") == new_content
    assert result_message(agent.history, "write-progress")[
        "content"
    ].startswith("File written successfully:")
    assert result_payload(agent.history, "allowed-retest")["exit_code"] == 0
