import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scripts.e2e_evaluator import (
    TASKS,
    CodingTask,
    HostCheckResult,
    TaskEvaluationResult,
    changed_files,
    collect_tool_call_counts,
    materialize_task,
    protected_files_unchanged,
    run_hidden_check,
    run_visible_tests,
    snapshot_workspace,
    summarize_results,
    task_passed,
)
from src.llm import LLMClient
from src.main import build_agent


HOST_CHECK_TIMEOUT_SECONDS = 30
OUTPUT_PREVIEW_CHARS = 1_600
EXPECTED_TOOL_NAMES = [
    "list_directory",
    "read_file",
    "edit_file",
    "write_file",
    "run_command",
    "verify_workspace",
]
FORBIDDEN_HIDDEN_FILES = {
    "answer.txt",
    "expected_solution.py",
    "hidden_cases.json",
    "hidden_test.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


def serialize(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def bounded_preview(value: str, max_chars: int = OUTPUT_PREVIEW_CHARS) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    marker = "\n[... host output omitted ...]\n"
    remaining = max_chars - len(marker)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining - head_chars
    return text[:head_chars] + marker + text[-tail_chars:]


def error_text(error: Exception) -> str:
    message = bounded_preview(str(error)) or "No error message."
    return f"{type(error).__name__}: {message}"


def append_error(current: str | None, new_error: str) -> str:
    return new_error if current is None else f"{current}; {new_error}"


class RecordingLLM:
    """Forward real requests unchanged while recording their inputs/outputs."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        call: dict[str, Any] = {
            "messages": deepcopy(messages),
            "tools": deepcopy(tools),
        }
        self.calls.append(call)
        response = self.client.chat(messages, tools=tools)
        call["response"] = response
        return response


def collect_tool_call_sequence(
    history: list[dict[str, Any]],
) -> list[str]:
    sequence: list[str] = []
    for message in history:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                sequence.append(name)
    return sequence


def assert_hidden_not_materialized(
    task: CodingTask,
    workspace: Path,
) -> None:
    fixture_sources = serialize(
        {
            "prompt": task.prompt,
            "files": task.files,
        }
    )
    require(
        task.hidden_check_code not in fixture_sources,
        f"{task.task_id} hidden evaluator leaked into its prompt or fixture.",
    )
    unexpected_hidden_files = [
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and path.name.casefold() in FORBIDDEN_HIDDEN_FILES
    ]
    require(
        not unexpected_hidden_files,
        f"{task.task_id} materialized hidden files: {unexpected_hidden_files}",
    )


def assert_hidden_not_sent_to_llm(
    task: CodingTask,
    recording: RecordingLLM,
) -> None:
    request_inputs = [
        {
            "messages": call["messages"],
            "tools": call["tools"],
        }
        for call in recording.calls
    ]
    require(
        task.hidden_check_code not in serialize(request_inputs),
        f"{task.task_id} hidden evaluator leaked into a real API request.",
    )


def failed_result(
    task: CodingTask,
    error: Exception | str,
) -> TaskEvaluationResult:
    detail = error_text(error) if isinstance(error, Exception) else error
    return TaskEvaluationResult(
        task_id=task.task_id,
        title=task.title,
        agent_completed=False,
        initial_visible_failed=False,
        visible_tests_passed=False,
        hidden_checks_passed=False,
        protected_files_unchanged=False,
        verification_tool_calls=0,
        verification_state_clean=False,
        temporary_workspace_cleaned=False,
        llm_calls=0,
        tool_calls=0,
        tool_call_counts={},
        changed_files=[],
        final_answer=None,
        error=detail,
    )


def run_one_task(
    task: CodingTask,
    task_index: int,
    task_count: int,
    client: LLMClient,
    verification_command: str,
) -> tuple[TaskEvaluationResult, HostCheckResult | None, HostCheckResult | None]:
    temporary_root_path: Path | None = None
    workspace_display = "<not created>"
    agent_completed = False
    initial_visible_failed = False
    visible_tests_passed = False
    hidden_checks_passed = False
    integrity_passed = False
    verification_state_clean = False
    llm_calls = 0
    tool_call_counts: dict[str, int] = {}
    tool_call_sequence: list[str] = []
    changed: list[str] = []
    final_answer: str | None = None
    task_error: str | None = None
    initial_check: HostCheckResult | None = None
    final_visible_check: HostCheckResult | None = None
    hidden_check: HostCheckResult | None = None
    recording: RecordingLLM | None = None
    agent: Any | None = None

    print("\n" + "=" * 60)
    print(f"Task {task_index}/{task_count}")
    print(f"ID: {task.task_id}")
    print(f"Title: {task.title}")
    print("=" * 60)

    try:
        with TemporaryDirectory(
            prefix=f"coding-agent-stage13-{task.task_id}-"
        ) as temporary_root:
            temporary_root_path = Path(temporary_root)
            workspace = temporary_root_path / "workspace"
            workspace_display = str(workspace)
            materialize_task(task, workspace)
            assert_hidden_not_materialized(task, workspace)

            initial_snapshot = snapshot_workspace(workspace)
            print(f"Temporary Workspace: {workspace}")
            print("Hidden checks configured: true (host-side only)")
            print("Protected Files: " + ", ".join(task.protected_files))

            initial_check = run_visible_tests(
                workspace,
                timeout=HOST_CHECK_TIMEOUT_SECONDS,
                basetemp=temporary_root_path / "initial-visible-pytest",
            )
            initial_visible_failed = (
                initial_check.exit_code not in (None, 0)
                and initial_check.timed_out is False
            )
            print(
                "Initial Visible Tests: "
                + (
                    "FAIL as expected"
                    if initial_visible_failed
                    else "INVALID FIXTURE RESULT"
                )
            )
            print(f"Initial Visible Tests exit_code: {initial_check.exit_code}")
            print(
                "Initial Visible Tests timed_out: "
                + str(initial_check.timed_out).lower()
            )
            require(
                initial_visible_failed,
                f"{task.task_id} initial visible tests must fail without timing out.",
            )

            recording = RecordingLLM(client)
            agent = build_agent(
                workspace,
                llm_client=recording,
                verification_commands=(verification_command,),
            )
            require(
                agent.tool_registry.names() == EXPECTED_TOOL_NAMES,
                f"{task.task_id} did not receive the production Tool Registry.",
            )
            require(
                agent.verification_tool_name == "verify_workspace",
                f"{task.task_id} Completion Gate is not enabled.",
            )
            verify_schema = next(
                schema
                for schema in agent.tool_registry.schemas()
                if schema["function"]["name"] == "verify_workspace"
            )
            require(
                verify_schema["function"]["parameters"]
                == {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "verify_workspace must not expose its Host command as arguments.",
            )
            pre_agent_sources = serialize(
                {
                    "system_prompt": agent.system_prompt,
                    "prompt": task.prompt,
                    "tools": agent.tool_registry.schemas(),
                    "verification_command": verification_command,
                }
            )
            require(
                task.hidden_check_code not in pre_agent_sources,
                f"{task.task_id} hidden evaluator leaked before Agent.run.",
            )

            try:
                final_answer = agent.run(task.prompt)
                agent_completed = True
            except Exception as error:
                task_error = append_error(task_error, error_text(error))

            history = agent.history
            tool_call_counts = collect_tool_call_counts(history)
            tool_call_sequence = collect_tool_call_sequence(history)
            llm_calls = len(recording.calls)
            assert_hidden_not_sent_to_llm(task, recording)

            calls_before_host_checks = len(recording.calls)
            history_before_host_checks = agent.history
            final_visible_check = run_visible_tests(
                workspace,
                timeout=HOST_CHECK_TIMEOUT_SECONDS,
                basetemp=temporary_root_path / "final-visible-pytest",
            )
            visible_tests_passed = final_visible_check.passed
            if agent_completed:
                hidden_check = run_hidden_check(task, workspace)
                hidden_checks_passed = hidden_check.passed
            require(
                len(recording.calls) == calls_before_host_checks
                and agent.history == history_before_host_checks,
                f"{task.task_id} Host checks changed Agent history or called LLM.",
            )

            final_snapshot = snapshot_workspace(workspace)
            integrity_passed = protected_files_unchanged(
                task,
                initial_snapshot,
                final_snapshot,
            )
            changed = changed_files(initial_snapshot, final_snapshot)
            verification_state_clean = (
                agent_completed
                and not agent.verification_required
                and agent.workspace_revision == agent.verified_revision
            )

            print("Tool Call Sequence: " + (
                " -> ".join(tool_call_sequence) if tool_call_sequence else "<none>"
            ))
    except Exception as error:
        task_error = append_error(task_error, error_text(error))
        if agent is not None:
            history = agent.history
            tool_call_counts = collect_tool_call_counts(history)
            tool_call_sequence = collect_tool_call_sequence(history)
            verification_state_clean = (
                agent_completed
                and not agent.verification_required
                and agent.workspace_revision == agent.verified_revision
            )
        if recording is not None:
            llm_calls = len(recording.calls)

    temporary_workspace_cleaned = (
        temporary_root_path is not None and not temporary_root_path.exists()
    )
    if not temporary_workspace_cleaned:
        task_error = append_error(
            task_error,
            f"Temporary workspace cleanup failed: {workspace_display}",
        )

    verification_calls = tool_call_counts.get("verify_workspace", 0)
    result = TaskEvaluationResult(
        task_id=task.task_id,
        title=task.title,
        agent_completed=agent_completed,
        initial_visible_failed=initial_visible_failed,
        visible_tests_passed=visible_tests_passed,
        hidden_checks_passed=hidden_checks_passed,
        protected_files_unchanged=integrity_passed,
        verification_tool_calls=verification_calls,
        verification_state_clean=verification_state_clean,
        temporary_workspace_cleaned=temporary_workspace_cleaned,
        llm_calls=llm_calls,
        tool_calls=sum(tool_call_counts.values()),
        tool_call_counts=tool_call_counts,
        changed_files=changed,
        final_answer=final_answer,
        error=task_error,
    )
    return result, final_visible_check, hidden_check


def print_host_check(label: str, check: HostCheckResult | None) -> None:
    if check is None:
        return
    print(f"{label} exit_code: {check.exit_code}")
    print(f"{label} timed_out: {str(check.timed_out).lower()}")
    if check.passed:
        return
    stdout_preview = bounded_preview(check.stdout)
    stderr_preview = bounded_preview(check.stderr)
    if stdout_preview:
        print(f"{label} stdout preview:\n{stdout_preview}")
    if stderr_preview:
        print(f"{label} stderr preview:\n{stderr_preview}")


def print_task_result(
    result: TaskEvaluationResult,
    visible_check: HostCheckResult | None,
    hidden_check: HostCheckResult | None,
) -> None:
    print("\nTask Result:")
    print("Agent completed: " + ("PASS" if result.agent_completed else "FAIL"))
    print(f"LLM calls: {result.llm_calls}")
    print(f"Tool calls: {result.tool_calls}")
    print("Tool counts:")
    for name in EXPECTED_TOOL_NAMES:
        print(f"  {name}: {result.tool_call_counts.get(name, 0)}")
    print(
        "Changed files: "
        + (", ".join(result.changed_files) if result.changed_files else "<none>")
    )
    print(
        "Visible host re-check: "
        + ("PASS" if result.visible_tests_passed else "FAIL")
    )
    print(
        "Hidden behavior checks: "
        + ("PASS" if result.hidden_checks_passed else "FAIL")
    )
    print(
        "Protected tests unchanged: "
        + ("PASS" if result.protected_files_unchanged else "FAIL")
    )
    print(
        "Verification tool called: "
        + ("PASS" if result.verification_tool_calls >= 1 else "FAIL")
        + f" ({result.verification_tool_calls})"
    )
    print(
        "Verification state clean: "
        + ("PASS" if result.verification_state_clean else "FAIL")
    )
    print(
        "Temporary Workspace cleanup: "
        + ("PASS" if result.temporary_workspace_cleaned else "FAIL")
    )
    if result.final_answer is not None:
        print("Final Answer:\n" + bounded_preview(result.final_answer))
    if result.error is not None:
        print("Error:\n" + bounded_preview(result.error))
    print_host_check("Visible host re-check", visible_check)
    print_host_check("Hidden check", hidden_check)
    print("TASK RESULT: " + ("PASS" if task_passed(result) else "FAIL"))


def print_summary(results: list[TaskEvaluationResult]) -> None:
    summary = summarize_results(results)
    print("\n" + "=" * 60)
    print("Stage 13 Summary")
    print("=" * 60)
    for index, result in enumerate(results, start=1):
        status = "PASS" if task_passed(result) else "FAIL"
        print(f"Task {index} - {result.title}: {status}")
    print(f"\nTasks: {summary.total}")
    print(f"Passed: {summary.passed}")
    print(f"Failed: {summary.failed}")
    print(
        "Visible Tests: "
        f"{sum(result.visible_tests_passed for result in results)}/"
        f"{summary.total} PASS"
    )
    print(
        "Hidden Checks: "
        f"{sum(result.hidden_checks_passed for result in results)}/"
        f"{summary.total} PASS"
    )
    print(
        "Protected Test Integrity: "
        f"{sum(result.protected_files_unchanged for result in results)}/"
        f"{summary.total} PASS"
    )
    print(
        "Verification Gate Usage: "
        f"{sum(result.verification_tool_calls >= 1 for result in results)}/"
        f"{summary.total} PASS"
    )
    print(
        "Temporary Workspaces: "
        f"{sum(result.temporary_workspace_cleaned for result in results)}/"
        f"{summary.total} cleaned"
    )
    if summary.all_passed:
        print("\nStage 13 End-to-End Coding Task Evaluation 成功")
    else:
        print("\nStage 13 End-to-End Coding Task Evaluation FAILED")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Stage 13 End-to-End Coding Task Evaluation")
    print("=" * 42)
    try:
        client = LLMClient()
    except Exception as error:
        print("Real LLM configuration failed: " + error_text(error))
        return 1

    print(f"\nModel: {client.model}")
    print(f"Tasks: {len(TASKS)}")
    print("Verification Gate: enabled")
    print("Hidden Evaluation: host-side / not exposed to Agent workspace")

    verification_command = make_command(
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
    )
    results: list[TaskEvaluationResult] = []
    for index, task in enumerate(TASKS, start=1):
        try:
            result, visible_check, hidden_check = run_one_task(
                task,
                index,
                len(TASKS),
                client,
                verification_command,
            )
        except Exception as error:
            result = failed_result(task, error)
            visible_check = None
            hidden_check = None
        results.append(result)
        print_task_result(result, visible_check, hidden_check)

    print_summary(results)
    return 0 if summarize_results(results).all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
