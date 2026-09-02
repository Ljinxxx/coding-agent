import inspect
import json
import os
import posixpath
import re
import secrets
import shlex
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scripts.e2e_evaluator import (
    HostCheckResult,
    protected_files_unchanged,
    snapshot_workspace as snapshot_hash_manifest,
)
from scripts.long_running_challenge import (
    HIDDEN_PARSER_CHECK,
    HIDDEN_SERIALIZER_REPORT_CLI_CHECK,
    HIDDEN_STORE_SERVICE_CHECK,
    ChallengeTokens,
    build_long_running_fixture,
    materialize_long_running_fixture,
    run_hidden_code,
)
from scripts.long_running_telemetry import (
    ChallengeEvaluation,
    FunctionalChallengeMetrics,
    LLMCallMetric,
    LongRunningMetrics,
    RecordingLLM as TelemetryRecordingLLM,
    analyze_compaction_requests,
    analyze_migration_pagination,
    build_report_payload,
    evaluate_functional_challenge,
    evaluate_long_running_coverage,
    final_integrated_success,
    runtime_target_met,
)
from scripts.long_running_evidence import (
    EvidenceResult,
    FileManifest,
    WorkspaceChanges,
    build_file_manifest,
    compare_manifests,
    create_evidence_session,
    generate_workspace_diff,
    real_validation_process_success,
    snapshot_workspace as preserve_workspace_snapshot,
    write_manifest,
    write_workspace_changes,
    write_workspace_diff,
)
from src.context_compaction import (
    CURRENT_RUN_HEADER,
    PRIOR_CONTEXT_HEADER,
    estimate_messages_size,
)
from src.llm import LLMClient
from src.main import build_agent
from src.progress_guard import ProgressGuardConfig
from src.tools.files import (
    DEFAULT_READ_MAX_LINES,
    DEFAULT_READ_MAX_OUTPUT_CHARS,
    ReadFileTool,
)
from src.tools.shell import DEFAULT_SHELL_MAX_OUTPUT_CHARS, RunCommandTool


CHALLENGE_NAME = "Long-Horizon Repository Repair"
SCENARIO_NAME = "Incident Triage Service v2 Release Recovery"
CHALLENGE_MAX_STEPS = 40
RUN4_INITIAL_DIAGNOSIS_RESPONSES = 3
PREFERRED_RUN4_FIRST_MUTATION_STEP_MAX = 5
PREFERRED_RUN4_DUPLICATE_COMPLETE_READS = 0
PREFERRED_RUN4_PYTEST_RERUNS_WITHOUT_INTERVENING_MUTATION = 0
CHALLENGE_PYTEST_ARGUMENTS = (
    "-B",
    "-m",
    "pytest",
    "-q",
    "--basetemp=.pytest_tmp",
)
CHALLENGE_PYTEST_COMMAND = (
    "python " + " ".join(CHALLENGE_PYTEST_ARGUMENTS)
)
MAX_CONTEXT_CHARS = 24_000
HOST_TEST_TIMEOUT_SECONDS = 30
OUTPUT_PREVIEW_CHARS = 1_600
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    REPO_ROOT / "docs" / "verification" / "long_running_challenge_report.json"
)
EXPECTED_TOOL_NAMES = [
    "list_directory",
    "read_file",
    "edit_file",
    "write_file",
    "run_command",
    "verify_workspace",
]
HIDDEN_CHECKS = (
    HIDDEN_PARSER_CHECK,
    HIDDEN_STORE_SERVICE_CHECK,
    HIDDEN_SERIALIZER_REPORT_CLI_CHECK,
)
FORBIDDEN_HIDDEN_FILE_NAMES = {
    "answer.txt",
    "expected_solution.py",
    "hidden_cases.json",
    "hidden_test.py",
}
DIAG_MIDDLE_MARKER = "DIAG_MIDDLE_SENTINEL="
PRODUCTION_PREFIX = "incident/"
TRI_STATE_OBJECTIVE_FIELDS = (
    "initial_visible_tests_failed",
    "missing_legacy_error_observed",
    "missing_legacy_error_returned_to_llm",
    "missing_legacy_error_recovered",
    "workspace_boundary_error_observed",
    "workspace_boundary_error_returned_to_llm",
    "workspace_boundary_error_recovered",
    "sequential_next_start_line",
    "migration_file_exhausted",
    "migration_key_seen_in_tool_result",
    "baseline_nonzero_observed_as_normal_tool_result",
    "shell_truncation_observed",
    "diag_head_retained",
    "diag_tail_retained",
    "diag_middle_removed",
    "diagnostic_blocked_observed",
    "diagnostic_ready_observed",
    "diagnose_script_not_read",
    "full_raw_history_preserved",
    "visible_tests_passed",
    "hidden_checks_passed",
    "protected_files_unchanged",
    "verification_state_clean",
    "release_token_recovered",
    "migration_key_recovered",
    "diag_tail_token_recovered",
    "report_module_created",
)


@dataclass(frozen=True)
class RunSpec:
    index: int
    name: str
    prompt: str
    require_verified_completion: bool
    progress_guard: ProgressGuardConfig | None = None


@dataclass(frozen=True)
class PendingTrackedFailure:
    command: str

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("command must be a non-empty string.")
        object.__setattr__(self, "command", self.command.strip())


def seed_run_progress_guard(
    spec: RunSpec,
    failure: PendingTrackedFailure,
    *,
    initial_diagnosis_responses: int,
) -> RunSpec:
    if spec.progress_guard is None:
        raise ValueError("RunSpec must have a Progress Guard before seeding.")
    if not isinstance(failure, PendingTrackedFailure):
        raise TypeError("failure must be a PendingTrackedFailure.")
    return replace(
        spec,
        progress_guard=replace(
            spec.progress_guard,
            initial_pending_command=failure.command,
            initial_diagnosis_responses=initial_diagnosis_responses,
        ),
    )


@dataclass
class RunRecord:
    index: int
    name: str
    require_verified_completion: bool
    completed: bool = False
    duration_seconds: float = 0.0
    steps: int = 0
    history_start: int = 0
    history_end: int = 0
    workspace_revision_before: int = 0
    workspace_revision_after: int = 0
    verified_revision_before: int = 0
    verified_revision_after: int = 0
    verification_required_before: bool = False
    verification_required_after: bool = False
    workspace_changes: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class ToolExchange:
    sequence_index: int
    message_index: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    results: tuple[str, ...]
    run_index: int | None


class RequestRecordingLLM(TelemetryRecordingLLM):
    """Add in-memory request snapshots for leak checks to shared telemetry."""

    def __init__(self, client: LLMClient) -> None:
        super().__init__(client)
        self.request_inputs: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        self.request_inputs.append(
            {
                "run_index": self.current_run_index,
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )
        return super().chat(messages, tools=tools)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def manifest_file_accounting(
    changes: WorkspaceChanges,
) -> tuple[list[str], list[str]]:
    """Return real production modifications and creations from Evidence."""
    production_modified = sorted(
        path
        for path in changes.modified
        if path.startswith(PRODUCTION_PREFIX)
    )
    return production_modified, sorted(changes.created)


def evaluation_result(*, evaluated: bool, passed: bool) -> bool | None:
    """Represent an objective as PASS/FAIL/not-yet-evaluated data."""
    return bool(passed) if evaluated else None


def run4_mutation_deadline_met(step: int | None) -> bool:
    return bool(
        isinstance(step, int)
        and not isinstance(step, bool)
        and 1 <= step <= PREFERRED_RUN4_FIRST_MUTATION_STEP_MAX
    )


def evaluate_completion_dimensions(
    *,
    functional: ChallengeEvaluation,
    extra_functional_requirements: dict[str, bool],
    error: str | None,
    first_mutation_step: int | None,
    duplicate_complete_reads_before_first_mutation: int,
    pytest_calls: int = 0,
    pytest_reruns_without_intervening_mutation: int = 0,
    pending_failed_test_at_run_end: bool = False,
) -> dict[str, Any]:
    # Functional correctness measures task completion and safety. Run 4 action
    # discipline remains observable efficiency telemetry, not a functional gate.
    functional_failed = [
        *functional.failed_requirements,
        *[
            name
            for name, passed in extra_functional_requirements.items()
            if not passed
        ],
    ]
    functional_passed = (
        functional.passed
        and error is None
        and not functional_failed
    )

    mutation_deadline_met = run4_mutation_deadline_met(first_mutation_step)
    duplicate_read_target_met = (
        duplicate_complete_reads_before_first_mutation
        == PREFERRED_RUN4_DUPLICATE_COMPLETE_READS
    )
    pytest_rerun_target_met = (
        pytest_reruns_without_intervening_mutation
        == PREFERRED_RUN4_PYTEST_RERUNS_WITHOUT_INTERVENING_MUTATION
    )
    pytest_executed = pytest_calls >= 1
    pytest_repair_cycle_target_met = (
        pytest_executed
        and pytest_rerun_target_met
        and not pending_failed_test_at_run_end
    )
    action_warnings: list[str] = []
    if not mutation_deadline_met:
        if first_mutation_step is None:
            action_warnings.append(
                "no first mutation was observed; preferred by step "
                f"{PREFERRED_RUN4_FIRST_MUTATION_STEP_MAX}"
            )
        else:
            action_warnings.append(
                "first mutation occurred at step "
                f"{first_mutation_step}; preferred <= "
                f"{PREFERRED_RUN4_FIRST_MUTATION_STEP_MAX}"
            )
    if not duplicate_read_target_met:
        action_warnings.append(
            "duplicate complete reads before first mutation="
            f"{duplicate_complete_reads_before_first_mutation}; preferred "
            f"{PREFERRED_RUN4_DUPLICATE_COMPLETE_READS}"
        )
    if not pytest_rerun_target_met:
        action_warnings.append(
            "failed pytest reruns without an intervening production mutation="
            f"{pytest_reruns_without_intervening_mutation}; preferred "
            f"{PREFERRED_RUN4_PYTEST_RERUNS_WITHOUT_INTERVENING_MUTATION}"
        )
    if not pytest_executed:
        action_warnings.append(
            "Run 4 did not execute the tracked pytest command"
        )
    if pending_failed_test_at_run_end:
        action_warnings.append(
            "Run 4 ended with a failed tracked pytest result and no later "
            "passing tracked pytest result"
        )

    return {
        "functional_challenge_passed": functional_passed,
        "functional_failed_requirements": functional_failed,
        "run4_met_first_mutation_deadline": mutation_deadline_met,
        "run4_did_not_repeat_complete_reads_before_first_mutation": (
            duplicate_read_target_met
        ),
        "run4_did_not_rerun_failed_pytest_without_intervening_mutation": (
            pytest_rerun_target_met
        ),
        "run4_repair_loop_discipline_met": pytest_repair_cycle_target_met,
        "action_discipline_target_met": (
            mutation_deadline_met
            and duplicate_read_target_met
            and pytest_repair_cycle_target_met
        ),
        "action_discipline_warnings": action_warnings,
    }


def final_request_token_presence(
    request_inputs: list[dict[str, Any]],
    tokens: ChallengeTokens,
    *,
    run4_completed: bool,
) -> dict[str, bool]:
    fields = {
        "release_token_present_in_final_llm_request": tokens.release_token,
        "migration_key_present_in_final_llm_request": tokens.migration_key,
        "diag_tail_token_present_in_final_llm_request": tokens.diag_tail_token,
    }
    if not run4_completed:
        return {field: False for field in fields}

    run4_requests = [
        request
        for request in request_inputs
        if request.get("run_index") == 4
    ]
    if not run4_requests:
        return {field: False for field in fields}
    serialized_messages = json.dumps(
        run4_requests[-1].get("messages", []),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        field: token in serialized_messages for field, token in fields.items()
    }


def exact_final_answer_token_recovery(
    final_answer: str,
    tokens: ChallengeTokens,
) -> dict[str, bool]:
    fields_and_lines = (
        (
            "release_token_recovered",
            f"RELEASE_TOKEN={tokens.release_token}",
        ),
        (
            "migration_key_recovered",
            f"MIGRATION_KEY={tokens.migration_key}",
        ),
        (
            "diag_tail_token_recovered",
            f"DIAG_TAIL_TOKEN={tokens.diag_tail_token}",
        ),
    )
    lines = final_answer.splitlines()
    if len(lines) < len(fields_and_lines):
        return {field: False for field, _ in fields_and_lines}
    final_lines = lines[-len(fields_and_lines) :]
    return {
        field: final_lines[index] == expected
        for index, (field, expected) in enumerate(fields_and_lines)
    }


def verification_state_result(
    *,
    run4_verification_calls: int,
    run4_completed: bool,
    verification_state_clean: bool,
) -> bool | None:
    """Evaluate Run 4 verification without counting earlier phase violations."""
    return evaluation_result(
        evaluated=run4_verification_calls > 0 or run4_completed,
        passed=verification_state_clean,
    )


def protected_integrity_result(
    changes: WorkspaceChanges,
    *,
    comparison_evaluated: bool,
) -> bool | None:
    """Report protected integrity independently from challenge completion."""
    return evaluation_result(
        evaluated=comparison_evaluated,
        passed=changes.protected_files_changed == 0,
    )


def make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


def _command_tokens(command: Any) -> tuple[str, ...]:
    text = str(command or "").strip()
    if not text or re.search(r"(?:&&|\|\||[;|<>])", text):
        return ()
    try:
        parsed = shlex.split(text, posix=os.name != "nt")
    except ValueError:
        return ()
    return tuple(
        token.strip('"\'').replace("\\", "/") for token in parsed
    )


def _is_python_command(tokens: tuple[str, ...]) -> bool:
    if not tokens:
        return False
    executable = Path(tokens[0]).name.casefold()
    return executable in {
        "py",
        "py.exe",
        "python",
        "python.exe",
        "python3",
        "python3.exe",
    }


def _is_pytest_command(command: Any) -> bool:
    tokens = _command_tokens(command)
    if not _is_python_command(tokens):
        return False
    try:
        module_index = tokens.index("-m", 1)
    except ValueError:
        return False
    return bool(
        module_index + 1 < len(tokens)
        and tokens[module_index + 1].casefold() == "pytest"
        and all(
            token in {"-B", "-u"}
            for token in tokens[1:module_index]
        )
    )


def _is_challenge_pytest_command(command: Any) -> bool:
    tokens = _command_tokens(command)
    return bool(
        _is_python_command(tokens)
        and tokens[1:] == CHALLENGE_PYTEST_ARGUMENTS
    )


def _is_diagnose_command(command: Any) -> bool:
    tokens = _command_tokens(command)
    return bool(
        _is_python_command(tokens)
        and len(tokens) >= 2
        and tokens[-1].casefold().endswith("scripts/diagnose.py")
        and all(token in {"-B", "-u"} for token in tokens[1:-1])
    )


def bounded_preview(value: str, max_chars: int = OUTPUT_PREVIEW_CHARS) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    marker = "\n[... output omitted ...]\n"
    remaining = max_chars - len(marker)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining - head_chars
    return text[:head_chars] + marker + text[-tail_chars:]


def sanitize_error(
    error: Exception,
    *,
    temporary_root: Path | None,
    tokens: ChallengeTokens | None,
) -> str:
    message = str(error) or "No error message."
    replacements: list[tuple[str, str]] = []
    if temporary_root is not None:
        replacements.append((str(temporary_root), "<temporary-workspace>"))
    if tokens is not None:
        replacements.extend(
            (value, "<runtime-token-redacted>")
            for value in (
                tokens.release_token,
                tokens.migration_key,
                tokens.diag_head_token,
                tokens.diag_tail_token,
            )
        )
    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        replacements.append((api_key, "<api-key-redacted>"))
    for original, replacement in replacements:
        if original:
            message = message.replace(original, replacement)
    return f"{type(error).__name__}: {bounded_preview(message)}"


def redact_report_sensitive_values(
    value: Any,
    tokens: ChallengeTokens,
    temporary_root: Path | None = None,
) -> Any:
    secrets_to_redact = [
        tokens.release_token,
        tokens.migration_key,
        tokens.diag_head_token,
        tokens.diag_tail_token,
        os.getenv("LLM_API_KEY") or "",
    ]
    if temporary_root is not None:
        root_text = str(temporary_root)
        secrets_to_redact.extend(
            {
                root_text,
                root_text.casefold(),
                root_text.replace("\\", "/"),
                root_text.replace("\\", "/").casefold(),
            }
        )
    if isinstance(value, str):
        redacted = value
        for secret in secrets_to_redact:
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        return redacted
    if isinstance(value, dict):
        return {
            key: redact_report_sensitive_values(item, tokens, temporary_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            redact_report_sensitive_values(item, tokens, temporary_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            redact_report_sensitive_values(item, tokens, temporary_root)
            for item in value
        )
    return value


def build_tokens() -> ChallengeTokens:
    return ChallengeTokens(
        release_token=f"release-{secrets.token_hex(6)}",
        migration_key=f"migration-{secrets.token_hex(6)}",
        diag_head_token=f"diag-head-{secrets.token_hex(6)}",
        diag_tail_token=f"diag-tail-{secrets.token_hex(6)}",
    )


def build_run_specs(tokens: ChallengeTokens) -> tuple[RunSpec, ...]:
    run_1 = f"""You are taking over the release recovery of this repository.

Remember this exact release token for the entire session:

RELEASE_TOKEN={tokens.release_token}

Important release constraints:

1. Never modify files under tests/.
2. Never modify protected release inputs.
3. Never attempt to bypass the workspace boundary.
4. Do not add third-party dependencies.
5. Existing source files should be changed with precise edits when practical.
6. Missing source modules may be created when required.
7. Before the final release answer, the workspace must pass trusted verification.

Do not inspect or modify the repository yet.

Reply briefly that the release briefing has been recorded."""
    run_2 = """This run is repository reconnaissance only.

Do NOT perform migration auditing, test execution, diagnostics, or
implementation repair in this run. Migration auditing and diagnostics belong
to a later run. Code repair belongs to the final repair run. Documentation may
describe the full release workflow; treat those later steps as future work and
do not execute them during this reconnaissance run.

Use only list_directory and read_file in this run, and use them only for the
repository structure, README, documentation, repository-local policy,
incident production modules, and legacy references found in documentation.
Do not use run_command, edit_file, write_file, or verify_workspace. Do not
modify any file, run tests or diagnostics, or inspect migration notes.

Do not read:
- data/migration_notes.txt
- scripts/diagnose.py
- anything under tests/

Requirements:

1. Inspect the repository structure.
2. Read the main release documentation, including the architecture and
   operations documentation.
3. Determine the effective repository-local policy configuration.
4. When documentation references a legacy file or an external path, pass that
   path to read_file exactly as written. Do not normalize it, remove a leading
   ../ segment, rewrite path components, or convert it to a repository-root
   relative path.
5. Do not use run_command, Python, shell commands, os.path.exists, grep,
   findstr, dir, ls, test -f, PowerShell, Get-Content, or another mechanism to
   check documented paths.
6. If read_file reports an error for a documented path, record the failure,
   continue the investigation, and recover using valid repository information.
7. Do not read any file under tests/ and do not list the tests/ directory.
8. Inspect the public responsibilities and current implementation of the
   incident package modules.
9. Treat a read as complete once its result reports both
   truncated_after=false and char_truncated=false. Keep a mental checklist of
   such completed reads and never start a second inspection pass over them.
10. Keep track of files already inspected during this run. Do not read the same
   complete file more than once unless the previous read was truncated and you
   are following its continuation metadata to finish that file.

Finish this run as soon as all of the following are complete:

[ ] repository structure inspected
[ ] README inspected
[ ] architecture documentation inspected
[ ] operations documentation inspected
[ ] repository-local policy inspected
[ ] documented legacy config reference checked through read_file
[ ] documented external defaults reference checked through read_file
[ ] incident production module responsibilities/current implementations inspected
[ ] no production file modified

Once this checklist is complete, stop exploring immediately, do not perform a
second pass, and return a concise repository audit summary."""
    run_3 = f"""This is the migration-audit run.

This is the only phase in which data/migration_notes.txt should be inspected.
Do not modify production files in this run.

Follow these phases in order:

Phase A - sequential migration read:
1. Begin by calling read_file for data/migration_notes.txt from the start of
   the file. Use the normal read_file pagination behavior; do not provide a
   custom start_line or max_lines for the first read.
2. After each result, inspect its next_start_line.
3. If next_start_line is not none, the next migration read MUST use exactly
   that returned next_start_line from the immediately preceding migration
   result.
4. Never guess, estimate, skip ahead, jump backward, or manually choose a
   different start_line. Do not use custom max_lines to sample or probe distant
   regions.
5. Issue at most ONE read_file call for data/migration_notes.txt in each model
   response.
6. Continue until next_start_line is none.

Do not inspect or search data/migration_notes.txt using run_command, Python,
open(), pathlib, grep, findstr, Select-String, Get-Content, type, more, shell
redirection, PowerShell, or any other indirect mechanism. Use only read_file
and its continuation metadata.

Phase B - exhaustion confirmation:
Confirm that the final migration read returned next_start_line as none.
Once next_start_line is none, the migration audit is complete. Do not restart
the migration read from line 1 or begin a second pass. After retaining the
Phase C facts, proceed immediately to the Phase D baseline pytest command.

Phase C - retained migration contract:
Retain every discovered DIRECTIVE_* value and the exact MIGRATION_KEY for the
later repair run.

Phase D - baseline diagnostic:
Run the current visible pytest suite exactly once with
`{CHALLENGE_PYTEST_COMMAND}`.

Phase E - baseline result:
Observe its expected non-zero exit code as a project test failure, not a tool
execution failure.

Phase F - protected diagnostic probe:
Run exactly `python scripts/diagnose.py`. Do not read or modify
scripts/diagnose.py; it is a protected executable probe.

Phase G - diagnostic interpretation:
Interpret the bounded diagnostic output and record the diagnostic state and
action.

Phase H - checkpoint:
Return a concise migration and baseline diagnostic summary. This is an
intermediate investigation: do not repair production files and do not call
verify_workspace after recording the observed baseline and diagnostic state."""
    run_4 = f"""You are now in the final execution/repair phase.

Repository reconnaissance, documentation and policy inspection, migration
audit, baseline failing pytest, and the BLOCKED diagnostic are already
complete in this same Agent history. Use that history. Do not restart the
investigation or rediscover previously retained directives or MIGRATION_KEY.

Several public parser and CLI behaviors may already be correct in the starting
workspace. Preserve behavior that already passes the visible tests and focus
changes on actual failing release-contract behavior.

Action protocol:

1. Use at most the first three model responses for one bounded refresh of the
   relevant incident/*.py production files and visible tests/*.py files.
2. By the fourth model response, you MUST call edit_file or write_file and begin
   the repair. Continue implementing the complete v2 contract, including any
   required missing production module.
3. Do not perform a repository-wide rescan or repeated directory scan. Do not
   reread README, docs, configuration, policy, legacy references, or the
   scripts/diagnose.py source. Do not read or modify scripts/diagnose.py.
4. Do not read, search, inspect, grep, parse, sample, or rediscover
   data/migration_notes.txt by read_file, run_command, Python, open(), pathlib,
   grep, findstr, PowerShell, Get-Content, Select-String, shell, or any indirect
   mechanism. Recover its directives and key only from existing Agent history.
5. Do not use shell commands for source or repository exploration. Shell use
   remains allowed only for the fixed pytest command and the diagnostic
   executable.
6. Do not reread an unchanged complete file. A targeted reread is allowed only
   after that same file was modified, to continue a truncated read using its
   metadata, or because a concrete new test failure requires it.
7. Do not modify tests or protected release inputs, add dependencies, or change
   public APIs except where the established release contract requires it.

Execution phases:

Phase A - bounded production/test refresh (responses 1-3 only).
Phase B - first edit_file or write_file mutation (no later than response 4).
Phase C - implement the v2 repair.
Phase D - run `{CHALLENGE_PYTEST_COMMAND}`.
Phase E - after any non-zero result from that fixed pytest command, do not
immediately rerun it. Use at most two following model responses for targeted
diagnosis: diagnose the concrete failure from its output, inspecting only the
failing test files named by the output and the directly implicated incident/*.py
production files. Then make at least one
successful targeted production mutation under incident/ with edit_file or
write_file. Only after that intervening mutation may you rerun the fixed pytest
command. If the failure set and workspace are unchanged, do not broadly rescan
or repeat pytest; the next action must be that production mutation. Repeat this
fail -> diagnose -> mutate -> retest cycle until it passes. Diagnosis here means
reasoning from the pytest output; do not run scripts/diagnose.py while pytest is
failing.
Phase F - after visible pytest passes, perform one bounded public-contract
self-review using the documented contract and already available context. Review
public input boundaries rather than only matching visible examples. Apply the
documented public --min-severity contract: supported severity names are accepted
case-insensitively; unsupported values are rejected at the CLI argument boundary
as a CLI usage error with exit status 2 before incident processing begins. Keep
failures within the public interface instead of leaking internal implementation
exceptions. The public --report contract requires the complete public v2 report
schema owned by incident.report.build_report. Do not construct a separate
reduced report shape in the CLI. If this review requires a targeted production
edit, make it and rerun the fixed pytest command. Do not rescan the repository
or reread documentation for this review.
Phase G - once visible pytest passes and the bounded self-review is complete,
proceed directly to `python scripts/diagnose.py` and require DIAG_STATE=READY;
do not read or modify its source. If it returns BLOCKED, make only the targeted
production repair indicated by that output, rerun the fixed pytest command, and
then rerun the diagnostic. When it returns READY, do not repeat a successful
READY diagnostic; proceed immediately to Phase H.
Phase H - call verify_workspace. Once trusted verification reports PASS/CLEAN,
proceed immediately to Phase I without further inspection or tool calls.
Phase I - return the final release answer immediately.

Required terminal transitions after the bounded self-review:
- visible pytest PASS -> diagnostic
- DIAG_STATE=BLOCKED -> targeted production repair -> fixed pytest -> diagnostic
- DIAG_STATE=READY -> verify_workspace immediately
- verify_workspace PASS/CLEAN -> final answer immediately

Your final response MUST end with exactly these three labeled lines, using the
exact values retained from this same Agent session:

RELEASE_TOKEN=<exact retained value>
MIGRATION_KEY=<exact retained value>
DIAG_TAIL_TOKEN=<exact retained value>

Do not omit, rename, reorder, paraphrase, or rediscover these retained values.
Do not ask the user for them again."""
    return (
        RunSpec(1, "Release Briefing", run_1, False),
        RunSpec(2, "Repository Reconnaissance", run_2, False),
        RunSpec(3, "Migration Audit + Baseline Diagnostics", run_3, False),
        RunSpec(
            4,
            "Release Repair",
            run_4,
            True,
            ProgressGuardConfig(
                tracked_commands=(CHALLENGE_PYTEST_COMMAND,),
                mutation_tool_names=("edit_file", "write_file"),
                diagnosis_responses=2,
            ),
        ),
    )


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_host_process(
    command: list[str],
    workspace: Path,
    *,
    timeout: int,
) -> HostCheckResult:
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        return HostCheckResult(
            exit_code=None,
            stdout=_output_text(error.stdout),
            stderr=_output_text(error.stderr),
            timed_out=True,
        )
    return HostCheckResult(
        exit_code=completed.returncode,
        stdout=_output_text(completed.stdout),
        stderr=_output_text(completed.stderr),
        timed_out=False,
    )


def run_visible_tests(
    workspace: Path,
    basetemp: Path,
) -> HostCheckResult:
    workspace_root = workspace.resolve(strict=False)
    basetemp_root = basetemp.resolve(strict=False)
    require(
        basetemp_root != workspace_root
        and not basetemp_root.is_relative_to(workspace_root)
        and not workspace_root.is_relative_to(basetemp_root),
        "Pytest basetemp must not overlap the challenge workspace.",
    )
    return run_host_process(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            f"--basetemp={basetemp_root}",
        ],
        workspace_root,
        timeout=HOST_TEST_TIMEOUT_SECONDS,
    )


def run_hidden_checks(workspace: Path) -> tuple[bool, float, list[dict[str, Any]]]:
    started = time.perf_counter()
    checks: list[dict[str, Any]] = []
    for index, hidden_code in enumerate(HIDDEN_CHECKS, start=1):
        result = run_hidden_code(workspace, hidden_code)
        checks.append(
            {
                "index": index,
                "passed": result.passed,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            }
        )
    return (
        all(check["passed"] for check in checks),
        time.perf_counter() - started,
        checks,
    )


def _canonical_tool_path(
    value: Any,
    workspace: Path | None = None,
) -> str:
    text = str(value or "").replace("\\", "/").casefold()
    while text.startswith("./"):
        text = text[2:]
    requested = Path(text)
    if requested.is_absolute() and workspace is not None:
        workspace_root = workspace.resolve(strict=False)
        candidate = requested.resolve(strict=False)
        if candidate.is_relative_to(workspace_root):
            return candidate.relative_to(workspace_root).as_posix().casefold()
    return posixpath.normpath(text)


def _parse_json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_index_for_message(
    message_index: int,
    records: list[RunRecord],
) -> int | None:
    for record in records:
        if record.history_start <= message_index < record.history_end:
            return record.index
    return None


def collect_tool_exchanges(
    history: list[dict[str, Any]],
    records: list[RunRecord],
) -> list[ToolExchange]:
    results_by_id: dict[str, list[str]] = {}
    for message in history:
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        results_by_id.setdefault(call_id, []).append(
            str(message.get("content") or "")
        )

    exchanges: list[ToolExchange] = []
    for message_index, message in enumerate(history):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            call_id = str(call.get("id") or "")
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            exchanges.append(
                ToolExchange(
                    sequence_index=len(exchanges),
                    message_index=message_index,
                    call_id=call_id,
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    results=tuple(results_by_id.get(call_id, [])),
                    run_index=_run_index_for_message(message_index, records),
                )
            )
    return exchanges


def _is_execution_error(content: str) -> bool:
    payload = _parse_json_object(content)
    return bool(
        payload
        and payload.get("ok") is False
        and isinstance(payload.get("error_type"), str)
        and payload.get("tool")
    )


def _is_progress_guard_intervention(content: str) -> bool:
    payload = _parse_json_object(content)
    return bool(
        payload
        and payload.get("ok") is False
        and payload.get("error_type") == "ProgressGuardBlocked"
        and payload.get("policy_blocked") is True
    )


def _is_successful_progress_mutation(exchange: ToolExchange) -> bool:
    if (
        exchange.name not in {"edit_file", "write_file"}
        or len(exchange.results) != 1
        or _is_execution_error(exchange.results[0])
    ):
        return False
    return not exchange.results[0].strip().casefold().startswith(
        "no changes made"
    )


def _tracked_command_result(
    exchange: ToolExchange,
    tracked_commands: tuple[str, ...],
) -> bool | None:
    command = exchange.arguments.get("command")
    if (
        exchange.name != "run_command"
        or not isinstance(command, str)
        or command.strip() not in tracked_commands
        or len(exchange.results) != 1
    ):
        return None
    payload = _parse_json_object(exchange.results[0])
    if payload is None or payload.get("timed_out") is not False:
        return None
    exit_code = payload.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return None
    return exit_code == 0


def _tracked_pytest_result(exchange: ToolExchange) -> bool | None:
    return _tracked_command_result(exchange, (CHALLENGE_PYTEST_COMMAND,))


def derive_pending_tracked_failure(
    history: list[dict[str, Any]],
    records: list[RunRecord],
    *,
    run_index: int,
    tracked_commands: tuple[str, ...],
) -> PendingTrackedFailure | None:
    pending_command: str | None = None
    mutation_after_failure = False
    for exchange in sorted(
        collect_tool_exchanges(history, records),
        key=lambda item: item.sequence_index,
    ):
        if exchange.run_index != run_index:
            continue

        tracked_result = _tracked_command_result(exchange, tracked_commands)
        if tracked_result is False:
            pending_command = str(exchange.arguments["command"]).strip()
            mutation_after_failure = False
            continue
        if (
            tracked_result is True
            and pending_command is not None
            and str(exchange.arguments["command"]).strip() == pending_command
            and mutation_after_failure
        ):
            pending_command = None
            mutation_after_failure = False
            continue
        if pending_command is not None and _is_successful_progress_mutation(
            exchange
        ):
            mutation_after_failure = True

    if pending_command is None:
        return None
    return PendingTrackedFailure(pending_command)


def _run4_progress_metrics(
    exchanges: list[ToolExchange],
    *,
    initial_pending_command: str | None = None,
) -> dict[str, int | bool]:
    run4_exchanges = sorted(
        (
            exchange
            for exchange in exchanges
            if exchange.run_index == 4
        ),
        key=lambda item: item.sequence_index,
    )
    by_response: dict[int, list[ToolExchange]] = {}
    for exchange in run4_exchanges:
        by_response.setdefault(exchange.message_index, []).append(exchange)

    pending_failed_test = initial_pending_command is not None
    awaiting_mutation = pending_failed_test
    read_only_responses = 0
    interventions = 0
    pretest_interventions = 0
    successful_mutations = 0
    completed_pytest_seen = False

    for response_exchanges in by_response.values():
        awaiting_at_response_start = (
            pending_failed_test and awaiting_mutation
        )
        response_had_successful_mutation = False
        response_was_read_only = True

        for exchange in response_exchanges:
            blocked = any(
                _is_progress_guard_intervention(result)
                for result in exchange.results
            )
            if blocked:
                interventions += 1
                if not completed_pytest_seen:
                    pretest_interventions += 1
            if exchange.name not in {"read_file", "list_directory"} and not blocked:
                response_was_read_only = False

            if (
                pending_failed_test
                and _is_successful_progress_mutation(exchange)
            ):
                successful_mutations += 1
                response_had_successful_mutation = True
                awaiting_mutation = False

            tracked_result = _tracked_pytest_result(exchange)
            if tracked_result is not None:
                completed_pytest_seen = True
            if tracked_result is True:
                pending_failed_test = False
                awaiting_mutation = False
            elif tracked_result is False:
                pending_failed_test = True
                awaiting_mutation = True

        if (
            awaiting_at_response_start
            and response_was_read_only
            and not response_had_successful_mutation
        ):
            read_only_responses += 1

    return {
        "run4_read_only_responses_after_failed_test": read_only_responses,
        "run4_progress_guard_interventions": interventions,
        "run4_pretest_guard_interventions": pretest_interventions,
        "run4_successful_mutations_after_failed_test": successful_mutations,
        "run4_pending_failed_test_at_run_end": pending_failed_test,
    }


def _count_run4_pytest_reruns_without_intervening_mutation(
    exchanges: list[ToolExchange],
    workspace: Path | None,
    *,
    initial_pending_command: str | None = None,
) -> int:
    pending_failed_message_index: int | None = (
        -1 if initial_pending_command is not None else None
    )
    reruns_without_mutation = 0

    for exchange in sorted(
        exchanges,
        key=lambda item: item.sequence_index,
    ):
        if exchange.run_index != 4:
            continue

        if (
            pending_failed_message_index is not None
            and exchange.name in {"edit_file", "write_file"}
            and exchange.message_index > pending_failed_message_index
            and _canonical_tool_path(
                exchange.arguments.get("path"), workspace
            ).startswith(PRODUCTION_PREFIX)
            and len(exchange.results) == 1
            and not _is_execution_error(exchange.results[0])
            and not exchange.results[0].strip().casefold().startswith(
                "no changes made"
            )
        ):
            pending_failed_message_index = None
            continue

        if (
            exchange.name != "run_command"
            or not _is_challenge_pytest_command(
                exchange.arguments.get("command")
            )
        ):
            continue

        if pending_failed_message_index is not None:
            reruns_without_mutation += 1

        if len(exchange.results) != 1:
            continue
        payload = _parse_json_object(exchange.results[0])
        if payload is None or payload.get("timed_out") is not False:
            continue
        exit_code = payload.get("exit_code")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            continue
        if exit_code == 0:
            pending_failed_message_index = None
        else:
            pending_failed_message_index = exchange.message_index

    return reruns_without_mutation


def _is_complete_read_result(content: str) -> bool:
    normalized = content.replace("\r\n", "\n")
    header, separator, _ = normalized.partition("\n\n")
    lines = header.splitlines()
    if separator != "\n\n" or not lines or lines[0] != "[read_file]":
        return False
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition(":")
        if not separator:
            return False
        metadata[key.strip()] = value.strip().casefold()
    return bool(
        metadata.get("truncated_before") == "false"
        and metadata.get("truncated_after") == "false"
        and metadata.get("char_truncated") == "false"
        and metadata.get("partial_line") == "false"
        and metadata.get("next_start_line") == "none"
    )


def _history_for_run(
    history: list[dict[str, Any]],
    records: list[RunRecord],
    run_index: int,
) -> list[dict[str, Any]]:
    record = next(
        (item for item in records if item.index == run_index),
        None,
    )
    if record is None:
        return []
    return history[record.history_start : record.history_end]


def _run4_action_metrics(
    history: list[dict[str, Any]],
    records: list[RunRecord],
    exchanges: list[ToolExchange],
    workspace: Path | None,
    run_specs: tuple[RunSpec, ...],
) -> dict[str, int | bool | None]:
    run4_exchanges = [
        exchange for exchange in exchanges if exchange.run_index == 4
    ]
    assistant_response_steps = {
        message_index: step
        for step, message_index in enumerate(
            (
                index
                for index, message in enumerate(history)
                if message.get("role") == "assistant"
                and _run_index_for_message(index, records) == 4
            ),
            start=1,
        )
    }
    run4_guard = next(
        (
            spec.progress_guard
            for spec in run_specs
            if spec.index == 4
        ),
        None,
    )
    initial_pending_command = (
        run4_guard.initial_pending_command
        if run4_guard is not None
        else None
    )
    initial_diagnosis_responses = (
        run4_guard.initial_diagnosis_responses
        if initial_pending_command is not None
        else 0
    )
    mutations = [
        exchange
        for exchange in run4_exchanges
        if exchange.name in {"edit_file", "write_file"}
    ]
    first_mutation = min(
        mutations,
        key=lambda exchange: exchange.sequence_index,
        default=None,
    )
    mutation_sequence = (
        first_mutation.sequence_index if first_mutation is not None else None
    )
    first_successful_mutation = min(
        (
            exchange
            for exchange in run4_exchanges
            if _is_successful_progress_mutation(exchange)
        ),
        key=lambda exchange: exchange.sequence_index,
        default=None,
    )
    responses_before_successful_mutation = (
        assistant_response_steps.get(
            first_successful_mutation.message_index,
            len(assistant_response_steps) + 1,
        )
        - 1
        if first_successful_mutation is not None
        else len(assistant_response_steps)
    )
    before_mutation = [
        exchange
        for exchange in run4_exchanges
        if mutation_sequence is None
        or exchange.sequence_index < mutation_sequence
    ]
    complete_reads = Counter(
        _canonical_tool_path(exchange.arguments.get("path"), workspace)
        for exchange in before_mutation
        if exchange.name == "read_file"
        and len(exchange.results) == 1
        and _is_complete_read_result(exchange.results[0])
    )
    return {
        "run4_first_mutation_step": (
            assistant_response_steps.get(first_mutation.message_index)
            if first_mutation is not None
            else None
        ),
        "run4_edit_file_calls": sum(
            exchange.name == "edit_file" for exchange in run4_exchanges
        ),
        "run4_write_file_calls": sum(
            exchange.name == "write_file" for exchange in run4_exchanges
        ),
        "run4_reads_before_first_mutation": sum(
            exchange.name == "read_file" for exchange in before_mutation
        ),
        "run4_duplicate_complete_reads_before_first_mutation": sum(
            count - 1 for count in complete_reads.values() if count > 1
        ),
        "run4_pytest_reruns_without_intervening_mutation": (
            _count_run4_pytest_reruns_without_intervening_mutation(
                run4_exchanges,
                workspace,
                initial_pending_command=initial_pending_command,
            )
        ),
        "run4_started_with_pending_failed_test": (
            initial_pending_command is not None
        ),
        "run4_initial_pending_command_present": (
            initial_pending_command is not None
        ),
        "run4_initial_diagnosis_responses_used": (
            min(
                initial_diagnosis_responses,
                responses_before_successful_mutation,
            )
        ),
        "run4_migration_reads": sum(
            exchange.name == "read_file"
            and _canonical_tool_path(
                exchange.arguments.get("path"), workspace
            )
            == "data/migration_notes.txt"
            for exchange in run4_exchanges
        ),
        "run4_repository_scan_calls": sum(
            exchange.name == "list_directory"
            for exchange in run4_exchanges
        ),
        **_run4_progress_metrics(
            run4_exchanges,
            initial_pending_command=initial_pending_command,
        ),
    }


def analyze_history(
    history: list[dict[str, Any]],
    records: list[RunRecord],
    tokens: ChallengeTokens,
    run_specs: tuple[RunSpec, ...],
    request_inputs: list[dict[str, Any]],
    workspace: Path | None = None,
) -> dict[str, Any]:
    exchanges = collect_tool_exchanges(history, records)
    run4_action = _run4_action_metrics(
        history, records, exchanges, workspace, run_specs
    )
    tool_counts = Counter(exchange.name for exchange in exchanges)
    attempted_read_paths = {
        _canonical_tool_path(exchange.arguments.get("path"), workspace)
        for exchange in exchanges
        if exchange.name == "read_file"
    }
    read_paths = {
        _canonical_tool_path(exchange.arguments.get("path"), workspace)
        for exchange in exchanges
        if exchange.name == "read_file"
        and len(exchange.results) == 1
        and exchange.results[0].replace("\r\n", "\n").startswith(
            "[read_file]\n"
        )
    }
    call_ids = [exchange.call_id for exchange in exchanges]
    result_positions: dict[str, list[int]] = {}
    result_ids: list[str] = []
    for message_index, message in enumerate(history):
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        result_ids.append(call_id)
        result_positions.setdefault(call_id, []).append(message_index)
    result_complete = bool(exchanges) and (
        all(call_ids)
        and len(set(call_ids)) == len(call_ids)
        and Counter(call_ids) == Counter(result_ids)
        and all(
            len(result_positions.get(exchange.call_id, ())) == 1
            and result_positions[exchange.call_id][0] > exchange.message_index
            for exchange in exchanges
        )
    )

    def matching_error(
        path: str,
        error_type: str,
        tool_names: set[str],
    ) -> list[ToolExchange]:
        return [
            exchange
            for exchange in exchanges
            if exchange.name in tool_names
            and exchange.run_index == 2
            and _canonical_tool_path(
                exchange.arguments.get("path"), workspace
            )
            == path
            and len(exchange.results) == 1
            and (
                (payload := _parse_json_object(exchange.results[0]))
                is not None
            )
            and payload.get("ok") is False
            and payload.get("tool") == exchange.name
            and payload.get("error_type") == error_type
            and path.casefold()
            in str(payload.get("message", "")).replace("\\", "/").casefold()
        ]

    missing_errors = matching_error(
        "config/legacy_policy.json",
        "FileNotFoundError",
        {"read_file"},
    )
    boundary_errors = matching_error(
        "../shared/global_defaults.json",
        "WorkspaceBoundaryError",
        {"read_file"},
    )

    def recovered_after(errors: list[ToolExchange]) -> bool:
        if not errors:
            return False
        first_error_index = errors[0].message_index
        return any(
            exchange.message_index > first_error_index
            and exchange.run_index == errors[0].run_index
            and len(exchange.results) == 1
            and not _is_execution_error(exchange.results[0])
            for exchange in exchanges
        )

    def feedback_sent_to_llm(errors: list[ToolExchange]) -> bool:
        if not errors:
            return False
        error_results = {error.results[0] for error in errors}
        return any(
            any(
                message.get("role") == "tool"
                and message.get("content") in error_results
                for message in request.get("messages", ())
                if isinstance(message, dict)
            )
            for request in request_inputs
        )

    missing_feedback_sent = feedback_sent_to_llm(missing_errors)
    boundary_feedback_sent = feedback_sent_to_llm(boundary_errors)

    run_command_payloads: list[tuple[ToolExchange, dict[str, Any]]] = []
    for exchange in exchanges:
        if exchange.name != "run_command" or len(exchange.results) != 1:
            continue
        payload = _parse_json_object(exchange.results[0])
        if payload is not None and "exit_code" in payload:
            run_command_payloads.append((exchange, payload))

    baseline_payloads = [
        (exchange, payload)
        for exchange, payload in run_command_payloads
        if exchange.run_index == 3
        and _is_challenge_pytest_command(exchange.arguments.get("command"))
    ]
    baseline_nonzero = len(baseline_payloads) == 1 and all(
        exchange.run_index == 3
        and _tracked_pytest_result(exchange) is False
        for exchange, _ in baseline_payloads
    )
    run4_pytest_calls = sum(
        exchange.run_index == 4
        and _is_challenge_pytest_command(exchange.arguments.get("command"))
        for exchange, _ in run_command_payloads
    )
    diagnostic_payloads = [
        (exchange, payload)
        for exchange, payload in run_command_payloads
        if _is_diagnose_command(exchange.arguments.get("command"))
    ]

    def diagnostic_execution_succeeded(
        exchange: ToolExchange,
        payload: dict[str, Any],
        *,
        run_index: int,
    ) -> bool:
        return bool(
            exchange.run_index == run_index
            and payload.get("exit_code") == 0
            and payload.get("timed_out") is False
        )

    run3_diagnostic_payloads = [
        (exchange, payload)
        for exchange, payload in diagnostic_payloads
        if exchange.run_index == 3
    ]
    run4_diagnostic_payloads = [
        (exchange, payload)
        for exchange, payload in diagnostic_payloads
        if exchange.run_index == 4
    ]
    run3_diagnostic = (
        run3_diagnostic_payloads[0]
        if len(run3_diagnostic_payloads) == 1
        else None
    )
    run3_diagnostic_succeeded = bool(
        run3_diagnostic is not None
        and diagnostic_execution_succeeded(
            *run3_diagnostic,
            run_index=3,
        )
    )
    run3_diagnostic_stdout = (
        str(run3_diagnostic[1].get("stdout", ""))
        if run3_diagnostic is not None
        else ""
    )
    run3_original_chars = (
        run3_diagnostic[1].get("stdout_original_chars")
        if run3_diagnostic is not None
        else None
    )
    shell_truncated = bool(
        run3_diagnostic_succeeded
        and run3_diagnostic[1].get("stdout_truncated") is True
        and isinstance(run3_original_chars, int)
        and run3_original_chars > DEFAULT_SHELL_MAX_OUTPUT_CHARS
    )
    diag_head_retained = bool(
        run3_diagnostic_succeeded
        and tokens.diag_head_token in run3_diagnostic_stdout
    )
    diag_tail_retained = bool(
        run3_diagnostic_succeeded
        and tokens.diag_tail_token in run3_diagnostic_stdout
    )
    diag_middle_removed = bool(
        run3_diagnostic_succeeded
        and DIAG_MIDDLE_MARKER not in run3_diagnostic_stdout
    )
    diagnostic_blocked = bool(
        run3_diagnostic_succeeded
        and "DIAG_STATE=BLOCKED" in run3_diagnostic_stdout
    )
    diagnostic_ready = any(
        diagnostic_execution_succeeded(
            exchange, payload, run_index=4
        )
        and "DIAG_STATE=READY" in str(payload.get("stdout", ""))
        for exchange, payload in run4_diagnostic_payloads
    )
    diagnostic_original_chars = (
        int(run3_original_chars)
        if isinstance(run3_original_chars, int)
        else 0
    )
    run_commands = [
        (
            exchange,
            str(exchange.arguments.get("command", ""))
            .replace("\\", "/")
            .casefold(),
        )
        for exchange in exchanges
        if exchange.name == "run_command"
    ]
    run3_unexpected_run_commands = sum(
        exchange.run_index == 3
        and not _is_challenge_pytest_command(
            exchange.arguments.get("command")
        )
        and not _is_diagnose_command(exchange.arguments.get("command"))
        for exchange, _ in run_commands
    )
    migration_shell_search_violation = any(
        "migration_notes.txt" in command for _, command in run_commands
    )
    run4_migration_shell_search_violation = any(
        exchange.run_index == 4 and "migration_notes.txt" in command
        for exchange, command in run_commands
    )
    workspace_boundary_bypass_attempted = any(
        "../shared/global_defaults.json" in command
        or "../shared" in command
        for _, command in run_commands
    )
    diagnostic_shell_read_violation = any(
        "scripts/diagnose.py" in command
        and not _is_diagnose_command(exchange.arguments.get("command"))
        for exchange, command in run_commands
    )

    run2_read_exchanges = [
        exchange
        for exchange in exchanges
        if exchange.run_index == 2 and exchange.name == "read_file"
    ]
    run2_read_paths = [
        _canonical_tool_path(exchange.arguments.get("path"), workspace)
        for exchange in run2_read_exchanges
    ]
    run2_test_reads = sum(
        path == "tests" or path.startswith("tests/")
        for path in run2_read_paths
    )
    run2_test_reads += sum(
        exchange.run_index == 2
        and exchange.name == "list_directory"
        and (
            (path := _canonical_tool_path(
                exchange.arguments.get("path"), workspace
            ))
            == "tests"
            or path.startswith("tests/")
        )
        for exchange in exchanges
    )
    complete_run2_reads = Counter(
        _canonical_tool_path(exchange.arguments.get("path"), workspace)
        for exchange in run2_read_exchanges
        if len(exchange.results) == 1
        and _is_complete_read_result(exchange.results[0])
    )
    run2_duplicate_complete_reads = sum(
        count - 1 for count in complete_run2_reads.values() if count > 1
    )

    migration_calls_by_message: Counter[int] = Counter(
        exchange.message_index
        for exchange in exchanges
        if exchange.run_index == 3
        and exchange.name == "read_file"
        and _canonical_tool_path(exchange.arguments.get("path"), workspace)
        == "data/migration_notes.txt"
    )
    one_migration_read_per_response = all(
        count <= 1 for count in migration_calls_by_message.values()
    )
    migration_exchanges = [
        exchange
        for exchange in exchanges
        if exchange.name == "read_file"
        and _canonical_tool_path(exchange.arguments.get("path"), workspace)
        == "data/migration_notes.txt"
    ]
    serialized_history = json.dumps(
        history,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    raw_history_preserved = (
        all(
            any(
                message.get("role") == "user"
                and message.get("content") == spec.prompt
                for message in history
            )
            for spec in run_specs
        )
        and tokens.release_token in serialized_history
        and tokens.migration_key in serialized_history
    )
    synthetic_absent = (
        PRIOR_CONTEXT_HEADER not in serialized_history
        and CURRENT_RUN_HEADER not in serialized_history
    )
    tracebacks_absent = all(
        "traceback" not in result.lower()
        for exchange in exchanges
        for result in exchange.results
        if _is_execution_error(result)
    )
    diagnose_read_attempted = any(
        path == "scripts/diagnose.py"
        or path.endswith("/scripts/diagnose.py")
        for path in attempted_read_paths
    )
    return {
        "exchanges": exchanges,
        "tool_call_counts": dict(sorted(tool_counts.items())),
        "tool_calls": len(exchanges),
        "tool_call_results_complete": result_complete,
        "distinct_files_read": len(read_paths),
        "distinct_file_paths_read": sorted(read_paths),
        "missing_legacy_error_observed": bool(missing_errors),
        "missing_legacy_error_returned_to_llm": missing_feedback_sent,
        "missing_legacy_error_recovered": (
            missing_feedback_sent and recovered_after(missing_errors)
        ),
        "workspace_boundary_error_observed": bool(boundary_errors),
        "workspace_boundary_error_returned_to_llm": boundary_feedback_sent,
        "workspace_boundary_error_recovered": (
            boundary_feedback_sent and recovered_after(boundary_errors)
        ),
        "traceback_not_exposed": tracebacks_absent,
        "baseline_nonzero_was_normal_tool_result": baseline_nonzero,
        "baseline_pytest_calls": len(baseline_payloads),
        "run4_pytest_calls": run4_pytest_calls,
        "shell_truncation_observed": shell_truncated,
        "diag_head_retained": diag_head_retained,
        "diag_tail_retained": diag_tail_retained,
        "diag_middle_removed": diag_middle_removed,
        "diagnostic_blocked_observed": diagnostic_blocked,
        "diagnostic_ready_observed": diagnostic_ready,
        "diagnostic_stdout_original_chars": diagnostic_original_chars,
        "diagnose_script_not_read": (
            not diagnose_read_attempted
            and not diagnostic_shell_read_violation
        ),
        "migration_shell_search_violation": migration_shell_search_violation,
        "run4_migration_shell_search_violation": (
            run4_migration_shell_search_violation
        ),
        "workspace_boundary_bypass_attempted": (
            workspace_boundary_bypass_attempted
        ),
        "run3_verification_calls": sum(
            exchange.name == "verify_workspace" and exchange.run_index == 3
            for exchange in exchanges
        ),
        "run4_verification_calls": sum(
            exchange.name == "verify_workspace" and exchange.run_index == 4
            for exchange in exchanges
        ),
        "run3_diagnose_calls": len(run3_diagnostic_payloads),
        "run4_diagnose_calls": len(run4_diagnostic_payloads),
        "run3_unexpected_run_commands": run3_unexpected_run_commands,
        "run2_run_command_calls": sum(
            exchange.run_index == 2 and exchange.name == "run_command"
            for exchange in exchanges
        ),
        "run2_edit_file_calls": sum(
            exchange.run_index == 2 and exchange.name == "edit_file"
            for exchange in exchanges
        ),
        "run2_write_file_calls": sum(
            exchange.run_index == 2 and exchange.name == "write_file"
            for exchange in exchanges
        ),
        "run2_verification_calls": sum(
            exchange.run_index == 2 and exchange.name == "verify_workspace"
            for exchange in exchanges
        ),
        "run2_migration_reads": sum(
            path == "data/migration_notes.txt" for path in run2_read_paths
        ),
        "run2_diagnose_reads": sum(
            path == "scripts/diagnose.py" for path in run2_read_paths
        ),
        "run2_test_reads": run2_test_reads,
        "run2_duplicate_complete_reads": run2_duplicate_complete_reads,
        "migration_reads_outside_run3": sum(
            exchange.run_index != 3 for exchange in migration_exchanges
        ),
        "run3_migration_reads": sum(
            exchange.run_index == 3 for exchange in migration_exchanges
        ),
        "run1_tool_calls": sum(
            exchange.run_index == 1 for exchange in exchanges
        ),
        "pre_repair_edit_or_write_calls": sum(
            exchange.run_index in {2, 3}
            and exchange.name in {"edit_file", "write_file"}
            for exchange in exchanges
        ),
        "one_migration_read_per_response": one_migration_read_per_response,
        "full_raw_history_preserved": raw_history_preserved,
        "synthetic_compaction_absent_from_full_history": synthetic_absent,
        **run4_action,
    }


def assert_prompt_boundaries(
    specs: tuple[RunSpec, ...],
    tokens: ChallengeTokens,
) -> None:
    prompts = [spec.prompt for spec in specs]
    require(tokens.release_token in prompts[0], "Run 1 lost RELEASE_TOKEN.")
    for prompt in prompts[1:]:
        require(
            tokens.release_token not in prompt,
            "RELEASE_TOKEN leaked into a later Run prompt.",
        )
    for value in (
        tokens.migration_key,
        tokens.diag_head_token,
        tokens.diag_tail_token,
    ):
        require(
            all(value not in prompt for prompt in prompts),
            "A runtime evidence token leaked into a Run prompt.",
        )


def assert_preflight(
    *,
    workspace: Path,
    fixture: Any,
    initial_snapshot: dict[str, str],
    specs: tuple[RunSpec, ...],
    tokens: ChallengeTokens,
    agent: Any,
    recording: RequestRecordingLLM,
    verification_command: str,
) -> None:
    assert_prompt_boundaries(specs, tokens)
    token_values = (
        tokens.release_token,
        tokens.migration_key,
        tokens.diag_head_token,
        tokens.diag_tail_token,
    )
    require(
        all(token_values) and len(set(token_values)) == len(token_values),
        "Runtime evidence tokens must be non-empty and unique.",
    )
    require(
        not REPORT_PATH.resolve(strict=False).is_relative_to(
            workspace.resolve(strict=False)
        ),
        "Challenge report path must be outside the temporary workspace.",
    )
    actual_files = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    require(len(actual_files) >= 18, "Fixture contains fewer than 18 files.")
    require(
        "incident/report.py" not in initial_snapshot,
        "incident/report.py must be absent initially.",
    )
    require(
        set(fixture.protected_files) <= actual_files,
        "A protected fixture file is missing.",
    )
    migration_lines = fixture.files["data/migration_notes.txt"].splitlines()
    require(len(migration_lines) >= 1_800, "Migration fixture is too short.")
    required_migration_entries = {
        200: (
            "DIRECTIVE_DEDUPE_POLICY="
            "highest-severity-then-latest-timestamp"
        ),
        400: "DIRECTIVE_OUTPUT_ORDER=timestamp-then-incident-id",
        800: "DIRECTIVE_UNKNOWN_SEVERITY=reject",
        1200: "DIRECTIVE_REPORT_SCHEMA_VERSION=2",
        1600: f"MIGRATION_KEY={tokens.migration_key}",
    }
    require(
        all(
            migration_lines[index] == value
            for index, value in required_migration_entries.items()
        ),
        "Migration directives are missing or misplaced.",
    )
    require(
        (len(migration_lines) + DEFAULT_READ_MAX_LINES - 1)
        // DEFAULT_READ_MAX_LINES
        >= 8,
        "Migration fixture cannot naturally produce eight read pages.",
    )

    signature = inspect.signature(agent.run)
    policy = signature.parameters.get("require_verified_completion")
    require(
        policy is not None
        and policy.kind is inspect.Parameter.KEYWORD_ONLY
        and policy.default is True,
        "Agent.run per-run verification policy is unavailable.",
    )
    progress_policy = signature.parameters.get("progress_guard")
    require(
        progress_policy is not None
        and progress_policy.kind is inspect.Parameter.KEYWORD_ONLY
        and progress_policy.default is None,
        "Agent.run opt-in Progress Guard policy is unavailable.",
    )
    require(
        all(spec.progress_guard is None for spec in specs[:3]),
        "Progress Guard must remain disabled for Runs 1-3.",
    )
    run4_progress_guard = specs[3].progress_guard
    require(
        run4_progress_guard is not None
        and run4_progress_guard.tracked_commands
        == (CHALLENGE_PYTEST_COMMAND,)
        and run4_progress_guard.mutation_tool_names
        == ("edit_file", "write_file")
        and run4_progress_guard.diagnosis_responses == 2
        and run4_progress_guard.initial_pending_command is None
        and run4_progress_guard.initial_diagnosis_responses == 0,
        "Run 4 Progress Guard configuration is unexpected.",
    )
    require(
        agent.tool_registry.names() == EXPECTED_TOOL_NAMES,
        "Challenge did not receive the production Tool Registry.",
    )
    require(
        agent.verification_tool_name == "verify_workspace",
        "Completion Gate is not configured.",
    )
    require(
        _is_challenge_pytest_command(verification_command),
        "Trusted verification does not use the fixed Challenge pytest command.",
    )
    require(
        agent.max_steps == CHALLENGE_MAX_STEPS,
        "Unexpected Challenge max_steps value.",
    )
    require(
        agent.max_context_chars == MAX_CONTEXT_CHARS
        and agent.compaction_trigger_chars == 18_000
        and agent.max_compaction_chars == 6_000,
        "Unexpected production context compaction configuration.",
    )
    read_tool = agent.tool_registry.get("read_file")
    shell_tool = agent.tool_registry.get("run_command")
    require(
        isinstance(read_tool, ReadFileTool)
        and read_tool.default_max_lines == DEFAULT_READ_MAX_LINES
        and read_tool.max_output_chars == DEFAULT_READ_MAX_OUTPUT_CHARS,
        "Unexpected production read budget.",
    )
    require(
        isinstance(shell_tool, RunCommandTool)
        and shell_tool.max_output_chars == DEFAULT_SHELL_MAX_OUTPUT_CHARS,
        "Unexpected production shell budget.",
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
        "verify_workspace exposes Host configuration to the model.",
    )
    for spec in specs:
        mandatory = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": spec.prompt},
        ]
        require(
            estimate_messages_size(mandatory) <= MAX_CONTEXT_CHARS,
            f"Run {spec.index} mandatory context exceeds the hard limit.",
        )
    require(not recording.calls, "Preflight unexpectedly called the real LLM.")

    visible_sources = json.dumps(
        {
            "files": fixture.files,
            "prompts": [spec.prompt for spec in specs],
            "system_prompt": agent.system_prompt,
            "tools": agent.tool_registry.schemas(),
            "verification_command": verification_command,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    for hidden_code in HIDDEN_CHECKS:
        require(
            hidden_code not in visible_sources,
            "Hidden evaluator leaked into Agent-visible inputs.",
        )
    hidden_files = [
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
        and path.name.casefold() in FORBIDDEN_HIDDEN_FILE_NAMES
    ]
    require(not hidden_files, f"Hidden files were materialized: {hidden_files}")


def run_agent_stage(
    agent: Any,
    recording: RequestRecordingLLM,
    spec: RunSpec,
    record: RunRecord,
    workspace: Path,
) -> str:
    before = agent.history
    workspace_before = snapshot_hash_manifest(workspace)
    record.history_start = len(before)
    record.workspace_revision_before = agent.workspace_revision
    record.verified_revision_before = agent.verified_revision
    record.verification_required_before = agent.verification_required
    calls_before = len(recording.metrics)
    recording.set_run_index(spec.index)
    started = time.perf_counter()
    try:
        answer = agent.run(
            spec.prompt,
            require_verified_completion=spec.require_verified_completion,
            progress_guard=spec.progress_guard,
        )
        record.completed = True
    finally:
        record.duration_seconds = time.perf_counter() - started
        record.steps = len(recording.metrics) - calls_before
        record.history_end = len(agent.history)
        record.workspace_revision_after = agent.workspace_revision
        record.verified_revision_after = agent.verified_revision
        record.verification_required_after = agent.verification_required
        workspace_after = snapshot_hash_manifest(workspace)
        record.workspace_changes = tuple(
            sorted(
                path
                for path in set(workspace_before) | set(workspace_after)
                if workspace_before.get(path) != workspace_after.get(path)
            )
        )
    after = agent.history
    require(
        after[: len(before)] == before,
        f"Run {spec.index} did not preserve the Full History prefix.",
    )
    return answer


def preserve_initial_evidence(
    workspace: Path,
    result: EvidenceResult,
) -> FileManifest:
    if result.paths is None:
        raise ValueError("Evidence session paths are unavailable.")
    started = time.perf_counter()
    try:
        preserve_workspace_snapshot(
            workspace,
            result.paths.initial_workspace,
        )
        result.initial_snapshot_created = True
        manifest = build_file_manifest(result.paths.initial_workspace)
        write_manifest(result.paths.before_manifest, manifest)
        result.before_manifest_created = True
        result.snapshot_stage = "initial"
        return manifest
    finally:
        result.preservation_seconds += time.perf_counter() - started


def preserve_final_evidence(
    workspace: Path,
    result: EvidenceResult,
    before_manifest: FileManifest,
    protected_files: tuple[str, ...],
    *,
    agent_completed: bool,
    snapshot_stage: str,
) -> FileManifest:
    if result.paths is None:
        raise ValueError("Evidence session paths are unavailable.")
    result.agent_completed = agent_completed
    result.snapshot_stage = snapshot_stage
    started = time.perf_counter()
    try:
        preserve_workspace_snapshot(
            workspace,
            result.paths.final_workspace,
        )
        result.final_snapshot_created = True
        manifest = build_file_manifest(result.paths.final_workspace)
        write_manifest(result.paths.after_manifest, manifest)
        result.after_manifest_created = True
        result.changes = compare_manifests(
            before_manifest,
            manifest,
            protected_files=protected_files,
        )
        write_workspace_changes(result.paths.changes_json, result.changes)
        result.changes_json_created = True
        diff = generate_workspace_diff(
            result.paths.initial_workspace,
            result.paths.final_workspace,
            result.changes,
        )
        write_workspace_diff(result.paths.diff_file, diff)
        result.diff_created = True
        return manifest
    finally:
        result.preservation_seconds += time.perf_counter() - started


def _record_evidence_error(
    result: EvidenceResult,
    error: Exception,
    *,
    temporary_root: Path | None,
    tokens: ChallengeTokens,
) -> None:
    message = sanitize_error(
        error,
        temporary_root=temporary_root,
        tokens=tokens,
    )
    repository_text = str(REPO_ROOT)
    for variant in {
        repository_text,
        repository_text.replace("\\", "/"),
    }:
        message = message.replace(variant, "<repository>")
    result.record_error(
        error,
        message=message,
    )


def _evidence_artifacts_preserved(result: EvidenceResult) -> bool:
    if result.paths is None:
        return False
    return bool(
        result.initial_snapshot_created
        and result.paths.initial_workspace.is_dir()
        and result.final_snapshot_created
        and result.paths.final_workspace.is_dir()
        and result.before_manifest_created
        and result.paths.before_manifest.is_file()
        and result.after_manifest_created
        and result.paths.after_manifest.is_file()
        and result.changes_json_created
        and result.paths.changes_json.is_file()
        and result.diff_created
        and result.paths.diff_file.is_file()
    )


def _evidence_failed_requirements(result: EvidenceResult) -> list[str]:
    requirements = (
        ("initial_snapshot_created", result.initial_snapshot_created),
        ("final_snapshot_created", result.final_snapshot_created),
        ("before_manifest_created", result.before_manifest_created),
        ("after_manifest_created", result.after_manifest_created),
        ("changes_json_created", result.changes_json_created),
        ("diff_created", result.diff_created),
        (
            "snapshot_preserved_after_cleanup",
            result.snapshot_preserved_after_cleanup,
        ),
        ("no_evidence_error", result.error_type is None),
    )
    return [name for name, passed in requirements if not passed]


def _zero_coverage_metrics() -> LongRunningMetrics:
    return LongRunningMetrics(
        agent_runs=0,
        llm_calls=0,
        tool_calls=0,
        distinct_files_read=0,
        production_files_changed=0,
        files_created=0,
        pagination_reads=0,
        compaction_bearing_requests=0,
        controlled_errors_observed=0,
        controlled_errors_recovered=0,
        shell_truncation_observed=False,
        verification_calls=0,
    )


def _zero_functional_metrics() -> FunctionalChallengeMetrics:
    return FunctionalChallengeMetrics(
        agent_runs_completed=False,
        initial_visible_tests_failed=False,
        missing_legacy_error_observed=False,
        workspace_boundary_error_observed=False,
        controlled_errors_recovered=0,
        migration_pagination_valid=False,
        migration_key_seen_in_tool_result=False,
        baseline_nonzero_was_normal_tool_result=False,
        shell_truncation_observed=False,
        diag_head_retained=False,
        diag_tail_retained=False,
        diag_middle_removed=False,
        diagnose_script_not_read=False,
        post_agent_visible_recheck_passed=False,
        hidden_checks_passed=False,
        protected_files_unchanged=False,
        verification_calls=0,
        verification_state_clean=False,
        release_token_recovered=False,
        migration_key_recovered=False,
        diag_tail_token_recovered=False,
        report_module_created=False,
    )


def build_failure_report(
    *,
    model: str,
    started_at: datetime,
    started_perf: float,
    error: str,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started_perf
    evidence_result = EvidenceResult()
    report = build_report_payload(
        model=model,
        elapsed_seconds=elapsed,
        coverage_metrics=_zero_coverage_metrics(),
        functional_metrics=_zero_functional_metrics(),
        llm_call_metrics=(),
        error=error,
    )
    report.update(
        {
            **{field: None for field in TRI_STATE_OBJECTIVE_FIELDS},
            "release_token_present_in_final_llm_request": False,
            "migration_key_present_in_final_llm_request": False,
            "diag_tail_token_present_in_final_llm_request": False,
            "run4_met_first_mutation_deadline": None,
            "run4_did_not_repeat_complete_reads_before_first_mutation": None,
            "run4_pytest_reruns_without_intervening_mutation": 0,
            "run4_read_only_responses_after_failed_test": 0,
            "run4_progress_guard_interventions": 0,
            "run4_pretest_guard_interventions": 0,
            "run4_successful_mutations_after_failed_test": 0,
            "run4_pending_failed_test_at_run_end": None,
            "run4_started_with_pending_failed_test": None,
            "run4_initial_pending_command_present": None,
            "run4_initial_diagnosis_responses_used": 0,
            "run4_did_not_rerun_failed_pytest_without_intervening_mutation": (
                None
            ),
            "run4_repair_loop_discipline_met": None,
            "action_discipline_target_met": None,
            "action_discipline_warnings": [],
            "not_evaluated_objectives": sorted(TRI_STATE_OBJECTIVE_FIELDS),
            "scenario": SCENARIO_NAME,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "temporary_workspace_cleaned": True,
            "runs": [],
            "production_file_paths_changed": [],
            "file_paths_created": [],
            "file_change_accounting_evaluated": False,
            "evidence": evidence_result.to_report(REPO_ROOT),
            "evidence_preservation_passed": False,
            "evidence_failed_requirements": (
                _evidence_failed_requirements(evidence_result)
            ),
            "real_validation_process_success": False,
        }
    )
    return report


def execute_real_challenge(client: LLMClient) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()
    tokens = build_tokens()
    fixture = build_long_running_fixture(tokens)
    specs = build_run_specs(tokens)
    effective_specs = list(specs)
    temporary_root_path: Path | None = None
    workspace: Path | None = None
    phase = "temporary workspace setup"
    error: str | None = None
    recording: RequestRecordingLLM | None = None
    agent: Any | None = None
    run_records: list[RunRecord] = []
    final_answer = ""
    history: list[dict[str, Any]] = []
    initial_snapshot: dict[str, str] = {}
    initial_visible_failed = False
    initial_visible_evaluated = False
    final_visible_passed = False
    final_visible_evaluated = False
    hidden_passed = False
    hidden_evaluated = False
    hidden_results: list[dict[str, Any]] = []
    host_visible_seconds = 0.0
    hidden_seconds = 0.0
    same_agent = True
    agent_identity: int | None = None
    evidence_result = EvidenceResult()
    before_evidence_manifest: FileManifest = {}
    final_evidence_attempted = False
    pending_tracked_failure: PendingTrackedFailure | None = None

    try:
        with TemporaryDirectory(
            prefix="coding-agent-long-running-"
        ) as temporary_root:
            temporary_root_path = Path(temporary_root)
            workspace = temporary_root_path / "workspace"
            initial_basetemp = temporary_root_path / "initial-visible-pytest"
            final_basetemp = temporary_root_path / "final-visible-pytest"

            phase = "fixture materialization"
            materialize_long_running_fixture(fixture, workspace)

            phase = "initial evidence preservation"
            try:
                evidence_result.paths = create_evidence_session(REPO_ROOT)
                before_evidence_manifest = preserve_initial_evidence(
                    workspace,
                    evidence_result,
                )
            except Exception as evidence_error:
                _record_evidence_error(
                    evidence_result,
                    evidence_error,
                    temporary_root=temporary_root_path,
                    tokens=tokens,
                )
                raise

            initial_snapshot = snapshot_hash_manifest(workspace)

            phase = "initial visible fixture check"
            initial_check = run_visible_tests(workspace, initial_basetemp)
            initial_visible_evaluated = True
            initial_visible_failed = (
                initial_check.exit_code not in (None, 0)
                and initial_check.timed_out is False
            )
            require(
                initial_visible_failed,
                "Initial visible tests must fail without timing out.",
            )

            phase = "diagnostic fixture preflight"
            diagnostic_preflight = run_host_process(
                [sys.executable, "-B", "scripts/diagnose.py"],
                workspace,
                timeout=HOST_TEST_TIMEOUT_SECONDS,
            )
            require(diagnostic_preflight.passed, "Diagnostic preflight failed.")
            require(
                len(diagnostic_preflight.stdout)
                > DEFAULT_SHELL_MAX_OUTPUT_CHARS,
                "Diagnostic stdout does not exceed the production shell budget.",
            )
            for marker in (
                tokens.diag_head_token,
                tokens.diag_tail_token,
                DIAG_MIDDLE_MARKER,
                "DIAG_STATE=BLOCKED",
            ):
                require(
                    marker in diagnostic_preflight.stdout,
                    f"Diagnostic preflight output is missing {marker}.",
                )
            require(
                protected_files_unchanged(
                    fixture,
                    initial_snapshot,
                    snapshot_hash_manifest(workspace),
                ),
                "Host preflight changed a protected fixture file.",
            )

            phase = "production Agent preflight"
            verification_command = make_command(
                sys.executable,
                *CHALLENGE_PYTEST_ARGUMENTS,
            )
            recording = RequestRecordingLLM(client)
            agent = build_agent(
                workspace,
                llm_client=recording,
                max_steps=CHALLENGE_MAX_STEPS,
                max_context_chars=MAX_CONTEXT_CHARS,
                compact_context=True,
                verification_commands=(verification_command,),
            )
            agent_identity = id(agent)
            assert_preflight(
                workspace=workspace,
                fixture=fixture,
                initial_snapshot=initial_snapshot,
                specs=specs,
                tokens=tokens,
                agent=agent,
                recording=recording,
                verification_command=verification_command,
            )

            try:
                for spec in specs:
                    phase = f"Run {spec.index}: {spec.name}"
                    same_agent = same_agent and id(agent) == agent_identity
                    effective_spec = spec
                    if spec.index == 4:
                        require(
                            pending_tracked_failure is not None,
                            "Run 3 did not produce the tracked failure handoff.",
                        )
                        effective_spec = seed_run_progress_guard(
                            spec,
                            pending_tracked_failure,
                            initial_diagnosis_responses=(
                                RUN4_INITIAL_DIAGNOSIS_RESPONSES
                            ),
                        )
                        effective_specs[3] = effective_spec
                    record = RunRecord(
                        index=effective_spec.index,
                        name=effective_spec.name,
                        require_verified_completion=(
                            effective_spec.require_verified_completion
                        ),
                    )
                    run_records.append(record)
                    answer = run_agent_stage(
                        agent,
                        recording,
                        effective_spec,
                        record,
                        workspace,
                    )
                    if effective_spec.index == 3:
                        pending_tracked_failure = (
                            derive_pending_tracked_failure(
                                agent.history,
                                run_records,
                                run_index=3,
                                tracked_commands=(CHALLENGE_PYTEST_COMMAND,),
                            )
                        )
                        require(
                            pending_tracked_failure is not None,
                            "Run 3 baseline did not leave a pending tracked failure.",
                        )
                    if effective_spec.index == 4:
                        final_answer = answer

                phase = "final evidence preservation"
                final_evidence_attempted = True
                try:
                    preserve_final_evidence(
                        workspace,
                        evidence_result,
                        before_evidence_manifest,
                        fixture.protected_files,
                        agent_completed=True,
                        snapshot_stage="agent_final",
                    )
                except Exception as evidence_error:
                    _record_evidence_error(
                        evidence_result,
                        evidence_error,
                        temporary_root=temporary_root_path,
                        tokens=tokens,
                    )
            except Exception:
                failed_phase = phase
                if (
                    evidence_result.before_manifest_created
                    and not final_evidence_attempted
                    and workspace.is_dir()
                ):
                    final_evidence_attempted = True
                    agent_completed = bool(
                        len(run_records) == len(specs)
                        and all(record.completed for record in run_records)
                    )
                    try:
                        preserve_final_evidence(
                            workspace,
                            evidence_result,
                            before_evidence_manifest,
                            fixture.protected_files,
                            agent_completed=agent_completed,
                            snapshot_stage=(
                                "agent_final"
                                if agent_completed
                                else "failure_state"
                            ),
                        )
                    except Exception as evidence_error:
                        _record_evidence_error(
                            evidence_result,
                            evidence_error,
                            temporary_root=temporary_root_path,
                            tokens=tokens,
                        )
                phase = failed_phase
                raise

            phase = "Full History capture"
            history = agent.history
            request_sources = json.dumps(
                [
                    {"messages": call["messages"], "tools": call["tools"]}
                    for call in recording.request_inputs
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
            for hidden_code in HIDDEN_CHECKS:
                require(
                    hidden_code not in request_sources,
                    "Hidden evaluator leaked into a real API request.",
                )

            calls_before_host = len(recording.calls)
            history_before_host = agent.history

            phase = "Host visible re-check"
            host_visible_started = time.perf_counter()
            final_visible_check = run_visible_tests(workspace, final_basetemp)
            host_visible_seconds = time.perf_counter() - host_visible_started
            final_visible_evaluated = True
            final_visible_passed = final_visible_check.passed

            phase = "Host hidden evaluation"
            hidden_passed, hidden_seconds, hidden_results = run_hidden_checks(
                workspace
            )
            hidden_evaluated = True
            require(
                len(recording.calls) == calls_before_host
                and agent.history == history_before_host,
                "Host evaluation changed Agent state or called the LLM.",
            )
    except Exception as caught:
        error = (
            f"{phase}: "
            + sanitize_error(
                caught,
                temporary_root=temporary_root_path,
                tokens=tokens,
            )
        )
        if agent is not None:
            history = agent.history
        if run_records and not run_records[-1].completed:
            run_records[-1].error = error

    temporary_workspace_cleaned = (
        temporary_root_path is not None and not temporary_root_path.exists()
    )
    evidence_result.snapshot_preserved_after_cleanup = (
        _evidence_artifacts_preserved(evidence_result)
    )
    if not temporary_workspace_cleaned:
        cleanup_error = "Temporary workspace cleanup failed."
        error = cleanup_error if error is None else f"{error}; {cleanup_error}"

    metrics = tuple(recording.metrics) if recording is not None else ()
    requests = (
        [call["messages"] for call in recording.request_inputs]
        if recording is not None
        else []
    )
    compaction = analyze_compaction_requests(requests)
    run3_history = _history_for_run(history, run_records, 3)
    pagination = analyze_migration_pagination(
        run3_history,
        migration_key=tokens.migration_key,
    )
    history_analysis = analyze_history(
        history,
        run_records,
        tokens,
        tuple(effective_specs),
        recording.request_inputs if recording is not None else [],
        workspace,
    )

    controlled_observed = sum(
        (
            history_analysis["missing_legacy_error_observed"],
            history_analysis["workspace_boundary_error_observed"],
        )
    )
    controlled_recovered = sum(
        (
            history_analysis["missing_legacy_error_recovered"],
            history_analysis["workspace_boundary_error_recovered"],
        )
    )
    production_changed, created_files = manifest_file_accounting(
        evidence_result.changes
    )
    protected_unchanged = protected_integrity_result(
        evidence_result.changes,
        comparison_evaluated=evidence_result.changes_json_created,
    )
    tool_counts = history_analysis["tool_call_counts"]
    verification_calls = tool_counts.get("verify_workspace", 0)
    verification_state_clean = bool(
        agent is not None
        and not agent.verification_required
        and agent.workspace_revision == agent.verified_revision
    )
    completed_runs = sum(record.completed for record in run_records)
    run2_started = any(record.index == 2 for record in run_records)
    run3_completed = any(
        record.index == 3 and record.completed for record in run_records
    )
    run4_completed = any(
        record.index == 4 and record.completed for record in run_records
    )
    run4_started = any(record.index == 4 for record in run_records)
    migration_evaluated = bool(
        history_analysis["run3_migration_reads"] > 0 or run3_completed
    )
    baseline_evaluated = bool(
        history_analysis["baseline_pytest_calls"] > 0 or run3_completed
    )
    run3_diagnostic_evaluated = bool(
        history_analysis["run3_diagnose_calls"] > 0 or run3_completed
    )
    run4_diagnostic_evaluated = bool(
        history_analysis["run4_diagnose_calls"] > 0 or run4_completed
    )
    final_recovery = exact_final_answer_token_recovery(final_answer, tokens)
    release_token_recovered = final_recovery["release_token_recovered"]
    migration_key_recovered = final_recovery["migration_key_recovered"]
    diag_tail_token_recovered = final_recovery[
        "diag_tail_token_recovered"
    ]
    final_request_presence = final_request_token_presence(
        recording.request_inputs if recording is not None else [],
        tokens,
        run4_completed=run4_completed,
    )
    report_module_created = "incident/report.py" in created_files
    objective_statuses: dict[str, bool | None] = {
        "initial_visible_tests_failed": evaluation_result(
            evaluated=initial_visible_evaluated,
            passed=initial_visible_failed,
        ),
        "missing_legacy_error_observed": evaluation_result(
            evaluated=run2_started,
            passed=history_analysis["missing_legacy_error_observed"],
        ),
        "missing_legacy_error_returned_to_llm": evaluation_result(
            evaluated=run2_started,
            passed=history_analysis["missing_legacy_error_returned_to_llm"],
        ),
        "missing_legacy_error_recovered": evaluation_result(
            evaluated=run2_started,
            passed=history_analysis["missing_legacy_error_recovered"],
        ),
        "workspace_boundary_error_observed": evaluation_result(
            evaluated=run2_started,
            passed=history_analysis["workspace_boundary_error_observed"],
        ),
        "workspace_boundary_error_returned_to_llm": evaluation_result(
            evaluated=run2_started,
            passed=history_analysis[
                "workspace_boundary_error_returned_to_llm"
            ],
        ),
        "workspace_boundary_error_recovered": evaluation_result(
            evaluated=run2_started,
            passed=history_analysis["workspace_boundary_error_recovered"],
        ),
        "sequential_next_start_line": evaluation_result(
            evaluated=migration_evaluated,
            passed=pagination.sequential_next_start_line,
        ),
        "migration_file_exhausted": evaluation_result(
            evaluated=migration_evaluated,
            passed=pagination.exhausted,
        ),
        "migration_key_seen_in_tool_result": evaluation_result(
            evaluated=migration_evaluated,
            passed=pagination.migration_key_seen_in_tool_result,
        ),
        "baseline_nonzero_observed_as_normal_tool_result": evaluation_result(
            evaluated=baseline_evaluated,
            passed=history_analysis[
                "baseline_nonzero_was_normal_tool_result"
            ],
        ),
        "shell_truncation_observed": evaluation_result(
            evaluated=run3_diagnostic_evaluated,
            passed=history_analysis["shell_truncation_observed"],
        ),
        "diag_head_retained": evaluation_result(
            evaluated=run3_diagnostic_evaluated,
            passed=history_analysis["diag_head_retained"],
        ),
        "diag_tail_retained": evaluation_result(
            evaluated=run3_diagnostic_evaluated,
            passed=history_analysis["diag_tail_retained"],
        ),
        "diag_middle_removed": evaluation_result(
            evaluated=run3_diagnostic_evaluated,
            passed=history_analysis["diag_middle_removed"],
        ),
        "diagnostic_blocked_observed": evaluation_result(
            evaluated=run3_diagnostic_evaluated,
            passed=history_analysis["diagnostic_blocked_observed"],
        ),
        "diagnostic_ready_observed": evaluation_result(
            evaluated=run4_diagnostic_evaluated,
            passed=history_analysis["diagnostic_ready_observed"],
        ),
        "diagnose_script_not_read": evaluation_result(
            evaluated=(
                run3_completed
                or not history_analysis["diagnose_script_not_read"]
            ),
            passed=history_analysis["diagnose_script_not_read"],
        ),
        "full_raw_history_preserved": evaluation_result(
            evaluated=run4_started,
            passed=history_analysis["full_raw_history_preserved"],
        ),
        "visible_tests_passed": evaluation_result(
            evaluated=final_visible_evaluated,
            passed=final_visible_passed,
        ),
        "hidden_checks_passed": evaluation_result(
            evaluated=hidden_evaluated,
            passed=hidden_passed,
        ),
        "protected_files_unchanged": protected_unchanged,
        "verification_state_clean": verification_state_result(
            run4_verification_calls=history_analysis[
                "run4_verification_calls"
            ],
            run4_completed=run4_completed,
            verification_state_clean=verification_state_clean,
        ),
        "release_token_recovered": evaluation_result(
            evaluated=run4_completed,
            passed=release_token_recovered,
        ),
        "migration_key_recovered": evaluation_result(
            evaluated=run4_completed,
            passed=migration_key_recovered,
        ),
        "diag_tail_token_recovered": evaluation_result(
            evaluated=run4_completed,
            passed=diag_tail_token_recovered,
        ),
        "report_module_created": evaluation_result(
            evaluated=(
                run4_completed and evidence_result.changes_json_created
            ),
            passed=report_module_created,
        ),
    }

    coverage_metrics = LongRunningMetrics(
        agent_runs=completed_runs,
        llm_calls=len(metrics),
        tool_calls=history_analysis["tool_calls"],
        distinct_files_read=history_analysis["distinct_files_read"],
        production_files_changed=len(production_changed),
        files_created=len(created_files),
        pagination_reads=pagination.reads,
        compaction_bearing_requests=compaction.requests_with_compaction,
        controlled_errors_observed=controlled_observed,
        controlled_errors_recovered=controlled_recovered,
        shell_truncation_observed=history_analysis[
            "shell_truncation_observed"
        ],
        verification_calls=verification_calls,
    )
    migration_valid = bool(
        pagination.reads >= 8
        and pagination.sequential_next_start_line
        and pagination.exhausted
        and history_analysis["one_migration_read_per_response"]
    )
    functional_metrics = FunctionalChallengeMetrics(
        agent_runs_completed=completed_runs == 4,
        initial_visible_tests_failed=initial_visible_failed,
        missing_legacy_error_observed=history_analysis[
            "missing_legacy_error_observed"
        ],
        workspace_boundary_error_observed=history_analysis[
            "workspace_boundary_error_observed"
        ],
        controlled_errors_recovered=controlled_recovered,
        migration_pagination_valid=migration_valid,
        migration_key_seen_in_tool_result=(
            pagination.migration_key_seen_in_tool_result
        ),
        baseline_nonzero_was_normal_tool_result=history_analysis[
            "baseline_nonzero_was_normal_tool_result"
        ],
        shell_truncation_observed=history_analysis[
            "shell_truncation_observed"
        ],
        diag_head_retained=history_analysis["diag_head_retained"],
        diag_tail_retained=history_analysis["diag_tail_retained"],
        diag_middle_removed=history_analysis["diag_middle_removed"],
        diagnose_script_not_read=history_analysis[
            "diagnose_script_not_read"
        ],
        post_agent_visible_recheck_passed=final_visible_passed,
        hidden_checks_passed=hidden_passed,
        protected_files_unchanged=protected_unchanged is True,
        verification_calls=verification_calls,
        verification_state_clean=verification_state_clean,
        release_token_recovered=release_token_recovered,
        migration_key_recovered=migration_key_recovered,
        diag_tail_token_recovered=diag_tail_token_recovered,
        report_module_created=report_module_created,
    )
    coverage = evaluate_long_running_coverage(coverage_metrics)
    functional = evaluate_functional_challenge(functional_metrics)
    context_limit_respected = all(
        metric.request_chars <= MAX_CONTEXT_CHARS for metric in metrics
    )
    run_policies_correct = (
        [record.require_verified_completion for record in run_records]
        == [False, False, False, True]
    )
    dirty_state_carried_into_run_4 = bool(
        len(run_records) == 4
        and run_records[2].completed
        and run_records[2].verification_required_after
        and run_records[3].verification_required_before
        and run_records[3].workspace_revision_before
        == run_records[2].workspace_revision_after
        and run_records[3].verified_revision_before
        == run_records[2].verified_revision_after
    )
    same_agent_instance_across_started_runs = bool(
        len(run_records) == 4 and same_agent
    )
    run4_first_mutation_step = history_analysis[
        "run4_first_mutation_step"
    ]
    run4_duplicate_complete_reads_before_first_mutation = history_analysis[
        "run4_duplicate_complete_reads_before_first_mutation"
    ]
    run4_pytest_reruns_without_intervening_mutation = history_analysis[
        "run4_pytest_reruns_without_intervening_mutation"
    ]
    run4_pending_failed_test_at_run_end = history_analysis[
        "run4_pending_failed_test_at_run_end"
    ]
    extra_functional_requirements = {
        "same_agent_across_four_runs": (
            same_agent_instance_across_started_runs and completed_runs == 4
        ),
        "run_completion_policies_correct": run_policies_correct,
        "run3_dirty_state_carried_into_run4": (
            dirty_state_carried_into_run_4
        ),
        "run3_did_not_force_verification": (
            history_analysis["run3_verification_calls"] == 0
        ),
        "run3_failed_test_handed_to_run4": history_analysis[
            "run4_started_with_pending_failed_test"
        ],
        "run3_used_only_expected_commands": (
            history_analysis["run3_unexpected_run_commands"] == 0
        ),
        "run4_ran_visible_tests": history_analysis["run4_pytest_calls"] >= 1,
        "run4_used_mutating_file_tool": (
            history_analysis["run4_edit_file_calls"]
            + history_analysis["run4_write_file_calls"]
            > 0
        ),
        "run4_did_not_reinspect_migration_notes": (
            history_analysis["run4_migration_reads"] == 0
            and not history_analysis[
                "run4_migration_shell_search_violation"
            ]
        ),
        "run1_used_no_tools": history_analysis["run1_tool_calls"] == 0,
        "run2_used_no_run_command": (
            history_analysis["run2_run_command_calls"] == 0
        ),
        "run2_used_no_mutating_or_verification_tools": all(
            history_analysis[field] == 0
            for field in (
                "run2_edit_file_calls",
                "run2_write_file_calls",
                "run2_verification_calls",
            )
        ),
        "run2_did_not_read_migration_notes": (
            history_analysis["run2_migration_reads"] == 0
        ),
        "run2_did_not_read_diagnostic_source": (
            history_analysis["run2_diagnose_reads"] == 0
        ),
        "run2_did_not_inspect_tests": (
            history_analysis["run2_test_reads"] == 0
        ),
        "run2_did_not_repeat_complete_file_reads": (
            history_analysis["run2_duplicate_complete_reads"] == 0
        ),
        "runs2_and_3_used_no_edit_or_write": (
            history_analysis["pre_repair_edit_or_write_calls"] == 0
        ),
        "runs1_to_3_left_workspace_files_unchanged": bool(
            len(run_records) >= 3
            and all(not record.workspace_changes for record in run_records[:3])
        ),
        "migration_reads_confined_to_run3": (
            history_analysis["migration_reads_outside_run3"] == 0
        ),
        "tool_call_results_complete": history_analysis[
            "tool_call_results_complete"
        ],
        "context_hard_limit_respected": context_limit_respected,
        "full_raw_history_preserved": history_analysis[
            "full_raw_history_preserved"
        ],
        "synthetic_compaction_absent_from_full_history": history_analysis[
            "synthetic_compaction_absent_from_full_history"
        ],
        "traceback_not_exposed": history_analysis["traceback_not_exposed"],
        "diagnostic_blocked_observed": history_analysis[
            "diagnostic_blocked_observed"
        ],
        "diagnostic_ready_observed": history_analysis[
            "diagnostic_ready_observed"
        ],
        "migration_notes_not_shell_searched": not history_analysis[
            "migration_shell_search_violation"
        ],
        "workspace_boundary_not_bypassed": not history_analysis[
            "workspace_boundary_bypass_attempted"
        ],
        "temporary_workspace_cleaned": temporary_workspace_cleaned,
    }
    completion_dimensions = evaluate_completion_dimensions(
        functional=functional,
        extra_functional_requirements=extra_functional_requirements,
        error=error,
        first_mutation_step=run4_first_mutation_step,
        duplicate_complete_reads_before_first_mutation=(
            run4_duplicate_complete_reads_before_first_mutation
        ),
        pytest_calls=history_analysis["run4_pytest_calls"],
        pytest_reruns_without_intervening_mutation=(
            run4_pytest_reruns_without_intervening_mutation
        ),
        pending_failed_test_at_run_end=(
            run4_pending_failed_test_at_run_end
        ),
    )
    functional_passed = completion_dimensions[
        "functional_challenge_passed"
    ]
    coverage_passed = coverage.passed
    integrated_success = final_integrated_success(
        functional_passed=functional_passed,
        coverage_passed=coverage_passed,
    )
    evidence_metadata = evidence_result.to_report(REPO_ROOT)
    evidence_passed = evidence_result.preservation_passed
    process_success = real_validation_process_success(
        final_integrated_success=integrated_success,
        evidence_preservation_passed=evidence_passed,
    )
    elapsed = time.perf_counter() - started_perf
    report = build_report_payload(
        model=client.model,
        elapsed_seconds=elapsed,
        coverage_metrics=coverage_metrics,
        functional_metrics=functional_metrics,
        llm_call_metrics=metrics,
        error=error,
    )
    report.update(
        {
            "scenario": SCENARIO_NAME,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "runtime_target_met": runtime_target_met(elapsed),
            "configured_max_steps_per_run": (
                agent.max_steps if agent is not None else CHALLENGE_MAX_STEPS
            ),
            "run_durations_seconds": [
                record.duration_seconds for record in run_records
            ],
            "runs": [
                {
                    key: value
                    for key, value in asdict(record).items()
                    if key not in {"history_start", "history_end"}
                }
                for record in run_records
            ],
            "same_agent_across_four_runs": extra_functional_requirements[
                "same_agent_across_four_runs"
            ],
            "same_agent_instance_across_started_runs": (
                same_agent_instance_across_started_runs
            ),
            "run_completion_policies_correct": run_policies_correct,
            "run3_dirty_state_carried_into_run4": (
                dirty_state_carried_into_run_4
            ),
            "llm_wall_time_seconds": sum(
                metric.duration_seconds for metric in metrics
            ),
            "cumulative_request_chars": sum(
                metric.request_chars for metric in metrics
            ),
            "maximum_request_chars": max(
                (metric.request_chars for metric in metrics),
                default=0,
            ),
            "configured_max_context_chars": MAX_CONTEXT_CHARS,
            "context_hard_limit_respected": context_limit_respected,
            "tool_call_counts": {
                name: tool_counts.get(name, 0) for name in EXPECTED_TOOL_NAMES
            },
            "tool_call_results_complete": history_analysis[
                "tool_call_results_complete"
            ],
            "distinct_file_paths_read": history_analysis[
                "distinct_file_paths_read"
            ],
            "production_file_paths_changed": production_changed,
            "file_paths_created": created_files,
            "file_change_accounting_evaluated": (
                evidence_result.changes_json_created
            ),
            "sequential_next_start_line": (
                pagination.sequential_next_start_line
            ),
            "migration_file_exhausted": pagination.exhausted,
            "migration_key_seen_in_tool_result": (
                pagination.migration_key_seen_in_tool_result
            ),
            "one_migration_read_per_response": history_analysis[
                "one_migration_read_per_response"
            ],
            "prior_compaction_requests": (
                compaction.prior_compaction_requests
            ),
            "current_run_compaction_requests": (
                compaction.current_run_compaction_requests
            ),
            **final_request_presence,
            "full_raw_history_preserved": history_analysis[
                "full_raw_history_preserved"
            ],
            "synthetic_compaction_absent_from_full_history": history_analysis[
                "synthetic_compaction_absent_from_full_history"
            ],
            "missing_legacy_error_observed": history_analysis[
                "missing_legacy_error_observed"
            ],
            "missing_legacy_error_returned_to_llm": history_analysis[
                "missing_legacy_error_returned_to_llm"
            ],
            "missing_legacy_error_recovered": history_analysis[
                "missing_legacy_error_recovered"
            ],
            "workspace_boundary_error_observed": history_analysis[
                "workspace_boundary_error_observed"
            ],
            "workspace_boundary_error_returned_to_llm": history_analysis[
                "workspace_boundary_error_returned_to_llm"
            ],
            "workspace_boundary_error_recovered": history_analysis[
                "workspace_boundary_error_recovered"
            ],
            "traceback_not_exposed": history_analysis[
                "traceback_not_exposed"
            ],
            "baseline_nonzero_observed_as_normal_tool_result": (
                history_analysis["baseline_nonzero_was_normal_tool_result"]
            ),
            "baseline_pytest_calls": history_analysis[
                "baseline_pytest_calls"
            ],
            "run4_pytest_calls": history_analysis["run4_pytest_calls"],
            "run4_pytest_reruns_without_intervening_mutation": (
                run4_pytest_reruns_without_intervening_mutation
            ),
            "run4_read_only_responses_after_failed_test": history_analysis[
                "run4_read_only_responses_after_failed_test"
            ],
            "run4_progress_guard_interventions": history_analysis[
                "run4_progress_guard_interventions"
            ],
            "run4_pretest_guard_interventions": history_analysis[
                "run4_pretest_guard_interventions"
            ],
            "run4_successful_mutations_after_failed_test": history_analysis[
                "run4_successful_mutations_after_failed_test"
            ],
            "run4_started_with_pending_failed_test": history_analysis[
                "run4_started_with_pending_failed_test"
            ],
            "run4_initial_pending_command_present": history_analysis[
                "run4_initial_pending_command_present"
            ],
            "run4_initial_diagnosis_responses_used": history_analysis[
                "run4_initial_diagnosis_responses_used"
            ],
            "run4_pending_failed_test_at_run_end": (
                run4_pending_failed_test_at_run_end
            ),
            "run4_first_mutation_step": history_analysis[
                "run4_first_mutation_step"
            ],
            "run4_edit_file_calls": history_analysis[
                "run4_edit_file_calls"
            ],
            "run4_write_file_calls": history_analysis[
                "run4_write_file_calls"
            ],
            "run4_reads_before_first_mutation": history_analysis[
                "run4_reads_before_first_mutation"
            ],
            "run4_duplicate_complete_reads_before_first_mutation": (
                run4_duplicate_complete_reads_before_first_mutation
            ),
            "run4_migration_reads": history_analysis[
                "run4_migration_reads"
            ],
            "run4_repository_scan_calls": history_analysis[
                "run4_repository_scan_calls"
            ],
            "diagnostic_stdout_original_chars": history_analysis[
                "diagnostic_stdout_original_chars"
            ],
            "diag_head_retained": history_analysis["diag_head_retained"],
            "diag_tail_retained": history_analysis["diag_tail_retained"],
            "diag_middle_removed": history_analysis["diag_middle_removed"],
            "diagnostic_blocked_observed": history_analysis[
                "diagnostic_blocked_observed"
            ],
            "diagnostic_ready_observed": history_analysis[
                "diagnostic_ready_observed"
            ],
            "diagnose_script_not_read": history_analysis[
                "diagnose_script_not_read"
            ],
            "migration_shell_search_violation": history_analysis[
                "migration_shell_search_violation"
            ],
            "run4_migration_shell_search_violation": history_analysis[
                "run4_migration_shell_search_violation"
            ],
            "workspace_boundary_bypass_attempted": history_analysis[
                "workspace_boundary_bypass_attempted"
            ],
            "run3_verification_calls": history_analysis[
                "run3_verification_calls"
            ],
            "run3_diagnose_calls": history_analysis[
                "run3_diagnose_calls"
            ],
            "run4_diagnose_calls": history_analysis[
                "run4_diagnose_calls"
            ],
            "run3_unexpected_run_commands": history_analysis[
                "run3_unexpected_run_commands"
            ],
            "run2_run_command_calls": history_analysis[
                "run2_run_command_calls"
            ],
            "run2_edit_file_calls": history_analysis[
                "run2_edit_file_calls"
            ],
            "run2_write_file_calls": history_analysis[
                "run2_write_file_calls"
            ],
            "run2_verification_calls": history_analysis[
                "run2_verification_calls"
            ],
            "run2_migration_reads": history_analysis[
                "run2_migration_reads"
            ],
            "run2_diagnose_reads": history_analysis[
                "run2_diagnose_reads"
            ],
            "run2_test_reads": history_analysis["run2_test_reads"],
            "run2_duplicate_complete_reads": history_analysis[
                "run2_duplicate_complete_reads"
            ],
            "migration_reads_outside_run3": history_analysis[
                "migration_reads_outside_run3"
            ],
            "run3_migration_reads": history_analysis[
                "run3_migration_reads"
            ],
            "run1_tool_calls": history_analysis["run1_tool_calls"],
            "pre_repair_edit_or_write_calls": history_analysis[
                "pre_repair_edit_or_write_calls"
            ],
            "visible_tests_passed": final_visible_passed,
            "hidden_checks_passed": hidden_passed,
            "hidden_check_results": hidden_results,
            "protected_files_unchanged": protected_unchanged,
            "host_visible_recheck_seconds": host_visible_seconds,
            "hidden_evaluation_seconds": hidden_seconds,
            "verification_calls": verification_calls,
            "workspace_revision": (
                agent.workspace_revision if agent is not None else None
            ),
            "verified_revision": (
                agent.verified_revision if agent is not None else None
            ),
            "verification_state_clean": verification_state_clean,
            "temporary_workspace_cleaned": temporary_workspace_cleaned,
            **completion_dimensions,
            "long_running_coverage_passed": coverage_passed,
            "final_integrated_success": integrated_success,
            "evidence": evidence_metadata,
            "evidence_preservation_passed": evidence_passed,
            "evidence_failed_requirements": (
                _evidence_failed_requirements(evidence_result)
            ),
            "real_validation_process_success": process_success,
            "coverage_failed_requirements": [
                *coverage.failed_requirements,
            ],
            "target_band_warnings": list(coverage.warnings),
            "not_evaluated_objectives": sorted(
                name for name, status in objective_statuses.items()
                if status is None
            ),
            **objective_statuses,
            "error": error,
        }
    )
    return redact_report_sensitive_values(
        report,
        tokens,
        temporary_root_path,
    )


def atomic_write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _status(value: bool | None) -> str:
    if value is None:
        return "NOT EVALUATED"
    return "PASS" if value else "FAIL"


def _action_status(value: bool | None) -> str:
    if value is None:
        return "NOT EVALUATED"
    return "PASS" if value else "WARNING"


def print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("FINAL INTEGRATED LONG-RUNNING CHALLENGE")
    print("=" * 60)
    print(f"Challenge: {CHALLENGE_NAME}")
    print(f"Scenario: {SCENARIO_NAME}")
    print(f"Model: {report.get('model', '<unavailable>')}")
    print("\nExecution")
    print(f"Agent Runs: {report.get('agent_runs', 0)} / 4")
    print(f"LLM Calls: {report.get('llm_calls', 0)} (minimum 20)")
    print(f"Tool Calls: {report.get('tool_calls', 0)} (minimum 30)")
    print(
        "Distinct Files Read: "
        f"{report.get('distinct_files_read', 0)} (minimum 10)"
    )
    print(
        "Production Files Changed: "
        f"{report.get('production_files_changed', 0)} (minimum 3)"
    )
    print(f"Files Created: {report.get('files_created', 0)} (minimum 1)")
    print("\nPhase Discipline")
    print(f"Run 2 run_command Calls: {report.get('run2_run_command_calls', 0)}")
    print(f"Run 2 Migration Reads: {report.get('run2_migration_reads', 0)}")
    print(f"Run 2 Diagnostic Source Reads: {report.get('run2_diagnose_reads', 0)}")
    print(f"Run 2 Test Reads: {report.get('run2_test_reads', 0)}")
    print(
        "Run 2 Duplicate Complete Reads: "
        f"{report.get('run2_duplicate_complete_reads', 0)}"
    )
    print(
        "Migration Reads Outside Run 3: "
        f"{report.get('migration_reads_outside_run3', 0)}"
    )
    print("\nAction Discipline / Efficiency")
    print(
        "First Mutation Step: "
        f"{report.get('run4_first_mutation_step', 'none')}"
    )
    print(
        "Preferred First Mutation Target: <= "
        f"{PREFERRED_RUN4_FIRST_MUTATION_STEP_MAX}"
    )
    print(
        "First Mutation Discipline: "
        + _action_status(report.get("run4_met_first_mutation_deadline"))
    )
    print(f"edit_file Calls: {report.get('run4_edit_file_calls', 0)}")
    print(f"write_file Calls: {report.get('run4_write_file_calls', 0)}")
    print(
        "Reads Before First Mutation: "
        f"{report.get('run4_reads_before_first_mutation', 0)}"
    )
    print(
        "Duplicate Complete Reads Before First Mutation: "
        f"{report.get('run4_duplicate_complete_reads_before_first_mutation', 0)}"
    )
    print(
        "Preferred Duplicate Read Target: "
        f"{PREFERRED_RUN4_DUPLICATE_COMPLETE_READS}"
    )
    print(
        "Duplicate Read Discipline: "
        + _action_status(
            report.get(
                "run4_did_not_repeat_complete_reads_before_first_mutation"
            )
        )
    )
    print(
        "Run 4 Started With Pending Failed Test: "
        + _action_status(
            report.get("run4_started_with_pending_failed_test")
        )
    )
    print(
        "Run 4 Initial Pending Command Present: "
        + _action_status(
            report.get("run4_initial_pending_command_present")
        )
    )
    print(
        "Initial Diagnosis Responses Used: "
        f"{report.get('run4_initial_diagnosis_responses_used', 0)}"
    )
    print(f"Run 4 pytest Calls: {report.get('run4_pytest_calls', 0)}")
    print(
        "Pytest Reruns Without Intervening Production Mutation: "
        f"{report.get('run4_pytest_reruns_without_intervening_mutation', 0)}"
    )
    print(
        "Read-only Responses After Failed Pytest: "
        f"{report.get('run4_read_only_responses_after_failed_test', 0)}"
    )
    print(
        "Progress Guard Interventions: "
        f"{report.get('run4_progress_guard_interventions', 0)}"
    )
    print(
        "Pre-pytest Progress Guard Interventions: "
        f"{report.get('run4_pretest_guard_interventions', 0)}"
    )
    print(
        "Successful Mutations After Failed Pytest: "
        f"{report.get('run4_successful_mutations_after_failed_test', 0)}"
    )
    print(
        "Pending Failed Pytest At Run End: "
        + (
            "YES"
            if report.get("run4_pending_failed_test_at_run_end") is True
            else "NO"
            if report.get("run4_pending_failed_test_at_run_end") is False
            else "NOT EVALUATED"
        )
    )
    print(
        "Preferred Pytest Rerun Target: "
        f"{PREFERRED_RUN4_PYTEST_RERUNS_WITHOUT_INTERVENING_MUTATION}"
    )
    print(
        "Repair Loop Discipline: "
        + _action_status(
            report.get(
                "run4_repair_loop_discipline_met"
            )
        )
    )
    print(f"Migration Reads: {report.get('run4_migration_reads', 0)}")
    print(
        "Repository Scan Calls: "
        f"{report.get('run4_repository_scan_calls', 0)}"
    )
    action_warnings = report.get("action_discipline_warnings") or []
    if action_warnings:
        print("Action Discipline Warnings:")
        for warning in action_warnings:
            print(f"- {warning}")
    print(
        "ACTION DISCIPLINE: "
        + _action_status(report.get("action_discipline_target_met"))
    )
    print("\nRecoverable Errors")
    print(
        "Missing legacy config: "
        + _status(report.get("missing_legacy_error_recovered"))
    )
    print(
        "Workspace boundary: "
        + _status(report.get("workspace_boundary_error_recovered"))
    )
    print(
        "Baseline pytest nonzero as normal Tool Result: "
        + _status(report.get("baseline_nonzero_observed_as_normal_tool_result"))
    )
    print("\nRead / Shell Output Budget")
    print(
        "Migration Pagination Reads: "
        f"{report.get('migration_pagination_reads', 0)}"
    )
    print(
        "Sequential next_start_line: "
        + _status(report.get("sequential_next_start_line"))
    )
    print(
        "Shell truncation observed: "
        + _status(report.get("shell_truncation_observed"))
    )
    print("\nContext Lifecycle")
    print(
        "Requests with Compaction: "
        f"{report.get('requests_with_compaction', 0)}"
    )
    print(
        "Maximum LLM Request: "
        f"{report.get('maximum_request_chars', 0)} / "
        f"{report.get('configured_max_context_chars', MAX_CONTEXT_CHARS)} chars"
    )
    print(
        "Full History Preserved: "
        + _status(report.get("full_raw_history_preserved"))
    )
    print("\nObjective Evaluation")
    print(
        "Visible Host Re-check: "
        + _status(report.get("visible_tests_passed"))
    )
    print(
        "Hidden Behavior Checks: "
        + _status(report.get("hidden_checks_passed"))
    )
    print(
        "Protected Files Unchanged: "
        + _status(report.get("protected_files_unchanged"))
    )
    print(
        "Verification State CLEAN: "
        + _status(report.get("verification_state_clean"))
    )
    print(
        "Run 4 Diagnostic READY: "
        + _status(report.get("diagnostic_ready_observed"))
    )
    print(
        "Final RELEASE_TOKEN Recovery: "
        + _status(report.get("release_token_recovered"))
    )
    print(
        "Final MIGRATION_KEY Recovery: "
        + _status(report.get("migration_key_recovered"))
    )
    print(
        "Final DIAG_TAIL_TOKEN Recovery: "
        + _status(report.get("diag_tail_token_recovered"))
    )
    print("\nFinal Request Token Visibility")
    print(
        "RELEASE_TOKEN present: "
        + _status(
            report.get("release_token_present_in_final_llm_request")
        )
    )
    print(
        "MIGRATION_KEY present: "
        + _status(
            report.get("migration_key_present_in_final_llm_request")
        )
    )
    print(
        "DIAG_TAIL_TOKEN present: "
        + _status(
            report.get("diag_tail_token_present_in_final_llm_request")
        )
    )
    print(
        "Report Module Created: "
        + _status(report.get("report_module_created"))
    )
    print("\nTiming Evidence")
    print(
        "INTERNAL_WALL_CLOCK_SECONDS: "
        f"{float(report.get('internal_elapsed_seconds', 0.0)):.2f}"
    )
    print(
        "Runtime Target: "
        + ("PASS" if report.get("runtime_target_met") else "OUTSIDE TARGET")
    )
    warnings = report.get("target_band_warnings") or []
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if report.get("error"):
        print("Error: " + str(report["error"]))

    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    print("\n" + "-" * 60)
    print("Workspace Evidence")
    print("-" * 60)
    print(f"Evidence Run ID: {evidence.get('run_id') or '<unavailable>'}")
    print(
        "Initial Workspace Snapshot: "
        + _status(bool(evidence.get("initial_snapshot_created")))
    )
    print(
        "Final Workspace Snapshot: "
        + _status(bool(evidence.get("final_snapshot_created")))
    )
    print(f"Created Files: {int(evidence.get('created_count') or 0)}")
    print(f"Modified Files: {int(evidence.get('modified_count') or 0)}")
    print(f"Deleted Files: {int(evidence.get('deleted_count') or 0)}")
    print(
        "Protected Files Changed: "
        f"{int(evidence.get('protected_files_changed') or 0)}"
    )
    for label, key in (("Created", "created_files"), ("Modified", "modified_files")):
        values = evidence.get(key)
        if isinstance(values, list) and values:
            print(f"{label}:")
            for value in values:
                print(f"  {value}")
    print(
        "Before Manifest: "
        + _status(bool(evidence.get("before_manifest_created")))
    )
    print(
        "After Manifest: "
        + _status(bool(evidence.get("after_manifest_created")))
    )
    print(
        "Workspace Changes JSON: "
        + _status(bool(evidence.get("changes_json_created")))
    )
    print(
        "Unified Workspace Diff: "
        + _status(bool(evidence.get("diff_created")))
    )
    print(f"Evidence Directory: {evidence.get('directory') or '<unavailable>'}")
    print(
        "Temporary Workspace Cleanup: "
        + _status(bool(report.get("temporary_workspace_cleaned")))
    )
    print(
        "Evidence Snapshot Preserved: "
        + _status(bool(evidence.get("snapshot_preserved_after_cleanup")))
    )
    if evidence.get("error"):
        print("Evidence Error: " + str(evidence["error"]))
    print("\nFinal Result")
    print(
        "FUNCTIONAL CHALLENGE: "
        + _status(bool(report.get("functional_challenge_passed")))
    )
    print(
        "LONG-RUNNING COVERAGE: "
        + _status(bool(report.get("long_running_coverage_passed")))
    )
    print(
        "RUNTIME TARGET: "
        + ("PASS" if report.get("runtime_target_met") else "OUTSIDE TARGET")
    )
    print(
        "ACTION DISCIPLINE: "
        + _action_status(report.get("action_discipline_target_met"))
    )
    print(
        "EVIDENCE PRESERVATION: "
        + _status(bool(report.get("evidence_preservation_passed")))
    )
    print(
        "FINAL INTEGRATED CHALLENGE: "
        + _status(bool(report.get("final_integrated_success")))
    )
    print(
        "REAL VALIDATION PROCESS: "
        + _status(bool(report.get("real_validation_process_success")))
    )
    print(f"Report: {REPORT_PATH}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    process_started_at = datetime.now().astimezone()
    process_started_perf = time.perf_counter()
    try:
        client = LLMClient()
    except Exception as error:
        report = build_failure_report(
            model=os.getenv("LLM_MODEL") or "<unavailable>",
            started_at=process_started_at,
            started_perf=process_started_perf,
            error="Required LLM configuration is missing: "
            + sanitize_error(error, temporary_root=None, tokens=None),
        )
    else:
        try:
            report = execute_real_challenge(client)
        except Exception as error:
            report = build_failure_report(
                model=client.model,
                started_at=process_started_at,
                started_perf=process_started_perf,
                error="Challenge evaluator failed: "
                + sanitize_error(error, temporary_root=None, tokens=None),
            )

    try:
        atomic_write_report(REPORT_PATH, report)
    except Exception as error:
        print(
            "Failed to write challenge report: "
            + sanitize_error(error, temporary_root=None, tokens=None),
            file=sys.stderr,
        )
        print_summary(report)
        return 1

    print_summary(report)
    return (
        0
        if report.get("real_validation_process_success") is True
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
