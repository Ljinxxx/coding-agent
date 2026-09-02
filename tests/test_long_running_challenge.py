import inspect
import json
import re
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import scripts.long_running_challenge as challenge_module
import scripts.long_running_evidence as evidence_module
import scripts.long_running_telemetry as telemetry_module
import scripts.verify_long_running_challenge_real as runner_module
from scripts.e2e_evaluator import (
    protected_files_unchanged,
    run_visible_tests,
    snapshot_workspace,
)
from scripts.long_running_challenge import (
    HIDDEN_PARSER_CHECK,
    HIDDEN_SERIALIZER_REPORT_CLI_CHECK,
    HIDDEN_STORE_SERVICE_CHECK,
    ChallengeTokens,
    build_long_running_fixture,
    generate_diagnose_source,
    generate_migration_notes,
    materialize_long_running_fixture,
    run_hidden_code,
)
from scripts.long_running_telemetry import (
    FunctionalChallengeMetrics,
    LLMCallMetric,
    LongRunningMetrics,
    RecordingLLM,
    analyze_compaction_requests,
    analyze_migration_pagination,
    build_report_payload,
    evaluate_functional_challenge,
    evaluate_long_running_coverage,
    final_integrated_success,
    runtime_target_met,
)
from src.context_compaction import (
    CURRENT_RUN_HEADER,
    PRIOR_CONTEXT_HEADER,
)


EXPECTED_INITIAL_FILES = {
    "README.md",
    "docs/architecture.md",
    "docs/operations.md",
    "config/policy.json",
    "data/migration_notes.txt",
    "data/sample_incidents.txt",
    "incident/__init__.py",
    "incident/constants.py",
    "incident/models.py",
    "incident/parser.py",
    "incident/store.py",
    "incident/service.py",
    "incident/serializer.py",
    "incident/cli.py",
    "scripts/diagnose.py",
    "tests/test_parser.py",
    "tests/test_store.py",
    "tests/test_service.py",
    "tests/test_serializer.py",
    "tests/test_report.py",
    "tests/test_cli.py",
}

EXPECTED_PROTECTED_FILES = {
    "config/policy.json",
    "data/migration_notes.txt",
    "scripts/diagnose.py",
    "tests/test_parser.py",
    "tests/test_store.py",
    "tests/test_service.py",
    "tests/test_serializer.py",
    "tests/test_report.py",
    "tests/test_cli.py",
}

FIXED_TOKENS = ChallengeTokens(
    release_token="release-unit-1234",
    migration_key="migration-unit-5678",
    diag_head_token="diag-head-unit-9012",
    diag_tail_token="diag-tail-unit-3456",
)


def _source(value: str) -> str:
    return dedent(value).lstrip()


def _write_files(workspace: Path, files: dict[str, str]) -> None:
    for relative_name, content in files.items():
        target = workspace / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")


def _reference_solution_files() -> dict[str, str]:
    return {
        "incident/__init__.py": "from incident.models import Incident\n",
        "incident/models.py": _source(
            """
            from dataclasses import dataclass


            @dataclass(frozen=True)
            class Incident:
                incident_id: str
                severity: str
                timestamp: int
                message: str
            """
        ),
        "incident/constants.py": _source(
            """
            SEVERITY_ORDER = {
                "LOW": 0,
                "MEDIUM": 1,
                "HIGH": 2,
                "CRITICAL": 3,
            }
            """
        ),
        "incident/parser.py": _source(
            """
            from incident.constants import SEVERITY_ORDER
            from incident.models import Incident


            def parse_incident(line: str) -> Incident:
                parts = line.split("|")
                if len(parts) != 4:
                    raise ValueError("incident line must contain four fields")
                incident_id, severity, timestamp, message = parts
                incident_id = incident_id.strip()
                severity = severity.strip().upper()
                if severity not in SEVERITY_ORDER:
                    raise ValueError(f"unknown severity: {severity}")
                return Incident(
                    incident_id=incident_id,
                    severity=severity,
                    timestamp=int(timestamp.strip()),
                    message=message.strip(),
                )
            """
        ),
        "incident/store.py": _source(
            """
            from incident.constants import SEVERITY_ORDER


            class IncidentStore:
                def __init__(self):
                    self._incidents = {}

                def get(self, incident_id):
                    return self._incidents.get(incident_id)

                def values(self):
                    return list(self._incidents.values())

                def upsert(self, incident):
                    current = self.get(incident.incident_id)
                    should_replace = current is None
                    if current is not None:
                        new_rank = SEVERITY_ORDER[incident.severity]
                        current_rank = SEVERITY_ORDER[current.severity]
                        should_replace = new_rank > current_rank or (
                            new_rank == current_rank
                            and incident.timestamp >= current.timestamp
                        )
                    if should_replace:
                        self._incidents[incident.incident_id] = incident
                    return should_replace
            """
        ),
        "incident/service.py": _source(
            """
            from incident.constants import SEVERITY_ORDER
            from incident.store import IncidentStore


            def select_actionable(incidents, min_severity):
                threshold = str(min_severity).strip().upper()
                if threshold not in SEVERITY_ORDER:
                    raise ValueError(f"unknown severity: {min_severity}")
                store = IncidentStore()
                for incident in list(incidents):
                    store.upsert(incident)
                selected = [
                    incident
                    for incident in store.values()
                    if SEVERITY_ORDER[incident.severity]
                    >= SEVERITY_ORDER[threshold]
                ]
                return sorted(
                    selected,
                    key=lambda item: (item.timestamp, item.incident_id),
                )
            """
        ),
        "incident/serializer.py": _source(
            """
            def serialize_incident(incident):
                return {
                    "id": incident.incident_id,
                    "severity": incident.severity,
                    "timestamp": incident.timestamp,
                    "message": incident.message,
                }
            """
        ),
        "incident/report.py": _source(
            """
            from incident.serializer import serialize_incident


            def build_report(received, actionable):
                actionable_items = list(actionable)
                counts = {
                    severity: sum(
                        item.severity == severity
                        for item in actionable_items
                    )
                    for severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
                }
                return {
                    "schema_version": 2,
                    "total_received": len(list(received)),
                    "total_actionable": len(actionable_items),
                    "severity_counts": counts,
                    "incidents": [
                        serialize_incident(item) for item in actionable_items
                    ],
                }
            """
        ),
        "incident/cli.py": _source(
            """
            import argparse
            import json

            from incident.constants import SEVERITY_ORDER
            from incident.parser import parse_incident
            from incident.report import build_report
            from incident.serializer import serialize_incident
            from incident.service import select_actionable


            def _severity(value):
                normalized = value.strip().upper()
                if normalized not in SEVERITY_ORDER:
                    raise argparse.ArgumentTypeError(
                        f"unknown severity: {value}"
                    )
                return normalized


            def main(argv=None):
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    "input_path",
                    nargs="?",
                    default="data/sample_incidents.txt",
                )
                parser.add_argument(
                    "--min-severity",
                    type=_severity,
                    default="MEDIUM",
                )
                parser.add_argument("--report", action="store_true")
                args = parser.parse_args(argv)
                with open(args.input_path, encoding="utf-8") as input_file:
                    received = [
                        parse_incident(line)
                        for line in input_file
                        if line.strip()
                    ]
                actionable = select_actionable(
                    received,
                    args.min_severity,
                )
                payload = (
                    build_report(received, actionable)
                    if args.report
                    else [serialize_incident(item) for item in actionable]
                )
                print(json.dumps(payload, sort_keys=True))
                return 0


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ),
    }


def _minimum_coverage_metrics() -> LongRunningMetrics:
    return LongRunningMetrics(
        agent_runs=4,
        llm_calls=20,
        tool_calls=30,
        distinct_files_read=10,
        production_files_changed=5,
        files_created=1,
        pagination_reads=8,
        compaction_bearing_requests=3,
        controlled_errors_observed=2,
        controlled_errors_recovered=2,
        shell_truncation_observed=True,
        verification_calls=1,
    )


def _passing_functional_metrics() -> FunctionalChallengeMetrics:
    return FunctionalChallengeMetrics(
        agent_runs_completed=True,
        initial_visible_tests_failed=True,
        missing_legacy_error_observed=True,
        workspace_boundary_error_observed=True,
        controlled_errors_recovered=2,
        migration_pagination_valid=True,
        migration_key_seen_in_tool_result=True,
        baseline_nonzero_was_normal_tool_result=True,
        shell_truncation_observed=True,
        diag_head_retained=True,
        diag_tail_retained=True,
        diag_middle_removed=True,
        diagnose_script_not_read=True,
        post_agent_visible_recheck_passed=True,
        hidden_checks_passed=True,
        protected_files_unchanged=True,
        verification_calls=1,
        verification_state_clean=True,
        release_token_recovered=True,
        migration_key_recovered=True,
        diag_tail_token_recovered=True,
        report_module_created=True,
    )


def _read_exchange(
    call_id: str,
    *,
    start_line: int,
    next_start_line: int | None,
    payload: str,
) -> list[dict[str, object]]:
    next_value = "none" if next_start_line is None else str(next_start_line)
    arguments = {
        "path": "data/migration_notes.txt",
        "start_line": start_line,
    }
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": (
                "[read_file]\n"
                "path: data/migration_notes.txt\n"
                f"lines: {start_line}-{start_line + 4} of 1800\n"
                "total_lines: 1800\n"
                f"truncated_before: {str(start_line > 1).lower()}\n"
                f"truncated_after: {str(next_start_line is not None).lower()}\n"
                "char_truncated: false\n"
                "partial_line: false\n"
                "original_selected_chars: 250\n"
                f"next_start_line: {next_value}\n"
                "notice: none\n\n"
                f"{payload}\n"
            ),
        },
    ]


def _tool_exchange(
    call_id: str,
    name: str,
    arguments: dict[str, object],
    result: str,
) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        },
    ]


def _run4_pytest_exchange(
    call_id: str,
    *,
    exit_code: int,
    timed_out: bool = False,
    command: str | None = None,
) -> list[dict[str, object]]:
    return _tool_exchange(
        call_id,
        "run_command",
        {
            "command": command or runner_module.CHALLENGE_PYTEST_COMMAND,
        },
        json.dumps(
            {
                "exit_code": exit_code,
                "timed_out": timed_out,
            }
        ),
    )


def _progress_guard_blocked_exchange(
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> list[dict[str, object]]:
    return _tool_exchange(
        call_id,
        name,
        arguments,
        json.dumps(
            {
                "ok": False,
                "tool": name,
                "error_type": "ProgressGuardBlocked",
                "message": "A successful mutation is required first.",
                "policy_blocked": True,
            }
        ),
    )


def _analyze_run4_history(
    history: list[dict[str, object]],
    workspace: Path | None = None,
) -> dict[str, object]:
    record = runner_module.RunRecord(
        index=4,
        name="Release Repair",
        require_verified_completion=True,
        completed=False,
        history_start=0,
        history_end=len(history),
    )
    return runner_module.analyze_history(
        history,
        [record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
        workspace,
    )


def _file_read_exchange(
    call_id: str,
    path: str,
    *,
    truncated_before: bool = False,
    truncated_after: bool = False,
    next_start_line: int | None = None,
) -> list[dict[str, object]]:
    next_value = "none" if next_start_line is None else str(next_start_line)
    return _tool_exchange(
        call_id,
        "read_file",
        {"path": path},
        (
            "[read_file]\n"
            f"path: {path}\n"
            "lines: 1-2 of 2\n"
            "total_lines: 2\n"
            f"truncated_before: {str(truncated_before).lower()}\n"
            f"truncated_after: {str(truncated_after).lower()}\n"
            "char_truncated: false\n"
            "partial_line: false\n"
            "original_selected_chars: 10\n"
            f"next_start_line: {next_value}\n"
            "notice: none\n\n"
            "content\n"
        ),
    )


def test_fixture_materialization_is_deterministic_and_isolated(
    tmp_path: Path,
) -> None:
    first_fixture = build_long_running_fixture(FIXED_TOKENS)
    second_fixture = build_long_running_fixture(FIXED_TOKENS)
    assert first_fixture == second_fixture

    run_specs = runner_module.build_run_specs(FIXED_TOKENS)
    assert [spec.require_verified_completion for spec in run_specs] == [
        False,
        False,
        False,
        True,
    ]
    assert all(spec.progress_guard is None for spec in run_specs[:3])
    run4_progress_guard = run_specs[3].progress_guard
    assert run4_progress_guard is not None
    assert run4_progress_guard.tracked_commands == (
        runner_module.CHALLENGE_PYTEST_COMMAND,
    )
    assert run4_progress_guard.mutation_tool_names == (
        "edit_file",
        "write_file",
    )
    assert run4_progress_guard.diagnosis_responses == 1
    assert FIXED_TOKENS.release_token in run_specs[0].prompt
    assert all(
        FIXED_TOKENS.release_token not in spec.prompt
        for spec in run_specs[1:]
    )
    for hidden_value in (
        FIXED_TOKENS.migration_key,
        FIXED_TOKENS.diag_head_token,
        FIXED_TOKENS.diag_tail_token,
    ):
        assert all(hidden_value not in spec.prompt for spec in run_specs)

    first_workspace = tmp_path / "first" / "workspace"
    second_workspace = tmp_path / "second" / "workspace"
    materialize_long_running_fixture(first_fixture, first_workspace)
    materialize_long_running_fixture(second_fixture, second_workspace)
    assert first_workspace != second_workspace
    assert snapshot_workspace(first_workspace) == snapshot_workspace(
        second_workspace
    )

    first_parser = first_workspace / "incident/parser.py"
    second_parser = second_workspace / "incident/parser.py"
    original = second_parser.read_text(encoding="utf-8")
    first_parser.write_text("changed only in first workspace\n", encoding="utf-8")
    assert second_parser.read_text(encoding="utf-8") == original


def test_run_agent_stage_forwards_only_the_run_spec_progress_guard(
    tmp_path: Path,
) -> None:
    class RecordingStub:
        def __init__(self) -> None:
            self.metrics: list[object] = []
            self.run_indices: list[int] = []

        def set_run_index(self, run_index: int) -> None:
            self.run_indices.append(run_index)

    class AgentStub:
        def __init__(self) -> None:
            self.history: list[dict[str, object]] = []
            self.workspace_revision = 0
            self.verified_revision = 0
            self.verification_required = False
            self.received: list[dict[str, object]] = []

        def run(self, prompt: str, **kwargs: object) -> str:
            self.received.append({"prompt": prompt, **kwargs})
            self.history.append({"role": "user", "content": prompt})
            return "done"

    recording = RecordingStub()
    agent = AgentStub()
    specs = runner_module.build_run_specs(FIXED_TOKENS)

    for spec in (specs[0], specs[3]):
        record = runner_module.RunRecord(
            index=spec.index,
            name=spec.name,
            require_verified_completion=spec.require_verified_completion,
        )
        assert runner_module.run_agent_stage(
            agent,
            recording,
            spec,
            record,
            tmp_path,
        ) == "done"

    assert agent.received[0]["progress_guard"] is None
    assert agent.received[1]["progress_guard"] is specs[3].progress_guard
    assert recording.run_indices == [1, 4]


def test_fixture_public_cli_contract_defines_unsupported_severity_usage_error(
) -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    readme = " ".join(fixture.files["README.md"].casefold().split())

    for public_contract_phrase in (
        "--min-severity",
        "case-insensitively",
        "unsupported values",
        "cli argument boundary",
        "cli usage error",
        "exit status 2",
        "before incident processing",
    ):
        assert public_contract_phrase in readme

    for implementation_hint in (
        "choices=",
        "type=str.upper",
        "parser.error(",
    ):
        assert implementation_hint not in readme

    public_fixture = "\n".join(fixture.files.values())
    assert HIDDEN_SERIALIZER_REPORT_CLI_CHECK not in public_fixture


def test_public_report_contract_is_consistent_across_public_surfaces() -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    readme = " ".join(fixture.files["README.md"].casefold().split())
    report_test = fixture.files["tests/test_report.py"]
    cli_test = fixture.files["tests/test_cli.py"]
    run_4 = " ".join(
        runner_module.build_run_specs(FIXED_TOKENS)[3]
        .prompt.casefold()
        .split()
    )
    report_fields = (
        "schema_version",
        "total_received",
        "total_actionable",
        "severity_counts",
        "incidents",
    )

    for phrase in (
        "--report",
        "incident.report.build_report",
        "complete v2 report schema",
        "exactly these top-level fields",
        "serialized actionable incidents",
        "required output order",
    ):
        assert phrase in readme
    for severity in ("low", "medium", "high", "critical"):
        assert severity in readme
    for field in report_fields:
        assert field in readme
        assert field in report_test
        assert field in cli_test
    assert "assert payload == {" in cli_test
    for phrase in (
        "complete public v2 report schema",
        "incident.report.build_report",
        "separate reduced report shape",
    ):
        assert phrase in run_4


def test_visible_cli_report_oracle_rejects_reduced_shape_and_accepts_complete_shape(
    tmp_path: Path,
) -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    workspace = tmp_path / "workspace"
    materialize_long_running_fixture(fixture, workspace)
    reference = _reference_solution_files()
    _write_files(workspace, reference)

    reference_cli = reference["incident/cli.py"]
    canonical_call = "build_report(received, actionable)"
    assert reference_cli.count(canonical_call) == 1
    reduced_expression = (
        '{"schema_version": 2, '
        '"total_received": len(received), '
        '"total_actionable": len(actionable), '
        '"actionable": ['
        "serialize_incident(item) for item in actionable]}"
    )
    _write_files(
        workspace,
        {
            "incident/cli.py": reference_cli.replace(
                canonical_call,
                reduced_expression,
            )
        },
    )

    rejected = run_visible_tests(
        workspace,
        basetemp=tmp_path / "reduced-report-pytest",
    )
    rejected_output = rejected.stdout + rejected.stderr
    assert rejected.timed_out is False
    assert rejected.exit_code not in (None, 0)
    assert "test_cli_report_flag_outputs_v2_report" in rejected_output

    _write_files(
        workspace,
        {"incident/cli.py": reference_cli},
    )
    accepted = run_visible_tests(
        workspace,
        basetemp=tmp_path / "complete-report-pytest",
    )
    assert accepted.passed, accepted.stdout + accepted.stderr


def test_public_report_contract_does_not_leak_hidden_concrete_inputs() -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    public_surfaces = "\n".join(
        (
            fixture.files["README.md"],
            fixture.files["tests/test_cli.py"],
            runner_module.build_run_specs(FIXED_TOKENS)[3].prompt,
        )
    )

    for hidden_literal in (
        "hidden-json",
        "hidden-low",
        "hidden-cli-low",
        "hidden-cli-high",
    ):
        assert hidden_literal not in public_surfaces


def test_run2_prompt_enforces_reconnaissance_phase_boundaries() -> None:
    assert runner_module.CHALLENGE_MAX_STEPS == 40
    assert "max_steps=CHALLENGE_MAX_STEPS" in inspect.getsource(
        runner_module.execute_real_challenge
    )
    run_2 = " ".join(
        runner_module.build_run_specs(FIXED_TOKENS)[1].prompt.casefold().split()
    )

    required_protocol = (
        "repository reconnaissance only",
        "use only list_directory and read_file in this run",
        "do not use run_command, edit_file, write_file, or verify_workspace",
        "do not read: - data/migration_notes.txt",
        "- scripts/diagnose.py",
        "- anything under tests/",
        "path to read_file exactly as written",
        "do not normalize it",
        "remove a leading ../ segment",
        "do not read any file under tests/",
        "do not list the tests/ directory",
        "truncated_after=false and char_truncated=false",
        "never start a second inspection pass",
        "do not read the same complete file more than once",
        "previous read was truncated",
        "following its continuation metadata",
        "once this checklist is complete, stop exploring",
        "do not perform a second pass",
        "treat those later steps as future work",
    )
    for phrase in required_protocol:
        assert phrase in run_2

    assert "config/legacy_policy.json" not in run_2
    assert "../shared/global_defaults.json" not in run_2


def test_run3_prompt_requires_exact_sequential_pagination() -> None:
    run_specs = runner_module.build_run_specs(FIXED_TOKENS)
    run_3 = " ".join(run_specs[2].prompt.casefold().split())
    run_4 = " ".join(run_specs[3].prompt.casefold().split())

    required_protocol = (
        "this is the only phase in which data/migration_notes.txt",
        "from the start of the file",
        "do not provide a custom start_line or max_lines for the first read",
        "must use exactly",
        "immediately preceding migration result",
        "never guess, estimate, skip ahead, jump backward",
        "at most one read_file call for data/migration_notes.txt",
        "continue until next_start_line is none",
        "once next_start_line is none",
        "the migration audit is complete",
        "do not restart",
        "from line 1",
        "proceed immediately",
        "baseline pytest command",
        "do not inspect or search data/migration_notes.txt using run_command",
        "use only read_file and its continuation metadata",
        "phase a - sequential migration read",
        "phase h - checkpoint",
        runner_module.CHALLENGE_PYTEST_COMMAND.casefold(),
        "run exactly `python scripts/diagnose.py`",
        "do not read or modify scripts/diagnose.py",
        "do not repair production files",
    )
    for phrase in required_protocol:
        assert phrase in run_3

    for leaked_value in (
        "highest-severity-then-latest-timestamp",
        "timestamp-then-incident-id",
        "directive_unknown_severity=reject",
        "directive_report_schema_version=2",
        FIXED_TOKENS.migration_key,
    ):
        assert leaked_value not in run_3

    assert "do not read or modify scripts/diagnose.py" in run_4
    assert "call verify_workspace" in run_4


def test_run4_prompt_enforces_execution_phase_and_action_deadline() -> None:
    run_4 = " ".join(
        runner_module.build_run_specs(FIXED_TOKENS)[3]
        .prompt.casefold()
        .split()
    )
    required_protocol = (
        "final execution/repair phase",
        "repository reconnaissance",
        "migration audit",
        "already complete",
        "do not restart the investigation",
        "first three model responses",
        "bounded refresh",
        "by the fourth model response",
        "edit_file or write_file",
        "repository-wide rescan",
        "reread readme, docs, configuration, policy",
        "do not reread an unchanged complete file",
        "after that same file was modified",
        "concrete new test failure",
        runner_module.CHALLENGE_PYTEST_COMMAND.casefold(),
        "diag_state=ready",
        "call verify_workspace",
    )
    for phrase in required_protocol:
        assert phrase in run_4

    assert "by the third model response" not in run_4
    assert "responses 1-2 only" not in run_4
    assert "no later than response 3" not in run_4


def test_run4_mutation_deadline_accepts_four_but_not_later_or_missing() -> None:
    assert runner_module.run4_mutation_deadline_met(4) is True
    assert runner_module.run4_mutation_deadline_met(5) is False
    assert runner_module.run4_mutation_deadline_met(None) is False


def test_run4_prompt_requires_generic_public_boundary_self_review() -> None:
    run_4 = " ".join(
        runner_module.build_run_specs(FIXED_TOKENS)[3]
        .prompt.casefold()
        .split()
    )

    for phrase in (
        "public input boundaries",
        "documented contract",
        "public --min-severity contract",
        "supported severity names are accepted case-insensitively",
        "unsupported values are rejected",
        "cli argument boundary",
        "cli usage error",
        "exit status 2",
        "before incident processing begins",
        "public interface",
        "internal implementation exceptions",
        "one bounded public-contract self-review",
        "public --report contract",
        "complete public v2 report schema",
        "incident.report.build_report",
        "separate reduced report shape",
    ):
        assert phrase in run_4

    for hidden_specific_hint in (
        "invalid",
        "notice",
        "unknown",
        "--min-severity invalid",
        "choices=",
        "type=str.upper",
        "parser.error(",
        "argumenttypeerror",
        "_severity",
        'split("|", 3)',
        'count("|")',
        "maxsplit",
    ):
        assert hidden_specific_hint not in run_4


def test_run4_prompt_requires_fail_diagnose_mutate_retest_cycle() -> None:
    run_4 = " ".join(
        runner_module.build_run_specs(FIXED_TOKENS)[3]
        .prompt.casefold()
        .split()
    )

    for phrase in (
        "after any non-zero result",
        "do not immediately rerun it",
        "at most one following model response",
        "diagnose the concrete failure from its output",
        "only the failing test files named by the output",
        "directly implicated incident/*.py production files",
        "successful targeted production mutation under incident/",
        "edit_file or write_file",
        "only after that intervening mutation",
        "failure set and workspace are unchanged",
        "do not broadly rescan or repeat pytest",
        "the next action must be that production mutation",
        "fail -> diagnose -> mutate -> retest",
        "do not run scripts/diagnose.py while pytest is failing",
    ):
        assert phrase in run_4


def test_run4_prompt_requires_direct_terminal_transitions() -> None:
    run_4 = " ".join(
        runner_module.build_run_specs(FIXED_TOKENS)[3]
        .prompt.casefold()
        .split()
    )

    for transition in (
        "visible pytest pass -> diagnostic",
        (
            "diag_state=blocked -> targeted production repair -> fixed pytest "
            "-> diagnostic"
        ),
        "diag_state=ready -> verify_workspace immediately",
        "verify_workspace pass/clean -> final answer immediately",
    ):
        assert transition in run_4


def test_run4_prompt_requires_exact_trailing_token_contract() -> None:
    run_4 = runner_module.build_run_specs(FIXED_TOKENS)[3].prompt
    normalized_run_4 = " ".join(run_4.casefold().split())
    exact_lines = (
        "RELEASE_TOKEN=<exact retained value>\n"
        "MIGRATION_KEY=<exact retained value>\n"
        "DIAG_TAIL_TOKEN=<exact retained value>"
    )

    assert (
        "Your final response MUST end with exactly these three labeled lines"
        in run_4
    )
    assert exact_lines in run_4
    for instruction in (
        "do not omit",
        "rename",
        "paraphrase",
        "rediscover",
    ):
        assert instruction in normalized_run_4

    for runtime_value in (
        FIXED_TOKENS.release_token,
        FIXED_TOKENS.migration_key,
        FIXED_TOKENS.diag_head_token,
        FIXED_TOKENS.diag_tail_token,
    ):
        assert runtime_value not in run_4


def test_run4_prompt_forbids_migration_reinspection_and_uses_history() -> None:
    run_4 = " ".join(
        runner_module.build_run_specs(FIXED_TOKENS)[3]
        .prompt.casefold()
        .split()
    )
    for phrase in (
        "use that history",
        "data/migration_notes.txt",
        "read_file",
        "run_command",
        "python",
        "open()",
        "grep",
        "recover its directives and key only from existing agent history",
        "scripts/diagnose.py source",
    ):
        assert phrase in run_4


def test_challenge_pytest_command_is_windows_safe_and_consistent() -> None:
    assert runner_module.CHALLENGE_PYTEST_COMMAND == (
        "python -B -m pytest -q --basetemp=.pytest_tmp"
    )
    run_specs = runner_module.build_run_specs(FIXED_TOKENS)
    assert runner_module.CHALLENGE_PYTEST_COMMAND in run_specs[2].prompt
    assert runner_module.CHALLENGE_PYTEST_COMMAND in run_specs[3].prompt
    trusted_command = runner_module.make_command(
        "python",
        *runner_module.CHALLENGE_PYTEST_ARGUMENTS,
    )
    assert runner_module._is_challenge_pytest_command(trusted_command)
    assert not runner_module._is_challenge_pytest_command(
        "python -B -m pytest -q"
    )
    assert not runner_module._is_challenge_pytest_command(
        runner_module.CHALLENGE_PYTEST_COMMAND + " && echo repeated"
    )
    assert "*CHALLENGE_PYTEST_ARGUMENTS" in inspect.getsource(
        runner_module.execute_real_challenge
    )


def test_snapshot_workspace_ignores_challenge_pytest_basetemp(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_files(
        workspace,
        {
            "incident/module.py": "value = 1\n",
            ".pytest_tmp/session/result.txt": "runtime only\n",
        },
    )
    snapshot = snapshot_workspace(workspace)
    assert set(snapshot) == {"incident/module.py"}


def test_fixture_contains_required_long_running_repository_structure(
    tmp_path: Path,
) -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    workspace = tmp_path / "workspace"
    materialize_long_running_fixture(fixture, workspace)

    actual_files = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    assert actual_files == EXPECTED_INITIAL_FILES
    assert len(actual_files) == 21
    assert len(actual_files) >= 18
    assert "incident/report.py" not in actual_files
    assert set(fixture.protected_files) == EXPECTED_PROTECTED_FILES

    visible_test_items = sum(
        len(re.findall(r"^def test_", content, flags=re.MULTILINE))
        for name, content in fixture.files.items()
        if name.startswith("tests/test_")
    )
    assert 22 <= visible_test_items <= 25

    initial = run_visible_tests(
        workspace,
        basetemp=tmp_path / "initial-pytest",
    )
    assert initial.timed_out is False
    assert initial.exit_code not in (None, 0)


def test_migration_notes_place_directives_across_late_pages() -> None:
    notes = generate_migration_notes(FIXED_TOKENS.migration_key)
    lines = notes.splitlines()
    assert len(lines) == 1800
    assert lines[200] == (
        "DIRECTIVE_DEDUPE_POLICY=highest-severity-then-latest-timestamp"
    )
    assert lines[400] == "DIRECTIVE_OUTPUT_ORDER=timestamp-then-incident-id"
    assert lines[800] == "DIRECTIVE_UNKNOWN_SEVERITY=reject"
    assert lines[1200] == "DIRECTIVE_REPORT_SCHEMA_VERSION=2"
    assert lines[1600] == f"MIGRATION_KEY={FIXED_TOKENS.migration_key}"
    assert FIXED_TOKENS.migration_key not in "\n".join(lines[:400])


def test_protected_integrity_detects_modification_and_deletion(
    tmp_path: Path,
) -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)

    modified_workspace = tmp_path / "modified"
    materialize_long_running_fixture(fixture, modified_workspace)
    modified_before = snapshot_workspace(modified_workspace)
    assert protected_files_unchanged(
        fixture,
        modified_before,
        snapshot_workspace(modified_workspace),
    )
    protected_test = modified_workspace / "tests/test_parser.py"
    protected_test.write_text("modified\n", encoding="utf-8")
    assert not protected_files_unchanged(
        fixture,
        modified_before,
        snapshot_workspace(modified_workspace),
    )

    deleted_workspace = tmp_path / "deleted"
    materialize_long_running_fixture(fixture, deleted_workspace)
    deleted_before = snapshot_workspace(deleted_workspace)
    (deleted_workspace / "scripts/diagnose.py").unlink()
    assert not protected_files_unchanged(
        fixture,
        deleted_before,
        snapshot_workspace(deleted_workspace),
    )


def test_hidden_parser_checks_reject_incomplete_visible_only_solution(
    tmp_path: Path,
) -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    workspace = tmp_path / "workspace"
    materialize_long_running_fixture(fixture, workspace)
    reference = _reference_solution_files()
    _write_files(
        workspace,
        {
            "incident/__init__.py": reference["incident/__init__.py"],
            "incident/models.py": reference["incident/models.py"],
            "incident/constants.py": reference["incident/constants.py"],
            "incident/parser.py": _source(
                """
                from incident.models import Incident


                def parse_incident(line):
                    incident_id, severity, timestamp, message = line.split("|")
                    return Incident(
                        incident_id,
                        severity,
                        int(timestamp),
                        message,
                    )
                """
            ),
        },
    )
    rejected = run_hidden_code(workspace, HIDDEN_PARSER_CHECK)
    assert rejected.passed is False

    _write_files(
        workspace,
        {"incident/parser.py": reference["incident/parser.py"]},
    )
    accepted = run_hidden_code(workspace, HIDDEN_PARSER_CHECK)
    assert accepted.passed, accepted.stdout + accepted.stderr


def test_hidden_store_and_service_checks_cover_unseen_contract_cases(
    tmp_path: Path,
) -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    workspace = tmp_path / "workspace"
    materialize_long_running_fixture(fixture, workspace)
    reference = _reference_solution_files()
    _write_files(workspace, reference)
    _write_files(
        workspace,
        {
            "incident/store.py": _source(
                """
                class IncidentStore:
                    def __init__(self):
                        self._incidents = {}

                    def get(self, incident_id):
                        return self._incidents.get(incident_id)

                    def values(self):
                        return list(self._incidents.values())

                    def upsert(self, incident):
                        self._incidents[incident.incident_id] = incident
                        return True
                """
            ),
            "incident/service.py": _source(
                """
                from incident.constants import SEVERITY_ORDER


                def select_actionable(incidents, min_severity):
                    threshold = SEVERITY_ORDER[min_severity.upper()]
                    return [
                        item for item in incidents
                        if SEVERITY_ORDER[item.severity] >= threshold
                    ]
                """
            ),
        },
    )
    rejected = run_hidden_code(workspace, HIDDEN_STORE_SERVICE_CHECK)
    assert rejected.passed is False

    _write_files(
        workspace,
        {
            "incident/store.py": reference["incident/store.py"],
            "incident/service.py": reference["incident/service.py"],
        },
    )
    accepted = run_hidden_code(workspace, HIDDEN_STORE_SERVICE_CHECK)
    assert accepted.passed, accepted.stdout + accepted.stderr


def test_hidden_report_and_cli_checks_cover_release_contract(
    tmp_path: Path,
) -> None:
    fixture = build_long_running_fixture(FIXED_TOKENS)
    workspace = tmp_path / "workspace"
    materialize_long_running_fixture(fixture, workspace)
    reference = _reference_solution_files()
    _write_files(workspace, reference)
    _write_files(
        workspace,
        {
            "incident/report.py": _source(
                """
                def build_report(received, actionable):
                    return {
                        "schema_version": 1,
                        "total_received": len(received),
                        "incidents": [],
                    }
                """
            ),
            "incident/cli.py": _source(
                """
                import json


                def main(argv=None):
                    print(json.dumps([]))
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ),
        },
    )
    rejected = run_hidden_code(
        workspace,
        HIDDEN_SERIALIZER_REPORT_CLI_CHECK,
    )
    assert rejected.passed is False

    _write_files(
        workspace,
        {
            "incident/report.py": reference["incident/report.py"],
            "incident/cli.py": reference["incident/cli.py"],
        },
    )
    accepted = run_hidden_code(
        workspace,
        HIDDEN_SERIALIZER_REPORT_CLI_CHECK,
    )
    assert accepted.passed, accepted.stdout + accepted.stderr


def test_long_running_coverage_requires_all_minimum_metrics() -> None:
    minimum = _minimum_coverage_metrics()
    assert evaluate_long_running_coverage(minimum).passed is True

    below_minimum = {
        "agent_runs": 3,
        "llm_calls": 19,
        "tool_calls": 29,
        "distinct_files_read": 9,
        "production_files_changed": 4,
        "files_created": 0,
        "pagination_reads": 7,
        "compaction_bearing_requests": 2,
        "controlled_errors_observed": 1,
        "controlled_errors_recovered": 1,
        "shell_truncation_observed": False,
        "verification_calls": 0,
    }
    for field, value in below_minimum.items():
        evaluation = evaluate_long_running_coverage(
            replace(minimum, **{field: value})
        )
        assert evaluation.passed is False
        assert any(
            field in requirement
            for requirement in evaluation.failed_requirements
        )


def test_manifest_accounting_uses_only_real_sha256_changes() -> None:
    before = {
        f"incident/module_{index}.py": {
            "sha256": f"unchanged-{index}",
            "size_bytes": index,
        }
        for index in range(8)
    }
    unchanged = evidence_module.compare_manifests(before, dict(before))
    production_changed, created = runner_module.manifest_file_accounting(
        unchanged
    )
    assert production_changed == []
    assert created == []

    after = {path: dict(entry) for path, entry in before.items()}
    after["incident/module_2.py"]["sha256"] = "modified-2"
    after["incident/module_6.py"]["sha256"] = "modified-6"
    after["incident/report.py"] = {
        "sha256": "created-report",
        "size_bytes": 100,
    }
    changes = evidence_module.compare_manifests(before, after)
    production_changed, created = runner_module.manifest_file_accounting(
        changes
    )
    assert production_changed == [
        "incident/module_2.py",
        "incident/module_6.py",
    ]
    assert created == ["incident/report.py"]


def test_objective_status_distinguishes_not_evaluated_from_failure() -> None:
    assert runner_module.evaluation_result(evaluated=False, passed=False) is None
    assert runner_module.evaluation_result(evaluated=True, passed=False) is False
    assert runner_module.evaluation_result(evaluated=True, passed=True) is True
    assert runner_module._status(True) == "PASS"
    assert runner_module._status(False) == "FAIL"
    assert runner_module._status(None) == "NOT EVALUATED"
    assert runner_module.protected_integrity_result(
        evidence_module.WorkspaceChanges(protected_unchanged=("tests/x.py",)),
        comparison_evaluated=True,
    ) is True
    assert runner_module.protected_integrity_result(
        evidence_module.WorkspaceChanges(protected_modified=("tests/x.py",)),
        comparison_evaluated=True,
    ) is False
    assert runner_module.protected_integrity_result(
        evidence_module.WorkspaceChanges(),
        comparison_evaluated=False,
    ) is None


def test_run3_verification_does_not_evaluate_run4_verification_state() -> None:
    history = _tool_exchange(
        "early-verify",
        "verify_workspace",
        {},
        json.dumps({"ok": True}),
    )
    run3_record = runner_module.RunRecord(
        index=3,
        name="Migration Audit + Baseline Diagnostics",
        require_verified_completion=False,
        completed=False,
        history_start=0,
        history_end=len(history),
    )
    analysis = runner_module.analyze_history(
        history,
        [run3_record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    assert analysis["run3_verification_calls"] == 1
    assert analysis["run4_verification_calls"] == 0
    assert runner_module.verification_state_result(
        run4_verification_calls=analysis["run4_verification_calls"],
        run4_completed=False,
        verification_state_clean=True,
    ) is None
    assert runner_module.verification_state_result(
        run4_verification_calls=1,
        run4_completed=False,
        verification_state_clean=False,
    ) is False


def test_early_abort_report_and_summary_preserve_tristate(capsys) -> None:
    report = runner_module.build_failure_report(
        model="unit-test-model",
        started_at=datetime.now().astimezone(),
        started_perf=time.perf_counter(),
        error="preflight failed",
    )
    for field in (
        "visible_tests_passed",
        "hidden_checks_passed",
        "verification_state_clean",
        "release_token_recovered",
        "migration_key_recovered",
        "diag_tail_token_recovered",
    ):
        assert report[field] is None
    assert json.loads(json.dumps(report))["visible_tests_passed"] is None

    for field in (
        "release_token_present_in_final_llm_request",
        "migration_key_present_in_final_llm_request",
        "diag_tail_token_present_in_final_llm_request",
    ):
        assert report[field] is False
        assert type(report[field]) is bool

    for field in (
        "run4_met_first_mutation_deadline",
        "run4_did_not_repeat_complete_reads_before_first_mutation",
        "run4_did_not_rerun_failed_pytest_without_intervening_mutation",
        "run4_repair_loop_discipline_met",
        "action_discipline_target_met",
    ):
        assert report[field] is None
    assert report["run4_pytest_reruns_without_intervening_mutation"] == 0
    assert report["run4_read_only_responses_after_failed_test"] == 0
    assert report["run4_progress_guard_interventions"] == 0
    assert report["run4_successful_mutations_after_failed_test"] == 0
    assert report["run4_pending_failed_test_at_run_end"] is None
    assert report["action_discipline_warnings"] == []

    report["hidden_checks_passed"] = False
    report["protected_files_unchanged"] = True
    runner_module.print_summary(report)
    output = capsys.readouterr().out
    assert "Visible Host Re-check: NOT EVALUATED" in output
    assert "Hidden Behavior Checks: FAIL" in output
    assert "Protected Files Unchanged: PASS" in output
    assert "Verification State CLEAN: NOT EVALUATED" in output
    assert "Run 4 pytest Calls: 0" in output
    assert "Pending Failed Pytest At Run End: NOT EVALUATED" in output
    assert "Repair Loop Discipline: NOT EVALUATED" in output
    assert "ACTION DISCIPLINE: NOT EVALUATED" in output


def test_target_band_warning_does_not_fail_functional_success() -> None:
    coverage = evaluate_long_running_coverage(
        replace(_minimum_coverage_metrics(), llm_calls=45, tool_calls=65)
    )
    functional = evaluate_functional_challenge(
        _passing_functional_metrics()
    )
    assert coverage.passed is True
    assert functional.passed is True
    assert any("llm_calls" in warning for warning in coverage.warnings)
    assert any("tool_calls" in warning for warning in coverage.warnings)
    assert final_integrated_success(
        functional_passed=functional.passed,
        coverage_passed=coverage.passed,
    )


def test_runtime_target_is_reported_separately_from_functional_result() -> None:
    assert runtime_target_met(240.0) is False
    assert runtime_target_met(300.0) is True
    assert runtime_target_met(900.0) is True
    assert runtime_target_met(901.0) is False
    assert final_integrated_success(
        functional_passed=True,
        coverage_passed=True,
    ) is True


def test_functional_success_ignores_action_discipline_efficiency_warnings(
    capsys,
) -> None:
    dimensions = runner_module.evaluate_completion_dimensions(
        functional=evaluate_functional_challenge(
            _passing_functional_metrics()
        ),
        extra_functional_requirements={"functional_outcome": True},
        error=None,
        first_mutation_step=7,
        duplicate_complete_reads_before_first_mutation=11,
        pytest_reruns_without_intervening_mutation=2,
    )

    assert dimensions["functional_challenge_passed"] is True
    assert dimensions["functional_failed_requirements"] == []
    assert dimensions["action_discipline_target_met"] is False
    assert dimensions["run4_met_first_mutation_deadline"] is False
    assert (
        dimensions[
            "run4_did_not_repeat_complete_reads_before_first_mutation"
        ]
        is False
    )
    assert (
        dimensions[
            "run4_did_not_rerun_failed_pytest_without_intervening_mutation"
        ]
        is False
    )
    assert dimensions["run4_repair_loop_discipline_met"] is False
    assert len(dimensions["action_discipline_warnings"]) == 3

    coverage = evaluate_long_running_coverage(_minimum_coverage_metrics())
    integrated = final_integrated_success(
        functional_passed=dimensions["functional_challenge_passed"],
        coverage_passed=coverage.passed,
    )
    process_success = evidence_module.real_validation_process_success(
        final_integrated_success=integrated,
        evidence_preservation_passed=True,
    )
    assert coverage.passed is True
    assert integrated is True
    assert process_success is True

    report = {
        **dimensions,
        "run4_first_mutation_step": 7,
        "run4_reads_before_first_mutation": 26,
        "run4_duplicate_complete_reads_before_first_mutation": 11,
        "run4_pytest_reruns_without_intervening_mutation": 2,
        "run4_pytest_calls": 6,
        "run4_repository_scan_calls": 4,
        "long_running_coverage_passed": coverage.passed,
        "runtime_target_met": True,
        "evidence_preservation_passed": True,
        "final_integrated_success": integrated,
        "real_validation_process_success": process_success,
    }
    runner_module.print_summary(report)
    output = capsys.readouterr().out
    assert "First Mutation Discipline: WARNING" in output
    assert "Duplicate Read Discipline: WARNING" in output
    assert "Run 4 pytest Calls: 6" in output
    assert "Repair Loop Discipline: WARNING" in output
    assert "ACTION DISCIPLINE: WARNING" in output
    assert "FUNCTIONAL CHALLENGE: PASS" in output
    assert "FINAL INTEGRATED CHALLENGE: PASS" in output
    assert "REAL VALIDATION PROCESS: PASS" in output


def test_action_discipline_passes_for_prompt_target_behavior(capsys) -> None:
    dimensions = runner_module.evaluate_completion_dimensions(
        functional=evaluate_functional_challenge(
            _passing_functional_metrics()
        ),
        extra_functional_requirements={"functional_outcome": True},
        error=None,
        first_mutation_step=3,
        duplicate_complete_reads_before_first_mutation=0,
        pytest_reruns_without_intervening_mutation=0,
    )

    assert dimensions["functional_challenge_passed"] is True
    assert dimensions["action_discipline_target_met"] is True
    assert dimensions["run4_met_first_mutation_deadline"] is True
    assert (
        dimensions[
            "run4_did_not_repeat_complete_reads_before_first_mutation"
        ]
        is True
    )
    assert (
        dimensions[
            "run4_did_not_rerun_failed_pytest_without_intervening_mutation"
        ]
        is True
    )
    assert dimensions["run4_repair_loop_discipline_met"] is True
    assert dimensions["action_discipline_warnings"] == []

    runner_module.print_summary(
        {
            **dimensions,
            "run4_first_mutation_step": 3,
            "run4_duplicate_complete_reads_before_first_mutation": 0,
            "run4_pytest_reruns_without_intervening_mutation": 0,
            "run4_pytest_calls": 2,
            "action_discipline_target_met": True,
        }
    )
    output = capsys.readouterr().out
    assert "First Mutation Discipline: PASS" in output
    assert "Duplicate Read Discipline: PASS" in output
    assert "Run 4 pytest Calls: 2" in output
    assert "Repair Loop Discipline: PASS" in output
    assert "ACTION DISCIPLINE: PASS" in output


def test_hidden_failure_still_fails_functional_when_action_discipline_passes(
) -> None:
    functional = evaluate_functional_challenge(
        replace(_passing_functional_metrics(), hidden_checks_passed=False)
    )
    dimensions = runner_module.evaluate_completion_dimensions(
        functional=functional,
        extra_functional_requirements={"functional_outcome": True},
        error=None,
        first_mutation_step=3,
        duplicate_complete_reads_before_first_mutation=0,
        pytest_reruns_without_intervening_mutation=0,
    )

    assert dimensions["functional_challenge_passed"] is False
    assert dimensions["functional_failed_requirements"] == [
        "hidden_checks_passed"
    ]
    assert dimensions["action_discipline_target_met"] is True


def test_pagination_analysis_requires_next_start_line_chain(
    tmp_path: Path,
) -> None:
    correct_history = [
        *_read_exchange(
            "read-1",
            start_line=1,
            next_start_line=137,
            payload="first page",
        ),
        *_read_exchange(
            "read-2",
            start_line=137,
            next_start_line=511,
            payload="second page",
        ),
        *_read_exchange(
            "read-3",
            start_line=511,
            next_start_line=None,
            payload=f"MIGRATION_KEY={FIXED_TOKENS.migration_key}",
        ),
    ]
    correct = analyze_migration_pagination(
        correct_history,
        migration_key=FIXED_TOKENS.migration_key,
    )
    assert correct.reads == 3
    assert correct.start_lines == (1, 137, 511)
    assert correct.sequential_next_start_line is True
    assert correct.exhausted is True
    assert correct.migration_key_seen_in_tool_result is True

    skipped_history = [
        *_read_exchange(
            "read-1",
            start_line=1,
            next_start_line=137,
            payload="first page",
        ),
        *_read_exchange(
            "read-2",
            start_line=801,
            next_start_line=None,
            payload=f"MIGRATION_KEY={FIXED_TOKENS.migration_key}",
        ),
    ]
    skipped = analyze_migration_pagination(
        skipped_history,
        migration_key=FIXED_TOKENS.migration_key,
    )
    assert skipped.sequential_next_start_line is False

    repeated_history: list[dict[str, object]] = []
    for pass_index in range(2):
        for page_index, start_line in enumerate(
            range(1, 1_602, 200),
            start=1,
        ):
            repeated_history.extend(
                _read_exchange(
                    f"pass-{pass_index + 1}-page-{page_index}",
                    start_line=start_line,
                    next_start_line=(
                        start_line + 200 if page_index < 9 else None
                    ),
                    payload=(
                        f"MIGRATION_KEY={FIXED_TOKENS.migration_key}"
                        if page_index == 9
                        else "routine page"
                    ),
                )
            )
    repeated = analyze_migration_pagination(
        repeated_history,
        migration_key=FIXED_TOKENS.migration_key,
    )
    assert repeated.reads == 18
    assert repeated.exhausted is True
    assert repeated.sequential_next_start_line is False

    error_result = json.dumps(
        {
            "ok": False,
            "tool": "read_file",
            "error_type": "FileNotFoundError",
            "message": "File not found: config/legacy_policy.json",
        }
    )
    recovered_history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "missing",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps(
                            {"path": "config/legacy_policy.json"}
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "missing", "content": error_result},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "policy",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "config/policy.json"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "policy",
            "content": "[read_file]\npath: config/policy.json\n\n{}",
        },
    ]
    run_record = runner_module.RunRecord(
        index=2,
        name="Repository Reconnaissance",
        require_verified_completion=False,
        completed=True,
        history_start=0,
        history_end=len(recovered_history),
    )
    request_inputs = [
        {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "missing",
                    "content": error_result,
                }
            ]
        }
    ]
    recovered_analysis = runner_module.analyze_history(
        recovered_history,
        [run_record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        request_inputs,
    )
    assert recovered_analysis["missing_legacy_error_returned_to_llm"] is True
    assert recovered_analysis["missing_legacy_error_recovered"] is True

    batched_history = [
        {
            "role": "assistant",
            "tool_calls": [
                recovered_history[0]["tool_calls"][0],
                recovered_history[2]["tool_calls"][0],
            ],
        },
        recovered_history[1],
        recovered_history[3],
    ]
    batched_record = replace(
        run_record,
        history_end=len(batched_history),
    )
    batched_analysis = runner_module.analyze_history(
        batched_history,
        [batched_record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        request_inputs,
    )
    assert batched_analysis["missing_legacy_error_recovered"] is False

    assert runner_module._is_pytest_command("python -B -m pytest -q")
    assert not runner_module._is_pytest_command("echo pytest && exit 1")
    assert runner_module._is_diagnose_command(
        "python -B scripts/diagnose.py"
    )
    assert not runner_module._is_diagnose_command(
        "gc scripts/diagnose.py"
    )

    workspace = tmp_path / "workspace"
    absolute_diagnose = (workspace / "scripts/diagnose.py").resolve()
    unsafe_history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "read-diagnose",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps(
                            {"path": str(absolute_diagnose)}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read-diagnose",
            "content": "[read_file]\npath: scripts/diagnose.py\n\nsource",
        },
    ]
    unsafe_record = replace(run_record, history_end=len(unsafe_history))
    unsafe_analysis = runner_module.analyze_history(
        unsafe_history,
        [unsafe_record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
        workspace,
    )
    assert unsafe_analysis["distinct_file_paths_read"] == [
        "scripts/diagnose.py"
    ]
    assert unsafe_analysis["diagnose_script_not_read"] is False


def test_run3_pagination_ignores_run4_migration_read() -> None:
    run3_history = [
        *_read_exchange(
            "run3-read-1",
            start_line=1,
            next_start_line=201,
            payload="page one",
        ),
        *_read_exchange(
            "run3-read-2",
            start_line=201,
            next_start_line=None,
            payload=f"MIGRATION_KEY={FIXED_TOKENS.migration_key}",
        ),
    ]
    run4_history = _read_exchange(
        "run4-illegal-read",
        start_line=801,
        next_start_line=None,
        payload="illegal sample",
    )
    history = [*run3_history, *run4_history]
    records = [
        runner_module.RunRecord(
            index=3,
            name="Migration Audit + Baseline Diagnostics",
            require_verified_completion=False,
            completed=True,
            history_start=0,
            history_end=len(run3_history),
        ),
        runner_module.RunRecord(
            index=4,
            name="Release Repair",
            require_verified_completion=True,
            completed=False,
            history_start=len(run3_history),
            history_end=len(history),
        ),
    ]

    pagination = analyze_migration_pagination(
        runner_module._history_for_run(history, records, 3),
        migration_key=FIXED_TOKENS.migration_key,
    )
    analysis = runner_module.analyze_history(
        history,
        records,
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    assert pagination.reads == 2
    assert pagination.sequential_next_start_line is True
    assert pagination.exhausted is True
    assert pagination.migration_key_seen_in_tool_result is True
    assert analysis["run3_migration_reads"] == 2
    assert analysis["migration_reads_outside_run3"] == 1
    assert analysis["one_migration_read_per_response"] is True


def test_run4_migration_shell_violation_is_independent() -> None:
    run3_history = [
        *_read_exchange(
            "run3-read-1",
            start_line=1,
            next_start_line=None,
            payload=f"MIGRATION_KEY={FIXED_TOKENS.migration_key}",
        )
    ]
    run4_history = _tool_exchange(
        "run4-shell-search",
        "run_command",
        {"command": "python -c \"open('data/migration_notes.txt').read()\""},
        json.dumps({"exit_code": 0, "timed_out": False}),
    )
    history = [*run3_history, *run4_history]
    records = [
        runner_module.RunRecord(
            index=3,
            name="Migration Audit + Baseline Diagnostics",
            require_verified_completion=False,
            completed=True,
            history_start=0,
            history_end=len(run3_history),
        ),
        runner_module.RunRecord(
            index=4,
            name="Release Repair",
            require_verified_completion=True,
            completed=False,
            history_start=len(run3_history),
            history_end=len(history),
        ),
    ]
    pagination = analyze_migration_pagination(
        runner_module._history_for_run(history, records, 3),
        migration_key=FIXED_TOKENS.migration_key,
    )
    analysis = runner_module.analyze_history(
        history,
        records,
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    assert pagination.sequential_next_start_line is True
    assert pagination.exhausted is True
    assert analysis["migration_shell_search_violation"] is True
    assert analysis["run4_migration_shell_search_violation"] is True


def test_run3_diagnostic_evidence_survives_run4_abort() -> None:
    diagnostic_stdout = "\n".join(
        (
            f"DIAG_HEAD_TOKEN={FIXED_TOKENS.diag_head_token}",
            "DIAG_STATE=BLOCKED",
            "DIAG_ACTION=implement-release-contract",
            f"DIAG_TAIL_TOKEN={FIXED_TOKENS.diag_tail_token}",
        )
    )
    run3_history = _tool_exchange(
        "run3-diagnose",
        "run_command",
        {"command": "python scripts/diagnose.py"},
        json.dumps(
            {
                "exit_code": 0,
                "timed_out": False,
                "stdout": diagnostic_stdout,
                "stdout_truncated": True,
                "stdout_original_chars": 50_000,
            }
        ),
    )
    records = [
        runner_module.RunRecord(
            index=3,
            name="Migration Audit + Baseline Diagnostics",
            require_verified_completion=False,
            completed=True,
            history_start=0,
            history_end=len(run3_history),
        ),
        runner_module.RunRecord(
            index=4,
            name="Release Repair",
            require_verified_completion=True,
            completed=False,
            history_start=len(run3_history),
            history_end=len(run3_history),
            error="AgentMaxStepsError",
        ),
    ]
    analysis = runner_module.analyze_history(
        run3_history,
        records,
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    assert analysis["shell_truncation_observed"] is True
    assert analysis["diag_head_retained"] is True
    assert analysis["diag_tail_retained"] is True
    assert analysis["diag_middle_removed"] is True
    assert analysis["diagnostic_blocked_observed"] is True
    assert analysis["diagnostic_ready_observed"] is False
    assert runner_module.evaluation_result(
        evaluated=analysis["run4_diagnose_calls"] > 0,
        passed=analysis["diagnostic_ready_observed"],
    ) is None


def test_executed_untruncated_run3_diagnostic_is_failure_not_unevaluated() -> None:
    history = _tool_exchange(
        "run3-untruncated-diagnose",
        "run_command",
        {"command": "python scripts/diagnose.py"},
        json.dumps(
            {
                "exit_code": 0,
                "timed_out": False,
                "stdout": "DIAG_STATE=BLOCKED",
                "stdout_truncated": False,
                "stdout_original_chars": 100,
            }
        ),
    )
    record = runner_module.RunRecord(
        index=3,
        name="Migration Audit + Baseline Diagnostics",
        require_verified_completion=False,
        completed=True,
        history_start=0,
        history_end=len(history),
    )
    analysis = runner_module.analyze_history(
        history,
        [record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    status = runner_module.evaluation_result(
        evaluated=analysis["run3_diagnose_calls"] > 0,
        passed=analysis["shell_truncation_observed"],
    )
    assert analysis["shell_truncation_observed"] is False
    assert status is False


def test_run4_action_telemetry_records_third_response_mutation() -> None:
    history = [
        *_file_read_exchange("run4-read-parser", "incident/parser.py"),
        *_file_read_exchange("run4-read-tests", "tests/test_parser.py"),
        *_tool_exchange(
            "run4-edit-parser",
            "edit_file",
            {"path": "incident/parser.py"},
            "Updated incident/parser.py",
        ),
    ]
    record = runner_module.RunRecord(
        index=4,
        name="Release Repair",
        require_verified_completion=True,
        completed=False,
        history_start=0,
        history_end=len(history),
    )
    analysis = runner_module.analyze_history(
        history,
        [record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    assert analysis["run4_first_mutation_step"] == 3
    assert analysis["run4_edit_file_calls"] == 1
    assert analysis["run4_write_file_calls"] == 0
    assert analysis["run4_reads_before_first_mutation"] == 2


def test_run4_first_mutation_step_uses_responses_not_tool_ordinal() -> None:
    first_read = _file_read_exchange("batched-parser", "incident/parser.py")
    second_read = _file_read_exchange("batched-tests", "tests/test_parser.py")
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                first_read[0]["tool_calls"][0],
                second_read[0]["tool_calls"][0],
            ],
        },
        first_read[1],
        second_read[1],
        *_tool_exchange(
            "second-response-edit",
            "edit_file",
            {"path": "incident/parser.py"},
            "Updated incident/parser.py",
        ),
    ]
    record = runner_module.RunRecord(
        index=4,
        name="Release Repair",
        require_verified_completion=True,
        completed=False,
        history_start=0,
        history_end=len(history),
    )
    analysis = runner_module.analyze_history(
        history,
        [record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    assert analysis["run4_first_mutation_step"] == 2
    assert analysis["run4_reads_before_first_mutation"] == 2


def test_run4_action_telemetry_detects_no_mutation() -> None:
    history: list[dict[str, object]] = []
    for index in range(4):
        history.extend(
            _file_read_exchange(
                f"run4-read-{index}",
                f"incident/module_{index}.py",
            )
        )
    record = runner_module.RunRecord(
        index=4,
        name="Release Repair",
        require_verified_completion=True,
        completed=False,
        history_start=0,
        history_end=len(history),
    )
    analysis = runner_module.analyze_history(
        history,
        [record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )

    assert analysis["run4_first_mutation_step"] is None
    assert analysis["run4_edit_file_calls"] == 0
    assert analysis["run4_write_file_calls"] == 0
    assert analysis["run4_reads_before_first_mutation"] == 4


def test_run4_pytest_repair_loop_counts_only_unmutated_failure_reruns() -> None:
    compliant_history = [
        *_run4_pytest_exchange("pytest-fail", exit_code=1),
        *_tool_exchange(
            "repair-parser",
            "edit_file",
            {"path": "incident/parser.py"},
            "File edited successfully: incident/parser.py",
        ),
        *_run4_pytest_exchange("pytest-pass", exit_code=0),
    ]
    assert (
        _analyze_run4_history(compliant_history)[
            "run4_pytest_reruns_without_intervening_mutation"
        ]
        == 0
    )

    repeated_history = [
        *_run4_pytest_exchange("pytest-fail-1", exit_code=1),
        *_file_read_exchange("read-failure-context", "incident/parser.py"),
        *_run4_pytest_exchange("pytest-fail-2", exit_code=1),
        *_run4_pytest_exchange("pytest-pass-after-retries", exit_code=0),
    ]
    assert (
        _analyze_run4_history(repeated_history)[
            "run4_pytest_reruns_without_intervening_mutation"
        ]
        == 2
    )


def test_run4_pytest_repair_loop_respects_assistant_response_boundaries() -> None:
    failed = _run4_pytest_exchange("batched-fail", exit_code=1)
    repair = _tool_exchange(
        "batched-repair",
        "edit_file",
        {"path": "incident/service.py"},
        "File edited successfully: incident/service.py",
    )
    passed = _run4_pytest_exchange("batched-pass", exit_code=0)
    same_response_history = [
        {
            "role": "assistant",
            "tool_calls": [
                failed[0]["tool_calls"][0],
                repair[0]["tool_calls"][0],
                passed[0]["tool_calls"][0],
            ],
        },
        failed[1],
        repair[1],
        passed[1],
    ]
    assert (
        _analyze_run4_history(same_response_history)[
            "run4_pytest_reruns_without_intervening_mutation"
        ]
        == 1
    )

    later_response_history = [
        *failed,
        {
            "role": "assistant",
            "tool_calls": [
                repair[0]["tool_calls"][0],
                passed[0]["tool_calls"][0],
            ],
        },
        repair[1],
        passed[1],
    ]
    assert (
        _analyze_run4_history(later_response_history)[
            "run4_pytest_reruns_without_intervening_mutation"
        ]
        == 0
    )


def test_run4_pytest_repair_loop_ignores_nonfailures_and_nonfixed_commands(
) -> None:
    missing_result_call = _run4_pytest_exchange(
        "pytest-missing-result",
        exit_code=1,
    )[0]
    nonfailure_history = [
        *_run4_pytest_exchange("pytest-pass-1", exit_code=0),
        *_run4_pytest_exchange("pytest-pass-2", exit_code=0),
        *_run4_pytest_exchange(
            "pytest-timeout",
            exit_code=1,
            timed_out=True,
        ),
        *_run4_pytest_exchange("pytest-after-timeout", exit_code=0),
        *_run4_pytest_exchange(
            "nonfixed-pytest",
            exit_code=1,
            command="python -B -m pytest -q",
        ),
        *_run4_pytest_exchange("pytest-after-nonfixed", exit_code=0),
        *_tool_exchange(
            "pytest-malformed-result",
            "run_command",
            {"command": runner_module.CHALLENGE_PYTEST_COMMAND},
            "not-json",
        ),
        *_run4_pytest_exchange("pytest-after-malformed", exit_code=0),
        missing_result_call,
        *_run4_pytest_exchange("pytest-after-missing", exit_code=0),
    ]
    assert (
        _analyze_run4_history(nonfailure_history)[
            "run4_pytest_reruns_without_intervening_mutation"
        ]
        == 0
    )

    failed_mutation_result = json.dumps(
        {
            "ok": False,
            "tool": "edit_file",
            "error_type": "ValueError",
            "message": "old_text was not found",
        }
    )
    ineffective_mutation_histories = (
        [
            *_run4_pytest_exchange("fail-before-test-edit", exit_code=1),
            *_tool_exchange(
                "edit-visible-test",
                "edit_file",
                {"path": "tests/test_parser.py"},
                "File edited successfully: tests/test_parser.py",
            ),
            *_run4_pytest_exchange("retry-after-test-edit", exit_code=0),
        ],
        [
            *_run4_pytest_exchange("fail-before-bad-edit", exit_code=1),
            *_tool_exchange(
                "failed-production-edit",
                "edit_file",
                {"path": "incident/parser.py"},
                failed_mutation_result,
            ),
            *_run4_pytest_exchange("retry-after-bad-edit", exit_code=0),
        ],
        [
            *_run4_pytest_exchange("fail-before-no-op", exit_code=1),
            *_tool_exchange(
                "no-op-production-edit",
                "edit_file",
                {"path": "incident/parser.py"},
                (
                    "No changes made: old_text and new_text are identical in "
                    "incident/parser.py."
                ),
            ),
            *_run4_pytest_exchange("retry-after-no-op", exit_code=0),
        ],
    )
    for history, expected_successful_mutations in zip(
        ineffective_mutation_histories,
        (1, 0, 0),
        strict=True,
    ):
        analysis = _analyze_run4_history(history)
        assert (
            analysis[
                "run4_pytest_reruns_without_intervening_mutation"
            ]
            == 1
        )
        assert (
            analysis["run4_successful_mutations_after_failed_test"]
            == expected_successful_mutations
        )


def test_single_failed_pytest_with_32_read_only_responses_is_pending_warning(
    capsys,
) -> None:
    history = _run4_pytest_exchange("latest-single-failure", exit_code=1)
    for index in range(32):
        history.extend(
            _file_read_exchange(
                f"latest-read-only-{index}",
                f"incident/module_{index % 6}.py",
            )
        )

    analysis = _analyze_run4_history(history)
    assert analysis["run4_pytest_calls"] == 1
    assert analysis["run4_pytest_reruns_without_intervening_mutation"] == 0
    assert analysis["run4_read_only_responses_after_failed_test"] == 32
    assert analysis["run4_progress_guard_interventions"] == 0
    assert analysis["run4_successful_mutations_after_failed_test"] == 0
    assert analysis["run4_pending_failed_test_at_run_end"] is True

    dimensions = runner_module.evaluate_completion_dimensions(
        functional=evaluate_functional_challenge(
            _passing_functional_metrics()
        ),
        extra_functional_requirements={"functional_outcome": True},
        error=None,
        first_mutation_step=3,
        duplicate_complete_reads_before_first_mutation=0,
        pytest_reruns_without_intervening_mutation=0,
        pending_failed_test_at_run_end=True,
    )
    assert dimensions["functional_challenge_passed"] is True
    assert (
        dimensions[
            "run4_did_not_rerun_failed_pytest_without_intervening_mutation"
        ]
        is True
    )
    assert dimensions["run4_repair_loop_discipline_met"] is False
    assert dimensions["action_discipline_target_met"] is False
    assert any(
        "ended with a failed tracked pytest result" in warning
        for warning in dimensions["action_discipline_warnings"]
    )

    runner_module.print_summary(
        {
            **dimensions,
            **analysis,
            "run4_first_mutation_step": 3,
        }
    )
    output = capsys.readouterr().out
    assert "Read-only Responses After Failed Pytest: 32" in output
    assert "Pending Failed Pytest At Run End: YES" in output
    assert "Repair Loop Discipline: WARNING" in output


def test_failed_pytest_mutate_retest_pass_clears_pending_progress() -> None:
    history = [
        *_run4_pytest_exchange("cycle-fail", exit_code=1),
        *_file_read_exchange("cycle-diagnosis", "incident/parser.py"),
        *_tool_exchange(
            "cycle-repair",
            "edit_file",
            {"path": "incident/parser.py"},
            "File edited successfully: incident/parser.py",
        ),
        *_run4_pytest_exchange("cycle-pass", exit_code=0),
    ]
    analysis = _analyze_run4_history(history)

    assert analysis["run4_read_only_responses_after_failed_test"] == 1
    assert analysis["run4_progress_guard_interventions"] == 0
    assert analysis["run4_successful_mutations_after_failed_test"] == 1
    assert analysis["run4_pending_failed_test_at_run_end"] is False

    dimensions = runner_module.evaluate_completion_dimensions(
        functional=evaluate_functional_challenge(
            _passing_functional_metrics()
        ),
        extra_functional_requirements={"functional_outcome": True},
        error=None,
        first_mutation_step=3,
        duplicate_complete_reads_before_first_mutation=0,
        pytest_reruns_without_intervening_mutation=analysis[
            "run4_pytest_reruns_without_intervening_mutation"
        ],
        pending_failed_test_at_run_end=analysis[
            "run4_pending_failed_test_at_run_end"
        ],
    )
    assert dimensions["run4_repair_loop_discipline_met"] is True
    assert dimensions["action_discipline_target_met"] is True


def test_progress_guard_intervention_is_advisory_not_functional_failure(
) -> None:
    history = [
        *_run4_pytest_exchange("guard-fail", exit_code=1),
        *_file_read_exchange("guard-diagnosis", "incident/parser.py"),
        *_progress_guard_blocked_exchange(
            "guard-blocked-read",
            "read_file",
            {"path": "incident/service.py"},
        ),
        *_tool_exchange(
            "guard-repair",
            "write_file",
            {"path": "incident/report.py", "content": "repair"},
            "File written successfully: incident/report.py",
        ),
        *_run4_pytest_exchange("guard-pass", exit_code=0),
    ]
    analysis = _analyze_run4_history(history)

    assert analysis["run4_read_only_responses_after_failed_test"] == 2
    assert analysis["run4_progress_guard_interventions"] == 1
    assert analysis["run4_successful_mutations_after_failed_test"] == 1
    assert analysis["run4_pending_failed_test_at_run_end"] is False
    assert "incident/parser.py" in analysis["distinct_file_paths_read"]
    assert "incident/service.py" not in analysis["distinct_file_paths_read"]

    dimensions = runner_module.evaluate_completion_dimensions(
        functional=evaluate_functional_challenge(
            _passing_functional_metrics()
        ),
        extra_functional_requirements={"functional_outcome": True},
        error=None,
        first_mutation_step=3,
        duplicate_complete_reads_before_first_mutation=0,
        pytest_reruns_without_intervening_mutation=0,
        pending_failed_test_at_run_end=False,
    )
    assert dimensions["functional_challenge_passed"] is True
    assert dimensions["action_discipline_target_met"] is True
    assert dimensions["action_discipline_warnings"] == []


def test_run4_duplicate_read_analyzer_handles_allowed_rereads() -> None:
    def duplicate_count(history: list[dict[str, object]]) -> int:
        record = runner_module.RunRecord(
            index=4,
            name="Release Repair",
            require_verified_completion=True,
            completed=False,
            history_start=0,
            history_end=len(history),
        )
        analysis = runner_module.analyze_history(
            history,
            [record],
            FIXED_TOKENS,
            runner_module.build_run_specs(FIXED_TOKENS),
            [],
        )
        return analysis[
            "run4_duplicate_complete_reads_before_first_mutation"
        ]

    duplicate_history = [
        *_file_read_exchange("complete-1", "incident/parser.py"),
        *_file_read_exchange("complete-2", "incident/parser.py"),
    ]
    post_edit_history = [
        *_file_read_exchange("before-edit", "incident/parser.py"),
        *_tool_exchange(
            "edit-parser",
            "edit_file",
            {"path": "incident/parser.py"},
            "Updated incident/parser.py",
        ),
        *_file_read_exchange("after-edit", "incident/parser.py"),
    ]
    continuation_history = [
        *_file_read_exchange(
            "truncated-page",
            "incident/large.py",
            truncated_after=True,
            next_start_line=3,
        ),
        *_file_read_exchange(
            "continued-page",
            "incident/large.py",
            truncated_before=True,
        ),
    ]

    assert duplicate_count(duplicate_history) == 1
    assert duplicate_count(post_edit_history) == 0
    assert duplicate_count(continuation_history) == 0


def test_run2_phase_analysis_detects_only_real_protocol_violations() -> None:
    compliant_history = [
        *_file_read_exchange("readme", "README.md"),
        *_file_read_exchange(
            "large-first",
            "docs/large.md",
            truncated_after=True,
            next_start_line=3,
        ),
        *_file_read_exchange(
            "large-next",
            "docs/large.md",
            truncated_before=True,
        ),
    ]
    compliant_record = runner_module.RunRecord(
        index=2,
        name="Repository Reconnaissance",
        require_verified_completion=False,
        completed=True,
        history_start=0,
        history_end=len(compliant_history),
    )
    compliant = runner_module.analyze_history(
        compliant_history,
        [compliant_record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )
    for field in (
        "run2_run_command_calls",
        "run2_edit_file_calls",
        "run2_write_file_calls",
        "run2_verification_calls",
        "run2_migration_reads",
        "run2_diagnose_reads",
        "run2_test_reads",
        "run2_duplicate_complete_reads",
    ):
        assert compliant[field] == 0

    violating_history = [
        *_file_read_exchange("readme-1", "README.md"),
        *_file_read_exchange("readme-2", "README.md"),
        *_file_read_exchange("migration", "data/migration_notes.txt"),
        *_file_read_exchange("diagnose", "scripts/diagnose.py"),
        *_file_read_exchange("test-file", "tests/test_parser.py"),
        *_tool_exchange("list-tests", "list_directory", {"path": "tests"}, "x"),
        *_tool_exchange(
            "shell",
            "run_command",
            {"command": "python -c open('data/migration_notes.txt').read()"},
            json.dumps({"exit_code": 0, "timed_out": False}),
        ),
        *_tool_exchange("edit", "edit_file", {"path": "incident/parser.py"}, "x"),
        *_tool_exchange("write", "write_file", {"path": "incident/new.py"}, "x"),
        *_tool_exchange("verify", "verify_workspace", {}, "x"),
    ]
    violating_record = replace(
        compliant_record,
        history_end=len(violating_history),
    )
    violating = runner_module.analyze_history(
        violating_history,
        [violating_record],
        FIXED_TOKENS,
        runner_module.build_run_specs(FIXED_TOKENS),
        [],
    )
    assert violating["run2_run_command_calls"] == 1
    assert violating["run2_edit_file_calls"] == 1
    assert violating["run2_write_file_calls"] == 1
    assert violating["run2_verification_calls"] == 1
    assert violating["run2_migration_reads"] == 1
    assert violating["run2_diagnose_reads"] == 1
    assert violating["run2_test_reads"] == 2
    assert violating["run2_duplicate_complete_reads"] == 1
    assert violating["migration_reads_outside_run3"] == 1
    assert violating["migration_shell_search_violation"] is True
    assert violating["diagnose_script_not_read"] is False


def test_boundary_probe_is_counted_only_when_read_file_enforces_it() -> None:
    def analyze(tool_name: str) -> dict[str, object]:
        boundary_result = json.dumps(
            {
                "ok": False,
                "tool": tool_name,
                "error_type": "WorkspaceBoundaryError",
                "message": (
                    "Path escapes workspace: ../shared/global_defaults.json"
                ),
            }
        )
        history = _tool_exchange(
            f"boundary-{tool_name}",
            tool_name,
            {"path": "../shared/global_defaults.json"},
            boundary_result,
        )
        record = runner_module.RunRecord(
            index=2,
            name="Repository Reconnaissance",
            require_verified_completion=False,
            completed=True,
            history_start=0,
            history_end=len(history),
        )
        return runner_module.analyze_history(
            history,
            [record],
            FIXED_TOKENS,
            runner_module.build_run_specs(FIXED_TOKENS),
            [],
        )

    assert analyze("list_directory")["workspace_boundary_error_observed"] is False
    assert analyze("read_file")["workspace_boundary_error_observed"] is True


def test_compaction_metrics_detect_real_request_markers() -> None:
    requests = [
        [{"role": "user", "content": "ordinary request"}],
        [
            {
                "role": "user",
                "content": f"{PRIOR_CONTEXT_HEADER}\nprior digest",
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "clipped prefix\n"
                    f"{CURRENT_RUN_HEADER}\ncurrent digest"
                ),
            }
        ],
        [
            {"role": "user", "content": PRIOR_CONTEXT_HEADER},
            {"role": "user", "content": CURRENT_RUN_HEADER},
        ],
    ]
    metrics = analyze_compaction_requests(requests)
    assert metrics.requests_with_compaction == 3
    assert metrics.prior_compaction_requests == 2
    assert metrics.current_run_compaction_requests == 2

    response = object()

    class Delegate:
        def __init__(self) -> None:
            self.received_messages = None
            self.received_tools = None

        def chat(self, messages, tools=None):
            self.received_messages = messages
            self.received_tools = tools
            return response

    delegate = Delegate()
    recording = RecordingLLM(delegate, current_run_index=3)
    request_messages = requests[-1]
    request_tools = [{"type": "function", "function": {"name": "demo"}}]

    assert recording.chat(request_messages, tools=request_tools) is response
    assert delegate.received_messages is request_messages
    assert delegate.received_tools is request_tools
    assert recording.calls[0].run_index == 3
    assert recording.calls[0].has_prior_compaction is True
    assert recording.calls[0].has_current_run_compaction is True


def test_request_recording_llm_snapshots_the_current_run_index() -> None:
    response = object()

    class Delegate:
        model = "unit-test-model"

        def chat(self, messages, tools=None):
            return response

    messages = [{"role": "user", "content": "final request"}]
    tools = [{"type": "function", "function": {"name": "demo"}}]
    recording = runner_module.RequestRecordingLLM(Delegate())
    recording.set_run_index(4)

    assert recording.chat(messages, tools=tools) is response
    assert recording.request_inputs == [
        {
            "run_index": 4,
            "messages": messages,
            "tools": tools,
        }
    ]
    assert recording.request_inputs[0]["messages"] is not messages
    assert recording.request_inputs[0]["tools"] is not tools


def test_final_request_token_presence_uses_only_the_last_run4_request() -> None:
    request_inputs = [
        {
            "run_index": 3,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"RELEASE_TOKEN={FIXED_TOKENS.release_token}\n"
                        f"MIGRATION_KEY={FIXED_TOKENS.migration_key}\n"
                        f"DIAG_TAIL_TOKEN={FIXED_TOKENS.diag_tail_token}"
                    ),
                }
            ],
            "tools": [],
        },
        {
            "run_index": 4,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"RELEASE_TOKEN={FIXED_TOKENS.release_token}\n"
                        f"MIGRATION_KEY={FIXED_TOKENS.migration_key}\n"
                        f"DIAG_TAIL_TOKEN={FIXED_TOKENS.diag_tail_token}"
                    ),
                }
            ],
            "tools": [],
        },
        {
            "run_index": 4,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"RELEASE_TOKEN={FIXED_TOKENS.release_token}\n"
                        f"MIGRATION_KEY={FIXED_TOKENS.migration_key}"
                    ),
                }
            ],
            "tools": [],
        },
    ]

    presence = runner_module.final_request_token_presence(
        request_inputs,
        FIXED_TOKENS,
        run4_completed=True,
    )
    assert presence == {
        "release_token_present_in_final_llm_request": True,
        "migration_key_present_in_final_llm_request": True,
        "diag_tail_token_present_in_final_llm_request": False,
    }
    assert all(type(value) is bool for value in presence.values())

    serialized_presence = json.dumps(presence, sort_keys=True)
    for runtime_value in (
        FIXED_TOKENS.release_token,
        FIXED_TOKENS.migration_key,
        FIXED_TOKENS.diag_head_token,
        FIXED_TOKENS.diag_tail_token,
    ):
        assert runtime_value not in serialized_presence

    assert runner_module.final_request_token_presence(
        request_inputs,
        FIXED_TOKENS,
        run4_completed=False,
    ) == {
        "release_token_present_in_final_llm_request": False,
        "migration_key_present_in_final_llm_request": False,
        "diag_tail_token_present_in_final_llm_request": False,
    }


def test_exact_final_answer_token_recovery_requires_three_trailing_lines() -> None:
    exact_lines = [
        f"RELEASE_TOKEN={FIXED_TOKENS.release_token}",
        f"MIGRATION_KEY={FIXED_TOKENS.migration_key}",
        f"DIAG_TAIL_TOKEN={FIXED_TOKENS.diag_tail_token}",
    ]
    expected_success = {
        "release_token_recovered": True,
        "migration_key_recovered": True,
        "diag_tail_token_recovered": True,
    }
    exact_answer = "Release complete.\n" + "\n".join(exact_lines)

    assert runner_module.exact_final_answer_token_recovery(
        exact_answer,
        FIXED_TOKENS,
    ) == expected_success

    invalid_answers = (
        "Release complete.\n" + "\n".join(exact_lines[1:]),
        exact_answer.replace(exact_lines[1], "MIGRATION_KEY=wrong-value"),
        exact_answer.replace("MIGRATION_KEY=", "MIGRATION_TOKEN="),
        exact_answer.replace(
            exact_lines[0],
            "prefix-" + exact_lines[0] + "-suffix",
        ),
        "Release complete.\n"
        + "\n".join((exact_lines[1], exact_lines[0], exact_lines[2])),
        exact_answer + "\nTrailing commentary.",
    )
    for invalid_answer in invalid_answers:
        recovery = runner_module.exact_final_answer_token_recovery(
            invalid_answer,
            FIXED_TOKENS,
        )
        assert not all(recovery.values())


def test_challenge_contains_no_artificial_sleep() -> None:
    source = "\n".join(
        [
            inspect.getsource(challenge_module),
            inspect.getsource(evidence_module),
            inspect.getsource(telemetry_module),
            inspect.getsource(runner_module),
            generate_diagnose_source(FIXED_TOKENS),
        ]
    ).lower()
    assert "time.sleep" not in source
    assert re.search(r"\bsleep\s*\(", source) is None


def test_report_serialization_contains_required_evidence_fields(
    tmp_path: Path,
) -> None:
    metrics = replace(
        _minimum_coverage_metrics(),
        llm_calls=45,
        tool_calls=65,
    )
    functional = _passing_functional_metrics()
    request_metrics = (
        LLMCallMetric(
            index=1,
            run_index=1,
            duration_seconds=0.25,
            request_chars=1234,
            has_prior_compaction=False,
            has_current_run_compaction=True,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        ),
    )
    report = build_report_payload(
        model="unit-test-model",
        elapsed_seconds=240.0,
        coverage_metrics=metrics,
        functional_metrics=functional,
        llm_call_metrics=request_metrics,
        error=None,
    )
    required_fields = {
        "challenge",
        "model",
        "internal_elapsed_seconds",
        "runtime_target_seconds",
        "runtime_target_met",
        "agent_runs",
        "llm_calls",
        "tool_calls",
        "distinct_files_read",
        "production_files_changed",
        "files_created",
        "migration_pagination_reads",
        "requests_with_compaction",
        "controlled_errors_observed",
        "controlled_errors_recovered",
        "shell_truncation_observed",
        "verification_calls",
        "long_running_coverage_passed",
        "functional_challenge_passed",
        "final_integrated_success",
        "target_band_warnings",
        "llm_call_metrics",
        "error",
    }
    assert required_fields <= set(report)
    assert report["runtime_target_met"] is False
    assert report["long_running_coverage_passed"] is True
    assert report["functional_challenge_passed"] is True
    assert report["final_integrated_success"] is True
    assert "messages" not in report["llm_call_metrics"][0]

    serialized = json.dumps(report, sort_keys=True)
    for secret in (
        FIXED_TOKENS.release_token,
        FIXED_TOKENS.migration_key,
        FIXED_TOKENS.diag_head_token,
        FIXED_TOKENS.diag_tail_token,
        "sk-unit-test-secret",
    ):
        assert secret not in serialized
    assert "api_key" not in serialized.lower()
    assert "authorization" not in serialized.lower()

    injected = runner_module.redact_report_sensitive_values(
        {
            "created_paths": [
                f"artifacts/{FIXED_TOKENS.release_token}.txt",
                f"artifacts/{FIXED_TOKENS.migration_key}.txt",
                (tmp_path / "workspace" / "secret.txt").as_posix(),
            ],
            "error": (
                f"unexpected {FIXED_TOKENS.diag_head_token} and "
                f"{FIXED_TOKENS.diag_tail_token}"
            ),
        },
        FIXED_TOKENS,
        tmp_path,
    )
    redacted_serialized = json.dumps(injected, sort_keys=True)
    assert "<redacted>" in redacted_serialized
    assert all(
        secret not in redacted_serialized
        for secret in (
            FIXED_TOKENS.release_token,
            FIXED_TOKENS.migration_key,
            FIXED_TOKENS.diag_head_token,
            FIXED_TOKENS.diag_tail_token,
        )
    )
    assert str(tmp_path).casefold() not in redacted_serialized.casefold()
