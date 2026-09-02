import json
import posixpath
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.context_compaction import (
    CURRENT_RUN_HEADER,
    PRIOR_CONTEXT_HEADER,
    estimate_messages_size,
)


RUNTIME_TARGET_SECONDS = (300, 900)
LLM_CALL_TARGET_BAND = (20, 40)
TOOL_CALL_TARGET_BAND = (30, 60)
MIGRATION_NOTES_PATH = "data/migration_notes.txt"


@dataclass(frozen=True)
class LLMCallMetric:
    index: int
    run_index: int
    duration_seconds: float
    request_chars: int
    has_prior_compaction: bool
    has_current_run_compaction: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LongRunningMetrics:
    agent_runs: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    distinct_files_read: int = 0
    production_files_changed: int = 0
    files_created: int = 0
    pagination_reads: int = 0
    compaction_bearing_requests: int = 0
    controlled_errors_observed: int = 0
    controlled_errors_recovered: int = 0
    shell_truncation_observed: bool = False
    verification_calls: int = 0


@dataclass(frozen=True)
class FunctionalChallengeMetrics:
    agent_runs_completed: bool = False
    initial_visible_tests_failed: bool = False
    missing_legacy_error_observed: bool = False
    workspace_boundary_error_observed: bool = False
    controlled_errors_recovered: int = 0
    migration_pagination_valid: bool = False
    migration_key_seen_in_tool_result: bool = False
    baseline_nonzero_was_normal_tool_result: bool = False
    shell_truncation_observed: bool = False
    diag_head_retained: bool = False
    diag_tail_retained: bool = False
    diag_middle_removed: bool = False
    diagnose_script_not_read: bool = False
    post_agent_visible_recheck_passed: bool = False
    hidden_checks_passed: bool = False
    protected_files_unchanged: bool = False
    verification_calls: int = 0
    verification_state_clean: bool = False
    release_token_recovered: bool = False
    migration_key_recovered: bool = False
    diag_tail_token_recovered: bool = False
    report_module_present: bool = False
    report_module_implemented: bool = False


@dataclass(frozen=True)
class ChallengeEvaluation:
    passed: bool
    failed_requirements: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MigrationPaginationMetrics:
    reads: int
    start_lines: tuple[int, ...]
    next_start_lines: tuple[int | None, ...]
    sequential_next_start_line: bool
    exhausted: bool
    migration_key_seen_in_tool_result: bool

    @property
    def valid(self) -> bool:
        return self.sequential_next_start_line and self.exhausted


@dataclass(frozen=True)
class CompactionRequestMetrics:
    requests_with_compaction: int
    prior_compaction_requests: int
    current_run_compaction_requests: int


def _content_has_marker(messages: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    has_prior = False
    has_current = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        has_prior = has_prior or PRIOR_CONTEXT_HEADER in content
        has_current = has_current or CURRENT_RUN_HEADER in content
    return has_prior, has_current


def _usage_integer(usage: Any, field: str) -> int | None:
    value = (
        usage.get(field)
        if isinstance(usage, Mapping)
        else getattr(usage, field, None)
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _provider_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return None, None, None
    return (
        _usage_integer(usage, "prompt_tokens"),
        _usage_integer(usage, "completion_tokens"),
        _usage_integer(usage, "total_tokens"),
    )


class RecordingLLM:
    """Forward LLM requests unchanged while recording bounded metadata."""

    def __init__(self, client: Any, *, current_run_index: int = 0) -> None:
        self.client = client
        self.current_run_index = current_run_index
        self.metrics: list[LLMCallMetric] = []

    @property
    def model(self) -> Any:
        return getattr(self.client, "model", None)

    @property
    def calls(self) -> tuple[LLMCallMetric, ...]:
        return tuple(self.metrics)

    def set_run_index(self, run_index: int) -> None:
        if isinstance(run_index, bool) or not isinstance(run_index, int):
            raise ValueError("run_index must be an integer.")
        if run_index < 0:
            raise ValueError("run_index must not be negative.")
        self.current_run_index = run_index

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        request_chars = estimate_messages_size(messages)
        has_prior, has_current = _content_has_marker(messages)
        call_index = len(self.metrics) + 1
        started = time.perf_counter()
        response: Any = None
        completed = False
        try:
            response = self.client.chat(messages, tools=tools)
            completed = True
            return response
        finally:
            duration = time.perf_counter() - started
            usage = (
                _provider_usage(response)
                if completed
                else (None, None, None)
            )
            self.metrics.append(
                LLMCallMetric(
                    index=call_index,
                    run_index=self.current_run_index,
                    duration_seconds=duration,
                    request_chars=request_chars,
                    has_prior_compaction=has_prior,
                    has_current_run_compaction=has_current,
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    total_tokens=usage[2],
                )
            )


TelemetryLLM = RecordingLLM


def analyze_compaction_requests(
    requests: Sequence[Sequence[Mapping[str, Any]] | Mapping[str, Any]],
) -> CompactionRequestMetrics:
    requests_with_compaction = 0
    prior_compaction_requests = 0
    current_run_compaction_requests = 0

    for request in requests:
        if isinstance(request, Mapping):
            candidate = request.get("messages", ())
        else:
            candidate = request
        if not isinstance(candidate, Sequence) or isinstance(
            candidate,
            (str, bytes),
        ):
            continue
        messages = [item for item in candidate if isinstance(item, Mapping)]
        has_prior, has_current = _content_has_marker(messages)
        requests_with_compaction += int(has_prior or has_current)
        prior_compaction_requests += int(has_prior)
        current_run_compaction_requests += int(has_current)

    return CompactionRequestMetrics(
        requests_with_compaction=requests_with_compaction,
        prior_compaction_requests=prior_compaction_requests,
        current_run_compaction_requests=current_run_compaction_requests,
    )


def _normalize_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return posixpath.normpath(value.replace("\\", "/")).casefold()


def _tool_arguments(call: Mapping[str, Any]) -> dict[str, Any] | None:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return arguments


def _positive_line(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _metadata_boolean(value: str | None) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _read_result_metadata(
    content: Any,
) -> tuple[int | None, bool | None, str] | None:
    if not isinstance(content, str):
        return None
    normalized = content.replace("\r\n", "\n")
    header, separator, payload = normalized.partition("\n\n")
    lines = header.splitlines()
    if separator != "\n\n" or not lines or lines[0] != "[read_file]":
        return None

    metadata: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition(":")
        if not separator:
            return None
        metadata[key.strip()] = value.strip()

    next_value = metadata.get("next_start_line")
    if next_value == "none":
        next_start_line: int | None = None
    else:
        try:
            next_start_line = int(next_value or "")
        except ValueError:
            return None
        if next_start_line < 1:
            return None

    truncated_after = _metadata_boolean(metadata.get("truncated_after"))
    if truncated_after is None:
        return None
    return next_start_line, truncated_after, payload


def analyze_migration_pagination(
    history: Sequence[Mapping[str, Any]],
    *,
    migration_key: str,
    migration_path: str = MIGRATION_NOTES_PATH,
) -> MigrationPaginationMetrics:
    result_messages: dict[str, tuple[int, Any]] = {}
    duplicate_result_ids: set[str] = set()
    for message_index, message in enumerate(history):
        call_id = message.get("tool_call_id")
        if message.get("role") != "tool" or not isinstance(call_id, str):
            continue
        if call_id in result_messages:
            duplicate_result_ids.add(call_id)
            continue
        result_messages[call_id] = (message_index, message.get("content"))

    expected_path = _normalize_path(migration_path)
    reads = 0
    starts: list[int] = []
    next_starts: list[int | None] = []
    truncated_after_values: list[bool] = []
    key_seen = False
    structurally_valid = True
    seen_call_ids: set[str] = set()

    for message_index, message in enumerate(history):
        if message.get("role") != "assistant":
            continue
        migration_calls_in_message = 0
        calls = message.get("tool_calls") or ()
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            structurally_valid = False
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            if function.get("name") != "read_file":
                continue
            arguments = _tool_arguments(call)
            if (
                arguments is None
                or _normalize_path(arguments.get("path")) != expected_path
            ):
                continue

            migration_calls_in_message += 1
            reads += 1
            call_id = call.get("id")
            if not isinstance(call_id, str) or call_id in seen_call_ids:
                structurally_valid = False
                continue
            seen_call_ids.add(call_id)

            start_line = _positive_line(arguments.get("start_line", 1))
            result_entry = result_messages.get(call_id)
            if (
                start_line is None
                or result_entry is None
                or call_id in duplicate_result_ids
                or result_entry[0] <= message_index
            ):
                structurally_valid = False
                continue

            parsed_result = _read_result_metadata(result_entry[1])
            if parsed_result is None:
                structurally_valid = False
                continue
            next_start_line, truncated_after, payload = parsed_result
            starts.append(start_line)
            next_starts.append(next_start_line)
            truncated_after_values.append(truncated_after)
            key_seen = key_seen or migration_key in payload

        if migration_calls_in_message > 1:
            structurally_valid = False

    complete_observations = (
        reads > 0
        and len(starts) == reads
        and len(next_starts) == reads
        and len(truncated_after_values) == reads
    )
    sequential = structurally_valid and complete_observations and starts[0] == 1
    if sequential:
        for index, start_line in enumerate(starts):
            next_start_line = next_starts[index]
            if next_start_line is not None and next_start_line <= start_line:
                sequential = False
                break
            if index > 0 and start_line != next_starts[index - 1]:
                sequential = False
                break

    exhausted = bool(
        structurally_valid
        and complete_observations
        and next_starts[-1] is None
        and truncated_after_values[-1] is False
    )
    return MigrationPaginationMetrics(
        reads=reads,
        start_lines=tuple(starts),
        next_start_lines=tuple(next_starts),
        sequential_next_start_line=sequential,
        exhausted=exhausted,
        migration_key_seen_in_tool_result=key_seen,
    )


def evaluate_long_running_coverage(
    metrics: LongRunningMetrics,
) -> ChallengeEvaluation:
    requirements = (
        ("agent_runs == 4", metrics.agent_runs == 4),
        ("llm_calls >= 20", metrics.llm_calls >= 20),
        ("tool_calls >= 30", metrics.tool_calls >= 30),
        ("distinct_files_read >= 10", metrics.distinct_files_read >= 10),
        (
            "production_files_changed >= 3",
            metrics.production_files_changed >= 3,
        ),
        ("pagination_reads >= 8", metrics.pagination_reads >= 8),
        (
            "compaction_bearing_requests >= 3",
            metrics.compaction_bearing_requests >= 3,
        ),
        (
            "controlled_errors_observed >= 2",
            metrics.controlled_errors_observed >= 2,
        ),
        (
            "controlled_errors_recovered >= 2",
            metrics.controlled_errors_recovered >= 2,
        ),
        (
            "shell_truncation_observed is true",
            metrics.shell_truncation_observed is True,
        ),
        ("verification_calls >= 1", metrics.verification_calls >= 1),
    )
    failed = tuple(name for name, passed in requirements if not passed)
    warnings: list[str] = []
    if not LLM_CALL_TARGET_BAND[0] <= metrics.llm_calls <= LLM_CALL_TARGET_BAND[1]:
        warnings.append(
            "llm_calls="
            f"{metrics.llm_calls} is outside target band "
            f"{LLM_CALL_TARGET_BAND[0]}-{LLM_CALL_TARGET_BAND[1]}"
        )
    if not TOOL_CALL_TARGET_BAND[0] <= metrics.tool_calls <= TOOL_CALL_TARGET_BAND[1]:
        warnings.append(
            "tool_calls="
            f"{metrics.tool_calls} is outside target band "
            f"{TOOL_CALL_TARGET_BAND[0]}-{TOOL_CALL_TARGET_BAND[1]}"
        )
    return ChallengeEvaluation(
        passed=not failed,
        failed_requirements=failed,
        warnings=tuple(warnings),
    )


def evaluate_functional_challenge(
    metrics: FunctionalChallengeMetrics,
) -> ChallengeEvaluation:
    requirements = (
        ("agent_runs_completed", metrics.agent_runs_completed),
        ("initial_visible_tests_failed", metrics.initial_visible_tests_failed),
        (
            "missing_legacy_error_observed",
            metrics.missing_legacy_error_observed,
        ),
        (
            "workspace_boundary_error_observed",
            metrics.workspace_boundary_error_observed,
        ),
        (
            "controlled_errors_recovered >= 2",
            metrics.controlled_errors_recovered >= 2,
        ),
        ("migration_pagination_valid", metrics.migration_pagination_valid),
        (
            "migration_key_seen_in_tool_result",
            metrics.migration_key_seen_in_tool_result,
        ),
        (
            "baseline_nonzero_was_normal_tool_result",
            metrics.baseline_nonzero_was_normal_tool_result,
        ),
        ("shell_truncation_observed", metrics.shell_truncation_observed),
        ("diag_head_retained", metrics.diag_head_retained),
        ("diag_tail_retained", metrics.diag_tail_retained),
        ("diag_middle_removed", metrics.diag_middle_removed),
        ("diagnose_script_not_read", metrics.diagnose_script_not_read),
        (
            "post_agent_visible_recheck_passed",
            metrics.post_agent_visible_recheck_passed,
        ),
        ("hidden_checks_passed", metrics.hidden_checks_passed),
        ("protected_files_unchanged", metrics.protected_files_unchanged),
        ("verification_calls >= 1", metrics.verification_calls >= 1),
        ("verification_state_clean", metrics.verification_state_clean),
        ("release_token_recovered", metrics.release_token_recovered),
        ("migration_key_recovered", metrics.migration_key_recovered),
        ("diag_tail_token_recovered", metrics.diag_tail_token_recovered),
        ("report_module_present", metrics.report_module_present),
        ("report_module_implemented", metrics.report_module_implemented),
    )
    failed = tuple(name for name, passed in requirements if not passed)
    return ChallengeEvaluation(
        passed=not failed,
        failed_requirements=failed,
    )


def runtime_target_met(
    elapsed_seconds: float,
    *,
    lower_seconds: float = RUNTIME_TARGET_SECONDS[0],
    upper_seconds: float = RUNTIME_TARGET_SECONDS[1],
) -> bool:
    return lower_seconds <= elapsed_seconds <= upper_seconds


def final_integrated_success(
    *,
    functional_passed: bool,
    coverage_passed: bool,
) -> bool:
    return functional_passed and coverage_passed


def _aggregate_provider_usage(
    metrics: Sequence[LLMCallMetric],
) -> tuple[bool, int | None, int | None, int | None]:
    if not metrics or any(
        metric.prompt_tokens is None
        or metric.completion_tokens is None
        or metric.total_tokens is None
        for metric in metrics
    ):
        return False, None, None, None
    return (
        True,
        sum(
            metric.prompt_tokens
            for metric in metrics
            if metric.prompt_tokens is not None
        ),
        sum(
            metric.completion_tokens
            for metric in metrics
            if metric.completion_tokens is not None
        ),
        sum(
            metric.total_tokens
            for metric in metrics
            if metric.total_tokens is not None
        ),
    )


def build_report_payload(
    *,
    model: str,
    elapsed_seconds: float,
    coverage_metrics: LongRunningMetrics,
    functional_metrics: FunctionalChallengeMetrics,
    llm_call_metrics: Sequence[LLMCallMetric],
    error: str | None,
) -> dict[str, Any]:
    request_metrics = tuple(llm_call_metrics)
    coverage = evaluate_long_running_coverage(coverage_metrics)
    functional = evaluate_functional_challenge(functional_metrics)
    integrated = final_integrated_success(
        functional_passed=functional.passed,
        coverage_passed=coverage.passed,
    )
    provider_usage = _aggregate_provider_usage(request_metrics)
    prior_compaction_requests = sum(
        metric.has_prior_compaction for metric in request_metrics
    )
    current_run_compaction_requests = sum(
        metric.has_current_run_compaction for metric in request_metrics
    )

    return {
        "challenge": "Long-Horizon Repository Repair",
        "scenario": "Incident Triage Service v2 Release Recovery",
        "model": model,
        "internal_elapsed_seconds": elapsed_seconds,
        "runtime_target_seconds": list(RUNTIME_TARGET_SECONDS),
        "runtime_target_met": runtime_target_met(elapsed_seconds),
        "agent_runs": coverage_metrics.agent_runs,
        "llm_calls": coverage_metrics.llm_calls,
        "llm_wall_time_seconds": sum(
            metric.duration_seconds for metric in request_metrics
        ),
        "provider_usage_available": provider_usage[0],
        "prompt_tokens": provider_usage[1],
        "completion_tokens": provider_usage[2],
        "total_tokens": provider_usage[3],
        "cumulative_request_chars": sum(
            metric.request_chars for metric in request_metrics
        ),
        "maximum_request_chars": max(
            (metric.request_chars for metric in request_metrics),
            default=0,
        ),
        "tool_calls": coverage_metrics.tool_calls,
        "distinct_files_read": coverage_metrics.distinct_files_read,
        "production_files_changed": coverage_metrics.production_files_changed,
        "files_created": coverage_metrics.files_created,
        "migration_pagination_reads": coverage_metrics.pagination_reads,
        "sequential_next_start_line": (
            functional_metrics.migration_pagination_valid
        ),
        "migration_key_seen_in_tool_result": (
            functional_metrics.migration_key_seen_in_tool_result
        ),
        "requests_with_compaction": (
            coverage_metrics.compaction_bearing_requests
        ),
        "prior_compaction_requests": prior_compaction_requests,
        "current_run_compaction_requests": current_run_compaction_requests,
        "controlled_errors_observed": (
            coverage_metrics.controlled_errors_observed
        ),
        "controlled_errors_recovered": (
            coverage_metrics.controlled_errors_recovered
        ),
        "missing_legacy_error_observed": (
            functional_metrics.missing_legacy_error_observed
        ),
        "workspace_boundary_error_observed": (
            functional_metrics.workspace_boundary_error_observed
        ),
        "baseline_nonzero_observed_as_normal_tool_result": (
            functional_metrics.baseline_nonzero_was_normal_tool_result
        ),
        "shell_truncation_observed": (
            coverage_metrics.shell_truncation_observed
        ),
        "diag_head_retained": functional_metrics.diag_head_retained,
        "diag_tail_retained": functional_metrics.diag_tail_retained,
        "diag_middle_removed": functional_metrics.diag_middle_removed,
        "diagnose_script_not_read": functional_metrics.diagnose_script_not_read,
        "initial_visible_tests_failed": (
            functional_metrics.initial_visible_tests_failed
        ),
        "visible_tests_passed": (
            functional_metrics.post_agent_visible_recheck_passed
        ),
        "hidden_checks_passed": functional_metrics.hidden_checks_passed,
        "protected_files_unchanged": (
            functional_metrics.protected_files_unchanged
        ),
        "verification_calls": coverage_metrics.verification_calls,
        "verification_state_clean": (
            functional_metrics.verification_state_clean
        ),
        "release_token_recovered": (
            functional_metrics.release_token_recovered
        ),
        "migration_key_recovered": (
            functional_metrics.migration_key_recovered
        ),
        "diag_tail_token_recovered": (
            functional_metrics.diag_tail_token_recovered
        ),
        "report_module_present": functional_metrics.report_module_present,
        "report_module_implemented": (
            functional_metrics.report_module_implemented
        ),
        "long_running_coverage_passed": coverage.passed,
        "functional_challenge_passed": functional.passed,
        "final_integrated_success": integrated,
        "coverage_failed_requirements": list(coverage.failed_requirements),
        "functional_failed_requirements": list(
            functional.failed_requirements
        ),
        "target_band_warnings": list(coverage.warnings),
        "llm_call_metrics": [asdict(metric) for metric in request_metrics],
        "error": error,
    }
