import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.tool_execution import ToolExecutionResult


COMMAND_TOOL_NAME = "run_command"
PROGRESS_GUARD_ERROR_TYPE = "ProgressGuardBlocked"


class ProgressGuardState(str, Enum):
    NORMAL = "normal"
    TEST_FAILED_DIAGNOSIS_ALLOWED = "test_failed_diagnosis_allowed"
    MUTATION_REQUIRED = "mutation_required"
    RETEST_ALLOWED = "retest_allowed"


def _normalized_nonempty_values(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple.")

    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} must contain only non-empty strings."
            )
        normalized.append(value.strip())
    return tuple(normalized)


@dataclass(frozen=True)
class ProgressGuardConfig:
    tracked_commands: tuple[str, ...]
    mutation_tool_names: tuple[str, ...]
    diagnosis_responses: int = 1

    def __post_init__(self) -> None:
        tracked_commands = _normalized_nonempty_values(
            self.tracked_commands,
            "tracked_commands",
        )
        mutation_tool_names = _normalized_nonempty_values(
            self.mutation_tool_names,
            "mutation_tool_names",
        )
        if COMMAND_TOOL_NAME in mutation_tool_names:
            raise ValueError(
                "run_command cannot be a Progress Guard mutation tool."
            )
        if (
            isinstance(self.diagnosis_responses, bool)
            or not isinstance(self.diagnosis_responses, int)
            or self.diagnosis_responses < 0
        ):
            raise ValueError(
                "diagnosis_responses must be a non-negative integer."
            )

        object.__setattr__(self, "tracked_commands", tracked_commands)
        object.__setattr__(
            self,
            "mutation_tool_names",
            mutation_tool_names,
        )


class ProgressGuard:
    def __init__(self, config: ProgressGuardConfig) -> None:
        self.config = config
        self.state = ProgressGuardState.NORMAL
        self.pending_command: str | None = None
        self.pending_failure_workspace_revision: int | None = None
        self._diagnosis_responses_remaining = 0
        self._failure_cycle = 0
        self._active_diagnosis_cycle: int | None = None

    def begin_response(self) -> None:
        if self.state is ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED:
            self._active_diagnosis_cycle = self._failure_cycle
        else:
            self._active_diagnosis_cycle = None

    def finish_response(self) -> None:
        active_cycle = self._active_diagnosis_cycle
        self._active_diagnosis_cycle = None
        if (
            active_cycle is None
            or self.state
            is not ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED
            or active_cycle != self._failure_cycle
        ):
            return

        self._diagnosis_responses_remaining -= 1
        if self._diagnosis_responses_remaining <= 0:
            self.state = ProgressGuardState.MUTATION_REQUIRED

    def allows(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        if self.state is ProgressGuardState.NORMAL:
            return True

        if self._is_mutation_tool(tool_name):
            return True

        is_pending_retest = self._is_pending_retest(tool_name, arguments)
        if self.state is ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED:
            return not self._is_tracked_command(tool_name, arguments)

        if self.state is ProgressGuardState.RETEST_ALLOWED:
            return is_pending_retest

        return False

    def blocked_result(self, tool_name: str) -> ToolExecutionResult:
        if self.state is ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED:
            message = (
                "A tracked repair check is currently failing. This retest "
                "was not executed because a successful workspace-changing "
                "mutation is required before retesting."
            )
        elif self.state is ProgressGuardState.RETEST_ALLOWED:
            message = (
                "A tracked repair check is still pending after workspace "
                "progress. This tool call was not executed. Only another "
                "allowed mutation or the exact pending tracked retest is "
                "allowed."
            )
        else:
            message = (
                "A tracked repair check is still failing at the current "
                "workspace state. The diagnostic response allowance has "
                "already been consumed, and this tool call was not executed. "
                "A successful workspace-changing mutation with an allowed "
                "mutation tool is required before further inspection or "
                "retesting."
            )

        return ToolExecutionResult(
            tool_name=tool_name,
            content=json.dumps(
                {
                    "ok": False,
                    "tool": tool_name,
                    "error_type": PROGRESS_GUARD_ERROR_TYPE,
                    "message": message,
                    "policy_blocked": True,
                },
                ensure_ascii=False,
            ),
            execution_ok=False,
            error_type=PROGRESS_GUARD_ERROR_TYPE,
        )

    def observe_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolExecutionResult,
        workspace_revision_before: int,
        workspace_revision_after: int,
    ) -> None:
        if self._is_tracked_command(tool_name, arguments):
            exit_code = self._completed_exit_code(result)
            if exit_code is not None:
                if exit_code == 0:
                    command = self._normalized_command(arguments)
                    if (
                        self.pending_command is None
                        or command == self.pending_command
                    ):
                        self._clear_pending_failure()
                else:
                    self._record_failure(
                        arguments,
                        workspace_revision_after,
                    )
                return

        if (
            self.pending_command is not None
            and self._is_mutation_tool(tool_name)
            and self._is_successful_mutation(
                result,
                workspace_revision_before,
                workspace_revision_after,
            )
        ):
            self.state = ProgressGuardState.RETEST_ALLOWED

    def _record_failure(
        self,
        arguments: dict[str, Any],
        workspace_revision: int,
    ) -> None:
        self.pending_command = self._normalized_command(arguments)
        self.pending_failure_workspace_revision = workspace_revision
        self._diagnosis_responses_remaining = (
            self.config.diagnosis_responses
        )
        self._failure_cycle += 1
        if self._diagnosis_responses_remaining > 0:
            self.state = (
                ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED
            )
        else:
            self.state = ProgressGuardState.MUTATION_REQUIRED

    def _clear_pending_failure(self) -> None:
        self.state = ProgressGuardState.NORMAL
        self.pending_command = None
        self.pending_failure_workspace_revision = None
        self._diagnosis_responses_remaining = 0

    def _is_pending_retest(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        return (
            tool_name == COMMAND_TOOL_NAME
            and self.pending_command is not None
            and self._normalized_command(arguments) == self.pending_command
        )

    def _is_mutation_tool(self, tool_name: str) -> bool:
        return (
            tool_name != COMMAND_TOOL_NAME
            and tool_name in self.config.mutation_tool_names
        )

    def _is_tracked_command(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        return (
            tool_name == COMMAND_TOOL_NAME
            and self._normalized_command(arguments)
            in self.config.tracked_commands
        )

    @staticmethod
    def _normalized_command(arguments: dict[str, Any]) -> str | None:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        return command.strip()

    @staticmethod
    def _completed_exit_code(result: ToolExecutionResult) -> int | None:
        if not result.execution_ok:
            return None
        try:
            payload = json.loads(result.content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict) or payload.get("timed_out") is not False:
            return None
        exit_code = payload.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            return None
        return exit_code

    def _is_successful_mutation(
        self,
        result: ToolExecutionResult,
        workspace_revision_before: int,
        workspace_revision_after: int,
    ) -> bool:
        failure_revision = self.pending_failure_workspace_revision
        return (
            result.execution_ok
            and failure_revision is not None
            and workspace_revision_after > workspace_revision_before
            and workspace_revision_after > failure_revision
            and not result.content.lstrip().startswith("No changes made:")
        )
