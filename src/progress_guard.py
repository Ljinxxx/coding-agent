import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any

from src.tool_execution import ToolExecutionResult
from src.tools.path_utils import (
    WorkspaceBoundaryError,
    resolve_workspace_path,
)


COMMAND_TOOL_NAME = "run_command"
VERIFICATION_TOOL_NAME = "verify_workspace"
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
    initial_pending_command: str | None = None
    initial_diagnosis_responses: int = 0
    allowed_mutation_paths: tuple[str, ...] | None = None

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
        initial_pending_command = self.initial_pending_command
        if initial_pending_command is not None:
            if (
                not isinstance(initial_pending_command, str)
                or not initial_pending_command.strip()
            ):
                raise ValueError(
                    "initial_pending_command must be a non-empty string or None."
                )
            initial_pending_command = initial_pending_command.strip()
            if initial_pending_command not in tracked_commands:
                raise ValueError(
                    "initial_pending_command must be one of tracked_commands."
                )
        if (
            isinstance(self.initial_diagnosis_responses, bool)
            or not isinstance(self.initial_diagnosis_responses, int)
            or self.initial_diagnosis_responses < 0
        ):
            raise ValueError(
                "initial_diagnosis_responses must be a non-negative integer."
            )
        if (
            initial_pending_command is None
            and self.initial_diagnosis_responses != 0
        ):
            raise ValueError(
                "initial_diagnosis_responses requires initial_pending_command."
            )
        allowed_mutation_paths = self.allowed_mutation_paths
        if allowed_mutation_paths is not None:
            if not isinstance(allowed_mutation_paths, tuple):
                raise ValueError(
                    "allowed_mutation_paths must be a tuple or None."
                )
            normalized_paths: list[str] = []
            for path in allowed_mutation_paths:
                if not isinstance(path, str) or not path.strip():
                    raise ValueError(
                        "allowed_mutation_paths must contain only non-empty "
                        "strings."
                    )
                normalized_paths.append(path.strip())
            allowed_mutation_paths = tuple(normalized_paths)

        object.__setattr__(self, "tracked_commands", tracked_commands)
        object.__setattr__(
            self,
            "mutation_tool_names",
            mutation_tool_names,
        )
        object.__setattr__(
            self,
            "initial_pending_command",
            initial_pending_command,
        )
        object.__setattr__(
            self,
            "allowed_mutation_paths",
            allowed_mutation_paths,
        )


class ProgressGuard:
    def __init__(
        self,
        config: ProgressGuardConfig,
        *,
        initial_workspace_revision: int = 0,
        workspace: Path | None = None,
    ) -> None:
        if (
            isinstance(initial_workspace_revision, bool)
            or not isinstance(initial_workspace_revision, int)
            or initial_workspace_revision < 0
        ):
            raise ValueError(
                "initial_workspace_revision must be a non-negative integer."
            )
        self.config = config
        self.state = ProgressGuardState.NORMAL
        self.pending_command: str | None = None
        self.pending_failure_workspace_revision: int | None = None
        self._diagnosis_responses_remaining = 0
        self._failure_cycle = 0
        self._active_diagnosis_cycle: int | None = None
        self._initial_pending_failure = False
        self._workspace = (
            Path(workspace).resolve(strict=False)
            if workspace is not None
            else None
        )
        self._allowed_mutation_paths = self._normalize_allowed_paths()

        if config.initial_pending_command is not None:
            self._activate_pending_failure(
                config.initial_pending_command,
                initial_workspace_revision,
                config.initial_diagnosis_responses,
                initial=True,
            )

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
        if (
            self._is_mutation_tool(tool_name)
            and not self._mutation_path_is_allowed(arguments)
        ):
            return False

        if self.state is ProgressGuardState.NORMAL:
            return True

        if self._is_mutation_tool(tool_name):
            return True

        is_pending_retest = self._is_pending_retest(tool_name, arguments)
        if self.state is ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED:
            if self._initial_pending_failure and tool_name in {
                COMMAND_TOOL_NAME,
                VERIFICATION_TOOL_NAME,
            }:
                return False
            return not self._is_tracked_command(tool_name, arguments)

        if self.state is ProgressGuardState.RETEST_ALLOWED:
            return is_pending_retest

        return False

    def blocked_result(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        candidate_arguments = arguments or {}
        if (
            self._is_mutation_tool(tool_name)
            and not self._mutation_path_is_allowed(candidate_arguments)
        ):
            allowed = "\n".join(
                f"- {path}" for path in self._allowed_mutation_paths or ()
            )
            message = (
                "This guarded run is restricted to the declared mutation "
                "surface. The requested path is outside the allowed mutation "
                "surface and was not modified. Allowed mutation paths:"
            )
            if allowed:
                message = f"{message}\n{allowed}"
        elif (
            self.state is ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED
            and self._initial_pending_failure
        ):
            message = (
                "A tracked repair check is already pending from an earlier "
                "run. This command or verification call was not executed. "
                "Use the bounded inspection allowance or make a successful "
                "workspace-changing mutation before running commands."
            )
        elif self.state is ProgressGuardState.TEST_FAILED_DIAGNOSIS_ALLOWED:
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
            self._initial_pending_failure = False
            self.state = ProgressGuardState.RETEST_ALLOWED

    def _record_failure(
        self,
        arguments: dict[str, Any],
        workspace_revision: int,
    ) -> None:
        command = self._normalized_command(arguments)
        if command is None:
            return
        self._activate_pending_failure(
            command,
            workspace_revision,
            self.config.diagnosis_responses,
            initial=False,
        )

    def _activate_pending_failure(
        self,
        command: str,
        workspace_revision: int,
        diagnosis_responses: int,
        *,
        initial: bool,
    ) -> None:
        self.pending_command = command
        self.pending_failure_workspace_revision = workspace_revision
        self._diagnosis_responses_remaining = diagnosis_responses
        self._failure_cycle += 1
        self._initial_pending_failure = initial
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
        self._initial_pending_failure = False

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

    def _normalize_allowed_paths(self) -> tuple[str, ...] | None:
        configured = self.config.allowed_mutation_paths
        if configured is None:
            return None
        if self._workspace is None:
            raise ValueError(
                "allowed_mutation_paths requires mutation tools with a "
                "workspace."
            )

        normalized: list[str] = []
        for path in configured:
            portable_path = path.replace("\\", "/")
            requested = Path(portable_path)
            windows_requested = PureWindowsPath(path)
            if (
                requested.is_absolute()
                or requested.drive
                or windows_requested.is_absolute()
                or windows_requested.drive
            ):
                raise ValueError(
                    "allowed_mutation_paths must contain workspace-relative "
                    "paths."
                )
            try:
                canonical = self._canonical_workspace_relative_path(path)
            except WorkspaceBoundaryError as error:
                raise ValueError(
                    "allowed_mutation_paths must stay within the workspace."
                ) from error
            if canonical not in normalized:
                normalized.append(canonical)
        return tuple(normalized)

    def _mutation_path_is_allowed(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        allowed = self._allowed_mutation_paths
        if allowed is None:
            return True
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return False
        try:
            canonical = self._canonical_workspace_relative_path(path)
        except WorkspaceBoundaryError:
            return False
        return canonical in allowed

    def _canonical_workspace_relative_path(self, path: str) -> str:
        if self._workspace is None:
            raise ValueError("Progress Guard workspace is not configured.")
        target = resolve_workspace_path(
            self._workspace,
            path.replace("\\", "/"),
        )
        relative = target.relative_to(self._workspace).as_posix()
        return os.path.normcase(relative).replace("\\", "/")

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
