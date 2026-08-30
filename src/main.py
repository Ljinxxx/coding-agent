import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.agent import (
    Agent,
    AgentContextLimitError,
    AgentLLMError,
    AgentMaxStepsError,
    AgentResponseError,
)
from src.llm import LLMClient
from src.tools.files import (
    DEFAULT_READ_MAX_LINES,
    DEFAULT_READ_MAX_OUTPUT_CHARS,
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from src.tools.registry import ToolRegistry
from src.tools.shell import DEFAULT_SHELL_MAX_OUTPUT_CHARS, RunCommandTool
from src.tools.verification import VerifyWorkspaceTool


SYSTEM_PROMPT = (
    "You are a coding agent working in the configured local workspace. "
    "Use the provided tools to inspect files, make changes, and run commands. "
    "Prefer edit_file for localized changes to existing files; use write_file "
    "for new files or complete replacements. "
    "read_file supports 1-based start_line and max_lines; follow its metadata "
    "to continue reading large files. "
    "Do not guess file contents or tool results. Return a concise final answer "
    "after completing the task."
)
VERIFICATION_PROMPT = (
    " The verification checks are controlled by the host. After changing the "
    "workspace, run verify_workspace successfully before returning the final "
    "answer; do not attempt to replace the configured checks."
)
DEFAULT_MAX_CONTEXT_CHARS = 60_000
COMPACTION_TRIGGER_NUMERATOR = 3
COMPACTION_RATIO_DENOMINATOR = 4


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_agent(
    workspace: Path,
    llm_client: Any | None = None,
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    compact_context: bool = False,
    verification_commands: Sequence[str] | None = None,
) -> Agent:
    workspace_root = Path(workspace).expanduser().resolve(strict=False)
    registry = ToolRegistry()
    registry.register(ListDirectoryTool(workspace_root))
    registry.register(
        ReadFileTool(
            workspace_root,
            default_max_lines=DEFAULT_READ_MAX_LINES,
            max_output_chars=DEFAULT_READ_MAX_OUTPUT_CHARS,
        )
    )
    registry.register(EditFileTool(workspace_root))
    registry.register(WriteFileTool(workspace_root))
    registry.register(
        RunCommandTool(
            workspace_root,
            max_output_chars=DEFAULT_SHELL_MAX_OUTPUT_CHARS,
        )
    )

    system_prompt = SYSTEM_PROMPT
    verification_tool_name: str | None = None
    if verification_commands:
        registry.register(
            VerifyWorkspaceTool(workspace_root, verification_commands)
        )
        system_prompt += VERIFICATION_PROMPT
        verification_tool_name = VerifyWorkspaceTool.name

    client = llm_client if llm_client is not None else LLMClient()
    compaction_trigger_chars = (
        max(
            1,
            max_context_chars
            * COMPACTION_TRIGGER_NUMERATOR
            // COMPACTION_RATIO_DENOMINATOR,
        )
        if compact_context
        else None
    )
    max_compaction_chars = (
        max(1, max_context_chars // COMPACTION_RATIO_DENOMINATOR)
        if compact_context
        else None
    )
    return Agent(
        client,
        registry,
        system_prompt=system_prompt,
        verbose=True,
        max_steps=20,
        max_context_chars=max_context_chars,
        compaction_trigger_chars=compaction_trigger_chars,
        max_compaction_chars=max_compaction_chars,
        verification_tool_name=verification_tool_name,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Coding Agent.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root (default: current working directory).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=_positive_integer,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Character-based context budget sent to the model "
            f"(default: {DEFAULT_MAX_CONTEXT_CHARS})."
        ),
    )
    parser.add_argument(
        "--verify",
        dest="verification_commands",
        action="append",
        metavar="COMMAND",
        help=(
            "Host-controlled verification command. Repeat for multiple checks; "
            "verification is disabled when omitted."
        ),
    )
    parser.add_argument(
        "--compact-context",
        action="store_true",
        help=(
            "Enable deterministic layered context compaction using "
            "host-derived budgets."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace = args.workspace.expanduser().resolve(strict=False)
    if not workspace.is_dir():
        print(f"Workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    task = input("Task: ").strip()
    if not task:
        print("Task cannot be empty.", file=sys.stderr)
        return 2

    try:
        agent = build_agent(
            workspace,
            max_context_chars=args.max_context_chars,
            compact_context=args.compact_context,
            verification_commands=args.verification_commands,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    try:
        final_answer = agent.run(task)
    except (
        AgentContextLimitError,
        AgentLLMError,
        AgentMaxStepsError,
        AgentResponseError,
    ) as error:
        print(f"Coding Agent failed: {error}", file=sys.stderr)
        return 1

    print("\nAssistant:")
    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
