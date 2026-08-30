from pathlib import Path
from typing import Any

from src.tools.base import BaseTool
from src.tools.path_utils import resolve_workspace_path


DEFAULT_READ_MAX_LINES = 200
DEFAULT_READ_MAX_OUTPUT_CHARS = 20_000


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List the direct contents of a directory in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path within the workspace.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve(strict=False)

    def execute(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        target = resolve_workspace_path(self.workspace, path)

        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries = sorted(target.iterdir(), key=lambda entry: entry.name)
        return "\n".join(
            f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries
        )


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a bounded, pageable UTF-8 text window from a workspace file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path within the workspace.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based line number at which reading starts.",
                "minimum": 1,
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of consecutive lines to read.",
                "minimum": 1,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path,
        *,
        default_max_lines: int = DEFAULT_READ_MAX_LINES,
        max_output_chars: int = DEFAULT_READ_MAX_OUTPUT_CHARS,
    ) -> None:
        self.workspace = Path(workspace).resolve(strict=False)
        self.default_max_lines = _positive_integer(
            default_max_lines,
            "default_max_lines",
        )
        self.max_output_chars = _positive_integer(
            max_output_chars,
            "max_output_chars",
        )

    def execute(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        target = resolve_workspace_path(self.workspace, path)

        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        start_line = _positive_integer(
            kwargs.get("start_line", 1),
            "start_line",
        )
        max_lines = _positive_integer(
            kwargs.get("max_lines", self.default_max_lines),
            "max_lines",
        )
        window_end = start_line + max_lines
        total_lines = 0
        complete_end: int | None = None
        first_unreturned_line: int | None = None
        original_selected_chars = 0
        payload_chars = 0
        payload_parts: list[str] = []
        output_closed = False
        char_truncated = False
        partial_line = False

        with target.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                total_lines = line_number
                if not start_line <= line_number < window_end:
                    continue

                original_selected_chars += len(line)
                if output_closed:
                    continue

                remaining_chars = self.max_output_chars - payload_chars
                if len(line) <= remaining_chars:
                    payload_parts.append(line)
                    payload_chars += len(line)
                    complete_end = line_number
                    continue

                char_truncated = True
                output_closed = True
                if complete_end is None:
                    payload_parts.append(line[:remaining_chars])
                    payload_chars += remaining_chars
                    partial_line = True
                else:
                    first_unreturned_line = line_number

        payload = "".join(payload_parts)
        if complete_end is None:
            line_range = "none"
        else:
            line_range = f"{start_line}-{complete_end} of {total_lines}"

        truncated_before = total_lines > 0 and start_line > 1
        truncated_after = (
            partial_line
            or first_unreturned_line is not None
            or (complete_end is not None and complete_end < total_lines)
        )
        if partial_line:
            next_start_line: int | str = "none"
            notice = (
                "first selected line exceeds character budget; partial line "
                "returned; line-based continuation unavailable"
            )
        elif first_unreturned_line is not None:
            next_start_line = first_unreturned_line
            notice = (
                "read payload stopped before an incomplete line; continue at "
                "next_start_line"
            )
        else:
            next_start_line = (
                complete_end + 1
                if complete_end is not None and complete_end < total_lines
                else "none"
            )
            notice = "none"
        header = "\n".join(
            [
                "[read_file]",
                f"path: {path}",
                f"lines: {line_range}",
                f"total_lines: {total_lines}",
                f"truncated_before: {str(truncated_before).lower()}",
                f"truncated_after: {str(truncated_after).lower()}",
                f"char_truncated: {str(char_truncated).lower()}",
                f"partial_line: {str(partial_line).lower()}",
                f"original_selected_chars: {original_selected_chars}",
                f"next_start_line: {next_start_line}",
                f"notice: {notice}",
            ]
        )
        return f"{header}\n\n{payload}"


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Replace exactly one matching literal text block in an existing "
        "UTF-8 file within the workspace."
    )
    mutates_workspace = True
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path within the workspace.",
            },
            "old_text": {
                "type": "string",
                "description": (
                    "Exact literal text to replace. It must be non-empty and "
                    "match exactly once; include surrounding context when needed."
                ),
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text. Use an empty string to delete.",
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve(strict=False)

    def execute(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        old_text = str(kwargs["old_text"])
        new_text = str(kwargs["new_text"])

        target = resolve_workspace_path(self.workspace, path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        content = target.read_text(encoding="utf-8")
        if not old_text:
            raise ValueError("old_text must not be empty.")

        first_index = content.find(old_text)
        if first_index == -1:
            raise ValueError(
                f"old_text was not found in the target file: {path}"
            )
        if content.find(old_text, first_index + 1) != -1:
            raise ValueError(
                "old_text matched multiple locations in the target file: "
                f"{path}. Provide more surrounding context in old_text."
            )
        if old_text == new_text:
            return (
                "No changes made: old_text and new_text are identical in "
                f"{path}."
            )

        updated_content = (
            content[:first_index]
            + new_text
            + content[first_index + len(old_text) :]
        )
        target.write_text(updated_content, encoding="utf-8")
        return (
            f"File edited successfully: {path} "
            "(replaced exactly one occurrence)."
        )


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a UTF-8 text file in the workspace."
    mutates_workspace = True
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path within the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve(strict=False)

    def execute(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        content = str(kwargs["content"])
        target = resolve_workspace_path(self.workspace, path)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return f"File written successfully: {path}"
