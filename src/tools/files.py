from pathlib import Path
from typing import Any

from src.tools.base import BaseTool
from src.tools.path_utils import resolve_workspace_path


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
    description = "Read a UTF-8 text file from the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path within the workspace.",
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
            raise FileNotFoundError(f"File not found: {path}")

        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        return target.read_text(encoding="utf-8")


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
