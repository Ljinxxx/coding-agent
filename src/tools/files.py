from pathlib import Path
from typing import Any

from src.tools.base import BaseTool


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List the direct contents of a directory in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path relative to the workspace.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def execute(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        target = self.workspace / path

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
                "description": "File path relative to the workspace.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def execute(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        target = self.workspace / path

        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        return target.read_text(encoding="utf-8")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a UTF-8 text file in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def execute(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        content = str(kwargs["content"])
        target = self.workspace / path

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return f"File written successfully: {path}"
