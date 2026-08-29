import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from src.tools.files import ListDirectoryTool, ReadFileTool, WriteFileTool
from src.tools.path_utils import WorkspaceBoundaryError
from src.tools.shell import RunCommandTool


def test_read_file_allows_paths_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    safe_file = workspace / "safe.txt"
    safe_file.write_text("stage11-safe", encoding="utf-8")
    tool = ReadFileTool(workspace)

    assert tool.execute(path="safe.txt") == "stage11-safe"
    assert tool.execute(path=str(safe_file.resolve())) == "stage11-safe"


def test_read_file_rejects_parent_path_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("stage11-secret", encoding="utf-8")

    with pytest.raises(
        WorkspaceBoundaryError,
        match=r"^Path escapes workspace: \.\./secret\.txt$",
    ):
        ReadFileTool(workspace).execute(path="../secret.txt")

    assert secret_file.read_text(encoding="utf-8") == "stage11-secret"


def test_read_file_rejects_absolute_path_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("stage11-outside", encoding="utf-8")

    with pytest.raises(WorkspaceBoundaryError, match="Path escapes workspace"):
        ReadFileTool(workspace).execute(path=str(outside_file.resolve()))


def test_write_file_rejects_escape_without_external_side_effect(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_directory = tmp_path / "outside-created"
    outside_file = outside_directory / "created.txt"

    with pytest.raises(WorkspaceBoundaryError, match="Path escapes workspace"):
        WriteFileTool(workspace).execute(
            path="../outside-created/created.txt",
            content="danger",
        )

    assert not outside_file.exists()
    assert not outside_directory.exists()


def test_list_directory_rejects_parent_path_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside-visible.txt"
    outside_file.write_text("stage11-outside", encoding="utf-8")

    with pytest.raises(
        WorkspaceBoundaryError,
        match=r"^Path escapes workspace: \.\.$",
    ):
        ListDirectoryTool(workspace).execute(path="..")

    assert outside_file.read_text(encoding="utf-8") == "stage11-outside"


def test_normalized_path_inside_workspace_is_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "safe.txt").write_text("stage11-normalized", encoding="utf-8")

    result = ReadFileTool(workspace).execute(path="src/../safe.txt")

    assert result == "stage11-normalized"


def test_run_command_starts_in_canonical_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    noncanonical_workspace = workspace / "nested" / ".."
    parts = [sys.executable, "-c", "import os; print(os.getcwd())"]
    command = (
        subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)
    )

    result = json.loads(
        RunCommandTool(noncanonical_workspace).execute(command=command)
    )

    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert result["stderr"] == ""
    assert Path(result["stdout"].strip()).resolve() == workspace.resolve()
