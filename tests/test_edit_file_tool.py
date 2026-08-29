import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.tools.files import EditFileTool
from src.tools.path_utils import WorkspaceBoundaryError
from src.tools.registry import ToolRegistry


def test_edit_file_schema_and_registry_metadata(tmp_path: Path) -> None:
    tool = EditFileTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)

    assert tool.name == "edit_file"
    assert tool.mutates_workspace is True
    assert tool.parameters == {
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
                "description": (
                    "Replacement text. Use an empty string to delete."
                ),
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }
    assert "literal" in tool.description
    assert registry.names() == ["edit_file"]
    assert registry.get("edit_file") is tool
    assert registry.schemas() == [tool.to_schema()]


def test_edit_file_replaces_one_exact_match_and_preserves_other_text(
    tmp_path: Path,
) -> None:
    target = tmp_path / "calculator.py"
    target.write_text(
        "def add(a, b):\n    return a - b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    result = EditFileTool(tmp_path).execute(
        path="calculator.py",
        old_text="def add(a, b):\n    return a - b",
        new_text="def add(a, b):\n    return a + b",
    )

    assert "replaced exactly one occurrence" in result
    assert target.read_text(encoding="utf-8") == (
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"
    )


def test_edit_file_rejects_missing_match_without_changing_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    original = "alpha\nbeta\n"
    target.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="old_text was not found"):
        EditFileTool(tmp_path).execute(
            path="notes.txt",
            old_text="gamma",
            new_text="delta",
        )

    assert target.read_text(encoding="utf-8") == original


def test_edit_file_rejects_multiple_matches_then_accepts_more_context(
    tmp_path: Path,
) -> None:
    target = tmp_path / "functions.py"
    original = (
        "def first():\n    return value\n\n"
        "def second():\n    return value\n"
    )
    target.write_text(original, encoding="utf-8")
    tool = EditFileTool(tmp_path)

    with pytest.raises(ValueError, match="multiple locations.*surrounding context"):
        tool.execute(
            path="functions.py",
            old_text="return value",
            new_text="return updated",
        )

    assert target.read_text(encoding="utf-8") == original

    tool.execute(
        path="functions.py",
        old_text="def second():\n    return value",
        new_text="def second():\n    return updated",
    )
    assert target.read_text(encoding="utf-8") == (
        "def first():\n    return value\n\n"
        "def second():\n    return updated\n"
    )


def test_edit_file_rejects_overlapping_match_locations(tmp_path: Path) -> None:
    target = tmp_path / "overlap.txt"
    target.write_text("aaa", encoding="utf-8")

    with pytest.raises(ValueError, match="multiple locations"):
        EditFileTool(tmp_path).execute(
            path="overlap.txt",
            old_text="aa",
            new_text="b",
        )

    assert target.read_text(encoding="utf-8") == "aaa"


def test_edit_file_rejects_empty_old_text_without_changing_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="old_text must not be empty"):
        EditFileTool(tmp_path).execute(
            path="notes.txt",
            old_text="",
            new_text="replacement",
        )

    assert target.read_text(encoding="utf-8") == "keep me"


def test_edit_file_allows_empty_new_text_to_delete_unique_match(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before\nremove me\nafter\n", encoding="utf-8")

    EditFileTool(tmp_path).execute(
        path="notes.txt",
        old_text="remove me\n",
        new_text="",
    )

    assert target.read_text(encoding="utf-8") == "before\nafter\n"


def test_edit_file_identical_text_is_no_op_without_disk_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("unique text", encoding="utf-8")

    def unexpected_write_text(
        _path: Path,
        _data: str,
        **_kwargs: Any,
    ) -> int:
        raise AssertionError("no-op edit must not rewrite the file")

    monkeypatch.setattr(Path, "write_text", unexpected_write_text)

    result = EditFileTool(tmp_path).execute(
        path="notes.txt",
        old_text="unique text",
        new_text="unique text",
    )

    assert "No changes made" in result


def test_edit_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File not found: missing.txt"):
        EditFileTool(tmp_path).execute(
            path="missing.txt",
            old_text="old",
            new_text="new",
        )


def test_edit_file_rejects_directory_path(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(IsADirectoryError, match="Not a file: folder"):
        EditFileTool(tmp_path).execute(
            path="folder",
            old_text="old",
            new_text="new",
        )


def test_edit_file_handles_unicode_text(tmp_path: Path) -> None:
    target = tmp_path / "unicode.txt"
    target.write_text("你好，旧世界！\n", encoding="utf-8")

    EditFileTool(tmp_path).execute(
        path="unicode.txt",
        old_text="旧世界",
        new_text="新世界 🌏",
    )

    assert target.read_text(encoding="utf-8") == "你好，新世界 🌏！\n"


def test_edit_file_rejects_parent_traversal_and_preserves_outside_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")

    with pytest.raises(WorkspaceBoundaryError, match="Path escapes workspace"):
        EditFileTool(workspace).execute(
            path="../outside.txt",
            old_text="outside",
            new_text="changed",
        )

    assert outside.read_text(encoding="utf-8") == "outside secret"


def test_edit_file_rejects_absolute_outside_path_and_preserves_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")

    with pytest.raises(WorkspaceBoundaryError, match="Path escapes workspace"):
        EditFileTool(workspace).execute(
            path=str(outside.resolve()),
            old_text="outside",
            new_text="changed",
        )

    assert outside.read_text(encoding="utf-8") == "outside secret"


def test_edit_file_rejects_symlink_escape_and_preserves_outside_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside = outside_directory / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    link = workspace / "linked"
    try:
        link.symlink_to(outside_directory, target_is_directory=True)
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"Symbolic links are unavailable: {symlink_error}")
        junction = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside_directory)],
            capture_output=True,
            text=True,
            check=False,
        )
        if junction.returncode != 0:
            pytest.skip(
                "Symbolic links and directory junctions are unavailable: "
                f"{symlink_error}; {junction.stderr.strip()}"
            )

    try:
        with pytest.raises(
            WorkspaceBoundaryError,
            match="Path escapes workspace",
        ):
            EditFileTool(workspace).execute(
                path="linked/outside.txt",
                old_text="outside",
                new_text="changed",
            )
    finally:
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            link.rmdir()

    assert outside.read_text(encoding="utf-8") == "outside secret"
