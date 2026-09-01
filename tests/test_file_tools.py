from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.files import ListDirectoryTool, ReadFileTool, WriteFileTool


def read_payload(result: str) -> str:
    return result.split("\n\n", 1)[1]


def test_list_directory(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "src").mkdir()

    result = ListDirectoryTool(tmp_path).execute(path=".")

    assert result == "hello.txt\nsrc/"


def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello stage4", encoding="utf-8")

    result = ReadFileTool(tmp_path).execute(path="hello.txt")

    assert read_payload(result) == "hello stage4"


def test_write_file(tmp_path: Path) -> None:
    target = tmp_path / "output.txt"
    assert not target.exists()

    result = WriteFileTool(tmp_path).execute(
        path="output.txt",
        content="stage4 write test",
    )

    assert result == "File written successfully: output.txt"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "stage4 write test"


def test_write_file_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "output.txt"
    target.write_text("old content", encoding="utf-8")

    result = WriteFileTool(tmp_path).execute(
        path="output.txt",
        content="new content",
    )

    assert result == "File written successfully: output.txt"
    assert target.read_text(encoding="utf-8") == "new content"


def test_write_file_skips_identical_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "output.txt"
    target.write_text("already current", encoding="utf-8")

    with patch.object(
        Path,
        "write_text",
        autospec=True,
        side_effect=AssertionError("identical content must not be rewritten"),
    ) as write_text:
        result = WriteFileTool(tmp_path).execute(
            path="output.txt",
            content="already current",
        )

    write_text.assert_not_called()
    assert result.startswith("No changes made:")
    assert target.read_text(encoding="utf-8") == "already current"


def test_write_then_read_file(tmp_path: Path) -> None:
    WriteFileTool(tmp_path).execute(path="hello.txt", content="hello stage4")

    result = ReadFileTool(tmp_path).execute(path="hello.txt")

    assert read_payload(result) == "hello stage4"


def test_read_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File not found: missing.txt"):
        ReadFileTool(tmp_path).execute(path="missing.txt")


def test_list_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Directory not found: missing"):
        ListDirectoryTool(tmp_path).execute(path="missing")


def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    WriteFileTool(tmp_path).execute(
        path="src/generated/result.txt",
        content="hello",
    )

    target = tmp_path / "src" / "generated" / "result.txt"
    assert target.read_text(encoding="utf-8") == "hello"
