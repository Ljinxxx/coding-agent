import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.tools.files import ReadFileTool
from src.tools.path_utils import WorkspaceBoundaryError
from src.tools.shell import RunCommandTool
from src.tools.verification import VerifyWorkspaceTool


def parse_read_result(result: str) -> tuple[dict[str, str], str]:
    header, payload = result.split("\n\n", 1)
    lines = header.splitlines()
    assert lines[0] == "[read_file]"
    metadata = dict(line.split(": ", 1) for line in lines[1:])
    return metadata, payload


def make_python_command(code: str) -> str:
    parts = [sys.executable, "-B", "-c", code]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def run_python(tool: RunCommandTool, code: str) -> dict[str, Any]:
    return json.loads(tool.execute(command=make_python_command(code)))


def test_read_small_file_returns_complete_content(tmp_path: Path) -> None:
    content = "first line\nmiddle line\nlast line\n"
    (tmp_path / "small.txt").write_text(content, encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    metadata, payload = parse_read_result(tool.execute(path="small.txt"))

    assert payload == content
    assert metadata == {
        "path": "small.txt",
        "lines": "1-3 of 3",
        "total_lines": "3",
        "truncated_before": "false",
        "truncated_after": "false",
        "char_truncated": "false",
        "partial_line": "false",
        "original_selected_chars": str(len(content)),
        "next_start_line": "none",
        "notice": "none",
    }
    parameters = tool.to_schema()["function"]["parameters"]
    assert set(parameters["properties"]) == {
        "path",
        "start_line",
        "max_lines",
    }
    assert parameters["required"] == ["path"]
    assert parameters["additionalProperties"] is False
    assert "max_output_chars" not in parameters["properties"]
    assert tool.mutates_workspace is False

    outside = tmp_path.parent / "outside-read-budget.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(WorkspaceBoundaryError, match="Path escapes workspace"):
        tool.execute(path="../outside-read-budget.txt")


def test_read_large_file_uses_default_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [f"Line {index:04d}\n" for index in range(1, 1001)]
    (tmp_path / "large.txt").write_text("".join(lines), encoding="utf-8")
    tool = ReadFileTool(
        tmp_path,
        default_max_lines=25,
        max_output_chars=1_000,
    )

    def reject_read_text(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("ReadFileTool must stream instead of read_text().")

    monkeypatch.setattr(Path, "read_text", reject_read_text)
    metadata, payload = parse_read_result(tool.execute(path="large.txt"))

    assert payload.splitlines() == [line.rstrip("\n") for line in lines[:25]]
    assert "Line 0026" not in payload
    assert metadata["lines"] == "1-25 of 1000"
    assert metadata["total_lines"] == "1000"
    assert metadata["truncated_before"] == "false"
    assert metadata["truncated_after"] == "true"
    assert metadata["char_truncated"] == "false"
    assert metadata["next_start_line"] == "26"
    assert len(payload) <= tool.max_output_chars


def test_read_file_supports_line_pagination(tmp_path: Path) -> None:
    lines = [f"Line {index:04d}\n" for index in range(1, 1001)]
    (tmp_path / "large.txt").write_text("".join(lines), encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    metadata, payload = parse_read_result(
        tool.execute(path="large.txt", start_line=201, max_lines=100)
    )
    assert payload == "".join(lines[200:300])
    assert metadata["lines"] == "201-300 of 1000"
    assert metadata["truncated_before"] == "true"
    assert metadata["truncated_after"] == "true"
    assert metadata["next_start_line"] == "301"

    tail_metadata, tail_payload = parse_read_result(
        tool.execute(path="large.txt", start_line=901, max_lines=200)
    )
    assert tail_payload == "".join(lines[900:])
    assert tail_metadata["lines"] == "901-1000 of 1000"
    assert tail_metadata["truncated_after"] == "false"
    assert tail_metadata["next_start_line"] == "none"

    eof_metadata, eof_payload = parse_read_result(
        tool.execute(path="large.txt", start_line=1200, max_lines=10)
    )
    assert eof_payload == ""
    assert eof_metadata["lines"] == "none"
    assert eof_metadata["total_lines"] == "1000"
    assert eof_metadata["truncated_before"] == "true"
    assert eof_metadata["truncated_after"] == "false"

    invalid_calls = [
        {"start_line": 0},
        {"start_line": -1},
        {"start_line": True},
        {"start_line": "1"},
        {"max_lines": 0},
        {"max_lines": -1},
        {"max_lines": False},
        {"max_lines": "10"},
    ]
    for invalid in invalid_calls:
        with pytest.raises(ValueError, match="positive integer"):
            tool.execute(path="large.txt", **invalid)


def test_read_file_hard_char_budget_limits_long_single_line(
    tmp_path: Path,
) -> None:
    multiline = "ABCDEFGHIJ\nKLMNOPQRST\nUVWXYZ"
    (tmp_path / "multi-line.txt").write_text(multiline, encoding="utf-8")
    multiline_tool = ReadFileTool(
        tmp_path,
        default_max_lines=3,
        max_output_chars=15,
    )
    multiline_metadata, multiline_payload = parse_read_result(
        multiline_tool.execute(path="multi-line.txt")
    )
    assert multiline_payload == "ABCDEFGHIJ\n"
    assert multiline_metadata["lines"] == "1-1 of 3"
    assert multiline_metadata["truncated_after"] == "true"
    assert multiline_metadata["char_truncated"] == "true"
    assert multiline_metadata["partial_line"] == "false"
    assert multiline_metadata["original_selected_chars"] == str(
        len(multiline)
    )
    assert multiline_metadata["next_start_line"] == "2"

    resumed_metadata, resumed_payload = parse_read_result(
        multiline_tool.execute(
            path="multi-line.txt",
            start_line=int(multiline_metadata["next_start_line"]),
            max_lines=1,
        )
    )
    assert resumed_payload == "KLMNOPQRST\n"
    assert "OPQRST" in resumed_payload
    assert resumed_metadata["lines"] == "2-2 of 3"
    assert resumed_metadata["char_truncated"] == "false"
    assert resumed_metadata["partial_line"] == "false"

    long_line = "你" * 50_000
    long_content = long_line + "\ntail-line\n"
    (tmp_path / "long-line.txt").write_text(
        long_content,
        encoding="utf-8",
    )
    tool = ReadFileTool(
        tmp_path,
        default_max_lines=2,
        max_output_chars=137,
    )

    metadata, payload = parse_read_result(
        tool.execute(
            path="long-line.txt",
            max_output_chars=999_999,
        )
    )

    assert payload == long_line[:137]
    assert len(payload) == tool.max_output_chars
    assert len(payload.encode("utf-8")) > tool.max_output_chars
    assert "tail-line" not in payload
    assert metadata["lines"] == "none"
    assert metadata["char_truncated"] == "true"
    assert metadata["partial_line"] == "true"
    assert metadata["truncated_after"] == "true"
    assert metadata["original_selected_chars"] == str(len(long_content))
    assert metadata["next_start_line"] == "none"
    assert metadata["notice"] == (
        "first selected line exceeds character budget; partial line "
        "returned; line-based continuation unavailable"
    )


def test_run_command_keeps_small_stdout_and_stderr(tmp_path: Path) -> None:
    tool = RunCommandTool(tmp_path, max_output_chars=200)
    result = run_python(
        tool,
        'import sys; sys.stdout.write("small-out"); '
        'sys.stderr.write("small-err")',
    )

    assert result == {
        "exit_code": 0,
        "stdout": "small-out",
        "stderr": "small-err",
        "timed_out": False,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_original_chars": 9,
        "stderr_original_chars": 9,
    }
    parameters = tool.to_schema()["function"]["parameters"]
    assert set(parameters["properties"]) == {"command", "timeout"}
    assert "max_output_chars" not in parameters["properties"]
    assert tool.mutates_workspace is True


def test_run_command_truncates_large_stdout_with_head_and_tail(
    tmp_path: Path,
) -> None:
    head = "STDOUT_HEAD_TOKEN"
    middle = "STDOUT_REMOVED_MIDDLE_TOKEN"
    tail = "STDOUT_TAIL_TOKEN"
    original = head + "A" * 4_000 + middle + "B" * 4_000 + tail
    tool = RunCommandTool(tmp_path, max_output_chars=240)

    result = run_python(
        tool,
        f"import sys; sys.stdout.write({original!r})",
    )

    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is False
    assert result["stdout_original_chars"] == len(original)
    assert len(result["stdout"]) <= tool.max_output_chars
    assert head in result["stdout"]
    assert tail in result["stdout"]
    assert middle not in result["stdout"]
    assert "stdout truncated: original" in result["stdout"]


def test_run_command_truncates_large_stderr_and_preserves_nonzero_exit(
    tmp_path: Path,
) -> None:
    head = "STDERR_HEAD_TOKEN"
    middle = "STDERR_REMOVED_MIDDLE_TOKEN"
    tail = "STDERR_TAIL_TOKEN"
    original = head + "E" * 25_000 + middle + "F" * 25_000 + tail
    command = make_python_command(
        "import sys; "
        'sys.stdout.write("small-out"); '
        f"sys.stderr.write({head!r} + 'E' * 25_000 + {middle!r} + "
        f"'F' * 25_000 + {tail!r}); "
        "raise SystemExit(3)"
    )
    tool = RunCommandTool(tmp_path, max_output_chars=240)

    result = json.loads(tool.execute(command=command))

    assert result["exit_code"] == 3
    assert result["timed_out"] is False
    assert result["stdout"] == "small-out"
    assert result["stdout_truncated"] is False
    assert result["stderr_truncated"] is True
    assert result["stderr_original_chars"] == len(original)
    assert len(result["stderr"]) <= tool.max_output_chars
    assert head in result["stderr"]
    assert tail in result["stderr"]
    assert middle not in result["stderr"]

    verification = json.loads(
        VerifyWorkspaceTool(tmp_path, [command]).execute()
    )
    assert verification["ok"] is False
    check = verification["checks"][0]
    assert check["exit_code"] == 3
    assert check["timed_out"] is False
    assert len(check["stderr"]) <= 20_000
    assert "stderr truncated: original" in check["stderr"]


def test_timeout_output_is_budgeted_and_invalid_configuration_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout_errors = [
        subprocess.TimeoutExpired(
            "first",
            1,
            output=b"BYTES_HEAD" + b"A" * 500 + b"BYTES_TAIL",
            stderr=None,
        ),
        subprocess.TimeoutExpired(
            "second",
            1,
            output="TEXT_HEAD" + "B" * 500 + "TEXT_TAIL",
            stderr=b"ERROR_HEAD" + b"C" * 500 + b"ERROR_TAIL",
        ),
        subprocess.TimeoutExpired("third", 1, output=None, stderr=None),
    ]

    def raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise timeout_errors.pop(0)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    tool = RunCommandTool(tmp_path, max_output_chars=100)

    first = json.loads(tool.execute(command="first", timeout=1))
    assert first["exit_code"] is None
    assert first["timed_out"] is True
    assert first["stdout_truncated"] is True
    assert first["stderr"] == ""
    assert first["stderr_original_chars"] == 0
    assert "BYTES_HEAD" in first["stdout"]
    assert "BYTES_TAIL" in first["stdout"]
    assert len(first["stdout"]) <= tool.max_output_chars

    second = json.loads(tool.execute(command="second", timeout=1))
    assert second["timed_out"] is True
    assert second["stdout_truncated"] is True
    assert second["stderr_truncated"] is True
    assert "TEXT_HEAD" in second["stdout"]
    assert "TEXT_TAIL" in second["stdout"]
    assert "ERROR_HEAD" in second["stderr"]
    assert "ERROR_TAIL" in second["stderr"]

    third = json.loads(tool.execute(command="third", timeout=1))
    assert third["timed_out"] is True
    assert third["stdout"] == third["stderr"] == ""
    assert third["stdout_original_chars"] == 0
    assert third["stderr_original_chars"] == 0

    invalid_factories = [
        lambda: ReadFileTool(tmp_path, default_max_lines=0),
        lambda: ReadFileTool(tmp_path, max_output_chars=-1),
        lambda: RunCommandTool(tmp_path, max_output_chars=0),
        lambda: RunCommandTool(tmp_path, max_output_chars=True),
    ]
    for invalid_factory in invalid_factories:
        with pytest.raises(ValueError, match="positive integer"):
            invalid_factory()
