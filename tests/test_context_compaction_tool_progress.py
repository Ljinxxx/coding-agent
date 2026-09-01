import json
from copy import deepcopy
from typing import Any

from src.context_compaction import (
    CURRENT_RUN_HEADER,
    PRIOR_CONTEXT_HEADER,
    build_compacted_context,
    build_context_units,
    estimate_messages_size,
)


def _assistant_tool_message(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }


def _exchange(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    result: str,
) -> list[dict[str, Any]]:
    return [
        _assistant_tool_message(call_id, name, arguments),
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        },
    ]


def _read_result(
    path: str,
    *,
    start_line: int,
    end_line: int,
    total_lines: int,
    next_start_line: int | None,
    exact_anchor: str | None = None,
) -> str:
    body_lines = [
        f"routine record {line_number:04d} " + "x" * 24
        for line_number in range(start_line, end_line + 1)
    ]
    if exact_anchor is not None:
        body_lines[len(body_lines) // 2] = exact_anchor
    next_value = (
        "none" if next_start_line is None else str(next_start_line)
    )
    return (
        "[read_file]\n"
        f"path: {path}\n"
        f"lines: {start_line}-{end_line} of {total_lines}\n"
        f"total_lines: {total_lines}\n"
        f"truncated_before: {str(start_line > 1).lower()}\n"
        f"truncated_after: {str(next_start_line is not None).lower()}\n"
        "char_truncated: false\n"
        "partial_line: false\n"
        f"original_selected_chars: {sum(map(len, body_lines))}\n"
        f"next_start_line: {next_value}\n"
        "notice: none\n\n"
        + "\n".join(body_lines)
        + "\n"
    )


def _compacted_text(context: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in context
        if CURRENT_RUN_HEADER
        in str(message.get("content") or "")
    )


def test_long_paginated_tool_progress_preserves_exact_state_and_budget(
) -> None:
    path = "data/large_release_ledger.txt"
    anchors = {
        2: "CHECKPOINT_ALPHA=value-a",
        4: "CHECKPOINT_BETA=value-b",
        6: "CHECKPOINT_GAMMA=value-c",
        9: "SESSION_KEY=AbC-12_XyZ",
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Audit the paginated ledger once."},
    ]
    for page in range(1, 10):
        start_line = (page - 1) * 200 + 1
        end_line = page * 200
        next_start_line = end_line + 1 if page < 9 else None
        arguments: dict[str, Any] = {"path": path}
        if page > 1:
            arguments["start_line"] = start_line
        messages.extend(
            _exchange(
                f"page-{page}",
                "read_file",
                arguments,
                _read_result(
                    path,
                    start_line=start_line,
                    end_line=end_line,
                    total_lines=1_800,
                    next_start_line=next_start_line,
                    exact_anchor=anchors.get(page),
                ),
            )
        )

    original = deepcopy(messages)
    units = build_context_units(messages[2:], start_index=2)
    first = build_compacted_context(
        messages,
        current_user_index=1,
        max_context_chars=5_000,
        max_compaction_chars=3_000,
    )
    second = build_compacted_context(
        messages,
        current_user_index=1,
        max_context_chars=5_000,
        max_compaction_chars=3_000,
    )
    summary = _compacted_text(first)

    assert len(units) == 9
    assert all(unit.kind == "tool_exchange" for unit in units)
    assert first == second
    assert messages == original
    assert "[Tool Progress]" in summary
    assert "tool: read_file" in summary
    assert f"path: {path}" in summary
    assert "returned_lines: 1601-1800 of 1800" in summary
    assert "truncated_after: false" in summary
    assert "next_start_line: none" in summary
    assert "next_start_line: 201" not in summary
    assert "read_status: exhausted" in summary
    for anchor in anchors.values():
        assert anchor in summary
    assert "[Tool Progress]" not in json.dumps(original)
    assert estimate_messages_size(first) <= 5_000
    assert len(summary) <= 3_000
    assert len(summary) < sum(
        len(str(message.get("content") or ""))
        for message in original
    ) // 5


def test_exhausted_progress_and_exact_state_survive_as_prior_context(
) -> None:
    path = "data/session_ledger.txt"
    anchors = {
        3: "CHECKPOINT_RED=value-r",
        6: "CHECKPOINT_BLUE=value-b",
        9: "SESSION_KEY=AbC-12_XyZ",
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "EARLY_CONSTRAINT=retain\n" + "old " * 600,
        },
        {"role": "assistant", "content": "constraint recorded"},
    ]
    for index in range(3):
        messages.extend(
            (
                {
                    "role": "user",
                    "content": f"old question {index}\n" + "q" * 1_000,
                },
                {
                    "role": "assistant",
                    "content": f"old answer {index}\n" + "a" * 800,
                },
            )
        )
    messages.append(
        {"role": "user", "content": "Audit the complete ledger once."}
    )
    for page in range(1, 10):
        start_line = (page - 1) * 200 + 1
        end_line = page * 200
        messages.extend(
            _exchange(
                f"prior-page-{page}",
                "read_file",
                {"path": path, "start_line": start_line},
                _read_result(
                    path,
                    start_line=start_line,
                    end_line=end_line,
                    total_lines=1_800,
                    next_start_line=(end_line + 1 if page < 9 else None),
                    exact_anchor=anchors.get(page),
                ),
            )
        )
    messages.extend(
        _exchange(
            "prior-diagnostic",
            "run_command",
            {"command": "python diagnose.py"},
            json.dumps(
                {
                    "exit_code": 1,
                    "stdout": (
                        "diagnostic head\n"
                        + "routine\n" * 500
                        + "DIAGNOSTIC_STATE=BLOCKED\n"
                        + "diagnostic tail"
                    ),
                    "stderr": "",
                    "timed_out": False,
                    "stdout_truncated": True,
                    "stderr_truncated": False,
                    "stdout_original_chars": 20_000,
                    "stderr_original_chars": 0,
                }
            ),
        )
    )
    messages.append({"role": "assistant", "content": "audit recorded"})
    current_user_index = len(messages)
    messages.append({"role": "user", "content": "Repair the workspace."})
    original = deepcopy(messages)

    context = build_compacted_context(
        messages,
        current_user_index=current_user_index,
        max_context_chars=5_000,
        max_compaction_chars=3_000,
    )
    prior = "\n".join(
        str(message.get("content") or "")
        for message in context
        if PRIOR_CONTEXT_HEADER in str(message.get("content") or "")
    )

    assert "EARLY_CONSTRAINT=retain" in prior
    assert f"path: {path}" in prior
    assert "returned_lines: 1601-1800 of 1800" in prior
    assert "next_start_line: none" in prior
    assert "next_start_line: 201" not in prior
    assert "read_status: exhausted" in prior
    for anchor in anchors.values():
        assert anchor in prior
    assert "DIAGNOSTIC_STATE=BLOCKED" in prior
    assert messages == original
    assert estimate_messages_size(context) <= 5_000


def test_structured_read_progress_keeps_latest_state_for_multiple_files(
) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect several large inputs."},
    ]
    for index, path in enumerate(
        ("data/file_a.txt", "data/file_b.txt", "data/file_c.txt"),
        start=1,
    ):
        messages.extend(
            _exchange(
                f"read-{index}",
                "read_file",
                {"path": path},
                _read_result(
                    path,
                    start_line=1,
                    end_line=200,
                    total_lines=200,
                    next_start_line=None,
                    exact_anchor=f"FILE_{index}_STATE=complete",
                ),
            )
        )

    context = build_compacted_context(
        messages,
        current_user_index=0,
        max_context_chars=2_400,
        max_compaction_chars=1_800,
    )
    summary = _compacted_text(context)

    for index, path in enumerate(
        ("data/file_a.txt", "data/file_b.txt", "data/file_c.txt"),
        start=1,
    ):
        assert f"path: {path}" in summary
        assert f"FILE_{index}_STATE=complete" in summary
    assert summary.count("read_status: exhausted") >= 3
    assert estimate_messages_size(context) <= 2_400


def test_run_command_progress_preserves_truncation_tail_and_domain_failure(
) -> None:
    stdout = (
        "HEAD_MARKER\n"
        + "routine output\n" * 500
        + "[... stdout truncated ...]\n"
        + "JOB_STATE=READY\n"
        + "TAIL_MARKER\n"
    )
    result = json.dumps(
        {
            "exit_code": 1,
            "stdout": stdout,
            "stderr": "",
            "timed_out": False,
            "stdout_truncated": True,
            "stderr_truncated": False,
            "stdout_original_chars": 50_000,
            "stderr_original_chars": 0,
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "user", "content": "Run the bounded diagnostic."},
        *_exchange(
            "command",
            "run_command",
            {"command": "python -m pytest -q"},
            result,
        ),
    ]

    context = build_compacted_context(
        messages,
        current_user_index=0,
        max_context_chars=1_700,
        max_compaction_chars=1_200,
    )
    summary = _compacted_text(context)

    assert "tool: run_command" in summary
    assert "command: python -m pytest -q" in summary
    assert "execution_ok: true" in summary
    assert "exit_code: 1" in summary
    assert "timed_out: false" in summary
    assert "stdout_truncated: true" in summary
    assert "stdout_original_chars: 50000" in summary
    assert "HEAD_MARKER" in summary
    assert "TAIL_MARKER" in summary
    assert "JOB_STATE=READY" in summary
    assert "error_type:" not in summary
    assert estimate_messages_size(context) <= 1_700


def test_structured_tool_errors_preserve_tool_path_and_error_type() -> None:
    failures = (
        (
            "boundary",
            "read_file",
            {"path": "../outside.txt"},
            "WorkspaceBoundaryError",
        ),
        ("unknown", "missing_tool", {}, "UnknownTool"),
        ("runtime", "custom_tool", {"value": 1}, "RuntimeError"),
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Recover from controlled failures."},
    ]
    for call_id, tool_name, arguments, error_type in failures:
        messages.extend(
            _exchange(
                call_id,
                tool_name,
                arguments,
                json.dumps(
                    {
                        "ok": False,
                        "tool": tool_name,
                        "error_type": error_type,
                        "message": f"controlled {error_type} failure",
                    }
                ),
            )
        )

    context = build_compacted_context(
        messages,
        current_user_index=0,
        max_context_chars=1_450,
        max_compaction_chars=1_300,
    )
    summary = _compacted_text(context)

    assert "path: ../outside.txt" in summary
    assert summary.count("execution_ok: false") == 3
    for _, tool_name, _, error_type in failures:
        assert f"tool: {tool_name}" in summary
        assert f"error_type: {error_type}" in summary
    assert estimate_messages_size(context) <= 1_450


def test_mutation_verification_and_generic_fallback_progress_survive(
) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Modify, inspect, and verify."},
        *_exchange(
            "write",
            "write_file",
            {"path": "src/new_module.py", "content": "x" * 4_000},
            "File written successfully: src/new_module.py",
        ),
        *_exchange(
            "edit",
            "edit_file",
            {
                "path": "src/existing.py",
                "old_text": "a" * 2_000,
                "new_text": "b" * 2_000,
            },
            "File edited successfully: src/existing.py",
        ),
        *_exchange(
            "custom",
            "future_tool",
            {"target": "artifact"},
            "GENERIC_HEAD\n" + "noise\n" * 500 + "GENERIC_TAIL",
        ),
        *_exchange(
            "verify",
            "verify_workspace",
            {},
            json.dumps(
                {
                    "ok": True,
                    "checks": [
                        {
                            "command": "python -m pytest -q",
                            "exit_code": 0,
                            "stdout": "23 passed",
                            "stderr": "",
                            "timed_out": False,
                        }
                    ],
                }
            ),
        ),
    ]

    context = build_compacted_context(
        messages,
        current_user_index=0,
        max_context_chars=2_050,
        max_compaction_chars=1_900,
    )
    summary = _compacted_text(context)

    assert "tool: write_file" in summary
    assert "path: src/new_module.py" in summary
    assert "tool: edit_file" in summary
    assert "path: src/existing.py" in summary
    assert "tool: future_tool" in summary
    assert "GENERIC_HEAD" in summary
    assert "GENERIC_TAIL" in summary
    assert "tool: verify_workspace" in summary
    assert "verification_ok: true" in summary
    assert "exit_code: 0" in summary
    assert "23 passed" in summary
    assert "x" * 500 not in summary
    assert "a" * 500 not in summary
    assert "b" * 500 not in summary
    assert estimate_messages_size(context) <= 2_050


def test_malformed_structured_results_use_deterministic_bounded_fallback(
) -> None:
    messages = [
        {"role": "user", "content": "Inspect malformed tool outputs safely."},
        *_exchange(
            "malformed-read",
            "read_file",
            {"path": "data/malformed.txt"},
            "[read_file]\nmalformed header\n\nREAD_FALLBACK_HEAD\n"
            + "read noise\n" * 400
            + "READ_FALLBACK_TAIL",
        ),
        *_exchange(
            "malformed-command",
            "run_command",
            {"command": "python broken.py"},
            "not-json\nCOMMAND_FALLBACK_HEAD\n"
            + "command noise\n" * 400
            + "COMMAND_FALLBACK_TAIL",
        ),
    ]
    original = deepcopy(messages)

    first = build_compacted_context(
        messages,
        current_user_index=0,
        max_context_chars=1_450,
        max_compaction_chars=1_300,
    )
    second = build_compacted_context(
        messages,
        current_user_index=0,
        max_context_chars=1_450,
        max_compaction_chars=1_300,
    )
    summary = _compacted_text(first)

    assert first == second
    assert messages == original
    assert "tool: read_file" in summary
    assert "path: data/malformed.txt" in summary
    assert "READ_FALLBACK_HEAD" in summary
    assert "READ_FALLBACK_TAIL" in summary
    assert "tool: run_command" in summary
    assert "COMMAND_FALLBACK_HEAD" in summary
    assert "COMMAND_FALLBACK_TAIL" in summary
    assert estimate_messages_size(first) <= 1_450
