import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


PRIOR_CONTEXT_HEADER = "[Compacted Prior Context - Harness Generated]"
CURRENT_RUN_HEADER = (
    "[Compacted Current-Run Progress - Harness Generated]"
)
_HARNESS_NOTICE = "This is not a new user request."
_USER_PREVIEW_CHARS = 800
_ASSISTANT_PREVIEW_CHARS = 500
_TOOL_ARGUMENT_PREVIEW_CHARS = 300
_TOOL_RESULT_PREVIEW_CHARS = 500
_RAW_SELECTION_OVERHEAD_CHARS = 64
_STRUCTURED_VALUE_PREVIEW_CHARS = 160
_CONTENT_EDGE_LINES = 3
_PER_UNIT_EXACT_ANCHORS = 6
_DIGEST_EXACT_ANCHORS = 24
_DIGEST_CONTENT_ANCHORS = 12
_DIGEST_LATEST_STATES = 12
_DIGEST_EXACT_ANCHOR_CHARS = 600
_DIGEST_CONTENT_ANCHOR_CHARS = 600
_DIGEST_LATEST_STATE_CHARS = 1_200
_KEY_VALUE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


@dataclass(frozen=True)
class ContextUnit:
    messages: tuple[dict[str, Any], ...]
    kind: str
    start_index: int
    end_index: int
    char_count: int


@dataclass(frozen=True)
class CompactionBlock:
    kind: str
    message: dict[str, Any]


@dataclass(frozen=True)
class ToolProgressSummary:
    workflow_key: str
    latest_state: str
    exact_anchors: tuple[str, ...]
    content_anchors: tuple[str, ...]
    rendered: str


def estimate_messages_size(messages: list[dict[str, Any]]) -> int:
    return len(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _is_tool_error(message: dict[str, Any]) -> bool:
    if message.get("role") != "tool":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("ok") is False


def _single_message_kind(message: dict[str, Any]) -> str:
    role = message.get("role")
    content = str(message.get("content") or "")
    if role == "system":
        return "system"
    if role == "user" and content.startswith("[Verification Required]"):
        return "harness_feedback"
    if role == "user":
        return "user"
    if role == "assistant":
        return "assistant_text"
    if role == "tool":
        return "tool_error" if _is_tool_error(message) else "tool_result"
    return "other"


def build_context_units(
    messages: list[dict[str, Any]],
    *,
    start_index: int = 0,
) -> list[ContextUnit]:
    units: list[ContextUnit] = []
    message_index = 0

    while message_index < len(messages):
        message = messages[message_index]
        grouped_messages = [message]
        end_index = message_index
        kind = _single_message_kind(message)
        tool_calls = message.get("tool_calls", [])

        if message.get("role") == "assistant" and tool_calls:
            expected_ids = {
                str(call.get("id"))
                for call in tool_calls
                if call.get("id") is not None
            }
            seen_ids: set[str] = set()
            next_index = message_index + 1
            while next_index < len(messages):
                result_message = messages[next_index]
                result_id = result_message.get("tool_call_id")
                if (
                    result_message.get("role") != "tool"
                    or str(result_id) not in expected_ids
                    or str(result_id) in seen_ids
                ):
                    break
                grouped_messages.append(result_message)
                seen_ids.add(str(result_id))
                end_index = next_index
                next_index += 1
                if seen_ids == expected_ids:
                    break

            kind = (
                "tool_error"
                if any(_is_tool_error(item) for item in grouped_messages)
                else "tool_exchange"
            )

        copied_messages = tuple(deepcopy(grouped_messages))
        absolute_start = start_index + message_index
        absolute_end = start_index + end_index
        units.append(
            ContextUnit(
                messages=copied_messages,
                kind=kind,
                start_index=absolute_start,
                end_index=absolute_end,
                char_count=estimate_messages_size(list(copied_messages)),
            )
        )
        message_index = end_index + 1

    return units


def _clip_text(
    text: str,
    max_chars: int,
    label: str,
    *,
    prefer_head: bool = False,
) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    marker = f"\n[... {label} omitted ...]\n"
    if len(marker) >= max_chars:
        return marker[:max_chars]

    remaining = max_chars - len(marker)
    head_chars = (
        (remaining * 2 + 2) // 3
        if prefer_head
        else (remaining + 1) // 2
    )
    tail_chars = remaining - head_chars
    tail = text[-tail_chars:] if tail_chars else ""
    return text[:head_chars] + marker + tail


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_arguments(call: dict[str, Any]) -> str:
    function = call.get("function", {})
    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        return json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _argument_mapping(call: dict[str, Any]) -> dict[str, Any] | None:
    arguments = call.get("function", {}).get("arguments", "")
    if isinstance(arguments, dict):
        return deepcopy(arguments)
    if not isinstance(arguments, str):
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_mapping(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _inline_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        rendered = " ".join(value.splitlines())
    else:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return _clip_text(
        rendered,
        _STRUCTURED_VALUE_PREVIEW_CHARS,
        "value",
    )


def _argument_lines(arguments: dict[str, Any] | None) -> list[str]:
    if arguments is None:
        return []

    lines: list[str] = []
    excluded = {"content", "old_text", "new_text"}
    for key in sorted(arguments):
        if key in excluded:
            continue
        label = key if key == "path" else f"argument.{key}"
        lines.append(f"{label}: {_inline_value(arguments[key])}")
    return lines


def _extract_exact_anchors(text: str) -> tuple[str, ...]:
    latest_by_key: dict[str, str] = {}
    for line in text.splitlines():
        if (
            len(line) > _STRUCTURED_VALUE_PREVIEW_CHARS
            or _KEY_VALUE_PATTERN.fullmatch(line) is None
        ):
            continue
        key = line.split("=", 1)[0]
        if key in latest_by_key:
            del latest_by_key[key]
        latest_by_key[key] = line
    return tuple(latest_by_key.values())[-_PER_UNIT_EXACT_ANCHORS:]


def _edge_anchors(text: str, label: str) -> tuple[str, ...]:
    lines = [line for line in text.splitlines() if line]
    if not lines:
        return ()

    selected = lines[:_CONTENT_EDGE_LINES]
    if len(lines) > _CONTENT_EDGE_LINES:
        selected.extend(lines[-_CONTENT_EDGE_LINES:])

    anchors: list[str] = []
    for index, line in enumerate(selected):
        position = "head" if index < _CONTENT_EDGE_LINES else "tail"
        preview = _clip_text(
            line,
            _STRUCTURED_VALUE_PREVIEW_CHARS,
            f"{label} {position}",
        )
        rendered = f"{label}_{position}: {preview}"
        if rendered not in anchors:
            anchors.append(rendered)
    return tuple(anchors)


def _read_result_parts(
    content: str,
) -> tuple[dict[str, str], str] | None:
    header, separator, body = content.partition("\n\n")
    lines = header.splitlines()
    if separator != "\n\n" or not lines or lines[0] != "[read_file]":
        return None

    metadata: dict[str, str] = {}
    for line in lines[1:]:
        key, colon, value = line.partition(":")
        if not colon:
            return None
        metadata[key.strip()] = value.strip()
    return metadata, body


def _progress_summary(
    *,
    workflow_key: str,
    lines: list[str],
    exact_anchors: tuple[str, ...] = (),
    content_anchors: tuple[str, ...] = (),
) -> ToolProgressSummary:
    rendered = "\n".join(["[Tool Progress]", *lines])
    return ToolProgressSummary(
        workflow_key=workflow_key,
        latest_state=rendered,
        exact_anchors=exact_anchors,
        content_anchors=content_anchors,
        rendered=rendered,
    )


def _summarize_tool_error(
    name: str,
    arguments: dict[str, Any] | None,
    payload: dict[str, Any],
) -> ToolProgressSummary:
    path = arguments.get("path") if arguments is not None else None
    identity = _inline_value(path) if path is not None else _normalized_identity(
        arguments
    )
    lines = [f"tool: {name}", "execution_ok: false"]
    lines.extend(_argument_lines(arguments))
    lines.append(
        f"error_type: {_inline_value(payload.get('error_type', 'unknown'))}"
    )
    lines.append(f"message: {_inline_value(payload.get('message', ''))}")
    return _progress_summary(
        workflow_key=f"error:{name}:{identity}",
        lines=lines,
    )


def _normalized_identity(arguments: dict[str, Any] | None) -> str:
    if not arguments:
        return "none"
    for key in ("path", "target", "command"):
        if key in arguments:
            return _inline_value(arguments[key])
    return _clip_text(
        json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        _STRUCTURED_VALUE_PREVIEW_CHARS,
        "identity",
    )


def _summarize_read_file(
    name: str,
    arguments: dict[str, Any] | None,
    content: str,
) -> ToolProgressSummary | None:
    parsed = _read_result_parts(content)
    if parsed is None:
        return None
    metadata, body = parsed
    path = metadata.get("path")
    if path is None and arguments is not None:
        raw_path = arguments.get("path")
        path = str(raw_path) if raw_path is not None else None
    path = path or "<unknown>"

    requested_start = 1
    requested_max: Any = "default"
    if arguments is not None:
        requested_start = arguments.get("start_line", 1)
        requested_max = arguments.get("max_lines", "default")

    lines = [
        f"tool: {name}",
        "execution_ok: true",
        f"path: {_inline_value(path)}",
        f"requested_start_line: {_inline_value(requested_start)}",
        f"requested_max_lines: {_inline_value(requested_max)}",
    ]
    field_mapping = (
        ("lines", "returned_lines"),
        ("total_lines", "total_lines"),
        ("truncated_before", "truncated_before"),
        ("truncated_after", "truncated_after"),
        ("char_truncated", "char_truncated"),
        ("partial_line", "partial_line"),
        ("next_start_line", "next_start_line"),
    )
    for source, label in field_mapping:
        if source in metadata:
            lines.append(f"{label}: {_inline_value(metadata[source])}")
    if (
        metadata.get("truncated_after", "").casefold() == "false"
        and metadata.get("next_start_line", "").casefold() == "none"
    ):
        lines.append("read_status: exhausted")

    return _progress_summary(
        workflow_key=f"read_file:{path}",
        lines=lines,
        exact_anchors=_extract_exact_anchors(body),
        content_anchors=_edge_anchors(body, "read_body"),
    )


def _summarize_run_command(
    name: str,
    arguments: dict[str, Any] | None,
    content: str,
) -> ToolProgressSummary | None:
    payload = _json_mapping(content)
    if payload is None or "exit_code" not in payload:
        return None
    command = "<unknown>"
    if arguments is not None and "command" in arguments:
        command = _inline_value(arguments["command"])

    lines = [
        f"tool: {name}",
        "execution_ok: true",
        f"command: {command}",
    ]
    for key in (
        "exit_code",
        "timed_out",
        "stdout_truncated",
        "stderr_truncated",
        "stdout_original_chars",
        "stderr_original_chars",
    ):
        if key in payload:
            lines.append(f"{key}: {_inline_value(payload[key])}")

    stdout = payload.get("stdout", "")
    stderr = payload.get("stderr", "")
    stdout_text = stdout if isinstance(stdout, str) else _inline_value(stdout)
    stderr_text = stderr if isinstance(stderr, str) else _inline_value(stderr)
    return _progress_summary(
        workflow_key=f"run_command:{command}",
        lines=lines,
        exact_anchors=_extract_exact_anchors(
            "\n".join((stdout_text, stderr_text))
        ),
        content_anchors=(
            *_edge_anchors(stdout_text, "stdout"),
            *_edge_anchors(stderr_text, "stderr"),
        ),
    )


def _summarize_verification(
    name: str,
    content: str,
) -> ToolProgressSummary | None:
    payload = _json_mapping(content)
    if payload is None or "ok" not in payload or "checks" not in payload:
        return None
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return None

    lines = [
        f"tool: {name}",
        "execution_ok: true",
        f"verification_ok: {_inline_value(payload['ok'])}",
        f"checks_count: {len(checks)}",
    ]
    exact_anchors: list[str] = []
    content_anchors: list[str] = []
    for index, check in enumerate(checks[:3], start=1):
        if not isinstance(check, dict):
            continue
        for key in ("command", "exit_code", "timed_out"):
            if key in check:
                lines.append(
                    f"check_{index}_{key}: {_inline_value(check[key])}"
                )
        for stream in ("stdout", "stderr"):
            value = check.get(stream, "")
            text = value if isinstance(value, str) else _inline_value(value)
            exact_anchors.extend(_extract_exact_anchors(text))
            content_anchors.extend(
                _edge_anchors(text, f"check_{index}_{stream}")
            )
    return _progress_summary(
        workflow_key=name,
        lines=lines,
        exact_anchors=tuple(exact_anchors)[-_PER_UNIT_EXACT_ANCHORS:],
        content_anchors=tuple(content_anchors),
    )


def _summarize_mutation(
    name: str,
    arguments: dict[str, Any] | None,
    content: str,
) -> ToolProgressSummary:
    lines = [f"tool: {name}", "execution_ok: true"]
    if arguments is not None and "path" in arguments:
        lines.append(f"path: {_inline_value(arguments['path'])}")
    lines.append(f"result: {_inline_value(content)}")
    return _progress_summary(
        workflow_key=f"{name}:{_normalized_identity(arguments)}",
        lines=lines,
    )


def _summarize_generic_tool(
    name: str,
    arguments: dict[str, Any] | None,
    content: str,
) -> ToolProgressSummary:
    lines = [f"tool: {name}", "execution_ok: true"]
    lines.extend(_argument_lines(arguments))
    return _progress_summary(
        workflow_key=f"{name}:{_normalized_identity(arguments)}",
        lines=lines,
        exact_anchors=_extract_exact_anchors(content),
        content_anchors=_edge_anchors(content, "result"),
    )


def _summarize_tool_call(
    call: dict[str, Any],
    result: dict[str, Any] | None,
) -> ToolProgressSummary:
    function = call.get("function", {})
    name = str(function.get("name", "<unknown>"))
    arguments = _argument_mapping(call)
    content = _message_text(result) if result is not None else ""
    payload = _json_mapping(content)

    if (
        payload is not None
        and payload.get("ok") is False
        and payload.get("error_type") is not None
    ):
        return _summarize_tool_error(name, arguments, payload)
    if name == "read_file":
        summary = _summarize_read_file(name, arguments, content)
        if summary is not None:
            return summary
    elif name == "run_command":
        summary = _summarize_run_command(name, arguments, content)
        if summary is not None:
            return summary
    elif name == "verify_workspace":
        summary = _summarize_verification(name, content)
        if summary is not None:
            return summary
    elif name in {"edit_file", "write_file"}:
        return _summarize_mutation(name, arguments, content)
    return _summarize_generic_tool(name, arguments, content)


def _tool_progress_summaries(
    unit: ContextUnit,
) -> tuple[ToolProgressSummary, ...]:
    if not unit.messages or unit.messages[0].get("role") != "assistant":
        return ()
    calls = unit.messages[0].get("tool_calls", [])
    if not isinstance(calls, list):
        return ()
    results = {
        str(message.get("tool_call_id", "")): message
        for message in unit.messages[1:]
        if message.get("role") == "tool"
    }
    return tuple(
        _summarize_tool_call(call, results.get(str(call.get("id", ""))))
        for call in calls
        if isinstance(call, dict)
    )


def _render_tool_exchange(unit: ContextUnit) -> str:
    summaries = _tool_progress_summaries(unit)
    if summaries:
        return "\n\n".join(summary.rendered for summary in summaries)
    return "[Tool Exchange]\n" + _clip_text(
        json.dumps(
            list(unit.messages),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        _TOOL_ARGUMENT_PREVIEW_CHARS + _TOOL_RESULT_PREVIEW_CHARS,
        "tool exchange",
    )


def _render_unit(unit: ContextUnit) -> str:
    if (
        unit.kind in {"tool_exchange", "tool_error"}
        and unit.messages
        and unit.messages[0].get("role") == "assistant"
        and unit.messages[0].get("tool_calls")
    ):
        return _render_tool_exchange(unit)

    message = unit.messages[0]
    content = _message_text(message)
    if unit.kind == "user":
        return "[User]\n" + _clip_text(
            content,
            _USER_PREVIEW_CHARS,
            "user text",
        )
    if unit.kind == "harness_feedback":
        return "[Harness Feedback]\n" + _clip_text(
            content,
            _USER_PREVIEW_CHARS,
            "harness feedback",
        )
    if unit.kind == "assistant_text":
        return "[Assistant]\n" + _clip_text(
            content,
            _ASSISTANT_PREVIEW_CHARS,
            "assistant text",
        )
    if unit.kind in {"tool_result", "tool_error"}:
        return "[Tool Result]\n" + _clip_text(
            content,
            _TOOL_RESULT_PREVIEW_CHARS,
            "tool result",
        )
    return f"[{unit.kind}]\n" + _clip_text(
        content,
        _ASSISTANT_PREVIEW_CHARS,
        "message",
    )


def _select_bounded_items(
    items: list[str],
    *,
    max_items: int,
    max_chars: int,
    prefer_recent: bool = False,
) -> list[str]:
    unique: list[str] = []
    for item in items:
        if item and item not in unique:
            unique.append(item)
    if not unique or max_items <= 0 or max_chars <= 0:
        return []

    if prefer_recent:
        priority = list(range(len(unique) - 1, -1, -1))
    else:
        priority = []
        left = 0
        right = len(unique) - 1
        while left <= right:
            priority.append(left)
            if left != right:
                priority.append(right)
            left += 1
            right -= 1

    selected_indexes: list[int] = []
    selected_chars = 0
    for index in priority:
        item_chars = len(unique[index]) + (1 if selected_indexes else 0)
        if selected_chars + item_chars > max_chars:
            continue
        selected_indexes.append(index)
        selected_chars += item_chars
        if len(selected_indexes) == max_items:
            break
    return [unique[index] for index in sorted(selected_indexes)]


def _structured_progress_sections(
    units: list[ContextUnit],
) -> tuple[str, str, str]:
    latest_by_workflow: dict[str, str] = {}
    exact_by_key: dict[str, str] = {}
    content_anchors: list[str] = []

    for unit in units:
        for summary in _tool_progress_summaries(unit):
            if summary.workflow_key in latest_by_workflow:
                del latest_by_workflow[summary.workflow_key]
            latest_by_workflow[summary.workflow_key] = summary.latest_state
            for anchor in summary.exact_anchors:
                key = anchor.split("=", 1)[0]
                if key in exact_by_key:
                    del exact_by_key[key]
                exact_by_key[key] = anchor
            content_anchors.extend(summary.content_anchors)

    latest_states = _select_bounded_items(
        list(latest_by_workflow.values()),
        max_items=_DIGEST_LATEST_STATES,
        max_chars=_DIGEST_LATEST_STATE_CHARS,
        prefer_recent=True,
    )
    exact_anchors = _select_bounded_items(
        list(exact_by_key.values()),
        max_items=_DIGEST_EXACT_ANCHORS,
        max_chars=_DIGEST_EXACT_ANCHOR_CHARS,
    )
    bounded_content = _select_bounded_items(
        content_anchors,
        max_items=_DIGEST_CONTENT_ANCHORS,
        max_chars=_DIGEST_CONTENT_ANCHOR_CHARS,
    )

    latest_section = "\n\n".join(latest_states)
    exact_section = (
        "[Exact Tool Result Anchors]\n" + "\n".join(exact_anchors)
        if exact_anchors
        else ""
    )
    content_section = (
        "[Bounded Tool Result Edges]\n" + "\n".join(bounded_content)
        if bounded_content
        else ""
    )
    return latest_section, exact_section, content_section


def _render_digest(header: str, units: list[ContextUnit]) -> str:
    ordinary_entries: list[str] = []
    for unit in units:
        if _tool_progress_summaries(unit):
            continue
        ordinary_entries.append(_render_unit(unit))

    ordinary = "\n\n".join(ordinary_entries)
    latest, exact, content = _structured_progress_sections(units)
    structured_parts = [part for part in (latest, exact, content) if part]
    if header == PRIOR_CONTEXT_HEADER:
        body_parts = [ordinary, content, latest, exact]
    else:
        body_parts = [latest, exact, content, ordinary]
    body = "\n\n".join(part for part in body_parts if part)
    if not body and structured_parts:
        body = "\n\n".join(structured_parts)
    return f"{header}\n{_HARNESS_NOTICE}\n\n{body}"


def _allocate_digest_budgets(
    desired_lengths: list[int],
    total_budget: int,
) -> list[int]:
    if not desired_lengths:
        return []
    if len(desired_lengths) == 1:
        return [min(desired_lengths[0], total_budget)]

    first_share = (total_budget + 1) // 2
    budgets = [
        min(desired_lengths[0], first_share),
        min(desired_lengths[1], total_budget - first_share),
    ]
    remaining = total_budget - sum(budgets)
    for index in range(len(budgets)):
        available = desired_lengths[index] - budgets[index]
        added = min(available, remaining)
        budgets[index] += added
        remaining -= added
    return budgets


def _build_compaction_messages(
    prior_units: list[ContextUnit],
    progress_units: list[ContextUnit],
    total_budget: int,
) -> list[CompactionBlock]:
    if total_budget <= 0:
        return []

    digests: list[tuple[str, str, str]] = []
    if prior_units:
        digests.append(
            (
                "prior",
                "prior context",
                _render_digest(PRIOR_CONTEXT_HEADER, prior_units),
            )
        )
    if progress_units:
        digests.append(
            (
                "current_run",
                "current-run progress",
                _render_digest(CURRENT_RUN_HEADER, progress_units),
            )
        )

    budgets = _allocate_digest_budgets(
        [len(content) for _, _, content in digests],
        total_budget,
    )
    result: list[CompactionBlock] = []
    for (kind, label, content), budget in zip(
        digests,
        budgets,
        strict=True,
    ):
        if budget <= 0:
            continue
        result.append(
            CompactionBlock(
                kind=kind,
                message={
                    "role": "user",
                    "content": _clip_text(
                        content,
                        budget,
                        label,
                        prefer_head=kind == "prior",
                    ),
                },
            )
        )
    return result


def _flatten_units(units: list[ContextUnit]) -> list[dict[str, Any]]:
    return [
        deepcopy(message)
        for unit in units
        for message in unit.messages
    ]


def _select_recent_units(
    units: list[ContextUnit],
    budget: int,
) -> tuple[list[ContextUnit], int]:
    selected: list[ContextUnit] = []
    selected_chars = 0
    for unit in reversed(units):
        if len(units) > 1 and len(selected) == len(units) - 1:
            break
        if selected_chars + unit.char_count > budget:
            break
        selected.insert(0, unit)
        selected_chars += unit.char_count
    return selected, selected_chars


def build_compacted_context(
    messages: list[dict[str, Any]],
    *,
    current_user_index: int,
    max_context_chars: int,
    max_compaction_chars: int,
) -> list[dict[str, Any]]:
    if not 0 <= current_user_index < len(messages):
        raise ValueError("current_user_index is outside the message history.")
    if messages[current_user_index].get("role") != "user":
        raise ValueError("current_user_index must reference a user message.")
    if max_context_chars < 1 or max_compaction_chars < 1:
        raise ValueError("Context budgets must be positive integers.")

    system_messages: list[dict[str, Any]] = []
    conversation_start = 0
    if messages and messages[0].get("role") == "system":
        system_messages = [deepcopy(messages[0])]
        conversation_start = 1

    units = build_context_units(
        messages[conversation_start:],
        start_index=conversation_start,
    )
    anchor_unit = next(
        (
            unit
            for unit in units
            if unit.start_index == current_user_index
            and unit.end_index == current_user_index
        ),
        None,
    )
    if anchor_unit is None:
        raise ValueError("Current user message is not an independent unit.")

    prior_units = [
        unit for unit in units if unit.end_index < current_user_index
    ]
    progress_units = [
        unit for unit in units if unit.start_index > current_user_index
    ]
    anchor_messages = _flatten_units([anchor_unit])
    mandatory_messages = [*system_messages, *anchor_messages]
    mandatory_size = estimate_messages_size(mandatory_messages)
    if mandatory_size > max_context_chars:
        raise ValueError("Mandatory context exceeds max_context_chars.")

    raw_budget = max(
        0,
        max_context_chars
        - mandatory_size
        - max_compaction_chars
        - _RAW_SELECTION_OVERHEAD_CHARS,
    )
    selected_raw, _ = _select_recent_units(
        [*prior_units, *progress_units],
        raw_budget,
    )

    summary_budget = max_compaction_chars
    while True:
        raw_starts = {unit.start_index for unit in selected_raw}
        prior_raw = [
            unit for unit in prior_units if unit.start_index in raw_starts
        ]
        progress_raw = [
            unit for unit in progress_units if unit.start_index in raw_starts
        ]
        prior_compacted = [
            unit for unit in prior_units if unit.start_index not in raw_starts
        ]
        progress_compacted = [
            unit
            for unit in progress_units
            if unit.start_index not in raw_starts
        ]
        blocks = _build_compaction_messages(
            prior_compacted,
            progress_compacted,
            summary_budget,
        )
        prior_summaries = [
            block.message for block in blocks if block.kind == "prior"
        ]
        progress_summaries = [
            block.message
            for block in blocks
            if block.kind == "current_run"
        ]
        context = [
            *system_messages,
            *prior_summaries,
            *_flatten_units(prior_raw),
            *anchor_messages,
            *progress_summaries,
            *_flatten_units(progress_raw),
        ]
        context_size = estimate_messages_size(context)
        if context_size <= max_context_chars:
            return deepcopy(context)

        if summary_budget > 0:
            overflow = context_size - max_context_chars
            summary_budget = max(0, summary_budget - max(1, overflow))
            continue
        if selected_raw:
            selected_raw.pop(0)
            continue
        return deepcopy(mandatory_messages)
