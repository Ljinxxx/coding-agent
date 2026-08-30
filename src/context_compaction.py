import json
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


def _render_tool_exchange(unit: ContextUnit) -> str:
    assistant_message = unit.messages[0]
    calls = assistant_message.get("tool_calls", [])
    call_names: dict[str, str] = {}
    parts = ["[Tool Exchange]"]

    assistant_content = _message_text(assistant_message)
    if assistant_content:
        parts.append(
            "[Assistant]\n"
            + _clip_text(
                assistant_content,
                _ASSISTANT_PREVIEW_CHARS,
                "assistant text",
            )
        )

    for call in calls:
        call_id = str(call.get("id", ""))
        function = call.get("function", {})
        name = str(function.get("name", "<unknown>"))
        call_names[call_id] = name
        arguments = _clip_text(
            _normalized_arguments(call),
            _TOOL_ARGUMENT_PREVIEW_CHARS,
            "tool arguments",
        )
        parts.append(f"[Tool Call]\n{name} arguments={arguments}")

    for result in unit.messages[1:]:
        tool_name = call_names.get(
            str(result.get("tool_call_id", "")),
            "<unknown>",
        )
        result_preview = _clip_text(
            _message_text(result),
            _TOOL_RESULT_PREVIEW_CHARS,
            "tool result",
        )
        parts.append(
            f"[Tool Result]\ntool={tool_name}\n{result_preview}"
        )

    return "\n".join(parts)


def _render_unit(unit: ContextUnit) -> str:
    if unit.kind in {"tool_exchange", "tool_error"}:
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


def _render_digest(header: str, units: list[ContextUnit]) -> str:
    entries = "\n\n".join(_render_unit(unit) for unit in units)
    return f"{header}\n{_HARNESS_NOTICE}\n\n{entries}"


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
