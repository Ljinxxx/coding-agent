import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.context_compaction import (
    CURRENT_RUN_HEADER,
    PRIOR_CONTEXT_HEADER,
    estimate_messages_size,
)
from src.tools.files import ReadFileTool
from src.tools.registry import ToolRegistry


MAX_CONTEXT_CHARS = 5_500
COMPACTION_TRIGGER_CHARS = 2_500
MAX_COMPACTION_CHARS = 1_400
READ_MAX_OUTPUT_CHARS = 900
SYSTEM_PROMPT = (
    "You are participating in a deterministic layered-context verification. "
    "Follow the current request and use only the provided tool when asked."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def tool_response(call_id: str, path: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name="read_file",
                    arguments=json.dumps(
                        {"path": path},
                        ensure_ascii=False,
                    ),
                ),
            )
        ],
    )


def compaction_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if str(message.get("content") or "").startswith(
            (PRIOR_CONTEXT_HEADER, CURRENT_RUN_HEADER)
        )
    ]


def native_tool_ids(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    call_ids = [
        str(call["id"])
        for message in messages
        for call in message.get("tool_calls", [])
    ]
    result_ids = [
        str(message["tool_call_id"])
        for message in messages
        if message.get("role") == "tool"
    ]
    return call_ids, result_ids


def require_atomic_native_exchanges(
    messages: list[dict[str, Any]],
) -> None:
    for index, message in enumerate(messages):
        calls = message.get("tool_calls", [])
        if message.get("role") != "assistant" or not calls:
            continue
        expected_ids = [str(call["id"]) for call in calls]
        results = messages[index + 1 : index + 1 + len(expected_ids)]
        require(
            len(results) == len(expected_ids)
            and all(result.get("role") == "tool" for result in results)
            and [str(result.get("tool_call_id")) for result in results]
            == expected_ids,
            "Native Assistant Tool Call 后没有紧邻全部匹配 Tool Results。",
        )


class FakeLLM:
    def __init__(self, expected_tools: list[dict[str, Any]]) -> None:
        self.expected_tools = expected_tools
        self.calls: list[dict[str, Any]] = []
        self.responses = [
            text_response("run-1-recorded"),
            text_response("run-2-recorded"),
            tool_response("fake-read-1", "notes_1.txt"),
            tool_response("fake-read-2", "notes_2.txt"),
            tool_response("fake-read-3", "notes_3.txt"),
            tool_response("fake-read-4", "notes_4.txt"),
            text_response("run-3-complete"),
        ]
        self.agent: Agent | None = None
        self.current_user_index: int | None = None
        self.deterministic_context = False
        self.history_snapshots: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        call_number = len(self.calls) + 1
        require(
            tools == self.expected_tools,
            f"第 {call_number} 次 Fake LLM 调用收到的 Tool Schema 不正确。",
        )
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )
        if self.agent is not None:
            self.history_snapshots.append(self.agent.history)
        require(
            call_number <= len(self.responses),
            "Fake LLM 收到了计划外的额外调用。",
        )

        if call_number == len(self.responses):
            require(
                self.agent is not None and self.current_user_index is not None,
                "Determinism 检查缺少 Agent 或 Current User Anchor。",
            )
            first = self.agent._build_context_messages(
                current_user_index=self.current_user_index
            )
            second = self.agent._build_context_messages(
                current_user_index=self.current_user_index
            )
            self.deterministic_context = first == second == messages

        return self.responses[call_number - 1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    early_constraint = f"EARLY_CONSTRAINT={uuid4().hex[:10]}"
    recent_marker = f"RECENT_MARKER={uuid4().hex[:10]}"
    current_marker = f"CURRENT_MARKER={uuid4().hex[:10]}"
    old_noise_middle = f"OLD_RAW_NOISE_MIDDLE={uuid4().hex[:10]}"
    old_noise = (
        "OLD_RAW_NOISE_START\n"
        + "A" * 1_350
        + f"\n{old_noise_middle}\n"
        + "B" * 1_350
        + "\nOLD_RAW_NOISE_END"
    )
    early_user = (
        f"{early_constraint}\n"
        "Keep this exact constraint available for later turns.\n"
        f"{old_noise}"
    )
    recent_user = f"{recent_marker}\nKeep this recent marker available."
    current_user = (
        f"{current_marker}\n"
        "Read notes_1.txt through notes_4.txt in order using read_file."
    )

    temporary_root_path: Path | None = None
    workspace_display = ""
    prior_digest_chars = 0
    progress_digest_chars = 0
    final_context_chars = 0
    raw_call_ids: list[str] = []

    with TemporaryDirectory(
        prefix="coding-agent-p1-3-fake-"
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        workspace_display = str(workspace)
        for index in range(1, 5):
            content = "".join(
                f"NOTES_{index}_LINE_{line:03d}: "
                + chr(64 + index) * 42
                + "\n"
                for line in range(1, 41)
            )
            (workspace / f"notes_{index}.txt").write_text(
                content,
                encoding="utf-8",
            )

        registry = ToolRegistry()
        registry.register(
            ReadFileTool(
                workspace,
                default_max_lines=100,
                max_output_chars=READ_MAX_OUTPUT_CHARS,
            )
        )
        fake_llm = FakeLLM(registry.schemas())
        agent = Agent(
            fake_llm,
            registry,
            system_prompt=SYSTEM_PROMPT,
            max_steps=5,
            max_context_chars=MAX_CONTEXT_CHARS,
            compaction_trigger_chars=COMPACTION_TRIGGER_CHARS,
            max_compaction_chars=MAX_COMPACTION_CHARS,
        )
        fake_llm.agent = agent

        print("P1-3 Layered Context Compaction 验证")
        print(f"\nTemporary Workspace:\n{workspace}")
        print("\nContext Configuration:")
        print(f"max_context_chars: {MAX_CONTEXT_CHARS}")
        print(
            "compaction_trigger_chars: "
            f"{COMPACTION_TRIGGER_CHARS}"
        )
        print(f"max_compaction_chars: {MAX_COMPACTION_CHARS}")

        print(f"\nRun 1:\n{early_constraint} added")
        require(
            agent.run(early_user) == "run-1-recorded",
            "Run 1 Fake Final 不正确。",
        )
        print(f"\nRun 2:\n{recent_marker} added")
        require(
            agent.run(recent_user) == "run-2-recorded",
            "Run 2 Fake Final 不正确。",
        )

        fake_llm.current_user_index = len(agent.history)
        print(f"\nRun 3:\n{current_marker} added")
        for index in range(1, 5):
            print(f"Tool Exchange {index}: read_file(notes_{index}.txt)")
        require(
            agent.run(current_user) == "run-3-complete",
            "Run 3 Fake Final 不正确。",
        )
        require(
            len(fake_llm.calls) == 7,
            "三次 run 加四次 Tool Exchange 应恰好调用 Fake LLM 七次。",
        )

        final_context = fake_llm.calls[-1]["messages"]
        synthetic_messages = compaction_messages(final_context)
        prior_messages = [
            message
            for message in synthetic_messages
            if str(message["content"]).startswith(PRIOR_CONTEXT_HEADER)
        ]
        progress_messages = [
            message
            for message in synthetic_messages
            if str(message["content"]).startswith(CURRENT_RUN_HEADER)
        ]
        require(prior_messages, "Final Context 缺少 Compacted Prior Context。")
        require(
            progress_messages,
            "Final Context 缺少 Compacted Current-Run Progress。",
        )
        prior_text = "\n".join(str(item["content"]) for item in prior_messages)
        progress_text = "\n".join(
            str(item["content"]) for item in progress_messages
        )
        serialized_context = json.dumps(final_context, ensure_ascii=False)
        prior_digest_chars = sum(
            len(str(message["content"])) for message in prior_messages
        )
        progress_digest_chars = sum(
            len(str(message["content"])) for message in progress_messages
        )
        compacted_chars = prior_digest_chars + progress_digest_chars
        final_context_chars = estimate_messages_size(final_context)

        require(
            final_context[0]
            == {"role": "system", "content": SYSTEM_PROMPT}
            and sum(
                message.get("role") == "system"
                for message in final_context
            )
            == 1,
            "Final Context 没有精确且唯一地保留 System Prompt。",
        )
        require(
            all(
                set(message) == {"role", "content"}
                for message in synthetic_messages
            ),
            "Synthetic Digest 必须是普通 role/content Message。",
        )

        require(
            early_constraint in prior_text,
            "EARLY_CONSTRAINT 没有保留在 Prior Digest。",
        )
        require(
            {"role": "user", "content": early_user} not in final_context,
            "Run 1 原始大 User Message 仍然作为 Raw Message 出现。",
        )
        require(
            old_noise not in serialized_context
            and old_noise_middle not in serialized_context,
            "原始旧大 noise 没有被移除。",
        )
        recent_is_raw = (
            {"role": "user", "content": recent_user} in final_context
        )
        require(
            recent_marker in serialized_context,
            "RECENT_MARKER 没有保留在 Final Context。",
        )
        require(
            final_context.count({"role": "user", "content": current_user}) == 1,
            "CURRENT_MARKER 没有作为唯一的原始 Current User 保留。",
        )
        require(
            "notes_1.txt" in progress_text,
            "较旧的 current-run Tool Exchange 没有进入 Progress Digest。",
        )
        raw_call_ids, raw_result_ids = native_tool_ids(final_context)
        require(
            "fake-read-1" not in raw_call_ids
            and "fake-read-1" not in raw_result_ids,
            "较旧 Tool Exchange 仍然处于 Native Raw Context。",
        )
        require(
            "fake-read-4" in raw_call_ids
            and "fake-read-4" in raw_result_ids,
            "最新 Tool Call / Result 没有保持 Native Raw。",
        )
        require(
            raw_call_ids == raw_result_ids,
            "Final Context 中存在 orphan Tool Call 或 Tool Result。",
        )
        require_atomic_native_exchanges(final_context)
        require(
            compacted_chars <= MAX_COMPACTION_CHARS,
            "所有 Synthetic Digest 的总字符数超过预算。",
        )
        require(
            final_context_chars <= MAX_CONTEXT_CHARS,
            "Final Context 超过 Stage 10 Hard Limit。",
        )
        require(
            fake_llm.deterministic_context,
            "同一 History 与 Anchor 的两次 Compaction 结果不一致。",
        )

        history = agent.history
        serialized_history = json.dumps(history, ensure_ascii=False)
        history_call_ids, history_result_ids = native_tool_ids(history)
        require(
            {"role": "user", "content": early_user} in history
            and {"role": "user", "content": recent_user} in history
            and {"role": "user", "content": current_user} in history,
            "Full History 缺少某个原始 User Message。",
        )
        require(
            early_constraint in serialized_history
            and recent_marker in serialized_history
            and current_marker in serialized_history,
            "Full History 缺少运行时随机 Marker。",
        )
        require(
            history_call_ids
            == [f"fake-read-{index}" for index in range(1, 5)]
            and history_result_ids == history_call_ids,
            "Full History 没有完整保留所有 Tool Calls / Results。",
        )
        first_seen_results: dict[str, dict[str, Any]] = {}
        for snapshot in fake_llm.history_snapshots:
            for message in snapshot:
                result_id = message.get("tool_call_id")
                if message.get("role") == "tool" and isinstance(result_id, str):
                    first_seen_results.setdefault(result_id, message)
        final_results = {
            str(message["tool_call_id"]): message
            for message in history
            if message.get("role") == "tool"
        }
        require(
            final_results == first_seen_results
            and all(
                str(message.get("content") or "").startswith("[read_file]\n")
                and f"path: notes_{index}.txt"
                in str(message.get("content") or "")
                and f"NOTES_{index}_LINE_001"
                in str(message.get("content") or "")
                for index, message in enumerate(
                    final_results.values(),
                    start=1,
                )
            ),
            "Full History 中原始 ReadFileTool Result 内容不完整或被改写。",
        )
        require_atomic_native_exchanges(history)
        require(
            compaction_messages(history) == []
            and PRIOR_CONTEXT_HEADER not in serialized_history
            and CURRENT_RUN_HEADER not in serialized_history,
            "Synthetic Compaction Message 被写入了 Full History。",
        )
        require(
            agent.workspace_revision == 0
            and agent.verified_revision == 0
            and not agent.verification_required,
            "只读 ReadFileTool 不应改变 Stage 12 Revision 状态。",
        )

        recent_raw_units = [
            message.get("content")
            for message in final_context
            if message.get("role") == "user"
            and message not in synthetic_messages
        ]

    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "Fake 验证的 Temporary Workspace 没有被自动清理。",
    )

    print("\nFinal Context Layers:")
    print("System: preserved")
    print("Compacted Prior Context: present")
    print(f"Compacted Prior chars: {prior_digest_chars}")
    print("Current User: CURRENT_MARKER present raw")
    print("Compacted Current-Run Progress: present")
    print(f"Compacted Current-Run chars: {progress_digest_chars}")
    print(
        "RECENT_MARKER location: "
        + ("recent raw" if recent_is_raw else "compacted prior digest")
    )
    print(f"Recent Raw User Units: {len(recent_raw_units)}")
    print(f"Native Raw Tool Call IDs: {', '.join(raw_call_ids)}")
    print(f"Context chars: {final_context_chars} / {MAX_CONTEXT_CHARS}")

    print("\n验证结果：")
    print("Compaction triggered：通过")
    print("Early constraint retained in compacted digest：通过")
    print("Old raw noise removed：通过")
    print("Recent marker preserved：通过")
    print("Current user preserved raw：通过")
    print("Old current-run progress compacted：通过")
    print("Recent Tool Exchange remains atomic raw：通过")
    print("No orphan Tool Result：通过")
    print("Compacted digest bounded：通过")
    print("Final Context bounded：通过")
    print("Full History remains complete：通过")
    print("Synthetic compaction message absent from Full History：通过")
    print("Deterministic compaction：通过")
    print(f"Temporary Workspace：{workspace_display}（已清理）")
    print("\nP1-3 Layered Context Compaction 验证成功")


if __name__ == "__main__":
    main()
