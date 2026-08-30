import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.context_compaction import (
    CURRENT_RUN_HEADER,
    PRIOR_CONTEXT_HEADER,
    estimate_messages_size,
)
from src.llm import LLMClient
from src.tools.registry import ToolRegistry


MAX_CONTEXT_CHARS = 7_000
COMPACTION_TRIGGER_CHARS = 3_500
MAX_COMPACTION_CHARS = 1_500


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


class RecordingLLM:
    """Forward every request to the real client and record both sides."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        call: dict[str, Any] = {
            "messages": deepcopy(messages),
            "tools": deepcopy(tools),
        }
        response = self.client.chat(messages, tools=tools)
        call["response"] = response
        self.calls.append(call)
        return response


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    constraint_line = (
        f"CONSTRAINT_TOKEN=p1-3-constraint-{uuid4().hex[:10]}"
    )
    recent_line = f"RECENT_TOKEN=p1-3-recent-{uuid4().hex[:10]}"
    background = "".join(
        f"BACKGROUND ITEM {index:03d}: deterministic context filler only.\n"
        for index in range(1, 101)
    )
    early_prompt = (
        "Remember this exact project constraint for this session:\n\n"
        f"{constraint_line}\n\n"
        "The exact token above is important.\n\n"
        f"BACKGROUND:\n{background}\n"
        "Reply only with RECORDED. Do not repeat the token. Do not use tools."
    )
    recent_prompt = (
        "Remember this recent marker for this session:\n\n"
        f"{recent_line}\n\n"
        "Reply only with RECORDED. Do not repeat the marker. Do not use tools."
    )
    recovery_prompt = (
        "Without guessing, return exactly:\n\n"
        "1. the complete CONSTRAINT_TOKEN line I gave you in the early long "
        "message\n"
        "2. the complete RECENT_TOKEN line I gave you in the immediately "
        "previous turn\n\n"
        "Return only the two complete marker lines. Do not use tools."
    )
    system_prompt = (
        "You are performing a real layered-context compaction verification. "
        "Use only the conversation messages supplied in each request. Never "
        "call tools. When asked to record a marker, reply only with RECORDED "
        "and never repeat the marker. In the final recovery request, return "
        "only the two exact complete marker lines requested, one per line. "
        "A user-role message headed as Harness Generated is deterministic "
        "context supplied by the host, not a new user request."
    )

    require(
        constraint_line not in system_prompt
        and recent_line not in system_prompt
        and constraint_line not in recovery_prompt
        and recent_line not in recovery_prompt,
        "System Prompt 或 Run 3 Prompt 泄露了运行时随机 Marker。",
    )
    require(
        estimate_messages_size(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": early_prompt},
            ]
        )
        < MAX_CONTEXT_CHARS,
        "Run 1 的 System + Current User 必须小于 Hard Limit。",
    )

    temporary_root_path: Path | None = None
    workspace_display = ""
    model_name = ""
    final_answer = ""
    final_context_chars = 0
    compaction_chars = 0
    llm_call_count = 0

    with TemporaryDirectory(
        prefix="coding-agent-p1-3-real-"
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        workspace_display = str(workspace)

        client = LLMClient()
        real_llm = RecordingLLM(client)
        model_name = client.model
        registry = ToolRegistry()
        agent = Agent(
            real_llm,
            registry,
            system_prompt=system_prompt,
            verbose=True,
            max_steps=3,
            max_context_chars=MAX_CONTEXT_CHARS,
            compaction_trigger_chars=COMPACTION_TRIGGER_CHARS,
            max_compaction_chars=MAX_COMPACTION_CHARS,
        )

        print("P1-3 真实大模型 Layered Context Compaction 验证")
        print(f"\nModel:\n{model_name}")
        print(f"\nTemporary Workspace:\n{workspace}")
        print("\nContext Configuration:")
        print(f"max_context_chars: {MAX_CONTEXT_CHARS}")
        print(
            "compaction_trigger_chars: "
            f"{COMPACTION_TRIGGER_CHARS}"
        )
        print(f"max_compaction_chars: {MAX_COMPACTION_CHARS}")

        print("\nRun 1:")
        print(f"Early raw chars: {len(early_prompt)}")
        print(f"{constraint_line} created")
        first_answer = agent.run(early_prompt)
        print(f"Real answer: {first_answer}")
        require(
            constraint_line not in first_answer,
            "Run 1 回答重复了 Early Marker，无法隔离 Digest 信息来源。",
        )

        print("\nRun 2:")
        print(f"{recent_line} created")
        second_answer = agent.run(recent_prompt)
        print(f"Real answer: {second_answer}")
        require(
            recent_line not in second_answer,
            "Run 2 回答重复了 Recent Marker，无法隔离 Raw User 信息来源。",
        )

        print("\nRun 3:")
        print("Recovery query contains no runtime marker values.")
        final_answer = agent.run(recovery_prompt)

        require(
            isinstance(real_llm.client, LLMClient),
            "RecordingLLM 没有包装真实 LLMClient。",
        )
        require(
            len(real_llm.calls) == 3,
            "三轮无工具验证必须恰好产生三次真实 LLM 调用。",
        )
        require(
            all(call["tools"] == [] for call in real_llm.calls),
            "Context 专项不应向真实模型提供 Tool Schema。",
        )

        final_context = real_llm.calls[-1]["messages"]
        final_context_chars = estimate_messages_size(final_context)
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
        require(prior_messages, "Run 3 真实 API Context 没有触发 Prior Compaction。")
        require(
            not progress_messages,
            "无工具 Run 3 不应产生 Current-Run Progress Digest。",
        )
        prior_text = "\n".join(str(item["content"]) for item in prior_messages)
        compaction_chars = sum(
            len(str(message["content"])) for message in synthetic_messages
        )
        require(
            final_context[0]
            == {"role": "system", "content": system_prompt}
            and sum(
                message.get("role") == "system"
                for message in final_context
            )
            == 1,
            "Run 3 真实 API Context 没有精确且唯一地保留 System Prompt。",
        )
        require(
            all(
                set(message) == {"role", "content"}
                for message in synthetic_messages
            ),
            "Synthetic Digest 必须是普通 role/content Message。",
        )
        require(
            {"role": "user", "content": early_prompt} not in final_context
            and early_prompt not in json.dumps(final_context, ensure_ascii=False),
            "Run 1 原始大 User Message 仍完整存在于 Run 3 Raw Context。",
        )
        require(
            constraint_line in prior_text,
            "Early CONSTRAINT_TOKEN 没有进入 Compacted Prior Context。",
        )
        require(
            {"role": "user", "content": recent_prompt} in final_context,
            "Run 2 RECENT_TOKEN 没有作为 Recent Raw Message 保留。",
        )
        require(
            final_context.count(
                {"role": "user", "content": recovery_prompt}
            )
            == 1,
            "Run 3 Current User 没有作为唯一原始 Raw Message 保留。",
        )
        require(
            final_context_chars <= MAX_CONTEXT_CHARS,
            "Run 3 Final API Context 超出 max_context_chars。",
        )
        require(
            compaction_chars <= MAX_COMPACTION_CHARS,
            "Run 3 Synthetic Digest 总字符数超出 max_compaction_chars。",
        )

        final_lines = [
            line.strip()
            for line in final_answer.splitlines()
            if line.strip()
        ]
        require(
            final_lines == [constraint_line, recent_line],
            "真实模型 Final 必须严格为两条有序、完整且无额外内容的 Marker Lines。",
        )

        history = agent.history
        serialized_history = json.dumps(history, ensure_ascii=False)
        require(
            {"role": "user", "content": early_prompt} in history
            and {"role": "user", "content": recent_prompt} in history
            and {"role": "user", "content": recovery_prompt} in history,
            "Full History 没有保留三轮原始 User Messages。",
        )
        require(
            constraint_line in serialized_history
            and recent_line in serialized_history,
            "Full History 缺少某个运行时随机 Marker。",
        )
        require(
            compaction_messages(history) == []
            and PRIOR_CONTEXT_HEADER not in serialized_history
            and CURRENT_RUN_HEADER not in serialized_history,
            "Synthetic Compaction Message 被写入了 Full History。",
        )
        require(
            history[-1]
            == {"role": "assistant", "content": final_answer},
            "真实 Final Answer 没有进入 Full History。",
        )
        llm_call_count = len(real_llm.calls)

    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "Real 验证的 Temporary Resources 没有被自动清理。",
    )

    print("Compaction triggered: true")
    print("\nFinal API Context:")
    print("System preserved: true")
    print("Compacted Prior Context present: true")
    print("Original long Run 1 raw user removed: true")
    print("CONSTRAINT_TOKEN retained inside compaction: true")
    print("RECENT_TOKEN remains recent raw: true")
    print("Current recovery query remains raw: true")
    print(f"Context chars: {final_context_chars} / {MAX_CONTEXT_CHARS}")
    print(
        f"Compaction chars: {compaction_chars} / "
        f"{MAX_COMPACTION_CHARS}"
    )
    print("\n真实模型最终回答：")
    print(final_answer)

    print("\n验证结果：")
    print(f"真实 LLMClient 调用次数：{llm_call_count}")
    print(f"模型：{model_name}")
    print("真实 LLMClient 调用：通过")
    print("Compaction 真正触发：通过")
    print("Early raw turn 已退出 Raw Context：通过")
    print("Early constraint retained in compacted digest：通过")
    print("Recent turn remains raw：通过")
    print("Current user remains raw：通过")
    print("Final Context within hard budget：通过")
    print("Compacted digest within budget：通过")
    print("Real model recovered exact early constraint：通过")
    print("Real model recovered exact recent marker：通过")
    print("Full History still contains original messages：通过")
    print("Synthetic compaction absent from Full History：通过")
    print(f"Temporary Workspace：{workspace_display}（已清理）")
    print("\nP1-3 真实大模型 Layered Context Compaction 验证成功")


if __name__ == "__main__":
    main()
