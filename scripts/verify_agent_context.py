import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.tools.registry import ToolRegistry


SYSTEM_PROMPT = "你是一个使用最近连续上下文回答问题的 Coding Agent。"
OLD_ASSISTANT = "较旧轮次已记录。"
RECENT_ASSISTANT = "最近轮次已记录。"
CURRENT_ASSISTANT = "已确认最近连续上下文。"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def estimate_messages_size(messages: list[dict[str, Any]]) -> int:
    return len(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def print_messages(messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "<missing-role>")
        content = message.get("content")
        print(f"{index}. {role}: {content}")


class FakeLLM:
    def __init__(self, expected_tools: list[dict[str, Any]]) -> None:
        self.expected_tools = expected_tools
        self.calls: list[list[dict[str, Any]]] = []
        self.responses = [
            OLD_ASSISTANT,
            RECENT_ASSISTANT,
            CURRENT_ASSISTANT,
        ]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        self.calls.append(deepcopy(messages))
        call_number = len(self.calls)

        require(
            tools == self.expected_tools,
            f"第 {call_number} 次 LLM 调用收到的工具 schema 不正确。",
        )
        require(
            call_number <= len(self.responses),
            "完成三轮上下文验证后不应继续调用 LLM。",
        )
        return text_response(self.responses[call_number - 1])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    old_marker = f"old-marker-{uuid4().hex[:8]}"
    recent_marker = f"recent-marker-{uuid4().hex[:8]}"
    old_user = f"请记录这个较旧轮次标记：{old_marker}"
    recent_user = f"请记录这个最近轮次标记：{recent_marker}"
    current_question = "请确认当前上下文是否仍然保留最近一轮对话。"

    system_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    old_turn = [
        {"role": "user", "content": old_user},
        {"role": "assistant", "content": OLD_ASSISTANT},
    ]
    recent_turn = [
        {"role": "user", "content": recent_user},
        {"role": "assistant", "content": RECENT_ASSISTANT},
    ]
    current_turn = [
        {"role": "user", "content": current_question},
    ]

    expected_final_context = system_messages + recent_turn + current_turn
    context_with_old_turn = old_turn + recent_turn + current_turn
    required_recent_size = estimate_messages_size(expected_final_context)
    required_with_old_size = estimate_messages_size(
        system_messages + context_with_old_turn
    )
    budget = required_recent_size

    require(
        required_recent_size <= budget,
        "动态 Budget 无法容纳 system、最近完整 Turn 和当前 Turn。",
    )
    require(
        required_with_old_size > budget,
        "动态 Budget 仍能容纳最旧 Turn，无法验证上下文裁剪。",
    )
    require(
        estimate_messages_size(system_messages + [old_turn[0]]) <= budget,
        "动态 Budget 无法支持第一轮必要 Context。",
    )
    require(
        estimate_messages_size(system_messages + [recent_turn[0]])
        <= budget,
        "动态 Budget 无法支持第二轮必要 Context。",
    )

    registry = ToolRegistry()
    llm = FakeLLM(registry.schemas())
    agent = Agent(
        llm,
        registry,
        system_prompt=SYSTEM_PROMPT,
        max_context_chars=budget,
    )

    print("Stage 10 上下文长度管理验证")
    print("\n运行时随机标记：")
    print(f"old marker: {old_marker}")
    print(f"recent marker: {recent_marker}")
    print(f"\nContext Budget:\n{budget} chars")

    print("\nRound 1")
    print(f"\nUser：\n{old_user}")
    first_answer = agent.run(old_user)
    print(f"\nAssistant：\n{first_answer}")
    require(first_answer == OLD_ASSISTANT, "第一轮 Fake LLM 回答不正确。")

    print("\nRound 2")
    print(f"\nUser：\n{recent_user}")
    second_answer = agent.run(recent_user)
    print(f"\nAssistant：\n{second_answer}")
    require(
        second_answer == RECENT_ASSISTANT,
        "第二轮 Fake LLM 回答不正确。",
    )

    history_before_current = agent.history
    require(
        history_before_current == system_messages + old_turn + recent_turn,
        "第三轮开始前的完整 History 与预期不一致。",
    )

    print("\nRound 3")
    print(f"\nUser：\n{current_question}")
    final_answer = agent.run(current_question)
    print(f"\nAssistant：\n{final_answer}")
    require(
        final_answer == CURRENT_ASSISTANT,
        "第三轮 Fake LLM 回答不正确。",
    )
    require(len(llm.calls) == 3, "三轮验证应恰好调用 Fake LLM 三次。")

    final_context = llm.calls[-1]
    final_context_size = estimate_messages_size(final_context)
    full_history = agent.history
    full_history_size = estimate_messages_size(full_history)
    expected_full_history = (
        system_messages
        + old_turn
        + recent_turn
        + current_turn
        + [{"role": "assistant", "content": CURRENT_ASSISTANT}]
    )

    final_context_text = json.dumps(final_context, ensure_ascii=False)
    full_history_text = json.dumps(full_history, ensure_ascii=False)

    require(
        final_context == expected_final_context,
        "最后一次 Fake LLM Context 未保留正确的最近连续消息顺序。",
    )
    require(
        sum(message.get("role") == "system" for message in final_context)
        == 1,
        "最后一次 Context 必须保留且只保留一条 system prompt。",
    )
    require(
        old_marker not in final_context_text,
        "最后一次 Context 未删除最旧 Turn 的 old marker。",
    )
    require(
        recent_marker in final_context_text,
        "最后一次 Context 未保留最近 Turn 的 recent marker。",
    )
    require(
        current_question in final_context_text,
        "最后一次 Context 未保留当前 Turn。",
    )
    require(
        final_context_size <= budget,
        "最后一次 Context Size 超过 max_context_chars。",
    )
    require(
        full_history == expected_full_history,
        "Context 裁剪修改了 Agent 的完整 History。",
    )
    require(
        old_marker in full_history_text,
        "完整 History 没有保留 old marker。",
    )
    require(
        recent_marker in full_history_text,
        "完整 History 没有保留 recent marker。",
    )
    require(
        current_question in full_history_text,
        "完整 History 没有保留当前 Turn。",
    )
    require(
        full_history_size > final_context_size,
        "完整 History Size 应大于裁剪后的 Context Size。",
    )

    print("\n完整 History：")
    print_messages(full_history)
    print("\n最后一次发送给 Fake LLM 的 Context：")
    print_messages(final_context)
    print("\n字符数：")
    print(f"max_context_chars: {budget}")
    print(f"final context size: {final_context_size}")
    print(f"full history size: {full_history_size}")

    print("\n验证结果：")
    print("完整 History 保留 old marker：通过")
    print("最终 Context 删除 old marker：通过")
    print("最终 Context 保留 recent marker：通过")
    print("最终 Context 保留 current turn：通过")
    print("system prompt 仍然存在：通过")
    print("最近 History 连续且顺序正确：通过")
    print("Context Size <= max_context_chars：通过")
    print("Full History 未被修改：通过")
    print("\nStage 10 上下文长度管理验证成功")


if __name__ == "__main__":
    main()
