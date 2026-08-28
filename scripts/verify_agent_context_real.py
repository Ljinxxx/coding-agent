import copy
import json
import sys
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.llm import LLMClient
from src.tools.registry import ToolRegistry


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


def split_history_into_turns(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    system_messages: list[dict[str, Any]] = []
    conversation_messages = messages

    if messages and messages[0].get("role") == "system":
        system_messages = [messages[0]]
        conversation_messages = messages[1:]

    turns: list[list[dict[str, Any]]] = []
    for message in conversation_messages:
        if message.get("role") == "user":
            turns.append([message])
        else:
            require(
                bool(turns),
                "History 中出现了不属于任何 User Turn 的消息。",
            )
            turns[-1].append(message)

    return system_messages, turns


class RecordingLLM:
    """真实调用 LLM，只额外记录每次发送给模型的 messages。"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        self.calls.append(copy.deepcopy(messages))
        return self.client.chat(messages, tools=tools)


def print_messages(messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "<missing>")
        content = message.get("content")
        print(f"{index}. {role}: {content}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    old_marker = f"context-old-{uuid4().hex[:8]}"
    recent_marker = f"context-recent-{uuid4().hex[:8]}"
    system_prompt = (
        "You are performing a context-window verification. Never call tools. "
        "When the user asks you to store a marker, reply only with 已记录。 "
        "Do not repeat that marker or any earlier marker in the acknowledgement. "
        "When the user later asks for the most recently stored marker, recover "
        "the exact complete value solely from the conversation messages that are "
        "available to you, and include it in your answer. Do not invent a marker "
        "and do not mention an older marker."
    )
    old_prompt = (
        f"请记住这个较早的会话标记：{old_marker}。"
        "只回复‘已记录。’，不要复述任何标记，不要调用工具。"
    )
    recent_prompt = (
        f"请记住这个最新的会话标记：{recent_marker}。"
        "只回复‘已记录。’，不要复述任何标记，不要调用工具。"
    )
    current_prompt = (
        "请仅根据本次请求随附的对话上下文，报告我最近一次让你记住的"
        "完整会话标记。回答必须包含完整标记；不要调用工具，也不要猜测。"
    )

    require(
        old_marker not in system_prompt and recent_marker not in system_prompt,
        "运行时随机标记不应提前出现在 system_prompt 中。",
    )
    require(
        old_marker not in current_prompt and recent_marker not in current_prompt,
        "第三轮 Prompt 不能重复提供任何运行时随机标记。",
    )

    real_llm = RecordingLLM(LLMClient())
    agent = Agent(
        real_llm,
        ToolRegistry(),
        system_prompt=system_prompt,
        verbose=True,
        max_steps=3,
        max_context_chars=None,
    )

    print("Stage 10 真实大模型上下文长度管理验证")
    print(f"\n较早随机标记：{old_marker}")
    print(f"最新随机标记：{recent_marker}")

    print(f"\nRound 1\n\nUser：\n{old_prompt}")
    old_answer = agent.run(old_prompt)
    print(f"\n真实模型回答：\n{old_answer}")

    print(f"\nRound 2\n\nUser：\n{recent_prompt}")
    recent_answer = agent.run(recent_prompt)
    print(f"\n真实模型回答：\n{recent_answer}")

    require(
        len(real_llm.calls) == 2,
        "前两轮无工具直接回答应恰好产生 2 次真实 LLM 调用。",
    )
    require(
        agent.max_context_chars is None,
        "前两轮构建完整 History 时不应启用上下文限制。",
    )

    history_before_third = agent.history
    system_messages, turns = split_history_into_turns(history_before_third)

    require(
        len(system_messages) == 1 and len(turns) == 2,
        "第三轮开始前的完整 History 应包含 system 和两个完整 Turn。",
    )
    require(
        turns[0][0].get("content") == old_prompt
        and turns[0][-1].get("content") == old_answer,
        "较早 Turn 没有完整保存在 Agent History 中。",
    )
    require(
        turns[1][0].get("content") == recent_prompt
        and turns[1][-1].get("content") == recent_answer,
        "最新 Turn 没有完整保存在 Agent History 中。",
    )

    current_user_message = {"role": "user", "content": current_prompt}
    retained_candidate = [
        *system_messages,
        *turns[1],
        current_user_message,
    ]
    full_candidate = [*history_before_third, current_user_message]
    retained_candidate_size = estimate_messages_size(retained_candidate)
    full_candidate_size = estimate_messages_size(full_candidate)
    budget = retained_candidate_size

    require(
        retained_candidate_size <= budget,
        "动态预算必须能够容纳 system、最新 Turn 和第三轮 User。",
    )
    require(
        full_candidate_size > budget,
        "动态预算必须排除包含较早 Turn 的完整第三轮 History。",
    )

    agent.max_context_chars = budget

    print("\n动态上下文预算：")
    print(f"保留候选大小：{retained_candidate_size}")
    print(f"完整候选大小：{full_candidate_size}")
    print(f"max_context_chars：{budget}")
    print(f"\nRound 3\n\nUser：\n{current_prompt}")

    final_answer = agent.run(current_prompt)
    print(f"\n真实模型回答：\n{final_answer}")

    require(
        len(real_llm.calls) == 3,
        "三轮无工具直接回答应恰好产生 3 次真实 LLM 调用。",
    )

    third_request = real_llm.calls[-1]
    third_request_size = estimate_messages_size(third_request)
    expected_roles = ["system", "user", "assistant", "user"]

    require(
        [message.get("role") for message in third_request]
        == expected_roles,
        "第三轮真实 API Context 的顺序不是 system/最新 User/最新 Assistant/当前 User。",
    )
    require(
        third_request == retained_candidate,
        "第三轮真实 API Context 没有完整保留 system、最新 Turn 和当前 User。",
    )
    require(
        third_request_size <= budget,
        "第三轮真实 API Context 超出了动态预算。",
    )

    serialized_third_request = json.dumps(
        third_request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    require(
        system_prompt in serialized_third_request,
        "第三轮真实 API Context 没有保留 system prompt。",
    )
    require(
        recent_marker in serialized_third_request,
        "第三轮真实 API Context 没有保留最新随机标记。",
    )
    require(
        current_prompt in serialized_third_request,
        "第三轮真实 API Context 没有保留当前 User Message。",
    )
    require(
        old_marker not in serialized_third_request,
        "第三轮真实 API Context 仍然包含应被裁剪的较早随机标记。",
    )

    final_history = agent.history
    serialized_final_history = json.dumps(
        final_history,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    require(
        old_marker in serialized_final_history,
        "上下文裁剪意外删除了完整 History 中的较早随机标记。",
    )
    require(
        recent_marker in serialized_final_history,
        "上下文裁剪意外删除了完整 History 中的最新随机标记。",
    )
    require(
        final_history[: len(history_before_third)] == history_before_third,
        "第三轮上下文裁剪修改了此前已经保存的完整 History。",
    )
    require(
        final_history[-2].get("content") == current_prompt
        and final_history[-1].get("content") == final_answer,
        "第三轮 User Message 或 Assistant Final Answer 没有进入完整 History。",
    )
    require(
        recent_marker in final_answer,
        "真实模型最终回答没有包含仅通过保留 Context 获得的最新随机标记。",
    )
    require(
        old_marker not in final_answer,
        "真实模型最终回答意外包含了已从 Context 裁剪的较早随机标记。",
    )

    final_history_size = estimate_messages_size(final_history)

    print("\n第三轮真实 API 请求中的 Context：")
    print_messages(third_request)

    print("\n三轮结束后的完整 Agent History：")
    print_messages(final_history)

    print("\n验证结果：")
    print("前两轮 max_context_chars=None 并构建完整 History：通过")
    print("动态预算可容纳 system + 最新 Turn + 当前 User：通过")
    print("动态预算不能容纳包含较早 Turn 的完整 History：通过")
    print("第三轮真实 API Context 始终保留 system prompt：通过")
    print(f"第三轮真实 API Context 保留最新标记 {recent_marker}：通过")
    print(f"第三轮真实 API Context 裁剪较早标记 {old_marker}：通过")
    print("第三轮真实 API Context 保留当前 User Message：通过")
    print("第三轮真实 API Context 消息顺序正确：通过")
    print(f"第三轮真实 API Context 大小：{third_request_size}")
    print(f"完整 Agent History 大小：{final_history_size}")
    print(f"动态上下文预算：{budget}")
    print("完整 Agent History 仍包含较早和最新随机标记：通过")
    print("真实模型最终回答包含最新随机标记：通过")

    print("\nStage 10 真实大模型上下文长度管理验证成功")


if __name__ == "__main__":
    main()
