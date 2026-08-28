import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.tools.registry import ToolRegistry


SYSTEM_PROMPT = "你是一个能够连续维护当前会话历史的 Coding Agent。"
FIRST_ASSISTANT = "recorded"
THIRD_ASSISTANT = "new-conversation-recorded"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def print_messages(messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "<missing-role>")
        content = message.get("content")
        print(f"{index}. {role}: {content}")


class FakeLLM:
    def __init__(
        self,
        token: str,
        expected_tools: list[dict[str, Any]],
    ) -> None:
        self.token = token
        self.expected_tools = expected_tools
        self.calls: list[list[dict[str, Any]]] = []

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

        if call_number == 1:
            return text_response(FIRST_ASSISTANT)

        if call_number == 2:
            return text_response(f"会话标记是：{self.token}")

        if call_number == 3:
            return text_response(THIRD_ASSISTANT)

        raise RuntimeError("完成三轮历史验证后不应继续调用 LLM。")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    token = f"history-{uuid4().hex[:8]}"
    first_user = f"请记住会话标记：{token}"
    second_user = "刚才的会话标记是什么？"
    third_user = "new-conversation"

    registry = ToolRegistry()
    llm = FakeLLM(token, registry.schemas())
    agent = Agent(
        llm,
        registry,
        system_prompt=SYSTEM_PROMPT,
    )

    print("Stage 9 对话历史管理验证")
    print(f"\n运行时随机标记：\n{token}")

    print("\nRound 1")
    print(f"\nUser：\n{first_user}")
    first_answer = agent.run(first_user)
    print(f"\nAssistant：\n{first_answer}")

    require(first_answer == FIRST_ASSISTANT, "第一轮 Assistant 回答不正确。")
    require(len(llm.calls) == 1, "第一轮应恰好调用 LLM 一次。")

    first_history = agent.history
    expected_first_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_user},
        {"role": "assistant", "content": FIRST_ASSISTANT},
    ]
    require(
        first_history == expected_first_history,
        "第一轮结束后的历史未按 system、user、assistant 顺序完整保存。",
    )

    print("\nRound 1 结束后的 Agent History：")
    print_messages(first_history)

    require(
        token not in second_user,
        "第二轮 User Prompt 不得再次包含运行时随机 token。",
    )

    print("\nRound 2")
    print(f"\nUser：\n{second_user}")
    second_answer = agent.run(second_user)
    print(f"\nAssistant：\n{second_answer}")

    require(len(llm.calls) == 2, "第二轮结束后应累计调用 LLM 两次。")
    require(
        token in second_answer,
        "第二轮 Assistant 回答没有返回第一轮历史中的随机 token。",
    )

    second_call_messages = llm.calls[1]
    expected_second_call = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_user},
        {"role": "assistant", "content": FIRST_ASSISTANT},
        {"role": "user", "content": second_user},
    ]
    require(
        second_call_messages == expected_second_call,
        "第二轮 LLM 未收到按顺序排列的第一轮完整历史和第二轮 User 消息。",
    )

    token_sources = [
        message
        for message in second_call_messages
        if token in str(message.get("content") or "")
    ]
    require(
        token_sources == [{"role": "user", "content": first_user}],
        "第二轮 LLM 请求中的随机 token 必须只来自第一轮 User 历史。",
    )
    require(
        sum(
            message.get("role") == "system"
            for message in second_call_messages
        )
        == 1,
        "第二轮 LLM 请求中的 system prompt 必须恰好出现一次。",
    )

    print("\n第二轮 LLM 接收到的 History：")
    print_messages(second_call_messages)

    second_history = agent.history
    require(
        second_history[-1]
        == {"role": "assistant", "content": second_answer},
        "第二轮 Final Assistant Message 未保存到 Agent History。",
    )

    print("\n跨 run 验证结果：")
    print("第一轮 User 已保存：通过")
    print("第一轮 Final Assistant 已保存：通过")
    print("第二轮收到第一轮完整 History：通过")
    print("历史消息顺序正确：通过")
    print("第二轮 Prompt 未重复提供随机 token：通过")
    print("随机 token 仅来自第一轮 History：通过")
    print("system prompt 仅出现一次：通过")

    print("\n执行 reset_history()")
    agent.reset_history()
    reset_history = agent.history

    require(
        reset_history == [{"role": "system", "content": SYSTEM_PROMPT}],
        "reset_history() 后应只保留一条原始 system prompt。",
    )
    require(
        token not in str(reset_history),
        "reset_history() 后仍然存在旧随机 token。",
    )
    require(
        first_user not in str(reset_history),
        "reset_history() 后仍然存在第一轮 User 消息。",
    )
    require(
        FIRST_ASSISTANT not in str(reset_history),
        "reset_history() 后仍然存在第一轮 Assistant 消息。",
    )

    print("\nreset 后 History：")
    print_messages(reset_history)
    print("\n旧随机 token 已清除：通过")
    print("旧 User / Assistant 历史已清除：通过")
    print("system prompt 保留且仅出现一次：通过")

    print("\nRound 3（reset 后的新会话）")
    print(f"\nUser：\n{third_user}")
    third_answer = agent.run(third_user)
    print(f"\nAssistant：\n{third_answer}")

    require(third_answer == THIRD_ASSISTANT, "第三轮 Assistant 回答不正确。")
    require(len(llm.calls) == 3, "第三轮结束后应累计调用 LLM 三次。")

    third_call_messages = llm.calls[2]
    expected_third_call = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": third_user},
    ]
    require(
        third_call_messages == expected_third_call,
        "reset 后第三轮 LLM 请求不应包含任何旧会话消息。",
    )
    require(
        token not in str(third_call_messages),
        "reset 后第三轮 LLM 请求仍然包含旧随机 token。",
    )
    require(
        first_user not in str(third_call_messages)
        and FIRST_ASSISTANT not in str(third_call_messages)
        and second_user not in str(third_call_messages)
        and second_answer not in str(third_call_messages),
        "reset 后第三轮 LLM 请求仍然包含旧 User 或 Assistant 历史。",
    )
    require(
        sum(
            message.get("role") == "system"
            for message in third_call_messages
        )
        == 1,
        "reset 后第三轮 LLM 请求中的 system prompt 必须恰好出现一次。",
    )

    print("\n第三轮 LLM 接收到的 History：")
    print_messages(third_call_messages)
    print("\nRound 3 未收到旧随机 token：通过")
    print("Round 3 未收到 reset 前的 User / Assistant 历史：通过")
    print("Round 3 仍保留一条 system prompt：通过")

    print("\nStage 9 对话历史管理验证成功")


if __name__ == "__main__":
    main()
