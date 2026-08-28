import copy
import sys
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.llm import LLMClient
from src.tools.registry import ToolRegistry


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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

    token = f"history-real-{uuid4().hex[:8]}"
    system_prompt = (
        "You are performing a conversation-history verification. "
        "Do not call tools. When the user asks you to remember a session marker, "
        "confirm that it has been recorded. If the user later asks for that "
        "marker, recover the exact value only from the conversation history and "
        "include the complete value in your answer. Do not invent a marker."
    )
    first_prompt = (
        f"请记住这个会话标记：{token}。"
        "只需要确认你已经记录，不要进行工具调用。"
    )
    second_prompt = "刚才我要求你记住的会话标记是什么？请回答完整标记。"

    require(
        token not in system_prompt,
        "运行时随机 token 不应提前出现在 system_prompt 中。",
    )
    require(
        token in first_prompt,
        "第一轮用户消息必须包含运行时随机 token。",
    )
    require(
        token not in second_prompt,
        "第二轮 Prompt 不能重复提供运行时随机 token。",
    )

    real_llm = RecordingLLM(LLMClient())
    agent = Agent(
        real_llm,
        ToolRegistry(),
        system_prompt=system_prompt,
        verbose=True,
        max_steps=3,
    )

    print("Stage 9 真实大模型对话历史验证")
    print(f"\n运行时随机会话标记：\n{token}")

    print(f"\nRound 1\n\nUser：\n{first_prompt}")
    first_answer = agent.run(first_prompt)
    print(f"\n真实模型回答：\n{first_answer}")

    first_history = agent.history
    require(
        [message.get("role") for message in first_history]
        == ["system", "user", "assistant"],
        "第一轮结束后的 Agent History 顺序不是 system/user/assistant。",
    )
    require(
        first_history[0].get("content") == system_prompt,
        "第一轮 History 中的 system message 不正确。",
    )
    require(
        first_history[1].get("content") == first_prompt,
        "第一轮 User Message 没有完整保存。",
    )
    require(
        first_history[2].get("content") == first_answer,
        "第一轮 Assistant Final Answer 没有完整保存。",
    )

    print(f"\nRound 2\n\nUser：\n{second_prompt}")
    second_answer = agent.run(second_prompt)
    print(f"\n真实模型回答：\n{second_answer}")

    require(
        len(real_llm.calls) == 2,
        "两轮无工具直接回答应恰好产生 2 次真实 LLM 调用。",
    )

    second_request = real_llm.calls[1]
    expected_second_roles = ["system", "user", "assistant", "user"]

    require(
        [message.get("role") for message in second_request]
        == expected_second_roles,
        "第二轮真实 API 请求的历史顺序不是 system/user/assistant/user。",
    )
    require(
        second_request[0].get("content") == system_prompt,
        "第二轮真实 API 请求中的 system message 不正确。",
    )
    require(
        second_request[1].get("content") == first_prompt,
        "第二轮真实 API 请求没有包含第一轮 User Message。",
    )
    require(
        second_request[2].get("content") == first_answer,
        "第二轮真实 API 请求没有包含第一轮 Assistant Message。",
    )
    require(
        second_request[3].get("content") == second_prompt,
        "第二轮真实 API 请求没有包含新的 User Message。",
    )
    require(
        token in str(second_request[1].get("content", "")),
        "随机 token 没有通过第一轮 User History 进入第二轮真实 API 请求。",
    )
    require(
        token not in str(second_request[0].get("content", ""))
        and token not in str(second_request[3].get("content", "")),
        "第二轮的 system message 或新 User Message 意外泄漏了随机 token。",
    )

    final_history = agent.history
    expected_final_roles = [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    require(
        [message.get("role") for message in final_history]
        == expected_final_roles,
        "两轮结束后的 Agent History 顺序不完整。",
    )
    require(
        final_history[:4] == second_request,
        "Agent 最终 History 与第二轮真实 API 请求中的历史不一致。",
    )
    require(
        final_history[4].get("content") == second_answer,
        "第二轮 Assistant Final Answer 没有进入 Agent History。",
    )
    require(
        token in second_answer,
        "真实模型第二轮回答没有包含历史中的随机 token。",
    )

    print("\n第二轮真实 API 请求中的 messages：")
    print_messages(second_request)

    print("\n两轮结束后的 Agent History：")
    print_messages(final_history)

    print("\n验证结果：")
    print(f"真实 LLM 调用次数：{len(real_llm.calls)}")
    print("第二轮 Prompt 未重复提供随机 token：通过")
    print("第二轮真实 API 请求包含第一轮 User：通过")
    print("第二轮真实 API 请求包含第一轮 Assistant：通过")
    print("第二轮真实 API 请求包含新的 User：通过")
    print("历史消息顺序正确：通过")
    print("同一 Agent 的最终 History 包含两轮对话：通过")
    print("随机 token 仅通过对话历史进入第二轮请求：通过")
    print("真实模型最终回答包含随机 token：通过")

    print("\nStage 9 真实大模型对话历史验证成功")


if __name__ == "__main__":
    main()
