import json
import sys
from types import SimpleNamespace
from typing import Any

from src.agent import Agent, AgentMaxStepsError
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the provided value."
    parameters = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
            },
        },
        "required": ["value"],
    }

    def __init__(self) -> None:
        self.execute_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.execute_count += 1
        return str(kwargs["value"])


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def tool_response(call_id: str, value: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name="echo",
                    arguments=json.dumps({"value": value}),
                ),
            )
        ],
    )


class SuccessfulFakeLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        self.call_count += 1

        if self.call_count == 1:
            return tool_response("success_call_1", "stage7")
        if self.call_count == 2:
            return text_response("stage7-final-answer")

        raise RuntimeError("场景 A 不应发生第 3 次 LLM 调用。")


class LoopingFakeLLM:
    def __init__(self, allowed_calls: int) -> None:
        self.allowed_calls = allowed_calls
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        self.call_count += 1

        if self.call_count > self.allowed_calls:
            raise RuntimeError("场景 B 不应发生第 4 次 LLM 调用。")

        return tool_response(
            f"loop_call_{self.call_count}",
            f"loop-{self.call_count}",
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_successful_termination() -> None:
    max_steps = 3
    expected_answer = "stage7-final-answer"
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    llm = SuccessfulFakeLLM()
    agent = Agent(llm, registry, verbose=True, max_steps=max_steps)

    print("\n场景 A：在步数上限内正常结束")
    print(f"  max_steps：{max_steps}")

    final_answer = agent.run("先调用 echo，再给出最终回答。")

    require(final_answer == expected_answer, "场景 A 的最终回答不正确。")
    require(llm.call_count == 2, "场景 A 应恰好调用 LLM 2 次。")
    require(tool.execute_count == 1, "场景 A 应恰好执行工具 1 次。")

    print(f"  实际步数（LLM 调用次数）：{llm.call_count}")
    print(f"  工具执行次数：{tool.execute_count}")
    print(f"  最终回答：{final_answer}")
    print("  验证结果：通过")


def verify_max_steps_termination() -> None:
    max_steps = 3
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    llm = LoopingFakeLLM(allowed_calls=max_steps)
    agent = Agent(llm, registry, verbose=True, max_steps=max_steps)

    print("\n场景 B：达到最大步数后强制终止")
    print(f"  max_steps：{max_steps}")

    try:
        agent.run("持续调用 echo。")
    except AgentMaxStepsError as error:
        caught_error = error
    else:
        raise RuntimeError("场景 B 应抛出 AgentMaxStepsError。")

    require(llm.call_count == max_steps, "场景 B 应恰好调用 LLM 3 次。")
    require(tool.execute_count == max_steps, "场景 B 应恰好执行工具 3 次。")
    require(str(max_steps) in str(caught_error), "终止异常应包含 max_steps。")

    print(f"  实际步数（LLM 调用次数）：{llm.call_count}")
    print(f"  工具执行次数：{tool.execute_count}")
    print(f"  捕获异常：{type(caught_error).__name__}: {caught_error}")
    print("  第 4 次 LLM 调用：未发生")
    print("  验证结果：通过")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Stage 7 循环终止机制验证")
    verify_successful_termination()
    verify_max_steps_termination()
    print("\nStage 7 循环终止机制验证成功")


if __name__ == "__main__":
    main()
