import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.tools.files import ReadFileTool
from src.tools.registry import ToolRegistry


MISSING_CALL_ID = "stage8_read_missing"
CORRECT_CALL_ID = "stage8_read_correct"
FINAL_ANSWER = "stage8-recovery-complete"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def read_file_response(call_id: str, path: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name="read_file",
                    arguments=json.dumps({"path": path}),
                ),
            )
        ],
    )


class FakeLLM:
    def __init__(
        self,
        token: str,
        expected_tools: list[dict[str, Any]],
    ) -> None:
        self.token = token
        self.expected_tools = expected_tools
        self.call_count = 0
        self.error_feedback_checked = False
        self.success_feedback_checked = False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        self.call_count += 1
        require(
            tools == self.expected_tools,
            f"第 {self.call_count} 轮收到的工具 schema 不正确。",
        )

        if self.call_count == 1:
            require(
                [message.get("role") for message in messages] == ["user"],
                "第 1 轮应只包含用户消息。",
            )
            return read_file_response(MISSING_CALL_ID, "missing.txt")

        if self.call_count == 2:
            self._check_error_feedback(messages)
            return read_file_response(CORRECT_CALL_ID, "correct.txt")

        if self.call_count == 3:
            self._check_success_feedback(messages)
            return text_response(FINAL_ANSWER)

        raise RuntimeError("恢复完成后不应发生第 4 次 LLM 调用。")

    def _check_error_feedback(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        require(
            [message.get("role") for message in messages]
            == ["user", "assistant", "tool"],
            "第 2 轮的消息角色顺序不正确。",
        )

        assistant_message = messages[-2]
        tool_message = messages[-1]
        assistant_calls = assistant_message.get("tool_calls")
        require(
            isinstance(assistant_calls, list) and len(assistant_calls) == 1,
            "第 2 轮缺少原始 assistant tool call。",
        )
        require(
            assistant_calls[0].get("id") == MISSING_CALL_ID,
            "第 2 轮 assistant tool call id 不正确。",
        )
        require(
            tool_message.get("tool_call_id") == MISSING_CALL_ID,
            "第 2 轮错误反馈的 tool_call_id 不正确。",
        )

        content = tool_message.get("content")
        require(isinstance(content, str), "工具错误反馈必须是 JSON 字符串。")
        try:
            error_payload = json.loads(content)
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError("工具错误反馈不是合法 JSON。") from error

        expected_payload = {
            "ok": False,
            "tool": "read_file",
            "error_type": "FileNotFoundError",
            "message": "File not found: missing.txt",
        }
        require(
            error_payload == expected_payload,
            f"FileNotFoundError JSON 不正确：{error_payload!r}",
        )
        self.error_feedback_checked = True

    def _check_success_feedback(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        require(
            [message.get("role") for message in messages]
            == ["user", "assistant", "tool", "assistant", "tool"],
            "第 3 轮的消息角色顺序不正确。",
        )

        assistant_message = messages[-2]
        tool_message = messages[-1]
        assistant_calls = assistant_message.get("tool_calls")
        require(
            isinstance(assistant_calls, list) and len(assistant_calls) == 1,
            "第 3 轮缺少原始 assistant tool call。",
        )
        require(
            assistant_calls[0].get("id") == CORRECT_CALL_ID,
            "第 3 轮 assistant tool call id 不正确。",
        )
        require(
            tool_message.get("tool_call_id") == CORRECT_CALL_ID,
            "第 3 轮成功反馈的 tool_call_id 不正确。",
        )
        require(
            tool_message.get("content") == self.token,
            "第 3 轮未收到 correct.txt 中的真实随机 token。",
        )
        self.success_feedback_checked = True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    workspace = (
        project_root / "docs" / "verification" / "stage08_demo_workspace"
    )
    correct_path = workspace / "correct.txt"
    missing_path = workspace / "missing.txt"
    token = f"stage8-{uuid4().hex[:8]}"

    workspace.mkdir(parents=True, exist_ok=True)
    try:
        missing_path.unlink(missing_ok=True)
        correct_path.write_text(token, encoding="utf-8")

        registry = ToolRegistry()
        registry.register(ReadFileTool(workspace))
        llm = FakeLLM(token, registry.schemas())
        agent = Agent(llm, registry, verbose=True, max_steps=5)

        print("Stage 8 错误处理与恢复验证")
        print(f"验证工作目录：{workspace}")
        print(f"运行时随机内容：{token}")
        print("第 1 轮将读取不存在的 missing.txt。\n")

        final_answer = agent.run(
            "读取 missing.txt；如果失败，请改为读取 correct.txt 后完成任务。"
        )

        require(final_answer == FINAL_ANSWER, "Agent 最终回答不正确。")
        require(llm.call_count == 3, "Agent 应恰好调用 LLM 3 次。")
        require(
            llm.error_feedback_checked,
            "未验证 missing.txt 的 FileNotFoundError 反馈。",
        )
        require(
            llm.success_feedback_checked,
            "未验证 correct.txt 的真实读取结果。",
        )
        require(not missing_path.exists(), "missing.txt 不应被创建。")
        require(
            correct_path.read_text(encoding="utf-8") == token,
            "correct.txt 的内容发生了变化。",
        )

        print(f"\nLLM 调用次数：{llm.call_count}")
        print("第 2 轮：已验证 FileNotFoundError JSON 与 tool_call_id")
        print("第 3 轮：已验证 correct.txt 的真实随机 token")
        print(f"Agent 最终回答：{final_answer}")
    finally:
        correct_path.unlink(missing_ok=True)
        missing_path.unlink(missing_ok=True)
        if workspace.exists() and not any(workspace.iterdir()):
            workspace.rmdir()

    print("\nStage 8 错误处理与恢复验证成功")


if __name__ == "__main__":
    main()
