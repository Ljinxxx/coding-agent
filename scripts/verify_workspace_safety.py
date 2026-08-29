import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.tools.files import ListDirectoryTool, ReadFileTool
from src.tools.registry import ToolRegistry


OUTSIDE_READ_CALL_ID = "stage11_read_outside"
LIST_WORKSPACE_CALL_ID = "stage11_list_workspace"
SAFE_READ_CALL_ID = "stage11_read_safe"
FINAL_ANSWER = "workspace-safety-complete"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def tool_response(
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        ],
    )


def parse_single_tool_call(
    message: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    require(
        isinstance(tool_calls, list) and len(tool_calls) == 1,
        "Assistant 消息必须包含且只包含一个 Tool Call。",
    )

    tool_call = tool_calls[0]
    function = tool_call.get("function")
    require(isinstance(function, dict), "Tool Call 缺少 function 数据。")

    arguments_text = function.get("arguments")
    require(isinstance(arguments_text, str), "Tool Call 参数必须是 JSON 字符串。")
    try:
        arguments = json.loads(arguments_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Tool Call 参数不是合法 JSON。") from error

    require(isinstance(arguments, dict), "Tool Call 参数必须是 JSON 对象。")
    return str(tool_call.get("id")), str(function.get("name")), arguments


def parse_error_result(content: Any) -> dict[str, Any]:
    require(isinstance(content, str), "Workspace Boundary 错误结果必须是 JSON 字符串。")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Workspace Boundary 错误结果不是合法 JSON。") from error

    require(isinstance(payload, dict), "Workspace Boundary 错误结果必须是 JSON 对象。")
    return payload


class FakeLLM:
    def __init__(
        self,
        outside_request: str,
        expected_tools: list[dict[str, Any]],
    ) -> None:
        # 仅预先提供要尝试的外部路径；两个文件的随机内容均不会传入。
        self.outside_request = outside_request
        self.expected_tools = expected_tools
        self.calls: list[list[dict[str, Any]]] = []
        self.boundary_feedback_checked = False
        self.directory_feedback_checked = False
        self.safe_read_feedback_checked = False
        self.discovered_safe_filename: str | None = None
        self.observed_safe_content: str | None = None

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
            require(
                [message.get("role") for message in messages] == ["user"],
                "第 1 次 LLM 调用应只包含 User Message。",
            )
            return tool_response(
                OUTSIDE_READ_CALL_ID,
                "read_file",
                {"path": self.outside_request},
            )

        if call_number == 2:
            self._check_boundary_feedback(messages)
            return tool_response(
                LIST_WORKSPACE_CALL_ID,
                "list_directory",
                {"path": "."},
            )

        if call_number == 3:
            self._check_directory_feedback(messages)
            require(
                self.discovered_safe_filename is not None,
                "Fake LLM 未从真实目录结果中发现安全文件。",
            )
            return tool_response(
                SAFE_READ_CALL_ID,
                "read_file",
                {"path": self.discovered_safe_filename},
            )

        if call_number == 4:
            self._check_safe_read_feedback(messages)
            return text_response(FINAL_ANSWER)

        raise RuntimeError("完成 Workspace Safety 验证后不应继续调用 LLM。")

    def _check_boundary_feedback(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        require(
            [message.get("role") for message in messages]
            == ["user", "assistant", "tool"],
            "第 2 次 LLM 调用没有收到完整的 Boundary Error Tool Result。",
        )

        call_id, tool_name, arguments = parse_single_tool_call(messages[-2])
        require(call_id == OUTSIDE_READ_CALL_ID, "外部读取 Tool Call ID 不正确。")
        require(tool_name == "read_file", "第一个 Tool Call 必须是 read_file。")
        require(
            arguments == {"path": self.outside_request},
            "第一个 read_file 没有请求预期的 Workspace 外文件。",
        )

        tool_message = messages[-1]
        require(
            tool_message.get("tool_call_id") == OUTSIDE_READ_CALL_ID,
            "Boundary Error 的 tool_call_id 与原 Tool Call 不匹配。",
        )
        payload = parse_error_result(tool_message.get("content"))
        require(payload.get("ok") is False, "Boundary Error 的 ok 必须为 false。")
        require(
            payload.get("tool") == "read_file",
            "Boundary Error 的 tool 字段不正确。",
        )
        require(
            payload.get("error_type") == "WorkspaceBoundaryError",
            "Workspace 外真实文件必须被 WorkspaceBoundaryError 拒绝。",
        )
        require(
            payload.get("message")
            == f"Path escapes workspace: {self.outside_request}",
            "WorkspaceBoundaryError 消息没有准确指出被拒绝的请求路径。",
        )
        self.boundary_feedback_checked = True

    def _check_directory_feedback(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        require(
            [message.get("role") for message in messages]
            == ["user", "assistant", "tool", "assistant", "tool"],
            "第 3 次 LLM 调用没有收到完整的目录检查 Tool Result。",
        )

        call_id, tool_name, arguments = parse_single_tool_call(messages[-2])
        require(call_id == LIST_WORKSPACE_CALL_ID, "目录检查 Tool Call ID 不正确。")
        require(tool_name == "list_directory", "错误恢复后必须检查 Workspace。")
        require(arguments == {"path": "."}, "Workspace 目录检查路径必须是 '.'。")

        tool_message = messages[-1]
        require(
            tool_message.get("tool_call_id") == LIST_WORKSPACE_CALL_ID,
            "目录 Tool Result 的 tool_call_id 与原 Tool Call 不匹配。",
        )
        content = tool_message.get("content")
        require(isinstance(content, str), "目录 Tool Result 必须是字符串。")
        entries = [entry for entry in content.splitlines() if entry]
        require(
            len(entries) == 1 and not entries[0].endswith("/"),
            "临时 Workspace 应只包含一个可读取的安全文件。",
        )
        self.discovered_safe_filename = entries[0]
        self.directory_feedback_checked = True

    def _check_safe_read_feedback(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        require(
            [message.get("role") for message in messages]
            == [
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "tool",
            ],
            "第 4 次 LLM 调用没有收到完整的安全文件 Tool Result。",
        )

        call_id, tool_name, arguments = parse_single_tool_call(messages[-2])
        require(call_id == SAFE_READ_CALL_ID, "安全读取 Tool Call ID 不正确。")
        require(tool_name == "read_file", "发现安全文件后必须使用 read_file。")
        require(
            arguments == {"path": self.discovered_safe_filename},
            "read_file 没有读取从目录结果中发现的安全文件。",
        )

        tool_message = messages[-1]
        require(
            tool_message.get("tool_call_id") == SAFE_READ_CALL_ID,
            "安全读取 Tool Result 的 tool_call_id 与原 Tool Call 不匹配。",
        )
        content = tool_message.get("content")
        require(
            isinstance(content, str) and bool(content),
            "安全文件的真实 Tool Result 不能为空。",
        )
        self.observed_safe_content = content
        self.safe_read_feedback_checked = True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    safe_filename = f"safe_{uuid4().hex[:8]}.txt"
    secret_filename = f"secret_{uuid4().hex[:8]}.txt"
    safe_token = f"safe-token-{uuid4().hex[:8]}"
    secret_token = f"secret-token-{uuid4().hex[:8]}"
    outside_request = f"../{secret_filename}"

    temporary_root_path: Path | None = None
    workspace_display = ""
    error_payload: dict[str, Any] = {}
    directory_result = ""
    final_answer = ""
    llm_call_count = 0

    with TemporaryDirectory(prefix="coding-agent-stage11-") as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        safe_path = workspace / safe_filename
        secret_path = temporary_root_path / secret_filename
        safe_path.write_text(safe_token, encoding="utf-8")
        secret_path.write_text(secret_token, encoding="utf-8")

        require(secret_path.exists(), "Workspace 外测试文件必须真实存在。")
        require(secret_path.is_file(), "Workspace 外测试目标必须是真实文件。")
        require(safe_path.exists(), "Workspace 内安全文件必须真实存在。")

        registry = ToolRegistry()
        registry.register(ListDirectoryTool(workspace))
        registry.register(ReadFileTool(workspace))
        llm = FakeLLM(outside_request, registry.schemas())
        agent = Agent(llm, registry, max_steps=6)

        user_task = (
            "请验证 Workspace Boundary。第一步必须使用 read_file 读取 "
            f"{outside_request}；如果访问被拒绝，请使用 list_directory 检查 "
            "Workspace，读取其中发现的唯一文本文件，最后完成任务。"
        )
        final_answer = agent.run(user_task)
        history = agent.history

        require(final_answer == FINAL_ANSWER, "Agent 最终回答不正确。")
        require(len(llm.calls) == 4, "完整安全恢复流程应调用 Fake LLM 四次。")
        require(
            llm.boundary_feedback_checked,
            "WorkspaceBoundaryError 没有进入后续 Fake LLM Context。",
        )
        require(
            llm.directory_feedback_checked,
            "合法 list_directory Tool Result 没有进入 Fake LLM Context。",
        )
        require(
            llm.safe_read_feedback_checked,
            "合法 read_file Tool Result 没有进入 Fake LLM Context。",
        )
        require(
            llm.discovered_safe_filename == safe_filename,
            "Fake LLM 没有通过真实目录结果发现随机安全文件。",
        )
        require(
            llm.observed_safe_content == safe_token,
            "Fake LLM 没有收到安全文件的真实随机内容。",
        )

        expected_roles = [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ]
        require(
            [message.get("role") for message in history] == expected_roles,
            "Agent History 没有保留完整的边界拒绝与恢复流程。",
        )

        first_call = parse_single_tool_call(history[1])
        list_call = parse_single_tool_call(history[3])
        safe_call = parse_single_tool_call(history[5])
        require(
            first_call
            == (
                OUTSIDE_READ_CALL_ID,
                "read_file",
                {"path": outside_request},
            ),
            "第一次真实 Tool Call 不是预期的 Workspace 外 read_file。",
        )
        require(
            list_call
            == (LIST_WORKSPACE_CALL_ID, "list_directory", {"path": "."}),
            "Boundary Error 后没有执行预期的合法目录检查。",
        )
        require(
            safe_call
            == (SAFE_READ_CALL_ID, "read_file", {"path": safe_filename}),
            "目录检查后没有读取发现的随机安全文件。",
        )

        boundary_tool_message = history[2]
        error_payload = parse_error_result(boundary_tool_message.get("content"))
        require(
            error_payload
            == {
                "ok": False,
                "tool": "read_file",
                "error_type": "WorkspaceBoundaryError",
                "message": f"Path escapes workspace: {outside_request}",
            },
            "Agent 产生的 WorkspaceBoundaryError 结构化结果不正确。",
        )
        require(
            boundary_tool_message.get("tool_call_id") == OUTSIDE_READ_CALL_ID,
            "结构化 Boundary Error 的 tool_call_id 不正确。",
        )
        require(
            any(
                message == boundary_tool_message
                for later_context in llm.calls[1:]
                for message in later_context
            ),
            "Boundary Error Tool Result 没有真正发送给后续 Fake LLM。",
        )

        directory_result = str(history[4].get("content"))
        require(
            history[4].get("tool_call_id") == LIST_WORKSPACE_CALL_ID,
            "目录 Tool Result 的 tool_call_id 不正确。",
        )
        require(
            directory_result == safe_filename,
            "真实 ListDirectoryTool 没有返回唯一随机安全文件。",
        )
        require(
            history[6].get("tool_call_id") == SAFE_READ_CALL_ID,
            "安全文件 Tool Result 的 tool_call_id 不正确。",
        )
        require(
            history[6].get("content") == safe_token,
            "真实 ReadFileTool 没有返回安全文件的随机 token。",
        )

        tool_results_text = json.dumps(
            [message for message in history if message.get("role") == "tool"],
            ensure_ascii=False,
        )
        history_text = json.dumps(history, ensure_ascii=False)
        llm_contexts_text = json.dumps(llm.calls, ensure_ascii=False)
        require(
            secret_token not in tool_results_text,
            "Workspace 外 secret token 泄露到了 Tool Result。",
        )
        require(
            secret_token not in history_text,
            "Workspace 外 secret token 泄露到了 Agent History。",
        )
        require(
            secret_token not in llm_contexts_text,
            "Workspace 外 secret token 泄露到了 Fake LLM Context。",
        )
        require(
            secret_token not in final_answer,
            "Workspace 外 secret token 泄露到了 Final Answer。",
        )
        require(
            secret_path.read_text(encoding="utf-8") == secret_token,
            "Workspace 外测试文件在拒绝访问后发生了变化。",
        )
        require(
            safe_path.read_text(encoding="utf-8") == safe_token,
            "Workspace 内安全文件在验证期间发生了变化。",
        )

        workspace_display = str(workspace)
        llm_call_count = len(llm.calls)

    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "Stage 11 Fake 验证的临时目录没有被清理。",
    )

    print("Stage 11 Workspace Safety 验证")
    print("\n临时安全验证环境：")
    print(f"Workspace: {workspace_display}")
    print(f"Workspace 内文件: {safe_filename}")
    print(f"Workspace 外文件: {secret_filename}")
    print("Workspace 外文件真实存在：通过")

    print("\nStep 1")
    print("Fake LLM Tool Call:")
    print(f'read_file("{outside_request}")')
    print("Tool Result:")
    print(json.dumps(error_payload, ensure_ascii=False, indent=2))

    print("\nStep 2")
    print("Fake LLM Tool Call:")
    print('list_directory(".")')
    print("Tool Result:")
    print(directory_result)

    print("\nStep 3")
    print("Fake LLM Tool Call:")
    print(f'read_file("{safe_filename}")')
    print("Tool Result:")
    print(safe_token)

    print("\nStep 4")
    print(f"Fake LLM Final Answer: {final_answer}")

    print("\n验证结果：")
    print(f"Fake LLM 调用次数：{llm_call_count}")
    print("Workspace 外文件真实存在：通过")
    print("第一次 Tool Call 请求 Workspace 外真实文件：通过")
    print("Workspace 外读取被 WorkspaceBoundaryError 主动拒绝：通过")
    print("Boundary Error 的 tool_call_id 正确：通过")
    print("Boundary Error Tool Result 返回 Fake LLM：通过")
    print("Boundary Error 后继续进行新的 Tool 决策：通过")
    print("Workspace 内目录可以列出：通过")
    print("随机安全文件通过目录结果发现：通过")
    print("Workspace 内安全文件可以读取：通过")
    print("safe token 正确读取：通过")
    print("secret token 未进入 Tool Result、History、LLM Context 或 Final Answer：通过")
    print("临时验证目录已清理：通过")
    print("\nStage 11 Workspace Safety 验证成功")


if __name__ == "__main__":
    main()
