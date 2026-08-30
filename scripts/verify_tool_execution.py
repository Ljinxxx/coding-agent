import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.tools.base import BaseTool
from src.tools.files import ReadFileTool
from src.tools.registry import ToolRegistry


SUCCESS_CALL_ID = "fake-success"
UNKNOWN_CALL_ID = "fake-unknown"
RUNTIME_CALL_ID = "fake-runtime"
READ_CALL_ID = "fake-read"
NORMAL_VALUE = "normal-path"
NORMAL_RESULT = f"echo:{NORMAL_VALUE}"
FAILURE_MESSAGE = "P1-4 intentional fake failure"
FINAL_ANSWER = "tool-execution-boundary-complete"
SYSTEM_PROMPT = (
    "You are participating in a deterministic tool-execution boundary "
    "verification. Follow the current request and use only the supplied "
    "tools."
)
USER_TASK = (
    "Exercise the configured tool-execution recovery workflow, then read "
    "target.txt and complete the task."
)
INTERNAL_MESSAGE_KEYS = {
    "execution_ok",
    "error_type",
    "tool_name_internal",
    "mutates_workspace",
    "executor_status",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def serialize(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def tool_response(
    *calls: tuple[str, str, dict[str, Any]],
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
            for call_id, tool_name, arguments in calls
        ],
    )


def parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    require(isinstance(function, dict), "Tool Call 缺少 function 对象。")
    arguments = function.get("arguments")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as error:
        raise RuntimeError("Tool Call arguments 不是合法 JSON。") from error
    require(isinstance(parsed, dict), "Tool Call arguments 必须是 JSON 对象。")
    return parsed


def parse_error_result(message: dict[str, Any]) -> dict[str, Any]:
    require(
        set(message) == {"role", "tool_call_id", "content"},
        "模型可见 Tool Error Message 含有非 Provider 字段。",
    )
    content = message.get("content")
    require(isinstance(content, str), "Structured Tool Error 必须是字符串。")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("Structured Tool Error 不是合法 JSON。") from error
    require(isinstance(payload, dict), "Structured Tool Error 必须是 JSON 对象。")
    require(
        set(payload) == {"ok", "tool", "error_type", "message"},
        "Structured Tool Error 字段不稳定。",
    )
    return payload


def parse_read_result(content: Any) -> tuple[dict[str, str], str]:
    require(isinstance(content, str), "read_file Tool Result 必须是字符串。")
    header, separator, payload = content.partition("\n\n")
    require(separator == "\n\n", "read_file Tool Result 缺少正文分隔。")
    header_lines = header.splitlines()
    require(
        bool(header_lines) and header_lines[0] == "[read_file]",
        "read_file Tool Result 缺少 metadata header。",
    )
    metadata: dict[str, str] = {}
    for line in header_lines[1:]:
        key, delimiter, value = line.partition(": ")
        require(bool(key) and bool(delimiter), "read_file metadata 格式错误。")
        require(key not in metadata, f"read_file metadata 字段重复：{key}。")
        metadata[key] = value
    return metadata, payload


def collect_protocol_events(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_messages: dict[str, tuple[int, dict[str, Any]]] = {}
    result_id_order: list[str] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        require(isinstance(call_id, str), "Tool Result 缺少有效 tool_call_id。")
        require(
            call_id not in result_messages,
            f"Tool Call {call_id} 产生了重复 Tool Result。",
        )
        require(
            set(message) == {"role", "tool_call_id", "content"},
            f"Tool Result {call_id} 含有非 Provider 字段。",
        )
        result_messages[call_id] = (message_index, message)
        result_id_order.append(call_id)

    events: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for assistant_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        require(isinstance(calls, list), "assistant tool_calls 必须是列表。")
        if not calls:
            continue

        expected_ids: list[str] = []
        for call in calls:
            require(isinstance(call, dict), "Tool Call 必须是对象。")
            call_id = call.get("id")
            require(isinstance(call_id, str), "Tool Call 缺少有效 id。")
            require(
                call_id not in seen_call_ids,
                f"Tool Call ID {call_id} 被重复使用。",
            )
            seen_call_ids.add(call_id)
            expected_ids.append(call_id)

        adjacent_results = messages[
            assistant_index + 1 : assistant_index + 1 + len(expected_ids)
        ]
        require(
            len(adjacent_results) == len(expected_ids)
            and all(item.get("role") == "tool" for item in adjacent_results)
            and [item.get("tool_call_id") for item in adjacent_results]
            == expected_ids,
            "Assistant Tool Calls 后未按原顺序紧邻恰好一个对应 Result。",
        )

        for call, expected_id in zip(calls, expected_ids, strict=True):
            require(
                expected_id in result_messages,
                f"Tool Call {expected_id} 缺少 Tool Result。",
            )
            result_index, result_message = result_messages[expected_id]
            require(
                result_index > assistant_index,
                f"Tool Call {expected_id} 的 Result 时序错误。",
            )
            function = call.get("function")
            require(isinstance(function, dict), "Tool Call 缺少 function。")
            name = function.get("name")
            require(isinstance(name, str), "Tool Call 缺少工具名称。")
            events.append(
                {
                    "assistant_index": assistant_index,
                    "result_index": result_index,
                    "id": expected_id,
                    "name": name,
                    "arguments": parse_tool_arguments(call),
                    "result_message": result_message,
                }
            )

    call_id_order = [str(event["id"]) for event in events]
    require(
        call_id_order == result_id_order,
        "Tool Result 数量或顺序与 Parsed Tool Calls 不一致。",
    )
    require(
        len(events) == len(result_messages),
        "History 中存在 orphan Tool Call 或 Tool Result。",
    )
    return events


class EchoDemoTool(BaseTool):
    name = "echo_demo"
    description = "Return the supplied value for execution verification."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, **kwargs: Any) -> str:
        value = str(kwargs["value"])
        self.calls.append(value)
        return f"echo:{value}"


class FailDemoTool(BaseTool):
    name = "fail_demo"
    description = "Intentional verification tool. Call only when instructed."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.execute_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.execute_count += 1
        raise RuntimeError(FAILURE_MESSAGE)


class FakeLLM:
    def __init__(self, expected_tools: list[dict[str, Any]]) -> None:
        self.expected_tools = deepcopy(expected_tools)
        self.calls: list[dict[str, Any]] = []
        self.round_one_results_checked = False
        self.read_result_checked = False
        self.recovered_marker: str | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        call_number = len(self.calls) + 1
        require(
            tools == self.expected_tools,
            f"第 {call_number} 轮 Fake LLM 收到的 Tool Schemas 不正确。",
        )
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )

        if call_number == 1:
            require(
                [message.get("role") for message in messages]
                == ["system", "user"],
                "第 1 轮应只包含 System 与真实 Current User。",
            )
            require(
                messages[-1] == {"role": "user", "content": USER_TASK},
                "第 1 轮 Current User Message 不正确。",
            )
            return tool_response(
                (SUCCESS_CALL_ID, "echo_demo", {"value": NORMAL_VALUE}),
                (UNKNOWN_CALL_ID, "ghost_tool", {}),
                (RUNTIME_CALL_ID, "fail_demo", {}),
            )

        if call_number == 2:
            self._check_round_one_results(messages)
            return tool_response(
                (READ_CALL_ID, "read_file", {"path": "target.txt"})
            )

        if call_number == 3:
            self._check_read_result(messages)
            return text_response(FINAL_ANSWER)

        raise RuntimeError("三轮 Fake 验证完成后不应继续调用 LLM。")

    def _check_round_one_results(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        events = collect_protocol_events(messages)
        require(
            [event["id"] for event in events]
            == [SUCCESS_CALL_ID, UNKNOWN_CALL_ID, RUNTIME_CALL_ID],
            "第 2 轮未收到第一轮全部 Tool Results，或顺序错误。",
        )
        require(
            [event["name"] for event in events]
            == ["echo_demo", "ghost_tool", "fail_demo"],
            "第一轮 Tool Call 名称或顺序错误。",
        )
        require(
            events[0]["arguments"] == {"value": NORMAL_VALUE}
            and events[1]["arguments"] == {}
            and events[2]["arguments"] == {},
            "第一轮 Tool Call 参数不正确。",
        )

        success_message = events[0]["result_message"]
        require(
            success_message.get("content") == NORMAL_RESULT,
            "Registered normal Tool 的原始返回内容被改写。",
        )

        unknown_message = events[1]["result_message"]
        unknown_payload = parse_error_result(unknown_message)
        require(
            unknown_payload
            == {
                "ok": False,
                "tool": "ghost_tool",
                "error_type": "UnknownTool",
                "message": "Tool 'ghost_tool' is not registered.",
            },
            "Unknown Tool 没有归一化为稳定 UnknownTool Error。",
        )

        runtime_message = events[2]["result_message"]
        runtime_payload = parse_error_result(runtime_message)
        require(
            runtime_payload
            == {
                "ok": False,
                "tool": "fail_demo",
                "error_type": "RuntimeError",
                "message": FAILURE_MESSAGE,
            },
            "Registered RuntimeError 没有归一化为稳定 Tool Error。",
        )
        for message in (unknown_message, runtime_message):
            content = str(message.get("content") or "")
            require(
                "Traceback" not in content
                and "File \"" not in content
                and str(Path.cwd()) not in content,
                "Structured Tool Error 泄漏了 Python traceback 或内部路径。",
            )
        self.round_one_results_checked = True

    def _check_read_result(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        events = collect_protocol_events(messages)
        require(
            [event["id"] for event in events]
            == [
                SUCCESS_CALL_ID,
                UNKNOWN_CALL_ID,
                RUNTIME_CALL_ID,
                READ_CALL_ID,
            ],
            "第 3 轮没有收到完整且有序的 ReadFileTool Result。",
        )
        read_event = events[-1]
        require(
            read_event["name"] == "read_file"
            and read_event["arguments"] == {"path": "target.txt"},
            "错误恢复后的 Tool Call 不是预期 read_file(target.txt)。",
        )
        metadata, payload = parse_read_result(
            read_event["result_message"].get("content")
        )
        require(metadata.get("path") == "target.txt", "Read metadata path 错误。")
        payload_lines = [line.strip() for line in payload.splitlines() if line.strip()]
        require(
            len(payload_lines) == 1
            and payload_lines[0].startswith("TARGET_MARKER=p1-4-fake-"),
            "真实 ReadFileTool Result 没有返回唯一随机 TARGET_MARKER。",
        )
        self.recovered_marker = payload_lines[0]
        self.read_result_checked = True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    target_marker = f"TARGET_MARKER=p1-4-fake-{uuid4().hex}"
    marker_suffix = target_marker.removeprefix("TARGET_MARKER=")
    temporary_root_path: Path | None = None
    workspace_display = ""
    history: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    fake_llm: FakeLLM | None = None

    with TemporaryDirectory(
        prefix="coding-agent-p1-4-fake-"
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        workspace_display = str(workspace)
        target_path = workspace / "target.txt"
        target_path.write_text(target_marker + "\n", encoding="utf-8")

        echo_tool = EchoDemoTool()
        fail_tool = FailDemoTool()
        registry = ToolRegistry()
        registry.register(echo_tool)
        registry.register(fail_tool)
        registry.register(ReadFileTool(workspace))
        schemas = registry.schemas()

        require(
            registry.names() == ["echo_demo", "fail_demo", "read_file"],
            "Fake 验证注册工具集合不正确。",
        )
        require(
            "ghost_tool" not in registry.names(),
            "ghost_tool 不应注册到 ToolRegistry。",
        )
        initial_sources = serialize(
            {
                "system": SYSTEM_PROMPT,
                "task": USER_TASK,
                "schemas": schemas,
                "failure_message": FAILURE_MESSAGE,
            }
        )
        require(
            target_marker not in initial_sources
            and marker_suffix not in initial_sources,
            "随机 TARGET_MARKER 被提前泄漏到 Prompt、Schema 或失败消息。",
        )

        fake_llm = FakeLLM(schemas)
        agent = Agent(
            fake_llm,
            registry,
            system_prompt=SYSTEM_PROMPT,
            verbose=True,
            max_steps=3,
            max_context_chars=None,
            verification_tool_name=None,
        )

        print("P1-4 Unified Tool Execution Boundary 验证")
        print(f"\nTemporary Workspace:\n{workspace}")
        print("\nRegistered Tools:")
        for name in registry.names():
            print(name)
        print("\nRound 1 Tool Calls:")
        print(f"1. echo_demo  id={SUCCESS_CALL_ID}")
        print(f"2. ghost_tool id={UNKNOWN_CALL_ID}")
        print(f"3. fail_demo  id={RUNTIME_CALL_ID}")

        final_answer = agent.run(USER_TASK)
        history = agent.history
        events = collect_protocol_events(history)

        require(final_answer == FINAL_ANSWER, "Fake Agent Final Answer 不正确。")
        require(len(fake_llm.calls) == 3, "Fake LLM 必须恰好调用三次。")
        require(
            fake_llm.round_one_results_checked,
            "Fake LLM 未在下一轮检查前三个 Tool Results。",
        )
        require(
            fake_llm.read_result_checked,
            "Fake LLM 未在下一轮检查真实 ReadFileTool Result。",
        )
        require(
            fake_llm.recovered_marker == target_marker,
            "Fake LLM 没有从真实 ReadFileTool Result 恢复随机 Marker。",
        )
        require(
            echo_tool.calls == [NORMAL_VALUE],
            "Registered normal Tool 没有恰好执行一次。",
        )
        require(
            fail_tool.execute_count == 1,
            "Unknown Tool Error 阻止了后续 FailDemo sibling 执行。",
        )

        expected_ids = [
            SUCCESS_CALL_ID,
            UNKNOWN_CALL_ID,
            RUNTIME_CALL_ID,
            READ_CALL_ID,
        ]
        require(
            [event["id"] for event in events] == expected_ids,
            "Full History 的 Tool Call IDs 或顺序错误。",
        )
        require(
            [event["name"] for event in events]
            == ["echo_demo", "ghost_tool", "fail_demo", "read_file"],
            "Full History 的 Tool Call 顺序错误。",
        )
        require(
            [message.get("role") for message in history]
            == [
                "system",
                "user",
                "assistant",
                "tool",
                "tool",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
            "Full History 的 Native Tool Call / Result 协议顺序不正确。",
        )
        require(
            history[-1] == {"role": "assistant", "content": FINAL_ANSWER},
            "Final Answer 没有原样写入 Full History。",
        )

        first_results = [event["result_message"] for event in events[:3]]
        second_request = fake_llm.calls[1]["messages"]
        require(
            all(message in second_request for message in first_results),
            "第一轮 Tool Results 没有全部进入紧接着的 Fake LLM messages。",
        )
        read_result = events[-1]["result_message"]
        require(
            read_result in fake_llm.calls[2]["messages"],
            "ReadFileTool Result 没有进入紧接着的 Fake LLM messages。",
        )

        for call in fake_llm.calls:
            for message in call["messages"]:
                require(
                    not (set(message) & INTERNAL_MESSAGE_KEYS),
                    "Harness 内部 ToolExecutionResult 字段泄漏给 Provider。",
                )

        require(
            target_marker not in serialize(fake_llm.calls[0]["messages"])
            and target_marker not in serialize(fake_llm.calls[1]["messages"]),
            "随机 TARGET_MARKER 在真实 ReadFileTool Result 前进入了模型消息。",
        )
        read_result_index = int(events[-1]["result_index"])
        require(
            target_marker not in serialize(history[:read_result_index]),
            "随机 TARGET_MARKER 在 ReadFileTool Result 前进入了 Full History。",
        )
        require(
            str(read_result.get("content") or "").count(target_marker) == 1,
            "ReadFileTool Result 没有且仅有一个完整随机 TARGET_MARKER。",
        )
        marker_sources: list[tuple[int, str, Any]] = []
        for call_index, call in enumerate(fake_llm.calls):
            for message in call["messages"]:
                if target_marker in serialize(message):
                    marker_sources.append(
                        (
                            call_index,
                            str(message.get("role")),
                            message.get("tool_call_id"),
                        )
                    )
        require(
            marker_sources == [(2, "tool", READ_CALL_ID)],
            "随机 TARGET_MARKER 首次模型可见来源不是 ReadFileTool Result。",
        )
        require(
            agent.workspace_revision == 0
            and agent.verified_revision == 0
            and not agent.verification_required,
            "P1-4 Fake 的只读/非 mutating Tools 不应改变 Verification State。",
        )
        require(
            target_path.read_text(encoding="utf-8") == target_marker + "\n",
            "临时 target.txt 在验证期间被意外修改。",
        )

    require(fake_llm is not None, "Fake LLM 验证未初始化。")
    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "Fake 验证的 Temporary Workspace 没有被自动清理。",
    )

    print("\nExecution Results:")
    print(f"echo_demo: execution_ok=true id={SUCCESS_CALL_ID}")
    print(
        "ghost_tool: execution_ok=false error_type=UnknownTool "
        f"id={UNKNOWN_CALL_ID} traceback leaked=false"
    )
    print(
        "fail_demo: execution_ok=false error_type=RuntimeError "
        f"id={RUNTIME_CALL_ID} traceback leaked=false"
    )
    print("\nRound 2:")
    print("Previous Tool Results received by Fake LLM: 3")
    print(f"Model recovered and requested read_file id={READ_CALL_ID}")
    print("\nRound 3:")
    print(f"Recovered marker: {fake_llm.recovered_marker}")
    print(f"Final: {FINAL_ANSWER}")

    print("\n验证结果：")
    print("Registered Tool normal result：通过")
    print("Unknown Tool structured error：通过")
    print("Runtime exception structured error：通过")
    print("No traceback leaked：通过")
    print("Tool call IDs preserved：通过")
    print("Exactly one result per call：通过")
    print("Execution order preserved：通过")
    print("Sibling execution continues after error：通过")
    print("Tool Results entered next LLM messages：通过")
    print("Internal execution fields absent from Provider messages：通过")
    print("Agent recovered after Tool Error：通过")
    print("Later ReadFileTool succeeded：通过")
    print("Random target not leaked before ReadFileTool Result：通过")
    print("Random target recovered from real Tool Result：通过")
    print("Full History protocol pairs valid：通过")
    print(f"Temporary Workspace：{workspace_display}（已清理）")
    print("\nP1-4 Unified Tool Execution Boundary 验证成功")


if __name__ == "__main__":
    main()
