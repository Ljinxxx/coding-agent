import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.agent import Agent
from src.llm import LLMClient
from src.tools.files import ListDirectoryTool, ReadFileTool, WriteFileTool
from src.tools.path_utils import WorkspaceBoundaryError, resolve_workspace_path
from src.tools.registry import ToolRegistry
from src.tools.verification import VerifyWorkspaceTool


INITIAL_SOURCE = "def add(a, b):\n    return a - b\n"
TEST_SOURCE = (
    "from calculator import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_read_payload(result: Any) -> str:
    require(isinstance(result, str), "read_file Tool Result 必须是字符串。")
    header, separator, payload = result.partition("\n\n")
    require(
        separator == "\n\n" and header.splitlines()[:1] == ["[read_file]"],
        "read_file Tool Result 缺少预期 metadata header。",
    )
    return payload


def make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


class RecordingLLM:
    """真实调用 LLM，只额外记录每次发送给模型的 messages 和 tools。"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        return self.client.chat(messages, tools=tools)


def parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call["function"]["arguments"]
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
    else:
        parsed = arguments
    require(isinstance(parsed, dict), "真实 Tool Call 参数必须是 JSON 对象。")
    return parsed


def parse_verification_result(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "checks" not in payload:
        return None
    return payload


def collect_tool_events(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_messages: dict[str, tuple[int, dict[str, Any]]] = {}
    for message_index, message in enumerate(history):
        tool_call_id = message.get("tool_call_id")
        if message.get("role") == "tool" and isinstance(tool_call_id, str):
            require(
                tool_call_id not in result_messages,
                f"Tool Call ID {tool_call_id} 出现了重复 Tool Result。",
            )
            result_messages[tool_call_id] = (message_index, message)

    events: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for message_index, message in enumerate(history):
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []):
            call_id = tool_call.get("id")
            require(isinstance(call_id, str), "真实 Tool Call 缺少有效 id。")
            require(
                call_id not in seen_call_ids,
                f"真实模型重复使用了 Tool Call ID {call_id}。",
            )
            seen_call_ids.add(call_id)
            require(call_id in result_messages, f"Tool Call {call_id} 缺少 Tool Result。")
            result_index, result_message = result_messages[call_id]
            require(
                result_index > message_index,
                f"Tool Call {call_id} 的 Tool Result 时序不正确。",
            )
            events.append(
                {
                    "order": len(events),
                    "assistant_index": message_index,
                    "result_index": result_index,
                    "id": call_id,
                    "name": tool_call["function"]["name"],
                    "arguments": parse_tool_arguments(tool_call),
                    "result_message": result_message,
                }
            )
    return events


def event_targets_path(
    event: dict[str, Any],
    workspace: Path,
    target: Path,
) -> bool:
    requested_path = event["arguments"].get("path")
    if not isinstance(requested_path, str):
        return False
    try:
        resolved = resolve_workspace_path(workspace, requested_path)
    except WorkspaceBoundaryError:
        return False
    return resolved == target.resolve(strict=False)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    verification_command = make_command(
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "test_calculator.py",
    )
    temporary_root_path: Path | None = None
    workspace_display = ""
    model_name = ""
    final_answer = ""
    tool_call_summary: list[str] = []
    verification_results: list[dict[str, Any]] = []
    host_verification: dict[str, Any] = {}
    llm_call_count = 0
    final_workspace_revision = -1
    final_verified_revision = -1
    last_mutation_order = -1
    successful_verification_order = -1
    final_message_index = -1
    gate_feedback_count = 0

    with TemporaryDirectory(
        prefix="coding-agent-stage12-real-"
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        calculator_path = workspace / "calculator.py"
        test_path = workspace / "test_calculator.py"
        calculator_path.write_text(INITIAL_SOURCE, encoding="utf-8")
        test_path.write_text(TEST_SOURCE, encoding="utf-8")

        verifier = VerifyWorkspaceTool(
            workspace,
            [verification_command],
            timeout=30,
        )
        registry = ToolRegistry()
        registry.register(ListDirectoryTool(workspace))
        registry.register(ReadFileTool(workspace))
        registry.register(WriteFileTool(workspace))
        registry.register(verifier)

        client = LLMClient()
        real_llm = RecordingLLM(client)
        model_name = client.model
        system_prompt = (
            "You are a coding agent performing a real Verification Loop task. "
            "Use only the provided tools. Inspect both calculator.py and its existing "
            "test before editing. Fix calculator.py without modifying the test file. "
            "Before completing, call verify_workspace with an empty JSON object; that "
            "tool runs the host-configured completion checks, and you cannot choose or "
            "replace its command. If verification fails, use the real failure details "
            "to repair the implementation and verify again. Do not guess or claim a "
            "test result. Give a concise final answer only after verification succeeds, "
            "and mention the modified file and the successful verification."
        )
        task = (
            "这是一个真实的 Coding Agent 验证任务。\n\n"
            "当前 Workspace 中有 calculator.py 和 test_calculator.py。"
            "实现中存在一个错误，请先读取源码与现有测试，再只修改 "
            "calculator.py，使现有测试通过。\n\n"
            "完成前必须调用无参数的 verify_workspace 进行 Host-controlled "
            "真实验证；如果失败，依据真实结果继续修复并再次验证。"
            "只有验证成功后才能给出最终回答。不要修改测试，不要猜测测试结果。"
        )
        require(
            "return a + b" not in system_prompt and "return a + b" not in task,
            "Prompt 不得提前向真实模型提供修复代码。",
        )
        verify_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "verify_workspace"
        )
        require(
            verify_schema["function"]["parameters"]
            == {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "真实模型可见的 verify_workspace Schema 不应接受 command。",
        )
        require(
            "run_command" not in registry.names(),
            "真实模型验证不应注册普通 RunCommandTool。",
        )

        agent = Agent(
            real_llm,
            registry,
            system_prompt=system_prompt,
            verbose=True,
            max_steps=12,
            verification_tool_name="verify_workspace",
        )

        print("Stage 12 真实大模型 Verification Loop 验证")
        print(f"\nModel：{model_name}")
        print(f"Temporary Workspace：{workspace}")
        print(f"Verification Command：{verification_command}")
        print(f"\nInitial calculator.py:\n{INITIAL_SOURCE}")
        print(f"Existing test_calculator.py:\n{TEST_SOURCE}")
        print(f"用户任务：\n{task}\n")

        final_answer = agent.run(task)
        history = agent.history
        events = collect_tool_events(history)

        require(
            isinstance(real_llm.client, LLMClient) and bool(real_llm.calls),
            "验证过程没有通过真实 LLMClient 发起请求。",
        )
        require(events, "真实模型没有产生任何 Tool Call。")
        require(
            all(call["tools"] == registry.schemas() for call in real_llm.calls),
            "真实 LLM API 请求没有持续收到当前 Tool Schema。",
        )

        read_events = [event for event in events if event["name"] == "read_file"]
        calculator_read_events = [
            event
            for event in read_events
            if event_targets_path(event, workspace, calculator_path)
            and (
                parse_read_payload(event["result_message"].get("content"))
                == INITIAL_SOURCE
            )
        ]
        test_read_events = [
            event
            for event in read_events
            if event_targets_path(event, workspace, test_path)
            and (
                parse_read_payload(event["result_message"].get("content"))
                == TEST_SOURCE
            )
        ]
        require(
            calculator_read_events,
            "真实模型没有成功读取 calculator.py 的初始真实内容。",
        )
        require(
            test_read_events,
            "真实模型没有成功读取 test_calculator.py 的初始真实内容。",
        )

        write_events = [event for event in events if event["name"] == "write_file"]
        require(write_events, "真实模型没有调用 write_file 修改 Workspace。")
        first_write_order = min(int(event["order"]) for event in write_events)
        require(
            all(
                int(event["order"]) < first_write_order
                for event in [calculator_read_events[0], test_read_events[0]]
            ),
            "真实模型没有在第一次 write_file 之前成功读取源码和测试。",
        )
        require(
            any(
                event_targets_path(event, workspace, calculator_path)
                for event in write_events
            ),
            "真实模型没有修改 calculator.py。",
        )
        require(
            not any(
                event_targets_path(event, workspace, test_path)
                for event in write_events
            ),
            "真实模型不应调用 write_file 修改 test_calculator.py。",
        )
        require(
            calculator_path.read_text(encoding="utf-8") != INITIAL_SOURCE,
            "真实模型完成后 calculator.py 仍是初始错误实现。",
        )
        require(
            test_path.read_text(encoding="utf-8") == TEST_SOURCE,
            "真实模型不应修改 test_calculator.py。",
        )

        verification_events = [
            event for event in events if event["name"] == "verify_workspace"
        ]
        require(verification_events, "真实模型没有调用 verify_workspace。")
        require(
            all(event["arguments"] == {} for event in verification_events),
            "真实模型向 verify_workspace 传入了不允许的参数。",
        )

        successful_verification_events: list[dict[str, Any]] = []
        for event in verification_events:
            payload = parse_verification_result(
                event["result_message"].get("content")
            )
            require(payload is not None, "verify_workspace 没有返回结构化结果。")
            verification_results.append(payload)
            checks = payload.get("checks")
            require(
                isinstance(checks, list) and bool(checks),
                "Verification Result 必须包含真实 checks。",
            )
            require(
                all(check.get("command") == verification_command for check in checks),
                "Verification Result 执行的不是 Host 配置命令。",
            )
            if payload.get("ok") is True:
                require(
                    all(
                        check.get("exit_code") == 0
                        and check.get("timed_out") is False
                        for check in checks
                    ),
                    "ok=true 的 Verification Result 存在失败或 timeout。",
                )
                successful_verification_events.append(event)

        require(
            successful_verification_events,
            "真实模型流程中没有任何一次成功 Verification。",
        )
        mutation_names = {
            name
            for name in registry.names()
            if registry.get(name).mutates_workspace
        }
        mutation_events = [
            event for event in events if event["name"] in mutation_names
        ]
        require(mutation_events, "没有记录到 mutation-capable Tool Call。")
        last_mutation = mutation_events[-1]
        final_success = successful_verification_events[-1]
        last_mutation_order = int(last_mutation["order"])
        successful_verification_order = int(final_success["order"])
        require(
            last_mutation_order < successful_verification_order,
            "最后一次 Workspace Mutation 没有发生在最终成功 Verification 之前。",
        )
        require(
            not any(
                event["name"] in mutation_names
                and event["order"] > successful_verification_order
                for event in events
            ),
            "最终成功 Verification 后又发生了新的 Workspace Mutation。",
        )

        final_message_indices = [
            index
            for index, message in enumerate(history)
            if message.get("role") == "assistant"
            and not message.get("tool_calls")
            and message.get("content") == final_answer
        ]
        require(final_message_indices, "Agent History 中缺少最终 Assistant Answer。")
        final_message_index = final_message_indices[-1]
        require(
            final_success["result_index"] < final_message_index,
            "Final Answer 没有发生在成功 Verification Result 之后。",
        )
        success_result_message = final_success["result_message"]
        require(
            any(
                success_result_message in call["messages"]
                for call in real_llm.calls
            ),
            "成功 Verification Result 没有进入后续真实 LLM messages。",
        )
        require(
            agent.workspace_revision == len(mutation_events),
            "Workspace Revision 与真实 mutation-capable Tool Call 数量不一致。",
        )
        require(
            agent.workspace_revision == agent.verified_revision
            and not agent.verification_required,
            "真实 Agent 最终没有处于成功验证后的 CLEAN 状态。",
        )
        require(bool(final_answer.strip()), "真实模型最终回答不能为空。")
        final_answer_folded = final_answer.casefold()
        require(
            "calculator.py" in final_answer_folded,
            "真实模型最终回答没有说明修改的文件。",
        )
        require(
            "verify" in final_answer_folded
            or "验证" in final_answer_folded
            or "测试" in final_answer_folded,
            "真实模型最终回答没有说明验证结果。",
        )

        gate_feedback_messages = [
            message
            for message in history
            if message.get("role") == "user"
            and "[Verification Required]" in str(message.get("content") or "")
        ]
        gate_feedback_count = len(gate_feedback_messages)
        for feedback in gate_feedback_messages:
            feedback_index = history.index(feedback)
            require(
                feedback_index > 0
                and history[feedback_index - 1].get("role") == "assistant"
                and not history[feedback_index - 1].get("tool_calls"),
                "Completion Gate Feedback 前缺少被拦截的 Assistant Final。",
            )
            require(
                any(feedback in call["messages"] for call in real_llm.calls),
                "Completion Gate Feedback 没有进入后续真实 LLM messages。",
            )

        host_verification = json.loads(verifier.execute())
        require(
            host_verification.get("ok") is True,
            "Agent 完成后 Host 独立复验最终 Workspace 未通过。",
        )
        require(
            all(
                check.get("exit_code") == 0
                and check.get("timed_out") is False
                for check in host_verification["checks"]
            ),
            "Host 独立复验包含失败或 timeout。",
        )

        tool_call_summary = [
            f"{event['order'] + 1}. {event['name']}"
            f"({json.dumps(event['arguments'], ensure_ascii=False)})"
            for event in events
        ]
        workspace_display = str(workspace)
        llm_call_count = len(real_llm.calls)
        final_workspace_revision = agent.workspace_revision
        final_verified_revision = agent.verified_revision

    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "Stage 12 真实模型验证的临时目录没有被清理。",
    )

    print("\n真实模型 Tool Call 顺序：")
    for summary in tool_call_summary:
        print(summary)
    print("\nVerification Results：")
    for index, payload in enumerate(verification_results, start=1):
        print(f"Verification {index}:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\n真实模型最终回答：")
    print(final_answer)

    print("\n验证结果：")
    print(f"真实 LLMClient 调用次数：{llm_call_count}")
    print(f"模型：{model_name}")
    print(f"Temporary Workspace：{workspace_display}（已清理）")
    print("真实模型读取 calculator.py：通过")
    print("真实模型读取 test_calculator.py：通过")
    print("真实模型修改 calculator.py：通过")
    print("test_calculator.py 保持不变：通过")
    print("真实模型调用无 command 参数的 verify_workspace：通过")
    print("Verification Command 真实执行：通过")
    print("至少一次 Verification 成功：通过")
    print(f"最后一次 Mutation 顺序位置：{last_mutation_order}")
    print(f"最终成功 Verification 顺序位置：{successful_verification_order}")
    print(f"Final History 顺序位置：{final_message_index}")
    print("last mutation < successful verification < final：通过")
    print("成功 Verification 后没有新的 Mutation：通过")
    print(f"Completion Gate 拦截次数：{gate_feedback_count}")
    print(f"最终 workspace_revision：{final_workspace_revision}")
    print(f"最终 verified_revision：{final_verified_revision}")
    print(f"Host 独立复验 ok={host_verification['ok']}：通过")
    print("临时 Workspace 已清理：通过")
    print("\nStage 12 真实大模型 Verification Loop 验证成功")


if __name__ == "__main__":
    main()
