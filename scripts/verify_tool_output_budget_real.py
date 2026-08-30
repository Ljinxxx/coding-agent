import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from secrets import randbelow
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.llm import LLMClient
from src.tools.files import ReadFileTool
from src.tools.registry import ToolRegistry
from src.tools.shell import RunCommandTool


LARGE_FILE_NAME = "large_notes.txt"
OUTPUT_SCRIPT_NAME = "generate_output.py"
TOTAL_LINES = 900
READ_MAX_LINES = 150
READ_MAX_OUTPUT_CHARS = 6_000
SHELL_MAX_OUTPUT_CHARS = 600
MIDDLE_FILL_CHARS = 12_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


class RecordingLLM:
    """Call the real client unchanged while recording API inputs."""

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


def collect_tool_events(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_messages: dict[str, tuple[int, dict[str, Any]]] = {}
    for message_index, message in enumerate(history):
        call_id = message.get("tool_call_id")
        if message.get("role") != "tool" or not isinstance(call_id, str):
            continue
        require(
            call_id not in result_messages,
            f"Tool Call ID {call_id} 出现了重复 Tool Result。",
        )
        result_messages[call_id] = (message_index, message)

    events: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for message_index, message in enumerate(history):
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            require(
                len(tool_calls) == 1,
                "本专项要求真实模型每轮只产生一个 Tool Call。",
            )

        for tool_call in tool_calls:
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


def parse_read_result(content: Any) -> tuple[dict[str, str], str]:
    require(isinstance(content, str), "read_file Tool Result 必须是字符串。")
    parts = content.split("\n\n", 1)
    require(len(parts) == 2, "read_file Tool Result 缺少元数据与正文分隔。")
    header, payload = parts
    header_lines = header.splitlines()
    require(
        bool(header_lines) and header_lines[0] == "[read_file]",
        "read_file Tool Result 缺少元数据头。",
    )

    metadata: dict[str, str] = {}
    for line in header_lines[1:]:
        key, separator, value = line.partition(": ")
        require(bool(separator) and bool(key), "read_file 元数据行格式无效。")
        require(key not in metadata, f"read_file 元数据字段 {key} 重复。")
        metadata[key] = value
    return metadata, payload


def parse_shell_result(content: Any) -> dict[str, Any]:
    require(isinstance(content, str), "run_command Tool Result 必须是字符串。")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("run_command Tool Result 不是有效 JSON。") from error
    require(isinstance(payload, dict), "run_command Tool Result 必须是 JSON 对象。")
    return payload


def serialize(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    target_token = f"p1-2-target-{uuid4().hex}"
    head_token = f"p1-2-head-{uuid4().hex}"
    middle_token = f"p1-2-middle-{uuid4().hex}"
    tail_token = f"p1-2-tail-{uuid4().hex}"
    target_line_number = 721 + randbelow(90)

    target_marker = f"TARGET_MARKER={target_token}"
    head_marker = f"HEAD_MARKER={head_token}"
    middle_marker = f"MIDDLE_MARKER={middle_token}"
    tail_marker = f"TAIL_MARKER={tail_token}"
    python_command = make_command(
        sys.executable,
        "-B",
        OUTPUT_SCRIPT_NAME,
    )

    system_prompt = (
        "You are performing a real Tool Output Budget integration task. "
        "Use only the provided read_file and run_command tools. Issue exactly "
        "one tool call per response. Your first action must be read_file with "
        f'exactly {{"path": "{LARGE_FILE_NAME}"}} so the host default window '
        "is exercised. If its metadata says more content remains, call "
        "read_file again with only path and start_line, setting start_line "
        "to the exact next_start_line value from the preceding result. Do not "
        "set max_lines. Continue until you actually observe the complete line "
        "beginning TARGET_MARKER=. Never use run_command, Python, or shell "
        f"commands to inspect {LARGE_FILE_NAME}. Never read "
        f"{OUTPUT_SCRIPT_NAME}. After observing TARGET_MARKER, call "
        "run_command exactly once with the exact host-provided command in the "
        "user task. Do not alter or wrap that command. Read HEAD_MARKER and "
        "TAIL_MARKER only from the real run_command result. Do not guess values. "
        "Your final answer must include the three complete lines "
        "TARGET_MARKER=..., HEAD_MARKER=..., and TAIL_MARKER=...."
    )
    task = (
        "这是一次真实 Tool Output Budget 集成验证。\n\n"
        f"Workspace 中有 {LARGE_FILE_NAME} 和 {OUTPUT_SCRIPT_NAME}。\n"
        f"1. 第一项操作必须仅使用 path={LARGE_FILE_NAME} 的默认 read_file。\n"
        "2. 第一页找不到 TARGET_MARKER 时，依据返回元数据中的 "
        "next_start_line 逐页继续；后续调用只传 path 和 start_line，"
        "不要传 max_lines，也不要跳页。\n"
        f"3. 禁止使用 run_command 或任何 Shell/Python 命令读取 "
        f"{LARGE_FILE_NAME}，也禁止读取 {OUTPUT_SCRIPT_NAME} 的源码。\n"
        "4. 真实读到完整 TARGET_MARKER 行后，必须且只能通过 "
        "run_command 执行下面这条 Host 提供的精确命令，不得修改、"
        "包装或替换命令：\n"
        f"{python_command}\n"
        "5. 从真实 run_command Tool Result 中读取 HEAD_MARKER 与 "
        "TAIL_MARKER。中间输出会被 Host 截断，不要猜测被截断内容。\n"
        "6. 最终回答必须逐行给出完整 TARGET_MARKER、HEAD_MARKER 和 "
        "TAIL_MARKER。所有值必须来自真实 Tool Result。"
    )

    hidden_values = (target_token, head_token, middle_token, tail_token)
    for hidden_value in hidden_values:
        require(
            hidden_value not in system_prompt
            and hidden_value not in task
            and hidden_value not in python_command,
            "运行时随机 marker 不得提前泄露给真实模型。",
        )

    temporary_root_path: Path | None = None
    workspace_display = ""
    model_name = ""
    final_answer = ""
    tool_call_summary: list[str] = []
    result_forwarding_summary: list[str] = []
    read_window_summary: list[str] = []
    shell_original_chars = -1
    shell_returned_chars = -1
    llm_call_count = 0
    final_workspace_revision = -1
    target_read_round = -1

    with TemporaryDirectory(
        prefix="coding-agent-p1-2-real-"
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()

        note_lines = [
            f"NOTE_LINE_{line_number:04d}=ordinary-{line_number:04d}\n"
            for line_number in range(1, TOTAL_LINES + 1)
        ]
        note_lines[target_line_number - 1] = f"{target_marker}\n"
        large_file_path = workspace / LARGE_FILE_NAME
        large_file_path.write_text("".join(note_lines), encoding="utf-8")

        expected_stdout = (
            f"{head_marker}\n"
            + "A" * MIDDLE_FILL_CHARS
            + f"\n{middle_marker}\n"
            + "B" * MIDDLE_FILL_CHARS
            + f"\n{tail_marker}\n"
        )
        output_script = (
            "import sys\n\n"
            f"sys.stdout.write({head_marker!r} + '\\n')\n"
            f"sys.stdout.write('A' * {MIDDLE_FILL_CHARS})\n"
            f"sys.stdout.write('\\n' + {middle_marker!r} + '\\n')\n"
            f"sys.stdout.write('B' * {MIDDLE_FILL_CHARS})\n"
            f"sys.stdout.write('\\n' + {tail_marker!r} + '\\n')\n"
        )
        output_script_path = workspace / OUTPUT_SCRIPT_NAME
        output_script_path.write_text(output_script, encoding="utf-8")

        read_tool = ReadFileTool(
            workspace,
            default_max_lines=READ_MAX_LINES,
            max_output_chars=READ_MAX_OUTPUT_CHARS,
        )
        shell_tool = RunCommandTool(
            workspace,
            max_output_chars=SHELL_MAX_OUTPUT_CHARS,
        )
        registry = ToolRegistry()
        registry.register(read_tool)
        registry.register(shell_tool)

        require(
            registry.names() == ["read_file", "run_command"],
            "本专项只能向真实模型注册 read_file 和 run_command。",
        )
        read_parameters = read_tool.to_schema()["function"]["parameters"]
        shell_parameters = shell_tool.to_schema()["function"]["parameters"]
        require(
            "max_output_chars" not in read_parameters["properties"]
            and "max_output_chars" not in shell_parameters["properties"],
            "Host-controlled output budget 不得暴露为模型参数。",
        )

        client = LLMClient()
        real_llm = RecordingLLM(client)
        model_name = client.model
        agent = Agent(
            real_llm,
            registry,
            system_prompt=system_prompt,
            verbose=True,
            max_steps=14,
            max_context_chars=None,
            verification_tool_name=None,
        )
        require(
            agent.verification_tool_name is None,
            "P1-2 Real 验证不得启用 Completion Gate。",
        )

        print("P1-2 真实大模型 Tool Output Budget 验证")
        print(f"\nModel：{model_name}")
        print(f"Temporary Workspace：{workspace}")
        print(f"Read default max lines：{READ_MAX_LINES}")
        print(f"Read character budget：{READ_MAX_OUTPUT_CHARS}")
        print(f"Shell per-stream character budget：{SHELL_MAX_OUTPUT_CHARS}")
        print(f"Host-provided command：{python_command}")
        print(f"\n用户任务：\n{task}\n")

        final_answer = agent.run(task)
        history = agent.history
        events = collect_tool_events(history)

        require(
            isinstance(real_llm.client, LLMClient) and bool(real_llm.calls),
            "验证过程没有通过真实 LLMClient 发起请求。",
        )
        require(events, "真实模型没有产生任何 Tool Call。")
        require(
            len(real_llm.calls) == len(events) + 1,
            "真实 API 调用次数应等于单 Tool Call 轮数加最终回答轮。",
        )
        require(
            all(call["tools"] == registry.schemas() for call in real_llm.calls),
            "真实 LLM API 请求没有持续收到当前 Tool Schema。",
        )

        for event in events:
            next_call_index = int(event["order"]) + 1
            require(
                event["result_message"]
                in real_llm.calls[next_call_index]["messages"],
                f"第 {event['order'] + 1} 个 Tool Result "
                "没有进入紧接着的真实 LLM API messages。",
            )
            result_forwarding_summary.append(
                f"Tool {event['order'] + 1} ({event['name']}) Result "
                f"-> Real API Call {next_call_index + 1}"
            )

        require(
            events[0]["name"] == "read_file"
            and events[0]["arguments"] == {"path": LARGE_FILE_NAME},
            "真实模型第一次必须使用 Host 默认窗口读取 large_notes.txt。",
        )
        require(
            all(event["name"] in {"read_file", "run_command"} for event in events),
            "真实模型产生了本专项之外的 Tool Call。",
        )

        read_events = [event for event in events if event["name"] == "read_file"]
        require(len(read_events) >= 2, "真实模型没有主动进行分页读取。")

        expected_start_line = 1
        target_event: dict[str, Any] | None = None
        first_metadata: dict[str, str] | None = None
        for read_index, event in enumerate(read_events):
            expected_arguments: dict[str, Any] = {"path": LARGE_FILE_NAME}
            if read_index > 0:
                expected_arguments["start_line"] = expected_start_line
            require(
                event["arguments"] == expected_arguments,
                "真实模型没有严格依据上一页 next_start_line 顺序分页，"
                "或传入了不允许的 max_lines。",
            )

            metadata, payload = parse_read_result(
                event["result_message"].get("content")
            )
            if first_metadata is None:
                first_metadata = metadata
            require(
                metadata.get("path") == LARGE_FILE_NAME
                and metadata.get("total_lines") == str(TOTAL_LINES),
                "Read Tool Result 的路径或总行数元数据不正确。",
            )
            require(
                metadata.get("char_truncated") == "false",
                "行分页 fixture 不应额外触发 Read 字符截断。",
            )
            require(
                len(payload) <= READ_MAX_OUTPUT_CHARS,
                "Read Tool Result 正文突破 Host 字符预算。",
            )
            read_window_summary.append(
                f"Read {read_index + 1}: lines={metadata.get('lines')}, "
                f"next_start_line={metadata.get('next_start_line')}, "
                f"payload_chars={len(payload)}"
            )

            if target_marker in payload:
                require(
                    target_event is None,
                    "随机 TARGET 出现在多个 Read Tool Result 中。",
                )
                target_event = event
                require(
                    payload.count(target_marker) == 1,
                    "目标 Read Tool Result 没有唯一包含完整随机 TARGET。",
                )
                continue

            require(
                target_event is None,
                "真实模型在发现 TARGET 后仍继续读取 large_notes.txt。",
            )
            require(
                metadata.get("truncated_after") == "true",
                "尚未找到 TARGET 时 Read 元数据却没有后续页。",
            )
            next_start_line = metadata.get("next_start_line", "")
            require(
                next_start_line.isdigit(),
                "Read 元数据没有提供可继续分页的 next_start_line。",
            )
            expected_start_line = int(next_start_line)

        require(first_metadata is not None, "缺少第一 Read 元数据。")
        require(
            first_metadata.get("lines")
            == f"1-{READ_MAX_LINES} of {TOTAL_LINES}"
            and first_metadata.get("truncated_after") == "true",
            "第一 Read 没有按 Host 默认行窗口截断。",
        )
        require(
            target_marker
            not in str(events[0]["result_message"].get("content") or ""),
            "随机 TARGET 不应出现在第一 Read Tool Result。",
        )
        require(target_event is not None, "后续 Read Tool Result 没有找到随机 TARGET。")
        require(
            target_event is read_events[-1],
            "真实模型找到 TARGET 后仍进行了额外 read_file 调用。",
        )
        target_read_round = read_events.index(target_event) + 1

        target_event_order = int(target_event["order"])
        target_result_index = int(target_event["result_index"])
        require(
            target_event_order > 0,
            "随机 TARGET 必须来自第一 Read 之后的分页 Tool Result。",
        )
        require(
            target_token not in serialize(history[:target_result_index]),
            "随机 TARGET 在目标 Read Tool Result 之前已进入 Agent History。",
        )
        require(
            all(
                target_token not in serialize(call["messages"])
                for call in real_llm.calls[: target_event_order + 1]
            ),
            "随机 TARGET 在目标 Read Result 之前已进入真实 API messages。",
        )

        run_events = [event for event in events if event["name"] == "run_command"]
        require(
            len(run_events) == 1,
            "真实模型必须且只能调用一次 run_command。",
        )
        run_event = run_events[0]
        require(
            int(run_event["order"]) > target_event_order,
            "run_command 必须发生在真实分页找到 TARGET 之后。",
        )
        run_arguments = run_event["arguments"]
        require(
            run_arguments.get("command") == python_command
            and set(run_arguments).issubset({"command", "timeout"}),
            "真实模型执行的不是 Host 提供的精确 Python 命令。",
        )
        require(
            LARGE_FILE_NAME not in str(run_arguments.get("command")),
            "真实模型不应使用 run_command 偷读 large_notes.txt。",
        )

        shell_result = parse_shell_result(
            run_event["result_message"].get("content")
        )
        require(
            shell_result.get("exit_code") == 0
            and shell_result.get("timed_out") is False,
            "Host 提供的真实 Python 命令执行失败或超时。",
        )
        stdout = shell_result.get("stdout")
        require(isinstance(stdout, str), "Shell stdout 必须是字符串。")
        require(
            shell_result.get("stdout_truncated") is True,
            "大 stdout 没有触发 Host-controlled 截断。",
        )
        require(
            shell_result.get("stdout_original_chars") == len(expected_stdout)
            and len(expected_stdout) > SHELL_MAX_OUTPUT_CHARS,
            "Shell stdout 原始字符数元数据不正确。",
        )
        require(
            len(stdout) <= SHELL_MAX_OUTPUT_CHARS
            and len(stdout) < len(expected_stdout),
            "Shell stdout Tool Result 突破预算或没有缩小。",
        )
        require(
            head_marker in stdout and tail_marker in stdout,
            "Shell Head/Tail marker 没有同时保留。",
        )
        require(
            middle_marker not in stdout
            and "stdout truncated: original" in stdout,
            "Shell 中部 marker 未被移除或缺少明确截断标记。",
        )
        require(
            shell_result.get("stderr") == ""
            and shell_result.get("stderr_truncated") is False
            and shell_result.get("stderr_original_chars") == 0,
            "生成脚本不应产生 stderr 或 stderr 截断。",
        )

        run_event_order = int(run_event["order"])
        run_result_index = int(run_event["result_index"])
        require(
            head_token not in serialize(history[:run_result_index])
            and tail_token not in serialize(history[:run_result_index]),
            "Shell Head/Tail 在真实 run_command Tool Result 之前已泄露。",
        )
        require(
            all(
                head_token not in serialize(call["messages"])
                and tail_token not in serialize(call["messages"])
                for call in real_llm.calls[: run_event_order + 1]
            ),
            "Shell Head/Tail 在 Tool Result 之前已进入真实 API messages。",
        )
        require(
            middle_token not in serialize(history),
            "被截断的 Shell middle marker 不应进入 Agent History。",
        )

        require(bool(final_answer.strip()), "真实模型最终回答不能为空。")
        require(
            target_marker in final_answer,
            "真实模型最终回答缺少 Read Tool Result 中的完整 TARGET_MARKER。",
        )
        require(
            head_marker in final_answer and tail_marker in final_answer,
            "真实模型最终回答缺少 Shell Tool Result 中的 Head/Tail marker。",
        )
        require(
            middle_token not in final_answer,
            "真实模型不应猜测被截断的 middle marker。",
        )

        final_messages = [
            (index, message)
            for index, message in enumerate(history)
            if message.get("role") == "assistant"
            and not message.get("tool_calls")
            and message.get("content") == final_answer
        ]
        require(final_messages, "Agent History 中缺少真实模型最终回答。")
        final_message_index = final_messages[-1][0]
        require(
            int(run_event["result_index"]) < final_message_index,
            "最终回答没有发生在 Shell Tool Result 之后。",
        )
        require(
            not any(
                message.get("role") == "user"
                and "[Verification Required]" in str(message.get("content") or "")
                for message in history
            ),
            "P1-2 专项不应出现 Completion Gate Feedback。",
        )
        require(
            agent.workspace_revision == 1
            and agent.verified_revision == 0
            and agent.verification_required,
            "普通 RunCommandTool 应保持 mutation metadata，且专项未启用验证。",
        )
        require(
            large_file_path.read_text(encoding="utf-8") == "".join(note_lines)
            and output_script_path.read_text(encoding="utf-8") == output_script,
            "真实验证过程中临时 fixture 被意外修改。",
        )

        tool_call_summary = [
            f"{event['order'] + 1}. {event['name']}"
            f"({json.dumps(event['arguments'], ensure_ascii=False)})"
            for event in events
        ]
        workspace_display = str(workspace)
        shell_original_chars = int(shell_result["stdout_original_chars"])
        shell_returned_chars = len(stdout)
        llm_call_count = len(real_llm.calls)
        final_workspace_revision = agent.workspace_revision

    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "P1-2 真实模型验证的临时目录没有被清理。",
    )

    print("\n真实模型 Tool Call 顺序：")
    for summary in tool_call_summary:
        print(summary)
    print("\nRead 分页窗口：")
    for summary in read_window_summary:
        print(summary)
    print("\nTool Result 进入后续真实 API messages：")
    for summary in result_forwarding_summary:
        print(summary)
    print("\n真实模型最终回答：")
    print(final_answer)

    print("\n验证结果：")
    print(f"真实 LLMClient 调用次数：{llm_call_count}")
    print(f"模型：{model_name}")
    print(f"Temporary Workspace：{workspace_display}（已清理）")
    print(f"随机 TARGET 实际行号：{target_line_number}")
    print("第一 read_file 使用 Host 默认窗口：通过")
    print("第一 Read bounded 且不含随机 TARGET：通过")
    print("真实模型依据 next_start_line 主动连续分页：通过")
    print(f"第 {target_read_round} 次 read_file 找到 TARGET：通过")
    print(f"后续 Read 找到随机 {target_marker}：通过")
    print("TARGET 在目标 Read Tool Result 前未泄露：通过")
    print("真实模型只执行 Host 提供的精确 Python 命令：通过")
    print("run_command 未用于读取 large_notes.txt：通过")
    print(f"Shell stdout 原始字符数：{shell_original_chars}")
    print(f"Shell stdout 返回字符数：{shell_returned_chars}")
    print("Shell stdout_truncated=true：通过")
    print(f"Shell Head 保留 {head_marker}：通过")
    print(f"Shell Tail 保留 {tail_marker}：通过")
    print("Shell middle marker 已移除：通过")
    print("每个 Tool Result 均进入紧接着的真实 API messages：通过")
    print("Agent History 中 Read/Shell Tool Result 均受预算约束：通过")
    print("Final 包含 TARGET、HEAD、TAIL 且未猜测 middle：通过")
    print(f"RunCommand workspace revision：{final_workspace_revision}")
    print("Completion Gate 未启用：通过")
    print("临时 Workspace 已清理：通过")

    print("\nP1-2 真实大模型 Tool Output Budget 验证成功")


if __name__ == "__main__":
    main()
