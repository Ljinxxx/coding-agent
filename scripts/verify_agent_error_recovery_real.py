import copy
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.llm import LLMClient
from src.tools.files import ListDirectoryTool, ReadFileTool
from src.tools.registry import ToolRegistry


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


def parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call["function"]["arguments"]
    if isinstance(arguments, str):
        return json.loads(arguments)
    return arguments


def extract_report_field(final_answer: str, label: str) -> str:
    """从模型最终回答中提取要求的结构化报告字段。"""

    prefixes = (f"{label}：", f"{label}:")

    for raw_line in final_answer.splitlines():
        line = raw_line.strip()

        if line.startswith(("- ", "* ")):
            line = line[2:].lstrip()

        line = line.replace("**", "")

        for prefix in prefixes:
            if line.startswith(prefix):
                return line[len(prefix) :].strip()

    raise RuntimeError(f"最终回答缺少字段：{label}。")


def normalize_report_value(value: str) -> str:
    return value.strip().strip("`").strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    workspace = (
        project_root
        / "docs"
        / "verification"
        / "stage08_real_demo_workspace"
    )

    workspace.mkdir(parents=True, exist_ok=True)

    token = f"stage8-real-{uuid4().hex[:8]}"
    target_name = f"recovery_target_{uuid4().hex[:6]}.txt"

    target_path = workspace / target_name
    missing_path = workspace / "missing.txt"

    missing_path.unlink(missing_ok=True)
    target_path.write_text(token, encoding="utf-8")

    require(
        not missing_path.exists(),
        "验证开始前 missing.txt 必须真实不存在。",
    )

    try:
        registry = ToolRegistry()
        registry.register(ListDirectoryTool(workspace))
        registry.register(ReadFileTool(workspace))

        real_llm = RecordingLLM(LLMClient())

        system_prompt = (
            "You are a coding agent performing an error-recovery verification. "
            "You must use the provided tools and must not guess filenames or file "
            "contents. Your FIRST response must contain exactly one tool call: "
            'read_file with path "missing.txt". '
            "Do not call any other tool in the first response. After receiving the "
            "tool error, do not stop. Use the available tools to inspect the "
            "workspace, locate a real text file, and read its actual contents. "
            "Your final answer must clearly distinguish the file that failed to be "
            "read from the file discovered and successfully read during recovery. "
            "It must explicitly state that missing.txt does not exist and that the "
            "initial read_file call failed. It must name the actual file discovered "
            "and successfully read during recovery and report that file's exact "
            "contents. Do not imply that missing.txt was eventually read "
            "successfully. In the final answer, include these five lines, with each "
            "line starting exactly with the shown Chinese label and without Markdown "
            "formatting around the label: 初始失败文件：, 失败原因：, 恢复操作：, "
            "恢复后读取文件：, 实际内容：. You may add concise explanatory text."
        )

        task = (
            "完成一次错误恢复任务：\n\n"
            "1. 第一项操作必须尝试使用 read_file 读取 missing.txt；\n"
            "2. 如果 missing.txt 不存在，不要结束任务；\n"
            "3. 必须根据工具返回的错误信息自行恢复；\n"
            "4. 使用提供的工具检查当前工作区，找到一个真实存在的文本文件；\n"
            "5. 使用 read_file 读取你找到的真实文件，禁止猜测文件内容；\n"
            "6. 最终回答必须明确说明：\n"
            "   - missing.txt 不存在，因此第一次读取失败；\n"
            "   - 第一次读取失败以后，你进行了什么恢复操作；\n"
            "   - 恢复过程中实际找到并成功读取的是哪个文件；\n"
            "   - 该文件的真实内容是什么；\n"
            "7. 最终回答不得让人误解为 missing.txt 后来被成功读取。\n\n"
            "最终回答请至少按照以下结构明确报告，每个字段单独一行，且字段名不得省略：\n"
            "初始失败文件：<文件名>\n"
            "失败原因：<原因>\n"
            "恢复操作：<简要说明>\n"
            "恢复后读取文件：<实际文件名>\n"
            "实际内容：<真实文件内容>\n\n"
            "禁止猜测文件名和文件内容，所有信息必须来自工具的真实返回结果。"
        )

        require(
            target_name not in system_prompt and token not in system_prompt,
            "动态目标文件名或随机内容不应提前出现在 system_prompt 中。",
        )

        require(
            target_name not in task and token not in task,
            "动态目标文件名或随机内容不应提前出现在用户任务中。",
        )

        agent = Agent(
            real_llm,
            registry,
            system_prompt=system_prompt,
            verbose=True,
            max_steps=8,
        )

        print("Stage 8 真实大模型错误恢复验证")
        print(f"验证工作目录：{workspace}")
        print("预期首先读取：missing.txt")
        print(f"真实目标文件：{target_name}")
        print(f"运行时随机内容：{token}")
        print(f"\n用户任务：\n{task}\n")

        final_answer = agent.run(task)

        require(
            len(real_llm.calls) >= 2,
            "真实模型没有在工具错误后进行下一轮调用。",
        )

        # ---------------------------------------------------------
        # 1. 确认第一次真实模型决策确实是 read_file("missing.txt")
        # ---------------------------------------------------------

        latest_history = real_llm.calls[-1]

        assistant_messages = [
            message
            for message in latest_history
            if message.get("role") == "assistant"
        ]

        require(
            assistant_messages,
            "没有记录到真实模型的 Tool Call。",
        )

        first_assistant = assistant_messages[0]
        first_calls = first_assistant.get("tool_calls")

        require(
            isinstance(first_calls, list) and len(first_calls) == 1,
            "第一次真实模型响应必须且只能包含一个 Tool Call。",
        )

        first_call = first_calls[0]

        require(
            first_call["function"]["name"] == "read_file",
            "真实模型第一次调用的不是 read_file。",
        )

        first_args = parse_tool_arguments(first_call)

        require(
            first_args.get("path") == "missing.txt",
            "真实模型第一次没有读取 missing.txt。",
        )

        first_call_id = first_call.get("id")

        require(
            isinstance(first_call_id, str),
            "第一次 read_file Tool Call 缺少有效 id。",
        )

        # ---------------------------------------------------------
        # 2. 确认 FileNotFoundError Tool Result 真正反馈给真实 LLM
        # ---------------------------------------------------------

        error_feedback_found = False

        for call_messages in real_llm.calls[1:]:
            for message in call_messages:
                if message.get("role") != "tool":
                    continue

                content = message.get("content")
                if not isinstance(content, str):
                    continue

                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    continue

                if (
                    message.get("tool_call_id") == first_call_id
                    and payload.get("ok") is False
                    and payload.get("tool") == "read_file"
                    and payload.get("error_type") == "FileNotFoundError"
                    and "missing.txt" in payload.get("message", "")
                ):
                    error_feedback_found = True
                    break

            if error_feedback_found:
                break

        require(
            error_feedback_found,
            "FileNotFoundError 没有进入后续真实 LLM 调用的 messages。",
        )

        # ---------------------------------------------------------
        # 3. 确认模型在错误后确实改变了决策
        # ---------------------------------------------------------

        recovery_tool_found = False
        workspace_inspection_found = False
        target_read_found = False
        target_read_call_ids: set[str] = set()

        for assistant_message in assistant_messages[1:]:
            tool_calls = assistant_message.get("tool_calls") or []

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                args = parse_tool_arguments(tool_call)

                if name != "read_file" or args.get("path") != "missing.txt":
                    recovery_tool_found = True

                if name == "list_directory" and args.get("path") == ".":
                    workspace_inspection_found = True

                if (
                    name == "read_file"
                    and args.get("path") == target_name
                ):
                    target_read_found = True
                    tool_call_id = tool_call.get("id")
                    if isinstance(tool_call_id, str):
                        target_read_call_ids.add(tool_call_id)

        require(
            recovery_tool_found,
            "真实模型收到错误后没有产生新的恢复性工具决策。",
        )

        require(
            workspace_inspection_found,
            '真实模型收到错误后没有调用 list_directory(".") 检查工作区。',
        )

        require(
            target_read_found,
            f"真实模型最终没有读取真实目标文件 {target_name}。",
        )

        target_result_found = False

        for call_messages in real_llm.calls[1:]:
            for message in call_messages:
                if (
                    message.get("role") == "tool"
                    and message.get("tool_call_id") in target_read_call_ids
                    and isinstance(message.get("content"), str)
                    and parse_read_payload(message["content"]) == token
                ):
                    target_result_found = True
                    break

            if target_result_found:
                break

        require(
            target_result_found,
            "真实目标文件的 Tool Result 没有进入后续真实 LLM 调用的 messages。",
        )

        # ---------------------------------------------------------
        # 4. 确认最终回答明确区分失败文件和恢复后读取的文件
        # ---------------------------------------------------------

        require(
            "missing.txt" in final_answer,
            "最终回答没有明确提到最初读取失败的 missing.txt。",
        )

        require(
            target_name in final_answer,
            "最终回答没有明确说明错误恢复后真正读取的目标文件。",
        )

        require(
            token in final_answer,
            "真实模型最终回答中没有包含目标文件的真实随机内容。",
        )

        initial_failed_file = normalize_report_value(
            extract_report_field(final_answer, "初始失败文件")
        )
        failure_reason = normalize_report_value(
            extract_report_field(final_answer, "失败原因")
        )
        recovery_action = normalize_report_value(
            extract_report_field(final_answer, "恢复操作")
        )
        recovered_file = normalize_report_value(
            extract_report_field(final_answer, "恢复后读取文件")
        )
        actual_content = normalize_report_value(
            extract_report_field(final_answer, "实际内容")
        )

        require(
            initial_failed_file == "missing.txt",
            "最终回答的‘初始失败文件’不是 missing.txt。",
        )

        failure_reason_folded = failure_reason.casefold()
        failure_markers = (
            "不存在",
            "未找到",
            "filenotfounderror",
            "file not found",
            "does not exist",
        )

        require(
            any(marker in failure_reason_folded for marker in failure_markers),
            "最终回答的‘失败原因’没有明确说明文件不存在。",
        )

        recovery_action_folded = recovery_action.casefold()
        recovery_markers = (
            "检查",
            "目录",
            "工作区",
            "list_directory",
            "inspect",
            "directory",
            "workspace",
        )

        require(
            any(marker in recovery_action_folded for marker in recovery_markers),
            "最终回答的‘恢复操作’没有说明对工作区或目录进行了检查。",
        )

        require(
            recovered_file == target_name,
            "最终回答的‘恢复后读取文件’不是实际读取成功的随机目标文件。",
        )

        require(
            actual_content == token,
            "最终回答的‘实际内容’不是目标文件的真实随机内容。",
        )

        require(
            "missing.txt" not in recovered_file,
            "最终回答错误地将 missing.txt 标记为恢复后读取成功的文件。",
        )

        require(
            target_path.read_text(encoding="utf-8") == token,
            "目标文件内容在验证过程中被意外修改。",
        )

        print("\n模型最终回答：")
        print(final_answer)

        print("\n验证结果：")
        print(f"真实 LLM 调用次数：{len(real_llm.calls)}")
        print("第一次 read_file(missing.txt)：通过")
        print("missing.txt 真实不存在并产生 FileNotFoundError：通过")
        print("FileNotFoundError 返回真实 LLM：通过")
        print("错误后重新进行工具决策：通过")
        print('错误后 list_directory(".") 检查工作区：通过')
        print(f"读取真实目标文件 {target_name}：通过")
        print("真实目标文件 Tool Result 返回真实 LLM：通过")
        print("最终回答明确说明 missing.txt 读取失败：通过")
        print(f"最终回答明确指出真实读取文件 {target_name}：通过")
        print("最终回答包含真实随机 token：通过")
        print("最终回答未混淆失败文件与恢复后文件：通过")

        print("\nStage 8 真实大模型错误恢复验证成功")

    finally:
        missing_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)

        if workspace.exists() and not any(workspace.iterdir()):
            workspace.rmdir()


if __name__ == "__main__":
    main()
