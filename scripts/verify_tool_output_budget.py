import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.tools.files import ReadFileTool
from src.tools.registry import ToolRegistry
from src.tools.shell import RunCommandTool


READ_DEFAULT_CALL_ID = "p1_2_read_default"
READ_PAGED_CALL_ID = "p1_2_read_paged"
RUN_SHELL_CALL_ID = "p1_2_run_shell"
FINAL_ANSWER = "tool-output-budget-complete"

TOTAL_LINES = 1_000
READ_DEFAULT_MAX_LINES = 120
READ_MAX_OUTPUT_CHARS = 8_000
READ_TAIL_PAGE_LINES = 25
SHELL_MAX_OUTPUT_CHARS = 800


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_command(*parts: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


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


def find_tool_message(
    messages: list[dict[str, Any]],
    call_id: str,
) -> dict[str, Any]:
    matches = [
        message
        for message in messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") == call_id
    ]
    require(
        len(matches) == 1,
        f"Expected exactly one Tool Result for {call_id}, got {len(matches)}.",
    )
    return matches[0]


def parse_read_result(content: Any) -> tuple[dict[str, str], str]:
    require(isinstance(content, str), "Read Tool Result must be text.")
    header, separator, payload = content.partition("\n\n")
    require(separator == "\n\n", "Read Tool Result is missing its payload separator.")
    header_lines = header.splitlines()
    require(
        bool(header_lines) and header_lines[0] == "[read_file]",
        "Read Tool Result is missing the [read_file] metadata header.",
    )

    metadata: dict[str, str] = {}
    for line in header_lines[1:]:
        key, delimiter, value = line.partition(": ")
        require(bool(delimiter), f"Malformed Read metadata line: {line!r}")
        metadata[key] = value
    return metadata, payload


def parse_shell_result(content: Any) -> dict[str, Any]:
    require(isinstance(content, str), "Shell Tool Result must be JSON text.")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Shell Tool Result is not valid JSON.") from error
    require(isinstance(payload, dict), "Shell Tool Result must be a JSON object.")
    return payload


def parse_line_range(value: str) -> tuple[int, int, int]:
    selected, delimiter, total = value.partition(" of ")
    require(bool(delimiter), f"Malformed Read line range: {value!r}")
    start, range_delimiter, end = selected.partition("-")
    require(bool(range_delimiter), f"Malformed Read selected range: {value!r}")
    try:
        return int(start), int(end), int(total)
    except ValueError as error:
        raise RuntimeError(f"Read line range is not numeric: {value!r}") from error


class FakeLLM:
    def __init__(
        self,
        *,
        expected_tools: list[dict[str, Any]],
        reader: ReadFileTool,
        runner: RunCommandTool,
        huge_content: str,
        target_token: str,
        shell_command: str,
        shell_output: str,
        shell_head_token: str,
        shell_middle_token: str,
        shell_tail_token: str,
    ) -> None:
        self.expected_tools = deepcopy(expected_tools)
        self.reader = reader
        self.runner = runner
        self.huge_content = huge_content
        self.target_token = target_token
        self.shell_command = shell_command
        self.shell_output = shell_output
        self.shell_head_token = shell_head_token
        self.shell_middle_token = shell_middle_token
        self.shell_tail_token = shell_tail_token

        self.agent: Agent | None = None
        self.calls: list[list[dict[str, Any]]] = []
        self.states: list[tuple[int, int, bool]] = []
        self.first_read_result = ""
        self.paged_read_result = ""
        self.shell_result_text = ""
        self.first_read_metadata: dict[str, str] = {}
        self.paged_read_metadata: dict[str, str] = {}
        self.shell_result: dict[str, Any] = {}
        self.paged_start_line = -1

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        require(self.agent is not None, "Fake LLM is not bound to the real Agent.")
        call_number = len(self.calls) + 1
        require(
            tools == self.expected_tools,
            f"Tool schemas changed on Fake LLM call {call_number}.",
        )
        require(
            messages == self.agent.history,
            f"Fake LLM call {call_number} did not receive the Agent Full History.",
        )
        self.calls.append(deepcopy(messages))
        self.states.append(
            (
                self.agent.workspace_revision,
                self.agent.verified_revision,
                self.agent.verification_required,
            )
        )

        if call_number == 1:
            require(
                not any(message.get("role") == "tool" for message in messages),
                "The first Fake LLM call unexpectedly received a Tool Result.",
            )
            return tool_response(
                READ_DEFAULT_CALL_ID,
                "read_file",
                {"path": "huge_file.txt"},
            )

        if call_number == 2:
            first_message = find_tool_message(messages, READ_DEFAULT_CALL_ID)
            first_content = first_message.get("content")
            metadata, payload = parse_read_result(first_content)
            start, end, total = parse_line_range(metadata.get("lines", ""))

            require(start == 1, "The default Read window did not start at line 1.")
            require(
                end == self.reader.default_max_lines,
                "The default Read window did not use the Host line budget.",
            )
            require(total == TOTAL_LINES, "The first Read reported the wrong total lines.")
            require(
                metadata.get("truncated_after") == "true",
                "The first Read did not report that later content exists.",
            )
            require(
                metadata.get("next_start_line") == str(end + 1),
                "The first Read reported an invalid next_start_line.",
            )
            require(
                len(payload) <= self.reader.max_output_chars,
                "The first Read payload exceeded the Host character budget.",
            )
            require(
                len(str(first_content)) < len(self.huge_content),
                "The first Read returned the entire large file.",
            )
            require(
                self.target_token not in str(first_content),
                "The late target token leaked into the first Read window.",
            )

            self.first_read_result = str(first_content)
            self.first_read_metadata = metadata
            self.paged_start_line = total - READ_TAIL_PAGE_LINES + 1
            return tool_response(
                READ_PAGED_CALL_ID,
                "read_file",
                {
                    "path": "huge_file.txt",
                    "start_line": self.paged_start_line,
                    "max_lines": READ_TAIL_PAGE_LINES,
                },
            )

        if call_number == 3:
            require(
                find_tool_message(messages, READ_DEFAULT_CALL_ID).get("content")
                == self.first_read_result,
                "The bounded first Read Result disappeared from Full History.",
            )
            paged_message = find_tool_message(messages, READ_PAGED_CALL_ID)
            paged_content = paged_message.get("content")
            metadata, payload = parse_read_result(paged_content)
            start, end, total = parse_line_range(metadata.get("lines", ""))

            require(
                start == self.paged_start_line,
                "The second Read did not use the metadata-derived tail page.",
            )
            require(end == total == TOTAL_LINES, "The tail Read range is incorrect.")
            require(
                metadata.get("truncated_before") == "true",
                "The tail Read did not report earlier content.",
            )
            require(
                metadata.get("truncated_after") == "false",
                "The tail Read incorrectly reported later content.",
            )
            require(
                self.target_token in payload,
                "The paged Read did not recover the late target token.",
            )
            require(
                payload.count(self.target_token) == 1,
                "The paged Read did not contain exactly one target token.",
            )
            require(
                len(payload) <= self.reader.max_output_chars,
                "The paged Read payload exceeded the Host character budget.",
            )

            self.paged_read_result = str(paged_content)
            self.paged_read_metadata = metadata
            return tool_response(
                RUN_SHELL_CALL_ID,
                "run_command",
                {"command": self.shell_command},
            )

        if call_number == 4:
            require(
                find_tool_message(messages, READ_DEFAULT_CALL_ID).get("content")
                == self.first_read_result,
                "The first Read Result was not retained through the shell round.",
            )
            require(
                find_tool_message(messages, READ_PAGED_CALL_ID).get("content")
                == self.paged_read_result,
                "The paged Read Result was not retained through the shell round.",
            )
            shell_message = find_tool_message(messages, RUN_SHELL_CALL_ID)
            shell_content = shell_message.get("content")
            payload = parse_shell_result(shell_content)
            stdout = payload.get("stdout")

            require(payload.get("exit_code") == 0, "The real shell command failed.")
            require(payload.get("timed_out") is False, "The shell command timed out.")
            require(isinstance(stdout, str), "Shell stdout is not text.")
            require(
                payload.get("stdout_truncated") is True,
                "Large shell stdout was not truncated.",
            )
            require(
                payload.get("stdout_original_chars") == len(self.shell_output),
                "Shell stdout_original_chars is incorrect.",
            )
            require(
                len(stdout) <= self.runner.max_output_chars,
                "Bounded shell stdout exceeded the Host budget.",
            )
            require(
                self.shell_head_token in stdout,
                "The shell Head token was not preserved.",
            )
            require(
                self.shell_tail_token in stdout,
                "The shell Tail token was not preserved.",
            )
            require(
                self.shell_middle_token not in stdout,
                "The shell middle token was not removed by truncation.",
            )
            require(
                payload.get("stderr") == ""
                and payload.get("stderr_truncated") is False
                and payload.get("stderr_original_chars") == 0,
                "Empty shell stderr metadata is incorrect.",
            )

            self.shell_result_text = str(shell_content)
            self.shell_result = payload
            return text_response(FINAL_ANSWER)

        raise RuntimeError("The Fake LLM was called after the four-step flow.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    target_token = f"TARGET_TOKEN={uuid4().hex}"
    shell_head_token = f"SHELL_HEAD_TOKEN={uuid4().hex}"
    shell_middle_token = f"SHELL_MIDDLE_TOKEN={uuid4().hex}"
    shell_tail_token = f"SHELL_TAIL_TOKEN={uuid4().hex}"
    target_line = TOTAL_LINES - 7

    huge_lines = []
    for line_number in range(1, TOTAL_LINES + 1):
        line = f"Line {line_number:04d}: payload-{line_number:04d}-" + ("x" * 24)
        if line_number == target_line:
            line += f" {target_token}"
        huge_lines.append(line + "\n")
    huge_content = "".join(huge_lines)

    shell_output = (
        f"{shell_head_token}\n"
        + ("A" * 6_000)
        + f"\n{shell_middle_token}\n"
        + ("B" * 6_000)
        + f"\n{shell_tail_token}\n"
    )
    emit_script = "import sys\nsys.stdout.write(" + repr(shell_output) + ")\n"

    temporary_root_path: Path | None = None
    workspace_display = ""
    final_answer = ""
    final_history: list[dict[str, Any]] = []
    final_revision_state = (-1, -1, False)
    fake: FakeLLM | None = None

    with TemporaryDirectory(prefix="coding-agent-p1-2-") as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        huge_path = workspace / "huge_file.txt"
        emit_path = workspace / "emit_large_output.py"
        huge_path.write_text(huge_content, encoding="utf-8")
        emit_path.write_text(emit_script, encoding="utf-8")

        require(huge_path.is_file(), "The temporary large file was not created.")
        require(emit_path.is_file(), "The temporary shell script was not created.")
        require(
            huge_path.read_text(encoding="utf-8").count(target_token) == 1,
            "The target token is not unique in the temporary large file.",
        )

        reader = ReadFileTool(
            workspace,
            default_max_lines=READ_DEFAULT_MAX_LINES,
            max_output_chars=READ_MAX_OUTPUT_CHARS,
        )
        runner = RunCommandTool(
            workspace,
            max_output_chars=SHELL_MAX_OUTPUT_CHARS,
        )
        shell_command = make_command(
            sys.executable,
            "-B",
            emit_path.name,
        )

        registry = ToolRegistry()
        registry.register(reader)
        registry.register(runner)
        require(
            registry.names() == ["read_file", "run_command"],
            "The Fake verification registry contains unexpected tools.",
        )
        require(
            "max_output_chars"
            not in runner.to_schema()["function"]["parameters"]["properties"],
            "The model can override the Host shell output budget.",
        )

        fake = FakeLLM(
            expected_tools=registry.schemas(),
            reader=reader,
            runner=runner,
            huge_content=huge_content,
            target_token=target_token,
            shell_command=shell_command,
            shell_output=shell_output,
            shell_head_token=shell_head_token,
            shell_middle_token=shell_middle_token,
            shell_tail_token=shell_tail_token,
        )
        agent = Agent(
            fake,
            registry,
            max_steps=4,
            max_context_chars=None,
            verification_tool_name=None,
        )
        fake.agent = agent

        print("P1-2 Tool Output Budget 验证")
        print(f"\nTemporary Workspace:\n{workspace}")
        print("\nRead Configuration:")
        print(f"default_max_lines: {reader.default_max_lines}")
        print(f"max_output_chars: {reader.max_output_chars}")
        print("\nShell Configuration:")
        print(f"max_output_chars per stream: {runner.max_output_chars}")
        print(f"command: {shell_command}")

        final_answer = agent.run(
            "Run the deterministic bounded read, paged read, and shell output flow."
        )
        final_history = agent.history

        require(final_answer == FINAL_ANSWER, "The Fake final answer is incorrect.")
        require(len(fake.calls) == 4, "The Fake LLM call count is not four.")
        require(
            fake.states
            == [
                (0, 0, False),
                (0, 0, False),
                (0, 0, False),
                (1, 0, True),
            ],
            f"Unexpected revision state sequence: {fake.states!r}",
        )
        require(
            agent.verification_tool_name is None
            and "verify_workspace" not in registry.names(),
            "Completion Gate was unexpectedly enabled.",
        )
        require(
            not any(
                "[Verification Required]" in str(message.get("content") or "")
                for message in final_history
            ),
            "Completion Gate feedback appeared in the P1-2-only flow.",
        )
        require(
            agent.workspace_revision == 1
            and agent.verified_revision == 0
            and agent.verification_required,
            "RunCommandTool mutation/revision semantics changed.",
        )
        require(
            reader.mutates_workspace is False and runner.mutates_workspace is True,
            "Read/Shell mutation metadata is incorrect.",
        )

        tool_messages = [
            message for message in final_history if message.get("role") == "tool"
        ]
        require(
            [message.get("tool_call_id") for message in tool_messages]
            == [READ_DEFAULT_CALL_ID, READ_PAGED_CALL_ID, RUN_SHELL_CALL_ID],
            "Tool Results were not saved to Full History in execution order.",
        )
        require(
            tool_messages[0].get("content") == fake.first_read_result
            and tool_messages[1].get("content") == fake.paged_read_result
            and tool_messages[2].get("content") == fake.shell_result_text,
            "Full History does not contain the exact bounded Tool Results.",
        )
        require(
            find_tool_message(fake.calls[1], READ_DEFAULT_CALL_ID).get("content")
            == fake.first_read_result,
            "The first Read Result did not enter the immediate next LLM messages.",
        )
        require(
            find_tool_message(fake.calls[2], READ_PAGED_CALL_ID).get("content")
            == fake.paged_read_result,
            "The paged Read Result did not enter the immediate next LLM messages.",
        )
        require(
            find_tool_message(fake.calls[3], RUN_SHELL_CALL_ID).get("content")
            == fake.shell_result_text,
            "The Shell Result did not enter the immediate next LLM messages.",
        )

        workspace_display = str(workspace)
        final_revision_state = (
            agent.workspace_revision,
            agent.verified_revision,
            agent.verification_required,
        )

    require(fake is not None, "The Fake LLM verification was not initialized.")
    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "The P1-2 temporary workspace was not cleaned up.",
    )

    first_start, first_end, first_total = parse_line_range(
        fake.first_read_metadata["lines"]
    )
    page_start, page_end, page_total = parse_line_range(
        fake.paged_read_metadata["lines"]
    )
    shell_stdout = str(fake.shell_result["stdout"])

    print("\nStep 1 - default bounded read")
    print(f"Tool Call: read_file(path='huge_file.txt')")
    print(f"Returned lines: {first_start}-{first_end} of {first_total}")
    print(f"truncated_after: {fake.first_read_metadata['truncated_after']}")
    print(f"next_start_line: {fake.first_read_metadata['next_start_line']}")
    print("First window bounded: 通过")
    print("Late target absent: 通过")

    print("\nStep 2 - metadata-driven paged read")
    print(
        "Tool Call: read_file(path='huge_file.txt', "
        f"start_line={fake.paged_start_line}, max_lines={READ_TAIL_PAGE_LINES})"
    )
    print(f"Returned lines: {page_start}-{page_end} of {page_total}")
    print(f"Recovered token: {target_token}")
    print("Late target recovered exactly once: 通过")

    print("\nStep 3 - bounded large shell stdout")
    print(f"stdout_original_chars: {fake.shell_result['stdout_original_chars']}")
    print(f"stdout_returned_chars: {len(shell_stdout)}")
    print(f"stdout_truncated: {fake.shell_result['stdout_truncated']}")
    print(f"Head token retained: {shell_head_token in shell_stdout}")
    print(f"Tail token retained: {shell_tail_token in shell_stdout}")
    print(f"Middle token removed: {shell_middle_token not in shell_stdout}")

    print("\nStep 4 - Fake Final")
    print(final_answer)

    print("\n验证结果：")
    print("Large Read bounded：通过")
    print("Read pagination metadata：通过")
    print("Late target recovered：通过")
    print("Large Shell stdout bounded：通过")
    print("Shell Head/Tail retained and middle removed：通过")
    print("Original size metadata：通过")
    print("Every Tool Result entered next-round messages：通过")
    print("Bounded Tool Results retained in Full History：通过")
    print("Completion Gate disabled：通过")
    print(
        "Revision state preserved："
        f"workspace_revision={final_revision_state[0]}, "
        f"verified_revision={final_revision_state[1]}, "
        f"verification_required={final_revision_state[2]}"
    )
    print(f"Temporary Workspace：{workspace_display}（已清理）")
    print("临时 Workspace 已清理：通过")
    print("\nP1-2 Tool Output Budget 验证成功")


if __name__ == "__main__":
    main()
