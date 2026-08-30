import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from src.agent import Agent
from src.llm import LLMClient
from src.tools.base import BaseTool
from src.tools.files import ReadFileTool
from src.tools.registry import ToolRegistry


TARGET_FILE_NAME = "target.txt"
FAILURE_MESSAGE = "P1-4 intentional verification failure"


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


class FailDemoTool(BaseTool):
    name = "fail_demo"
    description = "Intentional verification tool. Call this tool when instructed."
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


class RecordingLLM:
    """Forward requests unchanged to a real client and record both sides."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        call: dict[str, Any] = {
            "messages": deepcopy(messages),
            "tools": deepcopy(tools),
        }
        self.calls.append(call)
        response = self.client.chat(messages, tools=tools)
        call["response"] = response
        return response


def parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call["function"]["arguments"]
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Real Tool Call arguments are not valid JSON."
            ) from error
    else:
        parsed = arguments
    require(isinstance(parsed, dict), "Real Tool Call arguments must be an object.")
    return parsed


def parse_read_result(content: Any) -> tuple[dict[str, str], str]:
    require(isinstance(content, str), "read_file Tool Result must be a string.")
    header, separator, payload = content.partition("\n\n")
    require(separator == "\n\n", "read_file Tool Result is missing its separator.")
    header_lines = header.splitlines()
    require(
        bool(header_lines) and header_lines[0] == "[read_file]",
        "read_file Tool Result is missing its metadata header.",
    )

    metadata: dict[str, str] = {}
    for line in header_lines[1:]:
        key, field_separator, value = line.partition(": ")
        require(
            bool(key) and field_separator == ": ",
            "read_file Tool Result contains invalid metadata.",
        )
        require(key not in metadata, f"read_file metadata field {key} is duplicated.")
        metadata[key] = value
    return metadata, payload


def collect_tool_events(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_messages: dict[str, tuple[int, dict[str, Any]]] = {}
    for message_index, message in enumerate(history):
        call_id = message.get("tool_call_id")
        if message.get("role") != "tool":
            continue
        require(
            isinstance(call_id, str) and bool(call_id),
            "Tool Result has no valid id.",
        )
        require(
            call_id not in result_messages,
            f"Tool Call ID {call_id} has more than one Tool Result.",
        )
        require(
            set(message) == {"role", "tool_call_id", "content"},
            "Provider-facing Tool Result contains unexpected internal fields.",
        )
        result_messages[call_id] = (message_index, message)

    events: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for message_index, message in enumerate(history):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue
        require(
            isinstance(tool_calls, list) and len(tool_calls) == 1,
            "Each real tool-calling response must contain exactly one Tool Call.",
        )

        tool_call = tool_calls[0]
        call_id = tool_call.get("id")
        require(
            isinstance(call_id, str) and bool(call_id),
            "Tool Call has no valid id.",
        )
        require(call_id not in seen_call_ids, f"Tool Call ID {call_id} was reused.")
        require(call_id in result_messages, f"Tool Call {call_id} has no Tool Result.")
        seen_call_ids.add(call_id)

        result_index, result_message = result_messages[call_id]
        require(
            result_index == message_index + 1,
            f"Tool Result for {call_id} does not immediately follow its single call.",
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

    require(
        seen_call_ids == set(result_messages),
        "Full History contains an orphan Tool Call or Tool Result.",
    )
    return events


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    marker_token = f"p1-4-target-{uuid4().hex}"
    marker_line = f"TARGET_MARKER={marker_token}"
    system_prompt = (
        "You are performing a real unified tool-execution-boundary verification. "
        "Use only the provided tools and issue exactly one tool call per tool-using "
        "response. Your first response must contain exactly one fail_demo call with "
        "an empty argument object and no read_file call. Wait for its Tool Result. "
        "The failure is expected; after receiving it, do not stop and do not call "
        "fail_demo again. Your second response must contain exactly one read_file "
        f"call with exactly {json.dumps({'path': TARGET_FILE_NAME})}. Wait for the "
        "real read_file Tool Result. Your third response must contain no Tool Call "
        "and must be exactly the complete TARGET_MARKER=... line read from that "
        "result, with no Markdown, explanation, or surrounding text. Never guess "
        "the marker. Do not request multiple tools in one response."
    )
    task = (
        "This is a tool-error-recovery verification task.\n\n"
        "Follow these steps exactly:\n"
        "1. First call fail_demo with no arguments.\n"
        "2. The tool is expected to return an execution error.\n"
        "3. After receiving that Tool Result, do not stop.\n"
        f"4. Then call read_file on {TARGET_FILE_NAME} with only the path argument.\n"
        "5. Read the exact TARGET_MARKER from the real Tool Result.\n"
        "6. Return only that complete TARGET_MARKER line.\n\n"
        "Do not guess the marker. Do not call both tools in one response. "
        "Do not call fail_demo again after receiving its result."
    )

    hidden_sources = (
        system_prompt,
        task,
        FailDemoTool.description,
        FAILURE_MESSAGE,
    )
    require(
        all(
            marker_token not in source and marker_line not in source
            for source in hidden_sources
        ),
        "The runtime TARGET_MARKER leaked into a prompt or FailDemoTool text.",
    )

    temporary_root_path: Path | None = None
    workspace_display = ""
    model_name = ""
    final_answer = ""
    llm_call_count = 0
    fail_call_id = ""
    read_call_id = ""

    with TemporaryDirectory(prefix="coding-agent-p1-4-real-") as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        workspace_display = str(workspace)
        target_path = workspace / TARGET_FILE_NAME
        target_content = marker_line + "\n"
        target_path.write_text(target_content, encoding="utf-8")

        fail_tool = FailDemoTool()
        read_tool = ReadFileTool(workspace)
        registry = ToolRegistry()
        registry.register(fail_tool)
        registry.register(read_tool)
        require(
            registry.names() == ["fail_demo", "read_file"],
            "Real verification must expose only fail_demo and read_file.",
        )
        require(
            fail_tool.mutates_workspace is False
            and read_tool.mutates_workspace is False,
            "Real verification tools must both be non-mutating.",
        )
        require(
            marker_token not in serialize(registry.schemas()),
            "The runtime TARGET_MARKER leaked into a Tool Schema.",
        )

        client = LLMClient()
        real_llm = RecordingLLM(client)
        model_name = client.model
        agent = Agent(
            real_llm,
            registry,
            system_prompt=system_prompt,
            verbose=True,
            max_steps=3,
            max_context_chars=None,
            verification_tool_name=None,
        )
        require(
            agent.verification_tool_name is None,
            "P1-4 Real verification must not enable the Completion Gate.",
        )

        print("P1-4 Real LLM Unified Tool Execution Boundary verification")
        print(f"\nModel:\n{model_name}")
        print(f"\nTemporary Workspace:\n{workspace}")
        print("\nRandom TARGET_MARKER created.")
        print(f"{marker_line}")
        print(f"\nUser task:\n{task}\n")

        final_answer = agent.run(task)
        history = agent.history
        events = collect_tool_events(history)

        require(
            isinstance(real_llm.client, LLMClient),
            "RecordingLLM is not wrapping the real LLMClient.",
        )
        require(
            len(real_llm.calls) == 3
            and all("response" in call for call in real_llm.calls),
            "The verification must produce exactly three successful real LLM calls.",
        )
        require(
            all(call["tools"] == registry.schemas() for call in real_llm.calls),
            "A real API request did not receive the exact registered Tool Schemas.",
        )
        require(len(events) == 2, "The real model must make exactly two Tool Calls.")
        require(
            [event["name"] for event in events] == ["fail_demo", "read_file"],
            "Real Tool Call order must be fail_demo followed by read_file.",
        )
        require(
            events[0]["arguments"] == {}
            and events[1]["arguments"] == {"path": TARGET_FILE_NAME},
            "The real model used unexpected Tool arguments.",
        )

        response_call_counts = [
            len(getattr(call["response"], "tool_calls", None) or [])
            for call in real_llm.calls
        ]
        require(
            response_call_counts == [1, 1, 0],
            "Real responses must be one fail call, one read call, then a final answer.",
        )
        require(
            real_llm.calls[0]["response"].tool_calls[0].function.name == "fail_demo"
            and real_llm.calls[1]["response"].tool_calls[0].function.name
            == "read_file",
            "Recorded real responses do not match the required Tool Call sequence.",
        )

        fail_event, read_event = events
        require(
            fail_tool.execute_count == 1,
            "fail_demo must execute exactly once through the production boundary.",
        )
        fail_call_id = str(fail_event["id"])
        read_call_id = str(read_event["id"])
        require(
            fail_call_id != read_call_id,
            "The real model reused one Tool Call ID for two calls.",
        )

        fail_result = fail_event["result_message"]
        fail_content = fail_result.get("content")
        require(
            isinstance(fail_content, str),
            "fail_demo Tool Result must be a string.",
        )
        try:
            fail_payload = json.loads(fail_content)
        except json.JSONDecodeError as error:
            raise RuntimeError("fail_demo Tool Result is not valid JSON.") from error
        require(
            fail_payload
            == {
                "ok": False,
                "tool": "fail_demo",
                "error_type": "RuntimeError",
                "message": FAILURE_MESSAGE,
            },
            "fail_demo RuntimeError was not normalized to the stable Tool Error.",
        )
        require(
            "traceback" not in fail_content.casefold(),
            "The model-facing fail_demo Tool Error leaked a traceback.",
        )
        require(
            marker_token not in fail_content,
            "The fail_demo Tool Error leaked the runtime TARGET_MARKER.",
        )

        read_result = read_event["result_message"]
        read_metadata, read_payload = parse_read_result(read_result.get("content"))
        require(
            read_metadata.get("path") == TARGET_FILE_NAME
            and read_metadata.get("lines") == "1-1 of 1"
            and read_metadata.get("total_lines") == "1",
            "The real read_file Result has incorrect path or line metadata.",
        )
        require(
            read_metadata.get("truncated_before") == "false"
            and read_metadata.get("truncated_after") == "false"
            and read_metadata.get("char_truncated") == "false",
            "The one-line target unexpectedly triggered Read truncation.",
        )
        require(
            read_metadata.get("original_selected_chars") == str(len(target_content))
            and read_metadata.get("next_start_line") == "none",
            "The real read_file Result has incorrect size or continuation metadata.",
        )
        require(
            read_payload == target_content and read_payload.count(marker_line) == 1,
            "The random TARGET_MARKER did not come from the exact target.txt payload.",
        )

        for event in events:
            next_api_call = int(event["order"]) + 1
            next_messages = real_llm.calls[next_api_call]["messages"]
            assistant_message = history[int(event["assistant_index"])]
            require(
                next_messages.count(assistant_message) == 1
                and next_messages.count(event["result_message"]) == 1,
                f"Tool Result for {event['id']} did not enter the next real "
                "API request with its original assistant call exactly once.",
            )
        require(
            fail_result not in real_llm.calls[0]["messages"]
            and read_result not in real_llm.calls[1]["messages"],
            "A Tool Result appeared in API messages before its Tool Call completed.",
        )

        read_result_index = int(read_event["result_index"])
        marker_history_indexes = [
            index
            for index, message in enumerate(history)
            if marker_token in serialize(message)
        ]
        require(
            marker_history_indexes
            and marker_history_indexes[0] == read_result_index
            and marker_token not in serialize(history[:read_result_index]),
            "The runtime TARGET_MARKER entered Full History before the real "
            "Read Result.",
        )
        require(
            all(
                marker_token not in serialize(call["messages"])
                for call in real_llm.calls[:2]
            ),
            "The runtime TARGET_MARKER leaked into an API request before "
            "read_file returned.",
        )
        third_request_marker_messages = [
            message
            for message in real_llm.calls[2]["messages"]
            if marker_token in serialize(message)
        ]
        require(
            third_request_marker_messages
            and third_request_marker_messages[0] == read_result,
            "The marker's first real API source was not the read_file Tool Result.",
        )

        require(
            final_answer == marker_line,
            "The real model Final must exactly equal the complete TARGET_MARKER line.",
        )
        require(
            getattr(real_llm.calls[2]["response"], "content", None) == marker_line,
            "The recorded final real response was not the exact TARGET_MARKER line.",
        )
        require(
            [message.get("role") for message in history]
            == [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
            "Full History does not contain the exact three-round message sequence.",
        )
        require(
            history[0] == {"role": "system", "content": system_prompt}
            and history[1] == {"role": "user", "content": task}
            and history[-1] == {"role": "assistant", "content": marker_line},
            "Full History did not preserve the exact prompt, task, and final answer.",
        )
        require(
            not any(
                message.get("role") == "user"
                and "[Verification Required]" in str(message.get("content") or "")
                for message in history
            ),
            "P1-4 Real verification unexpectedly triggered Completion Gate feedback.",
        )
        require(
            agent.workspace_revision == 0
            and agent.verified_revision == 0
            and agent.verification_required is False,
            "Non-mutating fail/read tools unexpectedly changed verification state.",
        )
        require(
            target_path.read_text(encoding="utf-8") == target_content,
            "The temporary target.txt fixture was modified during verification.",
        )

        llm_call_count = len(real_llm.calls)

    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "P1-4 Real verification Temporary Workspace was not cleaned up.",
    )

    print("\nReal Tool Call order:")
    print(f"1. fail_demo (call_id={fail_call_id})")
    print(f"2. read_file (call_id={read_call_id})")
    print("\nTool Result -> Real API threading:")
    print("fail_demo Tool Result -> Real API Call 2: true")
    print("read_file Tool Result -> Real API Call 3: true")
    print("\nReal model Final:")
    print(final_answer)

    print("\nVerification result:")
    print(f"Real LLMClient call count: {llm_call_count}")
    print(f"Model: {model_name}")
    print(f"Temporary Workspace: {workspace_display} (cleaned)")
    print("Real LLMClient calls: PASS")
    print("First Tool Call is fail_demo: PASS")
    print("fail_demo execution_ok=false, error_type=RuntimeError: PASS")
    print("Tool Error traceback leaked: false")
    print("fail_demo tool_call_id preserved: PASS")
    print("fail_demo Result entered Real API Call 2: PASS")
    print("Model continued after the Tool Error: PASS")
    print("Second Tool Call is read_file: PASS")
    print("read_file execution_ok=true, real target.txt Result: PASS")
    print(f"Runtime marker remained hidden until Read Result: PASS ({marker_line})")
    print("Model returned the exact random TARGET_MARKER: PASS")
    print("Full History Tool Call / Result pairing: PASS")
    print("Temporary Workspace cleanup: PASS")
    print("\nP1-4 真实大模型 Unified Tool Execution Boundary 验证成功")


if __name__ == "__main__":
    main()
