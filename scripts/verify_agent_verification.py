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

from src.agent import Agent
from src.tools.files import WriteFileTool
from src.tools.registry import ToolRegistry
from src.tools.verification import VerifyWorkspaceTool


WRITE_WRONG_CALL_ID = "stage12_write_wrong"
VERIFY_FAIL_CALL_ID = "stage12_verify_fail"
WRITE_FIX_CALL_ID = "stage12_write_fix"
VERIFY_PASS_CALL_ID = "stage12_verify_pass"
PREMATURE_FINAL = "任务已经完成。"
FINAL_ANSWER = "verification-loop-complete"
WRONG_SOURCE = "def add(a, b):\n    return a - b\n"
CORRECT_SOURCE = "def add(a, b):\n    return a + b\n"


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


def parse_json_result(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    require(isinstance(content, str), "Verification Tool Result 必须是 JSON 字符串。")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Verification Tool Result 不是合法 JSON。") from error
    require(isinstance(payload, dict), "Verification Tool Result 必须是 JSON 对象。")
    return payload


class FakeLLM:
    def __init__(
        self,
        calculator_path: Path,
        expected_tools: list[dict[str, Any]],
    ) -> None:
        self.calculator_path = calculator_path
        self.expected_tools = expected_tools
        self.calls: list[list[dict[str, Any]]] = []
        self.states: list[tuple[int, int]] = []
        self.agent: Agent | None = None
        self.wrong_write_observed = False
        self.gate_feedback_observed = False
        self.failed_verification_observed = False
        self.fix_observed = False
        self.successful_verification_observed = False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        self.calls.append(deepcopy(messages))
        call_number = len(self.calls)
        require(
            tools == self.expected_tools,
            f"第 {call_number} 次 Fake LLM 调用收到的 Tool Schema 不正确。",
        )
        require(self.agent is not None, "Fake LLM 尚未绑定真实 Agent。")
        self.states.append(
            (
                self.agent.workspace_revision,
                self.agent.verified_revision,
            )
        )

        if call_number == 1:
            return tool_response(
                WRITE_WRONG_CALL_ID,
                "write_file",
                {"path": "calculator.py", "content": WRONG_SOURCE},
            )

        if call_number == 2:
            require(
                self.calculator_path.read_text(encoding="utf-8") == WRONG_SOURCE,
                "第一次真实 WriteFileTool 没有把错误实现写入 Workspace。",
            )
            self.wrong_write_observed = True
            return text_response(PREMATURE_FINAL)

        if call_number == 3:
            feedback = messages[-1]
            require(
                feedback.get("role") == "user"
                and "[Verification Required]" in str(feedback.get("content"))
                and "verify_workspace" in str(feedback.get("content")),
                "Completion Gate Feedback 没有进入下一次 Fake LLM Context。",
            )
            require(
                messages[-2]
                == {"role": "assistant", "content": PREMATURE_FINAL},
                "被阻止的提前 Final 没有作为合法 Assistant History 保存。",
            )
            self.gate_feedback_observed = True
            return tool_response(
                VERIFY_FAIL_CALL_ID,
                "verify_workspace",
                {},
            )

        if call_number == 4:
            tool_message = messages[-1]
            require(
                tool_message.get("role") == "tool"
                and tool_message.get("tool_call_id") == VERIFY_FAIL_CALL_ID,
                "第一次 Verification Result 的 Tool Call ID 不正确。",
            )
            payload = parse_json_result(tool_message)
            require(payload.get("ok") is False, "错误实现的真实 Verification 必须失败。")
            checks = payload.get("checks")
            require(
                isinstance(checks, list) and len(checks) == 1,
                "第一次 Verification 应返回一项真实检查。",
            )
            check = checks[0]
            require(check.get("exit_code") != 0, "失败 pytest 的 exit_code 不能是 0。")
            require(check.get("timed_out") is False, "失败 pytest 不应是 timeout。")
            failure_output = f"{check.get('stdout', '')}\n{check.get('stderr', '')}"
            require(
                "FAILED" in failure_output or "failed" in failure_output,
                "第一次 Verification Result 没有保留真实 pytest 失败信息。",
            )
            require(
                self.agent.verification_required,
                "Verification FAIL 后 Workspace 必须保持 DIRTY。",
            )
            self.failed_verification_observed = True
            return tool_response(
                WRITE_FIX_CALL_ID,
                "write_file",
                {"path": "calculator.py", "content": CORRECT_SOURCE},
            )

        if call_number == 5:
            require(
                self.calculator_path.read_text(encoding="utf-8") == CORRECT_SOURCE,
                "第二次真实 WriteFileTool 没有修复 calculator.py。",
            )
            self.fix_observed = True
            return tool_response(
                VERIFY_PASS_CALL_ID,
                "verify_workspace",
                {},
            )

        if call_number == 6:
            tool_message = messages[-1]
            require(
                tool_message.get("role") == "tool"
                and tool_message.get("tool_call_id") == VERIFY_PASS_CALL_ID,
                "第二次 Verification Result 的 Tool Call ID 不正确。",
            )
            payload = parse_json_result(tool_message)
            require(payload.get("ok") is True, "修复后的真实 Verification 必须成功。")
            check = payload["checks"][0]
            require(check.get("exit_code") == 0, "成功 pytest 的 exit_code 必须是 0。")
            require(check.get("timed_out") is False, "成功 pytest 不应 timeout。")
            require(
                "passed" in f"{check.get('stdout', '')} {check.get('stderr', '')}",
                "第二次 Verification Result 没有保留真实 pytest 成功信息。",
            )
            require(
                not self.agent.verification_required,
                "Verification PASS 后 Workspace 应处于 CLEAN 状态。",
            )
            self.successful_verification_observed = True
            return text_response(FINAL_ANSWER)

        raise RuntimeError("完成六步 Verification Loop 后不应继续调用 Fake LLM。")


def find_tool_result(
    history: list[dict[str, Any]],
    call_id: str,
) -> dict[str, Any]:
    message = next(
        item
        for item in history
        if item.get("role") == "tool"
        and item.get("tool_call_id") == call_id
    )
    return parse_json_result(message)


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
    first_verification: dict[str, Any] = {}
    second_verification: dict[str, Any] = {}
    host_verification: dict[str, Any] = {}
    final_answer = ""
    final_workspace_revision = -1
    final_verified_revision = -1
    llm_call_count = 0

    with TemporaryDirectory(prefix="coding-agent-stage12-") as temporary_root:
        temporary_root_path = Path(temporary_root)
        workspace = temporary_root_path / "workspace"
        workspace.mkdir()
        calculator_path = workspace / "calculator.py"
        test_path = workspace / "test_calculator.py"
        calculator_path.write_text(CORRECT_SOURCE, encoding="utf-8")
        test_path.write_text(
            "from calculator import add\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        verifier = VerifyWorkspaceTool(
            workspace,
            [verification_command],
            timeout=30,
        )
        registry = ToolRegistry()
        registry.register(WriteFileTool(workspace))
        registry.register(verifier)
        llm = FakeLLM(calculator_path, registry.schemas())
        agent = Agent(
            llm,
            registry,
            verbose=True,
            max_steps=6,
            verification_tool_name="verify_workspace",
        )
        llm.agent = agent

        print("Stage 12 Verification Loop 验证")
        print(f"\nTemporary Workspace:\n{workspace}")
        print(f"\nVerification Command:\n{verification_command}")
        print("\n确定性流程：错误写入 → 提前 Final → 验证失败 → 修复 → 验证成功 → Final\n")

        final_answer = agent.run(
            "请完成确定性的 Verification Loop 集成验证。"
        )
        history = agent.history
        first_verification = find_tool_result(history, VERIFY_FAIL_CALL_ID)
        second_verification = find_tool_result(history, VERIFY_PASS_CALL_ID)

        require(final_answer == FINAL_ANSWER, "Agent 最终回答不正确。")
        require(len(llm.calls) == 6, "完整 Fake Verification Loop 必须调用 LLM 六次。")
        require(
            llm.states
            == [(0, 0), (1, 0), (1, 0), (1, 0), (2, 0), (2, 2)],
            f"Workspace Revision 状态时序不正确：{llm.states!r}",
        )
        require(llm.wrong_write_observed, "未观察到错误代码真实写入。")
        require(llm.gate_feedback_observed, "未观察到 Completion Gate Feedback。")
        require(
            llm.failed_verification_observed,
            "未观察到第一次真实 Verification 失败。",
        )
        require(llm.fix_observed, "未观察到 Fake LLM 继续进行真实修复。")
        require(
            llm.successful_verification_observed,
            "未观察到第二次真实 Verification 成功。",
        )
        require(
            calculator_path.read_text(encoding="utf-8") == CORRECT_SOURCE,
            "最终 calculator.py 没有保留正确实现。",
        )
        require(
            agent.workspace_revision == 2
            and agent.verified_revision == 2
            and not agent.verification_required,
            "最终 Workspace Revision 没有与成功 Verification 对齐。",
        )

        assistant_indices: dict[str, int] = {}
        final_index = -1
        for index, message in enumerate(history):
            calls = message.get("tool_calls", [])
            for call in calls:
                assistant_indices[str(call.get("id"))] = index
            if message.get("role") == "assistant" and not calls:
                if message.get("content") == FINAL_ANSWER:
                    final_index = index

        last_mutation_index = assistant_indices[WRITE_FIX_CALL_ID]
        successful_verification_index = assistant_indices[VERIFY_PASS_CALL_ID]
        require(
            last_mutation_index < successful_verification_index < final_index,
            "最终顺序不是 last mutation < successful verification < final。",
        )

        host_verification = json.loads(verifier.execute())
        require(
            host_verification.get("ok") is True,
            "Agent 完成后 Host 独立复验未通过。",
        )

        workspace_display = str(workspace)
        llm_call_count = len(llm.calls)
        final_workspace_revision = agent.workspace_revision
        final_verified_revision = agent.verified_revision

    require(
        temporary_root_path is not None and not temporary_root_path.exists(),
        "Stage 12 Fake 验证的临时目录没有被清理。",
    )

    failed_check = first_verification["checks"][0]
    passed_check = second_verification["checks"][0]
    print("\nStep 1：真实 write_file 写入错误 calculator.py")
    print("Workspace State: workspace_revision=1, verified_revision=0")
    print(f"\nStep 2：Fake LLM Premature Final: {PREMATURE_FINAL}")
    print("Completion Gate: BLOCKED")
    print("Harness Feedback: [Verification Required]")
    print("\nStep 3：第一次 verify_workspace()")
    print(json.dumps(first_verification, ensure_ascii=False, indent=2))
    print("Workspace State: DIRTY")
    print("\nStep 4：真实 write_file 修复 calculator.py")
    print("Workspace State: workspace_revision=2, verified_revision=0")
    print("\nStep 5：第二次 verify_workspace()")
    print(json.dumps(second_verification, ensure_ascii=False, indent=2))
    print("Workspace State: CLEAN")
    print(f"\nStep 6：Fake LLM Final: {final_answer}")

    print("\n验证结果：")
    print(f"Temporary Workspace：{workspace_display}（已清理）")
    print(f"Fake LLM 调用次数：{llm_call_count}")
    print("错误代码真实写入：通过")
    print("未经验证的提前 Final 被 Completion Gate 阻止：通过")
    print("Verification Required Feedback 返回 Fake LLM：通过")
    print(f"第一次真实 Verification 失败，exit_code={failed_check['exit_code']}：通过")
    print("失败 pytest output 返回 Fake LLM：通过")
    print("失败后 Workspace 保持 DIRTY：通过")
    print("Fake LLM 继续修复：通过")
    print(f"第二次真实 Verification 成功，exit_code={passed_check['exit_code']}：通过")
    print("成功 Verification 对应最后一次 Mutation：通过")
    print("Verification 成功后 Final 被允许：通过")
    print(f"最终 workspace_revision：{final_workspace_revision}")
    print(f"最终 verified_revision：{final_verified_revision}")
    print(f"Host 独立复验 ok={host_verification['ok']}：通过")
    print("临时 Workspace 已清理：通过")
    print("\nStage 12 Verification Loop 验证成功")


if __name__ == "__main__":
    main()
