from pathlib import Path
from uuid import uuid4

from src.agent import Agent
from src.llm import LLMClient
from src.tools.files import ListDirectoryTool, ReadFileTool, WriteFileTool
from src.tools.registry import ToolRegistry
from src.tools.shell import RunCommandTool


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace = project_root / "docs" / "verification" / "stage06_demo_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    token = f"stage6-{uuid4().hex[:8]}"
    input_path = workspace / "stage06_input.txt"
    output_path = workspace / "stage06_output.txt"
    input_path.write_text(token, encoding="utf-8")
    output_path.unlink(missing_ok=True)

    try:
        registry = ToolRegistry()
        registry.register(ListDirectoryTool(workspace))
        registry.register(ReadFileTool(workspace))
        registry.register(WriteFileTool(workspace))
        registry.register(RunCommandTool(workspace))

        task = (
            "请使用提供的工具完成以下任务：\n"
            "1. 读取当前工作区中的 stage06_input.txt；\n"
            "2. 将读取到的内容原样写入 stage06_output.txt；\n"
            "3. 再读取 stage06_output.txt 进行确认；\n"
            "4. 最后告诉我 stage06_output.txt 中的实际内容。\n"
            "不要猜测文件内容，必须使用提供的工具获取真实结果。"
        )
        agent = Agent(
            LLMClient(),
            registry,
            system_prompt=(
                "You are a coding agent. Use the provided tools to complete the task. "
                "Do not guess file contents. Continue until the requested work is "
                "complete, then return a concise final answer with the confirmed "
                "content."
            ),
            verbose=True,
        )

        print(f"Stage 6 验证工作目录：{workspace}")
        print(f"运行时随机内容：{token}")
        print(f"\n用户任务：\n{task}\n")

        final_answer = agent.run(task)
        output_exists = output_path.exists()
        output_content = (
            output_path.read_text(encoding="utf-8") if output_exists else ""
        )

        print(f"\n模型最终回答：\n{final_answer}")
        print(f"\n实际输入文件内容：\n{token}")
        print(f"\n实际输出文件内容：\n{output_content}")

        checks = {
            "output file exists": output_exists,
            "output content matches input": output_content == token,
            "final answer is not empty": bool(final_answer.strip()),
            "final answer contains runtime token": token in final_answer,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]

        if failed_checks:
            failed = ", ".join(failed_checks)
            raise RuntimeError(f"Stage 6 Agent Loop verification failed: {failed}.")

        print("\nStage 6 Agent Loop 验证成功")
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        if workspace.exists() and not any(workspace.iterdir()):
            workspace.rmdir()


if __name__ == "__main__":
    main()
