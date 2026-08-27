from pathlib import Path

from src.tools.files import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    workspace = (
        project_root
        / "docs"
        / "verification"
        / "stage04_demo_workspace"
    )

    workspace.mkdir(parents=True, exist_ok=True)

    write_tool = WriteFileTool(workspace)
    list_tool = ListDirectoryTool(workspace)
    read_tool = ReadFileTool(workspace)

    print(f"验证工作目录：{workspace}")

    print("\n1. 写入文件")
    result = write_tool.execute(
        path="stage4_demo.txt",
        content="hello stage4",
    )
    print(result)

    print("\n2. 列出目录")
    print(list_tool.execute(path="."))

    print("\n3. 读取文件")
    content = read_tool.execute(path="stage4_demo.txt")
    print(content)

    print("\n4. 验证结果")
    if content == "hello stage4":
        print("Stage 4 文件写入与读取验证成功")
    else:
        raise RuntimeError("Stage 4 文件验证失败")


if __name__ == "__main__":
    main()