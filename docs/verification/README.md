# 阶段验证记录

## 第一阶段：大模型接入

- 验证目标：确认程序能够通过 AutoDL API 成功调用 DeepSeek-V4-Flash。
- 运行命令：`python -m src.main`
- 测试结果：程序成功接收用户输入，并返回模型生成的 Python 代码。
- 验证截图：`stage01_llm_connection.png`

## 第二阶段：工具定义与注册

- 验证目标：确认工具能够完成注册、查询和 Schema 导出，并能正确识别重复注册及不存在的工具。
- 验证命令：`python -m pytest tests/test_tool_registry.py -v`
- 测试结果：4 项测试全部通过。
- 验证截图：`stage02_tool_registry.png`

## 第三阶段：工具调用与模型输出解析

- 验证目标：验证程序能够正确解析模型普通文本、单个及多个工具调用，并能够识别非法工具参数；同时验证大模型能够实际产生工具调用。
- 模型输出解析验证命令：`python -m pytest tests/test_response_parser.py -v`
- 模型输出解析验证结果：4 项测试全部通过。
- 工具调用验证命令：`python scripts/verify_tool_calling.py`
- 工具调用验证结果：大模型成功调用测试工具，程序正确解析出工具名称 `echo` 和参数 `{'text': 'stage3-ok'}`。
- 验证截图：`stage03_response_parser.png`、`stage03_tool_calling.png`

## 第四阶段：本地文件操作工具

- 验证目标：验证程序能够在本地工作目录中正常列出目录、读取文件、创建和覆盖写入文件，并能够正确处理不存在的文件和目录。
- 自动化验证命令：`python -m pytest tests/test_file_tools.py -v`
- 自动化验证结果：8 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_file_tools`
- 人工验证结果：程序成功创建 `stage4_demo.txt`，能够列出该文件并正确读取其中的 `hello stage4` 内容。
- 验证截图：`stage04_file_tools.png`、`stage04_file_tools_manual.png`

## 第五阶段：本地命令执行工具

- 验证目标：验证程序能够在指定本地工作目录中执行真实命令，并正确获取命令的退出状态、正常输出、错误输出和超时状态。
- 自动化验证命令：`python -m pytest tests/test_shell_tool.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：6 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_shell_tool`
- 人工验证结果：程序成功在项目工作目录中执行真实本地命令，并正确获取文本输出、实际工作目录和 Python 计算结果。
- 验证截图：`stage05_shell_tool.png`、`stage05_shell_tool_manual.png`

## 第五阶段：本地命令执行工具

- 验证目标：验证程序能够在指定本地工作目录中执行真实命令，并正确获取命令的退出状态、正常输出、错误输出和超时状态。
- 自动化验证命令：`python -m pytest tests/test_shell_tool.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：6 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_shell_tool`
- 人工验证结果：程序成功通过 RunCommandTool 启动真实本地 Python 子进程。验证脚本进程与子进程 PID 不同，子进程使用当前项目虚拟环境中的 Python，并运行在指定项目工作目录中，同时能够正确返回标准输出、退出状态和超时状态。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：22 项测试全部通过。
- 验证截图：`stage05_shell_tool.png`、`stage05_shell_tool_manual.png`