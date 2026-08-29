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
- 人工验证结果：程序成功通过 RunCommandTool 启动真实本地 Python 子进程。验证脚本进程与子进程 PID 不同，子进程使用当前项目虚拟环境中的 Python，并运行在指定项目工作目录中，同时能够正确返回标准输出、退出状态和超时状态。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：22 项测试全部通过。
- 验证截图：`stage05_shell_tool.png`、`stage05_shell_tool_manual.png`

## 第六阶段：基础智能体执行循环

- 验证目标：验证模型能够自主发起工具调用，Agent 能够通过工具注册表执行对应本地工具，将真实工具结果反馈给模型，并持续循环直到模型返回最终文本。
- 自动化验证命令：`python -m pytest tests/test_agent.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：6 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_agent_loop`
- 人工验证结果：真实模型成功读取运行时随机生成的本地文件内容，并基于工具返回结果继续推理，将未知内容写入新的本地文件后再次确认，最终回答与实际文件内容一致，证明模型调用、工具执行和结果反馈形成完整闭环。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：28 项测试全部通过。
- 验证截图：`stage06_agent_loop.png`、`stage06_agent_loop_manual.png`

## 第七阶段：智能体循环终止机制

- 验证目标：验证 Agent 能够在正常任务中于最大步数限制内完成任务，并在模型持续请求工具、无法产生最终答案时严格限制模型调用次数并可靠终止，防止 Agent Loop 无限运行。
- 自动化验证命令：`python -m pytest tests/test_agent_termination.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：5 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_agent_termination`
- 人工验证结果：正常任务能够在步数限制内返回最终答案；持续返回 Tool Call 的可控模型在达到设定的最大步数后被 Agent 强制终止，实际模型调用次数与 `max_steps` 一致，未发生额外模型调用。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：33 项测试全部通过。
- 真实模型验证：使用当前 Stage 7 Agent 重新执行真实模型 Agent Loop，模型在 `max_steps=20` 的情况下于第 4 次 LLM 调用返回最终答案，Agent 随即正常终止，未继续执行后续 Step。
- 验证截图：`stage07_agent_termination.png`、`stage07_agent_termination_manual.png`、`stage07_real_llm_early_termination.png`

## 第八阶段：错误处理与恢复机制

- 验证目标：验证 Agent 能够将工具查找失败和工具执行异常转换为结构化 Tool Error Result 返回模型，使模型能够继续决策和恢复；同时验证 LLM 调用异常和模型响应解析错误会以明确的 Agent 层异常向上传播。
- 自动化验证命令：`python -m pytest tests/test_agent_error_handling.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：5 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_agent_error_recovery`
- 人工验证结果：真实 ReadFileTool 首次读取不存在文件产生 FileNotFoundError，Agent 将错误转换为 Tool Result 返回模型；后续模型调用选择正确文件并成功读取，最终正常结束。
- 真实模型集成验证命令：`python -m scripts.verify_agent_error_recovery_real`
- 真实模型集成验证结果：真实模型首先尝试读取不存在的 `missing.txt`，Agent 将真实 `FileNotFoundError` 转换为 Tool Error Result 返回模型；模型随后调用 `list_directory` 检查工作区，发现实际存在的随机目标文件，再通过 `read_file` 读取其真实随机内容，并在最终回答中明确区分首次失败文件与恢复后成功读取的文件。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：38 项测试全部通过。
- 验证截图：`stage08_error_recovery.png`、`stage08_error_recovery_manual.png`、`stage08_error_recovery_real_llm.png`

## 第九阶段：对话历史管理

- 验证目标：验证同一个 Agent 实例能够在多次 `run()` 之间保留完整 conversation history，包括 User、Assistant Tool Call、Tool Result 与 Assistant Final Answer，并支持显式重置历史和安全读取历史快照。
- 自动化验证命令：`python -m pytest tests/test_agent_history.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：6 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_agent_history`
- 人工验证结果：第二次 `run()` 能够接收到第一轮 User 与 Assistant 历史；执行 `reset_history()` 后旧会话内容被清除，同时 system prompt 正确保留，reset 后的新一轮调用不再收到旧会话。
- 真实模型集成验证命令：`python -m scripts.verify_agent_history_real`
- 真实模型集成验证结果：使用运行时随机会话标记进行两轮真实模型调用，第二轮 Prompt 未重复提供标记；Agent 将第一轮完整对话历史发送给真实模型，模型能够从历史中正确返回随机标记。
- Stage 8 回归验证：5 项测试全部通过。
- Stage 7 回归验证：5 项测试全部通过。
- Stage 6 回归验证：6 项测试全部通过。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：44 项测试全部通过。
- 验证截图：`stage09_conversation_history.png`、`stage09_conversation_history_manual.png`、`stage09_conversation_history_real_llm.png`

## 第十阶段：上下文长度管理

- 验证目标：验证 Agent 能够在保留完整 Conversation History 的同时，根据可配置字符预算为每次 LLM 调用构造受限 Context；旧历史按照完整 User Turn 从旧到新淘汰，保留 system prompt、当前 Turn 和最近连续完整 Turns，避免拆散 Tool Call / Tool Result。
- 上下文预算：使用完整消息列表的稳定 JSON 序列化长度作为模型无关的字符规模代理，包含 Tool Call、Tool Result 和参数等消息结构；该数值不等价于精确 Token Count，也不包含 Tool Schema 或 Provider 协议开销。
- 自动化验证命令：`python -m pytest tests/test_agent_context.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：7 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_agent_context`
- 人工验证结果：Fake LLM 最后一次调用的 Context 已移除最旧 Turn，但最近 Turn、当前 Turn 与 system prompt 仍被保留；同时 Agent Full History 中旧 Turn 仍然存在，最终 Context 大小没有超过动态字符预算。
- 真实模型集成验证命令：`python -m scripts.verify_agent_context_real`
- 真实模型集成验证结果：真实 API 最后一次请求使用动态字符预算构造 Context，旧随机标记已从发送给模型的 Context 中移除，而最近随机标记、当前 Turn 与 system prompt 被保留；完整 History 仍保留旧标记，真实模型能够利用保留的最近 Context 返回最近随机标记。
- Stage 9 回归验证：6 项测试全部通过。
- Stage 8 回归验证：5 项测试全部通过。
- Stage 7 回归验证：5 项测试全部通过。
- Stage 6 回归验证：6 项测试全部通过。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：51 项测试全部通过。
- 计划由用户后续手工保存的验证截图：`stage10_context_management.png`、`stage10_context_management_manual.png`、`stage10_context_management_real_llm.png`；本次未生成或伪造截图文件。

## 第十一阶段：工作区安全边界

- 验证目标：验证文件类工具通过 canonical path resolution 和 Workspace containment check 限制文件操作范围，阻止 `../` 路径穿越以及 Workspace 外绝对路径访问，同时保证正常 Workspace 内路径仍可使用。
- 文件工具安全边界：`list_directory`、`read_file`、`write_file` 均只能访问 canonical Workspace Root 内路径；Workspace 内相对路径和绝对路径均可使用，包含 `..` 但规范化后仍位于 Workspace 内的路径也可正常访问。
- 符号链接：路径检查基于 `Path.resolve(strict=False)` 得到的 canonical path，能够从设计上阻止已有 symlink 将文件工具重定向到 Workspace 外；固定 7 项测试不依赖 Windows symlink 创建权限。
- Shell 边界：`run_command` 固定从 canonical Workspace Root 作为 `cwd` 启动，但该机制不是操作系统级 Sandbox，任意 Shell 命令理论上仍可能显式访问宿主机其他路径。
- 自动化验证命令：`python -m pytest tests/test_workspace_safety.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：7 项测试全部通过。
- 人工验证命令：`python -m scripts.verify_workspace_safety`
- 人工验证结果：Fake LLM 首先通过 `read_file` 请求真实存在的 Workspace 外随机文件，真实文件工具以 `WorkspaceBoundaryError` 主动拒绝；错误经 Agent 作为 Tool Error Result 返回后，Fake LLM 继续调用 `list_directory` 发现并读取 Workspace 内随机安全文件，外部 secret 内容未进入 Tool Result、History、LLM Context 或最终回答。
- 真实模型集成验证命令：`python -m scripts.verify_workspace_safety_real`
- 真实模型集成验证结果：真实模型首先尝试访问真实存在的 Workspace 外随机文件，收到结构化 `WorkspaceBoundaryError` 后调用 `list_directory` 重新检查 Workspace，并通过 `read_file` 成功读取实际发现的随机安全文件；外部 secret 内容未进入任何 Tool Result 或最终模型回答。
- Stage 4 文件工具回归验证：8 项测试全部通过。
- Stage 5 Shell 工具回归验证：6 项测试全部通过。
- Stage 8 Error Handling 回归验证：5 项测试全部通过。
- Stage 9 History 回归验证：6 项测试全部通过。
- Stage 10 Context 回归验证：7 项测试全部通过。
- 完整回归验证命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归验证结果：58 项测试全部通过。
- 计划由用户后续手工保存的验证截图：`stage11_workspace_safety.png`、`stage11_workspace_safety_manual.png`、`stage11_workspace_safety_real_llm.png`；本次未生成或伪造截图文件。
