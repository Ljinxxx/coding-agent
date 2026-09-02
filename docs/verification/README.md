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
手工保存的验证截图：`stage10_context_management.png`、`stage10_context_management_manual.png`、`stage10_context_management_real_llm.png`

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
手工保存的验证截图：`stage11_workspace_safety.png`、`stage11_workspace_safety_manual.png`、`stage11_workspace_safety_real_llm.png`

## 第十二阶段：验证闭环机制

- 验证目标：在 Agent 观察到 Workspace 发生可能改变状态的操作后建立 Completion Gate，禁止模型在最后一次修改尚未通过宿主程序预设验证标准时直接结束任务。
- Host-controlled Verification：新增 `verify_workspace`；验证命令在构造工具时由宿主程序预先配置，Tool Schema 不接受 `command` 参数，模型只能决定何时调用验证，不能改变完成标准。
- Verification Result：工具从 canonical Workspace Root 依次执行真实命令并返回结构化 JSON，保留每项检查的 `command`、`exit_code`、`stdout`、`stderr` 和 `timed_out`；采用 fail-fast，只有全部配置命令退出码为 0 且均未超时时 `ok=true`。
- Workspace Revision：Agent 实例跟踪 `workspace_revision` 与 `verified_revision`。初始状态二者均为 0；mutation-capable Tool 一旦准备执行就保守增加 Workspace Revision，即使工具随后失败或超时也不回滚。
- Mutation Metadata：`write_file` 和普通 `run_command` 标记为可能修改 Workspace；`read_file`、`list_directory` 和 `verify_workspace` 保持只读元数据。成功 Verification 后再次执行 mutation-capable Tool 会立即使旧验证失效。
- Completion Gate：该能力通过 `verification_tool_name` 显式启用，默认 `None`，因此 Stage 1～11 的旧 Agent 行为保持兼容。启用后如果两个 Revision 不相等，无 Tool Call 的提前 Final 会被保存到完整 History，随后 Agent 追加合法 `user` 角色的 `[Verification Required]` Harness 控制反馈，并继续现有 Agent Loop。
- 状态与终止：Gate 后的下一次模型调用仍计入 Stage 7 `max_steps`，每次调用前仍重新经过 Stage 10 Context Builder；Verification State 跨同一 Agent 的多次 `run()` 保留，`reset_history()` 和 Agent Error 不会回滚 Workspace Revision。
- 失败恢复：Verification nonzero 或 timeout 是正常的 `ok=false` Tool Result，会返回模型继续修复；工具自身异常仍复用 Stage 8 Tool Error Result，不自动 rollback Workspace。
- Shell 说明：普通 `run_command` 即使退出码为 0 也不具有 Completion Authority，反而会使旧验证失效；只有 Host-controlled `verification_tool_name` 对应工具返回 JSON object 且 `ok` 严格为 `true` 时才更新 `verified_revision`。Verification 继承 Stage 11 的 canonical cwd，但仍不等同于 OS-level Sandbox。
- 自动化验证命令：`python -m pytest tests/test_agent_verification.py -v --basetemp=.pytest_tmp`
- 自动化验证结果：严格 7 项 Stage 12 测试全部通过。
- Fake LLM 验证命令：`python -m scripts.verify_agent_verification`
- Fake 验证结果：Fake LLM 通过真实 `WriteFileTool` 写入错误实现后故意提前 Final，Completion Gate 主动阻止；第一次 Host-controlled 真实 pytest 验证以退出码 1 失败，失败结果返回 Fake LLM 后继续真实修复，第二次 pytest 以退出码 0 成功，最终 `workspace_revision=2`、`verified_revision=2` 后才允许 Final；Agent 完成后的 Host 独立复验同样通过，临时 Workspace 已清理。
- 真实模型验证命令：`python -m scripts.verify_agent_verification_real`
- 真实模型验证状态：脚本已经实现真实 `LLMClient + RecordingLLM + Agent + File Tools + VerifyWorkspaceTool` 流程以及 Tool Call 顺序、Mutation/Verification/Final 顺序、成功结果进入真实模型 Context、最终 CLEAN 状态和 Host 独立复验等程序化检查；本次命令已尝试运行，但在首个真实 API 请求建立连接时受到当前 Codex 沙箱与执行策略限制，未获得真实模型响应或 Real LLM 成功结果，因此没有记录或声称 Real LLM 验证成功，需在用户本地 Windows 环境运行上述命令。
- 完整回归命令：`python -m pytest -v --basetemp=.pytest_tmp`
- 完整回归结果：65 项测试全部通过。
- 保存的验证截图：`stage12_verification_loop.png`、`stage12_verification_loop_manual_1.png`、`stage12_verification_loop_manual_2.png``stage12_verification_loop_real_llm_1.png`
`stage12_verification_loop_real_llm_2.png`、`stage12_full_regression.png`

## P0.5：正式入口配置收尾

- 默认运行：`python -m src.main`
- 自定义 Workspace：`python -m src.main --workspace ./demo`
- Context Budget：`--max-context-chars 60000` 控制发送给模型的字符数量代理预算，必须是正整数；它不是精确 Token Budget。
- Host-controlled Verification：使用可重复的 `--verify "COMMAND"` 配置一项或多项验证命令，例如 `python -m src.main --workspace ./demo --verify "python -m pytest -q"`。命令由用户或 Host 提供并按顺序执行；`verify_workspace` 的 Tool Arguments 不暴露命令参数，因此模型不能传入或替换验证命令，实际执行命令只会作为 Verification Result 的审计字段返回。
- 默认 Verification 行为：未提供 `--verify` 时，正式入口不注册 `verify_workspace` 且不启用 Completion Gate，也不会自动运行 pytest 或猜测项目类型。
- Shell 边界：Host 提供的验证字符串由当前操作系统 Shell 执行，并以 canonical Workspace 为 `cwd`；该机制不是 OS-level Sandbox。
- 正式入口自动化测试：`python -m pytest tests/test_main.py -v`，结果为 7 项测试全部通过。
- P0.5 完整回归：72 项测试全部通过。

## P1：EditFileTool 精确局部修改

- `edit_file` 的 Schema 仅包含必填字符串参数 `path`、`old_text`、`new_text`，并设置 `additionalProperties: false`；它只编辑 Workspace 内已存在的 UTF-8 普通文件。
- `old_text` 必须非空并且只有一个匹配位置：零次或多次匹配均报错且不写盘；`new_text` 可以为空以删除唯一片段；相同的新旧文本会返回 no-op 且不重写文件。
- 路径继续复用现有 `resolve_workspace_path()`，测试覆盖父路径、Workspace 外绝对路径以及 symlink/Windows junction 逃逸，均不能修改外部文件。
- 正式 `build_agent()` 按 `list_directory`、`read_file`、`edit_file`、`write_file`、`run_command` 的顺序注册基础工具。`edit_file.mutates_workspace = True`，因此无需修改 Agent 或 Verification 架构即可进入现有 revision 与 Completion Gate 流程。
- Fake LLM 集成验证覆盖 `read_file → edit_file → premature Final → Completion Gate → verify_workspace → Final`，同时验证未配置 Verification 时仍可正常局部编辑并结束。
- 完整回归命令：`python -m pytest -q`；结果：`86 passed in 7.43s`。
- 编译检查命令：`python -m compileall src`；结果：通过。
- 真实模型 smoke 使用仅含合成 `calculator.py` 和单项测试的隔离 Workspace，实际流程为 `read_file → edit_file → verify_workspace → Final`；模型把唯一的 `return a - b` 修改为 `return a + b`，Host 验证与独立复验均为 `1 passed`。

## P1-2：Read / Shell 输出预算与截断

- 验证目标：在 Tool Result 进入 Full History 前控制单次文件读取和 Shell 输出规模，避免超大文件或命令输出直接污染上下文。
- Read 分页：`read_file` 支持 1-based `start_line` 与 `max_lines`；默认 `start_line=1`，正式 Agent 的 Host 默认行窗口为 200 行。超出 EOF 时返回带总行数的空窗口，而不是抛出分页异常。
- Read Hard Budget：正式 Agent 的 Host 字符预算为 20,000 字符，约束文件内容 payload；固定的小型 metadata header 不计入 payload 预算。模型 Schema 不暴露 `max_output_chars`，即使单行极长也不能绕过预算。
- Read Metadata：返回路径、实际行范围、总行数、前后截断状态、字符截断状态、`partial_line`、原始选中字符数和 `next_start_line`。实现保持流式扫描，不把完整大文件保存在内存列表中；`original_selected_chars` 始终统计整个实际请求行窗口。普通多行读取只返回完整行：下一完整行放不下时不返回该行的局部内容，`lines` 停在最后一条完整行，`next_start_line` 指向首条未返回行。若所选第一行本身超过字符预算，则允许返回有界前缀，并明确报告 `partial_line=true`、`lines=none`、`next_start_line=none`，表示当前接口不提供字符级续读。
- Shell Budget：`run_command` 对 stdout 和 stderr 独立应用同一个 Host-controlled 字符预算；正式 Agent 每个 stream 的默认预算为 20,000 字符，模型 Schema 仍只有 `command` 与 `timeout`。
- Head/Tail：超预算 Shell 输出保留开头和结尾，中间插入包含 stream 名称与原始字符数的 truncation marker，marker 本身计入预算，最终单个 stream 长度不超过配置值。
- Shell Metadata：保留 `exit_code`、`stdout`、`stderr`、`timed_out`，并增加 `stdout_truncated`、`stderr_truncated`、`stdout_original_chars` 与 `stderr_original_chars`。
- 向后兼容：小文件和小 Shell 输出保持完整；nonzero exit 继续是普通 Tool Result；timeout 继续返回 `timed_out=true`，已捕获的 stdout/stderr 也经过相同预算处理。
- Context 分层：P1-2 控制进入 Full History 前的单次 Tool Result；Stage 10 的 `_build_context_messages()` 与整体 Context Budget 未修改，二者职责独立。
- Workspace Safety：Read 分页继续复用 Stage 11 `resolve_workspace_path()` canonical boundary；路径穿越和 Workspace 外绝对路径仍被拒绝。
- Verification：`ReadFileTool.mutates_workspace=False`、`RunCommandTool.mutates_workspace=True` 与 `VerifyWorkspaceTool` 的成功判定保持不变，P1-2 不改变 Stage 12 Completion Gate 语义。
- 专项测试命令：`python -m pytest tests/test_tool_output_budget.py -v --basetemp=.pytest_tmp`；结果：严格 8 项测试全部通过。Test 4 同时覆盖普通多行的完整行边界与下一页可恢复性，以及超长首行的有界局部返回和诚实 metadata。
- Fake LLM 验证命令：`python -m scripts.verify_tool_output_budget`；结果：默认 Read 返回 1–120 / 1000 且不含后段随机目标，分页读取 976–1000 后恢复目标；真实 Shell stdout 从 12,154 字符截断为 800 字符并保留 Head/Tail，所有有界 Tool Result 均进入下一轮 messages 与 Full History，临时 Workspace 已清理。
- Real LLM 验证命令：`python -m scripts.verify_tool_output_budget_real`；结果：DeepSeek-V4-Flash 使用真实 `LLMClient` 连续读取 6 个 150 行窗口，在实际第 798 行发现运行时随机 TARGET；随后仅执行 Host 提供的 Python 命令，stdout 从 24,171 字符截断为 600 字符并保留随机 Head/Tail。三类随机 marker 均由真实 Tool Result 获得，所有 Tool Result 均进入紧接着的真实 API messages，临时 Workspace 已清理。
- 正式入口 smoke：`python -m src.main --workspace <synthetic-workspace>` 通过；真实模型使用 `read_file` 读取合成 token `P1-2-CLI-SMOKE=boundary-repair-ok` 并准确返回，没有修改文件或执行命令，临时 Workspace 已清理。
- 完整回归命令：`python -m pytest -v --basetemp=.pytest_tmp`；结果：94 项测试全部通过（P1-2 开发前基线 86 项，本阶段严格新增 8 项；本次边界修复前后均为 94 项）。
- 建议截图命名（本次未生成截图）：`p1_2_tool_output_budget.png`、`p1_2_tool_output_budget_manual_1.png`、`p1_2_tool_output_budget_manual_2.png`、`p1_2_tool_output_budget_real_llm_1.png`、`p1_2_tool_output_budget_real_llm_2.png`、`p1_2_full_regression.png`。

## P1-3：分层 Context Compaction

- 验证目标：当 Full History 达到 Host 配置的 Trigger 后，为单次 LLM 请求构造有界的分层 Context View；早期信息转换为确定性摘要，真实 Current User 与最近连续 ContextUnits 保持原始协议结构，同时不破坏既有 Agent Loop。
- Full History：`self._messages` 继续保存完整真实历史，是唯一事实来源。Harness-generated synthetic messages 只存在于当前 API Request View，不写回 History；`history` 深拷贝和 `reset_history()` 保持 Stage 9 原语义。
- 配置与 Trigger：Agent 的 `compaction_trigger_chars` 与 `max_compaction_chars` 默认均为 `None`，因此默认继续使用 Stage 10 旧路径。正式入口通过 `--compact-context` 启用；Host 根据 `max_context_chars` 派生 75% Trigger 和 25% synthetic digest 总预算。完整 History 的稳定 JSON 字符数达到 Trigger 时才进入 Compaction。
- Deterministic Compaction：摘要由 Harness 对 ContextUnit 做结构化、抽取式渲染和有界 Head/Tail 裁剪，不调用额外 LLM，不保存独立摘要状态；相同 History、Current User Anchor 与预算会生成完全相同的 Context。进入 Digest 的 Tool Unit 现在额外生成通用 Structured Tool Progress：按 workflow 保留最新客观状态，并在固定单元/全局预算内保留 Tool Result 中符合通用规则的短 `KEY=VALUE` 精确锚点与少量正文 Head/Tail。
- Current User Anchor：`run()` 在追加真实用户消息前记录其精确 History index，并在该次循环的每次 LLM 调用中持续使用同一 index；Stage 12 的 `[Verification Required]` user-role Harness feedback 不会被误认成真实 Current User。
- Layering：最终顺序为完整 System、Compacted Prior Context、可用的 prior raw 后缀、完整 Current User、Compacted Current-Run Progress、recent raw progress。所有非 Anchor ContextUnits 按全局时间顺序从最新向前选择连续 Raw 后缀，第一个放不下时停止，不跳过较新大 Unit 去保留更老小 Unit。
- Tool Atomicity：一条 Assistant Tool Call 消息及其全部匹配 Tool Results 组成一个 ContextUnit；Raw 时整体保留原生协议，Compact 时整体转成普通 synthetic 文本，不会产生 orphan Tool Result。成功、Tool Error、`verify_workspace` 结果均使用相同规则。
- Compaction Budget：最多两个 synthetic blocks 共享同一个 `max_compaction_chars` 内容预算；普通 User / Assistant 与 Tool Progress 的 metadata、exact anchors、正文 edges 均有固定 Host 上限。最终 API messages 仍使用 Stage 10 的稳定 JSON 字符计数，并必须满足 `serialized_context_chars <= max_context_chars`；如果完整 System 与原始 Current User 自身超限，仍在 LLM 调用前抛出 `AgentContextLimitError`。
- Small-budget block identity：Prior / Current-Run block 的 `kind` 由 Harness 内部结构字段维护并贯穿 clipping；Harness 不再从可能被截断的 Header 文本反向推断类型，因此很小的 compaction budget 下两个 layer 也不会因 Header 截断而静默丢失。最终发送给 provider 的 message 仍只包含 `role` / `content`。
- Stage 12 Compatibility：Workspace Revision、Verification State、Completion Gate、`max_steps` 和错误恢复逻辑未修改。最新 Verification feedback 与 Tool Exchange 可以作为 recent raw ContextUnits 保留，较旧进度可以确定性压缩。
- P1-2 Compatibility：Read / Shell Tool Result 先经过 P1-2 单次输出预算，再进入 Full History 和 P1-3；recent raw 单元保留原 metadata，只有进入 Digest 后才生成结构化 progress。`read_file` 保留请求窗口、返回范围、截断字段与 `next_start_line`，最终页通用推导 `read_status: exhausted`；`run_command` 非零退出仍保持 `execution_ok: true` 的 Domain Failure 语义，并保留 exit / timeout / truncation metadata 与有界输出边缘。
- 本轮专项测试命令：`python -m pytest tests/test_context_compaction.py tests/test_context_compaction_tool_progress.py -v --basetemp=.pytest_tmp`；结果：严格 15 项测试全部通过，覆盖 9 页 pagination 的中间精确锚点、跨 Run Prior Context 的最终 exhausted state、多个文件 latest state、Shell 截断/非零退出、受控 Tool Error、write/edit/verification progress、未知/异常格式 fallback、确定性、Full History 不变与硬预算。
- Fake LLM 验证命令：`python -m scripts.verify_context_compaction`；结果：同一 Agent 完成三次 `run()` 和四次真实 `ReadFileTool` 交换。配置为 `5500 / 2500 / 1400`；最终 Context 为 4573 字符，Prior 与 Current-Run Digest 各 700 字符，最新两组 Tool Exchanges 保持原生且原子。由于当前运行工具进度比 Run 2 更新，`RECENT_MARKER` 按全局连续后缀规则进入 Prior Digest并完整保留；Full History 仍包含全部原始 User、Tool Call 和 Tool Result，synthetic messages 不在 History 中，确定性检查通过，临时 Workspace 已清理。
- 真实模型验证命令：`python -m scripts.verify_context_compaction_real`；结果：`DeepSeek-V4-Flash` 通过真实 `LLMClient` 完成恰好 3 次调用。Run 1 原始长消息为 5821 字符并在 Run 3 退出 Raw Context；随机 `CONSTRAINT_TOKEN` 保留在 886 字符的 Prior Digest 中，Run 2 的随机 `RECENT_TOKEN` 作为最近 Raw Message 保留，Run 3 Current User 保持原文。最终真实 API Context 为 1977 / 7000 字符，模型精确返回两条完整随机 Marker，Full History 完整且无 synthetic message，临时资源已清理。
- 正式入口验证：`python -m src.main --help` 已显示唯一新增的 `--compact-context` 开关；随后在隔离临时 Workspace 中运行 `python -m src.main --workspace <temp> --max-context-chars 12000 --compact-context`，真实 Agent 正常启动并返回 `P1-3-CLI-SMOKE-OK`，退出码为 0，临时目录已清理。
- 兼容回归：Stage 9 / Stage 10 / Stage 12 与正式入口相关的 35 项测试全部通过。
- 完整回归命令：`python -m pytest -v --basetemp=.pytest_tmp`；结果：102 项测试全部通过（P1-3 开发前基线 94 项，本阶段严格新增 8 项）。
- 建议截图命名（本次未生成截图）：`p1_3_context_compaction.png`、`p1_3_context_compaction_manual_1.png`、`p1_3_context_compaction_manual_2.png`、`p1_3_context_compaction_real_llm_1.png`、`p1_3_context_compaction_real_llm_2.png`、`p1_3_full_regression.png`。

## P1-4：统一 Tool Execution Result / Error Boundary

- 目标：将 Tool Registry 解析、本地工具执行、普通运行时异常捕获和模型可见 Structured Tool Error 集中到单一 Harness 执行边界；Parser 与 LLM API 错误继续位于更高层边界。
- Internal Result：Harness 内部使用冻结的 `ToolExecutionResult(tool_name, content, execution_ok, error_type)`；`execution_ok` 仅表示 Registered Tool 是否正常返回，不表示命令、验证或任务的 Domain Success。该内部结构不会直接发送给模型。
- One Call / One Result：每个已解析 Tool Call 都产生且只产生一个对应 `role=tool` Result，沿用模型给出的原始 `tool_call_id`；Provider message 仍只有 `role`、`tool_call_id` 与 `content`。
- Unknown Tool：未注册工具被规范化为 `error_type=UnknownTool` 的稳定 JSON Tool Error，不修改 Workspace Revision，也不让 Registry exception 终止 Agent Loop。
- Runtime Exception：Registered Tool 抛出的普通 `Exception` 被转换为原有四字段 JSON Tool Error；只包含 tool、异常类型和消息，不泄漏 Python traceback。`BaseException` 控制异常不被捕获。
- Domain Result：`run_command` 非零退出、timeout 和 `verify_workspace` 的 `ok=false` 都是 `execution_ok=true` 的普通 Tool Result；各 Tool 的既有业务 content 原样透传，不增加统一业务 envelope。
- Multi-Call：同一响应中的多个 Tool Calls 继续按模型原始顺序串行执行；一个 Unknown Tool 或 Runtime Error 不会跳过同批后续 sibling call。
- Stage 8 Compatibility：LLM API error、Model/Parser error、Tool Execution error 仍分别进入 `AgentLLMError`、`AgentResponseError`、model-facing Structured Tool Error；不自动重试 Tool。
- P1-2 Compatibility：Read/Shell Output Budget 仍由各 Tool 在正常返回前完成，P1-4 不进行二次截断或重新序列化。
- P1-3 Compatibility：Tool Result 进入 Full History 后继续作为对应 Tool Call 的原子 ContextUnit；Raw 与 Compacted 路径和预算算法未修改。
- Stage 12 Compatibility：Registered mutating tool 在实际执行前继续保守增加 `workspace_revision`，因此即使随后抛异常也会保持 DIRTY；Unknown Tool 不触发 mutation bookkeeping。Verify PASS 才同步 `verified_revision`，Verify FAIL 或 Verify Runtime Exception 均不会错误标记已验证，Completion Gate 原语义保持。
- 专项测试命令：`python -m pytest tests/test_tool_execution.py -v --basetemp=.pytest_tmp`；结果：严格 8 项测试全部通过，覆盖 success、unknown、runtime、one-call-one-result、multi-call、Shell domain failure、mutating exception 和 Verification Gate。
- 相关回归：`python -m pytest tests/test_agent_error_handling.py tests/test_agent_verification.py tests/test_tool_output_budget.py tests/test_context_compaction.py -v --basetemp=.pytest_tmp`；结果：Stage 8、Stage 12、P1-2 与 P1-3 共 28 项测试全部通过。
- Fake LLM 验证命令：`python -m scripts.verify_tool_execution`；结果：同批 `echo_demo → ghost_tool → fail_demo` 依次产生 normal / UnknownTool / RuntimeError Results，错误未阻断 sibling；下一轮真实 `ReadFileTool` 读取随机目标并完成恢复。Tool Result ID、数量、顺序、下一轮 messages、Full History、无 traceback、无内部字段泄漏和临时目录清理全部验证通过。
- 真实模型验证命令：`python -m scripts.verify_tool_execution_real`；结果：`DeepSeek-V4-Flash` 通过真实 `LLMClient` 恰好完成 3 次调用，严格执行 `fail_demo → read_file → Final`。RuntimeError Result 进入下一次真实 API messages，随机 `TARGET_MARKER` 首次来自真实 Read Tool Result，模型最终精确返回完整 marker，临时 Workspace 已清理。
- 正式入口：`python -m src.main --help` 通过；保留 `workspace`、Context、Verification 与 `--compact-context` 参数，没有新增 P1-4 CLI flag，也没有将验证专用 Demo Tools 注册到正式 Tool List。
- 完整回归命令：`python -m pytest -v --basetemp=.pytest_tmp`；结果：110 项测试全部通过（P1-4 开发前真实 baseline 102 项，本阶段严格新增 8 项）。
- 建议截图命名（本次未生成截图）：`p1_4_tool_execution.png`、`p1_4_tool_execution_manual.png`、`p1_4_tool_execution_real_llm_1.png`、`p1_4_tool_execution_real_llm_2.png`、`p1_4_full_regression.png`。

## Stage 13：End-to-End Coding Task Evaluation

- 目标：冻结现有 Agent 核心能力，在相互隔离的临时 Workspace 中使用真实 Production Agent 完成完整 Coding Task，并由 Host 进行确定性的最终客观评价。
- Task 1：单文件 Bug Fix；定位并修复 `calculator.add`，同时保持 `subtract` 的正确行为。
- Task 2：功能实现；根据公开需求实现 `normalize_words`，包括通用空白分隔、小写转换、空输入与标点保留。
- Task 3：多文件任务；实现 `UserStore.exists` 与 `register_user` 的重复用户、空白名称、名称规范化和存储契约。
- Workspace：每个任务从运行时生成的全新 `TemporaryDirectory` fixture 开始，使用新的 Agent 与 RecordingLLM，不共享 History 或 Verification State；三个任务结束后临时目录均已自动清理。
- Production Agent：真实 runner 直接复用 `src.main.build_agent()`，因此使用正式 System Prompt、20 steps、60,000 字符 Context、Read/Shell 输出预算、Unified Tool Execution Boundary，以及 `list_directory`、`read_file`、`edit_file`、`write_file`、`run_command`、`verify_workspace` 六个生产工具。
- Verification Gate：每个任务都通过非空 Host verification command 启用真实 Stage 12 Completion Gate；Agent 修改 Workspace 后必须调用 `verify_workspace` 并达到 `workspace_revision == verified_revision` 才能 Final。三个真实任务各调用 1 次 `verify_workspace`，最终状态均为 CLEAN。
- Visible Evaluation：Agent Workspace 内只包含 production source 与 visible pytest tests。Host 在 Agent 前确认三个初始 fixture 均真实失败，并在 Agent Final 后使用 `sys.executable -m pytest -q` 独立复验，结果为 3/3 PASS。
- Hidden Evaluation：额外行为断言只保存在 Host evaluator 中，在 Agent Final 后通过 Workspace 外的 `python -B -c` 子进程执行；hidden code 不写入 Workspace、不进入 Prompt、Tool Schema、Verification Result 或真实 API request，失败结果也不会回喂模型。真实结果为 3/3 PASS。
- Test Integrity：Agent 运行前后使用 SHA-256 检查 visible test 文件；修改或删除都会失败。真实运行中 `test_calculator.py`、`test_text_utils.py`、`test_service.py` 均保持不变，结果为 3/3 PASS。
- Objective PASS：单项必须同时满足初始 visible tests 失败、Agent 正常 Final、Host visible PASS、Host hidden PASS、protected test hash 不变、`verify_workspace >= 1`、Verification State CLEAN 和 Temporary Workspace 已清理；模型最终回答措辞不参与判定。
- Evaluator Tests：`python -m pytest tests/test_e2e_evaluator.py -v --basetemp=.pytest_tmp`；严格收集 6 项，结果为 `6 passed`。测试证明 visible-only 硬编码会被 Task 1/2 hidden checks 拒绝，Task 3 的四类错误实现会被拒绝，并覆盖 fixture 隔离、文件完整性和结果聚合。
- 相关回归：`python -m pytest tests/test_agent_verification.py tests/test_tool_execution.py tests/test_context_compaction.py tests/test_tool_output_budget.py -v --basetemp=.pytest_tmp`；Stage 12、P1-4、P1-3 与 P1-2 共 31 项测试全部通过。
- Real E2E：`python -m scripts.verify_e2e_coding_tasks_real`；`DeepSeek-V4-Flash` 串行完成 3/3。Task 1 为 7 次 LLM / 7 次 Tool，修改 `calculator.py`；Task 2 为 5 / 6，修改 `text_utils.py`；Task 3 为 5 / 7，修改 `service.py` 与 `store.py`。Visible、Hidden、Integrity、Verification Usage 与 Cleanup 均为 3/3 PASS。
- 正式入口：`python -m src.main --help` 通过，没有新增 Stage 13 CLI 参数。
- Full Regression：`python -m pytest -v --basetemp=.pytest_tmp`；开发前 baseline 为 110 项，本阶段严格新增 6 项，最终 `116 passed`。
- 建议截图命名（本次未生成截图）：`stage13_e2e_evaluator.png`、`stage13_e2e_task_1.png`、`stage13_e2e_task_2.png`、`stage13_e2e_task_3.png`、`stage13_e2e_summary.png`、`stage13_full_regression.png`。

## Final Integrated Long-Running Challenge

- 当前状态：第六次真实 Long-Running Challenge 已保留真实失败证据：Run 1–3 完成，Run 4 完成首轮 production repairs 后只执行 1 次 fixed visible pytest；该次测试失败后又产生 32 个没有成功 mutation 的 read-only model responses，直到 `40 / 40` steps 触发 `AgentMaxStepsError`，没有进入 diagnostic、trusted verification 或 Final。结构化 report、`.runs/long_running_challenge_20260901_193115.log` 和 `.runs/long_running_challenge/20260901_193117_6765e3e0/` Evidence 均保持不变；本次 58 次 LLM calls、221 次 Tool calls、约 473 秒、最大 request `23993 / 24000` chars。旧 telemetry 因 `run4_pytest_calls=1` 且 `run4_pytest_reruns_without_intervening_mutation=0` 将 Repair Loop 误报为 PASS；真实根因是 Prompt / Execution Budget 都属于 advisory guidance，Host 未强制阻止 failure 后持续只读。
- 第二次真实运行残留：Run 2 没有原样保留文档 external path 的前导 `../`，且仍检查了 tests 并重复完整读取；Run 3 的 visible pytest 受 Windows `%TEMP%/pytest-of-*` 权限错误干扰。Run 3 本身完成了合法的顺序 migration pagination 与 BLOCKED diagnostic，但原统计错误混入 Run 4 的违规 migration reads，并在 Run 4 abort 后把已成立的 Shell 截断证据标为 `NOT EVALUATED`。
- 第三次真实运行事实：4/4 Runs、36 次 LLM calls、67 次 Tool calls、19 个不同文件读取、29 个带 Compaction 的请求；7 个既有 production 文件修改并创建 `incident/report.py`。Run 3 首次 9 页顺序读取已到 `next_start_line=none`，但随后从 line 1 完整重启，最终 18 reads，因此 pagination 正确判定失败；三个 long-term exact token 未在最终回答恢复。Run 4 首次 mutation 发生在 step 4，随后 23 个 visible tests、READY diagnostic 与真实 verification 均通过；一个通用 public-input boundary hidden check 仍失败，所以 Functional Challenge 保持 FAIL。
- 第四轮最小修复：P1-3 增加通用 Structured Tool Progress、短 `KEY=VALUE` exact anchors、同 workflow latest-state 与 pagination exhausted 保留，且不修改 recent raw、Full History、预算或 Tool 原子性。Run 3 Prompt 明确 exhausted 后不得从 line 1 重启并立即进入 baseline；Run 4 健康 mutation deadline 与 evaluator 同步为 `<=4`，visible PASS 后增加一次基于已知 documented contract 的通用 public-input boundary self-review。最终回答采用严格尾部三行 token contract；report 新增三个只含布尔值的 final-request token visibility 字段，严格 recovery 判定与 request presence 相互独立。
- 第四轮真实验证前的本地验证：P1-3 专项 15 项、Long-Running / Evidence 专项 53 项、相关核心回归 99 项、完整回归 179 项全部通过；该本地阶段未执行真实收费模型。
- 第四次真实运行事实：4/4 Runs、48 次 LLM calls、118 次 Tool calls、20 个不同文件读取、41 个带 Compaction 的请求，最大 request 为 `23997 / 24000` chars；6 个 production 文件修改并创建 `incident/report.py`。Migration 仅顺序读取 9 页并到达 exhausted，三个 exact token 均进入最终 LLM request 且成功恢复；visible tests、READY diagnostic、Verification State CLEAN、Long-Running Coverage、约 516 秒 Runtime Target 与 Evidence Preservation 均通过。Hidden Group 3 因公开 CLI 契约未明确 unsupported `--min-severity` 的 usage-error 边界而失败。Run 4 首次 mutation 为 step 7，mutation 前有 26 次读取、其中 11 次重复完整读取；这些数据真实反映效率问题，但该 Run 已完成代码修复、visible tests、READY diagnostic 与 trusted verification。
- 第五轮最终最小修复与真实结果：fixture README 公开明确 `--min-severity` 的合法值大小写不敏感，unsupported 值必须在 incident processing 前于 CLI argument boundary 以 usage error / exit status 2 拒绝；具体 unseen input、Hidden evaluator 与 Visible tests 均不变。第五次真实运行共 58 次 LLM calls、168 次 Tool calls、51 个带 Compaction 的请求，约 653 秒且最大 request 为 `24000 / 24000` chars；Run 4 共运行 6 次 pytest、首次 mutation 为 step 7、mutation 前 21 次读取且有 7 次重复完整读取，最终在 `1 failed / 22 passed` 后耗尽 40 steps。根因定位为无进展 repair loop 与模型不可见剩余 step budget，而非 P1-3、Verification 或 Hidden 失败。
- 第六轮针对性修复与真实结果：`Agent` 在每次 LLM request 组装末端加入通用、瞬时的 Execution Budget，动态报告 current / max / remaining responses；该块不写入 Full History 或 Compaction digest，并在普通 Context 组装前按精确序列化大小预留容量，因此仍受原 `24000` 字符 hard limit 约束且不增加 `max_steps`。Run 4 Prompt 明确采用 failing pytest → 最多一次 targeted diagnosis → successful production mutation → retest 的循环。第六次真实运行证明这些 advisory 提示没有阻止模型在首个 failing pytest 后持续只读，也暴露了单次 pytest 时 `reruns_without_mutation=0` 的 Repair Loop 假阳性。
- 第七轮根因修复：增加通用、opt-in、per-run 的 Host-enforced Progress Guard；默认关闭，Runs 1–3 不启用，Run 4 仅配置 exact fixed pytest command、`edit_file` / `write_file` repair mutation 与 1 个 diagnosis response allowance。tracked pytest FAIL 后允许一轮 targeted diagnosis，随后 Host 在 Tool Executor 前阻止继续 read/list、其它 command 与 verification；只有成功 file mutation 才开放 exact retest，PASS 后恢复 NORMAL。阻止结果仍是与原 `tool_call_id` 一一配对的结构化 `ProgressGuardBlocked` Tool Result，进入真实 Full History，且不改 Tool schemas、Workspace Boundary、Tool Error Boundary、Stage 12 Verification Gate、Context Compaction 或 `40` steps ceiling。Telemetry 新增 failure 后 read-only response 数、Guard interventions、failure 后成功 mutation 数与 Run 结束 pending-failure 状态；Repair Loop 现在仅在“无未隔离 mutation 的 rerun”且“Run 结束无 pending failed test”时 PASS，intervention 只属于 Action Discipline evidence，不进入 Functional、Coverage、Final Integrated 或 Real Validation Process hard gate。第七次 Real Validation：**Pending**，本轮未执行真实收费模型。
- 第七轮补丁修复：补齐 `WriteFileTool` 的 identical-content no-op 语义；现有文件内容与请求内容按 UTF-8 exact string equality 完全相同时不再调用 `write_text`，并返回稳定的 `No changes made:` 结果。Stage 12 仍保守推进 Workspace revision 并保留 DIRTY state，但 Progress Guard 不再把该 no-op 当作 successful repair mutation；不同内容覆盖与新文件创建仍可开放 exact retest。下一次 Real Validation：**Pending**，本轮未执行真实收费模型。
- 最新真实 DeepSeek-V4-Flash 验证：Run 1–4 全部完成，Run 4 在 `29 / 40` steps 内结束；Visible Host Re-check、Diagnostic READY、trusted Verification CLEAN、Long-Running Coverage 与 Evidence Preservation 均通过。Hidden Group 1/2 通过，Group 3 失败；唯一 Functional failure 已定位为 benchmark 的公开 CLI contract 缺口：`--report` 的 visible oracle 未强制完整 canonical v2 report schema，因此 reduced CLI payload 通过了 visible verification，最终由 Hidden integration check 发现。
- Final benchmark-contract patch：fixture README 现已公开 `--report` 的完整 v2 schema；visible CLI integration test 对固定公开输入执行独立的完整 payload equality；Run 4 增加简短的 canonical report behavior reminder；deterministic regression 同时证明 reduced shape 被拒绝、complete shape 被接受，并防止 Visible/Hidden contract drift。Hidden evaluator 保持不变。下一次 Real Validation：**Pending**，本轮未执行真实收费模型。
- Challenge Steps：`src.main.build_agent()` 新增通用 keyword-only `max_steps` 透传，production 默认仍为 20；Long-Running runner 单独使用 40 作为 ceiling，不要求模型消耗满 40 steps，也没有增加 Challenge mode 特判。
- 挑战场景：在全新的 Temporary Workspace 中动态生成 Incident Triage Service v2 repository repair fixture，并由同一个正式 Agent 连续完成 Release Briefing、Repository Reconnaissance、Migration Audit + Baseline Diagnostics、Release Repair 四个 Run。
- Multi-run Completion Policy：`Agent.run(..., require_verified_completion=False)` 仅允许前三个中间调查 Run 在保留 DIRTY state 的情况下返回；第四个 Run 使用默认的 verified completion policy，必须通过真实 `verify_workspace` 后才能完成。Revision、mutating tool、Workspace Boundary 和 Tool Error Boundary 均保持正式语义。
- Repository Evidence：初始 21 个文件、23 个 visible tests、10+ distinct files read、5+ existing production files changed、至少 1 个新文件；`incident/report.py` 初始不存在并必须由 Agent 创建。
- Long-running Coverage：硬门槛为 4 Agent Runs、至少 20 次 LLM calls、30 次 Tool calls、8 次 migration pagination reads、3 个带 compaction 的真实请求、2 个受控错误恢复和至少 1 次 trusted verification。20–40 LLM、30–60 Tool 与 5–15 分钟仅作为目标带或独立 runtime 证据。
- Read / Shell Budget：模型必须根据真实 `next_start_line` 顺序读完 1800 行 migration notes；diagnostic stdout 超过正式 Shell budget，验证 Head/Tail 保留和中部 sentinel 移除。
- Context Lifecycle：启用正式 P1-3 Context Compaction；telemetry 使用 Stage 10 的真实 message-character 计算和当前 compaction markers，确认每个请求不超过 configured context limit，并验证 synthetic digest 不进入 Full History。
- Error Recovery / Safety：真实观察缺失 legacy config 与 workspace boundary rejection 的结构化 Tool Error，确认错误后继续使用合法 workspace evidence；tests、policy、migration notes 和 diagnostic script 均由 SHA-256 保护。
- Objective Evaluation：Agent Final 后由 Host 独立执行 visible pytest 和 workspace 外 hidden behavior checks；Hidden 结果不会写入 fixture、Prompt、Tool Result 或后续 Agent messages。
- Evidence Preservation：每次正式运行会创建唯一的 `.runs/long_running_challenge/<run_id>/`，在 Fixture materialization 后、首次 visible pytest 和 LLM 调用前保存 `initial_workspace/` 与 `file_manifest_before.json`；在 Run 4 Final 返回后、Host visible/hidden evaluation 前保存 `final_workspace/`、`file_manifest_after.json`、`workspace_changes.json` 与 `workspace_changes.diff`。
- Workspace Forensics：文件创建、修改、删除和未变化状态通过初始/最终 Workspace 的确定性 SHA-256 Manifest 对比得出，不根据候选 production paths、读取记录或 Tool Call 推测。`production_files_changed` 仅统计 `changes.modified` 中的 production 文件，`files_created` 直接来自 `changes.created`；统一 Diff 仅展示真实变化文件，并支持 created、modified、deleted 及 non-UTF8/binary change marker。
- Evidence Lifecycle：Temporary Workspace 在运行结束后仍会自动删除，取证副本会在删除前持久保存到项目自身的 `.runs/`。`.runs/` 已被 Git ignore，只用于本地原始取证，不提交 GitHub；公开结构化 report 只保存 repo-relative artifact path 和脱敏状态，不保存 API key、runtime token 原值或 Temporary Workspace 绝对路径。
- Result Separation：`FUNCTIONAL CHALLENGE`、`LONG-RUNNING COVERAGE`、`RUNTIME TARGET`、`ACTION DISCIPLINE` 与 `EVIDENCE PRESERVATION` 分别报告；`final_integrated_success` 仍只表示 Functional + Coverage，正式进程退出码额外要求 Evidence Preservation 成功。Action Discipline 使用独立 `PASS/WARNING` 语义；pending failed test 会产生 Repair Loop warning，Guard intervention 只计数而不产生失败。二者都不影响 Functional、Coverage、Final Integrated 或 Real Validation Process；其余单项检查继续使用 `PASS`、`FAIL`、`NOT EVALUATED` 三态。
- Timing / Report：runner 同时记录 challenge、每个 Run、每个 LLM call、Host visible 与 Hidden evaluation 的 monotonic duration；结构化结果写入 `docs/verification/long_running_challenge_report.json`，但不保存真实 token、API key、完整 Prompt、Full History 或 Temporary Workspace 路径。
- Provider Usage：当前正式 `LLMClient.chat()` 不暴露外层 completion usage，因此 token usage 会诚实报告为 unavailable / `null`；request message chars 作为独立、准确的字符指标记录，不伪装成 provider tokens。
- 运行日志：第六次完整 trace 保存在被 Git 忽略的 `.runs/long_running_challenge_20260901_193115.log`，对应 Evidence 为 `.runs/long_running_challenge/20260901_193117_6765e3e0/`；第五次 `.runs/long_running_challenge_20260901_172126.log`、第四次 `.runs/long_running_challenge_20260901_155654.log` 及更早证据继续保留。后续正式命令仍使用新的时间戳日志，不覆盖历史证据。
- 建议人工截图：`longrunning_01_audit_errors.png`、`longrunning_02_pagination_compaction.png`、`longrunning_03_multifile_repair.png`、`longrunning_04_verification_hidden.png`、`longrunning_05_summary_timing.png`。
- 正式 Real Validation 命令：`chcp 65001 > $null; [Console]::InputEncoding=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $OutputEncoding=[System.Text.UTF8Encoding]::new($false); $env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"; New-Item -ItemType Directory -Force .\.runs | Out-Null; $log=".\.runs\long_running_challenge_$((Get-Date).ToString('yyyyMMdd_HHmmss')).log"; $sw=[System.Diagnostics.Stopwatch]::StartNew(); python -m scripts.verify_long_running_challenge_real 2>&1 | Tee-Object -FilePath $log; $code=$LASTEXITCODE; $sw.Stop(); Write-Host "LOG_FILE=$log"; Write-Host "PYTHON_EXIT_CODE=$code"; Write-Host "EXTERNAL_WALL_CLOCK_SECONDS=$([math]::Round($sw.Elapsed.TotalSeconds,2))"; Write-Host "EXTERNAL_WALL_CLOCK=$($sw.Elapsed)"`
