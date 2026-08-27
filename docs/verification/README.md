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