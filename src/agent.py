import json
from copy import deepcopy
from typing import Any

from src.parser import ResponseParser
from src.tools.registry import ToolRegistry


def _format_tool_error(tool_name: str, error: Exception) -> str:
    return json.dumps(
        {
            "ok": False,
            "tool": tool_name,
            "error_type": type(error).__name__,
            "message": str(error),
        },
        ensure_ascii=False,
    )


class AgentMaxStepsError(RuntimeError):
    pass


class AgentLLMError(RuntimeError):
    pass


class AgentResponseError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry,
        system_prompt: str | None = None,
        verbose: bool = False,
        max_steps: int = 20,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.max_steps = max_steps
        self.parser = ResponseParser()
        self._messages = self._create_initial_history()

    @property
    def history(self) -> list[dict[str, Any]]:
        return deepcopy(self._messages)

    def reset_history(self) -> None:
        self._messages = self._create_initial_history()

    def _create_initial_history(self) -> list[dict[str, Any]]:
        if not self.system_prompt:
            return []

        return [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def run(self, user_input: str) -> str:
        self._messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        for step in range(1, self.max_steps + 1):
            self._log(f"Step {step}/{self.max_steps}")
            tool_schemas = self.tool_registry.schemas()
            try:
                response = self.llm_client.chat(
                    self._messages,
                    tools=tool_schemas,
                )
            except Exception as error:
                raise AgentLLMError(f"LLM request failed: {error}") from error

            try:
                parsed = self.parser.parse(response)
            except ValueError as error:
                raise AgentResponseError(
                    f"Failed to parse model response: {error}"
                ) from error

            if not parsed.has_tool_calls:
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": parsed.content,
                    }
                )
                return parsed.content

            self._messages.append(
                {
                    "role": "assistant",
                    "content": parsed.content or None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(
                                    tool_call.arguments,
                                    ensure_ascii=False,
                                ),
                            },
                        }
                        for tool_call in parsed.tool_calls
                    ],
                }
            )

            for tool_call in parsed.tool_calls:
                self._log(f"模型请求工具：{tool_call.name}")
                self._log(
                    "参数："
                    + json.dumps(tool_call.arguments, ensure_ascii=False)
                )

                try:
                    tool = self.tool_registry.get(tool_call.name)
                    result = tool.execute(**tool_call.arguments)
                except Exception as error:
                    self._log(f"工具执行失败：{tool_call.name}")
                    result = _format_tool_error(tool_call.name, error)

                self._log(f"工具结果：{result}")
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            if step < self.max_steps:
                self._log("再次调用模型")

        raise AgentMaxStepsError(
            f"Agent reached maximum step limit: {self.max_steps}"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Agent] {message}")
