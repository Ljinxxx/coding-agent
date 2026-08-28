import json
from typing import Any

from src.parser import ResponseParser
from src.tools.registry import ToolRegistry


class AgentMaxStepsError(RuntimeError):
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

    def run(self, user_input: str) -> str:
        messages: list[dict[str, Any]] = []

        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        for step in range(1, self.max_steps + 1):
            self._log(f"Step {step}/{self.max_steps}")
            response = self.llm_client.chat(
                messages,
                tools=self.tool_registry.schemas(),
            )
            parsed = self.parser.parse(response)

            if not parsed.has_tool_calls:
                return parsed.content

            messages.append(
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

                tool = self.tool_registry.get(tool_call.name)
                result = tool.execute(**tool_call.arguments)

                self._log(f"工具结果：{result}")
                messages.append(
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
