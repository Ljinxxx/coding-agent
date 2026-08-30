import json
from copy import deepcopy
from typing import Any

from src.context_compaction import (
    build_compacted_context,
    estimate_messages_size,
)
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


class AgentContextLimitError(RuntimeError):
    pass


class Agent:
    _VERIFICATION_REQUIRED_FEEDBACK = (
        "[Verification Required]\n"
        "The workspace has changed since the last successful verification.\n"
        "You must call {tool_name} and obtain a successful result before "
        "completing the task.\n"
        "This is harness-generated control feedback, not a new user request."
    )

    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry,
        system_prompt: str | None = None,
        verbose: bool = False,
        max_steps: int = 20,
        max_context_chars: int | None = None,
        compaction_trigger_chars: int | None = None,
        max_compaction_chars: int | None = None,
        verification_tool_name: str | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if max_context_chars is not None and max_context_chars < 1:
            raise ValueError(
                "max_context_chars must be greater than 0 or None."
            )
        if (compaction_trigger_chars is None) != (
            max_compaction_chars is None
        ):
            raise ValueError(
                "compaction_trigger_chars and max_compaction_chars must "
                "both be configured or both be None."
            )
        if compaction_trigger_chars is not None:
            if max_context_chars is None:
                raise ValueError(
                    "Context compaction requires max_context_chars."
                )
            if (
                isinstance(compaction_trigger_chars, bool)
                or not isinstance(compaction_trigger_chars, int)
                or compaction_trigger_chars < 1
            ):
                raise ValueError(
                    "compaction_trigger_chars must be a positive integer."
                )
            if (
                isinstance(max_compaction_chars, bool)
                or not isinstance(max_compaction_chars, int)
                or max_compaction_chars < 1
            ):
                raise ValueError(
                    "max_compaction_chars must be a positive integer."
                )
            if compaction_trigger_chars > max_context_chars:
                raise ValueError(
                    "compaction_trigger_chars cannot exceed "
                    "max_context_chars."
                )
        if (
            verification_tool_name is not None
            and verification_tool_name not in tool_registry.names()
        ):
            raise ValueError(
                "Verification tool "
                f"'{verification_tool_name}' is not registered."
            )

        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.max_steps = max_steps
        self.max_context_chars = max_context_chars
        self.compaction_trigger_chars = compaction_trigger_chars
        self.max_compaction_chars = max_compaction_chars
        self.verification_tool_name = verification_tool_name
        self.parser = ResponseParser()
        self._messages = self._create_initial_history()
        self._workspace_revision = 0
        self._verified_revision = 0

    @property
    def history(self) -> list[dict[str, Any]]:
        return deepcopy(self._messages)

    @property
    def workspace_revision(self) -> int:
        return self._workspace_revision

    @property
    def verified_revision(self) -> int:
        return self._verified_revision

    @property
    def verification_required(self) -> bool:
        return self._workspace_revision != self._verified_revision

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

    def _estimate_messages_size(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        return estimate_messages_size(messages)

    def _split_history_into_turns(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        system_messages: list[dict[str, Any]] = []
        conversation_messages = messages

        if messages and messages[0].get("role") == "system":
            system_messages = [messages[0]]
            conversation_messages = messages[1:]

        turns: list[list[dict[str, Any]]] = []
        for message in conversation_messages:
            if message.get("role") == "user":
                turns.append([message])
            else:
                turns[-1].append(message)

        return system_messages, turns

    def _build_legacy_context_messages(self) -> list[dict[str, Any]]:
        if self.max_context_chars is None:
            return deepcopy(self._messages)

        system_messages, turns = self._split_history_into_turns(
            self._messages
        )
        current_turn = turns[-1] if turns else []
        selected_turns = [current_turn] if current_turn else []
        context_messages = system_messages + current_turn
        required_size = self._estimate_messages_size(context_messages)

        if required_size > self.max_context_chars:
            raise AgentContextLimitError(
                "Required context size "
                f"{required_size} exceeds max_context_chars "
                f"{self.max_context_chars}."
            )

        for turn in reversed(turns[:-1]):
            candidate_turns = [turn, *selected_turns]
            candidate_messages = system_messages + [
                message
                for selected_turn in candidate_turns
                for message in selected_turn
            ]

            if (
                self._estimate_messages_size(candidate_messages)
                > self.max_context_chars
            ):
                break

            selected_turns = candidate_turns
            context_messages = candidate_messages

        return deepcopy(context_messages)

    def _build_context_messages(
        self,
        current_user_index: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.compaction_trigger_chars is None:
            return self._build_legacy_context_messages()

        if (
            self._estimate_messages_size(self._messages)
            < self.compaction_trigger_chars
        ):
            return self._build_legacy_context_messages()

        if current_user_index is None:
            raise ValueError(
                "current_user_index is required when context compaction "
                "is active."
            )

        system_messages = (
            [self._messages[0]]
            if self._messages
            and self._messages[0].get("role") == "system"
            else []
        )
        current_user = self._messages[current_user_index]
        required_messages = [*system_messages, current_user]
        required_size = self._estimate_messages_size(required_messages)
        if required_size > self.max_context_chars:
            raise AgentContextLimitError(
                "Required context size "
                f"{required_size} exceeds max_context_chars "
                f"{self.max_context_chars}."
            )

        context_messages = build_compacted_context(
            self._messages,
            current_user_index=current_user_index,
            max_context_chars=self.max_context_chars,
            max_compaction_chars=self.max_compaction_chars,
        )
        context_size = self._estimate_messages_size(context_messages)
        if context_size > self.max_context_chars:
            raise AgentContextLimitError(
                "Compacted context size "
                f"{context_size} exceeds max_context_chars "
                f"{self.max_context_chars}."
            )
        return context_messages

    def _verification_succeeded(self, result: str) -> bool:
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return False

        return isinstance(payload, dict) and payload.get("ok") is True

    def run(self, user_input: str) -> str:
        current_user_index = len(self._messages)
        self._messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        for step in range(1, self.max_steps + 1):
            self._log(f"Step {step}/{self.max_steps}")
            tool_schemas = self.tool_registry.schemas()
            context_messages = self._build_context_messages(
                current_user_index=current_user_index
            )
            try:
                response = self.llm_client.chat(
                    context_messages,
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
                if (
                    self.verification_tool_name is not None
                    and self.verification_required
                ):
                    feedback = self._VERIFICATION_REQUIRED_FEEDBACK.format(
                        tool_name=self.verification_tool_name
                    )
                    self._messages.append(
                        {
                            "role": "user",
                            "content": feedback,
                        }
                    )
                    self._log("Completion Gate：需要成功验证后才能结束")
                    if step < self.max_steps:
                        self._log("再次调用模型")
                    continue

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
                    if tool.mutates_workspace:
                        self._workspace_revision += 1
                    result = tool.execute(**tool_call.arguments)
                except Exception as error:
                    self._log(f"工具执行失败：{tool_call.name}")
                    result = _format_tool_error(tool_call.name, error)

                if (
                    tool_call.name == self.verification_tool_name
                    and self._verification_succeeded(result)
                ):
                    self._verified_revision = self._workspace_revision

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
