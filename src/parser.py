import json
from typing import Any

from src.response import ParsedResponse, ToolCall


class ResponseParser:
    def parse(self, message: Any) -> ParsedResponse:
        parsed_tool_calls: list[ToolCall] = []

        for tool_call in message.tool_calls or []:
            function = tool_call.function

            try:
                arguments = json.loads(function.arguments)
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(
                    f"Failed to parse arguments for tool '{function.name}' as JSON."
                ) from error

            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Arguments for tool '{function.name}' must be a JSON object."
                )

            parsed_tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=function.name,
                    arguments=arguments,
                )
            )

        return ParsedResponse(
            content=message.content or "",
            tool_calls=parsed_tool_calls,
        )
