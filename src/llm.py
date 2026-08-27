from typing import Any, overload

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from src.config import API_KEY, BASE_URL, MODEL, validate_config


class LLMClient:
    def __init__(self) -> None:
        validate_config()

        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

        self.model = MODEL

    @overload
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: None = None,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ChatCompletionMessage: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str | ChatCompletionMessage:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if tools is not None:
            request["tools"] = tools

        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message

        if tools is not None:
            return message

        return message.content or ""
