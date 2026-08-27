from openai import OpenAI

from src.config import API_KEY, BASE_URL, MODEL, validate_config


class LLMClient:
    def __init__(self) -> None:
        validate_config()

        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

        self.model = MODEL

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content or ""