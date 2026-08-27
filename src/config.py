import os

from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")


def validate_config() -> None:
    if not API_KEY:
        raise ValueError("LLM_API_KEY is not set.")

    if not BASE_URL:
        raise ValueError("LLM_BASE_URL is not set.")

    if not MODEL:
        raise ValueError("LLM_MODEL is not set.")