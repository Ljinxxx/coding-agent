from src.llm import LLMClient


def main() -> None:
    llm = LLMClient()

    task = input("Task: ").strip()

    if not task:
        print("Task cannot be empty.")
        return

    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant.",
        },
        {
            "role": "user",
            "content": task,
        },
    ]

    response = llm.chat(messages)

    print("\nAssistant:")
    print(response)


if __name__ == "__main__":
    main()