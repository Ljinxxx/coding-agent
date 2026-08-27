from types import SimpleNamespace

import pytest

from src.parser import ResponseParser


def test_parse_text_response() -> None:
    message = SimpleNamespace(
        content="Task completed.",
        tool_calls=None,
    )

    parsed = ResponseParser().parse(message)

    assert parsed.content == "Task completed."
    assert parsed.tool_calls == []
    assert parsed.has_tool_calls is False


def test_parse_tool_call() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="echo",
                    arguments='{"text": "hello"}',
                ),
            )
        ],
    )

    parsed = ResponseParser().parse(message)

    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].id == "call_1"
    assert parsed.tool_calls[0].name == "echo"
    assert parsed.tool_calls[0].arguments == {"text": "hello"}
    assert parsed.has_tool_calls is True


def test_parse_multiple_tool_calls() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="echo",
                    arguments='{"text": "first"}',
                ),
            ),
            SimpleNamespace(
                id="call_2",
                function=SimpleNamespace(
                    name="another_tool",
                    arguments='{"value": 2}',
                ),
            ),
        ],
    )

    parsed = ResponseParser().parse(message)

    assert [tool_call.name for tool_call in parsed.tool_calls] == [
        "echo",
        "another_tool",
    ]
    assert [tool_call.arguments for tool_call in parsed.tool_calls] == [
        {"text": "first"},
        {"value": 2},
    ]


def test_invalid_tool_arguments() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="echo",
                    arguments="{invalid json",
                ),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Failed to parse arguments for tool 'echo' as JSON\\.",
    ):
        ResponseParser().parse(message)
