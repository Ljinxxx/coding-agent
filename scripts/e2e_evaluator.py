import hashlib
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    title: str
    prompt: str
    files: dict[str, str]
    protected_files: tuple[str, ...]
    hidden_check_code: str


@dataclass(frozen=True)
class HostCheckResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and self.timed_out is False


@dataclass(frozen=True)
class TaskEvaluationResult:
    task_id: str
    title: str
    agent_completed: bool
    initial_visible_failed: bool
    visible_tests_passed: bool
    hidden_checks_passed: bool
    protected_files_unchanged: bool
    verification_tool_calls: int
    verification_state_clean: bool
    temporary_workspace_cleaned: bool
    llm_calls: int
    tool_calls: int
    tool_call_counts: dict[str, int]
    changed_files: list[str]
    final_answer: str | None
    error: str | None


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    passed: int
    failed: int
    all_passed: bool


TASK_1 = CodingTask(
    task_id="task_01_bugfix",
    title="Single-file bug fix",
    prompt=(
        "Fix the bug in this project so that all tests pass.\n\n"
        "Requirements:\n"
        "1. Preserve the existing public function names.\n"
        "2. `subtract` must keep its correct subtraction behavior.\n"
        "3. Do not modify any test file.\n"
        "4. Inspect the project and make only the necessary source-code changes.\n"
        "5. Run the available tests as needed.\n"
        "6. Before giving your final answer, use the workspace verification tool "
        "and ensure verification passes."
    ),
    files={
        "calculator.py": (
            "def add(a, b):\n"
            "    return a - b\n\n\n"
            "def subtract(a, b):\n"
            "    return a - b\n"
        ),
        "test_calculator.py": (
            "from calculator import add, subtract\n\n\n"
            "def test_add_basic():\n"
            "    assert add(2, 3) == 5\n\n\n"
            "def test_subtract_basic():\n"
            "    assert subtract(8, 3) == 5\n"
        ),
    },
    protected_files=("test_calculator.py",),
    hidden_check_code=(
        "from calculator import add, subtract\n\n"
        "assert add(7, 11) == 18\n"
        "assert add(-4, 9) == 5\n"
        "assert add(0, 0) == 0\n\n"
        "assert subtract(9, 4) == 5\n"
        "assert subtract(-4, 9) == -13\n"
        "assert subtract(3, -2) == 5\n"
    ),
)


TASK_2 = CodingTask(
    task_id="task_02_feature",
    title="Feature implementation",
    prompt=(
        "Implement `normalize_words(text)` in this project.\n\n"
        "Required behavior:\n"
        "1. Ignore leading and trailing whitespace.\n"
        "2. Treat any consecutive whitespace as a single separator between words.\n"
        "3. Convert every returned word to lowercase.\n"
        "4. Return a list of words.\n"
        "5. Only whitespace is treated as a separator; punctuation inside a token "
        "should remain unchanged.\n"
        "6. Empty or whitespace-only input must return an empty list.\n"
        "7. Do not modify any test file.\n"
        "8. Run the available tests as needed.\n"
        "9. Before giving your final answer, use the workspace verification tool "
        "and ensure verification passes."
    ),
    files={
        "text_utils.py": (
            "def normalize_words(text: str) -> list[str]:\n"
            '    raise NotImplementedError("normalize_words is not implemented")\n'
        ),
        "test_text_utils.py": (
            "from text_utils import normalize_words\n\n\n"
            "def test_basic_normalization():\n"
            '    assert normalize_words("  Hello   WORLD  ") == '
            '["hello", "world"]\n\n\n'
            "def test_mixed_whitespace():\n"
            '    assert normalize_words("A\\tB\\nC") == ["a", "b", "c"]\n'
        ),
    },
    protected_files=("test_text_utils.py",),
    hidden_check_code=(
        "from text_utils import normalize_words\n\n"
        'assert normalize_words("") == []\n'
        'assert normalize_words("      ") == []\n'
        'assert normalize_words("  MiXeD   CaSe ") == ["mixed", "case"]\n'
        'assert normalize_words("one\\ttwo\\nthree") == '
        '["one", "two", "three"]\n'
        'assert normalize_words("  Hello,   WORLD! ") == '
        '["hello,", "world!"]\n'
        'assert normalize_words("single") == ["single"]\n'
    ),
)


TASK_3 = CodingTask(
    task_id="task_03_multifile",
    title="Multi-file user registration",
    prompt=(
        "Complete the user registration functionality in this project.\n\n"
        "Requirements:\n\n"
        "1. Implement `UserStore.exists(user_id)` in store.py.\n"
        "2. Implement `register_user(store, user_id, name)` in service.py.\n"
        "3. A user ID that already exists must be rejected.\n"
        "4. A name is invalid if it becomes empty after stripping surrounding "
        "whitespace.\n"
        "5. For a valid name, strip surrounding whitespace before storing it.\n"
        "6. A successful registration must store the normalized name and return "
        "True.\n"
        "7. A rejected registration must leave the store unchanged and return "
        "False.\n"
        "8. Preserve the existing public APIs.\n"
        "9. Do not modify any test file.\n"
        "10. Run the available tests as needed.\n"
        "11. Before giving your final answer, use the workspace verification tool "
        "and ensure verification passes."
    ),
    files={
        "store.py": (
            "class UserStore:\n"
            "    def __init__(self):\n"
            "        self.users = {}\n\n"
            "    def exists(self, user_id):\n"
            '        raise NotImplementedError("exists is not implemented")\n\n'
            "    def add(self, user_id, name):\n"
            "        self.users[user_id] = name\n"
        ),
        "service.py": (
            "from store import UserStore\n\n\n"
            "def register_user(store: UserStore, user_id, name):\n"
            '    raise NotImplementedError("register_user is not implemented")\n'
        ),
        "test_service.py": (
            "from service import register_user\n"
            "from store import UserStore\n\n\n"
            "def test_store_exists():\n"
            "    store = UserStore()\n"
            "    assert store.exists(1) is False\n"
            '    store.add(1, "Alice")\n'
            "    assert store.exists(1) is True\n\n\n"
            "def test_register_valid_user():\n"
            "    store = UserStore()\n\n"
            '    assert register_user(store, 1, "Alice") is True\n'
            '    assert store.users[1] == "Alice"\n\n\n'
            "def test_duplicate_user_is_rejected():\n"
            "    store = UserStore()\n\n"
            '    assert register_user(store, 1, "Alice") is True\n'
            '    assert register_user(store, 1, "Bob") is False\n'
            '    assert store.users[1] == "Alice"\n\n\n'
            "def test_blank_name_is_rejected():\n"
            "    store = UserStore()\n\n"
            '    assert register_user(store, 2, "   ") is False\n'
            "    assert store.exists(2) is False\n"
        ),
    },
    protected_files=("test_service.py",),
    hidden_check_code=(
        "from service import register_user\n"
        "from store import UserStore\n\n"
        "store = UserStore()\n\n"
        "assert store.exists(42) is False\n\n"
        'assert register_user(store, 42, "  Alice  ") is True\n'
        "assert store.exists(42) is True\n"
        'assert store.users[42] == "Alice"\n\n'
        'assert register_user(store, 42, "Bob") is False\n'
        'assert store.users[42] == "Alice"\n\n'
        'assert register_user(store, 99, "\\t \\n") is False\n'
        "assert store.exists(99) is False\n\n"
        'assert register_user(store, 7, "Carol") is True\n'
        "assert store.exists(7) is True\n"
        'assert store.users[7] == "Carol"\n'
    ),
)


TASKS = (TASK_1, TASK_2, TASK_3)

_IGNORED_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".pytest_tmp"}
)


def materialize_task(task: CodingTask, workspace: Path) -> None:
    workspace_root = Path(workspace)
    workspace_root.mkdir(parents=True, exist_ok=True)
    if any(workspace_root.iterdir()):
        raise ValueError("Task workspace must be empty before materialization.")
    if not set(task.protected_files) <= set(task.files):
        raise ValueError("Every protected file must be part of the task fixture.")

    for name, content in task.files.items():
        relative_path = Path(name)
        if (
            relative_path.is_absolute()
            or bool(relative_path.drive)
            or ".." in relative_path.parts
        ):
            raise ValueError(f"Invalid task fixture path: {name}")
        target = workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ignored_runtime_file(relative_path: Path) -> bool:
    return (
        relative_path.suffix == ".pyc"
        or any(
            part in _IGNORED_DIRECTORY_NAMES
            for part in relative_path.parts
        )
    )


def snapshot_workspace(workspace: Path) -> dict[str, str]:
    workspace_root = Path(workspace)
    snapshot: dict[str, str] = {}
    for path in sorted(workspace_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(workspace_root)
        if _ignored_runtime_file(relative_path):
            continue
        snapshot[relative_path.as_posix()] = sha256_file(path)
    return snapshot


def protected_files_unchanged(
    task: CodingTask,
    before: dict[str, str],
    after: dict[str, str],
) -> bool:
    return all(
        name in before
        and name in after
        and before[name] == after[name]
        for name in task.protected_files
    )


def changed_files(
    before: dict[str, str],
    after: dict[str, str],
) -> list[str]:
    return sorted(
        name
        for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_host_process(
    command: list[str],
    workspace: Path,
    timeout: int,
) -> HostCheckResult:
    try:
        completed = subprocess.run(
            command,
            cwd=Path(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        return HostCheckResult(
            exit_code=None,
            stdout=_output_text(error.stdout),
            stderr=_output_text(error.stderr),
            timed_out=True,
        )
    return HostCheckResult(
        exit_code=completed.returncode,
        stdout=_output_text(completed.stdout),
        stderr=_output_text(completed.stderr),
        timed_out=False,
    )


def run_visible_tests(
    workspace: Path,
    *,
    timeout: int = 30,
    basetemp: Path | None = None,
) -> HostCheckResult:
    workspace_root = Path(workspace).resolve(strict=False)
    command = [sys.executable, "-m", "pytest", "-q"]
    if basetemp is not None:
        basetemp_root = Path(basetemp).resolve(strict=False)
        if (
            basetemp_root == workspace_root
            or basetemp_root.is_relative_to(workspace_root)
            or workspace_root.is_relative_to(basetemp_root)
        ):
            raise ValueError(
                "Pytest basetemp must not overlap the task workspace."
            )
        command.append(f"--basetemp={basetemp_root}")
    return _run_host_process(command, workspace_root, timeout)


def run_hidden_check(
    task: CodingTask,
    workspace: Path,
    *,
    timeout: int = 10,
) -> HostCheckResult:
    return _run_host_process(
        [sys.executable, "-B", "-c", task.hidden_check_code],
        Path(workspace),
        timeout,
    )


def collect_tool_call_counts(
    history: Sequence[dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for message in history:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                counts[name] += 1
    return dict(sorted(counts.items()))


def task_passed(result: TaskEvaluationResult) -> bool:
    return (
        result.agent_completed
        and result.initial_visible_failed
        and result.visible_tests_passed
        and result.hidden_checks_passed
        and result.protected_files_unchanged
        and result.verification_tool_calls >= 1
        and result.verification_state_clean
        and result.temporary_workspace_cleaned
    )


def summarize_results(
    results: Sequence[TaskEvaluationResult],
) -> EvaluationSummary:
    total = len(results)
    passed = sum(task_passed(result) for result in results)
    failed = total - passed
    return EvaluationSummary(
        total=total,
        passed=passed,
        failed=failed,
        all_passed=total > 0 and failed == 0,
    )
