from dataclasses import replace
from pathlib import Path

from scripts.e2e_evaluator import (
    TASKS,
    TaskEvaluationResult,
    changed_files,
    collect_tool_call_counts,
    materialize_task,
    protected_files_unchanged,
    run_hidden_check,
    run_visible_tests,
    sha256_file,
    snapshot_workspace,
    summarize_results,
    task_passed,
)


def _task(task_id: str):
    return next(task for task in TASKS if task.task_id == task_id)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _passing_result(task_id: str = "task-pass") -> TaskEvaluationResult:
    return TaskEvaluationResult(
        task_id=task_id,
        title="Passing task",
        agent_completed=True,
        initial_visible_failed=True,
        visible_tests_passed=True,
        hidden_checks_passed=True,
        protected_files_unchanged=True,
        verification_tool_calls=1,
        verification_state_clean=True,
        temporary_workspace_cleaned=True,
        llm_calls=4,
        tool_calls=6,
        tool_call_counts={"verify_workspace": 1},
        changed_files=["source.py"],
        final_answer="done",
        error=None,
    )


def test_task_fixtures_are_created_deterministically_and_isolated(
    tmp_path: Path,
) -> None:
    assert [task.task_id for task in TASKS] == [
        "task_01_bugfix",
        "task_02_feature",
        "task_03_multifile",
    ]
    assert _task("task_01_bugfix").files == {
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
    }
    assert _task("task_02_feature").files == {
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
    }
    assert _task("task_03_multifile").files == {
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
    }
    assert {
        task.task_id: task.protected_files for task in TASKS
    } == {
        "task_01_bugfix": ("test_calculator.py",),
        "task_02_feature": ("test_text_utils.py",),
        "task_03_multifile": ("test_service.py",),
    }

    workspaces: dict[str, Path] = {}
    initial_snapshots: dict[str, dict[str, str]] = {}
    for task in TASKS:
        workspace = tmp_path / task.task_id / "workspace"
        materialize_task(task, workspace)
        workspaces[task.task_id] = workspace

        relative_files = {
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file()
        }
        assert relative_files == set(task.files)
        assert set(task.protected_files) <= relative_files
        assert all(
            (workspace / name).read_text(encoding="utf-8") == content
            for name, content in task.files.items()
        )

        initial_snapshots[task.task_id] = snapshot_workspace(workspace)
        replica = tmp_path / f"{task.task_id}-replica" / "workspace"
        materialize_task(task, replica)
        assert snapshot_workspace(replica) == initial_snapshots[task.task_id]

        initial_visible = run_visible_tests(
            workspace,
            basetemp=tmp_path / f"{task.task_id}-initial-pytest",
        )
        assert initial_visible.timed_out is False
        assert initial_visible.exit_code not in (None, 0)
        assert initial_visible.passed is False

    task_1 = _task("task_01_bugfix")
    first = workspaces[task_1.task_id]
    replica = tmp_path / f"{task_1.task_id}-replica" / "workspace"
    _write(first / "calculator.py", "changed only in the first workspace\n")
    assert (replica / "calculator.py").read_text(encoding="utf-8") == (
        task_1.files["calculator.py"]
    )


def test_protected_file_integrity_detects_modification_and_deletion(
    tmp_path: Path,
) -> None:
    task = _task("task_01_bugfix")
    workspace = tmp_path / "workspace"
    materialize_task(task, workspace)
    before = snapshot_workspace(workspace)

    protected_name = task.protected_files[0]
    protected_path = workspace / protected_name
    assert before[protected_name] == sha256_file(protected_path)
    assert protected_files_unchanged(task, before, before) is True

    _write(workspace / "calculator.py", "def add(a, b):\n    return a + b\n")
    source_changed = snapshot_workspace(workspace)
    assert protected_files_unchanged(task, before, source_changed) is True

    _write(protected_path, "def test_tampered():\n    assert True\n")
    protected_changed = snapshot_workspace(workspace)
    assert protected_files_unchanged(task, before, protected_changed) is False

    _write(protected_path, task.files[protected_name])
    restored = snapshot_workspace(workspace)
    assert protected_files_unchanged(task, before, restored) is True

    protected_path.unlink()
    after_deletion = snapshot_workspace(workspace)
    assert protected_files_unchanged(task, before, after_deletion) is False
    assert changed_files(before, after_deletion) == [
        "calculator.py",
        "test_calculator.py",
    ]


def test_task1_hidden_evaluator_rejects_hardcoded_visible_case(
    tmp_path: Path,
) -> None:
    task = _task("task_01_bugfix")
    workspace = tmp_path / "workspace"
    materialize_task(task, workspace)
    before = snapshot_workspace(workspace)

    _write(
        workspace / "calculator.py",
        "def add(a, b):\n"
        "    if a == 2 and b == 3:\n"
        "        return 5\n"
        "    return 0\n\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n",
    )
    visible = run_visible_tests(
        workspace,
        basetemp=tmp_path / "task1-visible-pytest",
    )
    hidden = run_hidden_check(task, workspace)
    assert visible.passed is True
    assert hidden.timed_out is False
    assert hidden.exit_code not in (None, 0)
    assert hidden.passed is False

    _write(
        workspace / "calculator.py",
        "def add(a, b):\n"
        "    return a + b\n\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n",
    )
    assert run_hidden_check(task, workspace).passed is True
    assert changed_files(before, snapshot_workspace(workspace)) == [
        "calculator.py"
    ]


def test_task2_hidden_evaluator_covers_unseen_normalization_cases(
    tmp_path: Path,
) -> None:
    task = _task("task_02_feature")
    workspace = tmp_path / "workspace"
    materialize_task(task, workspace)
    before = snapshot_workspace(workspace)

    _write(
        workspace / "text_utils.py",
        "def normalize_words(text: str) -> list[str]:\n"
        "    visible_cases = {\n"
        '        "  Hello   WORLD  ": ["hello", "world"],\n'
        '        "A\\tB\\nC": ["a", "b", "c"],\n'
        "    }\n"
        "    return visible_cases.get(text, [])\n",
    )
    visible = run_visible_tests(
        workspace,
        basetemp=tmp_path / "task2-visible-pytest",
    )
    hidden = run_hidden_check(task, workspace)
    assert visible.passed is True
    assert hidden.timed_out is False
    assert hidden.passed is False

    _write(
        workspace / "text_utils.py",
        "def normalize_words(text: str) -> list[str]:\n"
        "    return text.lower().split()\n",
    )
    assert run_hidden_check(task, workspace).passed is True
    assert changed_files(before, snapshot_workspace(workspace)) == [
        "text_utils.py"
    ]


def test_task3_hidden_evaluator_checks_multifile_registration_contract(
    tmp_path: Path,
) -> None:
    task = _task("task_03_multifile")
    workspace = tmp_path / "workspace"
    materialize_task(task, workspace)
    before = snapshot_workspace(workspace)

    correct_exists = (
        "class UserStore:\n"
        "    def __init__(self):\n"
        "        self.users = {}\n\n"
        "    def exists(self, user_id):\n"
        "        return user_id in self.users\n\n"
        "    def add(self, user_id, name):\n"
        "        self.users[user_id] = name\n"
    )
    valid_service = (
        "from store import UserStore\n\n\n"
        "def register_user(store: UserStore, user_id, name):\n"
        "    normalized = name.strip()\n"
        "    if not normalized or store.exists(user_id):\n"
        "        return False\n"
        "    store.add(user_id, normalized)\n"
        "    return True\n"
    )
    flawed_variants = [
        (
            "exists is always false",
            correct_exists.replace(
                "return user_id in self.users",
                "return False",
            ),
            valid_service,
        ),
        (
            "duplicates overwrite",
            correct_exists,
            "from store import UserStore\n\n\n"
            "def register_user(store: UserStore, user_id, name):\n"
            "    store.add(user_id, name.strip())\n"
            "    return True\n",
        ),
        (
            "blank names are accepted",
            correct_exists,
            "from store import UserStore\n\n\n"
            "def register_user(store: UserStore, user_id, name):\n"
            "    if store.exists(user_id):\n"
            "        return False\n"
            "    store.add(user_id, name.strip())\n"
            "    return True\n",
        ),
        (
            "valid names are not stripped",
            correct_exists,
            "from store import UserStore\n\n\n"
            "def register_user(store: UserStore, user_id, name):\n"
            "    if store.exists(user_id) or not name.strip():\n"
            "        return False\n"
            "    store.add(user_id, name)\n"
            "    return True\n",
        ),
    ]

    for label, store_source, service_source in flawed_variants:
        _write(workspace / "store.py", store_source)
        _write(workspace / "service.py", service_source)
        assert run_hidden_check(task, workspace).passed is False, label

    _write(workspace / "store.py", correct_exists)
    _write(workspace / "service.py", valid_service)
    assert run_hidden_check(task, workspace).passed is True
    assert changed_files(before, snapshot_workspace(workspace)) == [
        "service.py",
        "store.py",
    ]


def test_evaluation_result_aggregation_requires_all_objective_checks() -> None:
    passing = _passing_result()
    assert task_passed(passing) is True
    assert task_passed(replace(passing, verification_tool_calls=2)) is True

    failing_results = [
        replace(passing, task_id="agent", agent_completed=False),
        replace(passing, task_id="initial", initial_visible_failed=False),
        replace(passing, task_id="visible", visible_tests_passed=False),
        replace(passing, task_id="hidden", hidden_checks_passed=False),
        replace(
            passing,
            task_id="protected",
            protected_files_unchanged=False,
        ),
        replace(passing, task_id="verify", verification_tool_calls=0),
        replace(
            passing,
            task_id="state",
            verification_state_clean=False,
        ),
        replace(
            passing,
            task_id="cleanup",
            temporary_workspace_cleaned=False,
        ),
    ]
    assert all(task_passed(result) is False for result in failing_results)

    mixed = summarize_results([passing, *failing_results])
    assert (mixed.total, mixed.passed, mixed.failed, mixed.all_passed) == (
        9,
        1,
        8,
        False,
    )
    all_passing = summarize_results(
        [replace(passing, task_id=f"pass-{index}") for index in range(3)]
    )
    assert (all_passing.total, all_passing.passed, all_passing.failed) == (
        3,
        3,
        0,
    )
    assert all_passing.all_passed is True
    assert summarize_results([]).all_passed is False

    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "verify_workspace"}},
            ],
        },
        {"role": "tool", "tool_call_id": "ignored", "content": "ok"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "verify_workspace"}},
            ],
        },
    ]
    assert collect_tool_call_counts(history) == {
        "read_file": 1,
        "verify_workspace": 2,
    }
