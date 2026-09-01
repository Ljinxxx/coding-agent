import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import scripts.long_running_evidence as evidence_module
import scripts.verify_long_running_challenge_real as runner_module
from scripts.long_running_evidence import (
    EvidenceResult,
    WorkspaceChanges,
    build_file_manifest,
    compare_manifests,
    create_evidence_session,
    evidence_paths_for_report,
    generate_workspace_diff,
    real_validation_process_success,
    snapshot_workspace,
    write_manifest,
    write_workspace_changes,
)


def _write_bytes(root: Path, relative_path: str, content: bytes) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _manifest_entry(content: bytes) -> dict[str, str | int]:
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def test_create_evidence_session_uses_repo_local_collision_safe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_module,
        "_new_run_id",
        lambda: "20260831_153501_a1b2c3d4",
    )

    paths = create_evidence_session(tmp_path)

    assert paths.run_id == "20260831_153501_a1b2c3d4"
    assert paths.root == (
        tmp_path
        / ".runs"
        / "long_running_challenge"
        / "20260831_153501_a1b2c3d4"
    )
    assert paths.initial_workspace == paths.root / "initial_workspace"
    assert paths.final_workspace == paths.root / "final_workspace"
    assert paths.before_manifest == paths.root / "file_manifest_before.json"
    assert paths.after_manifest == paths.root / "file_manifest_after.json"
    assert paths.changes_json == paths.root / "workspace_changes.json"
    assert paths.diff_file == paths.root / "workspace_changes.diff"

    sentinel = paths.root / "sentinel.txt"
    sentinel.write_text("do-not-overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_evidence_session(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "do-not-overwrite"


def test_workspace_snapshot_preserves_repository_files_and_excludes_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    destination = tmp_path / "evidence" / "initial_workspace"
    _write_bytes(workspace, "a.py", b"print('ok')\n")
    _write_bytes(workspace, "dir/b.txt", b"payload\n")
    _write_bytes(workspace, "tests/test_x.py", b"def test_x(): pass\n")
    _write_bytes(workspace, ".gitignore", b".env\n")
    _write_bytes(workspace, ".hidden", b"keep\n")
    _write_bytes(workspace, ".cache/value", b"keep\n")
    for name in (
        "__pycache__/x.pyc",
        ".pytest_cache/cache",
        ".pytest_tmp/temp",
        ".git/config",
        ".runs/private/artifact",
        "htmlcov/index.html",
        "nested/module.pyc",
        "nested/module.pyo",
        ".coverage",
        "coverage.xml",
        ".env",
        ".env.local",
    ):
        _write_bytes(workspace, name, b"excluded")
    before = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }

    snapshot_workspace(workspace, destination)

    assert (destination / "a.py").read_bytes() == b"print('ok')\n"
    assert (destination / "dir/b.txt").read_bytes() == b"payload\n"
    assert (destination / "tests/test_x.py").is_file()
    assert (destination / ".gitignore").is_file()
    assert (destination / ".hidden").is_file()
    assert (destination / ".cache/value").is_file()
    assert not (destination / "__pycache__").exists()
    assert not (destination / ".pytest_cache").exists()
    assert not (destination / ".pytest_tmp").exists()
    assert not (destination / ".git").exists()
    assert not (destination / ".runs").exists()
    assert not (destination / "htmlcov").exists()
    assert not (destination / "nested/module.pyc").exists()
    assert not (destination / "nested/module.pyo").exists()
    assert not (destination / ".coverage").exists()
    assert not (destination / "coverage.xml").exists()
    assert not (destination / ".env").exists()
    assert not (destination / ".env.local").exists()
    assert {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    } == before

    with pytest.raises(ValueError, match="inside the source workspace"):
        snapshot_workspace(workspace, workspace / "nested-evidence")


def test_file_manifest_is_sorted_binary_safe_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_bytes(snapshot, "z.txt", b"same\n")
    _write_bytes(snapshot, "nested/alpha.txt", "你好\n".encode())
    _write_bytes(snapshot, "binary.bin", b"\xff\x00\x80")

    first = build_file_manifest(snapshot)
    second = build_file_manifest(snapshot)

    assert list(first) == ["binary.bin", "nested/alpha.txt", "z.txt"]
    assert first == second
    assert first["binary.bin"] == _manifest_entry(b"\xff\x00\x80")
    assert first["nested/alpha.txt"] == _manifest_entry("你好\n".encode())

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_manifest(first_path, first)
    write_manifest(second_path, dict(reversed(list(first.items()))))
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    assert json.loads(first_path.read_text(encoding="utf-8")) == first


def test_manifest_comparison_classifies_all_changes_and_protected_files(
    tmp_path: Path,
) -> None:
    before = {
        "deleted.txt": _manifest_entry(b"deleted"),
        "modified.txt": _manifest_entry(b"before"),
        "protected/deleted.txt": _manifest_entry(b"protected deleted"),
        "protected/modified.txt": _manifest_entry(b"protected before"),
        "protected/stable.txt": _manifest_entry(b"stable"),
        "renamed-old.txt": _manifest_entry(b"renamed"),
        "unchanged.txt": _manifest_entry(b"same"),
    }
    after = {
        "created.txt": _manifest_entry(b"created"),
        "modified.txt": _manifest_entry(b"after"),
        "protected/modified.txt": _manifest_entry(b"protected after"),
        "protected/stable.txt": _manifest_entry(b"stable"),
        "renamed-new.txt": _manifest_entry(b"renamed"),
        "unchanged.txt": _manifest_entry(b"same"),
    }

    changes = compare_manifests(
        before,
        after,
        protected_files=(
            "protected/stable.txt",
            "protected/modified.txt",
            "protected/deleted.txt",
        ),
    )

    assert changes.created == ("created.txt", "renamed-new.txt")
    assert changes.modified == (
        "modified.txt",
        "protected/modified.txt",
    )
    assert changes.deleted == (
        "deleted.txt",
        "protected/deleted.txt",
        "renamed-old.txt",
    )
    assert changes.unchanged == (
        "protected/stable.txt",
        "unchanged.txt",
    )
    assert changes.protected_unchanged == ("protected/stable.txt",)
    assert changes.protected_modified == ("protected/modified.txt",)
    assert changes.protected_deleted == ("protected/deleted.txt",)
    assert changes.counts == {
        "created": 2,
        "modified": 2,
        "deleted": 3,
        "unchanged": 2,
    }

    changes_path = tmp_path / "workspace_changes.json"
    write_workspace_changes(changes_path, changes)
    payload = json.loads(changes_path.read_text(encoding="utf-8"))
    assert payload["counts"] == changes.counts
    assert payload["protected_modified"] == ["protected/modified.txt"]
    assert changes_path.read_bytes().endswith(b"\n")


def test_workspace_diff_is_deterministic_for_created_modified_and_deleted(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_bytes(before, "deleted.txt", b"remove me\n")
    _write_bytes(before, "empty-deleted.txt", b"")
    _write_bytes(before, "line-endings.txt", b"same\r\n")
    _write_bytes(before, "modified.txt", b"old value\n")
    _write_bytes(before, "unchanged.txt", b"same\n")
    _write_bytes(after, "created.txt", b"new file\n")
    _write_bytes(after, "empty-created.txt", b"")
    _write_bytes(after, "line-endings.txt", b"same\n")
    _write_bytes(after, "modified.txt", b"new value\n")
    _write_bytes(after, "unchanged.txt", b"same\n")
    changes = compare_manifests(
        build_file_manifest(before),
        build_file_manifest(after),
    )

    first = generate_workspace_diff(before, after, changes)
    second = generate_workspace_diff(before, after, changes)

    assert first == second
    assert first.index("final/created.txt") < first.index("initial/deleted.txt")
    assert first.index("initial/deleted.txt") < first.index(
        "initial/modified.txt"
    )
    assert "--- /dev/null\n+++ final/created.txt\n" in first
    assert "--- initial/deleted.txt\n+++ /dev/null\n" in first
    assert "--- /dev/null\n+++ final/empty-created.txt\n" in first
    assert "--- initial/empty-deleted.txt\n+++ /dev/null\n" in first
    assert "--- initial/modified.txt\n+++ final/modified.txt\n" in first
    assert "+new file" in first
    assert "-remove me" in first
    assert "-old value" in first and "+new value" in first
    assert (
        "Text file bytes changed without a line-content diff: line-endings.txt"
        in first
    )
    assert "unchanged.txt" not in first
    assert str(before) not in first and str(after) not in first


def test_workspace_diff_marks_non_utf8_changes_without_decode_failure(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_bytes(before, "binary.bin", b"\xff\x00before")
    _write_bytes(after, "binary.bin", b"\xff\x00after")
    _write_bytes(after, "created.bin", b"\xfe\x00new")
    changes = compare_manifests(
        build_file_manifest(before),
        build_file_manifest(after),
    )

    diff = generate_workspace_diff(before, after, changes)

    assert "Binary or non-UTF8 file changed: binary.bin" in diff
    assert "Binary or non-UTF8 file changed: created.bin" in diff
    assert hashlib.sha256(b"\xff\x00before").hexdigest() in diff
    assert hashlib.sha256(b"\xff\x00after").hexdigest() in diff
    assert "\ufffd" not in diff
    assert str(before) not in diff and str(after) not in diff


def test_evidence_artifacts_survive_temporary_workspace_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = create_evidence_session(tmp_path)
    result = EvidenceResult(paths=paths)

    with TemporaryDirectory(prefix="evidence-workspace-") as directory:
        workspace = Path(directory) / "workspace"
        _write_bytes(workspace, "original.txt", b"before\n")
        before = runner_module.preserve_initial_evidence(workspace, result)
        _write_bytes(workspace, "original.txt", b"after\n")
        _write_bytes(workspace, "created.txt", b"created\n")
        temporary_workspace = workspace
        runner_module.preserve_final_evidence(
            workspace,
            result,
            before,
            (),
            agent_completed=False,
            snapshot_stage="failure_state",
        )

    assert not temporary_workspace.exists()
    result.snapshot_preserved_after_cleanup = (
        runner_module._evidence_artifacts_preserved(result)
    )
    assert result.preservation_passed is True
    assert result.agent_completed is False
    assert result.snapshot_stage == "failure_state"
    assert (paths.initial_workspace / "original.txt").read_bytes() == b"before\n"
    assert (paths.final_workspace / "original.txt").read_bytes() == b"after\n"
    for artifact in (
        paths.before_manifest,
        paths.after_manifest,
        paths.changes_json,
        paths.diff_file,
    ):
        assert artifact.is_file()
        assert str(temporary_workspace) not in artifact.read_text(
            encoding="utf-8"
        )

    failed_paths = create_evidence_session(tmp_path)
    failed_result = EvidenceResult(paths=failed_paths)
    failed_workspace = tmp_path / "failed-workspace"
    _write_bytes(failed_workspace, "value.txt", b"before\n")
    failed_before = runner_module.preserve_initial_evidence(
        failed_workspace,
        failed_result,
    )
    _write_bytes(failed_workspace, "value.txt", b"after\n")

    def fail_diff_write(path: Path, diff: str) -> None:
        raise OSError("simulated diff write failure")

    monkeypatch.setattr(runner_module, "write_workspace_diff", fail_diff_write)
    with pytest.raises(OSError, match="simulated diff write failure"):
        runner_module.preserve_final_evidence(
            failed_workspace,
            failed_result,
            failed_before,
            (),
            agent_completed=True,
            snapshot_stage="agent_final",
        )
    assert failed_paths.root.is_dir()
    assert failed_paths.initial_workspace.is_dir()
    assert failed_paths.final_workspace.is_dir()
    assert failed_paths.before_manifest.is_file()
    assert failed_paths.after_manifest.is_file()
    assert failed_paths.changes_json.is_file()
    assert not failed_paths.diff_file.exists()


def test_report_evidence_metadata_is_relative_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_module,
        "_new_run_id",
        lambda: "20260831_153501_1234abcd",
    )
    paths = create_evidence_session(tmp_path)
    secret = "fake-api-secret-evidence"
    result = EvidenceResult(
        paths=paths,
        agent_completed=True,
        snapshot_stage="agent_final",
        initial_snapshot_created=True,
        final_snapshot_created=True,
        before_manifest_created=True,
        after_manifest_created=True,
        changes_json_created=True,
        diff_created=True,
        snapshot_preserved_after_cleanup=True,
        changes=WorkspaceChanges(created=("incident/report.py",)),
    )
    tokens = runner_module.ChallengeTokens(secret, "m", "h", "t")
    runner_module._record_evidence_error(
        result,
        RuntimeError(
            f"redact {secret} at {tmp_path} under {runner_module.REPO_ROOT}"
        ),
        temporary_root=tmp_path,
        tokens=tokens,
    )
    metadata = result.to_report(tmp_path)
    assert metadata["directory"] == (
        ".runs/long_running_challenge/20260831_153501_1234abcd"
    )
    assert metadata["initial_snapshot"] == "initial_workspace"
    assert metadata["final_snapshot"] == "final_workspace"
    serialized = json.dumps(
        runner_module.redact_report_sensitive_values(
            {"evidence": metadata},
            tokens,
        )
    )
    assert secret not in serialized
    assert str(tmp_path) not in serialized

    escaped = replace(paths, diff_file=tmp_path.parent / "outside.diff")
    with pytest.raises(ValueError, match="outside evidence directory"):
        evidence_paths_for_report(escaped, tmp_path)


@pytest.mark.parametrize(
    (
        "functional_passed",
        "coverage_passed",
        "evidence_passed",
        "expected_integrated",
        "expected_process",
    ),
    [
        (True, True, True, True, True),
        (True, True, False, True, False),
        (False, True, True, False, False),
        (True, False, True, False, False),
    ],
)
def test_evidence_result_is_separate_from_functional_result(
    functional_passed: bool,
    coverage_passed: bool,
    evidence_passed: bool,
    expected_integrated: bool,
    expected_process: bool,
) -> None:
    integrated = runner_module.final_integrated_success(
        functional_passed=functional_passed,
        coverage_passed=coverage_passed,
    )
    assert integrated is expected_integrated
    assert (
        real_validation_process_success(
            final_integrated_success=integrated,
            evidence_preservation_passed=evidence_passed,
        )
        is expected_process
    )


def test_runner_captures_evidence_at_host_only_boundaries() -> None:
    source = inspect.getsource(runner_module.execute_real_challenge)
    initial = source.index("preserve_initial_evidence(")
    initial_tests = source.index("run_visible_tests(")
    recording = source.index("RequestRecordingLLM(")
    final = source.index("preserve_final_evidence(")
    host_visible = source.index('phase = "Host visible re-check"')
    host_hidden = source.index('phase = "Host hidden evaluation"')

    assert initial < initial_tests < recording
    assert final < host_visible < host_hidden
    assert "agent.run(" not in inspect.getsource(
        runner_module.preserve_initial_evidence
    )
    assert "agent.run(" not in inspect.getsource(
        runner_module.preserve_final_evidence
    )
