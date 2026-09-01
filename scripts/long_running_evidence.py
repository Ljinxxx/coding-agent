import difflib
import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


_EVIDENCE_PARENT = Path(".runs") / "long_running_challenge"
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".pytest_tmp",
        ".runs",
        "__pycache__",
        "htmlcov",
    }
)
_EXCLUDED_FILE_NAMES = frozenset({".coverage", "coverage.xml"})
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
_HASH_CHUNK_SIZE = 1024 * 1024


ManifestEntry = dict[str, str | int]
FileManifest = dict[str, ManifestEntry]


@dataclass(frozen=True)
class EvidencePaths:
    run_id: str
    root: Path
    initial_workspace: Path
    final_workspace: Path
    before_manifest: Path
    after_manifest: Path
    changes_json: Path
    diff_file: Path


@dataclass(frozen=True)
class WorkspaceChanges:
    created: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    protected_unchanged: tuple[str, ...] = ()
    protected_modified: tuple[str, ...] = ()
    protected_deleted: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {
            "created": len(self.created),
            "modified": len(self.modified),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
        }

    @property
    def protected_files_changed(self) -> int:
        return len(self.protected_modified) + len(self.protected_deleted)

    def to_payload(self) -> dict[str, Any]:
        return {
            "created": list(self.created),
            "modified": list(self.modified),
            "deleted": list(self.deleted),
            "unchanged": list(self.unchanged),
            "protected_unchanged": list(self.protected_unchanged),
            "protected_modified": list(self.protected_modified),
            "protected_deleted": list(self.protected_deleted),
            "counts": self.counts,
        }


@dataclass
class EvidenceResult:
    paths: EvidencePaths | None = None
    agent_completed: bool = False
    snapshot_stage: str = "not_started"
    initial_snapshot_created: bool = False
    final_snapshot_created: bool = False
    before_manifest_created: bool = False
    after_manifest_created: bool = False
    changes_json_created: bool = False
    diff_created: bool = False
    snapshot_preserved_after_cleanup: bool = False
    changes: WorkspaceChanges = field(default_factory=WorkspaceChanges)
    preservation_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None

    @property
    def preservation_passed(self) -> bool:
        return bool(
            self.paths is not None
            and self.initial_snapshot_created
            and self.final_snapshot_created
            and self.before_manifest_created
            and self.after_manifest_created
            and self.changes_json_created
            and self.diff_created
            and self.snapshot_preserved_after_cleanup
            and self.error_type is None
        )

    def record_error(self, error: Exception, *, message: str | None = None) -> None:
        if self.error_type is None:
            self.error_type = type(error).__name__
            self.error_message = message or str(error) or "No error message."

    def to_report(self, repo_root: Path) -> dict[str, Any]:
        path_metadata = (
            evidence_paths_for_report(self.paths, repo_root)
            if self.paths is not None
            else {
                "run_id": None,
                "directory": None,
                "initial_snapshot": None,
                "final_snapshot": None,
                "before_manifest": None,
                "after_manifest": None,
                "changes": None,
                "diff": None,
            }
        )
        return {
            "preservation_passed": self.preservation_passed,
            **path_metadata,
            "agent_completed": self.agent_completed,
            "snapshot_stage": self.snapshot_stage,
            "initial_snapshot_created": self.initial_snapshot_created,
            "final_snapshot_created": self.final_snapshot_created,
            "before_manifest_created": self.before_manifest_created,
            "after_manifest_created": self.after_manifest_created,
            "changes_json_created": self.changes_json_created,
            "diff_created": self.diff_created,
            "snapshot_preserved_after_cleanup": (
                self.snapshot_preserved_after_cleanup
            ),
            "created_files": list(self.changes.created),
            "modified_files": list(self.changes.modified),
            "deleted_files": list(self.changes.deleted),
            "unchanged_files": list(self.changes.unchanged),
            "protected_unchanged": list(self.changes.protected_unchanged),
            "protected_modified": list(self.changes.protected_modified),
            "protected_deleted": list(self.changes.protected_deleted),
            "created_count": len(self.changes.created),
            "modified_count": len(self.changes.modified),
            "deleted_count": len(self.changes.deleted),
            "unchanged_count": len(self.changes.unchanged),
            "protected_files_changed": (
                self.changes.protected_files_changed
            ),
            "preservation_seconds": self.preservation_seconds,
            "error": (
                {
                    "error_type": self.error_type,
                    "message": self.error_message,
                }
                if self.error_type is not None
                else None
            ),
        }


def _new_run_id() -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{secrets.token_hex(4)}"


def create_evidence_session(repo_root: Path) -> EvidencePaths:
    repository = Path(repo_root).resolve(strict=False)
    run_id = _new_run_id()
    root = repository / _EVIDENCE_PARENT / run_id
    if not root.resolve(strict=False).is_relative_to(repository):
        raise ValueError("Evidence directory must remain inside the repository.")
    root.mkdir(parents=True, exist_ok=False)
    return EvidencePaths(
        run_id=run_id,
        root=root,
        initial_workspace=root / "initial_workspace",
        final_workspace=root / "final_workspace",
        before_manifest=root / "file_manifest_before.json",
        after_manifest=root / "file_manifest_after.json",
        changes_json=root / "workspace_changes.json",
        diff_file=root / "workspace_changes.diff",
    )


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _excluded_name(name: str, *, is_directory: bool) -> bool:
    normalized = name.casefold()
    if is_directory and normalized in _EXCLUDED_DIRECTORY_NAMES:
        return True
    if normalized in _EXCLUDED_FILE_NAMES:
        return True
    if normalized == ".env" or normalized.startswith(".env."):
        return True
    return Path(normalized).suffix in _EXCLUDED_SUFFIXES


def _snapshot_ignore(directory: str, names: list[str]) -> set[str]:
    current = Path(directory)
    ignored: set[str] = set()
    for name in names:
        candidate = current / name
        if _is_link_like(candidate) or _excluded_name(
            name,
            is_directory=candidate.is_dir(),
        ):
            ignored.add(name)
    return ignored


def snapshot_workspace(source_workspace: Path, destination: Path) -> None:
    source_input = Path(source_workspace)
    if _is_link_like(source_input):
        raise ValueError("Source workspace must not be a symlink or junction.")
    source = source_input.resolve(strict=True)
    target = Path(destination).resolve(strict=False)
    if not source.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {source}")
    if target == source or target.is_relative_to(source):
        raise ValueError("Snapshot destination must not be inside the source workspace.")
    if target.exists():
        raise FileExistsError(f"Snapshot destination already exists: {target}")
    shutil.copytree(
        source,
        target,
        ignore=_snapshot_ignore,
        copy_function=shutil.copy2,
        symlinks=True,
    )


def _normalize_relative_path(value: str) -> str:
    text = str(value).replace("\\", "/")
    posix_path = PurePosixPath(text)
    windows_path = PureWindowsPath(str(value))
    if (
        not text
        or posix_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or posix_path.as_posix() == "."
    ):
        raise ValueError(f"Expected a relative workspace path: {value}")
    return posix_path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(snapshot_root: Path) -> FileManifest:
    root = Path(snapshot_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Snapshot is not a directory: {root}")
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not _is_link_like(path)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    manifest: FileManifest = {}
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        manifest[relative_path] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return manifest


def _atomic_write_text(path: Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(4)}.tmp"
    )
    try:
        temporary.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_manifest(path: Path, manifest: Mapping[str, Mapping[str, Any]]) -> None:
    normalized = {
        _normalize_relative_path(name): {
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        }
        for name, entry in sorted(manifest.items())
    }
    _write_json(path, normalized)


def _normalized_manifest(
    manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    normalized: dict[str, Mapping[str, Any]] = {}
    for name, entry in manifest.items():
        path = _normalize_relative_path(name)
        if path in normalized:
            raise ValueError(f"Duplicate normalized manifest path: {path}")
        if not isinstance(entry.get("sha256"), str):
            raise ValueError(f"Manifest entry has no SHA-256 value: {path}")
        normalized[path] = entry
    return normalized


def compare_manifests(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    protected_files: Sequence[str] = (),
) -> WorkspaceChanges:
    before_normalized = _normalized_manifest(before)
    after_normalized = _normalized_manifest(after)
    before_paths = set(before_normalized)
    after_paths = set(after_normalized)
    created = tuple(sorted(after_paths - before_paths))
    deleted = tuple(sorted(before_paths - after_paths))
    common = before_paths & after_paths
    modified = tuple(
        sorted(
            path
            for path in common
            if before_normalized[path]["sha256"]
            != after_normalized[path]["sha256"]
        )
    )
    unchanged = tuple(sorted(common - set(modified)))
    protected = tuple(
        sorted(_normalize_relative_path(path) for path in protected_files)
    )
    missing_protected = sorted(set(protected) - before_paths)
    if missing_protected:
        raise ValueError(
            "Protected files are missing from the before manifest: "
            + ", ".join(missing_protected)
        )
    return WorkspaceChanges(
        created=created,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        protected_unchanged=tuple(
            path for path in protected if path in unchanged
        ),
        protected_modified=tuple(path for path in protected if path in modified),
        protected_deleted=tuple(path for path in protected if path in deleted),
    )


def write_workspace_changes(path: Path, changes: WorkspaceChanges) -> None:
    _write_json(path, changes.to_payload())


def _text_lines(content: bytes) -> list[str] | None:
    if b"\x00" in content:
        return None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        line if line.endswith("\n") else line + "\n"
        for line in normalized.splitlines(keepends=True)
    ]


def _binary_diff_marker(
    relative_path: str,
    before_content: bytes,
    after_content: bytes,
    *,
    before_exists: bool,
    after_exists: bool,
) -> str:
    before_hash = (
        hashlib.sha256(before_content).hexdigest()
        if before_exists
        else "none"
    )
    after_hash = (
        hashlib.sha256(after_content).hexdigest() if after_exists else "none"
    )
    return (
        f"Binary or non-UTF8 file changed: {relative_path}\n"
        f"before_sha256={before_hash} before_size={len(before_content)}\n"
        f"after_sha256={after_hash} after_size={len(after_content)}\n"
    )


def _text_byte_change_marker(
    relative_path: str,
    before_content: bytes,
    after_content: bytes,
) -> str:
    return (
        f"Text file bytes changed without a line-content diff: {relative_path}\n"
        f"before_sha256={hashlib.sha256(before_content).hexdigest()} "
        f"before_size={len(before_content)}\n"
        f"after_sha256={hashlib.sha256(after_content).hexdigest()} "
        f"after_size={len(after_content)}\n"
    )


def generate_workspace_diff(
    initial_workspace: Path,
    final_workspace: Path,
    changes: WorkspaceChanges,
) -> str:
    before_root = Path(initial_workspace).resolve(strict=True)
    after_root = Path(final_workspace).resolve(strict=True)
    changed_paths = sorted(
        set(changes.created) | set(changes.modified) | set(changes.deleted)
    )
    sections: list[str] = []
    for relative_path in changed_paths:
        normalized_path = _normalize_relative_path(relative_path)
        before_path = before_root / Path(normalized_path)
        after_path = after_root / Path(normalized_path)
        before_exists = before_path.is_file()
        after_exists = after_path.is_file()
        before_content = before_path.read_bytes() if before_exists else b""
        after_content = after_path.read_bytes() if after_exists else b""
        before_lines = _text_lines(before_content)
        after_lines = _text_lines(after_content)
        if before_lines is None or after_lines is None:
            sections.append(
                _binary_diff_marker(
                    normalized_path,
                    before_content,
                    after_content,
                    before_exists=before_exists,
                    after_exists=after_exists,
                )
            )
            continue

        from_file = (
            "/dev/null"
            if normalized_path in changes.created
            else f"initial/{normalized_path}"
        )
        to_file = (
            "/dev/null"
            if normalized_path in changes.deleted
            else f"final/{normalized_path}"
        )
        rendered = "".join(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=from_file,
                tofile=to_file,
                lineterm="\n",
            )
        ).rstrip("\n")
        if not rendered and normalized_path in changes.created:
            rendered = f"--- /dev/null\n+++ final/{normalized_path}"
        elif not rendered and normalized_path in changes.deleted:
            rendered = f"--- initial/{normalized_path}\n+++ /dev/null"
        elif not rendered and normalized_path in changes.modified:
            sections.append(
                _text_byte_change_marker(
                    normalized_path,
                    before_content,
                    after_content,
                )
            )
            continue
        if rendered:
            sections.append(rendered + "\n")
    return "\n".join(sections)


def write_workspace_diff(path: Path, diff: str) -> None:
    _atomic_write_text(path, diff)


def evidence_paths_for_report(
    paths: EvidencePaths,
    repo_root: Path,
) -> dict[str, str]:
    repository = Path(repo_root).resolve(strict=False)
    root = paths.root.resolve(strict=False)
    try:
        relative_root = root.relative_to(repository)
    except ValueError as error:
        raise ValueError("Evidence directory is outside the repository.") from error
    if relative_root.parts[:2] != (".runs", "long_running_challenge"):
        raise ValueError("Evidence directory is outside the expected .runs path.")

    artifacts = {
        "initial_snapshot": paths.initial_workspace,
        "final_snapshot": paths.final_workspace,
        "before_manifest": paths.before_manifest,
        "after_manifest": paths.after_manifest,
        "changes": paths.changes_json,
        "diff": paths.diff_file,
    }
    relative_artifacts: dict[str, str] = {}
    for name, path in artifacts.items():
        resolved = path.resolve(strict=False)
        try:
            relative_artifact = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Evidence artifact is outside evidence directory: {path}"
            ) from error
        relative_artifacts[name] = relative_artifact.as_posix()
    return {
        "run_id": paths.run_id,
        "directory": relative_root.as_posix(),
        **relative_artifacts,
    }


def real_validation_process_success(
    *,
    final_integrated_success: bool,
    evidence_preservation_passed: bool,
) -> bool:
    return final_integrated_success and evidence_preservation_passed
