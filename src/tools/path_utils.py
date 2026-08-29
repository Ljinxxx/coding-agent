from pathlib import Path


class WorkspaceBoundaryError(PermissionError):
    pass


def resolve_workspace_path(
    workspace: Path,
    requested_path: str | Path,
) -> Path:
    workspace_root = Path(workspace).resolve(strict=False)
    requested = Path(requested_path)

    if requested.is_absolute():
        candidate = requested.resolve(strict=False)
    else:
        candidate = (workspace_root / requested).resolve(strict=False)

    if not candidate.is_relative_to(workspace_root):
        raise WorkspaceBoundaryError(
            f"Path escapes workspace: {requested_path}"
        )

    return candidate
