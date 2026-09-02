import json
from dataclasses import dataclass
from pathlib import Path

from scripts.e2e_evaluator import (
    CodingTask,
    HostCheckResult,
    materialize_task,
    run_hidden_check,
)


@dataclass(frozen=True)
class ChallengeTokens:
    release_token: str
    migration_key: str
    diag_head_token: str
    diag_tail_token: str


@dataclass(frozen=True)
class LongRunningChallengeFixture:
    files: dict[str, str]
    protected_files: tuple[str, ...]
    hidden_check_code: str


PROTECTED_FILES = (
    "config/policy.json",
    "data/migration_notes.txt",
    "scripts/diagnose.py",
    "tests/test_parser.py",
    "tests/test_store.py",
    "tests/test_service.py",
    "tests/test_serializer.py",
    "tests/test_report.py",
    "tests/test_cli.py",
)

DIAG_MIDDLE_SENTINEL = "diagnostic-middle-record-not-in-bounded-output"


HIDDEN_PARSER_CHECK = r'''
from incident.parser import parse_incident


def require_value_error(value):
    try:
        parse_incident(value)
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError for: {value!r}")


incident = parse_incident(
    "  hidden-17  |  critical  |  731  |  preserve   inner  spacing  "
)
assert incident.incident_id == "hidden-17"
assert incident.severity == "CRITICAL"
assert incident.timestamp == 731
assert isinstance(incident.timestamp, int)
assert incident.message == "preserve   inner  spacing"

require_value_error("hidden-18|unknown|1|bad severity")
require_value_error("hidden-19|LOW|2")
require_value_error("hidden-20|LOW|3|message|extra")
'''.lstrip()


HIDDEN_STORE_SERVICE_CHECK = r'''
from incident.models import Incident
from incident.service import select_actionable
from incident.store import IncidentStore


store = IncidentStore()
store.upsert(Incident("hidden-a", "HIGH", 50, "original high"))
store.upsert(Incident("hidden-a", "LOW", 900, "newer but lower"))
assert store.get("hidden-a").message == "original high"

store.upsert(Incident("hidden-a", "CRITICAL", 10, "higher wins"))
assert store.get("hidden-a").message == "higher wins"

store.upsert(Incident("hidden-b", "MEDIUM", 30, "current"))
store.upsert(Incident("hidden-b", "MEDIUM", 29, "older"))
assert store.get("hidden-b").message == "current"
store.upsert(Incident("hidden-b", "MEDIUM", 31, "newer"))
assert store.get("hidden-b").message == "newer"

items = [
    Incident("zeta", "LOW", 1, "ignored"),
    Incident("same", "MEDIUM", 40, "old"),
    Incident("same", "HIGH", 20, "higher severity"),
    Incident("beta", "CRITICAL", 20, "tie by id"),
    Incident("alpha", "HIGH", 20, "tie by id"),
]
original = list(items)
selected = select_actionable(items, "high")
assert items == original
assert [(item.incident_id, item.timestamp) for item in selected] == [
    ("alpha", 20),
    ("beta", 20),
    ("same", 20),
]
assert select_actionable([], "LOW") == []
'''.lstrip()


HIDDEN_SERIALIZER_REPORT_CLI_CHECK = r'''
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from incident.cli import main
from incident.models import Incident
from incident.report import build_report
from incident.serializer import serialize_incident


incident = Incident("hidden-json", "HIGH", 817, "stable")
expected = {
    "id": "hidden-json",
    "severity": "HIGH",
    "timestamp": 817,
    "message": "stable",
}
assert serialize_incident(incident) == expected
assert serialize_incident(incident) == expected
assert set(serialize_incident(incident)) == {
    "id",
    "severity",
    "timestamp",
    "message",
}

received = [
    Incident("hidden-low", "LOW", 1, "not actionable"),
    incident,
]
actionable = [incident]
report = build_report(received, actionable)
assert report == {
    "schema_version": 2,
    "total_received": 2,
    "total_actionable": 1,
    "severity_counts": {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 1,
        "CRITICAL": 0,
    },
    "incidents": [expected],
}
empty_report = build_report([], [])
assert empty_report["total_received"] == 0
assert empty_report["total_actionable"] == 0
assert empty_report["incidents"] == []

with tempfile.TemporaryDirectory(prefix="incident-hidden-") as directory:
    input_path = Path(directory) / "incidents.txt"
    input_path.write_text(
        "hidden-cli-low|LOW|1|ignored\n"
        "hidden-cli-high|high|2|selected\n",
        encoding="utf-8",
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert main([str(input_path), "--min-severity", "high"]) == 0
    direct_payload = json.loads(output.getvalue())
    assert [item["id"] for item in direct_payload] == ["hidden-cli-high"]

    try:
        main([str(input_path), "--min-severity", "invalid"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Invalid severity must be rejected by the CLI")

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "incident.cli",
            str(input_path),
            "--min-severity",
            "high",
            "--report",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    process_payload = json.loads(completed.stdout)
    assert process_payload["schema_version"] == 2
    assert process_payload["total_received"] == 2
    assert process_payload["total_actionable"] == 1
    assert process_payload["incidents"][0]["id"] == "hidden-cli-high"
'''.lstrip()


HIDDEN_CHECK_CODE = (
    HIDDEN_PARSER_CHECK
    + "\n"
    + HIDDEN_STORE_SERVICE_CHECK
    + "\n"
    + HIDDEN_SERIALIZER_REPORT_CLI_CHECK
)


def generate_migration_notes(
    migration_key: str,
    *,
    line_count: int = 1800,
) -> str:
    if line_count < 1601:
        raise ValueError("line_count must include the migration key line.")

    directives = {
        201: (
            "DIRECTIVE_DEDUPE_POLICY="
            "highest-severity-then-latest-timestamp"
        ),
        401: "DIRECTIVE_OUTPUT_ORDER=timestamp-then-incident-id",
        801: "DIRECTIVE_UNKNOWN_SEVERITY=reject",
        1201: "DIRECTIVE_REPORT_SCHEMA_VERSION=2",
        1601: f"MIGRATION_KEY={migration_key}",
    }
    lines = [
        directives.get(
            line_number,
            (
                f"Line {line_number:04d}: routine migration record for "
                "incident release preparation"
            ),
        )
        for line_number in range(1, line_count + 1)
    ]
    return "\n".join(lines) + "\n"


def generate_diagnose_source(tokens: ChallengeTokens) -> str:
    return f'''import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incident.models import Incident


def release_contract_ready():
    try:
        from incident.parser import parse_incident
        from incident.report import build_report
        from incident.serializer import serialize_incident
        from incident.service import select_actionable
        from incident.store import IncidentStore

        parsed = parse_incident(
            "  diag-id  | high | 42 |  diagnostic   message  "
        )
        if (
            parsed.incident_id != "diag-id"
            or parsed.severity != "HIGH"
            or parsed.timestamp != 42
            or parsed.message != "diagnostic   message"
        ):
            return False

        store = IncidentStore()
        store.upsert(Incident("dedupe", "HIGH", 10, "high"))
        store.upsert(Incident("dedupe", "LOW", 99, "low"))
        if store.get("dedupe").severity != "HIGH":
            return False

        received = [
            Incident("b", "HIGH", 20, "second"),
            Incident("a", "MEDIUM", 10, "first"),
            Incident("a", "HIGH", 30, "replacement"),
        ]
        actionable = select_actionable(received, "medium")
        if [item.incident_id for item in actionable] != ["b", "a"]:
            return False
        serialized = serialize_incident(actionable[0])
        if set(serialized) != {{"id", "severity", "timestamp", "message"}}:
            return False
        report = build_report(received, actionable)
        return (
            report.get("schema_version") == 2
            and report.get("total_received") == 3
            and report.get("total_actionable") == 2
        )
    except Exception:
        return False


ready = release_contract_ready()
print("DIAG_HEAD_TOKEN={tokens.diag_head_token}")
for index in range(600):
    if index == 300:
        print("DIAG_MIDDLE_SENTINEL={DIAG_MIDDLE_SENTINEL}")
    else:
        print(
            f"Diagnostic record {{index:05d}}: deterministic incident "
            "release contract observation"
        )
print("DIAG_STATE=" + ("READY" if ready else "BLOCKED"))
print("DIAG_ACTION=" + ("verify-release" if ready else "implement-release-contract"))
print("DIAG_TAIL_TOKEN={tokens.diag_tail_token}")
'''


def _fixture_readme() -> str:
    return '''# Incident Triage Service

This repository is being migrated to the v2 incident release contract.
Historical deployment documentation references `config/legacy_policy.json`.
The v2 release also uses the repository-local policy at `config/policy.json`.

Incident input uses four pipe-delimited fields:

`incident_id|severity|timestamp|message`

The supported severity names are LOW, MEDIUM, HIGH, and CRITICAL. Tests are
authoritative for public API behavior. The `--min-severity` option accepts
these names case-insensitively. Unsupported values must be rejected at the CLI
argument boundary as a CLI usage error with exit status 2, before incident
processing begins. A programmatic `main(...)` call must raise `SystemExit(2)`
for an unsupported severity, not return 2, and validation must happen before
opening the input file. The `python -m incident.cli` module entrypoint must
invoke the same public CLI behavior.

When `--report` is used, the CLI must emit the same complete v2 report schema
defined by `incident.report.build_report`. The schema has exactly these
top-level fields: `schema_version`, `total_received`, `total_actionable`,
`severity_counts`, and `incidents`. `schema_version` is 2. `severity_counts`
contains actionable-incident counts for LOW, MEDIUM, HIGH, and CRITICAL, with
all four keys present even when a count is zero. `incidents` contains the
serialized actionable incidents in their required output order, using the
public `id`, `severity`, `timestamp`, and `message` serialization fields.
The public report function signature is
`incident.report.build_report(received, actionable)`. `received` contains all
parsed incidents before actionable selection. `actionable` contains the
selected, deduplicated, and ordered incidents.

Do not modify release inputs or tests.
'''


def _architecture_doc() -> str:
    return '''# Architecture

- `incident.models` owns the immutable Incident value object.
- `incident.parser` converts one input record into an Incident.
- `incident.store` owns canonical incident replacement.
- `incident.service` selects and orders actionable incidents.
- `incident.serializer` exposes the stable JSON shape.
- `incident.report` owns the v2 aggregate report and may need to be created.
- `incident.cli` connects parsing, selection, serialization, and reporting.

Deployment defaults are documented at `../shared/global_defaults.json`.
Reconcile that reference with the repository-local policy during the release
audit.
'''


def _operations_doc() -> str:
    return '''# Release Operations

1. Read the release briefing and preserve its token in conversation history.
2. Inspect repository-local documentation and configuration.
3. Reconcile documented policy references with repository-local evidence.
4. Audit the complete migration notes by following read pagination metadata.
5. Run the baseline visible tests and diagnostic command without hiding errors.
6. Repair production modules while preserving tests and protected inputs.
7. Run visible tests, diagnostics, and trusted workspace verification.

The diagnostic script is a protected executable probe. Run it as a command;
do not read or modify its source.
'''


def _initial_source_files() -> dict[str, str]:
    return {
        "incident/__init__.py": "from incident.models import Incident\n",
        "incident/models.py": '''from dataclasses import dataclass


@dataclass(frozen=True)
class Incident:
    incident_id: str
    severity: str
    timestamp: int
    message: str
''',
        "incident/constants.py": '''SEVERITY_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}
''',
        "incident/parser.py": '''from incident.constants import SEVERITY_ORDER
from incident.models import Incident


def parse_incident(line: str) -> Incident:
    parts = line.split("|")
    if len(parts) != 4:
        raise ValueError("incident line must contain four fields")
    incident_id, severity, timestamp, message = parts
    incident_id = incident_id.strip()
    severity = severity.strip().upper()
    if severity not in SEVERITY_ORDER:
        raise ValueError(f"unknown severity: {severity}")
    return Incident(
        incident_id=incident_id,
        severity=severity,
        timestamp=int(timestamp.strip()),
        message=message.strip(),
    )
''',
        "incident/store.py": '''class IncidentStore:
    def __init__(self):
        self._incidents = {}

    def get(self, incident_id):
        return self._incidents.get(incident_id)

    def values(self):
        return list(self._incidents.values())

    def upsert(self, incident):
        self._incidents[incident.incident_id] = incident
        return True
''',
        "incident/service.py": '''from incident.constants import SEVERITY_ORDER


def select_actionable(incidents, min_severity):
    threshold = SEVERITY_ORDER[min_severity]
    selected = [
        incident
        for incident in incidents
        if SEVERITY_ORDER[incident.severity] >= threshold
    ]
    return sorted(selected, key=lambda item: item.incident_id)
''',
        "incident/serializer.py": '''def serialize_incident(incident):
    return {
        "id": incident.incident_id,
        "severity": incident.severity,
        "timestamp": str(incident.timestamp),
        "message": incident.message,
    }
''',
        "incident/cli.py": '''import argparse
import json

from incident.constants import SEVERITY_ORDER
from incident.parser import parse_incident
from incident.serializer import serialize_incident
from incident.service import select_actionable


def _severity(value):
    normalized = value.strip().upper()
    if normalized not in SEVERITY_ORDER:
        raise argparse.ArgumentTypeError(f"unknown severity: {value}")
    return normalized


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/sample_incidents.txt",
    )
    parser.add_argument(
        "--min-severity",
        type=_severity,
        default="MEDIUM",
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    with open(args.input_path, encoding="utf-8") as input_file:
        received = [
            parse_incident(line) for line in input_file if line.strip()
        ]
    actionable = select_actionable(received, args.min_severity)
    if args.report:
        from incident.report import build_report

        payload = build_report(received, actionable)
    else:
        payload = [serialize_incident(item) for item in actionable]
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    }


def _visible_test_files() -> dict[str, str]:
    return {
        "tests/test_parser.py": '''import pytest

from incident.models import Incident
from incident.parser import parse_incident


def test_parse_incident_basic_record():
    assert parse_incident("A-1|HIGH|12|disk full") == Incident(
        "A-1", "HIGH", 12, "disk full"
    )


def test_parse_incident_normalizes_identifier_severity_and_timestamp():
    incident = parse_incident("  A-2  | medium |  17  | queued ")
    assert (incident.incident_id, incident.severity) == ("A-2", "MEDIUM")
    assert incident.timestamp == 17
    assert isinstance(incident.timestamp, int)


def test_parse_incident_rejects_unknown_severity():
    with pytest.raises(ValueError):
        parse_incident("A-3|NOTICE|18|unsupported")


def test_parse_incident_preserves_internal_message_content_and_four_fields():
    incident = parse_incident("A-4|LOW|19|  keep   these  spaces  ")
    assert incident.message == "keep   these  spaces"
    with pytest.raises(ValueError):
        parse_incident("A-4|LOW|19|message|extra")
''',
        "tests/test_store.py": '''from incident.models import Incident
from incident.store import IncidentStore


def test_store_inserts_and_exposes_incidents():
    store = IncidentStore()
    incident = Incident("A", "LOW", 1, "first")
    store.upsert(incident)
    assert store.get("A") == incident
    assert store.values() == [incident]


def test_store_replaces_with_higher_severity():
    store = IncidentStore()
    store.upsert(Incident("A", "LOW", 20, "low"))
    store.upsert(Incident("A", "HIGH", 10, "high"))
    assert store.get("A").message == "high"


def test_store_keeps_higher_existing_severity():
    store = IncidentStore()
    store.upsert(Incident("A", "CRITICAL", 10, "critical"))
    store.upsert(Incident("A", "MEDIUM", 99, "medium"))
    assert store.get("A").message == "critical"


def test_store_uses_latest_timestamp_for_equal_severity():
    store = IncidentStore()
    store.upsert(Incident("A", "HIGH", 20, "current"))
    store.upsert(Incident("A", "HIGH", 19, "older"))
    assert store.get("A").message == "current"
    store.upsert(Incident("A", "HIGH", 21, "newer"))
    assert store.get("A").message == "newer"
''',
        "tests/test_service.py": '''from incident.models import Incident
from incident.service import select_actionable


def test_select_actionable_empty_and_does_not_mutate_input():
    incidents = []
    assert select_actionable(incidents, "LOW") == []
    assert incidents == []


def test_select_actionable_filters_with_case_insensitive_threshold():
    incidents = [
        Incident("low", "LOW", 1, "low"),
        Incident("high", "HIGH", 2, "high"),
    ]
    assert [item.incident_id for item in select_actionable(incidents, "high")] == [
        "high"
    ]


def test_select_actionable_deduplicates_by_highest_severity():
    incidents = [
        Incident("same", "LOW", 50, "low"),
        Incident("same", "CRITICAL", 10, "critical"),
    ]
    selected = select_actionable(incidents, "LOW")
    assert [(item.incident_id, item.message) for item in selected] == [
        ("same", "critical")
    ]


def test_select_actionable_uses_latest_timestamp_after_severity_tie():
    incidents = [
        Incident("same", "MEDIUM", 10, "old"),
        Incident("same", "MEDIUM", 12, "new"),
    ]
    assert select_actionable(incidents, "LOW")[0].message == "new"


def test_select_actionable_orders_by_timestamp_then_id_without_mutation():
    incidents = [
        Incident("z", "HIGH", 20, "z"),
        Incident("b", "HIGH", 10, "b"),
        Incident("a", "HIGH", 10, "a"),
    ]
    original = list(incidents)
    selected = select_actionable(incidents, "MEDIUM")
    assert [(item.timestamp, item.incident_id) for item in selected] == [
        (10, "a"),
        (10, "b"),
        (20, "z"),
    ]
    assert incidents == original
''',
        "tests/test_serializer.py": '''from incident.models import Incident
from incident.serializer import serialize_incident


def test_serialize_incident_returns_release_schema_and_primitive_values():
    incident = Incident("A", "HIGH", 7, "message")
    assert serialize_incident(incident) == {
        "id": "A",
        "severity": "HIGH",
        "timestamp": 7,
        "message": "message",
    }


def test_serialize_incident_has_no_extra_fields():
    payload = serialize_incident(Incident("A", "LOW", 1, "message"))
    assert set(payload) == {"id", "severity", "timestamp", "message"}


def test_serialize_incident_is_deterministic():
    incident = Incident("A", "CRITICAL", 9, "stable")
    assert serialize_incident(incident) == serialize_incident(incident)
''',
        "tests/test_report.py": '''from incident.models import Incident


def test_build_report_has_v2_schema_and_totals():
    from incident.report import build_report

    incident = Incident("A", "HIGH", 1, "message")
    report = build_report([incident], [incident])
    assert report["schema_version"] == 2
    assert report["total_received"] == 1
    assert report["total_actionable"] == 1


def test_build_report_counts_actionable_severities():
    from incident.report import build_report

    received = [
        Incident("low", "LOW", 1, "low"),
        Incident("high", "HIGH", 2, "high"),
    ]
    report = build_report(received, [received[1]])
    assert report["severity_counts"] == {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 1,
        "CRITICAL": 0,
    }


def test_build_report_handles_empty_input():
    from incident.report import build_report

    report = build_report([], [])
    assert report["total_received"] == 0
    assert report["total_actionable"] == 0
    assert report["incidents"] == []


def test_build_report_serializes_actionable_incidents_in_given_order():
    from incident.report import build_report

    incidents = [
        Incident("A", "MEDIUM", 1, "first"),
        Incident("B", "CRITICAL", 2, "second"),
    ]
    assert [item["id"] for item in build_report(incidents, incidents)["incidents"]] == [
        "A",
        "B",
    ]
''',
        "tests/test_cli.py": '''import json
import subprocess
import sys

import pytest

from incident.cli import main


def _input_file(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text(
        "low|LOW|1|ignored\\n"
        "high|HIGH|2|selected\\n",
        encoding="utf-8",
    )
    return path


def _expected_report_payload():
    return {
        "schema_version": 2,
        "total_received": 2,
        "total_actionable": 1,
        "severity_counts": {
            "LOW": 0,
            "MEDIUM": 0,
            "HIGH": 1,
            "CRITICAL": 0,
        },
        "incidents": [
            {
                "id": "high",
                "severity": "HIGH",
                "timestamp": 2,
                "message": "selected",
            }
        ],
    }


def test_cli_outputs_default_json_list(tmp_path, capsys):
    input_path = _input_file(tmp_path)
    assert main([str(input_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in payload] == ["high"]


def test_cli_accepts_lowercase_minimum_severity(tmp_path, capsys):
    input_path = _input_file(tmp_path)
    assert main([str(input_path), "--min-severity", "high"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["severity"] == "HIGH"


def test_cli_rejects_unsupported_severity_with_usage_error(tmp_path):
    input_path = _input_file(tmp_path)
    with pytest.raises(SystemExit) as error:
        main([str(input_path), "--min-severity", "urgent"])
    assert error.value.code == 2


def test_cli_rejects_unsupported_severity_before_opening_input(monkeypatch):
    def forbidden_open(*args, **kwargs):
        raise AssertionError("input must not be opened before argument validation")

    monkeypatch.setattr("builtins.open", forbidden_open)
    with pytest.raises(SystemExit) as error:
        main(["public-do-not-open.txt", "--min-severity", "urgent"])
    assert error.value.code == 2


def test_cli_report_flag_outputs_v2_report(tmp_path, capsys):
    input_path = _input_file(tmp_path)
    assert main([str(input_path), "--report"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == _expected_report_payload()


def test_cli_module_report_outputs_complete_v2_payload(tmp_path):
    input_path = _input_file(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m", "incident.cli",
            str(input_path),
            "--report",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == _expected_report_payload()
''',
    }


def build_long_running_fixture(
    tokens: ChallengeTokens,
) -> LongRunningChallengeFixture:
    policy = {
        "default_min_severity": "MEDIUM",
        "severity_order": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    }
    files = {
        "README.md": _fixture_readme(),
        "docs/architecture.md": _architecture_doc(),
        "docs/operations.md": _operations_doc(),
        "config/policy.json": json.dumps(policy, indent=2) + "\n",
        "data/migration_notes.txt": generate_migration_notes(
            tokens.migration_key
        ),
        "data/sample_incidents.txt": (
            "INC-001|low|100|initial low event\n"
            " INC-001 | HIGH | 200 | escalated event \n"
            "INC-002|medium|150|initial medium event\n"
            "INC-002|MEDIUM|250|newer medium event\n"
            "INC-003|critical|125|critical event\n"
            "INC-004|LOW|50|low event\n"
        ),
        "scripts/diagnose.py": generate_diagnose_source(tokens),
        **_initial_source_files(),
        **_visible_test_files(),
    }
    return LongRunningChallengeFixture(
        files=files,
        protected_files=PROTECTED_FILES,
        hidden_check_code=HIDDEN_CHECK_CODE,
    )


def materialize_long_running_fixture(
    fixture: LongRunningChallengeFixture,
    workspace: Path,
) -> None:
    task = CodingTask(
        task_id="long_horizon_incident_repair",
        title="Long-Horizon Repository Repair",
        prompt="",
        files=fixture.files,
        protected_files=fixture.protected_files,
        hidden_check_code=fixture.hidden_check_code,
    )
    materialize_task(task, Path(workspace))


def run_hidden_code(
    workspace: Path,
    code: str,
    *,
    timeout: int = 10,
) -> HostCheckResult:
    task = CodingTask(
        task_id="long_horizon_hidden_check",
        title="Long-Horizon Host Hidden Check",
        prompt="",
        files={},
        protected_files=(),
        hidden_check_code=code,
    )
    return run_hidden_check(task, Path(workspace), timeout=timeout)
