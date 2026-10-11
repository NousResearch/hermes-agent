"""Native dispatcher-to-ledger contracts; all mutations use owned profile roots."""
import hashlib
import hmac
import json
import stat
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text("skills:\n  native_call_audit: true\n  ledger: true\n  ledger_max_bytes: 0\n  external_dirs: []\n", encoding="utf-8")
    from hermes_constants import get_hermes_home
    from agent.skill_utils import get_all_skills_dirs
    from tools import skill_manager_tool as sm
    assert get_hermes_home().resolve() == home.resolve()
    assert sm._skills_dir().resolve() == (home / "skills").resolve()
    assert all(p.resolve().is_relative_to(tmp_path.resolve()) for p in get_all_skills_dirs())
    return home


def create_args(name="audit-fixture"):
    return {"operations": [{"action": "create", "name": name, "content": f"---\nname: {name}\ndescription: Use when testing owned fixtures.\n---\n\nprivate-content-canary\n"}]}


def call(args, **ids):
    from model_tools import handle_function_call
    return handle_function_call("skill_manage", args, **ids)


def records(home):
    path = home / "skill_native_audit.jsonl"
    assert path.exists(), "opted-in native handler entry/completion journal must exist"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_native_dispatch_records_redacted_handler_span(sandbox):
    args = create_args()
    args["credential-shaped-arbitrary-name"] = "credential-value-canary"
    result = call(args, session_id="session-native", turn_id="turn-native", tool_call_id="call-native", api_request_id="request-native")
    assert json.loads(result)["success"] is True
    rows = records(sandbox)
    assert [r["event"] for r in rows] == ["handler_entry", "handler_completion"]
    entry, done = rows
    assert entry["invocation_id"] == done["invocation_id"]
    assert entry["native_ids"] == {"session_id": "session-native", "turn_id": "turn-native", "tool_call_id": "call-native", "api_request_id": "request-native", "task_id": None}
    key_path = sandbox / "skill_native_audit.key"
    key = key_path.read_bytes()
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert entry["args_hmac_sha256"] == hmac.new(key, canonical, hashlib.sha256).hexdigest()
    canonical_result = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert done["result_hmac_sha256"] == hmac.new(key, canonical_result, hashlib.sha256).hexdigest()
    assert done["handler_status"] == "success"
    text = (sandbox / "skill_native_audit.jsonl").read_text()
    for canary in ["private-content-canary", "credential-shaped-arbitrary-name", "credential-value-canary", "audit-fixture", result]:
        assert canary not in text
    assert key.hex() not in text
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((sandbox / "skill_native_audit.jsonl").stat().st_mode) == 0o600


def test_native_span_links_successfully_appended_ledger_evidence(sandbox):
    from tools import skill_ledger
    result = call(create_args(), session_id="session-native")
    assert json.loads(result)["success"] is True
    entry, done = records(sandbox)
    ledger_rows = skill_ledger.list_entries()
    assert len(ledger_rows) == 1
    ledger_row = ledger_rows[0]
    assert ledger_row.get("native_skill_call") == entry["invocation_id"]
    assert ledger_row["evidence"] == {"session_id": "session-native"}
    assert done["ledger_status"] == "appended"
    key = (sandbox / "skill_native_audit.key").read_bytes()
    expected = json.dumps(ledger_row["evidence"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert done["ledger_entries"] == [{"id": ledger_row["id"], "evidence_hmac_sha256": hmac.new(key, expected, hashlib.sha256).hexdigest()}]
    # The existing API and caller-owned evidence remain intact outside native dispatch.
    evidence = {"native_skill_call": "caller-owned", "private-evidence": "evidence-canary"}
    row_id = skill_ledger.append_entry("patch", "audit-fixture", evidence=evidence)
    assert skill_ledger.get_entry(row_id)["evidence"] == evidence
    assert "native_skill_call" not in skill_ledger.get_entry(row_id)
    assert evidence == {"native_skill_call": "caller-owned", "private-evidence": "evidence-canary"}
    assert "evidence-canary" not in (sandbox / "skill_native_audit.jsonl").read_text()


def test_fingerprints_final_middleware_args_and_raw_handler_result(sandbox, monkeypatch):
    from hermes_cli.plugins import _delivery_manager
    original = create_args("original-never-written")
    original["tool_call_id"] = "forged-argument-id"
    final = create_args("rewritten-fixture")
    final["private-arg-name-canary"] = "private-arg-value-canary"
    captured = []
    def rewrite(next_call, **kwargs):
        raw = next_call(final)
        captured.append(raw)
        return '{"middleware_response":"not-the-handler-result"}'
    manager = _delivery_manager()
    monkeypatch.setitem(manager._middleware, "tool_execution", [rewrite])
    response = call(original, task_id="task-native", api_request_id="request-native")
    assert json.loads(response) == {"middleware_response": "not-the-handler-result"}
    assert (sandbox / "skills" / "rewritten-fixture" / "SKILL.md").exists()
    assert not (sandbox / "skills" / "original-never-written").exists()
    entry, done = records(sandbox)
    key = (sandbox / "skill_native_audit.key").read_bytes()
    def digest(value):
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        return hmac.new(key, canonical, hashlib.sha256).hexdigest()
    assert entry["args_hmac_sha256"] == digest(final)
    assert entry["args_hmac_sha256"] != digest(original)
    assert entry["native_ids"] == {"task_id": "task-native", "api_request_id": "request-native", "session_id": None, "turn_id": None, "tool_call_id": None}
    assert done["result_hmac_sha256"] == digest(captured[0]), "fingerprint must cover exact raw handler result, not parsed JSON or transformed response"
    assert done["result_hmac_sha256"] != digest(response)
    text = (sandbox / "skill_native_audit.jsonl").read_text()
    for forbidden in ["forged-argument-id", "private-arg-name-canary", "private-arg-value-canary", "not-the-handler-result", "rewritten-fixture"]:
        assert forbidden not in text


def test_handler_failure_receipt_never_contains_error_text(sandbox):
    args = {"operations": [{"action": "create", "name": "private-name-canary", "content": "invalid-frontmatter-error-canary"}]}
    response = call(args)
    parsed = json.loads(response)
    assert parsed["success"] is False
    entry, done = records(sandbox)
    assert done["handler_status"] == "failure"
    assert done["ledger_status"] == "missing"
    assert done["ledger_entries"] == []
    text = (sandbox / "skill_native_audit.jsonl").read_text()
    assert parsed["error"] not in text
    assert "private-name-canary" not in text
    assert "invalid-frontmatter-error-canary" not in text
    assert entry["native_ids"] == dict.fromkeys(["task_id", "session_id", "tool_call_id", "turn_id", "api_request_id"])


def test_disabled_ledger_is_not_missing_or_committed_proof(sandbox):
    (sandbox / "config.yaml").write_text("skills:\n  native_call_audit: true\n  ledger: false\n  external_dirs: []\n")
    response = call(create_args())
    assert json.loads(response)["success"] is True
    assert (sandbox / "skills" / "audit-fixture" / "SKILL.md").exists()
    entry, done = records(sandbox)
    assert done["handler_status"] == "success"
    assert done["ledger_status"] == "disabled"
    assert done["ledger_entries"] == []
    assert not (sandbox / "skills" / ".curator_ledger.jsonl").exists()
    assert "committed" not in done


def test_staged_approval_is_not_handler_success_or_ledger_mutation(sandbox):
    (sandbox / "config.yaml").write_text("skills:\n  native_call_audit: true\n  ledger: true\n  write_approval: true\n  external_dirs: []\n")
    response = call(create_args())
    parsed = json.loads(response)
    assert parsed["success"] is True and parsed["staged"] is True
    assert not (sandbox / "skills" / "audit-fixture").exists()
    assert (sandbox / "pending" / "skills" / (parsed["pending_id"] + ".json")).exists()
    _, done = records(sandbox)
    assert done["handler_status"] == "approval_pending"
    assert done["ledger_status"] == "missing" and done["ledger_entries"] == []
    assert parsed["pending_id"] not in (sandbox / "skill_native_audit.jsonl").read_text()


def test_rolled_back_batch_retains_linked_historical_ledger_rows(sandbox):
    from tools import skill_ledger
    ops = create_args()["operations"]
    ops.append({"action": "patch", "name": "audit-fixture", "old_string": "never-present-canary", "new_string": "never-written-canary"})
    response = call({"operations": ops})
    assert json.loads(response)["success"] is False
    assert not (sandbox / "skills" / "audit-fixture").exists()
    entry, done = records(sandbox)
    ledger_rows = skill_ledger.list_entries()
    assert len(ledger_rows) == 1  # Successful create row is history, NOT a committed batch.
    assert ledger_rows[0]["native_skill_call"] == entry["invocation_id"]
    assert done["handler_status"] == "failure"
    assert done["batch_rollback"] == "rolled_back"
    assert done["ledger_status"] == "appended"
    assert done["ledger_entries"][0]["id"] == ledger_rows[0]["id"]
    assert "never-present-canary" not in (sandbox / "skill_native_audit.jsonl").read_text()


def test_runner_fixture_isolation(tmp_path):
    from hermes_constants import get_hermes_home
    root = Path(tempfile.gettempdir()).resolve()
    assert tmp_path.resolve().is_relative_to(root)
    assert get_hermes_home().resolve().is_relative_to(root)
    import os
    assert os.environ["HERMES_TEST_ISOLATION"]
    assert os.environ.get("HERMES_STATE_DB_GUARD_BYPASS") != "1"
    assert os.environ.get("HERMES_PM_TEST_ALLOW_AUTO_ENSURE") != "1"
    print(json.dumps({"tmp_path": str(tmp_path), "tempfile_root": str(root), "hermes_home": str(get_hermes_home())}))
