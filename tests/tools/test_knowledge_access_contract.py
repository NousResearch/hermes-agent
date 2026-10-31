import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import knowledge_access as ka
from tools.registry import registry


def _json(value):
    assert isinstance(value, str)
    parsed = json.loads(value)
    assert isinstance(parsed, dict)
    return parsed


def _tree_snapshot(root: Path):
    if not root.exists():
        return []
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            out.append(("link", rel, path.readlink().as_posix()))
        elif path.is_file():
            out.append(("file", rel, path.read_bytes()))
        elif path.is_dir():
            out.append(("dir", rel, b""))
    return out


def _write_skill(root: Path, rel: str, *, name: str, description: str = "test procedure", platforms=None, body="Procedure body"):
    skill_dir = root / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    if platforms is not None:
        lines.append("platforms:")
        lines.extend(f"  - {item}" for item in platforms)
    lines.extend(["---", "", body, ""])
    (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return skill_dir


def _write_registry(home: Path, body: str):
    target = home / "knowledge" / "KB_REGISTRY.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def test_schema_is_closed_three_type_surface():
    params = ka.KNOWLEDGE_ACCESS_SCHEMA["parameters"]
    assert params["additionalProperties"] is False
    assert set(params["properties"]["information_type"]["enum"]) == {
        "HISTORICAL_TRANSCRIPT", "PROCEDURE", "DOCUMENTARY_KNOWLEDGE"
    }
    assert set(params["properties"]) == ka.PUBLIC_ARGUMENT_KEYS
    forbidden = {"owner", "tool_name", "path", "db_path", "filesystem_path", "sql", "table", "column", "source_id", "provider", "registry_path"}
    assert not (set(params["properties"]) & forbidden)


def test_registry_dispatch_rejects_unknown_key_even_null():
    result = _json(registry.dispatch("knowledge_access", {
        "information_type": "HISTORICAL_TRANSCRIPT",
        "owner": None,
    }))
    assert result.get("success") is False or "error" in result
    assert "Unknown knowledge_access argument" in result.get("error", "")


def test_deferred_types_fail_closed_internally():
    for label in ("CURRENT_STATE", "HOT_CONTEXT"):
        with pytest.raises(ValueError):
            ka.execute_knowledge_access(information_type=label)


def test_historical_discovery_delegates_only_supported_args(monkeypatch):
    seen = {}
    db = object()

    def fake_dispatch(name, args, **kwargs):
        seen.update({"name": name, "args": dict(args), "kwargs": dict(kwargs)})
        return json.dumps({"success": True, "mode": "discovery", "results": []})

    monkeypatch.setattr(ka.registry, "dispatch", fake_dispatch)
    result = ka.execute_knowledge_access(
        information_type="HISTORICAL_TRANSCRIPT",
        query="alpha",
        limit=4,
        profile="default",
        after="7d",
        before="2026-09-20",
        db=db,
        current_session_id="current-1",
    )
    assert result["owner"] == "SESSION_SEARCH"
    assert result["result"]["shape"] == "discovery"
    assert seen["name"] == "session_search"
    assert seen["args"] == {
        "query": "alpha", "limit": 4, "profile": "default", "after": "7d", "before": "2026-09-20",
    }
    assert seen["kwargs"] == {"db": db, "current_session_id": "current-1"}


def test_historical_browse_defaults_and_rejects_silently_ignored_fields(monkeypatch):
    seen = {}

    def fake_dispatch(name, args, **kwargs):
        seen.update({"name": name, "args": dict(args), "kwargs": dict(kwargs)})
        return json.dumps({"success": True, "mode": "browse", "results": []})

    monkeypatch.setattr(ka.registry, "dispatch", fake_dispatch)
    result = ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT")
    assert result["result"]["shape"] == "browse"
    assert seen["name"] == "session_search"
    assert seen["args"]["query"] == ""
    assert seen["args"]["limit"] == 3
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", after="7d")
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", window=5)


def test_historical_read_and_scroll_are_distinct(monkeypatch):
    calls = []

    def fake_dispatch(name, args, **kwargs):
        calls.append((name, dict(args), dict(kwargs)))
        return json.dumps({"success": True, "mode": "ok"})

    monkeypatch.setattr(ka.registry, "dispatch", fake_dispatch)
    read_result = ka.execute_knowledge_access(
        information_type="HISTORICAL_TRANSCRIPT", session_id="session-1", profile="default")
    scroll_result = ka.execute_knowledge_access(
        information_type="HISTORICAL_TRANSCRIPT", session_id="session-1", around_message_id=42, window=7)
    assert read_result["result"]["shape"] == "read"
    assert scroll_result["result"]["shape"] == "scroll"
    assert calls[0][0] == "session_search"
    assert calls[0][1]["session_id"] == "session-1" and "around_message_id" not in calls[0][1]
    assert calls[1][0] == "session_search"
    assert calls[1][1]["session_id"] == "session-1"
    assert calls[1][1]["around_message_id"] == 42
    assert calls[1][1]["window"] == 7
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", around_message_id=1)
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", session_id="session-1", query="ignored")


def test_historical_integer_bounds_reject_bool_and_out_of_range():
    bad_calls = [
        {"limit": True}, {"limit": 0}, {"limit": 11},
        {"session_id": "s", "around_message_id": True},
        {"session_id": "s", "around_message_id": 1, "window": True},
        {"session_id": "s", "around_message_id": 1, "window": 0},
        {"session_id": "s", "around_message_id": 1, "window": 21},
    ]
    for extra in bad_calls:
        with pytest.raises(ValueError):
            ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", **extra)


def test_historical_native_failures_are_not_promoted(monkeypatch):
    monkeypatch.setattr(
        ka.registry, "dispatch",
        lambda name, args, **kwargs: json.dumps({"success": False, "error": "native fail"}),
    )
    with pytest.raises(ValueError, match="native fail"):
        ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", query="alpha")
    monkeypatch.setattr(ka.registry, "dispatch", lambda name, args, **kwargs: "not-json")
    with pytest.raises(ValueError, match="invalid JSON"):
        ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", query="alpha")

    def boom(name, args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(ka.registry, "dispatch", boom)
    with pytest.raises(ValueError, match="authority unavailable"):
        ka.execute_knowledge_access(information_type="HISTORICAL_TRANSCRIPT", query="alpha")


def test_procedure_reads_profile_create_and_external_roots_without_writes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    profile_root = home / "skills"
    create_root = home / "create-skills"
    external_root = home / "external-skills"
    for root in (profile_root, create_root, external_root):
        root.mkdir(parents=True)

    _write_skill(profile_root, "cat/alpha", name="alpha-proc", body="Alpha procedure")
    _write_skill(create_root, "beta", name="beta-proc", body="Beta procedure")
    _write_skill(external_root, "gamma", name="gamma-proc", body="Gamma procedure")
    _write_skill(profile_root, "disabled", name="disabled-proc")
    _write_skill(profile_root, "platform-miss", name="platform-miss", platforms=["definitely-not-this-platform"])
    _write_skill(profile_root, "large", name="large-proc", body="x" * (ka.MAX_SKILL_CONTENT_CHARS + 1))

    (home / "config.yaml").write_text(
        "skills:\n"
        "  create_dir: create-skills\n"
        "  external_dirs:\n"
        "    - external-skills\n"
        "  disabled:\n"
        "    - disabled-proc\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import skill_utils
    skill_utils._external_dirs_cache_clear()

    before = _tree_snapshot(home)
    a = ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="cat/alpha")
    b = ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="beta-proc")
    c = ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="gamma-proc")
    after = _tree_snapshot(home)

    assert before == after
    assert a["result"]["content"].endswith("Alpha procedure\n")
    assert a["result"]["relative_skill_path"] == "cat/alpha/SKILL.md"
    assert b["result"]["name"] == "beta-proc"
    assert c["result"]["name"] == "gamma-proc"
    for result in (a, b, c):
        serialized = json.dumps(result)
        assert str(home) not in serialized
        assert "_source_path" not in serialized

    with pytest.raises(ValueError, match="disabled"):
        ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="disabled-proc")
    with pytest.raises(ValueError, match="not supported"):
        ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="platform-miss")
    with pytest.raises(ValueError, match="exceeds"):
        ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="large-proc")


def test_procedure_unknown_ambiguous_and_unsafe_identifiers_fail(tmp_path, monkeypatch):
    home = tmp_path / "home"
    profile_root = home / "skills"
    external_root = home / "external-skills"
    profile_root.mkdir(parents=True)
    external_root.mkdir(parents=True)
    _write_skill(profile_root, "one", name="same-proc")
    _write_skill(external_root, "two", name="same-proc")
    (home / "config.yaml").write_text(
        "skills:\n  external_dirs:\n    - external-skills\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import skill_utils
    skill_utils._external_dirs_cache_clear()

    with pytest.raises(ValueError, match="Unknown"):
        ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="missing-proc")
    with pytest.raises(ValueError, match="Ambiguous"):
        ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="same-proc")
    for identifier in ("plug:thing", "../escape", "/absolute/path", r"C:\\escape"):
        with pytest.raises(ValueError):
            ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier=identifier)


def test_documentary_exact_registry_resolution_and_provider_fields_ignored(tmp_path):
    home = tmp_path / "home"
    token = set_hermes_home_override(home)
    try:
        _write_registry(home, """# registry\n```yaml\n- id: kb-alpha\n  path: docs/alpha.md\n  auxiliary_source: local-only\n- id: kb-beta\n  path: docs/beta.md\n```\n""")
        result = ka.execute_knowledge_access(information_type="DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha")
        assert result["result"] == {"kb_id": "kb-alpha", "canonical_path": "docs/alpha.md"}
        assert "auxiliary_source" not in json.dumps(result)
    finally:
        reset_hermes_home_override(token)


def test_documentary_registry_fail_closed_cases(tmp_path):
    missing = tmp_path / "missing.md"
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha", registry_path=missing)

    malformed = tmp_path / "bad.md"
    malformed.write_text("```yaml\nnot: [valid\n```\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha", registry_path=malformed)

    duplicate = tmp_path / "duplicate.md"
    duplicate.write_text("```yaml\n- id: kb-alpha\n  path: a\n- id: kb-alpha\n  path: b\n```\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        ka.execute_knowledge_access(information_type="DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha", registry_path=duplicate)

    unknown = tmp_path / "unknown.md"
    unknown.write_text("```yaml\n- id: kb-beta\n  path: b\n```\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        ka.execute_knowledge_access(information_type="DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha", registry_path=unknown)

    oversized = tmp_path / "oversized.md"
    oversized.write_bytes(b"x" * (ka.MAX_REGISTRY_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        ka.execute_knowledge_access(information_type="DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha", registry_path=oversized)


def test_branch_specific_null_is_not_silently_ignored():
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="PROCEDURE", skill_identifier="x", limit=None)
    with pytest.raises(ValueError):
        ka.execute_knowledge_access(information_type="DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha", query=None)


def test_candidate_source_has_no_design_only_assertions_or_private_markers():
    root = Path(ka.__file__).resolve().parents[1]
    candidate = [
        root / "agent" / "knowledge_router.py",
        root / "tools" / "knowledge_access.py",
        root / "tests" / "agent" / "test_knowledge_router_contract.py",
        root / "tests" / "tools" / "test_knowledge_access_contract.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in candidate)
    markers = [
        "assert" + " True",
        "/home/" + "jr-ubuntu",
        "bac" + "ardit",
        "hermes-" + "system",
        "hermes-" + "wiki",
        "g" + "brain",
        "date_" + "from",
        "date_" + "to",
        "memory_context_" + "retrieved",
        "hot_context_" + "retrieved",
    ]
    lowered = source.lower()
    for marker in markers:
        assert marker.lower() not in lowered
