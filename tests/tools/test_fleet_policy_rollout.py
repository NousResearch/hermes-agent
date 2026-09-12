"""Tests for the Fleet Policy rollout tool (tools/fleet_policy_rollout.py).

Real tool registry + real kanban DB on a temp HERMES_HOME; disposable
ten-profile fleet fixtures. Nothing here touches a live profile.
"""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

PROFILES = (
    "company", "design", "finance", "operations", "product",
    "qa", "research", "sales", "tech", "ux",
)

GATE_AUTHORS = {
    "gate:ci=pass": "qa",
    "gate:review=pass": "qa",
    "gate:rollback=pass": "operations",
}


def _seed_bundle(home, version="1.2.18"):
    """Full canonical bundle shape (mirrors fleet_policy.release_bundle
    RELEASE_PATHS): every required path present, manifest checksums real."""
    root = home / "releases" / version
    payload = root / "integrations" / "hermes" / "fleet-policy-plugin"
    for directory in ("config", "src/fleet_policy", "tests", "scripts",
                      "skills/company-os", "skills/rr-project"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    payload.mkdir(parents=True, exist_ok=True)
    # Canonical bundle layout also carries a root plugin.yaml (RELEASE_PATHS).
    (root / "plugin.yaml").write_text(
        'name: fleet-policy\nversion: "' + version + '"\n', encoding="utf-8")
    (payload / "plugin.yaml").write_text(
        'name: fleet-policy\nversion: "' + version + '"\ndescription: test payload\n',
        encoding="utf-8")
    (payload / "__init__.py").write_text("PAYLOAD_MARKER = 'ok'\n", encoding="utf-8")
    (root / "config" / "fleet-policy.yaml").write_text(
        "schema: hermes-fleet-policy\n", encoding="utf-8")
    for required_file in ("pyproject.toml", "uv.lock", "AGENTS.md", "APPROVALS.md",
                          "README.md", "CHANGELOG.md", "OPERATING_SYSTEM.md",
                          "PORTFOLIO.md"):
        (root / required_file).write_text("# placeholder\n", encoding="utf-8")
    for required_dir_marker in ("src/fleet_policy/__init__.py", "tests/test_placeholder.py",
                                "scripts/placeholder.py", "skills/company-os/SKILL.md",
                                "skills/rr-project/SKILL.md"):
        (root / required_dir_marker).write_text("# placeholder\n", encoding="utf-8")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema": "fleet-policy-release-bundle-v1",
                "files": [{"path": rel, "sha256": files[rel]} for rel in sorted(files)]}
    (root / "RELEASE-MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return root


def _setup(monkeypatch, tmp_path, version="1.2.18"):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_VENTURES_ROOT", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="rollout-test", assignee="tech")
        for marker, author in GATE_AUTHORS.items():
            kb.add_comment(conn, task_id, author=author, body=marker + " verified")
    finally:
        conn.close()
    profiles_root = home / "profiles"
    for profile in PROFILES:
        (profiles_root / profile / "plugins").mkdir(parents=True, exist_ok=True)
    # Orchestrator context: the kanban toolset (and thus the operator-only
    # rollout tool) resolves via profile config when HERMES_KANBAN_TASK is absent.
    (home / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    return task_id, _seed_bundle(home, version)


def _call(args):
    from tools import kanban_tools as kt
    return kt.handle_fleet_policy_rollout(args)


def _receipt(out):
    return json.loads(out)


# ---------------------------------------------------------------------------
# Registration / gating
# ---------------------------------------------------------------------------

def test_tool_registered_and_operator_only(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    import tools.kanban_tools  # noqa: F401  (ensures registration)
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake0001")
    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "fleet_policy_rollout" not in names

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "fleet_policy_rollout" in names


def test_worker_call_refused(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake0001")
    out = _call({"task_id": "whatever", "release_sha": "a" * 64, "version": "9.9.9"})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert "refused" in str(d.get("error", ""))


# ---------------------------------------------------------------------------
# Input / release-gate failures
# ---------------------------------------------------------------------------

def test_unknown_args_rejected(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    out = _call({"task_id": "t_x", "release_sha": "a" * 64, "version": "1.2.18",
                 "paths": ["C:/evil"]})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert "unknown argument" in str(d.get("error", ""))


def test_missing_bundle_rejected(monkeypatch, tmp_path):
    task_id, _ = _setup(monkeypatch, tmp_path)
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "9.9.9"})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert "release bundle gate failed" in str(d.get("error", ""))


def test_tampered_bundle_rejected(monkeypatch, tmp_path):
    task_id, release_root = _setup(monkeypatch, tmp_path)
    (release_root / "integrations" / "hermes" / "fleet-policy-plugin" / "plugin.yaml").write_text(
        'name: fleet-policy\nversion: "1.2.18"\ndescription: tampered\n', encoding="utf-8")
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert "release bundle gate failed" in str(d.get("error", ""))


def test_version_mismatch_rejected(monkeypatch, tmp_path):
    """plugin.yaml declaring a different version than requested must fail."""
    _setup(monkeypatch, tmp_path)
    home = tmp_path / "hermes-home"
    source = home / "releases" / "1.2.18"
    other = home / "releases" / "1.2.19"
    # Verbatim copy: verifies clean, but root + payload plugin.yaml still
    # declare 1.2.18 while 1.2.19 is requested → version gate must refuse.
    shutil.copytree(source, other)
    out = _call({"task_id": "t_x", "release_sha": "a" * 64, "version": "1.2.19"})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert "release plugin gate" in str(d.get("error", ""))
    assert "1.2.18" in str(d.get("error", ""))


# ---------------------------------------------------------------------------
# Evidence-gate failures
# ---------------------------------------------------------------------------

def _clear_comments(task_id):
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
        conn.commit()
    finally:
        conn.close()


def _add_comment(task_id, author, body):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        kb.add_comment(conn, task_id, author=author, body=body)
    finally:
        conn.close()


def test_gates_missing_rejected(monkeypatch, tmp_path):
    task_id, _ = _setup(monkeypatch, tmp_path)
    _clear_comments(task_id)
    _add_comment(task_id, "qa", "gate:ci=pass only")
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d.get("ok") is not True
    err = str(d.get("error", ""))
    assert "evidence gates failed" in err
    assert "gate:review=pass (by qa)" in err
    assert "gate:rollback=pass (by operations or tech)" in err


def test_gate_from_wrong_role_rejected(monkeypatch, tmp_path):
    task_id, _ = _setup(monkeypatch, tmp_path)
    _clear_comments(task_id)
    _add_comment(task_id, "qa", "gate:ci=pass")
    _add_comment(task_id, "qa", "gate:review=pass")
    _add_comment(task_id, "sales", "gate:rollback=pass")
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert "gate:rollback=pass (by operations or tech)" in str(d.get("error", ""))


def test_later_wrong_role_comment_invalidates_earlier_pass(monkeypatch, tmp_path):
    """The latest comment carrying a marker wins: a later wrong-role
    restatement invalidates an earlier correct-role pass."""
    task_id, _ = _setup(monkeypatch, tmp_path)
    _add_comment(task_id, "tech", "gate:review=pass later restatement")
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert "gate:review=pass (by qa)" in str(d.get("error", ""))


# ---------------------------------------------------------------------------
# Happy path + rollback
# ---------------------------------------------------------------------------

def test_successful_rollout_installs_on_all_ten_profiles(monkeypatch, tmp_path):
    task_id, release_root = _setup(monkeypatch, tmp_path)
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d["ok"] is True
    assert d["version"] == "1.2.18"
    assert d["release_sha"] == "a" * 64
    assert d["rolled_back"] is False
    assert len(d["installed"]) == 10
    assert set(d["installed"]) == set(PROFILES)
    assert all(v is None for v in d["backed_up"].values())
    receipt_files = {f["path"]: f["sha256"] for f in d["files"]}
    assert "plugins/fleet-policy/plugin.yaml" in receipt_files
    assert "plugins/fleet-policy/config/fleet-policy.yaml" in receipt_files
    home = tmp_path / "hermes-home"
    source_yaml = (release_root / "integrations" / "hermes" /
                   "fleet-policy-plugin" / "plugin.yaml").read_text(encoding="utf-8")
    for profile in d["installed"]:
        installed = home / "profiles" / profile / "plugins" / "fleet-policy"
        assert (installed / "plugin.yaml").read_text(encoding="utf-8") == source_yaml
        assert (installed / "config" / "fleet-policy.yaml").exists()


def test_successful_upgrade_backs_up_existing_install(monkeypatch, tmp_path):
    task_id, _ = _setup(monkeypatch, tmp_path)
    home = tmp_path / "hermes-home"
    target = home / "profiles" / "tech" / "plugins" / "fleet-policy"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text('version: "1.2.17"\n', encoding="utf-8")
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d["ok"] is True
    assert d["backed_up"]["tech"].startswith("fleet-policy.old-")
    backups = list((home / "profiles" / "tech" / "plugins").glob("fleet-policy.old-*"))
    assert len(backups) == 1
    assert (backups[0] / "plugin.yaml").read_text(encoding="utf-8") == 'version: "1.2.17"\n'
    assert (target / "plugin.yaml").read_text(encoding="utf-8").startswith(
        'name: fleet-policy\nversion: "1.2.18"\n')


def test_failed_profile_rollout_is_fully_rolled_back(monkeypatch, tmp_path):
    """Failing on a late profile (ux: missing plugins dir) must restore every
    earlier profile exactly: fresh payloads removed, the pre-existing tech
    install swapped back from its backup."""
    task_id, _ = _setup(monkeypatch, tmp_path)
    home = tmp_path / "hermes-home"
    plugins_root = home / "profiles"
    target = plugins_root / "tech" / "plugins" / "fleet-policy"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text('version: "1.2.17"\n', encoding="utf-8")
    shutil.rmtree(plugins_root / "ux" / "plugins")
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d.get("ok") is not True
    err = str(d.get("error", ""))
    assert "rolled back" in err
    assert "ux" in err
    assert not (plugins_root / "ux" / "plugins").exists()
    for profile in PROFILES:
        if profile in ("tech", "ux"):
            continue
        assert not (plugins_root / profile / "plugins" / "fleet-policy").exists(), profile
    assert (target / "plugin.yaml").read_text(encoding="utf-8") == 'version: "1.2.17"\n'


def test_incomplete_rollout_rolled_back(monkeypatch, tmp_path):
    """Defense-in-depth guard: a short install list triggers the same full
    rollback; the roster invariant holds on disk."""
    task_id, _ = _setup(monkeypatch, tmp_path)
    home = tmp_path / "hermes-home"
    plugins_root = home / "profiles"
    shutil.rmtree(plugins_root / "ux" / "plugins")
    out = _call({"task_id": task_id, "release_sha": "a" * 64, "version": "1.2.18"})
    d = _receipt(out)
    assert d.get("ok") is not True
    assert not (plugins_root / "qa" / "plugins" / "fleet-policy").exists()
    assert not (plugins_root / "tech" / "plugins" / "fleet-policy").exists()

