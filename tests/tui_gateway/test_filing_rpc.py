"""Tests for the filing.* JSON-RPC surface (correction-aware Projects filing).

The deployed contract library (~/project_filing_contract.py) is loaded for
real semantics; every path is pinned to tmp via PROJECT_FILING_CONTRACT and
monkeypatched loaders so nothing touches the user's live contract, hook, or
projects.db.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

import tui_gateway.server as server
from tui_gateway import filing_bridge

_DEPLOYED = Path.home() / "project_filing_contract.py"

pytestmark = pytest.mark.skipif(
    not _DEPLOYED.is_file(), reason="deployed filing contract library required")


def _call(method, params=None):
    handler = server._methods[method]
    return handler(1, params or {})


def _ok(method, params=None):
    resp = _call(method, params)
    assert "error" not in resp, resp.get("error")
    return resp["result"]


@pytest.fixture()
def real_contract_lib(tmp_path, monkeypatch):
    """Load the deployed contract library under a unique module name and point
    the bridge at it + a tmp contract file."""
    spec = importlib.util.spec_from_file_location("filing_contract_test", _DEPLOYED)
    assert spec is not None and spec.loader is not None
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    monkeypatch.setattr(filing_bridge, "load_contract_lib", lambda: lib)
    monkeypatch.setenv("PROJECT_FILING_CONTRACT", str(tmp_path / "filing.json"))
    return lib


@pytest.fixture()
def no_contract_lib(monkeypatch):
    monkeypatch.setattr(filing_bridge, "load_contract_lib", lambda: None)


def test_filing_methods_registered():
    for m in ("filing.status", "filing.rules", "filing.suggest",
              "filing.apply", "filing.reject"):
        assert m in server._methods


def test_status_reports_unavailable_without_library(no_contract_lib):
    result = _ok("filing.status")
    assert result["available"] is False
    assert result["rules"] == 0


def test_status_unavailable_library_errors_on_reads(no_contract_lib):
    resp = _call("filing.rules")
    assert resp["error"]["code"] == 5071


def test_status_counts_and_hook_presence(real_contract_lib, tmp_path, monkeypatch):
    lib = real_contract_lib
    data = lib.empty_contract()
    data, _ = lib.add_rule("/tmp/filing-test/alpha", "Alpha", source="t", data=data)
    data, _ = lib.add_exclusion("/tmp/filing-test/beta", None, source="t", data=data)
    lib.save_contract(data, filing_bridge.contract_path())
    # Pretend the hook is installed by pointing the bridge's hook lookup at tmp.
    hook_dir = tmp_path / "hooks" / "project-filing"
    hook_dir.mkdir(parents=True)
    (hook_dir / "handler.py").write_text("def handle(e, c): return None\n")
    (hook_dir / "HOOK.yaml").write_text("name: project-filing\nevents:\n  - agent:start\n")
    monkeypatch.setattr(
        filing_bridge, "load_filing_hook_lib", lambda: None)  # db-mirror absent here
    monkeypatch.setattr(
        filing_bridge, "hook_installed",
        lambda: (hook_dir / "handler.py").is_file()
        and "agent:start" in (hook_dir / "HOOK.yaml").read_text())
    result = _ok("filing.status")
    assert result["available"] is True
    assert result["hook_installed"] is True
    assert result["rules"] == 1
    assert result["exclusions"] == 1


def test_rules_lists_policy_with_audit(real_contract_lib):
    lib = real_contract_lib
    data = lib.empty_contract()
    data, _ = lib.add_rule("/tmp/filing-test/gamma", "Gamma", source="file it", data=data)
    lib.save_contract(data, filing_bridge.contract_path())
    result = _ok("filing.rules")
    assert result["rules"][0]["path"] == "/tmp/filing-test/gamma"
    assert result["rules"][0]["project"] == "Gamma"
    assert result["audit"][-1]["action"] == "file"


def test_apply_requires_arguments(real_contract_lib):
    resp = _call("filing.apply", {"path": "", "project": "X"})
    assert resp["error"]["code"] == 5072
    resp = _call("filing.apply", {"path": "/tmp/x"})
    assert resp["error"]["code"] == 5072


def test_apply_writes_rule_and_mirrors_db(real_contract_lib, tmp_path, monkeypatch):
    lib = real_contract_lib
    repo = tmp_path / "repo-delta"
    repo.mkdir()
    result = _ok("filing.apply", {"path": str(repo), "project": "Delta"})
    assert result["applied"] is True
    assert "repo-delta" in result["db"]
    contract = lib.load_contract(filing_bridge.contract_path())
    assert any(r["path"] == str(repo) and r["project"] == "Delta"
              for r in contract["rules"])
    # Idempotent: applying the same rule again reports no change note.
    again = _ok("filing.apply", {"path": str(repo), "project": "Delta"})
    assert again["note"] == ""


def test_reject_records_exclusion(real_contract_lib, tmp_path):
    lib = real_contract_lib
    repo = tmp_path / "repo-eps"
    repo.mkdir()
    result = _ok("filing.reject", {"path": str(repo), "project": "Eps"})
    assert result["recorded"] is True
    contract = lib.load_contract(filing_bridge.contract_path())
    assert any(e["path"] == str(repo) for e in contract["exclusions"])
    # Re-rejecting is a no-op but still reports honestly.
    again = _ok("filing.reject", {"path": str(repo), "project": "Eps"})
    assert again["recorded"] is False
    assert again["note"] == "already excluded"


def test_suggest_ladder_rule_then_cwd_then_exclusion(
        real_contract_lib, tmp_path, monkeypatch):
    lib = real_contract_lib
    from hermes_state import SessionDB

    db_home = tmp_path / "state.db"
    db = SessionDB(db_path=db_home)
    rule_dir = tmp_path / "ruled"
    cwd_dir = tmp_path / "cwded"
    excluded_dir = tmp_path / "vetoed"
    for d in (rule_dir, cwd_dir, excluded_dir):
        d.mkdir()
    db.create_session("s-ruled", "cli", cwd=str(rule_dir))
    db.append_message("s-ruled", "user", "work in ruled")
    db.create_session("s-cwd", "cli", cwd=str(cwd_dir))
    db.append_message("s-cwd", "user", "work in cwded")
    db.create_session("s-veto", "cli", cwd=str(excluded_dir))
    db.append_message("s-veto", "user", "work in vetoed")

    # Contract: rule maps `ruled` -> Ruled Project; exclusion vetoes `vetoed`.
    data = lib.empty_contract()
    data, _ = lib.add_rule(str(rule_dir), "Ruled", source="t", data=data)
    data, _ = lib.add_exclusion(str(excluded_dir), None, source="t", data=data)
    lib.save_contract(data, filing_bridge.contract_path())

    # projects.db: a project whose folder owns `cwded` (cwd_match tier).
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        pdb.create_project(conn, name="Cwded", folders=[str(cwd_dir)])
        # The "Ruled" project must exist for the contract-rule tier to resolve.
        pdb.create_project(conn, name="Ruled", folders=[])

    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        result = _ok("filing.suggest")
        by_session = {s["session_id"]: s for s in result["suggestions"]}
        assert by_session["s-ruled"]["reason"] == "contract_rule"
        assert by_session["s-ruled"]["project_name"] == "Ruled"
        assert by_session["s-cwd"]["reason"] == "cwd_match"
        assert "s-veto" not in by_session  # exclusion vetoes every suggestion

        # A contract rule that agrees with the cwd-derived filing is already
        # satisfied: filing that session is a no-op, not a suggestion.
        data = lib.load_contract(filing_bridge.contract_path())
        data, _ = lib.add_rule(str(cwd_dir), "Cwded", source="agree", data=data)
        lib.save_contract(data, filing_bridge.contract_path())
        result = _ok("filing.suggest")
        by_session = {s["session_id"]: s for s in result["suggestions"]}
        assert "s-cwd" not in by_session
    finally:
        db.close()
