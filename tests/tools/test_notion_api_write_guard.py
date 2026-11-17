"""The Notion write guard is enforced inside notion_api AND in the
model_tools execution backstop, and the Zendesk / Slack-posting precondition
(`notion.write_guard.enabled`) means exactly "the guard is enforcing"."""
import json

import pytest

import tools.notion_api_tool as nt
import tools.notion_write_guard as g
import tools.zendesk_tool as zt

ASSETS = "2" * 32
ROW = "b" * 32


def _guarded(**over):
    guard = {"enabled": True,
             "release_kits": {"data_sources": [ASSETS]},
             "prds": {"data_sources": ["3" * 32]},
             "ideas": {"data_sources": ["4" * 32]}}
    guard.update(over)
    return {"notion": {"write_guard": guard}, "agent_write_guard": True}


class _Proc:
    def __init__(self, stdout):
        self.returncode, self.stdout, self.stderr = 0, stdout, ""


@pytest.fixture
def world(monkeypatch):
    state = {"cfg": _guarded(), "runs": []}
    monkeypatch.setattr(nt.shutil, "which", lambda cmd: "/usr/local/bin/ntn")
    monkeypatch.setattr(nt, "_ntn_env", lambda: {"NOTION_API_TOKEN": "tok"})
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "read_raw_config", lambda: state["cfg"])
    row = {"object": "page", "id": ROW,
           "parent": {"type": "data_source_id", "data_source_id": ASSETS},
           "properties": {"Status": {"id": "s", "type": "select", "select": {"name": "In review"}}}}

    def fake_run(argv, input=None, **kw):
        state["runs"].append((argv, input))
        if argv[3:] == ["-X", "GET"] and argv[2].startswith("v1/pages/"):
            return _Proc(json.dumps(row))
        return _Proc('{"ok": true}')
    monkeypatch.setattr(nt.subprocess, "run", fake_run)
    g.clear_caches()
    return state


def _writes(state):
    return [r for r in state["runs"] if r[0][4] != "GET"]


def test_handler_refuses_before_running_the_write(world):
    out = json.loads(nt._notion_api_handler({
        "path": f"v1/pages/{ROW}", "method": "PATCH",
        "body": {"properties": {"Status": {"select": {"name": "Approved"}}}}}))
    assert "error" in out and "read-only" in out["error"]
    assert _writes(world) == []


def test_handler_refuses_delete(world):
    out = json.loads(nt._notion_api_handler({"path": f"v1/pages/{ROW}", "method": "DELETE"}))
    assert "DELETE" in out["error"]
    assert world["runs"] == []


def test_handler_reads_untouched(world):
    out = json.loads(nt._notion_api_handler({"path": f"v1/pages/{ROW}"}))
    assert out["ok"] is True


def test_unguarded_profile_unchanged(world):
    world["cfg"] = {}
    out = json.loads(nt._notion_api_handler({"path": f"v1/pages/{ROW}", "method": "DELETE"}))
    assert out["ok"] is True
    assert world["runs"][0][0] == ["ntn", "api", f"v1/pages/{ROW}", "-X", "DELETE"]


def test_config_read_failure_denies_writes(world, monkeypatch):
    import hermes_cli.config as hc

    def boom():
        raise OSError("disk")
    monkeypatch.setattr(hc, "read_raw_config", boom)
    out = json.loads(nt._notion_api_handler({"path": f"v1/pages/{ROW}", "method": "PATCH", "body": {}}))
    assert "error" in out
    assert json.loads(nt._notion_api_handler({"path": f"v1/pages/{ROW}"}))["ok"] is True


def test_backstop_blocks_without_reaching_handler(world, monkeypatch):
    import model_tools
    called = []
    monkeypatch.setattr(nt, "_notion_api_handler", lambda args, **kw: called.append(args) or "{}")
    out = json.loads(model_tools.handle_function_call(
        "notion_api", {"path": f"v1/pages/{ROW}", "method": "DELETE"}, skip_pre_tool_call_hook=True))
    assert "DELETE" in out["error"]
    assert called == []


def test_backstop_denial_function():
    assert g.denial_for_tool_call("web_search", {"query": "x"}) is None
    assert g.denial_for_tool_call("notion_api", {"path": "v1/pages/x"}) is None


# ── the contract with zendesk_create_draft ───────────────────────────────

AWG = {"agent_write_guard": True}

# (config, guard enforcing, zendesk / posting may act above `off`)
_MATRIX = [
    ({}, False, False),
    ({"notion": {}, **AWG}, False, False),
    ({"notion": {"write_guard": None}, **AWG}, False, False),
    ({"notion": {"write_guard": False}, **AWG}, False, False),
    ({"notion": {"write_guard": {"enabled": False}}, **AWG}, False, False),
    ({"notion": {"write_guard": {"enabled": True}}, **AWG}, True, True),   # incomplete → enforcing, denies all
    ({"notion": {"write_guard": True}, **AWG}, True, True),
    ({"notion": {"write_guard": {"enabled": "true"}}, **AWG}, True, True),  # malformed → enforcing
    (_guarded(), True, True),
    ({"notion": _guarded()["notion"]}, True, False),   # no agent_write_guard: guard denies, zendesk refuses
]


@pytest.mark.parametrize("config,enforcing,may_act", _MATRIX)
def test_zendesk_precondition_agrees_with_guard(config, enforcing, may_act):
    assert g.is_enforcing(config) is enforcing
    assert zt._write_guard_enabled(config) is may_act
    assert (g.autonomy_precondition(config) is None) is may_act
    # may_act implies enforcing: Zendesk never treats Approved as a human
    # signal while the agent could set it.
    assert enforcing or not may_act
    # Whenever Zendesk treats the row's Approved status as a human signal,
    # the guard must refuse the agent setting Approved itself.
    denial = g.check_request("PATCH", f"v1/pages/{ROW}",
                             {"properties": {"Status": {"select": {"name": "Approved"}}}},
                             cfg=config, lookup=_StaticLookup())
    assert (denial is not None) is enforcing


class _StaticLookup:
    def page(self, page_id):
        return {"object": "page", "id": ROW,
                "parent": {"type": "data_source_id", "data_source_id": ASSETS},
                "properties": {"Status": {"id": "s", "type": "select", "select": {"name": "Draft"}}}}

    def block(self, block_id):
        raise g.GuardLookupError("no")

    def data_source(self, ds):
        return {"properties": {}}

    def me(self):
        return "bot"
