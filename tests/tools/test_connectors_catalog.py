"""``manage_catalog`` (catalog install through the connection card).

Contracts:
- the tool reaches only desktop sessions, whatever a config selects, and is deferred by default
- only a desktop chat draws it
- the model sends catalog ids and an action; every other key is refused before anything runs
- an id the catalog does not know, or a plugin this OS cannot run, is drawn failed and never installed
- an approved row installs and reports what went live; the card never witnesses the outcome
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.connectors import live
from tools.connectors.catalog_tool import NOT_HERE, manage_catalog
from tools.connectors.contract import TargetState
from tools.connectors.run import apply_answer


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


def _entry(name, *, platforms=(), requires_env=()):
    return SimpleNamespace(
        name=name, repo=f"https://github.com/example/{name}", sha="a" * 40, subdir="", tier="official",
        description=f"Drives {name}. More text.", requires_hermes="", platforms=list(platforms),
        capabilities=SimpleNamespace(requires_env=list(requires_env)),
    )


class FakeInstaller:
    """The host side of a row: the catalog, the platform refusal and the installer."""

    def __init__(self, entries=(), *, refuse=None, install_error=""):
        self.entries = {e.name: e for e in entries}
        self.refusal = refuse or {}
        self.install_error = install_error
        self.installs = []

    def plugin_entry(self, name):
        return self.entries.get(name)

    def refuse(self, entry):
        if entry.name in self.refusal:
            raise RuntimeError(self.refusal[entry.name])

    def install_plugin(self, name, *, force, enable, ref, on_step):
        from hermes_constants import get_hermes_home

        self.installs.append({"name": name, "force": force, "enable": enable, "ref": ref,
                              "home": Path(get_hermes_home())})
        if self.install_error:
            return {"ok": False, "error": self.install_error}
        tools = [f"mcp__{name}__status", f"mcp__{name}__launch"]
        return {"ok": True, "plugin_name": name, "missing_env": [], "activation": {
            "live_now": {"mcp_servers": [{"name": name, "connected": True, "tools": tools}], "skills": []}}}

    def skill_meta(self, identifier):
        return None

    def install_skill(self, identifier, *, force):
        raise AssertionError("no skill in these tests")


def _card(answer, *, session_id="s1", profile_home=None):
    """A desktop card that answers the live operation a moment after it is drawn. ``profile_home`` is
    the session's profile, which ``connection.respond`` looks the operation up under."""
    seen = []

    def callback(payload):
        seen.append(payload)

        def respond():
            operation = live.get(session_id, payload["op_id"], profile_home=profile_home)
            if operation is not None:
                apply_answer(operation, json.dumps(answer(payload)))

        threading.Timer(0.01, respond).start()

    callback.seen = seen
    return callback


def _install(items, installer, card):
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        return json.loads(manage_catalog({"action": "install", "items": items}, session_id="s1",
                                         connection_callback=card, card_surface=True, installer=installer))


def _approve(env=None):
    return lambda payload: {"targets": [{"name": t["name"], "status": "approved", "env": env}
                                        for t in payload["targets"] if t["state"] == "pending"]}


def test_only_a_desktop_session_carries_the_tool():
    import model_tools
    import toolsets
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.tool_search import is_deferrable_tool_name

    # A config may name the toolset; `all` never brings it in.
    assert "manage_catalog" in model_tools._select_tool_names(["catalog", "web"], None, quiet_mode=True)
    assert "manage_catalog" not in model_tools._select_tool_names(None, None, quiet_mode=True)
    assert "manage_catalog" not in toolsets.resolve_toolset("all")
    # The agent's own session platform decides whether it keeps the tool.
    for platform in ("cli", "tui", "telegram", "cron", None):
        assert "manage_catalog" in toolsets.session_platform_tool_drops(platform), platform
    assert "manage_catalog" not in toolsets.session_platform_tool_drops("desktop")
    # Behind tool_search by default, like the other desktop surface tools.
    assert is_deferrable_tool_name("manage_catalog", frozenset(DEFAULT_CONFIG["tools"]["tool_search"]["defer"]))


def _desktop_child_toolsets():
    from tools.delegate_tool_toolsets import _resolve_child_toolsets
    parent = SimpleNamespace(enabled_toolsets=["catalog", "web"], disabled_toolsets=None)
    return _resolve_child_toolsets(parent, None, "leaf")


@pytest.mark.parametrize("platform", ["desktop", "cli", "subagent"])
def test_a_session_off_the_desktop_never_hears_of_the_tool_even_through_tool_search(monkeypatch, platform):
    """A CLI config naming ``catalog``, and a delegate child of a desktop chat, lose manage_catalog from
    the direct tools, the tool_search listing and the bridge's search, not just from the direct tools."""
    import model_tools
    from agent.agent_init import _load_tools

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    enabled, disabled = _desktop_child_toolsets() if platform == "subagent" else (["catalog", "web"], None)
    assert "catalog" in enabled
    agent = SimpleNamespace(quiet_mode=True, platform=platform, enabled_toolsets=enabled, disabled_toolsets=disabled)

    _load_tools(agent, enabled, disabled)
    found = json.loads(model_tools.handle_function_call(
        "tool_search", {"queries": ["catalog plugin install"]},
        enabled_toolsets=agent.enabled_toolsets, disabled_toolsets=agent.disabled_toolsets))

    carried = "manage_catalog" in json.dumps(agent.tools) or "manage_catalog" in json.dumps(found)
    assert carried is (platform == "desktop")


@pytest.mark.parametrize("args", [
    {"action": "install", "items": [{"kind": "plugin", "id": "x"}], "profile": "work"},
    {"action": "install", "items": [{"kind": "plugin", "id": "x", "sha": "b" * 40}]},
    {"action": "install", "items": [{"kind": "plugin", "id": "x", "url": "https://evil.example/x"}]},
])
def test_a_source_version_or_profile_from_the_model_is_refused_before_anything_runs(args):
    installer, card = FakeInstaller([_entry("x")]), _card(_approve())
    out = json.loads(manage_catalog(args, session_id="s1", connection_callback=card, card_surface=True,
                                    installer=installer))
    assert "error" in out and not card.seen and not installer.installs


def test_off_the_desktop_the_result_points_at_the_cli_and_opens_no_card():
    installer, card = FakeInstaller([_entry("x")]), _card(_approve())
    for callback, surface in ((None, True), (card, False)):
        out = json.loads(manage_catalog({"action": "install", "items": [{"kind": "plugin", "id": "x"}]},
                                        session_id="s1", connection_callback=callback, card_surface=surface,
                                        installer=installer))
        assert out["error"] == NOT_HERE
    assert not card.seen and not installer.installs and live.current("s1") is None


def test_unknown_and_unsupported_ids_fail_at_once_with_the_reason_and_are_never_installed():
    installer = FakeInstaller([_entry("nvidia-app", platforms=["windows"])],
                              refuse={"nvidia-app": "Plugin 'nvidia-app' is unavailable on darwin; "
                                                    "supported platforms: windows."})
    card = _card(lambda payload: {"settled_by": "continue"})
    out = _install([{"kind": "plugin", "id": "nope"}, {"kind": "plugin", "id": "nvidia-app"}], installer, card)
    assert not card.seen  # every row is already done: the model reads the reasons, no card waits
    rows = {t["name"]: t for t in out["targets"]}
    assert {t["state"] for t in rows.values()} == {TargetState.failed.value}
    assert "catalog" in rows["nope"]["detail"]
    assert "unavailable on darwin" in rows["nvidia-app"]["detail"]
    assert rows["nvidia-app"]["display"] == "Nvidia App" and rows["nvidia-app"]["platforms"] == ["windows"]
    assert not installer.installs


def test_an_approved_row_installs_into_the_chats_profile_and_lists_the_live_tools():
    from hermes_cli.profiles import create_profile
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    b_home = Path(create_profile("b", no_alias=True))
    installer = FakeInstaller([_entry("blender")])
    card = _card(_approve(), profile_home=str(b_home))
    token = set_hermes_home_override(str(b_home))
    try:
        out = _install([{"kind": "plugin", "id": "blender"}], installer, card)
    finally:
        reset_hermes_home_override(token)
    (row,) = out["targets"]
    assert card.seen[0]["targets"][0]["state"] == TargetState.pending.value  # nothing ran before the card
    assert row["state"] == TargetState.connected.value and row["target_profile"] == "b"
    assert row["tools"] == ["mcp__blender__status", "mcp__blender__launch"]
    (call,) = installer.installs
    assert (call["force"], call["enable"], call["ref"]) == (False, True, None)
    assert call["home"].resolve() == b_home.resolve()


def test_advanced_values_pick_the_profile_force_and_pin(tmp_path):
    from hermes_cli.profiles import create_profile

    work = create_profile("work", no_alias=True)
    installer = FakeInstaller([_entry("blender")])
    pin = "c" * 40
    out = _install([{"kind": "plugin", "id": "blender"}], installer,
                   _card(_approve({"target_profile": "work", "force": "1", "enable": "0", "ref": pin})))
    (row,) = out["targets"]
    (call,) = installer.installs
    assert (call["force"], call["enable"], call["ref"]) == (True, False, pin)
    assert call["home"].resolve() == Path(work).resolve() and row["target_profile"] == "work"
    # The row carries what the user approved, so a Try again after settle repeats it.
    assert row["approved"] == {"force": True, "enable": False, "ref": pin}
    assert row["enabled"] is False and "not enabled" in row["detail"]


def test_a_failed_row_keeps_its_reason_and_try_again_works_while_the_card_is_open():
    installer = FakeInstaller([_entry("blender"), _entry("krita")], install_error="clone failed: network down")
    card = _card(lambda payload: {"targets": [{"name": "blender", "status": "approved", "env": None}]})

    def wait_for(state):
        for _ in range(200):
            operation = live.current("s1")
            if operation is not None and operation.targets[0].state == state:
                return operation
            threading.Event().wait(0.01)
        raise AssertionError(f"blender never reached {state}")

    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("out", json.loads(manage_catalog(
            {"action": "install", "items": [{"kind": "plugin", "id": "blender"}, {"kind": "plugin", "id": "krita"}]},
            session_id="s1", connection_callback=card, card_surface=True, installer=installer))))
        thread.start()
        operation = wait_for(TargetState.failed)
        assert operation.targets[0].detail == "clone failed: network down"
        assert not operation.settled  # krita still waits on the user
        installer.install_error = ""
        apply_answer(operation, json.dumps({"targets": [{"name": "blender", "status": "approved"}]}))
        wait_for(TargetState.connected)
        apply_answer(operation, json.dumps({"targets": [{"name": "krita", "status": "skipped"}]}))
        thread.join(5)
    assert result["out"]["targets"][0]["state"] == TargetState.connected.value
    assert len(installer.installs) == 2

    # A failed row does not hold the turn to the deadline: once it is the last open row, the
    # operation settles and the row keeps its state and reason.
    installer.install_error = "clone failed: network down"
    out = _install([{"kind": "plugin", "id": "blender"}], installer, _card(_approve()))
    (row,) = out["targets"]
    assert out["settled_by"] == "all_resolved"
    assert (row["state"], row["detail"]) == (TargetState.failed.value, "clone failed: network down")
