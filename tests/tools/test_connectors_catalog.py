"""``manage_catalog`` (the setup profile's catalog install through the connection card).

Contracts:
- the tool reaches only a session of the ``role: setup`` profile, and only a desktop chat draws it
- the model sends catalog ids and an action; every other key is refused before anything runs
- an id the catalog does not know, or a plugin this OS cannot run, is drawn failed and never installed
- an approved row installs into ``default`` (or the profile the Advanced modal named) and reports
  what went live; the card never witnesses the outcome
"""

import contextlib
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

    def install_plugin(self, name, *, force, enable, ref):
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


def test_only_a_setup_profile_session_selects_the_tool(tmp_path, monkeypatch):
    import model_tools
    from hermes_cli.profiles import write_profile_meta
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = {}
    for name, role in (("default", None), ("hermes-setup", "setup")):
        homes[name] = tmp_path / name
        homes[name].mkdir()
        if role:
            write_profile_meta(homes[name], role=role)
    for name, expected in (("default", False), ("hermes-setup", True)):
        token = set_hermes_home_override(str(homes[name]))
        try:
            # Even an explicit request for the toolset does not reach a profile without the role.
            names = model_tools._select_tool_names(["setup", "web"], None, quiet_mode=True)
        finally:
            reset_hermes_home_override(token)
        assert ("manage_catalog" in names) is expected, name
    # The setup guide's one tool is sent directly, never hidden behind tool_search.
    from tools.tool_search import is_deferrable_tool_name
    assert not is_deferrable_tool_name("manage_catalog")


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


def test_unknown_and_unsupported_ids_are_drawn_failed_and_never_installed():
    installer = FakeInstaller([_entry("nvidia-app", platforms=["windows"])],
                              refuse={"nvidia-app": "Plugin 'nvidia-app' is unavailable on darwin; "
                                                    "supported platforms: windows."})
    card = _card(lambda payload: {"settled_by": "continue"})
    out = _install([{"kind": "plugin", "id": "nope"}, {"kind": "plugin", "id": "nvidia-app"}], installer, card)
    drawn = {t["name"]: t for t in card.seen[0]["targets"]}
    assert {t["state"] for t in drawn.values()} == {TargetState.failed.value}  # drawn with the reason
    rows = {t["name"]: t for t in out["targets"]}
    assert "catalog" in rows["nope"]["detail"]
    assert "unavailable on darwin" in rows["nvidia-app"]["detail"]
    assert rows["nvidia-app"]["display"] == "Nvidia App" and rows["nvidia-app"]["platforms"] == ["windows"]
    assert not installer.installs


def test_an_approved_row_installs_into_default_and_lists_the_live_tools(tmp_path):
    from hermes_cli.profiles import get_profile_dir, write_profile_meta
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    setup_home = tmp_path / "hermes-setup"
    setup_home.mkdir()
    write_profile_meta(setup_home, role="setup")
    installer = FakeInstaller([_entry("blender")])
    card = _card(_approve(), profile_home=str(setup_home))
    token = set_hermes_home_override(str(setup_home))  # the call comes from the setup chat
    try:
        out = _install([{"kind": "plugin", "id": "blender"}], installer, card)
    finally:
        reset_hermes_home_override(token)
    (row,) = out["targets"]
    assert card.seen[0]["targets"][0]["state"] == TargetState.pending.value  # nothing ran before the card
    assert row["state"] == TargetState.connected.value and row["target_profile"] == "default"
    assert row["tools"] == ["mcp__blender__status", "mcp__blender__launch"]
    (call,) = installer.installs
    assert (call["force"], call["enable"], call["ref"]) == (False, True, None)
    assert call["home"].resolve() == Path(get_profile_dir("default")).resolve() != setup_home.resolve()


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
    assert "not enabled" in row["detail"]


def test_a_failed_install_stays_failed_until_the_user_tries_again():
    installer = FakeInstaller([_entry("blender")], install_error="clone failed: network down")
    attempts = []

    def answer(payload):
        (target,) = payload["targets"]
        attempts.append(target["state"])
        return {"targets": [{"name": "blender", "status": "approved", "env": None}]}

    card = _card(answer)
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("out", json.loads(manage_catalog(
            {"action": "install", "items": [{"kind": "plugin", "id": "blender"}]}, session_id="s1",
            connection_callback=card, card_surface=True, installer=installer))))
        thread.start()
        for _ in range(200):
            operation = live.current("s1")
            if operation is not None and operation.targets[0].state == TargetState.failed:
                break
            threading.Event().wait(0.01)
        assert operation.targets[0].detail == "clone failed: network down"
        installer.install_error = ""
        apply_answer(operation, json.dumps({"targets": [{"name": "blender", "status": "approved"}]}))
        thread.join(5)
    assert result["out"]["targets"][0]["state"] == TargetState.connected.value
    assert len(installer.installs) == 2


# ── Credentials the card collected ──────────────────────────────────────────
#
# They have to be on disk before the install: enabling a plugin activates it in this process and its
# MCP servers read their keys from the profile's `.env`. So a failed install has to take them back
# out — nothing else ever will, and the row shows only the install error. Driven through
# ``_Runner._install`` directly: the ordering is the contract, and the card machinery is covered above.

@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME whose `.env` the install writes into, with the profile hop stubbed out."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.connectors import catalog as catalog_mod

    monkeypatch.setattr(catalog_mod, "target_scope", lambda profile: contextlib.nullcontext())
    token = set_hermes_home_override(str(tmp_path))
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def _env_values():
    from hermes_cli.config import invalidate_env_cache, load_env

    invalidate_env_cache()
    return load_env() or {}


def _run_install(installer, env):
    from tools.connectors.catalog import _Runner
    from tools.connectors.operation import Target

    runner = _Runner(installer)
    return runner._install(Target(name="blender", kind="plugin", action="install"), env)


def test_a_failed_install_does_not_leave_its_credential_behind(home):
    installer = FakeInstaller([_entry("blender")], install_error="clone failed: network down")

    with pytest.raises(RuntimeError, match="network down"):
        _run_install(installer, {"BLENDER_API_KEY": "sk-typed-by-the-user"})

    assert "BLENDER_API_KEY" not in _env_values()


def test_a_failed_install_restores_the_value_the_key_had_before(home):
    from hermes_cli.config import save_env_value

    save_env_value("BLENDER_API_KEY", "sk-the-one-that-works")
    installer = FakeInstaller([_entry("blender")], install_error="the security scan blocked this plugin")

    with pytest.raises(RuntimeError):
        _run_install(installer, {"BLENDER_API_KEY": "sk-typed-by-the-user"})

    assert _env_values().get("BLENDER_API_KEY") == "sk-the-one-that-works"


def test_a_successful_install_keeps_the_credential(home):
    installer = FakeInstaller([_entry("blender")])

    result = _run_install(installer, {"BLENDER_API_KEY": "sk-typed-by-the-user"})

    assert result["ok"] and _env_values().get("BLENDER_API_KEY") == "sk-typed-by-the-user"


def test_the_credential_is_on_disk_while_the_install_runs(home):
    """Activation happens inside install_plugin and reads the key from `.env`, so it must be there."""
    seen = {}

    class WatchingInstaller(FakeInstaller):
        def install_plugin(self, name, *, force, enable, ref):
            seen["during"] = _env_values().get("BLENDER_API_KEY")
            return super().install_plugin(name, force=force, enable=enable, ref=ref)

    _run_install(WatchingInstaller([_entry("blender")]), {"BLENDER_API_KEY": "sk-typed-by-the-user"})

    assert seen["during"] == "sk-typed-by-the-user"


def test_a_failed_skill_install_also_rolls_back(home):
    class ExplodingSkills(FakeInstaller):
        def install_skill(self, identifier, *, force):
            raise RuntimeError("the hub refused the bundle")

    installer = ExplodingSkills([_entry("blender")])
    installer.facts = {}
    from tools.connectors.catalog import _Runner
    from tools.connectors.operation import Target

    runner = _Runner(installer)
    runner.facts = {"blender": {"identifier": "blender"}}
    with pytest.raises(RuntimeError, match="refused the bundle"):
        runner._install(Target(name="blender", kind="skill", action="install"),
                        {"BLENDER_API_KEY": "sk-typed-by-the-user"})

    assert "BLENDER_API_KEY" not in _env_values()


def test_a_rejected_variable_name_takes_the_keys_written_before_it_back_out(home):
    """The write itself can fail partway: `_save_credentials` validates each name as it goes."""
    installer = FakeInstaller([_entry("blender")])

    with pytest.raises(ValueError):
        _run_install(installer, {"BLENDER_API_KEY": "sk-typed-by-the-user", "not-a-valid-name": "x"})

    assert "BLENDER_API_KEY" not in _env_values()
    assert not installer.installs  # and the install never ran
