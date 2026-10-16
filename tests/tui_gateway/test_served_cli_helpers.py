"""Served helpers use profile data without importing interactive CLI startup."""

import os
from pathlib import Path
import sys

import hermes_yaml as yaml
import pytest

from tui_gateway.method_ctx import rebind
from tui_gateway.methods_voice import _persist_wake_enabled
from tui_gateway.model_switch import _session_profile_runtime_scope

def test_drop_detection_keeps_cli_startup_state_out_of_served_profiles(tmp_path, monkeypatch):
    from tui_gateway import server

    runtime_scope = rebind(_session_profile_runtime_scope, vars(server))

    home = tmp_path / ".hermes"
    secondary = home / "profiles" / "worker"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_QUIET", "served-probe")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert "cli" not in sys.modules, "probe requires a fresh served process"
    observed = []
    for index, profile in enumerate((home, secondary, home)):
        (profile / "config.yaml").write_text(f"model: {{default: model-{index}}}\n")
        image = profile / "image.png"
        image.write_bytes(b"image")
        sid = f"drop-profile-{index}"
        session = {"agent": None, "profile_home": str(profile), "session_key": sid,
                   "attached_images": [], "image_counter": 0}
        monkeypatch.setitem(server._sessions, sid, session)
        with runtime_scope(session):
            response = server._methods["input.detect_drop"](index, {"session_id": sid, "text": str(image)})
        assert "error" not in response, response
        observed.append((response["result"]["path"], "cli" in sys.modules, os.environ.get("HERMES_QUIET")))
    assert observed == [(str(profile / "image.png"), False, "served-probe")
                        for profile in (home, secondary, home)]


@pytest.mark.parametrize("ignore_config", [False, True])
def test_served_config_writes_and_delegation_reads_keep_profile_scope(
    tmp_path, monkeypatch, ignore_config,
):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from hermes_cli.cli_config_load import _cli_config_defaults
    from hermes_constants import get_hermes_home
    from tools.delegate_tool_config import _load_config
    from tui_gateway import server

    runtime_scope = rebind(_session_profile_runtime_scope, vars(server))
    persist_wake = rebind(_persist_wake_enabled, vars(server))

    home = tmp_path / ".hermes"
    secondary = home / "profiles" / "worker"
    secondary.mkdir(parents=True)
    for path, limit in ((home, 11), (secondary, 23)):
        (path / "config.yaml").write_text(yaml.safe_dump({"delegation": {"max_iterations": limit}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_QUIET", "served-probe")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if ignore_config:
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    else:
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    environment = dict(os.environ)
    observed = []
    token = set_multiplex_context(True)
    try:
        for index, path in enumerate((home, secondary, home)):
            session = {"profile_home": str(path), "session_key": f"config-helper-{index}"}
            with runtime_scope(session):
                written = persist_wake(True)
                limit = _load_config()["max_iterations"]
                saved = yaml.safe_load((path / "config.yaml").read_text())
            observed.append((path, written, saved["wake_word"], limit))
        default = _cli_config_defaults()["delegation"]["max_iterations"]
        assert {"profiles": observed, "cli_imported": "cli" in sys.modules,
                "environment": dict(os.environ), "ambient_home": get_hermes_home()} == {
            "profiles": [(path, True, {"enabled": True}, default if ignore_config else limit)
                         for path, limit in ((home, 11), (secondary, 23), (home, 11))],
            "cli_imported": False, "environment": environment, "ambient_home": home}
    finally:
        reset_multiplex_context(token)
