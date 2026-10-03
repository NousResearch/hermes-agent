"""Interactive plugin skill discovery must bind complete profile runtime scope."""

import os
from pathlib import Path

import pytest

import tui_gateway.server as server


def test_plugin_skill_rpc_paths_isolate_secrets_across_profiles(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_cli import env_loader, plugins
    import hermes_constants
    import tui_gateway.launch_profile_policy as launch_policy

    launch_home = tmp_path / "relocated" / "hermes" / "default"
    profiles_root = launch_home / "profiles"
    homes = {"alpha": profiles_root / "alpha", "beta": profiles_root / "beta"}
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("PLUGIN_SCOPE_TOKEN", "launch-only-test-secret")
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_policy, "_snapshot", None)

    secrets = {"alpha": "alpha-test-secret", "beta": "beta-test-secret"}
    for label, home in homes.items():
        plugin = home / "plugins" / f"scope-{label}"
        skill = plugin / "skills" / "guide" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(f"name: scope-{label}\nversion: 0.1.0\n")
        (plugin / "__init__.py").write_text(
            "from pathlib import Path\nfrom agent.secret_scope import get_secret\n"
            "def register(ctx):\n"
            f"    if get_secret('PLUGIN_SCOPE_TOKEN') != {secrets[label]!r}:\n"
            "        raise RuntimeError('plugin secret scope mismatch')\n"
            "    ctx.register_skill('guide', Path(__file__).parent / 'skills' / 'guide' / 'SKILL.md')\n"
        )
        skill.write_text(
            f"---\nname: guide\ndescription: {label} guide.\n---\n\n{label} instructions.\n"
        )
        (home / "config.yaml").write_text(f"plugins:\n  enabled: [scope-{label}]\n")
        (home / ".env").write_text(f"PLUGIN_SCOPE_TOKEN={secrets[label]}\n")

    env_loader.reset_secret_source_cache()
    plugins._reset_plugin_managers_for_tests()
    try:
        # Session-less palette supports the relocated default root and each named profile.
        default_palette = server.handle_request({
            "id": "palette-default", "method": "commands.catalog", "params": {"profile": "default"},
        })
        assert "result" in default_palette, default_palette
        for label in ("alpha", "beta"):
            palette = server.handle_request({
                "id": f"palette-{label}", "method": "commands.catalog", "params": {"profile": label},
            })
            assert f"/scope-{label}:guide" in palette["result"]["skills"], palette

        # Exercise catalog, completion, slash.exec and command.dispatch A -> B -> A.
        for label in ("alpha", "beta", "alpha"):
            sid = f"scope-{label}-{len(server._sessions)}"
            server._sessions[sid] = {
                "session_key": sid, "agent": None, "profile_home": str(homes[label]),
            }
            command = f"/scope-{label}:guide"
            catalog = server.handle_request({
                "id": "cat", "method": "commands.catalog", "params": {"session_id": sid},
            })
            completed = server.handle_request({
                "id": "complete", "method": "complete.slash",
                "params": {"session_id": sid, "text": f"/scope-{label}:"},
            })
            dispatched = server.handle_request({
                "id": "dispatch", "method": "command.dispatch",
                "params": {"session_id": sid, "name": command[1:], "arg": "apply"},
            })
            slash = server.handle_request({
                "id": "slash", "method": "slash.exec",
                "params": {"session_id": sid, "command": command + " apply"},
            })
            assert command in catalog["result"]["skills"], catalog
            assert any(item["text"] == command[1:] and item["kind"] == "skill"
                       for item in completed["result"]["items"]), completed
            assert dispatched["result"]["type"] == "skill", dispatched
            assert f"{label} instructions." in dispatched["result"]["message"]
            assert slash.get("error", {}).get("code") == 4018, slash
            assert os.environ["PLUGIN_SCOPE_TOKEN"] == "launch-only-test-secret"
            assert hermes_constants.get_hermes_home_override() is None
        with pytest.raises(secret_scope.UnscopedSecretError):
            secret_scope.get_secret("PLUGIN_SCOPE_TOKEN")
    finally:
        plugins._reset_plugin_managers_for_tests()
        env_loader.reset_secret_source_cache()
