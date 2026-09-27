"""Real children started by the production spawn sites observe the secret scrub."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.tools._child_env_fixtures import child_env  # noqa: F401

_TIER1 = ("TELEGRAM_BOT_TOKEN", "GATEWAY_RELAY_SECRET")
_PROVIDER = "OPENAI_API_KEY"


def _plant(monkeypatch):
    for name in (*_TIER1, _PROVIDER):
        monkeypatch.setenv(name, f"fake-{name.lower()}")


def _probe_script(path: Path, out: Path, names) -> Path:
    path.write_text(
        f"#!{sys.executable}\nimport json, os\n"
        f"open({str(out)!r}, 'w').write(json.dumps({{n: os.environ.get(n) for n in {list(names)!r}}}))\n",
        encoding="utf-8")
    path.chmod(0o755)
    return path


def _compute_host_seen(child_env, monkeypatch, names):
    from tui_gateway.host_supervisor import HostSupervisor

    hello = ("import json, os, sys; print(json.dumps({'type': 'hello', 'seen': "
             f"{{n: os.environ.get(n) for n in {names!r}}}}}), flush=True); sys.stdin.readline()")
    sup = HostSupervisor(registry_path=child_env / "host.json", argv=[sys.executable, "-c", hello],
                         cwd=child_env, expected_build_sha="unknown", autostart=False)
    try:
        sup.start()
        return sup._hello["seen"]
    finally:
        sup.shutdown()


def _openviking_server_seen(child_env, monkeypatch, names):
    import subprocess

    import plugins.memory.openviking as ov

    out = child_env / "seen.json"
    probe = _probe_script(child_env / "openviking-server", out, names)
    monkeypatch.setattr(ov, "_local_openviking_port_is_open", lambda host, port: False)
    monkeypatch.setattr(ov.shutil, "which", lambda name: str(probe))
    real_popen, children = subprocess.Popen, []

    def _record(*args, **kwargs):
        children.append(real_popen(*args, **kwargs))
        return children[-1]

    monkeypatch.setattr(ov.subprocess, "Popen", _record)
    state, _ = ov._start_local_openviking_server("http://127.0.0.1:1933")
    assert state == ov._LOCAL_SERVER_STARTED
    children[0].wait(timeout=30)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("site", [
    "compute_host", pytest.param("openviking_server", marks=pytest.mark.platforms("posix"))])
def test_credentialed_children_keep_provider_keys_but_never_tier1_secrets(child_env, monkeypatch, site):
    # Both children call model providers (the turn process; openviking-server's embedding/VLM
    # models), so provider keys pass by design. Bot and relay tokens never do.
    _plant(monkeypatch)
    if site == "compute_host":
        own = {"HERMES_COMPUTE_HOST_HEARTBEAT_SECS": "15"}
        seen = _compute_host_seen(child_env, monkeypatch, [*_TIER1, _PROVIDER, *own])
    else:
        # The server finds ov.conf through OPENVIKING_CONFIG_FILE or HOME; Hermes' PYTHONPATH
        # would shadow its own site-packages (#78153).
        monkeypatch.setenv("OPENVIKING_CONFIG_FILE", str(child_env / "ov.conf"))
        monkeypatch.setenv("PYTHONPATH", str(child_env / "hermes-venv"))
        own = {"OPENVIKING_CONFIG_FILE": str(child_env / "ov.conf"), "HOME": str(child_env), "PYTHONPATH": None}
        seen = _openviking_server_seen(child_env, monkeypatch, [*_TIER1, _PROVIDER, *own])
    assert seen == {"TELEGRAM_BOT_TOKEN": None, "GATEWAY_RELAY_SECRET": None,
                    _PROVIDER: "fake-openai_api_key", **own}


@pytest.mark.platforms("posix")  # the stand-in binaries are shebang scripts
@pytest.mark.parametrize("site", ["lsp_server", "lsp_go_install", "raft_bridge", "buzz_cli"])
def test_third_party_children_never_see_hermes_credentials(child_env, monkeypatch, site):
    _plant(monkeypatch)
    out = child_env / "seen.json"
    own = {"lsp_server": "LSP_OWN_SETTING", "lsp_go_install": "GOBIN", "raft_bridge": "RAFT_CHANNEL_TOKEN",
           "buzz_cli": "BUZZ_PRIVATE_KEY"}[site]
    probe = _probe_script(child_env / "probe", out, [*_TIER1, _PROVIDER, own])

    if site == "lsp_server":
        from agent.lsp.client import LSPClient

        async def _run():
            client = LSPClient(server_id="probe", workspace_root=str(child_env), command=[str(probe)],
                               env={"LSP_OWN_SETTING": "kept"})
            await client._spawn()
            await client._proc.wait()
            for task in (client._reader_task, client._stderr_task):
                task.cancel()
        asyncio.run(_run())
    elif site == "lsp_go_install":
        from agent.lsp import install

        with patch.object(install.shutil, "which", return_value=str(probe)):
            install._install_go("example.com/probe@latest", "probe")
    elif site == "buzz_cli":
        from plugins.platforms.buzz.adapter import _exec_buzz

        asyncio.run(_exec_buzz(str(probe), [], relay_url="wss://relay.invalid", private_key="buzz-own"))
    else:
        from gateway.config import PlatformConfig
        from plugins.platforms.raft.adapter import RaftAdapter

        monkeypatch.setenv("RAFT_PROFILE", "probe")
        config = PlatformConfig(enabled=True, extra={"bridge_token": "bridge-own", "runtime_session": "default", "port": 0})
        adapter = RaftAdapter(config)
        with patch("plugins.platforms.raft.adapter.shutil.which", return_value=str(probe)):
            adapter._spawn_bridge(4321)
        adapter._bridge_process.wait(timeout=30)

    seen = json.loads(out.read_text(encoding="utf-8"))
    assert {k: seen[k] for k in (*_TIER1, _PROVIDER)} == dict.fromkeys((*_TIER1, _PROVIDER))
    assert seen[own]  # the child's own configuration still arrives
