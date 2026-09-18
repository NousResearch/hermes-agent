"""SSH backend egress (proxy.ssh_tunnel): tokens ride the tunnel, real keys never leave the host."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.environments import ssh as ssh_env
from tools.environments import ssh_egress
from tools.environments.egress_common import egress_env_overrides
from tools.environments.ssh import SSHEnvironment

REAL_KEY = "sk-or-REAL-SECRET"
TOKEN = "hpx-token-abc123"


def _egress() -> ssh_egress.SSHEgress:
    mapping = SimpleNamespace(real_env_name="OPENROUTER_API_KEY", proxy_token=TOKEN, alias_env_names=())
    return ssh_egress.SSHEgress(
        tunnel_port=9090, bind_host="172.17.0.1", ca_host_path=Path("/host/ca.crt"),
        env_overrides=egress_env_overrides("127.0.0.1", 9090, "$HOME/.hermes-egress-ca.crt", [mapping]))


@pytest.fixture
def make_env(monkeypatch):
    uploads: list[tuple[str, str]] = []
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "_run_ssh_checked", lambda self, *a, **k: None)
    monkeypatch.setattr(SSHEnvironment, "_scp_upload",
                        lambda self, host_path, remote_path: uploads.append((Path(host_path).read_text()
                                                                             if host_path.endswith(".env") else host_path,
                                                                             remote_path)))
    monkeypatch.setattr(ssh_env, "FileSyncManager", lambda **kw: SimpleNamespace(sync=lambda **k: None))
    monkeypatch.setenv("OPENROUTER_API_KEY", REAL_KEY)

    def _make(egress):
        monkeypatch.setattr(ssh_env, "ssh_egress_for_env", lambda: egress)
        return SSHEnvironment(host="example.com", user="testuser"), uploads
    return _make


def test_ssh_tunnel_forwards_proxy_and_ships_tokens_not_keys(make_env, monkeypatch):
    env, uploads = make_env(_egress())

    cmd = env._build_ssh_command()
    assert "-R" in cmd and "127.0.0.1:9090:172.17.0.1:9090" in cmd and "127.0.0.1:9091:172.17.0.1:9091" in cmd

    (ca_src, ca_dst), (env_text, env_dst) = uploads
    assert (ca_src, ca_dst) == ("/host/ca.crt", "/home/testuser/.hermes-egress-ca.crt")
    assert env_dst == "/home/testuser/.hermes-egress.env"  # beside, not inside, the synced ~/.hermes
    assert f"export OPENROUTER_API_KEY={TOKEN}" in env_text
    assert 'export SSL_CERT_FILE="$HOME/.hermes-egress-ca.crt"' in env_text
    assert 'export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--use-openssl-ca"' in env_text
    assert "HTTPS_PROXY=http://127.0.0.1:9090" in env_text
    assert REAL_KEY not in env_text

    monkeypatch.setattr(ssh_env, "resolve_passthrough_env", lambda **kw: ({}, set()))
    with patch.object(ssh_env, "_popen_bash") as popen:
        env._run_bash("echo hi")
    argv = popen.call_args.args[0]
    script = argv[-1]
    assert script.startswith("'. /home/testuser/.hermes-egress.env || ")  # sourced before the command
    assert "echo hi" in script and REAL_KEY not in " ".join(argv) and TOKEN not in " ".join(argv)

    # Control: tunnel off → today's behaviour, untouched.
    plain, _ = make_env(None)
    assert "-R" not in plain._build_ssh_command()
    with patch.object(ssh_env, "_popen_bash") as popen:
        plain._run_bash("echo hi")
    assert ".hermes-egress.env" not in popen.call_args.args[0][-1]


def test_ssh_tunnel_refuses_real_key_passthrough_and_half_configured_proxy(make_env, monkeypatch):
    env, _ = make_env(_egress())
    monkeypatch.setattr(ssh_env, "resolve_passthrough_env", lambda **kw: ({"OPENROUTER_API_KEY": REAL_KEY}, set()))
    with patch.object(ssh_env, "_popen_bash") as popen, pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        env._run_bash("echo hi")
    popen.assert_not_called()

    # Opted in but the daemon is down: fail closed, never fall back to real keys.
    stopped = SimpleNamespace(configured=True, pid=None, listening=False, tunnel_port=9090,
                              ca_cert_path=Path("/host/ca.crt"))
    with patch("hermes_cli.config.load_config", return_value={"proxy": {"enabled": True, "ssh_tunnel": True}}), \
         patch("agent.proxy_sources.iron_proxy.get_status", return_value=stopped), \
         pytest.raises(RuntimeError, match="not running"):
        ssh_egress.ssh_egress_for_env()
    with patch("hermes_cli.config.load_config", return_value={"proxy": {"enabled": True}}):
        assert ssh_egress.ssh_egress_for_env() is None  # docker-only posture unless ssh_tunnel is set
