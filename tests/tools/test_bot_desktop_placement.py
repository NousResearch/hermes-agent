"""Bot Desktop placement: the screen (and with it computer_use and the browser) follows the terminal backend.

Two invariants, both about the boundary a sandboxed user chose:
  1. A sandbox backend that cannot host a screen never falls back to the gateway host silently; it refuses
     unless ``bot_desktop.placement: gateway`` opts in. Docker/ssh/singularity resolve to the sandbox.
  2. Every RFB byte between the pane and a sandbox-hosted Xvnc, and cua-driver's MCP stdio, go through the
     terminal backend's exec prefix, never a host socket or a host binary.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from tools.bot_desktop import placement, runtime
from tools.environments import streams


@pytest.mark.parametrize(
    ("setting", "backend", "expected"),
    [
        ("auto", "local", placement.GATEWAY),
        ("auto", "docker", placement.TERMINAL),
        ("auto", "ssh", placement.TERMINAL),
        ("auto", "singularity", placement.TERMINAL),
        ("auto", "modal", placement.REFUSED),
        ("auto", "daytona", placement.REFUSED),
        ("gateway", "modal", placement.GATEWAY),
        ("gateway", "docker", placement.GATEWAY),
        ("terminal", "docker", placement.TERMINAL),
        ("terminal", "local", placement.GATEWAY),
    ],
)
def test_placement_follows_the_terminal_backend_and_refuses_unhostable_sandboxes(monkeypatch, setting, backend, expected):
    monkeypatch.setattr(placement, "_setting", lambda: setting)
    monkeypatch.setattr(placement, "_terminal_backend", lambda: backend)
    where = placement.resolve()
    assert where.where == expected
    if expected == placement.REFUSED:
        assert backend in where.reason and "placement: gateway" in where.reason


def test_refused_placement_blocks_start_and_names_the_opt_in(monkeypatch):
    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "modal")
    with pytest.raises(RuntimeError, match="placement: gateway"):
        runtime.start()


class _FakeDocker:
    """Stand-in for DockerEnvironment: only what ``streams.exec_prefix`` reads."""
    _docker_exe = "docker"
    _container_id = "c0ffee"

    def get_temp_dir(self):
        return "/tmp"  # no-tmp: ok — sandbox-side path


def test_sandbox_rfb_and_cua_ride_the_exec_prefix(monkeypatch):
    from tools.environments.docker import DockerEnvironment
    env = _FakeDocker()
    monkeypatch.setattr(streams, "exec_prefix", lambda e, *, user=None, interactive=True:
                        ["docker", "exec", "-i", "-u", user, e._container_id] if user else ["docker", "exec", "-i", e._container_id])
    spawned: list[list[str]] = []

    class _P:
        stdin = stdout = None
        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: spawned.append(argv) or _P())
    from tools.bot_desktop import sandbox_host
    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: "pn")
    sandbox_host.open_rfb_stream(env, "p1")
    assert spawned[0][:6] == ["docker", "exec", "-i", "-u", "pn", "c0ffee"]
    assert "rfb.sock" in spawned[0][-1] and "python3" in spawned[0][-1]

    command, args = sandbox_host.cua_mcp_invocation(env, "p1", {"DISPLAY": ":20"})
    assert command == "docker" and args[:5] == ["exec", "-i", "-u", "pn", "c0ffee"]
    assert "cua-driver mcp" in args[-1] and "export DISPLAY=:20" in args[-1]
    assert not isinstance(env, DockerEnvironment)  # the fake never touched a real daemon


def test_sandbox_status_offers_the_image_switch_instead_of_a_config_hint(monkeypatch):
    """A docker sandbox kept on the previous default image (no desktop stack) is the common
    upgraded-install case: the blocker becomes the switch offer and ``image_switch`` carries what
    the pane needs to approve it. Without a pending switch the plain image hint stands."""
    from hermes_cli import sandbox_image_switch as sw
    from tools.bot_desktop import sandbox_host

    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "docker")
    monkeypatch.setattr(placement, "terminal_environment", lambda *, create=True: _FakeDocker())
    monkeypatch.setattr(sandbox_host, "missing_binaries", lambda env: ["Xvnc"])
    monkeypatch.setattr(sandbox_host, "published_env", lambda env, profile: {})

    monkeypatch.setattr(sw, "pending", lambda: sw.PendingSwitch("old/base:1", "nousresearch/hermes-sandbox:desktop", ["hermes-a"]))
    st = runtime.status()
    assert st.image_switch == {"current_image": "old/base:1", "target_image": "nousresearch/hermes-sandbox:desktop", "containers": 1}
    assert "old/base:1" in st.blocker and "/root and /workspace" in st.blocker
    assert st.installed and not st.running

    monkeypatch.setattr(sw, "pending", lambda: None)
    st = runtime.status()
    assert st.image_switch is None
    assert "Xvnc" in st.blocker and "placement: gateway" in st.blocker


def test_remote_command_survives_the_ssh_remote_shell_reparse():
    """OpenSSH joins the words after the host with spaces and the remote login shell splits them again, so
    the ``bash -c <script>`` must travel as ONE quoted word there; docker/apptainer pass argv through and
    must not get that extra quoting. Both shapes are executed through a real bash the way each client's
    remote side would run them."""
    from tools.environments.ssh import SSHEnvironment

    class _FakeSSH(SSHEnvironment):
        def __init__(self):  # no daemon, no key: only the exec prefix is exercised
            self.host, self.user, self.port = "sandbox.example", "pn", 22

        def _build_ssh_command(self, extra_args=None, send_env=()):
            return ["ssh", f"{self.user}@{self.host}"]

    argv = ["printf", "%s|%s", "hello world", "it's $HOME"]
    child_env = {"DISPLAY": ":20", "AGENT_BROWSER_ARGS": "--a=1,--b='x y'"}

    over_ssh = streams.remote_command(_FakeSSH(), argv, child_env=child_env)
    assert over_ssh[:2] == ["ssh", "pn@sandbox.example"] and len(over_ssh) == 3
    remote_word = " ".join(over_ssh[2:])  # what sshd hands the login shell
    out = subprocess.run(["bash", "-c", remote_word], capture_output=True, text=True, check=True).stdout
    assert out == "hello world|it's $HOME"
    env_out = subprocess.run(["bash", "-c", " ".join(streams.remote_command(
        _FakeSSH(), ["sh", "-c", 'printf "%s" "$AGENT_BROWSER_ARGS"'], child_env=child_env)[2:])],
        capture_output=True, text=True, check=True).stdout
    assert env_out == "--a=1,--b='x y'"

    docker_shape = streams.remote_argv(["docker", "exec", "-i", "c0ffee"], argv, env=child_env)
    assert docker_shape[:6] == ["docker", "exec", "-i", "c0ffee", "bash", "-c"] and len(docker_shape) == 7
    out = subprocess.run(["bash", "-c", docker_shape[6]], capture_output=True, text=True, check=True).stdout
    assert out == "hello world|it's $HOME"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(runtime, "state_dir", lambda: home / "bot-desktop" / "default")
    return home


def _placed(monkeypatch, setting, backend):
    monkeypatch.setattr(placement, "_setting", lambda: setting)
    monkeypatch.setattr(placement, "_terminal_backend", lambda: backend)


def test_terminal_placement_starts_the_sandbox_screen_instead_of_running_on_the_host(monkeypatch, isolated_home):
    """The placement boundary is policy, not liveness: with ``placement: terminal`` and the screen not yet
    up, the browser preflight and the CUA invocation bring the sandbox screen up (no ``auto_start`` opt-in
    needed inside the user's own sandbox) and route there. Neither hands the tool a host command."""
    from tools import browser_tool_session as bts
    from tools.computer_use import cua_backend as cb

    _placed(monkeypatch, "terminal", "docker")
    started: list[str] = []
    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: bool(started))
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":20", "XAUTHORITY": "/x"} if started else {})
    monkeypatch.setattr(runtime, "start", lambda **kw: started.append("started"))
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: _FakeDocker())
    monkeypatch.setattr(runtime, "touch_activity", lambda: None)

    assert bts._browser_command_preflight() == {"browser_cmd": "agent-browser"}
    assert started == ["started"]

    from tools.bot_desktop import sandbox_host
    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: "pn")
    monkeypatch.setattr(streams, "exec_prefix", lambda e, *, user=None, interactive=True:
                        ["docker", "exec", "-i", "-u", user, e._container_id])
    (command, args), child_env = cb.sandbox_mcp_invocation()
    assert command == "docker" and "cua-driver mcp" in args[-1] and "export DISPLAY=:20" in args[-1]
    assert set(child_env) == {"PATH"}  # the host driver's env never reaches the sandbox driver


def test_refused_placement_never_falls_back_to_the_host(monkeypatch, isolated_home):
    from tools import browser_tool_session as bts
    from tools.computer_use import cua_backend as cb

    _placed(monkeypatch, "auto", "modal")
    res = bts._browser_command_preflight()
    assert res["success"] is False and "placement: gateway" in res["error"]
    with pytest.raises(RuntimeError, match="placement: gateway"):
        cb.sandbox_mcp_invocation()


def test_sandbox_screen_that_cannot_come_up_is_an_error_not_a_host_browser(monkeypatch, isolated_home):
    from tools import browser_tool_session as bts

    _placed(monkeypatch, "terminal", "docker")
    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: False)
    monkeypatch.setattr(runtime, "published_env", lambda: {})

    def _boom(**kw):
        raise RuntimeError("Bot Desktop needs Xvnc inside the terminal backend's sandbox")
    monkeypatch.setattr(runtime, "start", _boom)
    res = bts._browser_command_preflight()
    assert res["success"] is False and "Xvnc" in res["error"]


def test_start_adopting_a_screen_the_sandbox_kept_records_the_marker(monkeypatch, isolated_home):
    """Host state can vanish while the sandbox keeps its Xvnc (fresh HERMES_HOME, a stop() whose kill missed
    the launcher). start() finding the display already published must record it like a fresh launch, or
    status/thumbnail/stop never learn the screen is ours (found live against an ssh sandbox)."""
    from tools.bot_desktop import sandbox_host

    monkeypatch.setattr(sandbox_host, "_published", lambda env, rdir: {"DISPLAY": ":20", "XAUTHORITY": f"{rdir}/Xauthority"})
    monkeypatch.setattr(sandbox_host, "_remote_dir", lambda env, profile: f"/scratch/hermes-bot-desktop/{profile}")
    env = _FakeDocker()
    assert sandbox_host.start(env, "default", geometry="1280x800")["DISPLAY"] == ":20"
    marker = json.loads(sandbox_host._marker().read_text())
    assert marker["display"] == ":20" and marker["container"] == env._container_id and marker["profile"] == "default"


def test_marker_survives_a_gateway_restart_while_the_container_lives(monkeypatch, isolated_home):
    """After a restart the process-local terminal registry is empty but the persisted container (and the
    Xvnc in it) is still running. The marker must NOT be dropped on that observation: liveness comes from
    the recorded container, stop() re-attaches to it, and only a container that is gone drops the marker."""
    from tools.bot_desktop import sandbox_host

    _placed(monkeypatch, "terminal", "docker")
    sandbox_host._marker().parent.mkdir(parents=True)
    sandbox_host._marker().write_text('{"display": ":20", "dir": "/tmp/hermes-bot-desktop/default", "profile": "default", '
                                      '"backend": "DockerEnvironment", "container": "c0ffee", "docker": "docker"}')
    inspected: list[list[str]] = []
    alive = {"running": True}

    def _inspect(argv, **kw):
        inspected.append(argv)
        return subprocess.CompletedProcess(argv, 0 if alive["running"] else 1, "true\n" if alive["running"] else "", "")
    monkeypatch.setattr(sandbox_host.subprocess, "run", _inspect)
    sandbox_host._ALIVE_CACHE.clear()

    created: list[bool] = []

    def _env(*, create):
        created.append(create)
        return _FakeDocker() if create else None  # nothing registered in this (fresh) process
    monkeypatch.setattr(runtime, "_sandbox_env", _env)

    assert runtime.sandbox_screen_running() is True
    assert sandbox_host._marker().exists() and inspected[0][:3] == ["docker", "inspect", "-f"]

    stopped: list[object] = []
    monkeypatch.setattr(sandbox_host, "stop", lambda env, profile: stopped.append(env) or True)
    assert runtime.stop() is True
    assert isinstance(stopped[0], _FakeDocker) and created[-1] is True  # re-attached, then stopped THAT sandbox
    assert not sandbox_host._marker().exists()

    # The container was removed out from under us: the marker is stale and goes; nothing is (re)built.
    sandbox_host._marker().write_text('{"display": ":20", "backend": "DockerEnvironment", "container": "dead00", "docker": "docker"}')
    alive["running"] = False
    sandbox_host._ALIVE_CACHE.clear()
    created.clear()
    assert runtime.sandbox_screen_running() is False
    assert not sandbox_host._marker().exists() and True not in created


def test_stop_and_status_follow_the_recorded_owner_not_the_current_setting(monkeypatch, isolated_home):
    """Placement moved to ``gateway`` while a sandbox screen is still up: status keeps reporting the sandbox
    screen and stop() takes it down there, instead of both looking at the (empty) host."""
    from tools.bot_desktop import sandbox_host

    _placed(monkeypatch, "gateway", "docker")
    sandbox_host._marker().parent.mkdir(parents=True)
    sandbox_host._marker().write_text('{"display": ":20", "backend": "DockerEnvironment", "container": "c0ffee", "docker": "docker"}')
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda marker: True)
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: _FakeDocker())
    monkeypatch.setattr(sandbox_host, "missing_binaries", lambda env: [])
    monkeypatch.setattr(sandbox_host, "published_env", lambda env, profile: {"DISPLAY": ":20"})

    st = runtime.status()
    assert st.running and st.display == ":20" and st.placement.startswith("terminal:")

    stopped: list[object] = []
    monkeypatch.setattr(sandbox_host, "stop", lambda env, profile: stopped.append(env) or True)
    assert runtime.stop() is True and len(stopped) == 1


def test_forward_port_carries_bytes_both_ways_through_the_exec_stream(monkeypatch):
    """The sandbox's loopback (Chromium's CDP port) is reached from the host through a local listener whose
    connections ride an exec stream running the in-sandbox relay. An empty exec prefix runs that relay
    locally, so the whole path (listener → Popen relay → TCP target → back) is real."""
    import socket
    import threading

    monkeypatch.setattr(streams, "exec_prefix", lambda env, *, user=None, interactive=True: [])
    monkeypatch.setattr(streams, "shell_joined", lambda env: False)

    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    target_port = srv.getsockname()[1]

    def echo_upper():
        conn, _ = srv.accept()
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                conn.sendall(data.upper())
    threading.Thread(target=echo_upper, daemon=True).start()

    env = _FakeDocker()
    local = streams.forward_port(env, target_port)
    assert local != target_port
    assert streams.forward_port(env, target_port) == local, "one listener per (env, port)"

    with socket.create_connection(("127.0.0.1", local), timeout=10) as c:
        c.sendall(b"GET /json/version HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        c.settimeout(10)
        got = b""
        while b"\r\n\r\n" not in got:
            got += c.recv(65536)
    assert got.startswith(b"GET /JSON/VERSION HTTP/1.1")


def test_sandbox_cdp_endpoint_is_rewritten_to_the_forwarded_local_port(monkeypatch):
    from tools import browser_use_cli as buc
    from tools import browser_tool_session as bts

    monkeypatch.setattr(bts, "_browser_in_sandbox", lambda: True)
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: _FakeDocker())
    from tools.bot_desktop import sandbox_host
    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: "pn")
    monkeypatch.setattr(streams, "forward_port", lambda env, port, *, remote_host="127.0.0.1", user=None: 45000 + port % 7)
    assert buc._reach_sandbox_cdp("ws://127.0.0.1:9223/devtools/browser/abc") == "ws://127.0.0.1:45004/devtools/browser/abc"
    assert buc._reach_sandbox_cdp("wss://cloud.example/session/1") == "wss://cloud.example/session/1"
    monkeypatch.setattr(bts, "_browser_in_sandbox", lambda: False)
    assert buc._reach_sandbox_cdp("ws://127.0.0.1:9223/x") == "ws://127.0.0.1:9223/x"
