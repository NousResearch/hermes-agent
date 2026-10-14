"""Browser/desktop-control daemons Hermes starts itself die with the Hermes process that started them.

Each family in ``_OWNERS`` ran in a session of its own and was stopped only by Python cleanup (stop() /
atexit, or a LATER Hermes's record sweep), so a SIGKILLed or OOM-killed Hermes left it running. Each is now
an owned child (``process_identity.spawn_owned`` / ``own_process_groups``). These tests run the REAL spawn
path in a real owner process against a stand-in binary with the same process topology, SIGKILL the owner
(no Python cleanup of ours runs), and require the child to go, while an identical child owned by a
still-live owner stays.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from tests.waiting import wait_until

pytestmark = [pytest.mark.platforms("linux"),
              # The daemons reparent to init once their owner dies, so cleanup leaves this test's subtree:
              # real signal delivery to a real orphan is the point.
              pytest.mark.live_system_guard_bypass]

_REPO = str(Path(__file__).resolve().parents[2])

# A stand-in daemon: forks one helper that STAYS in its group (renderer / GPU process shape), writes
# both pids, then waits. Ignores SIGHUP like a real daemon. Probe verbs (``help``, ``--help``, ``status``)
# are answered by ``{pre}`` and exit before anything is forked.
_DAEMON = r'''#!{python}
import json, os, signal, subprocess, sys, time
{pre}
signal.signal(signal.SIGHUP, signal.SIG_IGN)
helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], stdin=subprocess.DEVNULL)
receipt = os.environ["STANDIN_RECEIPTS"] + "/" + os.path.basename(sys.argv[0]) + "-" + str(os.getpid())
with open(receipt + ".tmp", "w") as fh:
    json.dump({{"pid": os.getpid(), "helper": helper.pid}}, fh)
os.replace(receipt + ".tmp", receipt + ".json")
{body}
time.sleep(300)
'''

# lightpanda serve: answer CDP discovery on the requested port so launch_lightpanda() reports ready.
_LIGHTPANDA_PRE = 'if sys.argv[1:2] != ["serve"]: sys.exit(0)'
_LIGHTPANDA_BODY = r'''
import http.server, threading
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"webSocketDebuggerUrl": "ws://127.0.0.1:%d/devtools/browser/x"}' % port
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
threading.Thread(target=http.server.HTTPServer(("127.0.0.1", port), H).serve_forever, daemon=True).start()
'''

# Chrome: publish DevToolsActivePort in --user-data-dir as Chrome does.
_CHROME_PRE = 'if not any(a.startswith("--user-data-dir=") for a in sys.argv): sys.exit(0)'
_CHROME_BODY = r'''
data_dir = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
with open(os.path.join(data_dir, "DevToolsActivePort"), "w") as fh:
    fh.write("45678\n/devtools/browser/standin\n")
'''

# cua-driver serve --embedded --socket <path>; `cua-driver status --socket <path>` is the readiness probe.
_CUA_PRE = r'''
if sys.argv[1:2] == ["status"]:
    sys.exit(0 if os.path.exists(sys.argv[sys.argv.index("--socket") + 1]) else 1)
if sys.argv[1:2] != ["serve"]:
    sys.exit(0)
'''
_CUA_BODY = 'open(sys.argv[sys.argv.index("--socket") + 1], "w").close()'

_OWNERS = {
    "lightpanda": r'''
from tools import browser_lightpanda as lp
server, err = lp.launch_lightpanda("owned")
assert server is not None, err
''',
    "real-profile-chrome": r'''
import tools.browser_tool as bt
from tools import browser_tool_real_profile as rp
copy_dir = os.path.join(os.environ["HERMES_HOME"], "copy")
os.makedirs(copy_dir)
port, err = rp._launch_real_profile_chrome(os.environ["STANDIN_BIN"] + "/chrome", copy_dir)
assert port == 45678, err
''',
    "cua-embedded": r'''
from tools.computer_use import cua_backend_driver as drv
drv._resolve_mcp_invocation = lambda cmd, **kw: (cmd, ["mcp"])
from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon
d = _EmbeddedCuaDaemon(os.environ["STANDIN_BIN"] + "/cua-driver", "unrestricted")
d.start()
''',
}

_OWNER = r'''
import json, os, sys, time
sys.path.insert(0, {repo!r})
{body}
print("ready", flush=True)
sys.stdin.read()  # exits when the test closes the pipe: a killed pytest leaves no owner behind
'''

_STANDINS = {"lightpanda": ("lightpanda", _LIGHTPANDA_PRE, _LIGHTPANDA_BODY),
             "real-profile-chrome": ("chrome", _CHROME_PRE, _CHROME_BODY),
             "cua-embedded": ("cua-driver", _CUA_PRE, _CUA_BODY)}


def _alive(pid: int) -> bool:
    """Zombie-aware: a killed non-child lingers as a zombie until init reaps it."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class _World:
    def __init__(self, root: Path):
        self.root = root
        self.bin = root / "bin"
        self.receipts = root / "receipts"
        self.bin.mkdir()
        self.receipts.mkdir()
        for name, pre, body in _STANDINS.values():
            exe = self.bin / name
            exe.write_text(_DAEMON.format(python=sys.executable, pre=pre, body=body), encoding="utf-8")
            exe.chmod(0o755)
        self.owners: list[subprocess.Popen] = []

    def start_owner(self, family: str) -> subprocess.Popen:
        home = self.root / f"home-{family}-{len(self.owners)}"
        home.mkdir()
        env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), STANDIN_BIN=str(self.bin),
                   STANDIN_RECEIPTS=str(self.receipts), PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
                   PYTHONPATH=_REPO)
        # A Hermes launched from a shell leads its own job group; the owner-death supervisor refuses to
        # cover the group of the process it serves, so the owner must not share pytest's.
        owner = subprocess.Popen([sys.executable, "-c", _OWNER.format(repo=_REPO, body=_OWNERS[family])],
                                 env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, cwd=str(home), start_new_session=True)
        self.owners.append(owner)
        line = owner.stdout.readline()
        if line.strip() != "ready":
            owner.kill()
            _, err = owner.communicate(timeout=10)
            pytest.fail(f"{family} owner did not start: {line!r}\n{err[-3000:]}")
        return owner

    def children_of(self, family: str) -> list[dict]:
        name = _STANDINS[family][0]
        return [json.loads(r.read_text(encoding="utf-8-sig")) for r in self.receipts.glob(f"{name}-*.json")]

    def close(self):
        for receipt in self.receipts.glob("*.json"):
            info = json.loads(receipt.read_text(encoding="utf-8-sig"))
            for pid in (info["pid"], info["helper"]):
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)  # windows-footgun: ok — linux-marked test
        for owner in self.owners:
            if owner.poll() is None:
                owner.kill()
            owner.communicate(timeout=10)


@pytest.fixture
def world(tmp_path):
    w = _World(tmp_path)
    yield w
    w.close()


@pytest.mark.parametrize("family", sorted(_OWNERS))
def test_a_sigkilled_owner_takes_its_daemon_tree_and_a_live_owner_keeps_its_own(world, family):
    world.start_owner(family)  # the control: still alive when the victim dies
    control_kids = wait_until(lambda: world.children_of(family), what=f"{family} control child receipt")
    victim = world.start_owner(family)
    kids = wait_until(lambda: [k for k in world.children_of(family) if k not in control_kids],
                      what=f"{family} child receipt")
    assert len(kids) == 1 and len(control_kids) == 1
    child, ctl = kids[0], control_kids[0]
    assert os.getpgid(child["helper"]) == os.getpgid(child["pid"]), "stand-in helper left the daemon's group"  # windows-footgun: ok — linux-marked

    victim.kill()  # SIGKILL: no atexit, no finally, nothing of ours runs in the owner
    victim.wait(timeout=10)

    wait_until(lambda: not _alive(child["pid"]) and not _alive(child["helper"]), timeout=15,
               what=f"the {family} daemon and its helper to die with their SIGKILLed owner")
    assert _alive(ctl["pid"]) and _alive(ctl["helper"]), (
        f"the {family} daemon of a still-live owner was killed")


@pytest.mark.parametrize("started_here", [True, False])
def test_the_real_profile_attach_daemon_is_owned_only_when_this_attach_started_it(tmp_path, monkeypatch, started_here):
    """The ``hermes-real-profile`` session's socket dir is shared by every Hermes: a daemon an earlier,
    still-live Hermes started must stay that Hermes's, never this one's."""
    import subprocess as sp

    import tools.browser_tool as bt
    from tools import browser_tool_install as install
    from tools import browser_tool_real_profile as rp

    socket_dir = tmp_path / f"agent-browser-{bt._REAL_PROFILE_SESSION}"
    socket_dir.mkdir()
    pid_file = socket_dir / f"{bt._REAL_PROFILE_SESSION}.pid"
    pid_file.write_text("1111", encoding="utf-8")
    owned = []

    def fake_cli(argv, timeout, tag):
        if started_here:
            pid_file.write_text("2222", encoding="utf-8")  # the CLI forked a fresh daemon
        return sp.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(install, "_find_agent_browser", lambda: "agent-browser")
    monkeypatch.setattr(rp, "_capture_agent_browser_cli", fake_cli)
    monkeypatch.setattr(rp, "_agent_browser_get_cdp", lambda name: "http://127.0.0.1:45678")
    monkeypatch.setattr(rp._lifecycle, "_own_local_daemon", owned.append)

    cdp, err = rp._attach_agent_browser_to_real_profile(45678, str(tmp_path / "copy"))

    assert (cdp, err) == ("http://127.0.0.1:45678", None)
    assert owned == ([bt._REAL_PROFILE_SESSION] if started_here else [])
