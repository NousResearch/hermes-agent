import importlib.abc
import importlib.util
import io
import subprocess
import sys
import types
from pathlib import Path

from tui_gateway import slash_worker


def test_module_import_does_not_load_cli_before_watchdog_can_start():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; assert 'cli' not in sys.modules; "
            "import tui_gateway.slash_worker; assert 'cli' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=10,
    )


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, getppid=lambda: 999999) is True


def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert slash_worker._is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


def test_main_arms_watchdog_then_imports_cli_before_runtime_prep(monkeypatch):
    parent_pid = 424242
    events = []

    class FakeHermesCLI:
        def __init__(self, **kwargs):
            events.append(("cli", kwargs["resume"]))

    class FakeCliFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        # Record when ``cli`` is actually imported, not just when HermesCLI is built:
        # importing cli loads ~/.hermes/.env, which MCP discovery in runtime prep needs.
        def find_spec(self, name, path=None, target=None):
            return importlib.util.spec_from_loader(name, self) if name == "cli" else None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            events.append(("cli-import",))
            module.HermesCLI = FakeHermesCLI

    monkeypatch.delitem(sys.modules, "cli", raising=False)
    monkeypatch.setattr(sys, "meta_path", [FakeCliFinder(), *sys.meta_path])
    monkeypatch.setattr(
        slash_worker,
        "_start_parent_death_watchdog",
        lambda pid: events.append(("watchdog", pid)),
    )
    monkeypatch.setattr(
        slash_worker,
        "_prepare_slash_worker_runtime",
        lambda: events.append(("runtime",)),
    )

    def unexpected_getppid():
        raise AssertionError("spawn parent PID must come from argv")

    monkeypatch.setattr(slash_worker.os, "getppid", unexpected_getppid)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slash_worker",
            "--session-key",
            "session-1",
            "--parent-pid",
            str(parent_pid),
        ],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    slash_worker.main()

    assert events == [
        ("watchdog", parent_pid),
        ("cli-import",),
        ("runtime",),
        ("cli", "session-1"),
    ]


def test_windows_watchdog_keeps_observed_parent_over_spawn_pid(monkeypatch):
    # A venv python.exe redirector sits between the gateway and this interpreter on
    # Windows, so the gateway PID never equals os.getppid() there; arming on it
    # would make the watchdog kill a healthy worker on its first poll.
    armed = []
    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(HermesCLI=lambda **kw: types.SimpleNamespace()))
    monkeypatch.setattr(slash_worker, "_start_parent_death_watchdog", armed.append)
    monkeypatch.setattr(slash_worker, "_prepare_slash_worker_runtime", lambda: None)
    monkeypatch.setattr(slash_worker.os, "getppid", lambda: 777)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["slash_worker", "--session-key", "s", "--parent-pid", "424242"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    slash_worker.main()

    assert armed == [777]
