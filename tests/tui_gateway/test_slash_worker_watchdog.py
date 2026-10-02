import importlib.abc
import importlib.util
import io
import subprocess
import sys
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
        ("watchdog", slash_worker._watchdog_parent(parent_pid, is_windows=sys.platform == "win32")),
        ("cli-import",),
        ("runtime",),
        ("cli", "session-1"),
    ]


def test_watchdog_parent_uses_spawn_pid_off_windows():
    def unexpected_getppid():
        raise AssertionError("spawn parent PID must come from argv")

    assert slash_worker._watchdog_parent(424242, is_windows=False, getppid=unexpected_getppid) == 424242


def test_watchdog_parent_falls_back_to_observed_parent_without_spawn_pid():
    assert slash_worker._watchdog_parent(0, is_windows=False, getppid=lambda: 777) == 777


def test_watchdog_parent_keeps_observed_parent_on_windows():
    # A venv python.exe redirector sits between the gateway and this interpreter on
    # Windows, so the gateway PID never equals os.getppid() there; arming on it
    # would make the watchdog kill a healthy worker on its first poll.
    assert slash_worker._watchdog_parent(424242, is_windows=True, getppid=lambda: 777) == 777
