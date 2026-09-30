"""A CLI delivery turn that answers and then never exits must not block forever or lose the reply."""

import json
import sys
import time

import psutil
import pytest

from tools import bot_mode_dm, bot_relay
from tools.bot_failure_reasons import ALL_REASONS, AUTO_RETRYABLE, DELIVERY_TIMEOUT

_HANGS_AFTER_REPLY = (
    "import sys, time\n"
    "sys.stdin.read()\n"
    "print('the reply text', flush=True)\n"
    "time.sleep(60)\n"
)


@pytest.fixture
def hung_target(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_dm, "_delivery_timeout_seconds", lambda: 1)
    child = tmp_path / "hung_target.py"
    child.write_text(_HANGS_AFTER_REPLY, encoding="utf-8")
    dm = tmp_path / "message.txt"
    dm.write_text("hello", encoding="utf-8")
    return [sys.executable, str(child)], dm


@pytest.mark.parametrize("stdin_file", [False, True])
def test_hung_transport_is_bounded(hung_target, capfd, stdin_file):
    argv, dm = hung_target
    started = time.monotonic()
    with pytest.raises(bot_mode_dm.DeliveryTimeout) as caught:
        bot_mode_dm._run_delivery(argv, str(dm), stdin_file=stdin_file)
    assert time.monotonic() - started < 30
    assert caught.value.reason == DELIVERY_TIMEOUT
    # query-file mode re-emits the captured partial stdout; stdin mode's transport writes to the fd directly.
    assert "the reply text" in capfd.readouterr().out
    assert not dm.exists()


# A hung turn whose own child inherits stdout/stderr (a kernel, terminal or MCP child of a real
# agent turn). Killing only the direct child leaves the pipes open: on Windows ``subprocess.run``
# then waited on them forever; everywhere the grandchild outlived the delivery.
_HANGS_WITH_STDIO_GRANDCHILD = (
    "import pathlib, subprocess, sys, time\n"
    "sys.stdin.read()\n"
    "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "pathlib.Path(sys.argv[1]).write_text(str(grandchild.pid), encoding='utf-8')\n"
    "print('the reply text', flush=True)\n"
    "time.sleep(60)\n"
)


def _exited(pid: int, within: float = 10.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("stdin_file", [False, True])
def test_timeout_kills_the_tree_and_returns_despite_a_grandchild_holding_the_pipes(
        tmp_path, monkeypatch, capfd, stdin_file):
    monkeypatch.setattr(bot_mode_dm, "_delivery_timeout_seconds", lambda: 1)
    child = tmp_path / "hung_parent.py"
    child.write_text(_HANGS_WITH_STDIO_GRANDCHILD, encoding="utf-8")
    pid_file = tmp_path / "grandchild.pid"
    dm = tmp_path / "message.txt"
    dm.write_text("hello", encoding="utf-8")
    started = time.monotonic()
    with pytest.raises(bot_mode_dm.DeliveryTimeout):
        bot_mode_dm._run_delivery([sys.executable, str(child), str(pid_file)], str(dm), stdin_file=stdin_file)
    # Budget + the bounded post-kill drain, never the grandchild's 60s.
    assert time.monotonic() - started < 1 + bot_mode_dm._KILL_DRAIN_SECONDS + 5
    assert "the reply text" in capfd.readouterr().out
    assert _exited(int(pid_file.read_text(encoding="utf-8")))


def test_runner_reports_delivery_timeout_to_the_sender(hung_target, capsys):
    argv, dm = hung_target
    assert bot_mode_dm._delivery_main(["--run-delivery", "query-file", str(dm), *argv]) == 1
    out = capsys.readouterr().out
    # The reply the hung turn already produced is forwarded, then the typed refusal.
    assert "the reply text" in out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["reason"] == DELIVERY_TIMEOUT
    assert DELIVERY_TIMEOUT in ALL_REASONS and DELIVERY_TIMEOUT in AUTO_RETRYABLE


@pytest.mark.parametrize("configured, expected", [
    (None, 1800), (60, 60), (0, None), (-5, None), ("junk", 1800), ("", 1800), (1.5, 1)])
def test_delivery_timeout_config(monkeypatch, configured, expected):
    monkeypatch.setattr(bot_relay, "_bot_mode_cfg", lambda key, loader: configured)
    assert bot_mode_dm._delivery_timeout_seconds() == expected


def test_partial_emit_survives_a_closed_reader(monkeypatch):
    class _Closed:
        def write(self, _):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    monkeypatch.setattr(sys, "stdout", _Closed())
    monkeypatch.setattr(sys, "stderr", _Closed())
    exc = bot_mode_dm.subprocess.TimeoutExpired(["x"], 1, output=b"reply \xff", stderr=b"err")
    bot_mode_dm._emit_partial(exc)  # must not raise, so the caller's DeliveryTimeout is what surfaces
