"""A CLI delivery turn that answers and then never exits must not block forever or lose the reply."""

import json
import sys
import time

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


def test_runner_reports_delivery_timeout_to_the_sender(hung_target, capsys):
    argv, dm = hung_target
    assert bot_mode_dm._delivery_main(["--run-delivery", "query-file", str(dm), *argv]) == 1
    out = capsys.readouterr().out
    # The reply the hung turn already produced is forwarded, then the typed refusal.
    assert "the reply text" in out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["reason"] == DELIVERY_TIMEOUT
    assert DELIVERY_TIMEOUT in ALL_REASONS and DELIVERY_TIMEOUT in AUTO_RETRYABLE


@pytest.mark.parametrize("configured, expected", [(None, 1800), (60, 60), (0, None), (-5, None), ("junk", 1800)])
def test_delivery_timeout_config(monkeypatch, configured, expected):
    monkeypatch.setattr(bot_relay, "_bot_mode_cfg", lambda key, loader: configured)
    assert bot_mode_dm._delivery_timeout_seconds() == expected
