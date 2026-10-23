"""approvals.timeout <= 0 must mean unlimited on the input() fallback path too, not just the
CLI modal path (hermes_cli/cli_modal_mixin.py) or the gateway wait path
(tools/approval_gateway_wait.py). Before this fix, _read_choice's Thread.join(timeout=0) returned
immediately without waiting (the reader thread was still alive), reporting "no answer" even though
the user never actually got a chance to respond — an instant auto-deny instead of waiting forever.
"""
import threading
import time
from unittest.mock import patch

from tools import approval_prompt as mod


def test_zero_timeout_blocks_until_answered(monkeypatch):
    answered = threading.Event()

    def _slow_input(prompt):
        answered.wait(timeout=2)
        return "o"

    with patch.object(mod, "input", _slow_input):
        result = {}

        def _read():
            result["choice"] = mod._read_choice("prompt> ", timeout_seconds=0)

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        time.sleep(0.3)  # old code's join(timeout=0) would already have returned None here
        assert t.is_alive(), "_read_choice returned despite an unlimited (timeout<=0) deadline"
        answered.set()
        t.join(timeout=2)
        assert result["choice"] == "o"


def test_negative_timeout_also_blocks_until_answered(monkeypatch):
    answered = threading.Event()

    def _slow_input(prompt):
        answered.wait(timeout=2)
        return "o"

    with patch.object(mod, "input", _slow_input):
        result = {}

        def _read():
            result["choice"] = mod._read_choice("prompt> ", timeout_seconds=-5)

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        time.sleep(0.3)
        assert t.is_alive()
        answered.set()
        t.join(timeout=2)
        assert result["choice"] == "o"


def test_positive_timeout_still_times_out(monkeypatch):
    def _never_returns(prompt):
        time.sleep(5)
        return "o"

    with patch.object(mod, "input", _never_returns):
        result = mod._read_choice("prompt> ", timeout_seconds=0.2)
        assert result is None  # None = no answer within the deadline, still correctly bounded
