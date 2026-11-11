"""A browser_exec call holds its browser session until it returns.

The inactivity janitor closes sessions nobody used for ``browser.inactivity_timeout`` seconds. A single
browser_exec call may run for up to ``timeout_s`` (default 300 s, max 1800 s); the browser it drives is
in use for the whole call and must not be closed under it.
"""

import json
import stat
import threading

import pytest

import tools.browser_use_cli as bu_cli
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud
from tools import browser_session_lifecycle as lifecycle
from tests.waiting import wait_until


class _RecordingProvider:
    """A cloud browser provider (the ``BrowserProvider`` lifecycle trio) that records each close."""

    name = "recording"

    def __init__(self):
        self.closed: list[str] = []

    def is_available(self) -> bool:
        return True

    def create_session(self, task_id: str) -> dict:
        # Port 9 (discard) has no CDP listener: the supervisor attach fails fast and is non-fatal.
        return {"session_name": f"rec-{task_id}", "bb_session_id": f"bb-{task_id}",
                "cdp_url": "ws://127.0.0.1:9/devtools/browser/rec", "features": {}}

    def close_session(self, session_id: str) -> bool:
        self.closed.append(session_id)
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        self.closed.append(session_id)


@pytest.fixture
def session_state(monkeypatch):
    for name in ("_active_sessions", "_session_last_activity", "_session_owner_homes", "_cleanup_failures"):
        monkeypatch.setattr(bs, name, {})
    # The janitor's tick is driven by the test; the 30 s background loop stays off.
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)
    bs_cloud.reset_caches()
    yield
    bs_cloud.reset_caches()


def test_janitor_leaves_the_browser_of_a_running_exec_alone(tmp_path, monkeypatch, session_state):
    provider = _RecordingProvider()
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    started, release = tmp_path / "started", tmp_path / "release"
    cli = tmp_path / "browser-harness"
    cli.write_text(f"""#!/bin/sh
cat > /dev/null
: > "{started}"
i=0
while [ ! -f "{release}" ] && [ $i -lt 600 ]; do sleep 0.05; i=$((i+1)); done
echo done
""", encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])

    from model_tools import handle_function_call

    results: list[str] = []
    call = threading.Thread(target=lambda: results.append(handle_function_call(
        "browser_exec", {"code": "print(page_info())", "timeout_s": 60}, task_id="t-long")))
    call.start()
    try:
        wait_until(started.exists, what="the harness CLI to start")
        # The call has outlived browser.inactivity_timeout: the janitor's next tick finds the session idle.
        monkeypatch.setattr(bs, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 0)
        lifecycle.cleanup_inactive_sessions()

        assert provider.closed == []
        assert "t-long" in bs._active_sessions
    finally:
        release.touch()
        call.join(timeout=30)
    assert json.loads(results[0])["output"].strip() == "done"
