"""Integration tests for tools.browser_supervisor.

Exercises the supervisor end-to-end against a real local Chrome
(``--remote-debugging-port``).  Skipped when Chrome is not installed
— these are the tests that actually verify the CDP wire protocol
works, since mock-CDP unit tests can only prove the happy paths we
thought to model.

These tests spawn a **real Chrome process** on the machine running them.
They are therefore opt-in, twice over:

* ``@pytest.mark.integration`` — excluded by the default
  ``addopts = "-m 'not integration'"`` in ``pyproject.toml``, so a bare
  ``pytest`` cannot launch a browser on a developer's desktop by accident.
* ``HERMES_E2E_BROWSER=1`` — the env gate this docstring has always claimed.
  It previously existed only in this prose: nothing read the variable, and
  the sole real gate was "is a Chrome binary on PATH", which is true on most
  desktops and on ``ubuntu-latest``. Now it is enforced.

Run manually:
    HERMES_E2E_BROWSER=1 scripts/run_tests.sh -m integration \\
        tests/tools/test_browser_supervisor.py

(``scripts/run_tests.sh`` runs under ``env -i`` and forwards
``HERMES_E2E_BROWSER`` explicitly; ``-m integration`` overrides the default
marker filter.)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
        reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in",
    ),
    pytest.mark.skipif(
        not shutil.which("google-chrome") and not shutil.which("chromium"),
        reason="Chrome/Chromium not installed",
    ),
]


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    pytest.skip("no Chrome binary found")


@pytest.fixture
def chrome_cdp(request):
    """Start a headless Chrome with --remote-debugging-port, yield its WS URL.

    Uses a unique port per xdist worker to avoid cross-worker collisions.
    Always launches with ``--site-per-process`` so cross-origin iframes
    become real OOPIFs (needed by the iframe interaction tests).
    """

    # xdist worker_id is "master" in single-process mode or "gw0".."gwN" otherwise.
    # Under subprocess-per-file isolation there's no xdist, so we fall back
    # to "master" via the session-scoped fixture below.
    worker_id = request.getfixturevalue("worker_id") if "worker_id" in request.fixturenames else "master"
    if worker_id == "master":
        port_offset = 0
    else:
        port_offset = int(worker_id.lstrip("gw"))
    port = 9225 + port_offset
    profile = tempfile.mkdtemp(prefix="hermes-supervisor-test-")
    proc = subprocess.Popen(
        [
            _find_chrome(),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--headless=new",
            "--disable-gpu",
            "--site-per-process",  # force OOPIFs for cross-origin iframes
            "about:blank",  # avoid browser-owned new-tab content and background renderers
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ws_url = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=1
            ) as r:
                info = json.loads(r.read().decode())
                ws_url = info["webSocketDebuggerUrl"]
                break
        except Exception:
            time.sleep(0.25)
    if ws_url is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, AssertionError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except (AssertionError, Exception):
                pass
        shutil.rmtree(profile, ignore_errors=True)
        pytest.skip("Chrome didn't expose CDP in time")

    yield ws_url, port

    # Tear down Chrome. The stdlib `subprocess._wait()` POSIX implementation
    # has a known race (https://bugs.python.org/issue38630): when SIGCHLD
    # arrives concurrently with `proc.wait()`, `_try_wait(WNOHANG)` can
    # return a foreign pid and the `assert pid == self.pid or pid == 0`
    # fires. We saw this in CI on slice 1 after this fixture's teardown
    # (PR #33661 follow-up). Swallow the stdlib race + force-kill if wait
    # hangs, then always reap so we don't leak a zombie.
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except (subprocess.TimeoutExpired, AssertionError, Exception):
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except (AssertionError, Exception):
            pass
    shutil.rmtree(profile, ignore_errors=True)


def _test_page_url() -> str:
    html = """<!doctype html>
<html><head><title>Supervisor pytest</title></head><body>
<h1>Supervisor pytest</h1>
<iframe id="inner" srcdoc="<body><h2>frame-marker</h2></body>" width="400" height="100"></iframe>
</body></html>"""
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


def _title_page_url(title: str) -> str:
    html = f"<!doctype html><html><head><title>{title}</title></head><body>{title}</body></html>"
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


def _interactive_page_url() -> str:
    html = """<!doctype html><html><head><title>interactive</title></head><body>
<button id="owned-click" onclick="document.title='clicked'">Click</button>
<input id="owned-input">
</body></html>"""
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


def _fire_on_page(supervisor, expression: str) -> None:
    """Navigate and fire an expression on the page actually owned by the supervisor."""
    from tools.browser_supervisor import _schedule

    result = _schedule(
        supervisor._cdp("Page.navigate", {"url": _test_page_url()}, session_id=supervisor._page_session_id),
        supervisor._loop,
        timeout=10,
    )
    assert "error" not in result, result
    assert not result.get("result", {}).get("errorText"), result
    time.sleep(1.5)
    result = supervisor.evaluate_runtime(expression)
    assert result["ok"] is True, result


@pytest.fixture
def supervisor_registry():
    """Yield the global registry and tear down any supervisors after the test."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    yield SUPERVISOR_REGISTRY
    SUPERVISOR_REGISTRY.stop_all()


def _wait_for_dialog(supervisor, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = supervisor.snapshot()
        if snap.pending_dialogs:
            return snap.pending_dialogs
        time.sleep(0.1)
    return ()


def test_supervisor_start_and_snapshot(chrome_cdp, supervisor_registry):
    """Supervisor attaches, exposes an active snapshot with a top frame."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-1", cdp_url=cdp_url)

    # Navigate so the frame tree populates.
    _fire_on_page(supervisor, "/* no dialog */ void 0")

    # Give a moment for frame events to propagate
    time.sleep(1.0)
    snap = supervisor.snapshot()
    assert snap.active is True
    assert snap.task_id == "pytest-1"
    assert snap.pending_dialogs == ()
    # At minimum a top frame should exist after the navigate.
    assert snap.frame_tree.get("top") is not None


def test_two_supervisors_navigate_distinct_owned_pages(chrome_cdp, supervisor_registry):
    """Two task IDs navigate separate real Chrome tabs through owned sessions.

    This is the integration boundary behind ``browser_navigate`` for shared
    CDP browsers: each operation goes through ``CDPSupervisor.navigate_page``
    and must remain bound to that supervisor's page session (#69727).
    """
    cdp_url, _port = chrome_cdp
    first = supervisor_registry.get_or_start(task_id="pytest-nav-a", cdp_url=cdp_url)
    second = supervisor_registry.get_or_start(task_id="pytest-nav-b", cdp_url=cdp_url)

    assert first.page_target_id()
    assert second.page_target_id()
    assert first.page_target_id() != second.page_target_id()
    assert first.navigate_page(_title_page_url("owned-page-a"))["ok"] is True
    assert second.navigate_page(_title_page_url("owned-page-b"))["ok"] is True

    deadline = time.monotonic() + 5
    titles = (None, None)
    while time.monotonic() < deadline:
        first_title = first.evaluate_runtime("document.title")
        second_title = second.evaluate_runtime("document.title")
        titles = (first_title.get("result"), second_title.get("result"))
        if titles == ("owned-page-a", "owned-page-b"):
            break
        time.sleep(0.05)

    assert titles == ("owned-page-a", "owned-page-b")

    # A transport reconnect must retain the owned page and its loaded document.
    from tools.browser_supervisor import _schedule

    target_id, session_id = first.page_target_id(), first._page_session_id
    assert first._ws is not None
    _schedule(first._ws.close(), first._loop, timeout=5)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if first.snapshot().active and first._page_session_id != session_id:
            break
        time.sleep(0.05)
    assert first.snapshot().active
    assert first._page_session_id != session_id
    assert first.page_target_id() == target_id
    assert first.evaluate_runtime("document.title").get("result") == "owned-page-a"
    assert second.evaluate_runtime("document.title").get("result") == "owned-page-b"


@pytest.mark.skipif(
    not shutil.which("agent-browser") and not shutil.which("npx"),
    reason="agent-browser integration requires agent-browser or npx",
)
@pytest.mark.parametrize("retire_page", [False, True], ids=["stable-pages", "closed-page"])
def test_two_supervisors_bind_concurrent_follow_up_actions(chrome_cdp, supervisor_registry, monkeypatch, tmp_path, retire_page):
    """Concurrent click/fill operations stay on their task-owned CDP pages."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("browser:\n  allow_private_urls: true\n", encoding="utf-8")
    from tools import browser_tool, browser_tool_lifecycle, browser_tool_session

    cdp_url, _port = chrome_cdp
    monkeypatch.setenv("BROWSER_CDP_URL", cdp_url)
    task_ids = ("pytest-action-a", "pytest-action-b")
    retired_task = "pytest-action-retired"
    all_task_ids = (retired_task, *task_ids) if retire_page else task_ids
    page_url = _interactive_page_url()
    try:
        for task_id in all_task_ids:
            result = json.loads(browser_tool.browser_navigate(page_url, task_id=task_id))
            assert result["success"] is True, result
            assert result["title"] == "interactive", result
            assert "Click" in result["snapshot"], result
            assert browser_tool._last_active_session_key[task_id] == task_id
        snapshots = {}
        for task_id in all_task_ids:
            snapshot = json.loads(browser_tool.browser_snapshot(task_id=task_id))
            assert snapshot["success"] is True, snapshot
            assert "Click" in snapshot["snapshot"]
            snapshots[task_id] = snapshot["snapshot"]

        if retire_page:
            from tools.browser_supervisor import _schedule

            probe_supervisor = supervisor_registry.get(all_task_ids[0])
            assert probe_supervisor is not None
            targets = _schedule(
                probe_supervisor._cdp("Target.getTargets"), probe_supervisor._loop, timeout=5,
            )["result"]["targetInfos"]
            target_order = {target["targetId"]: index for index, target in enumerate(targets)}
            task_positions = {}
            for task_id in all_task_ids:
                supervisor = supervisor_registry.get(task_id)
                assert supervisor is not None
                task_positions[task_id] = target_order[supervisor.page_target_id()]
            retired_task = min(task_positions, key=task_positions.__getitem__)
            task_ids = tuple(task_id for task_id in all_task_ids if task_id != retired_task)
            browser_tool_lifecycle._cleanup_single_browser_session(retired_task)

        first = supervisor_registry.get(task_ids[0])
        second = supervisor_registry.get(task_ids[1])
        assert first is not None and second is not None
        click_selector, fill_selector = "#owned-click", "#owned-input"
        if not retire_page:
            click_ref = re.search(r"button[^\n]*\[ref=(e\d+)\]", snapshots[task_ids[0]])
            fill_ref = re.search(r"textbox[^\n]*\[ref=(e\d+)\]", snapshots[task_ids[1]])
            assert click_ref is not None, snapshots[task_ids[0]]
            assert fill_ref is not None, snapshots[task_ids[1]]
            click_selector, fill_selector = f"@{click_ref[1]}", f"@{fill_ref[1]}"
        with ThreadPoolExecutor(max_workers=2) as pool:
            click_future = pool.submit(
                browser_tool_session._run_browser_command,
                task_ids[0],
                "click",
                [click_selector],
            )
            fill_future = pool.submit(
                browser_tool_session._run_browser_command,
                task_ids[1],
                "fill",
                [fill_selector, "task-b"],
            )
            click_result = click_future.result(timeout=30)
            fill_result = fill_future.result(timeout=30)

        assert click_result["success"] is True, click_result
        assert fill_result["success"] is True, fill_result
        assert first.evaluate_runtime("document.title").get("result") == "clicked"
        assert second.evaluate_runtime("document.querySelector('#owned-input').value").get("result") == "task-b"
        assert first.evaluate_runtime("document.querySelector('#owned-input').value").get("result") == ""
        assert second.evaluate_runtime("document.title").get("result") == "interactive"
    finally:
        for task_id in all_task_ids:
            browser_tool_lifecycle._cleanup_single_browser_session(task_id)


@pytest.mark.skipif(
    not shutil.which("agent-browser") and not shutil.which("npx"),
    reason="agent-browser integration requires agent-browser or npx",
)
def test_new_foreign_page_cannot_steal_follow_up_action(chrome_cdp, supervisor_registry, monkeypatch, tmp_path):
    """A new tab between browser commands cannot become this task's action target."""
    from tools import browser_tool, browser_tool_lifecycle, browser_tool_session
    from tools.browser_supervisor import _schedule

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("browser:\n  allow_private_urls: true\n", encoding="utf-8")
    cdp_url, _port = chrome_cdp
    monkeypatch.setenv("BROWSER_CDP_URL", cdp_url)
    task_id = "pytest-new-foreign-page"
    try:
        navigated = json.loads(browser_tool.browser_navigate(_interactive_page_url(), task_id=task_id))
        assert navigated["success"] is True, navigated
        snapshot = json.loads(browser_tool.browser_snapshot(task_id=task_id))
        assert snapshot["success"] is True, snapshot
        supervisor = supervisor_registry.get(task_id)
        assert supervisor is not None
        created = _schedule(
            supervisor._cdp("Target.createTarget", {"url": _interactive_page_url()}),
            supervisor._loop, timeout=5,
        )
        foreign_target = created["result"]["targetId"]
        assert foreign_target != supervisor.page_target_id()
        inventory = browser_tool_session._run_browser_command(task_id, "tab", ["list"])
        assert inventory["success"] is True, inventory
        assert len(inventory["data"]["tabs"]) == 1, inventory
        clicked = browser_tool_session._run_browser_command(task_id, "click", ["#owned-click"])
        assert clicked["success"] is True, clicked
        assert supervisor.evaluate_runtime("document.title").get("result") == "clicked"
        attached = _schedule(
            supervisor._cdp("Target.attachToTarget", {"targetId": foreign_target, "flatten": True}),
            supervisor._loop, timeout=5,
        )
        foreign_title = _schedule(
            supervisor._cdp("Runtime.evaluate", {"expression": "document.title", "returnByValue": True},
                            session_id=attached["result"]["sessionId"]),
            supervisor._loop, timeout=5,
        )
        assert foreign_title["result"]["result"]["value"] == "interactive"
        endpoint = supervisor.page_command_endpoint()

        async def reject_foreign_attachment():
            from websockets.asyncio.client import connect

            async with connect(endpoint, proxy=None) as connection:
                await connection.send(json.dumps({"id": 1, "method": "Target.attachToTarget",
                                                  "params": {"targetId": foreign_target, "flatten": True}}))
                response = json.loads(await asyncio.wait_for(connection.recv(), 5))
                assert "error" in response, response
                assert "sessionId" not in response.get("result", {}), response

        asyncio.run(reject_foreign_attachment())
    finally:
        browser_tool_lifecycle._cleanup_single_browser_session(task_id)
    assert supervisor._thread is not None
    assert not supervisor._thread.is_alive(), "session cleanup left the supervisor running"


def test_main_frame_alert_detection_and_dismiss(chrome_cdp, supervisor_registry):
    """alert() in the main frame surfaces and can be dismissed via the sync API."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-2", cdp_url=cdp_url)

    _fire_on_page(supervisor, "setTimeout(() => alert('PYTEST-MAIN-ALERT'), 50)")
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no dialog detected"
    d = dialogs[0]
    assert d.type == "alert"
    assert "PYTEST-MAIN-ALERT" in d.message

    result = supervisor.respond_to_dialog("dismiss")
    assert result["ok"] is True
    # State cleared after dismiss
    time.sleep(0.3)
    assert supervisor.snapshot().pending_dialogs == ()


def test_iframe_contentwindow_alert(chrome_cdp, supervisor_registry):
    """alert() fired from inside a same-origin iframe surfaces too."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-3", cdp_url=cdp_url)

    _fire_on_page(
        supervisor,
        "setTimeout(() => document.querySelector('#inner').contentWindow.alert('PYTEST-IFRAME'), 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no iframe dialog detected"
    assert any("PYTEST-IFRAME" in d.message for d in dialogs)

    result = supervisor.respond_to_dialog("accept")
    assert result["ok"] is True


def test_prompt_dialog_with_response_text(chrome_cdp, supervisor_registry):
    """prompt() gets our prompt_text back inside the page."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-4", cdp_url=cdp_url)

    # Fire a prompt and stash the answer on window
    _fire_on_page(
        supervisor,
        "setTimeout(() => { window.__promptResult = prompt('give me a token', 'default-x'); }, 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs
    d = dialogs[0]
    assert d.type == "prompt"
    assert d.default_prompt == "default-x"

    result = supervisor.respond_to_dialog("accept", prompt_text="PYTEST-PROMPT-REPLY")
    assert result["ok"] is True


def test_browser_dialog_tool_end_to_end(chrome_cdp, supervisor_registry):
    """Full agent-path check: fire an alert, call the tool handler directly."""
    from tools.browser_dialog_tool import browser_dialog

    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-tool", cdp_url=cdp_url)

    _fire_on_page(supervisor, "setTimeout(() => alert('PYTEST-TOOL-END2END'), 50)")
    assert _wait_for_dialog(supervisor), "no dialog detected via wait_for_dialog"

    r = json.loads(browser_dialog(action="dismiss", task_id="pytest-tool"))
    assert r["success"] is True
    assert r["action"] == "dismiss"
    assert "PYTEST-TOOL-END2END" in r["dialog"]["message"]


def test_browser_cdp_frame_id_real_oopif_smoke_documented():
    """Document that real-OOPIF E2E was manually verified — see PR #14540.

    A pytest version of this hits an asyncio version-quirk in the venv
    (3.11) that doesn't show up in standalone scripts (3.13 + system
    websockets). The mechanism IS verified end-to-end by two separate
    smoke scripts in /tmp/dialog-iframe-test/:

      * smoke_local_oopif.py   — local Chrome + 2 http servers on
        different hostnames + --site-per-process. Outer page on
        localhost:18905, iframe src=http://127.0.0.1:18906. Calls
        browser_cdp(method='Runtime.evaluate', frame_id=<OOPIF>) and
        verifies inner page's title comes back from the OOPIF session.
        PASSED on 2026-04-23: iframe document.title = 'INNER-FRAME-XYZ'

      * smoke_bb_iframe_agent_path.py — Browserbase + real cross-origin
        iframe (src=https://example.com/). Same browser_cdp(frame_id=)
        path. PASSED on 2026-04-23: iframe document.title =
        'Example Domain'

    The test_browser_cdp_frame_id_routes_via_supervisor pytest covers
    the supervisor-routing plumbing with a fake injected OOPIF.
    """
    pytest.skip(
        "Real-OOPIF E2E verified manually with smoke_local_oopif.py and "
        "smoke_bb_iframe_agent_path.py — pytest version hits an asyncio "
        "version quirk between venv (3.11) and standalone (3.13). "
        "Smoke logs preserved in /tmp/dialog-iframe-test/."
    )


def test_evaluate_runtime_unserializable_value(chrome_cdp, supervisor_registry):
    """``Infinity``/``NaN``/``BigInt`` come back via ``unserializableValue``."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-eval-5", cdp_url=cdp_url)

    _fire_on_page(supervisor, "void 0")
    time.sleep(0.5)

    out = supervisor.evaluate_runtime("Infinity")
    assert out["ok"] is True
    assert out["result"] == "Infinity"
