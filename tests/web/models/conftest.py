"""Pytest fixtures for Playwright UI tests.

All tests import fixtures from here via conftest.py — no duplication.
"""
from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil
import pytest
from playwright.async_api import async_playwright, Page, Browser


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.kill()
        parent.wait(timeout=5)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        pass


def _worker_index() -> int:
    """Return the xdist worker index, or 0 for non-xdist runs."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    if worker.startswith("gw") and worker[2:].isdigit():
        return int(worker[2:])
    return 0


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_dashboard_port() -> int:
    if "HERMES_TEST_DASHBOARD_PORT" in os.environ:
        return int(os.environ["HERMES_TEST_DASHBOARD_PORT"])

    # Avoid port overflow by using compact non-overlapping ranges for workers
    base = 20000 + ((os.getpid() % 50) * 400) + (_worker_index() * 20)
    for port in range(base, base + 20):
        if _port_is_free(port):
            return port
    raise RuntimeError(f"No free dashboard test port in range {base}-{base + 19}")


DASHBOARD_PORT = _pick_dashboard_port()
DASHBOARD_URL = f"http://127.0.0.1:{DASHBOARD_PORT}"
MODELS_PAGE_URL = f"{DASHBOARD_URL}/models"


@pytest.fixture(scope="session", autouse=True)
def dashboard_server(tmp_path_factory):
    """Run an isolated dashboard for Playwright tests.

    The web tests must not hit a developer's real dashboard on port 9119 or
    mutate the real ~/.hermes/config.yaml. Each pytest worker gets its own
    dashboard process, port, and HERMES_HOME.
    """
    hermes_home = tmp_path_factory.mktemp("hermes_web_models_home")
    for child in ("sessions", "cron", "memories", "skills"):
        (hermes_home / child).mkdir(exist_ok=True)

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["HERMES_TEST_DASHBOARD_PORT"] = str(DASHBOARD_PORT)
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_path = Path(str(hermes_home)) / "dashboard-test.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            str(DASHBOARD_PORT),
            "--no-open",
            "--skip-build",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    atexit.register(_kill_process_tree, proc.pid)

    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_file.flush()
            output = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Dashboard test server exited early with code {proc.returncode}.\n{output}"
            )
        try:
            with urllib.request.urlopen(DASHBOARD_URL, timeout=1) as response:
                if response.status < 500:
                    break
        except Exception as exc:  # noqa: BLE001 - keep last readiness error for diagnostics
            last_error = exc
        time.sleep(0.25)
    else:
        _kill_process_tree(proc.pid)
        log_file.flush()
        output = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"Dashboard test server did not become ready at {DASHBOARD_URL}: {last_error}\n{output}"
        )

    yield

    _kill_process_tree(proc.pid)
    log_file.close()


@pytest.fixture
async def browser():
    """Launch a headless Chromium browser for the test suite."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser: Browser) -> Page:
    """Create a new page with the session token injected."""
    context = await browser.new_context()
    page = await context.new_page()

    # Fetch the session token from the running dashboard
    await page.goto(DASHBOARD_URL)
    token = await page.evaluate(
        "() => { const m = document.body.innerHTML.match(/__HERMES_SESSION_TOKEN__=\\\"([^\\\"]+)\\\"/); return m ? m[1] : null; }",
    )

    if token:
        await page.add_init_script(
            f"window.__HERMES_SESSION_TOKEN__ = \"{token}\";",
        )

    yield page
    await context.close()

@pytest.fixture(autouse=True)
async def reset_model_fallbacks(page: Page):
    """Start every web model test from a clean persisted fallback chain."""
    await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const response = await fetch('/api/model/fallbacks', {
            method: 'PUT',
            headers: {
                'X-Hermes-Session-Token': token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({fallbacks: []})
        });
        if (!response.ok) throw new Error(`fallback reset failed: ${response.status}`);
        return await response.json();
    }""")
    yield
