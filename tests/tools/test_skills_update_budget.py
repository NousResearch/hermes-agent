"""Real HTTP controls for invalid installs and bounded update-check work."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from tools import skills_hub_install as install
from tools.skills_hub import HubLockFile


@pytest.fixture
def upstream():
    calls = []
    release = threading.Event()
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            calls.append(self.path)
            if self.path == "/blocked":
                release.wait(10)
            elif self.path == "/paced":
                time.sleep(0.1)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Source:
        def source_id(self):
            return "fixture"

        def fetch(self, identifier):
            with urlopen(f"http://127.0.0.1:{server.server_port}/{identifier}", timeout=15) as response:
                response.read()
            finished.set()
            return None

    try:
        yield Source(), calls, release, finished
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def record(lock, name, path, identifier="ok"):
    lock.record_install(name, "fixture", identifier, "community", "safe", "old", path, ["SKILL.md"])


def test_invalid_install_never_contacts_upstream(tmp_path, monkeypatch, upstream):
    from tools import skills_hub as hub
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(hub, "SKILLS_DIR", root)
    (root / "healthy").mkdir()
    lock = HubLockFile()
    record(lock, "invalid", "invalid", "invalid")
    record(lock, "healthy", "healthy")
    data = lock.load()
    data["installed"]["invalid"]["install_path"] = "../outside"
    lock.save(data)
    source, calls, _, _ = upstream
    rows = install.check_for_skill_updates(lock=lock, sources=[source])
    assert [row["status"] for row in rows] == ["invalid_install", "unavailable"]
    assert calls == ["/ok"]


def test_update_budget_bounds_repeated_workers_and_whole_check(tmp_path, monkeypatch, upstream):
    from tools import skills_hub as hub
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(hub, "SKILLS_DIR", root)
    monkeypatch.setattr(install, "_FETCH_TIMEOUT_SECONDS", 0.2)
    source, calls, release, finished = upstream
    lock = HubLockFile()
    for index in range(30):
        name = f"skill-{index}"
        (root / name).mkdir()
        record(lock, name, name, "blocked")
    try:
        started = time.monotonic()
        for _ in range(3):
            rows = install.check_for_skill_updates(lock=lock, sources=[source])
            assert all(row["status"] == "unavailable" for row in rows)
        assert time.monotonic() - started < 2
        assert calls == ["/blocked"]
    finally:
        release.set()
        assert finished.wait(5)
    # Wait for fetch's finally to release capacity, without assuming scheduler order.
    deadline = time.monotonic() + 5
    while any(t.name == "skills-update-fetch" for t in threading.enumerate()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    calls.clear()
    for index in range(30):
        name = f"skill-{index}"
        record(lock, name, name, "paced")
    started = time.monotonic()
    install.check_for_skill_updates(lock=lock, sources=[source])
    assert time.monotonic() - started < 2
    assert 1 <= len(calls) <= 2
