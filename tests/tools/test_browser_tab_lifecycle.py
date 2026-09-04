from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

import pytest

from tools.browser_tab_lifecycle import (
    BrowserTabLifecycleGuard,
    _redact_url,
    finalize_browser_tab_owner,
    reap_expired_browser_tab_leases,
)


class MemoryGuard(BrowserTabLifecycleGuard):
    def __init__(self, *args, pages=None, **kwargs):
        self.pages = dict(pages or {})
        self.requests = []
        super().__init__(*args, **kwargs)

    def _require_live_browser_identity(self) -> None:
        # MemoryGuard's in-memory endpoint represents the browser_key passed by
        # each test. Network identity behavior is covered by _FakeCdp below.
        return None

    def _request_json(self, path: str, *, method: str = "GET"):
        self.requests.append((method, path))
        if path == "/json/list":
            return list(self.pages.values())
        if path.startswith("/json/close/"):
            target_id = path.rsplit("/", 1)[-1]
            self.pages.pop(target_id, None)
            return "Target is closing"
        if path.startswith("/json/new?"):
            target_id = f"baseline-{self.browser_key}"
            self.pages[target_id] = {
                "id": target_id,
                "type": "page",
                "url": "about:blank",
                "title": "",
            }
            return self.pages[target_id]
        raise AssertionError(path)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


def _page(target_id: str, url: str, title: str = "") -> dict:
    return {"id": target_id, "type": "page", "url": url, "title": title}


class _BytesResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class _FakeCdp:
    def __init__(self, websocket_url: str, pages: dict[str, dict]):
        self.websocket_url = websocket_url
        self.pages = pages
        self.requests: list[tuple[str, str]] = []

    def urlopen(self, request, *, timeout=5):
        del timeout
        if isinstance(request, str):
            url = request
            method = "GET"
        else:
            url = request.full_url
            method = request.get_method()
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        self.requests.append((method, path))
        if parsed.path == "/json/version":
            return _BytesResponse(
                json.dumps({"webSocketDebuggerUrl": self.websocket_url}).encode()
            )
        if parsed.path == "/json/list":
            return _BytesResponse(json.dumps(list(self.pages.values())).encode())
        if parsed.path.startswith("/json/close/"):
            target_id = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
            self.pages.pop(target_id, None)
            return _BytesResponse(b"Target is closing")
        if parsed.path == "/json/new":
            self.pages["new-blank"] = _page("new-blank", "about:blank")
            return _BytesResponse(json.dumps(self.pages["new-blank"]).encode())
        raise AssertionError((method, path))


def _browser_key(websocket_url: str) -> str:
    return hashlib.sha256(websocket_url.encode()).hexdigest()


def test_url_redaction_drops_credentials_query_and_fragment():
    assert (
        _redact_url("https://user:pass@example.com/path?a=secret#frag")
        == "https://example.com/path"
    )


def test_closes_created_and_repurposed_targets_then_leaves_one_blank(state_dir: Path):
    guard = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key="browser-a",
        owner_key="task-a",
        state_dir=state_dir,
        pages={"base": _page("base", "about:blank")},
    )
    assert guard.start() is None
    assert guard.target_id is not None
    guard.pages[guard.target_id] = _page(
        guard.target_id, "https://example.com/path?q=private", "Example"
    )
    guard.pages["popup"] = _page("popup", "https://example.com/popup", "Popup")

    report = guard.finish()

    assert report == {
        "managed": True,
        "ok": True,
        "created": 2,
        "repurposed": 0,
        "closed": 2,
        "leased": 0,
        "remaining_pages": 1,
        "errors": [],
    }
    assert list(guard.pages) == ["base"]
    with guard._connect() as db:
        rows = db.execute(
            """SELECT target_id, state, close_verified, url_redacted
               FROM browser_resources ORDER BY target_id"""
        ).fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("baseline-browser-a", "closed", 1),
        ("popup", "closed", 1),
    ]
    assert rows[0][3] == "https://example.com/path"


def test_explicit_lease_persists_then_next_unleased_call_closes_owned_target(state_dir: Path):
    pages = {"base": _page("base", "about:blank")}
    leased = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key="browser-a",
        owner_key="task-a",
        lease_minutes=20,
        lease_reason="continue pagination",
        state_dir=state_dir,
        pages=pages,
    )
    assert leased.start() is None
    assert leased.target_id is not None
    leased.pages[leased.target_id] = _page(
        leased.target_id, "https://example.com/page/1"
    )
    first = leased.finish()
    assert first["leased"] == 1
    assert first["closed"] == 0
    assert set(leased.pages) == {"base", "baseline-browser-a"}

    closing = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key="browser-a",
        owner_key="task-a",
        state_dir=state_dir,
        pages=leased.pages,
    )
    assert closing.start() is None
    second = closing.finish()
    assert second["closed"] == 1
    assert second["remaining_pages"] == 1
    assert list(closing.pages) == ["base"]


def test_browser_identity_scopes_leased_target_ids(state_dir: Path):
    leased = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key="browser-a",
        owner_key="turn-a",
        lease_minutes=20,
        lease_reason="continue",
        state_dir=state_dir,
        pages={"base": _page("base", "about:blank")},
    )
    assert leased.start() is None
    leased.finish()

    other = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9333",
        browser_key="browser-b",
        owner_key="turn-a",
        state_dir=state_dir,
        pages={"baseline-browser-a": _page("baseline-browser-a", "https://user.example")},
    )
    with other._connect() as db:
        assert other._owned_active_ids(db) == set()


def test_turn_finalizer_closes_leased_target(state_dir: Path, monkeypatch):
    websocket_url = "ws://127.0.0.1:9222/devtools/browser/browser-a"
    leased = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key=_browser_key(websocket_url),
        owner_key="turn-a",
        lease_minutes=20,
        lease_reason="continue",
        state_dir=state_dir,
        pages={"base": _page("base", "about:blank")},
    )
    assert leased.start() is None
    leased.finish()
    fake_cdp = _FakeCdp(websocket_url, leased.pages)
    monkeypatch.setattr(
        "tools.browser_tab_lifecycle.urllib.request.urlopen",
        fake_cdp.urlopen,
    )
    report = finalize_browser_tab_owner("turn-a", state_dir=state_dir)
    assert report == {"closed": 1, "failed": 0, "errors": []}
    assert any(path.startswith("/json/close/") for _, path in fake_cdp.requests)
    assert not any(path.startswith("/json/new") for _, path in fake_cdp.requests)
    with leased._connect() as db:
        state = db.execute(
            """SELECT state FROM browser_resources
               WHERE browser_key=? AND owner_key='turn-a'""",
            (_browser_key(websocket_url),),
        ).fetchone()[0]
    assert state == "closed"


def test_expired_lease_reaper_closes_target(state_dir: Path, monkeypatch):
    websocket_url = "ws://127.0.0.1:9222/devtools/browser/browser-a"
    leased = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key=_browser_key(websocket_url),
        owner_key="turn-a",
        lease_minutes=20,
        lease_reason="continue",
        state_dir=state_dir,
        pages={"base": _page("base", "about:blank")},
    )
    assert leased.start() is None
    leased.finish()
    with leased._connect() as db:
        db.execute(
            "UPDATE browser_resources SET lease_expires_at='2000-01-01T00:00:00+00:00'"
        )
    fake_cdp = _FakeCdp(websocket_url, leased.pages)
    monkeypatch.setattr(
        "tools.browser_tab_lifecycle.urllib.request.urlopen",
        fake_cdp.urlopen,
    )
    report = reap_expired_browser_tab_leases(state_dir=state_dir)
    assert report == {"closed": 1, "failed": 0, "errors": []}
    assert any(path.startswith("/json/close/") for _, path in fake_cdp.requests)
    assert not any(path.startswith("/json/new") for _, path in fake_cdp.requests)


def test_reaper_never_writes_to_restarted_browser_on_same_endpoint(
    state_dir: Path, monkeypatch
):
    old_websocket = "ws://127.0.0.1:9222/devtools/browser/old-guid"
    leased = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key=_browser_key(old_websocket),
        owner_key="turn-a",
        lease_minutes=20,
        lease_reason="continue",
        state_dir=state_dir,
        pages={"base": _page("base", "about:blank")},
    )
    assert leased.start() is None
    leased.finish()
    with leased._connect() as db:
        db.execute(
            "UPDATE browser_resources SET lease_expires_at='2000-01-01T00:00:00+00:00'"
        )

    restarted_pages = {"user-tab": _page("user-tab", "https://example.com")}
    fake_cdp = _FakeCdp(
        "ws://127.0.0.1:9222/devtools/browser/new-guid", restarted_pages
    )
    monkeypatch.setattr(
        "tools.browser_tab_lifecycle.urllib.request.urlopen",
        fake_cdp.urlopen,
    )

    report = reap_expired_browser_tab_leases(state_dir=state_dir)

    assert report["closed"] == 0
    assert report["failed"] == 1
    assert "browser identity changed" in report["errors"][0]
    assert fake_cdp.requests == [("GET", "/json/version")]
    assert restarted_pages == {"user-tab": _page("user-tab", "https://example.com")}
    with leased._connect() as db:
        row = db.execute(
            "SELECT state, close_verified, last_error FROM browser_resources"
        ).fetchone()
    assert tuple(row[:2]) == ("close_failed", 0)
    assert "browser identity changed" in row[2]


def test_unowned_nonblank_target_is_never_closed(state_dir: Path):
    guard = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key="browser-a",
        owner_key="task-a",
        state_dir=state_dir,
        pages={"other": _page("other", "https://example.org/user-work")},
    )
    assert guard.start() is None
    report = guard.finish()
    assert report["closed"] == 1
    assert "other" in guard.pages
    assert guard.pages["other"]["url"] == "https://example.org/user-work"
    closed_paths = [path for _, path in guard.requests if path.startswith("/json/close/")]
    assert all(not path.endswith("/other") for path in closed_paths)


def test_start_fails_closed_before_execution_when_snapshot_breaks(state_dir: Path):
    guard = MemoryGuard(
        enabled=True,
        endpoint="http://127.0.0.1:9222",
        browser_key="browser-a",
        owner_key="task-a",
        state_dir=state_dir,
    )

    def broken_snapshot():
        raise TimeoutError("CDP busy")

    guard._snapshot = broken_snapshot  # type: ignore[method-assign]
    error = guard.start()
    assert error is not None
    assert "could not prepare" in error
    assert "CDP busy" in error
