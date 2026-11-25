"""Tab cap: creating a tab over MAX_OPEN_PAGES closes the oldest other pages, never the new one."""

import pytest

from tools import browser_cdp_tool as cdp


def _pages(n: int) -> list[dict]:
    return [{"targetId": f"page-{i}", "type": "page"} for i in range(n)]


@pytest.fixture
def fake_cdp(monkeypatch):
    calls: list[tuple[str, dict]] = []
    state = {"targets": [], "fail_close": set()}

    async def _fake_cdp_call(endpoint, method, params, target_id, timeout):
        calls.append((method, dict(params)))
        if method == "Target.getTargets":
            return {"targetInfos": list(state["targets"])}
        if method == "Target.closeTarget":
            if params["targetId"] in state["fail_close"]:
                raise RuntimeError("cannot close")
            state["targets"] = [t for t in state["targets"] if t["targetId"] != params["targetId"]]
            return {"success": True}
        raise AssertionError(f"unexpected method {method}")

    monkeypatch.setattr(cdp, "_cdp_call", _fake_cdp_call)
    return state, calls


def test_no_close_at_or_below_cap(fake_cdp):
    state, calls = fake_cdp
    state["targets"] = _pages(cdp.MAX_OPEN_PAGES)
    assert cdp._enforce_page_cap("ws://x", "page-0", 5) == 0
    assert all(method != "Target.closeTarget" for method, _ in calls)


def test_closes_oldest_pages_above_cap_and_keeps_new_tab(fake_cdp):
    state, calls = fake_cdp
    state["targets"] = _pages(cdp.MAX_OPEN_PAGES + 3)
    closed = cdp._enforce_page_cap("ws://x", "page-0", 5)
    assert closed == 3
    closed_ids = [p["targetId"] for m, p in calls if m == "Target.closeTarget"]
    # page-0 is the new tab, so the oldest others (page-1..page-3) go
    assert closed_ids == ["page-1", "page-2", "page-3"]
    assert "page-0" in {t["targetId"] for t in state["targets"]}


def test_ignores_non_page_targets(fake_cdp):
    state, _ = fake_cdp
    state["targets"] = _pages(cdp.MAX_OPEN_PAGES) + [{"targetId": "sw-1", "type": "service_worker"}]
    assert cdp._enforce_page_cap("ws://x", None, 5) == 0


def test_one_failed_close_does_not_block_the_rest(fake_cdp):
    state, _ = fake_cdp
    state["targets"] = _pages(cdp.MAX_OPEN_PAGES + 2)
    state["fail_close"] = {"page-1"}
    assert cdp._enforce_page_cap("ws://x", None, 5) == 1


def test_cap_failure_never_raises(monkeypatch):
    async def _boom(*a, **k):
        raise OSError("endpoint gone")

    monkeypatch.setattr(cdp, "_cdp_call", _boom)
    assert cdp._enforce_page_cap("ws://x", None, 5) == 0
