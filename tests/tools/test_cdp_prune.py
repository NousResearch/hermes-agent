"""Tests for tools.cdp_prune — safe CDP tab pruning.

All coverage is driven by mocked CDP target lists and an injected fake client;
no test opens a socket or touches a real browser. The focus is the safety
contract: only disposable about:blank/newtab tabs are ever closed, real and
sensitive tabs are always preserved, and at least one page target survives.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from tools import cdp_prune as cp


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _page(target_id: str, url: str, *, ttype: str = "page", title: str = "") -> Dict[str, Any]:
    return {"id": target_id, "type": ttype, "url": url, "title": title}


class FakeCdpClient:
    """Injectable CDP client that records close() calls and never does I/O."""

    def __init__(self, targets: List[Dict[str, Any]], *, fail_ids=None, raise_on_list=False):
        self._targets = targets
        self.closed: List[str] = []
        self._fail_ids = set(fail_ids or ())
        self._raise_on_list = raise_on_list

    def list_targets(self) -> List[Dict[str, Any]]:
        if self._raise_on_list:
            raise ConnectionError("boom")
        return list(self._targets)

    def close_target(self, target_id: str) -> bool:
        self.closed.append(target_id)
        return target_id not in self._fail_ids


@pytest.fixture(autouse=True)
def _clear_prune_env(monkeypatch):
    """Ensure a clean env so gating defaults are deterministic."""
    for var in (
        "HERMES_CDP_PRUNE_ENABLED",
        "HERMES_CDP_PRUNE_DRY_RUN",
        "HERMES_CDP_PRUNE_SCRATCH",
        "HERMES_CDP_PRUNE_ON_BLOCK",
        "HERMES_CDP_PRUNE_ENDPOINT",
        "HERMES_CDP_PRUNE_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------------
# classify_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("about:blank", cp.CLASS_BLANK),
        ("", cp.CLASS_BLANK),
        ("about:blank#blocked", cp.CLASS_BLANK),
        ("chrome://newtab/", cp.CLASS_NEWTAB),
        ("chrome://newtab", cp.CLASS_NEWTAB),
        ("brave://newtab/", cp.CLASS_NEWTAB),
        ("chrome://new-tab-page/", cp.CLASS_NEWTAB),
        ("edge://newtab/", cp.CLASS_NEWTAB),
        ("data:text/html,<h1>scratch</h1>", cp.CLASS_SCRATCH),
        ("https://example.com/article", cp.CLASS_REAL),
        ("http://localhost:3000/app", cp.CLASS_REAL),
        ("file:///Users/x/report.html", cp.CLASS_REAL),
        # non-newtab internal pages are real (preserve intentional settings tabs)
        ("chrome://settings/", cp.CLASS_REAL),
        ("chrome://extensions/", cp.CLASS_REAL),
    ],
)
def test_classify_url_basic(url, expected):
    assert cp.classify_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://www.chase.com/dashboard",
        "https://sellercentral.amazon.com/orders",
        "https://mychart.example.org/patient",
        "https://accounts.google.com/signin",
        "https://www.irs.gov/payments",
        "https://app.coinbase.com/trade",
        "https://vendor-portal.acme.com/login",
    ],
)
def test_classify_url_sensitive(url):
    assert cp.classify_url(url) == cp.CLASS_SENSITIVE
    assert cp.is_sensitive(url) is True


def test_blank_and_newtab_are_not_sensitive():
    assert cp.is_sensitive("about:blank") is False
    assert cp.is_sensitive("chrome://newtab/") is False


# ---------------------------------------------------------------------------
# plan_prune — the core safety contract
# ---------------------------------------------------------------------------


def test_plan_closes_disposable_keeps_real():
    targets = [
        _page("t-real", "https://news.example.com"),
        _page("t-blank", "about:blank"),
        _page("t-newtab", "chrome://newtab/"),
    ]
    plan = cp.plan_prune(targets)
    assert set(plan.to_close) == {"t-blank", "t-newtab"}
    # real page keeps the lane alive -> no forced preservation
    assert plan.preserved_safety_target is None
    real = next(d for d in plan.decisions if d.target_id == "t-real")
    assert real.action == "skip" and real.url_class == cp.CLASS_REAL


def test_plan_never_closes_real_or_sensitive():
    targets = [
        _page("t-blank", "about:blank"),
        _page("t-bank", "https://chase.com/account"),
        _page("t-shop", "https://sellercentral.amazon.com"),
        _page("t-doc", "https://docs.example.com/page"),
    ]
    plan = cp.plan_prune(targets)
    assert plan.to_close == ["t-blank"]
    for tid in ("t-bank", "t-shop", "t-doc"):
        d = next(d for d in plan.decisions if d.target_id == tid)
        assert d.action == "skip"


def test_plan_preserves_one_page_when_all_disposable():
    """Coordinator lane with only its intentional about:blank must keep it."""
    targets = [
        _page("t-blank", "about:blank"),
        _page("t-nt1", "chrome://newtab/"),
        _page("t-nt2", "chrome://newtab/"),
    ]
    plan = cp.plan_prune(targets)
    # exactly one page survives, and it's the about:blank (preferred)
    assert plan.preserved_safety_target == "t-blank"
    assert set(plan.to_close) == {"t-nt1", "t-nt2"}
    kept = next(d for d in plan.decisions if d.target_id == "t-blank")
    assert kept.action == "skip" and kept.reason == "preserve-last-page"


def test_plan_preserves_newtab_when_no_blank():
    targets = [
        _page("t-nt1", "chrome://newtab/"),
        _page("t-nt2", "chrome://newtab/"),
    ]
    plan = cp.plan_prune(targets)
    assert plan.preserved_safety_target == "t-nt1"
    assert plan.to_close == ["t-nt2"]


def test_plan_single_blank_is_preserved_not_closed():
    targets = [_page("t-only", "about:blank")]
    plan = cp.plan_prune(targets)
    assert plan.to_close == []
    assert plan.preserved_safety_target == "t-only"


def test_plan_scratch_gated_off_by_default():
    targets = [
        _page("t-real", "https://x.example.com"),
        _page("t-scratch", "data:text/html,<b>hi</b>"),
    ]
    plan = cp.plan_prune(targets)
    assert plan.to_close == []  # scratch not closed by default
    d = next(d for d in plan.decisions if d.target_id == "t-scratch")
    assert d.url_class == cp.CLASS_SCRATCH and d.action == "skip"


def test_plan_scratch_closed_when_allowed():
    targets = [
        _page("t-real", "https://x.example.com"),
        _page("t-scratch", "data:text/html,<b>hi</b>"),
    ]
    plan = cp.plan_prune(targets, allow_scratch=True)
    assert plan.to_close == ["t-scratch"]


def test_plan_ignores_non_page_targets():
    targets = [
        _page("sw", "https://x.example.com/sw.js", ttype="service_worker"),
        _page("bg", "chrome-extension://abc/bg", ttype="background_page"),
        _page("wk", "about:blank", ttype="worker"),
        _page("t-blank", "about:blank", ttype="page"),
        _page("t-real", "https://x.example.com", ttype="page"),
    ]
    plan = cp.plan_prune(targets)
    # only the page-type about:blank is closed; workers/bg never touched
    assert plan.to_close == ["t-blank"]
    for tid in ("sw", "bg", "wk"):
        d = next(d for d in plan.decisions if d.target_id == tid)
        assert d.action == "skip" and d.url_class == "non_page"


def test_plan_handles_missing_target_id():
    targets = [{"type": "page", "url": "about:blank"}]  # no id
    plan = cp.plan_prune(targets)
    assert plan.to_close == []
    assert plan.decisions[0].reason == "no-target-id"


def test_plan_accepts_target_getTargets_shape():
    """Normalizer handles the CDP Target.getTargets 'targetId' key too."""
    targets = [
        {"targetId": "t1", "type": "page", "url": "about:blank"},
        {"targetId": "t2", "type": "page", "url": "https://real.example.com"},
    ]
    plan = cp.plan_prune(targets)
    assert plan.to_close == ["t1"]


def test_plan_counts_are_sanitized():
    targets = [
        _page("t-blank", "about:blank", title="Secret Bank Statement"),
        _page("t-real", "https://chase.com/acct?token=abcd", title="My Balance"),
    ]
    plan = cp.plan_prune(targets)
    blob = repr(plan.counts)
    # counts expose classes + numbers only, never URLs/titles/tokens
    assert "chase.com" not in blob
    assert "token" not in blob
    assert "Secret" not in blob
    assert plan.counts["class:blank"] == 1
    assert plan.counts["class:sensitive"] == 1


# ---------------------------------------------------------------------------
# prune_endpoint — I/O orchestration via injected fake client
# ---------------------------------------------------------------------------


def test_prune_endpoint_dry_run_closes_nothing():
    client = FakeCdpClient(
        [_page("t-blank", "about:blank"), _page("t-real", "https://x.example.com")]
    )
    summary = cp.prune_endpoint("http://127.0.0.1:18838", dry_run=True, client=client)
    assert client.closed == []  # nothing closed in dry-run
    assert summary["would_close"] == 1
    assert summary["dry_run"] is True
    assert summary["lane"] == "127.0.0.1:18838"


def test_prune_endpoint_live_closes_disposable_only():
    client = FakeCdpClient(
        [
            _page("t-blank", "about:blank"),
            _page("t-newtab", "chrome://newtab/"),
            _page("t-real", "https://x.example.com"),
            _page("t-bank", "https://chase.com/acct"),
        ]
    )
    summary = cp.prune_endpoint("http://127.0.0.1:18838", dry_run=False, client=client)
    assert set(client.closed) == {"t-blank", "t-newtab"}
    assert summary["closed"] == 2
    assert summary["close_failed"] == 0


def test_prune_endpoint_live_preserves_last_page():
    client = FakeCdpClient(
        [_page("t-blank", "about:blank"), _page("t-nt", "chrome://newtab/")]
    )
    summary = cp.prune_endpoint("http://127.0.0.1:18838", dry_run=False, client=client)
    # one page preserved; only one newtab actually closed
    assert client.closed == ["t-nt"]
    assert summary["preserved_last_page"] is True
    assert summary["closed"] == 1


def test_prune_endpoint_counts_close_failures():
    client = FakeCdpClient(
        [
            _page("t-blank", "about:blank"),
            _page("t-nt", "chrome://newtab/"),
            _page("t-real", "https://x.example.com"),
        ],
        fail_ids={"t-nt"},
    )
    summary = cp.prune_endpoint("http://127.0.0.1:18838", dry_run=False, client=client)
    assert summary["closed"] == 1
    assert summary["close_failed"] == 1


def test_prune_endpoint_unreachable_is_soft_error():
    client = FakeCdpClient([], raise_on_list=True)
    summary = cp.prune_endpoint("http://127.0.0.1:19999", dry_run=False, client=client)
    assert "error" in summary
    assert summary["closed"] == 0
    assert client.closed == []


# ---------------------------------------------------------------------------
# normalize_base_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("http://127.0.0.1:18838", "http://127.0.0.1:18838"),
        ("http://127.0.0.1:18838/json/version", "http://127.0.0.1:18838"),
        ("ws://127.0.0.1:18838/devtools/browser/abc", "http://127.0.0.1:18838"),
        ("wss://host:9222/devtools/browser/x", "https://host:9222"),
        ("127.0.0.1:18838", "http://127.0.0.1:18838"),
        ("", ""),
    ],
)
def test_normalize_base_url(endpoint, expected):
    assert cp.normalize_base_url(endpoint) == expected


# ---------------------------------------------------------------------------
# prune_after_transition — gating
# ---------------------------------------------------------------------------


def test_transition_disabled_by_default(monkeypatch):
    called = []
    monkeypatch.setattr(cp, "prune_endpoint", lambda *a, **k: called.append(1))
    assert cp.prune_after_transition(event="completed", task_id="t1") is None
    assert called == []


def test_transition_enabled_no_endpoint_is_noop(monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")
    monkeypatch.setattr(cp, "resolve_endpoints", lambda: [])
    called = []
    monkeypatch.setattr(cp, "prune_endpoint", lambda *a, **k: called.append(1))
    assert cp.prune_after_transition(event="completed", task_id="t1") is None
    assert called == []


def test_transition_completed_runs_dry_by_default(monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")
    monkeypatch.setattr(cp, "resolve_endpoints", lambda: ["http://127.0.0.1:18838"])
    seen = {}

    def _fake_prune(endpoint, *, dry_run, allow_scratch, timeout):
        seen["dry_run"] = dry_run
        seen["allow_scratch"] = allow_scratch
        return {"lane": "127.0.0.1:18838", "would_close": 3, "closed": 0}

    monkeypatch.setattr(cp, "prune_endpoint", _fake_prune)
    agg = cp.prune_after_transition(event="completed", task_id="t1")
    assert agg is not None
    assert agg["dry_run"] is True  # dry-run default even when enabled
    assert seen["dry_run"] is True
    assert seen["allow_scratch"] is False
    assert agg["would_close"] == 3
    assert agg["lanes"] == 1


def test_transition_live_when_dry_run_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")
    monkeypatch.setenv("HERMES_CDP_PRUNE_DRY_RUN", "0")
    monkeypatch.setenv("HERMES_CDP_PRUNE_SCRATCH", "1")
    monkeypatch.setattr(cp, "resolve_endpoints", lambda: ["http://127.0.0.1:18838"])
    seen = {}

    def _fake_prune(endpoint, *, dry_run, allow_scratch, timeout):
        seen["dry_run"] = dry_run
        seen["allow_scratch"] = allow_scratch
        return {"closed": 2}

    monkeypatch.setattr(cp, "prune_endpoint", _fake_prune)
    agg = cp.prune_after_transition(event="completed", task_id="t1")
    assert seen["dry_run"] is False
    assert seen["allow_scratch"] is True
    assert agg["closed"] == 2


def test_transition_block_requires_opt_in(monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")
    monkeypatch.setattr(cp, "resolve_endpoints", lambda: ["http://127.0.0.1:18838"])
    called = []
    monkeypatch.setattr(cp, "prune_endpoint", lambda *a, **k: called.append(1) or {})
    # block without opt-in -> no-op
    assert cp.prune_after_transition(event="blocked", task_id="t1") is None
    assert called == []


def test_transition_block_never_closes_scratch(monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")
    monkeypatch.setenv("HERMES_CDP_PRUNE_ON_BLOCK", "1")
    monkeypatch.setenv("HERMES_CDP_PRUNE_SCRATCH", "1")  # even with scratch on...
    monkeypatch.setattr(cp, "resolve_endpoints", lambda: ["http://127.0.0.1:18838"])
    seen = {}

    def _fake_prune(endpoint, *, dry_run, allow_scratch, timeout):
        seen["allow_scratch"] = allow_scratch
        return {}

    monkeypatch.setattr(cp, "prune_endpoint", _fake_prune)
    cp.prune_after_transition(event="blocked", task_id="t1")
    # ...block path forces scratch off
    assert seen["allow_scratch"] is False


def test_transition_never_raises(monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")

    def _boom():
        raise RuntimeError("resolve failed")

    monkeypatch.setattr(cp, "resolve_endpoints", _boom)
    # swallowed -> None, not an exception
    assert cp.prune_after_transition(event="completed", task_id="t1") is None


# ---------------------------------------------------------------------------
# resolve_endpoints
# ---------------------------------------------------------------------------


def test_resolve_endpoints_explicit_env(monkeypatch):
    monkeypatch.setenv(
        "HERMES_CDP_PRUNE_ENDPOINT",
        "http://127.0.0.1:18838, ws://127.0.0.1:18830/devtools/browser/x",
    )
    eps = cp.resolve_endpoints()
    assert eps == ["http://127.0.0.1:18838", "http://127.0.0.1:18830"]


def test_resolve_endpoints_dedupes(monkeypatch):
    monkeypatch.setenv(
        "HERMES_CDP_PRUNE_ENDPOINT",
        "http://127.0.0.1:18838,http://127.0.0.1:18838/json/version",
    )
    assert cp.resolve_endpoints() == ["http://127.0.0.1:18838"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_no_endpoint_returns_2(monkeypatch, capsys):
    monkeypatch.setattr(cp, "resolve_endpoints", lambda: [])
    rc = cp.main([])
    assert rc == 2
    assert "no endpoint" in capsys.readouterr().out


def test_cli_dry_run_default(monkeypatch, capsys):
    seen = {}

    def _fake_prune(endpoint, *, dry_run, allow_scratch, timeout):
        seen["dry_run"] = dry_run
        return {"lane": "127.0.0.1:18838", "would_close": 0}

    monkeypatch.setattr(cp, "prune_endpoint", _fake_prune)
    rc = cp.main(["--endpoint", "http://127.0.0.1:18838"])
    assert rc == 0
    assert seen["dry_run"] is True  # CLI defaults to dry-run
