"""Public dispatch contracts shared with the comparison PR (both config dialects)."""
import asyncio
import json

import pytest

from tools import web_tools


class Provider:
    def __init__(self, name, response):
        self.name = self.display_name = name
        self.response = response
        self.calls = []

    def is_available(self):
        return True

    def supports_search(self):
        return True

    def supports_extract(self):
        return True

    def search(self, query, limit=5):
        self.calls.append(query)
        return self.response

    def extract(self, urls, **kwargs):
        self.calls.append(list(urls))
        return self.response


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    import socket

    attempts = []

    def denied(*args, **kwargs):
        attempts.append("unexpected network call")
        raise AssertionError("Network access is not allowed in fallback contract tests")

    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    yield
    assert not attempts, "A caught exception hid an unexpected network call"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    # Real YAML loader, real registry tool handler, isolated on-disk cache.
    import tools.web_result_cache as cache
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "web:\n  search_backend: primary\n  extract_backend: primary\n"
        "  fallback_backend: secondary\n  fallback_enabled: true\n"
        "  fallback_backends: [secondary]\n  keyless_fallback: false\n"
        "  keyless_rescue: false\n", encoding="utf-8"
    )
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    if hasattr(cache, "search_memo"):
        monkeypatch.setattr(cache, "search_memo", cache.SearchMemo())
    providers = {}
    monkeypatch.setattr("agent.web_search_registry.get_provider", providers.get)
    monkeypatch.setattr(web_tools, "_is_backend_available", lambda name: name in providers)

    async def safe(*args, **kwargs):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", safe)
    return providers


@pytest.mark.parametrize("failed", [False, True], ids=["empty-success-is-terminal", "failed-primary-nonsticky"])
def test_search_only_fails_over_after_failure(configured, failed):
    response = {"success": False, "error": "outage"} if failed else {"success": True, "data": {"web": []}}
    primary = Provider("primary", response)
    secondary = Provider("secondary", {"success": True, "data": {"web": [{"url": "https://example.com", "title": "ok"}]}})
    configured.update(primary=primary, secondary=secondary)
    for _ in range(2):
        result = json.loads(web_tools.registry.get_entry("web_search").handler({"query": "same"}))
        assert result["success"]
        assert bool(result["data"]["web"]) == failed
    assert len(secondary.calls) == (2 if failed else 0)
    if failed:
        assert len(primary.calls) == 2


@pytest.mark.parametrize("mode", ["provider-policy", "global-policy", "partial", "empty", "reordered-policy" ])
def test_extract_policy_and_batch_identity(configured, monkeypatch, mode):
    urls = ["https://example.com/a", "https://example.com/b"]
    blocked = {"url": urls[0], "error": "Blocked by website policy", "blocked_by_policy": True}
    failure = {"url": urls[1], "error": "outage"}
    success = {"url": urls[0], "content": "primary text"}
    primary_rows = {
        "provider-policy": [blocked, failure],
        "global-policy": [blocked, failure],
        "partial": [success, failure],
        "empty": [],
        "reordered-policy": [failure, blocked],
    }[mode]
    primary = Provider("primary", primary_rows)
    secondary = Provider("secondary", [{"url": url, "content": "secondary text"} for url in urls])
    configured.update(primary=primary, secondary=secondary)
    if mode == "global-policy":
        monkeypatch.setattr("tools.website_policy.check_website_access", lambda url: {"message": "Blocked by website policy"})
    result = json.loads(asyncio.run(web_tools.registry.get_entry("web_extract").handler({"urls": urls})))
    rows = result["results"]
    assert [r["url"] for r in rows] == urls
    if mode == "global-policy":
        assert not primary.calls and not secondary.calls
        assert all(r["blocked_by_policy"] for r in rows)
    elif mode in {"provider-policy", "reordered-policy"}:
        assert rows[0]["blocked_by_policy"]
        assert secondary.calls == [[urls[1]]]
    elif mode == "partial":
        assert rows[0]["content"] == "primary text" and rows[1]["error"] == "outage"
        assert not secondary.calls
    else:
        assert all(r["content"] == "secondary text" for r in rows)
        assert secondary.calls == [urls]


@pytest.mark.parametrize("secondary", [False, True], ids=["ring", "configured"])
@pytest.mark.parametrize("kind", ["none", "wrapper", "missing-id", "unknown", "duplicate", "reordered", "later-only", "short-policy"])
def test_untrusted_retry_batches(configured, monkeypatch, secondary, kind):
    from tools import web_tools_rescue as rescue
    urls = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
    original = [{"url": url, "error": "primary outage"} for url in urls]
    original[0]["blocked_by_policy"] = True
    b = {"url": urls[1], "content": "B"}
    c = {"url": urls[2], "content": "C"}
    batches = {
        "none": None, "wrapper": {"results": [b]}, "missing-id": [None, {"content": "wrong"}],
        "unknown": [{"url": "https://other.example", "content": "wrong"}],
        "duplicate": [b, {"url": urls[1], "content": "wrong"}],
        "reordered": [c, b], "later-only": [c],
        "short-policy": [{"url": urls[2], "blocked_by_policy": True, "error": "policy"}],
    }
    batch = batches[kind]
    if secondary:
        provider = Provider("secondary", batch)
        configured["secondary"] = provider
        result, _ = asyncio.run(rescue._try_fallback_extract("primary", urls, original))
        result = original if result is None else result
        assert provider.calls == [urls[1:]]
    else:
        calls = []
        def ring(name, requested):
            calls.append(requested)
            return batch
        monkeypatch.setattr("plugins.web.keyless_mcp.extract_with_failover", ring)
        result = rescue._rescue_extract("primary", urls, original)
        assert calls == [urls[1:]]
    assert [r["url"] for r in result] == urls
    assert result[0] == original[0]
    assert result[1] == (b if kind in {"duplicate", "reordered"} else original[1])
    if kind in {"reordered", "later-only"}:
        assert result[2]["content"] == "C"
    elif kind == "short-policy":
        assert result[2]["blocked_by_policy"]
    else:
        assert result[2] == original[2]


def test_duplicate_requested_urls_keep_queue_identity():
    from tools.web_tools_rescue import _map_extract_rows_by_url
    url = "https://example.com"
    rows = [{"url": url, "content": "first"}, {"url": url, "content": "second"}, {"url": url, "content": "excess"}]
    assert _map_extract_rows_by_url([url, url], [0, 1], rows) == {0: rows[0], 1: rows[1]}


def test_extract_fallback_does_not_stick_in_primary_cache(configured):
    url = "https://example.com/nonsticky"
    primary = Provider("primary", [{"url": url, "error": "outage"}])
    secondary = Provider("secondary", [{"url": url, "content": "fallback"}])
    configured.update(primary=primary, secondary=secondary)
    for _ in range(2):
        result = json.loads(asyncio.run(web_tools.registry.get_entry("web_extract").handler({"urls": [url]})))
        assert result["results"][0]["content"] == "fallback"
    assert primary.calls == [[url], [url]]
    assert secondary.calls == [[url], [url]]


@pytest.mark.parametrize("primary_kind", ["none-row", "short", "duplicate", "unknown", "wrapper"])
@pytest.mark.parametrize("blocked_index", [0, 1])
@pytest.mark.parametrize("policy_marker", ["flag", "message"])
def test_incomplete_primary_retains_short_secondary_policy(
    configured, monkeypatch, primary_kind, blocked_index, policy_marker
):
    """A safely identified refusal survives even when neither batch is complete."""
    urls = ["https://example.com/a", "https://example.com/b"]
    failure = {"url": urls[1 - blocked_index], "error": "primary outage"}
    primary_rows = {
        "none-row": [None], "short": [failure],
        "duplicate": [failure, dict(failure)],
        "unknown": [{"url": "https://other.example", "error": "outage"}],
        "wrapper": {"results": [failure]},
    }[primary_kind]
    blocked = {"url": urls[blocked_index], "error": "Blocked by website policy after redirect"}
    if policy_marker == "flag":
        blocked.update(error="refused", blocked_by_policy=True)
    primary = Provider("primary", primary_rows)
    secondary = Provider("secondary", [None, blocked, {"content": "unidentified"}])
    configured.update(primary=primary, secondary=secondary)
    monkeypatch.setattr("tools.web_tools_extract._rescue_eligible", lambda provider: True)
    ring_calls = []

    def ring(name, requested):
        ring_calls.append(requested)
        return [{"url": url, "content": "ring text"} for url in requested]

    monkeypatch.setattr("plugins.web.keyless_mcp.extract_with_failover", ring)
    entry = web_tools.registry.get_entry("web_extract")
    assert entry is not None
    for _ in range(2):
        result = json.loads(asyncio.run(entry.handler({"urls": urls})))
        assert all(urls[blocked_index] not in batch for batch in ring_calls)
        rows = result["results"]
        assert [row["url"] for row in rows] == urls
        assert rows[blocked_index]["error"] == blocked["error"]
        assert rows[1 - blocked_index]["content"] == "ring text"
    assert ring_calls == [[urls[1 - blocked_index]]] * 2
    assert primary.calls == [urls, urls]
    assert secondary.calls == [urls, urls]


@pytest.mark.parametrize("kind", ["empty", "none", "later", "unknown", "reordered", "partial-success"])
def test_partial_policy_keeps_exact_failures_and_rescue_identity(configured, monkeypatch, kind):
    urls = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
    failure = {"url": urls[2], "error": "original C outage"}
    blocked = {"url": urls[1], "error": "Blocked by website policy after redirect"}
    a = {"url": urls[0], "content": "A"}
    c = {"url": urls[2], "content": "C"}
    primary = Provider("primary", [failure, None])
    secondary_rows = [blocked, c] if kind == "partial-success" else [blocked]
    configured.update(primary=primary, secondary=Provider("secondary", secondary_rows))
    monkeypatch.setattr("tools.web_tools_extract._rescue_eligible", lambda provider: True)
    calls = []

    def ring(name, requested):
        calls.append(requested)
        return {
            "empty": [], "none": None, "later": [None, c],
            "unknown": [{"url": "https://other.example", "content": "wrong"}],
            "reordered": [c, a], "partial-success": [a, c],
        }[kind]

    monkeypatch.setattr("plugins.web.keyless_mcp.extract_with_failover", ring)
    entry = web_tools.registry.get_entry("web_extract")
    assert entry is not None
    result = json.loads(asyncio.run(entry.handler({"urls": urls})))
    rows = result["results"]
    assert [row["url"] for row in rows] == urls
    assert rows[1]["error"] == blocked["error"]
    assert calls == ([] if kind == "partial-success" else [[urls[0], urls[2]]])
    if kind in {"later", "reordered", "partial-success"}:
        assert rows[2]["content"] == "C"
    else:
        assert rows[2]["error"] == failure["error"]
    if kind == "reordered":
        assert rows[0]["content"] == "A"
    else:
        assert rows[0]["error"]
