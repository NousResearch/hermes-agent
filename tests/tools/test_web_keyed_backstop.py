"""One-shot keyed backstop: a keyless call whose ring came back fully
throttled retries ONCE on the vendor's keyed path; the NEXT call starts on
the free ring again.

Mirror image of the keyless rescue in test_web_keyless_rescue.py.

Covers:
- eligibility: only fires on the ring's own exhaustion verdict, only for a
  keyless-mode ring vendor that actually has an API key, never for a vendor
  pinned "free", and never when a keyed call failed (that's the other
  rescue's job)
- search dispatcher: exhausted ring → keyed retry, annotated with
  backstopped_from + backend_error
- statelessness: the next dispatch goes back to the keyless ring
- force_keyed: scoped, resets on exit AND on exception, and does not leak
  to other vendors
- extract dispatcher: whole-batch throttling backstops; a partial failure
  and a policy block do not
- backstop failure: the original ring error survives with a note appended
"""

import asyncio
import json
from unittest.mock import patch

import pytest

import tools.web_tools as web_tools
from plugins.web import keyless_mcp

RING_EXHAUSTED = (
    "429 rate limit (all keyless vendors throttled: exa, parallel, tavily, "
    "firecrawl, keenable)"
)


class _KeylessExhaustedProvider:
    """Ring vendor whose keyless path reports every vendor throttled.

    Models the real dispatcher shape: the provider routes keyless by
    default and only takes its keyed branch while :func:`force_keyed` is
    active — which is exactly what the backstop does. Its keyed path
    succeeds, so the backstop has something real to fall back to.
    """

    name = "exa"
    display_name = "Exa"

    def __init__(self):
        self.search_calls = []
        self.extract_calls = []

    def supports_search(self):
        return True

    def supports_extract(self):
        return True

    def is_available(self):
        return True

    def _keyed(self):
        """True only under an active force_keyed override for this vendor.

        The real providers ask ``use_keyless()``; here we read the override
        directly so the double stays honest regardless of how the ambient
        tier resolves in a given test.
        """
        return keyless_mcp._forced_keyed.get() == self.name

    def search(self, query, limit=5):
        keyed = self._keyed()
        self.search_calls.append("keyed" if keyed else "keyless")
        if keyed:
            return {
                "success": True,
                "data": {"web": [{"url": "https://k", "title": "keyed result"}]},
            }
        return {"success": False, "error": RING_EXHAUSTED}

    def extract(self, urls, **kwargs):
        keyed = self._keyed()
        self.extract_calls.append("keyed" if keyed else "keyless")
        if keyed:
            return [
                {"url": u, "title": "keyed", "content": "body", "error": ""}
                for u in urls
            ]
        return [
            {"url": u, "title": "", "content": "", "error": "429 rate limit"}
            for u in urls
        ]


@pytest.fixture(autouse=True)
def _keyless_exa_env(monkeypatch):
    """Exa pinned to the free tier with a key on file, backstop opted in.

    This is the backstop's real domain: a ``free`` pin is the only config
    that runs the keyless ring while an API key exists to fall back to.

    ``keyed_backstop`` is set True explicitly because the feature ships
    OPT-IN (default false). Tests that assert the default itself override
    this fixture with an empty config -- see ``TestOptInDefault``.
    """
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda var: "sk-test-key" if var == "EXA_API_KEY" else "",
        raising=True,
    )
    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"keyed_backstop": True}
    )
    monkeypatch.setattr(web_tools, "_backstop_key_for",
                        lambda name: "sk-test-key" if name == "exa" else "")
    monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "free")
    monkeypatch.setattr(keyless_mcp, "keyless_enabled", lambda: True)
    monkeypatch.setattr(
        "agent.web_search_registry._keyless_tier_enabled", lambda: True
    )
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)


def _dispatch_search(monkeypatch, provider, query="q", limit=2):
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda name: provider
    )
    return json.loads(web_tools.web_search_tool(query, limit=limit))


def _dispatch_extract(monkeypatch, provider, urls):
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda name: provider
    )

    async def _allow_all(url, **kwargs):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
    return asyncio.run(web_tools.web_extract_tool(urls))


# ── force_keyed contextvar ───────────────────────────────────────────────

def test_force_keyed_scopes_and_resets():
    assert keyless_mcp.use_keyless("exa", "") is True
    with keyless_mcp.force_keyed("exa"):
        assert keyless_mcp.use_keyless("exa", "") is False
    assert keyless_mcp.use_keyless("exa", "") is True


def test_force_keyed_resets_on_exception():
    with pytest.raises(RuntimeError):
        with keyless_mcp.force_keyed("exa"):
            raise RuntimeError("boom")
    assert keyless_mcp.use_keyless("exa", "") is True, "override leaked"


def test_force_keyed_does_not_affect_other_vendors():
    with keyless_mcp.force_keyed("exa"):
        assert keyless_mcp.use_keyless("parallel", "") is True


# ── exhaustion detection ─────────────────────────────────────────────────

def test_ring_exhausted_matches_only_the_real_verdict():
    assert keyless_mcp.ring_exhausted(RING_EXHAUSTED) is True
    assert keyless_mcp.ring_exhausted("429 rate limit") is False
    assert keyless_mcp.ring_exhausted("HTTP 500 boom") is False
    assert keyless_mcp.ring_exhausted("") is False


def test_extract_ring_exhausted_requires_all_throttled():
    throttled = [{"error": "429 rate limit"}, {"error": "too many requests"}]
    mixed = [{"error": "429 rate limit"}, {"error": "404 not found"}]
    assert keyless_mcp.extract_ring_exhausted(throttled) is True
    assert keyless_mcp.extract_ring_exhausted(mixed) is False
    assert keyless_mcp.extract_ring_exhausted([]) is False


# ── eligibility ──────────────────────────────────────────────────────────

def test_eligible_for_exhausted_keyless_ring_vendor():
    assert web_tools._backstop_eligible(
        _KeylessExhaustedProvider(), RING_EXHAUSTED
    ) is True


def test_not_eligible_for_a_single_vendor_error():
    assert web_tools._backstop_eligible(
        _KeylessExhaustedProvider(), "HTTP 500 upstream exploded"
    ) is False


def test_not_eligible_without_an_api_key(monkeypatch):
    monkeypatch.setattr(web_tools, "_backstop_key_for", lambda name: "")
    assert web_tools._backstop_eligible(
        _KeylessExhaustedProvider(), RING_EXHAUSTED
    ) is False


def test_free_pin_is_the_target_case_not_an_exclusion():
    """A ``free`` pin is exactly when the ring runs with a key available.

    Excluding it would make the backstop unreachable; "never spend" is
    expressed with web.keyed_backstop: false instead.
    """
    assert web_tools._backstop_eligible(
        _KeylessExhaustedProvider(), RING_EXHAUSTED
    ) is True


def test_keyed_failure_never_reaches_the_backstop(monkeypatch):
    """A keyed backend that fails is the keyless rescue's job, not ours.

    The dispatcher checks _rescue_eligible first, so the backstop branch is
    an `elif` that never sees a keyed failure. Asserted so a future
    refactor of that chain can't silently double-rescue.
    """
    monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "paid")

    class _KeyedBoom(_KeylessExhaustedProvider):
        def search(self, query, limit=5):
            self.search_calls.append("keyed")
            return {"success": False, "error": "401 invalid api key"}

    provider = _KeyedBoom()
    # A keyed (paid-pinned) vendor failing is eligible for the KEYLESS rescue.
    assert web_tools._rescue_eligible(provider) is True

    ring_result = {"success": True, "data": {"web": [{"url": "https://r"}]}}
    with patch.object(
        keyless_mcp, "search_with_failover", return_value=ring_result
    ) as ring:
        payload = _dispatch_search(monkeypatch, provider)

    assert ring.called, "keyed failure must go to the keyless rescue"
    assert provider.search_calls == ["keyed"], "must not re-run the keyed path"
    assert payload["data"]["rescued_from"] == "exa"


def test_the_two_gates_are_mutually_exclusive(monkeypatch):
    """No single failure may satisfy both rescue and backstop."""
    provider = _KeylessExhaustedProvider()
    rescue = web_tools._rescue_eligible(provider)
    backstop = web_tools._backstop_eligible(provider, RING_EXHAUSTED)
    assert not (rescue and backstop), "a failure triggered both paths"


def test_disabled_by_config(monkeypatch):
    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"keyed_backstop": False}
    )
    assert web_tools._backstop_eligible(
        _KeylessExhaustedProvider(), RING_EXHAUSTED
    ) is False


def test_off_when_keyless_tier_disabled(monkeypatch):
    monkeypatch.setattr(
        "agent.web_search_registry._keyless_tier_enabled", lambda: False
    )
    assert web_tools._backstop_eligible(
        _KeylessExhaustedProvider(), RING_EXHAUSTED
    ) is False


# ── search dispatcher ────────────────────────────────────────────────────

def test_search_backstops_onto_the_keyed_path(monkeypatch):
    provider = _KeylessExhaustedProvider()
    payload = _dispatch_search(monkeypatch, provider)

    assert payload["success"] is True
    assert payload["data"]["web"][0]["title"] == "keyed result"
    assert payload["data"]["backstopped_from"] == "keyless"
    assert "exhausted" in payload["data"]["backend_error"]
    assert provider.search_calls == ["keyless", "keyed"]


def test_search_backstop_is_not_sticky(monkeypatch):
    """The next call must start on the free ring again."""
    provider = _KeylessExhaustedProvider()
    _dispatch_search(monkeypatch, provider, query="first")
    _dispatch_search(monkeypatch, provider, query="second")
    assert provider.search_calls == ["keyless", "keyed", "keyless", "keyed"]


def test_search_backstop_failure_keeps_the_ring_error(monkeypatch):
    class _BothFail(_KeylessExhaustedProvider):
        def search(self, query, limit=5):
            keyed = self._keyed()
            self.search_calls.append("keyed" if keyed else "keyless")
            if keyed:
                return {"success": False, "error": "401 invalid api key"}
            return {"success": False, "error": RING_EXHAUSTED}

    payload = _dispatch_search(monkeypatch, _BothFail())

    assert payload["success"] is False
    assert "all keyless vendors throttled" in payload["error"]
    assert "keyed backstop also failed" in payload["error"]
    assert "401 invalid api key" in payload["error"]


def test_search_backstop_survives_a_raising_keyed_path(monkeypatch):
    class _KeyedRaises(_KeylessExhaustedProvider):
        def search(self, query, limit=5):
            keyed = self._keyed()
            self.search_calls.append("keyed" if keyed else "keyless")
            if keyed:
                raise RuntimeError("sdk exploded")
            return {"success": False, "error": RING_EXHAUSTED}

    payload = _dispatch_search(monkeypatch, _KeyedRaises())

    assert payload["success"] is False
    assert "sdk exploded" in payload["error"]
    # The override must not survive a raising provider.
    assert keyless_mcp.use_keyless("exa", "") is True


# ── extract dispatcher ───────────────────────────────────────────────────

def test_extract_backstops_when_whole_batch_throttled(monkeypatch):
    provider = _KeylessExhaustedProvider()
    raw = _dispatch_extract(monkeypatch, provider, ["https://example.com/a"])
    assert provider.extract_calls == ["keyless", "keyed"]
    assert "keyed" in raw


def test_extract_partial_failure_is_not_backstopped(monkeypatch):
    class _PartialProvider(_KeylessExhaustedProvider):
        def extract(self, urls, **kwargs):
            self.extract_calls.append("keyless")
            return [
                {"url": urls[0], "title": "ok", "content": "body", "error": ""},
                {"url": urls[1], "title": "", "content": "", "error": "429 rate limit"},
            ]

    provider = _PartialProvider()
    _dispatch_extract(
        monkeypatch, provider, ["https://example.com/a", "https://example.com/b"]
    )
    assert provider.extract_calls == ["keyless"], "partial failure must not backstop"


def test_extract_policy_block_is_never_backstopped(monkeypatch):
    class _BlockedProvider(_KeylessExhaustedProvider):
        def extract(self, urls, **kwargs):
            self.extract_calls.append("keyless")
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": "blocked by website policy",
                    "blocked_by_policy": True,
                }
                for u in urls
            ]

    provider = _BlockedProvider()
    _dispatch_extract(monkeypatch, provider, ["https://blocked.example"])
    assert provider.extract_calls == ["keyless"], "policy block must not backstop"


class TestForcedKeyedReentrancy:
    """The forced-keyed override is scoped to ONE vendor name and must not
    turn a nested dispatch into a second silent spend.

    Raised in review: if a provider's keyed path ever performs its own nested
    ring dispatch (re-search, enrichment), that nested call inherits the
    ContextVar and spends again. No shipped provider does this today -- every
    vendor's ``use_keyless`` gate ``return``s immediately on the keyless branch
    and the keyed branch calls the vendor SDK directly -- so these tests pin
    the property rather than fix a live bug.
    """

    def test_override_is_scoped_to_one_vendor(self):
        """A different vendor inside the block is unaffected."""
        from plugins.web import keyless_mcp

        with keyless_mcp.force_keyed("exa"):
            # the forced vendor routes keyed
            assert keyless_mcp.use_keyless("exa", "key") is False
            # a DIFFERENT vendor still follows its own tier
            with patch.object(
                keyless_mcp, "provider_tier", lambda n: "free"
            ):
                assert keyless_mcp.use_keyless("tavily", "key") is True

    def test_nested_same_vendor_dispatch_would_stay_keyed(self):
        """Documents the inherited-context behaviour explicitly.

        If a future provider nests a same-vendor call inside its keyed path,
        that call stays keyed. That is the reviewer's concern made visible:
        the assertion below is the thing a future contributor must
        deliberately change if nesting is ever introduced.
        """
        from plugins.web import keyless_mcp

        with keyless_mcp.force_keyed("exa"):
            with keyless_mcp.force_keyed("exa"):
                assert keyless_mcp.use_keyless("exa", "key") is False
            # inner block exits, outer override still in force
            assert keyless_mcp.use_keyless("exa", "key") is False
        assert keyless_mcp._forced_keyed.get() is None

    def test_no_shipped_provider_nests_a_ring_dispatch(self):
        """Guard: a vendor's KEYED path must not call back into the ring.

        Fails if someone adds a nested ``*_with_failover`` call to a keyed
        branch, which is exactly the condition that would make the override
        leak into a second spend.
        """
        import ast
        import pathlib

        repo = pathlib.Path(__file__).resolve().parents[2]
        offenders = []
        for name in ("exa", "parallel", "tavily", "firecrawl", "keenable"):
            path = repo / "plugins" / "web" / name / "provider.py"
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            for fn in ast.walk(tree):
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if fn.name not in ("search", "extract"):
                    continue
                for node in ast.walk(fn):
                    if not isinstance(node, ast.If):
                        continue
                    test_src = ast.dump(node.test)
                    if "use_keyless" not in test_src:
                        continue
                    # Every statement guarded by use_keyless must terminate the
                    # function; the keyed path below it must be unreachable
                    # from inside the keyless branch.
                    if not any(
                        isinstance(s, ast.Return) for s in node.body
                    ):
                        offenders.append(f"{name}.{fn.name}: keyless branch does not return")
                    # the ELSE/fallthrough (keyed) path must not re-dispatch
                    after = [
                        n for n in ast.walk(fn)
                        if isinstance(n, ast.Call)
                        and getattr(n.func, "id", "") in (
                            "search_with_failover", "extract_with_failover",
                        )
                    ]
                    # all such calls must live inside the keyless branch
                    inside = [
                        n for n in ast.walk(node)
                        if isinstance(n, ast.Call)
                        and getattr(n.func, "id", "") in (
                            "search_with_failover", "extract_with_failover",
                        )
                    ]
                    if len(after) != len(inside):
                        offenders.append(
                            f"{name}.{fn.name}: ring dispatch outside the keyless branch"
                        )
        assert not offenders, "; ".join(offenders)


class TestOptInDefault:
    """The backstop must never spend without an explicit opt-in.

    Raised in review: defaulting this on makes paid-API spending a silent
    default for anyone with a key on file. It ships default-OFF; these tests
    pin that so a later refactor can't quietly flip it.
    """

    def test_config_default_is_false(self):
        """The shipped default in config_defaults.py is False."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["web"]["keyed_backstop"] is False

    def test_absent_key_means_disabled(self):
        """A config with no keyed_backstop entry must read as disabled."""
        with patch.object(web_tools, "_load_web_config", lambda: {}):
            assert web_tools._keyed_backstop_enabled() is False

    def test_explicit_true_enables_it(self):
        """Opting in works (and the enable path isn't dead)."""
        with patch.object(
            web_tools, "_load_web_config", lambda: {"keyed_backstop": True}
        ), patch(
            "agent.web_search_registry._keyless_tier_enabled", lambda: True
        ):
            assert web_tools._keyed_backstop_enabled() is True

    def test_disabled_backstop_is_never_eligible(self):
        """With the flag off, no failure shape can reach the backstop."""
        provider = _KeylessExhaustedProvider()
        with patch.object(web_tools, "_load_web_config", lambda: {}):
            assert web_tools._backstop_eligible(
                provider, keyless_mcp._RING_EXHAUSTED_MARKER
            ) is False
