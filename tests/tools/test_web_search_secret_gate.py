"""web_search_tool blocks secret-shaped queries before any provider dispatch.

Mirrors the existing gate on web_extract_tool (agent/redact._PREFIX_RE against
URLs): the search query is the other free-text field that leaves the machine
via a third-party backend, and until this test/fix it was ungated.

Every assertion here is on the MECHANISM (was provider.search() invoked?),
never on the error string — a string match is a proxy for the guard, not
proof it ran before the network call.
"""
import json
from unittest.mock import MagicMock, patch

import tools.web_tools as web_tools


def _fake_provider(name="tavily"):
    provider = MagicMock(name=f"{name}Provider")
    provider.supports_search = MagicMock(return_value=True)
    provider.search = MagicMock(return_value={"success": True, "data": {"web": []}})
    provider.name = name
    return provider


class TestWebSearchSecretGate:
    def test_secret_shaped_query_never_reaches_provider(self):
        """The mechanism: provider.search must have zero calls, not just a
        particular return shape. A test that only checks the JSON body could
        pass even if the provider fired and the result was overwritten after
        the fact — asserting call count is what proves the network leg
        never happened."""
        provider = _fake_provider()
        secret_query = "look into sk-abc123DEF456ghi789JKLmno012PQRstu345 usage"

        with patch("tools.web_tools._get_search_backend", return_value="tavily"), \
             patch("agent.web_search_registry.get_provider", return_value=provider), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(web_tools._debug, "log_call"), \
             patch.object(web_tools._debug, "save"):
            result = web_tools.web_search_tool(secret_query, limit=3)

        provider.search.assert_not_called()
        parsed = json.loads(result)
        assert parsed["success"] is False

    def test_non_secret_query_is_dispatched_normally(self):
        """Regression guard for the fix itself: a query with no credential-shaped
        substring must still reach the provider — the gate must not swallow
        ordinary calls no matter how the block is implemented."""
        provider = _fake_provider()

        with patch("tools.web_tools._get_search_backend", return_value="tavily"), \
             patch("agent.web_search_registry.get_provider", return_value=provider), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(web_tools._debug, "log_call"), \
             patch.object(web_tools._debug, "save"):
            result = web_tools.web_search_tool("python asyncio best practices", limit=3)

        provider.search.assert_called_once()
        parsed = json.loads(result)
        assert parsed == {"success": True, "data": {"web": []}}

    def test_false_positive_sweep_ordinary_research_queries(self):
        """Hyphens, underscores, and code-like fragments in normal research
        queries must not trip the gate. A gate that fires on legitimate use
        gets disabled — this is the check that would have caught that."""
        provider = _fake_provider()
        queries = [
            "site:example.com machine-learning best practices",
            "python asyncio.gather vs asyncio.wait_for",
            "how to fix TypeError: 'NoneType' object is not subscriptable",
            "docker-compose environment_variables best-practice",
            "aws_access_key_id environment variable rotation policy",
            "ssh-keygen -t ed25519 tutorial",
            "-inurl:login intitle:\"password reset\"",
        ]

        with patch("tools.web_tools._get_search_backend", return_value="tavily"), \
             patch("agent.web_search_registry.get_provider", return_value=provider), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(web_tools._debug, "log_call"), \
             patch.object(web_tools._debug, "save"):
            for q in queries:
                provider.search.reset_mock()
                web_tools.web_search_tool(q, limit=3)
                assert provider.search.called, f"false positive: blocked {q!r}"

    def test_non_string_query_does_not_crash_the_gate(self):
        """query is not defensively coerced to str the way limit is; a bad
        model emission (int, None) must not turn the guard itself into an
        unhandled TypeError. Downstream cache-key normalization has its own
        pre-existing non-string crash — out of scope here — but the secret
        gate specifically must not add a second failure mode."""
        provider = _fake_provider()

        with patch("tools.web_tools._get_search_backend", return_value="tavily"), \
             patch("agent.web_search_registry.get_provider", return_value=provider), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(web_tools._debug, "log_call"), \
             patch.object(web_tools._debug, "save"):
            result = web_tools.web_search_tool(12345, limit=3)  # type: ignore[arg-type]

        # The gate itself must not raise; whatever happens downstream is a
        # separate, pre-existing concern (query.strip() in the cache-key
        # path also lacks a string guard, unrelated to this fix).
        assert isinstance(result, str)

    def test_web_extract_tool_gate_is_unmodified(self):
        """Scope check: this fix touches web_search_tool only. web_extract_tool's
        own four-way URL gate (_PREFIX_RE over raw/unquoted/normalized forms) must
        keep working exactly as before."""
        from agent.redact import _PREFIX_RE

        assert _PREFIX_RE.search("sk-abc123DEF456ghi789JKLmno012PQRstu345")
        assert not _PREFIX_RE.search("https://example.com/docs/api-reference")
