"""Wall-clock caps on web_extract sub-phases.

Regression for the 2026-06-17 morning-brief outage: one flaky summarizer stream
(``peer closed connection ... incomplete chunked read``) retried through the
whole provider fallback chain and ran 3,362s — long past the 600s cron idle
limit — killing the brief. The per-attempt ``auxiliary.web_extract.timeout``
does not bound that aggregate; ``_web_extract_caps`` does. See
``tools/web_tools._web_extract_caps``.
"""

import asyncio

import pytest

import tools.web_tools as web_tools


@pytest.mark.asyncio
async def test_summarization_timeout_falls_back_to_raw_content(monkeypatch):
    """A summarizer that hangs past the cap degrades to truncated raw content
    rather than hanging the caller."""
    # Tiny cap; the summarizer "hangs" far past it.
    monkeypatch.setattr(web_tools, "_web_extract_caps", lambda: (0.05, 0.05))

    async def _hang(*args, **kwargs):
        await asyncio.sleep(5)
        return "SHOULD-NOT-RETURN"

    monkeypatch.setattr(web_tools, "_call_summarizer_llm", _hang)

    raw = "x" * 10_000
    # Outer guard: if the cap is not honored this raises instead of hanging CI.
    result = await asyncio.wait_for(
        web_tools.process_content_with_llm(raw, url="http://example.com", min_length=1),
        timeout=2.0,
    )

    assert result is not None
    assert "SHOULD-NOT-RETURN" not in result
    # Fell back to (truncated) raw content.
    assert result.startswith("x")


@pytest.mark.asyncio
async def test_summarization_succeeds_within_cap(monkeypatch):
    """A fast summarizer still returns its real summary unchanged."""
    monkeypatch.setattr(web_tools, "_web_extract_caps", lambda: (5.0, 5.0))

    async def _ok(*args, **kwargs):
        return "REAL SUMMARY"

    monkeypatch.setattr(web_tools, "_call_summarizer_llm", _ok)

    result = await web_tools.process_content_with_llm(
        "y" * 10_000, url="http://example.com", min_length=1
    )
    assert result == "REAL SUMMARY"


def test_caps_default_and_config_override(monkeypatch):
    # No explicit caps: summarize >= 2× per-attempt timeout (>=120 floor),
    # fetch == default.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **k: {"auxiliary": {"web_extract": {"timeout": 60}}},
    )
    summarize, fetch = web_tools._web_extract_caps()
    assert summarize >= 120.0
    assert fetch == web_tools._DEFAULT_FETCH_CAP_S

    # Explicit overrides win.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **k: {
            "auxiliary": {
                "web_extract": {"summarize_timeout": 7, "fetch_timeout": 9}
            }
        },
    )
    assert web_tools._web_extract_caps() == (7.0, 9.0)

    # A broken/empty config falls back to defaults instead of raising.
    def _boom(*a, **k):
        raise RuntimeError("no config")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    assert web_tools._web_extract_caps() == (
        web_tools._DEFAULT_SUMMARIZE_CAP_S,
        web_tools._DEFAULT_FETCH_CAP_S,
    )
