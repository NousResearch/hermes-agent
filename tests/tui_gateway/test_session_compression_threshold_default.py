"""An unset ``codex_responses_compact_threshold`` is "automatic", not "invalid".

``config_defaults`` materializes the key as ``None`` for profiles that never set it, so the live
config sync used to log ``Invalid compression.codex_responses_compact_threshold=None; using 200000``
on every session open — noise that reads like a broken config. Absence keeps the silent 200000
default (the same semantics as ``agent_init._compression_codex_settings``); only a present,
non-numeric value is worth a warning.
"""

from types import SimpleNamespace

import pytest

from tui_gateway import session_compression
from tui_gateway.session_compression import _apply_live_compression_config


@pytest.fixture(autouse=True)
def _logger_global():
    """``session_compression`` bodies are rebound onto ``server.py``'s globals in production
    (``method_ctx.bind_module``), so ``logger`` is bare there. Direct calls need the same name."""
    import logging

    session_compression.logger = logging.getLogger("tui_gateway.server")  # type: ignore[attr-defined]
    yield


def _agent():
    return SimpleNamespace(
        compression_enabled=True,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
        compression_idle_compact_after_seconds=0,
        context_compressor=None,  # early return after the threshold assignment
    )


def test_absent_threshold_keeps_default_without_warning(caplog):
    agent = _agent()

    _apply_live_compression_config(agent, {"compression": {}})

    assert agent.codex_responses_compact_threshold == 200_000
    assert "Invalid compression.codex_responses_compact_threshold" not in caplog.text


def test_explicit_none_threshold_is_automatic_without_warning(caplog):
    agent = _agent()

    _apply_live_compression_config(agent, {"compression": {"codex_responses_compact_threshold": None}})

    assert agent.codex_responses_compact_threshold == 200_000
    assert "Invalid compression.codex_responses_compact_threshold" not in caplog.text


def test_present_threshold_applies():
    agent = _agent()

    _apply_live_compression_config(agent, {"compression": {"codex_responses_compact_threshold": 500_000}})

    assert agent.codex_responses_compact_threshold == 500_000


def test_invalid_threshold_still_warns(caplog):
    agent = _agent()

    _apply_live_compression_config(agent, {"compression": {"codex_responses_compact_threshold": "lots"}})

    assert agent.codex_responses_compact_threshold == 200_000
    assert "Invalid compression.codex_responses_compact_threshold" in caplog.text


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_threshold_warns_and_falls_back(caplog, bad):
    """A threshold that cannot be a token count is a real mistake: warn and use the default."""
    agent = _agent()

    _apply_live_compression_config(agent, {"compression": {"codex_responses_compact_threshold": bad}})

    assert agent.codex_responses_compact_threshold == 200_000
    assert "Invalid compression.codex_responses_compact_threshold" in caplog.text


@pytest.mark.parametrize("bad", [True, False])
def test_bool_threshold_is_invalid_not_an_int(caplog, bad):
    """``True``/``False`` are ints in Python (1/0); they must not be read as a token count."""
    agent = _agent()

    _apply_live_compression_config(agent, {"compression": {"codex_responses_compact_threshold": bad}})

    assert agent.codex_responses_compact_threshold == 200_000
    assert "Invalid compression.codex_responses_compact_threshold" in caplog.text
