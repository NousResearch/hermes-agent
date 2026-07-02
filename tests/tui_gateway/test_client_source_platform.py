"""Client-source → platform attribution for the shared tui_gateway server.

The desktop app, dashboard SPA, Ink stdio TUI, and (future) mobile client all
drive the same JSON-RPC ``tui_gateway.server``. Before this feature every one of
them recorded turns as ``platform="tui"``; a client now self-declares a
``source`` on session.create/resume and it must:

  * flow through ``_make_agent`` to ``AIAgent(platform=...)`` (→ blackbox
    ``turns.platform`` → tokens.ace source chart), and
  * be sanitized, because it is untrusted client input feeding a persisted DB
    dimension.

These are behavior contracts (how source relates to the agent's platform), not
snapshots of a frozen value. Mirrors test_make_agent_provider.py's import style
(no sys.modules patching — that stubs get_hermes_home to a str and breaks the
tool-registry imports _make_agent pulls in).
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _sanitize_client_source — the untrusted-input guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("desktop", "desktop"),
        ("dashboard", "dashboard"),
        ("mobile", "mobile"),
        ("tui", "tui"),
        ("DESKTOP", "desktop"),  # case-normalized
        ("  desktop  ", "desktop"),  # trimmed
        ("web-app_2", "web-app_2"),  # allowed slug chars
        # Rejected → fall back to the stdio default, never raw-through:
        ("", "tui"),
        (None, "tui"),
        ("tui; DROP TABLE turns", "tui"),  # spaces/punctuation rejected
        ("<script>", "tui"),
        ("has spaces", "tui"),
        ("1leading-digit", "tui"),  # must start with a letter
        ("x" * 40, "tui"),  # over length cap
        (123, "tui"),  # non-string
        (True, "tui"),  # a JSON-RPC `true` must NOT coerce to the slug "true"
        (0, "tui"),  # non-string falsy
        ([], "tui"),  # non-string container
    ],
)
def test_sanitize_client_source(raw, expected):
    from tui_gateway.server import _sanitize_client_source

    assert _sanitize_client_source(raw) == expected


# ---------------------------------------------------------------------------
# _make_agent — source drives the agent's platform
# ---------------------------------------------------------------------------


def _make_agent_platform_for_source(source):
    """Invoke _make_agent with the heavy deps stubbed; return AIAgent's platform."""
    fake_runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test",
        "api_mode": "anthropic_messages",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    fake_cfg = {
        "model": {"default": "claude-opus-4-8", "provider": "anthropic"},
        "agent": {"system_prompt": "test"},
    }
    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-1", "key-1", source=source)
        return mock_agent.call_args.kwargs["platform"]


def test_make_agent_source_becomes_platform():
    """A declared client source is the agent's platform verbatim."""
    assert _make_agent_platform_for_source("desktop") == "desktop"


def test_make_agent_default_source_is_tui():
    """No source (stdio Ink / internal rebuild) keeps the historical default."""
    assert _make_agent_platform_for_source(None) == "tui"


def test_make_agent_sanitizes_hostile_source():
    """_make_agent doesn't trust its caller: a malformed source → tui."""
    assert _make_agent_platform_for_source("not a slug!") == "tui"
