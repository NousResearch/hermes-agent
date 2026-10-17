"""Platform-scoped approval mode: ``approvals.platform_overrides``.

``approvals.mode`` is global, so an operator running Hermes across several surfaces could
not require stricter approvals on a riskier one (a mobile chat where a fat-fingered tap is
easy) while keeping a fast flow on a trusted one (desktop CLI/dashboard).

``_get_approval_mode()`` therefore resolves the session platform first — the same
``HERMES_SESSION_PLATFORM`` identity ``_is_gateway_approval_context()`` already threads
through contextvars — and only then falls back to the global ``approvals.mode``.
Installs that never set ``platform_overrides`` (the default) are unaffected.
"""

from unittest.mock import patch

import pytest

from gateway.session_context import reset_session_vars, set_session_vars
from tools.approval_context import _get_approval_mode


@pytest.fixture(autouse=True)
def _clean_session_vars():
    reset_session_vars()
    yield
    reset_session_vars()


def _with_config(config):
    return patch("hermes_cli.config.load_config_readonly", return_value=config)


def test_platform_override_replaces_the_global_mode_for_that_platform():
    """The override wins on its own platform, ignores other platforms, and tolerates a
    hand-typed YAML key (``Telegram`` / surrounding spaces match ``telegram``)."""
    config = {
        "approvals": {
            "mode": "smart",
            "platform_overrides": {" Telegram ": "manual", "whatsapp": "OFF"},
        }
    }
    with _with_config(config):
        set_session_vars(platform="telegram")
        assert _get_approval_mode() == "manual"

        reset_session_vars()
        set_session_vars(platform="whatsapp")
        assert _get_approval_mode() == "off"

        # No override for this platform: the global mode still applies.
        reset_session_vars()
        set_session_vars(platform="discord")
        assert _get_approval_mode() == "smart"


def test_global_mode_applies_without_a_matching_override():
    """Backward compatibility: an absent/empty/malformed block, a platform-less session
    (CLI/TUI bind no platform), and an inherited null value keep the pre-override
    behavior — the global mode, or 'manual' when the global mode is unusable."""
    with _with_config({"approvals": {"mode": "off"}}):
        set_session_vars(platform="telegram")
        assert _get_approval_mode() == "off"

        reset_session_vars()
        assert _get_approval_mode() == "off"

    # A scalar where a mapping belongs is not a match — it must not strand the session
    # on an unreadable policy.
    with _with_config({"approvals": {"mode": "off", "platform_overrides": "manual"}}):
        set_session_vars(platform="telegram")
        assert _get_approval_mode() == "off"

    # `platform_overrides: {telegram: }` parses as null; like a null global `mode`, the
    # unset value fails closed to 'manual' instead of silently relaxing the surface.
    with _with_config({"approvals": {"mode": "off", "platform_overrides": {"telegram": None}}}):
        set_session_vars(platform="telegram")
        assert _get_approval_mode() == "manual"
