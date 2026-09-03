"""Desktop context-file policy mirrors the messaging gateway.

Desktop backends launch from the Hermes install tree, so a profile must be
able to skip that tree's project instructions without losing its SOUL identity.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch


def _runtime():
    return {
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
        "credential_pool": None,
    }


def _make_agent_kwargs(cfg, monkeypatch, *, platform="desktop"):
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    with ExitStack() as stack:
        stack.enter_context(patch("tui_gateway.server._load_cfg", return_value=cfg))
        stack.enter_context(patch("tui_gateway.server._get_db", return_value=MagicMock()))
        stack.enter_context(
            patch("tui_gateway.server._load_tool_progress_mode", return_value="compact")
        )
        stack.enter_context(
            patch("tui_gateway.server._load_reasoning_config", return_value=None)
        )
        stack.enter_context(patch("tui_gateway.server._load_service_tier", return_value=None))
        stack.enter_context(
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None)
        )
        stack.enter_context(
            patch(
                "tui_gateway.server._resolve_startup_runtime",
                return_value=("test-model", None),
            )
        )
        stack.enter_context(
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=_runtime(),
            )
        )
        mock_agent = stack.enter_context(patch("run_agent.AIAgent"))
        from tui_gateway.server import _make_agent

        _make_agent("sid-1", "key-1", platform_override=platform)
        return mock_agent.call_args.kwargs


def test_desktop_config_skips_project_context_but_keeps_soul_and_memory(monkeypatch):
    kwargs = _make_agent_kwargs(
        {"gateway": {"platforms": {"desktop": {"skip_context_files": True}}}},
        monkeypatch,
    )

    assert kwargs["skip_context_files"] is True
    assert kwargs["load_soul_identity"] is True
    assert kwargs["skip_memory"] is False


def test_desktop_skip_context_setting_does_not_apply_to_tui(monkeypatch):
    kwargs = _make_agent_kwargs(
        {"gateway": {"platforms": {"desktop": {"skip_context_files": True}}}},
        monkeypatch,
        platform="tui",
    )

    assert kwargs["skip_context_files"] is False
    assert kwargs["load_soul_identity"] is False
    assert kwargs["skip_memory"] is False


def test_platforms_list_shape_is_tolerated(monkeypatch):
    # The messaging gateway uses gateway.platforms as a plain list of enabled
    # platform names. Desktop must not crash on that shape and must not read a
    # skip out of it.
    kwargs = _make_agent_kwargs(
        {"gateway": {"platforms": ["desktop", "telegram"]}},
        monkeypatch,
    )

    assert kwargs["skip_context_files"] is False
    assert kwargs["load_soul_identity"] is False
    assert kwargs["skip_memory"] is False


def test_ignore_rules_still_skips_soul_and_memory(monkeypatch):
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    with patch("tui_gateway.server._load_cfg", return_value={}):
        from tui_gateway.server import _context_file_policy

        assert _context_file_policy({}, "desktop") == (True, False, True)
