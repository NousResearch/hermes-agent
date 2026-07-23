"""Browser tool gating in non-interactive sessions (#66393).

`check_browser_requirements()` is the `check_fn` for the browser tools, so a
False return removes them from the model's toolset. In a session with no human
to answer a first-use install prompt (gateway, cron, single-shot `chat -q` /
`-z`, daemon, pipe), the bare `npx` fallback is advertised-but-unusable and is
gated out.

The decision is *session*-level, not *process*-level: an explicit per-context
override or a bound gateway/cron signal gates the tool even when
``sys.stdin.isatty()`` is True (the leak flagged on PR #66422), while a plain
interactive TTY keeps the intentional install-on-demand flow.

The gate is per-tool via the check_fn, so it is precise: it removes only tools
whose check_fn is ``check_browser_requirements`` (the browser tools) and never
``web_search``, which shares the ``browser`` toolset bundle but has its own
check_fn. A delegate_task subagent inherits the gate for free in the cases that
matter — a gateway/cron/autonomous parent is itself non-interactive, so the
child assembled in its context sees the same gated verdict.
"""

import pytest

from tools import browser_tool
from tools.registry import registry, invalidate_check_fn_cache

# Browser tools are registered (toolset "browser") on import of tools.browser_tool.
_BROWSER_TOOL_NAMES = {
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
}


def _browser_visible_in_toolset() -> bool:
    """True when any browser tool survives the registry's check_fn filter, i.e.
    is exposed to the model. This is the real tool-definition path."""
    defs = registry.get_definitions(_BROWSER_TOOL_NAMES, quiet=True)
    return bool({d["function"]["name"] for d in defs})


@pytest.fixture
def local_mode(monkeypatch):
    """Neutralize every branch of check_browser_requirements() except the
    non-interactive gate, so tests exercise exactly that decision."""
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
    monkeypatch.setattr(
        browser_tool, "_requires_real_termux_browser_install", lambda cmd: False
    )
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    # If execution reaches past the npx gate, treat the engine as satisfiable
    # so a non-gated path returns True cleanly.
    monkeypatch.setattr(browser_tool, "_using_lightpanda_engine", lambda: True)
    # Default: no bound gateway/cron platform.
    monkeypatch.setattr(
        "gateway.session_context.get_session_env", lambda key, default="": default
    )
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    # Ensure no explicit override leaks in from another test.
    tok = browser_tool.set_browser_session_interactive(True)
    browser_tool.reset_browser_session_interactive(tok)
    # The registry check_fn cache is process-global; clear it around each test
    # so a warmed browser verdict neither pollutes nor is polluted by others.
    invalidate_check_fn_cache()
    yield
    invalidate_check_fn_cache()


def _raise_not_found(validate=False):
    raise FileNotFoundError("agent-browser")


def test_gated_when_agent_browser_absent_and_non_interactive(local_mode, monkeypatch):
    monkeypatch.setattr(browser_tool, "_find_agent_browser", _raise_not_found)
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: False)
    assert browser_tool.check_browser_requirements() is False


def test_npx_fallback_gated_in_non_interactive(local_mode, monkeypatch):
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: False)
    assert browser_tool.check_browser_requirements() is False


def test_npx_fallback_preserved_in_interactive_tty(local_mode, monkeypatch):
    """A plain interactive terminal keeps install-on-demand: the npx fallback
    is NOT gated, so the tool stays available."""
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)
    assert browser_tool.check_browser_requirements() is True


def test_explicit_non_interactive_overrides_tty(local_mode, monkeypatch):
    """The single-shot CLI case: stdin is an attached TTY, but the explicit
    context signal gates the npx fallback anyway."""
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)
    tok = browser_tool.set_browser_session_interactive(False)
    try:
        assert browser_tool.check_browser_requirements() is False
    finally:
        browser_tool.reset_browser_session_interactive(tok)
    # Scoped: after reset the TTY is interactive again.
    assert browser_tool.check_browser_requirements() is True


def test_gateway_platform_gates_despite_tty(local_mode, monkeypatch):
    """Real session context: a bound gateway platform gates the tool even when
    the process stdin is a TTY (leak-proof for in-process children)."""
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda key, default="": "telegram"
        if key == "HERMES_SESSION_PLATFORM"
        else default,
    )
    assert browser_tool.check_browser_requirements() is False


def test_cron_gated_by_cron_session_env(local_mode, monkeypatch):
    """Cron binds an empty platform, so it is matched by HERMES_CRON_SESSION
    rather than the platform var, even with a TTY."""
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert browser_tool._is_non_interactive_session() is True


def test_real_install_not_gated_even_when_non_interactive(local_mode, monkeypatch):
    """A real (non-npx) install is fine in any session — only the fragile npx
    fallback is gated."""
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda validate=False: "/usr/local/bin/agent-browser",
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: False)
    assert browser_tool.check_browser_requirements() is True


def test_gate_is_per_tool_and_spares_web_search():
    """Precision guard. Disabling the whole 'browser' toolset would strip
    web_search (it is bundled in TOOLSETS['browser']). Gating per check_fn does
    not: only tools whose check_fn is check_browser_requirements are removed, and
    web_search has its own check_fn."""
    import tools.web_tools  # noqa: F401  (registers web_search)

    nav = registry.get_entry("browser_navigate")
    ws = registry.get_entry("web_search")
    assert nav is not None and ws is not None
    assert nav.check_fn is browser_tool.check_browser_requirements
    assert ws.check_fn is not browser_tool.check_browser_requirements


def test_override_sets_and_resets():
    assert browser_tool._browser_session_interactive.get() is None
    tok = browser_tool.set_browser_session_interactive(False)
    try:
        assert browser_tool._is_non_interactive_session() is True
    finally:
        browser_tool.reset_browser_session_interactive(tok)
    assert browser_tool._browser_session_interactive.get() is None


def test_gateway_gated_through_registry(local_mode, monkeypatch):
    """A bound gateway platform gates the browser tool out of the assembled
    toolset even with a TTY, through the registry/tool-definition path."""
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda key, default="": "telegram"
        if key == "HERMES_SESSION_PLATFORM"
        else default,
    )
    invalidate_check_fn_cache()
    assert _browser_visible_in_toolset() is False


def test_subagent_worker_mark_gates_despite_tty(local_mode, monkeypatch):
    """A delegate child's run_conversation runs on a worker thread that starts
    with an empty contextvars context, so the parent's gateway platform signal
    does not propagate there; with an inherited TTY the browser check would be
    ungated on that thread, and a between-turns re-assembly could re-advertise
    browser and poison the parent's cache. The mark (paired with a reset around
    the child run) gates it on the worker (#66393)."""
    from concurrent.futures import ThreadPoolExecutor

    from tools.delegate_tool import (
        _mark_subagent_browser_non_interactive,
        _unmark_subagent_browser_non_interactive,
    )

    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)

    def worker(mark):
        tok = _mark_subagent_browser_non_interactive() if mark else None
        try:
            return browser_tool.check_browser_requirements()
        finally:
            _unmark_subagent_browser_non_interactive(tok)

    with ThreadPoolExecutor(max_workers=1) as ex:
        # Worker with a TTY and no propagated signal would advertise browser...
        assert ex.submit(worker, False).result() is True
        # ...the mark gates it on the worker thread.
        assert ex.submit(worker, True).result() is False


def test_subagent_mark_is_paired_and_restores():
    """The mark must reset after the child run. A single-task delegation runs
    the child *inline on the parent's thread* (pool-at-capacity, stateless
    channel, or a lone CLI `delegate_task`), so an unpaired set would
    permanently gate the interactive parent session for the rest of its life
    (#66393)."""
    from tools.delegate_tool import (
        _mark_subagent_browser_non_interactive,
        _unmark_subagent_browser_non_interactive,
    )

    assert browser_tool._browser_session_interactive.get() is None
    tok = _mark_subagent_browser_non_interactive()
    assert browser_tool._browser_session_interactive.get() is False
    _unmark_subagent_browser_non_interactive(tok)
    assert browser_tool._browser_session_interactive.get() is None


def test_cloud_provider_not_gated_by_npx_when_non_interactive(local_mode, monkeypatch):
    """The npx gate is local-mode only. A configured cloud provider hosts the
    browser and the CLI only drives the remote session, so an npx-only CLI stays
    available even non-interactively (cloud is checked before the npx gate)."""

    class _Provider:
        def is_configured(self):
            return True

    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: _Provider())
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: False)
    assert browser_tool.check_browser_requirements() is True


def test_isatty_exception_is_treated_non_interactive(local_mode, monkeypatch):
    def _boom():
        raise ValueError("closed fd")

    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", _boom)
    assert browser_tool.check_browser_requirements() is False
