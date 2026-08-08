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
    # main resolves CDP via the no-I/O _get_cdp_override_raw(); neutralize it too
    # so the fixture actually isolates the non-interactive gate (#69071 rebase).
    monkeypatch.setattr(browser_tool, "_get_cdp_override_raw", lambda: "")
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


def test_delegated_child_context_gates_browser_on_worker_despite_tty(local_mode, monkeypatch):
    """A delegate child's construction and run happen inside
    ``delegated_child_context()`` (agent/delegation_context.py). Even on a worker
    thread with an inherited TTY and no propagated gateway signal, being in that
    context gates the browser — the gate honors ``is_delegated_child_context()``
    (#69071 supersedes this PR's own subagent mark; the mechanism is now main's)."""
    from concurrent.futures import ThreadPoolExecutor

    from agent.delegation_context import delegated_child_context

    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)

    def worker(as_child):
        if as_child:
            with delegated_child_context():
                return browser_tool.check_browser_requirements()
        return browser_tool.check_browser_requirements()

    with ThreadPoolExecutor(max_workers=1) as ex:
        # Plain worker + TTY, no propagated signal → browser advertised...
        assert ex.submit(worker, False).result() is True
        # ...inside the delegated-child context → gated.
        assert ex.submit(worker, True).result() is False


def test_delegated_child_context_gates_and_resets(local_mode, monkeypatch):
    """``delegated_child_context()`` is the mechanism that marks a child; the gate
    must fire inside it and reset on exit so it never permanently gates the
    interactive parent (the pairing is main's context manager, not our own)."""
    from agent.delegation_context import (
        delegated_child_context,
        is_delegated_child_context,
    )

    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)

    assert is_delegated_child_context() is False
    assert browser_tool.check_browser_requirements() is True  # interactive parent
    with delegated_child_context():
        assert is_delegated_child_context() is True
        assert browser_tool.check_browser_requirements() is False  # gated in child context
    assert is_delegated_child_context() is False
    assert browser_tool.check_browser_requirements() is True  # restored, parent unaffected


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


# --- #69071 second pass: the gate must survive the two caches + child construction ---

def test_check_fn_cache_no_cross_session_leak(local_mode, monkeypatch):
    """#69071 Part 1: the browser check_fn is context-sensitive, so its verdict
    must not persist in the process-global check_fn cache across sessions on the
    same worker. Order-sensitive and deliberately does NOT invalidate between
    reads — the pre-fix bug was that a warmed interactive verdict was served to a
    later gateway/cron read (and vice versa)."""
    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)
    invalidate_check_fn_cache()  # start clean; no invalidate between the reads below

    tok = browser_tool.set_browser_session_interactive(True)
    try:
        assert _browser_visible_in_toolset() is True   # interactive -> visible (warms cache)
    finally:
        browser_tool.reset_browser_session_interactive(tok)

    tok = browser_tool.set_browser_session_interactive(False)
    try:
        assert _browser_visible_in_toolset() is False  # non-interactive -> gated, no leak
    finally:
        browser_tool.reset_browser_session_interactive(tok)

    tok = browser_tool.set_browser_session_interactive(True)
    try:
        assert _browser_visible_in_toolset() is True    # reverse order: no stale gate
    finally:
        browser_tool.reset_browser_session_interactive(tok)


def test_tool_defs_cache_keys_on_session_context(local_mode, monkeypatch):
    """#69071 Part 2: model_tools memoizes the assembled tool-definition list; the
    cache key now includes session interactivity, so a long-lived process does not
    serve one session type's cached list to the other."""
    import model_tools

    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()  # process-global, no TTL — start clean

    def browser_in_defs():
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["browser"], quiet_mode=True
        )
        return any(d["function"]["name"] == "browser_navigate" for d in defs)

    tok = browser_tool.set_browser_session_interactive(True)
    try:
        interactive = browser_in_defs()
    finally:
        browser_tool.reset_browser_session_interactive(tok)

    tok = browser_tool.set_browser_session_interactive(False)
    try:
        gated = browser_in_defs()
    finally:
        browser_tool.reset_browser_session_interactive(tok)

    assert interactive is True
    assert gated is False


def test_delegate_child_construction_snapshot_excludes_browser(local_mode, monkeypatch):
    """#69071 Part 3: a delegate child's tool list is snapshotted during
    construction, and main runs that construction inside ``delegated_child_context()``
    (verified: ``_build_child_agent`` wraps ``AIAgent()`` in it). Because the gate
    now honors ``is_delegated_child_context()``, that snapshot excludes browser even
    under an interactive parent TTY — so the child is never handed the offered-but-
    unusable npx browser. Assert the tool snapshot taken in that context has none."""
    import model_tools

    from agent.delegation_context import delegated_child_context

    monkeypatch.setattr(
        browser_tool, "_find_agent_browser", lambda validate=False: "npx agent-browser"
    )
    monkeypatch.setattr(browser_tool.sys.stdin, "isatty", lambda: True)  # interactive parent
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()

    def browser_in_defs():
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["browser"], quiet_mode=True
        )
        return any(d["function"]["name"] == "browser_navigate" for d in defs)

    assert browser_in_defs() is True  # interactive parent snapshot includes browser
    with delegated_child_context():
        # exactly the context child construction runs in — snapshot must exclude browser
        assert browser_in_defs() is False
    assert browser_in_defs() is True  # parent snapshot unaffected afterward
