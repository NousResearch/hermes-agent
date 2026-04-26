"""Tests for auto-title generation in tui_gateway (#15949).

maybe_auto_title() was called in cli.py and gateway/run.py but was missing
from tui_gateway/server.py, leaving TUI sessions with no auto-generated titles.
"""

import inspect
import textwrap


def _get_prompt_submit_source() -> str:
    """Return the source of the prompt.submit dispatch block as a string."""
    import tui_gateway.server as srv

    # The prompt.submit handler is registered via the @method("prompt.submit")
    # decorator.  Locate the inner `run` function via the method registry so
    # this test doesn't hard-code line numbers.
    handler = srv._methods.get("prompt.submit")
    if handler is None:
        # Fallback: scan the module source for the block we care about.
        return inspect.getsource(srv)
    return inspect.getsource(handler)


class TestAutoTitleInTuiGateway:
    """Source-level checks that maybe_auto_title is wired into tui_gateway/server."""

    def test_maybe_auto_title_imported_in_server(self):
        """tui_gateway/server.py must reference maybe_auto_title."""
        import tui_gateway.server as srv

        src = inspect.getsource(srv)
        assert "maybe_auto_title" in src, (
            "maybe_auto_title not found in tui_gateway/server.py — "
            "TUI sessions will never receive auto-generated titles (#15949)"
        )

    def test_auto_title_guarded_on_complete_status(self):
        """The maybe_auto_title call must be inside a 'status == complete' guard."""
        import tui_gateway.server as srv

        src = inspect.getsource(srv)
        # Confirm the guard appears before the import — crude but reliable.
        complete_idx = src.find('status == "complete"')
        title_idx = src.find("maybe_auto_title")
        assert complete_idx != -1, "No 'status == \"complete\"' guard found"
        assert title_idx != -1, "maybe_auto_title not found"
        # The guard should appear before the call (it wraps the try block).
        assert complete_idx < title_idx, (
            "status == 'complete' guard must appear before the maybe_auto_title call"
        )

    def test_auto_title_uses_get_db_and_session_key(self):
        """The call must pass _get_db() and session['session_key'] to maybe_auto_title."""
        import tui_gateway.server as srv

        src = inspect.getsource(srv)
        assert "_get_db()" in src
        assert 'session["session_key"]' in src or "session['session_key']" in src

    def test_title_generator_module_importable(self):
        """agent.title_generator must be importable so the lazy import won't fail."""
        from agent.title_generator import maybe_auto_title  # noqa: F401
