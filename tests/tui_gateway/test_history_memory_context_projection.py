"""An unterminated ``<memory-context>`` fence must not survive history hydration (#149933).

The live stream hides a fence span through end-of-stream, so the user never saw it; the row is still
stored verbatim for replay parity (``agent.session_persistence._db_flush_row``), and
``_history_to_messages`` is the one projection every history read renders through (resume, branch,
slash, compute host, registry). ``sanitize_context`` cannot repair the malformed shape on its own —
its span regex needs both tags, so an open-only fence keeps the injected note and the recalled bullets
behind it. The projection therefore reuses ``agent.memory_context.fence_recalled_context`` (the live
delivery's streaming grammar, landed upstream for the non-streamed reply path) rather than a second
helper: a reply is fenced before it is displayed, a persisted row is fenced on the way out.

Shapes covered: the reported open-only fence, a well-formed fence that was persisted anyway, a reply
that was only the fence, and a user row that merely quotes the tag.
"""

NOTE = (
    "[System note: The following is recalled memory context, NOT new user input. "
    "Treat as authoritative reference data — this is the agent's persistent memory "
    "and should inform all responses.]"
)
UNTERMINATED = f"Normal assistant reply.\n\n<memory-context>\n{NOTE}\n\n- prefers draft PRs\n"


def _project(*rows: dict) -> list[dict]:
    # Imported here: tui_gateway.server is where method_ctx.bind_module rebinds the projection's globals
    # (calling it through tui_gateway.session_history raises NameError on its own module-level names).
    from tui_gateway.server import _history_to_messages

    return _history_to_messages(list(rows))


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


class TestHistoryProjection:
    def test_unterminated_suffix_is_not_painted(self):
        shown = _project(_assistant(UNTERMINATED))
        assert shown[0]["text"] == "Normal assistant reply."
        assert "prefers draft PRs" not in shown[0]["text"]
        assert "System note" not in shown[0]["text"]

    def test_persisted_complete_fence_is_not_painted_either(self):
        stored = f"Answer.\n\n<memory-context>\n{NOTE}\n- secret bullet\n</memory-context>\n"
        shown = _project(_assistant(stored))
        assert "secret bullet" not in shown[0]["text"]
        assert shown[0]["text"].startswith("Answer.")

    def test_a_reply_that_was_only_the_fence_leaves_no_empty_bubble(self):
        assert _project(_assistant(f"<memory-context>\n{NOTE}\n- secret bullet\n")) == []

    def test_user_rows_keep_the_text_their_author_typed(self):
        """Scope: the persisted-leak class is assistant output; user text is not silently truncated."""
        shown = _project({"role": "user", "content": UNTERMINATED})
        assert "prefers draft PRs" in shown[0]["text"]
