"""Tests for the gateway-side clarify primitive (tools/clarify_gateway.py).

The clarify tool needs to ask the user a question and block the agent
thread until they respond.  These tests cover the module-level state
machine: register, wait, resolve via button, resolve via text-fallback,
"Other"-button text-capture flip, timeout, session boundary cleanup.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor



def _clear_clarify_state():
    """Reset module-level state between tests."""
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


class TestClarifyPrimitive:
    """Core register/wait/resolve mechanics."""

    def setup_method(self):
        _clear_clarify_state()

    def test_button_choice_resolves_wait(self):
        """resolve_gateway_clarify unblocks wait_for_response with the chosen string."""
        from tools import clarify_gateway as cm

        cm.register("id1", "sk1", "Pick one", ["A", "B", "C"])

        def resolver():
            time.sleep(0.05)
            cm.resolve_gateway_clarify("id1", "B")

        threading.Thread(target=resolver).start()
        result = cm.wait_for_response("id1", timeout=10.0)
        assert result == "B"

    def test_open_ended_auto_awaits_text(self):
        """Clarify with no choices is in text-capture mode immediately."""
        from tools import clarify_gateway as cm

        entry = cm.register("id2", "sk2", "Free form?", None)
        assert entry.awaiting_text is True

        # get_pending_for_session returns the entry so the gateway
        # text-intercept can find it.
        pending = cm.get_pending_for_session("sk2")
        assert pending is not None
        assert pending.clarify_id == "id2"

    def test_button_choice_does_not_auto_await(self):
        """Multi-choice clarify should NOT be in text-capture mode initially."""
        from tools import clarify_gateway as cm

        entry = cm.register("id3", "sk3", "Pick", ["X", "Y"])
        assert entry.awaiting_text is False
        assert cm.get_pending_for_session("sk3") is None

    def test_include_choice_prompts_returns_multi_choice_entry(self):
        """Gateway typed replies must see active choice prompts too."""
        from tools import clarify_gateway as cm

        cm.register("id3b", "sk3b", "Pick", ["X", "Y"])
        pending = cm.get_pending_for_session("sk3b", include_choice_prompts=True)
        assert pending is not None
        assert pending.clarify_id == "id3b"

    def test_resolve_text_response_maps_numeric_choice(self):
        """Typed numbers should resolve to the canonical choice string."""
        from tools import clarify_gateway as cm

        cm.register("id3c", "sk3c", "Pick", ["X", "Y"])
        assert cm.resolve_text_response_for_session("sk3c", "2") is True
        assert cm.wait_for_response("id3c", timeout=0.1) == "Y"

    def test_resolve_text_response_accepts_custom_other_text(self):
        """Arbitrary typed text should resolve as a custom Other answer when awaiting_text is True."""
        from tools import clarify_gateway as cm

        cm.register("id3d", "sk3d", "Pick", ["X", "Y"])
        # Flip to text-capture mode (user picked "Other")
        cm.mark_awaiting_text("id3d")
        custom = "None of those are valid options"
        assert cm.resolve_text_response_for_session("sk3d", custom) is True
        assert cm.wait_for_response("id3d", timeout=0.1) == custom

    def test_resolve_text_rejects_arbitrary_prose_for_native_multi_choice(self):
        """Native interactive multi-choice clarifies reject arbitrary prose unless awaiting_text is True."""
        from tools import clarify_gateway as cm

        # Native multi-choice (buttons, not awaiting text)
        cm.register("id-strict", "sk-strict", "Pick one", ["A", "B", "C"])

        # Arbitrary prose should be rejected
        assert cm.resolve_text_response_for_session("sk-strict", "just checking the visual UI") is False
        assert cm.resolve_text_response_for_session("sk-strict", "present 3 buttons") is False

        # Numeric choices should still work
        assert cm.resolve_text_response_for_session("sk-strict", "2") is True
        assert cm.wait_for_response("id-strict", timeout=0.1) == "B"

        # Exact label match should still work
        cm.register("id-strict2", "sk-strict2", "Pick", ["Option Alpha", "Option Beta"])
        assert cm.resolve_text_response_for_session("sk-strict2", "Option Alpha") is True
        assert cm.wait_for_response("id-strict2", timeout=0.1) == "Option Alpha"

    def test_text_fallback_mode_allows_any_text(self):
        """Text fallback mode (after base send_clarify calls mark_awaiting_text) accepts any text."""
        from tools import clarify_gateway as cm

        entry = cm.register("id-tf", "sk-tf", "Pick one", ["A", "B", "C"])
        assert entry.awaiting_text is False

        # Simulate base send_clarify calling mark_awaiting_text
        cm.mark_awaiting_text("id-tf")
        assert entry.awaiting_text is True

        # Now arbitrary text is accepted
        custom = "I choose a custom answer"
        assert cm.resolve_text_response_for_session("sk-tf", custom) is True
        assert cm.wait_for_response("id-tf", timeout=0.1) == custom

        # Numeric choices also work
        cm.register("id-tf2", "sk-tf2", "Pick", ["X", "Y"])
        cm.mark_awaiting_text("id-tf2")
        assert cm.resolve_text_response_for_session("sk-tf2", "1") is True
        assert cm.wait_for_response("id-tf2", timeout=0.1) == "X"

    def test_other_button_flips_to_text_mode(self):
        """mark_awaiting_text makes get_pending_for_session find the entry."""
        from tools import clarify_gateway as cm

        cm.register("id4", "sk4", "Pick", ["X", "Y"])
        assert cm.get_pending_for_session("sk4") is None

        flipped = cm.mark_awaiting_text("id4")
        assert flipped is True

        pending = cm.get_pending_for_session("sk4")
        assert pending is not None
        assert pending.clarify_id == "id4"

    def test_mark_awaiting_text_unknown_id(self):
        """mark_awaiting_text on a non-existent id returns False."""
        from tools import clarify_gateway as cm

        assert cm.mark_awaiting_text("nope") is False

    def test_timeout_returns_none(self):
        """wait_for_response returns None when no resolve fires within the timeout."""
        from tools import clarify_gateway as cm

        cm.register("id5", "sk5", "Q?", ["A"])
        result = cm.wait_for_response("id5", timeout=0.2)
        assert result is None

    def test_resolve_unknown_id_returns_false(self):
        """resolve_gateway_clarify is idempotent on unknown ids."""
        from tools import clarify_gateway as cm

        assert cm.resolve_gateway_clarify("nope", "anything") is False

    def test_resolve_after_wait_completes_is_noop(self):
        """A late resolve on a finished entry doesn't blow up."""
        from tools import clarify_gateway as cm

        cm.register("id6", "sk6", "Q?", ["A"])
        # Time out, entry gets cleaned up
        cm.wait_for_response("id6", timeout=0.1)
        # Late button click — should not raise
        result = cm.resolve_gateway_clarify("id6", "A")
        assert result is False

    def test_clear_session_cancels_pending_entries(self):
        """clear_session unblocks blocked threads with empty response."""
        from tools import clarify_gateway as cm

        cm.register("id7", "sk7", "Q?", ["A"])

        def waiter():
            return cm.wait_for_response("id7", timeout=10.0)

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(waiter)
            time.sleep(0.05)
            cancelled = cm.clear_session("sk7")
            assert cancelled == 1
            result = fut.result(timeout=10.0)
            # clear_session sets response="" then the wait returns it
            assert result == ""

    def test_has_pending(self):
        from tools import clarify_gateway as cm

        cm.register("id8", "sk8", "Q?", ["A"])
        assert cm.has_pending("sk8") is True
        assert cm.has_pending("nonexistent") is False

    def test_notify_register_unregister_clears_pending(self):
        """unregister_notify cancels any pending clarify so threads unwind."""
        from tools import clarify_gateway as cm

        cm.register("id9", "sk9", "Q?", ["A"])

        def waiter():
            return cm.wait_for_response("id9", timeout=10.0)

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(waiter)
            time.sleep(0.05)

            cm.register_notify("sk9", lambda entry: None)
            cm.unregister_notify("sk9")

            # unregister_notify calls clear_session; thread unwinds
            result = fut.result(timeout=10.0)
            assert result == ""

    def test_session_index_isolation(self):
        """Entries from different sessions don't leak across get_pending lookups."""
        from tools import clarify_gateway as cm

        cm.register("idA", "alpha", "Q?", None)  # auto-await text
        cm.register("idB", "beta", "Q?", None)   # auto-await text

        a = cm.get_pending_for_session("alpha")
        b = cm.get_pending_for_session("beta")
        assert a is not None and a.clarify_id == "idA"
        assert b is not None and b.clarify_id == "idB"

    def test_clarify_timeout_config_default(self):
        """get_clarify_timeout returns a positive int (default 3600)."""
        from tools import clarify_gateway as cm

        timeout = cm.get_clarify_timeout()
        # Default 3600s OR whatever is in the user's loaded config.
        # Floor check: must be a positive int, not crashed.
        assert isinstance(timeout, int)
        assert timeout > 0


class TestGatewayTextIntercept:
    """The gateway's _handle_message intercepts text replies to pending clarifies."""

    def setup_method(self):
        _clear_clarify_state()

    def test_get_pending_for_session_returns_oldest_text_awaiting(self):
        """When two clarifies are pending, get_pending_for_session returns the
        first that is awaiting_text (the older one if both)."""
        from tools import clarify_gateway as cm

        # Older multi-choice (not awaiting text)
        cm.register("first", "sk", "Q1?", ["A"])
        # Newer open-ended (awaiting text)
        cm.register("second", "sk", "Q2?", None)

        pending = cm.get_pending_for_session("sk")
        # The newer one is awaiting text; the older isn't.
        assert pending is not None
        assert pending.clarify_id == "second"

        # Now flip the first to text mode too.  Both are awaiting text,
        # FIFO returns the older one.
        cm.mark_awaiting_text("first")
        pending2 = cm.get_pending_for_session("sk")
        assert pending2 is not None
        assert pending2.clarify_id == "first"
    def test_text_fallback_enables_awaiting_text_for_multi_choice(self):
        """When base send_clarify renders choices as text, mark_awaiting_text
        is called so the gateway text-intercept can capture the reply."""
        from tools import clarify_gateway as cm

        entry = cm.register("id-tf", "sk-tf", "Pick one", ["A", "B", "C"])
        # Initially, multi-choice does NOT await text (button path)
        assert entry.awaiting_text is False

        # After the base send_clarify text fallback calls mark_awaiting_text:
        flipped = cm.mark_awaiting_text("id-tf")
        assert flipped is True

        # Now get_pending_for_session should find it
        pending = cm.get_pending_for_session("sk-tf")
        assert pending is not None
        assert pending.clarify_id == "id-tf"
        
        # Clean up
        cm.clear_session("sk-tf")


class TestCoverageGaps:
    """Cover remaining branches: signature(), get_entry miss, find_awaiting
    with deleted entry, cancel with None entry, timeout exception, get_notify."""

    def setup_method(self):
        _clear_clarify_state()

    def test_entry_signature(self):
        """_ClarifyEntry.signature() returns the expected dict."""
        from tools import clarify_gateway as cm

        entry = cm.register("sig1", "sk", "Q?", ["A", "B"])
        sig = entry.signature()
        assert sig["clarify_id"] == "sig1"
        assert sig["session_key"] == "sk"
        assert sig["question"] == "Q?"
        assert sig["choices"] == ["A", "B"]

    def test_entry_signature_no_choices(self):
        """signature() returns None for choices when open-ended."""
        from tools import clarify_gateway as cm

        entry = cm.register("sig2", "sk", "Q?", None)
        sig = entry.signature()
        assert sig["choices"] is None

    def test_wait_for_response_unknown_id_returns_none(self):
        """wait_for_response on a non-existent id returns None immediately."""
        from tools import clarify_gateway as cm

        assert cm.wait_for_response("nonexistent-id", timeout=0.1) is None

    def test_find_awaiting_skips_deleted_entry(self):
        """get_pending_for_session skips entries that were removed from _entries
        but still listed in _session_index."""
        from tools import clarify_gateway as cm

        cm.register("a1", "sk", "Q?", None)
        # Manually remove from _entries but leave in _session_index
        with cm._lock:
            cm._entries.pop("a1", None)
        # No entry to find → returns None
        assert cm.get_pending_for_session("sk") is None

    def test_clear_session_skips_deleted_entry(self):
        """clear_session skips entries that are None (already removed)."""
        from tools import clarify_gateway as cm

        cm.register("c1", "sk", "Q?", ["A"])
        # Manually remove from _entries but leave in _session_index
        with cm._lock:
            cm._entries.pop("c1", None)
        # Should return 0 cancelled (entry was already gone)
        cancelled = cm.clear_session("sk")
        assert cancelled == 0

    def test_get_clarify_timeout_exception_returns_default(self, monkeypatch):
        """get_clarify_timeout returns 3600 when load_config raises."""
        from tools import clarify_gateway as cm

        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cm.get_clarify_timeout() == 3600

    def test_get_notify_returns_callback(self):
        """get_notify returns the registered callback."""
        from tools import clarify_gateway as cm

        cb = lambda entry: None
        cm.register_notify("sk-notify", cb)
        assert cm.get_notify("sk-notify") is cb

    def test_get_notify_returns_none_when_not_registered(self):
        """get_notify returns None for an unregistered session."""
        from tools import clarify_gateway as cm

        assert cm.get_notify("unregistered") is None


class TestClarifyTimeoutResolution:
    """resolve_clarify_timeout is the single source of truth for the clarify
    timeout, shared by the CLI, TUI/desktop, and messaging-gateway paths."""

    def test_canonical_agent_key(self):
        from tools import clarify_gateway as cm

        assert cm.resolve_clarify_timeout({"agent": {"clarify_timeout": 900}}) == 900

    def test_legacy_clarify_key_overrides(self):
        """An explicitly-set legacy top-level clarify.timeout wins, for
        back-compat with users who set it before agent.clarify_timeout existed."""
        from tools import clarify_gateway as cm

        cfg = {"clarify": {"timeout": 42}, "agent": {"clarify_timeout": 900}}
        assert cm.resolve_clarify_timeout(cfg) == 42

    def test_default_when_unset(self):
        from tools import clarify_gateway as cm

        assert cm.resolve_clarify_timeout({}) == 3600

    def test_non_numeric_falls_back_to_default(self):
        from tools import clarify_gateway as cm

        assert cm.resolve_clarify_timeout({"agent": {"clarify_timeout": "nope"}}) == 3600

    def test_non_positive_preserved_as_unlimited_sentinel(self):
        """<= 0 is passed through verbatim — the waiting loops read it as
        'unlimited', so the resolver must not clamp it to a positive default."""
        from tools import clarify_gateway as cm

        assert cm.resolve_clarify_timeout({"agent": {"clarify_timeout": 0}}) == 0
        assert cm.resolve_clarify_timeout({"clarify": {"timeout": -1}}) == -1


class TestUnlimitedWait:
    """timeout <= 0 makes wait_for_response block until the answer arrives
    instead of auto-skipping."""

    def setup_method(self):
        _clear_clarify_state()

    def test_zero_timeout_waits_until_resolved(self):
        from tools import clarify_gateway as cm

        cm.register("u1", "sk", "Q?", ["A", "B"])
        result_box = {}

        def waiter():
            result_box["r"] = cm.wait_for_response("u1", timeout=0)

        t = threading.Thread(target=waiter)
        t.start()
        # An unlimited wait cannot finish while nothing resolves it: still
        # running after a comfortable margin (old code auto-skipped at once).
        t.join(timeout=1.5)
        assert t.is_alive()

        # Once resolved, the unlimited wait returns the real answer.
        cm.resolve_gateway_clarify("u1", "B")
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert result_box["r"] == "B"
