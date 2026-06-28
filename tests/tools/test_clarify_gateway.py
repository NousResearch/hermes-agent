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
        result = cm.wait_for_response("id1", timeout=2.0)
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
        """Arbitrary typed text should resolve as a custom Other answer."""
        from tools import clarify_gateway as cm

        cm.register("id3d", "sk3d", "Pick", ["X", "Y"])
        custom = "None of those are valid options"
        assert cm.resolve_text_response_for_session("sk3d", custom) is True
        assert cm.wait_for_response("id3d", timeout=0.1) == custom

    def test_parse_multi_select_response_exact_label_beats_letter_shortcut(self):
        from tools import clarify_gateway as cm

        assert cm.parse_multi_select_response("A", ["Alpha", "A", "Gamma"]) == ["A"]

    def test_resolve_text_response_rejects_custom_when_other_disallowed(self):
        from tools import clarify_gateway as cm

        cm.register("id-no-other", "sk-no-other", "Pick", ["X", "Y"], allow_other=False)
        assert cm.resolve_text_response_for_session("sk-no-other", "not offered") is False
        with cm._lock:
            entry = cm._entries.get("id-no-other")
        assert entry is not None
        assert entry.response is None
        assert not entry.event.is_set()

    def test_resolve_text_response_accepts_exact_choice_when_other_disallowed(self):
        from tools import clarify_gateway as cm

        cm.register("id-exact", "sk-exact", "Pick", ["Alpha", "A"], allow_other=False)
        assert cm.resolve_text_response_for_session("sk-exact", "A") is True
        assert cm.wait_for_response("id-exact", timeout=0.1) == "A"

    def test_multi_select_text_response_enforces_min_max_and_allow_other(self):
        from tools import clarify_gateway as cm

        cm.register(
            "id-ms-bounds",
            "sk-ms-bounds",
            "Pick",
            ["A", "B", "C"],
            multi_select=True,
            min_selections=2,
            max_selections=2,
            allow_other=False,
        )
        assert cm.resolve_text_response_for_session("sk-ms-bounds", "1") is False
        assert cm.resolve_text_response_for_session("sk-ms-bounds", "1,2,3") is False
        assert cm.resolve_text_response_for_session("sk-ms-bounds", "custom") is False
        assert cm.resolve_text_response_for_session("sk-ms-bounds", "1,3") is True
        assert cm.wait_for_response("id-ms-bounds", timeout=0.1) == "A, C"

    def test_invalid_text_response_message_mentions_bounds(self):
        from tools import clarify_gateway as cm

        entry = cm.register(
            "id-ms-message",
            "***",
            "Pick",
            ["A", "B", "C"],
            multi_select=True,
            min_selections=2,
            max_selections=2,
            allow_other=False,
        )

        message = cm.format_invalid_text_response_message(entry)
        assert "multi-select" in message
        assert "at least 2" in message
        assert "at most 2" in message

    def test_multi_select_metadata_is_stored_in_signature(self):
        """Gateway entries carry multi-select metadata for adapters/Desktop."""
        from tools import clarify_gateway as cm

        entry = cm.register(
            "id-ms",
            "sk-ms",
            "Pick several",
            ["Alpha", "Beta", "Gamma"],
            multi_select=True,
            min_selections=1,
            max_selections=2,
            allow_other=False,
        )

        assert entry.multi_select is True
        assert entry.min_selections == 1
        assert entry.max_selections == 2
        assert entry.allow_other is False
        assert entry.signature()["multi_select"] is True

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
            result = fut.result(timeout=2.0)
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
            result = fut.result(timeout=2.0)
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

    def test_parse_multi_select_response_accepts_numbers_and_letters(self):
        from tools import clarify_gateway as cm

        choices = ["Alpha", "Beta", "Gamma", "Delta"]

        assert cm.parse_multi_select_response("1,3", choices) == ["Alpha", "Gamma"]
        assert cm.parse_multi_select_response("A+C", choices) == ["Alpha", "Gamma"]
        assert cm.parse_multi_select_response("2 + 4", choices) == ["Beta", "Delta"]

    def test_parse_multi_select_response_exact_labels_and_custom_text(self):
        from tools import clarify_gateway as cm

        choices = ["Alpha", "Beta", "Gamma"]
        assert cm.parse_multi_select_response("Alpha, Gamma", choices) == ["Alpha", "Gamma"]
        assert cm.parse_multi_select_response("Something else", choices) is None
