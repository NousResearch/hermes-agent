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
            cm.resolve_gateway_clarify("id1", "B", session_key="sk1")

        threading.Thread(target=resolver).start()
        result = cm.wait_for_response("id1", timeout=10.0)
        assert result == "B"

    def test_first_resolution_wins(self):
        """A late cancellation must not overwrite an already-selected choice."""
        from tools import clarify_gateway as cm

        entry = cm.register("id-race", "sk-race", "Pick one", ["A", "B"])

        assert cm.resolve_gateway_clarify("id-race", "A", session_key="sk-race") is True
        assert cm.resolve_gateway_clarify("id-race", "", session_key="sk-race") is False
        assert entry.response == "A"

    def test_resolve_requires_matching_session_key(self):
        """Foreign or missing session_key must not resolve another session's clarify (#87780)."""
        from tools import clarify_gateway as cm

        entry = cm.register("cid-B", "sB", "Pick", ["A", "B"])
        assert cm.resolve_gateway_clarify("cid-B", "FORGED") is False
        assert cm.resolve_gateway_clarify("cid-B", "FORGED", session_key="sA") is False
        assert entry.response is None
        assert entry.event.is_set() is False
        assert cm.resolve_gateway_clarify("cid-B", "OK", session_key="sB") is True
        assert entry.response == "OK"

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


    def test_clear_session_preserves_resolved_response(self):
        """clear_session must not clobber an answer that already won.

        First-writer-wins (doryani-ai on PR #75732): a button callback that
        resolved the entry before session cleanup must keep its response.
        clear_session only cancels entries whose event is not yet set, so
        the racing waiter observes the real answer, not the empty sentinel.
        """
        from tools import clarify_gateway as cm

        cm.register("id-race", "sk-race", "Pick one", ["A", "B"])

        def waiter():
            return cm.wait_for_response("id-race", timeout=10.0)

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(waiter)
            time.sleep(0.05)
            # Button wins the race first...
            assert cm.resolve_gateway_clarify("id-race", "B", session_key="sk-race") is True
            # ...then session cleanup runs before the waiter wakes.
            cancelled = cm.clear_session("sk-race")
            assert cancelled == 0
            result = fut.result(timeout=10.0)
            # The real answer must survive cleanup, not the "" cancellation.
            assert result == "B"


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


    def test_wait_for_response_unknown_id_returns_none(self):
        """wait_for_response on a non-existent id returns None immediately."""
        from tools import clarify_gateway as cm

        assert cm.wait_for_response("nonexistent-id", timeout=0.1) is None


    def test_get_clarify_timeout_exception_returns_default(self, monkeypatch):
        """get_clarify_timeout returns 3600 when load_config raises."""
        from tools import clarify_gateway as cm

        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cm.get_clarify_timeout() == 3600


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
        cm.resolve_gateway_clarify("u1", "B", session_key="sk")
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert result_box["r"] == "B"


class TestMultiSelectTextFallback:
    """Multi-select clarifies via the gateway text fallback.

    The adapter's numbered-list fallback asks the user to reply with
    comma/space-separated numbers; _coerce_text_response must map those to a
    JSON array of choice labels (which _parse_multi_select_response on the
    tool side decodes into a list).
    """

    def setup_method(self):
        _clear_clarify_state()

    def _register_multi(self, cid="m1", choices=("A", "B", "C")):
        from tools import clarify_gateway as cm
        entry = cm.register(cid, "sk", "Pick some", list(choices), multi_select=True)
        # Text fallback path always flips awaiting_text on.
        cm.mark_awaiting_text(cid)
        return entry

    def test_register_stores_multi_select_flag(self):
        entry = self._register_multi()
        assert entry.multi_select is True
        assert entry.signature()["multi_select"] is True


    def test_multi_select_without_choices_is_ignored(self):
        """multi_select on an open-ended clarify is meaningless — dropped."""
        from tools import clarify_gateway as cm
        entry = cm.register("s2", "sk", "Q?", None, multi_select=True)
        assert entry.multi_select is False


    def test_duplicate_selections_deduped(self):
        import json
        from tools import clarify_gateway as cm
        entry = self._register_multi()
        coerced = cm._coerce_text_response(entry, "1, 1, 2")
        assert json.loads(coerced) == ["A", "B"]

    def test_resolve_text_response_end_to_end(self):
        """resolve_text_response_for_session delivers the JSON array to the waiter."""
        import json
        from tools import clarify_gateway as cm
        self._register_multi(cid="m3")
        result_box = {}

        def waiter():
            result_box["r"] = cm.wait_for_response("m3", timeout=5)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        assert cm.resolve_text_response_for_session("sk", "1,2") is True
        t.join(timeout=5)
        assert json.loads(result_box["r"]) == ["A", "B"]

    def test_single_select_regression_numeric(self):
        """Single-select coercion unchanged: '2' maps to the choice label string."""
        from tools import clarify_gateway as cm
        entry = cm.register("s3", "sk", "Q?", ["A", "B", "C"])
        assert cm._coerce_text_response(entry, "2") == "B"

    def test_single_select_regression_label(self):
        from tools import clarify_gateway as cm
        entry = cm.register("s4", "sk", "Q?", ["A", "B"])
        assert cm._coerce_text_response(entry, "b") == "B"


class TestNativeRejectClassification:
    """Rejected typed replies must distinguish free prose from bad selections.

    Free prose cancels/falls through (deadlock break). Selection-shaped but
    invalid replies (out-of-range number, unrecognised comma-list) keep the
    pending clarify armed so the user can retry.
    """

    def setup_method(self):
        _clear_clarify_state()

    def test_multi_select_out_of_range_is_invalid_selection(self):
        from tools import clarify_gateway as cm

        entry = cm.register(
            "ms-oor", "sk-ms", "Pick some", ["A", "B", "C"], multi_select=True,
        )
        assert entry.awaiting_text is False
        value, reason = cm._coerce_text_response_detailed(entry, "99")
        assert value is None
        assert reason == "invalid_selection"
        assert cm.attempt_text_response_for_session("sk-ms", "99") == (
            cm.TEXT_REJECTED_SELECTION
        )
        pending = cm.get_pending_for_session("sk-ms", include_choice_prompts=True)
        assert pending is not None
        assert not pending.event.is_set()

    def test_multi_select_bad_comma_list_is_invalid_selection(self):
        from tools import clarify_gateway as cm

        entry = cm.register(
            "ms-bad", "sk-ms2", "Pick some", ["A", "B", "C"], multi_select=True,
        )
        value, reason = cm._coerce_text_response_detailed(entry, "1,99")
        assert value is None
        assert reason == "invalid_selection"
        assert cm.attempt_text_response_for_session("sk-ms2", "nope,nope") == (
            cm.TEXT_REJECTED_SELECTION
        )
        pending = cm.get_pending_for_session("sk-ms2", include_choice_prompts=True)
        assert pending is not None
        assert not pending.event.is_set()

    def test_multi_select_free_prose_is_rejected_prose(self):
        from tools import clarify_gateway as cm

        entry = cm.register(
            "ms-prose", "sk-ms3", "Pick some", ["A", "B"], multi_select=True,
        )
        value, reason = cm._coerce_text_response_detailed(
            entry, "just checking the visual UI, no need to pass any data",
        )
        assert value is None
        assert reason == "prose"
        assert cm.attempt_text_response_for_session(
            "sk-ms3", "just checking the visual UI, no need to pass any data",
        ) == cm.TEXT_REJECTED_PROSE

    def test_single_select_out_of_range_is_invalid_selection(self):
        from tools import clarify_gateway as cm

        entry = cm.register("ss-oor", "sk-ss", "Pick one", ["A", "B"])
        value, reason = cm._coerce_text_response_detailed(entry, "9")
        assert value is None
        assert reason == "invalid_selection"
        assert cm.attempt_text_response_for_session("sk-ss", "9") == (
            cm.TEXT_REJECTED_SELECTION
        )

    def test_single_select_prose_is_rejected_prose(self):
        from tools import clarify_gateway as cm

        entry = cm.register("ss-prose", "sk-ss2", "Pick one", ["A", "B"])
        value, reason = cm._coerce_text_response_detailed(
            entry, "one more unrelated thought",
        )
        assert value is None
        assert reason == "prose"
