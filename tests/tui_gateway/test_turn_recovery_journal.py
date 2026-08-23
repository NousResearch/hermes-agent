"""Tests for the turn-recovery v2 journal and capability advertisement.

The wire contract these tests pin is defined by the Hermes Android client's
fail-closed parser (``lib/core/models/gateway_turn_contract.dart``). Every
assertion here mirrors a rejection branch in that parser: if the server emits a
capability block or reconcile page that violates one of these invariants, the
client silently falls back to the legacy transport instead of erroring, so a
contract regression is invisible in production. That is why these are strict
invariant assertions and not snapshot comparisons.
"""

import pytest

from tui_gateway.turn_recovery import (
    ATTACHMENT_DETACH_METHOD,
    CAPABILITY_VERSION,
    CORE_METHODS,
    EXECUTION_ROUTE,
    PROMPT_SUBMIT_VERSION,
    PROTOCOL_MAJOR,
    PROTOCOL_NAME,
    TurnJournal,
    TurnLimits,
    TurnRecoveryError,
    build_protocol_frame,
    build_turn_recovery_capability,
)


class _Clock:
    """Deterministic monotonic-ish clock; tests advance it explicitly."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def journal(clock: _Clock) -> TurnJournal:
    return TurnJournal(clock=clock)


def _digest(char: str = "a") -> str:
    return char * 64


# ---------------------------------------------------------------------------
# Capability advertisement
# ---------------------------------------------------------------------------


class TestCapabilityAdvertisement:
    def test_protocol_frame_matches_client_expectation(self):
        protocol = build_protocol_frame()
        assert protocol == {"name": PROTOCOL_NAME, "major": PROTOCOL_MAJOR}
        assert protocol["name"] == "hermes-jsonrpc"
        assert protocol["major"] == 2

    def test_capability_declares_exact_versions(self):
        cap = build_turn_recovery_capability()
        assert cap["version"] == CAPABILITY_VERSION == 2
        assert cap["prompt_submit_version"] == PROMPT_SUBMIT_VERSION == 2

    def test_capability_declares_safety_invariants(self):
        cap = build_turn_recovery_capability()
        # The client refuses the capability unless both are literally False.
        assert cap["shadow_only"] is False
        assert cap["automatic_resubmit"] is False
        assert cap["execution_route"] == EXECUTION_ROUTE
        assert cap["mobile_session_id_format"] == "canonical_lowercase_uuid"
        assert cap["client_turn_id_format"] == "canonical_lowercase_uuid"

    def test_method_sets_are_exact_without_attachments(self):
        cap = build_turn_recovery_capability(attachments_supported=False)
        assert set(cap["methods"]) == set(CORE_METHODS)
        assert set(cap["applies_to"]) == set(CORE_METHODS) | {"prompt.submit@2"}
        assert ATTACHMENT_DETACH_METHOD not in cap["methods"]

    def test_method_sets_extend_together_with_attachments(self):
        cap = build_turn_recovery_capability(attachments_supported=True)
        assert set(cap["methods"]) == set(CORE_METHODS) | {ATTACHMENT_DETACH_METHOD}
        assert set(cap["applies_to"]) == (
            set(CORE_METHODS) | {"prompt.submit@2", ATTACHMENT_DETACH_METHOD}
        )

    def test_method_lists_have_no_duplicates(self):
        # The client parses these with a strict set builder that rejects
        # duplicates outright.
        for attachments in (False, True):
            cap = build_turn_recovery_capability(attachments_supported=attachments)
            for key in ("methods", "applies_to"):
                assert len(cap[key]) == len(set(cap[key])), key

    def test_attachment_limits_present_only_when_supported(self):
        attachment_keys = (
            "max_attachments",
            "max_file_attachment_bytes",
            "max_image_attachment_bytes",
            "max_pdf_attachment_bytes",
            "max_attachment_registry_bytes",
        )
        without = build_turn_recovery_capability(attachments_supported=False)
        for key in attachment_keys:
            assert key not in without
        with_attachments = build_turn_recovery_capability(attachments_supported=True)
        for key in attachment_keys:
            assert with_attachments[key] > 0

    def test_retention_and_byte_limits_satisfy_client_inequalities(self):
        cap = build_turn_recovery_capability()
        assert cap["event_retention_seconds"] > 0
        assert cap["turn_retention_seconds"] >= cap["event_retention_seconds"]
        assert cap["max_event_bytes"] > 0
        assert cap["max_turn_bytes"] > 0
        assert 0 < cap["terminal_event_reserve_bytes"] <= cap["max_event_bytes"]
        assert cap["max_prompt_bytes"] > 0
        assert cap["reconcile_max_events"] > 0
        assert cap["reconcile_max_page_bytes"] >= cap["max_event_bytes"]

    def test_all_advertised_integers_are_real_ints(self):
        # The client uses an exact int check that rejects bools and floats.
        cap = build_turn_recovery_capability(attachments_supported=True)
        for key, value in cap.items():
            if key in {"methods", "applies_to"}:
                continue
            if isinstance(value, bool) or isinstance(value, str):
                continue
            assert isinstance(value, int), key

    def test_capability_reflects_custom_limits(self):
        limits = TurnLimits(
            event_retention_seconds=60,
            turn_retention_seconds=120,
            max_event_bytes=1024,
            max_turn_bytes=8192,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=10,
            reconcile_max_page_bytes=2048,
        )
        cap = build_turn_recovery_capability(limits=limits)
        assert cap["event_retention_seconds"] == 60
        assert cap["turn_retention_seconds"] == 120
        assert cap["reconcile_max_events"] == 10

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"event_retention_seconds": 0},
            {"turn_retention_seconds": 10, "event_retention_seconds": 20},
            {"terminal_event_reserve_bytes": 4096, "max_event_bytes": 1024},
            {"reconcile_max_page_bytes": 512, "max_event_bytes": 1024},
            {"max_prompt_bytes": 0},
            {"reconcile_max_events": 0},
        ],
    )
    def test_invalid_limits_are_rejected_at_construction(self, kwargs):
        base = dict(
            event_retention_seconds=60,
            turn_retention_seconds=120,
            max_event_bytes=1024,
            max_turn_bytes=8192,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=10,
            reconcile_max_page_bytes=2048,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            TurnLimits(**base)


# ---------------------------------------------------------------------------
# Journal: sequencing
# ---------------------------------------------------------------------------


class TestJournalSequencing:
    def test_open_turn_starts_at_seq_zero(self, journal: TurnJournal):
        turn = journal.open_turn(session_id="s1", client_turn_id="c1")
        assert turn.last_seq == 0
        assert turn.status == "accepted"

    def test_events_are_numbered_from_one_and_contiguous(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        seqs = [
            journal.append_event("s1", "c1", "message.start", "m1", {}).seq,
            journal.append_event(
                "s1", "c1", "message.delta", "m1", {"text": "a"}
            ).seq,
            journal.append_event(
                "s1", "c1", "message.delta", "m1", {"text": "b"}
            ).seq,
        ]
        assert seqs == [1, 2, 3]

    def test_last_seq_tracks_the_newest_event(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        journal.append_event("s1", "c1", "message.delta", "m1", {"text": "a"})
        assert journal.get_turn("s1", "c1").last_seq == 2

    def test_sequences_are_independent_per_turn(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.open_turn(session_id="s1", client_turn_id="c2")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        first_of_second = journal.append_event("s1", "c2", "message.start", "m2", {})
        assert first_of_second.seq == 1

    def test_sequence_never_reused_after_pruning(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=10,
            turn_retention_seconds=1000,
            max_event_bytes=1024,
            max_turn_bytes=65536,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=50,
            reconcile_max_page_bytes=4096,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        clock.advance(50)
        journal.prune()
        event = journal.append_event("s1", "c1", "message.delta", "m1", {"text": "x"})
        assert event.seq == 2


# ---------------------------------------------------------------------------
# Journal: idempotency
# ---------------------------------------------------------------------------


class TestJournalIdempotency:
    def test_reopening_same_client_turn_id_returns_existing_turn(
        self, journal: TurnJournal
    ):
        first = journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        second = journal.open_turn(session_id="s1", client_turn_id="c1")
        assert second.turn_id == first.turn_id
        assert second.created is False
        assert second.last_seq == 1

    def test_first_open_reports_created_true(self, journal: TurnJournal):
        assert journal.open_turn(session_id="s1", client_turn_id="c1").created is True

    def test_same_client_turn_id_in_other_session_is_a_distinct_turn(
        self, journal: TurnJournal
    ):
        a = journal.open_turn(session_id="s1", client_turn_id="c1")
        b = journal.open_turn(session_id="s2", client_turn_id="c1")
        assert a.turn_id != b.turn_id

    def test_turn_ids_are_unique(self, journal: TurnJournal):
        ids = {
            journal.open_turn(session_id="s1", client_turn_id=f"c{i}").turn_id
            for i in range(25)
        }
        assert len(ids) == 25


# ---------------------------------------------------------------------------
# Journal: event payload validation
# ---------------------------------------------------------------------------


class TestEventPayloadValidation:
    @pytest.mark.parametrize(
        "event_type,payload",
        [
            ("message.start", {}),
            ("message.delta", {"text": "hello"}),
            ("message.complete", {"text": "done", "status": "completed"}),
            ("turn.status", {"status": "running"}),
        ],
    )
    def test_valid_payloads_are_accepted(
        self, journal: TurnJournal, event_type, payload
    ):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        assert journal.append_event("s1", "c1", event_type, "m1", payload).seq == 1

    @pytest.mark.parametrize(
        "event_type,payload",
        [
            ("message.start", {"text": "unexpected"}),
            ("message.delta", {}),
            ("message.delta", {"text": 1}),
            ("message.delta", {"text": "a", "extra": "b"}),
            ("message.complete", {"text": "d"}),
            # message.complete must carry a terminal status.
            ("message.complete", {"text": "d", "status": "running"}),
            ("turn.status", {"status": "not-a-status"}),
            ("turn.status", {}),
            ("unknown.type", {}),
        ],
    )
    def test_invalid_payloads_are_rejected(
        self, journal: TurnJournal, event_type, payload
    ):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        with pytest.raises(TurnRecoveryError):
            journal.append_event("s1", "c1", event_type, "m1", payload)

    def test_rejected_event_does_not_consume_a_sequence_number(
        self, journal: TurnJournal
    ):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        with pytest.raises(TurnRecoveryError):
            journal.append_event("s1", "c1", "message.delta", "m1", {})
        assert journal.append_event("s1", "c1", "message.start", "m1", {}).seq == 1

    def test_append_to_unknown_turn_raises(self, journal: TurnJournal):
        with pytest.raises(TurnRecoveryError):
            journal.append_event("s1", "missing", "message.start", "m1", {})

    def test_oversized_event_is_rejected(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=60,
            turn_retention_seconds=120,
            max_event_bytes=64,
            max_turn_bytes=8192,
            terminal_event_reserve_bytes=32,
            max_prompt_bytes=4096,
            reconcile_max_events=10,
            reconcile_max_page_bytes=2048,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        journal.open_turn(session_id="s1", client_turn_id="c1")
        with pytest.raises(TurnRecoveryError):
            journal.append_event(
                "s1", "c1", "message.delta", "m1", {"text": "x" * 500}
            )


# ---------------------------------------------------------------------------
# Journal: status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    def test_status_updates_are_recorded(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.set_status("s1", "c1", "running")
        assert journal.get_turn("s1", "c1").status == "running"

    def test_terminal_status_is_final(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.set_status("s1", "c1", "completed")
        with pytest.raises(TurnRecoveryError):
            journal.set_status("s1", "c1", "running")

    def test_append_after_terminal_status_is_rejected(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.set_status("s1", "c1", "completed")
        with pytest.raises(TurnRecoveryError):
            journal.append_event("s1", "c1", "message.delta", "m1", {"text": "x"})

    def test_unknown_status_is_rejected(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        with pytest.raises(TurnRecoveryError):
            journal.set_status("s1", "c1", "bogus")

    def test_interrupt_marks_turn_interrupted_and_returns_last_seq(
        self, journal: TurnJournal
    ):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        result = journal.interrupt("s1", "c1")
        assert result.status == "interrupted"
        # last_seq must be monotonic: the interrupt itself emits turn.status.
        assert result.last_seq >= 1
        assert journal.get_turn("s1", "c1").status == "interrupted"

    def test_interrupt_is_idempotent(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        first = journal.interrupt("s1", "c1")
        second = journal.interrupt("s1", "c1")
        assert second.status == "interrupted"
        assert second.last_seq == first.last_seq


# ---------------------------------------------------------------------------
# Reconcile pages
# ---------------------------------------------------------------------------


class TestReconcileEventsMode:
    def _seed(self, journal: TurnJournal, deltas: int = 3):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        for index in range(deltas):
            journal.append_event(
                "s1", "c1", "message.delta", "m1", {"text": str(index)}
            )

    def test_full_replay_from_zero(self, journal: TurnJournal):
        self._seed(journal)
        page = journal.reconcile("s1", "c1", after_seq=0)
        assert page["mode"] == "events"
        assert page["automatic_resubmit"] is False
        assert [event["seq"] for event in page["events"]] == [1, 2, 3, 4]
        assert page["next_after_seq"] == 4
        assert page["last_seq"] == 4
        assert page["has_more"] is False

    def test_events_start_exactly_after_the_cursor(self, journal: TurnJournal):
        self._seed(journal)
        page = journal.reconcile("s1", "c1", after_seq=2)
        assert [event["seq"] for event in page["events"]] == [3, 4]

    def test_caught_up_cursor_returns_empty_page(self, journal: TurnJournal):
        self._seed(journal)
        page = journal.reconcile("s1", "c1", after_seq=4)
        assert page["events"] == []
        assert page["next_after_seq"] == 4
        assert page["has_more"] is False

    def test_pagination_respects_reconcile_max_events(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=600,
            turn_retention_seconds=1200,
            max_event_bytes=1024,
            max_turn_bytes=65536,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=2,
            reconcile_max_page_bytes=65536,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        self._seed(journal)
        page = journal.reconcile("s1", "c1", after_seq=0)
        assert [event["seq"] for event in page["events"]] == [1, 2]
        assert page["next_after_seq"] == 2
        assert page["has_more"] is True
        assert page["last_seq"] == 4

    def test_pagination_respects_page_byte_budget(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=600,
            turn_retention_seconds=1200,
            max_event_bytes=4096,
            max_turn_bytes=1_000_000,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=100,
            reconcile_max_page_bytes=4096,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        journal.open_turn(session_id="s1", client_turn_id="c1")
        for index in range(10):
            journal.append_event(
                "s1", "c1", "message.delta", "m1", {"text": "x" * 1000}
            )
        page = journal.reconcile("s1", "c1", after_seq=0)
        assert 0 < len(page["events"]) < 10
        assert page["has_more"] is True

    def test_at_least_one_event_is_returned_when_progress_is_possible(
        self, clock: _Clock
    ):
        # Never return an empty page while last_seq > after_seq: the client
        # treats that exact shape as a protocol violation and gives up.
        limits = TurnLimits(
            event_retention_seconds=600,
            turn_retention_seconds=1200,
            max_event_bytes=4096,
            max_turn_bytes=1_000_000,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=100,
            reconcile_max_page_bytes=4096,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.delta", "m1", {"text": "x" * 3000})
        page = journal.reconcile("s1", "c1", after_seq=0)
        assert len(page["events"]) == 1

    def test_page_cursor_progresses_to_completion(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=600,
            turn_retention_seconds=1200,
            max_event_bytes=1024,
            max_turn_bytes=1_000_000,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=2,
            reconcile_max_page_bytes=65536,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        self._seed(journal, deltas=6)
        cursor = 0
        collected = []
        for _ in range(10):
            page = journal.reconcile("s1", "c1", after_seq=cursor)
            collected.extend(event["seq"] for event in page["events"])
            cursor = page["next_after_seq"]
            if not page["has_more"]:
                break
        assert collected == list(range(1, 8))

    def test_has_more_is_exactly_cursor_behind_last_seq(self, journal: TurnJournal):
        self._seed(journal)
        for cursor in range(0, 5):
            page = journal.reconcile("s1", "c1", after_seq=cursor)
            assert page["has_more"] == (page["next_after_seq"] < page["last_seq"])

    def test_earliest_seq_is_positive(self, journal: TurnJournal):
        self._seed(journal)
        assert journal.reconcile("s1", "c1", after_seq=0)["earliest_seq"] >= 1

    def test_negative_cursor_is_rejected(self, journal: TurnJournal):
        self._seed(journal)
        with pytest.raises(TurnRecoveryError):
            journal.reconcile("s1", "c1", after_seq=-1)

    def test_cursor_ahead_of_last_seq_is_rejected(self, journal: TurnJournal):
        self._seed(journal)
        with pytest.raises(TurnRecoveryError):
            journal.reconcile("s1", "c1", after_seq=99)

    def test_unknown_turn_is_rejected(self, journal: TurnJournal):
        with pytest.raises(TurnRecoveryError):
            journal.reconcile("s1", "nope", after_seq=0)


class TestReconcileSnapshotMode:
    def _complete_turn(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        journal.append_event("s1", "c1", "message.delta", "m1", {"text": "hello"})
        journal.append_event(
            "s1",
            "c1",
            "message.complete",
            "m1",
            {"text": "hello", "status": "completed"},
        )
        journal.set_status("s1", "c1", "completed")

    def test_pruned_events_fall_back_to_snapshot(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=10,
            turn_retention_seconds=10_000,
            max_event_bytes=1024,
            max_turn_bytes=65536,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=50,
            reconcile_max_page_bytes=4096,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        self._complete_turn(journal)
        journal.set_final_message_ref("s1", "c1", 42)
        journal.set_attachment_manifest_digest("s1", "c1", _digest())
        clock.advance(100)
        journal.prune()

        page = journal.reconcile("s1", "c1", after_seq=0)
        assert page["mode"] == "snapshot"
        assert page["has_more"] is False
        assert page["automatic_resubmit"] is False
        snapshot = page["snapshot"]
        assert snapshot["status"] == "completed"
        assert snapshot["assistant"]["complete"] is True
        assert snapshot["assistant"]["text"] == "hello"
        assert snapshot["final_message_ref"] == 42
        assert snapshot["attachment_manifest_digest"] == _digest()

    def test_snapshot_cursors_are_pinned_to_last_seq(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=10,
            turn_retention_seconds=10_000,
            max_event_bytes=1024,
            max_turn_bytes=65536,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=50,
            reconcile_max_page_bytes=4096,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        self._complete_turn(journal)
        journal.set_final_message_ref("s1", "c1", 7)
        journal.set_attachment_manifest_digest("s1", "c1", _digest("b"))
        clock.advance(100)
        journal.prune()

        page = journal.reconcile("s1", "c1", after_seq=0)
        assert page["next_after_seq"] == page["last_seq"]
        assert page["snapshot"]["last_seq"] == page["last_seq"]

    def test_snapshot_requires_terminal_status(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=10,
            turn_retention_seconds=10_000,
            max_event_bytes=1024,
            max_turn_bytes=65536,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=50,
            reconcile_max_page_bytes=4096,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        clock.advance(100)
        journal.prune()
        # Still running with its events pruned: no honest answer exists, so the
        # server must fail loudly instead of fabricating a snapshot.
        with pytest.raises(TurnRecoveryError):
            journal.reconcile("s1", "c1", after_seq=0)

    def test_snapshot_digest_defaults_to_empty_manifest_digest(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=10,
            turn_retention_seconds=10_000,
            max_event_bytes=1024,
            max_turn_bytes=65536,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=50,
            reconcile_max_page_bytes=4096,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        self._complete_turn(journal)
        journal.set_final_message_ref("s1", "c1", 3)
        clock.advance(100)
        journal.prune()
        digest = journal.reconcile("s1", "c1", after_seq=0)["snapshot"][
            "attachment_manifest_digest"
        ]
        assert len(digest) == 64
        assert all(char in "0123456789abcdef" for char in digest)

    def test_invalid_digest_is_rejected(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        with pytest.raises(TurnRecoveryError):
            journal.set_attachment_manifest_digest("s1", "c1", "NOT-HEX")

    def test_invalid_final_message_ref_is_rejected(self, journal: TurnJournal):
        journal.open_turn(session_id="s1", client_turn_id="c1")
        with pytest.raises(TurnRecoveryError):
            journal.set_final_message_ref("s1", "c1", 0)


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


class TestRetention:
    def _limits(self) -> TurnLimits:
        return TurnLimits(
            event_retention_seconds=10,
            turn_retention_seconds=100,
            max_event_bytes=1024,
            max_turn_bytes=65536,
            terminal_event_reserve_bytes=256,
            max_prompt_bytes=4096,
            reconcile_max_events=50,
            reconcile_max_page_bytes=4096,
        )

    def test_fresh_events_survive_pruning(self, clock: _Clock):
        journal = TurnJournal(clock=clock, limits=self._limits())
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        clock.advance(5)
        journal.prune()
        assert journal.reconcile("s1", "c1", after_seq=0)["events"] != []

    def test_expired_turn_is_dropped_entirely(self, clock: _Clock):
        journal = TurnJournal(clock=clock, limits=self._limits())
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        clock.advance(1000)
        journal.prune()
        with pytest.raises(TurnRecoveryError):
            journal.reconcile("s1", "c1", after_seq=0)

    def test_turn_byte_cap_evicts_oldest_events_first(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=10_000,
            turn_retention_seconds=100_000,
            max_event_bytes=4096,
            max_turn_bytes=6000,
            terminal_event_reserve_bytes=512,
            max_prompt_bytes=4096,
            reconcile_max_events=100,
            reconcile_max_page_bytes=8192,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        journal.open_turn(session_id="s1", client_turn_id="c1")
        for _ in range(10):
            journal.append_event(
                "s1", "c1", "message.delta", "m1", {"text": "x" * 1000}
            )
        page = journal.reconcile("s1", "c1", after_seq=9)
        assert page["earliest_seq"] > 1
        assert page["last_seq"] == 10

    def test_terminal_event_is_admitted_within_its_reserve(self, clock: _Clock):
        limits = TurnLimits(
            event_retention_seconds=10_000,
            turn_retention_seconds=100_000,
            max_event_bytes=4096,
            max_turn_bytes=4000,
            terminal_event_reserve_bytes=2048,
            max_prompt_bytes=4096,
            reconcile_max_events=100,
            reconcile_max_page_bytes=8192,
        )
        journal = TurnJournal(clock=clock, limits=limits)
        journal.open_turn(session_id="s1", client_turn_id="c1")
        for _ in range(20):
            journal.append_event(
                "s1", "c1", "message.delta", "m1", {"text": "x" * 500}
            )
        event = journal.append_event(
            "s1",
            "c1",
            "message.complete",
            "m1",
            {"text": "final", "status": "completed"},
        )
        assert event.seq == 21
        assert journal.get_turn("s1", "c1").last_seq == 21

    def test_prune_is_idempotent(self, clock: _Clock):
        journal = TurnJournal(clock=clock, limits=self._limits())
        journal.open_turn(session_id="s1", client_turn_id="c1")
        journal.append_event("s1", "c1", "message.start", "m1", {})
        clock.advance(50)
        journal.prune()
        journal.prune()
        assert journal.get_turn("s1", "c1").last_seq == 1
