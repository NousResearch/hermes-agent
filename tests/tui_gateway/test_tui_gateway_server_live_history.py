"""The live/warm DISPLAY projection: the persisted display lineage reconciled with the in-memory tail."""
import contextlib
import types

from persistence.conversation.messages import ConversationMessage
from tui_gateway import server


def test_live_visible_history_falls_back_without_db_or_key(monkeypatch):
    in_memory = [{"role": "user", "content": "hi"}]
    # No DB handle available.
    assert server._live_visible_history({"session_key": "s"}, None, in_memory) == in_memory

    # DB available but the session has no persist key yet.
    class Messages:
        def read_conversation(self, *a, **k):  # pragma: no cover - not reached
            raise AssertionError("must not query without a session_key")

    persistence = types.SimpleNamespace(messages=Messages())
    monkeypatch.setattr(server, "_persistence_for", lambda _db: contextlib.nullcontext(persistence))
    assert server._live_visible_history({}, object(), in_memory) == in_memory


def test_live_visible_history_falls_back_when_db_empty(monkeypatch):
    """A brand-new live session whose first turn hasn't been flushed keeps its
    in-memory history rather than rendering empty."""
    in_memory = [{"role": "user", "content": "fresh turn not flushed yet"}]

    class EmptyMessages:
        def read_conversation(self, *a, **k):
            return ()

    persistence = types.SimpleNamespace(messages=EmptyMessages())
    monkeypatch.setattr(server, "_persistence_for", lambda _db: contextlib.nullcontext(persistence))
    assert server._live_visible_history({"session_key": "s"}, object(), in_memory) == in_memory


def test_live_visible_history_falls_back_when_db_raises(monkeypatch):
    in_memory = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    class BrokenMessages:
        def read_conversation(self, *a, **k):
            raise RuntimeError("db exploded")

    persistence = types.SimpleNamespace(messages=BrokenMessages())
    monkeypatch.setattr(server, "_persistence_for", lambda _db: contextlib.nullcontext(persistence))
    assert server._live_visible_history({"session_key": "s"}, object(), in_memory) == in_memory


def test_live_visible_history_keeps_candidate_and_fresh_tail(monkeypatch):
    """The hard case: the persisted candidate (missing from in-memory) AND a
    not-yet-flushed live turn (missing from the DB) must BOTH survive."""
    # Persisted display: has the verification candidate, lags the newest turn.
    db_display = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
    ]
    # In-memory model history: candidate collapsed out, but has a fresh turn 2.
    in_memory = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]

    class Messages:
        def read_conversation(self, key, request):
            return tuple(ConversationMessage.from_openai_message(message) for message in db_display)

    persistence = types.SimpleNamespace(messages=Messages())
    monkeypatch.setattr(server, "_persistence_for", lambda _db: contextlib.nullcontext(persistence))
    result = server._live_visible_history({"session_key": "s1"}, object(), in_memory)
    assert result == [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]


def test_reconcile_display_with_live_trusts_db_when_tail_absent():
    """If the DB tail isn't in memory (DB ahead / diverged), don't duplicate —
    serve the persisted display."""
    db_display = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    in_memory = [{"role": "user", "content": "unrelated"}]
    assert server._reconcile_display_with_live(db_display, in_memory) == db_display
    assert server._reconcile_display_with_live([], in_memory) == in_memory
    assert server._reconcile_display_with_live(db_display, []) == db_display


def test_live_visible_history_keeps_an_unflushed_repeat_of_the_last_turn_real_db(tmp_path):
    """The live tail is anchored on the last DB row's durable id, not its text: sending the same words again
    ("ok" -> "done") before the flush must not fold the new turn into the persisted one."""
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import conversation
    from tests.fixtures.storage_seed import seed_messages, seed_session

    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, "s1", "tui")
    seed_messages(db, "s1", {"role": "user", "content": "ok"}, {"role": "assistant", "content": "done"})
    live = conversation(db, "s1") + [
        {"role": "user", "content": "ok"}, {"role": "assistant", "content": "done"}]

    merged = server._live_visible_history({"session_key": "s1"}, db, live)
    assert [m["content"] for m in merged] == ["ok", "done", "ok", "done"]
    # Fully flushed: the same four rows, never doubled.
    for msg in live[2:]:
        seed_messages(db, "s1", {"role": msg["role"], "content": msg["content"]})
    flushed = conversation(db, "s1")
    assert [m["content"] for m in server._live_visible_history({"session_key": "s1"}, db, flushed)] == [
        "ok", "done", "ok", "done"]


def test_live_visible_history_matches_eager_resume_with_real_db(tmp_path):
    """E2E cross-builder consistency against a real store.

    A persisted verification candidate (finish_reason=verification_required)
    is collapsed out of the model history by repair_message_sequence but kept
    in the display lineage (#65919). The warm/live projection
    (_live_visible_history) must equal the eager session.resume display
    projection — both keeping the candidate — so switching to a live session
    shows the same substantive answer a cold resume would.
    """
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, "s1", "tui")
    seed_messages(
        db, "s1", {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "long substantive answer", "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply", "finish_reason": "stop"},
    )

    resume = db.messages.read_resume_conversations("s1")
    model_history = [m.as_openai_message() for m in resume.model]
    display_history = [m.as_openai_message() for m in resume.display]

    # The divergence #65919 introduced: candidate absent from the model
    # projection, present in the display projection.
    assert not any("long substantive" in (m.get("content") or "") for m in model_history)
    assert any("long substantive" in (m.get("content") or "") for m in display_history)

    # Eager session.resume serves the display projection.
    eager_messages = server._history_to_messages(display_history)
    # Warm/live reuse: in-memory history is the collapsed model projection.
    live_history = server._live_visible_history({"session_key": "s1"}, db, list(model_history))
    # They must agree — the candidate survives the warm switch.
    assert server._history_to_messages(live_history) == eager_messages
    assert any(m.get("text") == "long substantive answer" for m in eager_messages)


def test_live_visible_history_keeps_candidate_and_new_flushed_turn_real_db(tmp_path):
    """Real-DB variant of the combined case: a candidate from turn 1 AND a
    fully-flushed turn 2 both appear once."""
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, "s1", "tui")
    seed_messages(
        db, "s1", {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "candidate answer", "finish_reason": "verification_required"},
        {"role": "assistant", "content": "verified reply", "finish_reason": "stop"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "turn 2 reply", "finish_reason": "stop"},
    )

    model_history = [m.as_openai_message() for m in db.messages.read_resume_conversations("s1").model]
    live_history = server._live_visible_history({"session_key": "s1"}, db, list(model_history))
    texts = [m.get("text") for m in server._history_to_messages(live_history)]

    assert texts == [
        "turn 1",
        "candidate answer",
        "verified reply",
        "turn 2",
        "turn 2 reply",
    ]
