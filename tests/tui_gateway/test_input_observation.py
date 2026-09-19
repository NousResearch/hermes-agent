"""Behavior at the input/queue/publication boundaries, with inference controlled."""

import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.input_observation import new_input, project_inputs, snapshot, state
from tui_gateway.turn_observation import make_turn, turn_scope


def live(monkeypatch):
    session = {"running": True, "history_lock": threading.Lock(), "history": [], "session_key": "observation"}
    server._start_inflight_turn(session, "original")
    batch = new_input(session, "original", "original-ref")
    observation = make_turn("observe", session, inputs=batch)
    session["_turn_observation"] = observation
    session["inflight_turn"].update(turn=observation.wire(), input_batch=batch)
    frames = []
    session["transport"] = SimpleNamespace(write=lambda f: frames.append(f) or True)
    monkeypatch.setitem(server._sessions, "observe", session)
    observation.registered = True
    monkeypatch.setattr(server, "_session_info", lambda *args: {"submission_state": snapshot(session)})
    return session, observation, frames


def test_runner_ownership_refusal_retains_input_outcome(monkeypatch):
    session, _, frames = live(monkeypatch)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: "owned elsewhere")
    assert server._run_prompt_submit(1, "observe", session, "original") is False
    error = next(f["params"]["payload"] for f in frames if f["params"]["type"] == "error")
    assert error["inputs"][0]["ref"] == "original-ref"
    assert snapshot(session)["outcomes"][-1]["reason"] == "ownership_refused"
    assert not session["running"]
    assert server._inflight_snapshot(session) is None


def test_queued_dispatch_exception_is_correlated_and_clears_snapshot(monkeypatch):
    session, _, frames = live(monkeypatch)
    session["running"] = False
    batch = new_input(session, "next", "next-ref")
    server._enqueue_prompt(session, "next", None, input_batch=batch)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *args: False)
    def fail(*args, **kwargs):
        raise RuntimeError("dispatch unavailable")
    monkeypatch.setattr(server, "_run_prompt_submit", fail)
    assert server._drain_queued_prompt(1, "observe", session)
    error = next(f["params"]["payload"] for f in frames if f["params"]["type"] == "error")
    assert error["inputs"][0]["ref"] == "next-ref"
    assert snapshot(session)["outcomes"][-1]["reason"] == "dispatch_exception"
    assert snapshot(session)["queued"] == []
    assert server._inflight_snapshot(session) is None


def test_rendering_callback_cannot_publish_or_mutate_after_terminal(monkeypatch):
    session, observation, frames = live(monkeypatch)
    callbacks = []
    entered, release = threading.Event(), threading.Event()

    def feed(delta):
        entered.set()
        assert release.wait(5)

    def run_conversation(message, **kwargs):
        callbacks.append(kwargs["stream_callback"])
        return {}

    agent = SimpleNamespace(run_conversation=run_conversation)
    st = SimpleNamespace(agent=agent, history=[], tts_queue=None)
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *args: (threading.Event(), SimpleNamespace(join=lambda: None)))
    with turn_scope(observation):
        server._invoke_agent("observe", session, st, "original", "original", SimpleNamespace(feed=feed), [], None, None)
    worker = threading.Thread(target=lambda: callbacks[0]("late"))
    worker.start()
    try:
        assert entered.wait(5)
        with turn_scope(observation):
            server._emit("message.complete", "observe", {"status": "complete"})
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert [f["params"]["type"] for f in frames] == ["message.complete"]
    assert session["inflight_turn"]["assistant"] == ""
    assert frames[0]["params"]["payload"]["inputs"][0]["ref"] == "original-ref"
    assert snapshot(session)["outcomes"][0]["disposition"] == "terminal"


def test_correction_callout_cannot_mutate_replacement_execution(monkeypatch):
    session, observation, frames = live(monkeypatch)

    def redirect(text):
        with turn_scope(observation):
            server._emit("message.complete", "observe", {"status": "complete"})
        # Deliberate provider-side replacement attacks the post-call target check.
        server._start_inflight_turn(session, "replacement")
        session["inflight_turn"]["turn"] = {"id": "replacement"}
        return True

    session["agent"] = SimpleNamespace(redirect=redirect)
    result = server._apply_correction(1, session, "redirect", "for original", "redirected", sid="observe", visible=True)
    assert result["result"]["status"] == "redirected"
    assert result["result"]["submission"]["disposition"] == "unresolved"
    assert not session["inflight_turn"].get("corrections")
    assert not any(f["params"]["type"] == "message.input" for f in frames)


def test_two_equal_corrections_have_distinct_occurrences_and_one_execution(monkeypatch):
    session, observation, frames = live(monkeypatch)
    session["agent"] = SimpleNamespace(steer=lambda text: True)
    replies = [server._apply_correction(n, session, "steer", "same", "queued", sid="observe", visible=True)
               for n in (1, 2)]
    inputs = [f["params"] for f in frames if f["params"]["type"] == "message.input"]
    assert len(inputs) == 2
    assert all(e["turn"] == observation.wire() for e in inputs)
    assert inputs[0]["payload"]["inputs"] != inputs[1]["payload"]["inputs"]
    resumed = server._inflight_snapshot(session)
    assert resumed["user"] == "original"
    assert resumed["input_observations"] == [e["payload"] for e in inputs]
    assert [r["result"]["submission"]["input_id"] for r in replies] == [e["payload"]["inputs"][0]["id"] for e in inputs]


def test_unspecified_legacy_correction_has_no_new_text_projection(monkeypatch):
    session, _, frames = live(monkeypatch)
    session["agent"] = SimpleNamespace(steer=lambda text: True)
    server._apply_correction(1, session, "steer", "unspecified note", "queued", sid="observe")
    assert server._inflight_snapshot(session)["corrections"] == ["unspecified note"]
    assert "input_observations" not in server._inflight_snapshot(session)
    assert not any(f["params"]["type"] == "message.input" for f in frames)


def test_idle_clear_publishes_positive_cancellation(monkeypatch):
    session, _, frames = live(monkeypatch)
    session["running"] = False
    server._enqueue_prompt(session, "pending", None)
    occurrence = snapshot(session)["queued"][0]["inputs"][0]
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a: False)
    monkeypatch.setattr(server, "_clear_pending", lambda *a: None)
    server._interrupt_session_turn("observe", session)
    info = [f["params"]["payload"] for f in frames if f["params"]["type"] == "session.info"]
    assert info and info[-1]["submission_state"]["queued"] == []
    assert any(o["input"] == occurrence and o["disposition"] == "cancelled"
               for o in info[-1]["submission_state"]["outcomes"])


@pytest.mark.parametrize("ref", ["", "line\nbreak", "é", "x" * 65, None, 1])
def test_invalid_references_do_not_change_input_acceptance(ref):
    session = {"history_lock": threading.Lock()}
    batch = new_input(session, "words", ref)
    assert "ref" not in project_inputs(session, batch)["inputs"][0]


def test_same_reference_is_not_server_deduplication():
    session = {"history_lock": threading.Lock()}
    first, second = [new_input(session, "words", "retry") for _ in range(2)]
    a, b = [project_inputs(session, item)["inputs"][0] for item in (first, second)]
    assert a["id"] != b["id"] and a["ref"] == b["ref"]


def test_metadata_eviction_keeps_model_text_and_reports_uncertainty():
    session = {"history_lock": threading.Lock()}
    texts = [f"words {n}" for n in range(300)]
    for text in texts:
        server._enqueue_prompt(session, text, None)
    assert session["queued_prompt"]["text"] == "\n\n".join(texts)
    projected = snapshot(session)
    assert not projected["queued_complete"]
    assert not projected["queued"][0]["inputs_complete"]
    assert len(state(session).records) <= 256


def test_outcome_eviction_bounds_serialized_bytes_and_exposes_watermark():
    import json
    from tui_gateway.input_observation import record_outcome, MAX_BYTES, MAX_RECORDS
    session = {"history_lock": threading.Lock()}
    turn = {"id": "execution", "source": {"kind": "connection", "socket_id": "s" * 32}}
    for n in range(400):
        batch = new_input(session, "unchanged model work", "\\" * 64)
        record_outcome(session, batch, "terminal", turn=turn, status="complete")
    current = snapshot(session)
    assert 0 < len(current["outcomes"]) <= MAX_RECORDS
    assert len(json.dumps(current["outcomes"], separators=(",", ":"), ensure_ascii=True).encode("ascii")) <= MAX_BYTES
    watermark = current["outcomes_truncated_before_revision"]
    assert watermark is not None
    assert all(outcome["revision"] > watermark for outcome in current["outcomes"])
    assert all(outcome["disposition"] == "terminal" for outcome in current["outcomes"])


def test_queued_redirect_reply_survives_snapshot_eviction(monkeypatch):
    from tui_gateway import input_observation
    session, _, _ = live(monkeypatch)
    session["agent"] = None
    def evict(sid, record):
        with record["history_lock"]:
            for n in range(300):
                new_input(record, "other input", str(n))
    monkeypatch.setattr(input_observation, "publish_state", evict)
    response = server._methods["session.redirect"](1, {
        "session_id": "observe", "text": "next prompt", "submission_ref": "reply-ref"})
    assert response["result"]["submission"]["ref"] == "reply-ref"
    assert response["result"]["submission"]["disposition"] == "queued"
    assert snapshot(session)["queued"][0]["inputs_complete"] is False
    assert session["queued_prompt"]["text"] == "next prompt"


def test_sanitizer_reports_only_fully_removed_occurrences():
    session = {"history_lock": threading.Lock()}
    server._enqueue_prompt(session, "original", None)
    removed = snapshot(session)["queued"][0]["inputs"][0]
    server._enqueue_prompt(session, "follow up", None)
    kept = snapshot(session)["queued"][0]["inputs"][1]
    server._start_inflight_turn(session, "original")
    server._drop_queued_duplicates_of_inflight_user(session)
    current = snapshot(session)
    assert session["queued_prompt"]["text"] == "follow up"
    assert current["queued"][0]["inputs"] == [kept]
    assert [(o["input"], o["disposition"]) for o in current["outcomes"]] == [(removed, "absorbed")]


def test_new_admission_waits_for_correction_callout_but_terminal_does_not(monkeypatch):
    session, observation, frames = live(monkeypatch)
    entered, release, claiming, claimed = [threading.Event() for _ in range(4)]
    replies = []

    def redirect(text):
        entered.set()
        assert release.wait(10)
        return True

    session["agent"] = SimpleNamespace(redirect=redirect)
    correction = threading.Thread(target=lambda: replies.append(server._apply_correction(
        1, session, "redirect", "for original", "redirected", sid="observe", visible=True)))

    def claim():
        claiming.set()
        server._lock_in_submit_turn(2, "observe", session, "replacement", {}, False, None, None, None)
        claimed.set()

    correction.start()
    next_turn = threading.Thread(target=claim)
    try:
        assert entered.wait(5)
        with turn_scope(observation):
            server._emit("message.complete", "observe", {"status": "complete"})
        assert frames[0]["params"]["type"] == "message.complete"
        session["running"] = False
        next_turn.start()
        assert claiming.wait(5)
        assert not claimed.wait(2), "admission reused the agent during an outstanding callout"
    finally:
        release.set()
        correction.join(5)
        if next_turn.ident:
            next_turn.join(5)
    assert not correction.is_alive() and not next_turn.is_alive()
    assert claimed.is_set()
    assert replies[0]["result"]["submission"]["disposition"] == "unresolved"
    assert session["inflight_turn"]["user"] == "replacement"
    assert not session["inflight_turn"].get("corrections")


def test_close_waits_for_publication_before_removing_destination(monkeypatch):
    session, observation, frames = live(monkeypatch)
    entered, release, closing, closed = [threading.Event() for _ in range(4)]

    def write(frame):
        entered.set()
        assert release.wait(10)
        frames.append(frame)
        return True

    session["transport"] = SimpleNamespace(write=write)

    def publish():
        with turn_scope(observation):
            server._emit("message.delta", "observe", {"text": "last"})

    def close():
        closing.set()
        server._pop_session_by_id("observe")
        closed.set()

    publisher = threading.Thread(target=publish)
    closer = threading.Thread(target=close)
    publisher.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert closing.wait(5)
        assert not closed.wait(2)
    finally:
        release.set()
        publisher.join(5)
        if closer.ident:
            closer.join(5)
    assert not publisher.is_alive() and not closer.is_alive()
    assert closed.is_set()
    replacement = []
    monkeypatch.setitem(server._sessions, "observe", {"transport": SimpleNamespace(write=replacement.append)})
    with turn_scope(observation):
        assert server._emit("message.delta", "observe", {"text": "stale"}) is False
    assert len(frames) == 1 and not replacement


def test_parent_replaces_child_queue_state_and_publishes_on_completion(monkeypatch):
    session, observation, frames = live(monkeypatch)
    server._enqueue_prompt(session, "parent pending", None)
    expected = snapshot(session)
    server._relay_compute_host_rpc({"method": "event", "params": {
        "type": "session.info", "session_id": "observe", "payload": {
            "submission_state": {"revision": 999, "queued": []}}}})
    assert frames[-1]["params"]["payload"]["submission_state"] == expected
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *a: False)
    with turn_scope(observation):
        server._on_compute_host_turn_done(1, "observe", session, {"type": "turn.end", "session_info_emitted": True})
    assert frames[-1]["params"]["type"] == "session.info"
    assert frames[-1]["params"]["payload"]["submission_state"]["queued"] == expected["queued"]


def test_rejected_busy_correction_that_queues_has_no_failure_outcome(monkeypatch):
    session, _, _ = live(monkeypatch)
    session["agent"] = SimpleNamespace(steer=lambda text: False)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    batch = new_input(session, "follow up", "queued-ref")
    response = server._handle_busy_submit(1, "observe", session, "follow up", None, input_batch=batch)
    assert response["result"]["submission"]["disposition"] == "queued"
    assert not snapshot(session)["outcomes"]


def test_observed_compute_cancellation_retires_mirrored_request(monkeypatch):
    session, observation, frames = live(monkeypatch)
    request = {"jsonrpc": "2.0", "id": "srq-child-1", "method": "sudo",
               "params": {"session_id": "observe"}}
    server._relay_compute_host_rpc(request)
    assert session["_compute_host_open_request"]["id"] == request["id"]
    cancel = {"jsonrpc": "2.0", "method": "event", "params": {
        "type": "request.cancel", "session_id": "observe", "turn": observation.wire(),
        "payload": {"id": request["id"], "method": "sudo", "reason": "timeout"}}}
    assert server._relay_compute_host_rpc(cancel)
    assert "_compute_host_open_request" not in session
    assert frames[-1]["params"]["type"] == "request.cancel"


def test_dispatch_ignores_invalid_refs_but_rejects_forged_observation_authority(monkeypatch):
    session, _, _ = live(monkeypatch)
    calls = []
    session["agent"] = SimpleNamespace(steer=lambda text: calls.append(text) or True)
    for ref in ("", "line\nbreak", "é", "x" * 65, None, 1, [], {"id": "forged"}):
        response = server.handle_request({"id": "bad-ref", "method": "session.steer", "params": {
            "session_id": "observe", "text": "words", "submission_ref": ref,
            "input_visibility": "visible"}})
        assert "error" not in response, response
        assert "ref" not in response["result"]["submission"]
    assert calls == ["words"] * 8
    for method in ("session.steer", "session.redirect", "prompt.submit"):
        for key in ("turn", "input_batch", "input_id", "socket_id"):
            response = server.handle_request({"id": "forged", "method": method, "params": {
                "session_id": "observe", "text": "words", key: "forged"}})
            assert response["error"]["code"] == 4000
            assert key in response["error"]["message"]
    assert calls == ["words"] * 8

def test_batched_terminal_byte_eviction_keeps_retained_revisions_above_watermark():
    import json
    from tui_gateway.input_observation import record_outcome, MAX_BYTES, MAX_RECORDS

    session = {"history_lock": threading.Lock()}
    texts = [f"model work {n}" for n in range(MAX_RECORDS)]
    for text in texts:
        server._enqueue_prompt(session, text, None, input_batch=new_input(session, text, "\\" * 64))
    entry = session["queued_prompt"]
    inputs = project_inputs(session, entry["input_batch"])
    assert inputs["inputs_complete"] and len(inputs["inputs"]) == MAX_RECORDS
    turn = {"id": "t" * 32, "source": {"kind": "connection", "socket_id": "s" * 32}}
    record_outcome(session, entry["input_batch"], "terminal", turn=turn, status="complete")
    current = snapshot(session)
    outcomes = current["outcomes"]
    assert 0 < len(outcomes) < MAX_RECORDS  # Byte eviction within this single batch.
    assert [item["input"] for item in outcomes] == inputs["inputs"][-len(outcomes):]
    assert all(item["disposition"] == "terminal" and item["status"] == "complete"
               and item["turn"] == turn for item in outcomes)
    assert len(json.dumps(outcomes, separators=(",", ":"), ensure_ascii=True).encode("ascii")) <= MAX_BYTES
    watermark = current["outcomes_truncated_before_revision"]
    assert watermark is not None and all(item["revision"] > watermark for item in outcomes)
    assert entry["text"] == "\n\n".join(texts)
