"""Reasoning identity must preserve flat text and current app-server liveness contracts."""
from types import SimpleNamespace as NS

import pytest

from agent.codex_runtime import _consume_codex_event_stream, make_codex_app_server_event_bridge
from agent.stream_delivery import StreamDeliveryMixin
from tui_gateway import server


def _recording_agent(monkeypatch, native):
    sid = "reasoning-section-contract"
    wire, observers, activity = [], [], []
    monkeypatch.setattr(server, "_emit", lambda event, session_id, payload=None: wire.append((event, session_id, payload)))
    monkeypatch.setitem(server._sessions, sid, {"verbose": False, "show_reasoning": True})
    callbacks = server._agent_cbs(sid)
    agent = StreamDeliveryMixin()
    agent.session_id, agent.model, agent.provider, agent.platform = sid, "test-model", "openai-codex", "tui"
    agent.reasoning_event_callback = callbacks["reasoning_event_callback"] if native else None
    agent.reasoning_callback = callbacks["reasoning_callback"]
    agent._stream_reasoning_hooks_enabled = True
    agent._touch_activity = activity.append
    monkeypatch.setattr("agent.plugin_stream_hooks.enqueue_plugin_stream_hook", lambda event, **payload: observers.append((event, payload)))
    agent._claim_stream_writer()
    return agent, wire, observers, activity


def _flat(wire):
    return "".join(payload["text"] for event, _, payload in wire if event == "reasoning.delta")


def _observed(observers):
    return "".join(payload["delta"] for event, payload in observers if event == "on_stream_delta" and payload.get("kind") == "reasoning")


@pytest.mark.parametrize("native", [False, True])
def test_responses_summary_boundaries_reach_gateway_and_plugin_observer(monkeypatch, native):
    agent, wire, observers, _ = _recording_agent(monkeypatch, native)
    sections = [("rs_a", 0, "Inspect files"), ("rs_a", 1, "Compare results"), ("rs_b", 0, "Verify"), ("rs_c", 0, "Conclude")]
    events = []
    for item_id, index, text in sections:
        events.append(NS(type="response.reasoning_summary_part.added", item_id=item_id, summary_index=index))
        # More than one token in one source must not introduce extra separators.
        for chunk in (text[:3], text[3:]):
            events.append(NS(type="response.reasoning_summary_text.delta", item_id=item_id, summary_index=index, delta=chunk))
        events.append(NS(type="response.reasoning_summary_part.done", item_id=item_id, summary_index=index))
    events.append(NS(type="response.completed", response=NS(status="completed")))
    _consume_codex_event_stream(iter(events), model="test-model", on_reasoning_delta=agent._fire_reasoning_delta,
                               on_reasoning_event=agent._fire_reasoning_event if native else None)
    expected = "\n\n".join(text for _, _, text in sections)
    assert _flat(wire) == expected
    assert _observed(observers) == expected
    deltas = [payload for event, _, payload in wire if event == "reasoning.delta"]
    assert len(deltas) == 2 * len(sections)
    if native:
        assert [payload["reasoning_id"] for payload in deltas[::2]] == [f"{item_id}:summary:{index}" for item_id, index, _ in sections]
    else:
        assert all("reasoning_id" not in payload for payload in deltas)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("method,index_key,source_kind", [
    ("item/reasoning/summaryTextDelta", "summaryIndex", "summary"),
    ("item/reasoning/summaryDelta", "summary_index", "summary"),
    ("item/reasoning/textDelta", "contentIndex", "content"),
    ("item/reasoning/delta", "content_index", "content"),
])
def test_app_server_coordinates_keep_flat_output_liveness_and_matching_end(monkeypatch, native, method, index_key, source_kind):
    agent, wire, observers, activity = _recording_agent(monkeypatch, native)
    bridge = make_codex_app_server_event_bridge(agent)
    item = {"type": "reasoning", "id": "rs_app"}
    bridge({"method": "item/started", "params": {"item": item}})
    for index, text in [(0, "Inspect"), (0, " files"), (1, "Compare results")]:
        bridge({"method": method, "params": {"itemId": "rs_app", index_key: index, "delta": text}})
    bridge({"method": "item/completed", "params": {"item": item}})
    assert _flat(wire) == "Inspect files\n\nCompare results"
    assert _observed(observers) == _flat(wire)
    assert len(activity) == 5  # start, three real deltas, completion
    bridge({"method": method, "params": {"itemId": "rs_app", index_key: 1, "delta": ""}})
    bridge({"method": "server/heartbeat", "params": {}})
    assert len(activity) == 5  # empty payload/keepalive cannot hide a stalled turn
    if native:
        ids = [payload["reasoning_id"] for event, _, payload in wire if event == "reasoning.delta"]
        assert ids == [f"rs_app:{source_kind}:0", f"rs_app:{source_kind}:0", f"rs_app:{source_kind}:1"]
        ended = {payload["reasoning_id"] for event, _, payload in wire if event == "reasoning.end"}
        assert set(ids) <= ended
    # Main's other progress-only streams remain live even without display callbacks.
    for extra in ("item/commandExecution/outputDelta", "item/fileChange/outputDelta"):
        bridge({"method": extra, "params": {"delta": "working"}})
    assert len(activity) == 7

    # The event bridge survives multiple turns; reset source state on either turn boundary.
    for boundary in ("turn/completed", "turn/started"):
        bridge({"method": boundary, "params": {"turn": {"id": "next-turn"}}})
        wire.clear()
        observers.clear()
        fresh = {"type": "reasoning", "id": "rs_fresh_" + boundary}
        bridge({"method": "item/started", "params": {"item": fresh}})
        bridge({"method": method, "params": {"itemId": fresh["id"], index_key: 0, "delta": "Fresh"}})
        bridge({"method": "item/completed", "params": {"item": fresh}})
        assert _flat(wire) == "Fresh"
        assert _observed(observers) == "Fresh"

@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("groups", [
    (("Only",),),
    (("A", "B"),),
    (("A",), ("B",)),
    (("A", "B"), ("C",), ("D",)),
], ids=["single-part", "same-item", "different-items", "mixed"])
def test_nonstream_reemit_preserves_flat_text_and_native_identity(monkeypatch, native, groups):
    from copy import deepcopy
    from agent.chat_completion_helpers import _assistant_reasoning_text

    agent, wire, observers, _ = _recording_agent(monkeypatch, native)
    expected = "\n\n".join("\n".join(group) for group in groups)
    items = [{"type": "reasoning", "id": f"rs_{i}", "summary": [
        {"type": "summary_text", "text": text} for text in group
    ]} for i, group in enumerate(groups)]
    items.insert(0, {"type": "compaction", "encrypted_content": "opaque"})
    message = NS(reasoning=expected, codex_reasoning_items=items)
    original = deepcopy(items)
    agent._extract_reasoning = lambda message: message.reasoning
    agent.verbose_logging = False
    agent.stream_delta_callback = agent._stream_callback = None
    # Re-emitting another response must not inherit separators from the previous call.
    for _ in range(2):
        wire.clear()
        observers.clear()
        assert _assistant_reasoning_text(agent, message) == expected
        assert _flat(wire) == expected
        deltas = [payload for event, _, payload in wire if event == "reasoning.delta"]
        if native:
            assert _observed(observers) == expected
            ids = [f"rs_{i}:summary:{j}" for i, group in enumerate(groups) for j in range(len(group))]
            assert [payload["reasoning_id"] for payload in deltas] == ids
            for phase in ("reasoning.start", "reasoning.end"):
                assert [payload["reasoning_id"] for event, _, payload in wire if event == phase] == ids
        else:
            assert all("reasoning_id" not in payload for payload in deltas)
        assert items == original


@pytest.mark.parametrize("invalid", ["mismatched-text", "malformed-summary"])
def test_nonstream_reemit_keeps_flat_fallback_for_untrusted_identity(monkeypatch, invalid):
    from agent.chat_completion_helpers import _assistant_reasoning_text

    agent, wire, observers, _ = _recording_agent(monkeypatch, True)
    expected = "A\nB" if invalid == "malformed-summary" else "Already selected visible text"
    message = NS(reasoning=expected, codex_reasoning_items=[{
        "type": "reasoning", "id": "rs_untrusted", "summary": [
            {"type": "summary_text", "text": "A"},
            {"type": "summary_text", "text": None if invalid == "malformed-summary" else "B"},
        ],
    }])
    agent._extract_reasoning = lambda message: message.reasoning
    agent.verbose_logging = False
    agent.stream_delta_callback = agent._stream_callback = None
    assert _assistant_reasoning_text(agent, message) == expected
    assert _flat(wire) == expected
    assert all("reasoning_id" not in payload for event, _, payload in wire if event == "reasoning.delta")
    assert not any(event in {"reasoning.start", "reasoning.end"} for event, _, _ in wire)
    assert not _observed(observers)


def test_native_reasoning_obeys_current_session_display_policy(monkeypatch):
    from agent.chat_completion_helpers import _assistant_reasoning_text

    agent, wire, observers, _ = _recording_agent(monkeypatch, True)
    message = NS(reasoning="Inspect", codex_reasoning_items=[{
        "type": "reasoning", "id": "rs_visible",
        "summary": [{"type": "summary_text", "text": "Inspect"}],
    }])
    agent._extract_reasoning = lambda message: message.reasoning
    agent.verbose_logging = False
    agent.stream_delta_callback = agent._stream_callback = None
    # The callbacks live across setting changes; the current session owns display policy.
    for visible in (True, False, True):
        monkeypatch.setitem(server._sessions[agent.session_id], "show_reasoning", visible)
        wire.clear()
        observers.clear()
        assert _assistant_reasoning_text(agent, message) == message.reasoning
        if visible:
            assert [event for event, _, _ in wire] == ["reasoning.start", "reasoning.delta", "reasoning.end"]
            assert _flat(wire) == message.reasoning
        else:
            assert wire == []
        # Display policy is not transcript mutation or plugin-observer authorization.
        assert _observed(observers) == message.reasoning
