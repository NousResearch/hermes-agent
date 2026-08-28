"""Unit tests for stream_perf_hooks aggregation logic.

Covers StreamPerfCollector's turn-aggregation semantics:
  - TTFT = first_delta_at - started_at (only for API calls with a first chunk)
  - Generation window gen_ms = ended_at - first_delta_at
  - TPS accounting only counts output_tokens from calls with a generation
    window, so tool turns / queueing time never dilute TPS
  - Turn boundaries and session isolation
"""

import pytest

from tui_gateway.stream_perf_hooks import StreamPerfCollector


def _collector():
    return StreamPerfCollector()


class TestTurnLifecycle:
    def test_begin_turn_returns_empty(self):
        c = _collector()
        c.begin_turn("s1")
        assert c.end_turn("s1") is None

    def test_end_turn_without_begin_is_none(self):
        c = _collector()
        assert c.end_turn("s1") is None

    def test_end_turn_clears_state(self):
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 1.0)
        c.on_api_done("s1", 0, 0.5, 10.5, 200)
        summary = c.end_turn("s1")
        assert summary is not None
        assert summary["calls"] == 1
        # A second end should be None (already cleaned up)
        assert c.end_turn("s1") is None


class TestTtft:
    def test_ttft_is_first_delta_minus_started_at(self):
        c = _collector()
        c.begin_turn("s1")
        # started_at = 100.0, first delta at 114.0 -> TTFT = 14.0s
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        summary = c.end_turn("s1")
        assert summary["ttft_calls"] == 1
        assert summary["ttft_ms"] == pytest.approx(14000.0)

    def test_first_delta_only_recorded_once(self):
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_first_delta("s1", 0, 200.0)  # later deltas do not overwrite
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        summary = c.end_turn("s1")
        assert summary["ttft_ms"] == pytest.approx(14000.0)

    def test_call_without_first_delta_skips_ttft(self):
        # Tool turn: no text delta -> contributes no TTFT
        c = _collector()
        c.begin_turn("s1")
        c.on_api_done("s1", 0, 100.0, 120.0, 50)
        summary = c.end_turn("s1")
        assert summary["calls"] == 1
        assert summary["ttft_calls"] == 0
        assert summary["ttft_ms"] == 0.0

    def test_multiple_calls_accumulate(self):
        c = _collector()
        c.begin_turn("s1")
        # call 0: started=100, delta=114, ended=130, out=300
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        # call 1: started=131, delta=133, ended=145, out=400
        c.on_first_delta("s1", 1, 133.0)
        c.on_api_done("s1", 1, 131.0, 145.0, 400)
        summary = c.end_turn("s1")
        assert summary["calls"] == 2
        assert summary["ttft_calls"] == 2
        assert summary["ttft_ms"] == pytest.approx((14.0 + 2.0) * 1000)
        assert summary["gen_ms"] == pytest.approx((16.0 + 12.0) * 1000)
        assert summary["output_tokens"] == 700


class TestGenWindow:
    def test_gen_window_is_ended_minus_first_delta(self):
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        summary = c.end_turn("s1")
        # Generation window = 130 - 114 = 16s
        assert summary["gen_ms"] == pytest.approx(16000.0)

    def test_output_tokens_only_counted_for_window_calls(self):
        # Calls without a delta (tool turns) contribute no output_tokens,
        # avoiding TPS dilution
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        c.on_api_done("s1", 1, 131.0, 150.0, 500)  # tool turn, no delta
        summary = c.end_turn("s1")
        assert summary["calls"] == 2
        assert summary["ttft_calls"] == 1
        assert summary["output_tokens"] == 300  # only windowed calls count

    def test_batch_provider_degrades_gen_to_api_duration(self):
        # Batch return (provider buffers, first chunk ≈ last chunk): gen window
        # ≈ 0, degrade to the total API duration to avoid inflating TPS.
        # started=100, first=129.9, ended=130 -> gen=0.1s < 30% * 30s
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 129.9)
        c.on_api_done("s1", 0, 100.0, 130.0, 400)
        summary = c.end_turn("s1")
        assert summary is not None
        assert summary["ttft_ms"] == pytest.approx((129.9 - 100.0) * 1000)
        # gen degrades to the 30s total API duration
        assert summary["gen_ms"] == pytest.approx(30000.0)
        # TPS = 400 / 30 = 13.3 tok/s (end-to-end throughput)
        assert summary["output_tokens"] / (summary["gen_ms"] / 1000) == pytest.approx(400 / 30)

    def test_streaming_provider_keeps_gen_window(self):
        # True streaming: first chunk is early, generation window dominates,
        # so it does not degrade to the total API duration
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        summary = c.end_turn("s1")
        assert summary is not None
        assert summary["gen_ms"] == pytest.approx(16000.0)  # 16s window, no degrade


class TestRequestSentBaseline:
    def test_ttft_uses_request_sent_at_when_available(self):
        # request_sent_at (HTTP sent time) is later than started_at: TTFT uses
        # it to exclude agent-side request preparation
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0, request_sent_at=110.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        summary = c.end_turn("s1")
        assert summary is not None
        assert summary["ttft_ms"] == pytest.approx((114.0 - 110.0) * 1000)  # 4000ms

    def test_falls_back_to_started_at_without_request_sent(self):
        # Backends without request_sent_at fall back to the started_at
        # baseline (compatibility)
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)  # no request_sent_at passed
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        summary = c.end_turn("s1")
        assert summary is not None
        assert summary["ttft_ms"] == pytest.approx((114.0 - 100.0) * 1000)  # 14000ms

    def test_batch_degrades_gen_uses_request_sent_at(self):
        # Batch degrade: gen uses ended - request_sent_at (real HTTP duration)
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 129.9, request_sent_at=101.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 400)
        summary = c.end_turn("s1")
        assert summary is not None
        assert summary["gen_ms"] == pytest.approx((130.0 - 101.0) * 1000)  # 29s

    def test_reasoning_model_ttft_uses_first_chunk(self):
        # Reasoning model: the first text delta (1.2s) lands well after the
        # first chunk (0.4s); TTFT should use the earlier first packet ("the
        # model's first token back")
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 121.0, request_sent_at=100.0, first_chunk_at=104.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        summary = c.end_turn("s1")
        assert summary is not None
        assert summary["ttft_ms"] == pytest.approx((104.0 - 100.0) * 1000)  # 4s (first chunk), not 21s
        # gen also runs from the first chunk to the end: 130 - 104 = 26s
        # (within the batch-degrade threshold, unchanged)


class TestRealtimeUpdate:
    def test_on_update_fires_increment_after_api_done(self):
        updates = []
        c = StreamPerfCollector(on_update=lambda sid, perf: updates.append((sid, dict(perf))))
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        assert len(updates) == 1
        sid, perf = updates[0]
        assert sid == "s1"
        assert perf["calls"] == 1
        assert perf["ttft_calls"] == 1
        assert perf["ttft_ms"] == pytest.approx(14000.0)
        assert perf["gen_ms"] == pytest.approx(16000.0)
        assert perf["output_tokens"] == 300

    def test_on_update_not_fired_for_tool_only_call(self):
        updates = []
        c = StreamPerfCollector(on_update=lambda sid, perf: updates.append(perf))
        c.begin_turn("s1")
        c.on_api_done("s1", 0, 100.0, 130.0, 50)  # no delta (tool turn)
        assert updates == []

    def test_on_update_fires_per_call(self):
        updates = []
        c = StreamPerfCollector(on_update=lambda sid, perf: updates.append(dict(perf)))
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        c.on_first_delta("s1", 1, 133.0)
        c.on_api_done("s1", 1, 131.0, 145.0, 400)
        assert len(updates) == 2
        # Incremental semantics: each call pushes independently
        assert updates[0]["ttft_ms"] == pytest.approx(14000.0)
        assert updates[1]["ttft_ms"] == pytest.approx(2000.0)

    def test_set_on_update_replaces_callback(self):
        first = []
        second = []
        c = StreamPerfCollector(on_update=lambda sid, perf: first.append(perf))
        c.set_on_update(lambda sid, perf: second.append(perf))
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        assert first == []
        assert len(second) == 1


class TestSessionIsolation:
    def test_sessions_do_not_cross_talk(self):
        c = _collector()
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 114.0)
        c.on_api_done("s1", 0, 100.0, 130.0, 300)
        # s2 has its own independent pending state
        c.begin_turn("s2")
        c.on_first_delta("s2", 0, 8.0)
        c.on_api_done("s2", 0, 6.0, 20.0, 100)
        s1 = c.end_turn("s1")
        s2 = c.end_turn("s2")
        assert s1["ttft_ms"] == pytest.approx(14000.0)
        assert s2["ttft_ms"] == pytest.approx(2000.0)
        assert s1["output_tokens"] == 300
        assert s2["output_tokens"] == 100

    def test_api_done_without_first_delta_does_not_leak_pending(self):
        c = _collector()
        c.begin_turn("s1")
        c.on_api_done("s1", 3, 100.0, 120.0, 50)
        summary = c.end_turn("s1")
        assert summary is not None
        # The next turn should not carry leftover pending state
        c.begin_turn("s1")
        c.on_first_delta("s1", 0, 10.0)
        c.on_api_done("s1", 0, 8.0, 30.0, 200)
        s = c.end_turn("s1")
        assert s["ttft_ms"] == pytest.approx(2000.0)
