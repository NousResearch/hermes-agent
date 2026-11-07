"""The status bar's context readout across a model switch in the embedded TUI (``tui_gateway``).

``ContextCompressor.update_model`` adopts the new model's window and zeroes the previous model's
occupancy ("old usage cannot price a new model"), so a readout taken between the switch and the first
response of the new model knows the window but has no occupancy to report. The window is sent and the
occupancy figures are left out (``agent.context_breakdown.context_usage_fields``) rather than filled
in with zeros, which would paint an empty window that nothing measured; the TUI renders the absent
reading as ``—`` and drops the old model's figures because the frame states a new window
(``dropStaleContext``).
"""
from types import SimpleNamespace

from tui_gateway import server


def _agent(window: int, *, model: str = 'switched', used: int = 0):
    """The agent right after a switch: the new window, nothing counted in it yet."""
    return SimpleNamespace(model=model, provider='custom', session_id='sid', session_api_calls=0,
                           session_input_tokens=0, session_output_tokens=0, session_prompt_tokens=0,
                           session_completion_tokens=0, session_total_tokens=0,
                           session_reasoning_tokens=0, session_cache_read_tokens=0,
                           context_compressor=SimpleNamespace(context_length=window,
                                                              last_prompt_tokens=used,
                                                              compression_count=0))


def test_the_ticker_reports_the_new_window_with_nothing_counted():
    usage = server._get_usage(_agent(200_000))
    assert usage['context_max'] == 200_000
    assert 'context_used' not in usage and 'context_percent' not in usage
    assert usage['compressions'] == 0 and usage['model'] == 'switched'


def test_the_occupancy_returns_once_the_new_model_has_priced_a_prompt():
    usage = server._get_usage(_agent(200_000, used=50_000))
    assert (usage['context_used'], usage['context_max'], usage['context_percent']) == (50_000, 200_000, 25)


def test_the_session_info_snapshot_carries_the_new_window_to_the_status_bar():
    """``_emit_session_info`` after a committed switch (``model_switch._commit_agent_switch``) sends
    ``_session_info``, whose ``usage`` is this snapshot."""
    usage = server._session_usage_snapshot({'agent': _agent(200_000)})
    assert usage['context_max'] == 200_000
    assert 'context_used' not in usage and 'context_percent' not in usage


def test_a_window_the_route_does_not_report_keeps_the_context_block_empty():
    """No window known (a pick whose route reports none): nothing is fabricated, and the agent's own
    compressor read is still the only source -- the value never falls back to a previous model's."""
    usage = server._get_usage(_agent(0))
    assert 'context_max' not in usage and 'context_percent' not in usage
