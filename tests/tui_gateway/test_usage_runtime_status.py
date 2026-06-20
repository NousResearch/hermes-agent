from types import SimpleNamespace


def test_get_usage_includes_runtime_status_snapshot():
    from agent import runtime_status
    from tui_gateway.server import _get_usage

    runtime_status.clear_session("s-tui-runtime")
    runtime_status.record_phase("s-tui-runtime", run_mode="implement", target="tui", main_agent="executor")
    runtime_status.record_tool_completed("s-tui-runtime", "terminal", status="ok", duration_ms=17)
    runtime_status.record_skill("s-tui-runtime", "hermes-agent", event="view")
    runtime_status.record_task_summary("s-tui-runtime", {"total": 4, "completed": 2})

    agent = SimpleNamespace(
        session_id="s-tui-runtime",
        model="test/model",
        session_input_tokens=10,
        session_output_tokens=5,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=10,
        session_completion_tokens=5,
        session_total_tokens=15,
        session_api_calls=1,
        provider="test",
        base_url="",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=50,
            context_length=100,
            compression_count=1,
        ),
    )

    usage = _get_usage(agent)

    assert usage["runtime"]["run_mode"] == "implement"
    assert usage["runtime"]["target"] == "tui"
    assert usage["runtime"]["main_agent"] == "executor"
    assert usage["runtime"]["recent_tool"]["name"] == "terminal"
    assert usage["runtime"]["recent_tool"]["status"] == "ok"
    assert usage["runtime"]["recent_skill"]["name"] == "hermes-agent"
    assert usage["runtime"]["task"] == {
        "total": 4,
        "completed": 2,
        "in_progress": 0,
        "pending": 0,
        "cancelled": 0,
    }


def test_get_usage_records_background_process_count(monkeypatch):
    from agent import runtime_status
    from tui_gateway.server import _get_usage
    from tools.process_registry import process_registry

    runtime_status.clear_session("s-tui-bg")
    monkeypatch.setattr(process_registry, "count_running", lambda: 2)
    agent = SimpleNamespace(
        session_id="s-tui-bg",
        model="test/model",
        session_input_tokens=10,
        session_output_tokens=5,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=10,
        session_completion_tokens=5,
        session_total_tokens=15,
        session_api_calls=1,
        provider="test",
        base_url="",
        context_compressor=None,
    )

    usage = _get_usage(agent)

    assert usage["runtime"]["background_tasks"] == {"running": 2}


def test_get_usage_updates_runtime_from_agent_activity_summary():
    from agent import runtime_status
    from tui_gateway.server import _get_usage

    runtime_status.clear_session("s-tui-activity")
    agent = SimpleNamespace(
        session_id="s-tui-activity",
        model="test/model",
        session_input_tokens=10,
        session_output_tokens=5,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=10,
        session_completion_tokens=5,
        session_total_tokens=15,
        session_api_calls=1,
        provider="test",
        base_url="",
        context_compressor=None,
        get_activity_summary=lambda: {
            "current_tool": "terminal",
            "last_activity_desc": "running terminal",
            "budget_used": 3,
            "budget_max": 8,
        },
    )

    usage = _get_usage(agent)

    assert usage["runtime"]["phase"] == "tool"
    assert usage["runtime"]["run_mode"] == "agent"
    assert usage["runtime"]["wait"]["reason"] == "tool:terminal"
    assert usage["runtime"]["task"]["completed"] == 3
