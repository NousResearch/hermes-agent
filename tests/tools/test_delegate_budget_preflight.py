import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task


def _parent(*, toolsets=None, max_iterations=30, api_calls=0):
    return SimpleNamespace(
        base_url="https://example.test/v1",
        api_key="test-key",
        provider="openrouter",
        api_mode="chat_completions",
        model="test/model",
        platform="cli",
        enabled_toolsets=list(toolsets) if toolsets is not None else ["terminal", "file", "web"],
        valid_tool_names=set(),
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
        session_id="parent-session",
        max_iterations=max_iterations,
        _api_call_count=api_calls,
    )


def _completed_child(session_id="child-session"):
    child = MagicMock()
    child.session_id = session_id
    child._delegate_role = "leaf"
    return child


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 0})
@patch("tools.delegate_tool._build_child_agent")
def test_delegate_task_refuses_zero_effective_child_iterations_before_spawn(
    mock_build_child,
    _mock_config,
):
    result = json.loads(
        delegate_task(
            goal="Review correctness of the final diff before merge.",
            parent_agent=_parent(toolsets=["terminal", "file"]),
        )
    )

    mock_build_child.assert_not_called()
    assert result["error"] == "Delegation preflight failed"
    entry = result["results"][0]
    assert entry["status"] == "error"
    assert entry["review_evidence_status"] == "blocked_zero_budget"
    assert entry["preflight"]["effective_max_iterations"] == 0
    assert entry["preflight"]["expected_api_call_allowance"] == 0
    assert entry["child_session_id"] is None


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 8})
@patch("tools.delegate_tool._build_child_agent")
def test_delegate_task_refuses_review_when_parent_has_no_closeout_budget(
    mock_build_child,
    _mock_config,
):
    parent = _parent(toolsets=["terminal", "file"], max_iterations=6, api_calls=6)

    result = json.loads(
        delegate_task(
            goal="Review safety and correctness of the paid-live handoff.",
            parent_agent=parent,
        )
    )

    mock_build_child.assert_not_called()
    entry = result["results"][0]
    assert entry["review_evidence_status"] == "blocked_zero_budget"
    assert entry["preflight"]["parent_remaining_api_calls"] == 0
    assert entry["preflight"]["parent_closeout_reserve"] == 1
    assert entry["preflight"]["expected_api_call_allowance"] == 0


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 8})
@patch("tools.delegate_tool._build_child_agent")
def test_review_task_with_no_usable_tools_is_blocked_before_spawn(
    mock_build_child,
    _mock_config,
):
    result = json.loads(
        delegate_task(
            goal="Review correctness of the repository changes.",
            parent_agent=_parent(toolsets=[]),
        )
    )

    mock_build_child.assert_not_called()
    entry = result["results"][0]
    assert entry["review_evidence_status"] == "blocked_no_tools"
    assert entry["preflight"]["available_toolset_count"] == 0


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 8})
@patch("tools.delegate_tool._build_child_agent")
def test_review_task_without_repo_or_diff_access_is_blocked_before_spawn(
    mock_build_child,
    _mock_config,
):
    result = json.loads(
        delegate_task(
            goal="Give a verdict on safety and correctness of the diff.",
            parent_agent=_parent(toolsets=["web"]),
        )
    )

    mock_build_child.assert_not_called()
    entry = result["results"][0]
    assert entry["review_evidence_status"] == "blocked_no_repo_access"
    assert entry["preflight"]["available_toolset_count"] == 1
    assert set(entry["preflight"]["blocked_or_disabled_required_toolsets"]) == {
        "file",
        "terminal",
    }


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 8})
@patch("tools.delegate_tool._build_child_agent", return_value=_completed_child("child-text-only"))
@patch("tools.delegate_tool._run_single_child")
def test_explicit_text_only_review_can_run_but_is_not_counted_as_review_evidence(
    mock_run_child,
    mock_build_child,
    _mock_config,
):
    mock_run_child.return_value = {
        "task_index": 0,
        "status": "completed",
        "summary": "Text-only opinion.",
        "api_calls": 1,
        "duration_seconds": 0.1,
    }

    result = json.loads(
        delegate_task(
            goal="Text-only review from the prompt only; do not inspect repo files.",
            parent_agent=_parent(toolsets=[]),
        )
    )

    mock_build_child.assert_called_once()
    entry = result["results"][0]
    assert entry["review_evidence_status"] == "text_only_not_counted"
    assert entry["child_session_id"] == "child-text-only"
    assert result["review_evidence_summary"]["valid_count"] == 0
    assert result["review_evidence_summary"]["invalid_children"][0]["review_evidence_status"] == "text_only_not_counted"


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 8})
@patch("tools.delegate_tool._build_child_agent", return_value=_completed_child("child-valid"))
@patch("tools.delegate_tool._run_single_child")
def test_valid_bounded_review_with_repo_tools_and_budget_counts_as_review_evidence(
    mock_run_child,
    mock_build_child,
    _mock_config,
):
    mock_run_child.return_value = {
        "task_index": 0,
        "status": "completed",
        "summary": "Reviewed bounded diff.",
        "api_calls": 2,
        "duration_seconds": 0.2,
    }

    result = json.loads(
        delegate_task(
            goal="Review correctness of files src/a.py and tests/test_a.py only.",
            parent_agent=_parent(toolsets=["terminal", "file"]),
        )
    )

    mock_build_child.assert_called_once()
    entry = result["results"][0]
    assert entry["review_evidence_status"] == "valid_review"
    assert entry["child_session_id"] == "child-valid"
    assert result["review_evidence_summary"]["valid_count"] == 1
    assert result["review_evidence_summary"]["invalid_children"] == []
