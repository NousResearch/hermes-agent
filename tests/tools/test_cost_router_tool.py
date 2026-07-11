import json
import logging
import subprocess
from types import SimpleNamespace

from tools import cost_router_tool
from tools.cost_router_tool import (
    _build_acceptance_check,
    _normalise_route,
    _parse_worker_result,
    _profile_env_overrides,
    cost_router,
)
from tools.project_intake import build_project_card
from model_tools import get_tool_definitions
from toolsets import resolve_toolset


def test_cost_router_registered_in_default_tool_definitions():
    names = {t["function"]["name"] for t in get_tool_definitions(quiet_mode=True)}
    assert "cost_router" in names


def test_cost_router_in_delegation_toolset():
    assert "cost_router" in resolve_toolset("delegation")


def test_cost_router_luna_terra_sol_routing():
    assert _normalise_route("luna", "x", None) == ("luna", "worker-luna")
    assert _normalise_route("luna_economy", "x", None) == ("luna_economy", "worker-luna-economy")
    assert _normalise_route("worker-luna-economy", "x", None) == ("luna_economy", "worker-luna-economy")
    assert _normalise_route("worker-terra", "x", None) == ("terra", "worker-terra")
    assert _normalise_route("sol", "x", None) == ("sol", "worker-sol")
    assert _normalise_route(None, "classify URLs", None) == ("luna", "worker-luna")
    assert _normalise_route(None, "bulk classify URLs and dedupe JSON logs", None) == ("luna_economy", "worker-luna-economy")
    assert _normalise_route(None, "write a film article draft", None) == ("terra", "worker-terra")
    assert _normalise_route(None, "final review architecture conflict", None) == ("sol", "worker-sol")


def test_cost_router_selection_precedence_prefers_explicit_route_over_task_type_and_keywords():
    decision = cost_router_tool._select_route(
        "sol",
        "final architecture audit",
        "bulk JSON dedupe",
        task_type="classify",
        project={"route_candidate": "luna_economy", "id": "project-1", "slice_id": "slice-a"},
    )

    assert decision["tier"] == "sol"
    assert decision["profile"] == "worker-sol"
    assert decision["selection_mode"] == "explicit"
    assert decision["matched_rule"] == "route:sol"


def test_cost_router_selection_uses_project_route_before_task_type_and_keywords():
    decision = cost_router_tool._select_route(
        None,
        "final architecture audit",
        "",
        task_type="classify",
        project={"route_candidate": "luna_economy"},
    )

    assert decision["tier"] == "luna_economy"
    assert decision["selection_mode"] == "explicit"
    assert decision["matched_rule"] == "project.route_candidate:luna_economy"


def test_cost_router_selection_uses_task_type_before_incidental_keywords():
    decision = cost_router_tool._select_route(
        None,
        "final architecture audit",
        "",
        task_type="classify",
    )

    assert decision["tier"] == "luna"
    assert decision["profile"] == "worker-luna"
    assert decision["selection_mode"] == "task_type"
    assert decision["matched_rule"] == "task_type:classify"


def test_cost_router_selection_defaults_to_terra_when_no_rule_matches():
    decision = cost_router_tool._select_route(None, "Handle this bounded request.", None)

    assert decision["tier"] == "terra"
    assert decision["profile"] == "worker-terra"
    assert decision["selection_mode"] == "default"
    assert decision["matched_rule"] == "default:terra"


def test_cost_router_rejects_legacy_routes():
    for legacy in ["dsflash", "dspro", "gpt54", "gpt55", "worker-gpt55"]:
        try:
            _normalise_route(legacy, "x", None)
        except ValueError as exc:
            assert "luna" in str(exc) and "terra" in str(exc) and "sol" in str(exc)
        else:
            raise AssertionError(f"legacy route {legacy!r} unexpectedly accepted")


def test_cost_router_requires_goal():
    payload = json.loads(cost_router(goal=""))
    assert "error" in payload


def test_cost_router_logs_structured_result(monkeypatch, caplog):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout="session_id: worker-session-1\nworker output",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)

    with caplog.at_level(logging.INFO, logger="tools.cost_router_tool"):
        payload = json.loads(cost_router(goal="classify fields", route="luna", timeout=5))

    assert payload["tier"] == "luna"
    assert payload["route"] == "luna"
    assert payload["profile"] == "worker-luna"
    assert payload["selection_mode"] == "explicit"
    assert payload["matched_rule"] == "route:luna"
    assert payload["session_id"] == "worker-session-1"
    assert payload["output"] == "worker output"
    assert "cost_router.result tier=luna route=luna profile=worker-luna" in caplog.text
    assert "worker_session=worker-session-1" in caplog.text
    assert "exit_code=0" in caplog.text


def test_parse_worker_result_preserves_prose_and_normalizes_valid_footer():
    output = "Draft body in ordinary prose.\n\n<cost_router_result>\n" + json.dumps({
        "status": "complete",
        "deliverable": "Draft body above",
        "evidence": ["/tmp/draft.md", {"url": "https://example.test/report"}],
        "unverified_items": [],
        "controller_decisions": ["Decide whether to publish"],
    }) + "\n</cost_router_result>"

    prose, worker_result = _parse_worker_result(output, exit_code=0)

    assert prose == "Draft body in ordinary prose."
    assert worker_result == {
        "status": "complete",
        "deliverable": "Draft body above",
        "evidence": ["/tmp/draft.md", {"url": "https://example.test/report"}],
        "unverified_items": [],
        "controller_decisions": ["Decide whether to publish"],
        "contract_status": "valid",
        "controller_acceptance_required": True,
    }


def test_parse_worker_result_keeps_legacy_free_text_as_unverified_partial():
    prose, worker_result = _parse_worker_result("A useful free-text answer.", exit_code=0)

    assert prose == "A useful free-text answer."
    assert worker_result["status"] == "partial"
    assert worker_result["deliverable"] == "A useful free-text answer."
    assert worker_result["contract_status"] == "absent"
    assert worker_result["evidence"] == []
    assert worker_result["unverified_items"]
    assert worker_result["controller_decisions"]
    assert worker_result["controller_acceptance_required"] is True


def test_parse_worker_result_treats_malformed_footer_as_unverified_partial():
    output = "Prose survives.\n<cost_router_result>\n{not json}\n</cost_router_result>"

    prose, worker_result = _parse_worker_result(output, exit_code=0)

    assert prose == "Prose survives."
    assert worker_result["status"] == "partial"
    assert worker_result["contract_status"] == "malformed"
    assert worker_result["unverified_items"] == ["Structured worker-result footer was malformed."]


def test_parse_worker_result_preserves_valid_partial_and_blocked_statuses():
    for status in ("partial", "blocked"):
        footer = json.dumps({
            "status": status,
            "deliverable": None,
            "evidence": [],
            "unverified_items": ["verification remains"],
            "controller_decisions": ["Choose next step"],
        })
        prose, worker_result = _parse_worker_result(
            f"Worker notes.\n<cost_router_result>{footer}</cost_router_result>",
            exit_code=0,
        )

        assert prose == "Worker notes."
        assert worker_result["status"] == status
        assert worker_result["contract_status"] == "valid"


def test_parse_worker_result_nonzero_exit_cannot_report_complete():
    output = "Done.\n<cost_router_result>\n" + json.dumps({
        "status": "complete",
        "deliverable": "Done",
        "evidence": ["test output"],
        "unverified_items": [],
        "controller_decisions": [],
    }) + "\n</cost_router_result>"

    _, worker_result = _parse_worker_result(output, exit_code=2)

    assert worker_result["status"] == "blocked"
    assert "non-zero" in worker_result["unverified_items"][-1]
    assert worker_result["controller_acceptance_required"] is True


def test_parse_worker_result_absent_footer_with_nonzero_exit_is_blocked():
    prose, worker_result = _parse_worker_result("Legacy output.", exit_code=7)

    assert prose == "Legacy output."
    assert worker_result["status"] == "blocked"
    assert worker_result["contract_status"] == "absent"
    assert "non-zero" in worker_result["unverified_items"][-1]


def test_parse_worker_result_malformed_footer_with_nonzero_exit_is_blocked():
    output = "Useful prose.\n<cost_router_result>{bad json}</cost_router_result>"

    prose, worker_result = _parse_worker_result(output, exit_code=9)

    assert prose == "Useful prose."
    assert worker_result["status"] == "blocked"
    assert worker_result["contract_status"] == "malformed"
    assert "non-zero" in worker_result["unverified_items"][-1]


def test_parse_worker_result_truncated_footer_is_malformed_and_removed_from_prose():
    output = 'Useful prose.\n<cost_router_result>{"status":"partial"'

    prose, worker_result = _parse_worker_result(output, exit_code=0)

    assert prose == "Useful prose."
    assert worker_result["deliverable"] == "Useful prose."
    assert worker_result["status"] == "partial"
    assert worker_result["contract_status"] == "malformed"


def test_parse_worker_result_rejects_missing_or_wrong_typed_required_fields():
    valid = {
        "status": "complete",
        "deliverable": "result",
        "evidence": [],
        "unverified_items": [],
        "controller_decisions": [],
    }
    invalid_footers = []
    for field in valid:
        invalid = dict(valid)
        del invalid[field]
        invalid_footers.append(invalid)
    for field, value in (
        ("status", 1),
        ("status", []),
        ("deliverable", []),
        ("evidence", "proof"),
        ("evidence", [1]),
        ("unverified_items", "none"),
        ("unverified_items", [1]),
        ("controller_decisions", "decide"),
        ("controller_decisions", [1]),
    ):
        invalid = dict(valid)
        invalid[field] = value
        invalid_footers.append(invalid)

    for footer in invalid_footers:
        prose, worker_result = _parse_worker_result(
            f"Prose.\n<cost_router_result>{json.dumps(footer)}</cost_router_result>",
            exit_code=0,
        )

        assert prose == "Prose."
        assert worker_result["status"] == "partial"
        assert worker_result["contract_status"] == "malformed"


def test_cost_router_exit_zero_is_execution_success_not_controller_acceptance(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    footer = json.dumps({
        "status": "complete",
        "deliverable": "Classification table",
        "evidence": ["rows=3"],
        "unverified_items": [],
        "controller_decisions": [],
    })
    monkeypatch.setattr(
        cost_router_tool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"table prose\n<cost_router_result>\n{footer}\n</cost_router_result>",
            stderr="",
            returncode=0,
        ),
    )

    payload = json.loads(cost_router(goal="Classify these fields.", route="luna", timeout=5))

    assert payload["exit_code"] == 0
    assert payload["execution_status"] == "succeeded"
    assert payload["output"] == "table prose"
    assert payload["worker_result"]["status"] == "complete"
    assert payload["worker_result"]["controller_acceptance_required"] is True
    assert payload["acceptance_check"]["accepted"] is False
    assert payload["acceptance_check"]["controller_decision_required"] is True
    assert "worker_exit_success" in payload["acceptance_check"]["blocking_reasons"]
    assert "accepted" not in payload


def test_acceptance_check_exposes_all_controller_review_dimensions():
    project_card = {
        "acceptance_evidence": ["A tested artifact covers the requested behavior."],
    }
    worker_result = {
        "status": "complete",
        "deliverable": "/tmp/result.md",
        "evidence": ["pytest: 3 passed"],
        "unverified_items": ["One source remains unchecked"],
        "controller_decisions": ["Resolve conflict between slice A and B"],
        "contract_status": "valid",
        "controller_acceptance_required": True,
    }

    check = _build_acceptance_check(project_card, worker_result, exit_code=0)

    assert check["accepted"] is False
    assert check["controller_decision_required"] is True
    assert set(check["checklist"]) == {
        "deliverable_coverage",
        "evidence_sufficiency",
        "unsupported_claims",
        "cross_slice_conflicts",
        "verifiable_artifacts",
        "unresolved_items",
        "sol_review_warranted",
    }
    assert check["checklist"]["deliverable_coverage"]["status"] == "review"
    assert check["checklist"]["evidence_sufficiency"]["status"] == "review"
    assert check["checklist"]["unsupported_claims"]["status"] == "review"
    assert check["checklist"]["cross_slice_conflicts"]["status"] == "review"
    assert check["checklist"]["verifiable_artifacts"]["status"] == "review"
    assert check["checklist"]["unresolved_items"]["status"] == "attention"
    assert check["checklist"]["sol_review_warranted"]["status"] == "consider"
    assert check["sol_review"]["recommended"] is True
    assert check["sol_review"]["automatic"] is False


def test_acceptance_check_does_not_recommend_sol_for_ordinary_complete_work():
    check = _build_acceptance_check(
        {"acceptance_evidence": ["Rows cover every input."]},
        {
            "status": "complete",
            "deliverable": "Classification table",
            "evidence": ["3 inputs and 3 output rows"],
            "unverified_items": [],
            "controller_decisions": [],
            "contract_status": "valid",
            "controller_acceptance_required": True,
        },
        exit_code=0,
    )

    assert check["accepted"] is False
    assert check["sol_review"] == {
        "recommended": False,
        "automatic": False,
        "reasons": [],
    }
    assert check["checklist"]["sol_review_warranted"]["status"] == "not_warranted"


def test_cost_router_returns_safe_project_and_parent_session_telemetry(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    monkeypatch.setattr(
        cost_router_tool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="worker output", stderr="", returncode=0),
    )

    payload = json.loads(cost_router(
        goal="Handle this bounded request.",
        project={"id": "project-1", "slice_id": "slice-a", "route_candidate": "luna"},
        parent_session_id="controller-session-1",
        timeout=5,
    ))

    assert payload["selection_mode"] == "explicit"
    assert payload["matched_rule"] == "project.route_candidate:luna"
    assert payload["decision_metadata"] == {
        "project_id": "project-1",
        "slice_id": "slice-a",
        "parent_session_id": "controller-session-1",
    }


def test_project_card_keeps_simple_one_shot_request_single_with_reason():
    card = build_project_card("Classify these URLs by topic.", task_type="classify")

    assert card["user_goal"] == "Classify these URLs by topic."
    assert card["user_visible_deliverable"] == "Topic classifications for the provided URLs."
    assert card["dependencies"] == []
    assert card["acceptance_evidence"] == [
        "Classifications are provided for each URL and identify its topic."
    ]
    assert card["route_candidate"] == "luna"
    assert card["split_required"] is False
    assert card["no_split_reason"]
    assert card["split_triggers"] == []


def test_project_card_requires_split_for_each_mandatory_trigger():
    cases = [
        ("Research two sources and write a briefing.", "multi_source"),
        ("Research the topic and write a launch article.", "research_plus_writing"),
        ("Implement the endpoint and verify it with tests.", "implementation_plus_verification"),
        ("Choose the production migration strategy.", "high_impact_decision"),
        ("Create a report and a slide deck.", "multi_deliverable"),
    ]

    for goal, trigger in cases:
        card = build_project_card(goal)
        assert card["split_required"] is True
        assert trigger in card["split_triggers"]
        assert card["no_split_reason"] is None


def test_project_card_requires_split_for_each_chinese_mandatory_trigger():
    cases = [
        ("请分别产出市场报告和演示文稿。", "multi_deliverable"),
        ("对比多个来源的网站和论文，给出结论。", "multi_source"),
        ("先调研目标市场，再写一篇发布文章。", "research_plus_writing"),
        ("实现支付接口并运行测试验证。", "implementation_plus_verification"),
        ("请决定生产环境迁移的安全架构方案。", "high_impact_decision"),
    ]

    for goal, trigger in cases:
        card = build_project_card(goal)
        assert card["split_required"] is True
        assert trigger in card["split_triggers"]
        assert card["no_split_reason"] is None


def test_project_card_detects_bounded_multi_source_research_and_writing_regression():
    card = build_project_card("整理多个来源并写文章。")

    assert card["split_required"] is True
    assert card["split_triggers"] == ["multi_source", "research_plus_writing"]
    assert card["no_split_reason"] is None


def test_project_card_keeps_bounded_single_source_and_single_slice_requests_one_shot():
    for goal in [
        "整理这份会议记录。",
        "写一篇文章。",
        "Research the topic.",
        "Implement the endpoint.",
        "Explain the security architecture.",
    ]:
        card = build_project_card(goal)
        assert card["split_required"] is False, goal
        assert card["split_triggers"] == [], goal
        assert card["no_split_reason"], goal


def test_project_card_can_gate_a_compound_requirement_from_context():
    card = build_project_card("Prepare the release materials.", context="Research the topic and write a launch article.")

    assert card["split_required"] is True
    assert "research_plus_writing" in card["split_triggers"]


def test_cost_router_returns_project_card_without_routing_when_split_is_required(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")

    def fail_if_routed(*args, **kwargs):
        raise AssertionError("split-required work must not reach a worker route")

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fail_if_routed)

    payload = json.loads(cost_router(goal="Implement the endpoint and verify it with tests.", timeout=5))

    assert payload["routing_status"] == "split_required"
    assert payload["project_card"]["split_required"] is True
    assert "implementation_plus_verification" in payload["project_card"]["split_triggers"]


def test_cost_router_makes_intake_decision_before_route_selection(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    events = []
    real_build_project_card = cost_router_tool.build_project_card

    def observe_intake(*args, **kwargs):
        events.append("intake")
        return real_build_project_card(*args, **kwargs)

    def observe_route(*args, **kwargs):
        events.append("route")
        return {
            "tier": "luna",
            "profile": "worker-luna",
            "selection_mode": "keyword",
            "matched_rule": "keyword:classify",
        }

    monkeypatch.setattr(cost_router_tool, "build_project_card", observe_intake)
    monkeypatch.setattr(cost_router_tool, "_select_route", observe_route)
    monkeypatch.setattr(
        cost_router_tool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="worker output", stderr="", returncode=0),
    )

    payload = json.loads(cost_router(goal="Classify these URLs by topic.", timeout=5))

    assert payload["project_card"]["split_required"] is False
    assert events == ["intake", "route"]


def test_cost_router_keeps_legacy_one_shot_response_and_adds_card(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="session_id: worker-session-2\nworker output", stderr="", returncode=0)

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload = json.loads(cost_router(goal="Classify these URLs by topic.", route="luna", timeout=5))

    assert len(calls) == 1
    assert {"tier", "route", "profile", "session_id", "exit_code", "output", "project_card"} <= payload.keys()
    assert payload["tier"] == payload["route"] == payload["project_card"]["route_candidate"] == "luna"
    assert payload["project_card"]["split_required"] is False
    assert payload["project_card"]["no_split_reason"]


def test_cost_router_profile_env_overrides_preserve_deepseek_key(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    profile_dir = hermes_home / "profiles" / "worker-luna-economy"
    profile_dir.mkdir(parents=True)
    key = "sk-4c0268038ea44861aaaf2684c9873da4"
    (profile_dir / "config.yaml").write_text(
        "model:\n"
        "  provider: deepseek\n"
        "  default: deepseek-v4-flash\n"
        f"  api_key: {key}\n"
        "providers:\n"
        "  deepseek:\n"
        f"    api_key: {key}\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = _profile_env_overrides("worker-luna-economy")

    assert env["DEEPSEEK_API_KEY"] == key
    assert len(env["DEEPSEEK_API_KEY"]) == len(key)


def _completed(returncode, stdout="", stderr=""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_cost_router_falls_back_once_from_luna_to_terra_on_503(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return _completed(1, stdout="partial Luna result", stderr="provider returned HTTP 503")
        return _completed(0, stdout="session_id: terra-session\nTerra result")

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload = json.loads(cost_router(goal="Classify fields.", route="luna", allow_fallback=True, timeout=5))

    assert [call[call.index("--profile") + 1] for call in calls] == ["worker-luna", "worker-terra"]
    assert payload["routing_status"] == "completed"
    assert payload["tier"] == "terra"
    assert payload["fallback_from"] == "luna"
    assert payload["fallback_reason"] == "http_503"
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["output"] == "partial Luna result"
    assert payload["attempts"][0]["retryable"] is True
    assert payload["output"] == "Terra result"


def test_cost_router_falls_back_from_luna_economy_on_429(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    results = iter([
        _completed(1, stderr="429 Too Many Requests"),
        _completed(0, stdout="Terra result"),
    ])
    monkeypatch.setattr(cost_router_tool.subprocess, "run", lambda *args, **kwargs: next(results))

    payload = json.loads(cost_router(goal="Dedupe records.", route="luna_economy", allow_fallback=True, timeout=5))

    assert payload["fallback_from"] == "luna_economy"
    assert payload["fallback_reason"] == "http_429"
    assert [attempt["tier"] for attempt in payload["attempts"]] == ["luna_economy", "terra"]


def test_cost_router_project_policy_can_allow_fallback(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    results = iter([
        _completed(1, stderr="HTTP 503"),
        _completed(0, stdout="Terra result"),
    ])
    monkeypatch.setattr(cost_router_tool.subprocess, "run", lambda *args, **kwargs: next(results))

    payload = json.loads(cost_router(
        goal="Classify records.",
        route="luna",
        project={"allow_fallback": True},
        timeout=5,
    ))

    assert payload["fallback_from"] == "luna"
    assert len(payload["attempts"]) == 2


def test_cost_router_timeout_is_retryable_without_fallback(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output="partial", stderr="timeout detail")

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload = json.loads(cost_router(goal="Classify fields.", route="luna", allow_fallback=False, timeout=5))

    assert len(calls) == 1
    assert payload["routing_status"] == "retryable"
    assert payload["fallback_from"] is None
    assert payload["fallback_reason"] is None
    assert payload["attempts"][0]["failure_reason"] == "timeout"
    assert payload["attempts"][0]["output"] == "partial"


def test_cost_router_non_retryable_failure_never_falls_back(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed(1, stdout="worker draft", stderr="content policy rejected response")

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload = json.loads(cost_router(goal="Classify fields.", route="luna", allow_fallback=True, timeout=5))

    assert len(calls) == 1
    assert payload["routing_status"] == "failed"
    assert payload["attempts"][0]["retryable"] is False
    assert payload["attempts"][0]["output"] == "worker draft"
    assert payload["fallback_from"] is None


def test_cost_router_retryable_connection_failure_respects_disabled_fallback(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    monkeypatch.setattr(
        cost_router_tool.subprocess,
        "run",
        lambda *args, **kwargs: _completed(1, stderr="ConnectionError: connection refused"),
    )

    payload = json.loads(cost_router(goal="Classify fields.", route="luna", allow_fallback=False, timeout=5))

    assert payload["routing_status"] == "retryable"
    assert payload["attempts"][0]["failure_reason"] == "connection_failure"
    assert len(payload["attempts"]) == 1


def test_cost_router_terra_fallback_failure_stops_after_two_attempts_and_redacts_secrets(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    calls = []
    secret = "sk-" + "z" * 48

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return _completed(1, stderr=f"HTTP 503\nAuthorization: Bearer {secret}")
        return _completed(1, stdout=f"partial Terra\n{secret}", stderr=f"HTTP 503\nAuthorization: Bearer {secret}")

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload_text = cost_router(goal="Classify fields.", route="luna", allow_fallback=True, timeout=5)
    payload = json.loads(payload_text)

    assert len(calls) == 2
    assert payload["routing_status"] == "retryable"
    assert payload["fallback_from"] == "luna"
    assert payload["fallback_reason"] == "http_503"
    assert len(payload["attempts"]) == 2
    assert secret not in payload_text
    assert secret not in json.dumps(payload["attempts"])
    assert "Authorization: Bearer ***" in payload_text


def test_cost_router_does_not_classify_incidental_stdout_as_infrastructure_failure(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed(
            1,
            stdout="Draft discusses HTTP 429, 503, rate limits, and service unavailable.",
            stderr="content quality validation failed",
        )

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload = json.loads(cost_router(goal="Classify fields.", route="luna", allow_fallback=True, timeout=5))

    assert len(calls) == 1
    assert payload["routing_status"] == "failed"
    assert payload["attempts"][0]["retryable"] is False
    assert payload["attempts"][0]["failure_reason"] == "non_retryable_failure"
    assert payload["fallback_from"] is None


def test_cost_router_timeout_keeps_safe_bounded_partial_output_compatibility_alias(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    secret = "sk-" + "a" * 48

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd,
            kwargs["timeout"],
            output="x" * 5000 + "\n" + secret,
            stderr=" timeout tail",
        )

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload_text = cost_router(goal="Classify fields.", route="luna", timeout=5)
    payload = json.loads(payload_text)

    assert "partial_output" in payload
    assert len(payload["partial_output"]) <= 4000
    assert payload["partial_output"].endswith(" timeout tail")
    assert secret not in payload_text
    assert secret not in json.dumps(payload["attempts"])


def test_cost_router_timeout_falls_back_exactly_once_and_preserves_luna_error(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output="Luna partial", stderr="Luna timeout detail")
        return _completed(0, stdout="Terra result")

    monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
    payload = json.loads(cost_router(goal="Classify fields.", route="luna", allow_fallback=True, timeout=5))

    assert [call[call.index("--profile") + 1] for call in calls] == ["worker-luna", "worker-terra"]
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["error"] == "cost_router timed out after 5s"
    assert payload["attempts"][0]["output"] == "Luna partial"
    assert payload["attempts"][0]["stderr"] == "Luna timeout detail"
    assert payload["fallback_from"] == "luna"
    assert payload["fallback_reason"] == "timeout"
    assert payload["routing_status"] == "completed"


def test_cost_router_direct_terra_and_sol_never_fall_back(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")

    for route, expected_profile in (("terra", "worker-terra"), ("sol", "worker-sol")):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _completed(1, stderr="provider returned HTTP 503")

        monkeypatch.setattr(cost_router_tool.subprocess, "run", fake_run)
        payload = json.loads(cost_router(goal="Handle bounded work.", route=route, allow_fallback=True, timeout=5))

        assert len(calls) == 1
        assert calls[0][calls[0].index("--profile") + 1] == expected_profile
        assert len(payload["attempts"]) == 1
        assert payload["fallback_from"] is None
        assert payload["fallback_reason"] is None


def test_cost_router_preserves_original_luna_error_when_terra_fallback_fails(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    results = iter([
        _completed(1, stdout="Luna partial", stderr="provider returned HTTP 429"),
        _completed(1, stdout="Terra partial", stderr="content quality validation failed"),
    ])
    monkeypatch.setattr(cost_router_tool.subprocess, "run", lambda *args, **kwargs: next(results))

    payload = json.loads(cost_router(goal="Classify fields.", route="luna", allow_fallback=True, timeout=5))

    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["output"] == "Luna partial"
    assert payload["attempts"][0]["stderr"] == "provider returned HTTP 429"
    assert payload["attempts"][0]["error"] == "worker profile returned non-zero exit status"
    assert payload["attempts"][0]["failure_reason"] == "http_429"
    assert payload["attempts"][1]["failure_reason"] == "non_retryable_failure"
    assert payload["routing_status"] == "failed"


def test_cost_router_redacts_credentials_from_nested_return_metadata(monkeypatch):
    monkeypatch.setattr(cost_router_tool.shutil, "which", lambda name: "/usr/bin/hermes")
    secret = "sk-" + "q" * 48
    monkeypatch.setattr(
        cost_router_tool.subprocess,
        "run",
        lambda *args, **kwargs: _completed(0, stdout="worker result"),
    )

    payload_text = cost_router(
        goal=f"Classify fields using credential {secret}.",
        route="luna",
        project={"project_id": secret},
        parent_session_id=secret,
        timeout=5,
    )

    assert secret not in payload_text
    payload = json.loads(payload_text)
    assert secret not in json.dumps(payload["project_card"])
    assert secret not in json.dumps(payload["decision_metadata"])
