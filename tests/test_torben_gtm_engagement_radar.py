from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.gtm_engagement_radar import (
    discover_x_response_opportunities,
    run_gtm_engagement_radar,
    select_engagement_topics,
)


def _radar_payload() -> dict:
    return {
        "generated_at": "2026-06-25T17:00:00Z",
        "findings": [
            {
                "title": "Poisoned Playbooks: Knowledge Poisoning for AI Security Agents",
                "summary": "RAG security agents can be steered by poisoned knowledge sources.",
                "pillar": "security_ai",
                "content_route": "longform_article",
                "url": "https://arxiv.org/abs/2606.24402",
            },
            {
                "title": "Agentic AI Systems",
                "summary": "A survey frames agentic systems from foundations to production systems.",
                "pillar": "ai_engineering_leverage",
                "content_route": "longform_article",
                "url": "https://arxiv.org/abs/2606.24937",
            },
        ],
    }


def test_select_engagement_topics_builds_bounded_queries() -> None:
    topics = select_engagement_topics(_radar_payload(), max_topics=1)

    assert len(topics) == 1
    assert topics[0]["title"].startswith("Poisoned Playbooks")
    assert "AI security" in topics[0]["query"] or "agent security" in topics[0]["query"]
    assert topics[0]["source_url"] == "https://arxiv.org/abs/2606.24402"


def test_engagement_radar_stages_draft_only_reply_opportunities_and_dedupes(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "actions.json")
    state_path = tmp_path / "engagement-state.json"
    now = datetime(2026, 6, 25, 17, 0, tzinfo=timezone.utc)

    def fake_discover(*, topics, max_opportunities, now):
        return {
            "provider": "xai-oauth",
            "model": "grok-4.3",
            "x_search_enabled": True,
            "x_search_used": True,
            "opportunities": [
                {
                    "post_url": "https://x.com/example/status/1",
                    "author_handle": "operator",
                    "author_name": "Operator",
                    "post_summary": "A practitioner argues that RAG agents are impossible to secure.",
                    "why_reply": "Eric can add a concrete control-plane answer instead of a vague dunk.",
                    "reply_angle": "Bring provenance, evals, rollback, and source trust into the thread.",
                    "draft_reply": "The useful version is not 'never use RAG agents.' It is: prove source provenance, score retrieval drift, and make rollback boring.",
                    "score": 88,
                    "risk_notes": "Avoid sounding like a vendor pitch.",
                    "source_topic": topics[0]["title"],
                    "source_url": topics[0]["source_url"],
                },
                {
                    "post_url": "https://x.com/example/status/2",
                    "author_handle": "bait",
                    "post_summary": "Low-context AI rage bait.",
                    "why_reply": "Maybe engagement.",
                    "reply_angle": "Dunk.",
                    "draft_reply": "No.",
                    "score": 42,
                    "risk_notes": ["Low signal."],
                    "source_topic": topics[0]["title"],
                },
            ],
        }

    first = run_gtm_engagement_radar(
        _radar_payload(),
        ledger=ledger,
        state_path=state_path,
        now=now,
        discover=fake_discover,
    )
    second = run_gtm_engagement_radar(
        _radar_payload(),
        ledger=ledger,
        state_path=state_path,
        now=now + timedelta(minutes=30),
        discover=fake_discover,
    )

    assert first["wakeAgent"] is True
    assert first["selected_count"] == 1
    assert first["public_actions_taken"] == 0
    assert first["external_mutations"] == 0
    assert first["llm_judge"]["invoked"] is True
    assert first["llm_judge"]["model"] == "grok-4.3"
    assert first["llm_judge"]["x_search_used"] is True
    assert first["cron_audit"]["llm_invoked"] is True
    assert first["cron_audit"]["wake_reason"] == "llm_judged_reply_opportunities_selected"
    assert "Torben / GTM Response Radar" in first["text"]
    assert "LLM judge: Grok ran with x_search" in first["text"]
    assert "X algorithm lens" in first["text"]
    assert "repo snapshot 0bfc279" in first["text"]
    assert "Nothing has been posted" in first["text"]
    assert "draft_only" == first["actions"][0]["executor_state"]["mutation_status"]
    assert first["actions"][0]["executor_state"]["publishing_blocked_until"] == "separate_explicit_public_reply_approval"
    assert first["actions"][0]["executor_state"]["post_url"] == "https://x.com/example/status/1"
    assert first["actions"][0]["executor_state"]["llm_judged"] is True
    assert first["actions"][0]["executor_state"]["llm_score"] == 88
    assert first["actions"][0]["executor_state"]["risk_notes"] == ["Avoid sounding like a vendor pitch."]
    lens_source = first["actions"][0]["executor_state"]["x_algorithm_signal_lens"]["source"]
    assert lens_source["url"] == "https://github.com/xai-org/x-algorithm"
    assert lens_source["commit"] == "0bfc2795d308f90032544322747caacd535f75ae"
    assert first["actions"][0]["executor_state"]["x_algorithm_signal_lens"]["schema_version"] == 2
    assert second["wakeAgent"] is False
    assert second["reason"] == "no new response opportunities"
    assert second["selected_count"] == 0
    assert second["text"] == ""
    assert second["llm_judge"]["invoked"] is True
    assert second["llm_judge"]["status"] == "no_new_opportunities"
    assert second["llm_judge"]["x_search_used"] is True
    assert second["cron_audit"]["llm_invoked"] is True
    assert second["cron_audit"]["why_wake_agent"] is False
    assert second["cron_audit"]["wake_reason"] == "no_new_response_opportunities"


def test_engagement_radar_no_topics_has_explicit_non_llm_audit(tmp_path: Path) -> None:
    payload = run_gtm_engagement_radar(
        {"findings": []},
        ledger=ActionLedger(tmp_path / "actions.json"),
        state_path=tmp_path / "engagement-state.json",
        now=datetime(2026, 6, 25, 17, 0, tzinfo=timezone.utc),
        discover=lambda **_: {"opportunities": []},
    )

    assert payload["wakeAgent"] is False
    assert payload["reason"] == "no GTM topics available"
    assert payload["selected_count"] == 0
    assert payload["llm_judge"]["invoked"] is False
    assert payload["llm_judge"]["status"] == "not_invoked"
    assert payload["llm_judge"]["x_search_requested"] is False
    assert payload["cron_audit"]["llm_invoked"] is False
    assert payload["cron_audit"]["wake_reason"] == "no_gtm_topics_available"


def test_engagement_radar_preview_actions_preserve_llm_and_safety_metadata(tmp_path: Path) -> None:
    def fake_discover(*, topics, max_opportunities, now):
        return {
            "provider": "xai-oauth",
            "model": "grok-4.3",
            "x_search_enabled": True,
            "x_search_used": True,
            "opportunities": [
                {
                    "post_url": "https://x.com/example/status/3",
                    "author_handle": "researcher",
                    "post_summary": "A technical post about retrieval poisoning.",
                    "why_reply": "Eric can add a concrete control.",
                    "reply_angle": "Name source provenance as the fix.",
                    "draft_reply": "Treat the retrieved source like code: signed, versioned, reviewed.",
                    "score": 77,
                    "risk_notes": "Low risk; technical thread.",
                    "source_topic": topics[0]["title"],
                    "source_url": topics[0]["source_url"],
                }
            ],
        }

    payload = run_gtm_engagement_radar(
        _radar_payload(),
        ledger=ActionLedger(tmp_path / "actions.json"),
        state_path=tmp_path / "engagement-state.json",
        now=datetime(2026, 6, 25, 17, 0, tzinfo=timezone.utc),
        discover=fake_discover,
        stage_actions=False,
        mark_delivered=False,
    )

    state = payload["actions"][0]["executor_state"]
    assert state["mutation_status"] == "preview_only"
    assert state["llm_judged"] is True
    assert state["llm_score"] == 77
    assert state["risk_notes"] == ["Low risk; technical thread."]
    assert state["public_actions_taken"] == 0
    assert state["external_mutations"] == 0


def test_discover_x_response_opportunities_calls_grok_with_x_search_and_lens(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "opportunities": [
                                            {
                                                "post_url": "https://x.com/example/status/1",
                                                "author_handle": "operator",
                                                "author_name": "Operator",
                                                "post_summary": "A post about AI agent security.",
                                                "why_reply": "Eric has a concrete operator angle.",
                                                "reply_angle": "Talk control planes.",
                                                "draft_reply": "The control plane matters more than the demo.",
                                                "score": 81,
                                                "risk_notes": [],
                                                "source_topic": "Poisoned Playbooks",
                                                "source_url": "https://arxiv.org/abs/2606.24402",
                                            }
                                        ]
                                    }
                                ),
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://x.com/example/status/1",
                                        "title": "Example post",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "hermes_cli.signal_coo.gtm_engagement_radar.resolve_xai_http_credentials",
        lambda: {"provider": "xai-oauth", "api_key": "test-token", "base_url": "https://api.x.ai/v1"},
    )
    monkeypatch.setattr("hermes_cli.signal_coo.gtm_engagement_radar.requests.post", fake_post)
    now = datetime(2026, 6, 25, 17, 0, tzinfo=timezone.utc)

    result = discover_x_response_opportunities(
        topics=[
            {
                "title": "Poisoned Playbooks",
                "summary": "Knowledge poisoning can steer AI security agents.",
                "pillar": "security_ai",
                "source_url": "https://arxiv.org/abs/2606.24402",
                "query": "knowledge poisoning AI security agents",
            }
        ],
        max_opportunities=2,
        now=now,
    )

    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["json"]["model"] == "grok-4.3"
    assert captured["json"]["tools"] == [{"type": "x_search"}]
    assert captured["json"]["store"] is False
    prompt = captured["json"]["input"][1]["content"]
    assert "profile-click intent" in prompt
    assert "xai-org/x-algorithm" in prompt
    assert "Do not claim exact private X ranking weights" in prompt
    assert result["provider"] == "xai-oauth"
    assert result["x_search_enabled"] is True
    assert result["x_search_used"] is True
    assert result["citations"] == [{"url": "https://x.com/example/status/1", "title": "Example post"}]
    assert result["opportunities"][0]["post_url"] == "https://x.com/example/status/1"
