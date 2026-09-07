"""Renderer, action dispatcher and CLI surface tests for agent-led sharing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hermes_wisdom.agent_led import render
from hermes_wisdom.agent_led.actions import handle_action
from hermes_wisdom.agent_led.history import SuggestionHistory
from hermes_wisdom.agent_led.notify import STALE_ACTION_MESSAGE, DeliveryLedger, deliver, share_candidate_event, teammate_event
from hermes_wisdom.agent_led.schemas import CandidateRecommendation, RecipientRecommendation

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)


def _share_event():
    rec = CandidateRecommendation.model_validate(
        {
            "skill_name": "notes", "content_hash": "abcdef1234", "editorial_name": "Release Notes Drafter",
            "one_line_description": "Drafts release notes.", "what_it_does": "x", "how_user_relied_on_it": "weekly releases",
            "evidence": {"window_days": 7, "invocation_count": 4, "days_used": 2, "last_used_at": None, "examples": []},
            "why_coworkers_benefit": "everyone ships", "audience": "the team", "portability": [], "confidence": 0.5,
        }
    )
    return share_candidate_event(rec, organization_id="org", recipient_id="u", at=NOW)


def _teammate_event():
    rec = RecipientRecommendation.model_validate(
        {
            "schema_version": 1, "skill_id": "sk-9", "version": 2, "relevant": True, "editorial_name": "Incident Timeline",
            "outcome_one_liner": "Clean timelines.", "publisher_name": "Dana", "publisher_uses_it_for": "reviews",
            "evidence": {"window_days": 7, "invocation_count": 3, "days_used": 2, "last_used_at": None, "examples": []},
            "why_it_helps_recipient": "on-call", "installation_count": 12, "confidence": 0.7,
        }
    )
    return teammate_event(rec, organization_id="org", recipient_id="u", at=NOW)


def test_renderers_use_editorial_name_not_product_name():
    event = _teammate_event()
    html = render.render_telegram_html(event)
    assert "<b>Incident Timeline</b>" in html
    assert "New notification" not in html and "Pass" not in html
    assert html.count("<tg-button") == 3
    blocks = render.render_slack_blocks(event)
    header = next(b for b in blocks if b["type"] == "header")
    assert header["text"]["text"] == "Incident Timeline"
    actions = next(b for b in blocks if b["type"] == "actions")
    assert [e["text"]["text"] for e in actions["elements"]] == ["Mute", "View", "Install"]
    assert all(e["value"].startswith("wa:") for e in actions["elements"])
    desktop = render.render_desktop(event)
    assert desktop["skill_name"] == "Incident Timeline" and desktop["title"] == "New skill from your team"
    assert [a["label"] for a in desktop["actions"]] == ["Mute", "View", "Install"]
    assert render.render_plain(event).startswith("Hermes Collective Wisdom\n\nNew skill from your team")


def test_handle_action_not_now_records_dismissal_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_wisdom.agent_led.actions.load_policy", lambda client=None: __import__("hermes_wisdom.agent_led.policy", fromlist=["AgentLedPolicy"]).AgentLedPolicy())
    event = _share_event()
    ledger = DeliveryLedger(tmp_path / "l.json")
    history = SuggestionHistory(tmp_path / "h.json")
    deliver(event, sender=lambda _e: None, ledger=ledger, now=NOW)
    target = next(a.target for a in event.allowed_actions if a.id == "not_now")
    first = handle_action(target, ledger=ledger, history=history, now=NOW)
    assert first["ok"] and first["gateway"] == "pending_gateway"
    assert history.is_suppressed("notes", "abcdef1234", at=NOW + timedelta(days=29))
    second = handle_action(target, ledger=ledger, history=history, now=NOW)
    assert second["stale"] and second["message"] == STALE_ACTION_MESSAGE


def test_legacy_mute_only_opens_current_native_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_wisdom.agent_led.actions.load_policy", lambda client=None: __import__("hermes_wisdom.agent_led.policy", fromlist=["AgentLedPolicy"]).AgentLedPolicy())
    event = _teammate_event()
    ledger = DeliveryLedger(tmp_path / "l.json")
    history = SuggestionHistory(tmp_path / "h.json")
    deliver(event, sender=lambda _e: None, ledger=ledger, now=NOW)
    target = next(a.target for a in event.allowed_actions if a.id == "mute")
    ask = handle_action(target, ledger=ledger, history=history, now=NOW)
    assert ask["open_mute_settings"]
    done = handle_action(target, ledger=ledger, history=history, now=NOW, mute_choice="30d")
    assert done["open_mute_settings"]
    assert not history.is_muted("sk-9", at=NOW)
    assert ledger.resolve_action(target, at=NOW)["ok"]


def test_handle_action_install_never_claims_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_wisdom.agent_led.actions.load_policy", lambda client=None: __import__("hermes_wisdom.agent_led.policy", fromlist=["AgentLedPolicy"]).AgentLedPolicy())
    event = _teammate_event()
    ledger = DeliveryLedger(tmp_path / "l.json")
    deliver(event, sender=lambda _e: None, ledger=ledger, now=NOW)
    target = next(a.target for a in event.allowed_actions if a.id == "install")
    result = handle_action(target, ledger=ledger, history=SuggestionHistory(tmp_path / "h.json"), now=NOW)
    assert result["ok"] and result["installed"] is False
    assert result["next"] == "hermes wisdom install 'sk-9@v2' --plan --json"


def test_cli_parser_has_agent_led_verbs_with_json():
    from hermes_cli.subcommands.wisdom import build_wisdom_parser

    parser = argparse.ArgumentParser()
    build_wisdom_parser(parser.add_subparsers(dest="command"))
    for argv in (
        ["wisdom", "browse", "release", "--json"],
        ["wisdom", "list", "--json"],
        ["wisdom", "show", "sk-1", "--json"],
        ["wisdom", "review-week", "--force", "--dry-run", "--json"],
        ["wisdom", "share", "start", "notes", "--json"],
        ["wisdom", "dismiss", "notes", "abc", "--json"],
        ["wisdom", "mute", "1w", "--json"],
        ["wisdom", "act", "wa:share:abc", "--json"],
    ):
        args = parser.parse_args(argv)
        assert args.json is True
