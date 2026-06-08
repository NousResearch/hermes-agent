from __future__ import annotations

import json
from types import SimpleNamespace

from wisdom.capture import capture_text
from wisdom.models import WisdomConfig
from wisdom.service import apply


def test_llm_application_harness_stores_validated_proposals(monkeypatch, wisdom_db, wisdom_config):
    captured = capture_text(
        "Investing thought: I confuse a good thesis with a good position size.",
        config=wisdom_config,
        db=wisdom_db,
    ).capture
    assert captured is not None

    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        body = {
            "applications": [
                {
                    "application_type": "investment_rule",
                    "title": "Investment rule",
                    "body": "Size positions by survivability, liquidity, downside path, and forced-exit risk, not thesis confidence alone.",
                },
                {
                    "application_type": "checklist",
                    "title": "Position sizing checklist",
                    "body": "1. What loss can I survive? 2. What adverse move breaks the trade? 3. What forces exit? 4. Is liquidity adequate? 5. Am I sizing by conviction or survivability?",
                },
                {
                    "application_type": "decision_rule",
                    "title": "Risk decision rule",
                    "body": "If survivability or forced-exit risk is unclear, reduce size or pass even when the thesis is attractive.",
                },
            ]
        }
        message = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], model="codex-test")

    monkeypatch.setattr("wisdom.llm_apply.call_llm", fake_call_llm)
    config = _llm_config(wisdom_config)

    applications = apply(captured.id, config=config, db=wisdom_db)

    assert calls
    assert calls[0]["task"] == "wisdom_apply"
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["timeout"] == config.apply_timeout_seconds
    assert {app.application_type for app in applications} == {
        "investment_rule",
        "checklist",
        "decision_rule",
    }
    assert all(app.metadata["generator"] == "llm" for app in applications)
    assert all(app.metadata["model_used"] == "codex-test" for app in applications)
    assert "survivability" in applications[0].body.lower()


def test_llm_application_harness_falls_back_on_invalid_output(monkeypatch, wisdom_db, wisdom_config):
    captured = capture_text(
        "Business idea: Clients don't need rear-view mirrors, they need windshields.",
        config=wisdom_config,
        db=wisdom_db,
    ).capture
    assert captured is not None

    def fake_call_llm(**_kwargs):
        message = SimpleNamespace(content='{"applications":[{"application_type":"task_proposal","body":"too thin"}]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], model="bad-model")

    monkeypatch.setattr("wisdom.llm_apply.call_llm", fake_call_llm)

    applications = apply(captured.id, config=_llm_config(wisdom_config), db=wisdom_db)

    assert {app.application_type for app in applications} == {
        "client_language",
        "principle",
        "task_proposal",
    }
    assert all(app.metadata["generator_version"] == 2 for app in applications)
    assert "not just a record" in next(
        app.body.lower() for app in applications if app.application_type == "client_language"
    )


def test_llm_application_harness_falls_back_on_llm_error(monkeypatch, wisdom_db, wisdom_config):
    captured = capture_text(
        "Health note: Poor sleep changes my decision quality.",
        config=wisdom_config,
        db=wisdom_db,
    ).capture
    assert captured is not None

    def fake_call_llm(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("wisdom.llm_apply.call_llm", fake_call_llm)

    applications = apply(captured.id, config=_llm_config(wisdom_config), db=wisdom_db)

    assert {app.application_type for app in applications} == {
        "health_experiment",
        "decision_rule",
        "principle",
    }
    assert all(app.metadata["generator_version"] == 2 for app in applications)
    assert "sleep quality" in next(
        app.body.lower() for app in applications if app.application_type == "health_experiment"
    )


def test_deterministic_mode_does_not_call_llm(monkeypatch, wisdom_db, wisdom_config):
    captured = capture_text(
        "Life thought: I keep building systems because it feels safer than doing the uncomfortable human thing.",
        config=wisdom_config,
        db=wisdom_db,
    ).capture
    assert captured is not None

    def fake_call_llm(**_kwargs):
        raise AssertionError("LLM should not be called in deterministic mode")

    monkeypatch.setattr("wisdom.llm_apply.call_llm", fake_call_llm)

    applications = apply(captured.id, config=wisdom_config, db=wisdom_db)

    assert {app.application_type for app in applications} == {
        "principle",
        "writing_idea",
        "decision_rule",
    }
    assert all(app.metadata["generator_version"] == 2 for app in applications)


def _llm_config(config: WisdomConfig) -> WisdomConfig:
    return WisdomConfig(
        enabled=config.enabled,
        db_path=config.db_path,
        capture_mode=config.capture_mode,
        max_results=config.max_results,
        interpret_timeout_seconds=config.interpret_timeout_seconds,
        interpretation_mode=config.interpretation_mode,
        application_mode="llm",
        apply_timeout_seconds=12.0,
    )
