from __future__ import annotations

import json
import os

import pytest

from tools.briefing.executive_brief import (
    EXECUTION_STATES,
    BriefRecord,
    OperationalMetrics,
    classify_failure,
    duplicate_delivery_key,
    get_metrics,
    is_delivery_blocked,
    make_executive_brief,
    mark_delivered,
    mark_failed,
    redact_secrets,
    store_brief,
    store_delivery_blocklist,
    validate_brief_schema,
)


def _brief(**sections):
    return make_executive_brief(**sections)


def test_required_sections_present():
    brief = _brief()
    for section in [
        "calendar",
        "blackgold_priority_items",
        "deal_pipeline",
        "approval_queue",
        "banking_and_gci",
        "portfolio_alerts",
        "market_intelligence",
        "ai_workforce_status",
        "infrastructure_health",
        "failed_jobs",
        "financial_alerts",
        "weather",
        "scripture",
        "top_executive_priorities",
    ]:
        assert section in brief


def test_unavailable_defaults():
    brief = _brief()
    assert brief["weather"]["provenance"] == "unavailable"
    assert brief["calendar"]["provenance"] == "unavailable"


def test_provenance_override():
    brief = _brief(weather={"data": {"temp_c": 21}, "provenance": "live"})
    assert brief["weather"]["provenance"] == "live"
    assert brief["weather"]["data"]["temp_c"] == 21


def test_store_brief_persists(tmp_path):
    db = str(tmp_path / "briefs.db")
    brief = _brief(scripture="Provident")

    result = store_brief(brief, db_path=db)
    assert result["success"] is True
    assert result["id"] == brief["id"]
    assert result["execution_id"] == brief["execution_id"]

    import sqlite3

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT payload FROM briefs")
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert json.loads(row[0])["generated_at"] == brief["generated_at"]


def test_schema_validation_success():
    brief = _brief()
    assert validate_brief_schema(brief)["success"] is True


def test_schema_validation_missing_field():
    brief = _brief()
    brief.pop("execution_id", None)
    result = validate_brief_schema(brief)
    assert result["success"] is False
    assert "invalid_schema" in result["error"]["code"]


def test_delivery_state_transitions():
    brief = _brief()
    delivered = mark_delivered(brief, "telegram-123")
    assert delivered["delivery"]["status"] == "delivered"
    assert delivered["delivery"]["message_id"] == "telegram-123"
    assert delivered["execution_state"] == "delivered"


def test_failure_classification_and_retry():
    brief = _brief()
    failed = mark_failed(brief, "Telegram API timeout", classification="telegram_api", retry=True)
    assert failed["delivery"]["status"] == "failed"
    assert failed["delivery"]["error_classification"] == "telegram_api"
    assert failed["delivery"]["retry_count"] == 1


def test_failure_classification_mapping():
    assert classify_failure(Exception("auth token invalid")) == "authentication"
    assert classify_failure(Exception("telegram bot blocked")) == "telegram_api"
    assert classify_failure(Exception("network timeout")) == "network"
    assert classify_failure(Exception("missing config var")) == "configuration"
    assert classify_failure(Exception("database write failed")) == "storage"
    assert classify_failure(Exception("schema validation error")) == "validation"
    assert classify_failure(Exception("something weird")) == "unknown"


def test_duplicate_delivery_key_stable_across_retries():
    brief = _brief()
    brief = mark_failed(brief, "network timeout", classification="network", retry=True)
    key1 = duplicate_delivery_key(brief)
    brief = mark_failed(brief, "network timeout", classification="network", retry=True)
    key2 = duplicate_delivery_key(brief)
    assert key1 == key2


def test_idempotency_blocklist_prevents_duplicate_send(tmp_path):
    db = str(tmp_path / "briefs.db")
    brief = _brief()
    delivered = mark_delivered(brief, "telegram-456")
    key = duplicate_delivery_key(delivered)
    assert is_delivery_blocked(key, db_path=db) is False
    block = store_delivery_blocklist(key, db_path=db)
    assert block["success"] is True
    assert block["inserted"] is True
    again = store_delivery_blocklist(key, db_path=db)
    assert again["success"] is True
    assert again["inserted"] is False
    assert is_delivery_blocked(key, db_path=db) is True


def test_idempotency_survives_restart(tmp_path):
    db = str(tmp_path / "briefs.db")
    brief = _brief()
    delivered = mark_delivered(brief, "telegram-restart")
    key = duplicate_delivery_key(delivered)
    store_delivery_blocklist(key, db_path=db)
    assert is_delivery_blocked(key, db_path=db) is True
    # simulate new process / rerun by reopening connection
    import sqlite3

    conn1 = sqlite3.connect(db)
    cur1 = conn1.cursor()
    cur1.execute("SELECT 1 FROM delivery_blocklist WHERE idempotency_key=?", (key,))
    assert cur1.fetchone() is not None
    conn1.close()


def test_redact_secrets_hides_tokens():
    brief = _brief()
    brief["telegram_bot_token"] = "123456:ABC"
    brief["delivery"] = {"status": "failed", "error": "send failed with Bearer abcdef", "retry_count": 1}
    redacted = redact_secrets(brief)
    assert redacted["telegram_bot_token"] == "***REDACTED***"
    assert "error" not in redacted["delivery"]
    assert redacted["delivery"]["retry_count"] == 1


def test_redact_execution_log_masks_known_secret_patterns():
    brief = _brief()
    brief["execution_log"] = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "state": "failed", "message": "send failed with Bearer abc123def456"}
    ]
    redacted = redact_secrets(brief)
    assert "abc123def456" not in str(redacted["execution_log"])
    assert "Bearer" not in str(redacted["execution_log"])


def test_metrics_snapshot():
    metrics = get_metrics()
    metrics.record_success(duration_ms=100, delivery_latency_ms=20)
    metrics.record_failure()
    snapshot = metrics.snapshot()
    assert snapshot["successful_executions"] >= 1
    assert snapshot["consecutive_failures"] >= 1
    assert isinstance(snapshot["average_runtime_ms"], float)


def test_sections_marked_unavailable_do_not_fail_brief():
    brief = _brief()
    for section in [
        "calendar",
        "blackgold_priority_items",
        "deal_pipeline",
        "approval_queue",
        "banking_and_gci",
        "portfolio_alerts",
        "market_intelligence",
        "ai_workforce_status",
        "infrastructure_health",
        "failed_jobs",
        "financial_alerts",
        "weather",
        "scripture",
        "top_executive_priorities",
    ]:
        assert brief[section]["provenance"] == "unavailable"
    assert brief["execution_state"] == "generated"
