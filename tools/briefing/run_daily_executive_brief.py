from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

from tools.briefing.executive_brief import (
    ExecutionMetrics,
    append_execution_log,
    classify_failure,
    is_delivery_blocked,
    is_dry_run,
    make_executive_brief,
    mark_delivered,
    mark_failed,
    redact_secrets,
    store_brief,
    store_delivery_blocklist,
    validate_brief_schema,
)


def _schema_log_path() -> str:
    return os.environ.get("HERMES_DB", "/app/.hermes/executions.db").replace("executions.db", "brief_execution_logs.jsonl")


def log_execution_event(event: dict) -> None:
    path = _schema_log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(redact_secrets(event), ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger().exception("Failed to append structured execution log")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deliver", action="store_true", help="Send via Telegram")
    parser.add_argument("--db", default=os.environ.get("HERMES_DB", "/app/.hermes/executions.db"))
    parser.add_argument("--env", default=os.environ.get("HERMES_ENV", "unknown"))
    args = parser.parse_args(argv)

    metrics = ExecutionMetrics()
    try:
        brief = make_executive_brief()
    except Exception as exc:
        logging.getLogger().exception("Brief generation failed")
        metrics.finish(1)
        return 1

    brief["environment"] = args.env
    generated_at = datetime.now(timezone.utc).isoformat()
    brief["generated_at"] = generated_at
    brief["generated_timestamp"] = generated_at
    brief["execution_log"].append({"timestamp": generated_at, "state": "generated", "message": "Brief generated"})

    validation = validate_brief_schema(brief)
    if not validation.get("success"):
        classification = classify_failure(Exception(validation.get("error", "validation")))
        brief = mark_failed(brief, f"Schema validation failed: {validation.get('error')}", classification=classification)
        persist = store_brief(brief, db_path=args.db)
        metrics.finish(2)
        log_execution_event(
            {
                "type": "schema_validation_failure",
                "validation": validation,
                "persistence": persist,
                "metrics": {"duration_ms": metrics.duration_ms, "exit_code": metrics.exit_code},
            }
        )
        return 2

    persist = store_brief(brief, db_path=args.db)
    brief.setdefault("persistence", persist)
    if not persist.get("success"):
        brief = mark_failed(brief, f"Persistence failed: {persist.get('error')}", classification="storage")
        metrics.finish(3)
        log_execution_event(
            {
                "type": "persistence_failure",
                "persistence": persist,
                "metrics": {"duration_ms": metrics.duration_ms, "exit_code": metrics.exit_code},
            }
        )
        return 3

    delivery_status = {
        "success": False,
        "status": "skipped",
        "message_id": None,
        "telegram_delivery_timestamp": None,
        "retry_count": 0,
        "error_classification": None,
    }

    if args.deliver:
        if is_dry_run():
            delivery_status = {
                "success": True,
                "status": "dry_run",
                "message_id": None,
                "telegram_delivery_timestamp": None,
                "retry_count": 0,
                "error_classification": None,
            }
            brief = append_execution_log(brief, "delivered", "Dry run: Telegram delivery skipped")
        else:
            key = duplicate_delivery_key(brief)
            if is_delivery_blocked(key, db_path=args.db):
                delivery_status = {
                    "success": True,
                    "status": "duplicate_blocked",
                    "message_id": None,
                    "telegram_delivery_timestamp": None,
                    "retry_count": 0,
                    "error_classification": "unknown",
                }
                brief = append_execution_log(brief, "delivered", "Duplicate delivery prevented by blocklist")
            else:
                try:
                    from tools.briefing.executive_brief_telegram import deliver_brief
                    start = time.time()
                    result = deliver_brief(brief)
                    latency_ms = int((time.time() - start) * 1000)
                    delivery_status = {
                        "success": result.get("success", False),
                        "status": "sent" if result.get("success") else "failed",
                        "message_id": None,
                        "telegram_delivery_timestamp": None,
                        "retry_count": result.get("retry_count", 0),
                        "error_classification": None,
                    }
                    if result.get("success"):
                        msg_id = None
                        results = result.get("results", [])
                        if results and isinstance(results[0], dict):
                            msg_id = results[0].get("message_id")
                        ts = datetime.now(timezone.utc).isoformat()
                        brief = mark_delivered(brief, msg_id, timestamp=ts)
                        delivery_status["message_id"] = msg_id
                        delivery_status["telegram_delivery_timestamp"] = ts
                        store_delivery_blocklist(duplicate_delivery_key(brief), db_path=args.db)
                    else:
                        err = result.get("error", "unknown telegram error")
                        classification = classify_failure(Exception(str(err)))
                        brief = mark_failed(brief, str(err), classification=classification, retry=result.get("retry_count", 0) > 0)
                        delivery_status["error_classification"] = classification
                except Exception as exc:
                    logging.getLogger().exception("Telegram delivery failed")
                    classification = classify_failure(exc)
                    brief = mark_failed(brief, str(exc), classification=classification)
                    delivery_status = {
                        "success": False,
                        "status": "failed",
                        "message_id": None,
                        "telegram_delivery_timestamp": None,
                        "retry_count": brief.get("delivery", {}).get("retry_count", 0),
                        "error_classification": classification,
                    }

    brief["delivery"] = delivery_status
    brief.setdefault("persistence", persist)
    final_state = brief.get("execution_state", "generated")
    logging.getLogger().info(
        "Executive brief complete status=%s delivery=%s id=%s",
        final_state,
        delivery_status.get("status"),
        brief.get("id"),
    )
    log_execution_event(
        {
            "type": "cron_complete",
            "execution_id": brief.get("execution_id"),
            "status": final_state,
            "delivery": delivery_status,
            "metrics": {
                "duration_ms": metrics.duration_ms,
                "exit_code": 0 if final_state == "delivered" and delivery_status.get("status") != "failed" else 4,
            },
        }
    )
    print(json.dumps(redact_secrets(brief), ensure_ascii=False))
    return 0 if final_state == "delivered" and delivery_status.get("status") != "failed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
