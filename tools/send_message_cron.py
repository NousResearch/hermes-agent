"""Native cron send boundary; shared transports remain in send_message_tool."""

import json

from tools.send_message_senders import _error


def send_cron_message(args, **kw):
    """Enforce origin-only native sends for opted-in cron jobs."""
    from cron import outbound as cron_outbound
    from tools.send_message_tool import _handle_send
    if not cron_outbound.is_cron_messaging_session():
        return json.dumps(_error(
            "This cron job is not opted in to native send_message. "
            "Set allow_messaging=true on the job."
        ))

    if args.get("action", "send") != "send":
        return json.dumps(_error("Cron messaging supports only action=send."))
    target = str(args.get("target") or "origin").strip().lower()
    if target not in {"", "origin"}:
        return json.dumps(_error(
            "Opted-in cron jobs may send only to target='origin'."
        ))

    message = args.get("message")
    if not isinstance(message, str) or not message.strip():
        return json.dumps(_error("message is required"))

    try:
        message_key = cron_outbound.normalize_message_key(args.get("message_key"))
    except ValueError as exc:
        return json.dumps(_error(str(exc)))

    job_id = cron_outbound.current_cron_job_id()
    run_id = cron_outbound.current_cron_run_id()
    run = cron_outbound.current_run()
    origin = run.origin if run else None
    if not job_id or not run_id or not origin:
        return json.dumps(_error(
            "Cron outbound send is missing the job, run, or origin binding."
        ))

    try:
        claim = cron_outbound.claim_or_reuse(
            job_id=job_id,
            run_id=run_id,
            message_key=message_key,
            target="origin",
            body=message,
            platform=origin["platform"],
            chat_id=str(origin["chat_id"]),
            thread_id=origin.get("thread_id"),
        )
    except ValueError as exc:
        return json.dumps(_error(str(exc)))

    if claim["action"] == "reuse":
        return cron_outbound.dumps(cron_outbound.reuse_payload(claim["record"]))

    try:
        raw = _handle_send({
            "target": (
                f"{origin['platform']}:{origin['chat_id']}"
                + (f":{origin['thread_id']}" if origin.get("thread_id") else "")
            ),
            "message": message,
        })
    except Exception as exc:
        record = cron_outbound.mark_result(
            job_id=job_id,
            run_id=run_id,
            message_key=message_key,
            attempt=claim["record"]["attempt"],
            status="ambiguous",
            error=f"send engine raised {type(exc).__name__}",
        )
        return cron_outbound.dumps(cron_outbound.success_payload(record))
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        parsed = raw
    classified = cron_outbound.classify_send_result(parsed)
    record = cron_outbound.mark_result(
        job_id=job_id,
        run_id=run_id,
        message_key=message_key,
        attempt=claim["record"]["attempt"],
        status=classified["status"],
        transport_message_id=classified.get("transport_message_id"),
        error=classified.get("error"),
    )
    payload = cron_outbound.success_payload(record)
    if classified.get("error") and not payload.get("error"):
        payload["error"] = classified["error"]
    return cron_outbound.dumps(payload)
