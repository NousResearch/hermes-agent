import json
from pathlib import Path

import pytest

from scripts import vapi_postcall_worker as worker


def _event(event_type, call_id, *, tool_name=None, args=None, status="queued_for_agent_ingest", event_id=None):
    ev = {
        "event_id": event_id or f"evt-{event_type}-{tool_name or 'final'}",
        "event_type": event_type,
        "actor_type": "voice_call",
        "actor_ref": call_id,
        "status": status,
        "metadata": {},
    }
    if tool_name:
        ev["metadata"] = {
            "tool_name": tool_name,
            "tool_call_id": f"tc-{tool_name}",
            "call_id": call_id,
            "arguments": args or {},
        }
    return ev


def test_builds_postcall_job_only_after_final_summary_and_lead():
    call_id = "call-1"
    events = [
        _event("vapi_tool_call", call_id, tool_name="capture_voice_lead", args={
            "name": "Wendy Foliaco",
            "email": "sitiouno@gmail.com",
            "phone": "3059274824",
            "need": "Demostración de cotización personalizada para taller mecánico",
            "notes": "Cliente quiere ejemplo para taller mecánico.",
        }),
        _event("vapi_tool_call", call_id, tool_name="send_call_summary", args={
            "caller_name": "Wendy Foliaco",
            "outcome": "Demostración solicitada",
            "summary": "Se acordó enviar por correo una demostración de cotización personalizada para un taller mecánico.",
            "next_steps": "Zeus debe preparar la cotización de ejemplo y enviarla al correo confirmando recepción.",
            "follow_up_required": True,
        }),
        _event("vapi_call_ended", call_id, status="pending_agent_ingest"),
    ]

    jobs = worker.build_pending_jobs(events, processed_call_ids=set())

    assert len(jobs) == 1
    job = jobs[0]
    assert job.call_id == call_id
    assert job.lead["name"] == "Wendy Foliaco"
    assert job.to_email == "sitiouno@gmail.com"
    assert job.deliverable_kind == "mechanic_quote_demo"
    assert job.should_send_email is True


def test_does_not_build_job_before_call_has_ended():
    events = [
        _event("vapi_tool_call", "call-1", tool_name="capture_voice_lead", args={"email": "client@example.com"}),
        _event("vapi_tool_call", "call-1", tool_name="send_call_summary", args={"follow_up_required": True, "next_steps": "enviar demo por correo"}),
    ]

    assert worker.build_pending_jobs(events, processed_call_ids=set()) == []


def test_admin_accounting_request_uses_admin_accounting_demo_not_mechanic_template(tmp_path):
    call_id = "call-admin-1"
    events = [
        _event("vapi_tool_call", call_id, tool_name="capture_voice_lead", args={
            "name": "Eileen Owen",
            "company": "HMSK Business Solution LLC",
            "email": "e.owen@solucionesfiscalesconsulting.com",
            "need": "Formatos de cotización, facturas, cuentas de cobro, agenda, recordatorios y calendario tributario por cliente.",
            "notes": "Cliente quiere demo administrativo-contable, no taller mecánico.",
        }),
        _event("vapi_tool_call", call_id, tool_name="send_call_summary", args={
            "caller_name": "Eileen Owen",
            "summary": "Enviar formatos administrativos y contables para HMSK Business Solution LLC.",
            "next_steps": "Zeus debe preparar y enviar por correo cotización, factura, cuenta de cobro, agenda, recordatorios y calendario tributario.",
            "follow_up_required": True,
        }),
        _event("vapi_call_ended", call_id, status="pending_agent_ingest"),
    ]

    jobs = worker.build_pending_jobs(events, processed_call_ids=set())

    assert len(jobs) == 1
    assert jobs[0].deliverable_kind == "admin_accounting_demo"
    assert jobs[0].should_send_email is True
    subject, text, _html = worker._email_body(jobs[0])
    assert "administrativos y contables" in subject.lower()
    assert "factura / cuenta de cobro" in text
    pdf_path = worker.generate_admin_accounting_demo_pdf(jobs[0], tmp_path)
    assert pdf_path.exists()
    assert pdf_path.name.startswith("demo_formatos_admin_contable_hmsk-business-solution-llc")


def test_skips_already_processed_call_ids():
    events = [
        _event("vapi_tool_call", "call-1", tool_name="capture_voice_lead", args={"email": "client@example.com"}),
        _event("vapi_tool_call", "call-1", tool_name="send_call_summary", args={"follow_up_required": True, "next_steps": "enviar demo por correo"}),
        _event("vapi_call_ended", "call-1", status="pending_agent_ingest"),
    ]

    assert worker.build_pending_jobs(events, processed_call_ids={"call-1"}) == []


def test_sendgrid_payload_includes_pdf_attachment(tmp_path, monkeypatch):
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.4\nexample")
    captured = {}

    def fake_request(path, body, env=None):
        captured["path"] = path
        captured["body"] = body
        captured["env"] = env
        return {"ok": True, "configured": True, "adapter": "sendgrid", "status": 202, "x_message_id": "msg-1"}

    monkeypatch.setattr(worker, "_sendgrid_request", fake_request)

    result = worker.send_email_with_attachment(
        to_email="client@example.com",
        to_name="Client",
        subject="Demo",
        text="Body",
        html="<p>Body</p>",
        attachment_path=pdf,
        env={"SENDGRID_API_KEY": "key", "SENDGRID_FROM_EMAIL": "zeus@sitiouno.com", "SENDGRID_FROM_NAME": "Sophie de SitioUno"},
    )

    assert result["ok"] is True
    assert captured["path"] == "/v3/mail/send"
    assert captured["body"]["personalizations"][0]["to"] == [{"email": "client@example.com", "name": "Client"}]
    assert captured["body"]["attachments"][0]["filename"] == "demo.pdf"
    assert captured["body"]["attachments"][0]["type"] == "application/pdf"


def test_process_job_marks_state_after_success(tmp_path, monkeypatch):
    job = worker.PostCallJob(
        call_id="call-1",
        lead={"name": "Wendy Foliaco", "email": "client@example.com", "phone": "3059274824", "need": "cotización taller"},
        summary={"summary": "Enviar cotización", "next_steps": "preparar y enviar por correo", "follow_up_required": True},
        final_event={"event_id": "evt-final"},
        deliverable_kind="mechanic_quote_demo",
        should_send_email=True,
        to_email="client@example.com",
    )
    state = {"processed_call_ids": [], "processed_event_ids": []}

    monkeypatch.setattr(worker, "generate_mechanic_quote_demo_pdf", lambda job, output_dir: tmp_path / "demo.pdf")
    (tmp_path / "demo.pdf").write_bytes(b"pdf")
    monkeypatch.setattr(worker, "send_email_with_attachment", lambda **kwargs: {"ok": True, "status": 202, "x_message_id": "msg-1"})
    crm_calls = []
    monkeypatch.setattr(worker, "upsert_crm_records", lambda job, pdf_path, email_result: crm_calls.append((job.call_id, pdf_path, email_result)) or {"ok": True})

    result = worker.process_job(job, state=state, output_dir=tmp_path, env={"SENDGRID_API_KEY": "key", "SENDGRID_FROM_EMAIL": "from@example.com"})

    assert result["ok"] is True
    assert "call-1" in state["processed_call_ids"]
    assert crm_calls and crm_calls[0][0] == "call-1"
