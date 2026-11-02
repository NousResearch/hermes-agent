"""Cron standalone email delivery threads job metadata into the sender's Subject.

The email adapter's ``_standalone_send`` uses ``metadata["job_name"]`` for the
subject when the caller supplies it (``[Hermes] <job name>``); other plugin
standalone senders ignore the new kwarg. Behaviour contract: the metadata kwarg
must be accepted (not TypeError) by every standalone_sender_fn, and the email
subject falls back to the locale default when no job_name is present.
"""
import asyncio
import inspect

import pytest

from tools import send_message_senders
from tools.send_message_senders import _registry_standalone_send


class _FakeEntry:
    def __init__(self, fn):
        self.standalone_sender_fn = fn


@pytest.fixture
def fake_sender(monkeypatch):
    captured = {}

    async def sender(pconfig, chat_id, message, *, thread_id=None, metadata=None):
        captured.update(message=message, chat_id=chat_id, metadata=metadata)
        return {"success": True}

    monkeypatch.setattr(
        send_message_senders, "_plugin_standalone_sender",
        lambda name, **kw: (_FakeEntry(sender).standalone_sender_fn, None),
    )
    return captured


def test_metadata_kwarg_is_forwarded(fake_sender):
    meta = {"job_id": "abc123", "job_name": "Ivy skins watcher"}
    result = asyncio.run(_registry_standalone_send("email", None, "a@b.c", "body", metadata=meta))
    assert result == {"success": True}
    assert fake_sender["metadata"] == meta


def test_metadata_defaults_to_none(fake_sender):
    asyncio.run(_registry_standalone_send("email", None, "a@b.c", "body"))
    assert fake_sender["metadata"] is None


def test_email_adapter_subject_uses_job_name(monkeypatch):
    from plugins.platforms.email import adapter as email_adapter

    sent = {}

    class _Msg:
        def __setitem__(self, k, v):
            sent[k] = v

    monkeypatch.setattr(email_adapter, "MIMEText", lambda *a, **k: _Msg())
    monkeypatch.setattr(email_adapter, "_open_smtp", lambda *a, **k: _FakeSMTP())
    monkeypatch.setattr(email_adapter, "_get_secret", lambda name, default="": {
        "EMAIL_ADDRESS": "hermes@example.com",
        "EMAIL_PASSWORD": "pw",
        "EMAIL_SMTP_HOST": "smtp.example.com",
    }.get(name, default))
    monkeypatch.setattr(email_adapter, "_esecret_int", lambda name, default: default)
    monkeypatch.setattr(email_adapter, "_normalize_security", lambda *a, **k: "starttls")
    monkeypatch.setattr(email_adapter, "_tls_context", lambda *a, **k: None)

    result = asyncio.run(email_adapter._standalone_send(
        None, "me@example.com", "report body",
        metadata={"job_id": "j1", "job_name": "Release watcher"}))
    assert result.get("success")
    assert sent["Subject"] == "[Hermes] Release watcher"


def test_email_adapter_subject_falls_back_without_job_name(monkeypatch):
    from plugins.platforms.email import adapter as email_adapter

    sent = {}

    class _Msg:
        def __setitem__(self, k, v):
            sent[k] = v

    monkeypatch.setattr(email_adapter, "MIMEText", lambda *a, **k: _Msg())
    monkeypatch.setattr(email_adapter, "_open_smtp", lambda *a, **k: _FakeSMTP())
    monkeypatch.setattr(email_adapter, "_get_secret", lambda name, default="": {
        "EMAIL_ADDRESS": "hermes@example.com",
        "EMAIL_PASSWORD": "pw",
        "EMAIL_SMTP_HOST": "smtp.example.com",
    }.get(name, default))
    monkeypatch.setattr(email_adapter, "_esecret_int", lambda name, default: default)
    monkeypatch.setattr(email_adapter, "_normalize_security", lambda *a, **k: "starttls")
    monkeypatch.setattr(email_adapter, "_tls_context", lambda *a, **k: None)

    result = asyncio.run(email_adapter._standalone_send(None, "me@example.com", "hi"))
    assert result.get("success")
    assert sent["Subject"] == email_adapter.t("platform.email.standalone_subject")


class _FakeSMTP:
    def login(self, *a):
        pass

    def send_message(self, msg):
        pass

    def quit(self):
        pass
