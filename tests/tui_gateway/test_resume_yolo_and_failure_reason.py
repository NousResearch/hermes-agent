"""Regression tests for two bugs found investigating HermesUI NG reports:

1. ``session.resume`` mints a NEW runtime ``session_key`` every time
   (``_new_runtime_ids``), so the in-memory ``tools.approval._session_yolo``
   set — keyed on the OLD key — never covered it. YOLO silently reverted to
   per-command approval on every reconnect/resume even though the stored
   session row's ``model_config.yolo_mode`` and the UI toggle both still said
   it was on. ``tui_gateway/methods_session.py::_restore_session_yolo`` fixes
   this by mirroring the CLI's existing ``_restore_session_yolo``
   (hermes_cli/cli_session_mixin.py) for the gateway/TUI/desktop surface.

2. A failed turn's ``failure_reason`` (e.g. ``content_policy_blocked``,
   ``auth``, ``rate_limit``) was only copied into the client-facing
   ``message.complete`` payload inside the ``billing_block`` branch of
   ``_complete_turn_payload`` — every non-billing failure silently dropped
   it, so the client fell back to generic "request failed" copy instead of
   the agent's actual classification (e.g. "content policy" vs "auth").
"""

from __future__ import annotations

import types

import pytest


# ── 1. YOLO restored across session.resume ──────────────────────────────


def test_restore_session_yolo_enables_bypass_when_stored_flag_set(monkeypatch):
    from tui_gateway import methods_session as ms

    monkeypatch.setattr(ms, "_YOLO_MODE_FROZEN", False, raising=False)

    enabled_keys = []

    class _FakeSessionDB:
        @staticmethod
        def session_yolo_enabled(session_meta):
            return bool((session_meta or {}).get("model_config", {}).get("yolo_mode"))

    def _fake_enable(key):
        enabled_keys.append(key)

    def _fake_is_enabled(key):
        return key in enabled_keys

    monkeypatch.setitem(
        __import__("sys").modules, "hermes_state",
        types.SimpleNamespace(SessionDB=_FakeSessionDB),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "tools.approval",
        types.SimpleNamespace(
            _YOLO_MODE_FROZEN=False,
            enable_session_yolo=_fake_enable,
            is_session_yolo_enabled=_fake_is_enabled,
        ),
    )

    ms._restore_session_yolo("new-runtime-key", {"model_config": {"yolo_mode": True}})

    assert enabled_keys == ["new-runtime-key"]


def test_restore_session_yolo_noop_when_stored_flag_false(monkeypatch):
    from tui_gateway import methods_session as ms

    enabled_keys = []

    class _FakeSessionDB:
        @staticmethod
        def session_yolo_enabled(session_meta):
            return bool((session_meta or {}).get("model_config", {}).get("yolo_mode"))

    monkeypatch.setitem(
        __import__("sys").modules, "hermes_state",
        types.SimpleNamespace(SessionDB=_FakeSessionDB),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "tools.approval",
        types.SimpleNamespace(
            _YOLO_MODE_FROZEN=False,
            enable_session_yolo=lambda key: enabled_keys.append(key),
            is_session_yolo_enabled=lambda key: False,
        ),
    )

    ms._restore_session_yolo("new-runtime-key", {"model_config": {"yolo_mode": False}})

    assert enabled_keys == []


def test_restore_session_yolo_noop_without_session_key(monkeypatch):
    from tui_gateway import methods_session as ms

    # Must return before importing anything when the key is empty/None.
    ms._restore_session_yolo("", {"model_config": {"yolo_mode": True}})
    ms._restore_session_yolo(None, {"model_config": {"yolo_mode": True}})


# ── 2. failure_reason surfaced for every error, not just billing ───────────


def test_failure_reason_reaches_payload_for_non_billing_failure(monkeypatch):
    # _complete_turn_payload (defined in prompt_turn.py) is rebound onto server.py's globals
    # at import time (method_ctx.bind_module) so it resolves helpers like _get_usage /
    # render_message / _fail_inflight_turn from server.py's own namespace — call and patch it
    # through `server`, matching how the real turn loop invokes it.
    from tui_gateway import server

    session = {"history_lock": __import__("threading").Lock()}

    class _Result(dict):
        pass

    result = _Result(
        final_response="⚠️ The model provider refused this request (content policy).",
        error="content_policy_blocked: model declined (content_filter)",
        failed=True,
        failure_reason="content_policy_blocked",
        completed=False,
    )

    st = types.SimpleNamespace(
        result=result,
        agent=types.SimpleNamespace(provider="anthropic", model="claude-opus-5-5"),
        error_retained=False,
        error_detail="",
        prompt_text="hello",
        compression_count=None,
        terminal_callback=None,
        receipt_committed=False,
    )

    # Avoid needing the full agent/session machinery for unrelated helpers.
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_is_bot_mode_session", lambda session: False)
    monkeypatch.setattr(server, "_persisted_turn_receipt", lambda *a, **k: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: "")
    monkeypatch.setattr(server, "_append_inflight_delta", lambda *a, **k: None)
    monkeypatch.setattr(server, "_fail_inflight_turn", lambda *a, **k: None)
    monkeypatch.setattr(server, "_clear_inflight_turn", lambda *a, **k: None)
    monkeypatch.setattr(server, "_turn_failure_detail", lambda *a, **k: "")

    payload, raw, status = server._complete_turn_payload(session, st, None, 80)

    assert status == "error"
    assert payload.get("failure_reason") == "content_policy_blocked", (
        "failure_reason must reach the client for every failed turn, not just "
        "billing-wall failures — otherwise the client falls back to generic "
        "'request failed' copy instead of the agent's real classification."
    )
    # Non-billing failures must not fabricate a billing descriptor.
    assert "billing" not in payload
