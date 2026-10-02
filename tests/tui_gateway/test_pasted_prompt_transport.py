"""A Desktop large paste reaches the model as text, never as a path it has to open.

Desktop saves a big paste as ``<HERMES_HOME>/composer-pastes/pasted_content_*.txt`` and attaches
it as ``@file:``. ``file.attach`` used to copy it, like any file outside the chat's cwd, into
``attachments/``, which the turn's ``@file:`` guard refuses: the model got only that path plus
"path is outside the allowed workspace", and workspace-minded models asked the user to move the
file instead of starting the task.

Every test drives the real ``file.attach`` RPC and the real ``_prepare_turn_input`` (which builds
the prompt handed to the agent), stopping at the first step after the prompt is final.
"""

from __future__ import annotations

import base64
import logging
import threading
import types

import pytest

from tui_gateway import server

TASK = "You are the adversarial reviewer for ticket T-102.\n" + "".join(
    f"{n}. Attempt to falsify claim {n} against the source of truth.\n" for n in range(1, 300))


class _ReachedInference(Exception):
    """The turn got past context preparation (the prompt is what the agent would receive)."""


@pytest.fixture
def gw(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "getreta"
    repo.mkdir()
    (repo / "README.md").write_text("# GetReta\n", encoding="utf-8")

    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    for name in ("_wire_callbacks", "_apply_pending_model_switch", "_sync_agent_model_with_config",
                 "_sync_agent_compression_with_config", "_sync_agent_fallback_with_config",
                 "_sync_bot_capabilities", "_adopt_out_of_band_turns", "_register_session_cwd"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    emitted: list[tuple] = []
    monkeypatch.setattr(server, "_emit", lambda *a, **k: emitted.append(a))

    def reached(*a, **k):
        raise _ReachedInference()

    import agent.notification_presentation as presentation
    monkeypatch.setattr(presentation, "event_presentation_muted", reached)

    ready = threading.Event()
    ready.set()
    sid = "paste-transport"
    server._sessions[sid] = {
        "agent": types.SimpleNamespace(
            model="test/model", base_url="", api_key="", provider="", _config_context_length=128_000),
        "agent_ready": ready, "agent_error": None, "attached_images": [], "cols": 80,
        "cwd": str(repo), "history": [], "history_lock": threading.RLock(), "history_version": 0,
        "image_counter": 0, "profile_home": str(home), "running": False, "session_key": sid,
        "transport": None,
    }
    yield types.SimpleNamespace(home=home, repo=repo, sid=sid, emitted=emitted, tmp=tmp_path)
    server._sessions.pop(sid, None)


def desktop_paste(home, text: str | bytes, stamp: str = "2026-09-25_23-03-44-685_343de0"):
    """What apps/desktop/electron/composer-paste.ts writes for a large paste."""
    path = home / "composer-pastes" / f"pasted_content_{stamp}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


def attach(gw, **params) -> dict:
    resp = server._methods["file.attach"](1, {"session_id": gw.sid, **params})
    assert "error" not in resp, resp
    return resp["result"]


def prepare_turn(gw, text: str) -> str | None:
    """The prompt the agent would receive, or None when the turn was refused before inference."""
    st = server._TurnRun(agent=None, one_turn_restore=None, terminal_callback=None, receipt_committed=False)
    try:
        prepared = server._prepare_turn_input(gw.sid, server._sessions[gw.sid], st, text, [])
    except _ReachedInference:
        return st.prompt_text
    finally:
        from tools.approval_context import reset_current_session_key
        from tools.terminal_scope import reset_terminal_scope
        if st.scopes.terminal is not None:
            reset_terminal_scope(st.scopes.terminal)
        if st.scopes.secret is not None:
            server.reset_secret_scope(st.scopes.secret)
        if st.scopes.home is not None:
            server.reset_hermes_home_override(st.scopes.home)
        if st.scopes.approval is not None:
            reset_current_session_key(st.scopes.approval)
        server._clear_session_context(st.scopes.session_tokens)
    assert prepared is None
    return None


# 1 — short inline prompt
def test_short_prompt_stays_inline_without_resolution(gw, caplog):
    caplog.set_level(logging.DEBUG)
    text = "Review the README of this repo and list three risks."

    assert prepare_turn(gw, text) == text
    assert not (gw.home / "composer-pastes").exists()
    assert not (gw.home / "attachments").exists()
    assert "TRANSPORT_ARTIFACT" not in caplog.text


# 2 + 7 — long paste reaches the model as text, with no attachments/ path or workspace warning
@pytest.mark.parametrize("delivery", ["gateway-visible", "uploaded"])
def test_long_paste_reaches_the_model_as_text(gw, caplog, delivery):
    caplog.set_level(logging.INFO)
    paste = desktop_paste(gw.home, TASK)
    if delivery == "gateway-visible":
        staged = attach(gw, path=str(paste), name="Pasted content (12 KB)")
        assert staged["uploaded"] is False and staged["path"] == str(paste.resolve())
    else:  # remote gateway / container backend: the client's path is not on the gateway
        client_path = r"C:\Users\ben\AppData\Roaming\hermes\composer-pastes" + "\\" + paste.name
        paste.unlink()
        data_url = "data:text/plain;base64," + base64.b64encode(TASK.encode()).decode()
        staged = attach(gw, path=client_path, name="Pasted content (12 KB)", data_url=data_url)
        assert staged["path"] == str((gw.home / "composer-pastes" / paste.name).resolve())
        assert "TRANSPORT_ARTIFACT_MATERIALIZED" in caplog.text

    prompt = prepare_turn(gw, staged["ref_text"])

    assert prompt is not None
    assert prompt.count(TASK.strip()) == 1
    assert "/attachments/" not in prompt
    assert "outside the allowed workspace" not in prompt
    assert "--- Context Warnings ---" not in prompt
    assert not (gw.home / "attachments").exists()  # no second copy of the paste
    assert "TRANSPORT_ARTIFACT_RESOLVED" in caplog.text
    assert "falsify claim" not in caplog.text  # prompt contents are never logged


@pytest.mark.parametrize("chat_folder", ["another-project", "home"])
def test_paste_inlines_whatever_folder_the_chat_is_in(gw, chat_folder):
    """Nothing is keyed to one project: the paste resolves from any chat cwd, including one that
    contains ~/.hermes (the ref is then workspace-relative, not absolute)."""
    cwd = gw.tmp / "another-project" if chat_folder == "another-project" else gw.tmp
    cwd.mkdir(exist_ok=True)
    server._sessions[gw.sid]["cwd"] = str(cwd)
    staged = attach(gw, path=str(desktop_paste(gw.home, TASK)))

    prompt = prepare_turn(gw, staged["ref_text"])

    assert staged["uploaded"] is False
    assert prompt is not None and prompt.count(TASK.strip()) == 1
    assert "--- Context Warnings ---" not in prompt


# 3 — repo-restricted turn: the paste resolves, the rest of ~/.hermes stays out of reach
def test_repo_restricted_turn_gets_paste_but_not_other_attachments(gw):
    other = gw.home / "attachments" / "other-session-notes.txt"
    other.parent.mkdir(parents=True)
    other.write_text("OTHER-SESSION-SECRET\n", encoding="utf-8")
    staged = attach(gw, path=str(desktop_paste(gw.home, TASK)))

    prompt = prepare_turn(
        gw, f"{staged['ref_text']} and @file:{other} and @folder:{gw.home / 'attachments'} "
            f"and @file:../.hermes/attachments/{other.name}")

    assert prompt is not None and TASK.strip() in prompt
    assert "OTHER-SESSION-SECRET" not in prompt
    assert "- other-session-notes.txt" not in prompt  # no folder listing either
    assert prompt.count("path is outside the allowed workspace") == 3


# 4 — ordinary user files keep their existing staging and are not treated as prompt transport
def test_ordinary_file_attachments_keep_their_existing_staging(gw):
    downloads = gw.tmp / "Downloads"
    downloads.mkdir()
    report = downloads / "report.txt"
    report.write_text("REPORT-BODY\n", encoding="utf-8")
    lookalike = downloads / "pasted_content_notes.txt"  # a user file that merely looks like a paste
    lookalike.write_text("LOOKALIKE-BODY\n", encoding="utf-8")
    spec = gw.repo / "spec.pdf"
    spec.write_bytes(b"%PDF-1.7\n\x00\x01binary")

    staged_report = attach(gw, path=str(report))
    staged_lookalike = attach(gw, path=str(lookalike))
    uploaded = attach(gw, path="/Users/alice/Downloads/q3.txt", name="q3.txt",
                      data_url="data:text/plain;base64," + base64.b64encode(b"Q3").decode())
    staged_spec = attach(gw, path=str(spec))

    assert staged_report["path"] == str((gw.home / "attachments" / "report.txt").resolve())
    assert staged_lookalike["path"] == str((gw.home / "attachments" / lookalike.name).resolve())
    assert uploaded["path"] == str((gw.home / "attachments" / "q3.txt").resolve())
    assert (staged_spec["uploaded"], staged_spec["ref_text"]) == (False, "@file:spec.pdf")
    assert not (gw.home / "composer-pastes").exists()

    prompt = prepare_turn(gw, f"{staged_spec['ref_text']} {staged_lookalike['ref_text']}")
    assert prompt is not None  # the turn is not refused
    assert "📎 @file:spec.pdf (application/pdf" in prompt  # binary stays an on-disk pointer
    assert "\x00" not in prompt


# 5 — an unresolvable paste stops the turn before inference with a harness-level error
@pytest.mark.parametrize("damage", ["missing", "invalid-utf8", "binary"])
def test_unresolvable_paste_stops_before_the_model(gw, caplog, damage):
    paste = desktop_paste(gw.home, TASK)
    staged = attach(gw, path=str(paste))
    if damage == "missing":
        paste.unlink()
    elif damage == "invalid-utf8":
        paste.write_bytes(b"review \xff\xfe\xfa this")
    else:
        paste.write_bytes(b"review\x00\x00this")
    caplog.set_level(logging.WARNING)

    assert prepare_turn(gw, staged["ref_text"]) is None

    errors = [args[2] for args in gw.emitted if args[0] == "error"]
    assert len(errors) == 1
    assert "context injection refused: pasted content could not be loaded" in errors[0]["message"]
    assert "CONTEXT_INJECTION_FAILURE" in caplog.text


# 6 — several pastes: each inlined once, in the order they appear
def test_multiple_pastes_inline_in_order_exactly_once(gw):
    first = desktop_paste(gw.home, "FIRST-PASTE: review the email architecture.\n" * 50, stamp="a_000001")
    second = desktop_paste(gw.home, "SECOND-PASTE: then check DNS records.\n" * 50, stamp="b_000002")
    refs = [attach(gw, path=str(p))["ref_text"] for p in (first, second)]

    prompt = prepare_turn(gw, f"Do both. {refs[0]} then {refs[1]}")

    assert prompt is not None
    body = prompt.split("--- Attached Context ---", 1)[1]
    assert body.count("FIRST-PASTE: review") == 50 and body.count("SECOND-PASTE: then") == 50
    assert body.index("FIRST-PASTE") < body.index("SECOND-PASTE")
    assert body.count("📄 @file:") == 2
