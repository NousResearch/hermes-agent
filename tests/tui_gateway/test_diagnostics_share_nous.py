"""diagnostics.share_nous RPC — Desktop "Send Diagnostics" upload path.

Contract pinned:
* Reuses the CLI ``--nous`` pipeline (collect_share_bundle → build_nous_bundle
  → share_to_nous) with redaction FORCED on — the client cannot disable it.
* ``error_context`` and ``extra_files`` are redacted server-side, labels
  sanitized, sizes capped.
* Upload failures return a structured ``{ok: False, error}`` envelope, never a
  JSON-RPC error (the desktop renders them inline in the modal).
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from tui_gateway import server


def _handler():
    fn = server._methods.get("diagnostics.share_nous")
    assert fn is not None, "diagnostics.share_nous not registered"
    return fn


@pytest.fixture()
def captured_upload(monkeypatch, tmp_path):
    """Mock ONLY the network leg; the bundle pipeline runs for real."""
    captured: dict = {}

    def _fake_share(blob: bytes) -> dict:
        captured["blob"] = blob
        return {
            "viewUrl": "https://nas.example/view/abc123",
            "id": "abc123",
            "expiresAt": "2026-09-05T00:00:00Z",
        }

    import hermes_cli.diagnostics_upload as du

    monkeypatch.setattr(du, "share_to_nous", _fake_share)
    return captured


def _envelope(blob: bytes) -> dict:
    return json.loads(gzip.decompress(blob).decode("utf-8"))


@pytest.mark.parametrize(
    "secondary_params",
    ({"profile": "work"}, {"session_id": "work"}),
    ids=("explicit-profile", "session-profile"),
)
def test_share_nous_collects_inside_the_selected_profile_scope_a_b_a(
    monkeypatch, tmp_path, secondary_params,
):
    """Collection and upload stay in one full runtime scope: home, secrets and terminal policy."""
    from agent import secret_scope
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home, get_hermes_home_override
    from tools import terminal_tool
    from tools.terminal_scope import get_terminal_scope
    from tui_gateway import launch_profile_policy

    launch = tmp_path / ".hermes"
    work = launch / "profiles" / "work"
    for home, marker, backend in (
        (launch, "launch-profile-A", "local"),
        (work, "worker-profile-B", "docker"),
    ):
        (home / "logs").mkdir(parents=True)
        (home / ".env").write_text(f"DIAGNOSTIC_SCOPE_MARKER={marker}\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            f"terminal:\n  backend: {backend}\n  docker_image: ${{DIAGNOSTIC_SCOPE_MARKER}}\n",
            encoding="utf-8",
        )
        (home / "logs" / "agent.log").write_text(f"diagnostic marker: {marker}\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("DIAGNOSTIC_SCOPE_MARKER", "launch-profile-A")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_sessions", {
        "launch": {"profile_home": None},
        "work": {"profile_home": str(work)},
    })
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    # Mirror serve startup: bridge the launch profile once, then freeze it before any routed call.
    # The RPC itself must use ContextVars and leave this process-wide mapping byte-for-byte alone.
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=launch, project_env=tmp_path / "no-project-env")
    launch_profile_policy.activate_multi_profile_hosting()

    uploads = []

    def _fake_share(blob: bytes) -> dict:
        uploads.append({
            "home": str(get_hermes_home()),
            "secret": get_secret("DIAGNOSTIC_SCOPE_MARKER"),
            "terminal_bound": get_terminal_scope() is not None,
            "backend": terminal_tool._get_env_config()["env_type"],
            "envelope": _envelope(blob),
        })
        return {"viewUrl": "https://nas.example/view/scoped", "id": "scoped"}

    import hermes_cli.diagnostics_upload as du

    monkeypatch.setattr(du, "share_to_nous", _fake_share)
    environ_before = dict(os.environ)
    for index, params in enumerate(({"session_id": "launch"}, secondary_params, {"session_id": "launch"})):
        response = getattr(server, "handle_request")({
            "id": f"rid-scope-{index}", "method": "diagnostics.share_nous", "params": params,
        })
        assert response["result"]["ok"] is True, response

    assert [(row["home"], row["secret"], row["backend"]) for row in uploads] == [
        (str(launch), "launch-profile-A", "local"),
        (str(work), "worker-profile-B", "docker"),
        (str(launch), "launch-profile-A", "local"),
    ]
    assert all(row["terminal_bound"] for row in uploads)
    content = [json.dumps(row["envelope"], sort_keys=True) for row in uploads]
    assert "launch-profile-A" in content[0] and "worker-profile-B" not in content[0]
    assert "worker-profile-B" in content[1] and "launch-profile-A" not in content[1]
    assert "launch-profile-A" in content[2] and "worker-profile-B" not in content[2]
    assert dict(os.environ) == environ_before
    assert get_hermes_home_override() is None
    assert get_terminal_scope() is None


def test_share_nous_uploads_redacted_bundle(captured_upload):
    result = _handler()("rid-1", {})
    payload = result["result"]

    assert payload["ok"] is True
    assert payload["view_url"] == "https://nas.example/view/abc123"
    assert payload["upload_id"] == "abc123"

    envelope = _envelope(captured_upload["blob"])
    assert envelope["format"].startswith("hermes-debug-share/")
    assert envelope["redacted"] is True
    assert "report" in envelope["files"]


def test_share_nous_attaches_redacted_error_context(captured_upload):
    secret = "sk-abc123def456ghi789jkl012mno345pqr678"
    result = _handler()(
        "rid-2",
        {"error_context": f"layer: provider\ncode: rate_limit\nkey was {secret}"},
    )
    assert result["result"]["ok"] is True

    files = _envelope(captured_upload["blob"])["files"]
    context = files.get("error-context.txt", "")
    assert "layer: provider" in context
    assert secret not in context, "secret leaked through error_context redaction"


def test_share_nous_client_text_gets_upload_safe_log_redaction(captured_upload):
    """Client artifacts must ride the SAME redactor as backend logs
    (_redact_log_text): secrets AND email addresses — not just the bare
    secret pass, which leaves emails through (review finding on #92020)."""
    secret = "sk-abc123def456ghi789jkl012mno345pqr678"
    result = _handler()(
        "rid-2b",
        {
            "error_context": "user reported by alice@example.com",
            "extra_files": {"desktop.log": f"login bob@example.com token={secret}"},
        },
    )
    assert result["result"]["ok"] is True

    files = _envelope(captured_upload["blob"])["files"]
    assert "alice@example.com" not in files["error-context.txt"]
    assert "[REDACTED_EMAIL]" in files["error-context.txt"]
    assert "bob@example.com" not in files["client/desktop.log"]
    assert secret not in files["client/desktop.log"]


def test_share_nous_linkless_success_is_a_failure(monkeypatch):
    """ok:true with neither view_url nor id would strand the user with an
    unreferencable upload — surface it as a structured failure instead."""
    import hermes_cli.diagnostics_upload as du

    monkeypatch.setattr(du, "share_to_nous", lambda blob: {})

    result = _handler()("rid-2c", {})
    payload = result["result"]
    assert payload["ok"] is False
    assert payload["error"]


def test_share_nous_extra_files_sanitized_and_redacted(captured_upload):
    secret = "sk-abc123def456ghi789jkl012mno345pqr678"
    result = _handler()(
        "rid-3",
        {
            "extra_files": {
                "desktop.log": f"boot ok\ntoken={secret}\n",
                "../../etc/passwd": "nope",
                "ok name (1).txt": "fine",
                7: "not-a-str-label",
                "empty": "   ",
            }
        },
    )
    assert result["result"]["ok"] is True

    files = _envelope(captured_upload["blob"])["files"]
    assert "client/desktop.log" in files
    assert secret not in files["client/desktop.log"]
    # Path separators are stripped from labels; traversal shapes can't survive.
    assert not any("/etc/passwd" in k or ".." in k for k in files)
    assert "client/ok name (1).txt" in files
    # Non-string labels and blank bodies are dropped.
    assert not any(k.startswith("client/7") for k in files)
    assert "client/empty" not in files


def test_share_nous_upload_failure_is_structured(monkeypatch):
    import hermes_cli.diagnostics_upload as du

    def _boom(blob: bytes) -> dict:
        raise RuntimeError("NAS unavailable")

    monkeypatch.setattr(du, "share_to_nous", _boom)

    result = _handler()("rid-4", {})
    payload = result["result"]
    assert payload["ok"] is False
    assert "NAS unavailable" in payload["error"]
