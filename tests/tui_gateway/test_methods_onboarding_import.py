"""Spawned chat must survive the onboarding methods import and keep its explicit model.

Regression for the issue-139446 defect class. ``tui_gateway/methods_onboarding.py``
annotated ``_shared_nous_store() -> tuple[Path, ...]`` without importing ``Path``, so
the annotation blew up at module-import (def) time. Every spawned ``hermes chat``
rides that import chain (CLI ``/api/ws`` ticket handshake → ``gateway/run_api`` →
``tui_gateway.ws`` → ``tui_gateway.server`` → ``methods_onboarding``), so the ASGI
handshake died with NameError before any session existed and the CLI reported
``Gateway connection failed; no local fallback (InvalidStatus)`` — no session row,
no turn, no provider call, no warning outlet.

Contracts under test (design doc 契约规约):

* C3 — the module imports cleanly, and its ``_due_stamp()`` is callable against an
  isolated HERMES_HOME and returns a tuple (the latent call-time NameError path).
* C1 — a spawned one-shot chat with an explicit ``-m <model>`` and a named custom
  ``--provider`` runs end to end over a real gateway daemon: exit 0, the persisted
  session row, the fake provider's received wire requests and the user-visible
  reply all carry the REQUESTED model — never the profile default (no silent
  rerouting). Per-session usage-ledger attribution (1.P4) is out of this file's
  pinned scope: the single-writer daemon flushes it on its own schedule, not
  synchronously with the CLI's exit.
"""

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PROVIDER = "zen-relay"  # named custom provider; must not shadow a builtin name
EXPLICIT_MODEL = "zen-relay-fast"
DEFAULT_MODEL = "guard-default-must-not-route"


def test_methods_onboarding_imports_and_due_stamp_is_callable(tmp_path, monkeypatch):
    """C3: the import chain every spawned chat rides must not raise NameError, and the
    production ``_due_stamp()`` must run against an isolated HERMES_HOME and return a tuple
    (the latent call-time NameError path). ``bind_module`` publishes rebound handler copies
    onto the server namespace — the module's own attribute keeps the module globals, where
    the server-only helpers (``_profile_home`` & co) don't exist — so the bound copy is the
    one production dispatches through."""
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tui_gateway.methods_onboarding  # the crash site itself: def-time annotation at :72
    import tui_gateway.server  # /api/ws ticket handshake → ws → server:2146 → methods_onboarding

    assert tui_gateway.methods_onboarding.__name__ == "tui_gateway.methods_onboarding"
    stamp = tui_gateway.server._due_stamp()
    assert isinstance(stamp, tuple), f"_due_stamp() returned {type(stamp).__name__}: {stamp!r}"


def test_spawned_one_shot_chat_keeps_explicit_model_and_named_provider(tmp_path, model_peer):
    """C1 (one-shot ``-Q`` + ``--query-file`` form): with a profile default of a
    DIFFERENT model on the SAME named provider, an explicit ``--provider``/``-m``
    spawn must reach the fake provider and persist under the requested model — the
    profile default must appear nowhere, on the wire or in state.db."""
    home = tmp_path / "state"
    home.mkdir(mode=0o700)
    user = tmp_path / "user"
    user.mkdir()
    url = f"http://127.0.0.1:{model_peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        # Default route = same named provider, a DIFFERENT model: any silent reroute
        # of the spawn's explicit flags is then visible as DEFAULT_MODEL downstream.
        "model": {"provider": f"custom:{PROVIDER}", "default": DEFAULT_MODEL},
        "custom_providers": [{"name": PROVIDER, "base_url": url, "api_key": "loopback-only",
                              "api_mode": "chat_completions"}],
        "auxiliary": {"title_generation": {"enabled": False}},
        "memory": {"memory_enabled": False, "user_profile_enabled": False, "nudge_interval": 0},
        "skills": {"creation_nudge_interval": 0},
        "updates": {"check": False},
        "models_dev": {"url": "http://127.0.0.1:9/api.json"},
        "terminal": {"env": "local", "oneshot_completion_wait_seconds": 1},
    }), encoding="utf-8")
    query = tmp_path / "spawned-query.txt"
    query.write_text("Reply with WS_SHARED exactly once.", encoding="utf-8")
    # Retain native host location variables; never inherited credentials or another
    # session's runtime markers.
    env = {key: os.environ[key] for key in (
        "PATH", "LANG", "TZ", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
        "PATHEXT", "LOCALAPPDATA", "APPDATA",
    ) if key in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               HERMES_MANAGED_DIR=str(tmp_path / "managed"), TERMINAL_CWD=str(tmp_path),
               PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable, "-c", "from hermes_cli.main import main; main()", "chat", "-Q",
        "--provider", PROVIDER, "-m", EXPLICIT_MODEL, "--query-file", str(query),
    ]
    from tests.gateway.fixtures.local_recovery_probe import daemon

    with daemon(REPO_ROOT, home, env, barrier=False) as (owner, _descriptor):
        result = subprocess.run(command, cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, encoding="utf-8", timeout=90,
                                check=False)
        # The daemon served the whole turn; an ASGI-level crash mid-handshake kills it.
        assert owner.poll() is None, "gateway daemon exited during the spawn chat"
    diagnostic = (result.stdout[-2500:], result.stderr[-4000:])

    assert result.returncode == 0, diagnostic
    # The turn really executed end to end and the user saw the reply (1.P1).
    assert "LOCAL_ACK_WS_SHARED" in result.stdout, diagnostic
    assert "InvalidStatus" not in result.stderr, diagnostic
    assert "Gateway connection failed" not in result.stderr, diagnostic
    # Main's -Q prints `session_id: <id>` on stderr.
    session_id = next((line.split("session_id: ", 1)[1].strip()
                       for line in result.stderr.splitlines() if line.startswith("session_id: ")),
                      None)
    assert session_id, diagnostic

    with closing(sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)) as db:
        row = db.execute("SELECT model FROM sessions WHERE id=?", (session_id,)).fetchone()
        rerouted = db.execute("SELECT id FROM sessions WHERE model=?",
                              (DEFAULT_MODEL,)).fetchall()
        # Gateway-only ledger: a non-empty terminal admission is what pins this turn to
        # the real daemon session-authority path instead of a local fallback.
        admissions = db.execute("SELECT status FROM session_admissions").fetchall()
    assert row is not None, f"session {session_id} never persisted\n{diagnostic}"
    assert row[0] == EXPLICIT_MODEL, f"session row model {row[0]!r}\n{diagnostic}"
    assert not rerouted, f"sessions persisted under the profile default: {rerouted}"
    assert admissions and all(status == "terminal" for status, in admissions), admissions
    assert model_peer.requests, f"the fake provider received no completion request\n{diagnostic}"
    received = {request.get("model") for request in model_peer.requests}
    assert received == {EXPLICIT_MODEL}, sorted(filter(None, received))
    assert not any(request.get("model") == DEFAULT_MODEL for request in model_peer.requests)
    # 1.P5 narrowed to the defect class's crash signature: this tree's daemon log
    # legitimately carries an unrelated fail-open relay-plugin TypeError, but an
    # onboarding-module import crash is always a NameError on the /api/ws chain.
    daemon_log = (home / "restart.log").read_text(encoding="utf-8", errors="replace")
    assert "NameError" not in daemon_log, daemon_log[-4000:]
