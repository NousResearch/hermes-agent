"""Regression for #46303: Desktop session checkout attribution is per conversation."""

from hermes_cli.active_sessions import active_session_registry_snapshot
from tui_gateway import server


def test_desktop_leases_use_each_session_workspace_not_daemon_cwd(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    repo_a = tmp_path / "tts"
    repo_b = tmp_path / "engine"
    for repo in (repo_a, repo_b):
        (repo / ".git").mkdir(parents=True)

    # One Desktop backend hosts two conversations; its process cwd stays in repo A.
    monkeypatch.chdir(repo_a)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_canonical_ledger_refusal", lambda _session: None)

    sessions = []
    try:
        for name, repo in (("tts-session", repo_a), ("engine-session", repo_b)):
            session = {
                "session_key": name,
                "source": "desktop",
                "profile_home": str(home),
                "cwd": str(repo),
            }
            assert server._ensure_active_session_slot(name, session) is None
            sessions.append(session)

        roots = {
            entry["session_id"]: (entry.get("metadata") or {}).get("repo_root")
            for entry in active_session_registry_snapshot(registry_home=home)
        }
        assert roots == {
            "tts-session": str(repo_a.resolve()),
            "engine-session": str(repo_b.resolve()),
        }
    finally:
        for session in sessions:
            session["active_session_lease"].release()
