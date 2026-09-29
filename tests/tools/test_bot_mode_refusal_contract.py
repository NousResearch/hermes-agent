import json
import sys
from types import SimpleNamespace

from tools import bot_mode_dm


def test_delivery_uses_refusal_code_before_human_wording(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    for code, message, busy in (
        ("SESSION_NOT_OWNED", "Ce chat est occupé.", True),
        ("SESSION_COORDINATION_UNAVAILABLE", "Cannot verify whether session already has a live owner", False),
        ("SESSION_NOT_OWNED_EXTRA", "Different failure", False),
    ):
        dm = tmp_path / "message.txt"
        dm.write_text("isolated probe", encoding="utf-8")
        child = tmp_path / "child.py"
        child.write_text(f"import sys\nprint({f'hermes-refusal-reason: {code}'!r}, file=sys.stderr)\nprint({message!r}, file=sys.stderr)\nraise SystemExit(1)\n", encoding="utf-8")
        rc = bot_mode_dm._run_delivery([sys.executable, str(child)], str(dm), stdin_file=False)
        output = capsys.readouterr()
        if busy:
            # #116210: the refusal code still wins over the raw wording, but a recognized
            # SESSION_NOT_OWNED busy target is now queued for replay, not hard-failed.
            assert rc == 0
            assert json.loads(output.out)["status"] == "queued_busy"
        else:
            assert rc == 1
            assert not output.out
        assert not dm.exists()


def test_one_shot_cli_preserves_refusal_reason(monkeypatch, capsys):
    from cli import HermesCLI
    from hermes_cli import active_sessions
    refusal = active_sessions.ActiveSessionRefusal("Ce chat est occupé.", reason=active_sessions.SESSION_NOT_OWNED)
    monkeypatch.setattr(active_sessions, "try_acquire_active_session", lambda **kwargs: (None, refusal))
    cli = SimpleNamespace(_active_session_lease=None, session_id="isolated", config={})
    assert not HermesCLI._claim_active_session(cli, stderr=True)
    assert "hermes-refusal-reason: SESSION_NOT_OWNED\n" in capsys.readouterr().err
