import json
import sys
from types import SimpleNamespace

from tools import bot_mode_dm


def test_delivery_uses_refusal_code_before_human_wording(tmp_path, capsys):
    for code, message, busy in (
        ("SESSION_NOT_OWNED", "Ce chat est occupé.", True),
        ("SESSION_COORDINATION_UNAVAILABLE", "Cannot verify whether session already has a live owner", False),
        ("SESSION_NOT_OWNED_EXTRA", "Different failure", False),
    ):
        dm = tmp_path / "message.txt"
        dm.write_text("isolated probe", encoding="utf-8")
        child = tmp_path / "child.py"
        child.write_text(f"import sys\nprint({f'hermes-refusal-reason: {code}'!r}, file=sys.stderr)\nprint({message!r}, file=sys.stderr)\nraise SystemExit(1)\n", encoding="utf-8")
        assert bot_mode_dm._run_delivery([sys.executable, str(child)], str(dm), stdin_file=False) == 1
        output = capsys.readouterr()
        assert (json.loads(output.out)["reason"] == "target_busy") if busy else not output.out
        assert not dm.exists()


def test_one_shot_cli_preserves_refusal_reason(monkeypatch, capsys):
    from cli import HermesCLI
    from hermes_cli import active_sessions
    refusal = active_sessions.ActiveSessionRefusal("Ce chat est occupé.", reason=active_sessions.SESSION_NOT_OWNED)
    monkeypatch.setattr(active_sessions, "try_acquire_active_session", lambda **kwargs: (None, refusal))
    cli = SimpleNamespace(_active_session_lease=None, session_id="isolated", config={})
    assert not HermesCLI._claim_active_session(cli, stderr=True)
    assert "hermes-refusal-reason: SESSION_NOT_OWNED\n" in capsys.readouterr().err
