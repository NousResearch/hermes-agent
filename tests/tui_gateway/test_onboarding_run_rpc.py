"""``onboarding.state`` / ``onboarding.set_run`` read and write the ROOT profile's config.yaml, even
from a pooled backend launched under a named profile, and answer ``{run, eligible, session_id}``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tui_gateway import server
from agent.onboarding import PROFILE_BUILD_FLAG
from hermes_cli import profiles
from hermes_cli.config import read_user_config_raw
from tests.hermes_cli.anon_portal import as_local_desktop_backend


@pytest.fixture
def launched_under_work(tmp_path, monkeypatch) -> tuple[Path, Path]:
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    work = profiles.create_profile("work", no_alias=True)
    monkeypatch.setenv("HERMES_HOME", str(work))
    return root, work


def _call(method: str, params: dict | None = None) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": 5, "method": method, "params": params or {}})


def _onboarding(home: Path) -> dict:
    return read_user_config_raw(home / "config.yaml").get("onboarding") or {}


def test_state_answers_run_and_eligible(launched_under_work, monkeypatch):
    root, _work = launched_under_work
    # The user's own "work" profile is install history, so an unset flag is not due, and on the
    # desktop's local backend nothing carries a first chat: the trial is never claimed on a used machine.
    as_local_desktop_backend(monkeypatch)
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NOUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert _call("onboarding.state")["result"] == {"run": False, "eligible": False, "session_id": None, "run_id": None}

    # Onboarding runs in the default profile, so its key is what makes the run eligible.
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")
    assert _call("onboarding.state")["result"]["eligible"] is True


def test_set_run_writes_the_root_config(launched_under_work):
    root, work = launched_under_work

    assert _call("onboarding.set_run", {"run": True})["result"] == {"run": True, "eligible": False, "session_id": None, "run_id": None}
    assert _onboarding(root)["run"] is True
    assert "run" not in _onboarding(work)
    assert _call("onboarding.state")["result"]["run"] is True


def test_start_marks_the_first_chat_offer_in_the_root_config(launched_under_work):
    root, work = launched_under_work

    _call("onboarding.set_run", {"run": False, "mark_profile_offered": True})

    assert _onboarding(root)["run"] is False
    assert _onboarding(root)["seen"][PROFILE_BUILD_FLAG] is True
    assert PROFILE_BUILD_FLAG not in (_onboarding(work).get("seen") or {})


def test_skip_leaves_the_first_chat_offer_alone(launched_under_work):
    root, _work = launched_under_work

    _call("onboarding.set_run", {"run": False})

    assert _onboarding(root)["run"] is False
    assert PROFILE_BUILD_FLAG not in (_onboarding(root).get("seen") or {})


def test_a_failed_write_fails_the_call_and_writes_nothing(launched_under_work, monkeypatch):
    # The offer flag and run land in one write: if it fails the caller sees an error and neither is applied.
    from hermes_cli import config as config_mod

    root, _work = launched_under_work
    real_write = config_mod.atomic_config_write

    def refuse_the_offer_flag(path, data, **kwargs):
        if PROFILE_BUILD_FLAG in ((data.get("onboarding") or {}).get("seen") or {}):
            raise OSError("disk full")
        real_write(path, data, **kwargs)

    monkeypatch.setattr(config_mod, "atomic_config_write", refuse_the_offer_flag)

    assert "error" in _call("onboarding.set_run", {"run": False, "mark_profile_offered": True})
    assert "run" not in _onboarding(root)
    assert PROFILE_BUILD_FLAG not in (_onboarding(root).get("seen") or {})


def test_set_run_requires_a_bool(launched_under_work):
    assert "error" in _call("onboarding.set_run", {})
    assert "error" in _call("onboarding.set_run", {"run": "yes"})
