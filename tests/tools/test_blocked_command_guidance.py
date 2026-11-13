"""Tests for blocked-command recovery guidance (parser-limit + backgrounding)."""

import pytest

from tools.approval import _hardline_block_result
from tools.approval_detection import _PARSER_LIMIT_DESCRIPTION
from tools.terminal_tool import _foreground_background_guidance
from tools import approval_floors


class TestParserLimitRecovery:
    def test_parser_limit_block_saves_payload_and_names_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        cmd = "python3 -c '" + "x = 1; " * 900 + "'"
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, cmd)
        assert r["approved"] is False
        assert "RECOVERY" in r["message"]
        assert "blocked-scripts" in r["message"]
        import re as _re
        m = _re.search(r"saved to (\S+\.sh)", r["message"])
        assert m, r["message"]
        from pathlib import Path
        saved = Path(m.group(1))
        assert saved.exists()
        body = saved.read_text()
        assert cmd in body
        assert body.startswith("#!/usr/bin/env bash")
        assert f"bash {saved}" in r["message"]

    def test_save_failure_falls_back_to_manual_recipe(self, monkeypatch):
        monkeypatch.setattr(approval_floors, "_save_blocked_payload", lambda c: None)
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, "python3 -c 'x'")
        assert "write_file" in r["message"]
        assert "bash /path/script.sh" in r["message"]



    def test_real_hardline_blocks_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = _hardline_block_result("recursive delete of root filesystem", "rm -rf --no-preserve-root /")
        assert "RECOVERY" not in r["message"]
        assert "unconditional blocklist" in r["message"]
        # And nothing was saved for a genuine hardline block.
        assert not (tmp_path / ".hermes" / "cache" / "blocked-scripts").exists()

    def test_old_saved_payloads_cleaned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        import os
        d = tmp_path / ".hermes" / "cache" / "blocked-scripts"
        d.mkdir(parents=True)
        stale = d / "blocked-1-dead.sh"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, "python3 -c 'y'")
        assert not stale.exists()


class TestBackgroundGuidanceRecipes:
    def test_ampersand_block_names_exact_call_shape(self):
        msg = _foreground_background_guidance("python3 server.py &")
        assert msg is not None
        assert "WITHOUT the '&'" in msg
        assert "background=true" in msg

    def test_nohup_block_names_exact_call_shape(self):
        msg = _foreground_background_guidance("nohup ./worker.sh > /dev/null 2>&1")
        assert msg is not None
        assert "WITHOUT the wrapper" in msg
        assert "notify_on_complete=true" in msg

    def test_plain_command_unaffected(self):
        assert _foreground_background_guidance("echo hello") is None

    def test_quoted_ampersand_not_flagged(self):
        assert _foreground_background_guidance('git commit -m "a & b"') is None


class TestRepeatedBackgroundRejection:
    """Through terminal_tool: the fg/bg guidance escalates from the 3rd rejection per agent
    and turn, and only a background=true call resets it."""

    # Shapes the terminal still refuses (a lone trailing `&` now runs as background=true).
    AMP = "nohup python3 server.py > server.log 2>&1 &"
    FIXED_CALL = ('terminal(command="python3 server.py > server.log 2>&1", background=true, '
                  'notify_on_complete=true)')
    FAKE_START = {"output": "Background process started", "session_id": "proc_fake", "pid": 1,
                  "exit_code": 0, "error": None}

    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        import json
        from unittest.mock import patch
        from tools.terminal_tool import terminal_tool

        config = {"env_type": "local", "timeout": 30, "cwd": str(tmp_path), "host_cwd": None,
                  "modal_mode": "auto", "docker_image": "", "singularity_image": "",
                  "modal_image": "", "daytona_image": ""}
        spawned = []

        def fake_spawn(**kw):  # never start a real background process
            spawned.append(kw)
            return json.dumps(self.FAKE_START)

        with patch("tools.terminal_tool._get_env_config", return_value=config), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}), \
             patch("tools.terminal_tool.spawn_background_process", side_effect=fake_spawn):
            yield lambda command, **kw: json.loads(terminal_tool(command=command, **kw))

    @staticmethod
    def _escalated(result) -> bool:
        return "REPEATED REJECTION" in (result.get("error") or "")

    @pytest.fixture(autouse=True)
    def session_key(self):
        import uuid
        from tools.approval_context import (
            reset_current_observability_context, reset_current_session_key,
            set_current_observability_context, set_current_session_key,
        )
        key = f"conv-{uuid.uuid4().hex}"
        key_token = set_current_session_key(key)
        obs_tokens = set_current_observability_context(turn_id="turn-1")
        yield key
        reset_current_observability_context(obs_tokens)
        reset_current_session_key(key_token)

    def test_third_rejection_names_the_corrected_call(self, run):
        results = [run(self.AMP, task_id="t") for _ in range(3)]
        assert [self._escalated(r) for r in results] == [False, False, True]
        assert results[0]["error"] == results[1]["error"]
        assert self.FIXED_CALL in results[2]["error"]
        assert "different approach" not in results[2]["error"]

    def test_only_background_true_resets(self, run):
        run(self.AMP, task_id="t")
        assert run("echo ok", task_id="t")["exit_code"] == 0
        run(self.AMP, task_id="t")
        assert run("echo ok", task_id="t")["exit_code"] == 0
        assert self._escalated(run(self.AMP, task_id="t"))
        assert run("true", background=True, task_id="t").get("session_id")
        assert not self._escalated(run(self.AMP, task_id="t"))

    def test_no_task_id_never_escalates(self, run):
        assert not any(self._escalated(run(self.AMP)) for _ in range(4))

    def test_delegated_child_counts_apart_from_parent(self, run):
        from agent.delegation_context import delegated_child_context

        run(self.AMP, task_id="t")
        with delegated_child_context("child-1"):
            child = [run(self.AMP, task_id="child-t") for _ in range(2)]
        parent = run(self.AMP, task_id="t")
        assert not any(self._escalated(r) for r in [*child, parent])

    def test_new_turn_starts_a_new_count(self, run):
        from tools.approval_context import set_current_observability_context

        run(self.AMP, task_id="t")
        run(self.AMP, task_id="t")
        set_current_observability_context(turn_id="turn-2")  # restored by the session_key fixture
        assert not self._escalated(run(self.AMP, task_id="t"))

    def test_session_end_drops_parent_and_child_counters(self, run, session_key):
        from agent.delegation_context import delegated_child_context
        from tools.approval import clear_session

        def reject_twice():
            run(self.AMP, task_id="t")
            run(self.AMP, task_id="t")

        reject_twice()
        with delegated_child_context("child-1"):
            reject_twice()
        clear_session(session_key)
        assert not self._escalated(run(self.AMP, task_id="t"))
        with delegated_child_context("child-1"):
            assert not self._escalated(run(self.AMP, task_id="t"))

    @pytest.mark.parametrize("command,expected", [
        ("nohup ./worker.sh > log 2>&1 &", "./worker.sh > log 2>&1"),
        ('echo "a & b" &', 'echo "a & b"'),
        ("npm run dev", "npm run dev"),
        ("a & b", None),              # inline '&': two commands, not a mechanical fix
        ("sleep 9 & disown", None),
        ("setsid -f ./srv", None),
        ("x" * 300 + " &", None),     # too long to quote back
        ("python3 -m http.server 8000", "python3 -m http.server 8000"),
        # Shapes the trailing-& conversion refuses never get an exact resend.
        ('echo "$(sleep 9 & echo $!)" &', None),  # hidden second background job
        ("a; b &", None),
        ("nohup a; b &", None),
        ("hermes gateway run &", None),            # approval keys on the '&' itself
        ("nohup hermes gateway run", None),
    ])
    def test_corrected_command_is_only_offered_when_mechanical(self, command, expected):
        from tools.terminal_tool_guards import _backgrounded_command

        assert _backgrounded_command(command) == expected

    def test_exact_resend_keeps_the_rejected_calls_workdir(self, run, tmp_path):
        workdir = str(tmp_path / "srv app")
        results = [run("npm run dev", workdir=workdir, task_id="t") for _ in range(3)]
        assert [self._escalated(r) for r in results] == [False, False, True]
        assert results[0]["error"] == results[1]["error"]
        expected = (f'Resend exactly this: terminal(command="npm run dev", workdir="{workdir}", background=true, '
                    'notify_on_complete=true)')
        assert expected in results[2]["error"]

    def test_unrenderable_workdir_falls_back_to_the_placeholder(self, run, tmp_path):
        workdir = str(tmp_path / "a\nb")
        results = [run("npm run dev", workdir=workdir, task_id="t") for _ in range(3)]
        assert "Resend exactly this" not in results[2]["error"]
        assert ('terminal(command="<cmd>", workdir=<same workdir>, background=true, '
                'notify_on_complete=true)') in results[2]["error"]

    def test_escalation_uses_the_placeholder_when_no_exact_resend_exists(self, run):
        command = 'echo "$(sleep 9 & echo $!)" &'
        results = [run(command, task_id="t") for _ in range(3)]
        assert [self._escalated(r) for r in results] == [False, False, True]
        assert 'terminal(command="<cmd>", background=true, notify_on_complete=true)' in results[2]["error"]
        assert "Resend exactly this" not in results[2]["error"]

    def test_auto_background_is_not_counted(self, run):
        from tools import terminal_tool_guards as guards

        for _ in range(3):
            assert run("python3 server.py &", task_id="t")["session_id"] == "proc_fake"
        assert guards._fg_bg_owner("t") not in guards._FG_BG_REJECTIONS
        assert not self._escalated(run(self.AMP, task_id="t"))

    def test_unbound_turn_keeps_plain_guidance(self, run):
        from tools import terminal_tool_guards as guards
        from tools.approval_context import set_current_observability_context

        set_current_observability_context(turn_id="")  # restored by the session_key fixture
        results = [run(self.AMP, task_id="t") for _ in range(4)]
        assert not any(self._escalated(r) for r in results)
        assert {r["error"] for r in results} == {_foreground_background_guidance(self.AMP)}
        assert guards._fg_bg_owner("t") not in guards._FG_BG_REJECTIONS

    def test_owner_table_is_bounded(self, monkeypatch):
        from tools import terminal_tool_guards as guards

        monkeypatch.setattr(guards, "_FG_BG_REJECTIONS", {})
        monkeypatch.setattr(guards, "_FG_BG_MAX_OWNERS", 3)
        monkeypatch.setattr(guards, "_fg_bg_owner", lambda task_id: task_id)
        for i in range(10):
            guards._foreground_background_guidance(self.AMP, task_id=f"task-{i}")
        assert list(guards._FG_BG_REJECTIONS) == ["task-7", "task-8", "task-9"]
