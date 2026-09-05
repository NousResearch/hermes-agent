"""Tests for blocked-command recovery guidance (parser-limit + backgrounding)."""

import pytest

from tools.approval import _hardline_block_result
from tools.approval_detection import _PARSER_LIMIT_DESCRIPTION, _MALFORMED_EXEC_DESCRIPTION
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
        assert body.startswith("#!/bin/bash")
        assert f"bash {saved}" in r["message"]

    def test_save_failure_falls_back_to_manual_recipe(self, monkeypatch):
        import tools.approval as ap
        from tools import approval_floors
        monkeypatch.setattr(approval_floors, "_save_blocked_payload", lambda c: None)
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, "python3 -c 'x'")
        assert "write_file" in r["message"]
        assert "bash /path/script.sh" in r["message"]

    def test_no_command_falls_back_to_manual_recipe(self):
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION)
        assert "RECOVERY" in r["message"]
        assert "write_file" in r["message"]

    def test_malformed_exec_block_has_recovery_recipe(self):
        r = _hardline_block_result(_MALFORMED_EXEC_DESCRIPTION)
        assert "RECOVERY" in r["message"]

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

    def test_inspection_of_uvicorn_is_not_server_start(self):
        """`ps aux | grep uvicorn` inspects; the server word is only a grep
        operand, so the command must not be nudged to background mode."""
        assert _foreground_background_guidance("ps aux | grep uvicorn | grep -v grep | head -3") is None
        assert _foreground_background_guidance(
            "journalctl --user -u app-server -n 5 | grep uvicorn"
        ) is None
        assert _foreground_background_guidance(
            "kill 4158132 && sleep 2 && ps aux | grep uvicorn | grep -v grep | wc -l"
        ) is None

    def test_real_python_uvicorn_start_still_flagged(self):
        assert _foreground_background_guidance(
            "cd ~/proj && .venv/bin/python -m uvicorn src.server.main:app --port 8000"
        ) is not None
        # `timeout <N>` bounds the process — a smoke/restart test, not a
        # long-lived foreground server.
        assert _foreground_background_guidance(
            "cd ~/proj && timeout 15 .venv/bin/python -m uvicorn src.server.main:app --port 8001"
        ) is None

    def test_timeout_bounded_server_start_not_flagged(self):
        """A `timeout <N>` wrapper makes the command bounded, so even a server
        start cannot occupy the foreground forever."""
        assert _foreground_background_guidance(
            "systemctl --user stop app-webui.service 2>/dev/null; sleep 2; cd ~/proj && timeout 15 .venv/bin/python -m uvicorn src.server.main:app --host 0.0.0.0 --port 8001"
        ) is None
        assert _foreground_background_guidance(
            "cd ~/proj && MOCK=1 timeout 8 .venv/bin/python -m uvicorn src.server.main:app --host 127.0.0.1 --port 8011"
        ) is None
        assert _foreground_background_guidance("timeout 60s npm run dev") is None
        # unbounded still flagged
        assert _foreground_background_guidance("npm run dev") is not None

    def test_vite_build_is_not_server_start(self):
        """`vite build` is a one-shot build, not a long-lived dev server.
        Bare `vite` / `vite dev` are dev servers and stay flagged."""
        assert _foreground_background_guidance(
            "cd ~/proj/web && npx vite build 2>&1"
        ) is None
        assert _foreground_background_guidance("npx vite build") is None
        assert _foreground_background_guidance("vite") is not None
        assert _foreground_background_guidance(
            "cd ~/proj && npx vite dev"
        ) is not None

    def test_npm_docker_starts_still_flagged(self):
        assert _foreground_background_guidance("cd ~/proj && npm run dev") is not None
        assert _foreground_background_guidance("docker compose up") is not None

    def test_detached_docker_compose_up_is_not_server_start(self):
        """`docker compose up -d|--detach` returns immediately (daemon-owned),
        so the command line itself is NOT a long-lived foreground process."""
        assert _foreground_background_guidance("docker compose up -d") is None
        assert _foreground_background_guidance("docker compose up --detach") is None
        assert _foreground_background_guidance("docker compose up -d --force-recreate tei-reranker") is None
        assert _foreground_background_guidance(
            'cd ~/app && docker compose down 2>&1 && echo "=====UP=====" && docker compose up -d 2>&1'
        ) is None
        assert _foreground_background_guidance(
            "cd ~/app && docker compose up -d 2>&1 | tail -5 && sleep 15 && docker compose ps"
        ) is None
        # bare `up` (no detach) still keeps the foreground attached — guard stays
        assert _foreground_background_guidance(
            "cd ~/app && docker compose down && docker compose up"
        ) is not None
        # inspection with `up` as a grep operand is not a compose start either
        assert _foreground_background_guidance("docker ps | grep up") is None
