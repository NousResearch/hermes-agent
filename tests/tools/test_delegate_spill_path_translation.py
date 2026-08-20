"""Truncation footers must surface backend-readable cache paths, not host paths.

When the terminal backend is docker/modal/ssh the cache dirs are mounted at a
different path than the host path (e.g. /root/.hermes for docker), so a footer
advertising the host path makes read_file from inside the sandbox fail with
"File not found" and forces unnecessary re-dispatch.
"""

import os
import tempfile

import pytest

import tools.delegate_tool as dt


def test_delegate_summary_footer_translates_path_for_docker_backend(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        model_text, spill_path = dt._trim_summary_with_footer("X" * 50_000, 1000, 0)

        assert spill_path.startswith("/root/.hermes/"), spill_path
        assert str(td) not in spill_path
        assert "Full subagent output saved to: " in model_text
        assert f'read_file path="{spill_path}"' in model_text


def test_delegate_summary_footer_keeps_host_path_on_local_backend(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        model_text, spill_path = dt._trim_summary_with_footer("X" * 50_000, 1000, 0)

        assert spill_path.startswith(os.path.join(td, ".hermes")), spill_path
        assert os.path.exists(spill_path)
        assert f'read_file path="{spill_path}"' in model_text


def test_delegate_live_transcripts_translate_for_docker_backend(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        from tools.credential_files import to_agent_visible_cache_path
        from tools.delegation_live_log import create_live_transcripts

        _, _, paths = create_live_transcripts([{"goal": "hello"}], delegation_id="test-xyz")
        if not paths:
            pytest.skip("live transcripts disabled in this environment")
        # delegate_tool translates each display path through the same helper.
        visible = [to_agent_visible_cache_path(p) for p in paths]
        for p in visible:
            assert p.startswith("/root/.hermes/"), p
            assert str(td) not in p
