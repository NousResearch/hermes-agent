"""Regression for #110123: command suffixes are not secret reads."""

import pytest

from tools.threat_patterns import scan_for_threats


@pytest.mark.parametrize("scope", ["all", "context", "strict"])
@pytest.mark.parametrize("command", ["pw-cat", "logcat", "concat", "my.cat", "my_cat"])
def test_secret_read_requires_a_command_boundary(command, scope):
    assert "read_secrets" not in scan_for_threats(
        f"Play with `{command} --playback`; toggle systemd.environment", scope=scope
    )
    # A later actual command must still be scanned, even on the same line.
    for read in ["cat .env", "/bin/cat /app/.env", "cat ~/.aws/credentials",
                 "cat .netrc", "cat .pgpass", "cat .npmrc", "cat .pypirc",
                 "ｃａｔ .env"]:
        assert "read_secrets" in scan_for_threats(
            f"{command} --playback; {read}", scope=scope
        )


def test_project_context_keeps_command_documentation_but_blocks_real_reads(tmp_path):
    from agent.prompt_builder import build_context_files_prompt

    (tmp_path / ".git").mkdir()
    context = tmp_path / "AGENTS.md"
    documentation = "Play with `pw-cat --playback --raw`; toggle systemd.environment"
    context.write_text(documentation, encoding="utf-8")
    result = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
    assert documentation in result
    assert "[BLOCKED:" not in result

    context.write_text("cat /app/.env", encoding="utf-8")
    result = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
    assert "[BLOCKED: AGENTS.md" in result
    assert "read_secrets" in result
