"""The trajectory summariser must not ship raw secrets to a third party.

``_generate_summary`` sends the turns being compressed to OpenRouter. Those
turns are raw tool output: an API key printed by a terminal command or read out
of a file is still verbatim in the text. Both the sync and async paths built the
prompt from ``content`` directly, so the secret left the machine.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "trajectory_compressor.py"


def _prompt_bodies() -> list[str]:
    """The f-string prompt in each _generate_summary variant."""
    text = SOURCE.read_text(encoding="utf-8")
    return re.findall(r"TURNS TO SUMMARIZE:\n\{(\w+)\}", text)


def test_both_summary_paths_exist():
    """Sync and async; a fix that misses one still leaks."""
    assert len(_prompt_bodies()) == 2


def test_no_prompt_interpolates_raw_turns():
    assert "content" not in _prompt_bodies()


def test_every_prompt_interpolates_the_redacted_copy():
    assert set(_prompt_bodies()) == {"safe_content"}


def test_redaction_is_forced():
    """force=True: the caller cannot know whether a trajectory is trusted."""
    text = SOURCE.read_text(encoding="utf-8")
    calls = re.findall(r"redact_sensitive_text\(content, ([^)]*)\)", text)
    assert len(calls) == 2
    assert all("force=True" in c for c in calls)


def test_a_secret_in_the_turns_does_not_reach_the_prompt():
    """End-to-end through the real redactor, not a mock."""
    from agent.redact import redact_sensitive_text

    secret = "sk-ant-api03-" + "A" * 40
    turns = f"$ cat .env\nANTHROPIC_API_KEY={secret}\n"
    assert secret not in redact_sensitive_text(turns, force=True)
