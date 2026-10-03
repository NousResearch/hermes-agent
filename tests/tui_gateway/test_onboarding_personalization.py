"""profiles.remember_onboarding must fit USER.md's real budget and replace its own earlier entry.

USER.md is capped by ``memory.user_char_limit`` (1,375 by default) and already holds what the agent learned.
Before the fix the writer only checked a fixed 2,000-char cap, so a near-full USER.md failed the first build
with the model-facing "Consolidate now" error on every retry, and re-running onboarding added a second,
conflicting "Agreed during onboarding" entry.
"""

import pytest

from hermes_cli.profiles import get_profile_dir
from tools.memory_tool import ENTRY_DELIMITER
from tui_gateway.onboarding_personalization import remember_onboarding

ANSWERS = {"name": "Alex", "context": "ERP accounting sync", "theme": "dark", "accent": "violet",
           "layout": "wide", "focus": ["coding", "automation"], "connectors": ["Gmail", "Slack", "GitHub"]}


def _user_md():
    return get_profile_dir("default") / "memories" / "USER.md"


def _write_user_md(entries: list[str]) -> None:
    _user_md().parent.mkdir(parents=True, exist_ok=True)
    _user_md().write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def _entries() -> list[str]:
    return _user_md().read_text(encoding="utf-8").split(ENTRY_DELIMITER)


def test_near_full_user_md_keeps_the_facts_that_fit():
    existing = ["u" * 400, "v" * 400, "w" * 394]  # 1,200/1,375 chars
    _write_user_md(existing)

    assert remember_onboarding(ANSWERS)["saved"] is True
    assert remember_onboarding(ANSWERS)["saved"] is True  # "Retry first build" must not fail either

    entries = _entries()
    assert entries[:3] == existing
    assert len(entries) == 4 and entries[3].startswith("Agreed during onboarding:\nUser prefers to be called: Alex")
    assert len(ENTRY_DELIMITER.join(entries)) <= 1375


def test_rerunning_onboarding_replaces_the_earlier_entry():
    _write_user_md(["Prefers short answers"])

    remember_onboarding({"name": "Alex", "theme": "dark"})
    remember_onboarding({"name": "Alex", "theme": "light"})

    assert _entries() == ["Prefers short answers",
                          "Agreed during onboarding:\nUser prefers to be called: Alex\nDesktop theme: light"]


def test_full_user_md_fails_with_a_user_facing_message():
    _write_user_md(["x" * 1370])

    with pytest.raises(ValueError, match=r"USER\.md is full \(1,370/1,375 chars\)") as excinfo:
        remember_onboarding(ANSWERS)
    assert "Consolidate" not in str(excinfo.value)
    assert _entries() == ["x" * 1370]
