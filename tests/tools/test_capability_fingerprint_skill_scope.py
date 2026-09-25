"""The capability epoch tracks the skills the model can actually invoke.

``capability_fingerprint`` walks the profile's skills tree, and a change to its digest rebuilds the
stored Bot Chat system prompt — re-prefilling the whole prompt cache behind it. It must therefore
move for a real capability change and stay still for anything that is not one.

A raw ``**/SKILL.md`` glob counted files under the dirs every other reader of this tree prunes
(``agent.skill_utils.EXCLUDED_SKILL_DIRS``: ``.archive``, ``.curator_backups``, ``node_modules`` …)
and each skill's support dirs — files ``skills_list``/``skill_view`` never offer, ``skill_count``
never counts, and the prompt's own skills index never lists. So archiving a skill, or the curator
writing a backup, flipped the epoch and rebuilt a prompt whose skills section had not changed.
"""
from __future__ import annotations

import pytest

from tools import bot_mode_probe


@pytest.fixture
def home(tmp_path):
    h = tmp_path / ".hermes"
    h.mkdir()
    return h


def _install(home, relpath: str, name: str = "s") -> None:
    d = home / "skills" / relpath
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")


def test_installing_a_real_skill_still_moves_the_epoch(home):
    before = bot_mode_probe.capability_fingerprint(home)

    _install(home, "web/scraping", "scraping")

    assert bot_mode_probe.capability_fingerprint(home) != before


def test_archiving_a_skill_does_not_move_the_epoch(home):
    _install(home, "web/scraping", "scraping")
    _install(home, "web/keep", "keep")
    before = bot_mode_probe.capability_fingerprint(home)

    # What `skills archive` leaves behind: the package moved under `.archive/`.
    src = home / "skills" / "web" / "scraping"
    dst = home / "skills" / ".archive" / "scraping"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    after_archive = bot_mode_probe.capability_fingerprint(home)

    # It DID leave the invocable set, so the epoch moves once...
    assert after_archive != before
    # ...and the archived copy sitting there is not itself a capability: re-running is stable,
    # and adding more archived packages never moves it again.
    assert bot_mode_probe.capability_fingerprint(home) == after_archive
    _install(home, ".archive/another-old-skill", "another")
    assert bot_mode_probe.capability_fingerprint(home) == after_archive
