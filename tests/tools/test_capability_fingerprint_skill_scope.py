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

from agent.skill_utils import EXCLUDED_SKILL_DIRS
from tools import bot_mode_probe


@pytest.fixture(autouse=True)
def _fresh_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


@pytest.fixture
def home(tmp_path):
    h = tmp_path / ".hermes"
    (h / "profiles" / "researcher").mkdir(parents=True)
    (h / "profiles" / "researcher" / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    managed: true\n", encoding="utf-8")
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


@pytest.mark.parametrize("excluded", sorted(EXCLUDED_SKILL_DIRS))
def test_no_excluded_directory_can_move_the_epoch(home, excluded):
    _install(home, "web/keep", "keep")
    before = bot_mode_probe.capability_fingerprint(home)

    _install(home, f"{excluded}/pkg", "hidden")

    assert bot_mode_probe.capability_fingerprint(home) == before, excluded


def test_a_curator_backup_does_not_rebuild_every_bot_chat_prompt(home):
    """The curator writes into `.curator_backups/` on its own schedule — routine churn that must
    not invalidate a months-long Bot Chat's prompt cache."""
    _install(home, "web/keep", "keep")
    before = bot_mode_probe.capability_fingerprint(home)

    _install(home, ".curator_backups/2026-09-21T00-00-00Z/web/keep", "keep")

    assert bot_mode_probe.capability_fingerprint(home) == before


def test_a_skill_support_dir_is_not_a_second_skill(home):
    """`references/` inside a skill package is progressive-disclosure material, not an install."""
    _install(home, "web/keep", "keep")
    before = bot_mode_probe.capability_fingerprint(home)

    _install(home, "web/keep/references/deep-dive", "deep-dive")

    assert bot_mode_probe.capability_fingerprint(home) == before


def test_the_epoch_tracks_exactly_what_the_skills_walker_reports(home):
    """The fingerprint and every other reader of the tree must agree on what is installed."""
    from agent.skill_utils import iter_skill_index_files

    for rel in ("web/keep", "ops/deploy", ".archive/old", ".curator_backups/x/web/keep"):
        _install(home, rel, rel.rsplit("/", 1)[-1])

    walker = sorted(str(p.parent.relative_to(home / "skills"))
                    for p in iter_skill_index_files(home / "skills", "SKILL.md"))

    assert walker == ["ops/deploy", "web/keep"]
    # Same surface: removing a skill the walker reports moves the epoch, removing one it does not
    # report leaves it alone.
    before = bot_mode_probe.capability_fingerprint(home)
    (home / "skills" / ".archive" / "old" / "SKILL.md").unlink()
    assert bot_mode_probe.capability_fingerprint(home) == before
    (home / "skills" / "ops" / "deploy" / "SKILL.md").unlink()
    assert bot_mode_probe.capability_fingerprint(home) != before
