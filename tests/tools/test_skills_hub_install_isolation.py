"""Concurrent Skills Hub installs/scans stay isolated (#138805): each quarantine is its own tree,
the quarantine root cannot be redirected, and one profile runs one install at a time."""

import threading

import pytest

import hermes_cli.skills_hub as cli_hub
import tools.skills_hub as hub
from tools.skills_hub_install import (
    discard_quarantine, hub_install_lock, install_from_quarantine, quarantine_bundle,
)
from tools.skills_hub_models import SkillBundle


@pytest.fixture
def hub_dirs(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    hub_dir = skills_dir / ".hub"
    for name, value in {
        "SKILLS_DIR": skills_dir, "HUB_DIR": hub_dir, "LOCK_FILE": hub_dir / "lock.json",
        "QUARANTINE_DIR": hub_dir / "quarantine", "AUDIT_LOG": hub_dir / "audit.log",
        "TAPS_FILE": hub_dir / "taps.json", "INDEX_CACHE_DIR": hub_dir / "index-cache",
    }.items():
        monkeypatch.setattr(hub, name, value, raising=False)
    return skills_dir, hub_dir


def _bundle(body: str = "---\nname: demo\n---\n") -> SkillBundle:
    return SkillBundle(name="demo", files={"SKILL.md": body}, source="official",
                       identifier="official/demo", trust_level="builtin")


def test_concurrent_quarantines_of_one_name_never_share_a_tree(hub_dirs):
    first = quarantine_bundle(_bundle("---\nname: demo\n---\nfirst\n"))
    second = quarantine_bundle(_bundle("---\nname: demo\n---\nsecond\n"))

    assert first != second
    # The scanner names its result after the directory, so the bundle keeps its own name.
    assert first.name == second.name == "demo"
    assert (first / "SKILL.md").read_text(encoding="utf-8").endswith("first\n")
    assert (second / "SKILL.md").read_text(encoding="utf-8").endswith("second\n")

    discard_quarantine(first)
    assert not first.parent.exists()
    assert (second / "SKILL.md").exists()


@pytest.mark.require_symlinks
def test_symlinked_quarantine_root_is_refused(hub_dirs, tmp_path):
    _skills_dir, hub_dir = hub_dirs
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    hub_dir.mkdir(parents=True)
    (hub_dir / "quarantine").symlink_to(redirected, target_is_directory=True)

    with pytest.raises(ValueError, match="quarantine directory"):
        quarantine_bundle(_bundle())
    assert not list(redirected.iterdir())


def test_install_from_quarantine_leaves_no_staging_dir(hub_dirs):
    _skills_dir, hub_dir = hub_dirs
    q_path = quarantine_bundle(_bundle())
    result = type("Scan", (), {"verdict": "safe", "scan_provenance": None})()

    install_dir = install_from_quarantine(q_path, "demo", "", _bundle(), result)

    assert (install_dir / "SKILL.md").exists()
    assert list((hub_dir / "quarantine").iterdir()) == []


def test_install_fails_fast_while_another_install_holds_the_profile_lock(hub_dirs, monkeypatch):
    calls: list = []
    monkeypatch.setattr(cli_hub, "_install_skill", lambda *a, **k: calls.append(a) or (None, None, True))
    held, release = threading.Event(), threading.Event()

    def other_install():
        with hub_install_lock():
            held.set()
            release.wait(10)

    holder = threading.Thread(target=other_install)
    holder.start()
    try:
        assert held.wait(10)
        printed: list = []
        console = type("C", (), {"print": lambda self, *a, **k: printed.append(" ".join(map(str, a)))})()
        assert cli_hub.do_install("official/demo", console=console, skip_confirm=True) is False
        assert calls == []
        assert any("already running" in line for line in printed)
    finally:
        release.set()
        holder.join(10)

    assert cli_hub.do_install("official/demo", skip_confirm=True) is True
    assert len(calls) == 1
