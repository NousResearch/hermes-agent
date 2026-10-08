"""Every sanitized skill root must survive until its whole mount batch is ready."""

from pathlib import Path

from tools import credential_files
from tools.environments import skill_snapshot


def test_multiple_symlinked_skill_roots_remain_mountable(monkeypatch, tmp_path):
    roots = [tmp_path / "profile-skills", tmp_path / "external-skills"]
    for root, name in zip(roots, ("trading-agents", "other-skill")):
        skill = root / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(name, encoding="utf-8")
        (root / "ignored-link").symlink_to(tmp_path / "outside")

    monkeypatch.setattr(credential_files, "_safe_skills_tempdirs", [])
    monkeypatch.setattr(
        credential_files, "_skill_dir_roots",
        lambda _base: iter(zip(roots, ("/root/.hermes/skills", "/root/.hermes/external_skills/0"))),
    )
    monkeypatch.setattr(skill_snapshot, "safe_to_delete", lambda _path: True)

    first = credential_files.get_skills_directory_mount()
    assert len(first) == 2
    first_paths = [Path(item["host_path"]) for item in first]
    for root, path, name in zip(roots, first_paths, ("trading-agents", "other-skill")):
        assert path != root and path.is_dir()
        assert (path / name / "SKILL.md").read_text(encoding="utf-8") == name
        assert not (path / "ignored-link").exists()

    second = credential_files.get_skills_directory_mount()
    second_paths = [Path(item["host_path"]) for item in second]
    assert all(path.is_dir() for path in second_paths)
    assert all(not path.exists() for path in first_paths)
    for path in second_paths:
        credential_files._delete_safe_skills_copy(path)
