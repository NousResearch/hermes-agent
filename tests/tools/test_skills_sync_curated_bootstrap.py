from pathlib import Path


def _write_skill(root: Path, rel: str, name: str, body: str = "body") -> Path:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir):
    import tools.skills_sync as sync

    skills_dir = tmp_path / "home" / "skills"
    monkeypatch.setattr(sync, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sync, "MANIFEST_FILE", skills_dir / ".bundled_manifest")
    monkeypatch.setattr(sync, "_get_bundled_dir", lambda: bundled_dir)
    monkeypatch.setattr(sync, "_get_optional_dir", lambda: optional_dir)
    return sync, skills_dir


def test_default_bootstrap_mode_seeds_all(monkeypatch, tmp_path):
    """With HERMES_SKILLS_BOOTSTRAP unset, the default is seed-all — the
    long-standing behavior, so existing users see no change on update."""
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "software-development/plan", "plan")
    _write_skill(bundled_dir, "gaming/pokemon-player", "pokemon-player")
    _write_skill(optional_dir, "devops/watchers", "watchers")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)

    monkeypatch.delenv("HERMES_SKILLS_BOOTSTRAP", raising=False)
    result = sync.sync_skills(quiet=True)

    assert result["bootstrap_mode"] == "all"
    assert set(result["copied"]) == {"plan", "pokemon-player"}
    assert (skills_dir / "software-development" / "plan" / "SKILL.md").exists()
    assert (skills_dir / "gaming" / "pokemon-player" / "SKILL.md").exists()
    # Optional skills are not promoted outside curated mode.
    assert not (skills_dir / "devops" / "watchers" / "SKILL.md").exists()


def test_default_bootstrap_mode_never_prunes_seeded_skills(monkeypatch, tmp_path):
    """Updating with the default mode must not delete previously-seeded
    pristine skills, even ones outside the curated list."""
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    pokemon_src = _write_skill(bundled_dir, "gaming/pokemon-player", "pokemon-player")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)
    pokemon_dest = skills_dir / "gaming" / "pokemon-player"
    pokemon_dest.parent.mkdir(parents=True, exist_ok=True)
    sync.shutil.copytree(pokemon_src, pokemon_dest)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / ".bundled_manifest").write_text(
        f"pokemon-player:{sync._dir_hash(pokemon_src)}\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("HERMES_SKILLS_BOOTSTRAP", raising=False)
    result = sync.sync_skills(quiet=True)

    assert result["bootstrap_mode"] == "all"
    assert result["pruned"] == []
    assert (pokemon_dest / "SKILL.md").exists()
    assert "pokemon-player" in (skills_dir / ".bundled_manifest").read_text(
        encoding="utf-8"
    )


def test_none_bootstrap_mode_skips_seeding_without_pruning(monkeypatch, tmp_path):
    """HERMES_SKILLS_BOOTSTRAP=none disables seeding but never deletes what an
    earlier sync seeded — pruning is exclusive to opt-in curated mode."""
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "software-development/plan", "plan")
    pokemon_src = _write_skill(bundled_dir, "gaming/pokemon-player", "pokemon-player")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)
    pokemon_dest = skills_dir / "gaming" / "pokemon-player"
    pokemon_dest.parent.mkdir(parents=True, exist_ok=True)
    sync.shutil.copytree(pokemon_src, pokemon_dest)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / ".bundled_manifest").write_text(
        f"pokemon-player:{sync._dir_hash(pokemon_src)}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "none")
    result = sync.sync_skills(quiet=True)

    assert result["bootstrap_mode"] == "none"
    assert result["copied"] == []
    assert result["pruned"] == []
    assert (pokemon_dest / "SKILL.md").exists()
    assert not (skills_dir / "software-development" / "plan" / "SKILL.md").exists()


def test_curated_bootstrap_seeds_relevant_defaults_and_promoted_optional(
    monkeypatch,
    tmp_path,
):
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "software-development/plan", "plan")
    _write_skill(bundled_dir, "gaming/pokemon-player", "pokemon-player")
    _write_skill(optional_dir, "devops/watchers", "watchers")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "curated")
    result = sync.sync_skills(quiet=True)

    assert result["bootstrap_mode"] == "curated"
    assert "plan" in result["copied"]
    assert "watchers" in result["copied"]
    assert "pokemon-player" not in result["copied"]
    assert (skills_dir / "software-development" / "plan" / "SKILL.md").exists()
    assert (skills_dir / "devops" / "watchers" / "SKILL.md").exists()
    assert not (skills_dir / "gaming" / "pokemon-player" / "SKILL.md").exists()
    manifest = (skills_dir / ".bundled_manifest").read_text(encoding="utf-8")
    assert "plan:" in manifest
    assert "watchers:" in manifest


def test_curated_bootstrap_prunes_old_pristine_bundled_junk(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    pokemon_src = _write_skill(bundled_dir, "gaming/pokemon-player", "pokemon-player")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)
    pokemon_dest = skills_dir / "gaming" / "pokemon-player"
    pokemon_dest.parent.mkdir(parents=True, exist_ok=True)
    sync.shutil.copytree(pokemon_src, pokemon_dest)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / ".bundled_manifest").write_text(
        f"pokemon-player:{sync._dir_hash(pokemon_src)}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "curated")
    result = sync.sync_skills(quiet=True)

    assert result["pruned"] == ["pokemon-player"]
    assert not pokemon_dest.exists()
    assert "pokemon-player" not in (skills_dir / ".bundled_manifest").read_text(
        encoding="utf-8"
    )


def test_curated_bootstrap_preserves_user_modified_pruned_skill(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    pokemon_src = _write_skill(bundled_dir, "gaming/pokemon-player", "pokemon-player")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)
    pokemon_dest = skills_dir / "gaming" / "pokemon-player"
    pokemon_dest.mkdir(parents=True, exist_ok=True)
    (pokemon_dest / "SKILL.md").write_text(
        "---\nname: pokemon-player\n---\n\n# custom local edits\n",
        encoding="utf-8",
    )
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / ".bundled_manifest").write_text(
        f"pokemon-player:{sync._dir_hash(pokemon_src)}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "curated")
    result = sync.sync_skills(quiet=True)

    assert result["pruned"] == []
    assert result["prune_user_modified"] == ["pokemon-player"]
    assert (pokemon_dest / "SKILL.md").exists()
    assert "pokemon-player" in (skills_dir / ".bundled_manifest").read_text(
        encoding="utf-8"
    )


def test_all_bootstrap_mode_keeps_legacy_bundled_behavior(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "software-development/plan", "plan")
    _write_skill(bundled_dir, "gaming/pokemon-player", "pokemon-player")
    _write_skill(optional_dir, "devops/watchers", "watchers")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "all")
    result = sync.sync_skills(quiet=True)

    assert result["bootstrap_mode"] == "all"
    assert set(result["copied"]) == {"plan", "pokemon-player"}
    assert (skills_dir / "gaming" / "pokemon-player" / "SKILL.md").exists()
    assert not (skills_dir / "devops" / "watchers" / "SKILL.md").exists()


def test_all_bootstrap_mode_preserves_previous_promoted_optional(
    monkeypatch,
    tmp_path,
):
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "software-development/plan", "plan")
    watcher_src = _write_skill(optional_dir, "devops/watchers", "watchers")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)
    watcher_dest = skills_dir / "devops" / "watchers"
    watcher_dest.parent.mkdir(parents=True, exist_ok=True)
    sync.shutil.copytree(watcher_src, watcher_dest)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / ".bundled_manifest").write_text(
        f"watchers:{sync._dir_hash(watcher_src)}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "all")
    result = sync.sync_skills(quiet=True)

    assert result["bootstrap_mode"] == "all"
    assert result["pruned"] == []
    assert (watcher_dest / "SKILL.md").exists()
    assert "watchers:" in (skills_dir / ".bundled_manifest").read_text(
        encoding="utf-8"
    )


def test_curated_bootstrap_gates_macos_only_skills_off_macos(monkeypatch, tmp_path):
    """Off-macOS, curated mode must not seed the Apple-app automation skills —
    issue #19986's complaint is exactly these landing on a Linux install."""
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "software-development/plan", "plan")
    _write_skill(bundled_dir, "apple/imessage", "imessage")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "curated")
    monkeypatch.setattr(sync.sys, "platform", "linux")
    result = sync.sync_skills(quiet=True)

    assert "plan" in result["copied"]
    assert "imessage" not in result["copied"]
    assert not (skills_dir / "apple" / "imessage").exists()


def test_curated_bootstrap_keeps_macos_skills_on_macos(monkeypatch, tmp_path):
    """On macOS the Apple-app skills stay in the curated set."""
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "apple/imessage", "imessage")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "curated")
    monkeypatch.setattr(sync.sys, "platform", "darwin")
    result = sync.sync_skills(quiet=True)

    assert "imessage" in result["copied"]
    assert (skills_dir / "apple" / "imessage" / "SKILL.md").exists()


def test_curated_bootstrap_cleans_stale_promoted_optional_from_manifest(
    monkeypatch,
    tmp_path,
):
    """A promoted optional skill removed from the optional directory upstream
    must be cleaned from the manifest (the ``source_names`` union covers
    optional sources), while any local copy on disk is left untouched."""
    bundled_dir = tmp_path / "bundled"
    optional_dir = tmp_path / "optional"
    _write_skill(bundled_dir, "software-development/plan", "plan")
    sync, skills_dir = _isolated_sync(monkeypatch, tmp_path, bundled_dir, optional_dir)
    # Previously-seeded promoted optional skill: manifest-tracked local copy
    # whose upstream optional source has since been removed.
    watcher_dest = skills_dir / "devops" / "watchers"
    watcher_dest.mkdir(parents=True, exist_ok=True)
    (watcher_dest / "SKILL.md").write_text(
        "---\nname: watchers\n---\n\n# watchers\n",
        encoding="utf-8",
    )
    (skills_dir / ".bundled_manifest").write_text(
        f"watchers:{sync._dir_hash(watcher_dest)}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_SKILLS_BOOTSTRAP", "curated")
    result = sync.sync_skills(quiet=True)

    assert "watchers" in result["cleaned"]
    assert "watchers" not in (skills_dir / ".bundled_manifest").read_text(
        encoding="utf-8"
    )
    # Cleaning is manifest-only: the user's local copy stays on disk.
    assert (watcher_dest / "SKILL.md").exists()
