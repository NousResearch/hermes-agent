import os


def test_guard_inactive_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PROTECTED_GIT_ROOTS", raising=False)
    from tools.approval import check_protected_root_write_guard

    assert check_protected_root_write_guard(f"> {tmp_path / 'x'}", str(tmp_path)) is None


def test_guard_blocks_write_vectors(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
    from tools.approval import check_protected_root_write_guard

    for command in (
        f"cp source {root / 'a.py'}",
        f"echo x > {root / 'b.py'}",
        f"printf x | tee {root / 'c.py'}",
        f"rm -rf {root / 'd.py'}",
        f"sed -i s/a/b/ {root / 'e.py'}",
        f"rsync source {root / 'f.py'}",
        f"python - <<'PY'\nopen('{root / 'g.py'}', 'w').write('x')\nPY",
    ):
        result = check_protected_root_write_guard(command, str(tmp_path))
        assert result and result["status"] == "blocked", command


def test_guard_allows_linked_worktree_and_reads(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    worktree = root / ".worktrees" / "task"
    worktree.mkdir(parents=True)
    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
    from tools.approval import check_protected_root_write_guard

    assert check_protected_root_write_guard(f"> {worktree / 'x.py'}", str(tmp_path)) is None
    assert check_protected_root_write_guard(f"cat {root / 'README.md'}", str(tmp_path)) is None


def test_guard_resolves_relative_paths_after_cd(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
    from tools.approval import check_protected_root_write_guard

    result = check_protected_root_write_guard("cd repo && echo x > agent.py", str(tmp_path))
    assert result and result["status"] == "blocked"
