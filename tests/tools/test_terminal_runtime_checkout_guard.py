from tools import terminal_tool


def test_runtime_checkout_guard_blocks_direct_reset_under_authority(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    child = repo / "subdir"
    child.mkdir(parents=True)
    monkeypatch.setenv("HERMES_AGENT_REPO", str(repo))
    monkeypatch.setenv("HERMES_RELEASE_SHA", "abc123")

    message = terminal_tool._runtime_checkout_git_reset_block_message(
        "git reset --hard HEAD~1",
        str(child),
    )

    assert message is not None
    assert "runtime authority" in message
    assert "abc123" in message


def test_runtime_checkout_guard_blocks_git_c_reset_to_authority(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    monkeypatch.setenv("HERMES_AGENT_REPO", str(repo))

    message = terminal_tool._runtime_checkout_git_reset_block_message(
        f"git -C {repo} reset --hard HEAD",
        str(outside),
    )

    assert message is not None


def test_runtime_checkout_guard_blocks_cd_then_reset_to_authority(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    monkeypatch.setenv("HERMES_AGENT_REPO", str(repo))

    message = terminal_tool._runtime_checkout_git_reset_block_message(
        f"cd {repo} && git reset --hard HEAD",
        str(outside),
    )

    assert message is not None


def test_runtime_checkout_guard_allows_direct_reset_outside_authority(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    monkeypatch.setenv("HERMES_AGENT_REPO", str(repo))

    message = terminal_tool._runtime_checkout_git_reset_block_message(
        "git reset --hard HEAD",
        str(outside),
    )

    assert message is None


def test_runtime_checkout_guard_inactive_without_authority_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)

    message = terminal_tool._runtime_checkout_git_reset_block_message(
        "git reset --hard HEAD",
        str(tmp_path),
    )

    assert message is None
