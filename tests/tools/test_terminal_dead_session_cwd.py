"""A session's recorded terminal cwd whose folder was deleted (a removed worktree) no longer makes every
command fail ``cd``: the session's current workspace wins instead (Teknium's ruling that a missing
worktree must never block a session)."""


def test_deleted_recorded_cwd_yields_to_the_session_workspace(tmp_path):
    from tools.terminal_tool import _resolve_command_cwd, clear_session_cwd, record_session_cwd

    workspace, tree = tmp_path / "repo", tmp_path / "repo" / ".worktrees" / "feat"
    tree.mkdir(parents=True)
    record_session_cwd("dead-tree", str(tree))
    try:
        assert _resolve_command_cwd(workdir=None, default_cwd=str(workspace), session_key="dead-tree",
                                    env_type="local") == str(tree)
        tree.rmdir()
        assert _resolve_command_cwd(workdir=None, default_cwd=str(workspace), session_key="dead-tree",
                                    env_type="local") == str(workspace)
    finally:
        clear_session_cwd("dead-tree")
