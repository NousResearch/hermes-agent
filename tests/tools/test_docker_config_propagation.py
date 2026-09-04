from unittest.mock import MagicMock, patch


def test_file_and_execute_code_creation_honor_persist_settings():
    """Every Docker creation path forwards the reuse and orphan-reaper flags."""
    import tools.code_execution_tool as code_execution_tool
    import tools.file_tools as file_tools
    import tools.terminal_tool as terminal_tool

    config = {
        "env_type": "docker",
        "docker_image": "python:3.11",
        "cwd": "/workspace",
        "timeout": 30,
        "container_persistent": True,
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
    }
    created_env = MagicMock()

    with (
        patch.object(terminal_tool, "_active_environments", {}),
        patch.object(terminal_tool, "_last_activity", {}),
        patch.object(terminal_tool, "_creation_locks", {}),
        patch.object(terminal_tool, "_task_env_overrides", {}),
        patch.object(terminal_tool, "_get_env_config", return_value=config),
        patch.object(terminal_tool, "_create_environment", return_value=created_env) as create_env,
        patch.object(terminal_tool, "_start_cleanup_thread"),
        patch.object(file_tools, "_file_ops_cache", {}),
    ):
        file_tools._get_file_ops("file-tool-task")
        assert create_env.call_args.kwargs["container_config"]["docker_persist_across_processes"] is False
        assert create_env.call_args.kwargs["container_config"]["docker_orphan_reaper"] is False

        create_env.reset_mock()
        # The terminal backend deliberately collapses ordinary task IDs into
        # one shared container. Clear the first entry to exercise the separate
        # process-level creation path used by execute_code.
        terminal_tool._active_environments.clear()
        code_execution_tool._get_or_create_env("execute-code-task")
        assert create_env.call_args.kwargs["container_config"]["docker_persist_across_processes"] is False
        assert create_env.call_args.kwargs["container_config"]["docker_orphan_reaper"] is False
