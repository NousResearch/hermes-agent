def test_execute_code_rpc_uses_frozen_approved_target_config():
    from tools import code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    approved = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {"backend": "local", "cwd": "/approved"},
            },
        },
    }
    live = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {
                    "backend": "ssh",
                    "ssh_host": "new.example",
                    "ssh_user": "agent",
                },
            },
        },
    }
    targets_mod.set_execution_target_config_source(live)

    def handler(_name, args, task_id=None):
        selected = targets_mod.resolve_execution_target(args["target"])
        current = targets_mod.resolve_live_execution_target(args["target"])
        return selected.backend, selected.config.get("cwd"), current.backend

    try:
        assert code_mod._dispatch_rpc_tool(
            handler,
            "write_file",
            {"target": "alpha"},
            "task",
            approved,
        ) == ("local", "/approved", "ssh")
    finally:
        targets_mod.set_execution_target_config_source(None)
