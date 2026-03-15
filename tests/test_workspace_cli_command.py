from unittest.mock import MagicMock, patch


class TestWorkspaceCLICommand:
    def _make_cli(self):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {"quick_commands": {}}
        cli.console = MagicMock()
        cli.agent = None
        cli.conversation_history = []
        return cli

    def test_process_command_dispatches_workspace_handler(self):
        cli = self._make_cli()

        with patch.object(cli, "_handle_workspace_command") as handler:
            result = cli.process_command("/workspace status")

        assert result is True
        handler.assert_called_once_with("/workspace status")
