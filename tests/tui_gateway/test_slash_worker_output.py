"""Regression tests for slash-worker command output capture."""

from rich.text import Text


def test_slash_worker_captures_rich_output_without_ansi_codes():
    from tui_gateway.slash_worker import _run

    class _CLI:
        console = None

        def process_command(self, cmd):
            assert cmd == "/journey"
            self.console.print(Text("colored journey", style="rgb(255,0,128)"))

    output = _run(_CLI(), "/journey")

    assert output == "colored journey"
    assert "\x1b[" not in output
    assert "\033[" not in output
