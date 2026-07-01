"""The slash worker feeds desktop chat bubbles, which render plain text — so
any ANSI a worker-routed command emits (e.g. /journey's own Rich Console) must
be stripped from the worker's return value."""

from __future__ import annotations


class _FakeCLI:
    console = None

    def process_command(self, cmd: str) -> None:
        import sys

        sys.stdout.write("\x1b[38;2;1;2;3mcolored\x1b[0m plain")


def test_run_strips_ansi_from_output():
    from tui_gateway import slash_worker

    out = slash_worker._run(_FakeCLI(), "/anything")

    assert "\x1b[" not in out
    assert out == "colored plain"
