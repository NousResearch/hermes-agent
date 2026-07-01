"""Regression for #56533: slash-command output returned to a chat bubble must
not contain raw ANSI escape codes.

``_run`` builds a Rich Console with ``force_terminal=True`` (so tables / bar
charts lay out correctly), which emits 24-bit color escapes. That captured
string is delivered to the TUI/Desktop chat bubble as plain text, so the
escapes must be stripped before returning — otherwise they show up as literal
``?[38;2;...m`` garbage.
"""

import types

from tui_gateway import slash_worker


def _fake_cli(printer):
    """A minimal stand-in for HermesCLI: ``_run`` assigns ``.console`` and then
    calls ``.process_command``, which prints through that console."""
    cli = types.SimpleNamespace(console=None)

    def process_command(cmd):
        printer(cli.console)

    cli.process_command = process_command
    return cli


def test_run_strips_ansi_from_rich_colored_output():
    cli = _fake_cli(lambda console: console.print("[red]hello[/red] [green]world[/green]"))
    out = slash_worker._run(cli, "/journey")

    # No escape byte survives, but the visible text does.
    assert "\x1b" not in out
    assert "hello" in out
    assert "world" in out


def test_run_strips_truecolor_bar_chart_escapes():
    # Mirrors the /journey bar chart: 24-bit foreground colors are the exact
    # "?[38;2;...m" sequences the issue reports.
    def printer(console):
        console.print("[rgb(255,140,0)]████[/] node-a")
        console.print("[rgb(0,180,90)]██[/] node-b")

    out = slash_worker._run(_fake_cli(printer), "/journey")
    assert "\x1b" not in out
    assert "38;2;" not in out
    assert "node-a" in out
    assert "node-b" in out


def test_run_preserves_plain_text_unchanged():
    cli = _fake_cli(lambda console: console.print("no color here"))
    out = slash_worker._run(cli, "/plain")
    assert out == "no color here"
