"""Platform formatters for /memory slash command output.

Each formatter consumes the dict from MemoryStore.get_readout() and a
target selector ("all", "memory", or "user"). Returns a single string
ready for the platform's display layer.

- format_memory_cli: Rich markup for terminal (CLI)
- format_memory_markdown: GitHub-flavored markdown for chat platforms
"""

from __future__ import annotations

PREVIEW_CHARS = 200
WARN_PCT = 90


def _truncate(text: str) -> str:
    return text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")


def _sections(target: str) -> list[str]:
    if target == "memory":
        return ["memory"]
    if target == "user":
        return ["user"]
    return ["memory", "user"]


_HEADERS = {"memory": "MEMORY", "user": "USER PROFILE"}
_FILENAMES = {"memory": "MEMORY.md", "user": "USER.md"}


def format_memory_cli(data: dict, target: str = "all") -> str:
    lines: list[str] = []
    for key in _sections(target):
        section = data[key]
        warn = " [yellow]⚠ close to cap[/yellow]" if section["pct"] >= WARN_PCT else ""
        lines.append(
            f"[bold]{_FILENAMES[key]}[/bold]: "
            f"{section['char_count']:,}/{section['char_limit']:,} — "
            f"{section['pct']}%{warn}"
        )
    lines.append("")
    for key in _sections(target):
        section = data[key]
        lines.append(f"[bold cyan]── {_HEADERS[key]} ──[/bold cyan]")
        if section["entries"]:
            for i, entry in enumerate(section["entries"], 1):
                lines.append(f"{i}. {_truncate(entry)}")
        else:
            lines.append("[dim](empty)[/dim]")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_memory_markdown(data: dict, target: str = "all") -> str:
    lines: list[str] = []
    for key in _sections(target):
        section = data[key]
        warn = " ⚠️ close to cap" if section["pct"] >= WARN_PCT else ""
        lines.append(
            f"**{_FILENAMES[key]}**: "
            f"{section['char_count']:,}/{section['char_limit']:,} — "
            f"{section['pct']}%{warn}"
        )
    lines.append("")
    for key in _sections(target):
        section = data[key]
        lines.append(f"── {_HEADERS[key]} ──")
        if section["entries"]:
            for i, entry in enumerate(section["entries"], 1):
                lines.append(f"{i}. {_truncate(entry)}")
        else:
            lines.append("_(empty)_")
        lines.append("")
    return "\n".join(lines).rstrip()
