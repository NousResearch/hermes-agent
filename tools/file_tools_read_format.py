"""Presentation helpers for read_file results."""

import re


def validate_line_numbers(line_numbers: bool) -> None:
    if not isinstance(line_numbers, bool):
        raise ValueError("line_numbers must be a boolean (true or false).")


def format_read_content(content: str, line_numbers: bool) -> str:
    """Remove generated display gutters while preserving literal file prefixes."""
    if line_numbers:
        return content
    return re.sub(r"(?m)^[0-9]+(?:\||$)", "", content)
