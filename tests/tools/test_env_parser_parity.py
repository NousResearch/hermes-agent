"""Parser parity: every core reader of ~/.hermes/.env must agree.

Bug class ported from qwibitai/nanoclaw#3659: two .env parsers, two answers
for the same file. Hermes' canonical parser (``hermes_cli.config.load_env``)
reverses the ``\\"``/``\\\\`` escapes its own writer produces and handles
unpaired quotes conservatively; ``tools.skills_tool.load_env`` used to carry
a hand-rolled ``value.strip().strip('"\\'')`` loop that did neither — so a
credential containing ``"`` or ``\\`` (or an unterminated quote) resolved
differently in skill-requirement checks and sandbox env passthrough than
everywhere else. skills_tool now delegates to the canonical parser; these
tests pin the parity contract.
"""

from __future__ import annotations

from unittest.mock import patch


def _load_both(tmp_path, monkeypatch, contents: str):
    """Return (canonical, skills_tool) parses of the same .env file."""
    from hermes_cli.config import invalidate_env_cache, load_env

    env_path = tmp_path / ".env"
    env_path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import tools.skills_tool as skills_tool

    invalidate_env_cache()
    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            canonical = load_env()
            skills = skills_tool.load_env()
    finally:
        invalidate_env_cache()
    return canonical, skills


def test_escaped_quote_credential_parses_identically(tmp_path, monkeypatch):
    # The canonical writer (save_env_value) escapes " and \ inside double
    # quotes. The old skills_tool parser left the backslash in the value.
    canonical, skills = _load_both(
        tmp_path, monkeypatch, 'PASS="pa\\"ss"\nSLASH="a\\\\b"\n'
    )
    assert canonical["PASS"] == 'pa"ss'
    assert canonical["SLASH"] == "a\\b"
    assert skills == canonical


def test_unterminated_quote_parses_identically(tmp_path, monkeypatch):
    # The old skills_tool parser stripped the lone leading quote; the
    # canonical parser preserves it (not a matched pair).
    canonical, skills = _load_both(tmp_path, monkeypatch, 'B="unterminated\n')
    assert skills == canonical


def test_ordinary_quoted_values_parse_identically(tmp_path, monkeypatch):
    canonical, skills = _load_both(
        tmp_path,
        monkeypatch,
        'TZ="America/New_York"\n'
        "SINGLE='single quoted'\n"
        "PLAIN=bare\n"
        "export EXPORTED=fromshell\n"
        "# comment\n",
    )
    assert canonical["TZ"] == "America/New_York"
    assert canonical["SINGLE"] == "single quoted"
    assert canonical["PLAIN"] == "bare"
    assert canonical["EXPORTED"] == "fromshell"
    assert skills == canonical
