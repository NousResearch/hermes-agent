"""Regression tests for the P-0060 payload-guard fix (false malformed-payload blocks).

Two defects blocked benign commands as "malformed executable payload":

1. ``detect_hardline_command`` failed closed whenever the *normalized* detection
   variant failed to lex, even when the RAW command (what the shell actually sees)
   lexed fine. Normalization is a detection aid, not ground truth, so the guard now
   fires only when BOTH parses fail.
2. The grep-pattern tail lexer started mid-word (the command-word walk enters a grep
   nested in ``"$(grep ...)"`` after the opening quote) with no quote-state seed, so
   legitimately quoted operands parsed as unterminated. The lexer now takes the
   quote state recovered by a pre-scan from the segment start.

The fixtures under ``tests/fixtures/blocked_scripts_corpus/`` are the real
blocked-command corpus (4-line saved-script headers already stripped; CRLF line
endings are part of the real payloads). 26 commands the old guard falsely blocked
now pass; ``blocked-1788019127-ad3bc40a.sh`` (the ``["\\']`` quote idiom) is
genuinely unparsable and must stay blocked with the unparsable description.
"""

from pathlib import Path

import pytest

from tools.approval_detection import (
    _MALFORMED_EXEC_DESCRIPTION,
    _PARSER_LIMIT_DESCRIPTION,
    _UNPARSABLE_COMMAND_DESCRIPTION,
    _quoted_grep_pattern_spans,
    _scan_shell,
    _shell_tokens_with_spans,
    detect_hardline_command,
)
from tools.approval import _hardline_block_result

CORPUS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "blocked_scripts_corpus"
# Genuinely unparsable quoting — the fail-closed guard must keep blocking it.
STILL_BLOCKED = {"blocked-1788019127-ad3bc40a.sh"}


def corpus_body(name: str) -> str:
    """Fixture bodies are already header-stripped command payloads."""
    return (CORPUS_DIR / name).read_bytes().decode("utf-8", "replace").rstrip("\n")


def corpus_names():
    names = sorted(p.name for p in CORPUS_DIR.glob("blocked-*.sh"))
    assert len(names) == 27, f"expected 27 corpus fixtures, found {len(names)}"
    return names


class TestCorpusReplay:
    @pytest.mark.parametrize("name", corpus_names())
    def test_corpus_detect_verdicts(self, name):
        """Design acceptance: 26/27 pass detect_hardline_command; the genuinely
        unparsable file stays blocked with the unparsable-parse description."""
        body = corpus_body(name)
        blocked, description = detect_hardline_command(body)
        if name in STILL_BLOCKED:
            assert blocked, name
            assert description == _UNPARSABLE_COMMAND_DESCRIPTION, (name, description)
        else:
            assert not blocked, (name, description)

    def test_rescued_files_never_block_with_malformed_descriptions(self):
        for name in corpus_names():
            if name in STILL_BLOCKED:
                continue
            blocked, description = detect_hardline_command(corpus_body(name))
            if blocked:
                assert description not in (
                    _MALFORMED_EXEC_DESCRIPTION, _UNPARSABLE_COMMAND_DESCRIPTION,
                ), (name, description)

    def test_genuinely_unparsable_file_is_raw_malformed(self):
        """The one kept block fails closed on the raw parse itself."""
        body = corpus_body("blocked-1788019127-ad3bc40a.sh")
        _, malformed_raw = _quoted_grep_pattern_spans(body)
        assert malformed_raw


class TestMinimalRepros:
    @pytest.mark.parametrize("command", [
        # grep's substitution result interpolated into another command's argument.
        'echo "count=$(grep -c -i error log.txt)"',
        'sed -n "$(grep -n \'def x\' f.py | head -1 | cut -d: -f1),+40p" f.py',
    ])
    def test_nested_quoted_grep_substitution_passes(self, command):
        blocked, description = detect_hardline_command(command)
        assert not blocked, description

    @pytest.mark.parametrize("command", [
        # Quote idiom inside a grep PCRE — genuinely ambiguous, must fail closed.
        "grep -n -A 5 -E 'Popen|subprocess|exec|launch|cmd =|[\"\\']hermes[\"\\']' config.yaml",
        "grep -E 'a[\"\\']b' file",
        # Unclosed quote.
        "grep 'oops file",
        # Dangling backslash.
        "grep -e 'x' file \\",
    ])
    def test_genuinely_malformed_still_blocks(self, command):
        blocked, description = detect_hardline_command(command)
        assert blocked
        assert description == _UNPARSABLE_COMMAND_DESCRIPTION

    @pytest.mark.parametrize("command", [
        "grep -c -i error log.txt",
        "grep -n 'def x' f.py",
        'echo "plain $(uptime) text"',
        "rg -e 'pattern' src/",
    ])
    def test_benign_controls_stay_clean(self, command):
        blocked, description = detect_hardline_command(command)
        assert not blocked, description


class TestEdgeCases:
    @pytest.mark.parametrize("command", [
        "",                 # empty payload
        "grep '' f",        # empty pattern
        'grep "" f',
        "grep -n 'héllo→世界' f.py",  # unicode pattern
        "echo 'x $(grep y f'",  # unbalanced $( inside single quotes: pre-existing pass, documented
    ])
    def test_design_edge_cases(self, command):
        blocked, description = detect_hardline_command(command)
        assert not blocked, description


class TestUnparsableRecoveryMessage:
    def test_unparsable_block_saves_payload_with_accurate_header(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        cmd = "grep 'oops file"
        r = _hardline_block_result(_UNPARSABLE_COMMAND_DESCRIPTION, cmd)
        assert r["approved"] is False
        assert "RECOVERY" in r["message"]
        import re as _re
        m = _re.search(r"saved to (\S+\.sh)", r["message"])
        assert m, r["message"]
        saved = Path(m.group(1))
        assert saved.exists()
        body = saved.read_text(encoding="utf-8")
        assert cmd in body
        assert body.startswith("#!/bin/bash")
        # Header states the real cause, not the parser limit (design-exact phrasing).
        assert "could not be parsed for safety review" in body
        assert "command parser limit" not in body
        assert f"bash {saved}" in r["message"]

    def test_parser_limit_default_header_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        from tools.approval_floors import _save_blocked_payload
        path = Path(_save_blocked_payload("echo hi"))
        body = path.read_text(encoding="utf-8")
        assert "exceeded the inline command parser limit" in body
        assert "could not be parsed for safety review" not in body

    def test_malformed_exec_description_still_triggers_recovery(self):
        """The legacy sentinel description keeps its recovery path."""
        r = _hardline_block_result(_MALFORMED_EXEC_DESCRIPTION, "python3 -c 'x'")
        assert "RECOVERY" in r["message"]

    def test_real_hardline_blocks_get_no_recovery(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = _hardline_block_result("recursive delete of root filesystem", "rm -rf --no-preserve-root /")
        assert "RECOVERY" not in r["message"]
        assert not (tmp_path / ".hermes" / "cache" / "blocked-scripts").exists()


class TestLexerSeed:
    SEGMENT = 'echo "count=$(grep -c -i error log.txt)"'

    def _seed(self, segment: str, start: int):
        """Same pre-scan _quoted_grep_pattern_spans performs before the tail lex."""
        seed_quote = None
        for kind, i, j, quote in _scan_shell(segment, 0, len(segment)):
            if i >= start:
                break  # step belongs to the tail lexer
            seed_quote = quote
            if j <= start:
                continue
            break  # straddling step (subst/esc): its quote wins
        return seed_quote

    def test_seeded_tail_lexes_inside_enclosing_quote(self):
        """With the seed, the tail lexer stays inside the enclosing double quote:
        spaces are not separators and the closing quote balances the scan."""
        start = self.SEGMENT.index("grep")
        tokens = _shell_tokens_with_spans(self.SEGMENT, start, start_quote=self._seed(self.SEGMENT, start))
        assert tokens is not None
        # One token spanning to the segment end — the space after "grep" is
        # inside the enclosing double quote, not a word separator.
        assert tokens[0][0] == "grep -c -i error log.txt)"

    def test_unseeded_tail_still_fails_closed(self):
        """Without the seed the old behavior returns None (callers fail closed) —
        kept as a control so the seed parameter stays load-bearing."""
        start = self.SEGMENT.index("grep")
        assert _shell_tokens_with_spans(self.SEGMENT, start) is None


class TestSentinels:
    def test_parser_limit_description_unchanged(self):
        assert _PARSER_LIMIT_DESCRIPTION == "command parser limit exceeded"

    def test_malformed_exec_description_unchanged(self):
        assert _MALFORMED_EXEC_DESCRIPTION == "command parser limit or malformed executable payload"

    def test_unparsable_description_is_new_sentinel(self):
        assert _UNPARSABLE_COMMAND_DESCRIPTION == "command could not be parsed for safety review"
