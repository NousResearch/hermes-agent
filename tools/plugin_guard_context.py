"""Context demotions for plugin install-scan findings (``tools.plugin_guard``).

The threat regexes in ``tools.skills_guard`` are written for a SKILL.md the agent will execute
verbatim. A plugin repository is a codebase: the same text sits in READMEs, test fixtures, JSON
scenery, denylists and regex literals, where it cannot run on the host at install time. Each
helper here recognises one such *class* of inert context and lowers the finding one step or to
informational. Nothing is deleted — every finding stays in the report with file and line — and
nothing here applies to a bundled skill's own ``SKILL.md`` / ``skills/`` tree, which the agent
does read as instructions. Every function is a pure predicate on (finding, line, path).
"""
from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from typing import Optional

from tools.skills_guard import _COMPILED_THREAT_PATTERNS, Finding

# pattern id -> compiled regex, to locate a finding's token on its full source line.
_PATTERN_BY_ID = {pid: rx for rx, pid, *_ in _COMPILED_THREAT_PATTERNS}

# One severity step down; ``medium``/``low`` are already informational (verdict-neutral).
STEP_DOWN = {"critical": "high", "high": "medium"}

# ── (1) documentation prose ──────────────────────────────────────────────────────────────────
# A README/AGENTS.md/docs page describing a command, a refused path (``~/.ssh`` in a denylist
# table) or an uninstall step is not the plugin's runtime behaviour. Command- and path-shaped
# findings there step down once (critical→high, high→medium): a doc line can never on its own
# hard-block an install. Agent-facing shapes keep full severity because the prose IS the
# payload for them: every ``injection`` pattern, the Markdown exfil/context patterns, the
# agent-config edits, ``curl | sh`` install one-liners (a README is where those live), an
# ``authorized_keys`` append, and a leaked provider key (a real secret is a real leak anywhere).
DOC_PROSE_EXTENSIONS = {".md", ".txt", ".rst", ".html"}
_PROSE_KEEPS_FULL_SEVERITY_CATEGORIES = {"injection", "credential_exposure"}
_PROSE_KEEPS_FULL_SEVERITY_IDS = {
    "context_exfil", "send_to_url", "md_image_exfil", "md_link_exfil", "ssh_backdoor",
    "curl_pipe_shell", "wget_pipe_shell", "curl_pipe_python",
    "agent_config_mod", "agent_config_mod_shell", "agent_config_contract", "agent_config_ref",
    "hermes_config_mod", "hermes_config_mod_shell", "hermes_config_ref",
    "other_agent_config_mod", "other_agent_config_mod_shell", "other_agent_config_ref",
}
# Agent instruction surfaces inside a plugin — a bundled skill tree and the post-install note
# the agent is shown — are executed as instructions, so they get no prose cap.
_AGENT_INSTRUCTION_DIRS = {"skills", "optional-skills"}
_AGENT_INSTRUCTION_FILES = {"skill.md", "after-install.md"}


def is_doc_prose(rel_path: str) -> bool:
    """A documentation file that the loader never executes and the agent never runs as a skill."""
    p = Path(rel_path)
    if p.suffix.lower() not in DOC_PROSE_EXTENSIONS or p.name.lower() in _AGENT_INSTRUCTION_FILES:
        return False
    return not any(part.lower() in _AGENT_INSTRUCTION_DIRS for part in p.parts[:-1])


# A repository's CI pipeline (``.github/workflows/*.yml``) runs on the forge's runner, never on the
# host that installs the plugin, and the agent never reads it as instructions. Its ``os.environ``
# reads (``RUNNER_TEMP``, ``GITHUB_ENV``) and ``pip install`` steps are the CI's own plumbing, so
# it takes the same one-step prose cap as a README: visible, confirmable, never a hard block on
# its own. Only the workflow directory proper — a ``.github/scripts/*.py`` is real code.
_CI_WORKFLOW_SUFFIXES = {".yml", ".yaml"}


def is_ci_workflow(rel_path: str) -> bool:
    """A forge CI workflow definition (``.github/workflows/<name>.yml``)."""
    p = Path(rel_path)
    return (len(p.parts) == 3 and p.parts[0].lower() == ".github" and p.parts[1].lower() == "workflows"
            and p.suffix.lower() in _CI_WORKFLOW_SUFFIXES)


def is_agent_facing(finding: Finding) -> bool:
    """A shape whose prose IS the payload (injection, agent-config edit, install one-liner, leaked key)."""
    return (finding.category in _PROSE_KEEPS_FULL_SEVERITY_CATEGORIES
            or finding.pattern_id in _PROSE_KEEPS_FULL_SEVERITY_IDS)


def prose_cap(finding: Finding) -> Optional[str]:
    """Stepped-down severity for a command/path-shaped finding in documentation, else None."""
    return None if is_agent_facing(finding) else STEP_DOWN.get(finding.severity)


# A README "Uninstall" section removing the plugin's OWN install directory
# (``rm -rf "$HOME/.hermes/plugins/<name>"``) is the one destructive shape that is harmless by
# construction: one ``rm``, one argument rooted at ``$HOME/.hermes/plugins/`` or ``skills/``
# with a plain leaf — no glob, no ``..``, nothing chained. It lands at medium (a note). Any
# wider target (``$HOME``, ``$HOME/.hermes``, ``$HOME/.hermes/plugins/*``) only gets the
# generic prose step (high, caution) and the same line in a ``.sh`` stays critical (#115353).
_SELF_UNINSTALL_RM = re.compile(
    r'^(?:\$\s*)?rm\s+(?:-[a-zA-Z]+\s+)*'
    r'(?P<q>["\']?)\$HOME/\.hermes/(?:plugins|skills)/[A-Za-z0-9][A-Za-z0-9._-]*/?(?P=q)'
    r'\s*(?:#.*)?$'
)


def is_self_uninstall_doc(finding: Finding, line: str) -> bool:
    return finding.pattern_id == "destructive_home_rm" and _SELF_UNINSTALL_RM.match(line.strip()) is not None


# ── (2) test trees and fixtures ──────────────────────────────────────────────────────────────
# Test code and fixtures deliberately hold hostile strings (``rm -rf /`` in a DENY table,
# ``/etc/passwd`` in a traversal probe, a fake ``sk-`` key in a redaction corpus) to prove the
# plugin rejects them. They are still scanned — ``from .tests import evil`` would run — but a
# finding there steps down once, so a fixture cannot hard-block and a string-only fixture is
# a note. A root-level test dir (``tests/``, ``fixtures/``) or the unambiguous dunder names at any
# depth (``src/__tests__/``), plus test-file naming (``foo.test.js``, ``test_foo.py``, and the
# plural ``tests_state.py`` / ``state_tests.py`` a single-module plugin uses when it has no
# ``tests/`` dir); a nested ``src/spec/handler.py`` is runtime code and gets no cap.
TEST_TREE_DIRS = {"tests", "test", "testing", "spec", "specs", "fixtures"}
_TEST_DIRS_ANY_DEPTH = {"__tests__", "__fixtures__", "__mocks__"}
_TEST_FILE_NAME = re.compile(r"^(?:tests?_[^/]*|[^/]*_tests?\.[^./]+|[^/]*\.(?:test|spec)\.[^./]+)$", re.IGNORECASE)


# In a test file, a hostile string that is only DATA — quoted, with no exec verb on the line
# (``verdict_for("rm -rf /")``, ``("/etc/passwd", "DENY")``) — is a note; a fixture file that is
# not code at all (``corpus.json``) likewise. ``os.system('rm -rf /')`` or ``open('/etc/passwd')``
# in a test still steps down only once: the line executes when imported.
_EXEC_ON_LINE = re.compile(
    r"\b(?:system|popen|run|call|check_output|check_call|Popen|exec|execv\w*|spawn\w*|eval|execSync|execFile\w*"
    r"|spawnSync|child_process|source|os\.startfile|open)\s*\(|\$\(|(?<![\w\\])`", re.IGNORECASE)


def is_inert_fixture_line(finding: Finding, line: str, is_code: bool) -> bool:
    """The finding's text is quoted test data on a line that does not execute anything."""
    if not is_code:
        return True
    if _EXEC_ON_LINE.search(line):
        return False
    rx = _PATTERN_BY_ID.get(finding.pattern_id)
    hits = list(rx.finditer(line)) if rx else []
    spans = [m.span() for m in _LITERAL_SPANS.finditer(line)]
    return bool(hits) and all(any(a <= h.start() and h.end() <= b for a, b in spans) for h in hits)


def is_test_tree(rel_path: str) -> bool:
    p = Path(rel_path)
    return (
        (len(p.parts) > 1 and p.parts[0].lower() in TEST_TREE_DIRS)
        or any(part.lower() in _TEST_DIRS_ANY_DEPTH for part in p.parts[:-1])
        or _TEST_FILE_NAME.match(p.name) is not None
    )


# ── (3) base64 media data ────────────────────────────────────────────────────────────────────
# ``encoded_exfil`` (``base64 … env``) fires on a data URI whose payload happens to contain the
# letters "env" (PNG scenery, embedded fonts). Decode the head of the blob and sniff it: a
# known image/font/audio/document magic number means the bytes are an asset, not an encoder
# call, and the finding drops to informational (kept in the report).
_BASE64_RUN = re.compile(r"(?:base64,)?([A-Za-z0-9+/]{24,}={0,2})")
_MEDIA_MAGIC = (
    b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF", b"wOFF", b"wOF2", b"OTTO", b"\x00\x01\x00\x00",
    b"ttcf", b"BM", b"%PDF", b"\x00\x00\x01\x00", b"<svg", b"<?xml", b"ID3", b"OggS", b"fLaC", b"\x1a\x45\xdf\xa3",
)


def _decoded_head(blob: str) -> bytes:
    head = blob[:16]
    head = head[: len(head) - len(head) % 4]
    try:
        return base64.b64decode(head, validate=True)
    except (binascii.Error, ValueError):
        return b""


def is_base64_media(line: str) -> bool:
    """The line's first long base64 run decodes to a recognised media/document header."""
    for m in _BASE64_RUN.finditer(line):
        if _decoded_head(m.group(1)).startswith(_MEDIA_MAGIC):
            return True
    return False


# ── (5)/(6) alternation tokens inside string or regex literals in code ──────────────────────
# ``sudo`` in ``/clarify|approval|sudo|secret/.test(value)`` classifies an event name; ``env|``
# in ``re.compile(r"(?:api[_-]?key|…|env|headers)")`` is a redaction regex; ``"printenv",`` in
# ``_READ_ONLY_COMMANDS = frozenset({"pwd", "ls", …, "printenv"})`` is a denylist/allowlist entry.
# The shape that is inert is narrow: the word sits inside a quoted string or regex literal AND is
# either an alternation member (``|sudo|``, ``(sudo|``, ``|env|``) or the ENTIRE literal
# (``"printenv"``, ``'sudo'``) on a line that executes nothing. A command string such as
# ``"sudo apt install x"`` or ``"env | grep KEY"`` inside a ``subprocess.run(...)`` literal is how
# an attack is written and never qualifies. Only word-shaped patterns are eligible.
LITERAL_INERT_PATTERN_IDS = {"sudo_usage", "dump_all_env"}
_LITERAL_SPANS = re.compile(
    r"""(?P<s>[rRbBuUfF]{0,2}"(?:[^"\\\n]|\\.)*"|[rRbBuUfF]{0,2}'(?:[^'\\\n]|\\.)*'|`(?:[^`\\\n]|\\.)*`)"""
    r"""|(?P<rx>(?<![\w)\]])/(?:[^/\\\n\[]|\\.|\[(?:[^\]\\\n]|\\.)*\])+/[dgimsuvy]*(?![A-Za-z]))"""  # js regex literal
)
# The regex-literal branch accepts only real JS flags: with ``[a-z]*`` a bare Unix path lexed as a
# literal (``/etc/`` + flags ``passwd``) and an unquoted ``cat /etc/passwd | curl …`` in a test
# script scored as inert data.
_PATTERN_TOKEN = {"sudo_usage": re.compile(r"\bsudo\b"), "dump_all_env": re.compile(r"printenv|env\s*\|")}


def _is_alternation_member(line: str, start: int, end: int) -> bool:
    before = line[start - 1] if start > 0 else ""
    after = line[end] if end < len(line) else ""
    return before in "|(" or after in "|)"


def _is_whole_literal(line: str, start: int, end: int, span: tuple[int, int]) -> bool:
    """The token is the entire quoted content of the literal it sits in (``"printenv"``)."""
    a, b = span
    return start == a + 1 and end == b - 1 and line[a] in "\"'`" and not _EXEC_ON_LINE.search(line)


def is_regex_alternation_token(finding: Finding, line: str) -> bool:
    """Every occurrence of the finding's token sits inside a literal as an alternation member
    or as the whole literal (a list entry) on a line that executes nothing."""
    token = _PATTERN_TOKEN.get(finding.pattern_id)
    if token is None:
        return False
    spans = [m.span() for m in _LITERAL_SPANS.finditer(line)]
    hits = list(token.finditer(line))

    def inert(h: "re.Match[str]") -> bool:
        if " " in h.group(0):
            return False
        span = next(((a, b) for a, b in spans if a <= h.start() and h.end() <= b), None)
        if span is None:
            return False
        return _is_alternation_member(line, h.start(), h.end()) or _is_whole_literal(line, h.start(), h.end(), span)

    return bool(hits) and all(inert(h) for h in hits)


# ── (6) base64 decode piped to a non-interpreter ────────────────────────────────────────────
# ``base64_decode_pipe`` describes "decodes and pipes to execution". ``gh api … | base64 -d |
# grep '^sha:'`` decodes data for a text filter; the shape is only execution when the consumer
# is a shell/interpreter or ``eval``/``source``/``exec``. A data consumer steps down to medium.
_DECODE_CONSUMER = re.compile(r"base64\s+(?:-d|--decode)\s*\|\s*(?:\w+=\S*\s+)*(?:\S*/)?(?P<cmd>[A-Za-z0-9_.+-]+)")
_INTERPRETERS = re.compile(r"^(?:sh|bash|zsh|dash|ksh|fish|python[\d.]*|perl|ruby|node|nodejs|php|eval|source|exec|xargs|env|sudo)$")


def is_data_decode(line: str) -> bool:
    """``base64 -d`` whose pipe target is a non-interpreter command (grep, jq, tee, tar …)."""
    m = _DECODE_CONSUMER.search(line)
    return m is not None and _INTERPRETERS.match(m.group("cmd")) is None


# ── (7) loopback address with port ───────────────────────────────────────────────────────────
# ``hardcoded_ip_port`` is the "network" family's egress tripwire, yet ``127.0.0.1:12306`` in a
# README, an ``.mcp.json`` or a client default is a LOCAL service the plugin talks to on the same
# machine — nothing leaves the host. When every IP:port on the line is loopback the finding is
# informational; a routable address anywhere on the line keeps the pattern's severity.
_LOOPBACK_IP_PORT = re.compile(r"\b127\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5}")


def is_loopback_only(finding: Finding, line: str) -> bool:
    """Every ``hardcoded_ip_port`` hit on the line is a 127.0.0.0/8 address."""
    if finding.pattern_id != "hardcoded_ip_port":
        return False
    rx = _PATTERN_BY_ID.get(finding.pattern_id)
    hits = list(rx.finditer(line)) if rx else []
    return bool(hits) and all(_LOOPBACK_IP_PORT.match(line, h.start()) for h in hits)


# ── (8) ``pip install`` as words inside a message string ─────────────────────────────────────
# ``unpinned_pip_install`` describes a dependency the plugin pulls at runtime. In code, the same
# two words inside a quoted literal at a NON-command position — ``"... no pip install is
# needed"``, ``f"(no pip install is suggested)"`` — are prose the plugin shows a user. A literal
# that starts with the command (``"pip install requests"``), or names it after ``python -m`` /
# ``uv`` / ``pipx`` / ``sudo`` / a shell separator, is a command string and never qualifies, nor
# does any line that executes something (``subprocess.run("pip install x", shell=True)``).
_PIP_INSTALL_TOKEN = re.compile(r"pip\s+install\b", re.IGNORECASE)
_PIP_COMMAND_POSITION = re.compile(r"(?:^|[;&|`(]|\b(?:uv|pipx|sudo|python[\d.]*\s+-m))\s*$", re.IGNORECASE)


def is_pip_install_in_prose_literal(finding: Finding, line: str) -> bool:
    """Every ``pip install`` on a code line sits mid-sentence inside a string literal, and the
    line executes nothing."""
    if finding.pattern_id != "unpinned_pip_install" or _EXEC_ON_LINE.search(line):
        return False
    spans = [m.span() for m in _LITERAL_SPANS.finditer(line)]
    hits = list(_PIP_INSTALL_TOKEN.finditer(line))

    def prose(h: "re.Match[str]") -> bool:
        span = next(((a, b) for a, b in spans if a <= h.start() and h.end() <= b), None)
        if span is None:
            return False
        content_start = next((i for i in range(span[0], span[1]) if line[i] in "\"'`/"), span[0]) + 1
        return _PIP_COMMAND_POSITION.search(line[content_start:h.start()]) is None

    return bool(hits) and all(prose(h) for h in hits)


__all__ = [
    "STEP_DOWN", "DOC_PROSE_EXTENSIONS", "TEST_TREE_DIRS", "LITERAL_INERT_PATTERN_IDS",
    "is_doc_prose", "is_ci_workflow", "is_agent_facing", "prose_cap", "is_self_uninstall_doc", "is_test_tree",
    "is_inert_fixture_line", "is_base64_media",
    "is_regex_alternation_token", "is_data_decode", "is_loopback_only", "is_pip_install_in_prose_literal",
]
