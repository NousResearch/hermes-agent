"""Shared threat-pattern library (prompt injection / promptware / exfiltration).
Each pattern is ``(regex, pattern_id, scope)``; scope is cumulative: ``"all"`` everywhere,
``"context"`` adds promptware / C2 / role hijack for context files, memory and tool results
(warn-level: that content is not user-authored), ``"strict"`` adds aggressive checks only for
memory writes, where a block is resolvable. Skill installs are scanned by ``skills_guard``,
not here. New patterns must
anchor on C2 vocabulary or unambiguous attack behavior, NOT bossy English ("you must" is common
in legitimate AGENTS.md); filler between tokens is the bounded ``_FILLER``."""

from __future__ import annotations

import hermes_re as re
import itertools
import unicodedata
from bisect import bisect_left
from typing import Optional

# Hard cap on scanned text: scanners are advisory, so bound worst-case runtime.
# The whole input is still scanned — just in bounded chunks of this size — so a
# payload past the cap (which the renderer may still emit as a tail) is not
# silently dropped, while worst-case per-chunk runtime stays predictable.
MAX_SCAN_CHARS = 65_536
# Overlap between adjacent scan chunks.  A match no wider than the overlap that
# straddles a chunk edge lies whole inside the next chunk, so every quantifier
# below is bounded (``\s{1,64}``, not ``\s+``) and ``_compile`` rejects any
# pattern whose maximum match width exceeds this.
_CHUNK_OVERLAP = 4096
# A word character (``\w``) in a bounded run, spelled as "not whitespace, not ASCII punctuation or control": RE2
# copies a class once per repeat count, and a few hundred copies of Unicode ``\w`` cost tens of milliseconds to
# build (512 per filler exceed its program budget). The two are the same only on the filler view of the text
# (``_FILLER_BLIND``), which ``scan_for_threats`` gives every pattern that holds this class.
_WORD = r"[^\s\x00-\x2f\x3a-\x40\x5b-\x5e\x60\x7b-\x7f]"
# Bounded filler between key attack words.
_FILLER = rf"(?:{_WORD}{{1,64}}\s{{1,64}}){{0,8}}"
# Non-ASCII characters that are neither word nor whitespace: NUL in the filler view (same length, so offsets hold).
_FILLER_BLIND = re.compile(r"[^\x00-\x7f\w\s]")
# Env var reference ending in a secret-ish suffix (see exfil comment below).
_SECRET_VAR = rf"\$\{{?{_WORD}{{0,256}}(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b"
# "forget your original instructions" (#26977); shared with tools/skills_guard.py and cron/prompt_scan.py.
# The word before `forget` is consumed rather than looked behind so a "don't / never / do not forget" exclusion holds
# across any run of spaces or tabs. Negators are an explicit whole-word list (a suffix match let `IMPORTANT` / `Agent`
# negate). `_` (Markdown emphasis) counts as punctuation, not as part of the neighbouring word.
_NEGATOR_VERBS = ("do", "does", "did", "ca", "could", "would", "should", "wo", "must", "need", "is", "are", "was", "were",
                  "ai", "has", "have", "had")
_NEGATOR_PLAIN = ("cannot", "not", "never")
_NEGATOR = rf"(?:(?:{'|'.join(_NEGATOR_VERBS)})n[’']?t|{'|'.join(_NEGATOR_PLAIN)})"


class ExclusionMatcher:
    """Stands in for a compiled rule whose stdlib original held negative lookaheads (RE2 has none): the rule is
    the original without them. From a start, the original tries the ends its quantifiers reach in backtracking
    order and keeps the first that its lookaheads let through; ``ends(text, match)`` lists those ends in that
    order and ``excluded(text, match, end)`` evaluates the lookaheads for one end (a lookahead at a fixed point of
    the match, such as its start, is evaluated there). A start with no such end is no match, and the search goes
    on one character later; a match resumes the search at its end, as in stdlib. Offers ``search`` and
    ``finditer``, all the scanners call on a rule, and yields real match objects.

    A start can have as many ends as its line has characters, so an ``excluded`` that reads past its end must do
    that reading once per scan (``LineTail``), not once per end: that would make one line quadratic to scan. Such
    a rule is built with ``per_scan(pattern, rule)``: ``rule()`` returns a fresh ``(ends, excluded)`` pair, with
    whatever it learns about the text, at the start of every ``finditer`` call. The facts then serve every start
    and end of that one scan and go with it: the rules are module globals shared by threads and kept for the
    life of the process, so they hold no text and nothing learned from one."""

    def __init__(self, pattern: str, ends, excluded):
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._rule = lambda: (ends, excluded)

    @classmethod
    def per_scan(cls, pattern: str, rule) -> ExclusionMatcher:
        matcher = cls(pattern, None, None)
        matcher._rule = rule
        return matcher

    def finditer(self, text: str):
        ends, excluded = self._rule()
        pos = 0
        while (m := self._pattern.search(text, pos)) is not None:
            hit = next((hit for end in ends(text, m) if not excluded(text, m, end)
                        and (hit := self._pattern.fullmatch(text, m.start(), end)) is not None), None)
            if hit is None:
                pos = m.start() + 1
            else:
                yield hit
                pos = hit.end()

    def search(self, text: str):
        return next(self.finditer(text), None)


# Private-use code points that can mark positions in a copy of a text (MarkedMatcher): the first one the text lacks.
_MARK_RANGES = (range(0xE000, 0xF900), range(0xF0000, 0xFFFFE), range(0x100000, 0x10FFFE))


def _absent_mark(text: str) -> str:
    if "\ue000" not in text:
        return "\ue000"
    present = set(text)
    return next(mark for mark in map(chr, itertools.chain(*_MARK_RANGES)) if mark not in present)


class MarkedMatcher:
    """Stands in for a compiled rule whose stdlib original held a negative lookahead decided by one point of the
    match alone (where it starts, or where it ends), where ExclusionMatcher would cost too much: it searches again
    from the next character after every start it rejects, and that search reads on to where the rule's runs end,
    so a line of n rejected starts sharing one long run is read n times. ``cuts(text)`` lists, ascending, the
    positions the lookahead lets through. A private-use character the text lacks is inserted at each of them, and
    ``marked(mark)``, the rule requiring the mark where the lookahead stood (and taking it wherever else a cut can
    fall inside a match), is searched in that copy: one pass, which keeps the original's choice of start and end.
    Each hit is returned as the rule's own match on the text. Nothing is kept from one call to the next but the
    marked rules, one per mark character."""

    def __init__(self, pattern: str, marked, cuts):
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._marked, self._cuts, self._by_mark = marked, cuts, {}

    def finditer(self, text: str):
        cuts = self._cuts(text)
        if not cuts:
            return
        mark = _absent_mark(text)
        if (rule := self._by_mark.get(mark)) is None:
            rule = self._by_mark[mark] = re.compile(self._marked(mark), re.IGNORECASE)
        copy = mark.join(text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)]))
        marks = [cut + count for count, cut in enumerate(cuts)]  # where each mark stands in the copy
        pos = 0
        while (m := rule.search(copy, pos)) is not None:
            yield self._pattern.fullmatch(text, m.start() - bisect_left(marks, m.start()),
                                          m.end() - bisect_left(marks, m.end()))
            pos = m.end()

    def search(self, text: str):
        return next(self.finditer(text), None)


def only_end(text: str, m) -> tuple:
    """``ends`` of an ExclusionMatcher rule whose match has one end per start."""
    return (m.end(),)


class LineTail:
    """``tail(text, pos)``: whether a hit starts in ``text[pos:<end of pos's line>]``, for an exclusion that the
    original evaluated at every candidate end as a lookahead over the rest of the line (``(?!.*==)``): asked at n
    ends, that rescans the line n times. ``last_in_line(text, start, end)`` gives the start of the last hit in the
    line ``text[start:end]`` (-1 for none), and whether a position holds a hit must not depend on where the scan
    started; each line is then searched once, and every end costs a bisection. One instance serves one scan of
    one text (``ExclusionMatcher.per_scan`` makes it)."""

    def __init__(self, last_in_line):
        self._last_in_line = last_in_line
        self._breaks = None
        self._last_by_line: dict = {}

    def __call__(self, text: str, pos: int) -> bool:
        if self._breaks is None:
            self._breaks = _line_breaks(text)
        breaks = self._breaks
        line = bisect_left(breaks, pos)
        if (last := self._last_by_line.get(line)) is None:
            start = breaks[line - 1] + 1 if line else 0
            last = self._last_by_line[line] = self._last_in_line(
                text, start, breaks[line] if line < len(breaks) else len(text))
        return last >= pos


def _line_breaks(text: str) -> list:
    """Indexes of the newlines in ``text``."""
    breaks, i = [], text.find("\n")
    while i >= 0:
        breaks.append(i)
        i = text.find("\n", i + 1)
    return breaks


# The original's `(?!{_NEGATOR}[ \t])` at the start of the first branch. A match that starts on a negator came from
# that branch: the second one starts on `forget`, a space, a tab or a character that is not a letter.
_NEGATOR_THEN_SPACE = re.compile(rf"{_NEGATOR}[ \t]", re.IGNORECASE)


def _opens_with_negator(text: str, m, end: int) -> bool:
    return _NEGATOR_THEN_SPACE.match(text, m.start()) is not None


# FORGET_RULES_RAW_RE is the unbounded rule for the raw-line scanners (tools/skills_guard.py,
# cron/prompt_scan.py): they neither collapse whitespace nor chunk, so no width cap applies there. It is the stdlib
# original without its negator lookahead, which FORGET_RULES_EXCLUSION evaluates (see ExclusionMatcher).
FORGET_RULES_RAW_RE = (r"(?:(?:(?<![\w’'])|(?<=_))(?:[^\W_]|[’'])+[ \t]+"
                       r"|(?:^|[^\w’' \t]|_|(?<!\w)[’'])[ \t]*)"
                       r"forget\s+(?:about\s+)?(your|all|any|previous|prior|above|earlier)"
                       r"\s+(?:\w+\s+){0,3}(instructions|rules|guidelines)(?![^\W_])")
FORGET_RULES_EXCLUSION = (only_end, _opens_with_negator)
FORGET_RULES_RAW = ExclusionMatcher(FORGET_RULES_RAW_RE, *FORGET_RULES_EXCLUSION)
# FORGET_RULES_RE is the same rule bounded for the chunked scan_for_threats (width under _CHUNK_OVERLAP). RE2 has no
# lookaround, so the negator exclusion is not in this pattern: scan_for_threats first overwrites a `forget` that
# follows a whole-word negator (_NEGATOR_FORGET), and the prefix here only needs "anything, then spaces or tabs".
FORGET_RULES_RE = (r"(?:(?:^|[^ \t])[ \t]{1,64}"
                   r"|(?:^|[^\w’' \t]|_|(?:^|\W)[’']))"
                   r"forget\s{1,64}(?:about\s{1,64})?(your|all|any|previous|prior|above|earlier)"
                   r"\s{1,64}(?:\w{1,64}\s{1,64}){0,3}(instructions|rules|guidelines)(?![^\W_])")
# A negator is a whole word: it starts the text, or follows `_` or a character that is neither a word character nor
# an apostrophe. The `forget` after it becomes `xxxxxx`: same length, and still a word for the filler above.
_NEGATOR_FORGET = re.compile(rf"((?:^|[^\w’']|_){_NEGATOR}[ \t]{{1,64}})forget", re.IGNORECASE)
# A negator hard-wrapped onto the line above (`Do not⏎forget …`), emphasised (`**Don't** forget`) or followed by
# `ever` still negates: callers blank that `forget` before matching FORGET_RULES_RE. One line break at most, so a
# heading separated from the payload by a blank line cannot suppress it; the break may be CRLF (tool results,
# memory entries and cron prompts are not newline-normalised). Same length and newlines, so offsets hold.
# Two plain `[*_]*` with only optional tokens between them are safe here: RE2 does not backtrack.
_NEGATED_FORGET = re.compile(rf"(?<![\w*’'])([*_]*{_NEGATOR}[*_]*[ \t]*(?:\r?\n[ \t]*)?(?:ever[ \t]+)?[*_]*)"
                             r"forget(?![^\W_])", re.IGNORECASE)


def mask_negated_forget(text: str) -> str:
    """``text`` with every negated ``forget`` blanked; the input for FORGET_RULES_RE."""
    return _NEGATED_FORGET.sub(lambda m: m[1] + " " * 6, text)


# Verb prefix for "modify agent config" patterns. The ``[^\n]{0,2048}`` span is written as three repeats:
# RE2 caps a repeat count at 1000.
_MODIFY = r"(update|modify|edit|write|change|append|add\s{1,64}to)\s{1,64}[^\n]{0,1000}[^\n]{0,1000}[^\n]{0,48}"
# (regex, pattern_id, scope); scope ∈ {"all", "context", "strict"}
_PATTERNS: list[tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (rf'ignore\s{{1,64}}{_FILLER}(previous|all|above|prior)\s{{1,64}}{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s{1,64}prompt\s{1,64}override', "sys_prompt_override", "all"),
    (rf'disregard\s{{1,64}}{_FILLER}(your|all|any)\s{{1,64}}{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    # The determiner must follow `forget` directly and the filler is short, so docs like "don't forget to
    # follow all the setup instructions" or "forget your password? follow the reset instructions" pass.
    (FORGET_RULES_RE, "forget_rules", "all"),
    (rf'act\s{{1,64}}as\s{{1,64}}(if|though)\s{{1,64}}{_FILLER}you\s{{1,64}}{_FILLER}(have\s{{1,64}}no|don\'t\s{{1,64}}have)\s{{1,64}}{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    # `[^>]{0,2048}` in three parts: RE2 caps one repeat count at 1000.
    (r'<\s{0,64}div\s{1,64}style\s{0,64}=\s{0,64}["\'][^>]{0,1000}[^>]{0,1000}[^>]{0,48}display\s{0,64}:\s{0,64}none', "hidden_div", "all"),
    (
        rf"translate\s{{1,64}}[^\n]{{0,512}}\s{{1,64}}into\s{{1,64}}{_WORD}{{1,64}}(?:[\s-]{{1,64}}{_WORD}{{1,64}}){{0,2}}\s{{1,64}}and\s{{1,64}}(execute|run|eval)\b",
        "translate_execute",
        "all",
    ),
    (rf'do\s{{1,64}}not\s{{1,64}}{_FILLER}tell\s{{1,64}}{_FILLER}the\s{{1,64}}user', "deception_hide", "all"),

    # ── Role-play / identity hijack (scraped web content, poisoned context files) ──
    (rf'you\s{{1,64}}are\s{{1,64}}{_FILLER}now\s{{1,64}}(?:a|an|the)\s{{1,64}}', "role_hijack", "context"),
    (rf'pretend\s{{1,64}}{_FILLER}(you\s{{1,64}}are|to\s{{1,64}}be)\s{{1,64}}', "role_pretend", "context"),
    (rf'output\s{{1,64}}{_FILLER}(system|initial)\s{{1,64}}prompt', "leak_system_prompt", "context"),
    (rf'(respond|answer|reply)\s{{1,64}}without\s{{1,64}}{_FILLER}(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (rf'you\s{{1,64}}have\s{{1,64}}been\s{{1,64}}{_FILLER}(updated|upgraded|patched)\s{{1,64}}to', "fake_update", "context"),
    # Brainworm tell: identity override via spec. Verb pair anchored so "name your variables" is safe.
    (rf'\bname\s{{1,64}}yourself\s{{1,64}}{_WORD}{{1,64}}', "identity_override", "context"),

    # ── C2 / Brainworm-style promptware (context scope) ──────────────
    # Anchored on C2 vocabulary. "register as a node" appears in legitimate distributed-systems
    # docs, so this is WARN not block: a researcher reading the Brainworm post keeps their session.
    (r'register\s{1,64}(as\s{1,64})?a?\s{0,64}node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s{1,64}(to|with)\s{1,64}', "c2_heartbeat", "context"),
    (r'pull\s{1,64}(down\s{1,64})?(?:new\s{1,64})?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s{1,64}to\s{1,64}the\s{1,64}network\b', "c2_network_connect", "context"),
    # C2-specific verbs avoid the broader "you must X" false positive.
    (rf'you\s{{1,64}}must\s{{1,64}}(?:{_WORD}{{1,64}}\s{{1,64}}){{0,3}}(register|connect|report|beacon)\b', "forced_action", "context"),
    # Anti-forensic instructions: near-zero false positive in legitimate content.
    (r'only\s{1,64}use\s{1,64}one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (rf'never\s{{1,64}}{_FILLER}(?:create|write)\s{{1,64}}{_FILLER}(?:script|file)\s{{1,64}}{_FILLER}disk', "anti_forensic_disk", "context"),
    # Unsetting agent-runtime env vars is pure attack behavior (Brainworm sub-session bypass).
    (rf'unset\s{{1,64}}{_WORD}{{0,256}}(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC){_WORD}{{0,64}}', "env_var_unset_agent", "context"),

    # ── Known C2 / red-team framework names (warn-only) ─────────────
    # Every token must be a distinctive offensive-security brand: a common English word here
    # (e.g. "praxis", also a legitimate agent name) false-positives whole AGENTS.md / SOUL.md files.
    (r'\b(?:cobalt\s{0,64}strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s{1,64}(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s{1,64}and\s{1,64}control\b', "c2_explicit_long", "context"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    # The var name ends with \b so benign names containing KEY/TOKEN as substrings
    # ($TRILLIUM_ETAPI_URL) pass. API is deliberately absent: mid-name API is ubiquitous in
    # benign vars, and every real secret it caught ($OPENAI_API_KEY) already ends in KEY/TOKEN.
    # ``[^\n]{0,2048}`` is written as three repeats here and below: RE2 caps a repeat count at 1000.
    (rf'curl\s{{1,64}}[^\n]{{0,1000}}[^\n]{{0,1000}}[^\n]{{0,48}}{_SECRET_VAR}', "exfil_curl", "all"),
    (rf'wget\s{{1,64}}[^\n]{{0,1000}}[^\n]{{0,1000}}[^\n]{{0,48}}{_SECRET_VAR}', "exfil_wget", "all"),
    (r'cat\s{1,64}[^\n]{0,1000}[^\n]{0,1000}[^\n]{0,48}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s{1,64}[^\n]{0,1000}[^\n]{0,1000}[^\n]{0,48}\s{1,64}(to|at)\s{1,64}https?://', "send_to_url", "strict"),
    (rf'(include|output|print|share)\s{{1,64}}{_FILLER}(conversation|chat\s{{1,64}}history|previous\s{{1,64}}messages|full\s{{1,64}}context|entire\s{{1,64}}context)', "context_exfil", "strict"),

    # ── Persistence / SSH backdoor (strict scope — memory + skills) ──
    (r'authorized_keys', "ssh_backdoor", "strict"),
    # Write-verb gated like the *_config_mod rules: a bare path match blocked ordinary docs
    # ("check $HOME/.ssh is chmod 700"). ``>>?`` covers a leading redirect with no verb word;
    # ``open(`` covers the scripted-write shape; chmod/chown/sed/truncate/rm/touch/curl/wget/git
    # mutate the directory without an obvious copy verb.
    # The word boundary after the verb is spelled as what has to follow it: the path at once (it opens with a
    # non-word character) or one non-word character first. RE2's own ``\b`` is ASCII-only in that position.
    (r'(?:\b(?:echo|cat|cp|mv|dd|tee|install|printf|rsync|scp|ln|append|add|write'
     r'|sed|chmod|chown|truncate|rm|touch|curl|wget|git)(?:[^\w\n][^\n]{0,511})?|\bopen\s{0,64}\([^\n]{0,512}|>>?[^\n]{0,512})'
     r'(?:\$HOME/\.ssh|~/\.ssh)', "ssh_access", "strict"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict"),
    (rf'{_MODIFY}(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (rf'{_MODIFY}\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    # Group 1 is the start of the value: ``_ENV_NAME_VALUE`` below exempts a value that
    # is an environment-variable name (RE2 has no lookahead to do that in here).
    (r'(?:api[_-]?key|token|secret|password)\s{0,64}[=:]\s{0,64}["\']'
     r'([A-Za-z0-9+/=_-]{20})', "hardcoded_secret", "strict"),
]

# A ``hardcoded_secret`` value that is itself an environment-variable NAME
# (SHOUTY_SNAKE, ≥2 underscore-separated segments) up to its closing quote:
# ENV_PASSWORD = "MYPLUGIN_APP_PASSWORD" says where the credential lives, it does
# not embed one (#116221). Case-sensitive on purpose (no IGNORECASE here) — a
# lowercase snake value is the password-passphrase shape
# ("correct_horse_battery_staple"); requiring an underscore segment keeps
# underscore-free all-caps credentials (AWS AKIA…, base32) matched.
_ENV_NAME_VALUE = re.compile(r'[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+["\']')

# Invisible / bidirectional unicode used in injection attacks: zero-width space/non-joiner/joiner,
# word joiner, invisible times/separator/plus, BOM, LTR/RTL embedding + pop + overrides, LTR/RTL/
# first-strong isolates + pop (the same set as skills_guard.py INVISIBLE_CHARS), plus the Unicode TAG
# block U+E0000–U+E007F: invisible, maps 1:1 onto ASCII ("ASCII smuggling"); skills_guard lacks it (#78952).
INVISIBLE_CHARS = frozenset(
    "\u200b\u200c\u200d\u2060\u2062\u2063\u2064\ufeff"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
) | frozenset(map(chr, range(0xE0000, 0xE0080)))

# The RGI subdivision flags (England, Scotland, Wales: U+1F3F4 + tag letters + CANCEL TAG) are the one
# legitimate use of TAG chars. Only those three are exempt: a generic "flag-shaped" exemption lets a
# run of fake flags carry a smuggled sentence a few tag letters at a time.
_RGI_SUBDIVISION_FLAGS = ("gbeng", "gbsct", "gbwls")
EMOJI_TAG_SEQUENCE_RE = re.compile(
    "|".join("\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in code) + "\U000E007F" for code in _RGI_SUBDIVISION_FLAGS))

# Compiled per scope at import; inclusion is cumulative (all ⊂ context ⊂ strict).
_SCOPE_SETS = {"all": ("all", "context", "strict"), "context": ("context", "strict"), "strict": ("strict",)}


# Maximum match width of a table pattern, for the overlap check in ``_compile``. It knows the constructs the table
# uses and raises on anything else, so a new construct cannot ship without a width. The pattern text is read with
# string code, not with regexes: the check runs at import, and a process without the regex engine imports this module.
_UNBOUNDED = 2**32 - 1  # ``*``, ``+``, ``{n,}``: always wider than the overlap


def _max_width(pattern: str) -> int:
    """Longest text ``pattern`` can match, in characters."""
    width, end = _alternation_width(pattern, 0)
    if end != len(pattern):
        raise ValueError(f"threat_patterns: unbalanced ')' at {end} in {pattern!r}")
    return width


def _alternation_width(pattern: str, i: int) -> tuple[int, int]:
    """``(width, index)`` of the alternatives starting at ``i``, up to an unmatched ``)`` or the end."""
    widest = width = 0
    while i < len(pattern) and pattern[i] != ")":
        if pattern[i] == "|":
            widest, width, i = max(widest, width), 0, i + 1
            continue
        atom, i = _atom_width(pattern, i)
        count, i = _repeat_max(pattern, i)
        width += atom * count
    return max(widest, width), i


def _atom_width(pattern: str, i: int) -> tuple[int, int]:
    """``(width, index)`` of the atom at ``i``: a group, a class, an escape or one character."""
    ch = pattern[i]
    if ch == "(":
        opener = _group_opener(pattern, i)
        width, i = _alternation_width(pattern, i + len(opener))
        if pattern[i:i + 1] != ")":
            raise ValueError(f"threat_patterns: missing ')' in {pattern!r}")
        return (0 if opener[-1] in "=!" else width), i + 1  # a lookaround consumes nothing
    if ch == "[" and (end := _class_end(pattern, i)):
        return 1, end
    escaped = pattern[i + 1:i + 2]
    if ch in "*+?[" or (ch == "\\" and (not escaped or escaped in "0123456789xuUN")):
        raise ValueError(f"threat_patterns: cannot tell the width of {pattern[i:i + 2]!r} at {i} in {pattern!r}")
    if ch == "\\":
        return (0 if escaped in "AbBZ" else 1), i + 2
    return (0 if ch in "^$" else 1), i + 1


def _group_opener(pattern: str, i: int) -> str:
    """Text that opens the group at ``i``: ``(``, ``(?P<name>``, ``(?:`` with optional flags, or a lookaround."""
    rest = pattern[i + 2:] if pattern[i + 1:i + 2] == "?" else ""
    if rest[:2] == "P<":
        name = rest[2:rest.find(">")] if ">" in rest else ""
        return f"(?P<{name}>" if name and all(c.isalnum() or c == "_" for c in name) else "("
    n = len(rest) - len(rest.lstrip("aimsx"))
    if rest[n:n + 1] == "-" and (off := len(rest) - n - 1 - len(rest[n + 1:].lstrip("imsx"))):
        n += 1 + off
    if rest[n:n + 1] == ":":
        return "(?" + rest[:n + 1]
    n = rest[:1] == "<"
    return "(?" + rest[:n + 1] if rest[n:n + 1] in ("=", "!") else "("


def _class_end(pattern: str, i: int) -> int:
    """Index after the ``[...]`` class that opens at ``i``; 0 when nothing closes it."""
    first = i + 1 + (pattern[i + 1:i + 2] == "^")
    j = first + (pattern[first:first + 1] == "]")  # a leading ``]`` is a literal if another one closes the class
    while j < len(pattern) and pattern[j] != "]":
        j += 2 if pattern[j] == "\\" else 1
    if j < len(pattern):
        return j + 1
    return first + 1 if pattern[first:first + 1] == "]" else 0


def _repeat_max(pattern: str, i: int) -> tuple[int, int]:
    """``(maximum count, index)`` of the quantifier at ``i``; ``(1, i)`` when there is none."""
    ch = pattern[i:i + 1]
    low, comma, high = pattern[i + 1:max(pattern.find("}", i), i)].partition(",") if ch == "{" else ("", "", "")
    if ch and ch in "*+?":
        count, i = (1 if ch == "?" else _UNBOUNDED), i + 1
    elif (low or comma) and (low + high or "0").isdecimal():  # ``{n}``, ``{n,}``, ``{,m}``, ``{n,m}``, ``{,}``
        count, i = (int(high) if high else _UNBOUNDED if comma else int(low)), pattern.index("}", i) + 1
    else:
        return 1, i
    return count, i + (pattern[i:i + 1] == "?")  # a lazy quantifier is as wide


def _compile() -> dict[str, list[tuple[re.Pattern, str]]]:
    compiled: dict[str, list[tuple[re.Pattern, str]]] = {"all": [], "context": [], "strict": []}
    for pattern, pid, scope in _PATTERNS:
        if scope not in _SCOPE_SETS:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")
        if (width := _max_width(pattern)) > _CHUNK_OVERLAP:
            raise ValueError(f"threat_patterns: pattern {pid!r} can match {width} chars, "
                             f"wider than the {_CHUNK_OVERLAP}-char chunk overlap")
        for s in _SCOPE_SETS[scope]:
            compiled[s].append((re.compile(pattern, re.IGNORECASE), pid))
    return compiled


_COMPILED = _compile()
# Patterns that must be matched against the filler view (see ``_WORD``).
_FILLER_IDS = frozenset(pid for pattern, pid, _scope in _PATTERNS if _WORD in pattern)


_WS_RUN = re.compile(r"\s{2,}")


def _collapse_ws(m: re.Match) -> str:
    return "\n" if "\n" in m.group() else " "


def scan_for_threats(content: str, scope: str = "context") -> list[str]:
    """Matched pattern IDs in ``content`` for ``scope``; invisible codepoints are
    reported as ``"invisible_unicode_U+XXXX"``. Raises ValueError on an unknown scope."""
    if not content:
        return []
    if (patterns := _COMPILED.get(scope)) is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    # Invisible unicode is checked on the RAW content: NFKC below can strip these
    # codepoints.  Checked over the COMPLETE input, not just the first scan chunk;
    # the three RGI subdivision flags are blanked first (their TAG chars are legitimate).
    raw_chars = set(EMOJI_TAG_SEQUENCE_RE.sub(lambda m: " " * len(m[0]), content))
    findings: list[str] = [f"invisible_unicode_U+{ord(ch):04X}" for ch in raw_chars & INVISIBLE_CHARS]
    # Threat patterns — scan the COMPLETE input in bounded chunks with overlap so a
    # payload past ``MAX_SCAN_CHARS`` (which the renderer may still emit as a tail)
    # is not silently skipped.  Findings are deduplicated: a pattern can legitimately
    # fire in more than one overlapping chunk.  The overlap is at least every
    # pattern's maximum match width (checked in ``_compile``), so a match straddling
    # a chunk edge is seen whole by the next chunk.
    # NFKC folds full-width / compatibility variants (ｃａｔ → cat) against
    # homograph bypass.  It does NOT fold cross-script confusables (Cyrillic
    # ``а``) — that needs a TR#39 database.  Each whitespace run then collapses to
    # one char (newline-preserving, for the ``[^\n]`` spans) so the bounded ``\s``
    # quantifiers still see arbitrarily long whitespace as a single separator.
    folded = unicodedata.normalize("NFKC", content)
    normalised = _WS_RUN.sub(_collapse_ws, folded)
    filler_view = _FILLER_BLIND.sub("\x00", normalised)
    # Masked before the collapse: a blank line between a negator and `forget` must still break the negation.
    masked = _NEGATOR_FORGET.sub(lambda m: m[1] + "x" * 6, _WS_RUN.sub(_collapse_ws, mask_negated_forget(folded)))
    found_ids: set[str] = set()
    step = MAX_SCAN_CHARS - _CHUNK_OVERLAP
    for start in range(0, max(len(normalised), len(masked)), step):
        end = start + MAX_SCAN_CHARS
        for compiled, pid in patterns:
            text = masked if pid == "forget_rules" else filler_view if pid in _FILLER_IDS else normalised
            exempt = _ENV_NAME_VALUE if pid == "hardcoded_secret" else None
            if pid not in found_ids and _window_has_match(compiled, text, start, end, exempt):
                found_ids.add(pid)
                findings.append(pid)
    return findings


def _window_has_match(compiled: re.Pattern, text: str, start: int, end: int, exempt: Optional[re.Pattern]) -> bool:
    """Whether ``compiled`` has a real match starting in ``text[start:end]``.

    ``pos`` keeps the preceding text visible to ``\\b`` and lookbehinds, but
    ``endpos`` makes the window end look like end-of-string, so a trailing
    ``\\b`` or a negative lookahead can accept a match the full text rejects
    (``$MONKEY`` cut from ``$MONKEY_URL``) — a lookahead can read up to the
    window end even when the match itself ends earlier.  Unless the window
    reaches the end of the text, each candidate is re-matched at its absolute
    position against the complete text (cheap: widths are bounded); a rejected
    one resumes the search one char later so a real match behind it is not
    skipped.  A candidate whose group 1 starts where ``exempt`` (if given) matches
    the complete text does not count either.
    """
    pos = start
    while (m := compiled.search(text, pos, end)) is not None:
        if (end >= len(text) or compiled.match(text, m.start())) and not (
                exempt is not None and exempt.match(text, m.start(1))):
            return True
        pos = m.start() + 1
    return False


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """User-facing error for the first threat found, or None (block-on-first-hit paths)."""
    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (f"Blocked: content matches threat pattern '{pid}'. "
            f"Content is injected into the system prompt and must not contain "
            f"injection or exfiltration payloads.")


__all__ = ["EMOJI_TAG_SEQUENCE_RE", "INVISIBLE_CHARS", "MAX_SCAN_CHARS", "first_threat_message", "scan_for_threats"]
