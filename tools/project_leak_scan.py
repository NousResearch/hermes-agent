"""Project-specific content detector for the skill-review path.

The post-turn background review fork (and manual ``/refine``, which shares the
prompt) distils session lessons into skills that live in the active profile
and are read by EVERY project on the machine. A lesson carrying one project's
names and paths changes behaviour in unrelated projects, so the writing side
must GENERALIZE: keep the class of task, drop the instance.

This module is the detector behind the advisory ``project-specific-content``
linter rule. It is pure and stdlib-only: no I/O, no network, no clock. It
never rewrites files on disk; :func:`sanitize` rewrites a candidate *string*
so tests can prove nothing instance-shaped survives.

Design notes:

* Findings are merged into non-overlapping spans and rules apply in a fixed
  order, so output cannot depend on set-iteration order (PYTHONHASHSEED-safe).
* Dynamic ``project_tokens`` are matched longest-first for the same reason.
* ``secret`` findings are fully masked: :meth:`LeakFinding.redacted` returns
  no prefix at all for that category — half a credential is still a
  credential.
* The rule is a writing discipline, not a security boundary: skills that
  legitimately document the owner's own machine will still match, which is why
  the linter surfaces these as WARNING, never ERROR.

# ponytail: regex heuristics, not a classifier — version-like quads and short
hex runs are the known ceiling; tighten only on a real false-positive report.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

SECRET = "secret"

_ABSOLUTE_PATH = "absolute_path"
_PROJECT_NAME = "project_name"
_SESSION_ID = "session_id"
_NETWORK_ADDRESS = "network_address"
_BRANCH_OR_WORKTREE = "branch_or_worktree"
_PROJECT_FILENAME = "project_filename"
_CONCRETE_TEST_VALUE = "concrete_test_value"

# Fixed application order. Earlier rules win overlapping spans, so a path that
# embeds the project name is one absolute_path finding, not two.
_RULE_ORDER = (
    SECRET, _NETWORK_ADDRESS, _ABSOLUTE_PATH, _SESSION_ID,
    _BRANCH_OR_WORKTREE, _PROJECT_FILENAME, _PROJECT_NAME, _CONCRETE_TEST_VALUE,
)

_NEVER_ECHO_CATEGORIES = frozenset({SECRET})

# Documented placeholders / generic filler a skill may legitimately contain.
_PLACEHOLDER_RE = re.compile(
    r"EXAMPLE|YOUR_|MY_|PLACEHOLDER|SAMPLE[_-]?|FOO|BAR|XXX|<[^<>\n]*>|\.\.\.", re.IGNORECASE)

# Documented placeholders / generic filler a skill may legitimately contain.
_PLACEHOLDER_RES = re.compile(
    r"EXAMPLE|YOUR_|MY_|PLACEHOLDER|SAMPLE[_-]?|FOO|BAR|XXX|<[^<>\n]*>|\.\.\.", re.IGNORECASE)

_SECRET_RES = (
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9\-._~+/=]{8,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|"
        r"auth[_-]?token|private[_-]?key|client[_-]?secret)\s*[:=]\s*['\"]?"
        r"[\w\-.~+/=]{8,}['\"]?"),
    re.compile(r"\b(?:sk-[A-Za-z0-9\-]{8,}|ghp_[A-Za-z0-9]{8,}|gho_[A-Za-z0-9]{8,}|"
               r"AKIA[0-9A-Z]{16}|xox[bpas]-[A-Za-z0-9\-]{8,})\b"),
)

_NETWORK_RES = (
    re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
               r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?::\d{1,5})?\b"),
)

_UNIX_FIRST = ("home", "Users", "root", "data", "mnt", "srv", "var", "opt", "tmp",
               "private", "workspace", "workspaces", "repos?", "projects?", "code",
               "dev", "src", "etc")
_ABSOLUTE_PATH_RES = (
    re.compile(r"(?<![\w~./\-])/(?:" + "|".join(_UNIX_FIRST) + r")\b[^\s\"'<>|?*]*"),
    re.compile(r"\b[A-Za-z]:[\\/][A-Za-z0-9 _$\-+.(){}\[\]~%]+(?:[\\/][A-Za-z0-9 _$\-+.(){}\[\]~%]+)+"),
    re.compile(r"\\\\[\w.\-]+\\[^\s\"'<>]+"),
    re.compile(r"~/(?!\.?hermes/)[^\s\"'<>|?*]+"),
)
_WINDOWS_SYSTEM_PREFIXES = ("c:\\program files", "c:\\program files (x86)",
                            "c:\\windows", "c:\\programdata")

_SESSION_ID_RES = (
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
               r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\b(?:session[_-]?id|sess[_-]?id)\s*[:=]\s*['\"]?[\w\-]{4,}['\"]?",
               re.IGNORECASE),
    re.compile(r"\b[0-9a-fA-F]{16,}\b"),
)

_BRANCH_RES = (
    re.compile(r"\b(?:fix|feat|feature|bugfix|hotfix|chore|docs|refactor|perf|"
               r"test|wip|revert|release)/[A-Za-z0-9_.\-]+[0-9][A-Za-z0-9_./.\-]*"),
    re.compile(r"\bwt-\d[\w\-]*"),
    re.compile(r"\bworktree[_-]?\d[\w\-]*", re.IGNORECASE),
)

_STATUS_STEMS = ("STATUS", "STATE", "SESSION", "CONTEXT", "MEMORY")
_FILENAME_EXTS = ("json", "md", "yaml", "yml", "txt")
_FILENAME_PREFIX_ALLOW = ("MY_", "YOUR_", "EXAMPLE_", "TEST_", "SAMPLE_",
                          "PLACEHOLDER_", "FOO_", "DEMO_")
_PROJECT_FILENAME_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_\-]*(?:" + "|".join(_STATUS_STEMS) + r")[A-Za-z0-9_\-]*\."
    r"(?:" + "|".join(_FILENAME_EXTS) + r")\b")

_CONCRETE_TEST_VALUE_RE = re.compile(
    r"\b(?:test[_-]?(?:user|id|value|token|key|name|data)|fixture[_-]?\w*|"
    r"dummy[_-]?\w*|fake[_-]?\w*)\s*[:=]\s*['\"]?[\w@.+\-]+['\"]?", re.IGNORECASE)


@dataclass(frozen=True)
class LeakFinding:
    """One instance-shaped span. ``excerpt`` holds the raw match (never log a secret)."""

    category: str
    span: Tuple[int, int]
    excerpt: str

    def redacted(self) -> str:
        """Render for humans: 4-char prefix plus mask — except secrets, fully masked."""
        if self.category in _NEVER_ECHO_CATEGORIES:
            return "<masked:secret>"
        return f"{self.excerpt[:4]}…<masked:{len(self.excerpt)}>"


def _spans(patterns: Iterable[re.Pattern], text: str, category: str) -> List[LeakFinding]:
    out = []
    for rx in patterns:
        for m in rx.finditer(text):
            if _PLACEHOLDER_RES.search(m.group(0)):
                continue
            if category == _ABSOLUTE_PATH:
                lowered = m.group(0).lower()
                if lowered.startswith(_WINDOWS_SYSTEM_PREFIXES):
                    continue
            out.append(LeakFinding(category, (m.start(), m.end()), m.group(0)))
    return out


def _project_name_spans(text: str, tokens: Sequence[str]) -> List[LeakFinding]:
    ordered = sorted(set(t for t in tokens if t), key=lambda t: (-len(t), t))
    out = []
    for tok in ordered:
        for m in re.finditer(re.escape(tok), text, re.IGNORECASE):
            out.append(LeakFinding(_PROJECT_NAME, (m.start(), m.end()), m.group(0)))
    return out


def _project_filename_spans(text: str, tokens: Sequence[str]) -> List[LeakFinding]:
    out = []
    for m in _PROJECT_FILENAME_RE.finditer(text):
        word = m.group(0)
        if word.upper().startswith(_FILENAME_PREFIX_ALLOW):
            continue
        if _PLACEHOLDER_RES.search(word):
            continue
        out.append(LeakFinding(_PROJECT_FILENAME, (m.start(), m.end()), word))
    for tok in sorted(set(t for t in tokens if t), key=lambda t: (-len(t), t)):
        rx = re.compile(r"\b[\w.\-]*" + re.escape(tok) + r"[\w.\-]*\.(?:" +
                        "|".join(_FILENAME_EXTS) + r")\b", re.IGNORECASE)
        for m in rx.finditer(text):
            if _PLACEHOLDER_RES.search(m.group(0)):
                continue
            out.append(LeakFinding(_PROJECT_FILENAME, (m.start(), m.end()), m.group(0)))
    return out


def _merge(findings: List[LeakFinding]) -> List[LeakFinding]:
    """Merge overlapping spans; the earlier rule in _RULE_ORDER wins the overlap."""
    order = {cat: i for i, cat in enumerate(_RULE_ORDER)}
    ranked = sorted(findings, key=lambda f: (f.span[0], order.get(f.category, 99), f.span[1]))
    merged: List[LeakFinding] = []
    for f in ranked:
        if merged and f.span[0] < merged[-1].span[1]:
            s, e = merged[-1].span
            if f.span[1] > e:
                merged[-1] = LeakFinding(
                    merged[-1].category, (s, f.span[1]),
                    merged[-1].excerpt + f.excerpt[max(0, e - f.span[0]):])
            continue
        merged.append(f)
    return merged


def scan(text: str, *, project_tokens: Sequence[str] = ()) -> List[LeakFinding]:
    """Return merged, deterministically ordered findings for *text*."""
    findings = [
        *_spans(_SECRET_RES, text, SECRET),
        *_spans(_NETWORK_RES, text, _NETWORK_ADDRESS),
        *_spans(_ABSOLUTE_PATH_RES, text, _ABSOLUTE_PATH),
        *_spans(_SESSION_ID_RES, text, _SESSION_ID),
        *_spans(_BRANCH_RES, text, _BRANCH_OR_WORKTREE),
        *_project_filename_spans(text, project_tokens),
        *_project_name_spans(text, project_tokens),
        *_spans((_CONCRETE_TEST_VALUE_RE,), text, _CONCRETE_TEST_VALUE),
    ]
    return _merge(findings)


def sanitize(text: str, *, project_tokens: Sequence[str] = ()) -> str:
    """Replace every finding span with a category label (for tests; never touches disk)."""
    parts = []
    cursor = 0
    for f in scan(text, project_tokens=project_tokens):
        s, e = f.span
        parts.append(text[cursor:s])
        parts.append(f"[project-leak:{f.category}]")
        cursor = e
    parts.append(text[cursor:])
    return "".join(parts)


def session_tokens(*, cwd: Optional[str] = None) -> Tuple[str, ...]:
    """Best-effort project-name tokens from the session working directory.

    Returns the directory basename plus separator variants, longest-first, so a
    bare repository name in skill prose can be recognised too. Without tokens
    the detector still catches paths, ids, addresses, filenames and branches.
    """
    try:
        base = os.path.basename(os.path.normpath(cwd or os.getcwd()))
    except Exception:
        return ()
    if not base:
        return ()
    variants = {base, base.replace("-", "_"), base.replace("_", "-")}
    for part in re.split(r"[-_\s]+", base):
        if len(part) >= 4:
            variants.add(part)
    return tuple(sorted(variants, key=lambda t: (-len(t), t)))
