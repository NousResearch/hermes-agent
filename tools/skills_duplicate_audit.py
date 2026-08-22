"""
Deterministic near-duplicate candidate audit for skills — opt-in diagnostic,
not a merge decision.

The curator's consolidation pass is LLM-driven and opt-in (``curator.consolidate``,
default false), so most registries accumulate near-identical skills that nothing
ever collapses. This module answers a narrower and much cheaper question with no
model call and no external dependencies: *which skill pairs look alike enough
that a human should look?*

It reports **candidates**, never verdicts. Every signal here is lexical, so two
skills can share their name shape and every heading and still do genuinely
different things — the report exists to point a human at a short list, not to
authorize a merge. Nothing in this module writes to the skill library.

Pairs are reported on their own and never chained into groups: if A resembles B
and B resembles C, that does not make A and C duplicates, and collapsing them
into one cluster is how a "cleanup" silently deletes an unrelated skill.

CLI: ``hermes curator audit``
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple


class DuplicateCandidate(NamedTuple):
    """One pair of skills that look alike, with the evidence for saying so."""

    name_a: str
    name_b: str
    confidence: str  # "high" | "medium"
    signals: Tuple[str, ...]
    ownership_a: str
    ownership_b: str


# Lexical thresholds. Deliberately conservative: a missed duplicate costs a line
# of registry bloat, a false one costs a user's trust in the whole report.
_NAME_OVERLAP_HIGH = 0.5
_HEADING_OVERLAP_HIGH = 0.6
_DESC_SIMILARITY_MIN = 0.85

# Bodies shorter than this are too small to carry a meaningful hash match — a
# registry full of stub skills would otherwise report every stub against every
# other one.
_MIN_BODY_CHARS = 40

_CONFIDENCE_ORDER = {"high": 0, "medium": 1}

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

# Headings only mean something when they distinguish. Two layers strip the ones
# that don't:
#
# 1. CONTRIBUTING.md prescribes an exact section scaffold for SKILL.md, so a
#    well-authored skill is *expected* to carry these. An early version of this
#    scanner paired a voice-cloning skill with an S3 upload skill on
#    "When to Use / Procedure / Pitfalls" alone.
# 2. Skill families invent their own templates that no fixed list can predict —
#    the bundled fine-tuning skills share ten headings ("Working with this
#    skill", "For beginners", …) while doing entirely different things. So any
#    heading common across the registry is discounted too, whatever its origin.
_SCAFFOLD_HEADINGS = frozenset(
    {
        "when to use",
        "prerequisites",
        "how to run",
        "quick reference",
        "procedure",
        "pitfalls",
        "verification",
        "overview",
        "usage",
        "examples",
        "notes",
    }
)

# A heading carried by this fraction of the registry is boilerplate, not
# identity. Below _MIN_CORPUS_FOR_DF skills there is nothing to estimate from,
# so only the fixed scaffold above applies.
_BOILERPLATE_DF_RATIO = 0.15
_MIN_CORPUS_FOR_DF = 8


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Mirrors the normalization already used for near-duplicate transcript
    detection in the gateway, so "Git Commit Helper" and "git-commit_helper"
    compare equal.
    """
    if not text:
        return ""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def _name_tokens(name: str) -> Set[str]:
    """Token set for a skill name: ``ai-voice-cloning`` -> {ai, voice, cloning}."""
    return {tok for tok in _normalize(name.replace("-", " ").replace("_", " ")).split() if tok}


def _headings(body: str) -> Set[str]:
    """Distinctive normalized heading texts, used as a structural fingerprint.

    Template scaffold headings are dropped — see ``_SCAFFOLD_HEADINGS``.
    """
    found = {_normalize(h) for h in _HEADING_RE.findall(body)}
    return {norm for norm in found if norm and norm not in _SCAFFOLD_HEADINGS}


def _body_digest(body: str) -> str:
    """SHA-256 of the normalized body, or "" when the body is too short to judge."""
    norm = _normalize(body)
    if len(norm) < _MIN_BODY_CHARS:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Set overlap. Two empty sets share nothing, so they score 0, not 1."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _ownership(skill_name: str, skill_path: Path) -> str:
    """Human-facing origin label, with a note when the skill can't be curated.

    Path is checked before name: ``skills.external_dirs`` entries are
    externally owned and read-only, but ``skill_usage.provenance()`` has no
    concept of "external" — it classifies purely by name against the hub and
    bundled manifests. A skill discovered only in an external directory can
    share its name with an unrelated hub or bundled entry, in which case
    naming-only classification reports the wrong origin for the file this
    scan actually read. Checking the discovered path first avoids that.
    """
    try:
        from agent.skill_utils import is_external_skill_path

        if is_external_skill_path(skill_path):
            return "external, read-only"
    except Exception:
        pass

    try:
        from tools import skill_usage

        label = skill_usage.provenance(skill_name)
        if skill_usage.is_protected_builtin(skill_name):
            return f"{label}, protected"
        return label
    except Exception:
        # Usage state is optional context for the report, never a prerequisite
        # for producing it.
        return "unknown"


def _load_skills(skills_dirs: Optional[Sequence[Path]] = None) -> List[Dict[str, object]]:
    """Read every active SKILL.md into the fields the comparison needs.

    Archived skills live under ``.archive/`` and support directories hold
    progressive-disclosure data, not active skills; both are already excluded by
    ``iter_skill_index_files``, so restoring an archived skill is what brings it
    back into the audit.
    """
    from agent.skill_utils import (
        get_all_skills_dirs,
        iter_skill_index_files,
        parse_frontmatter,
    )

    roots: Iterable[Path] = skills_dirs if skills_dirs is not None else get_all_skills_dirs()

    skills: List[Dict[str, object]] = []
    seen: Set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for index_file in iter_skill_index_files(root, "SKILL.md"):
            try:
                content = index_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            frontmatter, body = parse_frontmatter(content)
            raw_name = frontmatter.get("name") if isinstance(frontmatter, dict) else None
            name = str(raw_name).strip() if raw_name else index_file.parent.name
            if not name or name in seen:
                continue
            seen.add(name)
            description = ""
            if isinstance(frontmatter, dict):
                description = str(frontmatter.get("description") or "")
            skills.append(
                {
                    "name": name,
                    "path": index_file.parent,
                    "tokens": _name_tokens(name),
                    "headings": _headings(body),
                    "digest": _body_digest(body),
                    "description": _normalize(description),
                }
            )
    skills.sort(key=lambda s: str(s["name"]))
    return skills


def _boilerplate_headings(skills: Sequence[Dict[str, object]]) -> Set[str]:
    """Headings so common in this registry that they say nothing about a skill."""
    if len(skills) < _MIN_CORPUS_FOR_DF:
        return set()
    counts: Counter = Counter()
    for skill in skills:
        counts.update(skill["headings"])  # type: ignore[arg-type]
    cutoff = max(2, math.ceil(_BOILERPLATE_DF_RATIO * len(skills)))
    return {heading for heading, count in counts.items() if count >= cutoff}


def _compare(a: Dict[str, object], b: Dict[str, object]) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Judge one pair. Returns (confidence, signals) or None when not a candidate."""
    signals: List[str] = []

    digest_a, digest_b = str(a["digest"]), str(b["digest"])
    identical_body = bool(digest_a) and digest_a == digest_b
    if identical_body:
        signals.append("identical normalized body hash")

    name_overlap = _jaccard(a["tokens"], b["tokens"])  # type: ignore[arg-type]
    heading_overlap = _jaccard(a["distinctive"], b["distinctive"])  # type: ignore[arg-type]
    desc_similarity = _ratio(str(a["description"]), str(b["description"]))

    if name_overlap >= _NAME_OVERLAP_HIGH:
        signals.append(f"high normalized-name overlap ({name_overlap:.2f})")
    if heading_overlap >= _HEADING_OVERLAP_HIGH:
        signals.append(f"matching document headings ({heading_overlap:.2f})")
    if desc_similarity >= _DESC_SIMILARITY_MIN:
        signals.append(f"similar descriptions ({desc_similarity:.2f})")

    # A shared name shape on its own is not evidence: `aws-s3-upload` and
    # `aws-s3-delete` are siblings, not duplicates. Require the content to agree
    # before a pair is worth a human's attention at all.
    content_agrees = (
        identical_body
        or heading_overlap >= _HEADING_OVERLAP_HIGH
        or desc_similarity >= _DESC_SIMILARITY_MIN
    )
    if not content_agrees:
        return None

    if identical_body or (
        name_overlap >= _NAME_OVERLAP_HIGH and heading_overlap >= _HEADING_OVERLAP_HIGH
    ):
        return "high", tuple(signals)
    return "medium", tuple(signals)


def scan_duplicates(
    skills_dirs: Optional[Sequence[Path]] = None,
) -> List[DuplicateCandidate]:
    """Report skill pairs that look like near-duplicates.

    Read-only. Pass *skills_dirs* to scan specific roots; the default is every
    configured skills directory. Ordering is deterministic (highest confidence
    first, then alphabetical) so the output can be diffed between runs.
    """
    skills = _load_skills(skills_dirs)

    boilerplate = _boilerplate_headings(skills)
    for skill in skills:
        skill["distinctive"] = skill["headings"] - boilerplate  # type: ignore[operator]

    candidates: List[DuplicateCandidate] = []
    for i, left in enumerate(skills):
        for right in skills[i + 1 :]:
            verdict = _compare(left, right)
            if verdict is None:
                continue
            confidence, signals = verdict
            name_a, name_b = str(left["name"]), str(right["name"])
            candidates.append(
                DuplicateCandidate(
                    name_a=name_a,
                    name_b=name_b,
                    confidence=confidence,
                    signals=signals,
                    ownership_a=_ownership(name_a, Path(str(left["path"]))),
                    ownership_b=_ownership(name_b, Path(str(right["path"]))),
                )
            )

    candidates.sort(key=lambda c: (_CONFIDENCE_ORDER.get(c.confidence, 9), c.name_a, c.name_b))
    return candidates


def summarize(candidates: Sequence[DuplicateCandidate]) -> Dict[str, int]:
    """Counts for the ``hermes curator status`` health block."""
    return {
        "possible_duplicate_pairs": len(candidates),
        "high_confidence_pairs": sum(1 for c in candidates if c.confidence == "high"),
        "medium_confidence_pairs": sum(1 for c in candidates if c.confidence == "medium"),
    }


def format_duplicate_report(candidates: Sequence[DuplicateCandidate]) -> str:
    """Plain-text report grouped by confidence."""
    if not candidates:
        return "Duplicate candidates:\n  None found."

    lines = ["Duplicate candidates:"]
    for label in ("high", "medium"):
        bucket = [c for c in candidates if c.confidence == label]
        if not bucket:
            continue
        lines.append("")
        lines.append(f"  {label.capitalize()} confidence ({len(bucket)}):")
        for cand in bucket:
            lines.append("")
            lines.append(f"    {cand.name_a} <-> {cand.name_b}")
            lines.append("      Signals:")
            for signal in cand.signals:
                lines.append(f"        - {signal}")
            lines.append("      Ownership:")
            lines.append(f"        - {cand.name_a}: {cand.ownership_a}")
            lines.append(f"        - {cand.name_b}: {cand.ownership_b}")
    lines.append("")
    lines.append("  Note: lexical candidates for human review, not merge decisions.")
    return "\n".join(lines)
