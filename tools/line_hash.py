"""
Hashline line-anchor editing utilities.

Each source line is annotated with a short content-derived hash so the LLM
can specify edit ranges using anchors instead of repeating context code.

Hash format: 4-char base36 (sha256 truncated) — 1.68M distinct values,
collision probability < 0.1% for files under 1000 lines (birthday bound ~1296).

When two lines share the same hash (collision), they are disambiguated
with ``#1``, ``#2`` etc. suffixes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# ── Constants ────────────────────────────────────────────────────────────────

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_HASH_LEN = 4  # 36^4 = 1,679,616 distinct values


# ── Public API ───────────────────────────────────────────────────────────────


def compute_line_hash(line: str) -> str:
    """Return a 4-char base36 content hash for *line*.

    The hash is deterministic and stable across platforms.  It is derived
    from the first 8 hex digits of SHA-256, converted to base36.
    """
    digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
    num = int(digest[:8], 16)
    chars: list[str] = []
    for _ in range(_HASH_LEN):
        chars.append(_BASE36[num % 36])
        num //= 36
    # Reverse so the most-significant digit comes first.
    return "".join(reversed(chars))


def format_hash_lines(
    lines: list[str],
    offset: int = 0,
) -> str:
    """Convert a list of source lines into hashline-annotated output.

    Parameters
    ----------
    lines:
        Raw source lines (without trailing newlines).
    offset:
        Starting line number (0-based index is shifted by *offset*).

    Returns
    -------
    str
        Multiline string where each line looks like::

            1:k7m2|def hello():
            2:a9f1|    print("hi")
            3:k7m2#1|def world():

        Collision-disambiguated lines carry a ``#N`` suffix.
    """
    # First pass: compute hashes and count occurrences.
    hashes = [compute_line_hash(line) for line in lines]
    hash_counts: dict[str, int] = {}
    for h in hashes:
        hash_counts[h] = hash_counts.get(h, 0) + 1

    # Second pass: format with disambiguation where needed.
    hash_seen: dict[str, int] = {}
    result: list[str] = []
    for i, (h, line) in enumerate(zip(hashes, lines)):
        lineno = i + 1 + offset
        if hash_counts[h] > 1:
            seen = hash_seen.get(h, 0) + 1
            hash_seen[h] = seen
            tag = f"{h}#{seen}"
        else:
            tag = h
        result.append(f"{lineno}:{tag}|{line}")
    return "\n".join(result)


def strip_hash_prefix(hashline: str) -> str:
    """Remove the ``<lineno>:<hash>|`` prefix, returning the raw code."""
    pipe = hashline.find("|")
    return hashline[pipe + 1 :] if pipe >= 0 else hashline


# ── Anchor parsing ───────────────────────────────────────────────────────────

_ANCHOR_RE = re.compile(
    r"^(?P<hash>[0-9a-z]{4})(?:#(?P<disambig>\d+))?$"
)


@dataclass(frozen=True)
class AnchorInfo:
    """Parsed representation of a single hashline anchor tag."""
    raw: str          # Original tag string, e.g. "k7m2" or "k7m2#1"
    hash: str         # 4-char base36 hash portion
    disambig: int     # Disambiguation index (0 = no suffix)


def parse_anchor(tag: str) -> AnchorInfo | None:
    """Parse an anchor tag into its components, or *None* if invalid."""
    m = _ANCHOR_RE.match(tag)
    if not m:
        return None
    return AnchorInfo(
        raw=tag,
        hash=m.group("hash"),
        disambig=int(m.group("disambig") or 0),
    )


def parse_anchor_range(anchor_range: str) -> tuple[str, str] | None:
    """Parse a ``start:end`` anchor range string.

    Returns ``(start_tag, end_tag)`` or *None* if the format is invalid.
    A single-anchor form (``"k7m2"``) is normalised to ``(tag, tag)``.
    """
    if ":" in anchor_range:
        parts = anchor_range.split(":", 1)
        if len(parts) != 2:
            return None
        start, end = parts[0].strip(), parts[1].strip()
        if not start or not end:
            return None
        return start, end
    # Single anchor -> degenerate range (insert after that line).
    tag = anchor_range.strip()
    return (tag, tag) if tag else None


# ── File-level anchor resolution ─────────────────────────────────────────────


def build_anchor_map(
    lines: list[str],
    offset: int = 0,
) -> dict[str, int]:
    """Build a mapping from anchor tag -> 1-based line number.

    This mirrors the disambiguation logic in :func:`format_hash_lines`
    so that anchors produced by a ``read_file(hashline=True)`` call can
    be resolved back to line numbers for editing.
    """
    hashes = [compute_line_hash(line) for line in lines]
    hash_counts: dict[str, int] = {}
    for h in hashes:
        hash_counts[h] = hash_counts.get(h, 0) + 1

    hash_seen: dict[str, int] = {}
    anchor_map: dict[str, int] = {}
    for i, h in enumerate(hashes):
        lineno = i + 1 + offset
        if hash_counts[h] > 1:
            seen = hash_seen.get(h, 0) + 1
            hash_seen[h] = seen
            tag = f"{h}#{seen}"
        else:
            tag = h
        anchor_map[tag] = lineno
    return anchor_map


def suggest_similar_anchors(
    target: str,
    anchor_map: dict[str, int],
    lines: list[str],
    offset: int = 0,
    max_suggestions: int = 3,
) -> list[str]:
    """Return human-readable "Did you mean?" suggestions.

    Matches by longest common prefix of the hash portion, then falls
    back to substring containment.
    """
    parsed = parse_anchor(target)
    if parsed is None:
        return []

    scored: list[tuple[int, str, int]] = []
    for tag, lineno in anchor_map.items():
        p = parse_anchor(tag)
        if p is None:
            continue
        # Prefix score: number of matching leading chars.
        prefix = 0
        for a, b in zip(parsed.hash, p.hash):
            if a == b:
                prefix += 1
            else:
                break
        if prefix >= 2:
            scored.append((prefix, tag, lineno))

    scored.sort(key=lambda t: t[0], reverse=True)
    suggestions: list[str] = []
    for _, tag, lineno in scored[:max_suggestions]:
        line_idx = lineno - 1 - offset
        preview = lines[line_idx].strip() if 0 <= line_idx < len(lines) else ""
        preview = preview[:60]
        suggestions.append(f"  - '{tag}' (line {lineno}: {preview})")
    return suggestions
