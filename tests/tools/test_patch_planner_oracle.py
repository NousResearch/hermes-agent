"""Bounded unhinted V4A oracle. Independent of the production planner.

Spec: tests/tools/ORACLE_SPEC.md. Expected answers are not taken from
apply_v4a_operations. Fault injection is test-only (_MUTATION).

Closed-domain fuzz reads HERMES_PATCH_ORACLE_SEED and
HERMES_PATCH_ORACLE_COUNT (each default 2000). For scripts/run_tests.sh,
use -- --oracle-seed=N --oracle-count=N: its hermetic environment drops
unlisted environment variables.

Generated similar-section rows and grammar-pin classes fail on that known
planner shape, so a seed or count that does not generate them (including 0)
passes. They are not xfail marks. Site comparison unavailable is a
SITE_UNAVAILABLE count, not a pass, xfail, or failure; the default fuzz
fails if that count grows above 60.
"""
from __future__ import annotations

import itertools
import os
import random
import re
from collections import Counter
from dataclasses import dataclass

import pytest

ALPHABET = (
    b"a", b"b", b"c", b"x", b"y", b"old", b"new", b"keep", b"tail",
    b"", b"a b", b"a\tb",
)
_ALPHA = frozenset(ALPHABET)
_NORM = {b"a b": b"a\tb", b"a\tb": b"a b"}
_HEAD = b"*** Begin Patch\n*** Update File: f\n@@\n"
_TAIL = b"*** End Patch\n"
_PREFIX = "Patch validation failed (no files were modified):\n  • f: "
_AMB_N = re.compile(r"ambiguous \((\d+) matches\)")
# The planner's Found-N diagnostic normally includes extra lines, so its live
# form is UNMAPPED; the single-line form is retained by the approved spec.
_FOUND_N = re.compile(r"Found (\d+) matches for old_string")
_NOT_FOUND = re.compile(
    r"hunk [0-9]+ \(no hint\) not found — "
    r"Could not find a match for old_string in the file\Z"
)
SEED = 20261106
SAMPLE_N = 32
_MUTATION = None


@dataclass(frozen=True)
class Write:
    site: tuple[int, int]
    new_text: bytes


@dataclass(frozen=True)
class Refuse:
    code: str


class OutOfDomain:
    def __repr__(self) -> str:
        return "OUT_OF_DOMAIN"


OOD = OutOfDomain()


def set_mutation(name: str | None) -> None:
    global _MUTATION
    _MUTATION = name


def _key(style, final, kind, target, tier, pos, card):
    rules = (1, 2, 3)
    if card != "one":
        rules += (5,)
    if kind == "delete":
        rules += (6,)
    return (style, final, kind, target, tier, pos, card, rules)


def decode_lines(data: bytes):
    """Physical (content, terminator) pairs. No empty trailing suffix."""
    if not data:
        return None
    lines = []
    i, n = 0, len(data)
    while i < n:
        j = data.find(b"\n", i)
        if j < 0:
            lines.append((data[i:], b""))
            break
        if j > i and data[j - 1:j] == b"\r":
            content, term, i = data[i:j - 1], b"\r\n", j + 1
        else:
            content, term, i = data[i:j], b"\n", j + 1
        if b"\n" in content or b"\r" in content:
            return None
        lines.append((content, term))
    if any(not c and not t for c, t in lines):
        return None
    return lines


def file_style(lines):
    terms = {t for _, t in lines if t}
    if len(terms) != 1:
        return None
    term = next(iter(terms))
    return term if term in (b"\n", b"\r\n") else None


def _winners(lines, search):
    exact = [i for i, (c, _) in enumerate(lines) if c == search]
    other = _NORM.get(search)
    norm = [i for i, (c, _) in enumerate(lines) if c == other] if other is not None else []
    if _MUTATION == "tier-order":
        if norm:
            return norm, "normalized"
        return (exact, "exact") if exact else ([], "none")
    if exact:
        return exact, "exact"
    return (norm, "normalized") if norm else ([], "none")


def parse_hunk(patch: bytes):
    if (
        not isinstance(patch, (bytes, bytearray))
        or not patch.startswith(_HEAD)
        or not patch.endswith(_TAIL)
        or b"\r" in patch
    ):
        return None
    body = patch[len(_HEAD):-len(_TAIL)]
    if b"@@" in body or not body.startswith(b"-"):
        return None
    nl = body.find(b"\n")
    if nl < 1:
        return None
    search, rest = body[1:nl], body[nl + 1:]
    if not rest:
        return search, None
    if not rest.startswith(b"+") or rest.count(b"\n") != 1 or not rest.endswith(b"\n"):
        return None
    repl = rest[1:-1]
    if repl == search or b"\n" in repl:
        return None
    return search, repl


def recognize(file, patch):
    parsed = parse_hunk(patch)
    if parsed is None or not isinstance(file, (bytes, bytearray)):
        return None
    search, repl = parsed
    if search not in _ALPHA or (repl is not None and repl not in _ALPHA):
        return None
    lines = decode_lines(bytes(file))
    if not lines or not 1 <= len(lines) <= 4:
        return None
    if any(c not in _ALPHA for c, _ in lines) or file_style(lines) is None:
        return None
    if _MUTATION != "allow-substring" and search and any(
        c != search and search in c for c, _ in lines
    ):
        return None
    return lines, search, repl, file_style(lines)


def evaluate(file: bytes, patch: bytes):
    rec = recognize(file, patch)
    if rec is None:
        return OOD
    lines, search, repl, style = rec
    winners, _tier = _winners(lines, search)
    if (
        _MUTATION != "ignore-unterminated"
        and repl is not None
        and any(not lines[i][1] for i in winners)
    ):
        return OOD
    if _MUTATION == "first-match" and len(winners) > 1:
        winners = winners[:1]
    if not winners:
        return Refuse("NOT_FOUND")
    if len(winners) != 1:
        return Refuse("AMBIGUOUS")
    index = winners[0]
    if _MUTATION == "keep-blank-on-delete" and repl is None:
        chunks = [style if j == index else c + t for j, (c, t) in enumerate(lines)]
        return Write((index, index + 1), b"".join(chunks))
    chunks = []
    for j, (c, t) in enumerate(lines):
        if j == index:
            if repl is not None:
                chunks.append(repl + style)
        else:
            chunks.append(c + t)
    return Write((index, index + 1), b"".join(chunks))


def tags_from_bytes(file, patch):
    rec = recognize(file, patch)
    if rec is None:
        return None
    lines, search, repl, style = rec
    winners, tier = _winners(lines, search)
    if repl is not None and any(not lines[i][1] for i in winners):
        return None
    n = len(winners)
    card = "zero" if n == 0 else "one" if n == 1 else "many"
    if card == "zero":
        tier = "none"
    pos = "none" if card != "one" else ("final" if winners[0] == len(lines) - 1 else "nonfinal")
    final = "unterminated" if lines[-1][1] == b"" else "terminated"
    kind = "delete" if repl is None else "replace"
    target = "blank" if search == b"" else "nonblank"
    style_name = "CRLF" if style == b"\r\n" else "LF"
    return _key(style_name, final, kind, target, tier, pos, card)


def classify_error(error) -> str:
    """Map PatchResult.error. UNMAPPED is not a refusal code."""
    if not isinstance(error, str) or not error.startswith(_PREFIX) or error.count("•") != 1:
        return "UNMAPPED"
    payload = error[len(_PREFIX):]
    if "\n" in payload or "empty search text" in payload:
        return "UNMAPPED"
    for rx in (_AMB_N, _FOUND_N):
        match = rx.search(payload)
        if match and int(match.group(1)) >= 2:
            return "AMBIGUOUS"
    if _NOT_FOUND.fullmatch(payload):
        return "NOT_FOUND"
    return "UNMAPPED"


def site_from_diff(diff: str, original: bytes):
    """Site consistency check from output bytes. None means unavailable, not a pass."""
    if not diff:
        return None
    hunks = []
    cur = None
    for line in diff.splitlines(keepends=True):
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)", line)
            if not match:
                return None
            cur = [int(match.group(1)), []]
            hunks.append(cur)
        elif cur is not None:
            cur[1].append(line)
    removed = [
        (h, line)
        for h in hunks
        for line in h[1]
        if line.startswith("-") and not line.startswith("---")
    ]
    if len(removed) != 1:
        return None
    hunk, _only = removed[0]
    cursor = hunk[0] - 1
    recorded = payload = None
    for line in hunk[1]:
        if line.startswith("\\"):
            return None
        if line.startswith("+") and not line.startswith("+++"):
            continue
        if line.startswith("-") and not line.startswith("---"):
            recorded, payload = cursor, line[1:]
            cursor += 1
        elif line.startswith(" "):
            cursor += 1
        else:
            return None
    if recorded is None or payload is None or not payload.endswith("\n"):
        return None
    lines = decode_lines(original)
    if not lines or not 0 <= recorded < len(lines):
        return None
    content, term = lines[recorded]
    if payload.encode("ascii") != content + term:
        return None
    if sum(c == content for c, _ in lines) != 1:
        return None
    return (recorded, recorded + 1)


def _render(contents, style, unterminated, search, repl):
    term = b"\r\n" if style == "CRLF" else b"\n"
    last = len(contents) - 1
    file = b"".join(
        content if unterminated and i == last else content + term
        for i, content in enumerate(contents)
    )
    body = b"-" + search + b"\n" + (b"" if repl is None else b"+" + repl + b"\n")
    return file, _HEAD + body + _TAIL


def _canon_repl(kind, search):
    if kind == "delete":
        return None
    return b"new" if search != b"new" else b"old"


def _add_witnesses(witnesses, contents, unterminated, search, repl, final, kind, target, tier, pos, card):
    for style in ("LF", "CRLF"):
        key = _key(style, final, kind, target, tier, pos, card)
        if key not in witnesses:
            witnesses[key] = (*_render(contents, style, unterminated, search, repl), key)


def enumerate_corpus(seed: int = SEED, sample_n: int = SAMPLE_N):
    """Exhaustive feasible classes, one enumerated witness each, then a sample."""
    rng = random.Random(seed)
    alpha = ALPHABET
    decoy = {
        s: frozenset(c for c in alpha if s and c != s and s in c) for s in alpha
    }
    norm_at = {i: alpha.index(_NORM[s]) if s in _NORM else None for i, s in enumerate(alpha)}
    witnesses = {}
    reservoir = []
    seen = 0
    for combo in itertools.chain.from_iterable(
        itertools.product(range(len(alpha)), repeat=n) for n in range(1, 5)
    ):
        n = len(combo)
        for unterminated in (False, True):
            if unterminated and (n == 1 or alpha[combo[-1]] == b""):
                continue
            contents = tuple(alpha[i] for i in combo)
            for si, search in enumerate(alpha):
                if decoy[search].intersection(contents):
                    continue
                exact = [i for i, c in enumerate(combo) if c == si]
                other = norm_at[si]
                norm = [i for i, c in enumerate(combo) if c == other] if other is not None else []
                if exact:
                    winners, tier = exact, "exact"
                elif norm:
                    winners, tier = norm, "normalized"
                else:
                    winners, tier = [], "none"
                card = "zero" if not winners else "one" if len(winners) == 1 else "many"
                if card == "zero":
                    tier = "none"
                pos = "none" if card != "one" else (
                    "final" if winners[0] == n - 1 else "nonfinal"
                )
                final = "unterminated" if unterminated else "terminated"
                target = "blank" if search == b"" else "nonblank"
                repl_ood = bool(winners) and unterminated and (n - 1) in winners
                for kind, blocked in (("delete", False), ("replace", repl_ood)):
                    if blocked:
                        continue
                    seen += 1
                    take = len(reservoir) < sample_n
                    slot = None
                    if not take:
                        slot = rng.randrange(seen)
                        take = slot < sample_n
                    repl = _canon_repl(kind, search)
                    _add_witnesses(witnesses, contents, unterminated, search, repl,
                                   final, kind, target, tier, pos, card)
                    if take:
                        style = "CRLF" if (slot or 0) % 2 else "LF"
                        sample_repl = repl
                        if kind == "replace" and slot is not None:
                            sample_repl = rng.choice([a for a in alpha if a != search])
                        key = _key(style, final, kind, target, tier, pos, card)
                        item = (
                            *_render(contents, style, unterminated, search, sample_repl),
                            key,
                        )
                        if len(reservoir) < sample_n:
                            reservoir.append(item)
                        else:
                            reservoir[slot] = item
    return witnesses, reservoir, {"seed": seed, "enumerated_parameter_sets": seen}


_CORPUS = None


def corpus():
    global _CORPUS
    if _MUTATION is not None:
        raise RuntimeError("corpus requires an unmutated oracle")
    if _CORPUS is None:
        _CORPUS = enumerate_corpus()
    return _CORPUS


# END ORACLE

def _patch(body: bytes) -> bytes:
    return body if body.startswith(b"***") else _HEAD + body + _TAIL


def _eq(got, expected) -> bool:
    return got is expected if expected is OOD else got == expected


# name, file, body or full patch, expected. Full patches start with ***.
SPEC = [
    ("r1-lf", b"a\nold\ntail\n", b"-old\n+new\n", Write((1, 2), b"a\nnew\ntail\n")),
    ("r1-crlf", b"a\r\nold\r\ntail\r\n", b"-old\n+new\n", Write((1, 2), b"a\r\nnew\r\ntail\r\n")),
    ("r1-unterminated-replace", b"a\nold", b"-old\n+new\n", OOD),
    ("r2-unique", b"old\ntail\n", b"-old\n+new\n", Write((0, 1), b"new\ntail\n")),
    ("r2-ambiguous", b"old\nold\n", b"-old\n+new\n", Refuse("AMBIGUOUS")),
    ("r2-not-found", b"a\nb\n", b"-old\n+new\n", Refuse("NOT_FOUND")),
    ("r2-normalized", b"a\tb\n", b"-a b\n+new\n", Write((0, 1), b"new\n")),
    ("r2-exact-over-normalized", b"a\tb\na b\n", b"-a b\n+new\n", Write((1, 2), b"a\tb\nnew\n")),
    ("r2-normalized-ambiguous", b"a\tb\na\tb\n", b"-a b\n+new\n", Refuse("AMBIGUOUS")),
    ("r2-normalized-other-way", b"a b\n", b"-a\tb\n+new\n", Write((0, 1), b"new\n")),
    ("r3-unique", b"a\nold\n", b"-old\n+new\n", Write((1, 2), b"a\nnew\n")),
    ("r3-hint", b"old\n", b"*** Begin Patch\n*** Update File: f\n@@ hint @@\n-old\n+new\n*** End Patch\n", OOD),
    ("r4-two-hunks", b"old\n", b"-old\n+new\n@@\n-old\n+new\n", OOD),
    ("r4-replacement-present", b"old\nnew\n", b"-old\n+new\n@@\n-old\n+new\n", OOD),
    ("r4-repeated", b"old\nold\n", b"-old\n+new\n@@\n-old\n+new\n", OOD),
    ("r5-not-found", b"a\n", b"-old\n+new\n", Refuse("NOT_FOUND")),
    ("r5-ambiguous", b"old\nold\n", b"-old\n+new\n", Refuse("AMBIGUOUS")),
    ("r5-second-hunk", b"old\n", b"-old\n+new\n@@\n-a\n+b\n", OOD),
    ("r6-delete", b"a\nold\nb\n", b"-old\n", Write((1, 2), b"a\nb\n")),
    ("r6-blank-delete", b"a\r\n\r\nb\r\n", b"-\n", Write((1, 2), b"a\r\nb\r\n")),
    ("r6-blank-ambiguous", b"a\n\nb\n\n", b"-\n", Refuse("AMBIGUOUS")),
    ("delete-unterminated-final", b"a\nold", b"-old\n", Write((1, 2), b"a\n")),
    ("blank-replacement", b"a\nold\nb\n", b"-old\n+\n", Write((1, 2), b"a\n\nb\n")),
    ("identical-bytes-normalized", b"a\tb\n", b"-a b\n+a\tb\n", Write((0, 1), b"a\tb\n")),
    ("equal-search-replacement", b"old\n", b"-old\n+old\n", OOD),
    ("substring-decoy", b"a b\n", b"-a\n+new\n", OOD),
    ("mixed-lf-crlf", b"a\nold\r\n", b"-old\n+new\n", OOD),
    ("empty-file", b"", b"-old\n+new\n", OOD),
    ("malformed-wrapper", b"old\n", b"*** Update File: f\n@@\n-old\n+new\n*** End Patch\n", OOD),
    ("ambiguous-unterminated-replace", b"old\nold", b"-old\n+new\n", OOD),
]
NEGATIVES = {
    "r1-unterminated-replace", "r3-hint", "r4-two-hunks", "r4-replacement-present",
    "r4-repeated", "r5-second-hunk", "equal-search-replacement", "substring-decoy",
    "mixed-lf-crlf", "empty-file", "malformed-wrapper", "ambiguous-unterminated-replace",
}
# These fail structural membership before search. Unterminated winners are
# structurally legal and become OOD only after the winning tier is known.
STRUCTURAL_NEGATIVES = NEGATIVES - {
    "r1-unterminated-replace", "ambiguous-unterminated-replace",
}
_TIER_POS_CARD = (
    ("none", "none", "zero"),
    ("exact", "final", "one"),
    ("exact", "nonfinal", "one"),
    ("exact", "none", "many"),
    ("normalized", "final", "one"),
    ("normalized", "nonfinal", "one"),
    ("normalized", "none", "many"),
)


def advertised_classes():
    keys = set()
    for style in ("LF", "CRLF"):
        for final in ("terminated", "unterminated"):
            for kind in ("replace", "delete"):
                for target in ("blank", "nonblank"):
                    for tier, pos, card in _TIER_POS_CARD:
                        if target == "blank" and tier == "normalized":
                            continue
                        if kind == "replace" and final == "unterminated" and pos == "final":
                            continue
                        # An empty unterminated suffix is not a physical line.
                        if final == "unterminated" and target == "blank" and pos == "final":
                            continue
                        keys.add(_key(style, final, kind, target, tier, pos, card))
    return keys


def _generated():
    witnesses, sample, stats = corpus()
    return list(witnesses.values()), sample, stats


@pytest.mark.parametrize("name,src,body,expected", SPEC, ids=[row[0] for row in SPEC])
def test_spec_table(name, src, body, expected):
    assert _eq(evaluate(src, _patch(body)), expected)


def test_membership_negatives_are_out_of_domain():
    seen = {name for name, *_ in SPEC if name in NEGATIVES}
    assert seen == NEGATIVES
    for name, src, body, expected in SPEC:
        if name not in NEGATIVES:
            continue
        assert expected is OOD
        assert evaluate(src, _patch(body)) is OOD
        if name in STRUCTURAL_NEGATIVES:
            assert recognize(src, _patch(body)) is None
        else:
            assert recognize(src, _patch(body)) is not None


def test_decoder_never_invents_empty_unterminated_suffix():
    assert decode_lines(b"a\n") == [(b"a", b"\n")]
    assert decode_lines(b"a\r\n") == [(b"a", b"\r\n")]
    assert decode_lines(b"a\n\n") == [(b"a", b"\n"), (b"", b"\n")]
    assert decode_lines(b"") is None
    witnesses, sample, _stats = _generated()
    for file, _patch_bytes, _key in witnesses + sample:
        lines = decode_lines(file)
        assert lines and 1 <= len(lines) <= 4
        assert all(content or term for content, term in lines)
        assert not (lines[-1] == (b"", b""))


def test_generator_covers_feasible_classes_without_asking_the_oracle_first():
    assert _MUTATION is None
    witnesses, sample, stats = _generated()
    assert stats["seed"] == SEED
    assert stats["enumerated_parameter_sets"] > len(witnesses)
    keys = {item[2] for item in witnesses}
    assert keys == advertised_classes()
    assert all(4 not in key[-1] for key in keys)
    rule4 = 0
    counts = {key: 0 for key in keys}
    for file, patch, key in witnesses + sample:
        assert evaluate(file, patch) is not OOD
        assert tags_from_bytes(file, patch) == key
        assert recognize(file, patch) is not None
        counts[key] = counts.get(key, 0) + 1
        rule4 += 4 in key[-1]
    assert rule4 == 0
    assert all(counts[key] >= 1 for key in advertised_classes())
    by_rule = {n: sum(1 for key in counts if n in key[-1]) for n in (1, 2, 3, 4, 5, 6)}
    assert by_rule[4] == 0
    assert by_rule[1] == by_rule[2] == by_rule[3] > 0
    assert by_rule[5] > 0 and by_rule[6] > 0


def test_mutation_sanity_breaks_spec_table():
    assert _MUTATION is None
    for _name, src, body, expected in SPEC:
        assert _eq(evaluate(src, _patch(body)), expected)
    for name in (
        "tier-order", "first-match", "keep-blank-on-delete",
        "ignore-unterminated", "allow-substring",
    ):
        set_mutation(name)
        try:
            assert any(
                not _eq(evaluate(src, _patch(body)), expected)
                for _n, src, body, expected in SPEC
            ), name
        finally:
            set_mutation(None)
    assert _MUTATION is None


@pytest.mark.parametrize(
    "error,code",
    [
        (_PREFIX + "hunk 1 (no hint) not found — Could not find a match for old_string in the file", "NOT_FOUND"),
        (_PREFIX + "hunk 12 (no hint) not found — Could not find a match for old_string in the file", "NOT_FOUND"),
        (_PREFIX + "hunk 1 (no hint) is ambiguous (2 matches)", "AMBIGUOUS"),
        (_PREFIX + "ambiguous (02 matches)", "AMBIGUOUS"),
        (_PREFIX + "Found 2 matches for old_string", "AMBIGUOUS"),
        (_PREFIX + "hunk 1 (no hint) not found — Found 3 matches for old_string", "AMBIGUOUS"),
        (_PREFIX + "hunk 1 (no hint) has empty search text — include surrounding context lines", "UNMAPPED"),
        (_PREFIX + "hunk 1 (no later ordering anchor) is ambiguous — include unique context lines", "UNMAPPED"),
        (_PREFIX + "hunk 1 (no hint) not found — Found 2 matches for old_string\n  L1: x", "UNMAPPED"),
        ("Could not apply hunk", "UNMAPPED"),
        (_PREFIX + "hunk 1 (no hint) not found — similar\nSimilar sections near lines 1", "UNMAPPED"),
        (_PREFIX + "ambiguous (1 matches)", "UNMAPPED"),
        (None, "UNMAPPED"),
    ],
)
def test_diagnostic_mapper(error, code):
    assert classify_error(error) == code


def _unified(old: bytes, new: bytes) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        old.decode("ascii").splitlines(keepends=True),
        new.decode("ascii").splitlines(keepends=True),
        fromfile="a/f", tofile="b/f",
    ))


def test_site_comparison_unavailable_is_not_a_pass():
    old = b"a\nold\ntail\n"
    assert site_from_diff(_unified(old, b"a\nnew\ntail\n"), old) == (1, 2)
    assert site_from_diff(_unified(b"a\r\nold\r\n", b"a\r\nnew\r\n"), b"a\r\nold\r\n") == (1, 2)
    assert site_from_diff(_unified(b"a\nold", b"a\n"), b"a\nold") is None
    assert site_from_diff(_unified(b"old\nold\n", b"new\nold\n"), b"old\nold\n") is None
    assert site_from_diff(_unified(b"a\nb\n", b"x\ny\n"), b"a\nb\n") is None
    assert site_from_diff("", old) is None
    assert site_from_diff(_unified(old, old), old) is None


def _production_code(success, writes, error):
    """Never classify partial or wrong-file writes as an ordinary outcome."""
    if success and len(writes) == 1 and writes[0][0] == "f":
        return "WRITE"
    if not success and not writes:
        return classify_error(error)
    return "UNMAPPED"


def test_production_code_requires_exact_write_count_and_path():
    error = _PREFIX + "hunk 1 (no hint) not found — Could not find a match for old_string in the file"
    assert _production_code(True, [("f", "x")], None) == "WRITE"
    assert _production_code(False, [], error) == "NOT_FOUND"
    assert _production_code(True, [], None) == "UNMAPPED"
    assert _production_code(True, [("g", "x")], None) == "UNMAPPED"
    assert _production_code(True, [("f", "x"), ("f", "y")], None) == "UNMAPPED"
    assert _production_code(False, [("f", "x")], error) == "UNMAPPED"


def _apply_prod(file: bytes, patch: bytes):
    from tools.file_operations_common import ReadResult, WriteResult
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    class Mem:
        def __init__(self, text):
            self.files = {"f": text}
            self.writes = []

        def read_file_raw(self, path):
            if path not in self.files:
                return ReadResult(error="missing", not_found=True)
            text = self.files[path]
            return ReadResult(content=text, total_lines=text.count("\n") + 1, file_size=len(text.encode()))

        def write_file(self, path, content, pre_content=None):
            self.writes.append((path, content))
            self.files[path] = content
            return WriteResult(bytes_written=len(content.encode()), verified=True)

    store = Mem(file.decode("ascii"))
    ops, err = parse_v4a_patch(patch.decode("ascii"))
    if err or not ops:
        return "UNMAPPED", file, ""
    result = apply_v4a_operations(ops, store)
    out = store.files["f"].encode()
    return _production_code(result.success, store.writes, result.error), out, result.diff or ""


def _compare(src, patch, expected):
    if expected is OOD:
        return "out-of-domain", None, src, None
    code, out, diff = _apply_prod(src, patch)
    if isinstance(expected, Write):
        if code == "WRITE" and out == expected.new_text:
            site = site_from_diff(diff, src)
            if site is None:
                return "site-unavailable", code, out, None
            if site != expected.site:
                return "finding", code, out, site
            return "agree-site", code, out, site
        return "finding", code, out, None
    if code == expected.code and out == src:
        return "agree-refuse", code, out, None
    return "finding", code, out, None


def test_comparison_never_accepts_wrong_output_or_refusal(monkeypatch):
    expected = Write((0, 1), b"new\n")
    monkeypatch.setattr(__import__(__name__, fromlist=["_apply_prod"]), "_apply_prod",
                        lambda *_: ("WRITE", b"wrong\n", ""))
    assert _compare(b"old\n", b"patch", expected)[0] == "finding"
    module = __import__(__name__, fromlist=["_apply_prod"])
    monkeypatch.setattr(module, "_apply_prod", lambda *_: ("WRITE", b"new\n", ""))
    assert _compare(b"old\n", b"patch", expected)[0] == "site-unavailable"
    monkeypatch.setattr(module, "_apply_prod", lambda *_: pytest.fail("OOD called planner"))
    assert _compare(b"old\n", b"patch", OOD)[0] == "out-of-domain"
    monkeypatch.setattr(module, "_apply_prod", lambda *_: ("WRITE", b"new\n", _unified(b"old\n", b"new\n")))
    assert _compare(b"old\n", b"patch", Write((1, 2), b"new\n"))[0] == "finding"


# Closed-domain differential fuzz. Expected values come from evaluate(), never
# from the planner. site-unavailable is SITE_UNAVAILABLE, not agreement.
# Grammar-pin classes fail when the known defect is observed. A different
# disagreement fails too. Similar-section samples are not a class pin:
# count may be zero, so only an observed sample fails that class.
_EMPTY_SEARCH = re.compile(
    r"Patch validation failed \(no files were modified\):\n"
    r"  • f: hunk 1 \(no hint\) has empty search text — "
    r"include surrounding context lines\Z"
)
_FOUND_EXTRA = re.compile(
    r"Patch validation failed \(no files were modified\):\n"
    r"  • f: hunk 1 \(no hint\) not found — Found (?P<n>[0-9]+) matches for "
    r"old_string\. Add a unique @@ hint @@ to this hunk or include unique "
    r"context lines in its search text\. Matches:\n"
    r"(?P<lines>(?:  L[0-9]+: \[source omitted\]\n)*"
    r"  L[0-9]+: \[source omitted\])\Z"
)
_SIMILAR = re.compile(
    r"Patch validation failed \(no files were modified\):\n"
    r"  • f: hunk 1 \(no hint\) not found — "
    r"Could not find a match for old_string in the file\n"
    r"\nSimilar sections near lines [0-9]+(?:, [0-9]+)*; "
    r"read the file for context\.\Z"
)
_PIN_REASON = {
    "strict-site": (
        "bytes match but site comparison is unavailable, which is not agreement"
    ),
    "strict-blank-delete": (
        "unique nonblank deletion writes a blank terminated line "
        "instead of removing the pair"
    ),
    "strict-empty-search": (
        "empty search text is UNMAPPED and leaves the file unchanged"
    ),
    "strict-found-extra": (
        "normalized ambiguity reports Found N matches plus match lines, "
        "so the mapper returns UNMAPPED"
    ),
}
_SIMILAR_REASON = (
    "known V4A defect similar-section diagnostic unmapped, fixed in t_6e12e276"
)
_DEFAULT_FUZZ = (2000, 2000)
# The old ceiling 59 assumed broken deletion; a correctly deleted repeated-
# context sample yields a multi-removal diff and is unavailable as well.
_SITE_UNAVAILABLE_CEILING = 60
_COLLECTION_XFAIL = frozenset({
    "strict-empty-search",
    "strict-found-extra",
})
_FUZZ_BUNDLES = {}
_ROW_CACHE = {}


class _KnownPlannerDefect(AssertionError):
    """Raised only for the specific currently documented planner defect."""


def _fuzz_settings():
    seed = int(os.environ.get("HERMES_PATCH_ORACLE_SEED", "2000"))
    count = int(os.environ.get("HERMES_PATCH_ORACLE_COUNT", "2000"))
    if seed < 0 or count < 0:
        raise ValueError("fuzz seed and count must be >= 0")
    return seed, count


def _fuzz_bundle():
    settings = _fuzz_settings()
    bundle = _FUZZ_BUNDLES.get(settings)
    if bundle is None:
        bundle = enumerate_corpus(*settings)
        _FUZZ_BUNDLES[settings] = bundle
    return bundle


def _class_id(key) -> str:
    style, final, kind, target, tier, pos, card, _rules = key
    return f"{style}-{final}-{kind}-{target}-{tier}-{pos}-{card}"


def _sorted_classes():
    return tuple(sorted(advertised_classes(), key=_class_id))


def _pin_kind(key) -> str:
    """Exact class pin. Not a delete-wide or not-found-wide group."""
    _style, final, kind, target, tier, pos, card, _rules = key
    if (
        kind == "delete" and target == "nonblank" and card == "one"
        and final == "unterminated" and pos == "final"
    ):
        return "strict-site"
    if (
        kind == "delete" and target == "nonblank" and card == "one"
        and tier in ("exact", "normalized")
    ):
        return "strict-blank-delete"
    if target == "blank":
        return "strict-empty-search"
    if target == "nonblank" and tier == "normalized" and card == "many":
        return "strict-found-extra"
    if target == "nonblank" and tier == "none" and card == "zero":
        return "mixed-not-found"
    return "must-agree"


def _blank_kept(src: bytes, index: int):
    lines = decode_lines(src)
    style = file_style(lines) if lines else None
    if not lines or style is None or not 0 <= index < len(lines):
        return None
    return b"".join(
        style if j == index else content + term
        for j, (content, term) in enumerate(lines)
    )


def _signature(src, expected, status, code, out) -> str:
    if status in ("agree-site", "agree-refuse"):
        return "agree"
    if (
        status == "site-unavailable"
        and isinstance(expected, Write)
        and out == expected.new_text
    ):
        return "site-unavailable-bytes-ok"
    if status == "finding" and code == "UNMAPPED" and out == src:
        return "unmapped-unchanged"
    if (
        status == "finding"
        and code == "WRITE"
        and isinstance(expected, Write)
        and out != expected.new_text
        and out == _blank_kept(src, expected.site[0])
    ):
        return "delete-keeps-blank-line"
    return "other"


def _found_extra(error: str) -> bool:
    match = _FOUND_EXTRA.fullmatch(error or "")
    if not match:
        return False
    count = int(match.group("n"))
    lines = [line for line in match.group("lines").splitlines() if line.startswith("  L")]
    return count >= 2 and len(lines) == count


def _planner_error(file: bytes, patch: bytes) -> str:
    from tools.file_operations_common import ReadResult, WriteResult
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    class Mem:
        def __init__(self, text):
            self.files = {"f": text}

        def read_file_raw(self, path):
            text = self.files[path]
            return ReadResult(
                content=text,
                total_lines=text.count("\n") + 1,
                file_size=len(text.encode()),
            )

        def write_file(self, path, content, pre_content=None):
            self.files[path] = content
            return WriteResult(bytes_written=len(content.encode()), verified=True)

    store = Mem(file.decode("ascii"))
    ops, err = parse_v4a_patch(patch.decode("ascii"))
    if err or not ops:
        return err or ""
    return apply_v4a_operations(ops, store).error or ""


@dataclass(frozen=True)
class _Row:
    origin: str
    sig: str
    status: str
    code: str
    out: bytes
    error: str
    expected: object
    src: bytes


def _class_rows(key):
    cache_key = (_fuzz_settings(), key)
    cached = _ROW_CACHE.get(cache_key)
    if cached is not None:
        return cached
    witnesses, sample, _stats = _fuzz_bundle()
    cases = [("witness", witnesses[key])]
    cases.extend(("sample", item) for item in sample if item[2] == key)
    rows = []
    for origin, (src, patch, tagged) in cases:
        expected = evaluate(src, patch)
        if expected is OOD or tags_from_bytes(src, patch) != tagged or tagged != key:
            raise AssertionError(f"{_class_id(key)} left the closed domain")
        status, code, out, _site = _compare(src, patch, expected)
        if not isinstance(code, str) or not isinstance(out, bytes):
            raise AssertionError(f"{_class_id(key)} comparison returned {status!r}")
        sig = _signature(src, expected, status, code, out)
        error = _planner_error(src, patch) if sig == "unmapped-unchanged" else ""
        rows.append(_Row(origin, sig, status, code, out, error, expected, src))
    _ROW_CACHE[cache_key] = tuple(rows)
    return _ROW_CACHE[cache_key]


def _fmt(row: _Row) -> str:
    return f"{row.origin} sig={row.sig} status={row.status} code={row.code}"


def _matches_pin(kind: str, row: _Row) -> bool:
    if kind == "strict-site":
        return row.sig == "site-unavailable-bytes-ok"
    if kind == "strict-blank-delete":
        return row.sig == "delete-keeps-blank-line"
    if kind == "strict-empty-search":
        return row.sig == "unmapped-unchanged" and bool(_EMPTY_SEARCH.fullmatch(row.error))
    if kind == "strict-found-extra":
        return row.sig == "unmapped-unchanged" and _found_extra(row.error)
    return False


def _is_similar_sample(row: _Row) -> bool:
    """Known generated-sample shape only. A witness with this text is not it."""
    return (
        row.origin == "sample"
        and row.sig == "unmapped-unchanged"
        and isinstance(row.expected, Refuse)
        and row.expected.code == "NOT_FOUND"
        and bool(_SIMILAR.fullmatch(row.error))
    )


def _outcome_bucket(kind: str, row: _Row) -> str:
    """PASS, XFAIL, SITE_UNAVAILABLE, or FAILURE. Unavailable is none of the others."""
    if row.sig == "site-unavailable-bytes-ok":
        return "SITE_UNAVAILABLE"
    if row.sig == "agree":
        return "PASS"
    if kind == "mixed-not-found" and _is_similar_sample(row):
        return "XFAIL"
    if kind in _COLLECTION_XFAIL and _matches_pin(kind, row):
        return "XFAIL"
    return "FAILURE"


def _enforce_site_unavailable_ceiling(seed: int, count: int, observed: int) -> None:
    if (seed, count) != _DEFAULT_FUZZ or observed <= _SITE_UNAVAILABLE_CEILING:
        return
    pytest.fail(
        f"SITE_UNAVAILABLE default count {observed} exceeds ceiling "
        f"{_SITE_UNAVAILABLE_CEILING}"
    )


def _finish_strict(rows, key, kind) -> None:
    seen = len(rows)
    rows = [row for row in rows if row.sig != "site-unavailable-bytes-ok"]
    if seen and not rows:
        pytest.fail(f"{_class_id(key)}: every row is SITE_UNAVAILABLE; nothing verified")
    if rows and all(row.sig == "agree" or _matches_pin(kind, row) for row in rows) and any(
        _matches_pin(kind, row) for row in rows
    ):
        raise _KnownPlannerDefect(_PIN_REASON[kind])
    if all(row.sig == "agree" for row in rows):
        return
    unrelated = "; ".join(
        _fmt(row) for row in rows if row.sig != "agree" and not _matches_pin(kind, row)
    )
    pytest.fail(f"unrelated disagreement in {_class_id(key)}: {unrelated}")


def _finish_mixed(rows, key) -> None:
    witness = [row for row in rows if row.origin == "witness"]
    assert len(witness) == 1
    assert witness[0].sig == "agree", (
        f"{_class_id(key)} witness must agree; refusing to xfail the class"
    )
    similar = 0
    bad = []
    for row in rows:
        if row.sig == "agree":
            continue
        if _is_similar_sample(row):
            similar += 1
            continue
        bad.append(row)
    if bad:
        pytest.fail(
            f"unrelated disagreement in {_class_id(key)}: "
            + "; ".join(_fmt(row) for row in bad)
        )
    # Do not return success while known-shape samples were dropped.
    if similar:
        raise _KnownPlannerDefect(f"{_SIMILAR_REASON} ({similar} generated samples)")


def test_shrink_replay_is_not_proof_of_prior_application():
    source = b"a\n"
    patch = _patch(b"-a b\n+a\n")
    expected = evaluate(source, patch)
    assert expected == Refuse("NOT_FOUND")
    status, code, out, _site = _compare(source, patch, expected)
    assert (status, code, out) == ("agree-refuse", "NOT_FOUND", source)


def test_planner_smoke_r1_lf():
    """Real planner, not the oracle: the spec's LF replacement must agree."""
    _name, src, body, expected = next(row for row in SPEC if row[0] == "r1-lf")
    patch = _patch(body)
    assert evaluate(src, patch) == expected
    status, code, out, site = _compare(src, patch, expected)
    assert (status, code, out, site) == (
        "agree-site", "WRITE", expected.new_text, expected.site,
    )
    print(f"oracle-fuzz smoke r1-lf {status} {code} site={site}")


def test_fuzz_defaults_are_2000(monkeypatch):
    monkeypatch.delenv("HERMES_PATCH_ORACLE_SEED", raising=False)
    monkeypatch.delenv("HERMES_PATCH_ORACLE_COUNT", raising=False)
    assert _fuzz_settings() == (2000, 2000)


def test_fuzz_seed_and_count_follow_the_environment(monkeypatch):
    monkeypatch.setenv("HERMES_PATCH_ORACLE_SEED", "7")
    monkeypatch.setenv("HERMES_PATCH_ORACLE_COUNT", "3")
    witnesses, sample, stats = _fuzz_bundle()
    assert stats["seed"] == 7
    assert len(witnesses) == 80
    assert len(sample) == 3
    assert all(evaluate(src, patch) is not OOD for src, patch, _key in sample)


def test_pin_partition_is_exact():
    groups = Counter(_pin_kind(key) for key in advertised_classes())
    assert dict(groups) == {
        "strict-site": 4,
        "strict-blank-delete": 12,
        "strict-empty-search": 28,
        "strict-found-extra": 8,
        "mixed-not-found": 8,
        "must-agree": 20,
    }
    assert _pin_kind(_key(
        "LF", "terminated", "delete", "nonblank", "exact", "none", "many",
    )) == "must-agree"
    assert _pin_kind(_key(
        "LF", "terminated", "delete", "nonblank", "none", "none", "zero",
    )) == "mixed-not-found"
    assert _pin_kind(_key(
        "LF", "unterminated", "delete", "nonblank", "exact", "final", "one",
    )) == "strict-site"
    assert _pin_kind(_key(
        "LF", "terminated", "replace", "nonblank", "exact", "nonfinal", "one",
    )) == "must-agree"


def test_site_unavailable_is_not_agreement():
    expected = Write((1, 2), b"a\n")
    assert _signature(b"a\nold", expected, "site-unavailable", "WRITE", b"a\n") == (
        "site-unavailable-bytes-ok"
    )
    assert _signature(b"a\nold", expected, "agree-site", "WRITE", b"a\n") == "agree"
    row = _Row(
        "witness", "site-unavailable-bytes-ok", "site-unavailable", "WRITE",
        b"a\n", "", expected, b"a\nold",
    )
    assert _outcome_bucket("strict-site", row) == "SITE_UNAVAILABLE"
    assert _outcome_bucket("must-agree", row) == "SITE_UNAVAILABLE"
    assert _outcome_bucket("strict-blank-delete", row) == "SITE_UNAVAILABLE"


def _similar_error(lines="1") -> str:
    return (
        "Patch validation failed (no files were modified):\n"
        "  • f: hunk 1 (no hint) not found — "
        "Could not find a match for old_string in the file\n"
        f"\nSimilar sections near lines {lines}; read the file for context."
    )


def _mixed_key():
    return next(key for key in _sorted_classes() if _pin_kind(key) == "mixed-not-found")


def test_byte_identical_replacement_is_site_unavailable_not_pass():
    """Search and replacement differ; output bytes do not. Not a pass."""
    src = b"a\tb\n"
    patch = _patch(b"-a b\n+a\tb\n")
    expected = evaluate(src, patch)
    assert expected == Write((0, 1), src)
    status, code, out, site = _compare(src, patch, expected)
    assert status == "site-unavailable"
    assert code == "WRITE"
    assert out == src
    assert site is None
    row = _Row(
        "sample",
        _signature(src, expected, status, code, out),
        status,
        code,
        out,
        "",
        expected,
        src,
    )
    assert _outcome_bucket("must-agree", row) == "SITE_UNAVAILABLE"


def test_final_unterminated_deletion_is_site_unavailable_not_pass():
    """Deleting an unterminated final line removes the pair and cannot be framed."""
    src = b"a\nold"
    patch = _patch(b"-old\n")
    expected = evaluate(src, patch)
    assert expected == Write((1, 2), b"a\n")
    status, code, out, site = _compare(src, patch, expected)
    assert status == "site-unavailable"
    assert code == "WRITE"
    assert out == b"a\n"
    assert site is None
    row = _Row(
        "witness",
        _signature(src, expected, status, code, out),
        status,
        code,
        out,
        "",
        expected,
        src,
    )
    assert _outcome_bucket("strict-site", row) == "SITE_UNAVAILABLE"


def test_class_parameters_have_no_xfail_markers():
    """Grammar pins are real failures, not collection xfail marks."""
    for key in _sorted_classes():
        param = _class_parameter(key)
        xfails = [mark for mark in param.marks if mark.name == "xfail"]
        assert not xfails, _class_id(key)


def test_mixed_absence_is_not_an_expected_failure():
    key = _mixed_key()
    agree = _Row("witness", "agree", "agree-refuse", "NOT_FOUND", b"a\n", "", Refuse("NOT_FOUND"), b"a\n")
    sample = _Row("sample", "agree", "agree-refuse", "NOT_FOUND", b"b\n", "", Refuse("NOT_FOUND"), b"b\n")
    _finish_mixed((agree, sample), key)


def test_mixed_similar_sample_is_a_known_defect_not_a_pass():
    key = _mixed_key()
    agree = _Row("witness", "agree", "agree-refuse", "NOT_FOUND", b"a\n", "", Refuse("NOT_FOUND"), b"a\n")
    similar = _Row(
        "sample", "unmapped-unchanged", "finding", "UNMAPPED", b"a\n", _similar_error("1, 2"),
        Refuse("NOT_FOUND"), b"a\n",
    )
    assert _is_similar_sample(similar)
    assert _outcome_bucket("mixed-not-found", similar) == "XFAIL"
    with pytest.raises(_KnownPlannerDefect, match="1 generated samples") as caught:
        _finish_mixed((agree, similar), key)
    assert "Similar sections" not in str(caught.value)
    assert "@@" not in str(caught.value)


def test_similar_shape_on_a_witness_is_not_xfailed():
    key = _mixed_key()
    witness = _Row(
        "witness", "unmapped-unchanged", "finding", "UNMAPPED", b"a\n", _similar_error(),
        Refuse("NOT_FOUND"), b"a\n",
    )
    assert not _is_similar_sample(witness)
    assert _outcome_bucket("mixed-not-found", witness) == "FAILURE"
    with pytest.raises(AssertionError, match="witness must agree") as caught:
        _finish_mixed((witness,), key)
    assert type(caught.value) is AssertionError


def test_mixed_unrelated_is_not_swallowed():
    key = _mixed_key()
    agree = _Row("witness", "agree", "agree-refuse", "NOT_FOUND", b"a\n", "", Refuse("NOT_FOUND"), b"a\n")
    bad = _Row(
        "sample", "other", "finding", "WRITE", b"nope", "@@ hint @@ leaked source",
        Refuse("NOT_FOUND"), b"a\n",
    )
    with pytest.raises(pytest.fail.Exception, match="unrelated disagreement") as caught:
        _finish_mixed((agree, bad), key)
    assert "@@" not in str(caught.value)
    assert "leaked" not in str(caught.value)
    assert type(caught.value) is not _KnownPlannerDefect


def test_site_unavailable_ceiling_applies_only_to_the_default_fuzz():
    _enforce_site_unavailable_ceiling(7, 20000, 400)
    _enforce_site_unavailable_ceiling(*_DEFAULT_FUZZ, _SITE_UNAVAILABLE_CEILING)
    _enforce_site_unavailable_ceiling(2000, 0, 100)
    with pytest.raises(pytest.fail.Exception, match="SITE_UNAVAILABLE"):
        _enforce_site_unavailable_ceiling(*_DEFAULT_FUZZ, _SITE_UNAVAILABLE_CEILING + 1)


def test_similar_section_case_has_no_xfail_marker():
    marks = getattr(
        test_not_found_with_similar_section_has_mapped_refusal, "pytestmark", []
    )
    if not isinstance(marks, (list, tuple)):
        marks = [marks]
    assert not any(getattr(mark, "name", None) == "xfail" for mark in marks)


def test_known_defect_pin_never_swallows_an_unrelated_mismatch():
    key = next(key for key in _sorted_classes() if _pin_kind(key) == "strict-blank-delete")
    row = _Row("sample", "other", "finding", "WRITE", b"wrong", "", Write((0, 1), b""), b"a\n")
    with pytest.raises(pytest.fail.Exception, match="unrelated disagreement"):
        _finish_strict((row,), key, "strict-blank-delete")


def test_fuzz_seed_and_counts():
    seed, count = _fuzz_settings()
    witnesses, sample, stats = _fuzz_bundle()
    items = [*witnesses.values(), *sample]
    assert stats["seed"] == seed
    assert len(witnesses) == 80
    assert len(sample) == min(count, stats["enumerated_parameter_sets"])
    counts = Counter()
    for src, patch, key in items:
        src.decode("ascii")
        patch.decode("ascii")
        assert evaluate(src, patch) is not OOD
        assert tags_from_bytes(src, patch) == key
        counts[key] += 1
    assert set(counts) == advertised_classes()
    assert all(counts[key] >= 1 for key in advertised_classes())
    rule_cases = {
        n: sum(v for key, v in counts.items() if n in key[-1])
        for n in (1, 2, 3, 4, 5, 6)
    }
    assert rule_cases[4] == 0
    assert rule_cases[1] == rule_cases[2] == rule_cases[3] == len(items)
    outcomes = Counter()
    for key in _sorted_classes():
        for row in _class_rows(key):
            outcomes[(_pin_kind(key), row.sig)] += 1
    print(
        "oracle-fuzz",
        {
            "seed": seed,
            "count": count,
            "enumerated_parameter_sets": stats["enumerated_parameter_sets"],
            "witnesses": len(witnesses),
            "sample": len(sample),
            "in_domain": len(items),
            "classes": len(counts),
            "rule_cases": rule_cases,
        },
    )
    print(
        "oracle-fuzz class-counts",
        {_class_id(key): counts[key] for key in _sorted_classes()},
    )
    print("oracle-fuzz outcomes", dict(outcomes))
    buckets = Counter()
    for key in _sorted_classes():
        kind = _pin_kind(key)
        for row in _class_rows(key):
            buckets[_outcome_bucket(kind, row)] += 1
    assert buckets["FAILURE"] == 0
    similar = sum(
        _is_similar_sample(row)
        for key in _sorted_classes()
        if _pin_kind(key) == "mixed-not-found"
        for row in _class_rows(key)
    )
    assert similar <= buckets["XFAIL"]
    assert buckets["PASS"] + buckets["XFAIL"] + buckets["SITE_UNAVAILABLE"] == len(items)
    _enforce_site_unavailable_ceiling(seed, count, buckets["SITE_UNAVAILABLE"])
    print(
        "oracle-fuzz ledger",
        {
            "PASS": buckets["PASS"],
            "XFAIL": buckets["XFAIL"],
            "SITE_UNAVAILABLE": buckets["SITE_UNAVAILABLE"],
            "FAILURE": 0,
            "similar_xfail": similar,
        },
    )


def test_mixed_not_found_witnesses_agree():
    witnesses, _sample, _stats = _fuzz_bundle()
    for key in advertised_classes():
        if _pin_kind(key) != "mixed-not-found":
            continue
        src, patch, _key = witnesses[key]
        expected = evaluate(src, patch)
        status, code, out, _site = _compare(src, patch, expected)
        assert status == "agree-refuse" and code == "NOT_FOUND" and out == src, key


def test_not_found_with_similar_section_has_mapped_refusal():
    src = b"a\n"
    patch = _patch(b"-tail\n+new\n")
    expected = evaluate(src, patch)
    assert expected == Refuse("NOT_FOUND")
    status, code, out, _site = _compare(src, patch, expected)
    if (status, code, out) == ("agree-refuse", "NOT_FOUND", src):
        return
    if code == "UNMAPPED" and out == src and _SIMILAR.fullmatch(_planner_error(src, patch)):
        raise _KnownPlannerDefect("similar-section suffix is not mapped")
    pytest.fail("unrelated not-found mismatch")


def _class_parameter(key):
    return pytest.param(key, id=_class_id(key))


@pytest.mark.parametrize("key", [_class_parameter(key) for key in _sorted_classes()])
def test_closed_domain_differential(key):
    rows = _class_rows(key)
    assert rows
    assert all(row.expected is not OOD for row in rows)
    kind = _pin_kind(key)
    failures = [row for row in rows if _outcome_bucket(kind, row) == "FAILURE"]
    if kind in ("must-agree", "strict-site"):
        # Byte-identical replacement and unterminated deletion have no framed
        # site. That bucket is not agreement and does not fail the class.
        assert not failures, (
            f"{_class_id(key)} unexpected disagreement; "
            "SITE_UNAVAILABLE is not a pass, xfail, or failure: "
            + "; ".join(_fmt(row) for row in failures)
        )
        return
    if kind == "mixed-not-found":
        _finish_mixed(rows, key)
        return
    _finish_strict(rows, key, kind)
