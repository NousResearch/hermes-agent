"""Bounded unhinted V4A oracle. Independent of the production planner.

Spec: tests/tools/ORACLE_SPEC.md. Expected answers are not taken from
apply_v4a_operations. Fault injection is test-only (_MUTATION).
"""
from __future__ import annotations

import itertools
import random
import re
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
    for n in range(1, 5):
        for combo in itertools.product(range(len(alpha)), repeat=n):
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
