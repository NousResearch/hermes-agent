# Unhinted V4A reference oracle: initial domain

Status: proposed specification, not a claim about current planner behavior. Base:
`74c994fdc32efc666475413bacd58756e6d9f885`. This spec deliberately excludes
features rather than making an unproven inference. It must not derive expected
answers by calling the production planner. An implementation belongs in tests,
is independent, and must be at most 400 lines.

## Outcome and provenance

Exactly one outcome: `WRITE(site=(start,end), new_text=bytes)` where `site` is
zero-based half-open **line indices**, not bytes or character offsets, in the
original file; `new_text` is the **entire
resulting file**, including exact terminators; `REFUSE(code)`; or
`OUT_OF_DOMAIN`. Refusal leaves all original files unchanged. Codes in this
version: `NOT_FOUND` and `AMBIGUOUS`; these are stable categories, not source
text or production diagnostic strings. Unsupported operations, syntax,
features, encodings, or situations are `OUT_OF_DOMAIN`, never `REFUSE`.
For baseline diagnostics, use case-sensitive substring matching on the entire
diagnostic: if `empty search text` appears, classify as `UNMAPPED`. Otherwise,
a substring matching numeric `ambiguous (N matches)` or
`Found N matches for old_string` with N >= 2 maps to `AMBIGUOUS` before any
`not found` label. Otherwise, map to `NOT_FOUND` only if the complete diagnostic
is `Could not find a match for old_string in the file`, or consists of a
`hunk N (no hint) not found` label, where N is decimal, followed by ` — ` and
that exact message. Do not match arbitrary `not found` substrings;
every other diagnostic is `UNMAPPED`. UNMAPPED is a finding, not either refusal
code. Do not infer a count from the input to reinterpret a production error.

For a successful whole-line replacement or deletion, `site` describes the
consumed original lines (including their individual terminators). One input line
is a `(content, terminator)` pair; terminator is LF, CRLF, or absent only on
the final line. Decode a file canonically: each terminator closes exactly one
physical line; only a nonempty suffix after the final terminator creates an
unterminated line. In particular `b'a\n'` has one line, not a second empty
unterminated line. The grammar rejects empty files and an empty unterminated
suffix. Untouched pairs are copied byte-for-byte. Deleted pairs vanish
including their terminator; each added replacement line has the file style's
terminator. For now, replacing an unterminated final line is OUT_OF_DOMAIN:
the assignment requires a separate unterminated class and file-style
replacement terminators, but the upstream contract does not settle whether a
replacement must add the missing terminator. This restricted grammar retains
unterminated-file cases whose selected line is terminated, and deletion of an
unterminated final line. Expand only after that EOF policy is resolved.

## Generator-owned closed grammar

Generate a tagged case, then independently check membership, not vice versa.
All strings below use Python bytes-literal escaping: `\t` denotes byte 0x09,
not backslash followed by `t`. Files contain 1–4 physical lines, each
content chosen from `{a,b,c,x,y,old,new,keep,tail,"", "a b", "a\tb"}`. File style is `LF` or
`CRLF`, uniformly across present terminators; final termination is either
`terminated` or `unterminated` (at least one terminator must be present to
establish the style in the latter class). The patch is exactly one existing
file update to path `f`, exactly one bare `@@` hunk, followed by either one
`-<content>` and one `+<content>`, or one `-<content>` and no addition. Context
is absent. Patch syntax itself uses LF. Search and replacement contents come
from the same fixed alphabet and replacement differs from search. A replacement
that produces byte-identical output is allowed when its content differs from
the search (for example normalized matching of `a b` to `a\tb`). No line
contains a newline byte. The three search cardinalities are 0, 1, or 2+
physical winning-tier whole-line candidates. Duplicate blank lines are generated as
ordinary candidates. Matching first selects byte-exact lines; only if there are
none, select lines using the one fixed normalization pair `a b` ≡ `a\tb`.
For replace cases, if any winning-tier candidate is unterminated, the case is
OUT_OF_DOMAIN, even when the tier is ambiguous. If no candidate wins, or all
winning candidates are terminated, this restriction does not exclude it.
There are no other normalization equivalences in this language. The generator
produces no substring candidate.
That is essential: absence of a whole-line match is not sufficient to prove a
production substring cannot match unless the generator checks it too. Reject a
file from this grammar if any non-equal content contains the search as a
substring; for empty search, nonempty contents are not substring candidates.

Each case has a deterministic `rules` set: `{1,2,3}` always, plus `5` iff
the winning cardinality is zero or many, plus `6` iff kind is delete.
Further tags are `style=LF|CRLF`, `final=terminated|unterminated`,
`kind=replace|delete`, `target=blank|nonblank` (the search content),
`tier=exact|normalized|none`, `position=final|nonfinal|none`, and
`cardinality=zero|one|many`. For a unique site, `position=final` iff its
line index is the last physical input line, else `nonfinal`; otherwise it is
`none`. `tier=none` iff cardinality is zero. The feasibility list consists of all
combinations realizable under the grammar, determined by exhaustive enumeration
of the bounded grammar before random sampling, and persisted with seed/counts. These tags are generator metadata, not advice to
the oracle. Generator assertions validate the grammar and tags from actual
bytes; every generated case must be independently accepted by the oracle and
must not return `OUT_OF_DOMAIN`. Count cases per rule and tag combination;
require nonzero count for every advertised feasible in-domain class and publish
the seed and counts. Seed each feasible combination with at least one enumerated
case before random sampling; missing any feasible class fails the run. Rule 4 has no generated in-domain class in this version;
record its count as zero rather than fabricating coverage. Compare production outcome and exact resulting file bytes;
compare site from a unified diff only as a consistency check derived from output
bytes, not independent evidence of the planner's internal site. Require exactly
one framed removed line; initialize a zero-based original cursor from the
`-start` of the unified-diff hunk containing that line minus one;
for every removed line record its cursor *before* advancing; advance the
cursor after context and removed lines, not additions. The removed line's
recorded cursor gives `(cursor,cursor+1)` only if the removed payload, including
its terminator, equals the complete original line pair at that cursor, and that
pair's content occurs at exactly one original line index. An unterminated
removed line cannot be framed reliably because the diff has no missing-newline
marker; report site comparison unavailable. If the diff has multiple removals,
omits the change, fails either check, or the content repeats, report
site comparison unavailable,
not a pass. A production
difference is a finding, never a
reason to discard a generated input or alter its expected value. Validate oracle
membership using separately constructed negative cases too.

For this grammar, an independently implemented recognizer first checks all
structural constraints in this entire grammar, including the exact wrapper in
the notation below, and returns `OUT_OF_DOMAIN` if any fail, before
performing search. For example `b'a b\n'` with `-a/+new` is OOD due to a
substring decoy; `b'old\n'` with `-old/+old` is OOD because replacement
content equals search content, not merely because resulting bytes are unchanged.
A blank `+` is replacement by a terminated blank line, distinct from no `+`,
which deletes the line. For example `b'a\nold\nb\n'` with `-old/+` yields
`WRITE((1,2), b'a\n\nb\n')`; deletion instead yields `b'a\nb\n'`.

For this grammar, enumerate all exact whole-line candidates globally; if none,
enumerate candidates under the fixed normalization pair. No
candidates gives `REFUSE(NOT_FOUND)`; multiple gives `REFUSE(AMBIGUOUS)`;
exactly one gives `WRITE` using the splice above. There is no cursor or hint.
A present `+` line, including a blank one, creates exactly one terminated
line. Replacements with content equal to search are excluded. The initial grammar does not cover Unicode, even
though a later expansion may introduce a fixed normalization list after its
semantics are proven. Blank search is allowed and uses identical cardinality.

## Rules 1–6 and hand-computed cases

Notation: `F=b'...'` is the input file; `P` is hunk body (bare `@@`, `-`,
`+`; omitted full wrapper is `*** Begin Patch\n*** Update File: f\n@@\n` + P +
`*** End Patch\n`). Escapes are bytes, not visual line breaks. `W((i,j),b'...')`
is WRITE; `R(code)` is REFUSE and leaves F unchanged; `OOD` is OUT_OF_DOMAIN.
Each example is independently runnable against the real planner after adding
the wrapper. Examples tagged OOD test membership, not planner correctness.

### 1. Consumed provenance and terminator policy

In-domain: one hunk; exact line splice and per-line terminators as above.
Multi-hunk provenance and written-context anchors are not covered (no
independent bounded model yet). A final unterminated line is its own class.

* F=`b'a\nold\ntail\n'`; P=`b'-old\n+new\n'` → W((1,2), `b'a\nnew\ntail\n'`).
* F=`b'a\r\nold\r\ntail\r\n'`; same P → W((1,2), `b'a\r\nnew\r\ntail\r\n'`).
* F=`b'a\nold'`; same P → OOD (unterminated final replacement); oracle does not write.

### 2. Global strict tier before lower tiers

In-domain: exact whole-line comparisons precede the fixed `a b`/`a\tb`
normalization pair. Other whitespace/tab normalization, Unicode and substring
fallback are not covered; the spec does not invent their general relation.

* F=`b'old\ntail\n'`; P=`b'-old\n+new\n'` → W((0,1), `b'new\ntail\n'`).
* F=`b'old\nold\n'`; same P → R(AMBIGUOUS), original `b'old\nold\n'`.
* F=`b'a\nb\n'`; same P → R(NOT_FOUND), original `b'a\nb\n'`.
* F=`b'a\tb\n'`; P=`b'-a b\n+new\n'` → W((0,1), `b'new\n'`).
* F=`b'a\tb\na b\n'`; same P → W((1,2), `b'a\tb\nnew\n'`).
* F=`b'a\tb\na\tb\n'`; same P → R(AMBIGUOUS), original `b'a\tb\na\tb\n'`.

### 3. Uniqueness and hints

In-domain: require exactly one candidate globally. Hints, hint windows,
cursors and hunk ordering are not modeled; all hinted patches are OOD, not a
refusal. Their units remain an open owner question.

* F=`b'a\nold\n'`; P=`b'-old\n+new\n'` → W((1,2), `b'a\nnew\n'`).
* F=`b'old\nold\n'`; same P → R(AMBIGUOUS), original `b'old\nold\n'`.
* F=`b'old\n'`; replace the wrapper's bare hunk marker with `@@ hint @@` and use P=`b'-old\n+new\n'` → OOD; oracle does not write.

### 4. Consumed search versus already-applied replacement

Not covered: requires at least two hunks and provenance across planned
edits. Multiple hunks are OOD even if a tempting no-op or near match exists;
none may be labeled REFUSE or WRITE by this oracle version.

* F=`b'old\n'`; P=`b'-old\n+new\n@@\n-old\n+new\n'` → OOD; original `b'old\n'`.
* F=`b'old\nnew\n'`; same P → OOD; original `b'old\nnew\n'`.
* F=`b'old\nold\n'`; same P → OOD; original `b'old\nold\n'`.

### 5. Plan before write, no leaked diagnostic content

In-domain: a single selection refusal leaves its sole input file unchanged;
reason codes contain no content. Multiple hunks/files, filesystem I/O failure,
and diagnostic text inspection are not covered by the reference model. Those
still require separate integration tests; this spec does not claim rollback on
I/O failure.

* F=`b'a\n'`; P=`b'-old\n+new\n'` → R(NOT_FOUND), original `b'a\n'`.
* F=`b'old\nold\n'`; same P → R(AMBIGUOUS), original `b'old\nold\n'`.
* F=`b'old\n'`; P=`b'-old\n+new\n@@\n-a\n+b\n'` → OOD; original `b'old\n'`.

### 6. Valid operations, pure and blank deletion

In-domain: whole-line deletion, including a blank line, deletes the consumed
line **and its terminator**, and uses ordinary uniqueness. Hint-selected partial
edits, addition-only EOF insertion, empty context anchors, written-context
hints, cursor-spanning hints, ordered multiple hunks and partial-line edits are
not covered in this initial grammar. Exclusion is not permission to regress
them in production.

* F=`b'a\nold\nb\n'`; P=`b'-old\n'` → W((1,2), `b'a\nb\n'`).
* F=`b'a\r\n\r\nb\r\n'`; P=`b'-\n'` → W((1,2), `b'a\r\nb\r\n'`).
* F=`b'a\n\nb\n\n'`; P=`b'-\n'` → R(AMBIGUOUS), original `b'a\n\nb\n\n'`.

## Observed baseline disagreements (findings, not spec revisions)

Direct execution of `parse_v4a_patch` and `apply_v4a_operations` at the base,
using a memory-backed reader/writer, produced these disagreements for the
hand-computed **in-domain** examples above. `site` was not independently
observable from the public apply result, so only outcomes and bytes were
compared. Rule 6's ordinary deletion writes `b'a\n\nb\n'`, not
`b'a\nb\n'`; its unique blank deletion refuses with `empty search text`, and
its duplicate blank deletion also refuses with `empty search text`, not the
specified ambiguous category. A separate probe used input `b'a\n'`, search
`SUPERUNIQUE`, replacement `new`; its baseline not-found diagnostic was
`Could not find a match for old_string in the file` and did not echo the
marker. These are baseline
findings, not evidence that
this proposed contract has already been implemented. The other in-domain
examples above had the expected success/refusal and resulting bytes. Out-of-
domain examples establish oracle membership only: real planner output cannot
be equated to an oracle outcome for them.

## Baseline red matrix (do not skip)

Run `tests/tools/test_patch_planner_targeting_matrix.py` on the exact baseline.
Expected-red node IDs, pinned at `74c994fd` (a red is a finding against
`docs/v4a-targeting-contract.md`, not an xfail or a skipped test):

* `test_success[exact-decoy-before]`
* `test_success[normalized-tab-decoy]`
* `test_success[normalized-space-decoy]`
* `test_refusal[repeat-consumed-prefix]`
* `test_refusal[repeat-consumed-prefix-footer]`
* `test_refusal[repeated-identical-no-hint]`
* `test_refusal[hint-cannot-bypass-tier]`

The independently examined baseline produced 7 failures and 17 passes. Pin
both the identities and counts; a changed list is investigated, not silently
accepted. Run the full matrix normally, never skip or xfail these tests. Its
coverage is broader than the deliberately narrow oracle domain and cannot be
used to claim excluded features are covered by the oracle.
