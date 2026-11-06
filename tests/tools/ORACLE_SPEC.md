# Unhinted V4A reference oracle: initial domain

Status: proposed specification, not a claim about current planner behavior. Base:
`74c994fdc32efc666475413bacd58756e6d9f885`. This spec deliberately excludes
features rather than making an unproven inference. It must not derive expected
answers by calling the production planner. An implementation belongs in tests,
is independent, and must be at most 400 lines.

## Outcome and provenance

Exactly one outcome: `WRITE(site=(start,end), new_text=bytes)` where `site` is
zero-based half-open in the **original file**, and `new_text` is the **entire
resulting file**, including exact terminators; `REFUSE(code)`; or
`OUT_OF_DOMAIN`. Refusal leaves all original files unchanged. Codes in this
version: `NOT_FOUND` and `AMBIGUOUS`; these are stable categories, not source
text or production diagnostic strings. Unsupported operations, syntax,
features, encodings, or situations are `OUT_OF_DOMAIN`, never `REFUSE`.
The harness maps production diagnostics to the category and reports an
unmappable result as a finding; it must not require textual diagnostic equality.

For a successful whole-line replacement or deletion, `site` describes the
consumed original lines (including their individual terminators). One input line
is a `(content, terminator)` pair; terminator is LF, CRLF, or absent only on
the final line. Untouched pairs are copied byte-for-byte. Deleted pairs vanish
including their terminator; each added replacement line has the file style's
terminator, even when replacing an unterminated final line. This is the explicit
contract for this narrow domain, not an observation about the baseline.

## Generator-owned closed grammar

Generate a tagged case, then independently check membership, not vice versa.
All strings below are literal bytes; files contain 1–8 physical lines, each
content chosen from `{a,b,c,x,y,old,new,keep,tail,"", "a b", "a\tb"}`. File style is `LF` or
`CRLF`, uniformly across present terminators; final termination is either
`terminated` or `unterminated` (at least one terminator must be present to
establish the style in the latter class). The patch is exactly one existing
file update to path `f`, exactly one bare `@@` hunk, followed by either one
`-<content>` and one `+<content>`, or one `-<content>` and no addition. Context
is absent. Patch syntax itself uses LF. Search and replacement contents come
from the same fixed alphabet and replacement differs from search. No line
contains a newline byte. The three search cardinalities are 0, 1, or 2+
physical winning-tier whole-line candidates. Duplicate blank lines are generated as
ordinary candidates. Matching first selects byte-exact lines; only if there are
none, select lines using the one fixed normalization pair `a b` ≡ `a\tb`.
There are no other normalization equivalences in this language. The generator
produces no substring candidate.
That is essential: absence of a whole-line match is not sufficient to prove a
production substring cannot match unless the generator checks it too. Reject a
file from this grammar if any non-equal content contains the search as a
substring; for empty search, nonempty contents are not substring candidates.

Each case has tags `rule`, `style=LF|CRLF`, `final=terminated|unterminated`,
`kind=replace|delete`, `target=blank|nonblank`, and
`cardinality=zero|one|many`. These tags are generator metadata, not advice to
the oracle. Generator assertions validate the grammar and tags from actual
bytes; every generated case must be independently accepted by the oracle and
must not return `OUT_OF_DOMAIN`. Count cases per rule and tag combination;
require nonzero count for every advertised feasible class and publish the
seed and counts. Compare production outcome and exact resulting file bytes;
compare a reported site only if it is observed directly from planner output,
not guessed from final bytes. A production difference is a finding, never a
reason to discard a generated input or alter its expected value. Validate oracle
membership using separately constructed negative cases too.

For this grammar, enumerate all exact whole-line candidates globally; if none,
enumerate candidates under the fixed normalization pair. No
candidates gives `REFUSE(NOT_FOUND)`; multiple gives `REFUSE(AMBIGUOUS)`;
exactly one gives `WRITE` using the splice above. There is no cursor or hint.
A nonempty replacement always creates exactly one terminated line. No-op
replacements are excluded. The initial grammar does not cover Unicode, even
though a later expansion may introduce a fixed normalization list after its
semantics are proven. Blank search is allowed and uses identical cardinality.

## Rules 1–6 and hand-computed cases

Notation: `F=b'...'` is the input file; `P` is hunk body (bare `@@`, `-`,
`+`; omitted full wrapper is `*** Begin Patch\n*** Update File: f\n@@\n` + P +
`*** End Patch\n`). Escapes are bytes, not visual line breaks. `W((i,j),b'...')`
is WRITE; `R(code)` is REFUSE and leaves F unchanged; `OOD` is OUT_OF_DOMAIN.
Each example is independently runnable against the real planner after adding
the wrapper. Examples tagged OOD test membership, not planner correctness.

### 1. Consumed provenance and newline preservation

In-domain: one hunk; exact line splice and per-line terminators as above.
Multi-hunk provenance and written-context anchors are not covered (no
independent bounded model yet). A final unterminated line is its own class.

* F=`b'a\nold\ntail\n'`; P=`b'-old\n+new\n'` → W((1,2), `b'a\nnew\ntail\n'`).
* F=`b'a\r\nold\r\ntail\r\n'`; same P → W((1,2), `b'a\r\nnew\r\ntail\r\n'`).
* F=`b'a\nold'`; same P → W((1,2), `b'a\nnew\n'`).

### 2. Global strict tier before lower tiers

In-domain: exact whole-line comparisons precede the fixed `a b`/`a\tb`
normalization pair. Other whitespace/tab normalization, Unicode and substring
fallback are not covered; the spec does not invent their general relation.

* F=`b'old\ntail\n'`; P=`b'-old\n+new\n'` → W((0,1), `b'new\ntail\n'`).
* F=`b'old\nold\n'`; same P → R(AMBIGUOUS), original `b'old\nold\n'`.
* F=`b'a\nb\n'`; same P → R(NOT_FOUND), original `b'a\nb\n'`.

### 3. Uniqueness and hints

In-domain: require exactly one candidate globally. Hints, hint windows,
cursors and hunk ordering are not modeled; all hinted patches are OOD, not a
refusal. Their units remain an open owner question.

* F=`b'a\nold\n'`; P=`b'-old\n+new\n'` → W((1,2), `b'a\nnew\n'`).
* F=`b'old\nold\n'`; same P → R(AMBIGUOUS), original `b'old\nold\n'`.
* F=`b'old\n'`; hunk `b'@@ hint @@\n-old\n+new\n'` → OOD; original `b'old\n'`.

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
compared. Rule 1's unterminated-final example writes `b'a\nnew'`, not
`b'a\nnew\n'`. Rule 6's ordinary deletion writes `b'a\n\nb\n'`, not
`b'a\nb\n'`; its unique blank deletion refuses with `empty search text`, and
its duplicate blank deletion also refuses with `empty search text`, not the
specified ambiguous category. Rule 5's missing-search diagnostic contains the
search string `old`, contrary to the no-source-text requirement, even though
it does refuse without writing. These are baseline findings, not evidence that
this proposed contract has already been implemented. The other in-domain
examples above had the expected success/refusal and resulting bytes. Out-of-
domain examples establish oracle membership only: real planner output cannot
be equated to an oracle outcome for them.

## Baseline red matrix (do not skip)

Run `tests/tools/test_patch_planner_targeting_matrix.py` on the exact baseline.
Expected-red node IDs, pinned at `74c994fd` (a red is a finding against the
proposed contract, not an xfail or a skipped test):

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
