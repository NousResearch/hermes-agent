# Native background work results

A long-running process can finish a finite work item before the process exits.
Readiness pattern notifications are rate-limited and may be disabled after a
flood; they are not a durable result channel. Native background processes can
opt into separate, bounded result obligations through `terminal.completion_work`.
Ordinary commands and readiness limits are unchanged.

## API and producer protocol

Use an existing native gateway session (including gateway-hosted local sessions),
the local terminal backend, `background: true`, and `pty: false`:

```json
{
  "command": "python producer.py",
  "background": true,
  "notify": false,
  "completion_work": [{"work_id": "build", "incarnation": "attempt-1"}]
}
```

The exact enrolled process writes a line to its combined output stream:

```text
HERMES_WORK_RESULT_V1 {"work_id":"build","incarnation":"attempt-1","status":"completed","evidence":"Tests passed; artifact: build/report.txt"}
```

The prefix includes one space; the line must end with LF. JSON must have exactly
`work_id`, `incarnation`, `status`, and `evidence`, all strings. Status is
`completed`, `failed`, or `cancelled`. Evidence is opaque untrusted text: a path
is not opened, a command is not executed, and the producer cannot supply routing
or parent fields. Unknown bindings, malformed JSON, duplicate fields, unsupported
statuses, and oversized frames do not create results. Fragments are accumulated
until LF; overlong lines are discarded through their delimiter. Protocol frames
do not also generate readiness notifications.

Limits are deliberate:

- 1–16 distinct bindings per spawn; each identifier is 1–128 characters.
- A `(work_id, incarnation)` cannot be enrolled twice for the same parent.
- At most 8192 decoded characters per frame, including prefix and delimiter;
  at most 4096 characters of evidence.
- At most 4096 ledger rows per profile. Full ledgers refuse enrollment; this
  feature does not automatically prune records or introduce a retention policy.
- A storage failure gets at most three attempts for the same frame, with 50 ms
  between attempts. Exhaustion retains unresolved state and the durable log.

## Delivery and lifecycle

Enrollment binds the process, profile, original parent and route before spawning.
The existing durable output log and checkpoint mechanisms retain the transport.
The first valid result is committed to the profile's SQLite ledger. Exact
repeats are idempotent. Conflicting results preserve the first payload and record
a conflict; a conflict observed before admission prevents delivery. Durable
payload evidence is redacted, while a digest distinguishes conflicting originals.

The existing gateway notification drain submits results to the original parent's
native automation FIFO, even with ordinary process notifications off. The
ledger disposition and FIFO admission commit in one transaction. Missing adapters,
a full queue, and temporary admission failures leave the obligation pending;
there is no direct-send fallback. A lost acknowledgement does not duplicate a
committed admission. Existing unknown-execution recovery applies without replaying
an already admitted result.

Stop fences enrollment and undelivered obligations. `/new` cannot deliver an old
result to the replacement conversation. Compression preserves the original
logical owner's obligation. Invalid persisted identifier types are preserved and
skipped before routing so healthy sibling obligations can still be admitted.

Two definite lifecycle observations have separate **native supervision notices**:
OS rejection before a child starts, and complete output EOF without a valid
terminal result. These notices say **task outcome unknown**, keep producer payload
and digest NULL, preserve the original lifecycle error, and do not authorize a
rerun. They use the same owner, transaction and Stop fences as results, independently
of `notify`. A process exit code alone never becomes a producer task status.
Unreadable or incompletely drained logs and exhausted storage retries remain
unknown; they do not fabricate a missing-result notice.

Spawn and process-status receipts expose `work_results`, including disposition,
lifecycle error, admission status/outcome, and `acceptance: "not_verified"`.
Admission means a native turn was queued, not that the evidence was verified or
the work accepted. Original OS exception identity is preserved if spawn fails;
secondary receipt failures cannot replace that exception.

## Tests and scope

The focused behavioral tests are:

```sh
scripts/run_tests.sh \
  tests/gateway/test_process_work_results.py \
  tests/gateway/test_work_result_boundaries.py \
  tests/gateway/test_work_result_lifecycle.py \
  tests/gateway/test_process_supervision_notices.py \
  tests/gateway/test_work_result_corrupt_rows.py \
  tests/tools/test_persistent_work_results.py \
  tests/tools/test_work_result_failure_boundaries.py \
  tests/tools/test_work_result_public_contract.py
```

They cover real SQLite/native admissions, finite child output, checkpoint recovery,
Stop races, conflicts, storage faults, profile isolation, supervision notices and
malformed sibling rows. Existing watch-pattern, completed-result, stdin-guard and
automation-admission tests provide regression coverage for shared paths.

This is opt-in native delivery, not retrospective recovery of arbitrary process
logs. It adds no scheduler, automatic rerun, exhaustion replay, external producer
transport, or evidence-acceptance authority. Physical disk outages and native
Windows process topology require separate platform validation.
