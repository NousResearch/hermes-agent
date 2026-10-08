# RoomLink cancellation retention

A missing response never permits a second execution. Stop addresses the original
task and execution generation. It either requests an existing run to stop or
records that the absent identity cannot start later.

Those exact records remain while their room authority is live. Grant refresh and
ordinary grant revocation followed by reauthorization do not erase them. A grant
identifier or a retention timeout cannot establish a new logical attempt.

The cancellation store compacts terminal cancellation records at two authoritative
lifecycle boundaries:

- The target accepts a verified successor authority epoch for that room member.
- Disband retires the current epoch using the grant's explicit `retire` permission,
  after the home has obtained Stop acknowledgments.

One durable watermark per room/home/member/target/profile replaces the retired
attempt records. It records the current authority gateway and epoch, plus the
highest explicitly retired epoch. Repeated epoch changes update this row. The
writer checks the captured request authority again, so a request authorized before
retirement cannot reserve work afterward. A later token cannot revive a retired
epoch. Existing live runs retain their status and stop intent until their owner
settles; retirement does not pretend that an executor stopped.

The watermark proves that an old authority cannot admit new work. It does **not**
prove that an individual old task never executed. When a request reaches this
boundary after its exact receipt was compacted, `run_history_retired` is an
ambiguous 409, not a fabricated cancelled/non-admitted task receipt. Consumers
must not use it as attachment-release or execution evidence.

New full grants include `retire`; narrower grants do not acquire it from a request
flag. Older endpoints and grants keep ordinary revocation through a narrowly
classified fallback, retaining their exact cancellation records. Old persisted
records acquire their authority metadata when their authenticated scope is next
observed; unidentifiable historical records remain retained.

This is lifecycle compaction, not an active-room quota. A still-live authority can
accumulate exact cancelled identities until Disband or an authority change. No
arbitrary lifetime cap disables an otherwise healthy room. Expiry alone, legacy
revocation without retirement permission, and never-observed historical scopes
do not justify forgetting non-replay evidence. SQLite reuses the freed pages;
compaction does not promise that the physical database file shrinks immediately.
