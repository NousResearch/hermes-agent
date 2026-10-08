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

One durable watermark per room origin/member/target/profile replaces the retired
attempt records. It records the current authority home, gateway and epoch, plus the
highest explicitly retired epoch. Repeated epoch changes update this row. The
writer checks the captured request authority again, so a request authorized before
retirement cannot reserve work afterward. A later token cannot revive a retired
epoch. Existing live runs retain their status and stop intent until their owner
settles; retirement does not pretend that an executor stopped.

A home move must explicitly bind the successor to the predecessor. The
target-owner invitation accepts `previous_authority` with the exact current
`home_install_id`, `authority_gateway_id` and `authority_epoch`; the new epoch
must be higher. The target retains an alias per authorized successor home to the
original watermark and hidden member session. The writer resolves these aliases
after restart, including for requests captured before the move. Repeating an
invitation for the exact current successor is safe after a lost reply. An
unrelated home using the same room and member identifiers has a separate origin;
its higher epoch cannot compact the original records or join existing origins.
Older records lacking authenticated predecessor coordinates remain conservative.

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

After grant expiry, the target owner can issue an invitation with
`retirement_only: true` for retained authority coordinates. Its signed permissions
are exactly `status` and `retire`: capability probing and retirement work without
a live reservation, while dispatch, Stop, approval and refresh remain unavailable.
Issuance does not recreate a reservation or clear revocation. This also permits a
fresh idempotent retirement after a lost response. An older epoch can retire only
through itself; a different gateway at the current epoch is rejected and a newer
successor remains usable. An expired bearer alone never authorizes this recovery.

This is lifecycle compaction, not an active-room quota. A still-live authority can
accumulate exact cancelled identities until Disband or an authority change. No
arbitrary lifetime cap disables an otherwise healthy room. Expiry alone, legacy
revocation without retirement permission, and never-observed historical scopes
do not justify forgetting non-replay evidence. SQLite reuses the freed pages;
compaction does not promise that the physical database file shrinks immediately.
