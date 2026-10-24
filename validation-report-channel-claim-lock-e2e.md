# Channel Claim-Lock — End-to-End Validation Report

Task: `t_24b1959a` — replay the 2026-10-23 incident, verify one answer
Incident channel: `8grzznya3trpbfxjwf5goe778h` (Mattermost DM lane)
Closure evidence for MM thread: `mircy79x43fmudxbtdjz4kknjw`

## Verdict

**PASS — all six checks green, deterministically, across 3 consecutive full-suite runs.**

The claim-lock work (intra-host SQLite gate + cross-instance relay gate) holds the
incident scenario: one question, the double-delivery trigger, a thread-keying collision,
and a second gateway on the same channel token. Every inbound message produced exactly
one session and one answer at every point of the run.

Validation found **two real defects in the merged implementation**. Both are fixed and
pinned in this branch (see "Defects found"). The incidents' failure modes were
reproduced first, then verified closed after the fixes.

## Harness

- Location: `tests/gateway/test_channel_claim_e2e_replay.py` (+ `_claim_e2e_support.py`, `_pipe_b_driver.py`)
- Stack: real `GatewayRunner` + Mattermost adapter + real SQLite state store, stubbed
  model transport. No fake session machinery.
- Branch: `integrate/claim-lock-e2e` — merge of `fix/channel-claim-lock` (`a38109346a`)
  + `feat/cross-instance-claim` (`5c81a558a7`), pushed to
  `https://github.com/weslien/hermes-agent` (branch `integrate/claim-lock-e2e`).
- Commits: `ba3ef3328b` (merge), `c59a1773f0` (harness), `13da5189cf` + `de07883dc5` (fixes).

## The six checks

### Check 1 — Exactly one active session per (platform, channel)

Test: `test_single_working_stream_and_single_compression_block` (+ the recorder in all replay tests).

- Every journal re-authorization resolved to the same session id:

```
OK session_id='20261023_213042_1cefea7a'
OK session_id='20261023_213042_1cefea7a'
OK session_id='20261023_213042_1cefea7a'   (per-message route check, repeated per turn phase)
```

- One `Working` heartbeat bubble per tick; no divergent streams:

```python
hearts = world.sendlog.heartbeat_edits()
assert hearts, "no heartbeat ticks captured (notify interval 1 s over slow turn)"
assert len({h["message_id"] for h in hearts}) == 1, "divergent heartbeat streams"
assert iters == sorted(iters)                # iteration counters monotone, no second counter
```

- `hermes_state_runtime.session_admissions` recorded exactly one admittance; the sampler
  saw `max_channel_sessions_seen == 1` throughout.

### Check 2 — Exactly one answer per inbound message (double delivery)

Test: `test_replay_double_delivery_single_answer`.

The same post id was delivered twice against a busy turn (queue mode). Journal:

```
INFO gateway.run:run_turn_persistence.py:104 response ready: platform=mattermost
     chat=8grzznya3trpbfxjwf5goe778h session=agent:main:mattermost:dm:8grzznya3trpbfxjwf5goe778h
     time=3.6s api_calls=1 response=18 chars
```

One `response ready` line. One answer hash in the adapter outbox. The duplicate
delivery stood down: `channel claim held by ...` — no second voice.

### Check 3 — Thread reply routes to the thread-root session

Test: `test_replay_thread_reply_routes_to_thread_root`.

A top-level post `post-t1` and a thread reply `post-t2` (root `t1`)
admitted into the same single lane; the answer carried the reply's hash.

```
ADMISSIONS: both target session=...213042_1cefea7a  (frozen route re-authorized OK)
```

### Check 4 — Queued events drain when the active session ends

Test: `test_queue_drain_runs_after_active_session_ends` (queue mode `HERMES_GATEWAY_BUSY_INPUT_MODE=queue`).

The follow-up message queued behind the live turn, then drained and answered after the
turn ended. The queued row left the FIFO; no re-auth pause.

### Check 5 — Lease expiry releases the claim; a waiting event admits cleanly

Test: `test_lease_expiry_releases_claim_and_waiting_event_admits`.

Production lease is 300 s (`CHANNEL_CLAIM_LEASE_SECONDS`). The identical expiry path
ran with a 2 s lease (module-level partial patch — the gate late-imports the functions).
Mid-turn `kill` (task cancel, no renew): the claim row survived to the lease window
(`lease_expire` held, holder session named), expired, was stolen, and the waiting event
admitted and answered.

### Check 6 — Cross-instance: the second gateway stands down, posts nothing

Test: `test_cross_instance_second_gateway_stands_down`.

Gateway A claimed the channel (claim files under the shared
`bot_relay/channel_claims/` routing root). Gateway B ran as a full stack in its own
process, own `HERMES_HOME`, own state DB, same relay root. Result:

```
sends: 0     edits: 0     session_keys: 0
[WARNING] gateway.session_ingress: channel claim held by other_instance=gatewayA-workstation; standing down
[WARNING] gateway.session_ingress: channel claim held by other_instance=gatewayA-workstation; standing down
[WARNING] gateway.session_ingress: channel claim held by other_instance=gatewayA-workstation; standing down
```

Gateway B admitted no session, posted nothing, and its journal names the holding
instance. Gateway A stayed the sole voice.

### Compression-config duplication guard

The compression-config duplication check reads pytest `caplog` (a fresh
`CaptureHandler` is evicted by the plugin loader's logging reconfiguration — that
eviction is why early journal reads appeared empty). The per-session compression-config
emission fired at most once per session:

```python
caplog_records = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
blocks = compression_block_lines(caplog_records)
assert len(blocks) <= 1, f"compression-config blocks duplicated: {blocks}"
```

## Determinism — acceptance criterion

Three consecutive full-suite runs, all six tests, zero failures:

| Run | Result | Wall time |
|-----|--------|-----------|
| det1 | 6 passed, 0 failed | 53.9 s |
| det2 | 6 passed, 0 failed | 52.7 s |
| det3 | 6 passed, 0 failed | 53.3 s |

Parent-surface regression after the fixes: `test_channel_claims.py` +
`test_bot_relay_relay_claims.py` 27/27; `test_msgraph_webhook.py` 13/13.

## Defects found (both fixed in this branch)

### D1 — Thread reply never drained after queue (§6.1 fold refused at re-auth)

Symptom: the thread reply queued correctly, then the drain-time re-authorization raised
`admission_conflict` at `check_native_route` (`gateway/session_authority.py:1052`):

```
REFUSE route=<dm:8grzznya3trpbfxjwf5goe778h:post-t1> session_route=<dm:8grzznya3trpbfxjwf5goe778h>
```

Cause: §6.1 lands a thread reply in the channel lane at admit time, but the frozen
`route` stays thread-scoped. The drain compared it against the channel-scoped session
route and refused what the fold had prescribed. The queued reply paused forever.

Fix: `13da5189cf` — `check_native_route` re-derives the channel-scope fold from the
event source and accepts it when it matches. Pinned by check 3.

### D2 — Cross-instance stand-down posted a second voice (§8.2)

Symptom: gateway B answered the user in the channel while gateway A owned the claim:

```
SEND: "Sorry, I encountered an error (ChannelClaimBlocked).\nchannel_claim_standdown\nTry again or use /reset to start a fresh session."
```

Cause: the relay gate raises `ChannelClaimBlocked` with reason `channel_claim_standdown`.
The catch arm logged the stand-down but fell through to `raise` — only
`channel_claimed_elsewhere` and `channel_claim_inflight` were silenced. The base error
handler turned the escape into a visible reply — the exact double-answer class the
2026-10-23 incident produced.

Fix: `de07883dc5` — the stand-down arm returns `None` like its two sibling arms. Pinned
by check 6: gateway B now posts nothing.

## Handling note for upstream

Both fixes apply to files owned by `fix/channel-claim-lock`. Cherry-pick `13da5189cf`
and `de07883dc5` onto that branch before release; the harness on
`integrate/claim-lock-e2e` guards both (`c59a1773f0`). After that, the branch is ready
for the upstreaming follow-up listed in `t_b2001624`.

## Closure excerpt for MM thread mircy79x43fmudxbtdjz4kknjw

> E2E validation of the channel claim-lock is complete (t_24b1959a). We replayed the
> 2026-10-23 pattern on the DM lane: one question, the double-delivery trigger, a
> thread-keying collision, and a second gateway on the same token. All six checks pass
> in 3 consecutive runs: one session, one answer, thread replies route to the thread
> root, queues drain, the 300 s lease releases on kill, and the second gateway stands
> down with no post. Validation caught two real gaps — a thread-reply drain stall and
> a visible error reply on the cross-instance stand-down; both are fixed and pinned by
> tests. Claim-lock is validated end-to-end and ready to close the incident.
