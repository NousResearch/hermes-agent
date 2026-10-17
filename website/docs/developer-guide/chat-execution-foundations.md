---
sidebar_position: 30
---

# Durable chat-execution accounting foundations

This is execution accounting and conservative fencing, **not automatic abandoned-chat recovery**.
It does not transfer Linear ownership, clear retained claims, replay interrupted work, or establish
that arbitrary external effects have settled. Automatic recovery remains unavailable.

## What is recorded

The gateway reserves a generation in the owning profile's `chat-execution.sqlite3` before submitting
an ordinary chat turn. The reservation is initially unbound: the actual session ID is attached only
when the physical worker starts. A database failure or recovery fence refuses admission; routing
keys are not substituted for session IDs.

Queued recursive follow-ups keep the logical routing/lease generation of their chat chain, but each
physical worker gets a fresh journal generation. Reservation atomically requires the exact latest
predecessor's recorded physical completion and real session identity. The new reservation is still
unbound until its own worker starts. Stop and stale-result checks continue to use the logical chain
generation; a later ordinary inbound admission advances beyond the journal's durable floor.

The physical worker records its own completion from its target's `finally`. Stop acknowledgement,
wrapper cancellation, future completion and gateway restart are not physical completion evidence.

Selected producers reserve immutable resource identities before offering executable work:

- Concurrent and sequential tool workers, including prepared terminal workers.
- Middleware continuation callbacks, including synchronous invocation and cross-thread handoff.
- Per-turn external memory-prefetch workers. Shared provider-lifetime writers are not assigned to
  whichever turn happened to initialize the provider.
- Linux local non-PTY background-process launches. PID and kernel start ticks are recorded after
  `Popen`; a PID cannot be known before the process exists.
- Background `delegate_task` units and manual `cronjob(action='run')` background runs, both
  `async_delegation_thread`. A unit is reserved only after the pool admits it (a capacity, retirement
  or journal refusal reserves nothing and returns `rejected`), while the dispatching worker is
  pending and current. It is detached: its pool thread may start, and its subtree may observe tools
  and reserve its own non-detached resources against the dispatching generation, after that worker
  completes or a successor generation starts. That holds only while the unit's resource is started
  and the session is not fenced; a copied context without one, or one whose unit has finished, is
  refused like any stale identity. A detached row starts once, only through the handle issued to its
  dispatcher in this process: not from a handle rebuilt from journal rows, not after its submission
  failed or was cancelled, not after a restart. Such rows stay `reserved` (unresolved). Backend
  retirement gates only dispatch admission; it does not stop a running unit, and retirement
  `prepare()` stays not-idle until the unit's worker exits.

Resource identity captures profile home, session key, generation, session ID, kind and a unique
physical resource ID. Registry notification-owner handoff does not change that execution identity.

## Conservative unresolved state

A thread/callback resource finishes only when its exact physical target unwinds. That does not
prove descendants, persistent provider jobs or remote effects have stopped. Unused continuations,
failed submissions, uncertain writes and unfinished resources remain unresolved.

Terminal-process resources are deliberately not completed by the generic thread-resource finish
API. Direct-child exit, PID absence, signals and checkpoint pruning do not prove process-tree
settlement. A lineage-recording failure after launch invokes existing cleanup, but cleanup is not
reported as settlement proof.

Production executions retain `effects=unknown`. Older unresolved generations remain relevant;
a newer completed turn cannot erase their uncertainty. Legacy records acquire no retrospective
proof. There is no production caller that declares `effects=none`, and no gateway/plugin recovery
route that invokes the journal's transfer primitive. Its test-only positive fixtures are not
production recovery acceptance or a supported claim-release mechanism.

## Boundaries not closed by this change

The registered producers above do not form a complete accounting boundary. In particular, provider
initialization and lifetime writers, middleware's own effects, delegate effects beyond their recorded
units and resources, other descendants, PTY/remote process domains, scope/environment preflight
effects, delivery, and domain-specific external mutations still need independently verifiable
settlement authority. Existing Stop observations are not replaced with durable recovery approval by
this change.

The journal is trusted-runtime state, not a security boundary against arbitrary same-UID code.
Do not use age, task status, an open session-catalog row, missing lease, or an empty-looking process
registry to authorize takeover.

## Verification and delivery

Source acceptance exercises actual admission, physical workers, callback handoff, memory prefetch
and local process launches with isolated temporary persistence and controlled real-thread/process
cleanup. Before-effect probes use independent database connections inside the launch seam.

These tests establish the bounded accounting contracts, not arbitrary effect settlement, live
containment or crash-safe ownership transfer. Merging the foundations is separate from image
publication, deployment and any future authorization to enable recovery. A future recovery feature
must supply genuine outcome verifiers and an idempotent fenced ownership-transfer contract before
it can make a positive recovery claim.
