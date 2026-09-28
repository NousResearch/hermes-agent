---
sidebar_position: 12
title: "Managed SSH rollouts"
description: "The managed SSH preparation, pinned-target, evidence, and recovery contracts used by Desktop."
---

# Managed SSH rollouts

This page is the implementation contract for Desktop-managed SSH rollouts. It is
written from the TypeScript/Electron and Hermes CLI contracts in this repository,
not from a claim that a live fleet provider is available.

> **Current status:** the provider has a local source, inventory, and assurance
> integration, but no qualifying real-SSH capacity measurement is supplied.
> It therefore reports `unverified-capacity`, `maxConcurrency: 0`, and
> `maxInstallations: 0`. The renderer must not fall back to the older one-install
> update surface when the rollout capability is unavailable. The tests described
> below do not establish a working native SSH fleet rollout in this build.

The main source files are:

- `apps/desktop/src/lib/managed-rollout-contract.ts` — wire types, phases,
  limits, and validators.
- `apps/desktop/electron/managed-ssh-update-service.ts` — per-install SSH
  admission, preparation, launch capability, and durable recovery.
- `apps/desktop/electron/managed-rollout-preflight.ts` — exact-target and
  protocol inspection.
- `apps/desktop/electron/managed-rollout-inventory.ts` and
  `managed-rollout-assurance.ts` — trusted inventory and assurance inputs.
- `apps/desktop/electron/managed-rollout-coordinator.ts` — serialized rollout
  transitions and wave admission.
- `apps/desktop/electron/managed-rollout-journal.ts` — durable records and
  unresolved fences.
- `apps/desktop/electron/managed-rollout-main-integration.ts` and
  `managed-rollout-desktop-runtime.ts` — the local adapters, shared update
  admission check, and owner-gated IPC registration.
- `apps/desktop/electron/main.ts` and `preload.ts` — the process-owned wiring
  and renderer bridge.

## Preparation and pinned rollout are different operations

The first operation makes a managed SSH installation eligible for review. The
second operation applies one already-reviewed target. Preparation must complete
before a pinned target can be frozen and admitted.

### Preparation: refresh eligibility without a pinned target

The service exposes an internal `prepare(connectionId)` contract that is
deliberately unpinned:

1. It resolves a registered connection and requires a Desktop-managed SSH
   source. Local, URL/HTTP, Cloud, and other source kinds are refused.
2. It refuses an `intent` or any pinned target. The preparation call cannot
   smuggle a SHA into the pre-review step.
3. The injected `prepareRemote` operation returns a receipt whose kind is
   `preparation` and whose correlation ID exactly matches the operation.
4. The receipt is recorded, then eligibility is refreshed. Only after that
   refresh may the caller freeze and review a target.

Preparation is not a fleet rollout and is not proof that a particular commit was
applied. It is the unpinned branch-tip/eligibility phase. A preparation failure
returns `preparation-failed`; an invalid source, concurrent operation, malformed
correlation ID, or pinned intent returns `refused`.

The current Settings workflow performs explicit preparation through the
existing individual `updateManaged` operation, then discards any earlier fleet
review and refreshes inventory. The production service does not inject
`prepareRemote` for its separate internal `prepare()` method, so that method
cannot be called as a second live preparation route. Both the individual
operation and fleet dispatch borrow the same Desktop update gate and operation
maps; startup refuses the rollout provider if those identities differ.

### Pinned rollout: apply only the reviewed target

A pinned target is the exact object represented by `RolloutTarget`:

```json
{
  "repositoryId": "...",
  "branch": "...",
  "sha": "<40 lowercase hexadecimal characters>",
  "protocol": 1
}
```

The plan also binds every installation to an installation fingerprint, source
fingerprint, required scope set, and a reviewed source binding. The reviewed
source binding contains the repository root, credential-free origin, resolved
origin ref, target SHA, assurance profile, assurance evidence digest, and
assurance generation.

The coordinator's launch edge is intentionally narrow:

1. Capture and verify inventory, source, assurance, exact Git object, protocol,
   and target identity.
2. Record intent for one target in the current wave.
3. Persist `launch-authorized` durably.
4. Issue an opaque, single-use launch capability to the managed SSH service.
5. Consume that capability only at the remote mutation edge.

A coordinator update without a reviewed pinned intent is refused. Conversely,
an intent-bearing update outside coordinator admission is refused. A receipt or
matching SHA by itself is not a promotion proof.

The contract accepts at most **500 installation rows** and runs updates
**serially**. Eight probe slots may be used for a bounded evidence sweep; they
are not update slots. A later advertised installation limit must come from
measured real-SSH capacity and may be lower than 500.
Promotion is either manual or `auto-after-canary` in the coordinator contract;
the canary still requires explicit manual approval before automatic promotion
can advance later waves. The currently advertised provider capacity is zero, so
these are release bounds, not a claim that the current build can use them.

## Protocol floor and exact-object inspection

The protocol resource is fixed at
`hermes_cli/update_rollout_protocol.json`. In this checkout its content is:

```json
{"protocol": 1}
```

Protocol 1 is the compatibility floor for the managed rollout contract. The
current preflight parser is intentionally strict: the resource must be valid
UTF-8 JSON, contain exactly the `protocol` key, and declare the integer `1` at
the exact reviewed Git object. Missing, malformed, oversized, or different
protocol metadata is rejected. This release does **not** silently treat a
higher protocol as compatible; support for a future protocol requires an
explicit contract change and tests.

The target SHA is inspected as a commit, and the protocol is read from that
exact object rather than from the working tree or an unrelated branch tip. The
reviewed source check additionally verifies the expected origin/ref, repository
root, branch, commit ancestry, and protocol resource before the source is marked
verified.

## Supported topology and identity boundary

The rollout contract is for a roster of **registered Desktop-managed SSH
installations** that expose the same reviewed repository identity. A registry
can contain local, remote, Cloud, and SSH connections, but only the SSH kind is
eligible for this managed SSH update lifecycle. The existing service rejects
other kinds without touching their processes.

Each installation is identified independently from its display label or address:

- `installId` identifies the remote installation.
- The installation fingerprint binds that ID to the canonical code root and
  repository ID.
- The source fingerprint binds the installation to the connection ID, config
  revision, verified host-key fingerprint, remote user, port, configured profile,
  and configured code path.
- Alias connection IDs may describe the same installation, but aliases may not
  collide across installations.
- Required scope IDs are captured and compared as a set. A missing, duplicate,
  changed, or incomplete scope set is not eligible for promotion.

Repository origins are normalized without carrying credentials into identity.
Origins containing a password, query, fragment, whitespace, or control bytes are
rejected. Do not place tokens, passwords, private keys, or connection strings in
plans, fixtures, documentation, or test output.

This topology is narrower than Desktop's general multi-connection registry. A
local process, HTTP(S) gateway, Cloud entry, or arbitrary SSH host is not a
managed rollout target merely because it appears in the registry; it must be a
verified registered SSH installation with a coherent inventory row and reviewed
source binding.

## Recheck, recover, and unknown state

A remote mutation can be authorized before the process terminates or its receipt
is observed. The state machine therefore refuses to guess after a restart. The
reducer's `restart` edge enters `reconciling` and sets `continuationRequired`,
and its `reconcile-unknown` edge turns an authorized or observed attempt into
`unverified` with the rollout in `attention-required`.

Live hydration follows the same refusal without guessing. When a restart reopens
a durable record, the provider restores the persisted state directly with
`continuationRequired` set: a record whose attempts include an unverified,
recovery-required, failed, or refused attempt returns as `attention-required`,
and an otherwise running record returns as `paused`. It never replays the
reducer's `reconciling` phase, and it never redispatches the attempt —
conclusive settlement stays required. A `restart-reconcile` event is persisted
whenever the hydrated state differs from the persisted snapshot.

**Unknown is neither success nor failure.** It is a durable admission that the
last launch has no conclusive settlement. It must not be redispatched merely
because the process restarted.

| Operation | Purpose | May apply the update? | Successful result |
|---|---|---:|---|
| **Recheck / reprobe** | Read-only observation of the existing correlation. | No | A terminal observation may settle the original attempt; a non-terminal observation leaves it unverified. |
| **Recover** | Re-establish restore/clearance for an attempt whose fence still needs proved clearance. | No | Exact correlation plus positive clearance; the durable fence is released only when clearance is proved, and the attempt's own state and outcome are not changed. |
| **New pinned rollout** | A separately admitted reviewed target. | Yes | A new authorization, not a retry of an unknown launch. |

`reprobe` is admissible only for authorized, observed, unverified, or
recovery-required attempts. Its result must carry the attempt's exact correlation
ID. A non-terminal result deliberately leaves the state unchanged. `recover` is
admissible for every attempt that can still carry an unresolved obligation —
unverified and recovery-required, plus settled failed, refused, updated, and
already-current attempts whose fence still needs proved clearance. A mismatched
correlation or unproved clearance is refused. For a launched attempt, the
receipt found during recovery must itself prove the reviewed request — its
recorded requested and post-update SHAs must equal the pinned target — before
clearance can be proved; a receipt that never recorded which request it
answered, or answered a different one, leaves clearance unproved. A prepared
pre-launch record may correctly have no receipt; there the durable scope
record and clear remote markers govern. Recovery does not mark an update
successful, does not relabel a failed/refused outcome, and does not issue a new
launch capability. Proved clearance releases the installation's durable fence
and is recorded as its own `recovery-cleared` evidence kind, carrying the
structured clearance artifact (clear markers, removal of the original durable
obligation, and whether the launched request was proved) that the durable
journal revalidates before release; it does not clear the attempt's own
recovery state — an unverified attempt remains unverified and may still report
`recoveryRequired`, because recovery does not settle the launch.

For the single-install service, durable recovery reopens the transport, waits
for the required restore clearance, closes the transport, and restores each
recorded scope. Failed scope restores remain pending for a later launch. Primary
routing cannot be mutated while an update, preparation, recovery, restore owner,
or durable recovery record remains active.

## Fences and durable evidence

The journal records an unresolved fence for an installation when an authorized
launch has no conclusive settlement. A fence is keyed by rollout ID and install
ID and retains its correlation ID, reason, and recording time. The unresolved
index survives process restart and is visible to admission/promotion checks.

A fence cannot be removed by archive, exclusion, or a successful-looking local
transition. `managed-rollout-journal.ts` releases it only on one of two distinct
evidence kinds recorded for the same rollout ID, correlation ID, and
installation: `settlement-validated`, a receipt-backed success whose requested
and post-update SHAs equal the reviewed target, or `recovery-cleared`, proved
recovery clearance of the original durable scope obligation, which must carry
the structured clearance artifact and can never be a bare assertion. A fence
also never accepts a foreign tag: fence evidence is bound to the rollout that
owns the obligation, and a fence tagged for another rollout cannot be added to
a record or released by that record's facts. Recovery clearance never asserts
that the update applied and is never recorded as a success settlement.
Pruning a settled journal record with an unresolved fence leaves a tombstone
in the unresolved index so the debt remains enumerable.

Promotion is also evidence-gated. A healthy target must have a fresh observation
for the exact sweep, the admitted SHA, complete and ready scopes, clear update
marker, clear recovery state, verified process identity, and a correlated
successful receipt whose requested and post-update SHAs equal the reviewed SHA.
The settlement path requires that proof from both receipts — the service's own
receipt and the live observation receipt — so a cooperating adapter cannot
substitute an observation receipt for a service receipt that never proved the
request. A live Main observation is itself a complete readiness projection:
success is only projected when the receipt proved the reviewed target and the
installation is ready with clear markers, coordinator readiness for the
correlation, a complete scope capture, and every scope restored with a verified
process identity at the reviewed SHA. A receipt-backed known failure (refused
or failed) keeps its own classification even when a follow-up live observation
cannot run; everything else stays unknown.
The live sweep is bounded to the current and next wave: the 240-probe budget is
a deliberate bound, not a claim that every earlier wave is re-probed. Earlier
waves are revalidated against their recorded settlement evidence — each
attempt's persisted receipt correlation, requested SHA, post-update SHA, health,
and scope proof must still match the in-memory attempt, so a legacy receipt that
never recorded its requested SHA cannot carry a wave forward — and any
unresolved fence, or a persisted attempt whose recorded scope proof no longer
covers its required scope set, blocks promotion. That earlier-wave check is a
comparison of persisted local evidence: it does not re-probe earlier waves
remotely.

## Current support limits

The current Electron integration stops at the capacity gate before a runnable
fleet provider:

- `hermes:managed-rollouts:capabilities` returns protocol `1`,
  `available: false`, reason `unverified-capacity`, and zero capacity while no
  qualifying real-SSH capacity measurement is supplied. If local dependencies
  are missing, it reports `trusted-rollout-dependencies-unavailable` instead.
- The renderer treats the bridge as optional. If it is absent, the managed
  rollout section is not rendered. If the capability is unavailable, polling
  and commands are not used.
- The renderer's contract explicitly says not to fall back to
  `connections.updateManaged`; that path is the separate single-install
  lifecycle.
- The local source, inventory, and assurance readers are wired to the provider.
  Their unit and local-integration tests do not establish a verified remote
  operating envelope or increase advertised capacity.
- The Desktop Playwright SSH acceptance file reports explicit skips without an
  approved disposable fixture provider. It does not connect to an SSH host and
  does not prove native SSH behavior in that state.

The currently available single-install path, where exposed by the Desktop
build, is not a fleet rollout. It is restricted to one registered managed SSH
connection, drains the exact Desktop-owned scopes, performs the remote update,
proves a correlated receipt, and restores prior scopes. URL/HTTP and Cloud
sources are refused. Treat that path and the rollout contract as separate
surfaces until the provider capability becomes available.

## Verification guidance

Contract tests should use injected readers, transports, journals, and local
fixtures. They may prove state transitions, validation, durable fencing, and
receipt correlation; they must not be described as native SSH or real-fleet
validation. Non-zero capability additionally requires a qualifying disposable
real-SSH measurement with a supported installation count. The reviewed
integrated tree must pass exact target binding, unknown-state fencing,
recheck/recovery separation, and promotion evidence on the real adapters.
