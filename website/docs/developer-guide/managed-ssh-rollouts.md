---
sidebar_position: 12
title: "Managed SSH rollouts"
description: "The managed SSH preparation, pinned-target, evidence, and recovery contracts used by Desktop."
---

# Managed SSH rollouts

This page is the implementation contract for Desktop-managed SSH rollouts. It is
written from the TypeScript/Electron and Hermes CLI contracts in this repository,
not from a claim that a live fleet provider is available.

> **Current status:** the Electron main process currently advertises the managed
> rollout capability as unavailable with reason
> `trusted-assurance-provider-unavailable`, `maxConcurrency: 0`, and
> `maxInstallations: 0`. The renderer must not fall back to the older one-install
> update surface when the rollout capability is unavailable. The state machine,
> validation, journal, and recovery seams described below are therefore the
> contract for the provider integration; they are not evidence of a working
> native SSH fleet rollout in this build.

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
- `apps/desktop/electron/main.ts` and `preload.ts` — the currently fail-closed
  capability boundary.

## Preparation and pinned rollout are different operations

The first operation makes a managed SSH installation eligible for review. The
second operation applies one already-reviewed target. Preparation must complete
before a pinned target can be frozen and admitted.

### Preparation: refresh eligibility without a pinned target

`createManagedSshUpdateService().prepare(connectionId)` is deliberately
unpinned:

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

The contract allows at most **500 installations** and **4 concurrent updates**.
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
is observed. The state machine therefore refuses to guess after a restart.
`restart` enters `reconciling` and sets `continuationRequired`. An authorized or
observed attempt that cannot be conclusively settled is reconciled to
`unverified`, and the rollout enters `attention-required`.

**Unknown is neither success nor failure.** It is a durable admission that the
last launch has no conclusive settlement. It must not be redispatched merely
because the process restarted.

| Operation | Purpose | May apply the update? | Successful result |
|---|---|---:|---|
| **Recheck / reprobe** | Read-only observation of the existing correlation. | No | A terminal observation may settle the original attempt; a non-terminal observation leaves it unverified. |
| **Recover** | Re-establish restore/clearance for an unverified or recovery-required attempt. | No | Exact correlation plus positive clearance; recovery state is cleared only when proved. |
| **New pinned rollout** | A separately admitted reviewed target. | Yes | A new authorization, not a retry of an unknown launch. |

`reprobe` is admissible only for authorized, observed, unverified, or
recovery-required attempts. Its result must carry the attempt's exact correlation
ID. A non-terminal result deliberately leaves the state unchanged. `recover` is
admissible only for unverified or recovery-required attempts; a mismatched
correlation or unproved clearance is refused. Recovery does not mark an update
successful and does not issue a new launch capability.

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
transition. `managed-rollout-journal.ts` requires a
`settlement-validated` evidence fact for the same correlation ID and installation
before releasing it. Pruning a settled journal record with an unresolved fence
leaves a tombstone in the unresolved index so the debt remains enumerable.

Promotion is also evidence-gated. A healthy target must have a fresh observation
for the exact sweep, the admitted SHA, complete and ready scopes, clear update
marker, clear recovery state, verified process identity, and a correlated
successful receipt whose requested and post-update SHAs equal the reviewed SHA.
Prior waves are revalidated before promotion, and any fence or changed required
scope set blocks promotion.

## Current support limits

The current Electron integration intentionally stops before a runnable fleet
provider:

- `hermes:managed-rollouts:capabilities` returns protocol `1`,
  `available: false`, reason `trusted-assurance-provider-unavailable`, and zero
  capacity.
- The renderer treats the bridge as optional. If it is absent, the managed
  rollout section is not rendered. If the capability is unavailable, polling
  and commands are not used.
- The renderer's contract explicitly says not to fall back to
  `connections.updateManaged`; that path is the separate single-install
  lifecycle.
- No trusted source reader, coherent inventory reader, or assurance reader is
  wired into the Electron provider boundary in this build.
- The Desktop Playwright acceptance file is intentionally named and skipped
  until a credential-free provider/fixture adapter exists. It does not connect
  to an SSH host and does not prove native SSH behavior.

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
validation. A provider integration is ready for enablement only when it supplies
trusted source, inventory, and assurance readers, exposes non-zero capabilities,
and passes independent tests for exact target binding, unknown-state fencing,
recheck/recovery separation, and promotion evidence.
