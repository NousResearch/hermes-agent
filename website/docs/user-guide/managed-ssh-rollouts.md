---
sidebar_position: 9
title: "Managed SSH rollouts"
description: "What Desktop-managed SSH preparation, pinned rollouts, and recovery mean in the current build."
---

# Managed SSH rollouts

Hermes Desktop has a contract for updating several registered SSH installations
against one reviewed revision. In the current build, the fleet rollout provider
is **not enabled**. No qualifying real-SSH capacity measurement has been
supplied, so Desktop reports `unverified-capacity` and zero rollout capacity.
There is no supported way to start a managed fleet rollout yet.

:::warning Current availability
The rollout screens and protocol are fail-closed. Do not work around an
unavailable capability by treating the older one-install update button as a
fleet rollout, by supplying an arbitrary SSH target, or by retrying a target
whose result is unknown.
:::

## What is different about the two stages?

### Preparation

Preparation uses the existing individual managed SSH updater for the selected
installation. It follows that connection's configured branch tip before a fleet
target is reviewed and pinned. It does **not** contain a fleet target SHA and
does not prove that a later reviewed revision was installed. Desktop shows the
individual update and restore result with its correlation ID, discards any old
fleet review, and refreshes the inventory before a new target can be reviewed.

### Pinned rollout

A pinned rollout would apply one exact reviewed commit to the admitted wave of
installations. The target includes the repository, branch, 40-character commit
SHA, and protocol `1`. Each installation is also bound to its own installation
and source identity. The coordinator records authorization before the remote
mutation and accepts only the reviewed target.

The plan input accepts at most 500 installation rows. Updates run one at a
time. That input limit is not a claim of supported fleet size: the current
provider advertises zero installations and zero concurrency until real-SSH
capacity is measured on an approved disposable topology.

## Which connections are supported?

When the provider is enabled, this lifecycle is for **registered Desktop-managed
SSH connections**. Desktop's connection registry also supports local, remote
HTTP(S), and Cloud entries, but those are not managed SSH rollout targets.

The installation must have a coherent identity: an installation ID, canonical
code root, repository identity, connection configuration, verified host-key
fingerprint, remote user/port, profile, configured code path, required scopes,
and a reviewed source binding. Duplicate aliases or changed scope/source data
make the plan ineligible.

Never put a password, token, private key, or connection string in a rollout
plan, screenshot, bug report, or test fixture. Repository identity is
credential-free by design.

## Protocol 1 is a compatibility floor

The reviewed commit must contain this exact resource:

```json
{"protocol": 1}
```

The current implementation reads it from the exact reviewed Git object. Missing,
malformed, oversized, or different protocol metadata is refused. A higher value
is not silently assumed to work in this release; support for a future protocol
requires an explicit compatibility change.

## If an attempt becomes unknown

A restart can happen after launch authorization but before a conclusive receipt.
In that case Desktop marks the attempt **unverified** and requires attention. It
does not guess whether the remote update happened.

- **Recheck / reprobe** is read-only. It observes the existing correlation and
  can settle that original attempt only when the observation is terminal and
  matches its correlation ID. It does not launch another update.
- **Recover** restores the recorded scopes and proves clearance for an
  unverified or recovery-required attempt. It also does not launch another
  update and does not declare the target successful.
- A new update is a new pinned admission, not an automatic retry of an unknown
  launch.

An unresolved fence remains until settlement evidence for the same installation
and correlation is validated. Archiving or pruning a record does not erase that
fence. This prevents a restart, stale screen, or local success-looking message
from turning an uncertain remote state into a second mutation.

## What is available now?

The current Desktop build may still expose the separate single-install managed
SSH update path. Where shown, it applies to one registered SSH installation and
owns its drain, update, receipt, and scope restoration lifecycle. It is not a
fleet rollout and does not provide wave promotion, fleet inventory, or the
managed rollout provider.

The fleet capability is currently reported as:

- protocol: `1`
- available: `false`
- reason: `unverified-capacity`
- maximum concurrency: `0`
- maximum installations: `0`

If the managed rollout section is absent or reports that reason, there is no
available fleet Start action. The separate supported single-install path may
still be available for a registered managed SSH connection.

## Validation boundary

The repository contains contract and injected-fixture tests for validation,
state transitions, durable fences, and receipt correlation. Those tests do not
establish that a real SSH host or native fleet provider was contacted. The
Desktop Playwright SSH acceptance cases report skips without an approved
disposable fixture provider; they refuse non-test targets and never accept
credentials. Those skips are not remote verification.
