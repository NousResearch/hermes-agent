---
title: "Core Testflight — Verify exact TestFlight builds without auth or device loops"
sidebar_label: "Core Testflight"
description: "Verify exact TestFlight builds without auth or device loops"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Core Testflight

Verify exact TestFlight builds without auth or device loops.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/core-testflight` |
| Path | `optional-skills/devops/core-testflight` |
| Version | `3.4.0` |
| Author | Peter Beddow, Hermes Agent |
| License | MIT |
| Platforms | macos |
| Tags | `Apple`, `TestFlight`, `App Store Connect`, `Xcode Cloud`, `verification` |
| Related skills | [`accessiblecapture-app-install`](/docs/user-guide/skills/optional/devops/devops-accessiblecapture-app-install) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Core TestFlight Skill

Use this skill for Accessible Core iOS distribution and TestFlight work. It preserves the
Xcode Cloud → TestFlight pipeline, separates Apple distribution from phone control, and
never replaces TestFlight with a local device installation. The central rule is to use
the narrowest already-authenticated path for each operation and to resume an existing
run instead of starting a duplicate.

## When to Use

Use when Peter asks to distribute an exact iOS build, make it available in TestFlight,
install it through TestFlight, or verify an installed build. Do not use a local IPA or
direct-device installation as a substitute for TestFlight.

## Prerequisites

Record these exact values before acting:

- bundle ID, marketing version, requested build number, source commit, GitHub ref,
  Xcode Cloud workflow, beta group, and tester label;
- the native checkout and its current release receipt template/provenance checker;
- the current delivery mode: `TESTFLIGHT` or `DIRECT_DEVICE`.

Use the established authenticated tools already available on the workstation:

- **GitHub/ref work:** the authenticated Git client or `gh` path. Resolve the named ref
  and verify its full 40-character SHA. Do not use Core Secrets for GitHub operations.
- **Xcode Cloud:** the existing authenticated Xcode Cloud/App Store Connect client or
  approved build-start helper. Read the workflow, resolve the Git ref, start a run, and
  poll that run through the same path. Do not invent a workflow or use a local archive.
- **App Store Connect readiness:** the existing `scripts/app_store_connect.py` verifier.
  When it genuinely needs Apple API credentials, resolve the literal
  `apple/app-store-connect` reference through the existing `core-secrets` skill using
  the `WKST-01` principal. Keep fields and short-lived JWTs in memory only.

Never print or persist credentials, tokens, private keys, or connection strings.

## Authorization and authentication boundary

Approval is a state, not a prompt loop. When Peter has approved the bounded operation,
record that approval in the current work state and do not ask again for the same build,
workflow, ref, or run. If approval is absent for an externally visible write, ask once
before starting it. Do not ask for credentials; use the established authenticated path.

Use this order:

1. Resolve GitHub refs with authenticated GitHub tools.
2. Resolve and start/poll Xcode Cloud with its established authenticated path.
3. Use Core Secrets only when the Apple API client actually requires the approved
   `apple/app-store-connect` reference.
4. Never open App Store Connect in a browser to repair an API or credential problem.
   Browser login is allowed only for a named operation that the existing API cannot do,
   after reporting that limitation.

If an authenticated tool returns an authorization, credential, or transport failure,
report the exact redacted cause and stop that operation. Do not retry with another secret
store, request approval again, prompt for a password, or restart the workflow.

## Operations and evidence

Keep these stages independent and report each one:

1. **GitHub source** — the authoritative ref resolves to the requested commit.
2. **Xcode Cloud run** — the exact workflow/run was accepted and its source commit
   matches; an archive must succeed before a build exists.
3. **Apple processing** — the exact App Store Connect build record has the requested
   build number and acceptable processing state.
4. **TestFlight availability** — compliance, beta state, expiration, exact group
   membership, and tester eligibility permit testing.
5. **Installation** — the tester selects that exact build in TestFlight and taps Install
   on PHN-01. This is an iPhone action, not a Mac-side CoreDevice action.
6. **Installed-build verification** — fresh structured device evidence matches the
   bundle ID, marketing version, and exact `CFBundleVersion`.

Never infer one stage from another. TestFlight visibility is not installation evidence,
and an unavailable phone tunnel is not an Apple distribution failure.

## Idempotent Xcode Cloud procedure

1. **Recover before starting.** Read the current receipt, saved run ID, and recent API
   state. If a run already exists for the requested commit/workflow, poll it; do not
   start another run. If the run is terminal, read its terminal state before deciding
   whether a new approved run is necessary.
2. **Verify GitHub.** Use authenticated GitHub tools to resolve the exact named ref.
   Require the returned SHA to equal the requested 40-character commit. If it does not,
   stop with the mismatch and identify the ref that must be corrected.
3. **Verify the Xcode Cloud target.** Read the existing workflow and its repository/ref
   configuration through the established Xcode Cloud path. Require the requested commit
   to be the source of the run. Do not create or alter a workflow.
4. **Start once.** Only when no matching active run exists, start the existing workflow
   for the exact ref/commit. Record the returned run ID immediately. A successful start
   means only that the run was accepted; it is not archive or TestFlight success.
5. **Poll one run.** Read the run until it is terminal. Record run status, source commit,
   archive status, produced build number, and any redacted failure reason. Do not restart
   because a run is pending or running.
6. **Verify Apple processing.** Once a build number exists, use the exact build number
   with `scripts/app_store_connect.py`; never select “latest” or by list position. Require
   the build record, linked marketing version, processing state, expiration, QC,
   export-compliance state, and beta state.
7. **Verify TestFlight.** Confirm the exact build ID is in the intended beta group and
   that at least one enrolled tester is eligible. Read back every bounded Apple write.
8. **Handle installation.** In `TESTFLIGHT` mode, do not invoke `devicectl` merely to
   install or prove Apple readiness. Report the exact build and
   `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE` when the phone-side Install action is
   outside available agent control. If the phone is paired but has no usable tunnel,
   report `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE` and do not request reconnection.
9. **Verify device state only when selected.** In `DIRECT_DEVICE` mode, use fresh
   CoreDevice readback and require exact bundle ID, marketing version, and build number.
   A missing/stale readback is unverified. Never install a local build.
10. **Write a current receipt.** Preserve historical receipts. Record GitHub SHA/ref,
    workflow/run IDs, source commit, archive result, App Store Connect record/build,
    processing/QC/compliance, beta group/tester eligibility, delivery mode, installation
    action or device evidence, capture time, and outcome. Run the existing receipt and
    provenance checkers.

## Read-only Apple verifier

Run the verifier for the exact target:

```text
python3 scripts/app_store_connect.py \
  --bundle-id <bundle-id> \
  --build <build-number> \
  --beta-group <exact-group-name>
```

It must return a redacted machine-readable result with `verification=API`. Require
`processingState=VALID` (or an equivalent testable state), non-expired availability,
complete compliance, exact beta-group membership, and an eligible tester. A successful
API read does not prove that PHN-01 has installed the build.

## Blocker contract

A blocker is not a reason to restart. Stop and report:

- **Exact cause:** the redacted tool/API result, stage, and target operation.
- **One action:** the single owner or environment action that would remove that cause.
- **What remains unchanged:** no duplicate run, credential, browser login, local install,
  or unrelated device check was performed.

Examples:

- GitHub ref cannot resolve: owner action is to restore authenticated GitHub access or
  correct the named ref; do not ask for Apple credentials.
- Xcode Cloud start/poll is terminated before a response: owner action is to permit the
  already-approved authenticated Xcode Cloud request in an environment with outbound
  Apple API access; do not request approval again or start a second run.
- App Store Connect rejects the API key: report the HTTP status/title/detail without
  secrets; owner action is to repair that approved Apple API authorization.
- PHN-01 has no usable tunnel in `TESTFLIGHT`: no owner action is required for Apple
  readiness; report the exact build as available for TestFlight installation and do not
  request a connection check.

## Delivery mode and WKST-02 boundary

Record `delivery_mode` before acting:

- `TESTFLIGHT`: Apple-side checks and TestFlight state only. Phone control is optional.
- `DIRECT_DEVICE`: CoreDevice is required for the explicitly selected direct-device
  operation. If it fails, stop that operation and return to `TESTFLIGHT`; do not retry
  automatically or use WKST-02.

WKST-02 is never an automatic fallback. Use it only when Peter explicitly names that
remote route, preserving its existing restrictions and authentication.

## Status contract

Use these outcomes exactly:

- `DONE`: current installed-build evidence matches the target build.
- `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE`: Apple readiness is verified, but Hermes
  cannot control/read PHN-01 from the current workstation.
- `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE`: Apple readiness is verified and the
  phone-side TestFlight Install action remains outside available agent control.
- `BLOCKED`: required owner input/approval, authorized Apple authentication, or a named
  direct-device operation is unavailable.
- `FAILED`: Apple distribution or a bounded Apple write genuinely failed.

Do not combine device-control status with Apple distribution status.

## Verification report

Report these fields separately and do not omit a stage:

```text
GitHub commit: <40-character SHA>
GitHub ref: <exact ref and resolved SHA>
Xcode Cloud: <workflow/run/source/archive/build/status>
App Store Connect: <record/build/processing/QC/compliance>
TestFlight: <group/tester/availability>
Target: <marketing version/build/bundle>
Installation: <TestFlight action or verified exact device readback>
Device control: <available/unavailable and exact reason>
Outcome: DONE / TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE / INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE / BLOCKED / FAILED
```
