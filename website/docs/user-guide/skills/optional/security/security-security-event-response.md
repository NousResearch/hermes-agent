---
title: "Security Event Response — Use when a CVE, leaked secret or license issue hits code"
sidebar_label: "Security Event Response"
description: "Use when a CVE, leaked secret or license issue hits code"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Security Event Response

Use when a CVE, leaked secret or license issue hits code.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/security/security-event-response` |
| Path | `optional-skills/security/security-event-response` |
| Version | `0.1.0` |
| Author | rlaope (github.com/rlaope), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `incident-response`, `cve`, `secret-leak`, `credential-rotation`, `license`, `supply-chain` |
| Related skills | [`oss-forensics`](../../optional/security/security-oss-forensics.md), [`application-threat-model`](../../optional/security/security-application-threat-model.md), [`data-share-review`](../../optional/security/security-data-share-review.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Security Event Response Skill

Respond to a security event on code that already exists: a CVE or advisory in a dependency, a secret
committed or pushed, a dependency whose license may not fit the product, or an advisory that forces
a major upgrade. It produces an event record, a reachability-adjusted severity, an ordered
containment plan, and a closure verdict that stays open until the fix or the rotation is observed.
It does not command a production outage, and it is not legal advice.

Ported from [rlaope/oh-my-hermes](https://github.com/rlaope/oh-my-hermes) (MIT), `omh-security-event-response`
and `omh-security-safety-review/references/credential-rotation.md`.

## When to Use

- "We committed an API key / pushed a `.env`, what now?"
- "Triage CVE-XXXX-YYYY / this GHSA / this Dependabot security alert against our repo."
- "Can we ship with this GPL/AGPL/unlicensed dependency?"
- A planned credential rotation (no exposure) — use `references/credential-rotation.md` alone.

Don't use for: a routine version bump with no advisory, forensics on a compromised upstream
repository (`oss-forensics`), or design-time threat modeling (`application-threat-model`).

## Prerequisites

None beyond the standard toolset. Evidence comes from the repo (`read_file` on lockfiles,
`search_files` for call sites), the advisory (`web_extract` on the GHSA/NVD page), and command output
the user or `terminal` produces (`git log`, `npm audit`, `pip-audit`, `osv-scanner`, an issuer's
CLI). Rotation and revocation run on the user's issuer accounts; do them only when the user asks
and has the access.

## Procedure

1. **Classify and record.** Event kind, source, affected package or credential class, and the
   exposure window — separately from suspected impact. Unknown window for a secret = exposed since
   its first push (`git log -S` on a *fingerprint* such as the key prefix, never the full value).
2. **Order containment** from `references/event-containment-order.md`:
   - **Leaked secret** (§1): replacement issued and deployed → old credential revoked → a call with
     the old credential *observed failing* → issuer access log audited for the window → history
     rewrite only if needed → prevention. Rotation always precedes any rewrite: a rewrite does not
     reach clones, forks, CI caches or scrapers that already read the push. Use
     `references/credential-rotation.md` for per-type sequences and the overlap window.
   - **CVE/advisory** (§2): installed version from the lockfile (every transitive path), then
     reachability — call sites that reach the vulnerable function or input, or the observed absence
     of any — then severity adjusted by that reachability, then upgrade/pin/override/disable.
   - **License** (§3): SPDX id from package metadata and the repo `LICENSE`, not a README badge;
     verdict fits / conflicts / needs counsel for the stated distribution model.
3. **Track states.** Each step is `prepared`, `observed` or `closed`. A command's exit code is not
   observation of its effect (a revoke that returned 0 has not proven the key dead).
4. **Close.** Closed only when every closing step is observed: old credential rejected, fixed version
   in the lockfile *and* the shipped build, or the license decision recorded. Otherwise name each
   step still prepared. A fix that needs a major upgrade gets its own plan; the event stays open with
   the workaround recorded as a workaround.

## Pitfalls

- **Never print, paste or search for the secret value** — reference it by type, prefix and location.
  Hermes masks secret-shaped tool output by default (`security.redact_secrets`), but a value typed
  into a command, a commit message or an issue is already out.
- **Force-push first is the classic mistake.** Every clone made before it keeps a live credential.
- **A compromised credential inverts the overlap rule:** revoke first and accept the outage.
- **Revocation proof is a failure with the old credential** (401/403), not a connection error, a 404
  or a still-successful call. Tokens minted before revocation may live to their expiry — say so.
- **Version-number safety is not reachability.** Cite the path or its observed absence; with neither,
  keep the advisory's severity and mark the adjustment unverified.

## Verification

- The event record has kind, source, window and affected asset.
- Every severity call cites a reachable path or an observed absence.
- In a secret plan, rotation and the old-credential rejection precede any history rewrite.
- The closure verdict lists each closing step with its observed evidence, or reads open.
