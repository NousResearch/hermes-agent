---
title: "Security Assurance — Review MDM, SBOM, supply-chain, PPAP, and device evidence"
sidebar_label: "Security Assurance"
description: "Review MDM, SBOM, supply-chain, PPAP, and device evidence"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Security Assurance

Review MDM, SBOM, supply-chain, PPAP, and device evidence.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/security/security-assurance` |
| Version | `0.1.0` |
| Author | jun (junkawasaki), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `security`, `evidence`, `mdm`, `sbom`, `inventory` |
| Related skills | [`oss-forensics`](../../optional/security/security-oss-forensics.md), [`security-records`](../../bundled/security/security-security-records.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Security Assurance Skill

Review operator-reported evidence for the Mithril security app: mobile MDM/MTD, SBOM, SASE, ICT supply chain, PPAP replacement, and device roster reconciliation. The script records the same decisions as the app.mithril.fund views. It does not enroll devices, send files, query a live registry, or issue a compliance certificate.

## When to Use

- The user asks to review MDM, MTD, SBOM, SASE, supply-chain, PPAP, or device-inventory evidence.
- The user wants the app.mithril.fund assurance, mobile, or operations review run from the agent.

Don't use for:

- Device enrollment, remote commands, live MTD sensors, or an authorization decision.
- Uploading a file, sending a link, or checking a password.
- Querying an SBOM registry or operating a SASE edge.
- Investigating a compromised repository. Use `oss-forensics` for that.

## Prerequisites

- Python 3 on `PATH`. The standard library is enough.
- Evidence the operator already has. Leave a control `unknown` when they did not report it.

## How to Run

Write a JSON request with `write_file`, then run this skill's script through `terminal`. Do not put file bytes, passwords, addresses, or links in the JSON.

```
terminal(command="python skills/security/security-assurance/scripts/review.py controls mobile --view mtd --platform ios")
terminal(command="python skills/security/security-assurance/scripts/review.py review --input request.json")
```

When this skill is installed outside the repo, run `scripts/review.py` from the skill directory instead of the repo-relative path.

Request shapes:

| surface | required fields |
|---|---|
| `transfer` | `method`: `password-zip`, `email-attachment`, or `managed-link`. `answers`: `yes`, `no`, or `unknown` per control id |
| `mobile` | `view`: `mdm` or `mtd`. MTD also needs `platform`: `ios` or `android`. `answers`: `observed`, `missing`, or `unknown` |
| `assurance` | `view`: `sbom`, `sase`, or `scrm`. Same answer tokens as mobile |
| `devices` | `roster` (`id`, `platform`, optional `owner`) and `reports` (`id`, `platform`, `reported_at`). `now` and `max_age_ms` are epoch milliseconds. `reported-at` is accepted |

`controls <surface>` prints the control ids before you ask for answers.

## Quick Reference

| Result | Meaning |
|---|---|
| `replace-ppap` | Password ZIP plus a separate password mail is rejected even when every other answer is yes |
| `review-attachment` | An ordinary email attachment stays in manual review |
| `planned-only` | A managed link whose six settings are all `yes`. This does not mean a provider enforced them |
| `incomplete-plan` | A managed link still has `no` or `unknown` |
| `unmeasured` | Blank roster or blank reports. Coverage is null |
| `self-reported-only` | Every selected control was reported present. This is not a certificate |
| `insufficient-evidence` | At least one control is still unknown and none is a reported gap |
| `reported-finding` / `reported-gap` | The operator reported a missing control |

A device report whose age is exactly `max_age_ms` is current. Only an older report is `stale`. Supported report platforms are `macos`, `ios`, `ipados`, `android`, `windows`, and `linux`.

## Procedure

1. Choose one surface from the request. Completion: the surface is `transfer`, `mobile`, `assurance`, or `devices`.
2. Run `controls` for that surface and collect only the answers the operator supplies. Completion: every unstated control stays `unknown`, and no file content was requested.
3. Write `request.json` and run `review --input`. Completion: the process prints one JSON object and exits 0.
4. Report `decision` or `conclusion`, the counts, and each `unmeasured` or gap row with its source. Completion: `self-reported-only` and `planned-only` are described as self-report, and no enrollment, transfer, or command was attempted.
5. For a supply-chain gap, include that control's `next_step` as a prompt for the operator. Completion: the prompt is the script's `next_step` text.

## Pitfalls

- Password ZIP stays `replace-ppap` when the other answers are all yes.
- An empty roster or an empty report list is `unmeasured`, not a fleet of zero devices.
- MTD accepts only `ios` and `android`. MDM does not take a platform.
- All-`observed` evidence ends as `self-reported-only`. Do not upgrade that to compliant, authorized, or enforced.
- CISA BOD 23-01's seven-day cadence is a US federal civilian scope. This review uses `max_age_ms` only as the operator's age threshold.
- Exit code 2 and an `error` field mean the input was refused. Fix the request. Do not invent a result.

## Verification

Run the same `request.json` again and confirm the decision, counts, and row statuses match. For a code change to this skill, `tests/skills/test_security_assurance_skill.py` must pass.
