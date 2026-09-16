---
title: "Security Review — Use when reviewing code or designs for security."
sidebar_label: "Security Review"
description: "Use when reviewing code or designs for security"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Security Review

Use when reviewing code or designs for security.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/security/security-review` |
| Path | `optional-skills/security/security-review` |
| Version | `0.1.0` |
| Author | Thomas Yiu, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `security`, `code-review`, `secure-design`, `audit` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered.
:::

# Security Review Skill

Perform a risk-prioritized, evidence-based security review of code, configuration, infrastructure, APIs, or design documents. Report confirmed findings separately from hypotheses and keep testing within the user's authorized scope.

## When to Use

- Reviewing a diff, repository, API, deployment, cloud configuration, or security-sensitive design.
- Looking for authentication, authorization, input-validation, secret-handling, data-protection, or supply-chain weaknesses.
- Preparing remediation guidance or release-gate findings.

Don't use for: unauthorized probing, exploitation, persistence, evasion, or handling exposed credentials.

## Prerequisites

- Review target and intended scope; preserve the repository's current state.
- Threat model or security requirements when available.
- Test commands and environment constraints from project documentation.

## Procedure

1. Establish scope, trust assumptions, sensitive assets, entry points, and review baseline. Completion criterion: target files/components and exclusions are listed.
2. Inspect structure and change surface before deep review using `search_files`, `read_file`, and `terminal` as appropriate. Completion criterion: all relevant changed or exposed components are accounted for.
3. Review identity and access control: authentication, authorization at every sensitive operation, tenant isolation, session lifecycle, privilege boundaries, and fail-closed behavior. Completion criterion: each sensitive action has an identified enforcement point.
4. Review data handling: validation, canonicalization, injection resistance, serialization, cryptography and key management, secrets, logging, privacy, and error disclosure. Completion criterion: each external input and sensitive output has a documented control assessment.
5. Review resilience and supply chain: rate limits, resource exhaustion, dependency versions, unsafe defaults, update provenance, deserialization, filesystem/process use, and security headers or network policy where relevant. Completion criterion: material abuse paths and dependencies are covered.
6. Validate findings with the least-invasive available evidence. Prefer static inspection and existing tests; do not run intrusive tests without explicit authorization. Completion criterion: each finding cites file/line or component evidence and a reproducible reasoning path.
7. Rank findings by exploitability, impact, exposure, and confidence. Completion criterion: every finding has severity, confidence, affected scope, remediation, and verification advice.

## Output Format

For each finding provide:

- ID and title
- Severity and confidence
- Affected component and exact evidence
- Security impact
- Preconditions and attack path, described defensively
- Remediation and safer alternative
- Regression test or verification step

End with: reviewed scope, clean areas, assumptions, and follow-up checks.

## Pitfalls

- Do not call a pattern a vulnerability without tracing data flow and authorization context.
- Do not report scanner output without manual triage and affected-version evidence.
- Do not include secrets, tokens, or personal data in findings; redact them.
- Avoid broad rewrites when a narrowly scoped control fixes the issue.

## Verification

Ensure every finding is actionable, deduplicated, evidence-backed, and mapped to a test or acceptance criterion. Re-run relevant tests or static checks after remediation and confirm the reported issue is no longer present.