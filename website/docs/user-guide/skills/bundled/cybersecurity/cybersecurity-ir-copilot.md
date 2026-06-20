---
title: "Ir Copilot — Structured incident response copilot workflow for triage, containment, recovery, and reporting"
sidebar_label: "Ir Copilot"
description: "Structured incident response copilot workflow for triage, containment, recovery, and reporting"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ir Copilot

Structured incident response copilot workflow for triage, containment, recovery, and reporting.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/cybersecurity/ir-copilot` |
| Version | `1.0.0` |
| Tags | `cybersecurity`, `incident-response`, `soc`, `playbook` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Incident Response Co-pilot

Guides an analyst through a structured IR lifecycle using the `ir_incident` playbook tool, from initial detection through containment to post-incident report.

## When to use

- On-call analyst receives a security alert and needs structured tracking
- SOC team opens a new incident channel and wants a living incident record
- Post-incident: generate a formatted report from the session evidence log

## IR Lifecycle

### 1. Triage (≤ 15 min)

```
ir_incident(action="create", title="<alert title>", severity="P2", description="<context>")
```

Immediately after:
- Call `extract_iocs` on the alert body to pull any embedded indicators
- Add each IOC as evidence: `ir_incident(action="add_evidence", evidence_type="ioc", value=<ioc>, source="alert")`
- Add a timeline entry: `ir_incident(action="add_timeline", event="Incident opened — initial triage started", actor=<analyst>)`

### 2. Investigation

For each IOC extracted:
- Run `threat_intel(action="ioc_reputation", ...)` and add result as a note
- Run `vuln_triage` for any CVEs in scope
- Record every finding as a timeline event with timestamps
- Add log snippets as `evidence_type="log"` entries

Use `delegate_task` to run parallel subagents:
- Subagent A: SIEM queries / log review
- Subagent B: EDR telemetry / endpoint artefacts
- Subagent C: External TI enrichment

### 3. Containment

```
ir_incident(action="update", status="contained", ...)
ir_incident(action="add_timeline", event="<host> isolated from network", actor=<analyst>)
```

### 4. Eradication & Recovery

Continue updating timeline with remediation steps. Update status to `eradicated` then `closed`.

### 5. Report

```
ir_incident(action="report", incident_id=<id>)
```

Copy the markdown output into your ticketing system or post to the incident channel via `send_message`.

## Severity guide

| Label | Criteria |
|-------|----------|
| P1    | Active ransomware, data exfiltration in progress, crown-jewel compromise |
| P2    | Confirmed malware C2, lateral movement detected, privileged account compromise |
| P3    | Suspicious activity under investigation, phishing with no confirmed payload |
| P4    | Low-confidence alert, informational, false positive candidate |

## Notes

- The IR store lives in-session. Use `memory` tool to persist key facts across sessions.
- All subagent-demoted follow-ups are queued (see #30170 protection) — you won't accidentally kill a running investigation subagent by sending a follow-up.
- Enable `HERMES_CYBER_AUDIT=true` to get a tamper-evident NDJSON log of every action for post-incident review.
