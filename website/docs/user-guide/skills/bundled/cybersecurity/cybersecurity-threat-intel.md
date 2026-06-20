---
title: "Threat Intel — Enrich IOCs and CVEs into concise, actionable threat intelligence briefs using Cyber Edition tools"
sidebar_label: "Threat Intel"
description: "Enrich IOCs and CVEs into concise, actionable threat intelligence briefs using Cyber Edition tools"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Threat Intel

Enrich IOCs and CVEs into concise, actionable threat intelligence briefs using Cyber Edition tools.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/cybersecurity/threat-intel` |
| Version | `1.0.0` |
| Tags | `cybersecurity`, `threat-intelligence`, `ioc`, `cve` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Threat Intelligence Synthesis

Enriches raw indicators and CVEs into a concise, actionable intelligence brief using the built-in cyber tools.

## When to use

- Analyst pastes a list of IPs, hashes, domains, or CVE IDs and asks "what are these?"
- Incoming alert contains suspicious artefacts that need rapid context
- Security team needs a TI brief before a patching call

## Workflow

1. **Extract IOCs** from any freeform input with `extract_iocs`
2. **Enrich in parallel** using `delegate_task` — one subagent per IOC type:
   - IP/domain/hash → `threat_intel` with `ioc_reputation`
   - CVE IDs → `threat_intel` with `cve_lookup`
3. **Map to ATT&CK** — for each malware family or tactic mentioned, call `threat_intel` with `mitre_ttp`
4. **Synthesise** the subagent results into a structured brief:
   - Executive summary (2–3 sentences)
   - IOC table (indicator | type | verdict | source)
   - ATT&CK TTP table (technique ID | name | tactic | detection hint)
   - Recommended actions (block list entries, hunt queries, patch priority)

## Output format

```
## Threat Intelligence Brief — <date>

**Summary:** <2–3 sentences>

### IOC Verdicts
| Indicator | Type | Verdict | Score | Source |
|-----------|------|---------|-------|--------|
...

### ATT&CK Coverage
| Technique | Name | Tactic | Detection |
|-----------|------|--------|-----------|
...

### Recommended Actions
1. ...
```

## Notes

- Call `vuln_triage` for any CVEs before making patch recommendations — EPSS score changes the priority.
- AbuseIPDB requires `ABUSEIPDB_API_KEY`; VirusTotal requires `VT_API_KEY`. Without these, CVE and ATT&CK lookups still work (NVD and TAXII are free).
- For large IOC lists (>20 indicators), fan out with `delegate_task` into batches of 5 to stay within API rate limits.
