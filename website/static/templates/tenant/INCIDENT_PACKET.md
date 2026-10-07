# Incident Packet

Use this after a tenant automation caused or may have caused harm, noise, data exposure, bad writes, or unexpected external actions.

## Incident summary

What happened?

## Impact

- Tenant: `<tenant-name>`
- Started at: `<timestamp>`
- Ended at: `<timestamp or ongoing>`
- Affected users/systems: `<summary>`
- Data exposure suspected: `yes/no/unknown`
- External messages sent: `yes/no/unknown`
- Data written or deleted: `yes/no/unknown`

## Immediate containment

What was stopped, paused, revoked, or rolled back?

- [ ] Cron paused
- [ ] Gateway stopped
- [ ] Tool disabled
- [ ] Credentials rotated/revoked
- [ ] External messages corrected
- [ ] Other: `<describe>`

## Relevant component

- Type: `cron | skill | script | gateway | tool | config | other`
- Name: `<component-name>`
- Path: `<relative path, if safe and authorized>`

## Sanitized timeline

```text
<timestamp> <event>
<timestamp> <event>
```

Do not include secrets, raw private messages, customer records, or full session transcripts unless explicitly authorized for a narrow review.

## Suspected cause

What likely caused the incident?

## Authorized review scope

- Allowed: `<specific files, commands, or logs>`
- Excluded: `memory/, sessions/, unrelated skills/, credentials, raw customer data`

## Recovery plan

1. `<step>`
2. `<step>`
3. `<step>`

## Verification before resume

```bash
# smoke test or validation command
```

Expected result:

```text
<safe expected output>
```

## Follow-up actions

- [ ] Add regression test or guardrail
- [ ] Update cron manifest
- [ ] Update support docs
- [ ] Rotate credentials if needed
- [ ] Notify affected parties if needed
