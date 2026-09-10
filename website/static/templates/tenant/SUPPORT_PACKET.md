# Support Packet

Use this when a tenant asks an operator/admin agent for help without opening the whole profile.

## Summary

One or two sentences describing the issue.

## Affected tenant

- Tenant: `<tenant-name>`
- Profile: `~/.hermes/profiles/<tenant-name>/`
- Workspace: `~/.hermes/profiles/<tenant-name>/workspace/`

## Affected component

- Type: `cron | skill | script | gateway | tool | config | other`
- Name: `<component-name>`
- Path: `<relative path inside workspace/profile, if authorized>`

## Expected behavior

What should have happened?

## Actual behavior

What happened instead?

## Sanitized errors

```text
<paste sanitized errors here>
```

Do not include secrets, raw customer data, private memory, or full session transcripts.

## Already checked

- [ ] Profile exists
- [ ] Workspace exists
- [ ] Relevant file exists
- [ ] Credentials were not pasted into this packet
- [ ] Cron/gateway status checked if relevant

## Authorized review scope

List exactly what the operator/admin may inspect.

- Allowed: `<specific files, commands, or logs>`
- Excluded: `memory/, sessions/, unrelated skills/, credentials, raw customer data`

## Desired outcome

What help is requested?

- [ ] Diagnose only
- [ ] Propose fix
- [ ] Apply fix in tenant workspace
- [ ] Create PR/patch
- [ ] Other: `<describe>`
