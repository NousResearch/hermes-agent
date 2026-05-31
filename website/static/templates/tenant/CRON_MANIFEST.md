# Cron Manifest

Document tenant-owned cron jobs here. Keep one section per job.

## `<job-name>`

- Status: `draft | active | paused | retired`
- Owner: `<tenant-name>`
- Schedule: `<cron expression or natural schedule>`
- Profile: `~/.hermes/profiles/<tenant-name>/`
- Workspace path: `cron/<job-name>/`
- Script/entrypoint: `<path>`
- Delivery target: `<platform/channel/user/file>`
- Data sources: `<files/APIs/services>`
- Writes data: `yes/no`
- Sends external messages: `yes/no`
- Risk level: `low | medium | high`
- Agent reasoning required: `yes/no`

### Purpose

What this job does and why it exists.

### Expected output

Describe the normal output shape. If nothing changed, say whether the job should stay silent.

### Authorization

Who approved this job and what resources it is allowed to access.

### Verification

```bash
# command to run a one-shot smoke test
```

Expected result:

```text
<expected safe output>
```

### Rollback / pause

```bash
# command or operator steps to pause/remove/restore this job
```

### Notes

Record known failure modes, rate limits, and privacy constraints.
