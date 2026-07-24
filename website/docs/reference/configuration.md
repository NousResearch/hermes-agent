---
title: "Configuration Reference"
description: "Reference entries for Hermes config.yaml settings"
---

# Configuration Reference

Most user-facing configuration guidance lives in [User Guide → Configuration](/user-guide/configuration). This reference page records individual config keys that are useful to link from feature docs.

## `agent.startup_context_audit`

Controls the redacted startup context-budget audit.

```yaml
agent:
  startup_context_audit: off  # off | summary | status | debug_file
```

| Value | Effect |
|---|---|
| `off` | Default. No startup audit for normal sessions. |
| `status` | Collect once at startup and expose `/context-audit`; no prompt injection. |
| `summary` | Collect once and append a compact redacted summary to the first prompt. |
| `debug_file` | Collect once, expose `/context-audit`, and write redacted JSON under `~/.hermes/sessions/context_audits/`. |

The report contains labels, size counts, hashes, necessity ranks, and optimization hints. It does not dump raw prompt text, memory/profile content, secrets, or full tool schemas.

Recommendations are advisory. Review and apply any runtime config changes manually, especially for running gateway/profile setups.
