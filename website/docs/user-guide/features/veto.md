---
title: Veto Policy Guard
---

Hermes can run terminal commands, edit files, browse websites, create cron jobs,
send messages, and delegate work to subagents. Veto adds a policy checkpoint
before those tool calls execute.

The bundled `veto` plugin uses Hermes' `pre_tool_call` hook for clear user-facing
blocks, and Hermes also checks Veto at the final tool registry dispatch layer so
direct internal dispatch cannot bypass policy.

## Install

Install the Python Veto SDK in the same environment as Hermes:

```bash
pip install veto
```

Then enable the bundled plugin:

```bash
hermes plugins enable veto
```

You can check the runtime state from Hermes:

```bash
/veto-status
```

## Configuration

Add a `veto` section to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - veto

veto:
  enabled: true
  config_dir: ~/.hermes/veto
  validation_mode: local
  mode: strict
  fail_open: false
```

If `config_dir` does not exist and `validation_mode` is `local`, Hermes uses the
bundled Veto defaults. Those defaults block high-risk terminal commands, writes
to common secret paths, and browser navigation to private network targets. They
also require approval before creating cron jobs or sending cross-platform
messages.

## Local Policy Example

Create a local policy directory:

```bash
mkdir -p ~/.hermes/veto/rules
```

Write `~/.hermes/veto/veto.config.yaml`:

```yaml
mode: strict
validation:
  mode: local
rules:
  directory: rules
  recursive: true
```

Write `~/.hermes/veto/rules/block-rm.yaml`:

```yaml
rules:
  - id: block-rm-rf
    name: Block recursive deletes
    enabled: true
    severity: critical
    action: block
    tools: [terminal]
    conditions:
      - field: arguments.command
        operator: matches
        value: "rm\\s+-[^\\n]*(r[^\\n]*f|f[^\\n]*r)"
```

Now a Hermes tool call like this is denied before the terminal tool runs:

```json
{
  "tool": "terminal",
  "arguments": {
    "command": "rm -rf /tmp/demo"
  }
}
```

Hermes returns a tool error similar to:

```text
Veto blocked terminal: rule=block-rm-rf
```

## Cloud or Self-hosted Veto

For a remote Veto PDP, switch validation mode and provide the endpoint/API key:

```yaml
veto:
  enabled: true
  validation_mode: cloud
  endpoint: http://localhost:8001
  api_key_env: VETO_API_KEY
  fail_open: false
```

Environment overrides are available for containerized deployments:

```bash
export HERMES_VETO_ENABLED=1
export HERMES_VETO_VALIDATION_MODE=cloud
export HERMES_VETO_CONFIG_DIR=/data/veto
export VETO_ENDPOINT=http://veto-server:8001
export VETO_API_KEY=...
```

Keep `fail_open: false` for production. Use `fail_open: true` only for observe
or development runs where Hermes should continue if Veto is temporarily
unavailable.
