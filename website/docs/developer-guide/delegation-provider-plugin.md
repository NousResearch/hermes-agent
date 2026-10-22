---
title: Delegation Provider Plugins
sidebar_label: Delegation provider plugins
---

# Delegation Provider Plugins

A delegation provider decides **how each `delegate_task` task becomes a child run**: which model,
provider, endpoint and reasoning effort the subagent uses, or which Hermes profile it runs as. It can
also add per-task parameters to the `delegate_task` schema, such as an enum of profile names or of
titled model options, so the model picks a route per task.

Hermes core keeps the whole subagent lifecycle: spawning, isolation, leaf/orchestrator roles, depth
limits, concurrency, background completion units, stop fan-out, result assembly, `output_schema`
validation and progress. A provider only answers "which route does this task take".

The active provider is `delegation.backend` in `config.yaml`. The built-in `default` strategy is
today's behaviour: every child follows the `delegation` section (`delegation.provider`,
`delegation.model`, `delegation.base_url`, ...), or inherits the parent's route when those are empty.
A configured backend that is not installed or reports unavailable falls back to `default` with one
warning, and its schema parameters disappear with it.

## Skeleton

```python
# ~/.hermes/plugins/delegation-tiers/__init__.py
from agent.delegation_provider import DelegationProvider, DelegationRouteError


class Tiers(DelegationProvider):
    @property
    def name(self) -> str:
        return "tiers"                      # delegation.backend: tiers

    def task_parameters(self, config):
        tiers = config.get("tiers") or []  # your own delegation.<key> settings
        return {"tier": {
            "type": "string",
            "enum": [t["name"] for t in tiers],
            "description": "; ".join(f"{t['name']}: {t['description']}" for t in tiers),
        }}

    def description(self, config):
        return "Set `tier` per task to pick the model that fits it."

    def route_task(self, task, config):
        for t in config.get("tiers") or []:
            if t["name"] == task.get("tier"):
                return {"provider": t["provider"], "model": t["model"], "label": t["name"]}
        if task.get("tier"):
            raise DelegationRouteError(f"Unknown tier {task['tier']!r}.")
        return dict(config)                 # no tier: the default route


def register(ctx):
    ctx.register_delegation_provider(Tiers())
```

```yaml
# plugin.yaml
name: delegation-tiers
version: 1.0.0
description: Pick a model tier per subagent.
```

```bash
hermes plugins enable delegation-tiers
hermes config set delegation.backend tiers
```

## The interface

`DelegationProvider` (`agent/delegation_provider.py`) subclasses the shared `ProviderBase`. Every
method receives `config`, the `delegation` section of the active profile's `config.yaml`, so provider
settings live next to the other delegation knobs.

| Member | Required | Contract |
|---|---|---|
| `name` | yes | The `delegation.backend` value. `default` is reserved for the built-in strategy. |
| `route_task(task, config)` | yes | Return the route dict for one task. `task` carries `goal`, `context`, `index` and the values the model supplied for your parameters. Raise `DelegationRouteError` to refuse; its message is the tool error the model reads. |
| `task_parameters(config)` | no | `{name: json_schema}` merged into `tasks.items.properties`. Core enforces `enum` before `route_task` runs and refuses core-owned names (`goal`, `context`, `index`, `output_schema`, `images`, `group`, `role`, `toolsets`, `provider`, `model`, `base_url`, `api_key`, `api_mode`, `acp_command`, `acp_args`). |
| `description(config)` | no | Guidance appended to the `delegate_task` description. |
| `is_available()` | no | No network calls. `False` makes delegation use `default`. |

### Routes

A route is a dict with any of:

- `model`, `provider`, `base_url`, `api_key`, `api_mode`, `request_overrides`, `reasoning_effort`,
  `fallback_providers`: shaped exactly like the `delegation` section and resolved by the same
  credential resolver. An empty route inherits the parent's route; `dict(config)` is the default route.
- `profile`: the name of a Hermes profile. The child runs **as that profile**: its `config.yaml`,
  primary model and credentials (a `model.default` alias or named provider resolves exactly as at
  CLI startup), credential pool, fallback chain, reasoning (`agent.reasoning_effort` and
  `agent.reasoning_overrides`), skills, `SOUL.md` and terminal policy. Core binds that profile's
  scope (HERMES_HOME override, secret scope, multiplex semantics, terminal policy) around the child's
  construction, its credential lease and every conversation turn, the same binding a multiplexed
  gateway applies to a routed turn, so a secret missing from the target profile is never read from
  the launching profile's environment. The child's transcript stays in the launching profile's
  session database (it is the parent's work) and it keeps the parent's toolset. Other route keys are
  ignored on a profile route.
- `label`: short text reported as `route` on the child's result (an option title, a profile name).

Unknown keys are dropped.

## Prompt caching

The `delegate_task` definition, including your parameters and description, is built once when a
session's tool list is assembled. `task_parameters` and `description` must be deterministic for a
given `config`: same config, same bytes. Never derive them from the clock, the network or anything
that changes during a conversation.

## Process isolation

Every argument and return value is plain JSON, so a provider runs unchanged under
`plugins.isolation: host`: Hermes holds a proxy that subclasses `DelegationProvider` and every call
runs in the plugin host process.

## Testing

Load the plugin through the real discovery path against a temporary `HERMES_HOME` whose
`config.yaml` sets `plugins.enabled` and `delegation.backend`, then assert on behaviour: the
`delegate_task` definition from `tools.registry.registry.get_definitions({"delegate_task"})` carries
your parameter, and `tools.delegate_tool.delegate_task(...)` hands each child the route you chose.
`tests/tools/test_delegation_backends.py` in the Hermes repo is a complete example.
