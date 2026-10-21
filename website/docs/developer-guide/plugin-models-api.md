---
title: Plugin Model Catalog
sidebar_label: Model catalog (ctx.models)
---

# Plugin Model Catalog

`ctx.models` is the read-only model catalog for plugins that let a user pick a provider and model
(saved subagent routes, per-command model overrides, a settings form). It answers the same questions
the `hermes model` picker answers, from the same sources, so a plugin does not need to import private
`hermes_cli.models` helpers that can move or change on any release.

| Method | Returns |
|---|---|
| `ctx.models.providers()` | Provider ids the `hermes model` picker offers, plus `"custom"`. |
| `ctx.models.ready(provider)` | `True` when this profile has usable credentials for `provider`. |
| `ctx.models.list(provider)` | Model ids `provider` lists (the picker's disk-cached catalog). `[]` when Hermes cannot list that provider's models, for example a custom endpoint. |
| `ctx.models.parse(spec)` | `("provider", "model")` for `"provider:model"`. Aliases resolve to the canonical id (`deep-seek:...` gives `deepseek`). Raises `ValueError` for a bare model, an empty side or an unknown provider. |

```python
import difflib

def validate_route(ctx, spec: str) -> tuple[str, str]:
    provider, model = ctx.models.parse(spec)  # ValueError: not provider:model
    if not ctx.models.ready(provider):
        raise ValueError(f"No credentials for {provider!r} in this profile. Run `hermes auth` first.")
    known = ctx.models.list(provider)
    if known and model not in known:
        close = difflib.get_close_matches(model, known, n=3)
        raise ValueError(f"{provider!r} does not list {model!r}. Close matches: {', '.join(close)}")
    return provider, model
```

## Profile scope

Every call answers for the profile that loaded the plugin, not for whichever profile happens to be
active in the process. A gateway serving several profiles gives each profile's plugin its own answer:
`ready()` reads that profile's `.env`, secret sources and credential pool, and never a key another
profile put in the process environment. You do not need to bind anything yourself.

## Cost

`providers()`, `ready()` and `parse()` read local config and credentials. `list()` returns the
cached catalog when one is fresh and may fetch from the provider on a cold cache, so call it when you
need to validate a model, not on every hook.

`ctx.models` also works under `plugins.isolation: host`; each call runs in Hermes. Across that boundary
`parse()` arrives as a two-item list, which unpacks the same way.
