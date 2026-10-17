---
sidebar_position: 11
title: Model Catalog
description: Remotely-hosted manifest driving curated model picker lists for OpenRouter and Nous Portal.
---

# Model Catalog

Hermes fetches curated model lists for **OpenRouter** and **Nous Portal** from a JSON manifest hosted alongside the docs site. This lets maintainers update picker lists without shipping a new `hermes-agent` release.

When the manifest is unreachable (offline, network blocked, hosting failure), Hermes silently falls back to the in-repo snapshot that ships with the CLI. The manifest never breaks the picker — worst case you see whatever list was bundled with your installed version.

## Live manifest URL

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

Published on every merge to `main` via the existing `deploy-site.yml` GitHub Pages pipeline. The source of truth lives in the repo at `website/static/api/model-catalog.json`.

## Schema

```json
{
  "version": 1,
  "updated_at": "2026-04-25T22:00:00Z",
  "metadata": {},
  "providers": {
    "openrouter": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2",         "description": "default", "default": true},
        {"id": "moonshotai/kimi-k3",   "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2", "default": true},
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k3"}
      ]
    }
  }
}
```

Field notes:

- **`version`** — integer schema version. Future schemas bump this; Hermes refuses manifests with versions it doesn't understand and falls back to the hardcoded snapshot.
- **`metadata`** — free-form dict at the manifest, provider, and model level. Any keys. Hermes ignores unknown fields, so you can annotate entries (`"tier": "paid"`, `"tags": [...]`, etc.) without coordinating a schema change.
- **`description`** — OpenRouter-only. Drives picker badge text (`"recommended"`, `"free"`, `"default"`, or empty). Nous Portal doesn't use this.
- **`default`** — exactly one entry per provider may carry `"default": true`. That model is the **silent default**: what Hermes lands on when the user never selected a model (GUI onboarding confirm card, `provider` configured with no `model`, empty `model.default`). Read cache-only at runtime (`get_default_model_from_cache`) so hot resolution paths never hit the network; when no cached manifest exists, Hermes falls back to the in-repo `PREFERRED_SILENT_DEFAULT_MODEL` constant, which must match the labeled entry. This lets maintainers rotate the silent default without shipping a release. It is deliberately a capable low-cost model, never the priciest flagship.
- **Pricing and context length** are NOT in the manifest. Those come from live provider APIs (`/v1/models` endpoints, models.dev) at fetch time.

## Fetch behavior

| When | What happens |
|---|---|
| `/model` or `hermes model` | Fetches if disk cache is stale, else uses cache |
| Gateway running | Background refresh every `ttl_minutes` (default 20), so the picker never lags the published manifest by more than one window |
| Disk cache fresh (< TTL) | No network hit |
| Network failure with cache | Silent fallback to cache, one log line |
| Network failure, no cache | Silent fallback to in-repo snapshot |
| Manifest fails schema validation | Treated as unreachable |

Cache location: `~/.hermes/cache/model_catalog.json`.

### Per-provider model lists in the GUI picker

The Desktop, TUI and dashboard pickers (`model.options`) build each provider's row from the disk-cached live catalog (`~/.hermes/provider_models_cache.json`) or, when nothing is cached yet, the curated list. Opening the picker never waits on a provider's `/v1/models` probe or on an auth probe: stale or missing catalogs are refreshed in a background thread and land on the next open, so one slow, rate-limited or unreachable provider cannot hold the whole picker on its loading state. **Refresh models** (or `/model --refresh`) is the explicit action that busts the cache and probes every provider live.

## Config

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_minutes: 20
  providers: {}
```

Set `enabled: false` to disable remote fetch entirely and always use the in-repo snapshot (this also disables the gateway's background refresh). `ttl_minutes` sets both the cache lifetime and the gateway refresh cadence; the legacy `ttl_hours` key is still honoured if you set it explicitly.

### Per-provider override URLs

Third parties can self-host their own curation list using the same schema. Point a provider at a custom URL:

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

The overriding manifest only needs to populate the provider block(s) it cares about. Other providers continue to resolve against the master URL.

### Recipe: expose the full OpenRouter catalog in the picker

The curated list trades coverage for curation: on a typical day the OpenRouter picker shows the
curated entries that pass the live `tools` filter — a few dozen models out of roughly 450 that
OpenRouter serves, most of which advertise tool calling and are perfectly usable. If you would
rather browse (or search) the complete tool-capable catalog, override the `openrouter` block with
a **superset manifest** generated from the live catalog.

Generating the manifest is a small one-time script. It pulls OpenRouter's `/api/v1/models`, keeps
every text-output model that advertises `tools`, keeps the shipped curated ids at the head of the
list (so badges and the recommended entry keep their positions), appends everything else
newest-first, and writes a schema-valid manifest:

```python
# ~/.hermes/scripts/refresh_openrouter_catalog.py — run manually or on a schedule
import json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))
from hermes_cli.models import _OPENROUTER_CATALOG_URL, _fetch_live_catalog_index, _urlopen_model_catalog_request
from hermes_cli.models_catalog_static import OPENROUTER_MODELS

OUT = Path.home() / ".hermes" / "catalog" / "openrouter-full.json"

def usable(item):  # text output + tool calling, the picker's own hard requirements
    arch = item.get("architecture") or {}
    outs = arch.get("output_modalities")
    text_out = "text" in outs if isinstance(outs, list) else str(arch.get("modality", "")).endswith("->text")
    params = item.get("supported_parameters")
    tool_call = ("tools" in params) if isinstance(params, list) else True
    return text_out and tool_call

items, by_id = _fetch_live_catalog_index(str(_OPENROUTER_CATALOG_URL), 20.0, _urlopen_model_catalog_request)
head = [mid for mid, _ in OPENROUTER_MODELS if mid in by_id and usable(by_id[mid])]
rest = sorted((mid for mid, m in by_id.items() if mid not in set(head) and usable(m)),
              key=lambda mid: by_id[mid].get("created") or 0, reverse=True)

manifest = {"version": 1, "updated_at": "…", "metadata": {"source": "full-catalog override"},
            "providers": {"openrouter": {"models": [{"id": mid} for mid in head + rest]}}}
OUT.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(dir=str(OUT.parent)); os.close(fd)
Path(tmp).write_text(json.dumps(manifest, indent=1)); os.replace(tmp, OUT)
```

Wire the picker to it — `file://` URLs work, since the manifest fetch goes through `urllib`:

```yaml
model_catalog:
  providers:
    openrouter:
      url: file:///C:/Users/<you>/.hermes/catalog/openrouter-full.json   # POSIX: file:///home/<you>/...
```

Then run the script once, refresh the picker (/model, **Refresh models**, or `/model --refresh`),
and the interactive surfaces show the full tool-capable list, curated flagships first. Schedule
the script (cron, Task Scheduler, or a Hermes cron job) and new OpenRouter models appear on their
own within one refresh window. Uninstalling is a one-line config removal.

Why the manifest only ever needs to be a **superset**: `fetch_openrouter_models()` intersects the
override with the live catalog and re-applies the tool-support filter itself (see the live filter
described above), so retired or tool-less ids self-clean on every refresh — a stale override can
never surface a dead model. The overriding manifest deliberately replaces only the `openrouter`
block; Nous Portal and every other provider keep resolving against the master URL.

Caveats:

- **Messaging-gateway inline pickers** (Telegram/Discord `/model` grids) render at most ~50 rows
  regardless of the manifest, so the override buys the **interactive** surfaces: the CLI `/model`
  picker, the TUI picker, the Desktop picker, and the dashboard. Inline grids keep their curation
  cap by design; a model past the cap is still reachable as an explicit `/model <id>` command.
- The override re-fetched on every picker open (per-provider overrides skip the disk cache), so
  the file stays local and cheap to read.
- Models that do not advertise `tools` stay hidden even when listed — that filter is intentional
  (agents fail hard at the first tool call without it), and the override does not attempt to lift
  it.

For a maintained implementation of this recipe (ordering, atomic writes, offline-safe refresh,
cron wiring), see the community write-up: <https://github.com/donwrightdesigns/hermes-openrouter-full-picker>

### Hiding providers from the picker

`excluded_providers` lets you hide specific providers from the `/model` picker even when valid credentials exist. Useful when credentials are present for legacy or testing providers that shouldn't appear in normal use (e.g. an old Copilot or OpenRouter token still cached in `auth.json` or discovered via the `gh` CLI).

```yaml
model_catalog:
  excluded_providers:
    - copilot
    - openrouter
    - openai
```

The exclusion is matched case-insensitively against every key a provider can surface under — the Hermes id and models.dev id (built-in mapped providers), the overlay pid and resolved Hermes slug (overlay providers), and the canonical slug (canonical providers) — so a single entry like `copilot` hides the provider regardless of which section emits it. It is honored by every `/model` picker surface: the gateway interactive/text pickers, the TUI picker, and the interactive `hermes model` CLI picker. An empty list (or omitting the key) has no effect.

## Updating the manifest

Maintainers:

```bash
# Re-generate from the in-repo hardcoded lists (keeps manifest in sync after
# editing OPENROUTER_MODELS or _PROVIDER_MODELS["nous"] in hermes_cli/models.py).
python scripts/build_model_catalog.py
```

Then PR the resulting change to `website/static/api/model-catalog.json` to `main`. The docs site auto-deploys on merge and the new manifest is live within a few minutes.

You can also hand-edit the JSON directly for fine-grained metadata changes that don't belong in the in-repo snapshot — the generator script is a convenience, not the single source of truth.
