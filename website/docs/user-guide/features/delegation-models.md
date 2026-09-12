---
title: Delegation Models and Backups
description: Choose persistent delegation defaults and a separate ordered fallback chain in Desktop.
---

# Delegation models and backups

In **Desktop → Settings → Models → Delegation models**, choose a default
provider and model for newly delegated work, followed by the backups to try
when a model request reaches the existing provider-recovery path.

The selectors use the same provider/model catalog as the other model controls.
A model ID not in the catalog can be entered manually. Leaving the provider
blank means **Use main provider**; a non-empty model with a blank provider is
an independent model-only override. **Use main model** resets both fields.
The picker persists empty strings for inheritance, not a literal `auto` model.

Choose **Apply delegation settings** after completing the selection. Provider
changes and incomplete fallback rows remain local drafts until then. The save
writes only the changed `delegation` keys and verifies them by reading the same
profile back. Changing a delegation setting does not change the primary
conversation's model or top-level fallback chain. Existing delegated workers
are not rerouted; the defaults apply when new children are constructed.

The settings profile selector controls the configuration being edited. Its
provider catalog, configuration read, save, and read-back use the same gateway
and profile scope. A profile switch discards uncommitted drafts. A failed save
retains edits for retry; an external delegation-config change requires an
explicit reload before applying a stale draft.

## Fallback behavior

| Selection | Stored value | Behavior |
| --- | --- | --- |
| Default behavior | Key absent or `null` | Unpinned children inherit the parent's chain. Explicit provider, endpoint, or model-only pins do not silently inherit it. |
| Use main fallbacks | `"inherit"` | Explicitly share the parent's chain, including with a pinned delegation model. |
| No fallbacks | `[]` | Disable fallback for delegated children. |
| Choose delegation fallbacks | List of provider/model entries | Use precisely this ordered, delegation-scoped chain, including with a pin. |

Add, remove, or move backup rows on the same screen. Each row keeps its route
metadata when moved or when only its model changes. Replacing a row's provider
clears that row's old endpoint/credential overrides, so the old route cannot
silently win over the new provider.

The selected default is a persistent preference, not a source-code constant.
Recovery uses the existing child `AIAgent` fallback machinery; this feature
adds neither a second retry loop nor automatic rewrites of your saved default.
The runtime fallback contract is owned by merged
[#105347](https://github.com/NousResearch/hermes-agent/pull/105347); the Desktop
editor only projects and persists that contract.
Failure classification, retry budgets, and terminal errors remain those of the
existing provider recovery system. Failure to resolve the initial delegation
credentials still fails preflight; a fallback chain does not override that
preflight authorization/transport check.

## Configuration

The same settings can be stored in the selected profile's `config.yaml`:

```yaml
delegation:
  provider: custom:workers
  model: worker-primary
  fallback_providers:
    - provider: custom:worker-backup
      model: worker-backup
    - provider: custom:local-worker
      model: local-worker
```

Provider names and model IDs above are examples for configured custom
providers, not built-in model defaults. Entries use the shared top-level
fallback entry format, including optional `base_url`, `key_env`, `api_mode`,
and other supported route metadata. Credentials continue to resolve through
the provider and profile's existing credential machinery.

Canonical `delegation.fallback_providers` takes precedence. For read
compatibility, `delegation.fallback_chain` and then
`delegation.fallback_model` are consulted only when the canonical value is
absent or `null`. A canonical `[]` never resurrects an alias. The explicit
`"parent"` string is accepted as an alias for `"inherit"`. Editing fallback
behavior in Desktop writes the canonical key and clears the old aliases, so
resetting to default cannot bring an old backup back accidentally.

A malformed declaration is not partially applied: it warns and retains the
pin-aware default from the existing fallback-matrix work. Thus a pinned child
gets no implicit parent chain on a configuration error. The editor requires a
valid policy and complete backup pairs before applying.

## Existing direct endpoints

A pre-existing `delegation.base_url` override is shown as **Existing direct
endpoint** rather than mislabelled as the main provider. Editing only its model
preserves the endpoint. Choosing a provider or **Use main model** explicitly
clears the old delegation endpoint, inline credential, wire-mode, and request
overrides so that the selected provider's configuration becomes authoritative.
Other delegation settings, including reasoning effort and iteration limits,
are preserved. The old Advanced delegation-model row opens the same scoped editor instead
of retaining a second, stale pair of raw inputs. The Models page adds a
discoverable control rather than moving runtime behavior into the renderer.

The control checks the selected backend's config-schema capability before
allowing edits. An older backend without delegation-scoped fallback support
shows an update notice instead of saving inert settings. Existing settings
are unchanged. An unavailable capability response shows a retry, not a false
unsupported verdict. A successful config save alone does not establish runtime
support: older backends can persist unknown keys without using them.

## Contribution lineage

This integration adapts webtecnica's [guided Desktop picker, #67523](https://github.com/NousResearch/hermes-agent/pull/67523)
and projects the runtime contract now owned by kshitijk4poor's merged
[#105347](https://github.com/NousResearch/hermes-agent/pull/105347). That runtime
work salvaged Ayush Nangia's [pin × config matrix, #80479](https://github.com/NousResearch/hermes-agent/pull/80479),
including spfcraze's model-only-pin correction, and builds on Ayush Nangia's
#65052, Axl Ibiza's #80421, wz-heng's #80438, and Teknium's #80465 pin
protection. Explicit parent inheritance follows devatnull's #81072; alias
compatibility follows TurgutKural's #101017 without adopting its conflicting
empty-list default. Reports and product requirements came from DavidMetcalfe
(#67347), mlahatte (#65038), and ScotterMonk (#94629).

The separate Dashboard part of #67347/#67557 is not implemented by this
Desktop integration. Original PRs retain their discussion, authorship, and
review reservations; this document does not declare them merged or closed.
