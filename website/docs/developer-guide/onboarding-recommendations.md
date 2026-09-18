---
title: Onboarding recommendations
sidebar_label: Onboarding recommendations
---

# Onboarding recommendations

Guided onboarding suggests outcomes from the user's app preferences and observed capabilities. It does not grant access or implement a second connection lifecycle. The feature remains inside the existing `HERMES_GUEST_ONBOARDING` flow.

## What takes priority

The existing `machineSetupLeads()` decision still leads with machine setup for a newly set-up machine or a recognized RTX/DGX Spark, with other tasks behind **Something else**. The fresh-machine signal is the existing age heuristic, not proof of the OS installation date. A Spark with unknown age gets a hardware-specific callout, not a claim that its OS is new.

Otherwise the guide favors relevant app-backed tasks, with a connection-free alternative. Detecting an app earns at most one option; the remaining choices come from the user's goals and other capabilities. Several ideas for the same app are appropriate only when explicitly requested. Choosing an inbox task means using real inbox data after permission, not building a mock inbox when permission is missing. A skipped or unavailable required connection leaves that task blocked; the user can choose a different task or supply data.

## Catalog metadata

Read the actual catalog at runtime. An entry does not need new recommendation metadata: detection falls back to its catalog name (with slug separators treated as spaces), and the model derives tasks from its description. Removing an entry removes it from future snapshots. No product-specific entry, package or install recipe belongs in the onboarding change.

Catalog owners may optionally provide richer hints:

```yaml
suggest:
  keywords: [example, modeling]
  hosts: [example.com]
  applications: [Example Studio]
  requires_app: true
  examples:
    - Light a product render in Example Studio
```

- `keywords` and `hosts` retain their existing meaning. `applications`, `examples` and `requires_app` are optional additions.
- `applications` contains up to 16 safe labels/aliases, each at most 80 characters. No paths, shell arguments, regexes or scripts.
- `examples` contains up to six printable single-line outcomes, at most 240 characters each. The model distills those into the existing option chips' shorter labels.
- `requires_app` is true only when the local application is a prerequisite. Desktop-app presence can be a relevance signal for a cloud service without being a requirement.
- Example capabilities and setup prerequisites must be checked against the integration's actual documentation. A catalog entry is not proof of account access, entitlement, add-on readiness or a live tool connection.

The managed-app picker keeps its curated leaders while making other enabled rows from the live catalog searchable. A newly deployed managed connector is no longer discarded merely because its slug is absent from the leader list. No new portal metadata endpoint is required.

## Discovery and scope

`GET /api/mcp/catalog?detect_apps=true` adds:

- `entries[].detected_apps`: matching catalog-derived application names or optional explicit aliases only.
- `discovery`: `{scope: "backend", status: "ok" | "unavailable", platform: string}`.

The default catalog request performs no app discovery. The optional scan checks standard application locations and exact safe PATH candidates on the **backend machine**, where its MCP processes run. It never launches apps, starts MCPs, installs packages, reads application documents or contacts the network. It returns neither a full inventory nor filesystem paths.

The scan has directory, entry and time budgets. It is not an exhaustive installed-software inventory. Access failures, an exhausted budget or an unsupported host report `unavailable`; positive observations remain useful, and missing observations are not evidence of absence. macOS app bundles, Linux desktop entries and Windows common program directories have platform-specific readers. Windows registry enumeration is not included.

Desktop pins the catalog request to the guide or handoff's backend/profile. It never combines a desktop-local app observation with a remote MCP host. A temporary-profile API probe can retain the actual OS home for read-only app discovery without reading the installed profile's configuration.

## Recommendation and execution boundaries

Recommendations rank explicit task relevance and selected apps before configuration and app-presence signals. Disabled integrations are not automatically resurfaced without explicit interest. Required local apps need positive detection or an existing configuration before they qualify. Each seed contains a small selection of candidates, not the entire catalog.

The guide receives a read-only snapshot when its session is created. The handoff refreshes that snapshot for the working profile and includes the selected candidates' full setup notes. Neither path mutates an existing system prompt, caches connection authority, or performs setup. An old or unavailable catalog falls back to the existing onboarding behavior.

| Evidence | Meaning | Next action |
| --- | --- | --- |
| App detected, MCP absent | Application signal, setup still needed | Existing `manage_connections` install approval |
| MCP configured but disabled, explicitly requested | Configuration exists, intentionally not active | Existing enable approval |
| MCP configured and enabled | Configuration only; connection unverified | Discover its available tools and verify before use |
| Managed app not connected | Account permission required | Existing managed connect card |
| Skipped, unavailable or missing session capability | Not authorized/usable | Explain the blocked task; no bypass or automatic retry |

For MCP targets, `status` is **not** a supported `manage_connections` action. The snapshot supplies `setupAction` (`install`, `enable`, or null); authorization is requested only when actually required. Managed account status remains a separate supported action. This distinction is verified against actual generated tool arguments, not only prompt text.

The legacy composer suggestion provider can only complete hosted HTTP OAuth. It excludes local-app and non-OAuth entries so new metadata cannot route a local editor into that unrelated flow. Onboarding uses the shared connection operation instead.

## Compatibility with the connection-operation work

The implementation consumes the public `manage_connections` tool and the existing catalog API. It does not modify connection-operation states, generated RPC contracts, account identifiers, OAuth callbacks, retry ownership, watcher behavior or settled-result handling.

It was checked against the current integration and Sid's pending [connection-operation PR](https://github.com/NousResearch/hermes-agent/pull/111008). That PR moves MCP execution into the backend; recommendations do not depend on which side currently executes the operation. Optional catalog fields preserve older clients, and missing fields preserve older backends. Sid's cancelled portal toolkit-metadata endpoint is not a dependency.

There is one intentional product-policy overlap in the onboarding runbooks and their tests: the old requirement to produce a no-account substitute is replaced by the accepted task's actual prerequisites. That overlap must be reconciled when the branches meet, not retained as contradictory instructions. Compatibility with an unknown future breaking API change is not implied.

## Verification boundaries

Contract tests cover catalogue parsing, opt-in read-only discovery, A/B/A profile isolation, missing metadata, ranking, backend-pinned seed creation, legal MCP setup actions and preservation of fresh-machine/Spark priority.

Live inference over synthetic onboarding turns can verify the recommended outcome and generated setup action. Native discovery can verify a real installed app signal. Neither is proof of OAuth completion or execution inside Blender: that still requires the application's add-on/server, the user's approval and a harmless live tool check. Read the actual returned entry's setup notes; onboarding does not supply its own integration recipe.
