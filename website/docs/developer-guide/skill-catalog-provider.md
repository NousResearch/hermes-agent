---
sidebar_label: "Skill catalog providers"
title: "Plugin-served skill catalogs"
description: "Expose a plugin catalog through the Skills HTTP API with provider-owned review, installation and recovery"
---

# Plugin-served skill catalogs

:::caution Experimental
`ctx.register_skill_catalog`, `SkillCatalogProvider` and `ctx.skill_catalog_mutations` are experimental. Signatures, wire fields and registration/mutation handle shapes may change. Keep integration in a small adapter that returns plain data. Registration may return `None` on refusal. Both ordinary in-process startup and `plugins.isolation: host` are supported. Host transport keeps the filesystem commit and its locks inside the process that writes.
:::

A catalog provider exposes skills through the existing Skills HTTP routes. Core composes discovery and supplies a common review contract. The plugin performs install, update and uninstall under its own policy and records. Provider packages never enter the hub bundle installer or its lock file.

- **Optional discovery:** a registered catalog supplies a source descriptor for clients such as Desktop's Capabilities > Skills view. Clients can add a source filter and badge without a separate plugin tab.
- **Live management:** ownership exists only while the provider is registered. Completed disable leaves installed files as ordinary local skills.
- **Stock behavior:** profiles without a registered provider retain the original route responses, including profiles that used one before. Core retains no provider ownership after disable.
- **Separate consumer:** Collective Wisdom implements its adapter separately. Core imports no consumer-specific code. This API does not itself provide the Desktop UI or the consumer plugin.

The [contract appendix](./skill-catalog-provider-contract.md) defines wire fields, HTTP routes, mutation admission and recovery.

## Register a provider

```python
from hermes_cli.skill_catalog import SkillCatalogProvider


class TeamCatalog(SkillCatalogProvider):
    name = "team"
    label = "Team skills"
    implementation_revision = "1"

    # Implement the read, review and effect methods in the contract appendix.


def register(ctx):
    return ctx.register_skill_catalog(TeamCatalog())
```

This skeleton shows registration only. The base class returns `unsupported` for methods the adapter has not implemented. A usable catalog implements the required methods and keeps its installation journal in provider-owned durable storage.

Declare `skills.catalog` in `plugin.yaml` and obtain the operator's grant through normal plugin consent:

```yaml
name: team-catalog
version: "1.0.0"
capabilities:
  - skills.catalog
```

The owning profile's `plugins.entries.<plugin-key>.granted_capabilities` is authoritative at admission and dispatch. Unreadable configuration denies dispatch. The consent text is:

> Show this plugin's skill catalog in Hermes. Catalog searches may contact its service. When you confirm an install, update or uninstall, the plugin performs it in this profile under its own policy. Disabling the plugin leaves its installed files as ordinary local skills.

Registration uses the scoped registrar and load ledger. Failed or abandoned loads publish no catalog. Static identity uses literal class or instance data, not properties or custom attribute lookup. The host child validates an immutable descriptor separately from the callable proxy, so snapshots never evaluate plugin properties.

## Execution and activation

Methods are synchronous and exchange bounded JSON data:

- **Discovery:** `availability`, `search`, `inspect`, optional `scan`, `installed` and optional `check`.
- **Authority:** `review`, `reviewed`, `confirm` and read-only `operation`.
- **Local commit:** the owning context's mutation helper coordinates every filesystem effect, including recovery and existing plugin entrypoints.

Core dispatches plugin methods on profile-owned workers, not gateway infrastructure threads. Bootstrap has separate capacity from public search. Caller deadlines bound waiting, not the lifetime of a still-running writer. Timed-out work retains capacity until it exits.

HTTP activation preserves ordinary isolation. An already-registered in-process provider works normally. A cold HTTP request may join ordinary host-isolated discovery after a metadata eligibility check. It does not import a cold in-process manager. That manager registers through ordinary startup or reload instead. Secondary profiles retain their existing host boundary.

Bootstrap waits at most five seconds independently of loader settings. Without an admitted registration, the response remains stock and has no provider loading/error fields. A late registration can serve a later request only if its manager generation, enable state and consent remain current. Host bootstrap output goes to bounded, redacted diagnostics, not the response or inherited terminal stream. Core does not redirect process-global stdout.

Provider-aware routing applies to request bodies up to 256 KiB. Larger bodies are handled exactly as a host without providers handles them.

## Identity and profile scope

Core stamps the profile, plugin key, registration generation, request id and deadline. Clients cannot supply a home or credentials. Invalid profiles fail rather than borrowing another profile's provider or secrets. Every effect requires authorization for the selected profile.

A source is `plugin.<owner_digest>.<catalog>`, where `owner_digest` is the full lowercase SHA-256 of the exact UTF-8 manifest registry key. A reference is `<source_id>:<item_id>`. Labels remain short human-readable text. A plugin reusing a vacated catalog name does not inherit another owner's source identity or old confirmations. Changing the plugin registry key changes its identity.

Metadata reuses the hub discovery fields with a serialized provider block. Core stamps source and owner fields and display trust remains `community`. Version, installed version and management state are distinct. Provider policy determines whether confirmation is allowed. A badge does not grant trust or bypass policy.

## Review, confirmation and recovery

```text
client -> review -> display exact plan -> explicit confirmation
                                           |
                              stored token + exact finding ids
                                           |
                              provider atomic operation claim
                                           |
                                guarded local commit -> result
```

A review binds the exact item and version, package digest, complete file plan, findings, policy, enrolled owner and prior installation revision. A token lasts at most 15 minutes before first use. A confirmable plan is complete and has the exact required acknowledgement set. Blocked or incomplete plans have no token. Setup guidance is display-only and never authorizes command execution.

Recovery precedes fresh-action prerequisites:

1. **Completed replay** returns the saved operation read-only, including after expiry, uninstall or target reuse. It creates no reservation and cannot republish ownership.
2. **Pending recovery** reuses the original provider operation binding. A restart requires a new live admission and verification of the provider's resumable journal. It never restores an old permit.
3. **Unstarted authority** enters fresh target admission only after current review and installation checks.

Core retains no tokens. Its disposable projection binds each admitted transition to a reservation, provider operation and expected path revision. Exact settlement repeats are idempotent. Stale owner, generation or revision refuses.

The writer holds its own operating-system exclusion for the entire filesystem critical section, with shared ancestor holds and an exclusive target hold. Parent death cannot release the writer's descriptors. Inside the hold the provider checks actual bytes and original identity against its journal before writing. Core's generic presence check does not replace that provider-specific check.

When a terminal provider journal exists but core settlement failed, core reconciles the projection from the read-only operation record without another file effect. The HTTP operation and installed rows show `settlement: reconciling` until finalization succeeds. A provider may also publish a locally committed pending result before remote reconciliation. A later remote refusal preserves those committed local files.

## HTTP composition

Existing hub routes gain provider branches for source discovery, search, preview, scan, checks and effects. New review, confirm and operation routes use the same authenticated profile scope. Provider references never enter hub bundle resolution, quarantine, locking or CLI spawning. Unqualified Update all keeps hub updates and returns provider review candidates instead of applying them.

Public search and provider work use separate capacity. The public fan-out and fallback launch independently of provider waits. Results keep stock public order and trust logic, and provider rows deduplicate by canonical reference. A non-streaming combined response may wait within the provider cap. This is not a zero-added-latency promise for provider-enabled searches.

Clients can bootstrap optional descriptors through the existing official-catalog request even with no hub installations. With no provider, `provider_sources` is absent. Client catalog and review state must be keyed by connection and profile. Switching either discards late replies and clears selections.

This delivery adds no `hermes skills` parser, activation probe, flags, commands or provider exit-code contract. The plain-data API can support a later CLI caller. Existing generic mutators still share the filesystem safeguards while an HTTP provider is live.

## Installed ownership and disable

`installed()` supplies the provider's local records, not a remote library listing. Core validates skills-root-relative targets and joins `/api/skills` by path rather than display name. Managed rows expose provider provenance, current revision, actions and settlement. Generic edit, delete, archive, rename, restore and hub replacement refuse overlapping live claims or reservations through shared final-path admission.

Unload fences new calls and commits, waits for already-entered short filesystem commits, then drops the provider's projection. It deletes no skill files. An active writer can produce a transient busy refusal, so retry unload after that writer exits. With the last registration gone, stale projection bytes confer no ownership and the original local behavior applies.

Re-enable calls `installed()` again. The provider classifies deleted, edited and replaced paths from its own records. Core validates those claims, never infers ownership by scanning arbitrary trees and never automatically deletes or overwrites a target. Ambiguous recovery against a changed target refuses.

Author-source enrollment is metadata-only and creates no managed claim. An edited update that needs a preservation copy at a second target refuses without writing files. Automatic preservation awaits a multi-target reservation contract. Guidance to archive or rename does not bypass live ownership. Provider reclassification or completed disable is required before a generic action can touch that claim.

## Model exposure and delivery limits

Registration injects no metadata into prompts or history. Deliberate HTTP reads remain untrusted tool data. Approved local installations follow ordinary next-session skill discovery and prompt-prefix rules. There is no new unconditional load-time scanner or model-facing install tool.

The core API includes generic fixture providers and pure-main stock-response controls. It does not establish a release or deployment claim for a consumer. A consumer's real product acceptance includes browse beyond one page, preview, exact-version review and install, next-session use, update, uninstall, disable/edit/delete/re-enable, policy refusal, source failure, profile switching and commit-then-timeout recovery.

CLI delegation, new tabs, ratings, payments, ranking, public index submission and automatic installation suggestions are outside this API. Provider policy remains separate from hub trust.
