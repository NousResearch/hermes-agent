---
title: "Federation Role Registry"
description: "Audit and safely seed the governed Hermes specialist federation."
---

# Federation Role Registry

Hermes profiles are the deployable unit for federation specialists. The checked-in role registry defines the expected departments, role boundaries, skills, toolsets, schedules, and handoff graph. It covers:

- Federal Core: intake, safety, resource, operations, release, and independent audit
- Engineering: implementation, contracts, CI repair, and architecture
- Research and Intelligence: observation, pathology, prior art, experiments, replication, red teaming, performance, validation, and synthesis
- Knowledge Commons and Library: discovery, source verification, cataloging, acquisition, retrieval, preservation, and librarianship
- Shared Memory: intake, curation, and independent validation
- Arts Department: creative direction, concept art, storyboarding, writing, image/media production, editing, and visual review
- Content Studio: nerdy-content discovery, paper/transcript ingestion, classification, citation verification, digest editing, and publication curation

## Audit coverage

Run this from the Hermes checkout or an installed editable environment:

```bash
hermes federation audit
hermes federation audit --json
```

The audit distinguishes three states:

- `installed`: the canonical role profile already exists
- `covered_by_existing`: an existing profile alias provides equivalent coverage
- `missing`: no exact profile or declared alias exists

It also reports whether each role's referenced bundled skills are available. This is a planning report only; it does not modify profiles.

## Seed specialist profiles

Seeding is always a dry-run unless `--apply` is explicit:

```bash
hermes federation seed
hermes federation seed --department arts_media
hermes federation seed --department knowledge_commons
hermes federation seed --role librarian --role nerdy-content-scout
```

When the plan is correct, apply it:

```bash
hermes federation seed --apply
```

Use `--create-alias` only when wrapper aliases are wanted for newly seeded profiles. Existing profiles are never overwritten, cloned, or reconfigured by the default command. If an exact role name is already present as an older profile, use `--refresh-existing` to adopt it: its existing `SOUL.md` is preserved and receives a federation addendum, while its primary, fallback, auxiliary, and identity metadata are updated from the registry. Each newly created profile receives a role-specific `SOUL.md` and `federation_role.json` identity record.

The registry also defines ten durable Bot Mode groups. Seed them alongside the profiles with `hermes federation seed --apply --groups`; repeat runs are idempotent and preserve unrelated Bot Mode metadata.

The seeded profiles appear in `hermes profile list`, the desktop Bot Mode roster, and the normal `hermes -p <role-id> chat` path. Bot Mode receives the role record through `profiles.list`, allowing future roster surfaces to group or filter specialists without reading profile files directly.

## Authority boundary

The registry describes coordination and capability; it does not grant permissions. `advisory`, `operator_gated`, and `write_scoped` are explicit role boundaries, and every generated identity instructs the specialist to preserve provenance, treat external content as untrusted, and avoid turning research or creative output into runtime or acceptance authority. Operators still control credentials, live execution, merges, deployments, and publication.

The source of truth is `configs/federation/roles.json`. Update the manifest and its tests when adding a department or role; do not silently add ad hoc profiles that bypass the registry.
