---
title: "Task Completion Verification"
description: "Exact-object, state-typed completion semantics for Hermes repository work."
sidebar_position: 1000
---

# Task Completion Verification

*All Gods Must Die* now carries a sixth architectural law:

> **No task is complete until the requested state exists at the authoritative target and is verified on the exact object.**

This closes a recurrent agent failure class: promoting evidence of activity into evidence of completion. A patch is not a branch. A branch is not a pull request. A pull request is not an approval. An approval is not a merge. A merge is not a release. A release is not an operational proof.

## Why this is architecture

Hermes already requires adversarial verification for load-bearing transformations. The producer cannot certify its own extraction or translation; independent checks adjudicate the result. Repository execution needs the same separation.

A local artifact can prove `prepared` or `locally_verified`. It cannot prove that GitHub changed. A successful API call can prove only what a post-write read-back confirms. A green workflow run proves only the exact SHA it tested. A historical pull request remains provenance after a typed `superseded_by` edge moves canonical ownership elsewhere.

The negative rule is equally important:

> **No receipt may be inherited from a local draft, previous head, adjacent commit, parent object, superseded artifact, or intended write.**

## Completion contract

Every work order resolves to a completion contract:

```text
(authoritative system,
 exact target object,
 requested predicate,
 admissible evidence,
 acceptance authority,
 required graph closure)
```

The requested predicate determines what must be proven. "Prepare a patch" may terminate locally. "Fix this PR" requires a target branch or PR update and post-write read-back. "Merge it" requires a merge receipt. "Verify production" requires a real-surface behavior receipt bound to the deployed version.

## Completion state vector

One green word cannot represent the real state. Hermes records six independent axes:

| Axis | States |
|---|---|
| Materialization | `absent`, `local_only`, `target_present` |
| Integrity | `unverified`, `locally_verified`, `read_back`, `exact_object_verified` |
| Governance | `unreviewed`, `reviewed`, `accepted`, `rejected` |
| Integration | `not_submitted`, `open`, `closed_unmerged`, `merged`, `released` |
| Operation | `not_exercised`, `hermetic_verified`, `operationally_verified` |
| Lineage | `active`, `blocked`, `superseded`, `retracted`, `abandoned` |

The completion claim is the exact conjunction the contract required. It may be weaker than the system hoped. It may never be stronger than the receipts.

## Minimum receipts

| Predicate | Minimum evidence |
|---|---|
| `prepared` | local artifact path and content digest |
| `locally_verified` | local command, exit status, exact input digest |
| `submitted` | target URL/object ID plus post-write read-back |
| `exact_object_verified` | exact object version plus checks bound to that version |
| `accepted` | designated authority verdict on the exact verified object |
| `merged` | merge commit or default-branch reachability |
| `released` | immutable release identity and digest |
| `operationally_verified` | real execution-surface receipt bound to the delivered version |
| `blocked` | failed operation, exact error, and boundary owner |
| `superseded` | typed successor edge and explicit retirement of the old object |

## Webhook case study

The supplied Task Completion Verification record captured two locally prepared fixes whose GitHub publication attempts failed with `403 Resource not accessible by integration`. The patches had SHA-256 digests and passed Git's structural parser, but no branch changed and no public state was altered.

That execution was correctly classified as:

```text
materialization: local_only
integrity: locally_verified
integration: not_submitted
lineage: blocked
```

The graph later moved.

- [PR #85002](https://github.com/NousResearch/hermes-agent/pull/85002) became the canonical Task 7 object and, at the August 21, 2026 reconciliation, carried exact-head successful CI at `355b50a30c184f1c70df172e018dedd240bd7400`.
- [PR #85523](https://github.com/NousResearch/hermes-agent/pull/85523) is closed, unmerged, and retained as historical Task 10 provenance.
- [PR #90236](https://github.com/NousResearch/hermes-agent/pull/90236) is the canonical Task 10 successor and, at the same reconciliation, carried exact-head successful CI at `416c5e48aefcccfe945f7d4118415a3b992e017a`.

The later target objects do not retroactively make the earlier 403 a successful write. The earlier 403 does not deny the later target state. Both events remain in the mutation journal.

## Namespaced campaign predicates

The god-file campaign's `SHIPPED` standard remains valid, but it is campaign-specific. It means that the agreed slice exists as an open, individually interlocked pull request with the campaign's required reviews and evidence.

Represent it as `KAG_SHIPPED`, not as a generic synonym for `MERGED`, `PUBLISHED`, `RELEASED`, or `OPERATIONALLY_VERIFIED`.

## Enforcement

The repository package includes:

- `docs/research/task-completion-record.schema.json` - JSON Schema for completion records;
- `docs/research/task-completion-verification-ledger.json` - the dated webhook packet and current GitHub reconciliation as separate, valid records;
- `docs/research/task-completion-verification-amendment.md` - the formal doctrine source;
- `docs/research/all-gods-must-die-adversarially-verified-transformation-v1.1.tex` - composition source for the successor release;
- `docs/research/all-gods-must-die-v1.1-release-manifest.json` - source and generated-output digests with typed lineage.

A verifier must reject any record that borrows CI from another SHA, uses a local path as proof of remote mutation, marks a superseded object active, omits a write failure, or names a stronger predicate than its receipts support.

## Revised declaration

1. No system component may accumulate authority without becoming legible.
2. No claim may outrank its evidence.
3. No actor may certify itself.
4. No rule counts until it executes.
5. No debt gets to hide behind institutional forgetting.
6. **No task is complete until the requested state exists at the authoritative target and is verified on the exact object.**
