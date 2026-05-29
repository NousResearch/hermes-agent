# Kynver Brain & Enforcement Layer — Roadmap

Hermes Forge is the first runtime adapter on Kynver AgentOS. Operating hooks
(memory, sessions, todos/plan progress, skills, audit) ship in
`plugins/memory/kynver/`. The **brain/enforcement layer** lives in Kynver core
and tightens over time without forking Hermes machine-control tools.

## Layering

| Layer | Owner | Responsibility |
| --- | --- | --- |
| **Runtime adapter** (Hermes Forge) | `plugins/memory/kynver/` | Bridge local tools; mirror state; fail-open degraded mode |
| **AgentOS control plane** | Kynver `src/modules/agent-os/` | Plans, progress rows/focus, tasks, sessions, memory API |
| **Harness runtime** | `@kynver-app/runtime` | Worktrees, workers, heartbeats, completion payloads |
| **Brain / enforcement** (roadmap) | Kynver core | Admission, review gates, policy, audit receipts, trust |

## Near-term (M8–M10)

1. **Unified operating audit stream** — correlate `session.events`, task events,
   plan progress events, and Hermes `agent_loop_tool_observed` into one queryable
   timeline per `KYNVER_TASK_ID`.
2. **Review-gate enforcement on completion** — eager idempotent reviewer tasks when
   workers finish; structured blockers when report/deep review rows are open.
3. **Plan progress propose/confirm** — workers emit `partial` only; MCP/session
   agents propose/confirm `done` (no worker CLI `done`).
4. **Degraded-mode operator surfacing** — dashboard badge when Forge runs with
   `provider.degraded` metadata from Kynver health failures.

## Mid-term (M11–M14)

1. **Policy brain** — rate limits, tool allowlists, and escalation paths keyed by
   `executorRef` and plan row role lanes.
2. **Transaction audit receipts** — DID-signed execution receipts from adapters;
   Phase 7 ingestion validates authorization tokens and enriches trust scores.
3. **Cross-runtime skill provenance** — treat Kynver skill bodies as untrusted
   user content (already tagged in Forge); enforcement blocks elevation into
   system policy without explicit operator approval.
4. **Standalone plugin extraction** — publish `hermes-kynver-agentos` for profiles
   that cannot ship in-tree memory providers; Forge keeps the reference wiring.

## Long-term

- **Dispatcher brain** owns lease, reclaim, and `running` vs `in_progress` semantics.
- **Landing/next-action router** creates child tasks with lane-expert personas.
- **Enforcement closes the loop** between verification at registration and
  continuous compliance from receipts and behavioral baselines.

## References

- [Kynver AgentOS runtime contract](./kynver-agentos-runtime-contract.md)
- Kynver plan: `docs/superpowers/plans/2026-05-26-hermes-forge-kynver-first-operating-tools.md` (monorepo)
- Plan progress focus: `docs/superpowers/plans/2026-05-28-agentos-plan-progress-in-progress-focus.md` (monorepo)
