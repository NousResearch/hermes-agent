# Board redesign: Backlog → Intake → Delivery

The KENSEI dashboard Kanban view was redesigned (2026-06) to replace a flat
12+ column raw-status board with three surfaces that mirror the upstream /
downstream model used by Jira, Azure DevOps and Linear.

**The board UI does not live in this repo.** It is the standalone
`kensei-dashboard` app (`~/repos/kensei-dashboard`, Vite + React + Tailwind v4,
served by its own FastAPI backend on `:9123`). The page is
`web/src/pages/Kanban.tsx`. This document records how that surface maps onto the
feature pipeline owned by *this* repo, and the engine paths its backend calls.

## Surfaces

| Surface | Meaning | Engine statuses |
|---------|---------|-----------------|
| **Backlog** | Raw idea pool, pre-shaping | `backlog` |
| **Intake** | Discovery + approval pipeline (per-feature gate stepper) | `triage, research, prd, spec, council, sign_off` |
| **Delivery** | Active build (flat agile board) | `tech_review, decompose, ready, scheduled, todo → To&nbsp;Do/Ready` · `running, execute → In&nbsp;Progress` · `blocked → Blocked` · `review, pr+qa → PR` · `audit, final_sign_off → QA` · `document, done → Done` |

Every engine status maps to exactly one surface, so no card is ever hidden (the
"unmapped status = invisible card" trap). Express features grey their skipped
gates (PRD, Council) in the stepper. Delivery cards carry an epic colour-chip and
support Group-by None/Epic/Assignee swimlanes. A unified "Awaiting you" strip
surfaces human gates (`sign_off`, `final_sign_off`).

## Engine paths the dashboard backend uses

The dashboard backend imports `hermes_cli` from this repo and goes through the
canonical, event-logged `kanban_db` API (never raw writes). New dashboard routes
(`kensei-dashboard/backend/pipeline.py` + `app.py`):

- `POST /api/pipeline/promote/{id}` — promote a Backlog idea into Intake: sets
  `tier=full`, `status=research`, `pipeline_stage=research`, and appends a
  `pipeline_advanced` event. Mirrors the triage processor's promotion.
- `POST /api/pipeline/approve/{id}` — approve a human gate: writes a
  `human_approved` event (same contract as `hermes feature sign-off`); the
  gateway dispatcher advances the task on its next tick.
- `GET /api/pipeline/epics` — board-aware parent/child linkage read from
  `task_links` plus `mode_of` (pipeline_mode per task) for epic chips, swimlane
  grouping, and express greying. Read-only.

## Operating notes

- The dashboard `uvicorn` runs without `--reload`; after a backend change run
  `sudo systemctl restart kensei-dashboard`. Frontend changes need `pnpm build`
  in `kensei-dashboard/web` (served statically from `web/dist`).
- WIP limits are intentionally **not** shown yet: real per-column limits need a
  config surface; hardcoded numbers would be arbitrary. Column counts are shown.
- The redesign is a presentation layer plus two thin backend actions. The
  pipeline engine (statuses, dispatcher, gates, events) is unchanged.
