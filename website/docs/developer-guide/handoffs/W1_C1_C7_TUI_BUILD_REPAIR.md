# W1 C1/C7 TUI Build Repair & Type-Contract Fix

**Status:** COMPLETE — 2026-07-12 22:34 UTC

## Summary

Fixed 19 TypeScript type failures blocking `npx tsc --noEmit -p tsconfig.json` in `ui-tui/`, plus hardened `scripts/run_tests.sh` for clean-checkout safety.

## Changed Files

All paths under `ui-tui/` unless noted.

### `src/app/interfaces.ts`
- **UiState.agentMode** (line 185) — added `agentMode: AgentMode` field (required). Root cause of 8 type errors at 6 call sites.
- **ComposerRefs.submitRef** (line 234) — widened type to `(value: string, showUserMessage?: boolean, skipOptimization?: boolean) => void`.
- **UseComposerStateOptions.submitRef** (line 253) — same widening.
- **submission.submitRef** (line 321) — same widening.
- **AppLayoutActions** — added `answerAskUserQuestions` and `answerPromptOptimization` handlers.
- **AppOverlaysProps** — added `onAskUserQuestionsAnswer` and `onPromptOptimizationChoice` callbacks.

### `src/gatewayTypes.ts`
- Added `ask_user_questions.request` variant to the `GatewayEvent` discriminated union (before `clarify.request`). Payload shape: `{ questions: Array<{ question, header?, multiSelect?, options? }>, request_id: string }`. Fixed 3 type errors.

### `src/app/useSubmission.ts`
- `submit()` function widened to accept optional `showUserMessage?` and `skipOptimization?` params, matching the call sites in `useMainApp.ts`.

### `src/app/useMainApp.ts`
- `submitRef.current(...)` calls in `answerPromptOptimization` handler — dropped the dead `true, true` args that didn't thread through to `submitPrompt()` on this code path. The underlying `submit()` → `dispatchSubmission()` → `send()` path never used them.

### `src/components/appOverlays.tsx`
- Added `onPromptOptimizationChoice` to `Pick<AppOverlaysProps, ...>` selection to match the current interface.
- Fixed TS18047 (`overlay.askUserQuestions` possibly null) — narrowed via local const `askUserQuestions = overlay.askUserQuestions` inside the truthy guard.

### `scripts/run_tests.sh` (repo root)
- Refactored venv detection to use bash array `PY_LAUNCHER` instead of string `PYTHON`.
- Added `uv run -- python3` fallback when no `.venv`/`venv` exists and `uv` is available.
- Better error message with install instructions for both venv and uv.

## Evidence

- **TUI typecheck:** `npx tsc --noEmit -p tsconfig.json` — exit 0, 0 errors, 0 warnings.
- **TUI test suite:** 75 test files pass (763 tests), 32 fail pre-existing (`packages/hermes-ink` needs dist build — unrelated to this change).
- **Python memory-tool battery:** 87/87 pass (unaffected, preserved from C9 fix).

## Error Register Resolution

| # | Count | Root Cause | Status |
|:--|:------|:-----------|:-------|
| P2 | 8 | `agentMode` not in `UiState` interface | FIXED — added required field |
| P3 | 3 | `ask_user_questions.request` missing from `GatewayEvent` union | FIXED — added variant |
| P4 | 2 | `submitRef.current` 3-arg call vs 1-arg type | FIXED — widened type, dropped dead args |
| P5 | 6 | Missing overlay props/actions + null guard | FIXED — added 4 interface fields + local const narrowing |

## Scope Notes

- **No active gateway, cron, service, or state touched.** All changes are source-only.
- **No `any`, `@ts-ignore`, or broad casts used.** Every fix is strict type-contract enforcement.
- **No new features or command behaviour changes.**
