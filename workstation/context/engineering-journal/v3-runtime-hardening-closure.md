# V3.1–V3.4 Runtime Hardening Closure

Date: 2026-09-11  
Base: `main@d77901a6857cf90f9401a90377de3b6ee5254bef` plus current Workstation working tree

## Implemented

| Roadmap area | Canonical implementation | Evidence |
|---|---|---|
| Evidence/event/resource state | `workstation/runtime.py` | durable state, bounded queue, deadline and reconnect tests |
| Independent supervision/recovery | `workstation/supervisor.py`, `workstation/recovery_cli.py` | checkpoint/restart tests and Windows process smoke |
| Routine promotion/replay | `workstation/routines.py`, `workstation/memory.py` | validation gate, deterministic replay and drift tests |
| Persistent workers | `workstation/workers.py` | queue, steer, wait, stop, reconstruction, journal and wake-up tests |
| Session/memory lifecycle | `workstation/session_lifecycle.py`, `workstation/memory.py` | lease, migration rollback, compaction and snapshot tests |
| Replay/evaluation | `workstation/replay.py`, `workstation/evaluation.py` | redacted offline replay, model fork, regression and soak tests |
| Isolation/protocol governance | `workstation/isolation.py`, `workstation/protocols.py` | control-plane, degraded boot, MCP bounds, qualification and adapter tests |

## Verification

- `.venv\Scripts\python.exe -m pytest workstation/tests -q -p no:cacheprovider` → **141 passed**;
- `.venv\Scripts\python.exe -m compileall -q workstation` → passed;
- `workstation\doctor.ps1` → environment, committed integration, component lock and license checks passed;
- `workstation\install.ps1 -SkipDependencies` and normal `workstation\install.ps1` → committed integration, lock, license, Python editable install, `npm ci`, import and checkout-clean checks passed;
- `npm run build`, `npm run pack` and `npm run dist:win:nsis` → production Desktop build, unpacked artifact and Windows NSIS target passed; `test-desktop.mjs all` validated the thin-installer payload and native binaries;
- `apps\desktop\npm run typecheck` → **0 errors**;
- packaged GUI E2E (`e2e/launch-packaged-app.spec.ts`) → **5 passed (26.4s)** after lazy Workstation Browser startup, per-sandbox state isolation and packaged-loader injection; see H-021;
- `v3-runtime-hardening-smoke.py` → supervisor recovery, cross-process lease and stale-evidence markers passed;
- `h010-native-browser-session-state-smoke.mjs` → `H010_CLASSIFICATION=VALIDATED`;
- broad Desktop Vitest aggregate → 7,389 passed / 36 failed / 5 skipped, retained as KI-006 rather than masked.

## Evidence boundary

This closes the V3.1–V3.4 implementation contract layer. It does not claim
clean-machine release promotion, full client wiring, cross-engine browser
coverage or long-duration production soak. Those gates remain explicit in
`ROADMAP.md`, `CURRENT_STATE.md` and `KNOWN_ISSUES.md`.
