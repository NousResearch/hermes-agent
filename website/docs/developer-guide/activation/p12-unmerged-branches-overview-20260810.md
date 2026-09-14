# P12 Long-Context Work — Two Unmerged Branches (for MrTrenchTrucker review)

**Date:** 2026-08-10
**Author:** Sahil (via Kensei)
**Purpose:** Share two pieces of P12 work that exist as unmerged branches in the
KenseiAgent repo, to check whether Turbohaul-Manager already addresses these
concerns directly (or has a solution in flight).

---

## Summary

Two self-contained modules were developed for the P12 256K memory policy but
**never merged into KenseiAgent main and never submitted as PRs to
Turbohaul-Manager**. They were preserved as git tags + worktrees. Before
deciding whether to merge, discard, or PR them, we want to confirm they are
not already solved upstream.

Both modules are stdlib-only, dependency-free, and unit-tested.

---

## Branch 1: `feat/turbohaul-mode-switch-20260809`
**Tag:** `preserve/turbohaul-mode-manager-20260809`
**Commit:** `d0b66dc958` — `feat(inference): add request-scoped Turbohaul GPU-KV/RAM-KV mode manager`

### Files touched
| File | Lines | Role |
|---|---|---|
| `scripts/turbohaul_mode_manager.py` | 308 | The mode manager |
| `tests/scripts/test_turbohaul_mode_manager.py` | 382 | Unit tests |

### What it implements
A request-scoped **mode switch between GPU-KV (fast path) and RAM-KV
(long-context) serving** for Turbohaul:

- Normal Turbohaul operation = GPU-KV mode (KV cache in VRAM, weights GPU0-only,
  fast path).
- An explicit long-context route enters RAM-KV mode (`--no-kv-offload`, KV cache
  in host RAM, weights still GPU0-only) for huge-context work, then **restores
  GPU-KV mode afterward**.
- Restore is guaranteed on success, on body error, and on timeout.
- **Single-resident lock:** at most one request may hold RAM-KV mode at a time
  (same constraint as Turbohaul's `max_parallel_sidecars=1`). A second
  concurrent long-context request → `ModeBusyError`.
- Re-entering for the same request id is idempotent (no double physical switch).
- The physical switch is an **injectable seam** (`mode_switcher` callable) — the
  module itself is dependency-free; the integration task decides how the live
  sidecar actually changes flags.
- Timeout semantics: `run_long_context` runs the body in a single-worker
  executor, raises `ModeTimeoutError` on deadline, **still restores**, and lets
  the timed-out thread finish in the background (never kills a llama-server
  worker mid-request — slot-state corruption risk).
- API: `enter_long_context(request_id)`, `restore_normal(request_id)`,
  `effective_mode()`, `active_requests()`, `run_long_context(request_id, fn)`,
  context-manager form `with mgr.long_context(request_id)`.
- Every transition logged as a structured one-line record with request id.

### Design intent
This is the **cold-respawn / mode-switch control** layer: who may flip a
sidecar from GPU-KV to RAM-KV and back, with what exclusivity and timeout
guarantees. It does NOT itself spawn/stop engines — it coordinates the switch
and guarantees restore.

---

## Branch 2: `t_e9639915/gpu-kv`
**Tag:** `preserve/p12-gpu-kv-allocator-20260809`
**Commit:** `0add54b170` — `feat(inference): GPU-resident KV-cache allocator as default fast path (P12)`

### Files touched
| File | Lines | Role |
|---|---|---|
| `scripts/p12_kv_cache_gpu_allocator.py` | 639 | GPU KV allocator (fast path) |
| `scripts/bench_p12_gpu_allocator.py` | 150 | Benchmark harness |
| `tests/scripts/test_p12_kv_cache_gpu_allocator.py` | 466 | Unit tests |

### What it implements
A **GPU-resident KV-cache allocator** — the default fast serving path for
ordinary contexts:

- Contexts ≤ `KV_CACHE_OFFLOAD_THRESHOLD` (default 128K tokens) allocate KV
  cache **on the GPU by default** — the fast path for daily work.
- A **VRAM budget check runs BEFORE allocation**; if the request cannot fit in
  available GPU memory (or the allocator is unavailable), it fails GRACEFULLY:
  logs and hands off to the offload path via an `OffloadHandoff` result. It
  never crashes the serving loop.
- Reads the placement policy from `scripts/kv_cache_policy.py` (`decide_tier()`)
  — does NOT duplicate decision logic; only executes GPU allocation when the
  policy says GPU_RESIDENT.
- **GPU1 isolation preserved** — main lane's KV cache allocated on GPU0 only.
- Minimal-fragmentation design: small pool manager over `torch.empty` blocks
  with a sorted free-list for reuse; `fragmentation_ratio` metric; `compact()`
  helper (never calls `torch.cuda.empty_cache()` itself).
- Allocation is synchronous (matches current hot path) — no added latency;
  async copies are the offload path's job.
- Entry point: `allocate_kv_cache()`; CLI `main()` for manual smoke tests.

### Design intent
This is the **fast-path resource manager**: ensure GPU memory is available
before allocating KV, keep fragmentation low, and hand off to RAM-KV when GPU
memory is tight — without ever crashing the serving loop.

---

## Relationship to the 4 merged PRs

Sahil has 4 merged PRs in Turbohaul-Manager (all approved Aug 7, "we took all
four"):
- PR #9 `fix(lifecycle): reap only recorded engine identities`
- PR #10 `feat(queue): reserve admission for main lane`
- PR #11 `test(worker): remove timing-dependent queue assertion`
- PR #12 `test(api): align negative keep-alive contract`

Those PRs touch `src/turbohaul/{manager,queue,config,singleton,state}.py` —
**engine lifecycle, ownership, queue admission**. They do NOT touch KV-cache
allocation or mode switching.

**These two branches are a different layer:** KV-cache memory management
(GPU vs RAM) and the mode-switch coordination for long-context serving.

---

## Questions for MrTrenchTrucker

1. Does Turbohaul-Manager already have (or plan) a GPU-KV / RAM-KV **mode
   switch** for long-context serving — i.e. flipping `--no-kv-offload` per
   request with restore guarantees? (Branch 1)
2. Does Turbohaul-Manager already have a **GPU KV-cache allocator / VRAM budget
   check with graceful offload handoff**? (Branch 2)
3. If neither: would these be welcome as PRs to Turbohaul-Manager (adapted to
   its architecture — e.g. integrated with the manifest system), or are they
   better left as KenseiAgent-side helpers?

---

## Recommendation (Sahil, preliminary)

**Hold both branches for now.** They are preserved (tags + worktrees), cost
nothing, and the decision to merge/PR/discard should follow MrTrenchTrucker's
answer on whether the manager already solves this.
