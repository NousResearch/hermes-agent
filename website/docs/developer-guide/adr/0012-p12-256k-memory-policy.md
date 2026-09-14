# ADR 0012 — P12 256K memory policy: GPU0-only weights, opt-in CPU offload

**Date:** 2026-08-09
**Status:** Accepted
**Deciders:** Sahil (decision), Kensei (implementation record)

## Context

The P12 local inference stack runs on the dual-RTX-3090 rig through Turbohaul.
There are two resident lanes:

- **Main lane (Sirvir primary)** — Darwin 28B, served by
  `scripts/systemd/turbohaul-main.service`. Weights are loaded onto GPU0
  (`-ngl 99 --mlock --no-mmap`, no `CUDA_VISIBLE_DEVICES` override).
- **Aux lane (Sirvir fast)** — Qwopus 27B, served by
  `scripts/systemd/turbohaul-aux.service`, pinned to GPU1 via
  `CUDA_VISIBLE_DEVICES=1`. GPU1 also hosts image/media/speech work.

P12 introduced the capability to promote the main model to a 256K-context
configuration. Large-context inference (256K) cannot always fit the full
working set in GPU0 VRAM, which raises the temptation to spill either model
weights or context/KV-cache data to CPU/system RAM. Spilling weights would
destroy the fast path for all daily work and could contaminate the GPU1
isolation contract.

## Decision

1. **Model weights are GPU0-only. They must never be offloaded to CPU or
   system RAM, and never placed on GPU1.** The main model's weight tensors
   stay pinned to GPU0 for the entire process lifetime. This is the
   non-negotiable baseline.

2. **CPU/system-RAM offload is permitted for context/KV-cache data ONLY** —
   and only for rare, genuinely huge-context workloads (e.g. 256K-context
   jobs). This exception is **explicitly opt-in** (e.g. a dedicated flag/env
   var such as `P12_ALLOW_HUGE_CONTEXT_OFFLOAD=1`). It must never be the
   default, and it must never move model weights.

3. **Normal daily work stays on the fast path:** weights on GPU0, no offload
   of any kind, context/KV-cache resident on GPU0.

4. **GPU1 isolation is preserved at all times.** GPU1 belongs to the aux lane
   and media/speech work. Neither the main model's weights nor its
   context/KV-cache may be placed on GPU1, even during huge-context jobs.
   Huge-context offload, when opted in, targets CPU/system RAM — never GPU1.

## What this means in practice

- Default runtime configuration has **no weight-offload behaviour**.
- The fast path is the only path normal daily work can take: main weights on
  GPU0, no CPU/system-RAM spill, GPU1 untouched.
- A huge-context job that needs more memory than GPU0 can hold must
  explicitly opt in, and may only spill context/KV-cache (never weights) to
  CPU/system RAM, while GPU1 stays isolated.
- Any code path that attempts to place weights on CPU, system RAM, or GPU1 is
  a violation of this policy and should fail fast with an actionable error.

## Revisit triggers

- A model whose weights genuinely cannot fit GPU0 even at minimum context
  (would require re-evaluating the GPU0-only baseline).
- A Turbohaul/Turboquant change that adds first-class weight-offload or
  multi-GPU weight sharding (revisit the isolation contract then).
- A future decision to give the main lane more than one GPU (supersedes the
  GPU1-isolation clause for that lane).

## References

- `~/brain/conventions/performance-rules.md` — the canonical **P12 performance rule**: the main model must feel fast for ordinary daily work; 256K context is a required capability; occasional near-256K jobs may run slower. ADR 0012's GPU0-only weight policy and opt-in offload are the mechanism that keeps the fast path fast while preserving the 256K capability.
- `scripts/systemd/turbohaul-main.service` — main lane, GPU0.
- `scripts/systemd/turbohaul-aux.service` — aux lane, GPU1.
- `scripts/sirvir_turbohaul_observer.py` — Sirvir pressure/contraction policy
  observer for the same stack.
