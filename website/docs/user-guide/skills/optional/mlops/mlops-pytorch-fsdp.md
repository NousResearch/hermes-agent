---
title: "Pytorch Fsdp — Guide PyTorch FSDP setup, sharding, and debugging"
sidebar_label: "Pytorch Fsdp"
description: "Guide PyTorch FSDP setup, sharding, and debugging"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pytorch Fsdp

Guide PyTorch FSDP setup, sharding, and debugging.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/pytorch-fsdp` |
| Path | `optional-skills/mlops/pytorch-fsdp` |
| Version | `1.1.0` |
| Author | yinjianxxx; original documentation by Orchestra Research |
| License | MIT |
| Dependencies | `torch>=2.0,<3` |
| Platforms | linux, macos |
| Tags | `pytorch`, `fsdp`, `distributed-training`, `sharding`, `checkpointing` |
| Related skills | `accelerate`, `torchtitan`, [`pytorch-lightning`](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# PyTorch FSDP Skill

Use this skill to design, migrate, or debug PyTorch Fully Sharded Data
Parallel training. It keeps the always-loaded instructions concise and sends
API details to the bundled reference archive or current PyTorch documentation.

## When to Use

- A model no longer fits with ordinary data parallel training.
- You need to choose between composable FSDP2 and wrapper-based FSDP1.
- A distributed job hangs, runs out of memory, or diverges after sharding.
- Checkpoint save, load, or resharding behaves differently across world sizes.
- Mixed precision, CPU offload, or activation checkpointing needs validation.

Do not use FSDP only because multiple GPUs are available. Start with ordinary
DistributedDataParallel when each rank can hold the model and optimizer state.

## Prerequisites

- A supported PyTorch 2.x installation on every rank.
- A launch method such as `torchrun` and a working process-group backend.
- One process per accelerator for NCCL-based GPU training.
- Shared, durable storage when checkpoints must survive node loss.
- The same code, model structure, and collective order on every rank.

Use `terminal` to record the installed PyTorch and accelerator versions before
choosing APIs:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## How to Run

1. Capture the target PyTorch version, device type, world size, model size,
   checkpoint format, and memory budget.
2. Choose FSDP2 for new composable sharding when its required APIs are present.
   Preserve FSDP1 for existing code built around the wrapper API.
3. Use `read_file` on `references/index.md`, then open only the relevant
   section of `references/other.md`.
4. Confirm current signatures in the official PyTorch documentation before
   copying an archived example.
5. Prove the configuration with a two-rank smoke test before scaling out.

## Quick Reference

| Need | Preferred surface | Check first |
|---|---|---|
| New composable sharding | `torch.distributed.fsdp.fully_shard` | PyTorch version and mesh |
| Existing wrapper code | `FullyShardedDataParallel` | wrap policy and device placement |
| Multi-rank checkpoints | `torch.distributed.checkpoint` | save/load participation by rank |
| Reduced activation memory | activation checkpointing | recomputation cost and wrap order |
| Lower communication cost | mixed precision | reduction dtype and numerically sensitive ops |
| CPU memory tradeoff | CPU offload | transfer cost and optimizer behavior |
| Launch and rendezvous | `torchrun` | rank, world size, address, and port |

Official references:

- FSDP2: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- FSDP1: https://docs.pytorch.org/docs/stable/fsdp.html
- Distributed checkpointing:
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html

## Procedure

### 1. Establish a correct unsharded baseline

Run one device and then ordinary DistributedDataParallel with the same batch,
seed, optimizer, and loss. Record peak memory, throughput, and the first few
loss values. A sharded run is not a useful diagnostic baseline by itself.

Completion criterion: the baseline trains deterministically enough to compare
loss and checkpoint behavior with the sharded run.

### 2. Initialize one process per device

Set the local device before constructing CUDA modules and initialize the
process group exactly once:

```python
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
```

Use a backend appropriate for the actual device. Do not silently switch
backends to hide an NCCL or rendezvous failure.

Completion criterion: every rank reports the expected global rank, local
device, backend, and world size, then exits cleanly.

### 3. Apply sharding at deliberate boundaries

For FSDP2, shard repeated inner blocks before the root module:

```python
from torch.distributed.fsdp import fully_shard

for block in model.blocks:
    fully_shard(block)
fully_shard(model)
```

For an existing FSDP1 codebase, keep the wrapper path explicit:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
```

The exact policy depends on model structure and PyTorch version. Avoid wrapping
the same parameters through overlapping manual and automatic policies.

Completion criterion: all trainable parameters have one intended sharding
owner and a single optimizer is created after sharding.

### 4. Add memory features one at a time

Enable mixed precision, activation checkpointing, or CPU offload separately.
Measure peak device memory and step time after each change. Keep normalization,
loss computation, and other sensitive operations at a safe dtype.

Completion criterion: each feature has an observed memory benefit and no
unexplained change in loss or optimizer state.

### 5. Design checkpointing before long runs

Choose whether checkpoints must load at the same or a different world size.
Make every rank follow the documented save/load protocol, write to a unique
checkpoint directory, and include optimizer state when training will resume.

Completion criterion: a fresh process group can load the checkpoint, run one
step, and save again without missing or unexpected keys.

### 6. Scale through controlled stages

Validate in this order:

1. One process with the production model code.
2. Two local ranks with a tiny batch.
3. All local devices.
4. Two nodes with the production rendezvous and storage path.
5. The intended world size.

Change only one dimension at a time. Record the first stage where behavior
diverges.

Completion criterion: the smallest failing topology is reproducible.

### 7. Debug hangs and rank divergence

Use `TORCH_DISTRIBUTED_DEBUG=DETAIL` to surface collective mismatches and
`NCCL_DEBUG=INFO` for NCCL initialization or transport failures. Add a
monitored barrier around suspected phase boundaries when the backend supports
it. Compare per-rank logs by rank and step instead of merging them by time.

Completion criterion: identify the first rank and operation that diverges,
not merely the rank that times out last.

## Pitfalls

1. **Creating the optimizer before sharding.** Parameter ownership or objects
   may change; construct the optimizer after the final sharding transform.
2. **Wrapping arbitrary module boundaries.** Overlapping policies can shard
   the same parameters twice or leave large blocks unsharded. Inspect the
   resulting module tree and parameter ownership.
3. **Changing several memory features together.** A lower memory peak does not
   identify which feature helped or which one changed convergence.
4. **Saving from rank zero without checking the API contract.** Distributed
   checkpoint APIs may require every rank to participate even when only one
   process writes metadata.
5. **Assuming a checkpoint is world-size independent.** Test the exact
   resharding path needed by recovery or inference before a long run.
6. **Treating the last timeout as the root cause.** Collective failures often
   originate on a different rank at an earlier operation.
7. **Copying archived signatures verbatim.** PyTorch distributed APIs evolve;
   check the documentation for the installed version.

## Verification

Before accepting an FSDP configuration, verify all of the following:

- A one-device baseline and a two-rank sharded run have comparable early loss.
- Every rank reports the intended device, backend, and world size.
- Peak memory and step time are recorded for each enabled memory feature.
- The optimizer is created after the final sharding transform.
- A saved model and optimizer state can be loaded by a fresh process group.
- One resumed step completes without missing keys, unexpected keys, or NaNs.
- The smallest multi-node topology completes initialization and one step.
- Current PyTorch documentation confirms the APIs used by the installed
  version.

If any check fails, return to the smallest reproducible topology and change
one sharding, precision, checkpoint, or launch decision at a time.
