---
sidebar_position: 10
title: "Trajectories & Training Format"
description: "How Hermes saves trajectories, normalizes tool calls, and produces training-friendly outputs"
---

# Trajectories & Training Format

Hermes can save conversation trajectories for training, evaluation, and batch data generation workflows.

Primary files:

- `agent/trajectory.py`
- `run_agent.py`
- `batch_runner.py`
- `trajectory_compressor.py`

## What trajectories are for

Trajectory outputs are used for:

- SFT data generation
- debugging agent behavior
- benchmark/evaluation artifact capture
- post-processing and compression pipelines

## Normalization strategy

Hermes converts live conversation structure into a training-friendly format.

Important behaviors include:

- representing reasoning in explicit markup
- converting tool calls into structured XML-like regions for dataset compatibility
- grouping tool outputs appropriately
- separating successful and failed trajectories

## Persistence boundaries

Trajectory files do **not** blindly mirror all runtime prompt state.

Some prompt-time-only layers are intentionally excluded from persisted trajectory content so datasets are cleaner and less environment-specific.

## Batch runner

`batch_runner.py` emits richer metadata than single-session trajectory saving, including:

- model/provider metadata
- toolset info
- partial/failure markers
- tool statistics

## Related docs

- [Environments, Benchmarks & Data Generation](./environments.md)
- [Agent Loop Internals](./agent-loop.md)
