# Hermes Workstation workload benchmark

`workload_baseline.json` is a provider-free, anonymized structural baseline for
the Workstation contracts. Run it with:

```bash
python -m workstation.workload_benchmark
```

The fixture intentionally contains no private transcripts or `.db` files. It
uses the canonical worker registry, evidence projection, policy engine, event
bus and operational-reference compaction envelope. Large tool-result counters
model reference-first transport using deterministic synthetic payloads; they do
not create a second artifact/result store.

CI should compare counters and bounded ratios against the versioned baseline,
not wall-clock timings. Update the version only when the workload contract or
its fixture changes deliberately.

The durable-execution regression runs the production compiler, canonical
SQLite plans and ArtifactStore without a paid provider:

```bash
python -m workstation.benchmarks.benchmark_execution_paradigm --durable-regression
```

`durable_execution_baseline.json` records a sampled 100-item run with a
transient read failure, restart after 37 completed items and one exception.
Planner boundaries and baseline inline transport are modeled; the separate
conversation integration test verifies two fake-provider calls for 100
operations. Assertions check bounded output and zero completed-item replay,
not exact serialized sizes or latency. Required physical operations still
scale with item count. Paid tokens and live browser behavior are not measured.
