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
