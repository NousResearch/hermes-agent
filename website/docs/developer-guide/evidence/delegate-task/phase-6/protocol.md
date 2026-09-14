Phase 6 backend-neutral experiment protocol

This protocol is frozen for any future separately approved experiment; it is not a production interface.

Provider-neutral read contract
- retrieve(query, scope, limit, now, applicability) -> ordered entries
- Each entry must contain content, provenance, source_scope, created_at, applicability and confidence.
- The adapter must be read-only: no add, update, supersede, usage increment, promotion or durable write.
- Scope is an opaque profile/collection/session value owned by the caller; adapters may not broaden it.
- Missing/ambiguous provenance returns no entry rather than guessing.

Controlled arms
1. memoryless child (current default)
2. explicitly opted-in scoped retrieval
3. broad retrieval (adversarial control only)
4. persistent executor only if separately approved

Frozen variables
Model, provider, routing, prompt, tools, budget, task corpus, judge, seed and concurrency remain identical across arms.

Predefined gates
- correctness: measurable task-quality gain over memoryless arm;
- leakage: zero cross-profile/source-scope hits;
- trap resistance: zero accepted poisoned/spoofed/contradictory memory answers;
- writes: zero child durable-memory writes;
- cost/latency: no unapproved increase and bounded p95 latency;
- reproducibility: repeated seeded trials with raw outputs and hashes.

NO-GO condition
If no provider-neutral read adapter and frozen comparable corpus exist, do not build or enable the feature. Retain memoryless children and reopen only under a separate release decision.
