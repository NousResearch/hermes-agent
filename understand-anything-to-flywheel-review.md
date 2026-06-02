# Understand-Anything → Flywheel Integration Review

> **Related:** Implementation strategy is in `.plans/ua-incorporation-strategy.md`.
> **Last updated:** 2026-05-30

## Executive Summary

Understand-Anything (UA) is a full-stack codebase analysis engine that produces
interactive knowledge graph dashboards. It combines tree-sitter static analysis,
LLM-powered summarization, and a React graph UI. **Flywheel should NOT port UA
wholesale** — the integration surface is too large and brings significant risks.
Instead, extract 3–4 targeted patterns that deliver the highest signal-to-noise
for general coding tasks.

---

## 1. Artifact Inventory (What UA Actually Produces)

| Artifact | Format | Location | Size (typical) |
|---|---|---|---|
| `knowledge-graph.json` | JSON nodes+edges+layers+tour | `.understand-anything/` | 50 KB–2 MB+ |
| `meta.json` | JSON analysis metadata | `.understand-anything/` | <1 KB |
| `fingerprints.json` | JSON structural signatures | `.understand-anything/` | 100 KB–5 MB |
| `domain-graph.json` | JSON domain model variant | `.understand-anything/` | 50 KB–500 KB |
| `intermediate/*.json` | Temp agent outputs | `.understand-anything/intermediate/` | 10 KB–500 KB each |
| Dashboard server | Vite dev server on localhost | launched by UA | Node.js process |

**Graph schema complexity**: 21 node types, 35 edge types across 8 categories,
plus `DomainMeta`, `KnowledgeMeta`, `TourStep`, `Layer`, `ProjectMeta`, and
21+ alias mappings to normalize LLM output. Zod validation = ~660 lines.

**Dependency count in UA core**: `fuse.js`, `web-tree-sitter` (WASM), 10+
language-specific tree-sitter configs, 8 framework detectors, 25+ type/edge
parsers, Zod for validation, d3-force for dashboard layout. Total: ~80 source
files in packages/core alone.

---

## 2. Risk Analysis

### 2.1 Context Bloat (HIGH RISK)

| Scenario | Problem | Impact |
|---|---|---|
| Full graph in system prompt | 50 KB–2 MB of JSON injected into every conversation | Token waste, cache invalidation, degraded response quality |
| Graph + edges + layers + tour | UA's `formatContextForPrompt()` serializes ALL matched nodes + ALL connected edges + ALL layers | 1-hop expansion of 15 seed nodes → 100–500 nodes easily in a medium repo |
| Always-on loading | If Flywheel auto-loads graph on every project entry, it adds 3–10 tokens per file permanently | Compounds across long conversations |
| Search results in context | UA's `context-builder.ts` dumps nodes, edges, layers as markdown tables | Each node ≈ 50–100 tokens; 50 nodes = 2,500–5,000 tokens |
| Stale graph in context | Serving outdated graph alongside new file reads creates conflicting context | Agent makes wrong decisions based on superseded analysis |

**Verdict**: The UA pattern of injecting graph-derived context into prompts is
useful for deep codebase questions but catastrophic as a default behavior for
general coding tasks.

### 2.2 Security Risks (MODERATE-HIGH)

| Risk | Detail | UA Mitigation | Flywheel Gap |
|---|---|---|---|
| Path leakage | Absolute disk paths embedded in graph nodes | `sanitiseFilePaths()` converts to relative | Must replicate this if Flywheel persists graphs |
| Dashboard file serve | `/file-content.json` endpoint serves arbitrary files | Gated by access token + graph-derived path allowlist | Flywheel has no web server — irrelevant unless adding dashboard |
| `.understand-anything/` in repo root | Persisted artifacts visible to all collaborators | N/A — by design | If Flywheel writes to `.gitignore`-excluded dirs, same issue |
| Intermediate artifacts | Agent outputs written to disk before graph assembly | Cleaned up after graph assembly | Must ensure no temp files leak into git or persist |
| `.env` exposure via graph | `.env` files may be classified as "config" nodes | UA's scanner has explicit `.env` handling but still includes them in graph | Flywheel must exclude secrets regardless of category |

**Verdict**: UA has reasonable path sanitization. The main risk is Flywheel
adopting the artifact storage pattern without matching the sanitization +
access-control discipline.

### 2.3 Maintenance Burden (HIGH)

| Aspect | Cost | Why |
|---|---|---|
| 21 node types × 35 edge types | Schema drift | Every new node type requires validation updates, alias mappings, dashboard UI changes |
| 10+ tree-sitter extractors | Parser maintenance | Tree-sitter grammar updates break extractors; WASM binaries need version pinning |
| Zod validation chain | Test maintenance | 660 lines of validation with auto-fix logic = fragile; LLMs generate unexpected shapes |
| Alias normalizers (100+ entries) | Hidden complexity | NODE_TYPE_ALIASES + EDGE_TYPE_ALIASES = 120+ mappings that drift silently |
| Dashboard dependency | React + Vite + React Flow + d3 + Zustand + Tailwind | Entire frontend stack for a visualization feature |
| Merge/update pipeline | Incremental graph updates | `mergeGraphUpdate()` + fingerprint comparison adds 300+ lines of change-detection logic |
| `web-tree-sitter` WASM | Build + deployment | WASM assets must be shipped; version incompatibility is common |

**Verdict**: The UA core is 5,000+ lines of specialized code. Adopting it as a
dependency or porting it means inheriting ~3 years of domain-specific bugfixes
and edge-case handling.

### 2.4 UX Risks (MODERATE)

| Risk | Scenario | Consequence |
|---|---|---|
| Analysis latency | UA scans can take 2–10 minutes on medium projects | User waits or gets frustrated; Flywheel becomes "slow" |
| Dashboard over-engineering | Full graph visualization vs. simple "what changed" questions | Users get a spaceship when they need a wrench |
| Staleness confusion | Graph says "function X exists" but file was refactored | Agent answers based on graph, user sees different code |
| Command proliferation | UA has `/understand`, `/understand-dashboard`, `/understand-chat`, `/understand-diff`, `/understand-explain`, `/understand-onboard` | UX fragmentation; users don't know which command to use |
| Hidden artifacts | `.understand-anything/` directory appears in file listings | Users confused by mysterious generated files |
| Always-on analyzer | Background scanning on every save | CPU/battery drain; users blame Flywheel for slowdown |

---

## 3. Anti-Patterns to Avoid

### ❌ Port the entire graph schema
The 21 node types and 35 edge types exist because UA supports wiki/article/claim/knowledge
nodes plus code, domain, and infrastructure. Flywheel only needs code comprehension.

### ❌ Inject graphs into system prompts
Never serialize the full graph or even a large subgraph into system context. Use
graphs as a retrieval index, not as context payload.

### ❌ Embed the React dashboard
A standalone React SPA for graph visualization is 3,000+ lines and introduces
a full frontend dependency tree. If Flywheel needs visualization, use a terminal
output or a lightweight static HTML page.

### ❌ Always-on background analysis
Running extractors on every file change creates I/O pressure and CPU usage. UA
does this only on explicit command invocation; Flywheel should do the same.

### ❌ Persist intermediate artifacts
UA's intermediate JSON files (agent outputs) are cleaned up after assembly.
Flywheel should not persist intermediate results at all — produce final output
and discard working data.

### ❌ Depend on web-tree-sitter
Tree-sitter WASM is powerful but introduces a brittle dependency. For Flywheel's
needs, simple regex/AST-light extraction may suffice for most languages.

### ❌ Complex alias normalization
The 120+ alias mappings exist because UA's LLMs generate non-canonical types.
If Flywheel controls the extraction (deterministic, not LLM-based), aliases are
unnecessary.

---

## 4. Patterns Worth Extracting

### ✅ Fingerprint-based change detection (HIGH VALUE)
UA's `fingerprint.ts` extracts structural signatures (function names, params,
return types, imports, exports) and compares them to determine whether changes
are structural vs. cosmetic. This is a 300-line, self-contained module that
solves the "do I need to re-analyze?" problem elegantly.

**Adaptation**: Port `extractFileFingerprint`, `compareFingerprints`, and
`ChangeLevel` enum. Drop the tree-sitter dependency — use simple regex
extraction for function/class signatures.

### ✅ Fuzzy search over extracted metadata (MODERATE VALUE)
UA's `search.ts` uses Fuse.js for fuzzy matching over node names, summaries,
tags, and language notes. This gives semantic-ish search without embeddings.

**Adaptation**: Lightweight search over file summaries + function/class names.
Could be a simple regex-first approach with optional Fuse.js as fallback.

### ✅ Deterministic file enumeration + classification (HIGH VALUE)
UA's `scan-project.mjs` replaces LLM-based scanning with a deterministic
extension/filename → language/category lookup table. This eliminates a slow,
expensive LLM call for every project scan.

**Adaptation**: Port the `LANGUAGE_BY_EXT`, `CATEGORY_BY_EXT`, and
`INFRA_FILENAMES` tables. These are data-only and directly reusable.

### ✅ Staleness detection (MODERATE VALUE)
UA's `staleness.ts` uses git diff to detect changes since the last analysis.
Combined with fingerprints, this enables incremental updates instead of full
re-scans.

**Adaptation**: Simple git-diff wrapper + commit hash tracking. Can be 30 lines.

---

## 5. Proposed Controls and Gates

### 5.1 Context Budget Gating
- **Rule**: Never inject more than N tokens of graph-derived context per turn
- **Implementation**: Hard cap at 30 nodes × 100 tokens ≈ 3,000 tokens maximum
- **Trigger**: Only activate graph context when user asks architecture-level questions
- **Default**: Off for ordinary edit/debug/read-file tasks

### 5.2 Artifact Size Caps
- **Rule**: Knowledge graph file must not exceed 2 MB
- **Enforcement**: Reject analysis of repos that produce graphs > 2 MB
- **Fallback**: Use file-level summaries only (no function/class granularity)

### 5.3 Staleness TTL
- **Rule**: Graph is valid for 24 hours or until the next commit, whichever comes first
- **Enforcement**: Compare `analyzedAt` + commit hash on each access
- **Override**: `flywheel scan --force` for fresh analysis

### 5.4 Security Gates
- **Path sanitization**: All stored paths must be relative to project root
- **Secret exclusion**: `.env`, `*credentials*`, `*secret*`, `*.pem` files
  are never graphed or indexed
- **Access control**: If a dashboard is built, it requires an ephemeral token
  (UA's pattern) and only serves files that appear as `filePath` in the graph

### 5.5 Dependency Gates
- **No React/SPAs**: Any visualization must be terminal-native or static HTML
- **No tree-sitter WASM**: Use regex-based extraction for MVP; evaluate native
  tree-sitter only if accuracy is insufficient
- **No external search engines**: Fuse.js is borderline acceptable (single-file,
  no native deps); embedding APIs are explicitly excluded
- **Maximum 2 new npm/pip packages**: Each new dependency must pass a
  value-to-weight analysis

### 5.6 User Experience Gates
- **Explicit invocation only**: Analysis runs only when user requests it
- **Progress feedback**: Show scan progress (% files analyzed) during execution
- **Clear opt-out**: `flywheel forget <project>` deletes all artifacts
- **No silent background work**: Analysis never runs without user awareness

---

## 6. Approval Plan

### Phase 1: Foundation (Week 1–2)
**Scope**: File enumeration, classification, fingerprint extraction (no graphs)

| Deliverable | Lines of Code | Dependencies | Risk |
|---|---|---|---|
| `scan_project()` — deterministic file enumeration | ~150 | None | Low |
| Language/category detection tables | ~100 | Data only | Low |
| Fingerprint extraction + comparison | ~200 | `hashlib` (stdlib) | Low |
| Staleness detection via git diff | ~40 | `subprocess` | Low |
| Unit tests for each module | ~300 | pytest | Low |

**Gate**: All tests pass. No new runtime dependencies added.

### Phase 2: Orchestration Layer (Week 3–4) — approval package
**Scope**: Make the Phase 1 scan scripts usable by agents without adding an always-on graph, dashboard, or heavyweight CLI surface.

> Reconciled 2026-05-30: the earlier “lightweight summaries + CLI” concept is deferred. The executable Phase 2 scope is the orchestration layer in `.plans/phase-2-flywheel-ua-integration.md`.

| Deliverable | Lines of Code | Dependencies | Risk |
|---|---|---|---|
| `scripts/code-scan/extract_imports.py` — regex import map from Phase 1 scan JSON | ~180 | stdlib only | Low |
| `skills/code-analysis/code-scan/SKILL.md` — JIT skill for scan orchestration + narrative synthesis | ≤80 lines | existing skill loader | Low |
| `skills/code-analysis/validation-gate/SKILL.md` — deterministic validation result interpreter | ≤80 lines | existing skill loader | Low |
| Optional `requesting-code-review` integration — opt-in only, no default review slowdown | docs/skill patch only unless approved | none | Moderate |
| Tests and fixtures for script + skill contracts | ~250 | pytest | Low |

**Gate**: Phase 1 exists and passes first; Phase 2 remains explicit-invocation/JIT only; both SKILL.md files together load ≤100 lines of prompt context; import extraction and validation run with zero new runtime dependencies.

### Phase 3: Graph Lite (Week 5–6, APPROVAL GATE)
**Scope**: Minimal graph for architecture questions only

| Deliverable | Lines of Code | Dependencies | Risk |
|---|---|---|---|
| Node types: file, function, class, module only (4 of 21) | ~100 | None | Low |
| Edge types: imports, contains, calls only (3 of 35) | ~50 | None | Low |
| Graph persistence (JSON, <2 MB cap) | ~100 | `json` (stdlib) | Low |
| Context builder (max 30 nodes, 1-hop edges) | ~150 | None | Moderate |
| `flywheel ask` command that uses graph context | ~200 | — | Moderate |

**Gate**: Graph files never exceed 2 MB. Context injection never exceeds
3,000 tokens. No performance regression on baseline coding tasks.

### Phase 4: Incremental Updates (Week 7–8)
**Scope**: Avoid full re-scans on subsequent analyses

| Deliverable | Lines of Code | Risk |
|---|---|---|
| Detect changed files via fingerprints | ~80 | Low |
| Merge-only graph updates | ~120 | Moderate |
| Invalidation tracking | ~60 | Low |

**Gate**: Incremental update takes <10% of initial scan time.

---

## 7. What to Reject (Out of Scope)

| Feature | Reason | Alternative |
|---|---|---|
| Full 21-node-type schema | Over-engineered, most types irrelevant to coding | 4 types max for Phase 3 |
| React dashboard | 3,000+ lines + heavy deps | Terminal output or static HTML |
| LLM-based file analysis | Slow, non-deterministic, expensive | Regex + AST-light extraction |
| Domain graphs | Separate concern from code comprehension | Future skill, not Flywheel |
| Tour/guide generation | Nice-to-have, adds 100+ lines | Manual user guidance |
| Knowledge/wiki nodes | Different use case entirely | Separate tool |
| Embedding search | Requires API + storage + vector DB | Fuzzy search is sufficient |
| Auto-diff overlay | Complex state management | Simple `flywheel diff` command |
| web-tree-sitter WASM | Build complexity + deployment burden | Regex extraction first |
| Alias normalization (120+ entries) | Solves LLM output variability | Deterministic extraction = no aliases needed |

---

## 8. Decision Matrix

| Pattern | Extract? | Priority | Risk if Adopted | Mitigation |
|---|---|---|---|---|
| Deterministic file scanner | ✅ Yes | P0 | Low | Direct port, data-only |
| Fingerprint change detection | ✅ Yes | P0 | Low | Port core logic, simplify tree-sitter dep |
| Fuzzy search over metadata | ✅ Yes | P1 | Low | Optional Fuse.js or regex-first |
| Staleness detection | ✅ Yes | P1 | Low | Git diff wrapper |
| Knowledge graph schema | ⚠️ Reduced | P2 | High | 4 node types, 3 edge types only |
| Context builder for prompts | ⚠️ Gated | P2 | High | 30-node max, opt-in only |
| Dashboard | ❌ No | — | High | Terminal output instead |
| LLM-based summarization | ❌ No | — | High | Deterministic extraction |
| Full merge pipeline | ⚠️ Deferred | P3 | Moderate | Phase 4 only |
| Alias normalization | ❌ No | — | Moderate | Unneeded with deterministic extraction |

---

## 9. Summary Recommendation

1. **Extract, don't port**: Take the file scanner, fingerprint system, staleness
   check, and optional fuzzy search as standalone Python modules (~650–800 LOC
   total).

2. **Reject the graph dashboard complex**: The React SPA, full schema, LLM
   analyzers, and merge pipeline add 10× the maintenance cost for 2× the value.

3. **Gate everything behind explicit invocation**: Flywheel analysis commands
   should never run automatically. Users opt-in per-project.

4. **Enforce hard limits**: 2 MB max graph, 30-node context cap, 30s scan time
   for repos ≤500 files, zero new heavy dependencies.

5. **Validate at each phase**: Each phase has explicit gates. Phase 3 (graph)
   requires explicit approval before starting — don't let scope creep happen.
