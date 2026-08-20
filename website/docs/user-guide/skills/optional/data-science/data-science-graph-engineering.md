---
title: "Graph Engineering — Build knowledge graphs and agent task graphs"
sidebar_label: "Graph Engineering"
description: "Build knowledge graphs and agent task graphs"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Graph Engineering

Build knowledge graphs and agent task graphs.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/data-science/graph-engineering` |
| Path | `optional-skills/data-science/graph-engineering` |
| Version | `1.0.0` |
| Author | codejunkie99 (ported to Hermes by Hermes Agent) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `knowledge-graph`, `graphrag`, `ontology`, `extraction`, `orchestration`, `data-science` |
| Related skills | [`gitnexus-explorer`](/docs/user-guide/skills/optional/research/research-gitnexus-explorer), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Graph Engineering

Use when asked to build a knowledge graph, extract entities/relations/events from text, design
an ontology, dedupe/merge entities, add graph memory or GraphRAG to an agent, orchestrate
multi-agent workflows as a task graph, or teach graph engineering (teaching mode explains each
stage with worked examples and visual diagram artifacts).

> **Hermes adaptation notes** (read once, then follow the original body below):
> - **Task-graph half → Hermes primitives.** The diamond pattern maps directly onto
>   `delegate_task` batch mode (parallel workers) with a separate verifier task — never let a
>   worker grade its own output in its own context. The plan node is your `todo` list; the
>   human gate is Hermes' approval flow (route irreversible actions — send, publish, delete,
>   deploy — through the user, not around them). "One writer per file" and loop caps apply
>   verbatim to subagent fan-outs.
> - **Small-scale storage default.** For stage 2 at agent scale, plain typed edges in
>   JSON/JSONL or SQLite (queried via `execute_code`) beats standing up Neo4j — reach for a
>   graph database only when multi-hop query volume justifies it.
> - **Diagrams.** Teaching mode's visual artifacts work well as mermaid blocks in chat, or as
>   Excalidraw/self-contained HTML files for keepable artifacts (see the `excalidraw` skill).
> - **Fusion pitfall (stage 8).** Initial-letter acronym matching misses intra-word acronyms —
>   "SEU" for "So·uth·east University" yields "SU" under a naive first-letters rule. Use alias
>   lists, subsequence matching, or LLM adjudication for acronym candidates, not initial-letter
>   rules alone.
> - **Quality gate at pilot scale (stage 7).** "≥90% precision on a 50-item sample" means
>   *all* items when the pilot has fewer than 50.
> - Ported from [codejunkie99/graph-engineering](https://github.com/codejunkie99/graph-engineering)
>   (MIT), itself distilled and translated from Southeast University's graduate Knowledge
>   Graph course (npubird/KnowledgeGraphCourse, Prof. Peng Wang).

Graph engineering is the discipline of designing the structures agents work through — not the
prompts. It has two halves:

1. **Knowledge graphs** — what agents remember. Nodes are entities and facts, edges are
   relationships with time and provenance. This file's 9-stage pipeline covers it, distilled
   from Southeast University's graduate KG course
   (https://github.com/npubird/KnowledgeGraphCourse, Prof. Peng Wang), translated to English
   and adapted for LLM-era agents.
2. **Task graphs** — how agents work. Nodes are jobs, edges are execution dependencies:
   parallel fan-out, separate verifier contexts, the stop rule, the human gate.
   Read [references/task-graphs.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/task-graphs.md) when the request is about
   orchestrating agents rather than building memory.

Core mental model: a knowledge graph is a **product with a schema**, not a pile of triples.
Quality comes from the pipeline order — model the domain BEFORE extracting, fuse BEFORE storing,
evaluate at every stage.

## Teaching Mode

When the user wants to LEARN graph engineering (rather than build something), teach it — do
not just execute. Rules:

1. Anchor every stage in the user's own domain: ask for one real project or dataset, then use
   it as the running example through all stages.
2. **Generate visual artifacts as you teach.** Concepts in this discipline are shapes; show
   them. For each major concept, produce a small diagram the user can keep — mermaid diagrams
   (flowchart for the pipeline and task graphs, `graph LR` for example ontologies and
   subgraphs) or a single self-contained HTML page when interactivity helps. At minimum:
   the 9-stage pipeline, a 3-type ontology drawn from the user's domain, one extracted
   subgraph (5-10 nodes) from a real sample, and the diamond pattern with the user's own jobs
   as nodes.
3. Teach in the pipeline's order, one stage per exchange, each ending with a small exercise
   ("write 3 competency questions for your project") before moving on.
4. Close by assembling what was built during the lesson into a starter `ontology.yaml` and a
   drawn task graph for the user's first real build.

## The 9-Stage Pipeline

Run stages in order. For small projects stages 4-6 collapse into one extraction pass, but never
skip stages 3 (ontology) or 8 (fusion) — they are where real-world graphs fail.

1. **Scope & value test** — Confirm a graph beats a simpler structure. A graph pays off when
   queries are multi-hop ("who worked with X on projects using Y"), when entities recur across
   documents, or when relationships ARE the data. If lookups are single-hop, use a table and stop.

2. **Knowledge representation choice** — Pick how facts are encoded: property graph
   (Neo4j-style, pragmatic default), RDF triples (interop/standards), or plain typed edges in
   JSON/SQLite (small scale). Decide now how time and provenance attach to every fact.

3. **Ontology modeling** — Define entity types, relation types (with domain/range), and
   attributes BEFORE extraction. Start minimal: 5-15 entity types, 10-30 relation types.
   Two rules from the course: every relation gets a precise verb name (`ACQUIRED`, not
   `RELATED_TO`), and if two types are always queried together, merge them.
   Details and worked examples: [references/modeling.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/modeling.md)

4. **Entity extraction (NER)** — Extract typed entities from sources. Method ladder: exact
   rules/dictionaries for closed vocabularies → LLM extraction with the ontology in the prompt
   for open text. Always extract with span + source pointer for provenance.

5. **Relation extraction** — Extract typed edges between recognized entities. Constrain the
   LLM to the ontology's relation list with domain/range checks; reject edges whose endpoints
   have incompatible types. This one validation step removes most hallucinated structure.

6. **Event extraction** — For dynamic domains (news, logs, transactions), extract events as
   first-class nodes (trigger + typed arguments + time), not just static edges.
   Extraction methods, prompt patterns, and failure modes for stages 4-6:
   [references/extraction.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/extraction.md)

7. **Quality gate** — Before fusion, sample and score: entity precision (are extracted
   entities real and correctly typed?), relation precision (does the source sentence actually
   assert the edge?). Fix the prompt/rules, not the output, then re-run. Target ≥90% precision
   on a 50-item sample before proceeding — recall improves with more passes; bad precision
   poisons the graph permanently.

8. **Knowledge fusion** — Merge duplicates within and across sources: same real-world entity,
   different surface forms ("SEU" = "Southeast University" = "东南大学"). Blocking + matching +
   merge policy. Skipping this is the #1 cause of useless graphs.
   Matching strategies: [references/fusion-and-llm.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/fusion-and-llm.md)

9. **Serve to LLMs (KG × LLM)** — Make the graph useful to agents: GraphRAG retrieval
   (subgraph → context), graph-as-memory (agent writes facts back through stages 4-8), and
   LLM-as-reasoner over paths. Patterns and pitfalls:
   [references/fusion-and-llm.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/fusion-and-llm.md)

## Working Rules

- **Schema first, always.** Extraction without an ontology produces a "graph" that is really a
  word cloud with arrows. If the user resists schema design, build the minimal 5-type ontology
  from 3 sample documents and show it for approval.
- **Provenance on every fact.** Each node/edge stores `source`, `extracted_at`, and confidence.
  Non-negotiable — fusion (stage 8) and trust both depend on it.
- **Incremental over big-bang.** Process a 10-document pilot through all 9 stages before
  scaling. The pilot exposes ontology gaps at 1% of the cost.
- **LLM extraction is stage machinery, not the pipeline.** The LLM slots into stages 4-6;
  the surrounding schema, validation, and fusion are what make the output a knowledge graph.

## Reference Files

- [references/curriculum.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/curriculum.md) — Full translated curriculum of the
  source course with per-lecture summaries and links to the original Chinese slide decks.
  Read when the user wants theory depth, the academic grounding, or the original materials.
- [references/modeling.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/modeling.md) — Knowledge representation & ontology
  engineering (course lectures 2-3). Read during stages 2-3.
- [references/extraction.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/extraction.md) — Entity, relation, and event
  extraction from rules to LLM prompting (lectures 4-7). Read during stages 4-7.
- [references/fusion-and-llm.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/data-science/graph-engineering/references/fusion-and-llm.md) — Knowledge fusion and
  KG × LLM integration (lectures 8-9). Read during stages 8-9.

## Credits

Ported from [codejunkie99/graph-engineering](https://github.com/codejunkie99/graph-engineering)
(MIT). That project is distilled and translated from 东南大学《知识图谱》研究生课程 (Southeast
University graduate course on Knowledge Graphs), Prof. Peng Wang —
https://github.com/npubird/KnowledgeGraphCourse. All original lecture PDFs are in Chinese;
the skill is an independent English distillation adapted for AI-agent workflows.
