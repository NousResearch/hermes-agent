---
sidebar_position: 5
title: "LLM Wiki Memory"
description: "Local-first, source-backed durable memory for Hermes Agent"
---

# LLM Wiki Memory

LLM Wiki is a local-first memory provider that stores durable knowledge as an inspectable Markdown wiki and indexes it for semantic retrieval. It is designed for knowledge you want to curate, cite, edit, back up, diff, and publish like documentation.

It is intentionally **not** an automatic transcript summarizer. Normal conversation turns are not written into the wiki. Hermes can search and read the wiki during chat, but durable writes require explicit tool calls and are blocked in unsafe contexts. Use it when you want source-backed memory that stays under your control instead of a hidden hosted memory database.

## When to use it

Use LLM Wiki for:

- project architecture and operating manuals
- durable policy and conventions
- research notes and source-backed conclusions
- memory that should remain inspectable in Git or a folder backup
- local/offline-first deployments

Use other memory paths for other jobs:

- **Built-in `MEMORY.md` / `USER.md`**: tiny boot-critical facts and preferences.
- **Session search**: chronological recall of past conversations.
- **Honcho/Mem0/Supermemory/etc.**: automatic user modeling or hosted semantic memory.

## Install

LLM Wiki's provider package is bundled with Hermes. Heavy vector-store dependencies are optional and lazy-loaded so normal Hermes installs stay small. The `llm-wiki` extra installs Qdrant plus Hermes's shared vector infrastructure dependency, `vector-core`, from a commit-pinned public GitHub ref until `vector-core` is published on PyPI.

For an explicit install, use the optional dependency group:

```bash
pip install 'hermes-agent[llm-wiki]'
```

If you only configure the provider, Hermes can still discover it without `qdrant-client` or `vector-core` installed. The first real engine/tool use will ask the lazy-dependency system to install the pinned optional dependencies, unless lazy installs are disabled in your security config.

LLM Wiki also expects:

- a Qdrant HTTP endpoint, default `http://localhost:6333`
- an OpenAI-compatible embedding endpoint, default `http://localhost:22222`
- an OpenAI-compatible chat endpoint for `wiki_query`, default `http://localhost:8011/v1`

For a local Qdrant server:

```bash
docker run -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## Configure

Run the memory setup wizard and select `llm_wiki`:

```bash
hermes memory setup
```

Manual `~/.hermes/config.yaml` example:

```yaml
memory:
  provider: llm_wiki

wiki:
  path: ~/.hermes/wiki/personal
  name: personal
  embedding:
    url: http://localhost:22222
    model: Qwen3-Embedding-8B
    dim: 4096
    # Optional; defaults to <wiki path>/.cache/embeddings.sqlite3
    cache_path: ~/.hermes/wiki/personal/.cache/embeddings.sqlite3
    cache_max_entries: 100000
  vector_store:
    url: http://localhost:6333
    collection_prefix: hermes_wiki
  llm:
    url: http://localhost:8011/v1
    model: gpt-5.5
```

Do not put secrets in wiki pages. If an endpoint needs an API key, keep it in `.env` or your provider-specific config; do not commit it with the wiki.

:::tip Local privacy profile
For sensitive personal or company memory, point both embedding and chat URLs at local OpenAI-compatible services. Search indexing sends wiki/source text to the embedding endpoint, and `wiki_query` sends retrieved snippets plus your question to the configured chat endpoint. Local endpoints keep that loop on your machine.
:::

## Wiki layout

A wiki is a folder with Markdown files:

```text
~/.hermes/wiki/personal/
├── SCHEMA.md
├── index.md
├── log.md
├── entities/
├── concepts/
├── comparisons/
├── queries/
└── raw/
    ├── articles/
    ├── papers/
    └── transcripts/
```

Raw sources live under `raw/`. Generated or curated pages cite those sources with provenance markers such as:

```markdown
This fact came from a source. ^[raw/articles/example.md]
```

## Tools

When enabled, the provider exposes these tools:

| Tool | Purpose | Mutates durable memory? |
| --- | --- | --- |
| `wiki_status` | Show wiki path, page counts, collection stats, recent activity | No |
| `wiki_orient` | Read orientation/index information | No |
| `wiki_search` | Semantic search over indexed wiki chunks | No |
| `wiki_read` | Read a wiki page by slug or relative path | No |
| `wiki_query` | Ask an LLM-backed question against wiki context | No by default |
| `wiki_lint` | Validate links, provenance, source hashes, vector health | No by default |
| `wiki_ingest` | Ingest a curated source file | Dry-run by default; primary context only |
| `wiki_reindex` | Rebuild vector index and index.md | Yes; gated |

Safe defaults:

- `wiki_query`: `file_result=false`, `log_query=false`
- `wiki_lint`: `write_log=false`
- `wiki_ingest`: `dry_run=true`; blocked outside the primary agent context because even dry-runs read local files and submit their content for analysis
- `wiki_search.limit` is clamped to a small bounded range
- `wiki_read` rejects path traversal and truncates oversized pages
- `sync_turn()` is a no-op; chat turns are not automatically canonized
- prefetch is bounded, cited, and retrieval-only
- writes are blocked outside the primary agent context

## Operational workflow

1. Put a source document somewhere on disk.
2. Ask Hermes to dry-run ingestion first:

   ```text
   Use wiki_ingest on /path/to/source.md with dry_run=true.
   ```

3. Review the proposed ingest.
4. If it is safe and intentional, run the actual ingest from a primary Hermes session:

   ```text
   Ingest /path/to/source.md into LLM Wiki with dry_run=false.
   ```

5. Run lint and reindex when needed:

   ```text
   Run wiki_lint with write_log=false, then wiki_reindex if the index is stale.
   ```

6. Keep a small retrieval eval file for important recall expectations and run it after changes:

   ```yaml
   # ~/.hermes/wiki/personal/evals/retrieval.yaml
   cases:
     - query: How autonomous should Hermes be?
       expected_pages:
         - concepts/user-autonomy-operating-policy.md
         - entities/hermes.md
       top_k: 5
   ```

   ```bash
   hermes-wiki-eval ~/.hermes/wiki/personal/evals/retrieval.yaml --config ~/.hermes/config.yaml --pretty
   # or: python -m hermes_wiki.eval ...
   ```

   `--config` is authoritative when supplied: the file must exist and contain the intended `wiki:` settings; the runner will not silently fall back to another profile/config.

   Retrieval evals are read-only. They validate that expected page paths appear in search results; they do not generate answers, write wiki pages, ingest sources, or reindex vectors. They still embed eval query text and read from the configured vector store, so use local/private endpoints for sensitive eval cases.

7. Inspect retrieval hits for one query when evals fail or recall looks suspicious:

   ```bash
   hermes-wiki-introspect "How autonomous should Hermes be?" \
     --config ~/.hermes/config.yaml \
     --expected-page concepts/user-autonomy-operating-policy.md
   hermes-wiki-introspect "What should Hermes call the user?" --config ~/.hermes/config.yaml --json
   hermes-wiki-introspect "Qwen3-Embedding-8B" --config ~/.hermes/config.yaml --search-mode hybrid --json
   ```

   Retrieval introspection is read-only. It reports the original query, search mode, top-k, ranked chunk hits, scores, page/source paths, deduplicated page coverage, and missing expected pages. `--search-mode` can be `dense` (default/backward compatible), `sparse` (lexical payload matching for literal names, commands, config keys, and paths), or `hybrid` (dense+sparse Reciprocal Rank Fusion). It does not generate answers, write wiki pages, ingest sources, reindex vectors, log queries, or queue proposals. Like evals, dense/hybrid modes may embed the inspected query and read from the configured vector store; sparse mode scans indexed payload text.

8. Draft proposed durable-memory updates without automatically ingesting them:

   ```bash
   hermes-wiki-propose \
     --title "Remember a stable preference" \
     --rationale "Prevents repeated correction" \
     --change "Add the preference to the relevant source-backed page" \
     --source "discord:<message-or-thread-id>" \
     --target concepts/example.md
   ```

   By default this prints markdown only. Add `--queue --config ~/.hermes/config.yaml` to explicitly write a pending review artifact under `<wiki>/proposals/`. `--queue` requires `--config` and fails closed rather than falling back to an ambient/default wiki path. Queueing does not ingest, reindex, or mutate canonical wiki pages.

9. Review queued proposal artifacts explicitly:

   ```bash
   hermes-wiki-proposals --config ~/.hermes/config.yaml list
   hermes-wiki-proposals --config ~/.hermes/config.yaml show <slug>
   hermes-wiki-proposals --config ~/.hermes/config.yaml accept <slug> --note "Applied through curated ingest"
   hermes-wiki-proposals --config ~/.hermes/config.yaml reject <slug> --note "Not durable enough"
   hermes-wiki-proposals --config ~/.hermes/config.yaml close <slug>
   ```

   The lifecycle command manages proposal review artifacts only. `accept`, `reject`, and `close` require explicit `--config` and update the proposal frontmatter status under `<wiki>/proposals/`; they do not edit canonical wiki pages, ingest sources, or reindex vectors. Accepting a proposal means the reviewed artifact has been accepted by a separate curated workflow, not that this command silently applied target-page edits.

10. Generate a read-only maintenance report:

   ```bash
   hermes-wiki-maintenance --config ~/.hermes/config.yaml
   hermes-wiki-maintenance --config ~/.hermes/config.yaml --json
   ```

   This checks broken wikilinks, orphan pages, pending proposals, and pages without source coverage. By default it only prints a report. `--write-report reports/maintenance.md` requires explicit `--config` and writes only under the dedicated `reports/` namespace; it does not ingest sources, reindex vectors, or mutate canonical entity/concept pages.

11. Run the agent-native caretaker loop for cron/watchdog-style memory health:

   ```bash
   hermes-wiki-caretaker --config ~/.hermes/config.yaml
   hermes-wiki-caretaker --config ~/.hermes/config.yaml --quiet
   hermes-wiki-caretaker --config ~/.hermes/config.yaml --json
   ```

   The caretaker combines maintenance checks with retrieval evals from `<wiki>/evals/retrieval.yaml` when present, then classifies next actions for Hermes (`review_pending_proposal`, `repair_broken_link`, `fix_retrieval_regression`, etc.). It is read-first and agent-native: no ingest, no reindex, no canonical page mutation, no query logging, and no chat-model calls. `--quiet` prints nothing when there are no blockers, making it suitable for watchdog scripts that only alert on retrieval regressions or hard maintenance errors. `--write-report reports/caretaker.md` requires explicit `--config` and writes only under `reports/`.

12. Draft proposal artifacts from caretaker findings:

   ```bash
   hermes-wiki-caretaker-propose --config ~/.hermes/config.yaml
   hermes-wiki-caretaker-propose --config ~/.hermes/config.yaml --json
   hermes-wiki-caretaker-propose --config ~/.hermes/config.yaml --queue --json
   ```

   The caretaker proposal orchestrator turns actionable caretaker findings into reviewable `MemoryProposal` drafts: broken-link repairs, graph-link strengthening, missing-source evidence work, and retrieval-regression fixes. It prints drafts by default. `--queue` requires explicit `--config` and writes only under `<wiki>/proposals/`; it still does not edit canonical pages, ingest sources, or reindex vectors.

## Safety model

LLM Wiki treats durable memory like a curated knowledge base, not a scratchpad.

- No automatic writes from normal conversation flow.
- No whole-wiki prompt dumps.
- No path traversal in `wiki_read`.
- Durable write tools are blocked in cron/subagent/batch/compression/retrieval contexts.
- Ingest source categories are allowlisted before they become paths.
- Reindexing upserts replacement vectors before deleting stale chunks, so embedding failures do not wipe existing search results.
- Dense embedding calls use vector-core's persistent SQLite embedding cache and OpenAI-compatible embedding client. Hermes namespaces cache keys by model, embedding dimension, and text content hash; cache values are JSON-serialized and LRU-evicted by `wiki.embedding.cache_max_entries`.
- Raw sources are hash-checked by lint so accidental drift is visible.
- Markdown pages are human-readable and can be reviewed with normal Git tools.

## Troubleshooting

**Provider not available**

Install the extra and restart Hermes:

```bash
pip install 'hermes-agent[llm-wiki]'
```

**Search fails**

Check Qdrant and embedding endpoint health. The embedding URL should be OpenAI-compatible; Hermes normalizes a URL like `http://localhost:22222` to `http://localhost:22222/v1` internally.

**Lint reports empty vector index**

Run `wiki_reindex` from a primary agent context after confirming Qdrant and the embedding endpoint are available.

**Writes are blocked**

This is expected in cron, subagents, batch jobs, and retrieval-only contexts. Re-run the operation from a primary interactive agent session, or keep `dry_run=true` for inspection.
