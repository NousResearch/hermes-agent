## Community Memory Providers

Beyond the 8 built-in providers, the following **standalone memory provider plugins** are available as community-maintained packages. These integrate with Hermes via the same `MemoryProvider` ABC and plugin discovery system — install them from pip or clone from their repos, and Hermes picks them up automatically.

### Memex Zero RAG

[Memex Zero RAG](https://github.com/JPeetz/MeMex-Zero-RAG) is a **knowledge compilation system** using the Zero-RAG pattern (Karpathy's LLM Wiki). Unlike traditional providers that store individual facts as vector embeddings, Memex compiles sources into a structured, cross-referenced wiki that compounds knowledge over time.

| | |
|---|---|
| **Best for** | Deep research, multi-session project knowledge, anti-hallucination-critical workflows, teams that want git-native knowledge management |
| **Requires** | `pip install memex-hermes` + a running Memex MCP server or local wiki directory |
| **Data storage** | Local filesystem (git repository) |
| **Cost** | Free (open-source, Apache 2.0) |
| **Tools** | `memex_search` (semantic + keyword hybrid search), `memex_ingest` (add sources), `memex_read` (read wiki concepts), `memex_status` (wiki health) |

**Key differentiators:**

- **Anti-hallucination protocol** — every claim in the wiki has a mandatory `[Source: path]` citation. Pages with >20% unsourced content are auto-quarantined.
- **Knowledge decay** — confidence scoring with time-based decay, revalidation queue, and heat-map prioritization.
- **Conflict detection** — dedicated contradiction tracking with human-in-the-loop resolution. Immutable nodes with dependency taint propagation.
- **Local-first, no API key required** — all processing is local. The wiki is a directory of markdown files in a git repo.
- **Multi-modal ingestion** — PDF with OCR, voice via Whisper, web clipping, markdown.
- **OKF-compatible** — Memex already implements a superset of Google's Open Knowledge Format (OKF) v0.1 conventions.
- **Hybrid search** — BM25 keyword + semantic search with configurable weighting.
- **Knowledge graph** — interactive vis.js force-directed graph visualization.
- **MCP server** — SSE transport for multi-agent access (Hermes, OpenClaw, Claude Code can share one wiki).

**Setup:**

```bash
# Install the Memex Hermes plugin
pip install memex-hermes

# Or clone from source
git clone https://github.com/JPeetz/memex-hermes-plugin.git
cd memex-hermes-plugin
pip install -e .

# Set up Memex Zero RAG (if not already running)
git clone https://github.com/JPeetz/MeMex-Zero-RAG.git
cd MeMex-Zero-RAG
pip install -e .
python mcp/server.py --transport sse --port 3001

# Configure Hermes
hermes memory setup  # select "memex"
```

**Config file:** `$HERMES_HOME/memex.json`

| Key | Default | Description |
|---|---|---|
| `endpoint` | `http://localhost:3001` | Memex MCP server URL |
| `wiki_path` | `~/MeMex-Zero-RAG/wiki/` | Path to the Memex wiki directory |
| `max_recall` | `10` | Max items injected into context per turn |
| `search_results` | `10` | Max results for memex_search tool |

**Plugin source:** [github.com/JPeetz/memex-hermes-plugin](https://github.com/JPeetz/memex-hermes-plugin)

To add your own community memory provider, publish a standalone plugin following the [Memory Provider Plugin guide](/developer-guide/memory-provider-plugin) and open a PR to add it to this section.