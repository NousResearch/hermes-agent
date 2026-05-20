# Obsidian Vector MCP

Stdio MCP server that exposes `search_obsidian` — semantic search over a
pre-built SQLite vector index of your Obsidian vault / LLM Wiki.

## Prerequisites

1. Backend script at `~/.hermes/scripts/llm_wiki_vector_search.py`
2. Ollama running with `mxbai-embed-large` (or another embedding model)
3. Index built: `python ~/.hermes/scripts/llm_wiki_vector_search.py index --wiki ~/llm-wiki`

## MCP client registration

**claude_desktop_config.json / Cursor / Codex:**
```json
{
  "mcpServers": {
    "obsidian-vector": {
      "command": "python",
      "args": ["/path/to/hermes-agent/scripts/obsidian_vector_mcp.py"]
    }
  }
}
```

**Hermes `~/.hermes/config.yaml`:**
```yaml
mcp_servers:
  obsidian-vector:
    command: python
    args:
      - /path/to/hermes-agent/scripts/obsidian_vector_mcp.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OBSIDIAN_VECTOR_BACKEND` | `~/.hermes/scripts/llm_wiki_vector_search.py` | Path to backend script |
| `LLM_WIKI_VECTOR_INDEX` | `~/.hermes/indexes/llm-wiki-vector.sqlite` | SQLite index path |
| `LLM_WIKI_EMBEDDING_BASE_URL` | `http://127.0.0.1:11434/v1` | Embedding endpoint |
| `LLM_WIKI_EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model name |
| `LLM_WIKI_EMBEDDING_MODE` | `ollama` | `ollama` or `hash` (offline/deterministic) |

## Tool: `search_obsidian`

```
search_obsidian(query, limit=8, source=None, include_content=False) -> str
```

Returns JSON:
```json
{
  "query": "attention mechanism",
  "count": 3,
  "results": [
    {
      "title": "Attention Mechanism",
      "path": "transformers/attention.md",
      "heading": "Attention Mechanism",
      "chunk_index": 0,
      "snippet": "Transformers rely on self-attention…",
      "score": 0.923,
      "source": "llm-wiki",
      "metadata": {"model": "mxbai-embed-large", "index": "/path/to/index.sqlite"},
      "content": "..."  // only when include_content=true
    }
  ]
}
```

## Smoke test (no Ollama needed)

```bash
LLM_WIKI_EMBEDDING_MODE=hash python scripts/obsidian_vector_mcp.py --smoke-test
```
