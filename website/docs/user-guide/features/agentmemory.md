---
sidebar_position: 100
title: "AgentMemory"
description: "Persistent cross-agent memory for Hermes via agentmemory, MCP, and the optional memory provider plugin"
---

# AgentMemory

[agentmemory](https://github.com/rohitg00/agentmemory) is a persistent memory engine for AI coding agents. It captures what agents do across sessions, compresses that activity into searchable memory, and retrieves the right context when a future session starts.

For Hermes users, the most important difference is that agentmemory is not only a place to store notes. It is a local memory service that can be shared by multiple coding agents:

- Hermes
- Claude Code
- Cursor
- Gemini CLI
- OpenCode
- Codex CLI
- Cline
- Aider
- Any MCP or REST-capable agent

:::info External integration
AgentMemory is an external project that integrates with Hermes through MCP today. A deeper Hermes memory-provider plugin is available from the agentmemory repository and is being discussed for upstream integration.
:::

## When to Use AgentMemory

Use AgentMemory when you want Hermes to remember project work across many sessions or share coding memory with other agents.

Good use cases:

- You switch between Hermes, Claude Code, Cursor, OpenCode, and other tools on the same repository.
- You keep re-explaining project architecture, test commands, framework choices, and past bug fixes.
- Your `MEMORY.md`, `USER.md`, `CLAUDE.md`, `.cursorrules`, or memory-bank files are becoming too large.
- You want semantic recall, not only keyword search.
- You want a local, self-hosted memory server with a viewer.
- You want memory lifecycle features such as versioning, decay, contradiction handling, and auto-forget.

AgentMemory is especially useful for long-running software projects where the useful context is not only who the user is, but what happened inside the codebase:

```text
Session 1: "Add auth to the API"
  Hermes creates JWT middleware, chooses jose, adds tests, fixes failures.
  AgentMemory captures the files touched, commands run, errors seen, and decisions made.

Session 2: "Now add rate limiting"
  Hermes can retrieve:
    - Auth uses jose middleware in src/middleware/auth.ts
    - Token validation tests live in test/auth.test.ts
    - jose was chosen over jsonwebtoken for Edge compatibility
    - The API already has an auth middleware pattern to reuse

Result:
  No re-explaining. No copy-pasting old logs. Hermes starts from project memory.
```

## How It Complements Hermes Memory

Hermes already has several memory and context systems:

| Hermes feature | What it is good for |
|---|---|
| `USER.md` | Durable facts about the human: preferences, writing style, expectations |
| `MEMORY.md` | Compact environment facts, project conventions, and lessons |
| Session search | Finding older conversation transcripts |
| Skills | Reusable procedures that improve over time |
| Context files | Explicit project instructions such as `AGENTS.md` |
| Memory providers | External memory backends such as Honcho |

AgentMemory adds a separate layer focused on searchable, cross-agent, project-level memory:

| Hermes built-in | AgentMemory adds |
|---|---|
| Flat memory files | Structured observations with files, commands, facts, concepts, and outcomes |
| Keyword-oriented session search | BM25 + vector + knowledge graph retrieval |
| One Hermes profile/session at a time | Shared memory across Hermes and other agents |
| Manual memory curation | Automatic capture through hooks, MCP tools, and REST calls |
| Context loaded into prompt | Token-budgeted retrieval of only relevant memories |

Think of the built-in Hermes memory as curated notes and preferences. Think of AgentMemory as the searchable project memory database behind those notes.

## AgentMemory vs Honcho

Honcho and AgentMemory solve different memory problems.

| Capability | Honcho | AgentMemory |
|---|---|---|
| Primary focus | Deep user modeling and personalization | Project and coding-agent memory |
| Best for | Understanding the human across conversations | Remembering codebase work across sessions and tools |
| Context type | User representations, peer cards, conclusions, session summaries | Observations, files, commands, decisions, bugs, fixes, lessons |
| Multi-agent model | Separate peers in a shared workspace | Shared memory service across MCP/REST-capable agents |
| Retrieval style | Semantic search over conclusions and context | BM25 + vector + graph search over observations |
| Setup style | Hermes memory provider | MCP server today, optional memory-provider plugin |
| Storage | Honcho Cloud or self-hosted Honcho | Local/self-hosted AgentMemory server by default |

Use Honcho when you want Hermes to understand the user better.

Use AgentMemory when you want Hermes and other coding agents to remember what happened in a repository.

You can also use them for different layers of memory: Honcho for user modeling, AgentMemory for project recall.

## Reported Benchmarks

AgentMemory reports the following results in its public benchmark docs:

| Metric | Reported value |
|---|---:|
| LongMemEval-S retrieval R@5 | 95.2% |
| LongMemEval-S retrieval R@10 | 98.6% |
| MRR | 88.2% |
| Token reduction vs loading large built-in memory | Up to 92% |
| External database required | 0 |
| Default REST API | `http://localhost:3111` |
| Real-time viewer | `http://localhost:3113` |

:::note Benchmark caveat
These numbers come from AgentMemory's benchmark reports. Compare memory benchmarks carefully because projects often use different datasets, embedding models, corpus sizes, and evaluation methods.
:::

## Quick Setup

Hermes includes dedicated AgentMemory commands so you do not need to edit `~/.hermes/config.yaml` by hand.

### 1. Start the AgentMemory server

```bash
npx @agentmemory/agentmemory
```

By default, the REST API listens on:

```text
http://localhost:3111
```

The viewer runs on:

```text
http://localhost:3113
```

### 2. Configure Hermes

For MCP tools only:

```bash
hermes agentmemory setup
```

For MCP plus the deeper memory-provider lifecycle hooks:

```bash
hermes agentmemory setup --provider
```

The command writes this MCP server config for you:

```yaml
mcp_servers:
  agentmemory:
    command: npx
    args: ["-y", "@agentmemory/mcp"]
```

With `--provider`, it also enables:

```yaml
memory:
  provider: agentmemory
```

Restart Hermes or start a new Hermes session so MCP servers reload.

### 3. Verify the setup

```bash
hermes agentmemory status
hermes agentmemory doctor
```

You can also verify the server directly:

```bash
curl http://localhost:3111/agentmemory/health
```

Expected response:

```json
{"status":"healthy"}
```

### 4. Open the viewer

```bash
hermes agentmemory viewer
```

Or open:

```text
http://localhost:3113
```

Use the viewer to inspect sessions, memories, lessons, feature flags, and captured activity.

## CLI Commands

```bash
hermes agentmemory setup          # Configure MCP integration
hermes agentmemory setup --provider  # Configure MCP and enable provider hooks
hermes agentmemory status         # Show server, viewer, MCP, and provider status
hermes agentmemory doctor         # Diagnose setup and print fixes
hermes agentmemory viewer         # Open the real-time viewer
hermes agentmemory mcp            # Configure only MCP
hermes agentmemory provider       # Enable provider lifecycle hooks
hermes agentmemory disable        # Disable AgentMemory as the active provider
```

## Optional: Memory Provider Plugin

AgentMemory also includes a Hermes memory-provider plugin in its repository:

```text
integrations/hermes
```

This gives deeper integration than MCP alone. The plugin maps Hermes memory lifecycle hooks to the AgentMemory REST API.

| Hermes lifecycle hook | AgentMemory behavior |
|---|---|
| `prefetch(query)` | Retrieves relevant memories before response generation |
| `sync_turn(user, assistant)` | Captures each conversation turn asynchronously |
| `on_session_end()` | Marks the session complete for summarization |
| `on_pre_compress()` | Re-injects relevant context before compaction |
| `on_memory_write()` | Mirrors `MEMORY.md` writes to AgentMemory |
| `system_prompt_block()` | Injects a project profile at session start |

To test the plugin manually from a checkout of the AgentMemory repository:

```bash
cp -r integrations/hermes ~/.hermes/plugins/memory/agentmemory
npx @agentmemory/agentmemory
```

Then set Hermes to use the provider if the plugin is available in your installation:

```yaml
memory:
  provider: agentmemory
```

:::warning Plugin availability
The MCP integration works without copying plugin files. The memory-provider plugin is an external integration unless it has been bundled with your Hermes installation.
:::

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGENTMEMORY_URL` | `http://localhost:3111` | AgentMemory REST server URL |
| `AGENTMEMORY_SECRET` | none | Optional auth token for protected instances |

## Example Workflow

Here is a realistic use case for a developer working on the same repository with multiple agents.

### Day 1: Hermes fixes auth

You ask Hermes:

```text
Add JWT auth to this FastAPI service and write tests.
```

Hermes edits files, runs tests, fixes failures, and explains the final design. AgentMemory captures observations like:

```text
Project uses FastAPI with routers under app/api.
Auth middleware lives in app/middleware/auth.py.
JWT validation uses python-jose.
Tests are run with uv run pytest.
A previous failure came from missing Authorization header normalization.
```

### Day 2: Claude Code debugs database performance

You use Claude Code instead of Hermes. Since Claude Code can connect to the same AgentMemory server, it can save new observations:

```text
N+1 query fixed in app/repositories/users.py.
SQLAlchemy selectinload is the preferred pattern for user dashboard queries.
Regression test added in tests/test_user_dashboard.py.
```

### Day 3: Hermes adds rate limiting

You return to Hermes and ask:

```text
Add per-user rate limiting to authenticated API routes.
```

Hermes can now retrieve both sets of memories:

- The auth middleware pattern from Day 1
- The test command and test layout
- The database performance fix from Day 2
- The preferred SQLAlchemy loading pattern

That lets Hermes avoid re-discovery and make a smaller, safer change.

## Best Practices

- Keep `USER.md` for stable facts about the human.
- Keep `MEMORY.md` for compact facts the agent should always know.
- Use AgentMemory for large project histories, tool traces, code decisions, bug fixes, and cross-agent recall.
- Use Skills for repeatable procedures, not raw memory.
- Keep AgentMemory local unless you intentionally expose it.
- Set `AGENTMEMORY_SECRET` if the server is reachable beyond localhost.
- Use the viewer to audit what is being stored.
- Prefer retrieved context over dumping huge logs or full memory files into prompts.

## Troubleshooting

### Hermes cannot see AgentMemory tools

Check that MCP is configured and Hermes has been restarted:

```bash
hermes mcp list
hermes mcp test agentmemory
```

Also confirm the server is running:

```bash
curl http://localhost:3111/agentmemory/health
```

### The viewer is empty

Run the demo to seed sample sessions:

```bash
npx @agentmemory/agentmemory demo
```

Then refresh:

```text
http://localhost:3113
```

### Retrieval works only by exact words

Check whether embeddings are enabled in AgentMemory. Without embeddings, AgentMemory can still use BM25 keyword retrieval, but semantic recall is weaker.

Run:

```bash
npx @agentmemory/agentmemory doctor
```

### Remote server does not connect

Set the server URL explicitly:

```bash
export AGENTMEMORY_URL=http://127.0.0.1:3111
```

If the server is protected, also set:

```bash
export AGENTMEMORY_SECRET=your-token
```

## Related Links

- [AgentMemory repository](https://github.com/rohitg00/agentmemory)
- [AgentMemory Hermes integration](https://github.com/rohitg00/agentmemory/tree/main/integrations/hermes)
- [Hermes MCP](./mcp.md)
- [Hermes Memory](./memory.md)
- [Memory Providers](./memory-providers.md)
- [Honcho Memory](./honcho.md)
