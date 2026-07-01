---
sidebar_position: 13
title: "Delegation & Parallel Work"
description: "When and how to use subagent delegation — patterns for parallel research, code review, and multi-file work"
---

# Delegation & Parallel Work

Hermes can spawn isolated child agents to work on tasks in parallel. Each subagent gets its own conversation, terminal session, and toolset. Only the final summary comes back — intermediate tool calls never enter your context window.

For the full feature reference, see [Subagent Delegation](/user-guide/features/delegation).

---

## When to Delegate

**Good candidates for delegation:**
- Reasoning-heavy subtasks (debugging, code review, research synthesis)
- Tasks that would flood your context with intermediate data
- Parallel independent workstreams (research A and B simultaneously)
- Fresh-context tasks where you want the agent to approach without bias

**Use something else:**
- Single tool call → just use the tool directly
- Mechanical multi-step work with logic between steps → `execute_code`
- Tasks needing user interaction → subagents can't use `clarify`
- Quick file edits → do them directly
- Durable long-running work that must outlive the current turn → `cronjob` or `terminal(background=True, notify_on_complete=True)`. `delegate_task` is **synchronous**: if the parent turn is interrupted, active children are cancelled and their work is discarded.

---

## Pattern: Parallel Research

Research three topics simultaneously and get structured summaries back:

```
Research these three topics in parallel:
1. Current state of WebAssembly outside the browser
2. RISC-V server chip adoption in 2025
3. Practical quantum computing applications

Focus on recent developments and key players.
```

Behind the scenes, Hermes uses:

```python
delegate_task(tasks=[
    {
        "goal": "Research WebAssembly outside the browser in 2025",
        "context": "Focus on: runtimes (Wasmtime, Wasmer), cloud/edge use cases, WASI progress",
        "toolsets": ["web"]
    },
    {
        "goal": "Research RISC-V server chip adoption",
        "context": "Focus on: server chips shipping, cloud providers adopting, software ecosystem",
        "toolsets": ["web"]
    },
    {
        "goal": "Research practical quantum computing applications",
        "context": "Focus on: error correction breakthroughs, real-world use cases, key companies",
        "toolsets": ["web"]
    }
])
```

All three run concurrently. Each subagent searches the web independently and returns a summary. The parent agent then synthesizes them into a coherent briefing.

---

## Pattern: Code Review

Delegate a security review to a fresh-context subagent that approaches the code without preconceptions:

```
Review the authentication module at src/auth/ for security issues.
Check for SQL injection, JWT validation problems, password handling,
and session management. Fix anything you find and run the tests.
```

The key is the `context` field — it must include everything the subagent needs:

```python
delegate_task(
    goal="Review src/auth/ for security issues and fix any found",
    context="""Project at /home/user/webapp. Python 3.11, Flask, PyJWT, bcrypt.
    Auth files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py
    Test command: pytest tests/auth/ -v
    Focus on: SQL injection, JWT validation, password hashing, session management.
    Fix issues found and verify tests pass.""",
    toolsets=["terminal", "file"]
)
```

:::warning The Context Problem
Subagents know **absolutely nothing** about your conversation. They start completely fresh. If you delegate "fix the bug we were discussing," the subagent has no idea what bug you mean. Always pass file paths, error messages, project structure, and constraints explicitly.
:::

---

## Pattern: Compare Alternatives

Evaluate multiple approaches to the same problem in parallel, then pick the best:

```
I need to add full-text search to our Django app. Evaluate three approaches
in parallel:
1. PostgreSQL tsvector (built-in)
2. Elasticsearch via django-elasticsearch-dsl
3. Meilisearch via meilisearch-python

For each: setup complexity, query capabilities, resource requirements,
and maintenance overhead. Compare them and recommend one.
```

Each subagent researches one option independently. Because they're isolated, there's no cross-contamination — each evaluation stands on its own merits. The parent agent gets all three summaries and makes the comparison.

---

## Pattern: Multi-File Refactoring

Split a large refactoring task across parallel subagents, each handling a different part of the codebase:

```python
delegate_task(tasks=[
    {
        "goal": "Refactor all API endpoint handlers to use the new response format",
        "context": """Project at /home/user/api-server.
        Files: src/handlers/users.py, src/handlers/auth.py, src/handlers/billing.py
        Old format: return {"data": result, "status": "ok"}
        New format: return APIResponse(data=result, status=200).to_dict()
        Import: from src.responses import APIResponse
        Run tests after: pytest tests/handlers/ -v""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Update all client SDK methods to handle the new response format",
        "context": """Project at /home/user/api-server.
        Files: sdk/python/client.py, sdk/python/models.py
        Old parsing: result = response.json()["data"]
        New parsing: result = response.json()["data"] (same key, but add status code checking)
        Also update sdk/python/tests/test_client.py""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Update API documentation to reflect the new response format",
        "context": """Project at /home/user/api-server.
        Docs at: docs/api/. Format: Markdown with code examples.
        Update all response examples from old format to new format.
        Add a 'Response Format' section to docs/api/overview.md explaining the schema.""",
        "toolsets": ["terminal", "file"]
    }
])
```

:::tip
Each subagent gets its own terminal session. They can work on the same project directory without stepping on each other — as long as they're editing different files. If two subagents might touch the same file, handle that file yourself after the parallel work completes.
:::

---

## Pattern: Gather Then Analyze

Use `execute_code` for mechanical data gathering, then delegate the reasoning-heavy analysis:

```python
# Step 1: Mechanical gathering (execute_code is better here — no reasoning needed)
execute_code("""
from hermes_tools import web_search, web_extract

results = []
for query in ["AI funding Q1 2026", "AI startup acquisitions 2026", "AI IPOs 2026"]:
    r = web_search(query, limit=5)
    for item in r["data"]["web"]:
        results.append({"title": item["title"], "url": item["url"], "desc": item["description"]})

# Extract full content from top 5 most relevant
urls = [r["url"] for r in results[:5]]
content = web_extract(urls)

# Save for the analysis step
import json
with open("/tmp/ai-funding-data.json", "w") as f:
    json.dump({"search_results": results, "extracted": content["results"]}, f)
print(f"Collected {len(results)} results, extracted {len(content['results'])} pages")
""")

# Step 2: Reasoning-heavy analysis (delegation is better here)
delegate_task(
    goal="Analyze AI funding data and write a market report",
    context="""Raw data at /tmp/ai-funding-data.json contains search results and
    extracted web pages about AI funding, acquisitions, and IPOs in Q1 2026.
    Write a structured market report: key deals, trends, notable players,
    and outlook. Focus on deals over $100M.""",
    toolsets=["terminal", "file"]
)
```

This is often the most efficient pattern: `execute_code` handles the 10+ sequential tool calls cheaply, then a subagent does the single expensive reasoning task with a clean context.

---

## Pattern: Cross-Profile Delegation

Sometimes you need separate Hermes agents with different configurations, API keys, and session histories — a personal agent and a business agent, for example. While `delegate_task` spawns subagents within the **same** profile, you can task another entire profile using the CLI directly:

```bash
# Task a fresh session in another profile
hermes -p business chat -q "Create a calendar event for tomorrow at 2pm" --quiet

# Resume a previous session for follow-up context
hermes -p business chat -r 20260701_110633_cd0d9f -q "Actually, move it to 3pm" --quiet

# Resume the most recent session
hermes -p business chat --continue -q "Follow-up task" --quiet
```

The `--quiet` flag suppresses banners and progress spinners so only the result is returned. Each profile has its own `config.yaml`, `.env`, sessions database, memory, and gateway — so tasks run with that profile's credentials, API keys, and bot tokens.

### Real-world example: Business + Personal separation

Set up two profiles — one for personal use and one for business:

```bash
hermes profile create business
hermes profile create personal
```

Each gets its own command alias automatically:

```bash
business chat -q "Send invoice reminder" --quiet   # One-shot business task
personal chat                                       # Your daily agent
```

From your personal agent session, you can task the business profile:

```bash
hermes -p business chat -q \
  'Create a calendar event titled "Team Standup" for tomorrow 10:00-10:30' \
  --quiet
```

All sessions are logged under the business profile's database — review them later by running `business chat` or `hermes -p business chat` directly.

### Fresh vs. continued sessions

| Mode | Command | When |
|------|---------|------|
| **Fresh** | `-q "task"` | Most tasks — no prior context needed |
| **Resume** | `-r SESSION_ID -q "..."` | Task builds on a previous conversation |
| **Continue** | `--continue -q "..."` | Quick follow-up on the most recent session |

### How it works

Each profile alias (like `business` in the examples) is a thin wrapper at `~/.local/bin/<name>` that runs `hermes -p <name> "$@"`. Under the hood, `-p` sets `HERMES_HOME` to the profile's directory, so all state — config, credentials, sessions, cron jobs — scopes to that profile. Nothing crosses over unless you explicitly pass it.

---

## Toolset Selection

Choose toolsets based on what the subagent needs:

| Task type | Toolsets | Why |
|-----------|----------|-----|
| Web research | `["web"]` | web_search + web_extract only |
| Code work | `["terminal", "file"]` | Shell access + file operations |
| Full-stack | `["terminal", "file", "web"]` | Everything except messaging |
| Read-only analysis | `["file"]` | Can only read files, no shell |

Restricting toolsets keeps the subagent focused and prevents accidental side effects (like a research subagent running shell commands).

---

## Constraints

- **Default 3 parallel tasks**: batches default to 3 concurrent subagents (configurable via `delegation.max_concurrent_children` in config.yaml, no hard ceiling, only a floor of 1)
- **Nested delegation is opt-in**: leaf subagents (default) cannot call `delegate_task`, `clarify`, `memory`, or `execute_code`. Orchestrator subagents (`role="orchestrator"`) retain `delegate_task` for further delegation, but only when `delegation.max_spawn_depth` is raised above the default of 1 (floor 1, no ceiling); the other three remain blocked. Disable globally via `delegation.orchestrator_enabled: false`.

### Tuning Concurrency and Depth

| Config | Default | Range | Effect |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | Parallel batch size per `delegate_task` call |
| `max_spawn_depth` | 1 | >=1 | How many delegation levels can spawn further |

Example: running 30 parallel workers with nested subagents:

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **Separate terminals** — each subagent gets its own terminal session with separate working directory and state
- **No conversation history** — subagents see only the `goal` and `context` the parent agent passes when calling `delegate_task`
- **Default 50 iterations** — set `max_iterations` lower for simple tasks to save cost
- **Not durable** — `delegate_task` is synchronous and runs inside the parent turn. If the parent is interrupted (new user message, `/stop`, `/new`), all active children are cancelled (`status="interrupted"`) and their work is discarded. For work that must outlive the current turn, use `cronjob` or `terminal(background=True, notify_on_complete=True)`.

---

## Tips

**Be specific in goals.** "Fix the bug" is too vague. "Fix the TypeError in api/handlers.py line 47 where process_request() receives None from parse_body()" gives the subagent enough to work with.

**Include file paths.** Subagents don't know your project structure. Always include absolute paths to relevant files, the project root, and the test command.

**Use delegation for context isolation.** Sometimes you want a fresh perspective. Delegating forces you to articulate the problem clearly, and the subagent approaches it without the assumptions that built up in your conversation.

**Check results.** Subagent summaries are just that — summaries. If a subagent says "fixed the bug and tests pass," verify by running the tests yourself or reading the diff.

---

*For the complete delegation reference — all parameters, ACP integration, and advanced configuration — see [Subagent Delegation](/user-guide/features/delegation).*
