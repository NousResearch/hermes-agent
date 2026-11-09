# Kanban Hub & Bridge — Installation Guide

## Overview

- **kanban-hub** — Centralized Flask server for cross-node task routing, worker discovery, and user management. Runs on port `9900`.
- **kanban-bridge** — Plugin inside `hermes-agent` that connects a local Hermes node to the Hub. Polls for tasks, submits results, discovers remote workers.

```text
Node A (bridge) ◄──► Hub Server (kanban-hub) ◄──► Node B (bridge)
```

### Local Task Progress

Each Hermes node uses a local Kanban board to track task progress. When a task is created (locally or received from the Hub), it moves through the Kanban pipeline (`ready → running → done / blocked / gave_up`). The bridge monitors this local Kanban state — when a Hub-assigned task reaches `done`, the bridge automatically submits the result back to the Hub.

---

## Part 1: Install Kanban Hub (Server)

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone or copy kanban-hub directory
cd /path/to/kanban-hub

# 2. Install dependencies
pip install -r requirements.txt    # flask>=3.0

# 3. Start the Hub server
python hub.py
# Default: http://0.0.0.0:9900
```

The Hub uses SQLite (`hub.db`) — no external database needed. The database is auto-created on first run via `init_db()`.

### Register a User on Hub

Open `http://0.0.0.0:9900`. Each node that connects to the Hub needs an account (email + secret).

### Authentication

All API calls require headers:

```text
X-Hub-Name: your-email@example.com
X-Hub-Secret: your-secret
```

---

## Part 2: Install Kanban Bridge (Plugin)

### Step 1 — Install the plugin

You can install the plugin in two ways.

#### Option A — Install from Git (recommended)

Hermes installs plugins directly from a Git repository. Once `kanban-bridge`
is published to a public (or accessible) Git repo, install it with:

```bash
# owner/repo shorthand (GitHub)
hermes plugins install https://github.com/miniframework/hermes-plugin-kanban-bridge --enable

# or a full Git URL
hermes plugins install https://github.com/miniframework/hermes-plugin-kanban-bridge --enable
```

Manage it afterwards:

```bash
hermes plugins list                       # show installed plugins
hermes plugins update kanban-bridge       # pull latest changes
hermes plugins disable kanban-bridge      # disable without removing
hermes plugins remove kanban-bridge       # uninstall
```

#### Option B — Copy plugin manually

Copy the `kanban-bridge` directory into the Hermes Agent plugins folder:

```bash
# Linux / macOS
cp -r kanban-bridge/ <hermes-agent>/plugins/kanban-bridge/

# Windows
xcopy /E kanban-bridge\ <hermes-agent>\plugins\kanban-bridge\
```

The plugin directory structure:

```text
plugins/kanban-bridge/
├── __init__.py        # Plugin entry point (register, slash commands, CLI)
├── bridge.py          # HubBridge — inbound task polling, approval, completion
├── hub_client.py      # HTTP client for Hub API
├── api.py             # API helpers
├── discovery.py       # Worker discovery logic
├── publisher.py       # HubSubmitter — outbound task submission
└── plugin.yaml        # Plugin metadata
```

`plugin.yaml`:

```yaml
name: kanban-bridge
version: 0.1.0
description: "Multi-host kanban bridge: forward tasks to remote Hermes instances via HTTP webhook and receive results back."
author: zhibinwang
hooks:
  - pre_gateway_dispatch
```

### Step 2 — Configure bridge in `config.yaml`

Add the bridge section under `kanban`:

```yaml
kanban:
  bridge:
    enabled: true
    mode: hub
    hub_url: http://<hub-server-ip>:9900
    self_name: your-email@example.com     # must match Hub account
    secret: your-secret                    # must match Hub account credits
    poll_interval: 5                       # seconds between Hub polls
    approval_required: false               # set true to require human approval
    approval_channel: ""                   # Discord channel ID (if approval_required)

plugins:
  enabled:
  - kanban-bridge
```

### Step 3 — Verify

```bash
# Check bridge is loaded
hermes kanban-bridge status

# Discover remote workers
hermes kanban-bridge discover
```

### Step 4 — Share local workers (optional)

```bash
hermes kanban-bridge share worker1
hermes kanban-bridge set-concurrency worker1 3
hermes kanban-bridge set-credits worker1 10
```

Or auto-publish in `config.yaml`:

```yaml
kanban:
  bridge:
    publish_workers:
      - profile: worker1
        share: true
```

---

## Part 3: Install Skills

Copy the skill files into the Hermes Agent skills directory:

```bash
# Linux / macOS
cp -r kanban-bridge-skill/ <hermes-agent>/skills/devops/kanban-bridge/
cp -r kanban-orchestrator-skill/ <hermes-agent>/skills/devops/kanban-orchestrator/

# Windows
xcopy /E kanban-bridge-skill\ <hermes-agent>\skills\devops\kanban-bridge\
xcopy /E kanban-orchestrator-skill\ <hermes-agent>\skills\devops\kanban-orchestrator\
```

Skill directory structure:

```text
skills/devops/
├── kanban-bridge/
│   ├── SKILL.md       # Discovery & routing constraints
│   └── SOUL.md        # Trigger rule: activate both skills together
└── kanban-orchestrator/
    └── SKILL.md       # Decomposition playbook, anti-temptation rules
```

### Configure default SOUL.md

Add the trigger rule to your default profile's `SOUL.md` (at the Hermes data root, e.g. `~/.hermes/SOUL.md`):

```markdown
# Default Profile

## Kanban Trigger Rule

When the user's request mentions **any** of these keywords or intents,
**always** activate both `kanban-orchestrator` and `kanban-bridge` together:

- worker, agent, profile, delegate, assign, route
- kanban, task, subtask, fan-out, parallel
- hub, remote, bridge, cross-node
- "let X do it", "ask X to", "send to X"
```

---

## Part 4: CLI Command Reference

| Command | Purpose |
|---------|---------|
| `hermes kanban-bridge status` | Show bridge status (hub URL, self name, inbound/outbound counts) |
| `hermes kanban-bridge discover` | List all remote workers on the Hub with SOUL.md descriptions |
| `hermes kanban-bridge workers` | List all workers registered on the Hub with active task counts |
| `hermes kanban-bridge tasks` | List tasks on the Hub assigned to this node |
| `hermes kanban-bridge share <profile>` | Share a local profile as a worker on the Hub |
| `hermes kanban-bridge unshare <profile>` | Remove a shared worker from the Hub |
| `hermes kanban-bridge set-concurrency <profile> <n>` | Set max concurrent tasks for a worker |
| `hermes kanban-bridge set-credits <profile> <n>` | Set credit cost per task for a worker |
| `hermes kanban-bridge approve <hub_id>` | Approve a completed task result |
| `hermes kanban-bridge reject <hub_id>` | Reject a completed task result |

Slash commands: same subcommands available as `/kanban-bridge <subcommand>` inside a Hermes session.

Programmatic tool: `kanban_discover()` — returns remote worker list for orchestrator task planning.

---

## Part 5: Usage Examples

### Example 1: Two-Node Setup (Alice + Bob)

```bash
# --- Hub server ---
cd kanban-hub && pip install -r requirements.txt && python hub.py

# --- Bob's node (worker) ---
hermes kanban-bridge share worker1
hermes kanban-bridge set-concurrency worker1 3

# --- Alice's node (orchestrator) ---
hermes kanban-bridge discover
# Output:
#   ## bob@example.com:worker1 [REMOTE]
#   - Description: you are coder review
#   - Load: 0/3
```

Alice creates a task routed to Bob:

```python
kanban_create(
    title="Review auth module",
    assignee="hub:bob@example.com:worker1",
    body="Review the new OAuth implementation for security issues.",
)
```

### Example 2: Mixed Local + Remote Task Graph

```python
t1 = kanban_create(
    title="prepare dataset",
    assignee="data-eng",                          # local worker
    body="Clean and split training data.",
)["task_id"]

t2 = kanban_create(
    title="train model",
    assignee="hub:bob@example.com:gpu-worker",    # remote worker
    body="Fine-tune model on prepared data.",
    parents=[t1],
)["task_id"]

t3 = kanban_create(
    title="evaluate results",
    assignee="analyst",                            # local worker
    body="Compare model metrics against baseline.",
    parents=[t2],
)["task_id"]
```

### Example 3: Approval Workflow

```yaml
kanban:
  bridge:
    approval_required: true
    approval_channel: "1483362021591220224"
```

1. Hub task completes locally → bridge sends Discord notification with hub ID.
2. User replies `approve hub_abc123` or `reject hub_abc123` in Discord.
3. Or via CLI: `hermes kanban-bridge approve hub_abc123`.

### Example 4: Semantic Matching

```bash
$ hermes kanban-bridge discover

## alice@example.com:worker1 [REMOTE]
- Description: full-stack developer, React and Python expert
- Load: 1/5

## alice@example.com:worker2 [REMOTE]
- Description: you are coder review, security audit specialist
- Load: 0/3

## bob@example.com:gpu-worker [REMOTE]
- Description: ML training on GPU cluster, PyTorch and TensorFlow
- Load: 2/4
```

User says "让 reviewer 检查代码" → match `worker2` (description: "coder review") → assign to `hub:alice@example.com:worker2`.

---

## Verification Checklist

| Step | Command | Expected |
|------|---------|----------|
| Hermes installed | `hermes --version` | Version number |
| Hub running | `curl http://<hub-ip>:9900/api/workers` | JSON response |
| Bridge loaded | `hermes kanban-bridge status` | Shows hub URL and self name |
| Discovery works | `hermes kanban-bridge discover` | Lists remote workers |
| Task routing | Create task with `hub:<email>:<profile>` | Task appears on remote node |

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `hermes: command not found` | Hermes not installed or not in PATH | Re-run install script, check PATH |
| Bridge plugin not loaded | Plugin not in `plugins/` directory | Copy `kanban-bridge/` to `plugins/` |
| Bridge status shows error | Wrong `hub_url` or Hub not running | Check Hub is running, verify IP/port |
| 403 on API calls | `self_name`/`secret` mismatch | Re-register on Hub or fix config |
| `discover` returns empty | No nodes have shared workers | Run `hermes kanban-bridge share` on another node |
| Task stuck in `submitted` | Remote node offline or typo in assignee | Check remote node, verify spelling |
| Approval stuck in pending | `approval_channel` not set or `DISCORD_BOT_TOKEN` missing | Configure both |

---

## Feature: Multi-Hub Federation

W2 Bridge Agent dual-registration across multiple Hubs, enabling cross-hub task routing and orchestration.
