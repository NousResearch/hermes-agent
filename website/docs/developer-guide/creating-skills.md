---
sidebar_position: 3
title: "Creating Skills"
description: "How to create skills for Hermes Agent — SKILL.md format, guidelines, and publishing"
---

# Creating Skills

Skills are the preferred way to add new capabilities to Hermes Agent. They're easier to create than tools, require no code changes to the agent, and can be shared with the community.

Treat a skill as maintained procedural knowledge, not as a one-off prompt. Before adding one, search the installed skills, both in-repo skill trees, and the Skills Hub. If an existing skill already owns the same trigger and outcome, update it instead of creating a competing sibling.

## Should it be a Skill, Plugin, or Tool?

Make it a **Skill** when:
- The capability can be expressed as instructions + shell commands + existing tools
- It wraps an external CLI or API that the agent can call via `terminal` or `web_extract`
- Any secrets can be declared with `required_environment_variables` and the workflow does not need a custom auth lifecycle
- Examples: arXiv search, git workflows, Docker management, PDF processing, email via CLI tools

Put repeated deterministic processing in the skill's `scripts/` directory.
When the capability needs structured runtime I/O, a managed auth lifecycle,
binary or streaming data, or real-time events, prefer an MCP server, plugin, or
service-gated tool. Add a new core model tool only when the capability is
fundamental to nearly every user and cannot be reached through existing tools.

## Decide Whether to Create or Update

1. **Search first.** Use `skills_list` to survey installed skills and `skill_view` to read likely matches. In a checkout, use `search_files` across `skills/` and `optional-skills/`; use `hermes skills search <query>` for Hub candidates. Read two or three peers in the closest category before deciding.
2. **Update when the intent overlaps.** Extend an existing skill when it already covers the same trigger class, tool, or outcome. Use a targeted patch for a small correction and a full rewrite only for a genuine overhaul. Remove stale or superseded wording instead of layering another version of the same rule.
3. **Create when the behavior is distinct.** A new skill needs a clear trigger boundary, a reusable procedure, and a verification step that do not fit an existing skill. Avoid a narrow sibling when an umbrella skill can absorb the workflow cleanly.

`skill_manage(action="create")` always creates a user-local skill under the active profile's `$HERMES_HOME/skills/`; it does **not** create source files in this repository. For an in-repo contribution, use Hermes file tools such as `write_file` and `patch` in the checkout.

## Choose the Destination

| Destination | Use it for | Authoring and distribution |
|---|---|---|
| Bundled: `skills/<category>/<name>/` | Broad workflows useful to most Hermes users | Edit the repository source. It ships and is seeded into user profiles by default. |
| Official optional: `optional-skills/<category>/<name>/` | Official but niche, paid-service, platform-specific, heavyweight, or dependency-heavy workflows | Edit the repository source. Users install it with `hermes skills install official/<category>/<name>`. |
| User-local: `$HERMES_HOME/skills/[<category>/]<name>/` | Personal procedures, private workflows, and prototypes | Let Hermes call `skill_manage(action="create")`, or edit the active profile directly. The default home is `~/.hermes`, but profiles can use another root. |
| External Hub or standalone repository | Community, organization-specific, or third-party integrations that should not be maintained in the Hermes core repository | Author in its own repository and publish through the Skills Hub or a tap. Installation copies it into `$HERMES_HOME/skills/`. |

Heavy or niche skills do not become bundled merely because their implementation is polished. Likewise, do not commit a personal skill to `skills/` only to make it discoverable locally.

## Skill Directory Structure

All four destinations use the same self-contained layout:

```text
<skill-root>/[<category>/]<skill-name>/
├── SKILL.md              # Required: compact instructions
├── scripts/              # Optional: deterministic helpers
├── references/           # Optional: detail loaded on demand
├── templates/            # Optional: reusable text/config templates
└── assets/               # Optional: files used in produced output
```

Create only the directories the skill needs. Keep detailed or branch-specific material in `references/`, but keep the core workflow in `SKILL.md`. Do not add auxiliary `README.md`, changelog, or installation-guide files that Hermes will not use.

## SKILL.md Format

```markdown
---
name: my-skill
description: "Diagnose service failures with a repeatable workflow."
version: 1.0.0
author: Your Name
license: MIT
platforms: [macos, linux]          # Optional — restrict to specific OS platforms
                                   #   Valid: macos, linux, windows
                                   #   Omit to load on all platforms (default)
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
    requires_toolsets: [web]            # Optional — only show when these toolsets are active
    requires_tools: [web_search]        # Optional — only show when these tools are available
    fallback_for_toolsets: [browser]    # Optional — hide when these toolsets are active
    fallback_for_tools: [browser_navigate]  # Optional — hide when these tools exist
    config:                              # Optional — config.yaml settings the skill needs
      - key: my.setting
        description: "What this setting controls"
        default: "sensible-default"
        prompt: "Display prompt for setup"
    blueprint:                              # Optional — marks this skill a runnable automation
      schedule: "0 9 * * *"              #   cron expr / "every 2h" / ISO timestamp
      deliver: origin                    #   optional (default origin)
      prompt: "Task instruction for each run"  # optional
      no_agent: false                    # optional
required_environment_variables:          # Optional — env vars the skill needs
  - name: MY_API_KEY
    prompt: "Enter your API key"
    help: "Get one at https://example.com"
    required_for: "API access"
---

# API Incident Triage Skill

State in two or three sentences what the skill does and what is outside its scope.

## When to Use
List positive triggers and important counter-triggers.

## Prerequisites
List required Hermes toolsets, MCP servers, credentials, platforms, and setup.

## How to Run
Show the primary invocation path using Hermes-native tools.

## Quick Reference
Summarize the few commands or decisions used most often.

## Procedure
Give ordered steps with checkable completion criteria.

## Pitfalls
Known failure modes and how to handle them.

## Verification
How the agent confirms it worked.
```

The body order above is the merge standard for every new or modernized bundled, optional, or contributed skill: `# <Skill> Skill`, a two-to-three-sentence introduction, then `When to Use`, `Prerequisites`, `How to Run`, `Quick Reference`, `Procedure`, `Pitfalls`, and `Verification`.

:::warning Description hardline
The runtime validator accepts descriptions up to 1024 characters for backward compatibility. Repository review is stricter: `description` must be **60 characters or fewer**, contain one sentence, end with a period, state the capability rather than the implementation, avoid marketing words, and not repeat the skill name. The description is present in skill listings before the body loads, so keep all trigger-critical wording there and put detail in the body.
:::

### Platform-Specific Skills

Skills can restrict themselves to specific operating systems using the `platforms` field:

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders)
platforms: [macos, linux]     # macOS and Linux
platforms: [windows]          # Windows only
```

When set, the skill is automatically hidden from the system prompt, `skills_list()`, and slash commands on incompatible platforms. If omitted or empty, the skill loads on all platforms (backward compatible).

### Conditional Skill Activation

Skills can declare dependencies on specific tools or toolsets. This controls whether the skill appears in the system prompt for a given session.

```yaml
metadata:
  hermes:
    requires_toolsets: [web]           # Hide if the web toolset is NOT active
    requires_tools: [web_search]       # Hide if web_search tool is NOT available
    fallback_for_toolsets: [browser]   # Hide if the browser toolset IS active
    fallback_for_tools: [browser_navigate]  # Hide if browser_navigate IS available
```

| Field | Behavior |
|-------|----------|
| `requires_toolsets` | Skill is **hidden** when ANY listed toolset is **not** available |
| `requires_tools` | Skill is **hidden** when ANY listed tool is **not** available |
| `fallback_for_toolsets` | Skill is **hidden** when ANY listed toolset **is** available |
| `fallback_for_tools` | Skill is **hidden** when ANY listed tool **is** available |

**Use case for `fallback_for_*`:** Create a skill that serves as a workaround when a primary tool isn't available. For example, a `duckduckgo-search` skill with `fallback_for_tools: [web_search]` only shows when the web search tool (which requires an API key) is not configured.

**Use case for `requires_*`:** Create a skill that only makes sense when certain tools are present. For example, a web scraping workflow skill with `requires_toolsets: [web]` won't clutter the prompt when web tools are disabled.

### Environment Variable Requirements

Skills can declare environment variables they need. When a skill is loaded via `skill_view`, its required vars are automatically registered for passthrough into sandboxed execution environments (terminal, execute_code).

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: "Tenor API key"               # Shown when prompting user
    help: "Get your key at https://tenor.com"  # Help text or URL
    required_for: "GIF search functionality"   # What needs this var
```

Each entry supports:
- `name` (required) — the environment variable name
- `prompt` (optional) — prompt text when asking the user for the value
- `help` (optional) — help text or URL for obtaining the value
- `required_for` (optional) — describes which feature needs this variable

Users can also manually configure passthrough variables in `config.yaml`:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_VAR
    - ANOTHER_VAR
```

See `skills/apple/` for examples of macOS-only skills.

## Secure Setup on Load

Use `required_environment_variables` when a skill needs an API key or token. Missing values do **not** hide the skill from discovery. Instead, Hermes prompts for them securely when the skill is loaded in the local CLI.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

The user can skip setup and keep loading the skill. Hermes never exposes the raw secret value to the model. Gateway and messaging sessions show local setup guidance instead of collecting secrets in-band.

:::tip Sandbox Passthrough
When your skill is loaded, any declared `required_environment_variables` that are set are **automatically passed through** to `execute_code` and `terminal` sandboxes — including remote backends like Docker and Modal. Your skill's scripts can access `$TENOR_API_KEY` (or `os.environ["TENOR_API_KEY"]` in Python) without the user needing to configure anything extra. See [Environment Variable Passthrough](/user-guide/security#environment-variable-passthrough) for details.
:::

Legacy `prerequisites.env_vars` remains supported as a backward-compatible alias.

### Config Settings (config.yaml)

Skills can declare non-secret settings that are stored in `config.yaml` under the `skills.config` namespace. Unlike environment variables (which are secrets stored in `.env`), config settings are for paths, preferences, and other non-sensitive values.

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
      - key: myplugin.domain
        description: Domain the plugin operates on
        default: ""
        prompt: Plugin domain (e.g., AI/ML research)
```

Each entry supports:
- `key` (required) — dotpath for the setting (e.g., `myplugin.path`)
- `description` (required) — explains what the setting controls
- `default` (optional) — default value if the user doesn't configure it
- `prompt` (optional) — prompt text shown during `hermes config migrate`; falls back to `description`

**How it works:**

1. **Storage:** Values are written to `config.yaml` under `skills.config.<key>`:
   ```yaml
   skills:
     config:
       myplugin:
         path: ~/my-data
   ```

2. **Discovery:** `hermes config migrate` scans all enabled skills, finds unconfigured settings, and prompts the user. Settings also appear in `hermes config show` under "Skill Settings."

3. **Runtime injection:** When a skill loads, its config values are resolved and appended to the skill message:
   ```
   [Skill config (from ~/.hermes/config.yaml):
     myplugin.path = /home/user/my-data
   ]
   ```
   The agent sees the configured values without needing to read `config.yaml` itself.

4. **Manual setup:** Users can also set values directly:
   ```bash
   hermes config set skills.config.myplugin.path ~/my-data
   ```

:::tip When to use which
Use `required_environment_variables` for API keys, tokens, and other **secrets** (stored in `~/.hermes/.env`, never shown to the model). Use `config` for **paths, preferences, and non-sensitive settings** (stored in `config.yaml`, visible in config show).
:::

### Credential File Requirements (OAuth tokens, etc.)

Skills that use OAuth or file-based credentials can declare files that need to be mounted into remote sandboxes. This is for credentials stored as **files** (not env vars) — typically OAuth token files produced by a setup script.

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

Each entry supports:
- `path` (required) — file path relative to `~/.hermes/`
- `description` (optional) — explains what the file is and how it's created

When loaded, Hermes checks if these files exist. Missing files trigger `setup_needed`. Existing files are automatically:
- **Mounted into Docker** containers as read-only bind mounts
- **Synced into Modal** sandboxes (at creation + before each command, so mid-session OAuth works)
- Available on **local** backend without any special handling

:::tip When to use which
Use `required_environment_variables` for simple API keys and tokens (strings stored in `~/.hermes/.env`). Use `required_credential_files` for OAuth token files, client secrets, service account JSON, certificates, or any credential that's a file on disk.
:::

See the `skills/productivity/google-workspace/SKILL.md` for a complete example using both.

## Skill Guidelines

### Use Hermes-Native Tools

Tool names in `SKILL.md` prose must be native Hermes tools or tools from an MCP server named in `## Prerequisites`. Prefer `search_files` over `find` or `ls`, `read_file` over `cat`, `head`, or `tail`, and `patch` over `sed` or `awk`. Use `write_file` for new files, `terminal` to execute a bundled helper, `web_extract` for URL content, and tools such as `browser_navigate`, `vision_analyze`, or `delegate_task` only when that capability is actually required.

Do not copy tool names from another agent framework and assume Hermes provides them. If the workflow requires an MCP server, name it and document its setup. Keep third-party CLIs and non-trivial shell pipelines behind a helper in `scripts/` where practical; invoke that helper through `terminal` instead of making ad hoc shell construction the skill's primary interaction surface.

### Minimize External Dependencies

Prefer standard-library helpers and existing Hermes tools. If a dependency is necessary, declare it in `## Prerequisites`, provide a repeatable setup path, audit `platforms:` against its real OS support, and test the failure shown when it is absent.

For repository contributions, keep any required `.env.example` addition inside a clearly delimited block owned by the skill. Do not reformat or refresh unrelated entries while adding the skill's credentials.

### Credit the Human Contributor

Put the human contributor's real name and GitHub handle first in `author`; Hermes Agent can be a secondary collaborator. If Hermes drafted the skill, do not replace the human contributor's credit with the tool's name.

### Progressive Disclosure

Put the common path and decision points in `SKILL.md`; move detailed APIs, schemas, and branch-specific material into directly linked files under `references/`. Aim for about 100 lines for a simple skill and 200 for a complex one. The main file should remain usable without loading every reference.

### Keep One Source of Truth

State each rule once, next to the step it controls. On update, remove the wording that the new instruction replaces, prune obsolete examples and references, and verify that related skills still resolve. Repeated guidance drifts and consumes context; a shorter, sharper update is usually better than an appended caveat.

### Include Helper Scripts

For XML/JSON parsing or other non-trivial deterministic logic, include and test a helper in `scripts/` instead of asking the model to recreate it inline. Reference it with `${HERMES_SKILL_DIR}/scripts/<name>` so the installed location does not matter.

### Deliver media as documents (`[[as_document]]`)

If your skill produces a high-resolution screenshot, chart, or any image where lossy preview compression would hurt — emit the literal directive `[[as_document]]` somewhere in the response (commonly the last line). The gateway strips the directive and delivers every extracted media path in that response as a downloadable file attachment instead of an inline image bubble. See [Skill output and media delivery](../user-guide/features/skills.md#skill-output-and-media-delivery) for the full semantics.

#### Referencing bundled scripts from SKILL.md

When a skill is loaded, the activation message exposes the absolute skill directory as `[Skill directory: /abs/path]` and also substitutes two template tokens anywhere in the SKILL.md body:

| Token | Replaced with |
|---|---|
| `${HERMES_SKILL_DIR}` | Absolute path to the skill's directory |
| `${HERMES_SESSION_ID}` | The active session id (left in place if there is no session) |

So a SKILL.md can tell the agent to run a bundled script directly with:

```markdown
To analyse the input, run:

    node ${HERMES_SKILL_DIR}/scripts/analyse.js <input>
```

The agent sees the substituted absolute path and invokes the `terminal` tool with a ready-to-run command — no path math, no extra `skill_view` round-trip. Disable substitution globally with `skills.template_vars: false` in `config.yaml`.

#### Inline shell snippets (opt-in)

Skills can also embed inline shell snippets written as `` !`cmd` `` in the SKILL.md body. When enabled, each snippet's stdout is inlined into the message before the agent reads it, so skills can inject dynamic context:

```markdown
Current date: !`date -u +%Y-%m-%d`
Git branch: !`git -C ${HERMES_SKILL_DIR} rev-parse --abbrev-ref HEAD`
```

This is **off by default** — any snippet in a SKILL.md runs on the host without approval, so only enable it for skill sources you trust:

```yaml
# config.yaml
skills:
  inline_shell: true
  inline_shell_timeout: 10   # seconds per snippet
```

Snippets run with the skill directory as their working directory, and output is capped at 4000 characters. Failures (timeouts, non-zero exits) show up as a short `[inline-shell error: ...]` marker instead of breaking the whole skill.

## Validate, Test, and Forward-Test

Treat these as three different checks. A valid file can still contain a broken helper, and a tested helper can still produce a skill that agents do not follow reliably.

### 1. Validate the Contract

For an in-repo skill, validate frontmatter, naming, the 60-character description hardline, and modern section order. This standalone check mirrors the relevant merge requirements while the runtime validator remains backward-compatible with older skills:

```bash
python3 - skills/<category>/<name>/SKILL.md <<'PY'
from pathlib import Path
import re
import sys
import yaml

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
assert text.startswith("---"), "frontmatter must start at byte 0"
end = re.search(r"\n---\s*\n", text[3:])
assert end, "frontmatter is not closed"
frontmatter = yaml.safe_load(text[3 : 3 + end.start()])
assert isinstance(frontmatter, dict)
assert frontmatter["name"] == path.parent.name
assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", frontmatter["name"])
description = str(frontmatter["description"])
assert len(description) <= 60, len(description)
assert description.endswith(".")
body = text[3 + end.end() :]
sections = [
    "## When to Use", "## Prerequisites", "## How to Run",
    "## Quick Reference", "## Procedure", "## Pitfalls", "## Verification",
]
positions = [body.find(section) for section in sections]
assert all(position >= 0 for position in positions), positions
assert positions == sorted(positions), positions
PY
```

Also audit `platforms:` against imports and commands in `scripts/`, confirm declared Hermes tools/toolsets exist, and check that every `metadata.hermes.related_skills` entry resolves in the intended distribution.

### 2. Test Deterministic Behavior

Put skill tests in `tests/skills/test_<skill>_skill.py`. Use standard-library modules, pytest, and `unittest.mock`; do not make live network calls. Test helper scripts directly, plus any regression-prone metadata or output contract:

```bash
scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q
```

Run the helper itself on a representative fixture as well. A mocked unit test is not enough for file I/O, subprocess, path, or platform behavior.

### 3. Forward-Test the Instructions

Use a fresh Hermes session that resolves the edited skill, then give it a realistic task and raw input rather than a review prompt or the expected answer:

```bash
hermes chat --toolsets skills,file,terminal -q \
  "Use the <skill-name> skill to <solve a realistic task>."
```

Add the other toolsets declared by the skill to the example command. For repository source, point a disposable test profile's `skills.external_dirs` at the checkout's `skills/` or `optional-skills/` root, or sync the source into that profile before running the command. Do not assume the current long-lived session has reloaded a changed skill.

For a complex skill, repeat the forward test in independent fresh sessions. Include the primary success path, a realistic failure path, and a nearby request that should remain outside the skill's scope. Check whether Hermes selects the skill from its description, reads only the references it needs, uses valid native tools, follows the procedure, and performs the documented verification. Tighten the skill and rerun the same class of task when any of those checks fails.

## Create and Update with Hermes

For skills resolved by the active profile:

- Create with `skill_manage(action="create", name=..., content=..., category=...)`. Supply the complete `SKILL.md`; creation always targets `$HERMES_HOME/skills/` and fails on a name collision anywhere Hermes can discover.
- Prefer `skill_manage(action="patch", ...)` for a focused correction. Read the current file first and include enough unique surrounding text.
- Use `skill_manage(action="edit", ...)` only for a full rewrite, and supply the complete updated `SKILL.md`.
- Add or replace supporting files with `skill_manage(action="write_file", ...)`; paths must be under `references/`, `templates/`, `scripts/`, or `assets/`.

The update actions locate an existing user-created, bundled, Hub-installed, or configured external-directory skill and edit that resolved copy. Updating an installed copy does not update its upstream repository. For bundled or official optional source contributions, edit `skills/` or `optional-skills/` in the checkout with `patch` and `write_file`. Do not use `skill_manage(action="create")`: it targets the active profile, not the repository. Keep the source, its focused tests, and any generated documentation changes in the same contribution.

### Generated Skill Documentation

Pages under `website/docs/user-guide/skills/bundled/` and `website/docs/user-guide/skills/optional/` are generated from the corresponding `SKILL.md` files. Do not edit those pages directly. After changing an in-repo skill, regenerate the pages and catalogs from the repository root:

```bash
python3 website/scripts/generate-skill-docs.py
```

Review the generated diff, but continue to make content corrections in the source `SKILL.md`.

## Blueprints: skills that are also automations

A **blueprint** is an ordinary skill that additionally declares a schedule in its frontmatter. Add a `metadata.hermes.blueprint` block and the skill becomes a shareable, runnable automation:

```yaml
metadata:
  hermes:
    tags: [blueprint, email]
    blueprint:
      schedule: "0 8 * * *"     # presence of `blueprint:` marks it runnable
      deliver: telegram          # optional (default: origin)
      prompt: "Summarize my unread email and today's calendar."  # optional
      no_agent: false            # optional
```

Because a blueprint **is** a skill, it flows through the entire skills pipeline unchanged — search, inspect, install, security scan, provenance, taps, the centralized index, and `hermes skills publish` for sharing. Nothing new to learn.

**Installing a blueprint.** When you install a skill that carries a `blueprint:` block, Hermes registers it as a **suggested cron job** rather than scheduling it. Scheduling is **opt-in** — installing never silently creates a recurring job. You review and accept it via `/suggestions`:

```bash
hermes skills install owner/morning-brief
# → Blueprint: 'morning-brief' is an automation (schedule 0 8 * * *).
#   Added to your suggestions — run /suggestions to schedule or dismiss it.

# then, in a session:
/suggestions             # lists pending suggestions, numbered
/suggestions accept 1    # creates the cron job
/suggestions dismiss 1   # never offer it again
```

Blueprints are one **source** of the unified Suggested Cron Jobs surface — the same place curated starter automations and (later) usage-pattern and integration suggestions appear. See [Suggested Cron Jobs](#suggested-cron-jobs) below.

**Sharing an automation you built.** A blueprint loaded by a cron job (`hermes cron create --skill <name> ...`) can be exported back to a SKILL.md and published like any other skill, so an automation you tuned for yourself becomes a one-command install for someone else.

The blueprint layer adds no new object type, store, or transport — the blueprint is a skill, the schedule is a cron job, and sharing is the existing publish/tap/index path.

## Suggested Cron Jobs

Hermes can *propose* automations and let you accept them with one tap, instead of making you assemble cron jobs by hand. Every proposal flows through one surface — the `/suggestions` command — regardless of where it came from:

| Source | Trigger |
|--------|---------|
| `catalog` | Curated starter automations (`/suggestions catalog`) — daily briefing, important-mail monitor, weekly review, workday-start reminder |
| `blueprint` | You installed a skill carrying a `blueprint:` block |
| `usage` | The background review noticed a recurring ask a schedule would serve |
| `integration` | You connected an account (Gmail, GitHub, ...) and the obvious automations are offered |

```bash
/suggestions             # list pending
/suggestions accept N    # schedule suggestion N (creates the cron job)
/suggestions dismiss N   # dismiss it — latched, never re-offered
/suggestions catalog     # add the curated starter automations
```

Accepting a suggestion calls the same `cron.jobs.create_job` the `cronjob` tool uses — there is no second job engine. Suggestions **never** auto-create jobs; acceptance is always explicit. Dismissed suggestions latch by a stable key so the same proposal is never re-offered. The pending list is capped so it never becomes a nag wall.

The **important-mail monitor** catalog entry is the poll→classify→surface pattern: it scores inbox items with a cheap classifier model (`auxiliary.monitor` in `config.yaml`) and delivers only the ones above an urgency threshold, staying silent otherwise.

## Publishing Skills

### To the Skills Hub

```bash
hermes skills publish skills/my-skill --to github --repo owner/repo
```

### To a Custom Repository

Add your repo as a tap:

```bash
hermes skills tap add owner/repo
```

Users can then search and install from your repository.

## Security Scanning

All hub-installed skills go through a security scanner that checks for:

- Data exfiltration patterns
- Prompt injection attempts
- Destructive commands
- Shell injection

Trust levels:
- `builtin` — ships with Hermes (always trusted)
- `official` — from `optional-skills/` in the repo (built-in trust, no third-party warning)
- `trusted` — from openai/skills, anthropics/skills, huggingface/skills
- `community` — non-dangerous findings can be overridden with `--force`; `dangerous` verdicts remain blocked

Hermes can now consume third-party skills from multiple external discovery models:
- direct GitHub identifiers (for example `openai/skills/k8s`)
- `skills.sh` identifiers (for example `skills-sh/vercel-labs/json-render/json-render-react`)
- well-known endpoints served from `/.well-known/skills/index.json`

If you want your skills to be discoverable without a GitHub-specific installer, consider serving them from a well-known endpoint in addition to publishing them in a repo or marketplace.
