# Recipes — Shareable Setup Bundles

Recipes let you share a working Hermes setup — cron automations, remote MCP
integrations, recommended skills, and a starter prompt — as a single YAML
file that anyone can preview and install with one command.

Inspired by Poke's Recipes feature (shareable one-link setups), adapted for a
self-hosted, security-first agent: **secrets never travel**, installs are
**consent-first**, and recipes can never carry executable configuration.

## Sharing your setup

```bash
hermes recipe export \
  --name "AI Research Kit" \
  --description "Daily arXiv digest + Exa search" \
  --jobs "ai digest" \
  --mcp exa \
  --skills official/research/arxiv \
  --starter-prompt "Give me today's paper highlights" \
  -o research-kit.yaml
```

Share `research-kit.yaml` anywhere — a gist, a repo, Discord. Anything with
an https URL works as an install source.

What export includes and strips:

- **Cron jobs** — prompt, schedule, skills, toolset restrictions. Run
  history, chat-specific delivery targets, and `script`/`monitor_script`
  fields are stripped (script jobs can't be exported at all — they're
  host-specific and executable).
- **MCP servers** — remote (http/sse) servers only, minus every credential:
  `headers`, `env`, and any secret-shaped key are removed and recorded by
  *name* in `required_secrets` so installers know what to supply.
- **Skills** — referenced by hub identifier, not bundled; installs go
  through the normal `hermes skills install` quarantine/consent flow.

## Installing a recipe

```bash
# Preview first — shows everything the recipe would add
hermes recipe show https://example.com/research-kit.yaml

# Install (interactive confirmation; --yes to skip)
hermes recipe install research-kit.yaml
```

Install behavior:

- **Cron jobs are created paused.** Review them, then
  `hermes cron resume <id>` — or pass `--enable` to activate immediately.
- **MCP servers are merged into `config.yaml`.** Existing entries with the
  same name are never overwritten. Server URLs are validated through the
  SSRF guard before being written.
- **Missing credentials are called out.** Any `required_secrets` the recipe
  recorded are printed with instructions — recipes never contain keys.
- **Skills are suggested, not auto-installed.** The output prints the
  `hermes skills install …` commands so the hub's trust flow stays in charge.

## Security model

A recipe is data, not a program:

- stdio MCP servers (`command:`) are refused on export *and* install.
- Cron `script`, `monitor_script`, `no_agent`, and `workdir` fields are
  refused on install.
- Secret-shaped fields anywhere in a recipe fail validation.
- URL sources are fetched through the SSRF-safe client with a 256 KiB cap.

## Recipe format

```yaml
recipe: 1
name: AI Research Kit
description: Daily arXiv digest + Exa search
author: yourhandle
starter_prompt: Give me today's paper highlights
skills:
  - official/research/arxiv
cron_jobs:
  - name: ai digest
    prompt: Summarize today's AI news
    schedule: 0 8 * * *
    deliver: local
mcp_servers:
  exa:
    transport: http
    url: https://mcp.exa.ai/mcp
required_secrets:
  exa:
    - api_key
```
