---
sidebar_position: 10
title: "Full Hermes Gateway Setup Runbook"
description: "Interactive runbook for setting up Hermes as an always-on assistant with model providers, Discord, WhatsApp, app integrations, and a local Ollama fallback."
---

# Full Hermes Gateway Setup Runbook

This runbook is for a new Hermes agent that needs to reproduce a production-style personal assistant setup with minimum user input.

The target state is:

- Hermes installed and healthy on the host machine.
- A strong hosted model is the primary provider.
- API keys or OAuth logins are configured through Hermes' own auth/config flows.
- Core toolsets are enabled.
- Gateway is installed as a persistent service.
- Discord is connected.
- WhatsApp can be connected, including a private self-chat/group pattern.
- Optional app integrations are discovered and configured one by one.
- A local Ollama model is configured as an emergency fallback for quota, rate-limit, or provider outages.
- The final setup is verified from CLI and from each messaging platform.

Do not collect secrets in chat. Put secrets in `~/.hermes/.env`, Hermes auth storage, the OS keychain, or provider-specific OAuth storage. Redact secrets in logs and summaries.

## Bootstrap Prompt for the New Hermes Agent

Paste this prompt into the new Hermes session:

```text
You are setting up Hermes Agent as my always-on personal assistant.

Goal:
- Install or repair Hermes on this machine.
- Configure a primary hosted model provider.
- Determine whether this machine is the master orchestrator Hermes computer or a secondary Hermes node.
- Configure Discord gateway access.
- Configure WhatsApp gateway access only on the master orchestrator unless I explicitly ask for WhatsApp on this machine.
- Configure app integrations I choose from a checklist.
- Configure a local Ollama fallback so Hermes can keep responding if the hosted provider hits limits or quota.
- Install Hermes Gateway as a persistent service and verify end-to-end messages.

Operating rules:
- Be interactive but minimize questions. Discover everything you can from the machine first.
- Ask only one question at a time when user input is required.
- Never ask me to paste a secret into chat unless there is no safer route. Prefer `hermes auth add`, provider OAuth, browser login, or editing the env file printed by `hermes config env-path`.
- Redact all API keys, tokens, passwords, cookies, and connection strings as `[REDACTED]`.
- Before changing config, back it up.
- After each integration, verify it with the CLI or a harmless test message.
- If something breaks, diagnose root cause before trying fixes.

Start by loading the `hermes-agent` skill, then run this runbook from discovery through verification.
```

## Phase 0: Safety and Discovery

Run discovery before asking for anything:

```bash
hermes --version || true
hermes config path || true
hermes config env-path || true
hermes config check || true
hermes doctor || true
hermes gateway status || true
command -v ollama || true
lsof -nP -iTCP:11434 -sTCP:LISTEN || true
```

Find the Hermes home and config paths:

```bash
CONFIG_PATH=$(hermes config path)
ENV_PATH=$(hermes config env-path)
echo "Config: $CONFIG_PATH"
echo "Env: $ENV_PATH"
```

Back up config before edits:

```bash
cp "$CONFIG_PATH" "$CONFIG_PATH.backup-$(date +%Y%m%d-%H%M%S)"
```

If Hermes is not installed, install it:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Then run:

```bash
hermes setup
hermes config check
hermes doctor
```

## Phase 1: Minimum User Questions

Ask these only after discovery, and only if the answer cannot be inferred.

Ask one at a time:

1. "Is this computer the master orchestrator Hermes node, a secondary Hermes node, or are you not sure?"
2. "Which hosted model provider should be primary? Recommended: OpenRouter, Anthropic, OpenAI, or Nous Portal."
3. "Should I enable Discord gateway on this machine? Recommended yes for the master orchestrator."
4. "Should I enable WhatsApp gateway on this machine? Recommended only for the master orchestrator unless you explicitly want WhatsApp attached here."
5. "Which optional app integrations do you want now? Choose any: GitHub, Google Workspace/Gmail/Calendar/Drive, Slack, Telegram, Notion, Linear, Spotify, Home Assistant, email/IMAP, SMS, Matrix, Feishu/Lark, WeCom, custom MCP servers."
6. "Do you want local Ollama fallback? Recommended yes for the master orchestrator and any always-on nodes."

Use the answer to question 1 to set defaults:

- **Master orchestrator:** configure persistent gateway, Discord, WhatsApp/private-group access, local fallback, cron, memory, and the broadest app integration set.
- **Secondary node:** configure the primary model, core toolsets, local fallback if useful, and only the app/platform integrations specifically needed on that machine. Do not attach WhatsApp by default.
- **Not sure:** infer from uptime and role. A 24/7 desktop/server/Mac mini that should receive phone or Discord messages is usually the master orchestrator; a laptop or task-specific workstation is usually secondary.

For every selected integration, walk the user through only the credential step that cannot be automated.

## Phase 2: Primary Model Provider

Preferred safe auth flow:

```bash
hermes auth add
```

For OAuth providers:

```bash
hermes login --provider nous
hermes login --provider openai-codex
```

For API-key providers, if `hermes auth add` is insufficient, print the env path and tell the user exactly what variable to add:

```bash
hermes config env-path
```

Common env vars:

```bash
OPENROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
GROQ_API_KEY=...
HF_TOKEN=...
```

After auth, choose model/provider:

```bash
hermes model
hermes config check
hermes chat -q "Reply with exactly: model-ok" --quiet
```

Expected response contains:

```text
model-ok
```

## Phase 3: Core Toolsets

Enable toolsets needed for a full assistant:

```bash
hermes tools list
hermes tools enable terminal
hermes tools enable file
hermes tools enable web
hermes tools enable browser
hermes tools enable vision
hermes tools enable skills
hermes tools enable memory
hermes tools enable session_search
hermes tools enable delegation
hermes tools enable cronjob
```

Some tool changes require a new session or gateway restart.

## Phase 4: Discord Gateway

Use the official wizard when possible:

```bash
hermes gateway setup
```

When Discord is selected, the user generally needs to:

1. Create or open an app at `https://discord.com/developers/applications`.
2. Create a bot user.
3. Copy the bot token into the Hermes env/config flow, not into chat.
4. Invite the bot to the target Discord server with required permissions.
5. Decide the home channel and whether DMs, mentions, or free-response channels are allowed.

If editing manually, use `hermes config env-path` for secrets and `hermes config edit` for non-secret config.

Verification:

```bash
hermes gateway restart
hermes gateway status
tail -100 ~/.hermes/logs/gateway.log
```

Look for a marker like:

```text
✓ discord connected
Gateway running
```

Then send a harmless message in the Discord home channel and confirm Hermes replies.

## Phase 5: WhatsApp Gateway

WhatsApp should normally be attached only to the master orchestrator Hermes computer. This prevents multiple Hermes nodes from competing for the same WhatsApp session or replying in the same private chat. If this machine is secondary, skip this phase unless the user explicitly wants WhatsApp on this node.

Run the WhatsApp setup/pairing flow:

```bash
hermes gateway setup
# or, when available:
hermes whatsapp
```

If a terminal QR code is hard to scan, prefer a generated PNG QR image if the bridge writes one, commonly under:

```text
~/.hermes/whatsapp/whatsapp-qr.png
```

For a private WhatsApp group pattern:

1. Create a temporary group with another person.
2. Remove the other person so it remains a private group.
3. Pair WhatsApp Web/bridge with Hermes.
4. Send a test message in the group.
5. Run bridge debug or inspect logs to discover the group JID. Group IDs look like `120...@g.us`.
6. Configure that group as the home/allowlisted chat.

Typical config shape:

```yaml
platforms:
  whatsapp:
    enabled: true
    extra:
      bridge_port: 3010
      group_policy: allowlist
      group_allow_from:
        - <group_jid>
      free_response_chats:
        - <group_jid>
      require_mention: false
```

Use port `3010` if port `3000` is occupied.

Verification:

```bash
hermes gateway restart
hermes gateway status
tail -100 ~/.hermes/logs/gateway.log
```

Look for:

```text
✓ whatsapp connected
```

Then send a harmless message from the WhatsApp group and confirm Hermes replies.

## Phase 6: Optional App Integrations Checklist

Use this as an interactive menu. Ask the user which items they want, then configure only those.

### GitHub

Check auth:

```bash
gh auth status || true
```

If needed:

```bash
gh auth login
```

Verify:

```bash
gh repo list --limit 5
```

### Google Workspace / Gmail / Calendar / Drive

Load and follow the Google Workspace skill if available. Prefer OAuth. Verify with a harmless list command, such as listing calendars or Drive files. Never print OAuth tokens.

### Slack

Use `hermes gateway setup` or Slack platform docs. The user needs a Slack app/bot token and signing secret. Put both in env/config, not chat. Verify with gateway logs and a test message.

### Telegram

Use `hermes gateway setup`. The user creates a bot with BotFather and enters the token through env/config. Verify with gateway logs and a direct bot message.

### Notion

Use an internal integration token and database/page IDs. Store the token in env. Verify with a harmless page/database read.

### Linear

Use a Linear API key or OAuth if configured. Store in env. Verify with a team/project query.

### Spotify

Use the Spotify skill or provider setup. Prefer OAuth. Verify by listing devices or current playback.

### Home Assistant

Use a long-lived access token and base URL. Store token in env. Verify by listing states or devices.

### Email / IMAP / SMTP

Use the email skill if available. Prefer app passwords or OAuth-specific flows. Verify with a mailbox list or recent message search; do not send test mail unless the user approves recipients.

### MCP Servers

List configured servers:

```bash
hermes mcp list
```

Add servers interactively:

```bash
hermes mcp add <name>
hermes mcp test <name>
hermes mcp configure <name>
```

## Phase 7: Local Ollama Fallback

Use this when the primary provider is hosted but the user wants a local backup for quota/rate-limit outages.

Check Ollama:

```bash
command -v ollama
lsof -nP -iTCP:11434 -sTCP:LISTEN
ollama list
```

If Ollama is missing, install it from `https://ollama.com` or the platform package manager.

Pick the best local model available. For this example, the base model is `qwen3:8b`.

Create a 64K wrapper Modelfile:

```Modelfile
FROM qwen3:8b
PARAMETER num_ctx 65536
PARAMETER temperature 0.2
SYSTEM "You are a local backup model for Hermes Agent. Be concise and direct. Do not reveal hidden reasoning; provide only the final answer."
```

Save and create the wrapper:

```bash
mkdir -p ~/.hermes
$EDITOR ~/.hermes/ollama-qwen3-hermes.Modelfile
ollama create qwen3:8b-hermes -f ~/.hermes/ollama-qwen3-hermes.Modelfile
ollama show qwen3:8b-hermes --parameters
```

Add fallback provider and explicit context override to `~/.hermes/config.yaml`:

```yaml
fallback_providers:
  - provider: custom
    model: qwen3:8b-hermes
    base_url: http://127.0.0.1:11434/v1
    api_key: no-key-required

custom_providers:
  - name: ollama-local
    base_url: http://127.0.0.1:11434/v1
    api_key: no-key-required
    api_mode: chat_completions
    model: qwen3:8b-hermes
    context_length: 65536
    models:
      qwen3:8b-hermes:
        context_length: 65536

providers:
  ollama-local:
    name: Ollama Local
    base_url: http://127.0.0.1:11434/v1
    api_key: no-key-required
    default_model: qwen3:8b-hermes
    transport: chat_completions
```

Validate:

```bash
hermes config check
hermes chat --provider ollama-local --model qwen3:8b-hermes \
  -q 'Reply with exactly: local-ok' \
  --toolsets '' \
  --quiet
```

Expected:

```text
local-ok
```

Pitfall: Ollama may report the base model's training context even after setting `num_ctx 65536`. The `custom_providers.models.<model>.context_length: 65536` override is what makes Hermes accept the local fallback as satisfying the 64K minimum.

## Phase 8: Install or Restart Gateway Service

Install gateway as a persistent service if not already installed:

```bash
hermes gateway install
hermes gateway start
```

For config changes:

```bash
hermes gateway restart
```

Verify:

```bash
hermes gateway status
tail -200 ~/.hermes/logs/gateway.log
```

Expected final markers depend on enabled platforms, but should include:

```text
Gateway running
Cron ticker started
```

And for enabled platforms:

```text
✓ discord connected
✓ whatsapp connected
✓ telegram connected
✓ slack connected
```

## Phase 9: Final End-to-End Verification

Run these checks:

```bash
hermes config check
hermes doctor
hermes chat -q "Reply with exactly: cli-ok" --quiet
hermes chat --provider ollama-local --model qwen3:8b-hermes -q "Reply with exactly: local-ok" --toolsets '' --quiet || true
hermes gateway status
tail -100 ~/.hermes/logs/gateway.log
```

Then verify each selected messaging platform manually:

- Discord: send a message in the home/free-response channel.
- WhatsApp: send a message in the private group or configured chat.
- Telegram/Slack/etc.: send a simple test message.

Expected responses:

```text
cli-ok
local-ok
```

And a normal gateway reply in each messaging app.

## Phase 10: Final Handoff Summary Template

At the end, report this to the user with secrets redacted:

```text
Hermes setup complete.

Primary provider:
- Provider: <provider>
- Model: <model>
- Auth: configured via <auth method>, secret redacted

Gateway:
- Installed as service: yes/no
- Enabled platforms: <platform list>
- Home channels/chats: <non-secret IDs or names>

Local fallback:
- Ollama running: yes/no
- Endpoint: http://127.0.0.1:11434/v1
- Model: <wrapper model>
- Context override: 65536
- Direct local test: passed/failed

App integrations:
- GitHub: configured/not configured
- Google Workspace: configured/not configured
- Slack: configured/not configured
- Notion: configured/not configured
- Linear: configured/not configured
- Spotify: configured/not configured
- Home Assistant: configured/not configured
- Other: <list>

Verification:
- hermes config check: passed/failed
- hermes doctor: passed/failed
- CLI model test: passed/failed
- Gateway status: running/not running
- Platform message tests: <results>

Files changed:
- ~/.hermes/config.yaml
- ~/.hermes/.env or auth store, secrets redacted
- Any Modelfile or service files

Known issues / next steps:
- <only real unresolved items>
```

## Troubleshooting

### Gateway says platform connected but no replies

Check platform-specific allowlists, free-response channels, mention requirements, and DM authorization:

```bash
hermes gateway status
grep -i "error\|failed\|ignored\|allow\|mention" ~/.hermes/logs/gateway.log | tail -100
```

### Config changes do not take effect

Restart the gateway:

```bash
hermes gateway restart
```

For toolset changes, start a new session or reset the current session.

### Local fallback is too slow

Use a smaller model, reduce context if acceptable, add RAM/VRAM, or keep fallback as emergency-only. For a persistent service, keep the model warm:

```bash
curl http://127.0.0.1:11434/api/generate \
  -d '{"model": "qwen3:8b-hermes", "keep_alive": "24h"}'
```

### Provider keys fail

Use Hermes auth commands first:

```bash
hermes auth list
hermes auth add
hermes auth reset <provider>
```

Then check `hermes config env-path` and verify the relevant env var is present without printing the secret.

### WhatsApp QR does not scan

Use the PNG QR if available, enlarge the terminal, or retry pairing. Avoid scanning a distorted compact terminal QR.

## Minimal Input Principle

The new Hermes should never ask the user to perform work it can do itself. It should:

1. Discover installed tools and current config.
2. Back up files before editing.
3. Use wizards and OAuth flows where available.
4. Ask for a credential only at the moment it is required.
5. Give the exact URL or command for creating the credential.
6. Wait for the user to confirm completion.
7. Store the secret safely without echoing it back.
8. Immediately verify the integration.
9. Continue to the next selected app.
