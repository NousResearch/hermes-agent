---
title: Setup Codex / 配置宝典
sidebar_position: 9
---

# Setup Codex / Hermes配置宝典

Setup Codex is the built-in, read-only setup guide in the Hermes Dashboard. It is designed as an official “configuration cockpit” for new users: pick a setup goal, read the current local state, copy the suggested commands, and complete the change manually.

Open it from the Dashboard sidebar as **配置宝典**, or visit:

```text
http://127.0.0.1:9119/setup-codex
```

## MVP safety boundary

The first version is intentionally read-only:

- It does **not** write `config.yaml`.
- It does **not** write `.env`.
- It does **not** execute shell commands.
- It does **not** restart the gateway.
- It does **not** return secret values to the browser.

The page can show that a credential exists, but never the credential itself.

## Read-only state API

The page reads its status from:

```http
GET /api/setup-codex/state
```

This endpoint returns setup-ready state such as:

- Hermes home, config path, and env path.
- Current model/provider names.
- Whether model base URL or API key fields are configured.
- Delegation/subagent settings.
- Gateway running state and platform runtime state.
- Telegram, Feishu/Lark, API server, Discord, and Slack configuration presence.
- Platform toolsets and disabled/enabled toolsets.
- Secret key names and counts only, with values redacted.
- Safety flags confirming that the guide cannot write files or execute commands.

The endpoint must never return API keys, tokens, OAuth credentials, cookies, passwords, private keys, connection strings, or raw `.env` values.

## Guide topics

The Dashboard page includes starter cards for:

- New Agent
- Telegram
- Feishu / Lark
- Model selection
- Tool permissions / toolsets
- Subagents / delegation
- Gateway troubleshooting
- Security checks

Each topic provides small steps and copyable commands. Users run commands themselves in a terminal.

### New Agent guided wizard

The **新建 Agent** card includes a small command generator. It asks for:

- agent type, such as personal assistant, coding assistant, group-chat bot, or research assistant;
- profile name, validated with the safe pattern `^[a-z][a-z0-9-]{1,31}$`;
- entry platform, such as CLI, Telegram, Feishu/Lark, or API server;
- a recommended toolset template.

The generated commands are still copy-only. The Dashboard does not create the profile, write config, write `.env`, execute the command, or restart Gateway.

### Windows launcher prototype

A conservative launcher prototype lives at:

```text
scripts/setup_codex_launcher.py
```

It is intended to be packaged as **Hermes配置宝典.exe** later. The prototype:

- checks `http://127.0.0.1:9119/api/status`;
- starts `hermes dashboard --host 127.0.0.1 --port 9119` only if no dashboard is reachable;
- waits for readiness, then opens `http://127.0.0.1:9119/setup-codex`;
- refuses non-localhost hosts, and never passes `--host 0.0.0.0` or `--insecure`.

Run it directly during development:

```bash
python scripts/setup_codex_launcher.py --no-open
python scripts/setup_codex_launcher.py
```

## Future phases

Later versions may add generated YAML patches and controlled apply flows. Those flows must preserve the same security model:

1. Generate a human-readable diff first.
2. Redact all secret values.
3. Require explicit confirmation before applying changes.
4. Keep high-risk operations, such as gateway restart or tool permission expansion, behind an additional confirmation step.
