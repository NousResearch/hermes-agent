---
sidebar_position: 9
title: "Privacy and Data"
description: "What leaves your machine, when, to whom, what is opt-in, and the config key that turns each path off"
---

# Privacy and Data

Hermes runs on your machine and keeps its state there. This page lists every path by which data leaves the machine, what is on by default, what is opt-in, and the config key or command that turns each one off. Config keys are `config.yaml` paths in the active profile's home (`~/.hermes/` for the default profile; see [Profiles](./profiles.md)).

## What stays local

Sessions, transcripts, memory, skills, cron jobs, logs, `config.yaml`, `.env` and `auth.json` are files under the profile's home directory. Nothing uploads them on its own. They leave the machine only as part of a model request (below), a debug upload you run, or a backup or sync you set up yourself.

Hermes has no crash reporter. On Linux the Desktop app keeps Chromium crash minidumps in a local database with uploading turned off. Crash and process-exit counts are part of shared metrics, which are off unless you opt in.

## At a glance

| Data path | Default | Goes to | Turn it off |
|---|---|---|---|
| [Main model calls](#model-providers) | On (it is how the agent works) | The provider in `model.provider` / `model.base_url` | Use a local endpoint (`model.provider: custom`, `model.base_url: http://localhost:...`) |
| [Auxiliary model calls](#auxiliary-models) | On | `auxiliary.<task>.provider`; `auto` = your main model first | Pin each `auxiliary.<task>` to a local endpoint and keep the main model local (a failing pin falls back to it) |
| [Shared metrics](#shared-metrics) | **Off** (opt-in) | `telemetry.shared_metrics.endpoint` (Nous telemetry service) | `telemetry.shared_metrics.enabled: false` and `send: false`, or `hermes setup telemetry` |
| [Passive update checks](#update-checks) | On | GitHub (`api.github.com`), or `git ls-remote` on a non-GitHub `origin` | `updates.check: false` (CLI, TUI, dashboard); not the Desktop app |
| [Desktop update checks](#update-checks) | On | GitHub (source installs) or the Desktop release feed (packaged apps) | No setting |
| [Model registry](#model-registry) | On | `https://models.dev/api.json` | `models_dev.url` points it at a mirror; no off switch |
| [Plugin update checks](#plugin-update-checks) | On while a gateway runs | Each installed plugin's feed, git remote or PyPI | `plugins.auto_update_check_hours: 0` |
| [Nous free-tier identity](#nous-account-free-tier-and-connectors) | On (created on first use) | Nous Portal | `nous.guest: false` |
| [Connector search](#nous-account-free-tier-and-connectors) | On once a Nous identity exists | Nous connector gateway | `tools.connectors.enabled: false` |
| [Tool backends](#tool-backends) | Only for tools you use | The backend you select per tool category | Pick another backend in `hermes tools` |
| [Debug uploads](#debug-uploads) | Never automatic | paste.rs / dpaste.com, or Nous storage with `--nous` | Don't run it; `hermes debug share --local` shows the report without uploading |
| [Messaging, MCP, memory providers, plugins](#integrations-you-add) | Only when configured | The service you configured | Remove the integration |
| [Desktop link titles](#desktop-link-titles) | On | The linked website | No setting |
| [Computer-use driver telemetry](#computer-use-driver) | **Off** | The driver's upstream analytics | `computer_use.cua_telemetry: false` (default) |

## Model providers

Every turn sends the conversation to the provider you configured: the system prompt (including memory and context files), the message history, tool definitions, tool results, and any files or images you attach. The provider's own data policy applies to that traffic. Choose the provider with `hermes model`; the result is stored in `model.provider`, `model.default` and `model.base_url`. To keep inference on the machine, point it at a local server (see [Local Models](./local-models.md)).

[Fallback providers](./features/fallback-providers.md) in `fallback_providers` receive the same request when the primary fails. Leave the list empty to never send to another provider.

On OpenRouter, the Vercel AI Gateway, Fireworks, OpenCode Zen, Kimi Coding and the Perplexity web backend, Hermes adds app-attribution headers to every request, such as `HTTP-Referer: https://hermes-agent.nousresearch.com` and `X-Title: Hermes Agent` (some routes also send `User-Agent: HermesAgent/<version>`). They name the app and its version, not you. They are always sent on those routes, and no key turns them off.

### Auxiliary models

Side tasks call a model too: context compression sends the older part of the conversation to a summarizer, title generation sends the first exchange, `vision` sends the image, and the `approval` classifier sends a flagged command. Each task reads `auxiliary.<task>.provider` and `auxiliary.<task>.model` (see [Auxiliary Models](./configuration.md#auxiliary-models)).

`provider: auto` (the default) uses your main provider and model. If that call fails, Hermes tries the task's `auxiliary.<task>.fallback_chain`, then your `fallback_providers`. Only when no main provider is selected (`model.provider` empty or `auto`) does it go on to other providers you have credentials for, in order: OpenRouter, Nous Portal, a custom endpoint, native Anthropic, then direct API-key providers.

Setting `provider` and `model` (or `base_url`) on an `auxiliary.<task>` block sends that task to the endpoint you name, but a pin is not a guarantee. When the pinned endpoint cannot serve the call (connection failure, timeout, or quota or rate limit after retries), Hermes tries the task's `fallback_chain` and then your main model. To keep side tasks on the machine, pin them to a local endpoint and keep the main model and any `fallback_chain` entries local too. `auxiliary.free_only: true` skips paid OpenRouter models in the auto chain. `auxiliary.title_generation.model_upgrade_enabled: false` keeps the instant local title and never calls a model for it.

## Shared metrics

Shared metrics are aggregate usage counters that Hermes can send to Nous Research once a day. They are **off by default**, and nothing is collected or sent until you say yes.

**What is sent.** Bounded counters only: activity, session length, outcomes, error classes, model and provider names, token totals, built-in tool, command and catalog names, update and install results, crash and process-exit counts, startup and reply timing, messaging-platform health, which features and settings are used, and coarse machine facts (Hermes version, OS family, architecture, install method, RAM range, GPU type). Each field is checked against a fixed allowlist before it is stored. Prompts, responses, file contents, paths, setting values, endpoints, error text, and session IDs are never included. The full field list is in the [shared-metrics developer reference](../developer-guide/relay-shared-metrics.md#current-slices).

**Identity.** Each package carries an `install_id`: a random UUID generated for the profile, with no hardware, account or user information in it. It stays the same across days until you delete `<profile home>/telemetry/shared_metrics/`, which creates a new one.

**When it is asked.** Once per profile, at the end of `hermes setup`, before the first interactive `hermes` chat, in a strip above the Desktop composer, or in a dashboard banner. "No thanks" is the default in the terminal. Managed installs are never asked.

**Config keys.**

```yaml
telemetry:
  shared_metrics:
    enabled: false   # collect counters locally
    send: false      # upload packages to the Nous telemetry service
    endpoint: https://telemetry.nousresearch.com/v1/telemetry
```

- `send` does nothing unless `enabled` is also true. Both are written together by every consent surface.
- The endpoint must be HTTPS (plain HTTP is accepted only for localhost). It can be changed only in `config.yaml`, not by an environment variable.
- Only packages whose whole collection period falls inside a time when `send` was on are uploaded. Data from before you opted in, or from a period when sending was off, is never sent.
- Setting `send: false` stops uploads at once. It does not delete packages already sent; there is no self-service remote delete. Server retention is in the [developer reference](../developer-guide/relay-shared-metrics.md#appendix-a-remote-exporter-decisions-phase-2).

**Turn it off or on.** `hermes setup telemetry`, the shared-metrics row in `hermes tools`, Desktop **Settings › Safety › Privacy & network**, or edit the two keys above.

## Update checks

**CLI, TUI and dashboard.** On startup Hermes checks whether a newer version exists, at most once every 24 hours per install (one hour after a failed check). The check asks the GitHub REST API (`api.github.com`) for the tip of your branch and, when your checkout does not already contain it, calls the compare endpoint with your installed commit. A checkout whose `origin` is not a GitHub repository runs `git ls-remote` on that remote instead. Set `updates.check: false` to stop these passive checks. `hermes update` and `hermes update --check` still contact GitHub when you run them. See [Update Behavior](./configuration.md#update-behavior).

**Desktop app.** The Desktop app checks on launch, then every 24 hours and when its window gains focus if 24 hours have passed. Source installs run the same GitHub check; packaged apps read the Desktop release feed. `updates.check` does not govern these checks, and there is no setting to turn them off.

## Model registry

Hermes downloads `https://models.dev/api.json` (context windows, capabilities and pricing per model) on startup and refreshes it in the background, at most every 4 hours, using a conditional request. It is a plain download; nothing about your sessions is sent. `models_dev.url` points it at a mirror, such as one behind a corporate proxy. There is no off switch.

## Plugin update checks

While a gateway runs, it checks installed plugins for updates once every `plugins.auto_update_check_hours` hours (default `24`). The check reads each plugin's update feed, git remote, or PyPI entry. It never installs anything unless you also set `plugins.auto_apply: true`. Set `plugins.auto_update_check_hours: 0` to turn the check off; `hermes plugins check-updates` still runs it on demand.

## Nous account, free tier, and connectors

**Free-tier identity.** With `nous.guest: true` (the default), Hermes can create a free Nous identity for you. This happens in two places only: the Desktop app's first-run welcome trial, and the first hosted-connector action (`manage_connections`, or connecting an app on the Desktop Connectors page) when you have no Nous account. Creating it calls the Nous Portal (`/api/anonymous/create`). Each later token exchange sends a client block with the app name, Hermes version, surface, and OS platform. Set `nous.guest: false` to never create one.

**Connector search.** Connectors are hosted apps, such as Gmail, served by the Nous tool gateway. Once you have a Nous identity (a free-tier identity, or a signed-in account the Portal has enabled connectors for), every `tool_search` query is also sent to the connector gateway, and connector tool calls run there. Until an identity exists, `tool_search` sends nothing to the gateway. Set `tools.connectors.enabled: false` to never contact it. See [Tool Search › Connectors](./features/tool-search.md#connectors-remote-tools).

**Signed-in Nous accounts** refresh their credentials in the background. `nous.keepalive_interval_seconds: 0` stops the refresh thread.

## Tool backends

Web search and extraction, image generation, text-to-speech, speech-to-text, and cloud browsers send the tool's input (search query, URL, prompt, text, audio, page actions) to the backend selected for that category: `web.backend` (or `web.search_backend` / `web.extract_backend`), `image_gen.provider`, `tts.provider`, `stt.provider`, `browser.cloud_provider`. The value `nous` routes through the [Nous Tool Gateway](./features/tool-gateway.md); a vendor name sends directly to that vendor with your key. Change the selection in `hermes tools`. A tool you never call sends nothing.

## Debug uploads

Nothing uploads debug data automatically. These commands upload when you run them:

| Command | Goes to | Contents |
|---|---|---|
| `hermes debug share` | paste.rs (fallback: dpaste.com) as a **public** URL | System info and recent agent, gateway, dashboard and Desktop logs (up to 512 KB each). Asks for confirmation first; `--yes` skips it. |
| `/debug` in the CLI or TUI | Same as above | Same as above. Typing the command is the confirmation. |
| `/debug` in a messaging chat | Same as above | System info and log tails only, no full logs. Runs without a confirmation step. |
| `hermes debug share --nous` (and Desktop's send-diagnostics dialog) | Nous diagnostics storage, viewable only by Nous staff | Same bundle. Auto-deletes after 14 days. |

Secrets (API keys, tokens, passwords) are redacted before upload unless you pass `--no-redact`. Personal data is **not** redacted: your display name, platform user ID, message content in the logs, and local paths go out as written. Deleting a paste.rs upload is best effort: Hermes queues the delete for 6 hours after the upload, and it runs only when a gateway's hourly sweep or your next `hermes debug` or `/debug` command gets to it. With no gateway running and no later debug command, the paste stays up; a delete that keeps failing is abandoned a day later. dpaste.com fallback pastes cannot be deleted and stay public for 1 day from `hermes debug share` (change it with `--expire`) and from `/debug` in a messaging chat, and for 7 days from `/debug` in the CLI or TUI. `hermes debug share --local` and `/debug local` show the report without uploading. See [`hermes debug`](../reference/cli-commands.md#hermes-debug).

## Integrations you add

These send data only after you configure them, and only to the service you chose:

- **Messaging platforms** (Telegram, Discord, Slack, ...): messages to and from the agent go through that platform. See [Messaging](./messaging/index.md).
- **MCP servers**: tool arguments go to the server you added under `mcp_servers`.
- **Memory providers** (for example [Honcho](./features/honcho.md)): conversation content goes to the provider set in `memory.provider`.
- **Plugins and skills**: they run with Hermes's permissions and can make their own network calls. Review what you install.

## Desktop link titles

When the Desktop app shows a link without a title, it fetches the page title and favicon directly from that website, with a browser user agent, so the link reads as a name instead of a URL. The site sees a request from your IP. There is no setting to turn this off.

## Computer-use driver

The computer-use driver ships with its own upstream analytics turned on. Hermes turns it off in every driver process it starts unless you set `computer_use.cua_telemetry: true`.

## Related

- [Security](./security.md): approvals, sandboxing, and secret redaction.
- [Secure Hermes on a work machine](../guides/secure-hermes-on-a-work-machine.md)
- [Network isolation](./egress/network-isolation.md): restrict where Hermes can connect at all.
