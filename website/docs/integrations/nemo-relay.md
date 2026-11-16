---
sidebar_position: 5
title: "NeMo Relay"
description: "Record Hermes model and tool calls, export traces, and configure NeMo Relay plugins"
---

# NeMo Relay

[NeMo Relay](https://docs.nvidia.com/nemo/relay/v0.11.1/about-nemo-relay/overview) records Hermes sessions, turns, model calls, and tool calls. Use it to investigate a failed run, see retries and delegated work, or send traces to a collector such as Arize Phoenix.

Hermes supports Relay directly. Start Hermes normally; you do not need a separate Relay daemon, the `nemo-relay run` command, or a Hermes observability plugin.

:::note Different from Hermes Relay
NeMo Relay handles agent observability and middleware. [Hermes Relay](../user-guide/messaging/relay.md) connects a Hermes gateway to messaging platforms.
:::

## Before you start

Complete the normal Hermes setup and confirm that you can run a conversation with your model provider.

Hermes uses `nemo-relay>=0.11,<0.12`. Its dependency configuration installs Relay on **Python 3.14** on these platforms:

| Operating system | Architecture |
|---|---|
| macOS | Apple silicon |
| Linux, including glibc and musl distributions | x86_64 or ARM64 |
| Windows | x86_64 or ARM64 |

Intel Macs and Android/Termux are not supported by this integration. A missing Relay binding does not prevent Hermes from running, but Relay recording and middleware are unavailable. Use the normal Hermes install or update process rather than changing packages in its managed environment. See Relay's [support matrix](https://docs.nvidia.com/nemo/relay/v0.11.1/reference/support-matrix).

### Existing Relay configuration

Hermes loads Relay's normal user configuration, followed by the higher-precedence system configuration. An administrator may therefore already have configured exporters or middleware for this machine. Run `hermes doctor` to see which files apply.

`HERMES_NEMO_RELAY_PLUGINS_TOML` is an **optional override**: it replaces the user file, not the system file. The Quick Start uses it to select one explicit file. You do not need to set it when using Relay's standard configuration locations.

An enabled Relay observability component also recognizes standard `OTEL_EXPORTER_OTLP_*ENDPOINT` environment variables for automatic export. Check inherited OTLP settings before running a local-only test. See [configuration](#how-configuration-is-applied) and [data and privacy](#data-and-privacy).

## Quick Start: write a local trace

### 1. Create the configuration file

Save the following as `relay-plugins.toml` in the directory where you will start Hermes:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4
enable_full_payloads = false

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "./hermes-relay"
filename = "events.jsonl"
mode = "append"
```

This writes [ATOF events](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/atof) to `./hermes-relay/events.jsonl`. Relative output paths use Hermes's working directory, not the configuration file's directory. Use an absolute output path for a service.

`enable_full_payloads = false` reduces repeated model history; it does **not** remove prompts, responses, tool arguments, or tool results. Treat the file as sensitive.

### 2. Select the file and run Hermes

On macOS or Linux:

```bash
export HERMES_NEMO_RELAY_PLUGINS_TOML="$PWD/relay-plugins.toml"
hermes chat --oneshot -q "Use the terminal tool to run pwd, then report the result."
```

On Windows PowerShell:

```powershell
$env:HERMES_NEMO_RELAY_PLUGINS_TOML = Join-Path (Get-Location) "relay-plugins.toml"
hermes chat --oneshot -q "Use the terminal tool to run pwd, then report the result."
```

The override applies to Hermes processes started from that shell. For regular use, configure Relay's user file instead, or set the override in the environment that starts your Hermes service. Restart Hermes after changing Relay configuration.

### 3. Check the output

On macOS or Linux:

```bash
wc -l ./hermes-relay/events.jsonl
tail -n 1 ./hermes-relay/events.jsonl | python3 -m json.tool
```

On Windows PowerShell:

```powershell
(Get-Content .\hermes-relay\events.jsonl).Count
Get-Content .\hermes-relay\events.jsonl -Tail 1
```

Look for records named `hermes.session`, `hermes.turn`, and the model operation. If Hermes ran the requested tool, there will also be `terminal` records. Each scope normally has a start and end record with the same `uuid`; `parent_uuid` links it to its parent. The number of records depends on the run, and append mode keeps previous runs in the same file.

If no file appears, run `hermes doctor` and check `hermes logs --level WARNING`. A Relay initialization failure does not necessarily stop Hermes from answering.

Removing the override restores normal Relay discovery; it does **not** turn off a user or system exporter. To stop exporting, update the configuration that enabled the destination and restart Hermes. `OTEL_SDK_DISABLED=true` disables Relay's OTLP exporters, but not ATOF or ATIF.

## What Hermes records

Relay follows Hermes's `AIAgent` conversation path. This includes conversations started through the CLI, TUI, Desktop, dashboard, ACP, messaging gateway, API server, scheduled jobs, batch jobs, and the Python library when they use that path.

| Activity | Recorded information |
|---|---|
| Session | A `hermes.session` agent scope and the logical working directory when known. |
| Turn | A `hermes.turn` child scope, working directory when known, and outcome. |
| Model call | A `hermes.logical_llm_call` scope with a child for each provider attempt, including retries and fallbacks. Request and response data, available token usage, provider/model metadata, timing, and outcome. |
| Streaming | Stream lifecycle, chunk receipt metadata, available usage and finish information, and a reconstructed final response when supported. Chunk marks are not a copy of every streamed text fragment. |
| Tool call | Tool name, arguments, result, call ID, timing, and outcome. Skill, memory, scheduler, and MCP tools appear as ordinary tool calls. |
| Delegated agent | A child session under the spawning turn when Hermes has that relationship. |

A typical trace reads from `hermes.session` to `hermes.turn`, then to a logical model call, provider attempt, tool, or delegated session. OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages use [provider-aware codecs](https://docs.nvidia.com/nemo/relay/v0.11.1/integrate-into-frameworks/provider-codecs). Other routes use a generic JSON projection and may lack normalized provider fields.

This does not automatically record slash commands, message delivery, voice transcription, scheduler bookkeeping, or work inside an external MCP server or another process. A tool scope shows the Hermes-level call, not everything the called service did internally.

Relay's scope and trajectory IDs are not interchangeable with `hermes sessions` IDs. See the [ATOF format](https://docs.nvidia.com/nemo/relay/v0.11.1/reference/atof-event-format) when joining exported records.

## Export to a collector

Keep a local ATOF file while setting up a remote destination. It helps distinguish a Hermes recording problem from an exporter or collector problem.

### Arize Phoenix

Start a local Phoenix instance using [Phoenix's installation instructions](https://arize.com/docs/phoenix/self-hosting). For a local Docker test:

```bash
docker run -d --rm --name hermes-relay-phoenix \
  -p 127.0.0.1:6006:6006 \
  arizephoenix/phoenix:latest
```

Add this block under the existing observability component in `relay-plugins.toml`:

```toml
[components.config.opentelemetry]
enabled = true

[[components.config.opentelemetry.endpoints]]
type = "openinference"
endpoint = "http://127.0.0.1:6006/v1/traces"
transport = "http_binary"
service_name = "hermes-agent"

[components.config.opentelemetry.endpoints.resource_attributes]
"openinference.project.name" = "hermes-relay"
```

Start a fresh Hermes process and complete a request. Open [http://127.0.0.1:6006](http://127.0.0.1:6006) and select the `hermes-relay` project. Allow a few seconds for batched export. Stop the local test with `docker stop hermes-relay-phoenix`.

This example binds Phoenix to your machine and has no authentication. Use your collector's authentication and deployment instructions for shared installations. The Relay [OpenInference reference](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/openinference) covers the exported fields.

### Other OpenTelemetry collectors

Use the same `opentelemetry` section with an endpoint supported by your collector. Relay supports `full`, `gen_ai`, and `openinference` trace projections. These are different views of the same activity; choose one for each destination rather than exporting several copies unintentionally.

With Relay's observability component enabled, standard OTLP endpoint environment variables can configure automatic export without an explicit endpoint block. Explicitly configured endpoints and automatic exporters can coexist. Review inherited endpoint and header variables before adding another destination. See [OpenTelemetry configuration](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/opentelemetry).

### ATIF trajectories

To write completed trajectories, add:

```toml
[components.config.atif]
enabled = true
agent_name = "Hermes Agent"
model_name = "unknown"
output_directory = "./hermes-relay"
filename_template = "trajectory-{session_id}.json"
```

[ATIF](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/atif) writes a trajectory when its top-level agent scope closes. A long-running gateway session may therefore show turn spans before it has a completed session trajectory. The historical `{session_id}` filename placeholder identifies the agent trajectory scope; it need not match the document's top-level `session_id`.

## Configure Relay plugins

Relay plugins are separate from Hermes plugins. Configure Relay components and dynamic plugins through Relay's [`plugins.toml`](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/plugin-configuration-files), not `hermes plugins enable`.

- **Exporters** receive observability events.
- **Observability sanitizers**, such as PII redaction, change exported copies without changing the real model request or tool result.
- **Execution middleware** can block or change live model or tool calls. This can change what the user sees.

### Redact common values from exports

Add a second component to the same file:

```toml
[[components]]
kind = "pii_redaction"
enabled = true

[components.config]
version = 1

[[components.config.profiles]]
mode = "builtin"
priority = 80

[components.config.profiles.builtin]
action = "redact"
detector = "email"

[[components.config.profiles]]
mode = "builtin"
priority = 90

[components.config.profiles.builtin]
action = "redact"
detector = "api_key"
```

This redacts detected emails and API keys from copied observability data. It does not redact the live tool or model payload, and it does not cover every kind of sensitive data. See [PII redaction](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/pii-redaction/configuration) for detectors and configuration.

### Load a dynamic plugin

Install the plugin and its dependencies using its author's instructions, then add its manifest to the Relay configuration:

```toml
[[plugins.dynamic]]
manifest = "./plugins/example/relay-plugin.toml"
```

The manifest path is relative to the configuration file. If the plugin documents configuration fields, put them under `[plugins.dynamic.config]`.

Hermes passes configuration to Relay's plugin host, which applies its validation, compatibility, integrity, and trust rules. A manifest entry alone does not bypass those checks. Use a plugin release compatible with your Relay version and follow Relay's [installation and validation instructions](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/discoverable-plugins).

Hermes does not download artifacts or install dependencies when a conversation starts. Native plugins run inside the Hermes process; worker plugins run in separate processes, but are not security sandboxes. Restart Hermes after changing the configuration and exercise the plugin's documented behavior to confirm that it works.

:::caution Enforcement coverage
Hermes can continue without Relay plugins after initialization fails. It also skips managed recording and middleware for a later overlapping turn in one session, or a model/tool call nested inside an active managed callback. Do not treat this integration as the only security boundary when every operation must be checked.
:::

For other built-in components, including adaptive caching, pricing, and resource metrics, use the [Relay plugin guides](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/about). Check whether a component changes live execution or only copied events. In particular, observability redaction does not sanitize response-cache contents.

## How configuration is applied

- Without an override, Relay reads its user file and then its system file.
- With `HERMES_NEMO_RELAY_PLUGINS_TOML`, the selected file replaces the user file. The system file still takes precedence.
- Dynamic plugin records follow Relay's configuration and policy rules, not a separate Hermes loader.
- The first hosted profile initializes one Relay plugin host for the process. Other profiles in that process share its middleware, exporter destinations, plugin code, and plugin credentials.
- Sessions remain separately scoped, but that is not separate plugin policy. Use separate Hermes processes when profiles need different destinations or trust settings.
- Changes take effect after the process restarts. Reading a file with no enabled component or selected dynamic plugin does not by itself enable managed middleware.

Run `hermes doctor` to inspect the effective file paths and validation findings. See [Relay's configuration reference](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/plugin-configuration-files) for discovery locations and merge rules.

## Continuous gateway sessions

A gateway session can stay open for days. To close session-level spans and trajectories periodically, configure segmentation in Hermes's `config.yaml`:

```yaml
gateway:
  telemetry:
    session_segments:
      on_compaction: false
      max_turns: 0
```

Set `on_compaction: true` to rotate the Relay session scope after compaction, or set `max_turns` to a positive number to rotate after that many completed turns. Rotation happens at a turn boundary and does not reset the Hermes conversation. The defaults above leave segmentation disabled.

## Data and privacy

ATOF, ATIF, and OpenTelemetry can include prompts, responses, tool arguments and results, working directories, metadata, and error details. Review the destination and the data before exporting.

- `enable_full_payloads = false` reduces repeated history, not all sensitive content.
- Check machine-wide Relay configuration and inherited OTLP endpoint variables; removing an explicit file override is not an export opt-out.
- Prefer SDK or environment authentication for model providers. Custom request fields and `extra_headers` can be present in copied payloads; do not assume custom credentials are stripped.
- Configure and test an observability sanitizer when needed. It changes the exported copy, not data sent to models or tools.
- A normal shutdown lets Relay finish queued work. A crash or forced exit can lose buffered records. Each remote exporter has its own delivery behavior.

### Shared metrics are separate

Hermes's optional [shared metrics](../developer-guide/relay-shared-metrics.md) collect bounded usage counters rather than full conversations. Collection and remote sending have separate settings:

```yaml
telemetry:
  shared_metrics:
    enabled: true
    send: false
```

This example enables local collection, not remote sending by the shared-metrics sender. `send: false` does not disable Relay exporters, including automatic export from inherited OTLP settings. Configuring a Relay exporter does not enable shared metrics. When both are enabled, shared-metrics records can also reach Relay's configured subscribers. Dedicated skill lifecycle and load counters belong to this optional mode; normal skill-tool arguments and results remain ordinary tool data.

## Migrate an older setup

The former Hermes `observability/nemo_relay` plugin and the old `HERMES_NEMO_RELAY_ATOF_*` / `HERMES_NEMO_RELAY_ATIF_*` exporter variables are no longer the configuration path.

`hermes update` migrates supported legacy exporter variables in Hermes's `.env`. To run the migration yourself:

```bash
hermes migrate relay
```

Use `hermes migrate relay --all-profiles` for all profile homes. Unless an explicit override is already set, the migration validates the generated configuration, writes `<hermes home>/relay-plugins.toml`, sets the override in that profile's `.env`, and comments out the old variables. The generated file replaces the user file; system configuration still applies. Restart Hermes and check the output.

Remove the former Hermes plugin entry from `plugins.enabled` as well; the exporter migration does not do that for you. For manual migration, move exporter settings into `plugins.toml`. The [built-in plugin migration note](../user-guide/features/built-in-plugins.md#nemo-relay-native-integration-migration-note) includes a complete migrated configuration.

## Troubleshooting

| Symptom | Check |
|---|---|
| No trace file | Run `hermes doctor`, check Relay initialization warnings, and confirm the output directory is writable. If using the override, check that its file exists. |
| File is in an unexpected directory | Relative output paths use Hermes's working directory. Use an absolute path for services. |
| Duplicate records | A start/end pair sharing a UUID is normal. For duplicate starts or ends, check old append-mode output and whether multiple exporters target the same destination. |
| Hermes works but a Relay plugin does not | Initialization can fail without stopping Hermes. Check validation, plugin compatibility, integrity, trust policy, and startup warnings. |
| Configuration changes have no effect | Restart the owning Hermes process. Check system configuration and the first profile's process-wide configuration. |
| Turns appear but the parent session does not | The session scope may still be open. End it normally or configure segmentation. |
| A model or tool call is missing | Check overlapping turns, nested managed callbacks, work outside an agent turn, and the selected provider's codec support. |
| Unexpected remote export | Inspect user/system configuration and inherited `OTEL_EXPORTER_OTLP_*ENDPOINT` variables. Explicit Relay endpoints and automatic export can coexist. |

For more detail, enable [Relay operational logging](https://docs.nvidia.com/nemo/relay/v0.11.1/reference/operational-logging) before starting a fresh Hermes process. Relay's operational logs are separate from its exported OpenTelemetry logs.

## Related documentation

- [Relay observability](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/about)
- [Relay plugin development](https://docs.nvidia.com/nemo/relay/v0.11.1/build-plugins/about)
- [Hermes environment variables](../reference/environment-variables.md#nemo-relay)
- [Hermes sessions](../user-guide/sessions.md)
- [Hermes delegation](../user-guide/features/delegation.md)
