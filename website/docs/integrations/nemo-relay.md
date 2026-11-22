---
sidebar_position: 5
title: "NeMo Relay"
description: "Record Hermes model and tool calls, export traces, and configure NeMo Relay plugins"
---

# NeMo Relay

[NeMo Relay](https://docs.nvidia.com/nemo/relay/v0.11.1/about-nemo-relay/overview) lets you follow a Hermes conversation through its model requests, tool calls, and delegated agents. You can see which commands ran, what they returned, which model requests were retried, and where the run spent its time.

Hermes supports Relay directly. Start Hermes normally; you do not need a separate Relay daemon, the `nemo-relay run` command, or a Hermes observability plugin.

Start with a local trace file, then choose whether to keep files, send traces to a dashboard, or save completed trajectories for analysis. Relay plugins can also redact exported data and add cost and resource measurements.

Beyond observability, Hermes also runs model and tool calls through Relay's [middleware](https://docs.nvidia.com/nemo/relay/v0.11.1/about-nemo-relay/concepts/middleware), including asynchronous middleware. Configured plugins can inspect or modify requests and results, apply policy checks, and serve cached responses. Observability redaction is separate: it changes exported copies without changing the live model or tool payload.

:::note Different from Hermes Relay
NeMo Relay handles agent observability and middleware. [Hermes Relay](../user-guide/messaging/relay.md) connects a Hermes gateway to messaging platforms.
:::

## What gets recorded

Relay records work that passes through Hermes's `AIAgent` conversation path. That includes conversations from the CLI, TUI, Desktop, dashboard, ACP, messaging gateway, API server, scheduled and batch jobs, and the Python library when they use that path.

| Activity | What you can inspect |
|---|---|
| Sessions and turns | Start/end times, parent-child relationships, turn outcome, and the working directory when known. |
| Model requests and responses | Provider and model, retained messages and request fields, model output, timing, and outcome. |
| Retries and fallbacks | A separate provider attempt under the same logical model call. This helps distinguish one user-facing call from several attempts to complete it. |
| Token usage and cost | Usage reported by the provider. The optional pricing plugin adds estimates when it has matching model prices and usable usage data. |
| Tool calls | Tool name, call ID, arguments, returned result, timing, and managed-call outcome. For a terminal command, inspect the returned output and exit status as well: an error returned as data need not mark the span as failed. |
| Streaming | Stream lifecycle, chunk receipt metadata, available usage and finish information, and a reconstructed final response when the provider format supports it. |
| Delegated agents | Child Hermes sessions under the turn that started them when Hermes has that relationship. |
| Skills, memory, and MCP | Calls made through Hermes tools, with their arguments and results. These are ordinary tool records, not a separate trace of everything inside those systems. |

The available fields depend on the provider, codec, plugins, and export format. OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages use [provider-aware codecs](https://docs.nvidia.com/nemo/relay/v0.11.1/integrate-into-frameworks/provider-codecs). Other routes use a generic JSON projection and may lack normalized model fields.

### Reading one run

Suppose you ask Hermes to investigate a failing test. In the trace, look for:

1. **`hermes.session`** for the conversation and **`hermes.turn`** for your request.
2. **`hermes.logical_llm_call`** for a model call, with a child for each provider attempt, such as `openai.chat_completions`.
3. **`terminal`** or other tool scopes for the commands Hermes ran, their arguments, results, and duration.
4. Further model calls and tools under the turn as Hermes works toward its answer. Delegated Hermes agents can have their own child sessions.

This lets you distinguish a slow provider request from a slow tool, or a successful tool run from a model request that needed several attempts.

### What is not automatically recorded

Relay does not capture every part of Hermes or your machine. Slash commands, message delivery, voice transcription, and scheduler bookkeeping are outside this conversation path. A call to an external MCP server records the Hermes-level request and result, not the server's internal tools, subprocesses, or network activity.

Streaming chunk marks contain receipt metadata, not every raw text fragment. Resource measurements are optional: the resource-metrics plugin adds host/process readings, not automatic per-tool or per-sandbox accounting. See [available plugins](#available-relay-plugins).

## Choose an output

You can export the same run in more than one format from one observability component. The formats serve different purposes and do not preserve every detail in the same way.

| Output | What it contains | Choose it for |
|---|---|---|
| [ATOF](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/atof) | Individual scope start/end events and marks, written incrementally as JSONL. | Inspecting recorded fields, debugging a run, or sending events to a custom collector. |
| [ATIF](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/atif) | A completed trajectory in JSON, organized into conversation and tool steps. It is written when the top-level agent scope closes. | Offline analysis and preparing evaluation or training datasets. |
| [OpenTelemetry / OTLP](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/opentelemetry) | Traces sent to an OTLP collector, with separate configuration for logs and metrics. Completed spans can appear before the whole session ends. | Viewing timing, failures, and nested calls in Phoenix or another observability backend. |

**Start with ATOF** if you want to check exactly what Hermes is recording. Use **OpenTelemetry** for a trace dashboard and **ATIF** for a completed trajectory file. ATIF is not a recording that can automatically replay every external tool or model call.

The [Quick Start](#quick-start-write-a-local-trace) writes ATOF. The examples below add [Phoenix](#arize-phoenix), [other OpenTelemetry collectors](#other-opentelemetry-collectors), or [ATIF](#atif-trajectories).

## Quick Start: write a local trace

### Before you start

Complete the normal Hermes setup and confirm that you can run a conversation with your model provider.

Hermes uses `nemo-relay>=0.11,<0.12`. Its dependency configuration installs Relay on **Python 3.14** on these platforms:

| Operating system | Architecture |
|---|---|
| macOS | Apple silicon |
| Linux, including glibc and musl distributions | x86_64 or ARM64 |
| Windows | x86_64 or ARM64 |

Intel Macs and Android/Termux are not supported by this integration. A missing Relay binding does not prevent Hermes from running, but Relay recording and middleware are unavailable. Use the normal Hermes install or update process rather than changing packages in its managed environment. See Relay's [support matrix](https://docs.nvidia.com/nemo/relay/v0.11.1/reference/support-matrix).

Run `hermes doctor` to check whether Relay configuration already applies to the machine. Hermes reads Relay's user and system files; the optional override below replaces only the user file. An enabled observability component can also pick up inherited OTLP endpoint settings. Review [configuration](#how-configuration-is-applied) before assuming a test will export only to a local file.

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

Relay offers three OpenTelemetry trace projections. A projection chooses how Relay's events become spans and attributes; it is not another file format.

| Projection | Use it for |
|---|---|
| `openinference` | OpenInference-compatible agent, model, and tool spans, including the Phoenix example above. |
| `gen_ai` | Standard OpenTelemetry GenAI fields for collectors that expect those conventions. It omits Relay-specific attributes and non-metric marks from traces. |
| `full` | Relay's scope hierarchy, metadata, and non-metric marks when you need its detailed event view in an OTLP backend. |

Choose a projection supported by your collector. Different projections must use separate destinations; Relay rejects conflicting projection types at the same endpoint. Configure log and metric pipelines separately if you need those signals.

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

[ATIF](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/atif) writes a trajectory when its top-level agent scope closes. A long-running gateway session may therefore show turn spans before it has a completed session trajectory; see [continuous gateway sessions](#continuous-gateway-sessions).

The filename uses a Relay trajectory identifier, not necessarily the ID shown by `hermes sessions`. If you need to join ATIF files to traces, use Relay's [correlation reference](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/about#correlating-trajectories-and-traces).

## Available Relay plugins

Relay plugins are separate from [Hermes plugins](../user-guide/features/plugins.md). Configure them in Relay's `plugins.toml`, not with `hermes plugins enable`. The following components are included in the Relay Python package used by Hermes:

| Plugin | What it adds | Does it change live calls? |
|---|---|---|
| [`observability`](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/about) | ATOF and ATIF files, or OpenTelemetry export. | No; it exports recorded activity. |
| [`pii_redaction`](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/pii-redaction/about) | Removes detected sensitive values from observability copies before export. | No; model and tool payloads stay unchanged. |
| [`pricing`](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/model-pricing) | Estimates model-call costs from configured prices and reported usage. | No; an estimate requires matching prices and usable usage data. |
| [`resource_metrics`](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/resource-metrics) | CPU, memory, process, disk, network, and supported GPU measurements through polling or snapshots. Polling is off until enabled. | No call rewriting. Some readings are system/device-wide; availability depends on the platform and drivers. |
| [`adaptive`](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/adaptive/about) | Prompt-cache planning, response caching, and other adaptive features. | It can. A cache hit can skip the real provider or tool call. |
| [`nemo_guardrails`](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/nemo-guardrails/about) | Legacy Guardrails policy checks. **Deprecated; do not choose it for a new setup.** | Yes; policy checks can affect live execution. |

[NeMo Switchyard](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/switchyard) is a separately installed NVIDIA plugin for model routing. It is not bundled with Hermes. Use a release compatible with your Relay version and follow its installation instructions.

For an observability setup, start with `observability`, add `pii_redaction` if needed, and then consider pricing or resource measurements. Execution-changing plugins need testing with your actual provider and tools. Observability redaction does not sanitize response-cache contents.

## Configure Relay plugins

The examples below extend the Quick Start's `relay-plugins.toml`. For the other components, follow the configuration links in the catalogue. Relay's [configuration reference](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/plugin-configuration-files) describes the file structure.

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
