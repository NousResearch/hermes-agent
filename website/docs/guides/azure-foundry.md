---
sidebar_position: 15
title: "Microsoft Foundry"
description: "Use Hermes Agent with Microsoft Foundry — OpenAI-style and Anthropic-style endpoints, auto-detection of transport and deployed models"
---

# Microsoft Foundry

Hermes Agent's `azure-foundry` provider supports Microsoft Foundry (formerly Azure AI Foundry) and Azure OpenAI. A single Foundry resource can host models with two different wire formats:

- **OpenAI-style** — `POST /v1/chat/completions` on endpoints like `https://<resource>.openai.azure.com/openai/v1`. Used for GPT-4.x, GPT-5.x, Llama, Mistral, and most open-weight models.
- **Anthropic-style** — `POST /v1/messages` on endpoints like `https://<resource>.services.ai.azure.com/anthropic`. Used when Microsoft Foundry serves Claude models via the Anthropic Messages API format.

The setup wizard probes your endpoint and auto-detects which transport it uses, which deployments are available, and each model's context length.

## Prerequisites

- A Microsoft Foundry or Azure OpenAI resource with at least one deployment
- The deployment's endpoint URL
- **Either** an API key (from the Azure Portal under "Keys and Endpoint") **or** the **Azure AI User** RBAC role on the Foundry resource if you plan to use Microsoft Entra ID (the keyless path Microsoft recommends). Some tenants may show the role as **Foundry User** during Microsoft's rename rollout.

## Quick Start

```bash
hermes model
# → Select "Azure Foundry"
# → Enter your endpoint URL
# → Choose Authentication:
#     1. API key
#     2. Microsoft Entra ID  (managed identity / workload identity / az login)
# → (Entra) Hermes probes DefaultAzureCredential; on success it never asks for a key
# → (API key) Enter your API key
# Hermes probes the endpoint and auto-detects transport + models
# → Pick a model from the list (or type a deployment name manually)
```

The wizard will:

1. **Sniff the URL path** — URLs ending in `/anthropic` are recognised as Microsoft Foundry Claude routes.
2. **Probe `GET <base>/models`** — if the endpoint returns an OpenAI-shaped model list, Hermes switches to `chat_completions` and prefills a picker with the returned deployment IDs.
3. **Probe Anthropic Messages shape** — fallback for endpoints that do not expose `/models` but do accept the Anthropic Messages format.
4. **Fall back to manual entry** — private/gated endpoints that reject every probe still work; you pick the API mode and type a deployment name by hand.

Context length for the chosen model is resolved via Hermes' standard metadata chain (`models.dev`, provider metadata, and hardcoded family fallbacks) and stored in `config.yaml` so the model can size its own context window correctly.

## Microsoft Entra ID (keyless, RBAC) — recommended

Microsoft recommends [keyless authentication with Microsoft Entra ID](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id) for production Foundry workloads. Hermes supports Entra ID for **both** API surfaces:

- **OpenAI-style** (`api_mode: chat_completions` / `codex_responses`) — GPT-4/5, Llama, Mistral, DeepSeek, etc.
- **Anthropic-style** (`api_mode: anthropic_messages`) — Claude models on Microsoft Foundry.

Foundry's RBAC is per-resource (`Azure AI User` grants both surfaces; some tenants may display `Foundry User`) and Microsoft documents the same inference scope (`https://ai.azure.com/.default`) for both. Under the hood:

- OpenAI-style uses the OpenAI Python SDK's native callable `api_key=` contract — the SDK mints a fresh JWT per request automatically.
- Anthropic-style uses an `httpx.Client` with a request event hook installed by `agent.azure_identity_adapter.build_bearer_http_client`, because the Anthropic SDK does not accept callable `auth_token` natively. The hook rewrites `Authorization: Bearer <fresh-jwt>` per outbound request. Same Microsoft RBAC, same Foundry scope — the SDK contract is the only difference.

### Why use Entra ID?

- No long-lived API keys to rotate or revoke.
- RBAC-driven access — grant or remove `Azure AI User` on the Foundry resource, no config rewrite needed.
- Access and audit logs are segmented by assignee instead of all callers sharing one static key.
- Single auth surface for Azure VMs, AKS pods, App Service, Functions, Container Apps, and Foundry Agent Service via managed identity.
- Workload identity and service-principal flows for CI/CD pipelines.

### One-time setup (Azure side)

1. In the Azure Portal, open your Foundry resource → **Access control (IAM)** → **Add → Add role assignment**.
2. Pick the **Azure AI User** role (or **Foundry User** if your tenant has the renamed role).
3. Assign it to:
   - **Your user account** for local development with `az login`.
   - **A managed identity or workload identity** for Azure-hosted compute (recommended for production).
   - **A Foundry Agent Service hosted agent's agent identity** when Hermes runs inside a hosted agent.
   - **A service principal** for CI/CD pipelines when workload identity is not available.
4. Wait ~5 minutes for the role to propagate.

Azure CLI equivalent:

```bash
az role assignment create \
  --assignee <principal-or-agent-identity-client-id> \
  --role "Azure AI User" \
  --scope <foundry-resource-id>
```

### One-time setup (Hermes side)

```bash
hermes model
# → Select "Azure Foundry"
# → Enter your endpoint URL
# → Authentication: 2 (Microsoft Entra ID)
# → (optional) user-assigned managed identity client ID
# → (optional) Azure tenant ID
# → Hermes probes DefaultAzureCredential() and reports which inner
#    credential succeeded (e.g. AzureCliCredential, ManagedIdentityCredential)
```

The wizard runs a bounded preflight probe (10 s timeout). On failure it offers to "save anyway, validate later" — useful when configuring on a machine that doesn't yet have credentials but will at runtime (e.g. preparing config for a managed-identity deployment).

`azure-identity` is installed automatically on first use via Hermes' lazy-install path. To pre-install:

```bash
pip install azure-identity
```

### Configuration written to `config.yaml`

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  auth_mode: entra_id
  default: gpt-4o
  context_length: 128000
  entra:
    scope: https://ai.azure.com/.default        # only when overriding the default
```

Hermes only manages one Entra-specific knob in `config.yaml`:

- **`scope`** — the OAuth resource scope. Defaults to Microsoft's documented inference scope (`https://ai.azure.com/.default`). Override only if your resource was provisioned against a non-standard audience.

Everything else (tenant, service principal secret, federated token file, sovereign cloud authority, broker preferences) is read by `azure-identity` directly from the standard `AZURE_*` environment variables — see the [credential resolution order](#credential-resolution-order) below. Set those in `~/.hermes/.env` or your deployment environment, exactly as Microsoft's SDK reference describes.

No secrets land in `~/.hermes/.env` for Entra mode — `azure-identity` caches tokens in-process (and where available, in your OS keychain / `~/.IdentityService`).

### Credential resolution order

`azure-identity`'s `DefaultAzureCredential` walks this chain on each token request, stopping at the first credential that returns a token:

1. **Environment credential** — `AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET` (or `AZURE_CLIENT_CERTIFICATE_PATH` / `AZURE_FEDERATED_TOKEN_FILE`).
2. **Workload Identity** — `AZURE_FEDERATED_TOKEN_FILE` (AKS federated tokens / OIDC).
3. **Managed Identity** — IMDS endpoint (`169.254.169.254`) for virtual machines; `IDENTITY_ENDPOINT` for App Service / Functions / Container Apps. Foundry Agent Service hosted agents use the hosted agent's agent identity.
4. **Visual Studio Code** — Azure account extension.
5. **Azure CLI** — `az login` session.
6. **Azure Developer CLI** — `azd auth login`.
7. **Azure PowerShell** — `Connect-AzAccount`.
8. **Broker** (Windows / WSL only) — Web Account Manager.

Interactive browser credential is excluded by default for unattended Hermes runs; use Azure CLI, Azure Developer CLI, managed identity, workload identity, or service principal credentials instead.

### Deployment patterns

**Local development:**
```bash
az login
hermes model   # pick Azure Foundry → Entra ID
hermes         # uses your az login token
```

**Azure VM / Functions / App Service / Container Apps (system-assigned managed identity):**
1. Enable system-assigned identity on the compute resource.
2. Grant the identity `Azure AI User` (or `Foundry User`) on the Foundry resource.
3. Set `model.auth_mode: entra_id` in config.yaml — no env vars needed.

**Azure VM / Functions / App Service / Container Apps (user-assigned managed identity):**
- Set `AZURE_CLIENT_ID` to the user-assigned identity's client ID so `DefaultAzureCredential` picks the right one.

**Foundry Agent Service hosted agent:**
- Create the hosted agent and grant that agent's identity `Azure AI User` (or `Foundry User`) on the Foundry resource. Hermes uses `ManagedIdentityCredential` from inside the hosted agent; role assignment belongs on the agent identity, not just the parent project or your user.

**AKS Workload Identity (replaces AAD Pod Identity):**
- Annotate the pod's service account with the workload identity client ID.
- The pod's federated token file is auto-detected via `AZURE_FEDERATED_TOKEN_FILE`.
- `model.auth_mode: entra_id` works without further config changes.

**Service principal in CI:**
- Set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` in the runner env.

#### Sovereign clouds (Government, China)

Export `AZURE_AUTHORITY_HOST` (e.g. `https://login.microsoftonline.us` for Azure Government, `https://login.partner.microsoftonline.cn` for Azure China). `azure-identity` reads it directly.

### Health checks

`hermes doctor` runs a 10 s probe against `DefaultAzureCredential` when `model.auth_mode: entra_id`, reporting which inner credential won (env vars present, managed identity endpoint reachable, etc.).

`hermes auth` shows a structured status block:

```
azure-foundry (Microsoft Entra ID):
  Endpoint: https://my-resource.openai.azure.com/openai/v1
  Scope: https://ai.azure.com/.default
  Status: configured; live token probe is skipped here
```

### Limitations

- **Anthropic-style endpoints use an httpx event hook.** The Anthropic Python SDK does not accept a callable `auth_token` natively (≤ 0.86.0). Hermes installs a request event hook on a custom `httpx.Client` that mints a fresh JWT per outbound request and rewrites `Authorization: Bearer <jwt>`. This is functionally equivalent to the OpenAI SDK's native `Callable[[], str]` contract but adds one indirection layer. If the Anthropic SDK adds first-class callable-auth support in a future release, Hermes will switch to it transparently.
- **Batch jobs and `multiprocessing.Pool`.** The Entra token provider is a closure that cannot be pickled across process boundaries. `batch_runner.py` automatically drops the callable from the worker config and lets each worker process rebuild its own provider from `config.yaml` — no user action required, but each worker pays one chain walk at startup.
- **No bearer JWT persistence in `auth.json`.** Hermes does not duplicate `azure-identity`'s internal token cache; cold starts walk the credential chain on first inference.

## Configuration (written to `config.yaml`)

After running the wizard you'll see something like this:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # or "anthropic_messages"
  default: gpt-5.4-mini              # your deployment / model name
  context_length: 400000             # auto-detected
```

And in `~/.hermes/.env`:

```
AZURE_FOUNDRY_API_KEY=<your-azure-key>
```

## OpenAI-style endpoints (GPT, Llama, etc.)

Azure OpenAI's v1 GA endpoint accepts the standard `openai` Python client with minimal changes:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  default: gpt-5.4
```

Important behaviour:

- **GPT-5.x, codex, and o-series auto-route to the Responses API.** Microsoft Foundry deploys GPT-5 / codex / o1 / o3 / o4 models as Responses-API-only — calling `/chat/completions` against them returns `400 "The requested operation is unsupported."`. Hermes detects these model families by name and upgrades `api_mode` to `codex_responses` transparently, even when `config.yaml` still reads `api_mode: chat_completions`. GPT-4, GPT-4o, Llama, Mistral, and other deployments stay on `/chat/completions`.
- **`max_completion_tokens` is used automatically.** Azure OpenAI (like direct OpenAI) requires `max_completion_tokens` for gpt-4o, o-series, and gpt-5.x models. Hermes sends the right parameter based on the endpoint.
- **Pre-v1 endpoints that require `api-version`.** If you have a legacy base URL like `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview`, Hermes extracts the query string and forwards it via `default_query` on every request (the OpenAI SDK otherwise drops it when joining paths).

## Anthropic-style endpoints (Claude via Microsoft Foundry)

For Claude deployments, use the Anthropic-style route:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_mode: anthropic_messages
  default: claude-sonnet-4-6
```

Important behaviour:

- **`/v1` is stripped from the base URL.** The Anthropic SDK appends `/v1/messages` to every request URL — Hermes removes any trailing `/v1` before handing the URL to the SDK to avoid double-`/v1` paths.
- **`api-version` is sent via `default_query`, not appended to the URL.** Azure Anthropic requires an `api-version` query string. Baking it into the base URL produces malformed paths like `/anthropic?api-version=.../v1/messages` and returns 404. Hermes passes `api-version=2025-04-15` via the Anthropic SDK's `default_query` instead.
- **Bearer auth is used instead of `x-api-key`.** Azure's Anthropic-compatible route requires `Authorization: Bearer <key>` rather than Anthropic's native `x-api-key` header. Hermes detects `azure.com` in the base URL and routes the API key through the SDK's `auth_token` field so the right header reaches the upstream.
- **1M context window beta header is kept.** Azure still gates the 1M-token Claude context (Opus 4.6/4.7, Sonnet 4.6) behind the `anthropic-beta: context-1m-2025-08-07` header. Hermes keeps that beta header on Azure paths (it's stripped from native Anthropic OAuth requests because some subscriptions reject it, but Azure requires it).
- **OAuth token refresh is disabled.** Azure deployments use static API keys. The `~/.claude/.credentials.json` OAuth token refresh loop that applies to Anthropic Console is explicitly skipped for Azure endpoints to prevent the Claude Code OAuth token from overwriting your Azure key mid-session.

## Alternative: `provider: anthropic` + Azure base URL

If you already have `provider: anthropic` configured and just want to point it at Microsoft Foundry for Claude, you can skip the `azure-foundry` provider entirely:

```yaml
model:
  provider: anthropic
  base_url: https://my-resource.services.ai.azure.com/anthropic
  key_env: AZURE_ANTHROPIC_KEY
  default: claude-sonnet-4-6
```

With `AZURE_ANTHROPIC_KEY` set in `~/.hermes/.env`. Hermes detects `azure.com` in the base URL and short-circuits around the Claude Code OAuth token chain so the Azure key is used directly with `x-api-key` auth.

`key_env` is the canonical snake_case field name; `api_key_env` (and the camelCase `keyEnv` / `apiKeyEnv`) are accepted as aliases. If both `key_env` and `AZURE_ANTHROPIC_KEY`/`ANTHROPIC_API_KEY` are set, the `key_env`-named env var wins.

## Model discovery

Azure does **not** expose a pure-API-key endpoint to list your *deployed* model deployments. Deployment enumeration requires Azure Resource Manager authentication (`az cognitiveservices account deployment list`) with an Azure AD principal, not the inference API key.

What Hermes can do:

- Azure OpenAI v1 endpoints (`<resource>.openai.azure.com/openai/v1`) expose `GET /models` with the resource's **available** model catalog. Hermes uses this list to prefill the model picker.
- Microsoft Foundry `/anthropic` routes: detected via URL path, model name entered manually.
- Private / firewalled endpoints: manual entry with a friendly "couldn't probe" message.

You can always type a deployment name directly — Hermes does not validate against the returned list.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Primary API key for Microsoft Foundry / Azure OpenAI (api_key mode) |
| `AZURE_FOUNDRY_BASE_URL` | Endpoint URL (set via `hermes model`; env var is used as a fallback) |
| `AZURE_ANTHROPIC_KEY` | Used by `provider: anthropic` + Azure base URL (alternative to `ANTHROPIC_API_KEY`) |
| `AZURE_TENANT_ID` | Entra ID tenant for service-principal flows |
| `AZURE_CLIENT_ID` | Entra ID client ID (service principal, workload identity, or user-assigned managed identity) |
| `AZURE_CLIENT_SECRET` | Service principal secret |
| `AZURE_CLIENT_CERTIFICATE_PATH` | Service principal cert (alternative to secret) |
| `AZURE_FEDERATED_TOKEN_FILE` | Workload Identity federated token path (AKS) |
| `AZURE_AUTHORITY_HOST` | Sovereign cloud authority host override |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | Managed Identity endpoint for App Service, Functions, and Container Apps; VMs usually use IMDS instead |

The Azure SDK reads the `AZURE_*` env vars directly. Hermes never inspects them other than to report which sources are present in `hermes doctor` output.

## Troubleshooting

**401 Unauthorized on gpt-5.x deployments.**
Azure serves gpt-5.x on `/chat/completions`, not `/responses`. Hermes handles this automatically when the URL contains `openai.azure.com`, but if you see a 401 with an `Invalid API key` body, check that `api_mode` in your `config.yaml` is `chat_completions`.

**404 on `/v1/messages?api-version=.../v1/messages`.**
This is the malformed-URL bug from pre-fix Azure Anthropic setups. Upgrade Hermes — the `api-version` parameter is now passed via `default_query` rather than baked into the base URL, so the SDK can't corrupt it during URL joining.

**Wizard says "Auto-detection incomplete."**
The endpoint rejected both the `/models` probe and the Anthropic Messages probe. This is normal for private endpoints behind a firewall or with an IP allow-list. Fall back to manual API mode selection and type your deployment name — everything still works, Hermes just can't prefill the picker.

**Wrong transport picked.**
Run `hermes model` again and the wizard will re-probe. If the probe still picks the wrong mode, you can edit `config.yaml` directly:

```yaml
model:
  provider: azure-foundry
  api_mode: anthropic_messages   # or chat_completions
```

**Entra ID: "credential chain exhausted" or 401 Unauthorized after switching to `auth_mode: entra_id`.**
- Run `az login` to refresh your developer session (the cached token may have expired).
- Verify the `Azure AI User` (or `Foundry User`) role assignment took effect: `az role assignment list --assignee <user-or-identity-id>` should list it on your Foundry resource. Role propagation can take up to 5 minutes.
- For user-assigned managed identities, double-check `AZURE_CLIENT_ID` matches the identity attached to the compute resource.
- Run `hermes doctor` — the Azure Entra probe reports whether token acquisition succeeded and includes a remediation hint.

**Entra ID: wizard preflight hangs or times out.**
The 10 s preflight is a soft check. Choose "Save anyway and validate later" and run `hermes doctor` after deploying to the target environment. Common causes include an unreachable token service or stale local login state — prefer workload identity in CI, set `AZURE_TENANT_ID`+`AZURE_CLIENT_ID`+`AZURE_CLIENT_SECRET` when using a service principal, or run `az login` for local development.

**401 on Anthropic-style endpoint with Entra ID.**
Verify the same `Azure AI User` (or `Foundry User`) role is assigned on the Foundry resource (it covers both `/openai/v1` and `/anthropic` paths). If the OpenAI-style probe works during the wizard but `claude-*` requests fail at runtime, the most common cause is a stale `model.entra.scope` left over from an earlier wizard run — delete the `entra.scope` line from `config.yaml` so the runtime falls back to the default `https://ai.azure.com/.default` scope.

## Related

- [Environment variables](/docs/reference/environment-variables)
- [Configuration](/docs/user-guide/configuration)
- [AWS Bedrock](/docs/guides/aws-bedrock) — the other major cloud provider integration
- [Microsoft: Configure Entra ID for Foundry](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id) — upstream documentation for the keyless path
