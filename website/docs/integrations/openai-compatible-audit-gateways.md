---
title: "OpenAI-Compatible Audit Gateways"
sidebar_label: "Audit Gateways"
sidebar_position: 3
---

# OpenAI-Compatible Audit Gateways

Hermes can route model traffic through an OpenAI-compatible gateway when an
operator needs provider-neutral audit evidence, centralized routing policy, or
local runtime governance outside Hermes itself. Hermes remains responsible for
the agent loop; the gateway supplies an independent record of provider routing.

This pattern does not replace Hermes planning, tools, skills, memory, approval
UX, or Mixture-of-Agents behavior. A gateway receipt also does not prove that a
model response is correct.

## Gateway requirements

The gateway must expose an OpenAI-compatible `/v1` endpoint and preserve the
request and response shapes expected by the selected Hermes provider. For
security or compliance use, prefer a gateway that can:

- write append-only, content-minimized request receipts;
- verify receipt integrity with a hash chain or comparable mechanism;
- avoid retaining raw prompts and responses by default;
- record hashed actor, team, organization, or host-agent identifiers; and
- export evidence in a documented, reviewable format.

ModelRouter is one example, but the configuration below is vendor-neutral.

## Configure the primary model

Hermes supports request-attribution headers through `model.default_headers`.
Keep credentials in environment variables and reference them from
`config.yaml`; do not put API keys in identity headers.

```yaml title="~/.hermes/config.yaml"
model:
  provider: custom
  default: your-model-id
  base_url: https://audit-gateway.example.com/v1
  api_key: ${AUDIT_GATEWAY_API_KEY}
  default_headers:
    X-Audit-Actor-ID: service-account-id
    X-Audit-Team-ID: platform
    X-Audit-Org-ID: example-org
    X-Audit-Host-Agent-ID: hermes-prod
```

Use the header names defined by your gateway. For example, ModelRouter may use
`X-ModelRouter-Actor-ID`, `X-ModelRouter-Team-ID`,
`X-ModelRouter-Org-ID`, and `X-ModelRouter-Host-Agent-ID`.

Header values should be stable, non-secret identifiers. Configure the gateway
to hash them before durable storage. Never send tokens, raw prompts, private
content, or other secrets as attribution headers.

## Scope headers to a named endpoint

When only one named provider needs gateway headers, use that provider's
`extra_headers` instead of global `model.default_headers`:

```yaml title="~/.hermes/config.yaml"
custom_providers:
  - name: audited-endpoint
    base_url: https://audit-gateway.example.com/v1
    key_env: AUDIT_GATEWAY_API_KEY
    api_mode: chat_completions
    extra_headers:
      X-Audit-Actor-ID: service-account-id
      X-Audit-Team-ID: platform
```

`extra_headers` is scoped to the matching endpoint and overrides a same-named
SDK or provider default. Header values are treated as sensitive configuration
and should not be logged.

## Verify the route

1. Start the audit gateway and confirm its `/v1/models` endpoint is reachable.
2. Configure Hermes with one of the supported header mechanisms above.
3. Send a bounded test request through Hermes.
4. Confirm the gateway recorded the expected actor and route without retaining
   raw content contrary to your policy.
5. Export and independently verify the content-minimized receipt.

If any step fails, restore the previous provider configuration and investigate
the gateway before sending production traffic. A gateway can observe or route
requests, but it must not be treated as Hermes tool authority or approval.
