# qwen-ops Routed Client

`qwen-ops` is a gateway-only Hermes client for local, low-cost, advisory
operations work. It is not a separate router, authority model, ledger, or
runner.

Use `/qwen-ops`, `/qwen`, `/qwenops`, or `/qwen_ops` from a messaging gateway
chat that is already allowed to talk to Hermes. Telegram's command menu shows
the canonical command as `/qwen_ops` because Telegram command names cannot
contain hyphens.

## Scope

- Hermes remains the routing and control layer.
- Existing chat allowlists, group/DM behavior, session handling, and
  slash-command access policy still apply.
- qwen-ops uses the shared `/model qwen3-coder --provider qwen-oauth` path for
  session model selection.
- qwen-ops prompts are sent through the normal Hermes turn path, so existing
  token and budget accounting applies.
- qwen-ops has no merge, approve, comment, label, write, or runner authority.

## Commands

`/qwen-ops` or `/qwen on`

Switches the current gateway session through the shared `/model` handler.

`/qwen-ops <prompt>`

Switches the session through the shared `/model` handler, then sends the prompt
through normal Hermes message handling with an advisory/local-ops scope prefix.

`/qwen-ops escalate [codex|gpt-5.5|opus] <blocker>`

Renders a shared escalation packet only. It does not queue, invoke, or execute a
frontier-model request. Any execution must come from existing shared Hermes
policy that explicitly allows it.

## Non-Claims

qwen-ops does not claim production readiness, launch readiness, live customer
data readiness, live OAuth/banking/account-connection readiness,
money-movement authority, tax/accounting/finality authority, protected
approval, GitHub green, or merge readiness.
