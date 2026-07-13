---
title: X (Twitter) Search
description: Search X (Twitter) posts and threads from within the agent using xAI's built-in x_search Responses tool — works with either a SuperGrok OAuth login or an XAI_API_KEY.
sidebar_label: X (Twitter) Search
sidebar_position: 7
---

# X (Twitter) Search

The `x_search` tool lets the agent search X (Twitter) posts, profiles, and threads directly. It's backed by xAI's built-in `x_search` tool on the Responses API at `https://api.x.ai/v1/responses` — Grok itself runs the search server-side and returns synthesized results with citations to the originating posts.

**Use this instead of `web_search`** when you specifically want current discussion, reactions, or claims **on X**. For general web pages, keep using `web_search` / `web_extract`.

## Authentication

`x_search` registers when **either** xAI credential path is available:

| Credential | Source | Setup |
|------------|--------|-------|
| **SuperGrok OAuth** (preferred) | Browser login at `accounts.x.ai`, refreshed automatically | `hermes auth add xai-oauth` — see [xAI Grok OAuth (SuperGrok Subscription)](../../guides/xai-grok-oauth.md) |
| **`XAI_API_KEY`** | Paid xAI API key | Set in `~/.hermes/.env` |

Both hit the same endpoint with the same payload — the only difference is the bearer token. **When both are configured, SuperGrok OAuth wins** so x_search runs against your subscription quota instead of paid API spend.

The tool's `check_fn` runs the xAI credential resolver every time the model's tool list is rebuilt. A `True` return means the bearer is fetchable AND non-empty AND (if it had expired) successfully refreshed. Revoked tokens with a failed refresh hide the tool from the schema; the model simply can't see it.

## Enabling the tool

Off by default. Enable in `hermes tools`:

```bash
hermes tools
# → 🐦 X (Twitter) Search   (press space to toggle on)
```

The picker offers two credential choices:

1. **xAI Grok OAuth (SuperGrok Subscription)** — opens the browser to `accounts.x.ai` if you're not already logged in
2. **xAI API key** — prompts for `XAI_API_KEY`

Either choice satisfies the gating. You can pick whichever credentials you already have; the tool works identically with both. If both end up configured, OAuth is preferred at call time.

## Configuration

```yaml
# ~/.hermes/config.yaml
x_search:
  # xAI model used for the Responses call.
  # grok-4.20-reasoning is the recommended default; any Grok model
  # with x_search tool access works.
  model: grok-4.20-reasoning

  # Request timeout in seconds. x_search can take 60–120s for
  # complex queries — the default is generous. Minimum: 30.
  timeout_seconds: 180

  # Number of automatic retries on 5xx / ReadTimeout / ConnectionError.
  # Each retry backs off (1.5x attempt seconds, capped at 5s).
  retries: 2
```

## Tool parameters

The agent calls `x_search` with these arguments:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string (required) | What to look up on X. |
| `allowed_x_handles` | string array | Optional list of handles to include **exclusively** (max 10). Leading `@` is stripped. |
| `excluded_x_handles` | string array | Optional list of handles to exclude (max 10). Mutually exclusive with `allowed_x_handles`. |
| `from_date` | string | Optional `YYYY-MM-DD` start date. |
| `to_date` | string | Optional `YYYY-MM-DD` end date. |
| `enable_image_understanding` | boolean | Ask xAI to analyze images attached to matching posts. |
| `enable_video_understanding` | boolean | Ask xAI to analyze videos attached to matching posts. |

The tool returns JSON with:

- `answer` — synthesized text response from Grok
- `citations` — citations returned by the Responses API top-level field
- `inline_citations` — `url_citation` annotations extracted from the message body (each with `url`, `title`, `start_index`, `end_index`)
- `credential_source` — `"xai-oauth"` if OAuth resolved, `"xai"` if API key resolved
- `model`, `query`, `provider`, `tool`, `success`

## Example

Talking to the agent:

> What are people on X saying about the new Grok image features? Focus on responses from @xai.

The agent will:

1. Call `x_search` with `query="reactions to new Grok image features"`, `allowed_x_handles=["xai"]`
2. Get back a synthesized answer plus a list of citations linking to specific posts
3. Reply with the answer and references

## Troubleshooting

### "No xAI credentials available"

The tool surfaces this when both auth paths fail. Either set `XAI_API_KEY` in `~/.hermes/.env` or run `hermes auth add xai-oauth` and complete the browser login. Then restart your session so the agent re-reads the tool registry.

### "`x_search` is not enabled for this model"

The configured `x_search.model` doesn't have access to the server-side `x_search` tool. Switch to `grok-4.20-reasoning` (the default) or another Grok model that supports it. Check the [xAI documentation](https://docs.x.ai/) for the current list.

### Tool doesn't appear in the schema

Two possible causes:

1. **Toolset not enabled.** Run `hermes tools` and confirm `🐦 X (Twitter) Search` is checked.
2. **No xAI credentials.** The check_fn returns False, so the schema stays hidden. Run `hermes auth status` to confirm xai-oauth login state, and check that `XAI_API_KEY` is set (if you're using the API-key path).

## See Also

- [xAI Grok OAuth (SuperGrok Subscription)](../../guides/xai-grok-oauth.md) — the OAuth setup guide
- [Web Search & Extract](web-search.md) — for general (non-X) web search
- [Tools Reference](../../reference/tools-reference.md) — full tool catalog
