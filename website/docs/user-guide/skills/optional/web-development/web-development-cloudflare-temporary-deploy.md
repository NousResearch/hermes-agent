---
title: "Cloudflare Temporary Deploy — Deploy a Worker live, no account, via wrangler --temporary"
sidebar_label: "Cloudflare Temporary Deploy"
description: "Deploy a Worker live, no account, via wrangler --temporary"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cloudflare Temporary Deploy

Deploy a Worker live, no account, via wrangler --temporary.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/cloudflare-temporary-deploy` |
| Path | `optional-skills/web-development/cloudflare-temporary-deploy` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `cloudflare`, `workers`, `wrangler`, `deploy`, `temporary`, `agent`, `serverless`, `web-development` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cloudflare Temporary Deploy Skill

Deploy a Cloudflare Worker to a live `workers.dev` URL with zero account setup, using `wrangler deploy --temporary`. Cloudflare provisions a throwaway account, deploys, and prints a claim URL valid for 60 minutes; unclaimed accounts auto-delete. This gives an agent a tight write → deploy → verify loop without any OAuth, signup, or token copy-paste.

This skill does NOT cover production deploys (use `wrangler login` + a permanent account for those), nor non-Worker Cloudflare products beyond the temporary-account limits below.

## When to Use

Load this skill when the user wants to:

- **Ship agent-written code to a live URL** without first creating a Cloudflare account — "deploy this and give me a link"
- **Iterate in a background/autonomous session** where a browser OAuth step would be a hard stop
- **Prototype or evaluate Workers** quickly with a throwaway, claimable target
- **Build a self-verifying deploy loop** — deploy, `curl` the live URL, confirm output matches the code, redeploy

## When NOT to Use

- **Production or CI/CD** → use a permanent account (`wrangler login` or `CLOUDFLARE_API_TOKEN`). `--temporary` errors out if any credential is present.
- **Wrangler is already authenticated** → `--temporary` returns an error by design. Run `wrangler logout` first only if the user explicitly wants a throwaway deploy.
- **Long-lived hosting** → temporary deployments are deleted after 60 minutes unless claimed.

## Prerequisites

- **Wrangler 4.102.0 or later.** This is the version that introduced `--temporary`. Earlier versions do not have it. Verify with `npx wrangler@latest --version`.
- **Node 18+ / npm** (or `npx`, `yarn`, `pnpm`). No global install needed — `npx wrangler@latest` works.
- **No Cloudflare credentials present.** `--temporary` only works when Wrangler is unauthenticated: no OAuth login, no `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_API_KEY` env var, no `~/.wrangler` / `~/.config/.wrangler` cached OAuth. Use the `terminal` tool's environment as-is; do not set those vars.
- Network egress to `cloudflare.com` and `workers.dev`.
- Using `--temporary` accepts Cloudflare's Terms of Service and Privacy Policy.

## How to Run

Use the `terminal` tool for every step. Always pin the version (`wrangler@latest` or `wrangler@4.102.0` or newer) so you don't accidentally run an old global wrangler that lacks the flag.

1. **Scaffold a minimal Worker** (skip if the project already exists). A Worker needs a `wrangler.toml` (or `wrangler.jsonc`) and an entry script. Minimal TypeScript example — write these with `write_file`:

   `wrangler.jsonc`:
   ```jsonc
   {
     "name": "hello-agent",
     "main": "src/index.ts",
     "compatibility_date": "2025-01-01"
   }
   ```

   `src/index.ts`:
   ```typescript
   export default {
     async fetch(): Promise<Response> {
       return new Response("hello cloudflare");
     },
   };
   ```

2. **Deploy with `--temporary`** from the project directory:
   ```
   npx wrangler@latest deploy --temporary
   ```
   The proof-of-work check adds a short automatic delay. On success Wrangler prints an `Account: <name> (created)` (or `(reused)`) line, a `Claim URL`, and the live `https://<worker>.<account>.workers.dev` URL.

3. **Parse the URLs** from that output. Run the helper to extract them reliably instead of eyeballing:
   ```
   npx wrangler@latest deploy --temporary 2>&1 | python3 scripts/parse_deploy_output.py
   ```
   (Resolve `scripts/parse_deploy_output.py` to this skill's absolute path.) It prints JSON: `{"live_url", "claim_url", "account", "account_state", "expires_minutes", "deployed"}`.

4. **Verify the deploy is actually live** — do not trust the deploy log alone. `curl` the live URL and confirm the body matches what the code returns:
   ```
   curl -sS <live_url>
   ```

5. **Iterate.** Edit the code, redeploy with the same `npx wrangler@latest deploy --temporary`. Within the 60-minute window Wrangler reuses the cached temporary account (`Account: <name> (reused)`), so the URL stays stable. `curl` again to confirm the change.

6. **Hand the claim URL to the user.** Tell them: open it within 60 minutes to keep the deployment and any resources; if they don't claim it, everything auto-deletes. Treat the claim URL as a secret — it grants ownership of the account.

## Quick Reference

| Step | Command |
|---|---|
| Check version (need 4.102.0+) | `npx wrangler@latest --version` |
| Deploy (no account) | `npx wrangler@latest deploy --temporary` |
| Deploy + parse URLs | `npx wrangler@latest deploy --temporary 2>&1 \| python3 scripts/parse_deploy_output.py` |
| Verify live | `curl -sS <live_url>` |
| Clear cached temp account | `npx wrangler@latest logout` |

### Temporary account product limits

| Product | Limit on a temporary account |
|---|---|
| Workers | Deploys to `workers.dev` |
| Static Assets | Up to 1,000 files, 5 MiB each |
| KV | Allowed |
| D1 | 1 database, 100 MB per DB / 100 MB total |
| Durable Objects | Allowed |
| Hyperdrive | 2 configs, 10 connections |
| Queues | Up to 10 |
| SSL/TLS certs | Allowed |

## Pitfalls

- **`--temporary` is not in `wrangler deploy --help` and is not a global flag.** It is intentionally hidden and surfaced dynamically: when an unauthenticated `wrangler deploy` fails, Wrangler prints "rerun with `--temporary`". Don't conclude the flag is missing just because `--help` omits it — check the version instead.
- **Old global wrangler.** A stale globally-installed `wrangler` (`< 4.102.0`) silently lacks the flag. Always invoke `npx wrangler@latest` (or a pinned `>=4.102.0`) so you control the version.
- **Auth present → hard error.** If `wrangler login` was ever run, or `CLOUDFLARE_API_TOKEN`/`CLOUDFLARE_API_KEY` is set, `--temporary` errors. Either unset the var for this shell or `wrangler logout`. Never strip a user's real credentials without telling them.
- **Rate limiting.** Creating temporary accounts too fast fails. Reuse the cached account (just redeploy) within the 60-minute window instead of forcing a new one; if rate-limited, wait or use a permanent account.
- **60-minute hard expiry, not extendable.** If the deploy must outlive an hour, the user must claim it. Surface this clearly.
- **`curl` may briefly serve the old body after a redeploy.** `workers.dev` has a short edge cache; the `(reused)` line plus a new `Current Version ID` confirm the deploy succeeded even if `curl` shows stale content for a few seconds. Re-curl, or add a cache-busting query string, before concluding a redeploy failed.
- **Don't log the claim URL into shared transcripts as "just a link."** It is credential-equivalent.

## Verification

- `npx wrangler@latest --version` returns `>= 4.102.0`.
- `npx wrangler@latest deploy --temporary` prints a `workers.dev` live URL and a `claim-preview?claimToken=` claim URL.
- `curl -sS <live_url>` returns the exact body the Worker code produces.
- A second deploy reports `Account: <name> (reused)` and the live URL is unchanged.
- The parser script's self-test passes: `python3 scripts/parse_deploy_output.py --selftest`.
