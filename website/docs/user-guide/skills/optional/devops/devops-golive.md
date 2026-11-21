---
title: "Golive — Ship an agent-built app live on the user's own accounts"
sidebar_label: "Golive"
description: "Ship an agent-built app live on the user's own accounts"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Golive

Ship an agent-built app live on the user's own accounts.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/golive` |
| Path | `optional-skills/devops/golive` |
| Version | `1.0.0` |
| Author | mikehasa (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `deploy`, `hosting`, `vercel`, `netlify`, `supabase`, `stripe`, `dns`, `go-live` |
| Related skills | [`github`](../../bundled/software-development/software-development-github.md), [`cloudflare-temporary-deploy`](../../optional/web-development/web-development-cloudflare-temporary-deploy.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# golive Skill

Ported from mikehasa/golive-skill (MIT) at 9dba601; CLI pinned to npm golive@0.1.0-alpha.4.

Takes an app you wrote from repo to live production on accounts the **human owns**, with providers
they choose: hosting (Vercel, Netlify automated; others guided), database/auth (Supabase, Neon),
payments (Stripe), email (Resend) and DNS (Cloudflare, GoDaddy, Porkbun). The `golive` CLI performs
supported provider writes only after the human approves a plan, then runs live checks and writes
evidence with explicit limits. It does NOT buy domains, create accounts, run app migrations or
prove the whole app works; never turn an infrastructure check into that claim.

## When to Use

The user wants to ship, deploy, go live, launch, publish or put an app online; or asks to wire env
vars, webhooks, Supabase Auth settings (signup, email confirmation, password policy, redirects), a
real signup → confirmation email → login journey, password recovery, account isolation, email DNS
or a custom domain. Not for throwaway previews or tunnels (use `cloudflare-temporary-deploy`).

## Prerequisites

- Node.js **>= 20** and `npx` on PATH (`node --version`). The CLI is fetched from npm on first use.
- The app's repo checked out locally; you run every command from its root (or with `--cwd <dir>`).
- A human present in chat: account logins, purchases and every plan approval are theirs.
- Provider CLIs as needed (`vercel`, `netlify`, `supabase`, `resend`), installed by the human.
- `HERMES_SKILL_DIR` is this skill's directory; if unset, use the directory containing this SKILL.md.

## How to Run

```bash
export GOLIVE_UPDATE_CHECK=0
npx -y golive@${GOLIVE_VERSION:-0.1.0-alpha.4} <command> --json
```

Every command prints one JSON document. Exit code `2` means "worked, but something needs
attention": read the JSON. The npm pin **is** the version control: never run `update-check`,
`install`, `update`, `rollback` or `scripts/install-cli.mjs`; `GOLIVE_UPDATE_CHECK=0` silences the
CLI's own release check. Bump `GOLIVE_VERSION` deliberately, never mid-run.

## Quick Reference

| Command | Purpose | Writes to providers? |
|---|---|---|
| `version --json` | Release identity of the pinned CLI | no |
| `detect --json` | Framework, providers in code, env var names, critical findings | no |
| `menu --json` | Provider options per axis, `automated` vs guided | no |
| `init --stack k=v,... [--domain d] [--email-from x] [--project axis=id] [--webhook-path p] [--events a,b] [--stripe-publishable test=pk_test_…,live=pk_live_…] --json` | Write `golive.yaml` | no (local file) |
| `credentials --setup --json` / `--prompt NAME --lang en --json` / `--remove NAME --yes --json` | Private local credentials file; native masked entry (macOS); remove one entry | no (local file) |
| `doctor --json` | Is each provider reachable/logged in; `howToFix` for the human | no |
| `plan --json` | Read-only step list with `planId`, `steps[].needs`, `handoffs` | no |
| `apply --plan <id> --yes [--confirm-live] [--confirm-dns] [--confirm-destroy] [--only id,id] --json` | Execute the approved plan | **yes** |
| `teardown --json` → `apply --plan <id> --yes --confirm-destroy [--confirm-dns] --json` | Inverse plan: only resources golive created | **yes (deletes)** |
| `verify [--only id,id] --json` | Live checks; writes `.golive/report.json`, `GOLIVE_REPORT.md` | opt-in auth probes only |
| `status --json` | Recorded baselines vs reads now; exit 2 = act; never a gate | no |
| `handoff [--write] [--force] --json` | Human-only items and whether closed; `--write` emits `GOLIVE_HANDOVER.md` | no |

## Hard rules (never break these)

1. **Never print, echo, `cat`, or paste a secret value** (`.env` files, API keys, tokens, database
   URLs, `~/.config/golive/credentials`). Refer to secrets by name. The CLI never prints them.
2. **Secrets never go through this chat.** Never ask the human to paste a secret key or token here.
   Prefer provider integrations or supported local secret transport. When guided setup has no safe
   automated route, the human may enter a value directly in the destination dashboard in their own
   browser; you must not view or capture it. If they paste one into chat anyway, don't use it; tell
   them it is now in the transcript and should be rotated. The one exception: Stripe **publishable**
   keys (`pk_test_…`, `pk_live_…`) are public and may be given in chat. Never `sk_`, `rk_`, `whsec_`.
3. **No provider/account writes until the human approves the plan.** Local credential setup,
   `init` and report files may be prepared during onboarding. Explain `plan` and get a clear yes in
   chat before `apply`. Pass `--confirm-live` (live payments, production data, or a **first**
   production deploy to a destination golive has never deployed; also the `auth:test-user`,
   `auth:isolation`, `auth-signup` probe and `auth:recovery` writes), `--confirm-dns` (DNS records) or
   `--confirm-destroy` (deletions) only if the human explicitly approved those categories. Say why
   you are asking for each: `steps[].needs` names the flags a step requires; a first production
   deploy needs `--confirm-live` because approving the plan approves what that deploy contains, not
   the first write to production itself. Later deploys of that target need no extra flag.
4. **Never buy anything or create accounts for them.** Signups, payment methods, identity checks
   (KYC) and domain purchases are handoffs the human does in their browser.
5. **A handoff is closed only by a passing check**, not by anyone saying "done". `done: false` is
   open. `done: null` (a `manual` item, or its check skipped) cannot be verified by golive: confirm
   it with the human and name it as **not verified by golive** in your final summary. If a skipped
   check's evidence says the step is recorded done in `.golive/state.json`, the work ran and only
   this invocation could not re-check it — say that, not that it is unproven.
6. **Stay neutral.** Present provider options without steering. If they already use something, keep it.
7. **No human, no `apply`.** On a headless surface (cron, a background job, a session with no human
   answering in chat) stop after `plan`: write the plan document, report the `planId`, and refuse to
   run `apply` or `teardown … apply`. "What's next?" or an old approval is never consent.

## How the human connects accounts

golive runs in *your* shell (the Hermes `terminal` tool), so a token the human `export`s in their own
terminal never reaches it. In order of preference:

1. **The vendor's browser login** when the adapter supports the required operations
   (`vercel login`, `supabase login`, `resend login`, `netlify login`, `neon auth`). The human runs it
   in **their own terminal window** (Terminal app or IDE terminal), **not through Hermes**: the
   `terminal` tool has no interactive TTY, so these logins fail or hang there. Nothing is copied.
   Check the CLI is on PATH afterwards. Never suggest a `--token` / `--key` login flag, even when a
   CLI error hint does: it puts the secret on the command line. If macOS Keychain or the vendor login
   raises a system prompt, explain which app asks and why ("Allow" answers once, "Always Allow"
   records it); never collect their Mac password or imitate an OS prompt.
2. **Native token entry on macOS** when a manual API key is actually needed. Give the exact
   variable name, provider token page, scope and permissions first, then run
   `credentials --prompt NAME --lang en --json` (`zh` when appropriate) — the name only, never a
   value. A masked local dialog stores it without returning it to you. Do not inspect the dialog,
   clipboard, credential file or raw child output. `saved` → re-run `doctor`. Confirm before
   `--replace`. `cleanupRequired: true` → repair local cleanup only (see troubleshooting.md); do not
   re-prompt. A cancellation means stop and wait. `envOverride: true` → the process env wins; say so.
3. **Manual fallback: the credentials file** `~/.config/golive/credentials` (path shown by `doctor`).
   Run `credentials --setup --json` first (creates a private empty file, preserves contents, returns
   metadata only; never read it). Give the human the exact `NAME=value` variable name, token page,
   scope and permissions; they edit in their own editor (for nano spell out **Ctrl+O → Enter →
   Ctrl+X**). You never enter token values.
4. The token exported in the shell Hermes was launched from (then restart Hermes).

`credentials --remove NAME --yes` deletes one entry and returns metadata only; pass `--yes` only
when the human asked to remove that specific credential. Removing golive's copy does not revoke it.
Vercel deploys always run through the Vercel CLI (`npm i -g vercel`); `VERCEL_TOKEN` only replaces
`vercel login`. Present only the applicable entry method from `doctor`'s `howToFix`, in the human's
language, and load `references/<provider>.md` once a provider is selected.

## Procedure

1. **Detect** — `detect --json`. Report framework, providers in code, env var *names*. Fix every
   **critical** finding in code first (`secret-in-client-env`, `config-inlines-all-env`); golive
   writes no server secrets to that host until they are gone. Gate: no critical findings remain.
2. **Choose providers** — `menu --json`, then `init --stack … --json`. Ask only about pieces the app
   needs and lacks; keep what the repo already uses; mark automated vs guided; offer "Other (guided,
   best effort)" using the menu's id or a lowercase-hyphen id, never `other`. Ask about a custom
   domain and the email "from" address; distinguish registrar from DNS host. Account and project
   are separate choices — pass `--project` for a deliberately chosen existing project. Gate:
   `golive.yaml` written and the stack read back to the human.
3. **Accounts** — `doctor --json`. For each `ok: false`, hand over its `howToFix` (section above).
   Guided providers return `ok: false` / exit 2 by design; that is not a login failure. Gate: every
   automated provider `ok: true`.
4. **Plan** — `plan --json`. Write `docs/GOLIVE-<stage>-PLAN.md` with `write_file` (IDs included),
   then put a short consent summary **directly in chat**: destination per axis (account/team/org →
   project, new or existing, region), what is created/changed, test/live mode, cost or unknown
   cost, which steps `need` `--confirm-live`/`--confirm-dns`/`--confirm-destroy`, `handoffs` and
   `unmappedEnv`. Name the `planId` and ask for an explicit yes to those exact destinations. Gate:
   explicit yes in chat (rule 7). Changing a destination needs a fresh plan and approval.
5. **Apply** — `apply --plan <planId> --yes [flags] --json`. Report each step. `failed`/`blocked`:
   read `error`/`next`, fix, re-run (completed steps skip). If a write may have reached the provider,
   reconcile via troubleshooting.md before repeating any creation. "plan changed" or DNS records
   changed → `plan` again, approve again. Re-`plan` after a successful apply until only zero-write
   pins remain (webhook and site URL appear after the first deploy). Write
   `docs/GOLIVE-<stage>-RESULT.md`. Gate: no failed/blocked steps.
6. **Verify** — `verify --json` (writes `GOLIVE_REPORT.md`). `skip` means blocked or not applicable,
   **never passed**. If `accounts` fails, fix logins and re-run. Then `handoff --json`: summarize the
   live URL, what passed, what is open and every `done: null`/skipped item as not verified by golive,
   with an owner for each. Offer `handoff --write --json` for the ownership document.
7. **Status** — `status --json` before a release and after any run that changed providers. Report
   `expected` (recorded) vs `observed` (now) with `subject`; `action: verify` → `verify --only
   <checkId>`; `reconcile` → plan/approve/apply; `human` → ask. Never use `status` as a gate or
   re-baseline by hand. Teardown: `teardown --json`, show the list, explicit approval, then `apply`
   with `--confirm-destroy` (plus `--confirm-dns`/`--confirm-live` as `needs` says).

Keep the current stage visible at handoffs: **completed / next step / what you need from them**.
On "what's next?", read `golive.yaml`, non-secret `.golive/state.json` and the latest plan/result;
resume, don't restart. `.env`, credentials and vendor login files are never context to read. Follow
the human's language. Recommend adding `.golive/`, `GOLIVE_REPORT.md`, `GOLIVE_HANDOVER.md` to the
app's `.gitignore`.

## Troubleshoot, then resume

When a setup command fails, resolve that specific failure before continuing. Keep the app directory,
chosen stack, approved plan and completed resource IDs; onboarding does not restart. For `command
not found`, PATH/version differences, failed login or an interrupted provider operation load
`references/troubleshooting.md`. Use narrow diagnostics that cannot expose credentials, verify the
repair with the appropriate CLI/account check, and return to the same stage. Explain **what failed /
what now passes / the next step**. A repaired command authorizes no new destinations, paid
operations or changed plan.

## Pitfalls

1. The CLI is **alpha** (`0.1.0-alpha.4`); breaking changes between alphas are likely. The pin
   protects the flow — bump `GOLIVE_VERSION` deliberately, re-run `version --json`, and never between
   `plan` and `apply` (a changed release invalidates the approval).
2. `npx -y` needs network on first use (~1 MB bundle cached under `~/.npm/_npx`). Offline → stop.
3. Do not run `update-check`, `install*`, `update`, `rollback`, `update-policy`, `recover-lock` or
   `scripts/install-cli.mjs`: they manage upstream's self-installed copies, which this port has none of.
4. Interactive vendor logins hang under the `terminal` tool. The human runs them in their own window.
5. `credentials --prompt` (masked dialog) is macOS-only; on Linux use the credentials-file fallback.
6. Preview/promote/rollback (`release.*`), `auth-isolation`, `production-release` are implemented
   and mock-covered upstream, **not live-validated**; never present them as proven until a report
   says `pass`. Vercel exposes no production-deployment read, so those checks skip there.
7. `auth.e2e`, `auth.recovery`, `auth.isolation` create real accounts / rotate passwords in the
   human's project (`--confirm-live`); say so plainly before they opt in. A same-name existing
   project is never adopted silently; in throwaway tests choose a fresh name.
8. Reports and handover files carry account names and resource IDs (no secret values): tell the
   human to review before sharing.

## Verification

- `node --version` >= 20; `version --json` returns `"version": "0.1.0-alpha.4"` and a `release.bundleDigest`.
- `detect --json` returns `ok: true` for the app root (framework not `unknown` for a real app).
- After `apply`, `plan --json` shows only zero-write project pins; `verify --json` has no `fail`, and
  every `skip`/`done: null` is named in your summary as not verified by golive.
- `status --json` exits 0, or each exit-2 item is explained to the human.

## References

- `references/flow.md` — the full seven-step CLI contract verbatim (JSON shapes, step ids,
  confirm-flag semantics, check-id table, approval/resume boundary). Load when explaining any
  `plan`/`apply`/`verify`/`status` output in detail.
- `references/plan-and-verify.md` — detect findings, plan steps and ordering, handoffs, why each
  check skips, what `status` compares. Load when a step or check needs justification.
- `references/guided.md` — load when the chosen provider is not automated ("Other").
- `references/troubleshooting.md` — load on setup failures, CLI/PATH mismatches, resuming after repair.
- `references/vercel.md`, `references/netlify.md` — load when `hosting=` is that provider.
- `references/supabase.md`, `references/neon.md` — load when `db=`/`auth=` is that provider.
- `references/stripe.md` — load when `payments=stripe`. `references/resend.md` — when `email=resend`.
- `references/cloudflare-dns.md`, `references/godaddy.md`, `references/porkbun.md` — load when
  `dns=` is that provider.

Load with `skill_view(name="golive", file_path="references/<topic>.md")`.
