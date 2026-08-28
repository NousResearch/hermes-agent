---
title: "Stripe Projects — Provision SaaS services + sync creds via Stripe Projects"
sidebar_label: "Stripe Projects"
description: "Provision SaaS services + sync creds via Stripe Projects"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stripe Projects

Provision SaaS services + sync creds via Stripe Projects.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/payments/stripe-projects` |
| Path | `optional-skills/payments/stripe-projects` |
| Version | `0.1.0` |
| Author | Teknium (teknium1), Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `Payments`, `Stripe`, `Projects`, `Provisioning`, `Infrastructure` |
| Related skills | [`stripe-link-cli`](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli), [`mpp-agent`](/docs/user-guide/skills/optional/payments/payments-mpp-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Stripe Projects Skill

Wraps the [Stripe Projects](https://projects.dev) CLI plugin so Hermes can provision SaaS services (Neon, Twilio, Vercel, etc.), generate and sync credentials into the user's `.env`, and manage billing across providers from one place.

Gated `[linux, macos]` while the broader payments cluster matures on Windows. The Stripe CLI itself is cross-platform; this gate is a posture for the cluster, not a hard limit.

## When to Use

Trigger phrases:

- "set up &lt;provider>", "provision &lt;Neon|Twilio|Vercel|...>", "create a database"
- "give me a &lt;Postgres|Redis|Twilio number|...> for this project"
- "manage my stack credentials", "rotate this key", "upgrade my plan"
- "what providers can I add?"

If the user already has a provider account, this skill can still connect it with `stripe projects link <provider>`. If the user wants to use an existing provider resource, such as an existing database or Vercel project, check provider support first; many providers currently support provisioning new resources but not importing existing ones.

## Prerequisites

- Stripe CLI installed (Homebrew on macOS, package manager on Linux, or download from https://docs.stripe.com/stripe-cli/install)
- Stripe Projects plugin installed
- A Stripe account. If the user doesn't have one yet, the CLI can guide them through sign-in or account creation in the browser during setup.

## Install

macOS:

```
brew install stripe/stripe-cli/stripe
stripe plugin install projects
```

Linux: follow the platform-specific install at https://docs.stripe.com/stripe-cli/install, then:

```
stripe plugin install projects
```

## How to Run

All commands run through the `terminal` tool from inside the user's project directory (the CLI writes `.env` and `.projects/vault/vault.json` into the CWD).

## Procedure

### 1. Initialize the project

```
cd <project-root>
stripe projects init
```

This creates `.projects/vault/vault.json` (encrypted credential store) and prepares the project to receive providers.

### 2. Discover available providers

```
stripe projects catalog
```

Lists every provider Stripe Projects supports — databases, hosting, auth, AI, analytics, messaging, etc.

### 3. Add a service

```
stripe projects add <provider>/<service>
```

Examples:

- `stripe projects add neon/postgres`
- `stripe projects add twilio/sms`
- `stripe projects add runloop/sandbox`

The CLI provisions the service in the user's own account with the provider, generates credentials, syncs them into `.env`, and records the resource in the vault. The user may need to confirm a tier selection or pricing prompt.

### 4. Verify

```
stripe projects list
```

Should show the newly added provider and its `.env` keys.

### 5. Manage / upgrade / remove

```
stripe projects upgrade <provider>     # tier change
stripe projects remove <provider>      # deprovision
stripe projects rotate <provider>      # rotate credentials
```

## Pitfalls

- **`.env` writes are real writes.** The CLI appends to whatever `.env` is in the project root. If the user's `.env` is gitignored (normal), the keys land safely; if not, this skill could be a credential-leak vector. Always check `.gitignore` first.
- **Per-project state.** `.projects/vault/vault.json` is per-project. Provisioning the same service in two different projects creates two separate resources — and two bills.
- **Billing happens on Stripe's side.** Tier prompts during `add`/`upgrade` are real charges; surface them to the user before confirming.
- **Provider availability changes.** The catalog grows; if a provider the user names isn't listed, `stripe projects catalog | grep <name>` first instead of failing the `add` call.
- **Credentials in vault are encrypted but `.env` is plaintext.** Standard `.env` hygiene applies — never commit it.
- **Removing a service does NOT always destroy the underlying resource.** Some providers leave a paused/dormant resource behind. Check the provider's own dashboard after `remove` for high-cost services (managed databases especially).

## Verification

```
stripe projects --version && stripe projects list
```

Exit code 0 inside an initialized project means the plugin is healthy.
