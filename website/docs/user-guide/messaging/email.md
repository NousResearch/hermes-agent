---
sidebar_position: 7
title: "Email"
description: "Set up Hermes Agent as an email assistant via IMAP/SMTP"
---

# Email Setup

Hermes can receive and reply to emails using standard IMAP and SMTP protocols. Send an email to the agent's address and it replies in-thread — no special client or bot API needed. Works with Gmail, Outlook, Yahoo, Fastmail, or any provider that supports IMAP/SMTP.

:::info Gateway adapter only: no external dependencies
This page covers the Email gateway adapter, which uses Python's built-in `imaplib`, `smtplib`, and `email` modules. No additional packages or external services are required for this gateway path.
:::

This is separate from the bundled [Himalaya email skill](/docs/user-guide/skills/bundled/email/email-himalaya), which lets the agent manage email through terminal commands and requires the external `himalaya` CLI plus a Himalaya config file.

| Use case | What to configure | External dependency |
|---|---|---|
| Let people email the Hermes agent and receive replies | Email gateway adapter on this page | None beyond an IMAP/SMTP email account |
| Let the agent inspect, compose, move, and manage mailbox messages from terminal tools | Himalaya email skill | `himalaya` CLI and `~/.config/himalaya/config.toml` |

**Email is an externally reachable, prompt-injection surface.** Before relying on it, read [Access Control](#access-control) (who can reach the agent) and [Threat model](#threat-model) below. By default every sender is denied, and an allowlisted sender is trusted only after the receiving mail server's SPF/DKIM/DMARC verdicts validate the `From:` domain.

---

## Prerequisites

- **A dedicated email account** for your Hermes agent (don't use your personal email)
- **IMAP enabled** on the email account
- **An app password** if using Gmail or another provider with 2FA

### Gmail Setup

1. Enable 2-Factor Authentication on your Google Account
2. Go to [App Passwords](https://myaccount.google.com/apppasswords)
3. Create a new App Password (select "Mail" or "Other")
4. Copy the 16-character password — you'll use this instead of your regular password

### Outlook / Microsoft 365

1. Go to [Security Settings](https://account.microsoft.com/security)
2. Enable 2FA if not already active
3. Create an App Password under "Additional security options"
4. IMAP host: `outlook.office365.com`, SMTP host: `smtp.office365.com`

### Other Providers

Most email providers support IMAP/SMTP. Check your provider's documentation for:
- IMAP host and port (usually port 993 with SSL)
- SMTP host and port (usually port 587 with STARTTLS)
- Whether app passwords are required

### Proton Mail Bridge / local relays

Proton Mail Bridge (and similar local relays such as a self-hosted MTA) listen on
loopback with **STARTTLS** and a self-signed certificate, so the defaults
(implicit TLS on IMAP 993, verified certificates) won't connect. Override the
transport in `~/.hermes/config.yaml`:

```yaml
platforms:
  email:
    enabled: true
    extra:
      imap_host: 127.0.0.1
      imap_security: starttls     # tls (default) | starttls | plain
      imap_tls_verify: false      # Bridge uses a self-signed cert
      smtp_host: 127.0.0.1
      smtp_security: starttls     # default: tls on port 465, starttls otherwise
      smtp_tls_verify: false
```

and set `EMAIL_IMAP_PORT=1143` / `EMAIL_SMTP_PORT=1025` alongside your Bridge
credentials in `~/.hermes/.env`. Unknown `*_security` values log a warning and
fall back to the secure default. Only disable `*_tls_verify` for loopback hosts —
Hermes logs a warning when verification is off for any other host.

---

## Step 1: Configure Hermes

The easiest way:

```bash
hermes gateway setup
```

Select **Email** from the platform menu. The wizard prompts for your email address, password, IMAP/SMTP hosts, and allowed senders.

### Manual Configuration

Add to `~/.hermes/.env`:

```bash
# Required
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # App password (not your regular password)
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# Security (recommended)
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com

# Optional
EMAIL_IMAP_PORT=993                    # Default: 993 (IMAP SSL)
EMAIL_SMTP_PORT=587                    # Default: 587 (SMTP STARTTLS)
EMAIL_POLL_INTERVAL=15                 # Seconds between inbox checks (default: 15)
EMAIL_HOME_ADDRESS=your@email.com      # Default delivery target for cron jobs
EMAIL_AUTHSERV_ID=mx.google.com        # Pin trusted Authentication-Results server (see Access Control)
```

Almost all access-control and outbound-safety settings live in `config.yaml` under `platforms.email.extra` — most importantly `read_only` for an inbound-only feed, `require_authenticated_sender`, and `authserv_id`. See [Access Control](#access-control) and [Inbound-only mode](#inbound-only-mode).

---

## Step 2: Start the Gateway

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

On startup, the adapter:
1. Tests IMAP and SMTP connections (the SMTP connection test is **skipped** in [inbound-only mode](#inbound-only-mode))
2. Marks all existing inbox messages as "seen" (only processes new emails)
3. Starts polling for new messages

---

## How It Works

### Receiving Messages

The adapter polls the IMAP inbox for UNSEEN messages at a configurable interval (default: 15 seconds). For each new email:

- **Subject line** is included as context (e.g., `[Subject: Deploy to production]`)
- **Reply emails** (subject starting with `Re:`) skip the subject prefix — the thread context is already established
- **Attachments** are cached locally:
  - Images (JPEG, PNG, GIF, WebP) → available to the vision tool
  - Documents (PDF, ZIP, etc.) → available for file access
- **HTML-only emails** have tags stripped for plain text extraction
- **Self-messages** are filtered out to prevent reply loops
- **Automated/noreply senders** are silently ignored — addresses matching `noreply`, `no-reply`, `no_reply`, `donotreply`, `do-not-reply`, `mailer-daemon`, `postmaster`, `bounce`, `notifications@`, `automated@`, `auto-confirm`, `auto-reply`, or `automailer`, and emails carrying `Auto-Submitted` (anything but `no`), `Precedence: bulk|list|junk`, `X-Auto-Response-Suppress`, or `List-Unsubscribe` headers
- **Sender `From:` authentication is checked before the allowlist is matched.** SPF/DKIM/DMARC verdicts are read from the `Authentication-Results` header stamped by the receiving mail server (see [Access Control](#access-control)). An allowlisted address whose `From:` domain is not authenticated is dropped, because the `From:` header is attacker-controlled and never authenticated by IMAP delivery.

### Sending Replies

Replies are sent via SMTP with proper email threading:

- **In-Reply-To** and **References** headers maintain the thread
- **Subject line** preserved with `Re:` prefix (no double `Re: Re:`)
- **Message-ID** generated with the agent's domain
- Responses are sent as plain text (UTF-8)

### Quiet email sessions (display overrides)

Email is a permanent-message mailbox — there is no editing, deleting, or streaming a message after it's sent. A single email-triggered session can otherwise churn out dozens of SMTP sends from ordinary live-work events: interim assistant commentary, tool-progress updates, heartbeats, long-running notifications, and busy-ack detail would each become their own email.

To prevent that, Email uses a **minimal display tier**: `tool_progress: off`, `streaming: false`, `interim_assistant_messages: false`, `long_running_notifications: false`, `busy_ack_detail: false`. Since the inbound-only hardenings, Email's minimal defaults are **pinned**: a global `display.<key>` setting no longer cascades into Email and silently raises its verbosity. Only an explicit `display.platforms.email.<key>` override beats the minimal tier.

To keep intent visible to other operators (and to stay protected even if default tiers change), set them explicitly in `config.yaml`:

```yaml
display:
  platforms:
    email:
      interim_assistant_messages: false
      tool_progress: off          # off | new | all | verbose | log
      streaming: false
      long_running_notifications: false
      busy_ack_detail: false
```

`tool_progress` accepts the same values as the global setting; `off` and `false` both normalize to `off`.

These settings are complementary to — **not a substitute for** — [inbound-only mode](#inbound-only-mode). Display overrides reduce *which* events are produced in the first place. `read_only` gate-keeps *every* outbound send at the SMTP boundary regardless of verbosity.

### Inbound-only mode

Use this supported `config.yaml` setting when Email is an authenticated input feed and Hermes must **never** automatically reply to its sender:

```yaml
platforms:
  email:
    extra:
      read_only: true
```

`read_only` is a **config.yaml-only** switch (the derived `EMAIL_READ_ONLY` environment variable is **not honored** — setting it has no effect). This keeps inbound-only mode explicit and deliberate in the machine's config rather than a silent env flag.

With `read_only: true`, the adapter:

- **Keeps all inbound behavior unchanged** — it still connects over IMAP, polls, dispatches new messages, creates normal Email-source sessions, and the full agent work, final answer, and tool activity still appear in the session for Hermes Desktop.
- **Skips the SMTP connection test at startup** and does **not require `EMAIL_SMTP_HOST`** — the mailbox is a pure read feed.
- **Suppresses every outbound send before SMTP** — task replies, interim commentary, progress, final answers, approval prompts, media follow-ups, image batches, document sends, and delivery retries. Each suppressed send returns `success` with message id `read-only-suppressed`, so the gateway's delivery ledger marks it delivered and never enters a retry loop.
- **Applies the same suppression at the standalone SMTP boundary** used by cron, report, and one-shot deliveries — no Email route can escape inbound-only mode.
- **Logs an audit line per suppressed send** (INFO) with recipient, subject, session, and delivery kind. It does **not** log the body, attachment paths, or credentials.

**Interplay with display overrides:** `read_only` is the enforce-at-egress control; the [quiet display settings](#quiet-email-sessions-display-overrides) are the reduce-what's-produced control. Use both. `read_only` guarantees no email is sent even if verbosity is raised; quiet display keeps the session transcript clean and avoids hundreds of pointless suppression audit lines.

:::warning
Do **not** configure an Email cron/report target while `platforms.email.extra.read_only: true` is enabled — those deliveries are suppressed too and will silently never arrive. Disable inbound-only mode only when you intentionally want Hermes to send Email again. The planned [draft approval (P1)](#draft-approval-p1--planned) removes the need to pick one or the other.
:::

### File Attachments

The agent can send file attachments in replies. Include `MEDIA:/path/to/file` in the response and the file is attached to the outgoing email.

### Skipping Attachments

To ignore all incoming attachments (for malware protection or bandwidth savings), add to your `config.yaml`:

```yaml
platforms:
  email:
    extra:
      skip_attachments: true
```

When enabled, attachment and inline parts are skipped before payload decoding. The email body text is still processed normally.

---

## Access Control

Email follows a **default-deny** model: a message is admitted and dispatched only when it satisfies the allowlist (or an explicit open-access opt-in) **and** — when the allowlist is what grants access — the sender's `From:` domain passes mail authentication. The logic is checked in two layers and BOTH must agree for a sender to reach the agent.

### 1. Adapter intake gate (per-message)

The adapter drops a message before it ever becomes a session event if:

- it is a **self-message**, or
- it matches the **automated/noreply** patterns, or
- the sender is **not in `EMAIL_ALLOWED_USERS`** and open access (`EMAIL_ALLOW_ALL_USERS` / `GATEWAY_ALLOW_ALL_USERS`) is not enabled. With no allowlist and no allow-all, **every** sender is silently dropped — default-deny at the intake boundary. (This gate keys on the per-platform `EMAIL_ALLOWED_USERS`; see the **global-allowlist nuance** note below.)

### 2. Gateway authorization

A dispatched event is authorized by the gateway if the sender is allowlisted (`EMAIL_ALLOWED_USERS` or the global `GATEWAY_ALLOWED_USERS`), DM-paired, or covered by an allow-all flag; otherwise it is denied (default `ignore` for unexpected Email senders).

:::note Global-allowlist nuance for Email
The gateway itself also honors `GATEWAY_ALLOWED_USERS`. However, because the adapter's **intake gate is keyed on `EMAIL_ALLOWED_USERS` only**, a sender listed in `GATEWAY_ALLOWED_USERS` but not in `EMAIL_ALLOWED_USERS` is dropped at intake and never reaches the gateway. In practice, **`EMAIL_ALLOWED_USERS` is the operative allowlist for Email** — keep your email senders there.
:::

### Decision table

| Configuration | Who can reach the agent |
|---|---|
| No `EMAIL_ALLOWED_USERS`, no allow-all | **No one.** All senders silently dropped (default-deny). |
| `EMAIL_ALLOWED_USERS=alice@x.com,bob@y.com` | Only those addresses — **and** each must pass mail authentication (below) unless you explicitly disable it. |
| `EMAIL_ALLOW_ALL_USERS=true` (or `GATEWAY_ALLOW_ALL_USERS=true`) | **Any** sender on the internet — opens the whole world to a terminal-capable agent. Not recommended; use only deliberately. |
| `platforms.email.extra.unauthorized_dm_behavior: pair` | Opts Email into the pairing flow instead of ignoring unknown senders. Because Email's intake gate rejects unknown senders first unless the mailbox is effectively open (allow-all), pairing for Email in practice requires opening intake — **an explicit allowlist is the supported and recommended control.** |

:::warning
**Do not set `EMAIL_ALLOW_ALL_USERS` or `EMAIL_TRUST_FROM_HEADER` on an inbound-only (or any externally reachable) mailbox.** `EMAIL_ALLOW_ALL_USERS` opens the mailbox to every sender. `EMAIL_TRUST_FROM_HEADER` disables the mail-authentication gate below and lets an allowlist be keyed on a forged `From:` header.
:::

### Sender authentication (`require_authenticated_sender` + `authserv_id`)

The `From:` header is **attacker-controlled and is never authenticated by IMAP delivery**. Mail applications connect to inboxes over TLS, but TLS authenticates the *transport*, not the authorship of the `From:` address. An allowlist keyed on `From:` alone is therefore spoofable — the fix that drove this design is tracked as GHSA-rxqh-5572-8m77.

Hermes closes this by reading the `Authentication-Results` header that your **receiving mail server** (the one you IMAP into) stamps after running SPF/DKIM/DMARC. A sender is trusted only when that header records a **pass** for the `From:` domain:

- `dmarc=pass`, **or**
- `spf=pass` aligned with the `From:` domain, **or**
- `dkim=pass` aligned with the `From:` domain (via `header.d`)

Configuration:

- `platforms.email.extra.require_authenticated_sender: true` — **default on** when it matters (an allowlist is actively granting access). When on, an allowlisted sender whose `From:` domain does not carry a pass verdict is dropped with a warning that includes the failure reason.
- `platforms.email.extra.authserv_id: mx.google.com` (or the env mirror `EMAIL_AUTHSERV_ID`) — **pins** the trusted `Authentication-Results` instance to your own receiving server (e.g. `mx.google.com` for Gmail). Use this when you know your provider's receiving hostname. Default (unpinned) trusts the topmost `Authentication-Results` header, which protects against forged `From:` but not against a malformed message whose forged header sorts first — pinning closes that last gap.

**Fail-closed:** a message with no `Authentication-Results` header, no instance from the pinned `authserv_id`, or non-pass verdicts is not admitted. If your mail server does not stamp `Authentication-Results` at all, your realistic options are to fix the delivery/verdicts at the server, or explicitly disable the check and accept the spoofing risk (`platforms.email.extra.require_authenticated_sender: false`, or the env mirror `EMAIL_TRUST_FROM_HEADER=true`). The auth gate is deliberately **skipped** when allow-all is active, because an operator who opted into any-sender access has already chosen to accept that risk.

---

## Threat model

Treat an email-only mailbox as a **remote, unauthenticated command surface with credential exposure baked in**. Understand these before exposing the agent by email.

**Who can command the mailbox.** Only the senders that survive the two-layer check above: allowlisted addresses whose `From:` domain passes SPF/DKIM/DMARC (or an explicit open-access opt-in, which should never be used for an externally reachable mailbox). An attacker forging an allowlisted `From:` is dropped because the mail-authentication verdicts won't validate the forged domain — unless you disabled `require_authenticated_sender` or the receiving server stamps no verdicts at all.

**Quoted/forwarded content is data, not authority.** Anything inside an email body — a quote from a third party, a forwarding chain, an attachment, a copied salesperson — is **untrusted content** to the agent, never a directive. The allowlist gates *who may transmit*; it does not make the transmission trustworthy. A human in the allowlist can still be socially engineered into forwarding an attack. Bake that separation into standing instructions and require explicit operator confirmation for consequential actions. (The [dedicated-address guide](/docs/guides/agent-email-address) applies the same rule to the Himalaya skill path.)

**A compromised owner mailbox is a control credential — not just a confidentiality leak.** The `.env` App Password grants full IMAP read access to every inbound request and send authority over the account. Worse, the agent holding that credential also has terminal access to the machine: gaining the mailbox can be a foothold toward that. Protect `~/.hermes/.env` (`chmod 600`), use a dedicated mailbox, rotate the App Password if you suspect exposure, and never reuse the agent's account for password resets, account recovery, or 2FA on other services.

**Provider quotas are a backstop, not a control.** Gmail-class providers cap outbound SMTP volume per account per day and limit concurrent IMAP connections. These bound the blast radius of a runaway send loop (the historical incident that shaped this page produced 74 outbound emails from a single session), but a flood that stays under the quota still happens. Gate sends at the adapter — with [inbound-only mode](#inbound-only-mode) or the planned [draft approval](#draft-approval-p1--planned) — rather than relying on the provider to stop you. Similarly, keep only **one** gateway instance polling the mailbox so the provider's concurrent-IMAP limit isn't tripped and replies aren't duplicated.

---

## Draft approval (P1) — planned

:::info Status: planned / in progress — not yet released
This section describes the **P1 design intent** (issue #99876 family, branch `feat/email-draft-approval`). It is **not merged** at the base of these docs. Config keys, storage paths, and behavior below are subject to change before release — verify against the merged release notes before relying on them.
:::

Inbound-only mode solves "never send", but it is binary: there is no supported way to let the agent *compose* email yet still gate the *sending* on human review. P1's goal is review-then-send — kill the "send fast, regret later" class while keeping outbound available.

Planned building blocks:

- **Durable draft store.** Every outbound email is first written as a draft to disk, surviving restarts. Nothing touches SMTP until explicitly approved.
- **One-shot atomic approval.** Approving a draft authorizes exactly **that** draft, sent exactly **once**. Approval is atomic and immediate: there are no standing "always allow this sender / this session" grants implied.
- **Expiry.** Drafts carry a time-to-live; untouched drafts are dropped unapproved, so no pending send lingers indefinitely.
- **Budgets / circuit breaker.** A per-period send budget and a tripwire that halts further sends until an explicit reset — bounding recurrence of a flood even with outbound enabled.
- **Interrupt fence.** A pending approval is isolated from concurrent activity: inbound mail, `/steer`, interrupts, or a new turn cannot amend, approve, or bypass a pending draft once it has been submitted.
- **Desktop surface.** The Hermes Desktop app is the approval surface — a draft renders as an approvable card (recipient, subject, body summary) with approve/deny that commits the one-shot decision.

Until P1 lands, the supported way to prevent automatic outbound is [inbound-only mode](#inbound-only-mode).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"IMAP connection failed"** at startup | Verify `EMAIL_IMAP_HOST` and `EMAIL_IMAP_PORT`. Ensure IMAP is enabled on the account. For Gmail, enable it in Settings → Forwarding and POP/IMAP. |
| **"SMTP connection failed"** at startup | Verify `EMAIL_SMTP_HOST` and `EMAIL_SMTP_PORT`. Check that your password is correct (use App Password for Gmail). If you intend an inbound-only mailbox, set `platforms.email.extra.read_only: true` so the SMTP test is skipped. |
| **Messages not received** | Check `EMAIL_ALLOWED_USERS` includes the sender's email. Check spam folder — some providers flag automated replies. |
| **"Authentication failed"** | For Gmail, you must use an App Password, not your regular password. Ensure 2FA is enabled first. |
| **Agent reads mail but never replies** | Confirm `platforms.email.extra.read_only` is not set. Confirm the sender is in `EMAIL_ALLOWED_USERS`. If the log shows `Dropping sender with unauthenticated From:`, the sender's mail authentication failed or the receiving server stamps no verdicts — see [Access Control](#access-control). |
| **Duplicate replies** | Ensure only one gateway instance is running. Check `hermes gateway status`. |
| **Slow response** | The default poll interval is 15 seconds. Reduce with `EMAIL_POLL_INTERVAL=5` for faster response (but more IMAP connections). |
| **Replies not threading** | The adapter uses In-Reply-To headers. Some email clients (especially web-based) may not thread correctly with automated messages. |
| **Many outbound emails from one task** | Set the [quiet display overrides](#quiet-email-sessions-display-overrides) so progress/interim events aren't emailed, and/or use [inbound-only mode](#inbound-only-mode). |

---

## Security

:::warning
**Use a dedicated email account.** Don't use your personal email — the agent stores the password in `.env` and has full inbox access via IMAP. A compromised mailbox doubles as a control credential; see [Threat model](#threat-model).
:::

- Use **App Passwords** instead of your main password (required for Gmail with 2FA)
- Set `EMAIL_ALLOWED_USERS` to restrict who can interact with the agent, and keep `require_authenticated_sender` on (do not set `EMAIL_TRUST_FROM_HEADER`)
- **Do not** set `EMAIL_ALLOW_ALL_USERS` / `GATEWAY_ALLOW_ALL_USERS` for a mailbox reachable from the internet
- The password is stored in `~/.hermes/.env` — protect this file (`chmod 600`)
- IMAP uses SSL (port 993) and SMTP uses STARTTLS (port 587) by default — connections are encrypted

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | Yes | — | Agent's email address |
| `EMAIL_PASSWORD` | Yes | — | Email password or app password |
| `EMAIL_IMAP_HOST` | Yes | — | IMAP server host (e.g., `imap.gmail.com`) |
| `EMAIL_SMTP_HOST` | Yes* | — | SMTP server host (e.g., `smtp.gmail.com`). *Not required when `platforms.email.extra.read_only: true`. |
| `EMAIL_IMAP_PORT` | No | `993` | IMAP server port |
| `EMAIL_SMTP_PORT` | No | `587` | SMTP server port |
| `EMAIL_POLL_INTERVAL` | No | `15` | Seconds between inbox checks |
| `EMAIL_ALLOWED_USERS` | No | — | Comma-separated allowed sender addresses — the operative Email allowlist |
| `EMAIL_HOME_ADDRESS` | No | — | Default delivery target for cron jobs (not used while inbound-only mode is on) |
| `EMAIL_ALLOW_ALL_USERS` | No | `false` | Accept **any** sender — open access, not recommended |
| `EMAIL_AUTHSERV_ID` | No | — | Pin the trusted `Authentication-Results` server (e.g. `mx.google.com`); mirror of `platforms.email.extra.authserv_id` |
| `EMAIL_TRUST_FROM_HEADER` | No | `false` | Skip sender mail-authentication entirely — trust the `From:` header unauthenticated (spoofable, not recommended); mirror of `platforms.email.extra.require_authenticated_sender: false` |
| `EMAIL_READ_ONLY` | No | — | **Not honored.** Inbound-only mode is config.yaml-only: `platforms.email.extra.read_only: true` |

The access-control and outbound-safety switches below are configured in `config.yaml` under `platforms.email.extra`: `read_only`, `require_authenticated_sender`, `authserv_id`, `skip_attachments`, `unauthorized_dm_behavior`.