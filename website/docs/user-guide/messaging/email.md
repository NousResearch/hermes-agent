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
```

---

## Step 2: Start the Gateway

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

On startup, the adapter:
1. Tests IMAP and SMTP connections
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
- **Automated/noreply senders** are silently ignored — `noreply@`, `mailer-daemon@`, `bounce@`, `no-reply@`, and emails with `Auto-Submitted`, `Precedence: bulk`, or `List-Unsubscribe` headers

### Sending Replies

Replies are sent via SMTP with proper email threading:

- **In-Reply-To** and **References** headers maintain the thread
- **Subject line** preserved with `Re:` prefix (no double `Re: Re:`)
- **Message-ID** generated with the agent's domain
- Responses are sent as plain text (UTF-8)

### File Attachments

The agent can send file attachments in replies. Include `MEDIA:/path/to/file` in the response and the file is attached to the outgoing email.

### Skipping Attachments

To ignore all incoming attachments (for malware protection or bandwidth savings), add to your `config.yaml`:

```yaml
platforms:
  email:
    skip_attachments: true
```

When enabled, attachment and inline parts are skipped before payload decoding. The email body text is still processed normally.

---

## Access Control

Email access is stricter by default than chat-style platforms:

1. **`EMAIL_ALLOWED_USERS` set** → only emails from those addresses are processed
2. **No allowlist set** → unknown senders are ignored silently
3. **`EMAIL_ALLOW_ALL_USERS=true`** → any sender is accepted (use with caution)
4. **`platforms.email.unauthorized_dm_behavior: pair`** → unknown senders receive a pairing code

:::warning
**Use a dedicated inbox and configure `EMAIL_ALLOWED_USERS` for normal operation.** Email pairing is opt-in because shared inboxes often contain unrelated unread messages, and Hermes should not reply to those contacts by default.
:::

---

## Review-First Outbound Policy

By default every accepted sender gets an automatic reply. For a business
mailbox you can flip the adapter into **review-first** mode: only
authenticated mail from an explicit allowlist is answered automatically —
every other proposed reply is stored in the IMAP **Drafts** folder for you
to review and send yourself, and it never touches SMTP.

```yaml
platforms:
  email:
    outbound_policy: review_first    # absent/direct (default) | review_first
    auto_send_authenticated_senders:
      - operator@example.com         # exact addresses, case-insensitive
    authserv_id: example-mx.com      # REQUIRED under review_first: the
                                     # authserv-id your receiving server
                                     # stamps in Authentication-Results
    drafts_mailbox: Drafts           # where proposed replies are stored
    sent_mailbox: Sent               # archival copy of transmitted mail
    save_sent: true                  # IMAP APPEND after each SMTP send
    agent_initiated_sends: draft     # draft (default) | send — agent-composed
                                     # outbound mail
    untrusted_draft_limit_per_sender_hour: 4   # abuse limits for stranger drafts
    untrusted_draft_limit_global_hour: 20
```

These are behavior settings and live in `config.yaml` only — there are no
environment-variable equivalents (`.env` stays credentials-only).

Under `review_first`:

- **Every email-triggered model turn runs with zero callable tools** —
  authenticated allowlisted senders included. Email content can never
  trigger tools, resolver errors fail closed to zero tools, and a
  zero-tool turn is never delegated to a remote proxy agent. The legacy
  `untrusted_sender_toolsets` key is ignored with a warning.
- **Automatic sending requires all three of**: an exact (case-insensitive)
  match in `auto_send_authenticated_senders`, a passing
  `Authentication-Results` verdict (DMARC, or aligned SPF/DKIM) on the
  specific inbound message being replied to, and a configured
  `authserv_id` pin. Evaluation is pinned to that authserv-id with EXACT
  equality (pin the literal token your server stamps) and is
  position-verified: only an `Authentication-Results` header above the
  message's first transport `Received:` header — the region only the
  receiving server can write — counts, so a forged header traveling
  inside an attacker's message can never authenticate it. A missing pin,
  a header from any other authserv-id, a pinned header below the
  `Received:` chain, or ambiguous duplicates all fail closed. A matching
  `From:` header with failed or missing authentication is drafted, never
  sent.
- **Everything else drafts.** An empty or malformed allowlist fails closed:
  every reply is drafted. A missing trust record (for example after a
  gateway restart) also drafts — send authority is never inferred from the
  recipient address alone.
- **Untrusted senders are additionally segregated**: their turns run in a
  session keyed apart from the trusted sender's conversation, so forged
  mail can never poison trusted history.
- **Gateway notifications still work — for allowlisted recipients.** Cron
  deliveries and restart/startup notifications are gateway-composed and
  carry internal provenance, but they transmit only to addresses on
  `auto_send_authenticated_senders`. `EMAIL_HOME_ADDRESS` is a
  notification destination, never send authority: a home address that is
  not allowlisted receives drafts.
- **Agent-initiated mail drafts by default.** Out-of-process sends (the
  send-message tool, CLI one-shots, cron fallback) carry no gateway
  provenance and always draft unless `agent_initiated_sends: send` is
  explicitly configured — the recipient address grants nothing. The
  toggle applies ONLY to those out-of-process sends: anchor-less
  in-process sends without gateway provenance (e.g. redelivered replies
  after a restart) draft unconditionally regardless of the toggle.
- **Sent mail is archived** to `sent_mailbox` via IMAP APPEND after each
  successful SMTP send (no Bcc-to-self). An archival failure is logged and
  never causes the message to be re-sent.
- **Abuse limits** cap stranger draft generation per sender and globally
  per hour, drop duplicate inbound `Message-ID`s, and truncate oversized
  bodies — all before any model work, with clear log lines.
- **Who gets drafts**: access control runs first. With
  `EMAIL_ALLOW_ALL_USERS=true`, every non-automated sender reaches
  tool-free draft generation (subject to the abuse limits); senders
  rejected by the access-control layer (`EMAIL_ALLOWED_USERS` etc.) are
  dropped before any draft is generated.
- Unknown `outbound_policy` values fail closed into `review_first`.
  Existing installations without the key keep today's always-send
  behavior.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"IMAP connection failed"** at startup | Verify `EMAIL_IMAP_HOST` and `EMAIL_IMAP_PORT`. Ensure IMAP is enabled on the account. For Gmail, enable it in Settings → Forwarding and POP/IMAP. |
| **"SMTP connection failed"** at startup | Verify `EMAIL_SMTP_HOST` and `EMAIL_SMTP_PORT`. Check that your password is correct (use App Password for Gmail). |
| **Messages not received** | Check `EMAIL_ALLOWED_USERS` includes the sender's email. Check spam folder — some providers flag automated replies. |
| **"Authentication failed"** | For Gmail, you must use an App Password, not your regular password. Ensure 2FA is enabled first. |
| **Duplicate replies** | Ensure only one gateway instance is running. Check `hermes gateway status`. |
| **Slow response** | The default poll interval is 15 seconds. Reduce with `EMAIL_POLL_INTERVAL=5` for faster response (but more IMAP connections). |
| **Replies not threading** | The adapter uses In-Reply-To headers. Some email clients (especially web-based) may not thread correctly with automated messages. |

---

## Security

:::warning
**Use a dedicated email account.** Don't use your personal email — the agent stores the password in `.env` and has full inbox access via IMAP.
:::

- Use **App Passwords** instead of your main password (required for Gmail with 2FA)
- Set `EMAIL_ALLOWED_USERS` to restrict who can interact with the agent
- The password is stored in `~/.hermes/.env` — protect this file (`chmod 600`)
- IMAP uses SSL (port 993) and SMTP uses STARTTLS (port 587) by default — connections are encrypted

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | Yes | — | Agent's email address |
| `EMAIL_PASSWORD` | Yes | — | Email password or app password |
| `EMAIL_IMAP_HOST` | Yes | — | IMAP server host (e.g., `imap.gmail.com`) |
| `EMAIL_SMTP_HOST` | Yes | — | SMTP server host (e.g., `smtp.gmail.com`) |
| `EMAIL_IMAP_PORT` | No | `993` | IMAP server port |
| `EMAIL_SMTP_PORT` | No | `587` | SMTP server port |
| `EMAIL_POLL_INTERVAL` | No | `15` | Seconds between inbox checks |
| `EMAIL_ALLOWED_USERS` | No | — | Comma-separated allowed sender addresses |
| `EMAIL_HOME_ADDRESS` | No | — | Default delivery target for cron jobs |
| `EMAIL_ALLOW_ALL_USERS` | No | `false` | Allow all senders (not recommended) |
