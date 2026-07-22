---
title: "Delta Chat"
description: "Connect Hermes to Delta Chat with a local JSON-RPC server, Chatmail account setup, secure invites, access control, groups, attachments, and cron delivery"
---

# Delta Chat

[Delta Chat](https://delta.chat/en/help) is an end-to-end encrypted messenger
built on interoperable email and Chatmail relays. Hermes runs the local
[`deltachat-rpc-server`](https://pypi.org/project/deltachat-rpc-server/) as a
child process; Delta Chat Core owns the network connection, account credentials,
encryption keys, contacts, and attachment storage.

The recommended setup creates a new Chatmail identity. You do not need to find
IMAP/SMTP settings or put a mailbox password in Hermes' `.env` file.

## First-time setup

### 1. Install the RPC server

Install it in the same Python environment as Hermes:

```bash
python -m pip install deltachat-rpc-server
deltachat-rpc-server --version
```

The package supplies prebuilt binaries for common Linux, macOS, and Windows
targets. If you install a standalone binary instead, put it on `PATH` or give
its path to the setup wizard.

### 2. Create the bot account

```bash
hermes gateway setup
```

Choose **Delta Chat**. The wizard will:

1. locate `deltachat-rpc-server`;
2. create a profile-scoped account store;
3. create a Chatmail account on the relay you select;
4. set the bot display name;
5. save the generated address and resolved binary path in `config.yaml`; and
6. print a secure invite link and, in a compatible terminal, a QR code.

The default account store is
`<HERMES_HOME>/platforms/deltachat`. It contains mailbox credentials and private
encryption keys, so back it up securely and do not share it. The wizard makes
the directory owner-only where the operating system supports Unix permissions.

### 3. Add the bot in Delta Chat

In the Delta Chat app, choose **+ → Scan QR Code** and scan the QR printed by
the wizard. For a remote setup, copy the invite link through a private channel
and open it on the device running Delta Chat.

Use the secure invite, not just the generated email address. The invite carries
the identity material Delta Chat needs for a verified end-to-end encrypted
contact. Keep the gateway online while the secure-join exchange completes.

### 4. Start the gateway and approve yourself

```bash
hermes gateway start
```

Send the bot a direct message. With the safe default `dm_policy: pairing`, it
returns a one-time code. Approve that sender on the Hermes host:

```bash
hermes pairing approve deltachat <CODE>
```

You can now chat with Hermes normally, send voice notes or attachments, and use
the shared gateway slash commands.

## Configuration

The wizard writes the important values for you. A complete manual configuration
looks like this:

```yaml
platforms:
  deltachat:
    enabled: true
    typing_indicator: false

    # Direct messages: pairing | allowlist | disabled
    dm_policy: pairing
    allow_from: []

    # Groups are opt-in: allowlist | open | disabled
    group_policy: disabled
    group_allow_from: []
    require_mention: true
    free_response_channels: []

    # Optional default for /sethome, cron, and notifications.
    home_channel:
      platform: deltachat
      chat_id: "42"
      name: "Delta Chat notifications"

    extra:
      email: "hermes@example.chat"
      display_name: "Hermes Agent"
      data_dir: "/absolute/path/to/deltachat-account-store"
      rpc_server_path: "/absolute/path/to/deltachat-rpc-server"
      avatar_image: "/absolute/path/to/avatar.png"
      show_invite_link: true
      join_invite_link: ""
```

There are deliberately no user-facing `DELTACHAT_*` environment variables.
Behavioral settings belong in `config.yaml`; the Delta Chat account database is
the credential store.

### Use an existing account store

An email address alone is not enough to reconstruct a Delta Chat account. To
reuse an existing bot identity, set `extra.email` to the configured account's
full address and set `extra.data_dir` to the `DC_ACCOUNTS_PATH` directory that
contains that account. The adapter selects the matching configured account and
will not ask Hermes to store its mailbox password.

For a scriptable two-step bootstrap, `extra.email` may temporarily contain a
relay marker such as `@nine.testrun.org`. On first start Hermes creates the
account and stops with the generated full address. Replace the marker with that
address before restarting. The setup wizard is easier because it performs both
steps and updates the YAML automatically.

## Access control

Hermes defaults to pairing for direct messages and disables groups. To skip
pairing for known senders, use an email allowlist:

```yaml
platforms:
  deltachat:
    dm_policy: allowlist
    allow_from:
      - alice@example.org
      - bob@example.net
```

Email comparisons are case-insensitive. `dm_policy: open` still requires the
gateway's explicit global allow-all security opt-in and is not recommended for
an internet-reachable bot.

## Groups and mentions

Enable groups explicitly and list the members allowed to invoke Hermes:

```yaml
platforms:
  deltachat:
    group_policy: allowlist
    group_allow_from:
      - alice@example.org
    require_mention: true
```

With `require_mention: true`, a group message must contain either the configured
display name (`Hermes Agent`) or the account's local-part (`@hermes`). Add a
numeric Delta Chat chat ID to `free_response_channels` if one trusted group
should not require mentions.

## Attachments and outbound delivery

The adapter receives images, video, voice notes, and arbitrary files through
Hermes' media cache. Outbound images, documents, video, animations, and voice
notes are sent as native Delta Chat attachments. Text replies can quote the
incoming message.

Use a numeric chat ID for unambiguous automation targets:

```bash
hermes send deltachat:42 "The report is ready."
```

The adapter can also resolve a configured contact email, a unique contact name,
or a unique chat name. For unattended cron delivery, set the home channel with
`/sethome` from the desired chat or configure `home_channel` as shown above,
then use `deliver: deltachat`.

## Operations and troubleshooting

Check the adapter and watch its logs with:

```bash
hermes gateway status
hermes logs --follow
```

- **`deltachat-rpc-server was not found`** — install it in the Hermes
  environment, rerun setup, or set `extra.rpc_server_path` to the executable.
- **Account not configured in the data directory** — the email and `data_dir`
  refer to different account stores. Point to the original store or rerun the
  wizard to create a new identity.
- **The contact cannot message the bot** — add it with the secure invite and
  leave the gateway online until secure join finishes. Then approve its pairing
  code or add its address to `allow_from`.
- **Group messages are ignored** — groups default to disabled. Enable an
  explicit group policy, allow the sender, and mention the bot unless that chat
  is listed in `free_response_channels`.
- **Two gateways fight over one account** — Hermes locks the account store per
  profile. Give each concurrently running profile its own `data_dir` and Delta
  Chat identity.
- **Service startup cannot find a newly installed binary** — rerun
  `hermes gateway setup` so the absolute binary path is saved, then reinstall
  the gateway service if its captured `PATH` changed.

