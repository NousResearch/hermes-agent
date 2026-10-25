---
title: "Running Hermes behind an external agent broker"
description: "Keep service credentials and raw data out of the agent: delegate privileged work to a broker that holds them and returns schema-bounded results"
sidebar_position: 3
---

# Running Hermes behind an external agent broker

The [egress proxy](./iron-proxy) keeps API keys out of the terminal sandbox, and
[network isolation](./network-isolation) limits where the sandbox can connect.
Both still let the agent itself use the services those keys open: if Hermes can
read your mailbox through a tool, a prompt-injected Hermes can read your
mailbox too.

An **external agent broker** goes one step further. Hermes holds no credential
for the sensitive service at all, and no direct tool for it. A separate
process, the broker, holds the credentials and runs a fixed, operator-written
job ("summarise today's unread mail"). Hermes may ask for that job and gets
back a small, structured result. It never gets the credential or the raw data.

This page describes the pattern and what Hermes provides for it. It is not tied
to one product; any broker that does the four things below fits.

## When to use it

- Hermes reads untrusted input (the web, inbound email, group chats) **and**
  should be able to act on something sensitive (mail, calendar, a CRM, a
  brokerage account, internal APIs).
- You want writes ("reply to that mail", "move the meeting") to need a
  person's yes, given somewhere Hermes cannot reach.
- You want an audit trail of every privileged action outside the agent's own
  logs.

If Hermes only needs model-provider keys, the [egress proxy](./iron-proxy) and
the [secret sources](../secrets/) are enough.

## The pattern

```
 you ──chat──▶ Hermes ──MCP (agent key)──▶ broker ──credentials──▶ mail, calendar, CRM …
                 ▲                           │
                 └── schema-bounded result ◀─┘
                                             │
               your phone / another device ◀─┘  approves new access and every write
```

1. **The agent's key is in a 0600 file, read by a bridge command.** Hermes
   talks to the broker as an MCP server. The `mcp_servers` entry in
   `config.yaml` names only a command and its arguments; the command (the
   bridge) reads the agent's broker key from a file owned by the Hermes user
   with mode `0600`, and refuses the file if it is a symlink, belongs to
   another user, or is readable by anyone else. `config.yaml`, which people
   copy, back up and paste into bug reports, never holds the key. This also
   fits Hermes's [MCP credential handling](../security#mcp-credential-handling):
   MCP subprocesses get a filtered environment, so a bridge that reads its own
   file does not depend on a key leaking through the environment.

   ```yaml
   mcp_servers:
     my-broker:
       command: "/usr/local/bin/broker-cli"       # the broker's bridge
       args: ["mcp-bridge", "--namespace", "home"] # no key here
   ```

   The key identifies Hermes to the broker and is worth only what the broker
   grants it: a fixed list of jobs, a budget, and nothing else. Hermes can read
   its own key; that is expected. What matters is that Hermes cannot read the
   **broker's** keys (next point).

2. **Credentials are entered and injected outside the agent.** You give the
   broker your mail or CRM credentials in the broker's own interface (on
   another device, or through the broker's CLI as its own OS user), never in
   the Hermes chat. The broker injects them into the work it runs. Run the
   broker as a **different OS user** (or on another machine) from Hermes:
   Hermes's terminal tool can read anything its own user can, so a broker that
   shares Hermes's user shares its credential store with the agent. A broker
   that implements this pattern should refuse that setup unless you accept it
   explicitly.

3. **Results come back through approved egress schemas.** Hermes asks for a
   job by name with **typed inputs** (a date, an enum of folders), not with
   free-text instructions, so a steered Hermes cannot talk the privileged side
   into doing something else. The job's answer is one JSON document validated
   against a schema the operator approved, with bounded fields ("at most 30
   items, one line of at most 200 characters each, sender domain only").
   Everything else the job produced, including intermediate files and raw
   error messages, stays with the broker. An answer that is too large or does
   not match the schema is refused, not truncated. Free-text fields that
   derive from third-party content should come back marked as untrusted.

   The same rule covers what else a job hands back:

   - **Files.** A job that renders a report or a PDF returns only the files
     the operator declared for that job. Text comes back bounded and marked
     untrusted; a binary file comes back as a short-lived download link for
     the person, not as bytes in the model's context. The broker re-checks
     the agent's grant on every read, so revoking access also revokes links
     already handed out.
   - **Accounts.** When the person has two accounts at one provider (work
     and personal mail), each is its own connection with its own label and
     credential. The agent asks the person which one instead of guessing,
     and the result names the account it came from; an answer from one
     account is never reused for another.
   - **Progress and reuse.** While a job runs, the agent can see where it is
     ("step 2 of 3") without reading the job's own text. For a follow-up
     about something a read-only job fetched minutes ago, the agent may ask
     for that earlier answer instead of a new run, within a bound it states
     per call; a job that writes, or whose reach changed since, always runs
     again.

4. **Approval happens on another device.** Anything that widens what the
   agent can reach (a new connection, a new credential, a new job, a higher
   budget) needs a person's yes on a device that is paired with the broker,
   such as a phone. So does every write action, either one by one or within
   standing limits the person set on that device (say, "replies to this one
   address"), unless the operator allowed a narrow class of writes to run
   unattended (see [Limits](#limits)). Hermes can see that something is waiting for approval; no tool
   it holds can approve it, and no tool it holds can widen those limits.

### What the person can see and undo

The broker is also where the person, not the agent, looks back:

- **What the broker remembers.** If the broker keeps memory for Hermes (a
  long-term memory provider, or what its jobs learned), the person can list,
  export and forget it from their own device, without the content passing
  through the model. Forgetting stops recall; a broker that keeps an audit
  record should say how long it keeps it.
- **What is running.** Hermes can cancel a job it started, including one
  that is waiting for input or for another job. A job waiting for the
  person's approval is answered (approve or reject), not cancelled from under
  them.
- **What happened.** Every job, approval and write is recorded by the
  broker, outside Hermes's own logs.

### Hermes's own command approvals on the same device

Hermes's [dangerous-command approvals](../security#dangerous-command-approval)
are separate from the broker's: they guard what Hermes does on its own machine.
A plugin can route them to the same device through an
[approval transport](../features/plugins#approval-transports): register it,
then select it in `config.yaml`:

```yaml
security:
  approval:
    transport: my-broker            # the transport's registered name
    # transport_fallback: builtin   # optional: fall back to the built-in prompt
```

A selected transport replaces the built-in prompt surfaces. If the transport
fails or the broker is unreachable, Hermes denies the command, unless you set
`transport_fallback: builtin`. Hermes keeps detection, the allowed scopes, the
timeout and the final decision; the transport only presents the request and
returns the answer. A modified Hermes, or one with the transport deselected,
does not ask the device, so the device is a surface and a record, not the
enforcement point.

## Limits

- **Unattended runs do not reach another device.** Cron jobs, single-query
  runs (`hermes chat -q`), and sessions on unattended platforms (webhook, or
  `api_server` with no client that can answer) take Hermes's unattended path
  (`approvals.cron_mode`, `approvals.single_query_mode`,
  `approvals.unattended_mode`) and never call an approval transport. With the
  default `deny`, a dangerous command in such a run is refused at once;
  nobody is asked. `hermes -z` asks nobody either: it runs in YOLO mode.
  Likewise, an unattended run cannot wait for a person to approve a broker
  write: plan unattended jobs as read-only, and expect anything that needs a
  yes to stay pending or be refused, not to happen.
- **Writes an operator lets run unattended are not approved one by one.** A
  broker may let the operator allow a narrow class of writes without a
  person's yes (labelling issues from an allowlist, say). That is the
  operator's decision, made in the broker's own configuration and out of the
  agent's reach; it should be off by default and named as unattended.
- **A reused answer can be stale.** Reusing a recent read saves a run but
  answers from the time of that run. Ask for a fresh read after changing a
  connection or when "now" matters.
- **A summary of sensitive data is still sensitive.** The broker guarantees
  that Hermes never held the credential and never saw content outside the
  approved schema. It does not make the result harmless; the result goes to
  Hermes's model provider like the rest of the conversation.
- **Prompt injection is reduced, not prevented.** An attacker who controls a
  mail can still phrase an instruction as a plausible one-line summary.
  Bounded fields, untrusted-content markers and the write approval limit what
  that can achieve; a read-only job cannot be made to act.
- **Your conversation is not protected by the broker.** Whatever you type to
  Hermes goes to Hermes's model provider. The broker protects the systems
  behind it, not the chat.
- **The schemas and jobs are operator-written and trusted.** A job whose
  instructions say "include the full body" defeats the schema up to its field
  bounds; that is why the bounds matter.

## Example: Vornik

[Vornik](https://docs.vornik.io/guides/hermes-setup/) is one self-hosted broker
that implements this pattern, with a Hermes plugin in the
[plugin catalog](/plugins/vornik-companion). It runs operator-approved
workflows with typed inputs and schema-validated results, keeps credentials in
its own store (one labelled connection per account), approves new access and
writes on a paired phone (each write, or within limits set there), and offers
an approval transport for Hermes's own command prompts. The plugin also
reports a running job's step, reuses a recent read-only answer on request,
returns only the files a workflow declares, and can serve as Hermes's
long-term memory, which the person lists, exports and forgets from the paired
phone. Issue triage on GitHub runs with the GitHub App's credential kept in
the broker; its label and close writes are off by default and, when enabled,
are one of the operator-allowed unattended writes described above. Its setup
guide covers running it as a separate OS user next to Hermes.

Other brokers that implement the same four properties are used the same way:
an MCP entry whose command reads a 0600 key file, and nothing sensitive in
Hermes's config or environment.

## See also

- [Egress credential-injection proxy (iron-proxy)](./iron-proxy)
- [Network egress isolation (Docker)](./network-isolation)
- [Secrets](../secrets/)
- [Security: MCP credential handling](../security#mcp-credential-handling)
- [Plugins: approval transports](../features/plugins#approval-transports)
