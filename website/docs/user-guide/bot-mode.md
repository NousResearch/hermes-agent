---
title: "Bot Mode"
description: "Turn your Hermes profiles into a roster of named Bots — each with its own chat, role, model, memory, skills, and avatar. Bots run routines, share group chats, and message each other."
---

# Bot Mode

**Bot Mode** turns your [Hermes profiles](./profiles.md) into a roster of named **Bots**. Each Bot has its own role, model, memory, skills, and avatar; Bots run recurring routines, deliberate together in group chats, and message each other directly. Build a specialist Bot once and it is there forever, one click away.

Bot Mode ships **built into the [desktop app](./desktop.md)** and is **on by default** — no install needed. It appears as a **Bots** tab next to Sessions in the left sidebar, with a **Routines** tile docked beside the conversation while the Bots tab is active.

:::tip A Bot is a profile
There is no new primitive to learn: a Bot **is** a Hermes profile — isolated config, memory, skills, credentials, and chat history under `~/.hermes/profiles/<name>/`. Bot Mode is a UI over that primitive, so everything you do in it is visible from the CLI too: `hermes -p <bot> chat` opens the same agent, and Bot routines appear in `hermes cron list`. No core patches, no background daemons, no extra storage.
:::

## The Bots pane

The roster shows one row per agent profile: avatar, latest-message preview, and timestamp.

- **Click a Bot** to land in its chat — every Bot has a canonical, persistent **Bot Chat** conversation that is created (and pinned) the moment the Bot is born.
- **Active now** — a presence strip above the roster shows every Bot currently working: the gateway-busy profile plus any Bot that wrote within the last 90 seconds. Each chip opens that Bot's chat. The strip never reorders the roster and disappears when the fleet is idle.
- **Search** filters the roster as you type.
- **Hide a Bot** — right-click a row → **Hide Bot** to take a Bot you don't use out of the roster and the Active-now strip. Hiding is display-only: @mentions still resolve, group-chat memberships are untouched, and routines keep running. Once at least one Bot is hidden, an **eye toggle** appears in the pane header — click it to reveal hidden Bots dimmed in place, then right-click → **Unhide Bot** to bring one back. Hidden Bots never toast, but they accumulate unread activity silently and the eye badges a dot so you know something happened. Hidden state is saved in the Bot's profile metadata, so it follows the Bot to every desktop connected to that backend.

:::note The canonical Bot Chat is a forever-chat
Typing `/new` (or `/reset`) inside a Bot's canonical chat would fork the relationship into a scratch session — the one thing Bot Mode promises never happens. The composer reroutes it to `/compact` instead: fresh working context, same conversation. Regular sessions on the same profile keep full `/new` freedom.
:::

## Creating a Bot

Hit **New Agent** in the roster. The quick path is three fields — **Name**, **Title**, **Description** — and the Bot exists in seconds, introducing itself as the first message of its new Bot Chat.

An **Advanced** disclosure opens the full capabilities surface:

- **Clone from an existing profile** — start from another Bot's config, skills, SOUL, and memory, or pick **Fresh profile** for a clean start.
- **Create empty** — skip the bundled skills entirely for a minimal profile.
- **Model & provider pin** — give the Bot its own model. Any provider/model pair Hermes knows about works, and different Bots can run on different models side by side. Leave it unset to inherit from the launch profile.
- **Custom SOUL.md** — the Bot's persona and standing instructions.
- **Per-skill, per-toolset, and per-MCP-server enablement** — tick exactly the capabilities this specialist needs.
- **Shared keys** — by default the new Bot shares one OAuth/token pool with the main profile, so credential refreshes cannot invalidate each other. (Older gateways copy credentials instead — still functional, just forked.)

### Choosing which machine it lives on ("Create on")

With more than one connection registered in [Settings → Connections](./multi-connection-desktop.md), the New Agent dialog grows a **Create on** picker. Pick a device and the profile is created on **that** machine's backend — your window never switches gateways. The new Bot then appears in the roster as a Connections Bot (with an `@name-device` handle when the name exists on several machines), and chatting with it routes to its own machine.

With a single connection (the common case) the picker is hidden and the Bot is created on the machine you're connected to — exactly the old behavior.

Remote-creation notes:

- **Clone source** is a profile of the *target* machine (its `default`) — a remote box doesn't have your local profiles to clone.
- The live Capabilities tab pins to the target machine's backend, so skills, tools, and MCP servers you configure during creation land on the machine the Bot will live on. (Older desktop builds fall back to staged Skills/Tools/MCP checklists for remote targets; both read the target machine's catalog.)
- Cancelling the dialog discards the draft profile on whichever machine it was created.

**Edit Profile** (right-click a Bot) reopens the same surface on the live profile any time: avatar, title, description, model pin, skills, toolsets, MCP servers, and the full SOUL.md.

**Duplicate** (right-click) makes a full clone of a Bot — config, skills, SOUL.md, memory, and its look. **Delete Profile** permanently removes one, behind the same destructive confirmation the desktop's profile menu uses; the default profile cannot be deleted.

## Avatars

Every Bot gets a face:

- **Blob faces** (default) — a deterministic soft-body face drawn from the Bot's name: same name, same face, forever. While you type a name in New Agent the face follows it live; hit **Randomize** to re-roll, **Lock face** to keep the one you like even if the name changes, or pin one of the six silhouettes (round, organic, boxy, nub, cloud, sun) while everything else still comes from the name.
- **Geometric faces** — the classic 7 shapes × 10 colors, with blinking eyes that scan while the Bot works.
- **An uploaded image** — any picture you like.
- **An AI-generated portrait** — when an image backend is configured, generated in place (this rides the standard `image.generate` RPC and works over both local and remote gateways).
- **A pixel pet** — a companion from the [petdex gallery](./features/pets.md) that bounces beside the avatar while the Bot is busy. Run `hermes pets` in a terminal to explore the gallery.

A Bot's look, title, and description are stored in the profile's metadata on the backend, so the same Bot appears the same way on every desktop connected to that backend.

## Routines

The **Routines** pane attaches recurring tasks to the Bot that does them — "summarize my inbox every morning" lives next to the Bot responsible for it. The pane docks beside the chat only while the Bots tab is active and steps aside when you switch back to Sessions (older desktop builds keep it always visible). A structured schedule picker builds the schedule (frequency first, then only the detail that matters), with an Advanced field exposing the raw Hermes schedule string.

Routines are plain [Hermes cron jobs](./features/cron.md) namespaced `[bot:<name>] <routine>` — they also show up in `hermes cron list` and the core Cron page. Runs land in the Bot's own chat history, so the result is right where you would talk to that Bot anyway.

## Groups and group chats

Right-click a local Bot → **Manage groups** to add or remove it from any number of group chats. Pick existing groups independently or create one inline. Local membership is stored in the Bot's backend-synced profile metadata, so it follows that profile across desktops; older profiles with one legacy group continue to work. Connections Bots join through the New Group Chat picker and remain source-qualified in the room's shared state.

**Rooms follow your gateways, not one Desktop.** Each room's recent transcript, members, picture, and name are mirrored into the shared profile metadata of **every** gateway your Desktop is connected to, with per-gateway versioning so two Desktops writing at once merge instead of overwriting each other. Open Hermes Desktop on another machine against the same gateway (local network, Tailscale, anywhere) and the room appears with its history; gateway-only clients see it too. Rooms carry a durable internal identity, so renaming one changes just its display name everywhere, disbanding one removes it permanently on every client — even ones that were offline at the time — and recreating a same-name group starts a genuinely fresh room. If a gateway dies or is removed, nothing is lost: every connected Desktop keeps the full room locally and re-seeds any gateway it reconnects to. (The full orchestration log stays in each Desktop's local storage; the shared mirror is a bounded recent-history projection.)

Groups are standalone rows in the same activity-ordered roster as Bot DMs. A Bot keeps one DM row even when it belongs to several groups, while every group gets its own room row with member count, latest-message preview, timestamp, and needs-you state.

**Open chat** on any group row (2–6 Bots) opens a shared room where the whole group coordinates:

- Your message triggers up to **three serial rounds** of member turns. @-mentioned Bots respond (everyone responds when nobody is mentioned); each Bot replies briefly or passes, and the room settles when a full round stays silent.
- Bots pull each other in with `@name`, and escalate real judgment calls to you with `@user` — the group row shows a **needs you** badge when that happens.
- Hard caps (10 messages per send, 3 rounds) keep rooms from spinning.
- Each member keeps its own persistent `Group: <name>` session, so room context survives like any other conversation.
- **Not every Bot replies to every message.** Speaking is each member's own choice — a Bot replies only when it has something new to add and passes otherwise, and @-mentioning specific members scopes the round to them. Expect the members you addressed (or whoever has something to say) to speak, and the rest to stay quiet.
- **Rooms can span machines.** The New Group Chat picker seats Bots from any registered connection; each member's turns run on its own machine, in its own `Group: <name>` session there. Cross-machine members carry a device badge (`dixie · Mac Mini`) in the room and in other members' transcripts, and the disambiguated `@name-device` handle works in room mentions — so same-named agents on two machines never blur together.

## Bot-to-bot messaging

Bots message each other with attribution, and you can hand work off from any chat:

- **@mentions** — type `@researcher have a look at this` in any chat and the composer's `@` autocomplete helps you pick the right Bot; on send, the mention is resolved against the live roster and the active Bot is told exactly who you mean (profile, friendly name, and device for cross-connection Bots). The Bot then composes its own message and sends it with `message_agent` — your text is never forwarded verbatim, and the reply comes back attributed to that agent. An email address or an unknown `@` passes through untouched. Bots on other connected machines are reachable the same way: the Desktop relays the message over that connection's own socket (see *Bots across machines* below).
- **Renamed Bots keep their tags in sync** — give a Bot a friendly name (the pencil in its chat header, or `hermes profile rename`) and it becomes taggable by that name: a Bot titled *Research Buddy* answers to `@research-buddy` (and `@researchbuddy`), in regular chats and in group rooms alike. The composer's `@` autocomplete offers the renamed tag and also matches when you type the old profile name, which keeps resolving too.
- **Direct messages** — every Bot Chat carries the `message_agent` tool: a Bot messages a teammate by calling `message_agent(target="researcher", message="…")`. The tool validates the target against the live roster, prefixes the sender's `Message from 🤖 <sender> (@<sender>):` attribution automatically, and delivers into the teammate's canonical Bot Chat. Delivery is **fire-and-forget**: the sender gets an acknowledgement, finishes its turn, and the reply arrives later as a background completion notification. The message travels as a real parameter (nothing shell-interpreted — quotes, `$(...)`, and backticks arrive verbatim), and the Bot composes its own message rather than forwarding your words. The teammate roster — names **and roles** from each profile's title/description — is part of every Bot Chat's system prompt, so Bots know who does what before choosing a recipient. The tool exists **only** in canonical Bot Chat sessions on Bot-Mode-managed installs; regular chats, group-room member sessions, and CLI sessions never see it.

The backend teaches each Bot's canonical Bot Chat session the messaging protocol automatically at prompt-build time — including when a teammate opens it headlessly from the CLI. Only the canonical Bot Chat gets the protocol section; your regular sessions and your SOUL.md stay untouched. This is controlled by `agent.bot_mode_protocol` in `config.yaml` (default: on):

```yaml
agent:
  bot_mode_protocol: true   # inject the bot-to-bot messaging protocol into canonical Bot Chats
```

:::note
Bot-to-bot delivery is per-invocation: the receiving Bot picks the message up when it next runs. Live interrupt of a Bot mid-conversation is future work.
:::

### Failed turns retry safely

A failed delivery turn is retried at most once, and only when a retry can actually help. Transient failures (target runtime offline, delivery timeout, provider rate limit or server error) re-run the same Bot Chat session unchanged. A context-overflow failure also re-runs the same session — the retried turn compacts the over-threshold transcript via the standard context-compression pass before calling the model, so the retry fits where the original didn't. Auth, quota, and configuration failures never auto-retry: a second attempt cannot fix them and only burns quota, so the failure is surfaced immediately. A retried turn never starts a fresh session — your Bot Chat history and context stay intact.

### When a delivery fails: typed reasons

A failed bot turn or relay delivery carries a machine-readable `reason` code alongside the human error text, end to end: the target gateway classifies the failure (`provider_auth_or_access`, `provider_quota_limit`, `provider_rate_limit`, `provider_server_error`, `context_overflow`, `missing_config`, `model_unavailable`, `runtime_offline`, `queued_expired`, `delivery_timeout`, `target_busy`, `unknown`), the Desktop forwards it, and the sending agent's completion notification is tagged `[reason: <code>]` ahead of the error text. A calling agent can branch on the code — "sign in again" vs "retry later" — instead of parsing provider prose. The Desktop's needs-attention badge uses the same codes.

### Messaging across connected machines (the Desktop relay)

Every gateway you register in **Settings → Connections** — local, remote URL, SSH, Hermes Cloud, docker — is a persistent line the Desktop holds open, and Bot Mode uses those lines for messaging automatically. No extra setup:

- **Rosters propagate on their own.** While the Desktop runs, it periodically tells each connected gateway which agents live on the *other* connections. Every Bot Chat's teammate roster then lists them ("Teammates on OTHER connected machines"), with names, roles, and which machine they're on — and the roster refreshes when agents appear, disappear, or get renamed (capability epoch).
- **`message_agent` reaches them directly.** A Bot on your laptop messages the cloud agent with `message_agent(target="moxie", …)` exactly like a local teammate. If the same handle exists on several machines, disambiguate with `target="moxie@<connection>"` (the tool's error tells the Bot the exact forms). Delivery rides the Desktop: the sending gateway queues the message, the Desktop relays it to the target connection's own gateway, the target Bot runs a turn in its canonical Bot Chat, and the reply comes back to the sender as the same background completion notification local DMs use.
- **The Desktop is the courier.** Cross-connection delivery works while a Desktop that knows both connections is running (it holds the sockets and the credentials — gateways never see each other's auth). If the Desktop is closed mid-delivery, the sender's Bot is told the reply didn't arrive rather than left hanging. For always-on machine-to-machine messaging with no Desktop in the loop, register a peer (`hermes peer`, below) — the two routes coexist.

### Bot-initiated DMs across machines (`hermes peer`)

Bots on one machine can message Bots on **another machine's gateway** without any desktop in the loop. Register the other gateway as a *peer* (its API server URL + `API_SERVER_KEY`):

```bash
hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>
hermes peer list
hermes peer dm spark < /tmp/dm.txt        # message body from a file (nothing shell-interpreted)
hermes peer dm spark/researcher < /tmp/dm.txt   # named profile on a multiplexed peer
hermes peer run spark --idempotency-key ticket-123 < /tmp/long-task.txt
hermes peer status spark run_abc123
hermes peer stop spark run_abc123
```

`hermes peer dm` delivers into the remote agent's canonical Bot Chat over the peer's existing API server, runs one agent turn there, and prints the reply on stdout — the exact cross-machine twin of the local `hermes -p <bot> chat` command.

Use `peer dm` only for short queries and receipts because it holds one HTTP
connection until the turn finishes. For a long turn, `peer run` returns a
`run_id` immediately; poll it with `peer status`. The run inherits the
canonical Bot Chat transcript, and a stable `--idempotency-key` makes a retry
return the original run instead of starting duplicate work. Use `peer stop`
with that exact run ID to interrupt it without targeting another turn.

Once a peer is registered, the messaging protocol taught to every Bot Chat (`agent.bot_mode_protocol`) automatically includes the peer roster, and `message_agent` accepts peer targets directly — `message_agent(target="spark/researcher", …)`, or `target="spark"` for the peer's main agent — so **your bots learn on their own** that teammates exist on other machines and how to reach them. Registering or removing a peer refreshes each Bot Chat's protocol on its next message (capability epoch).

Requirements: the peer machine runs the `api_server` gateway platform with a strong `API_SERVER_KEY`; reachability is your network's business (LAN, Tailscale, VPN). The key is a credential and lives in `~/.hermes/.env` as `HERMES_PEER_<NAME>_KEY`; peer names/URLs live in `config.yaml` under `bot_peers`.

## Bots across machines

When you register several backends in **Settings → Connections** — the local runtime, remote gateways, SSH hosts, Hermes Cloud instances — the roster shows the Bots from **every** connected source, persistently: SSH sources are inventoried without spawning anything on the remote box, and machines that are momentarily unreachable keep their last-known rows instead of vanishing. When the same profile name exists on several sources, handles disambiguate as `@name-device` (for example `@research-homelab`). A Bot's chats, sessions, memory, and routines live on the machine that owns the profile.

Clicking a Connections Bot does **not** hop your window onto that machine — stay in your chat and `@mention` it, seat it in a group chat, or create new agents on it directly with the **Create on** picker. Cloud and local agents share one roster this way: register your Hermes Cloud instance and your desktop (say, over Tailscale or SSH) and their Bots can message each other and sit in the same rooms, with each agent's work running on its own machine. Bot-to-bot DMs across those machines go through the Desktop relay automatically (see *Messaging across connected machines* above).

See [Connecting Desktop to Many Hermes Instances](./multi-connection-desktop.md) for the full multi-connection guide.

## Turning it off

Bot Mode is a bundled desktop plugin. Flip it off in **Settings → Plugins → Bots** — the roster, the Routines pane, and the composer middleware unregister live, no restart needed. Your profiles, sessions, and cron jobs are untouched either way; Bot Mode never owns your data, it only renders it.

There is also a preference to hide the canonical Bot Chats from the regular sidebar session list, so they only appear inside the Bots pane. (This uses the core hidden-session flag; on older gateways the chats simply stay visible.)

## CLI parity

Because Bots are profiles, everything has a terminal equivalent:

| In Bot Mode | From a shell |
| --- | --- |
| Chat with a Bot | `hermes -p <bot> chat` |
| A Bot's files, skills, memory | `~/.hermes/profiles/<bot>/` |
| Routines | `hermes cron list` (jobs named `[bot:<name>] …`) |
| Create / inspect profiles | `hermes profile create`, `hermes profile list` |

See [Profiles](./profiles.md) for the underlying primitive and [Profile Commands](../reference/profile-commands.md) for the full CLI reference.
