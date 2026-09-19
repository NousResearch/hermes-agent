---
title: "Bot Mode"
description: "Turn your Hermes profiles into a roster of named Bots — each with its own chat, role, model, memory, skills, and avatar. Bots run routines, share group chats, and message each other."
---

# Bot Mode

**Bot Mode** turns your [Hermes profiles](./profiles.md) into a roster of named **Bots**. Each Bot has its own role, model, memory, skills, and avatar; Bots run recurring routines, deliberate together in group chats, and message each other directly. Build a specialist Bot once and it is there forever, one click away.

Bot Mode ships **built into the [desktop app](./desktop.md)** and is **on by default** — no install needed. It appears as a **Bots** tab next to Sessions in the left sidebar, with a **Routines** tile docked beside the conversation while the Bots tab is active.

:::tip A Bot is a profile
There is no new primitive to learn: a Bot **is** a Hermes profile — isolated config, memory, skills, credentials, and chat history under `~/.hermes/profiles/<name>/`. Bot Mode is a UI over that primitive, so everything you do in it is visible from the CLI too: `hermes -p <bot> chat` opens the same agent, and Bot routines appear in `hermes cron list`. No core patches, no background daemons, no extra storage.

See [Profiles, agents, and bots](./profiles.md#profiles-agents-and-bots) for how
Bot Mode relates to messaging bots and delegated subagents.
:::

## The Bots pane

The roster shows one row per agent profile: avatar, latest-message preview, and timestamp.

- **Click a Bot** to land in its chat — every Bot has a canonical, persistent **Bot Chat** conversation that is created (and pinned) the moment the Bot is born. A row click always opens that Bot Chat (the same conversation the row previews), even when you have other tabs open for the Bot; those tabs stay open beside it. In the tab strip the Bot Chat is captioned with the Bot's name, so two open Bots are told apart at a glance.
- **Right-click a Bot → Open recent session** to jump to its most recently active ordinary conversation — a cron run, a delegated job, a side thread — as a tab beside the Bot Chat. The row click itself never changes; a Bot with no other conversation yet just opens its Bot Chat.
- **Active now** — the roster's activity filter includes the owner of the focused live turn, Bots that wrote within the last 90 seconds, and Bots with a recent worker heartbeat. A connected gateway alone does not mean a Bot is working.
- **Search** filters the roster as you type.
- **Hide a Bot** — right-click a row → **Hide Bot** to take a Bot you don't use out of the roster and the Active-now strip. Hiding is display-only: @mentions still resolve, group-chat memberships are untouched, and routines keep running. Once at least one Bot is hidden, an **eye toggle** appears in the pane header — click it to reveal hidden Bots dimmed in place, then right-click → **Unhide Bot** to bring one back. Hidden Bots never toast, but they accumulate unread activity silently and the eye badges a dot so you know something happened. Hidden state is saved in the Bot's profile metadata, so it follows the Bot to every desktop connected to that backend.

:::note The canonical Bot Chat is a forever-chat
Typing `/new` (or `/reset`) inside a Bot's canonical chat would fork the relationship into a scratch session — the one thing Bot Mode promises never happens. The composer reroutes it to `/compact` instead: fresh working context, same conversation. Regular sessions on the same profile keep full `/new` freedom.

Archiving a Bot Chat from the sidebar retires it: the next click on the Bot starts a fresh conversation that becomes the new canonical Bot Chat. The retired chat stays archived and hidden — its history is preserved in the database, but it is no longer reachable from the Bot or the archive view. The automatic idle-archive sweep (`sessions.auto_archive`) never retires a Bot Chat; only an explicit archive does.
:::

### Organize bots into sections

Sections are folders you make yourself — **Clients**, **Team**, whatever fits — as a second axis beside the automatic per-gateway grouping. With no sections created the roster is the plain list it always was.

- **Create one** from the pane's **+** menu → **New section**, or right-click a Bot → **Move to section** → **New section…** (that files the Bot into it as you create it).
- **File a Bot** by dragging its row onto a section — the target highlights while you hover, and **Esc** cancels the drag — or right-click → **Move to section** and pick one. **Remove from section** puts it back in **Unassigned**.
- **Rename, reorder, or delete** a section from its heading's right-click menu (or the **⋯** that appears on hover); double-click a heading to rename. Headings fold like the gateway headings do.
- **Deleting a section never deletes Bots** — they return to **Unassigned**, and the toast offers **Undo**. No confirmation is asked.

Membership — the section's id **and name** — is stored in each Bot's profile metadata (`ui_meta`), so a Bot's section follows it to every desktop connected to that backend: a second desktop rebuilds the section headings from its members the first time it loads the roster, and a rename made on one desktop reaches the others the same way. The section list itself (order, empty sections) is kept per desktop, so deleting a section on one machine leaves an empty heading on the others until you delete it there too. Bots filed before names rode along carry only the id; the desktop that created the section stamps the name onto them automatically the next time its roster loads, and other desktops pick the section up from there. When the roster shows more than one gateway, sections nest inside each gateway's bucket.

## Creating a Bot

Hit **New Agent** in the roster. The quick path is three fields — **Name**, **Title**, **Description** — and the Bot exists in seconds, introducing itself as the first message of its new Bot Chat.

An **Advanced** disclosure opens the full capabilities surface:

- **Clone from an existing profile** — start from another Bot's config, skills, SOUL, and memory, or pick **Fresh profile** for a clean start.
- **Create empty** — skip the bundled skills entirely for a minimal profile.
- **Model & provider pin** — give the Bot its own model. Any provider/model pair Hermes knows about works, and different Bots can run on different models side by side. Leave it unset to inherit from the launch profile.
- **Custom SOUL.md** — the Bot's persona and standing instructions.
- **Per-skill, per-toolset, and per-MCP-server enablement** — tick exactly the capabilities this specialist needs.
- **Copy API keys from the main profile** — on by default. Each Bot gets its own credential store: static API keys are copied in, while single-use OAuth logins (Anthropic, OpenAI Codex, xAI) are not copied — sign the Bot in itself with `hermes -p <name> auth add <provider>`. See [Every profile owns its credentials](./profiles.md#every-profile-owns-its-credentials).

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
- **Geometric faces** — the classic 7 shapes × 10 colors. During a focused live turn, the owning Bot leans and looks upward with three animated dots, then eases back to idle when the turn ends. Ownership includes the connection, so same-named Bots on different gateways do not borrow the pose. Background workers keep their existing working animation; photos, blob faces and sigils keep their own rendering.
- **An uploaded image** — any picture you like.
- **An AI-generated portrait** — when an image backend is configured, generated in place (this rides the standard `image.generate` RPC and works over both local and remote gateways).
- **A pixel pet** — a companion from the [petdex gallery](./features/pets.md) that bounces beside the avatar while the Bot is busy. Run `hermes pets` in a terminal to explore the gallery.

A Bot's look, title, and description are stored in the profile's metadata on the backend, so the same Bot appears the same way on every desktop connected to that backend.

## Routines

The **Routines** pane attaches recurring tasks to the Bot that does them — "summarize my inbox every morning" lives next to the Bot responsible for it. The pane docks beside the chat only while the Bots tab is active and steps aside when you switch back to Sessions (older desktop builds keep it always visible). Closing it with its ✕ hides it until you next leave and re-enter the Bots tab, when it comes back as the collapsed right-edge tab. A structured schedule picker builds the schedule (frequency first, then only the detail that matters), with an Advanced field exposing the raw Hermes schedule string.

Routines are plain [Hermes cron jobs](./features/cron.md) namespaced `[bot:<name>] <routine>` — they also show up in `hermes cron list` and the core Cron page. Runs land in the Bot's own chat history, so the result is right where you would talk to that Bot anyway.

## Groups and group chats

Messages sent during a member turn queue behind the active room drive, including
replies in the same thread. They do not interrupt that member or mark unseen
messages as read. Stop cancels queued continuations and holds the members until
resumed: @mention a held member, or address the whole room (`@all …` /
`@everyone …`, any wording) to release everyone — `@all resume` is not a
required incantation. `stop`, `halt` or `pause` holds a member only when the
word sits next to its @mention (`stop @bot`, `@bot please pause`); the same word
elsewhere in the sentence is read as ordinary prose, so a German `halt` no longer
silences the bot it was sent to — but such a message does not release a held
member either (`@bot please just stop now` never wakes it; repeat the stop next
to the mention to hold it). A quiet room watches timed-out members for another 21 minutes after the
foreground wait ends; this observation window does not extend the turn itself.
Unresolved member failures remain visible in the collapsed Activity summary after
the room settles — including a turn the member's backend itself failed (bad
credentials, provider errors), which is reported the moment the gateway
reports it instead of looking like twenty minutes of thinking. Expand Activity for the turn sequence; re-address the member to
try again. An ambiguous submit failure is not automatically resubmitted.


Right-click a local Bot → **Manage groups** to add or remove it from any number of group chats. Pick existing groups independently or create one inline. Local membership is stored in the Bot's backend-synced profile metadata, so it follows that profile across desktops; older profiles with one legacy group continue to work. Connections Bots join through the New Group Chat picker and remain source-qualified in the room's shared state.

To edit an existing room as a unit, use **Manage members** — from the room header's people icon or the **Manage members…** button in **Group settings**. The checklist pre-selects the current members and lists every local and Connections Bot; add or remove several at once and **Save members** applies the new roster (still 2–6 Bots) while the room, its history and each member's room session stay in place. A removed Bot takes no further turns; an added Bot reads the recent room history on its first turn. **Cancel** changes nothing.

**Rooms follow your gateways, not one Desktop.** Each room's recent transcript, members, picture, and name are mirrored into the shared profile metadata of **every** gateway your Desktop is connected to, with per-gateway versioning so two Desktops writing at once merge instead of overwriting each other. Open Hermes Desktop on another machine against the same gateway (local network, Tailscale, anywhere) and the room appears with its history; gateway-only clients see it too. Rooms carry a durable internal identity, so renaming one changes just its display name everywhere, disbanding one removes it permanently on every client — even ones that were offline at the time — and recreating a same-name group starts a genuinely fresh room. If a gateway dies or is removed, nothing is lost: every connected Desktop keeps the full room locally and re-seeds any gateway it reconnects to. (The full orchestration log stays in each Desktop's local storage; the shared mirror is a bounded recent-history projection.)

A group's identity is editable, at creation and after:

- **New Group Chat** lets you set an optional **room picture** alongside the name — upload one from your device or generate one (when an image backend is configured), using the same pipeline as Bot avatars. The picture appears in the roster row (in place of the default group glyph) and leads the title in the room header.
- The room header's **gear** button opens **Group settings**, where you can **rename** the group or set, replace, or remove its picture any time. A rename carries everything with it — the room log, each member's room session, memberships, and the picture — so no history is lost. Renaming to a name that's already taken is rejected rather than silently suffixed.

Groups are standalone rows in the same activity-ordered roster as Bot DMs. A Bot keeps one DM row even when it belongs to several groups, while every group gets its own room row with member count, latest-message preview, timestamp, and needs-you state.

A room row organizes like a Bot row. Right-click it → **Pin to top** to keep a daily-driver room above the unpinned Bots and rooms (**Unpin** puts it back into recency order); the pin is saved with the room on this Desktop. Right-click → **Move to section** files the room into one of your [sections](#organize-bots-into-sections) — or drag the row onto a section heading — and **Remove from section** returns it to the group-chat bucket. A room's section is stored on its room record (rooms have no profile), so it stays local to this Desktop like the section list itself.

The room composer (and the reply-in-thread composer) starts as a single row and grows as your prompt wraps or gains **Shift+Enter** newlines, up to half the window (at most 24rem); past that it scrolls inside the box so the transcript keeps its space. **Enter** sends.

Use the **Move up** and **Move down** arrows beside a room to choose its position among rooms. Until the first move, the existing pinned-first, recent-activity order is unchanged. After a move, room order is saved on this Desktop and survives reloads; new rooms follow the explicitly ordered rooms within their pinned or unpinned band. Moves cannot cross the pinned boundary, and filtering does not discard hidden rooms from the saved order. These controls reorder actual Group Chat rooms, not user-created Bot folders, and do not change membership or gateway ownership.

**Open chat** on any group row (2–6 Bots) opens a shared room where the whole group coordinates:

- **One visible conversation.** Public messages and each member's reply stay readable in arrival order, with the speaker's name and timestamp. Starting another topic does not collapse earlier replies. **Reply in thread** continues that topic without reordering the room; **Activity** is a secondary status view, not a replacement for messages. Private Bot Chats remain separate.
- Your message triggers up to **three serial rounds** of member turns. @-mentioned Bots respond (everyone responds when nobody is mentioned); each Bot replies briefly or passes, and the room settles when a full round stays silent.
- Teammates can hand off to the primary Bot with `@hermes`, including in older saved rooms; Bots on other gateways keep their device-qualified tags (for example, `@default-vera`).
- Bots pull each other in with `@name`, and escalate real judgment calls to you with `@user` — the group row shows a **needs you** badge when that happens. Pending questions and command approvals also light that badge; resolving the last prompt clears only prompt attention, not an independent mention. A command approval in the room answers on the click itself — `once`, `session`, `always` or `deny` sends at once, with no second button to find. Prompts follow a renamed room, while disbanding retires them even if a member's in-flight poll arrives later.
- Hard caps (10 messages per send, 3 rounds) keep rooms from spinning.
- Each member keeps its own persistent room session, so room context survives like any other conversation.
- **Not every Bot replies to every message.** Speaking is each member's own choice — a Bot replies only when it has something new to add and passes otherwise, and @-mentioning specific members scopes the round to them. Expect the members you addressed (or whoever has something to say) to speak, and the rest to stay quiet.
- **Rooms keep running when you close the Desktop.** When every member of a room lives on the same gateway, that gateway owns turn scheduling through a durable driver: closing Hermes Desktop (or losing its connection) does not stop a room mid-discussion, and the Desktop simply catches up from the room's log when it reconnects. `groups.capabilities` on the gateway reports `driver: true` when this applies. More than one room worker may share a home — the messaging gateway (`hermes gateway run`) and the Desktop's own backend (`hermes serve`) both run one — and whichever holds the room's driver lease runs the next turn; a member's room session is held only for the duration of its turn, so the lease can move between workers without a turn being refused. Room state lives in the install's root `shared-state.db`; installs that created rooms before that file existed (when rooms were still kept in the root `state.db`) get those rooms copied across once, on the first open after updating, so pre-existing Group Chats stay reachable. Rooms whose members span several machines are different: each member's turns run on its own gateway, and the cross-connection courier described under *Bot-to-bot messaging* still applies to them.

- **Mentions read as identities.** In the transcript a routed `@bot` mention, the human handoff `@user`, and the broadcasts `@everyone` / `@all` render as inline references (accent text, not pills); unknown `@words` and e-mail addresses stay plain. Hover a Bot's message and use **Reply to @handle** to seed `@handle ` into the composer, so your next send goes to that Bot only (when a same-named Bot from a Connection shares the room, the tag is device-qualified, e.g. `@reviewer-mini` / `@reviewer-local`) — **Reply in thread** still continues the whole thread. If a reply box for a *different* thread is open, **Reply to** seeds the main composer instead and so starts a fresh thread; use **Reply in thread** to continue that thread.
- **Rooms can span machines.** **New Group Chat** is available as soon as two Bots are selectable across all your registered connections — one Bot on this device plus one on another gateway is enough. The picker seats Bots from any registered connection; each member's turns run on its own machine, in its own room session there. Cross-machine members carry a device badge (`dixie · Mac Mini`) in the room and in other members' transcripts, and the disambiguated `@name-device` handle works in room mentions — so same-named agents on two machines never blur together. A member's turn that outlives this Desktop — you quit or it crashed mid-turn — finishes on the member's own machine: the room posts that reply the next time it is driven, and does not re-drive the member while it is still working.
- **Rooms show the same identity as the roster.** Transcript rows, the “X is thinking…” line and Activity rows resolve each member's title and avatar by its owning connection and profile — re-titling a Bot or changing its picture updates every room it sits in, remote members keep their own titles, and two same-named Bots on different connections never borrow each other's avatar. When two members would still read identically, Activity appends the connection label (`Reviewer · Mac Mini`).
- **Plugins can watch members work.** The durable room log records `turn.started` and `turn.settled`; what a member does in between (tools, approvals, streamed text) is projected to plugins through the [`on_room_member_activity`](./features/hooks.md#on_room_member_activity) hook with room, member and turn coordinates, so community clients can build tool cards and live member status on top of Group Chat without reading Hermes internals.

## Bot-to-bot messaging

Bots message each other with attribution, and you can hand work off from any chat:

- **@mentions** — type `@researcher have a look at this` in any chat and the composer's `@` autocomplete helps you pick the right Bot; on send, the mention is resolved against the live roster and the active Bot is told exactly who you mean (profile, friendly name, and device for cross-connection Bots). The Bot then composes its own message and sends it with `message_agent` — your text is never forwarded verbatim, and the reply comes back attributed to that agent. An email address or an unknown `@` passes through untouched. Bots on other connected machines are reachable the same way: the Desktop relays the message over that connection's own socket (see *Bots across machines* below).
- **Renamed Bots keep their tags in sync** — give a Bot a friendly name (the pencil in its chat header, or `hermes profile rename`) and it becomes taggable by that name: a Bot titled *Research Buddy* answers to `@research-buddy` (and `@researchbuddy`), in regular chats and in group rooms alike. The composer's `@` autocomplete offers the renamed tag and also matches when you type the old profile name, which keeps resolving too. This includes the primary Bot: rename it *Maia* and group-turn prompts introduce it as `@maia`, its inter-agent messages sign as `Message from 🤖 Maia (@hermes)`, and teammates can `message_agent(target="maia")` it — `@hermes` stays a working alias.
- **Direct messages** — every Bot Chat carries the `message_agent` tool: a Bot messages a teammate by calling `message_agent(target="researcher", message="…")`. The target is the teammate's profile name, its friendly name (`hermes profile rename` or the Bot Mode title — `Scribe`, `Dr. Foo`) or the `@`-tag the Desktop inserts for it (`@scribe`, `@dr-foo`, `@drfoo`); `@hermes` always means the primary Bot. A profile name is matched first, so it can never be hijacked by another Bot's friendly name, and a friendly name shared by two Bots is refused with the roster instead of guessing. The tool validates the target against the live roster, prefixes the sender's `Message from 🤖 <friendly name> (@<handle>):` attribution automatically, and delivers into the teammate's canonical Bot Chat. Delivery is **fire-and-forget**: the sender gets a *dispatch* acknowledgement (`status: queued` plus a `delivery_id` — and a `process_id` for the background delivery process — means the message was handed to that process, not that it was delivered), finishes its turn, and that process's completion notification carries the outcome — the reply, or the delivery failure. The message travels as a real parameter (nothing shell-interpreted — quotes, `$(...)`, and backticks arrive verbatim), and the Bot composes its own message rather than forwarding your words. The teammate roster — names **and roles** from each profile's title/description — is part of every Bot Chat's system prompt, so Bots know who does what before choosing a recipient. The tool exists **only** in canonical Bot Chat sessions on Bot-Mode-managed installs; regular chats, group-room member sessions, and CLI sessions never see it.

Local messages also reach a Bot Chat that stays open in Desktop or the TUI. The receiving backend keeps ownership: it reads durable ingress on its existing notification poller, admits immediately when idle, or waits until the running turn and already queued human prompts finish. A `queued` acknowledgement confirms durable admission, **not** a completed reply. The target profile retains the delivery ID and receipt under `runtime/bot_live_delivery/`; `settled` confirms completion. A crashed or cancelled imported turn is not automatically replayed, and pending work pinned to a departed owner remains inspectable rather than being silently rerun. Do not resend a delivery whose outcome is unknown. Older backends without live-delivery capability retain the existing ownership refusal; restart that backend after upgrading.

- **Staying silent** — a Bot that has nothing to add may end a turn with one of the [intentional silence tokens](./messaging/index.md#intentional-silence-tokens) (`[SILENT]`, `NO_REPLY`, …). The Bot Chat keeps that turn in its transcript but renders nothing, and a teammate that messaged it gets an empty reply instead of the token. Failed turns and prose that merely mentions a token are shown as-is.

The backend teaches each Bot's canonical Bot Chat session the messaging protocol automatically at prompt-build time — including when a teammate opens it headlessly from the CLI. Only the canonical Bot Chat gets the protocol section; your regular sessions and your SOUL.md stay untouched. This is controlled by `agent.bot_mode_protocol` in `config.yaml` (default: on):

```yaml
agent:
  bot_mode_protocol: true   # inject the bot-to-bot messaging protocol into canonical Bot Chats
```

### What actually makes a chat a Bot Chat

`agent.bot_mode_protocol` is only the master switch. Before the protocol section — or the `message_agent` tool — is injected, two further conditions must hold, and on a desktop install the Bots pane satisfies both for you the moment it creates a Bot:

1. **The session is titled exactly `Bot Chat`.** That exact title is the canonical chat's identity (it is what the desktop plugin's createCanonicalChat uses and what `hermes -p <bot> chat -c "Bot Chat"` resumes); any other title, or an untitled scratch session, gets neither the section nor the tool.
2. **At least one profile on the install carries a `ui_meta: { hermes-bots: … }` block in its `profile.yaml`.** This is the "Bot-Mode-managed" marker the desktop plugin writes for every Bot it owns; the gate scans every profile, so one marked profile marks the whole install. There is no CLI command that writes it.

The practical consequence: on a **headless install with no desktop app** (gateway plus Telegram, say) nothing ever writes either marker, so `message_agent` is unreachable even though the docs-level switch is on — bots message each other fine over the messaging platform, but the agent-side `message_agent` tool never appears. To enable it headless, satisfy both conditions by hand:

```bash
# the canonical forever-chat, created once per Bot (later runs resume it)
hermes -p <bot> chat -c "Bot Chat" --create-if-missing
```

```yaml
# ~/.hermes/profiles/<any-bot>/profile.yaml — an empty block is enough to mark the install
ui_meta:
  hermes-bots: {}
```

From the next turn in that Bot Chat the teammate roster, the protocol section, and `message_agent` are all picked up — including over `hermes -p <bot> chat` on a purely terminal box.

:::note
Bot-to-bot delivery is per-invocation: the receiving Bot picks the message up when it next runs. Live interrupt of a Bot mid-conversation is future work.
:::

### Failed turns retry safely

Local one-shot delivery preserves the active-session refusal code separately from
its human-readable message. `SESSION_NOT_OWNED` produces `target_busy`; an
unreadable coordination registry is not mislabeled as another owner. Older local
CLIs without the code marker still use the historical refusal wording.

A failed delivery turn is retried at most once, and only when a retry can actually help. Transient failures (target runtime offline, delivery timeout, provider rate limit or server error) re-run the same Bot Chat session unchanged. A context-overflow failure also re-runs the same session — the retried turn compacts the over-threshold transcript via the standard context-compression pass before calling the model, so the retry fits where the original didn't. Auth, quota, and configuration failures never auto-retry: a second attempt cannot fix them and only burns quota, so the failure is surfaced immediately. A retried turn never starts a fresh session — your Bot Chat history and context stay intact. The re-run resumes the message the failed attempt already wrote into the Bot Chat instead of appending it again, so the recipient's transcript carries exactly one copy of the DM.

When a target has no live Desktop or TUI owner, local delivery opens that
profile's canonical Bot Chat through a quiet CLI turn. The transport prefers
the Hermes entrypoint beside the sending runtime's Python interpreter, so an
unrelated or older `hermes` on a service's `PATH` cannot take precedence when
that sibling entrypoint exists. `--in ~` selects the working directory; the
explicit Bot Chat title is resolved in the target profile's session database.

### When a delivery fails: typed reasons

A failed bot turn or relay delivery carries a machine-readable `reason` code alongside the human error text, end to end: the target gateway classifies the failure (`provider_auth_or_access`, `provider_quota_limit`, `provider_rate_limit`, `provider_server_error`, `context_overflow`, `missing_config`, `model_unavailable`, `runtime_offline`, `queued_expired`, `delivery_timeout`, `target_busy`, `unknown`), the Desktop forwards it, and the sending agent's completion notification is tagged `[reason: <code>]` ahead of the error text. A calling agent can branch on the code — "sign in again" vs "retry later" — instead of parsing provider prose. The Desktop's needs-attention badge uses the same codes.

### Messaging across connected machines (the Desktop relay)

Every gateway you register in **Settings → Connections** — local, remote URL, SSH, Hermes Cloud, docker — is a persistent line the Desktop holds open, and Bot Mode uses those lines for messaging automatically. No extra setup:

- **Rosters propagate on their own.** While the Desktop runs, it periodically tells each connected gateway which agents live on the *other* connections. Every Bot Chat's teammate roster then lists them ("Teammates on OTHER connected machines"), with names, roles, and which machine they're on — and the roster refreshes when agents appear, disappear, or get renamed (capability epoch).
- **`message_agent` reaches them directly.** A Bot on your laptop messages the cloud agent with `message_agent(target="moxie", …)` exactly like a local teammate. If the same handle exists on several machines, disambiguate with `target="moxie@<connection>"` (the tool's error tells the Bot the exact forms). Delivery rides the Desktop: the sending gateway queues the message, the Desktop relays it to the target connection's own gateway, the target Bot runs a turn in its canonical Bot Chat, and the reply comes back to the sender as the same background completion notification local DMs use. Messages to different Bots are delivered side by side, so one Bot's long turn never delays another Bot's mail (or ages it past `bot_mode.envelope_ttl_seconds`); messages to the *same* Bot are delivered in order, one turn at a time.
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

:::note One-way reachability (NAT)
Cross-gateway links are direct gateway-to-gateway connections — Desktop is a
viewer, not a relay. A gateway behind home NAT can dial out to a public peer
(laptop → VPS works), but the reverse direction has no inbound route
(VPS → home fails) unless your network provides one. If your Group Chat spans
a NAT boundary, put the room's authority on the host every participant can
reach (typically the public VPS), or bridge the network with Tailscale/VPN.
:::

### Transferring hosted room authority

Authority takeover is an **operator recovery procedure**, not an atomic handover.
Use the existing JSON-RPC methods `groups.promote` and `groups.demote` on the
appropriate gateway. There are no `groups.peer.promote` or `groups.peer.demote`
methods; `groups.capabilities` lists the methods your gateway supports.

:::warning Fence the old writer before promotion
Before sending `confirm: true`, establish that the previous authority **cannot
commit**, and keep that fence in place until it has been demoted. Stop its
room-writing processes and prevent automatic restart, or use an equivalent
infrastructure fence. A network timeout, disconnecting Desktop, or `groups.stop`
is not proof: the old gateway may still be running, and stopping a turn does not
revoke room authority. If you cannot establish the fence, do not promote.
:::

1. **Check replica coverage.** On the replacement gateway, inspect
   `groups.replica_state` with `{"room_id":"ROOM_ID"}` and compare `last_seq`
   with `latest_seq`. Require a complete replica before planned takeover;
   promotion itself does not check this coverage. `groups.replicate` reports
   `caught_up` after ingesting pages returned by `groups.log`; peer registration
   alone does not prove the replacement has the room history. Caught-up status
   describes the last replicated page, not proof the old writer has stopped or
   that no newer events exist. For a planned move, quiesce writers, replicate
   through the final cursor, then maintain the fence. For disaster recovery,
   account for any history that never reached the replica.
2. **Promote only while the old writer is fenced.** On the replacement:

   ```json
   {"jsonrpc":"2.0","id":1,"method":"groups.promote","params":{"room_id":"ROOM_ID","confirm":true,"reason":"planned-handover"}}
   ```

   `room_id` and `confirm: true` are required; `reason` is optional and defaults
   to `authority-unreachable`. Confirmation is your assertion that the previous
   authority cannot commit, **not** a request to fence it automatically. Without
   confirmation the call returns error `4118`. A successful result reports
   `authority_gateway_id` and `authority_epoch` (the replicated epoch plus one).
3. **Demote the old authority before returning it to service.** Keep its normal
   room writers fenced while applying this RPC through a controlled recovery
   connection on the old gateway. Replace the example gateway ID and epoch with
   the exact values returned by the successful promotion:

   ```json
   {"jsonrpc":"2.0","id":2,"method":"groups.demote","params":{"room_id":"ROOM_ID","observed_gateway_id":"NEW_GATEWAY_ID","observed_epoch":2}}
   ```

   All three parameters are required. Do not guess a future epoch: demotion
   requires evidence of a newer authority, not an invented value. It records
   `authority.lost` and adopts the observed lineage; repeating the same lineage
   is idempotent. If the old host is unavailable, keep it fenced and perform
   this step before restoring its normal writers.
4. **Verify and reconnect.** Read `groups.state` on both gateways and compare
   `room.authority_gateway_id` and `room.authority_epoch` with the promotion
   result. Old-authority sends must be refused; direct clients to the replacement.
   Demotion fences writes; it does not merge histories or automatically turn the
   old authoritative store into a synchronized replica.

Promoting while the old gateway remains writable allows both independent
`state.db` stores to accept messages and develop divergent histories. A higher
epoch on the replacement does not remotely disable the old writer; equal epochs
are not required for split-brain. If histories have already diverged, fence
writers and preserve both histories for recovery rather than assuming that
promotion, demotion, or replay will merge them.

## Bots across machines

When you register several backends in **Settings → Connections** — the local runtime, remote gateways, SSH hosts, Hermes Cloud instances — the roster shows the Bots from **every** connected source, persistently: SSH sources are inventoried without spawning anything on the remote box, and machines that are momentarily unreachable keep their last-known rows instead of vanishing. When the same profile name exists on several sources, handles disambiguate as `@name-device` (for example `@research-homelab`). A Bot's chats, sessions, memory, and routines live on the machine that owns the profile.

Clicking a Connections Bot does **not** hop your window onto that machine — stay in your chat and `@mention` it, seat it in a group chat, or create new agents on it directly with the **Create on** picker. Cloud and local agents share one roster this way: register your Hermes Cloud instance and your desktop (say, over Tailscale or SSH) and their Bots can message each other and sit in the same rooms, with each agent's work running on its own machine. Bot-to-bot DMs across those machines go through the Desktop relay automatically (see *Messaging across connected machines* above).

See [Connecting Desktop to Many Hermes Instances](./multi-connection-desktop.md) for the full multi-connection guide.

## Warm Bot Backends (how many bots run at once)

Each local Bot runs in its own backend process, and Desktop keeps at most **Settings → Advanced → Warm Bot Backends** of them alive at once (default 3, ~60 MB each). Idle backends are reaped after the idle timeout next to that setting (default 10 minutes); the `Hermes backend for profile "<name>" exited (1)` line in `desktop.log` that follows an idle-reap message is that cleanup, not a crash. A Bot you open while every slot is busy waits up to 30 seconds for a slot, then fails with *timed out waiting for a free local slot*.

Reads of another Bot's chat history and background transcript refreshes do **not** take a slot — only an interactive open or a running turn does. If you drive a large fleet (group chats with many members, or Kanban dispatch across many profiles), raise Warm Bot Backends toward the number of Bots you expect to be active at the same time and give the machine the memory to match. Setting it higher than the profiles you actually use only adds startup work.

## Turning it off

Bot Mode is a bundled desktop plugin. Flip its **Desktop** switch off in **Capabilities → Plugins → Bots** — the roster, the Routines pane, and the composer middleware unregister live, no restart needed. Your profiles, sessions, and cron jobs are untouched either way; Bot Mode never owns your data, it only renders it.

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
