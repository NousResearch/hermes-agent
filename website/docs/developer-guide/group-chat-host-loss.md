# Group Chat host loss: copies, voters and moving a group

A Group Chat has one host: the installation that orders its log and dispatches its turns. This page
explains how a group survives losing that host, and then documents the contracts that keep every
copy of its history, decide who may continue it, and protect what the host acknowledged. The
moves themselves (fencing, the four kinds of proof, reconciling accepted work) build on these
contracts.

:::note Where each part lands
- [#99107](https://github.com/NousResearch/hermes-agent/pull/99107): the verified-transition marks
  and the four proof kinds.
- [#104601](https://github.com/NousResearch/hermes-agent/pull/104601), the contracts on this page:
  a copy on every member, custodians and successors, voters and modes, majority protection,
  configuration changes one at a time, heartbeats and the lease layer's hook points, following a
  group across moves, catch-up from any custodian, and retiring copies after a move.
- [#105079](https://github.com/NousResearch/hermes-agent/pull/105079): fences at the participants,
  one promise per epoch, host leases and the sleep-aware clock.
- [#105197](https://github.com/NousResearch/hermes-agent/pull/105197): the moves themselves
  (handover, majority certificate, careful evidence, one tap), reconciling accepted work, the status
  contract, the CLI and the notices.

Until #105079 and #105197 land, a group already keeps copies, voters and protection, but nothing
moves it: the takeover gate stays closed and an unproven change of host stays quarantined. Where the
lease layer's code isn't installed, a host offers no automatic moves at all (`automatic: false`), so
its groups ask first.
:::

## At a glance

- Every member computer keeps the full ordered history of the group, with an explicit opt-out. A
  Bot added later from someone else's computer keeps the group's full history too, including the
  messages from before it joined; Hermes Desktop warns the owner when adding one.
- When the host goes away, the group continues on another computer the owner allowed:
  **automatically by majority** with three or more always-on computers, **automatically after 3
  minutes of silence** with exactly two (careful mode, which can run the group on both if the two are
  only cut off from each other), and **on one tap** otherwise.
- The move fences the old epoch at every computer it reaches, catches up from the most complete
  copy, and reconciles the accepted work that copy records; uncertain work never reruns by itself. In
  majority mode the successor knows every dispatched task; in the other modes, work the old host
  dispatched in its last seconds can be unknown to it. A returning host rejoins as a copy.

```text
 Host stops or goes silent
   |
   +-- planned (stop, sleep, "Move to...") ---> old host signs a HANDOVER --> the chosen computer continues
   |
   +-- 3+ voters ("majority" mode) ------------> host's lease runs out (~20 s); a cut-off host stops itself
   |                                             standby collects promises from a majority (CERTIFIED)
   |                                             --> continues within about a minute; never two hosts by itself
   |
   +-- exactly 2 voters ("careful" mode) ------> 180 s of silence in both directions, and the standby is online
   |                                             standby signs EVIDENCE --> continues; the owner is warned
   |
   +-- owner chose "Ask me first", or no ------> owner taps "Continue on <best computer>" (ATTESTED)
       always-on standby ("ask" mode)

 Every path: fence the old epoch at participants -> adopt the most complete copy ->
             marked authority.transition -> reconcile accepted work -> old host returns as a copy
```

*Voters* are the host plus the always-on computers that may host the group: designated by the
owner and allowed by their own operator. Laptops can continue a group when asked, but never vote.

```text
  Clients: Desktop · messaging · CLI   (attach only; status and actions go to the host)
                         |
                         v
      +---------------- HOST (epoch N) ----------------+
      |  orders the log, dispatches turns               |
      |  majority mode: runs only while it holds a lease|<-------+ lease grants
      +--------+-----------------------------+----------+        | (voters only)
               | pages + heartbeats, authenticated by room       |
               | grants: voters about every 5 s, others          |
               | each minute                                     |
               v                             v                   |
      +-- VOTER COPY ----------+    +-- VOTER COPY -----------+  |
      | always-on, may host    |    | always-on, may host     |--+
      | full ordered log       |    | full ordered log        |
      | grants lease; fences   |    | grants lease; fences    |
      +------------------------+    +-------------------------+
      +-- COPY (laptop, or a computer not both allowed and designated) --+
      | full ordered log; no vote; refuses old-epoch work after a move   |
      +------------------------------------------------------------------+
```

## Decisions, and why

| Decision | Why | Rejected alternatives | Lands in |
|---|---|---|---|
| **Every member computer keeps the full ordered log**, with an explicit opt-out, including one whose Bot joined later. The owner can add backup computers that hold a copy without a Bot. | Any eligible survivor can continue with the complete history. Desktop warns the owner before adding a Bot from someone else's computer. | Three fixed coordination hosts; opt-in copies; history only from joining onward. | #104601 |
| **One host per epoch.** A host change is accepted only with a verified, marked `authority.transition`. Anything unmarked stays quarantined. | Every move is single-owner, monotonic and auditable. | Leaderless or CRDT logs. | #99107 |
| **Old epochs are fenced at the participants**, with one promise per epoch. | An old host's late work is refused wherever it lands. | Trusting the old host to stop. | #105079 |
| **Successors are owner-controlled.** A computer can host only if its operator allowed it *and* the owner designated it. | A group never moves somewhere nobody chose. | Algorithmic picks among all members. | #104601 |
| **Majority mode (3+ voters):** the host keeps running only while a majority grants its lease, and a voter never backs a takeover while its grant is live. | Any two majorities share a voter, so an automatic move never leaves two hosts starting work. Nothing hosted is needed. | An external lock service; a hosted referee by default. | voters: #104601; leases: #105079; moves: #105197 |
| **Careful mode (exactly 2 voters)** is our proposed default and the one place this layer goes past the bar: it moves without a lease or quorum, so it does **not** prevent split brain. The owner can switch a group to "Ask me first"; making "ask" the default for two voters is a one-line change in `mode_of`. | With two computers a dead host and a cut link look identical. We judged that a group stalled until its owner sees the notice and taps is more likely, and usually more harmful, than a split. A split happens when both stay online but lose each other for 3 minutes or more (for example an expired overlay-network key); it lasts until a device that reaches both passes on the move, or the link returns, and the owner chooses which history to keep. | Always asking (stalls overnight); moving on a timeout alone, without the both-ways silence, the standby's online check and a signed record of them. | mode and switch: #104601; the move: #105197 |
| **Voters change one at a time**, each change stored on a majority of the old and the new voters before the next. A move keeps the voters: the previous host stays a voter (when always on) and a successor. | No configuration change can leave two majorities that disagree, a majority group stays one after a move, and the owner can move it back. | Changing the voter set freely; dropping the old host at a move. | #104601 |
| **Copies pass history between themselves only as far as the host signed it.** Every page and heartbeat carries a head the host signs; catch-up stores nothing a head doesn't vouch for. | A member computer can't add messages, keys or voters to another computer's copy. | Trusting a custodian's own signature or watermark. | #104601 |
| **Durability in majority mode:** a task is dispatched only after a majority stores its admission, and a send reports `protected: true` once a majority stores it. In the other modes the tail is shown at risk and sends report no `protected`. | Nothing reported as protected (`protected: true`) is lost on an automatic move, and the successor knows every dispatched task. | Asynchronous copies with silent loss. | #104601 |
| **Room authority is not session authority.** A move changes only who orders the log and dispatches turns. Each Bot's sessions stay with its own computer. | It stays compatible with one gateway per installation. | Moving Bots or their sessions. | #104601, #105197 |

## Why it is safe

- **Never two hosts from an automatic move in majority mode.** A host must hold lease grants from a
  majority of voters, and a voter refuses to promise a takeover while its grant is live. Any takeover
  majority shares a voter with the host's majority, so a takeover completes only after the old lease
  ended. An owner's **Continue anyway**, or **Continue on…** while no majority can be reached, can
  still split the group; both carry the 'only if … is really offline' caution.
- **Nothing reported as protected (`protected: true`) is lost on an automatic move.** A send reports
  it only in majority mode, once a majority of voters stores it; the successor's majority includes a
  holder, and catch-up adopts the most complete copy.
- **Uncertain work never reruns by itself.** In majority mode a task is dispatched only after its
  `task.admitted` is stored on a majority, so the successor knows about it. In the other modes, work
  the old host dispatched in its last seconds can be unknown to the successor. Unknown outcomes are
  shown and never rerun.
- **No custodian can add to the group's history.** The host signs a head for every page and
  heartbeat, and a copy takes history from another copy only as far as such a head vouches for it,
  checked with the host's pinned key. So keys, configurations and voters come only from history the
  host wrote.
- **A copy can't take over by accident.** A copy follows a new host only through transitions it
  verified, signed with per-installation room identity keys pinned from the room's own records.
  The host's own pages are authenticated by room grants; catch-up requests and replies between
  custodians are signed with room identity keys.
- **Changing voters can't open a gap.** A configuration that drops a voter takes effect only once
  a majority of the old voters stores it, so a cut-off standby that still holds the older
  configuration never finds a majority that follows it.
- **Careful mode is the only *automatic* path where a split can happen.** It happens when both
  computers keep running but can't reach each other in either direction for 3 minutes or more while
  the standby's online check passes, and lasts until a device that reaches both hands the move to the
  old host, or until they reconnect.
- **A paused host appends nothing.** In majority mode, while the host has no lease (or hands the
  group over), it writes no configuration and admits no work, so a partition can't create post-split
  events there; without the lease layer's answer it fails closed the same way, in careful mode too.
  In careful mode a host that is cut off but online keeps writing; when the two meet, its later
  messages are set aside (`continued_on_two`).

The rest of this page documents the contracts in [#104601](https://github.com/NousResearch/hermes-agent/pull/104601).
The marks are in [#99107](https://github.com/NousResearch/hermes-agent/pull/99107), the fences and
leases in [#105079](https://github.com/NousResearch/hermes-agent/pull/105079), and the moves in
[#105197](https://github.com/NousResearch/hermes-agent/pull/105197).

## Custodians and successors

Every member installation is a **custodian**: its member grant carries `replicate` unless its
operator opts out (`groups.peer.invite` with `replication: false`), and the host copies the whole
history there, one bounded page at a time. Several Bots on one installation share one copy. That
includes a Bot added later from someone else's computer: that computer keeps the group's full
history, including the messages from before its Bot joined, and Hermes Desktop warns the owner
when adding one. `groups.capabilities` names whose computer it is beforehand (`room_identity`:
`install_id`, `name`, `operator_name` from `gateway.owner_name` or null, and `always_on`). An
owner can also add a **custodian-only** installation, such as a backup VPS without a Bot:
`groups.peer.invite` with `custody_only: true` there mints a copy-only grant (member id
`custody:installation`, permissions `replicate` and `status`, never `dispatch`), and
`groups.custody.add {room_id, target_url, catalog, grant, successor?}` on the host enrolls it after
a live probe. `groups.custody.remove {room_id, install_id}` stops copying there. The host copies the
same way, on a copy-only grant, to a custodian that none of its member routes reaches: after a move,
the previous host's Bots ran on it, so it gives the new host such a grant when it steps down
(#105197), and the new host's pushes, with its lease requests, reach it there.

A copy-only grant lasts at most 30 days, so its custodian renews it through its acknowledgments:
once less than a week of it is left (or a quarter of its life), the acknowledgment of a push from
the host it follows carries `custody.renewed_grant`, the same grant with only its life moved, and
the host keeps it as the route's grant. A quiet group's keepalives carry it too. A route its
custodian refused as unauthorized is probed again every hour and resumes once its grant is
accepted, or as soon as a renewed grant arrives.

A custodian is a **successor**, one that may continue the group, only when both are true:

- its own operator allowed it: grant permission `successor` (`groups.peer.invite(successor=true)`),
  or later `groups.custody.allow {room_id, successor}` on that computer;
- the room's owner designated it: `groups.custody.designate {room_id, install_id, successor}` on the
  host. Designations keep their order: that is the owner's order of successors.

Each installation reports whether it is **always on**: it has no battery, unless its config says
otherwise with `group_chat.always_on: true` or `false`. A computer whose battery state can't be
read is not always on. The report rides on the capabilities probe and on every acknowledgment.

| Method | Where | Who | Result |
|---|---|---|---|
| `groups.custody.status {room_id}` | host or copy | `session:read` | custodians, voters, protection (below) |
| `groups.custody.designate` | host | the room's owner (`session:control`) | `{room_id, install_id, successor, configuration_seq}` |
| `groups.custody.add` / `.remove` | host | the room's owner (`session:control`) | `{room_id, install_id, configuration_seq}` |
| `groups.custody.allow` | the custodian | its operator (`session:operator`) | `{room_id, install_id, allowed, confirmed}` |
| `groups.custody.automatic {room_id, enabled}` | host | the room's recorded owner or the operator; else `not_owner` | `{room_id, automatic, configuration_seq, pending}`: `pending` stays true until the switch is in a configuration stored on a majority of the voters |

The owner's actions (`designate`, `add`, `remove`, `automatic`) are refused (`permission_denied`)
to a messaging chat other people read (transport `messaging:shared:`), which carries the owner's
subject but never acts for the owner. Errors are JSON-RPC `4001` with `error.data.reason`, such as
`room_custody_invalid`, `peer_target_mismatch`, `not_owner`, `permission_denied` or
`invalid_params`.

## Room identity keys

Each installation has one Ed25519 room identity key, derived by a domain-separated HMAC from the
RoomLink secret in its installation root. Every profile, and one multiplex gateway serving them,
signs as the same installation, and no new private key is stored. Signatures are
`ed25519-v1.<base64url>` over `domain + "\0" + canonical JSON`. Keys are pinned per room and
installation: the host pins a member's key from its authenticated probe, and every custodian pins
the keys its copy's configurations name. A different key for a pinned installation is refused,
never adopted (`hosted_room_identity.verify_locked`).

## The configuration: `custody.configured`

The host records the room's custodians in the room's own log, so every copy carries them. The
event (system actor `custody-control`, id `system:custody-configured:<n>`) has exactly this payload:

```json
{
  "custodians": [{"install_id": "install:…", "public_key": "<hex>", "endpoint": "https://…",
                  "role": "authority | custodian | custodian_only", "successor": true,
                  "always_on": true, "voter": true, "name": "Mac mini", "operator_name": "Dana"}],
  "owner_name": "Dana",
  "automatic": true,
  "voters": ["install:host", "install:…"]
}
```

- Custodians are sorted by `install_id`, with exactly one `authority`: the current host, never its
  own successor. Names are display labels only, never identities.
- `voters` is the host first, then its always-on successors in the owner's order, at most seven.
  `voter` marks exactly those.
- `mode_of(configuration)` is `majority` with three or more voters, `careful` with exactly two, and
  `ask` with one or when `automatic` is off. `automatic` is the owner's switch where the lease layer's
  code is installed, and off elsewhere: a host offers only moves something can carry out.
- **One change at a time.** A configuration changes at most one voter, or the automatic switch,
  and the next change waits until a majority of both the voters before and after it stores it.
  `voter_sets` in the status names the sets whose majorities count now: two while a change is not
  settled. Names, endpoints and non-voting custodians change at once.
- After a verified move, the new host appends its first configuration with
  `reconfigure_after_transition_locked`: it becomes the authority and the first voter; the previous
  host stays a custodian that may continue the group again, and a voter right after the new host when
  it is always on, so the voters don't change and the owner can move the group back; every other
  custodian keeps its fields, the voters their order, and the group its automatic switch. A move
  settles every change of voters before it: the new host counts only changes made since.

## Watermarks, protection and the tail at risk

Each copy, and the host's own room, has a durable watermark `(epoch, seq, event_hash)`, where
`event_hash` chains every event of that exact prefix (`sha256` over a domain-separated genesis
and each normalized event; checkpoints every 128 events keep it bounded). A custodian acknowledges
every page with its watermark, written in the same transaction as the page, and the host keeps an
acknowledgment only when it matches its own chain: a divergent copy is never counted and stops
its route, and an older Hermes without watermarks is `unsupported` and never counted.

- `at_risk_after_seq` is the highest seq that at least one successor holds. Every later event is at
  risk of being lost with the host, and clients say so.
- `protected_seq` is the highest seq that a majority of every current voter set holds, counting the
  host. `wait_protected(room_id, seq, timeout)` waits for it.
- **Majority mode waits.** A queued task runs only once a majority stores its `task.admitted`; until
  then `waiting_for_copies {task_id, seq}` names it. `groups.send` waits up to `min(10 s, the lease
  left)` and returns `protected`. An unprotected send (`protected: false`) stays in the log, inside
  the tail at risk, and clients offer it again after a move, deduplicated by its event id.
- **Other modes never wait, and report no `protected`.** Dispatch there doesn't wait for copies, so
  a message offered again after a move could run its turn twice; clients instead show messages
  missing after a move with a manual **Send again**.
- A host that doesn't serve the room (`serving`) starts no queued task, in any mode.
- Each dispatch decision appends `task.admitted` (system actor `room-driver`, id
  `system:task-admitted:<sha256(task_id)[:32]>:<generation>`) when the room has more than one
  custodian, so a successor can reconcile it.

`groups.custody.status` returns, on the host, every custodian with `role`, `state`, `successor`,
`voter`, `always_on`, `allowed`, `designated`, its verified `watermark`, `acknowledged_at`,
`last_seen` and `divergent`, and for the group `at_risk_after_seq`, `protected_seq`, `automatic`,
`voters`, `voter_sets`, `mode`, `waiting_for_copies`, the configuration and `head` (below). On a
copy it reports the configuration it holds, what the host last reported, and the head that vouches
for the copy.

## Heads the host signs

With every page and heartbeat the host sends `custody.head`, signed with its room identity key
under `hermes.group.custody.head.v1`:

```json
{"room_id": "…", "host": "install:…", "epoch": 2, "seq": 412, "chain_hash": "<64 hex>",
 "signature": "ed25519-v1.…"}
```

`chain_hash` is the custody chain over the prefix `1..seq` (a watermark's `event_hash` at that seq),
`seq` the page's last event, and `epoch` the host's. A custodian keeps the head only when it is
signed by the host its copy follows (checked with the key it pinned for that host), names that
host's epoch, and matches its own chain; it keeps the latest per epoch, and one from an earlier
epoch stays valid for its prefix. A host that continues its own group at a fresh epoch (the split
rule, keeping it, or **Continue anyway**) keeps its own last head of the epoch it leaves
(`keep_own_head_locked`, in the writer that appends the change): nobody else can sign it, and a
copy that missed that epoch's last pushes needs it to cross the change. `heads_locked` lists the
heads kept here, one per epoch, and `vouched_head_locked` returns the head that vouches for the
history held here (signed on the spot on the host), for status and for the receipts a move collects.

## Heartbeats and the lease layer's hooks

A caught-up custodian still hears from the host: a voter about every 5 seconds, any other
custodian every minute, as an empty page carrying the custody report and the head. The lease layer
plugs in through `hosted_room_custody.register_lease_hooks()`. Until it does, a host in majority or
careful mode appends nothing, and the other hooks do nothing:

| Hook | Called | Contract |
|---|---|---|
| `lease_request_provider(room_id)` | on the host, building each push to a voter | returns `{epoch, duration_s, until?, sent_at}` or None; attached unchanged as `lease_request` |
| `lease_grant_hook(room_id, epoch, authority_install_id, request)` | on a custodian, for every push it stored from the host it follows: any push is contact | `request` is None when the push asked for no lease; an answer to a request (`{granted_until_s}` or `{refused, events?}`) goes back as `lease_grant` |
| `lease_ack_hook(room_id, voter_install_id, lease_grant, sent_at)` | on the host, for each acknowledgment | `sent_at` is the request's own, taken before the send |
| `lease_remaining_provider(room_id)` | on the host, for a send's wait | seconds the majority lease still holds, or None |
| `serving_provider(room_id)` | before any append here, and before a queued task starts | False while the host is paused: no `custody.configured`, no `task.admitted` (`room_host_paused`), no dispatch. Without an answer (no provider, or None), a host in majority or careful mode is paused too |

## Following a group across moves

A copy follows a new host only through transitions it verified (`ingest_page(...,
_verify_transition=fn)`). Pages are verified in order, inside one writer: the events before each
`authority.transition` are stored first, with the custodians of any `custody.configured` among
them pinned, then `fn(conn, event)` checks the transition against that history and marks it, and
only then is it inserted. A page that carries a move, the new host's configuration and a later
move verifies the later one against the configuration written before it. Every other event must
belong to its span: its epoch, and for a gateway actor that span's host. A page may lag its
sender: a new host relays the old host's history before its own transition, and the copy moves
only when the transition arrives. Without a verifier, a change of host is refused.

## Who may ask a computer to continue a group

Continuing always runs on the eligible computer's own gateway (`target_not_local` otherwise), and
the caller must act for the room's owner on that computer:

- the subject recorded as the owner there (on the host, the room's creator; on another computer,
  whoever consented there; after a move, the owner who continued it), or
- that computer's operator.

Anyone else gets `not_owner`. The same rule covers `prepare`, `promote` and `keep`.

## Continuing by hand (`groups.succession.prepare`, `groups.succession.promote`)

When nothing moves the group by itself, the owner continues it on one eligible computer:
pause, preserve, continue by hand. Nothing is elected.

`prepare` changes nothing. It asks every configured computer, signed with this installation's
room identity key, whom it follows. It refuses with `host_reachable` while the host answers or a
restart it announced is still running. Otherwise it returns a preview: how far this copy is
behind the most complete reachable one, the tail at risk, inherited work, the host's Bots that
stay unavailable, and cautions. A copy counts only as far as a head the host signed vouches for it
(each answer carries its `heads`), never by its computer's own word, and the tail at risk is what the
host told this computer it had beyond the most complete vouched copy.

`promote` runs the owner's decision in recorded steps, so a crash resumes instead of repeating
one:

1. **Fencing.** Every reachable computer fences the host's epoch `N` in its Runs store and promises
   `N+1` to this computer (`fence_and_promise`, #105079). It answers with a signed **fence
   receipt** carrying its watermark, a digest of its run evidence and, where its operator did not
   opt out, fresh member grants for this computer. Receipts are fences, not votes.
   - A computer that already promised the epoch to another one refuses. The attempt then stops
     with `room_authority_promised`, naming the other computer. An operator's retry moves past
     every epoch seen so far.
2. **Catching up.** The successor adopts the most complete fenced copy. Pages come from the
   custodian that holds it, and each transition in them is verified on the way.
3. **The transition.** The successor writes one `authority.transition`: proof kind `attested`, the
   owner's statement signed with its key and every receipt. It is marked verified in the same
   transaction (#99107). The payload also carries the notice clients render (`text`, `from_name`,
   `to_name`, `offline_since`, `reason`, `at_risk`), outside the proof.
4. **Reconciling, then finishing.**
   - The successor records itself as authority in the next `custody.configured`. The old host
     stays a custodian and a successor (a voter when always on), so the owner can move the group
     back once it is online again.
   - It classifies inherited work, takes ownership, and registers its members' routes from the
     grants.
   - It announces the transition to every computer. Each one verifies the proof against its own
     copy before it follows: the successor was eligible and signed, its own receipt and every
     other one are genuine, the replaced host was the configured one, and nothing more complete
     was left behind.

A copy that holds more of the old host's events than the successor adopted sets them aside as a
separate branch when it learns the move. Nothing is merged.

## Accepted work

Outside majority mode, dispatch never waits for copies: a tail no eligible successor holds is
reported as at risk, and clients keep unsent messages in their outbox.

Every dispatch decision is announced with `task.admitted` (#104601). The successor classifies
each admission without a published outcome, using the run evidence in the receipts:

| State | Meaning | What the successor does |
| --- | --- | --- |
| `completed` / `elsewhere` | The participant ran it, or is running it. | Observes that run with its new grant. Status and Stop passed to it with the fence. Then it publishes the outcome. |
| `unknown` | No evidence either way, or it was for a Bot on the old host, where it may have run. | Keeps it as indeterminate work, never "never ran". Today's controls apply. |

A Bot local to the group's original home runs only there. On any other host, a new turn for it is
deferred with `turn.deferred {reason: "waiting_for_host", resource, host_name}`, proof that it never
ran, so the room's next turn proceeds; it takes part again once the group moves back.

At the participant, a successor's dispatch of a task it already admitted under an earlier epoch
re-attaches to that run (`room_task_inherited`) instead of running it again. A new generation runs
only after every earlier attempt ended without success.

A participant keys a hosted member session to the room's original home, so a successor continues
the same conversation. A room that never moved keeps exactly today's session id.

## The old host returns

The old host asks the group's computers about its epoch, at start and whenever a peer refuses
its work. It pauses at once while another computer holds a later promise, and executes and
appends nothing. Once it sees a verified transition out of its epoch, it steps down
(`demote_to_custody`), in one writer transaction:

- its events after the shared history move into a branch, readable with
  `groups.succession.branch_log` (`session:read`, as the room's owner on this computer or its
  operator);
- the shared prefix becomes a copy that follows the new host;
- its driver, link and policy state goes.

It then catches up, verifying the transition itself, and reports what it did while cut off. The
new host keeps that report beside its own reconciliation. The old host executes nothing from the
branch.

A host can also be paused by a promise nobody keeps: the computer its next step was promised to gave
up halfway. After twice the careful window (6 minutes), the host continues past that step by itself,
but only when every computer that could have continued the group (each eligible successor, and the
computer the step was promised to) answers that it hosts nothing, holds no later transition, follows no
later host and promised no later step to anyone else. It then fences every computer it reaches at a
fresh epoch, and each of those computers must promise it (in majority mode a majority of the voters
too), so no step anyone took, or might still take, runs beside it. It writes a transition attested with
`RECOVER_TEXT` and `stalled: {install_id, epoch}` (`reason: automatic`), and every computer checks
those receipts before it follows. If any of them can't be heard from, the host stays paused
(`paused.reason: step_not_taken`, `waiting_for` naming them) until they answer, and the owner may
continue it anyway.

The old host keeps a copy from then on. It records its consent to a copy-only grant for every later
host, sends the new host that grant with its report (and any later candidate with its fence receipt),
and keeps its own consent to continue the group unless its operator said otherwise. The new host
saves the grant as a custody route and pushes the history to it like to any backup, with its lease
requests when the old host votes; the owner can move the group back to it.

## Continued on two computers

Two hosts can hold the group at once: a partition and two moves (two taps, a careful move while the
host kept writing, or "continue anyway" racing an automatic move). The first contact between them,
directly or through a copy, records `continued_on_two` on both. The group keeps running on one of
them by a rule every computer applies the same way: the higher epoch; on a tie `certified`, then
`evidence`, then `attested`; then the lower installation id.

- **The host the rule keeps** keeps serving. On a tie it continues at a fresh epoch, signed as the
  rule's choice (`decided_by: "rule"`, which anyone can check against the two transitions), so every
  computer can follow it whichever host it followed. Every copy follows it; a copy that followed the
  other host sets its extra events aside.
- **The other host** stops serving at once (sends are refused with `room_authority_conflict`) but keeps
  its room and its own messages, and its Bots take the kept host's work meanwhile.

Status on both shows `conflict {hosts, start, end, running_on}` until the owner chooses, with no
timeout. The owner chooses with `groups.succession.keep`, from either computer:

- **Keeping the running host** ("keep going") resolves it: the other host steps down to a copy, its
  own transition, messages and verified mark (`move_transition_mark_to_branch`) set aside in a branch.
- **Keeping the other host** switches: the running host steps aside first, then the kept one
  continues at a fresh epoch above every epoch known, with the owner's choice signed by the computer
  it was made on. In majority mode that needs a majority's promises, so the two never serve at once.

## Moving by itself: the host's lease (majority mode)

`gateway/hosted_room_succession_automatic.py` registers the lease layer's hooks when the
succession upkeep starts.

- **The lease.** With each push to a voter (about every 5 seconds) the host asks for
  `LEASE_SECONDS` (20 s). The voter grants it in its fence store (#105079) on its sleep-counting
  clock, and promises no later epoch to anyone while it runs. The host counts each grant from the
  moment it asked, less 1 % for drift and a one-second margin. It admits, dispatches and appends
  only while it holds grants from a majority of every voter set its protection needs now (both,
  while a change of voters is pending), counting itself. A heartbeat loop that stalled for 10
  seconds, or a detected sleep, voids every grant it held.
- **Paused to stay safe.** Without that majority the host pauses (`state: paused`,
  `paused.reason: lost_majority`): sends are refused with `room_host_paused`, the driver gets no
  binding, and `serving_provider` stops custody's own appends. While paused it asks the group about
  its epoch every 10 seconds, so it steps down as soon as any computer tells it of a move.
- **The standby.** Standbys are the other voters, in the owner's order. One that has heard nothing
  from the host for the lease plus 10 seconds, then 10 seconds per rank and some jitter:
  1. asks every computer first, signed and changing nothing, whether it would promise: not while
     it holds the host's lease, not past an epoch it already fenced or promised elsewhere, and not
     with a later configuration than the standby's;
  2. only when a majority would, asks for promises (`vote`, citing its `configuration_seq`; a
     voter holding a later configuration refuses with `configuration_stale`);
  3. with a majority, signs a certificate of the receipts (proof kind `certified`), catches up from
     the most complete promiser, writes the marked transition with `reason: automatic`, and
     finishes like any continuation, as the room's recorded owner there.

  A refusal for another candidate (`room_authority_promised`) or a reachable host
  (`host_reachable`) ends the attempt; the standby backs off and tries again at a later epoch.
  Because it asks before it fences, a standby cut off alone never fences itself.
- **Planned restart.** Before announcing `host_restarting`, a host in majority mode asks its voters
  to extend the lease until `until` (at most five minutes). Nothing waits for it: a voter that
  didn't hear in time lets the lease run out, and the restart window keeps the standbys waiting.

## Careful moves for two voters

With exactly two voters the standby takes over only when all of these hold:

- it has heard nothing from the host in either direction for 180 seconds: neither the host's
  pushes nor its own probes, every 10 seconds, nor a fence request or answer. A copy that starts
  following a host (after a move, stepping down or going back) counts its silence from then;
- its own online check passes: a TCP connection to a public endpoint it already uses (its model
  provider, or a messaging platform it runs), nothing new contacted. Loopback, link-local and private
  addresses (RFC 1918, ULA, 100.64/10) don't count, so a local model server can't make a cut-off
  computer look online;
- no `host_restarting` window is open.

It then fences every computer it reaches, catches up, and signs the evidence (proof kind
`evidence`: `{room_id, from_epoch, to_epoch, successor, last_seq, last_hash, silent_since,
silent_for_s}`), with `reason: automatic`.

The host pauses itself (`paused.reason: isolated`) after 90 seconds without the standby **and** a
failed online check of its own, and stays paused across a restart until it hears the standby again.
When the two meet again:

- a host that wrote nothing after the split becomes a copy, silently;
- a host that kept writing records `continued_on_two`, and so does the new host when the old one
  reports it. The new host keeps running (its epoch is higher); the old one keeps its messages apart
  until the owner chooses. `conflict.start` is the evidence's `silent_since`, `conflict.end` when it
  was found.

On the new host after a careful move, `moved_in` names the old host, and the owner may go back:
`groups.succession.keep {install_id: old host}` there steps the new host down at once, like keeping
the old host after `continued_on_two`. Both stay until the old host is a copy again.

After a move made by the owner (`attested`), a returning host's later messages are set aside
instead (above): the owner already chose.

## Continuing a paused host anyway

`groups.succession.continue_anyway {room_id}` on a host paused to stay safe, for its owner:
it fences every computer it reaches at a fresh epoch and writes a transition attested with the
owner's statement (`ANYWAY_TEXT`). A computer that already promised that step to another refuses,
and the call fails with `room_authority_promised` naming it; a host paused for a step that was never
taken (`step_not_taken`) continues past that step. The host then serves without a lease
until a majority answers again. On a backup in majority mode, **Continue on…** is offered only when
no majority is reachable, with the caution `voters_unreachable`.

A host paused because its lease layer isn't running (`no_lease_layer`) holds no lease, so continuing
it anyway also turns automatic moves off for the group: it writes a configuration with
`automatic: false` (ask mode), which every standby gets with the next push, so none moves the group by
itself beside it. Status offers it as `{action: "continue_anyway", turns_off_automatic: true}`, and the
owner turns automatic moves back on with `groups.custody.automatic`.

## Handing a group over on purpose

A gateway that stops or quits without restarting, `groups.succession.move {room_id,
target_install_id}` on the host (the owner), and Desktop's sleep hook
`groups.succession.handover_all {reason}` (the operator; moves only the groups the caller owns and
returns `{moved, skipped, reason}`) all hand over the same way:

1. The host stops admitting (`moving.reason: handover`) and lets the turns already running settle,
   publishing their outcomes into the history it hands over: up to 20 seconds on a stop, about a
   second per group before sleep.
   - The owner's `move` returns at once. With turns still running, status shows
     `moving {step: "waiting_for_turns", running}` and the owner's action `{action: "move_now"}`;
     upkeep hands over once they settle, after 15 minutes at most, or at once on
     `groups.succession.move_now {room_id}` (`session:control`, the owner).
2. It signs `{room_id, from_epoch, to_epoch, successor, last_seq, last_hash}` with its room
   identity key (`hermes.group.succession.handover.v1`, proof kind `handover`). `last_hash` is the
   custody chain hash through `last_seq`. Beside it, never in the log, it signs a release token
   `{room_id, from_epoch, to_epoch, successor, last_seq, signed_at, boot}` with its own
   sleep-counting clock and boot (`hermes.group.succession.handover-release.v1`). From then on it
   counts no lease grant it asked for earlier, and it asks for none while it hands over.
3. The standby catches up to exactly that history, checks it against the host's pinned key, fences
   every computer it reaches, and continues with `reason: handover`. Turns still running at the
   signature are counted in the transition's `at_risk` and inherited as `unknown`.
   - Each voter gives the host's lease back only for a valid token, when the host last asked for
     that lease before `signed_at` in the same boot and its own copy isn't past `last_seq`.
     Otherwise the lease runs out by itself: a token kept after a failed handover can't release a
     lease the resumed host serves on.
4. The host steps down to a copy. Nothing is set aside. It stays a successor, so the owner can move
   the group back.

On any refusal the host resumes and keeps the reason in `last_attempt`. A handover is never
completed without the signed statement. A restart in the middle resumes the host when it hadn't
signed yet, or when the standby answers that it holds nothing beyond the host's epoch; otherwise the
host waits to learn the outcome.

## Ending a split from any device

- **`groups.succession.learn {room_id, events}`** (`session:read`, as the room's owner on this computer
  or its operator): the `authority.transition` and `custody.configured` events after this computer's
  epoch, in log order with the event before the first transition, as `groups.log` returns them (a
  missing `authority_epoch` is derived). The first
  transition out of this computer's epoch is verified with pinned keys against its own history; a
  forged, replayed or unverifiable chain changes nothing. A host it supersedes steps down, or
  records `continued_on_two` after an automatic move it wrote past.
- **Refusals carry the news.** A voter refusing an older epoch's lease, and a backup refusing an
  older epoch's fence, return the verified chain that superseded it (`events`).
- **Catching up.** A host that stepped down catches up from the new host, or from any other
  computer of the group when the new host can't be reached.

## Status for clients

`groups.succession.status` (`session:read`, as the room's owner on this computer or its operator)
answers from this computer's own records. Every field is a code or a parameter, and times are Unix
seconds:

- `state` is `ok`, `paused`, `host_unreachable`, `host_restarting`, `moving`, `continued_on_two` or
  `moved_away`;
- the rest describes the host, this computer, the backups (with readiness, `voter` and
  `always_on`), the tail at risk, any move (`moving.reason`: `manual`, `automatic` or `handover`),
  conflict (`hosts`, `start`, `end`) or set-aside branch, inherited work and the owner's actions;
- `automatic`: `{mode, state, standby, voters, enabled, pending, reason?, offline?, needed?}`.
  `state` is `ready`, `not_ready` (`reason: voters_offline`, `offline`), `unavailable`
  (`reason: needs_computers`, `needed`) or `off`. `enabled` is the switch as the configuration holds
  it; `pending` the value the owner asked for while that change settles with the voters, else `null`;
- `conflict.running_on`: the host the group keeps running on while `continued_on_two` is shown;
- `moving`: `{to, step, started_at, reason}`, with `running` while a move waits for its turns
  (`step: "waiting_for_turns"`);
- `backups[].readiness`: `caught_up`, `behind`, `offline`, `unknown`, `unsupported`, or on the host
  `needs_reauthorization`: its copy to that computer is refused for lack of permission, for example a
  grant that lapsed while the host couldn't reach it, until the grant is accepted again or the owner
  adds that computer again. That computer can't take a move or vote meanwhile;
- `paused`: `{reason, since, waiting_for}` on a host paused to stay safe. `reason` is
  `lost_majority`, `isolated`, `no_lease_layer` (its lease layer isn't running, so it can't take
  part in automatic moves) or `step_not_taken` (its next step was promised to a computer that never
  took it, and `waiting_for` can't yet confirm that nothing else happened). Until that wait is over,
  such a host shows `moving` to the computer the step was promised to;
- `unavailable_reason`: why no continue is offered: `not_owner`, `no_successor`,
  `successor_behind_offline`, `host_reachable`, or `takeover_waiting` while a reachable majority
  should move the group by itself. Five minutes into `takeover_waiting`, `continue` is offered too;
- `moved_in`: `{from, at, proof_kind}` on a new host after an automatic move or a handover, until
  the old host is a copy again;
- `unavailable_bots`: `[{member_id, name, on: {install_id, name, reachable}}]` on a new host: Bots
  that run only on another computer (a local Bot on the group's original home, a peer member on the
  old host), and whether that computer has answered lately. When `reachable` is true and
  `actions[move].targets` lists it, clients offer **Move back**, a planned handover.

`actions[continue].targets` and `actions[move].targets` come best placed first: always on, then
readiness, then the owner's order. `move` lists the successors that hold a verified copy and
answered lately, the old host too once it is back as a copy. On the host, a custodian that has
acknowledged no push for 30 seconds (a voter) or 3 minutes (any other) counts as offline. The
host's owner also gets `{action: "move", targets}`, `{action: "automatic", enabled}`
(`groups.custody.automatic`), `{action: "designate", targets}` with a switch for every computer that
keeps a copy, `{action: "move_now"}` while a move waits for its turns, and on a paused host
`{action: "continue_anyway"}` (with `turns_off_automatic: true` when its lease layer isn't running).

The host counts as offline only after a bounded window without an answer (five minutes of signed
heartbeat queries), never after one missed poll. The best-placed eligible computer then tells the
owner once per incident, through the owner's private chats when messaging offers them, otherwise
through the home channel. Another eligible computer waits five minutes and speaks only while
every better-placed one still looks offline from it.

Errors are JSON-RPC `4001` with `error.data.reason`. `room_authority_promised` adds `data.other`,
and `target_not_local` adds `data.target`.

The host appends a quiet `succession.state` event on every change. Clients connected only to a
backup poll the status every 15 seconds.

## Catching up from any custodian

When the host is gone, the most complete copy may be anywhere.
`fetch_custodian_pages(db, room_id=, source_install_id=, after_seq=, limit=)` asks another
custodian for a page of its history at `POST /v1/room-members/custody/pages`. The request names
both installations, a nonce and its issue time, and is signed with the requester's room identity
key (`hermes.group.custody.pages.v1`). The source answers only an installation its own
configuration lists, within five minutes of issue, and signs its reply over the nonce
(`hermes.group.custody.pages-reply.v1`); the reply relays the head that vouches for its history.

`catch_up_from_custodian(db, room_id=, source_install_id=, head=None)` stores only what a head the
host signed vouches for:

- the head must be signed by the host this copy follows (the range then holds no change of host),
  or by the successor of a change of host that is the very first event fetched, which the copy then
  verifies; signatures check against keys pinned before the range;
- pages are fetched from the copy's last event up to the head's `seq`, never further, so any
  unvouched tail is dropped;
- the chain over them must reach the head's `chain_hash` before anything is stored, so keys,
  configurations and voters come only from inside a host-signed prefix;
- on any mismatch nothing is stored.

Catch-up resumes from the copy's own watermark, from any custodian, and never needs the host.

## Retiring copies after a move

A participant's operator can enroll the retirement of its copy with the room's first home
(`groups.replication.prepare` there, `groups.replication.enroll` here); Disband then retires the copy
with a notice signed by that enrollment's key. Once the group moves, that home's notice is refused:
the copy no longer follows it. The host the copy follows now inherits the obligation instead. The
participant's probe reports the enrollment to the copy's current verified authority (the successor
named by the latest marked transition in the copy's own log), that host's Disband closes it, and its
notice is signed with its room identity key (`hermes.group.replica.retirement.authority-notice.v1`)
and checked against the key the copy pinned for it. Every other check stays: the current active
enrollment, the destination installation, the copy's namespace and roster, a read-only signature
check before the writer and again inside it, and idempotent receipts.

## Reading a copy

`groups.list`, `groups.state` and `groups.log` show a copy as a read-only room (`copy: true`,
revision 0, no driver) to the installation's operator or to the room's recorded owner there.
Nobody else sees it, and a copy is never written to.

## One Gateway compatibility

- **Installation scope.** `install_id` and room identity keys come from the installation root, so
  per-profile gateways and one multiplex gateway are the same custodian, with one copy.
- **Room authority is not session authority.** Custody moves no sessions; a move changes only who
  orders the room's log.
- **Room logs keep their continuous prefix.** A copy is a hash-chained, contiguous prefix of the
  room's log within the room's budget. The relaxations that session replay may adopt ("a valid
  replay or an authoritative snapshot") don't apply to room logs.
- **Mixed versions.** An installation without room identity keys or watermarks is `unsupported`:
  shown, never counted, never a voter. A copy whose Hermes has no succession endpoints is never
  assumed fenced: `prepare` cautions `participant_not_fenced` with its name and count.
- **A successor's re-dispatch is not a second owner.** It maps onto the participant's existing
  admission (`room_task_inherited`), and each Bot's conversation stays with its own installation's
  canonical session authority.
- **A restart through Hermes is not loss.** Before restarting in place (`SIGUSR1` or
  `request_restart`, as `hermes gateway restart` and `/restart` do) the host announces
  `succession.state {state: "host_restarting", until}`. Backups report `host_restarting` until
  `until` plus a grace period, offer no continuation and send no notice. A restart from outside
  Hermes (`systemctl`, `docker`, `launchctl kickstart -k`, `--replace`) arrives as `SIGTERM`: the
  gateway stops, so it hands its groups over first.
- **Scopes.** Continuation grants, fence receipts and succession controls stay room-scoped: never
  installation-wide operator rights. Peers are found only among the room's recorded custodians.

## Limits

- Careful mode doesn't prevent a split: if both computers stay online but can't reach each other for
  3 minutes, both run until a device that reaches both passes on the move, or they reconnect.
- Outside majority mode dispatch doesn't wait for copies: messages after `at_risk_after_seq` are lost
  if the host never returns, and work it dispatched in that tail is unknown to the successor.
- Catch-up from another custodian crosses a change of host only as the first event it fetches. A
  copy further behind first catches up to the old host's last head (a custodian keeps one per epoch,
  `heads_locked`), or follows the new host's own pushes, which relay the old history first.
- A copy started after the room moved learns its first host from the history it replays; until
  its copy reaches the first move, its own gateway-actor checks are by consistency within each span.
- A copy-only grant renews only while its host reaches the custodian: one that runs out unreached
  needs a fresh grant (the owner adds the backup computer again).
- Not built: a content-free referee vote, an automatic move back to the original host when it
  returns (the owner can move the group back), and votes from laptops.
- After a move, Bots that ran on the old host stay unavailable until the group moves back; Hermes
  offers **Move back** once that computer is online. Re-enrolling them as peers of the new host is
  future work.
- After a stop, sleep or **Move now**, turns that were still running show as unknown on the new host
  and never rerun by themselves; the owner can Retry them.
- A host whose lease layer isn't running (`paused.reason: no_lease_layer`) pauses its groups in
  majority and careful mode; its owner can still continue it anyway, which turns automatic moves off
  for the group until the owner turns them back on.
- A standby's attempt that a voter refused halfway can leave that voter's promise of the next epoch
  standing: it can no longer grant the host a lease until the group moves on. Asking first makes this
  rare; the host keeps its majority from the other voters, and the owner can continue it anyway.
- In majority mode, right after a host starts, its sends are refused (`room_host_paused`) until its
  first heartbeat round gives it a lease: about a second while the voters answer.
- The careful online check uses only public endpoints this computer already talks to: its model
  provider, and Telegram, Discord, Slack or WhatsApp. A computer with none of them never moves
  carefully.
