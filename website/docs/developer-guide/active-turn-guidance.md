---
title: Active-turn guidance
description: Negotiated ACP guidance with owner-correlated incorporation receipts
---

# Active-turn guidance

This opt-in ACP extension delivers literal text only to the expected active
turn, with correlated admission and incorporation receipts. It does not replace
`session.steer` or queued-admission controls: clients that cannot safely replay
uncertain input need an active-only operation and a separate status lookup.
Only ordinary in-process gateway-backed ACP turns support this contract;
managed workers and non-ACP turns do not advertise it.

ACP clients request `_meta["ai.nous/active-turn-guidance"] = {"version":1}`.
The viewer advertises that key with `version:1`, `statusLookup:true`,
`literalText:true`, `terminalReceipts:true` only after the authenticated owner
advertises `active-turn-guidance-v1`. Legacy initialize does not boot an owner.
The prompt metadata under that key must be an object with a nonblank `promptId`
string up to 128 characters; invalid values fail before submission. Absent metadata
retains legacy prompts. No version string enables this extension.

The owner creates the inbox atomically with the primary claim, binds it to the
executing agent and closes it on every execution exit. Stop closes the same
generation. Non-ACP and managed worker turns have no inbox.
`session.guidance.turn(session_id, admission_id)` returns
`state:started|ended|unavailable`, `runtimeInstanceId` (instance plus authority
epoch), `sessionId`, random `turnId`, admission `promptId` and public `receipts`.
The ACP viewer maps the admission to the host prompt ID. Only an owner-confirmed
started snapshot produces `_ai.nous/guidanceTurn`; primary terminal metadata carries
the ended snapshot. A failed receipt lookup leaves primary output/usage intact.

Start snapshots and extension announcements are optional and bounded to three
seconds. Read, projection or notification failures leave guidance unavailable;
they neither replace a committed primary response nor stop its event pump.
Terminal snapshot failure likewise omits auxiliary metadata. These boundaries
catch ordinary exceptions, not primary cancellation. No failure resubmits input
or invents an acceptance receipt.

When load/resume resolves a saved ACP alias to a canonical owner session, all
guidance RPCs retain the canonical session ID at the owner. The viewer projects
notifications, admission/status receipts and terminal metadata (including nested
receipts) back to its existing held ACP alias, just like ordinary updates. Only
the expected canonical ID is translated: foreign identities remain uncorrelated.
Owner authorization and host correlation checks are unchanged.

`_ai.nous/steer` accepts `runtimeInstanceId`, `sessionId`, `expectedTurnId`,
`messageId`, literal `text`; it forwards to `session.guidance.admit` with snake_case
parameter names. Receipts echo those identities (`turnId` replaces `expectedTurnId`),
plus `disposition`, `state` and acceptance `sequence`. Admission requires session
control permission. Text is nonblank UTF-8, at most 262144 bytes; message IDs are
nonempty strings up to 128 chars. No prompt.submit, slash dispatch or future queue.

`accepted_active` starts `pending_incorporation`. Actual iteration preparation
appends accepted text as one user row in sequence order after an assistant/tool
row, preserving the cached prefix. Only then does it become `incorporated`. If native steering already added a user
row at that boundary, guidance stays pending until another legal boundary; it
never merges into a persisted steer row or creates consecutive user rows.
Ordinary provider normalization may trim outer whitespace; command-like text stays
text. Pending input closes `unconsumed_at_turn_end` or, on Stop,
`cancelled_before_incorporation`. Neither can enter a successor. Incorporation
does not imply model compliance.

`_ai.nous/guidanceStatus` has the same identities without text; the authorized owner
method is `session.guidance.status`, requiring session read permission. Duplicates
recover the same receipt; changed turn/text is `message_id_conflict`. Definitive
refusals use `state:rejected` and `invalid_input`, `rejected_turn_ended`,
`rejected_cancelled`, `rejected_turn_mismatch`, `rejected_session_mismatch`,
`message_id_conflict` or `receipt_capacity_reached`. `unknown_message` and
`rejected_instance_mismatch` cannot prove what a previous owner did. Missing,
future or contradictory receipt values must stay unknown at the host.

Each live session retains up to 4096 accepted IDs across generations without
eviction; each generation has its own engine inbox. Full capacity refuses new IDs.
Reconnect reads receipts without resubmission. Status does not restore execution.
The ledger is in memory for this live session/authority epoch, not crash durable.
Restart or retirement may lose it: the host persists unknown delivery and never
automatically replays input. There is no crash-durable exactly-once claim.

Canonical tests: `tests/gateway/test_active_turn_guidance.py` uses the real
authenticated gateway and AIAgent; `tests/acp_adapter/test_gateway_guidance.py`
uses default stdio ACP. `guidance_gateway_fixture.py` launches an ordinary gateway
for external ACP consumer tests, controlling only loopback model responses and
lost ACP replies. It substitutes neither owner nor conversation loop.
`tests/acp_adapter/test_guidance_boundaries.py` checks auxiliary failure isolation,
cancellation propagation and alias boundaries; `tests/gateway/test_guidance_permissions.py`
checks independent read/control permissions on the canonical dispatcher and SQLite.

`tests/agent/test_active_guidance_with_native_steer.py` exercises the real
iteration preparation with both input sources and verifies prefix stability,
receipt state and exactly one incorporation at the next legal boundary.
