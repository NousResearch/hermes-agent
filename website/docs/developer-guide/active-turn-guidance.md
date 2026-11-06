# Local active turn guidance contract, version 1

This local extension targets ordinary in-process ACP execution on gateway base
`79658be93ba36dc0ff5878b07b6a4c5df29dd79d`. The direct-ACP prototype is on a separate
historical branch. Neither patch is an upstream release.

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

`_ai.nous/steer` accepts `runtimeInstanceId`, `sessionId`, `expectedTurnId`,
`messageId`, literal `text`; it forwards to `session.guidance.admit` with snake_case
parameter names. Receipts echo those identities (`turnId` replaces `expectedTurnId`),
plus `disposition`, `state` and acceptance `sequence`. Admission requires session
control permission. Text is nonblank UTF-8, at most 262144 bytes; message IDs are
nonempty strings up to 128 chars. No prompt.submit, slash dispatch or future queue.

`accepted_active` starts `pending_incorporation`. Actual iteration preparation
appends accepted text as one user row in sequence order after an assistant/tool
row, preserving the cached prefix. Only then does it become `incorporated`.
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
for the companion Multica Go/HTTP/PostgreSQL tests, controlling only loopback model
responses and lost ACP replies. It substitutes neither owner nor conversation loop.
Multica's `docs/hermes-active-turn-guidance.md` contains the reproduction recipe.
