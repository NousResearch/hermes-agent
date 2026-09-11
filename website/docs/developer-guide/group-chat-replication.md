---
title: Participant History Copies
description: Opt-in passive Group Chat history delivery and its recovery limits.
---

# Participant History Copies

A hosted Group Chat has one authority gateway. This opt-in layer copies its ordered history to participating gateways without using Desktop or invoking a Bot. It is a prerequisite for surviving authority-host loss, not automatic takeover.

## Enable Through The Existing API

This initial implementation is API-first. Existing Group Chats and invitations keep their current permissions unless explicitly opted in.

1. Read `groups.capabilities` on both gateways. The complete lifecycle needs `authenticated_replication` and `replica_retirement`.
2. For an existing, locally owned group, call home-side `groups.replication.prepare` with its `room_id`, the participant's `target_install_id`, and its installation-root `endpoint`. Keep the returned public enrollment. This reserves the eventual cleanup obligation without revealing its private closing value.
3. Through the participant gateway owner's connection, call `groups.replication.enroll` with that enrollment. Setup is once per participating installation, shared by its profiles. A Bot's ordinary delegated grant cannot authorize it. The owner-authenticated HTTP equivalent is `POST /v1/group-replicas/enroll`.
4. Add the exact boolean `replication: true` to an ordinary target-side `groups.peer.invite` request, preserving the existing room, authority, epoch, member and profile coordinates. The HTTP equivalent is `POST /v1/room-members/invitations`.
5. Register the resulting scoped grant and catalog on the home with `groups.peer.register`. Do not retain a broad API key in that route. The publisher confirms the installed enrollment through its scoped capability probe before copying.
6. Inspect home `groups.state` under `driver_status.replication` and participant `groups.replica_state` for coverage. Retirement delivery appears separately from Bot execution status.

Legacy unversioned copying supports fixed-roster, initial-authority (epoch 1) groups. Passive history and retirement format 2 additionally support committed authority transitions with a pinned owner-enrolled descriptor and a fresh exact-scope target-issued grant. They do not change the participant roster or prove execution authority. Earlier data-only invitations remain readable in their limited mode; owner-authorized retirement enrollment is needed for automatic cleanup.

Invitation and capability responses advertise a `passive_replication` sibling outside the unchanged RoomLink protocol-2 catalog: `history_versions: [1,2]`, `retirement_versions: [1,2]`, and `work_record_versions: [1]`. Missing or incompatible lineage support blocks later-epoch history without downgrading it to the legacy format. For a non-executing invitation, set `passive_only: true` with `replication: true`; this grants only `status` and `replicate`, plus `work_records` only when separately opted in. It never grants dispatch, Allow or Stop.

Grant values are secrets. Never put them in examples, logs, screenshots or issue reports. Ordinary invitations do not gain replica-write access, and replication does not enable execution or change a Bot's approval policy. Grant refresh retains the original permissions and authorization horizon.

## Delivery And Recovery

- The existing RoomLink client sends UTF-8 JSON over verified HTTPS, with plain HTTP allowed only on loopback. Credentials are not forwarded to arbitrary redirects.
- The recipient verifies its target-issued grant and exact group/member/authority scope. Reservation and revocation are checked again in the same SQLite transaction as the history write.
- Two bounded background workers copy pages independently of Bot turns. Profiles sharing one participant gateway use shared target coverage and an OS file lock, so their deliveries cannot overtake an unresolved page.
- The pending page coordinates are durable before transmission. A lost response or publisher restart retries that page before newer history. Matching duplicates do not create new events; gaps and conflicting history are not silently discarded.
- The original membership remains fixed. Committed rename events can update the copied name; a terminal disband cannot be followed by more events.
- An unavailable participant backs off without holding the Bot driver's locks. Optional publisher failures are reported separately from ordinary Group Chat execution.

Coordination files under `room_replication_locks` contain no credentials. Do not unlink them while services can be running: replacing a locked file can create two independent lock owners. They are local process coordination, not a distributed authority lease.

## Retain Unfinished-Work Records

History alone cannot show every accepted task or pending Stop. An additional, explicit invitation option retains bounded task and receipt metadata on participants. It does not transfer execution rights.

Follow the history-copy and retirement-enrollment setup above. When issuing a new target-side `groups.peer.invite` or `POST /v1/room-members/invitations`, add both flags to the existing invitation coordinates:

```json
{"replication": true, "work_records": true}
```

Work records remain **version 1 and initial-authority only** in this merge/compatibility baseline. A later enrolled sender can copy history, but cannot deliver v1 work records under an old prefix header. The publisher exposes `work_record_status: unsupported_lineage` and leaves old pending payloads, revisions, digests and outcomes unchanged. Existing retained records remain historical evidence, not current post-transfer work coverage; producer-scoped storage and work-record v2 are a separate implementation gate.

Require `work_records_version: 1` in the invitation response before treating this option as supported. The scoped capability probe also reports that version. An older response or a history-only grant does not enable work-record delivery. Existing grants do not silently acquire the new permission; keep a working route unchanged if the target cannot support the requested option.

After registering the returned grant normally, the existing publisher delivers work records once their corresponding history prefix has arrived. Inspect source `groups.state` under `driver_status.replication.work_records`; participant `groups.replica_state` exposes `work_records` with the observed prefix, revision, task phases, receipt coordinates, Stop facts and limitations. This is an API-first surface, not a new Desktop or messaging recovery screen.

The publisher freezes a bounded snapshot and its history anchor before catching up. New conversation messages do not replace that pending snapshot or require the group to become quiet. Once the anchor is acknowledged, an eligible work-record route gets a delivery turn while newer history can continue copying. Lost acknowledgments retry the same revision and digest. If every work-record route is blocked, history can still use a healthy route; this does not remove the block or authorize new permissions.

Task phases can change without a new message, so records use their own durable content revision. A lost acknowledgment retries the same pending record before replacing it. A refused route does not prevent an explicitly authorized alternate from carrying that record; known refusals remain scoped to the route generation. Retirement still rejects late writes and reclaims record payloads without erasing the retired group identity.

Records exclude prompts, result bodies, commands, file bytes, private paths, broad credentials and signing secrets. They retain identifiers and digests, not a Bot's private context or running process. The source snapshot is consistent within its SQLite transaction; it is not a distributed transaction with the participant's live run store.

Limits are 128 tasks, 256 receipts and 128 KiB per record, with a shared 4 MiB/512-row budget for record storage. Unsupported or oversized captures explicitly report unavailable evidence, not a complete-looking truncated list. Core records do not certify the field build's additional approval/retry/input journals. No maximum freshness lag is promised under sustained history growth.

**Available metadata is not permission to resume.** A missing receipt does not mean a task never ran; an uncopied final update can still be lost. These records are passive evidence, not a complete execution checkpoint, an approval decision or a takeover certificate. `source_loss_safe` remains false.

## Disband And Copy Retirement

The existing irreversible disband fence, acknowledged Stop and ordinary-grant revocation remain the execution-safety boundary. Only canonical home disband makes a retirement notice deliverable. The private journal remains available after live routes and retained canonical payload have been removed.

Setup installs a context-bound commitment to a one-purpose closing value. After disband, the home sends that value to `POST /v1/group-replicas/retire` using the dedicated `HermesReplicaRetirement` authorization scheme. It cannot read history, run a Bot, access files, renew a grant or change authority. It remains usable independently of the ordinary grant's expiry/revocation, so cleanup does not need fresh execution permission.

The recipient records explicit local copy retirement, not a synthetic `room.disbanded` event. Partial or empty history remains partial or empty. Late pages and same-ID reenrollment are denied; payload reclamation retains the retirement and namespace facts. Active copies and quarantined evidence are not evicted to conceal capacity problems.

Only the gateway owner can revoke this separate cleanup permission through `groups.replication.revoke` or the owner-authenticated HTTP enrollment-revocation endpoint. Replacing an enrollment requires the exact expected enrollment and state; it cannot reactivate an old closing value. Revoking cleanup does not revoke ordinary Bot grants, and vice versa.

Use an installation-root endpoint, not `/p/<profile>`, so removing a profile does not remove the cleanup route. Stale endpoints, incomplete enrollment, revoked cleanup permission or a lost home key remain visible pending conditions, not successful delivery. A materialized post-disband notice survives a later home-key loss; the implementation does not silently rotate existing commitments or add an old-key ring.

## What A Copy Does Not Prove

`source_loss_safe` deliberately remains false. Acknowledged replica coverage describes copied history, not a quorum-committed room or proof that every accepted source write survived.

The source's unreplicated tail can be lost if it disappears. Accepted execution receipts, pending external effects, cancellation, permission revocation and the policy cursor need a complete recovery contract before any successor can run work. Unknown outcomes must not become automatic retries.

Files are not mirrored by this layer. A copied attachment reference does not prove that its bytes are available on another gateway. Retirement allows authorized cleanup when the final history page did not arrive, but it does not claim to have completed that missing history.

Neither `groups.promote` nor `groups.demote` is enabled here. A timeout, an operator confirmation flag or a higher epoch number is not proof that the old coordinator can no longer commit. Automatic continuation requires a separately verified, globally exclusive authority decision.

## Focused Validation

Use the canonical `scripts/run_tests.sh` runner with the repository's installed test environment. The receiver and publisher tests use disposable SQLite stores, real signatures and reservations, real loopback HTTP, dropped acknowledgments and separate publisher processes. They do not use production credentials or provider calls.

Relevant owners are `test_hosted_room_replica_ingress.py`, `test_api_server_room_replicas.py`, `test_hosted_room_replication.py` and `test_hosted_room_replication_http.py`, plus the existing room, grant, RPC and peer-HTTP regressions. Field acceptance across independently hosted gateway processes remains a separate gate from these tests.

Retirement adds `test_hosted_room_replica_retirement.py`, `test_api_replica_retirement.py` and `test_replica_retirement_disband_rpc.py`: owner authorization, transaction rollback, lost acknowledgments, key loss, partial-copy retention, denied resurrection and the registered disband control path.
