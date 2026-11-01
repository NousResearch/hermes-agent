# Opt-in s6 launch admission prerequisite

The native `S6ServiceManager.register_profile_gateway` API accepts
`launch_admission=True` only with `start_now=False`. Existing registration and launch defaults
are unchanged. The existing native s6 child sentinel now classifies a launch as
supervised even when enrollment is missing/invalid; this is refusal-only ownership
information, never positive enrollment proof. A new slot is built hidden and persistently
down, with an enrollment describing its exact directory identity, native launcher
hash, profile home identity and expected runtime UID. Registration does not alter
an existing slot or adopt a preexisting launcher.

The renderer invokes native admission after its normal UID-drop branch. Admission
requires the exact enrollment and launcher, the actual current child PID reported
by s6, and the same live parent supervisor PID/start/boot identity, executable and
slot working directory. It then execs the ordinary gateway **without `--replace`**;
enrollment gives no authority to displace another gateway.

The original answering gateway's existing local `identify` response can carry
`launch_admission`. It is recomputed from that answering process and the native
enrollment, not accepted from a caller or ambient declaration. It names the launch
profile; the launch-only builder keeps `runtime_profile_binding_verified` false.
The runtime's original `identify` handler may enrich that receipt from its existing
live `SessionAuthorities` registry and canonical `local_bots` names. Only a ready,
non-draining original runtime with an exact matching launch authority can report
`runtime_profile_binding_verified: true`. Its bounded `runtime_profile_binding`
names `scope: observed_served_profiles`, the original runtime instance and each
served profile's canonical name, directory device/inode and native authority epoch.
The registry is read before/after projection; missing/changed/ambiguous/retiring or
unavailable authority stays unverified. More than128profiles yields no positive
partial projection. Saved `served_profiles` status, caller labels and the
supervisor sentinel cannot establish this binding. `behavior_accepted` stays false.
Legacy, missing, invalid or inaccessible enrollment yields no positive receipt.
A verified receipt classifies this launch as supervised `s6`, preserving refusal
by consumers that support manual owners only. This is a live observation, not atomic idle admission or a permission to stop,
replace or activate an owner. It cannot be replayed as a grant; consumers still
need the original current run, applicable profile scope, cancellation and native
idle/holdoff fencing. The set may change after this observation.

This is a cooperative same-UID operator contract, not isolation against malicious
code running as that UID. It is Linux-only and fails closed when proc identity or
s6 status cannot be read. In particular, a root supervisor whose proc executable
or working directory is unreadable after the UID drop remains unsupported; do not
widen privileges to make admission pass. Qualification of that root-to-user path,
startup/successor fencing, atomic idle drain, durable hold/recovery integration and
independently installed recovery is still required before activation. Existing
idle-drain contribution144790 is separate and must not be duplicated or treated as
a maintained runtime pin before canonical merge/release qualification.

`scripts/e2e_s6_launch_admission.py` exercises private real s6 registration and the
actual gateway control boundary. Use a compatible existing Hermes interpreter,
a fresh private short fixture path, and no owner credentials. It uses installed
`S6_KEEP_ENV=1` to keep its cleared environment rather than importing the host
container environment. The foreground competitor explicitly uses the documented
`--no-supervise` route, never the container's global service redirect. It requests
no model turns and does not establish user-facing activation or production proof.

`scripts/e2e_s6_served_profile.py` exercises one native default multiplexer serving
an eligible named profile. A new standalone named gateway is refused by current
native policy and is not a workaround; the journey does not use `--force` or a
standalone compatibility shim. Markerless homes are not profiles and must not gain
bindings. Its preflight proves private service/log/socket targets, cleared
environment and `S6_KEEP_ENV=1` before any supervisor is started. A same-UID Linux
open-directory alias may keep a socket path short while resolving all files into
the private fixture. No shared `/run/service` lifecycle operation or logger target
is allowed. This supports the served-profile prerequisite only; root-to-user,
new-owner fencing, method activation and production remain separate gates.

`scripts/e2e_s6_dynamic_authority.py` uses the same private fixture and native
control socket to unserve and re-serve its named profile, then stop and start only
its enrolled private slot. It checks that removal excludes the old authority,
re-serving preserves the home identity while advancing its epoch, and a successor
under the same supervisor/enrollment has a different launch/runtime and newer
profile epochs. Each lifecycle operation is issued once; pending replies are
reconciled by observation instead of replay. No model turn is requested.

These comparisons qualify observation freshness, not enforcement of a stale-token
refusal: there is no activation token in this contract. A saved receipt cannot
substitute for the current answering runtime. Retiring live writers, an observed
draining window, overflow, atomic idle admission, root-to-user enrollment and
shared activation remain separate qualification gates. The journey does not
manufacture those states or duplicate the native idle-admission contribution.

## Bound idle consumer prerequisite

The local integration carries the existing native `pause-if-idle` implementation
from upstream PR144790 (`62e405383bf7de73f9adbeec2e2613bd94501587`). It is an
unmerged dependency, not an installed runtime pin. The ordinary updater's
`pause-if-idle` contract remains unchanged.

An original runtime advertising `launch_binding_idle_protocol: 1` also exposes
`pause-if-idle-bound`. Its parameters are `expected_binding` (the exact original
`identify.launch_admission` object) and optional update `target`. The handler
recomputes the original kernel enrollment and complete live served-authority set
on the gateway loop, compares every JSON field/type, then calls the existing
native idle decision in that same loop step, with no await in between. Missing,
unavailable or different evidence returns `pausing: false`, `binding_refused:
true`, protocol1 and a fixed `binding_status` of `unavailable` or `mismatch`.
Matching evidence reports `binding_status: matched`; only the native idle decision
can report `pausing: true`. A matched but busy/stopping response is not acceptance.
A timeout/unknown response is not permission to replay or publish method bytes.

This is a cooperative same-UID freshness precondition, not authentication, a token
grant or an independent supervisor hold. Callers must use the bound verb and
verify its protocol/result; they must never fall back to the legacy updater or
`pause-for-update` when binding is required. It does not install an Arwen recovery
adapter, permit shared activation, or clear the held original effect journal.

`scripts/e2e_s6_bound_idle.py` drives genuine missing/partial/wrong-profile,
removed/re-served-authority and successor refusals through the original private
native control boundary. Every refusal must leave the same current runtime ready
with the same complete binding. It deliberately requests no matching pause or
model turn. Matching admission, busy-work/draining/retiring/overflow, installed
independent recovery and full activation remain separate qualification gates.

## Private positive admission and recovery journey

`scripts/e2e_s6_positive_bound_idle.py` reuses the real private multiplexer
fixture. It applies s6's supported `-O` (once at most) to that private slot,
verifies the original child remains up with identical binding and wanted-up
false, then issues exactly one matching `pause-if-idle-bound` request. It requires
protocol1, matched evidence and actual native pause acceptance, observes the
original kernel identity gone and the slot held down for two seconds, and
explicitly brings up only that same private slot. Recovery must produce a new
launch/runtime and newer epochs for both original homes while preserving their
identity/config and enrolled launcher bytes and modes. The existing persistent
`down` file is not removed. No global gateway/service CLI or model turn is used.

The native `detached=False, via_service=True` path exits with service-restart
status; this journey does not launch a detached helper. The supported s6 command
is documented at https://skarnet.org/software/s6/s6-svc.html. This is a finite
same-supervisor hold observation and disposable process recovery. It is not
supervisor-loss recovery, independently installed host recovery or a method
transaction hold/release adapter. A wrapper that rewrites the enrolled launcher's
bytes invalidates its native hash proof and cannot be silently adopted. Genuine
busy/stopping and restart-wait ticket behavior need real owned activity; this
zero-turn journey neither manufactures pending work nor claims those gates.

## Opt-in original owner-journal launch guard

New offline registrations may supply `launch_guard={"path": absolute_path,
"owner": expected_document_owner, "lock": absolute_original_lock_path}` along with native admission and start_now=false.
The canonical private journal must already exist directly in the selected home,
be a bounded regular single-link mode0600 file of the runtime UID, and carry that
owner. The reference is part of native enrollment and its original identify
receipt. Defaults and unenrolled launches are unchanged; no environment key or
model tool is added.

Before exec, the after-UID native gate revalidates its kernel/supervisor/slot
identity and reads the original journal. An optional `launch_hold` uses owner
`hermes-s6-launch-guard/v1`, the exact slot, SHA256 of enrollment and state held or
released. Held waits without opening a gateway; release permits normal native
exec. The existing private original-owner lock is held shared across the final read and exec; reservation writers take it exclusively. A held gate drops the lock before waiting. Lock contention emits only a fixed category and its own PID. Missing/invalid/unsafe journals or unknown holds (including an explicit null hold) refuse. Native only reads
this owner journal; the concrete owner adapter controls its transaction lock,
reservation ordering, original cancellation, exact-file recovery and release.
It adds no task/hold store, expiry, release command or launcher rewrite.

This remains cooperative same-UID control. An owner may not silently replace or
remove a held original journal. Process interruption and private service proof do
not establish power-loss durability, independent host installation/recovery or
shared activation. Canonical maintained release qualification remains required.
