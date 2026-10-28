---
title: Private invocation prototype v1
---

# Private invocation prototype v1

This is proposed source, not an advertised or released Hermes API. It extends
PR124699's handler-context lease for a trusted integration continuing an admitted
native Slack conversation into one exact authenticated MCP action. Its local
companion depends on that PR's immutable `f18cf58336b8a4528b3726e286c094a342e9b080`
(replayed with original authorship onto the selected main base). No bundled
third-party integration or production activation is included.

## Plugin surface

A trusted plugin registers an `Issuer` through `ctx.invocations.register(issuer)`.
Its existing `plugins.entries.<plugin>.mcp_allowlist` still gates servers.
`ctx.invocations.current()` returns a lease only inside the stock native,
authorized, routed `post_gateway_admission` callback. Internal events, imported
source dictionaries, restored history and unrelated tasks cannot create one.
The callback's handler lease closes on return, error, timeout or cancellation.
An installed Python plugin is trusted inside the process; this API does not
sandbox malicious plugin Python.

```python
lease = ctx.invocations.current()
if lease is None:
    raise PermissionError("native admitted origin required")
ref = lease.retain(job_id=job_id, actions=["call"], expires_at=expires_at)
result = lease.call_mcp(server, "call", {"calls": {"path": path, "input": value}},
                       operation_id=operation_id)
```

`expires_at` is integer epoch seconds, at most 24 hours from retention. Retention
captures the original origin and configured recipient/app/build/account selection;
it cannot widen or renew a prior delegation. A job ID is unique within the issuer.
The private locator has no public bearer export or serialization method. A trusted
host recovering its own job calls `issuer.retained(job_id)` to load its existing
locator, then, inside an owned registry handler execution:

```python
lease = ctx.invocations.rebind(ref, operation={
    "server": server, "action": "call", "operationId": operation_id,
})
```

Every rebind, issue, redemption and forward checks the original scope with current
provider policy. Expiry, revocation, account/build/config changes or absent retained
state deny new effects. `issuer.revoke(ref)` prevents subsequent use, including
forwarding a redeemed permit. Hangup does not implicitly revoke accepted work.
The current source covers Slack only; missing native workspace/message/route
identity refuses. Other native transports need their own reviewed variants.

## Trusted configuration

Construct `hermes_cli.private_invocation_store.Issuer(path, namespace, targets,
policy, peer_principals, forward_context=...)` in trusted plugin/operator code.
Use the owning profile's private plugin data directory for SQLite; protect the
parent directory, original database and WAL/SHM files together. No behavior
configuration or identity comes from environment variables. No issuer starts
merely because a plugin or catalog is discovered.

`targets` is operator-owned and keyed by the granted MCP server name. Each entry
pins `recipientPrincipal`, `appId`, `buildId`, `toolPath`,
`selectedAccountsDigest`, `action: "call"`, `bindingId`, and `forwardUrl`.
`selectedAccountsDigest` hashes `{appAccountId,userAccountId}` (null is allowed).
The receiver must independently resolve its actual authenticated principal and
selected accounts/app/build before redemption. Static target configuration binds
the recipient and never supplies the messaging origin.

`policy(original_origin, configured_target, action)` returns literal `True` only
after bounded synchronous validation of the original connector/profile and current
scope. A saved source fingerprint alone is insufficient. This trusted policy must
also be available to the separately running issuer process after restart. It must
not perform remote effects or long network I/O while the short journal transaction
is held. Policy errors refuse. The source does not supply private integration policy.

`peer_principals` maps SHA256 of actual DER client certificates to independently
configured principals. Inbound TLS requires client certificates; outbound fixed
HTTPS uses an operator-provided verifying client TLS context. No caller controls
issuer URLs, trust roots, accounts or forwarding destinations.

## Exact carrier and issuer protocol

The negotiated initialize experimental descriptor must be exactly:

```json
{"privateInvocation":{"version":1,"header":"Private-Invocation","directCallOnly":true}}
```

The owned modern streamable HTTP client requires pinned MCP2.3.0 and
HTTPX2 2.15.1. It sets one opaque ASCII reference (at most
512 bytes) only on its matching outgoing `tools/call` HTTP request. The normal MCP
SDK sender and result renderer remain in use. The RPC lock serializes attachment;
final tool arguments are hashed and independently matched against the actual
request body. Static headers, redirects, unsupported transport/descriptor,
protected programs, batches and approval resumes refuse. Private requests bypass
transport recovery retries. Ordinary unprotected calls keep their existing path.
An app that requires protection must enforce that requirement at the receiver.

`digestVersion` is `rfc8785-sha256-v1`. This prototype implements the exact RFC8785
integer-only I-JSON subset: Unicode scalar strings, UTF-16 key ordering, booleans,
null, arrays, objects and safe integers. Floats, unsafe integers, lone surrogates
and other Python types refuse; there is no default JSON-sort substitute. Digests
are lowercase SHA256 hex over canonical UTF-8. Receivers may support a wider
RFC8785 domain but must agree on this subset.

The private endpoint is `POST /private-invocation/v1/redeem` over configured mTLS:

```text
{version:1, reference, binding:{digestVersion,recipientPrincipal,appId,buildId,
 toolPath,accountsDigest,action:"call",inputDigest}}
→ {version:1,invocationId,operationId,expiresAt,lease}
```

The issuer derives namespace/invocation/operation/expiry from its durable opaque
reference. Wire `expiresAt` is epoch milliseconds, at most 60 seconds from issue.
All bindings are immutable. Redemption reserves atomically; repeat redemption
refuses and retains uncertainty. It does not infer remote absence.

`POST /private-invocation/v1/forward` receives
`{version:1,lease,binding,bindingId,input}` with the same receiver binding. It checks
current original policy, actual mTLS peer, expiry and exact input digest again,
then durably reserves before one fixed HTTPS dispatch. The configured destination
receives private `{version:1,binding,origin,input}` and returns exactly `{result}`.
The endpoint returns `{version:1,result}` to the same authenticated caller. A
returned result is an observation, not independent proof of business completion.
Loss after reservation remains unknown and cannot be retried automatically.
Cancellation does not claim undo. No bearer, permit or origin belongs in public
app schemas, results, models, traces or logs.

## Real conformance entrypoint

Run from the modified repository in a separately provisioned development fixture:

```bash
python -m hermes_cli.private_invocation_conformance \
  --fixture fixture.private.json --output observations-private --seconds 60
```

The private manifest has exactly `home`, `gatewayPython`, `issuerFactory`,
`journal`, `cgroup`, `readbackUrl` and `readbackTlsFactory`. `gatewayPython` is the
owned development interpreter with the declared full native gateway dependencies.
The trusted issuer factory returns `(issuer,address,TLSContext)`. The readback TLS
factory supplies only the authenticated TLS context; the driver performs an actual
independent HTTPS GET to read the app effect, refusing redirects. Supply the real paired modified receiver and the
native plugin policy/consumer in the disposable gateway home. The driver starts
the actual Hermes gateway and actual issuer in separate processes; genuine humans
send native Slack events. It refuses a missing or nonempty owned cgroup or limits
over 1 CPU, 450 MiB and 12 tasks. It never provisions resources or fabricates
native events. It records private stdout/stderr and durable observations; exit2
means the full acceptance matrix remains unqualified even after observations.

Required behavioral qualification includes concurrent origin/task/account isolation,
negative/tamper/replay/expiry, retained restart/rebind/revocation, lost-write
reconciliation and result/model/trace confinement, plus ordinary-call compatibility.
No positive behavioral execution, model call or acceptance is claimed by this source.
The paired downstream worker catalog still needs an explicit direct-call adaptation.
