---
sidebar_position: 8
title: "Kubernetes"
description: "Running the agent's shell commands in a Kubernetes session pod"
---

# Kubernetes terminal backend

The `kubernetes` backend runs each agent shell command in a **session pod** instead of inside the Hermes container, isolating commands from the Hermes process, its ServiceAccount token, and the agent container's filesystem.

This page is self-contained: every object you need is on it, as YAML you can copy.

:::info The session pod is an execution boundary, not a secrets boundary
Registered credential files and skills **are** synced into the session pod on session start, because skills need them. Only the ServiceAccount-token isolation is absolute.

It is also **not a per-user boundary**. One Hermes process uses one session pod for every conversation it serves — browser, cron, and each chat-platform user alike. See [Session scope](#session-scope).
:::

## The config block is the object

`terminal.kubernetes` is shaped like the manifest it creates. `apiVersion`, `kind`, `metadata` and `spec` mean exactly what they mean in any manifest, so a pod you already have is a copy-paste away, and `kubectl explain pod.spec` documents most of this schema.

```yaml
terminal:
  backend: kubernetes
  kubernetes:
    namespace: hermes-agents
    apiVersion: v1
    kind: Pod
    metadata:
      labels:
        app.kubernetes.io/managed-by: hermes-agent
    spec:
      containers:
        - name: workspace
          image: ubuntu:26.04
          command:
            - sleep
            - infinity
          workingDir: /workspace
          volumeMounts:
            - name: workspace
              mountPath: /workspace
            - name: tmp
              mountPath: /tmp
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
      volumes:
        - name: workspace
          emptyDir: {}
        - name: tmp
          emptyDir: {}
      shareProcessNamespace: true
      restartPolicy: Never
      terminationGracePeriodSeconds: 1
      automountServiceAccountToken: false
      enableServiceLinks: false
      hostNetwork: false
      hostPID: false
      hostIPC: false
      activeDeadlineSeconds: 14400
      securityContext:
        seccompProfile:
          type: RuntimeDefault
```

This `spec` is also the **built-in default**: leave `terminal.kubernetes.spec` unset and this minimal ephemeral `ubuntu:26.04` pod is exactly what you get. `hermes setup` writes the block into your `config.yaml` so the pod you get is also the pod you can read. The default sets no UID and no `runAsNonRoot`, so it admits and starts on any cluster: where an admission controller assigns UIDs (OpenShift SCC) the session runs as the assigned user, and everywhere else it runs as the image's user inside the pod boundary. To require non-root, [set it in `spec`](#running-as-non-root-optional). Everything below explains why each field is there and what breaks without it.

**A non-empty `spec` replaces the default entirely.** It is posted to the API server verbatim: there is no base underneath it and no merge, so the pod you get is either exactly the default above or exactly the spec in your own `config.yaml` — never a blend of the two.

## Parameters

### `namespace`

*string, default `""`.* Where session pods are created. Empty resolves from the kubeconfig context's default namespace first (kubectl semantics: a context with no namespace set means `default`), then the projected ServiceAccount namespace file in-cluster (`/var/run/secrets/kubernetes.io/serviceaccount/namespace`). Deliberately not env-var resolvable.

Run Hermes in a **dedicated namespace with nothing else in it**. `create pods` in a namespace is a powerful verb, and this is the containment that matters.

### `kubeconfig` / `context`

*strings, default `""`.* `kubeconfig` is the path to the kubeconfig file. `context` selects a context in it; both are ignored in-cluster.

### `apiVersion` / `kind`

*strings, defaults `v1` / `Pod`.* Together they say which object Hermes creates **and knows how to drive** — `v1`/`Pod` is the only pair implemented today, and support for other objects (an agent-sandbox API, say) keys off this same pair rather than a new config key.

An unsupported pair fails in-process, naming what is supported:

```
kubernetes backend: unsupported apiVersion/kind apps/v1/Deployment.
Supported: v1/Pod. Set terminal.kubernetes.apiVersion and
terminal.kubernetes.kind.
```

It fails in-process because nothing downstream could tell you which kinds your build implements — posting an unknown one returns a 404 on a REST path you never wrote.

### `metadata`

*mapping, default `{}`.* Labels, annotations, finalizers — posted as written. Two keys are always Hermes':

| Field | Behaviour |
|---|---|
| `metadata.name` | **Always Hermes'.** Computed per process; it is how the pod is found again on the next command, so a supplied name would simply be lost. |
| `metadata.namespace` | **Always Hermes'.** From `namespace` above. |
| `metadata.ownerReferences` | Hermes **appends** its own (the agent pod) to whatever you set. Both survive — the pod is collected once all owners are gone. Hermes' reference is also the ownership proof, so it is never omitted. |
| `metadata.labels` | The `owned_selector` labels and an instance label (`hermes.nousresearch.com/instance`) are added **only when absent**. |
| everything else | Untouched. |

### `spec`

*mapping, default: the pod above.* The whole PodSpec. Unset means the documented default; a non-empty value replaces it wholesale and is posted verbatim. See [What Hermes needs from your spec](#what-hermes-needs-from-your-spec).

### `exec_container_name`

*string, default `workspace`.* **A pointer into your `spec`**, not something that creates a container. Setting it alone changes nothing about the pod; if it names a container your spec does not declare, the failure is caught before any pod is created:

```
terminal.kubernetes.exec_container_name is 'workspace' but spec.containers
declares ['devbox']. Every command would fail with "session pod has no
container 'workspace'" — after the pod is created and pulled, which is a slow
way to learn it.
```

### `owned_selector`

*mapping, default `{}` meaning `{app.kubernetes.io/managed-by: hermes-agent}`.* The labels that mark an object as this backend's. Used twice: stamped into `metadata.labels` when absent, and matched before a 409 is treated as "resume this pod".

Configurable so an operator can relabel session pods for their own platform conventions (chargeback, a different `managed-by` value) without falling out of the ownership check against their own pods. The ownerReference UID remains the actual proof; this is the cheap filter — **and it is what the NetworkPolicy below selects on**, so if you change it, change the policy to match. A non-empty value **replaces** the default entirely; nothing is merged, so rebranding does not silently keep a label you never wrote.

### `ready_timeout_seconds`

*integer, default `120`.* How long Hermes waits for the pod to become Ready. Not pod shape, so it stays config. Raise it for slow image pulls or a kata/sandboxed runtime — cold starts there routinely exceed the default.

### `owner_reference`

*`auto` (default) or `off`.* Stamps the agent's own pod as owner so session pods are garbage-collected when the agent dies. "Dies" means the agent **Pod object is deleted** — ownerReference GC follows object deletion, so a crash-looping agent container keeps its pod, and therefore keeps its session pod.

:::warning Quote it
YAML 1.1 parses an unquoted `off` as the boolean `false`. Hermes accepts both, but `owner_reference: "off"` is unambiguous.
:::

With `off`, nothing collects a session pod whose agent died except `spec.activeDeadlineSeconds` — which is why you should set one. (The idle reaper is in-process; it dies with the agent and cannot clean up after it.)

### `trusted_sandbox`

*boolean, default `true`.* Whether a session pod is treated as a disposable sandbox. When `true` (the default), commands skip the dangerous-command approval layer — the pod, not a prompt, is the boundary.

Set it to `false` when a spec grants access to node-owned state (a `hostPath` volume, `hostNetwork`/`hostPID`/`hostIPC`, or a privileged container) and you want a human back in the loop: commands then run the normal approval flow, so `approvals.mode` and `approvals.deny` govern them and the hardline blocklist applies. Hermes does not inspect the spec for you — this is the operator's assertion about how disposable the pod is.

### Not parameters

There is deliberately no key for the image, resources, service account, node selector, tolerations, runtime class, security context, persistence, or the active deadline. **All of those are PodSpec**, so they live in `spec`. The shared `container_cpu` / `container_memory` / `container_disk` / `container_persistent` settings are **not read** by this backend.

## What Hermes needs from your spec

`spec` is yours, and the API server decides whether it is well-formed. But a handful of fields are things the *backend* depends on, and a pod missing them is valid Kubernetes and useless to Hermes — an omission `fieldValidation=Strict` and a dry-run both wave through. So Hermes checks them itself: `preflight_spec` runs at config time and in `hermes doctor`.

| Your `spec` must have | Or else |
|---|---|
| a container named `exec_container_name` | every command fails with `session pod has no container 'workspace'` |
| a command that keeps it running (`sleep infinity`) | the entrypoint exits and, under `restartPolicy: Never`, the pod reaches phase `Succeeded`; it never becomes Ready |
| `workingDir` on that container | **this is the session's cwd** — Hermes reads it from here, which is why there is no separate mount-path key. Omitted, sessions start in `/workspace` no matter where you mounted anything |
| a writable volume at that `workingDir` | `builtin cd -- <workingDir>` fails on every command |
| a writable `/tmp` | `init_session()` cannot snapshot the environment, so cwd and env silently stop persisting between commands |
| `shareProcessNamespace: true` | `sleep` as PID 1 never reaps, so a backgrounded command's wrapper zombifies and background completion is never detected |
| `terminationGracePeriodSeconds: 1` | teardown waits the default 30s, on the interrupt path, because `sleep infinity` ignores SIGTERM |
| `activeDeadlineSeconds` | nothing bounds a session pod whose agent died without an ownerReference |
| `automountServiceAccountToken: false` | the ServiceAccount token is projected into the agent's shell |

These are warnings, not errors, except the exec container (which cannot work at all) and a read-only root filesystem with no `/tmp` mount.

:::caution `shareProcessNamespace` and sidecars
Hermes needs the shared PID namespace to detect background completion. A shared PID namespace is also a **shared `/proc`**: with any second container in the pod, the agent's shell can read that container's `/proc/<pid>/environ` and `/proc/<pid>/root` — including a Secret volume mounted only into it — and can signal its processes. Keep credential-holding sidecars out of the session pod.
:::

## Example Kubernetes Objects

Two, and only the first is required.

### 1. RBAC for the agent's ServiceAccount

Replace `<AGENT_NAMESPACE>` and `<AGENT_SA>`.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: hermes-session-exec
  namespace: <AGENT_NAMESPACE>
rules:
  # Lifecycle of session pods. `get` also resolves the agent's OWN pod for the
  # ownerReference, and reads pod status for readiness diagnostics.
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["create", "get", "delete"]
  # BOTH verbs. The python client opens exec as a websocket-upgrading GET, but
  # the API server refuses it with 403 unless `create` is also granted
  # (verified against Kubernetes 1.36). Granting only `get` produces a healthy
  # startup on which every command then fails.
  - apiGroups: [""]
    resources: ["pods/exec"]
    verbs: ["get", "create"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: hermes-session-exec
  namespace: <AGENT_NAMESPACE>
subjects:
  - kind: ServiceAccount
    name: <AGENT_SA>
    namespace: <AGENT_NAMESPACE>
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: hermes-session-exec
```

Verify with `hermes doctor`, which issues a `SelfSubjectAccessReview` for each verb and dry-runs the pod it would submit. RBAC is where in-cluster deployments actually fail.

### 2. NetworkPolicy (recommended)

Without one, a session pod reaches the Kubernetes API and every ClusterIP Service in the cluster.

```yaml
# Default-deny for session pods.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hermes-session-default-deny
  namespace: <AGENT_NAMESPACE>
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/managed-by: hermes-agent   # == your owned_selector
  policyTypes: ["Ingress", "Egress"]
  ingress: []
  egress:
    # DNS. Port 5353 is NOT optional on OpenShift: OVN-Kubernetes applies
    # egress ACLs AFTER service DNAT, so a query to the DNS service on :53 is
    # rewritten to the CoreDNS pod on :5353 before the policy is evaluated. A
    # rule naming only 53 never matches and every lookup times out — the
    # symptom is a session that resolves nothing while raw-IP egress works.
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels: {k8s-app: kube-dns}
      ports:
        - {protocol: UDP, port: 53}
        - {protocol: TCP, port: 53}
        - {protocol: UDP, port: 5353}
        - {protocol: TCP, port: 5353}
---
# OPTIONAL: most agents need the public internet (pip, npm, git). Apply this
# ALONGSIDE the default-deny, not instead of it — the except-list below covers
# RFC1918, which on most clusters contains the service CIDR, so this policy
# does not grant DNS.
#
# COMPLETE THE except-LIST FOR YOUR CLUSTER. On a cluster whose nodes, API
# server or service network sit on publicly routable addresses, add them.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hermes-session-allow-internet
  namespace: <AGENT_NAMESPACE>
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/managed-by: hermes-agent
  policyTypes: ["Egress"]
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except: ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"]
```

:::warning The selector is configuration, not a guarantee
These policies select on `app.kubernetes.io/managed-by: hermes-agent`, which is the **default value of `owned_selector`**. An operator who rebrands that label gets session pods these policies do not select, with unrestricted egress and nothing on the cluster logging it. The only warning you get is from Hermes itself: `preflight_spec` warns when `owned_selector` no longer carries the default `managed-by` value these policies select on — a static config check at config time and in `hermes doctor`; Hermes never reads your live NetworkPolicies. If you rebrand, change the `podSelector`s to match.
:::

## Agent deployment requirements

The Hermes pod itself needs the ServiceAccount bound above, and the client: `pip install 'hermes-agent[kubernetes]'`. The client lazy-installs on first use where PyPI is reachable, but an in-cluster deployment usually cannot reach PyPI — install it up front (the published image pre-bakes it). Until the client is importable, the terminal tool stays disabled.

**Downward API is optional.** Hermes resolves its own pod identity by looking itself up on the pod hostname. To inject it explicitly (runtime identity, not user configuration — do not put these in `.env.example`):

```yaml
env:
  - name: HERMES_POD_NAME
    valueFrom: {fieldRef: {fieldPath: metadata.name}}
  - name: HERMES_POD_UID
    valueFrom: {fieldRef: {fieldPath: metadata.uid}}
```

When identity cannot be resolved, Hermes logs a warning; session pods then carry no `ownerReference` and are bounded only by your `activeDeadlineSeconds`.

## OpenShift notes

**Session pods inherit the AGENT's SCC, not `restricted-v2`.** This is the most important thing on this page. SCC admission evaluates the SCCs available to the identity that *creates* the pod — always the agent's ServiceAccount — and that SA typically needs a `runAsUser: RunAsAny` SCC of its own, because the Hermes image starts as container-root under s6. A session pod with `runAsUser: 0` is therefore **admitted**, with preflight, `fieldValidation=Strict` and `hermes doctor` all green.

So do not assume `restricted-v2` confines your session pods. Binding a *stricter* SCC to a session ServiceAccount does not confine them either: SCC admission evaluates the **union** of the SCCs available to the creating identity and to the pod's ServiceAccount, so the agent SA's permissive SCC is always in the pool and a spec that requests root is admitted through it. To genuinely confine session pods, give them their **own namespace** (`terminal.kubernetes.namespace`) with Pod Security Admission enforcing `restricted` — PSA judges the pod itself, regardless of which identity created it — and put the RBAC Role/RoleBinding in that namespace.

* **Leave `runAsUser` unset** where the agent SA *is* confined to `restricted-v2`: that SCC assigns one from the namespace's `openshift.io/sa.scc.uid-range`, and a hard-coded value outside it is rejected.
* **Set resource limits** if a `ResourceQuota` covers `limits.*` — a requests-only pod is rejected. Include `ephemeral-storage`, since an `emptyDir` workspace is otherwise unbounded.
* **Sandboxed containers (kata):** set `spec.runtimeClassName: kata` and raise `ready_timeout_seconds`; cold starts routinely exceed 120s.

## Running as non-root (optional)

The default spec does not set `runAsNonRoot`, so it starts anywhere — but that means on a cluster where nothing assigns UIDs, sessions run as the image's user (root, for `ubuntu`). To require non-root, set both fields:

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
```

Both, because `runAsNonRoot: true` alone fails on a root-default image with `container has runAsNonRoot and image will run as root` unless something assigns a UID (on OpenShift the SCC does — leave `runAsUser` unset there). No `fsGroup` is needed: the kubelet creates `emptyDir` directories world-writable.

A namespace labelled `pod-security.kubernetes.io/enforce=restricted` also requires this: the Pod Security `restricted` profile rejects a spec without `runAsNonRoot`. Label the session-pod namespace, not the agent's — the agent pod starts as container-root and an enforced `restricted` level rejects it.

## Storage

Session workspaces are **stateless by default** — an `emptyDir` that dies with the pod — and there is deliberately no persistence key. Storage is plain PodSpec.

```yaml
# Per-session volume, dynamically provisioned. Also how you bound its size.
# The generated claim is owned by the pod and deleted with it (generic
# ephemeral volume semantics) — a reaped pod takes this workspace with it.
spec:
  volumes:
    - name: workspace
      ephemeral:
        volumeClaimTemplate:
          spec:
            accessModes: [ReadWriteOnce]
            storageClassName: fast-nvme
            resources: {requests: {storage: 10Gi}}
    - name: tmp
      emptyDir: {}
```

```yaml
# A claim you pre-provisioned. You own its lifecycle; Hermes never creates,
# adopts or deletes it.
spec:
  volumes:
    - name: workspace
      persistentVolumeClaim: {claimName: hermes-workspace}
    - name: tmp
      emptyDir: {}
```

Two cautions. **Restate `tmp`** — you are writing the whole `volumes` list, and omitting it produces `spec.containers[0].volumeMounts[1].name: Not found: "tmp"` at create time. And **the shared idle reaper still applies**: after `terminal.lifetime_seconds` (default 300) of tool-call inactivity the pod is destroyed regardless of what it mounted, so raise it for session-length workspaces.

A dry-run cannot see a non-existent `storageClassName` or `claimName`, so a typo there yields a green `hermes doctor` and then a readiness timeout.

## Environment variables

Like storage, environment for the session pod is plain PodSpec — there is no separate env key, because `spec.containers[].env` already covers all three shapes:

```yaml
spec:
  containers:
    - name: workspace
      env:
        # 1. A static knob, written literally.
        - name: DEBUG
          value: "1"
        # 2. Forwarded from the AGENT's environment: config.yaml expands
        #    ${VAR} from the Hermes process's environment (shell or
        #    ~/.hermes/.env) at load time.
        - name: DEPLOY_ENV
          value: "${DEPLOY_ENV}"
        # 3. A token, injected by the kubelet from a Secret in the session
        #    pod's namespace. The value never passes through Hermes, its
        #    config, or the pod manifest.
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef: {name: hermes-session-secrets, key: github-token}
```

Know where each value ends up before picking a shape. A `${VAR}` reference is expanded **before** the pod is created, so the resolved value is written into the pod manifest — readable by anyone with `get pods` in the namespace, and recorded in the API-server audit log of the create. That is fine for a deploy-target name and wrong for a token: secrets belong in shape 3, where only a Secret *reference* appears in the manifest. (This is the same reasoning that makes credential-file sync stream over exec stdin rather than argv.)

## Session scope

A "session pod" is one pod per Hermes **process**, not one per conversation: a browser session, scheduled crons and every chat-platform user served by one gateway all execute in the **same** pod. One user can read a file another just wrote, and the synced credential files are readable from every session.

This is deliberate for subagents. But it means the filesystem, installed packages, shell environment and the idle reaper are all shared. (Each session keeps its own cwd.) The gateway's `group_sessions_per_user` setting does not change this: it isolates *conversation* state per user, but every one of those sessions still executes in the one shared pod.

**Do not rely on the session pod as a per-user boundary.** Run a Hermes process per trust domain, or keep the gateway single-tenant.

## Validation: who checks what

Hermes validates **nothing about the pod's content** in-process. It makes exactly two checks of its own, and both are questions the API server has no opinion on:

* **the `apiVersion`/`kind` pair** — does this build know how to drive that object?
* **`preflight_spec`** — can Hermes *use* this pod? A pod whose containers are all named `sidecar` is perfectly legal Kubernetes and useless as an exec target.

Everything else is delegated:

* **Well-formedness** — every create passes `fieldValidation=Strict`, so an unknown or misspelled field is a `400` naming the exact JSON path. (The python client discards the API server's `Warning: 299 - unknown field` header, which makes the default `Warn` behaviour indistinguishable from success.) `hermes doctor` submits your pod as a `dry_run=All` create for the same reason.
* **What the pod may be** — SCC, Pod Security Admission, NetworkPolicy and RBAC. These are the cluster administrator's tools, they are authoritative, and an in-process approximation would be redundant where it agreed and wrong where it did not.

## What this does not protect against

* `create pods` in a namespace remains a powerful verb. Run Hermes in a dedicated namespace with nothing else in it.
* Session pods share the cluster network unless the NetworkPolicy is applied.
* **Sessions are not isolated from each other** — see [Session scope](#session-scope).
* **Exec requests are recorded in the API-server audit log.** The client puts every `command` element into the request URL, and kube-apiserver records `requestURI` at Metadata level and above. Hermes' file-sync therefore streams over the exec **stdin** channel and never through argv — but a command the *agent* runs still appears verbatim, including anything piped through its heredoc stdin. Do not treat the audit log as a place secrets cannot reach.
