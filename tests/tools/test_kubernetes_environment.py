"""Unit tests for the Kubernetes session-pod execution backend.

Ported from upstream PR #37591 and re-specified for this fork's config surface:
every setting is a ``terminal.kubernetes.*`` config.yaml key bridged as ONE
internal JSON env var, and the pod shape is ONE ``spec`` posted to the API
server VERBATIM — the shipped default when unset, no base and no merge when
set. Validating and constraining
the pod is the cluster's job (``fieldValidation=Strict``, SCC / PSA / admission
policy), so these tests pin the passthrough, the ownership and adoption rules,
the preflight checks the cluster cannot make, and the exec loop — not a local
judge of pod content.

No cluster required: the kubernetes client is stubbed into ``sys.modules`` and
manifest builders run with ``api=None``.
"""

import base64
import io
from copy import deepcopy
from pathlib import Path
import json
import time
import sys
import tarfile
import threading
import types as _types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from tools.environments.kubernetes import (
    DEFAULT_KUBERNETES_CONFIG,
    INSTANCE_LABEL,
    KubernetesEnvironment,
    PodProvisioner,
    PodRef,
    WorkspaceProvisioner,
    merge_kubernetes_config,
    render_session_object,
    sanitize_name,
)

# The manifest builders are import-safe without the kubernetes client (its
# imports are function-local), which is what lets this module import at all
# on a machine that never installed the SDK.


@pytest.fixture(autouse=True)
def _stub_kubernetes(monkeypatch):
    """Stub the kubernetes client so the backend imports without a cluster."""
    if "kubernetes" in sys.modules:
        return
    k = _types.ModuleType("kubernetes")
    k.client = _types.ModuleType("kubernetes.client")
    k.config = _types.ModuleType("kubernetes.config")
    k.stream = _types.ModuleType("kubernetes.stream")
    ws_mod = _types.ModuleType("kubernetes.stream.ws_client")
    exc_mod = _types.ModuleType("kubernetes.client.exceptions")

    class ApiException(Exception):
        def __init__(self, status=0, reason=""):
            self.status = status
            self.reason = reason
            super().__init__(f"{status}: {reason}")

    exc_mod.ApiException = ApiException
    k.client.exceptions = exc_mod

    # Every exec builds its OWN ApiClient: kubernetes.stream.stream()
    # monkeypatches api_client.request, so sharing one corrupts it under
    # concurrency.  These are real (tiny) classes rather than MagicMock so
    # CoreV1Api(api_client) does not read as MagicMock(spec=...).
    class _StubApiClient:
        def __init__(self, configuration=None):
            self.configuration = configuration
            self.closed = False

        def close(self):
            self.closed = True

    def _stub_core_v1_api(api_client=None, **kwargs):
        api = MagicMock()
        api.api_client = api_client if api_client is not None else _StubApiClient()
        return api

    class _V1Preconditions:
        def __init__(self, uid=None, resource_version=None):
            self.uid = uid
            self.resource_version = resource_version

    class _V1DeleteOptions:
        def __init__(self, preconditions=None, grace_period_seconds=None, **kw):
            self.preconditions = preconditions
            self.grace_period_seconds = grace_period_seconds

    k.client.ApiClient = _StubApiClient
    k.client.CoreV1Api = _stub_core_v1_api
    k.client.CustomObjectsApi = MagicMock
    k.client.V1Preconditions = _V1Preconditions
    k.client.V1DeleteOptions = _V1DeleteOptions
    k.config.load_incluster_config = lambda: None
    k.config.load_kube_config = lambda **kw: None
    k.stream.stream = MagicMock()
    ws_mod.STDIN_CHANNEL = 0
    ws_mod.V5_CHANNEL_PROTOCOL = "v5.channel.k8s.io"
    k.stream.ws_client = ws_mod
    monkeypatch.setitem(sys.modules, "kubernetes", k)
    monkeypatch.setitem(sys.modules, "kubernetes.client", k.client)
    monkeypatch.setitem(sys.modules, "kubernetes.client.exceptions", exc_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", k.config)
    monkeypatch.setitem(sys.modules, "kubernetes.stream", k.stream)
    monkeypatch.setitem(sys.modules, "kubernetes.stream.ws_client", ws_mod)


#: The shipped starter object. Tests that are not ABOUT the spec use it, so a
#: change to what Hermes needs from a pod breaks here as well as in
#: test_the_shipped_starter_template_renders_a_usable_pod.
def _starter_object():
    """The object `hermes setup` writes. It used to be read from a shipped
    k8s/ YAML; that directory is gone (the docs carry the objects now) and the
    docs page is pinned against this same constant by
    test_the_documented_starter_object_is_the_shipped_one."""
    from copy import deepcopy

    from tools.environments.kubernetes import STARTER_SESSION_OBJECT

    return deepcopy(STARTER_SESSION_OBJECT)


def _tpl_merge(base, overlay):
    """Deep-merge helper for BUILDING a test spec. Not backend behaviour: the
    backend posts `spec` verbatim and has no merge of any kind. This exists so
    a test that cares about one field does not restate a whole PodSpec."""
    out = deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _tpl_merge(out[key], value)
        elif key == "containers" and isinstance(value, list) and isinstance(out.get(key), list):
            merged = [deepcopy(c) for c in out[key]]
            index = {c["name"]: i for i, c in enumerate(merged) if isinstance(c, dict)}
            for entry in value:
                name = entry.get("name") if isinstance(entry, dict) else None
                if name in index:
                    merged[index[name]] = _tpl_merge(merged[index[name]], entry)
                else:
                    merged.append(deepcopy(entry))
            out[key] = merged
        else:
            out[key] = deepcopy(value)
    return out


def _kcfg(**overrides):
    """Config with a COMPLETE spec, so tests state exactly what is posted.

    `pod_template` here is a TEST convenience: a {metadata?, spec?} fragment
    folded onto the starter object, so a test can name the one field it cares
    about. Pass `pod_template_raw` to configure exactly what you wrote (which
    is what the backend actually posts)."""
    raw = overrides.pop("pod_template_raw", _SENTINEL)
    fragment = overrides.pop("pod_template", None)
    if raw is not _SENTINEL:
        obj = raw if isinstance(raw, dict) else {}
        overrides["metadata"] = obj.get("metadata", {}) if isinstance(obj, dict) else {}
        overrides["spec"] = obj.get("spec") if isinstance(obj, dict) else obj
    else:
        starter = _starter_object()
        merged = _tpl_merge(
            {"metadata": starter.get("metadata", {}), "spec": starter["spec"]},
            fragment,
        )
        overrides["metadata"] = merged["metadata"]
        overrides["spec"] = merged["spec"]
    return merge_kubernetes_config(overrides)


_SENTINEL = object()


OWNER_REF = {
    "apiVersion": "v1",
    "kind": "Pod",
    "name": "hermes-agent-0",
    "uid": "11111111-1111-1111-1111-111111111111",
    "controller": False,
    "blockOwnerDeletion": False,
}

MANAGED_BY = "app.kubernetes.io/managed-by"


# ---------------------------------------------------------------------------
# Value types / helpers
# ---------------------------------------------------------------------------


def test_provisioner_is_abstract():
    with pytest.raises(TypeError):
        WorkspaceProvisioner()  # ABC — cannot instantiate


def test_sanitize_name_makes_rfc1123_safe_names():
    """RL/benchmark task ids carry uppercase, ``_`` and ``:``; the API server
    rejects those in a pod name with a confusing 422."""
    assert sanitize_name("default") == "default"
    slug = sanitize_name("Task_ID:With/Junk")
    assert slug.replace("-", "").isalnum()
    assert slug.islower()
    # Distinct inputs that share a prefix must not collide after truncation.
    a = sanitize_name("x" * 200 + "aaa")
    b = sanitize_name("x" * 200 + "bbb")
    assert a != b
    assert len(a) <= 40


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_partial_config_still_yields_every_default():
    """cli.py and gateway/run.py bridge the RAW config without deep-merging
    DEFAULT_CONFIG, so a user who sets one key produces a partial payload."""
    merged = merge_kubernetes_config({"namespace": "hermes-agents"})
    assert merged["namespace"] == "hermes-agents"
    assert merged["kind"] == "Pod"
    assert merged["apiVersion"] == "v1"
    assert merged["exec_container_name"] == "workspace"
    # Nested partials merge rather than replace the block.
    nested = merge_kubernetes_config({"spec": {"hostPID": True}})
    assert nested["spec"]["hostPID"] is True
    assert nested["kind"] == "Pod"


def test_merge_does_not_mutate_defaults():
    merged = merge_kubernetes_config({"spec": {}})
    merged["spec"]["hostPID"] = True
    assert DEFAULT_KUBERNETES_CONFIG["spec"] == {}
    merged = merge_kubernetes_config({"owned_selector": {"a": "b"}})
    merged["owned_selector"]["c"] = "d"
    assert DEFAULT_KUBERNETES_CONFIG["owned_selector"] == {}


def test_every_nested_default_is_empty():
    """merge_kubernetes_config does a SHALLOW update, which is only equivalent
    to the old recursive merge while every dict-valued default is empty.

    Add a non-empty nested default (here or in the config_defaults mirror) and
    that key silently flips from replace to merge — which is exactly the bug
    `owned_selector` had: an operator who rebranded it kept a default label
    they never wrote and could not remove."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    for name, value in DEFAULT_KUBERNETES_CONFIG.items():
        if isinstance(value, dict):
            assert value == {}, f"{name} has a non-empty dict default"
    mirror = DEFAULT_CONFIG["terminal"]["kubernetes"]
    for name, value in mirror.items():
        if isinstance(value, dict):
            assert value == {}, f"config_defaults mirror: {name} is non-empty"


def test_an_empty_owned_selector_means_the_managed_by_label():
    """The default is empty on purpose, and that is load-bearing twice.

    A non-empty mapping default gets flattened into the web settings schema as
    `...owned_selector.app.kubernetes.io/managed-by` — the only schema path in
    the codebase with a `/` inside a segment. The dashboard splits paths on
    `.`, so any edit or category Reset rewrote it as nested objects and every
    pod create then 422'd on an invalid label value, with the backend dead
    until config.yaml was hand-edited.

    It also makes replace-vs-merge moot: an operator who rebrands the selector
    would otherwise deep-merge onto the default and silently keep a
    `managed-by: hermes-agent` they never wrote and could not remove."""
    from tools.environments.kubernetes import owned_selector

    assert owned_selector(merge_kubernetes_config({})) == {MANAGED_BY: "hermes-agent"}
    assert owned_selector(merge_kubernetes_config(
        {"owned_selector": {"platform.example.com/owner": "ml-team"}},
    )) == {"platform.example.com/owner": "ml-team"}

    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.web_server import _build_schema_from_config

    schema = _build_schema_from_config(DEFAULT_CONFIG)
    assert not [k for k in schema if "/" in k], \
        "a schema path with a / breaks the dashboard's dotted-path splitter"


def test_there_is_no_in_process_config_validator():
    """The one Hermes-side decision (the provisioner enum) raises in the
    environment factory; everything else — quantities, RFC-1123 names, mount
    collisions, unknown fields — is the API server's to reject under
    fieldValidation=Strict. An in-process approximation would be redundant
    where it agreed and wrong where it did not."""
    import tools.environments.kubernetes as k8s_mod

    assert not hasattr(k8s_mod, "validate_kubernetes_config")


def test_kind_is_the_provisioner_seam():
    """apiVersion/kind say which object Hermes creates and knows how to drive.
    Dispatching on the pair means a second kind — an agent-sandbox CRD, say —
    needs a dispatch entry and NO new config key.

    Asserted as equality so shrinking OR quietly re-growing the table is a
    deliberate act with a test to update, not a drive-by."""
    from tools.environments.kubernetes import (
        PROVISIONERS_BY_KIND, resolve_provisioner_kind,
    )

    assert PROVISIONERS_BY_KIND == {("v1", "Pod"): "pod"}
    assert resolve_provisioner_kind({"apiVersion": "v1", "kind": "Pod"}) == "pod"
    # Empty means the default Pod, not a failure: merge_kubernetes_config
    # supplies both, but a hand-built dict in a caller should not explode.
    assert resolve_provisioner_kind({}) == "pod"


def test_an_unsupported_kind_names_what_is_supported():
    """The one Hermes-side decision. Posting an unknown kind blindly would come
    back as a 404 on a REST path the operator never wrote."""
    from tools.environments.kubernetes import resolve_provisioner_kind

    with pytest.raises(ValueError, match="unsupported apiVersion/kind apps/v1/Deployment"):
        resolve_provisioner_kind({"apiVersion": "apps/v1", "kind": "Deployment"})
    with pytest.raises(ValueError, match="Supported: v1/Pod"):
        resolve_provisioner_kind({"kind": "SandboxClaim"})


def test_hard_cut_keys_are_not_in_the_schema():
    """The PodSpec-shaped keys collapsed into `spec`, and the stateless cut
    removed the persistence and image sugar on top. No aliases, no shim."""
    # Named, not counted: a bare count fails for any legitimate new key and
    # names no offender. This fails for the same input and says which key.
    assert set(DEFAULT_KUBERNETES_CONFIG) == {
        "namespace", "kubeconfig", "context",
        "apiVersion", "kind", "metadata", "spec",
        "exec_container_name", "owned_selector",
        "ready_timeout_seconds", "owner_reference", "trusted_sandbox",
    }, sorted(DEFAULT_KUBERNETES_CONFIG)
    gone = {
        "image_pull_policy", "image_pull_secrets", "service_account",
        "automount_service_account_token", "runtime_class_name",
        "node_selector", "tolerations", "labels", "annotations", "env",
        "security_context", "resources", "pod_template_overrides",
        # The stateless / claim-based cut:
        "image", "persistent", "volume",
        # The claim provisioner's admin-side block went with it.
        "sandbox",
    }
    # `pod_template` and `provisioner` were themselves cut: the first split
    # into metadata + spec, the second is implied by `kind`.
    assert not ((gone | {"pod_template", "provisioner",
                         "workspace_mount_path", "active_deadline_seconds"})
                & set(DEFAULT_KUBERNETES_CONFIG))


# ---------------------------------------------------------------------------
# Pod template
# ---------------------------------------------------------------------------


def _provisioner(**overrides):
    return PodProvisioner(
        _kcfg(**overrides), "hermes", api=None, owner_reference=OWNER_REF
    )


def test_pod_uses_emptydir_and_carries_ownerref():
    """Stateless by design: the workspace is an emptyDir that dies with the
    pod. There is no PVC surface at all."""
    pod = _provisioner().pod_manifest("abc")
    vols = {v["name"]: v for v in pod["spec"]["volumes"]}
    assert vols["workspace"]["emptyDir"] == {}
    mount = pod["spec"]["containers"][0]["volumeMounts"][0]
    assert mount["mountPath"] == "/workspace"
    assert pod["metadata"]["ownerReferences"][0]["uid"] == OWNER_REF["uid"]
    assert not any("persistentVolumeClaim" in v for v in pod["spec"]["volumes"])


def test_there_is_no_pvc_surface():
    """The stateless cut removed pvc_name/pvc_manifest/_ensure_pvc wholesale —
    nothing in the backend can create, adopt or mount a claim."""
    p = _provisioner()
    for attr in ("pvc_name", "pvc_manifest", "_ensure_pvc", "_assert_pvc_is_ours"):
        assert not hasattr(p, attr), attr


def test_pod_manifest_omits_ownerref_when_owner_unknown():
    """K8s rejects an ownerReference with an empty name/uid with a 422."""
    p = PodProvisioner(_kcfg(), "hermes", api=None, owner_reference=None)
    pod = p.pod_manifest("abc")
    assert "ownerReferences" not in pod["metadata"]


def test_the_deadline_backstop_comes_from_the_template_not_a_config_key():
    """`active_deadline_seconds` was a config key that Hermes injected into the
    spec. It is PodSpec, so it lives in the template with everything else — and
    that means Hermes no longer guarantees one. The starter template sets it;
    a template that does not is unbounded, which is the operator's call to
    make explicitly rather than ours to make silently."""
    p = PodProvisioner(
        _kcfg(pod_template={"spec": {"activeDeadlineSeconds": 999}}),
        "hermes", api=None, owner_reference=OWNER_REF,
    )
    assert p.pod_manifest("abc")["spec"]["activeDeadlineSeconds"] == 999

    bare = PodProvisioner(
        _kcfg(pod_template_raw={"spec": {"containers": [
            {"name": "workspace", "image": "alpine:3.20"}]}}),
        "hermes", api=None, owner_reference=OWNER_REF,
    )
    assert "activeDeadlineSeconds" not in bare.pod_manifest("abc")["spec"]


def test_the_starter_template_is_a_sane_starting_point():
    """There is no default base any more — this is the SHIPPED TEMPLATE, the
    thing the missing-template error tells operators to copy. It has to produce
    a pod that starts, stays up, can be exec'd into, and satisfies OpenShift
    restricted-v2 without extra RBAC, because that is what it promises."""
    pod = _provisioner().pod_manifest("abc")
    spec = pod["spec"]
    assert spec["restartPolicy"] == "Never"
    assert spec["automountServiceAccountToken"] is False
    # No serviceAccountName: the namespace `default` SA, which is what a
    # vanilla cluster has. Defaulting to the no-perms SA that the RBAC on the Kubernetes docs page
    # ships would make the out-of-box config fail to schedule
    # (`serviceaccount "hermes-session-noperms" not found`) on every cluster
    # that has not applied that manifest. The token is off either way.
    assert "serviceAccountName" not in spec
    assert spec["hostNetwork"] is False
    assert spec["hostPID"] is False
    assert spec["hostIPC"] is False
    assert spec["enableServiceLinks"] is False
    assert spec["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    sc = spec["containers"][0]["securityContext"]
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"]["drop"] == ["ALL"]
    # No UID fields at either level: the default must admit and start on any
    # cluster. restricted-v2 rejects a hardcoded runAsUser outside the
    # namespace range, and runAsNonRoot without a UID fails at container
    # start wherever no admission controller assigns one (kubeadm, k3s,
    # kind, EKS, GKE, AKS defaults). Non-root is the operator's opt-in.
    assert "runAsUser" not in spec["securityContext"]
    assert "runAsNonRoot" not in spec["securityContext"]
    assert "fsGroup" not in spec["securityContext"]
    assert "runAsUser" not in sc
    assert "runAsNonRoot" not in sc
    # A plain distro base: the shipped template promises a shell and
    # coreutils, nothing more.
    assert spec["containers"][0]["image"] == "ubuntu:26.04"
    # The shared terminal.container_* knobs are NOT read by this backend:
    # resources are PodSpec and belong in the template, commented out there.
    assert "resources" not in spec["containers"][0]
    # PID 1 is `sleep`, which never reaps — the shared PID namespace makes
    # the sandbox `pause` process the reaper, so background completion
    # detection works.
    assert spec["shareProcessNamespace"] is True


def test_pod_template_reaches_the_manifest():
    """The user layer: anything that is merely PodSpec goes here."""
    tolerations = [{"key": "kata", "operator": "Exists", "effect": "NoSchedule"}]
    p = _provisioner(pod_template={
        "metadata": {"labels": {"team": "hermes"},
                     "annotations": {"io.katacontainers.x": "4096"}},
        "spec": {
            "runtimeClassName": "kata",
            "nodeSelector": {"node-role.kubernetes.io/worker": ""},
            "tolerations": tolerations,
            "imagePullSecrets": [{"name": "regcred"}],
            "priorityClassName": "high",
            "containers": [{"name": "workspace",
                            "imagePullPolicy": "Always",
                            "env": [{"name": "HTTPS_PROXY", "value": "http://p:3128"}]}],
        },
    })
    pod = p.pod_manifest("abc")
    spec = pod["spec"]
    assert spec["runtimeClassName"] == "kata"
    assert spec["nodeSelector"] == {"node-role.kubernetes.io/worker": ""}
    assert spec["tolerations"] == tolerations
    assert spec["imagePullSecrets"] == [{"name": "regcred"}]
    assert spec["priorityClassName"] == "high"
    assert pod["metadata"]["labels"]["team"] == "hermes"
    assert pod["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "hermes-agent"
    assert pod["metadata"]["annotations"]
    container = spec["containers"][0]
    assert container["imagePullPolicy"] == "Always"
    assert {"name": "HTTPS_PROXY", "value": "http://p:3128"} in container["env"]


def test_the_starter_template_mounts_tmp():
    """init_session() writes its env snapshot under /tmp, so a session pod
    needs a writable one. Hermes no longer supplies it — the starter template
    does, and README.md lists it as REQUIRED-BY-HERMES. A template that omits
    it against a read-only root filesystem breaks env tracking between
    commands."""
    pod = _provisioner().pod_manifest("abc")
    mounts = {m["mountPath"] for m in pod["spec"]["containers"][0]["volumeMounts"]}
    assert "/tmp" in mounts
    assert any(v["name"] == "tmp" for v in pod["spec"]["volumes"])


def test_the_session_cwd_is_read_from_the_template():
    """There is no workspace_mount_path key any more. It stated the same fact
    as the exec container's workingDir, so the two could disagree — a pod that
    mounted the workspace at one path while every command ran `builtin cd` into
    another. One source of truth removes the disagreement."""
    from tools.environments.kubernetes import session_cwd

    assert session_cwd(_kcfg()) == "/workspace"
    assert session_cwd(_kcfg(pod_template={"spec": {"containers": [
        {"name": "workspace", "workingDir": "/home/agent"}]}})) == "/home/agent"
    # It follows exec_container_name, not list position.
    two = _kcfg(exec_container_name="devbox", pod_template_raw={"spec": {"containers": [
        {"name": "sidecar", "workingDir": "/wrong"},
        {"name": "devbox", "workingDir": "/right"},
    ]}})
    assert session_cwd(two) == "/right"


def test_a_template_without_a_working_dir_falls_back():
    """The container then starts in the image's WORKDIR, which Hermes cannot
    read. A documented constant beats guessing, and TERMINAL_CWD still wins."""
    from tools.environments.kubernetes import FALLBACK_SESSION_CWD, session_cwd

    bare = _kcfg(pod_template_raw={"spec": {"containers": [
        {"name": "workspace", "image": "alpine:3.20"}]}})
    assert session_cwd(bare) == FALLBACK_SESSION_CWD
    # Pointing at a container that does not exist cannot resolve one either;
    # that failure belongs at exec, not here.
    assert session_cwd(_kcfg(exec_container_name="nope")) == FALLBACK_SESSION_CWD


def test_pod_name_is_instance_scoped():
    """_resolve_container_task_id collapses nearly everything to "default", so
    without a discriminator two agents in one namespace fight over the same pod
    name — the second 409s, "reuses" the first's workspace, and later deletes it."""
    a = PodProvisioner(_kcfg(), "hermes", api=None, owner_reference=OWNER_REF)
    b = PodProvisioner(
        _kcfg(), "hermes", api=None,
        owner_reference={**OWNER_REF, "uid": "22222222-2222-2222-2222-222222222222"},
    )
    assert a.workspace_name("default") != b.workspace_name("default")


# ---------------------------------------------------------------------------
# PodProvisioner against a mocked API
# ---------------------------------------------------------------------------


def _running_pod(labels=None, owners=None, containers=("workspace",)):
    cond = SimpleNamespace(type="Ready", status="True")
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="hermes-ws", labels=labels or {}, owner_references=owners or []
        ),
        spec=SimpleNamespace(
            containers=[SimpleNamespace(name=n) for n in containers]
        ),
        status=SimpleNamespace(
            phase="Running", conditions=[cond], container_statuses=[]
        ),
    )


def _provisioner_with_api(api, **overrides):
    return PodProvisioner(
        _kcfg(**overrides), "hermes", api=api, owner_reference=OWNER_REF
    )


def test_ensure_creates_pod_only():
    api = MagicMock()
    api.read_namespaced_pod.return_value = _running_pod()
    p = _provisioner_with_api(api)

    ref = p.ensure("abc")

    api.create_namespaced_pod.assert_called_once()
    # Without fieldValidation=Strict an unknown field is accepted with 201 and
    # silently dropped, and the python client discards the API server's
    # "Warning: 299 - unknown field" header, so nothing is logged.
    assert api.create_namespaced_pod.call_args.kwargs["field_validation"] == "Strict"
    assert ref.pod_name == p.workspace_name("abc")
    assert ref.container == "workspace"


def test_ensure_reuses_our_own_pod_on_conflict():
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    api.read_namespaced_pod.return_value = _running_pod(
        labels={"app.kubernetes.io/managed-by": "hermes-agent"},
        owners=[SimpleNamespace(uid=OWNER_REF["uid"])],
    )
    p = _provisioner_with_api(api)

    ref = p.ensure("abc")
    assert ref.pod_name == p.workspace_name("abc")


def test_ensure_refuses_to_hijack_another_agents_pod():
    """A blanket 409-as-reuse leaks another agent's workspace (and later
    deletes it) on a shared cluster."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    api.read_namespaced_pod.return_value = _running_pod(
        labels={"app.kubernetes.io/managed-by": "someone-else"},
    )
    p = _provisioner_with_api(api)

    with pytest.raises(RuntimeError, match="not created by this Hermes instance"):
        p.ensure("abc")


def test_wait_ready_surfaces_image_pull_failures_immediately():
    api = MagicMock()
    api.read_namespaced_pod.return_value = SimpleNamespace(
        metadata=SimpleNamespace(name="p", labels={}, owner_references=[]),
        status=SimpleNamespace(
            phase="Pending",
            conditions=[],
            container_statuses=[
                SimpleNamespace(
                    name="workspace",
                    state=SimpleNamespace(
                        waiting=SimpleNamespace(
                            reason="ImagePullBackOff", message="pull access denied"
                        ),
                        terminated=None,
                    ),
                )
            ],
        ),
    )
    p = _provisioner_with_api(api, ready_timeout_seconds=30)
    with pytest.raises(RuntimeError) as excinfo:
        p.ensure("abc")
    assert "ImagePullBackOff" in str(excinfo.value)
    assert "pull access denied" in str(excinfo.value)


def test_destroy_deletes_the_pod():
    api = MagicMock()
    p = _provisioner_with_api(api)
    p.destroy(PodRef("hermes", "hermes-ws-abc", "workspace"))
    api.delete_namespaced_pod.assert_called_once()
    kwargs = api.delete_namespaced_pod.call_args.kwargs
    assert kwargs["name"] == "hermes-ws-abc"
    assert kwargs["namespace"] == "hermes"
    # `sleep infinity` as PID 1 ignores SIGTERM; without grace 0 every teardown
    # waits the full 30s default, on the interrupt path.
    assert kwargs["grace_period_seconds"] == 0


# ---------------------------------------------------------------------------
# KubernetesEnvironment
# ---------------------------------------------------------------------------


class _FakeWSClient:
    """Mimics kubernetes.stream WSClient for one exec call."""

    def __init__(self, stdout="", returncode=0, open_cycles=1, raise_on_update=None,
                 subprotocol="v4.channel.k8s.io", update_sleep=0.0):
        self._update_sleep = update_sleep
        self._stdout = stdout
        self._returncode = returncode
        self._cycles = open_cycles
        self._raise_on_update = raise_on_update
        self.closed = False
        self.subprotocol = subprotocol
        self.stdin_writes: list[str] = []
        self.channels_closed: list[int] = []

    def write_stdin(self, data):
        self.stdin_writes.append(data)

    def close_channel(self, channel):
        self.channels_closed.append(channel)

    def is_open(self):
        if self.closed or self._cycles <= 0:
            return False
        self._cycles -= 1
        return True

    def update(self, timeout=None):
        if self._raise_on_update:
            raise self._raise_on_update
        if self._update_sleep:
            time.sleep(self._update_sleep)

    def peek_stdout(self):
        return bool(self._stdout)

    def read_stdout(self):
        s, self._stdout = self._stdout, ""
        return s

    def peek_stderr(self):
        return False

    def read_stderr(self):
        return ""

    def close(self):
        self.closed = True

    @property
    def returncode(self):
        if isinstance(self._returncode, Exception):
            raise self._returncode
        return self._returncode


def _make_k8s_env(monkeypatch, exec_results, api=None):
    """exec_results: list of _FakeWSClient factories / tuples per exec call."""
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)

    provisioner = MagicMock()
    provisioner.workspace_name.return_value = "hermes-ws-abc"
    provisioner.namespace = "hermes"
    provisioner.ensure.return_value = PodRef("hermes", "hermes-ws-abc", "workspace")
    # Explicit: a MagicMock would auto-create `_api`, and every field read off
    # the resulting fake pod (deletion_timestamp, phase) would be a truthy
    # sentinel — making every exec error look like a dead pod. Tests that DO
    # want the liveness check pass a real fake via `api`.
    provisioner._api = api

    calls = {"i": 0}

    def fake_stream(*args, **kwargs):
        idx = min(calls["i"], len(exec_results) - 1)
        calls["i"] += 1
        spec = exec_results[idx]
        if isinstance(spec, _FakeWSClient):
            return spec
        out, rc = spec
        return _FakeWSClient(stdout=out, returncode=rc)

    monkeypatch.setattr("kubernetes.stream.stream", fake_stream)

    env = KubernetesEnvironment(
        provisioner=provisioner,
        task_id="abc",
        cwd="/workspace",
        timeout=30,
        api=api,
        sync_files=False,  # exercised directly below (see the file-sync tests)
    )
    env._exec_calls = calls
    return env


def test_basic_command(monkeypatch):
    # exec calls: (1) init_session bootstrap, (2) the actual command
    env = _make_k8s_env(monkeypatch, [("", 0), ("hello\n", 0)])
    result = env.execute("echo hello")
    assert "hello" in result["output"]
    assert result["returncode"] == 0


def test_nonzero_exit_code(monkeypatch):
    env = _make_k8s_env(monkeypatch, [("", 0), ("nope\n", 127)])
    result = env.execute("bad_cmd")
    assert result["returncode"] == 127


def test_exec_stream_error_returns_message_not_silence(monkeypatch):
    client = _FakeWSClient(raise_on_update=RuntimeError("connection reset"))
    env = _make_k8s_env(monkeypatch, [("", 0), client])
    result = env.execute("something")
    assert "connection reset" in result["output"]
    assert result["returncode"] != 0


def test_cleanup_calls_provisioner_destroy(monkeypatch):
    env = _make_k8s_env(monkeypatch, [("", 0)])
    env.cleanup()
    env._provisioner.destroy.assert_called_once()
    args, _kwargs = env._provisioner.destroy.call_args
    assert args[0].pod_name == "hermes-ws-abc"


def test_cleanup_is_idempotent(monkeypatch):
    env = _make_k8s_env(monkeypatch, [("", 0)])
    env.cleanup()
    env.cleanup()
    assert env._provisioner.destroy.call_count == 1


def test_cleanup_retries_destroy_once_then_escalates(monkeypatch, caplog):
    """A failed teardown is retried once; the final failure is an ERROR that
    names the pod and the backstop, and the environment stays torn down."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    env._provisioner.destroy.side_effect = RuntimeError("apiserver hiccup")
    with caplog.at_level("WARNING"):
        env.cleanup()
    assert env._provisioner.destroy.call_count == 2
    assert env._torn_down is True
    assert env._pod_ref is None
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors and "activeDeadlineSeconds" in errors[0].getMessage()
    # Still idempotent: a second cleanup never re-runs the failed destroy.
    env.cleanup()
    assert env._provisioner.destroy.call_count == 2


def test_cleanup_retry_runs_on_a_refreshed_client(monkeypatch):
    """The retry must rebuild the API client first: at interpreter exit the
    kubernetes client's own atexit hook may have already deleted the temp
    cert files it materialized from a kubeconfig with embedded cert data
    (kind, minikube), so every request on the cached client dies with an SSL
    FileNotFoundError. Reproduced live on kind: without the refresh, teardown
    fails and the session pod leaks until activeDeadlineSeconds."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    calls = []

    def _destroy(ref):
        calls.append("destroy")
        if calls.count("destroy") == 1:
            raise RuntimeError("SSLError(FileNotFoundError: temp cert gone)")

    env._provisioner.destroy.side_effect = _destroy
    env._provisioner.refresh_api.side_effect = lambda: calls.append("refresh")
    env.cleanup()
    # refresh happens between the failed attempt and the successful retry
    assert calls == ["destroy", "refresh", "destroy"]


def test_cleanup_mid_stream_does_not_break_inflight_exec(monkeypatch):
    """cleanup() nulls _pod_ref, but an in-flight exec holds its own captured
    ref/stream: the running command must finish, not crash on the null."""
    client = _FakeWSClient(open_cycles=6)
    env = _make_k8s_env(monkeypatch, [("", 0), client])
    handle = env._run_bash("echo hi")
    env.cleanup()
    # rc 0 proves the in-flight exec completed normally; the error path
    # returns rc 1 with an [kubernetes exec error] note instead.
    assert handle.wait(timeout=5) == 0
    assert env._pod_ref is None


def test_cancel_does_not_destroy_the_pod(monkeypatch):
    """_wait_for_process calls _kill_process() on an ORDINARY TIMEOUT as well
    as on a user interrupt (base.py:1127). Destroying the pod there deleted
    /workspace — every file the agent had just written — on any command that
    ran past its timeout, with no notice to the agent. Closing the websocket
    is enough: the kubelet reaps the exec'd process."""
    client = _FakeWSClient(open_cycles=10_000)
    env = _make_k8s_env(monkeypatch, [("", 0), client])
    handle = env._run_bash("sleep 1")
    handle.kill()
    # Wait for the worker to unwind — kill() may land before the exec thread
    # has even published its stream, and the close happens in its finally.
    handle.wait(timeout=5)
    assert client.closed is True
    env._provisioner.destroy.assert_not_called()
    assert env._pod_ref is not None


def test_session_recovers_when_the_pod_disappears(monkeypatch):
    """activeDeadlineSeconds / an operator TTL / an eviction can delete the pod
    under us. Without this the session bricks: every later command 404s with
    empty output until the idle reaper evicts the environment."""
    from kubernetes.client.exceptions import ApiException

    gone = _FakeWSClient(raise_on_update=ApiException(status=404, reason="Not Found"))
    env = _make_k8s_env(monkeypatch, [("", 0), gone, ("", 0), ("back\n", 0)])
    env.execute("ls")
    assert env._pod_ref is None

    env._provisioner.ensure.reset_mock()
    result = env.execute("echo back")
    env._provisioner.ensure.assert_called_once()
    assert env._pod_ref is not None
    assert "back" in result["output"]


def test_kill_racing_stream_open_is_not_erased(monkeypatch):
    """exec_fn used to set _cancelled = False AFTER _open_stream returned, so a
    kill() that landed while the stream was still opening was erased and the
    command kept running."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    opening = threading.Event()
    release = threading.Event()
    client = _FakeWSClient(open_cycles=10_000)

    def slow_stream(*args, **kwargs):
        opening.set()
        release.wait(5)
        return client

    monkeypatch.setattr("kubernetes.stream.stream", slow_stream)
    handle = env._run_bash("sleep 3600")
    assert opening.wait(5)
    handle.kill()          # cancel() runs with _active_stream still None
    release.set()
    assert handle.wait(timeout=5) == 130
    assert client.closed is True


def test_failed_ensure_does_not_orphan_a_pod(monkeypatch):
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    provisioner = MagicMock()
    provisioner.namespace = "hermes"
    provisioner.workspace_name.return_value = "hermes-ws-abc"
    provisioner.ensure.side_effect = TimeoutError("not Ready after 120s")

    with pytest.raises(TimeoutError):
        KubernetesEnvironment(
            provisioner=provisioner, task_id="abc",
            cwd="/workspace", timeout=30, sync_files=False,
        )
    provisioner.destroy.assert_called_once()


def test_agent_visible_cache_base_is_stable_across_cd(monkeypatch):
    env = _make_k8s_env(monkeypatch, [("", 0)])
    base = env.agent_visible_cache_base()
    env.cwd = "/somewhere/else"
    assert env.agent_visible_cache_base() == base == "/workspace/.hermes"


# ---------------------------------------------------------------------------
# terminal_tool integration
# ---------------------------------------------------------------------------


def _install_fake_backend(monkeypatch):
    captured = {}

    class _FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import tools.environments.kubernetes as k8s_mod

    monkeypatch.setattr(k8s_mod, "KubernetesEnvironment", _FakeEnv)
    monkeypatch.setattr(
        k8s_mod, "PodProvisioner", lambda *a, **kw: MagicMock(name="pod")
    )
    monkeypatch.setattr(k8s_mod, "load_core_api", lambda kcfg: MagicMock())
    monkeypatch.setattr(k8s_mod, "resolve_owner_reference", lambda *a, **kw: None)
    return captured, _FakeEnv


def test_factory_builds_kubernetes_env(monkeypatch):
    import tools.terminal_tool as tt

    captured, fake_cls = _install_fake_backend(monkeypatch)
    env = tt._create_environment(
        env_type="kubernetes", image="img:1", cwd="/workspace", timeout=30,
        container_config={"container_persistent": False,
                          "kubernetes": {"namespace": "hermes",
                                         "spec": _starter_object()["spec"]}},
        task_id="abc",
    )
    assert isinstance(env, fake_cls)
    assert captured["task_id"] == "abc"
    # Stateless: the environment takes no persistence, image or resource
    # arguments at all — the pod shape carries all of that.
    for gone in ("persistent", "image", "resources"):
        assert gone not in captured, captured


@pytest.mark.parametrize("bad", [
    {"kind": "Deployment", "apiVersion": "apps/v1"},
    {"kind": "SandboxClaim", "apiVersion": "extensions.agents.x-k8s.io/v1beta1"},
])
def test_factory_rejects_a_kind_it_has_no_provisioner_for(monkeypatch, bad):
    """The ONE Hermes-side config decision: kind selects which Kubernetes API
    is called, so an unimplemented kind never becomes a request the server
    would answer with a confusing 404. Everything else is the server's to
    validate."""
    import tools.terminal_tool as tt

    _install_fake_backend(monkeypatch)
    with pytest.raises(ValueError, match="unsupported apiVersion/kind"):
        tt._create_environment(
            env_type="kubernetes", image="img:1", cwd="/workspace", timeout=30,
            container_config={"kubernetes": dict({"namespace": "hermes"}, **bad)},
            task_id="abc",
        )


def test_check_requirements_kubernetes_missing_client(monkeypatch):
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "kubernetes"})
    import importlib.util as _ilu

    real_find_spec = _ilu.find_spec

    def fake_find_spec(name, *a, **k):
        if name == "kubernetes":
            return None
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(tt.importlib.util, "find_spec", fake_find_spec)
    assert tt.check_terminal_requirements() is False


def test_check_requirements_kubernetes_present(monkeypatch):
    import tools.terminal_tool as tt

    monkeypatch.setattr(
        tt, "_get_env_config",
        lambda: {"env_type": "kubernetes", "kubernetes": {"namespace": "hermes"}},
    )
    monkeypatch.setattr(tt.importlib.util, "find_spec", lambda name, *a, **k: object())
    assert tt.check_terminal_requirements() is True


def test_check_requirements_does_not_validate_config(monkeypatch):
    """Requirements gate = "is the client importable", nothing else. Config
    problems surface as the factory's ValueError or the API server's 400 —
    both name the offender, which a False here never could."""
    import tools.terminal_tool as tt

    monkeypatch.setattr(
        tt, "_get_env_config",
        lambda: {"env_type": "kubernetes", "kubernetes": {"provisioner": "nope"}},
    )
    monkeypatch.setattr(tt.importlib.util, "find_spec", lambda name, *a, **k: object())
    assert tt.check_terminal_requirements() is True


def test_live_terminal_tool_kubernetes_container_config(monkeypatch):
    """Regression pin for the two spots the upstream PR missed at first: the
    image-selection ladder and the container_config builder in terminal_tool().

    Drives the real terminal_tool() entry path with _create_environment stubbed,
    so no cluster is needed.
    """
    import uuid

    import tools.terminal_tool as tt

    unique_task_id = f"k8s-regression-{uuid.uuid4().hex}"
    with tt._env_lock:
        tt._active_environments.pop(unique_task_id, None)
        # _resolve_container_task_id collapses non-isolation task ids back to
        # "default", so a cached env another test left there would be reused and
        # _create_environment never called.
        tt._active_environments.pop("default", None)

    captured = {}

    def _fake_create_environment(**kwargs):
        captured.update(kwargs)
        mock_env = MagicMock()
        mock_env.execute.return_value = {"output": "", "returncode": 0}
        return mock_env

    monkeypatch.setattr(tt, "_create_environment", _fake_create_environment)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **kw: {"approved": True})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)

    monkeypatch.setenv("TERMINAL_ENV", "kubernetes")
    # The ONLY kubernetes env var: the internal JSON bridge payload. There is
    # deliberately no TERMINAL_KUBERNETES_POD_SA / _IMAGE / _NAMESPACE.
    # The image is authored in `spec` like every other pod field, and the
    # `kubernetes_image` display value is DERIVED from the rendered object.
    monkeypatch.setenv(
        "TERMINAL_KUBERNETES",
        json.dumps({"namespace": "hermes", "exec_container_name": "devbox",
                    "spec": {"containers": [
                        {"name": "devbox",
                         "image": "quay.io/hermes/session:1"}]}}),
    )

    # An isolation-keyed override keeps _resolve_container_task_id from
    # collapsing this session onto the shared "default" container.
    tt._task_env_overrides[unique_task_id] = {"env_type": "kubernetes"}
    try:
        tt.terminal_tool(command="echo hi", task_id=unique_task_id, force=True)
    finally:
        with tt._env_lock:
            tt._active_environments.pop(unique_task_id, None)
        tt._task_env_overrides.pop(unique_task_id, None)

    assert captured, "_create_environment was never called"
    cc = captured.get("container_config")
    assert cc is not None, "container_config was None — kubernetes missing from the builder"
    assert cc["kubernetes"]["exec_container_name"] == "devbox"
    assert cc["kubernetes"]["namespace"] == "hermes"
    # Defaults survive a partial payload.
    assert cc["kubernetes"]["kind"] == "Pod"
    assert cc["kubernetes"]["apiVersion"] == "v1"


def test_kubernetes_backend_default_cwd_comes_from_the_spec(monkeypatch):
    """The bridge reads it out of `spec`, so there is no second place to state
    it and nothing to keep in sync."""
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_ensure_terminal_env_bridged", lambda: None)
    monkeypatch.setenv("TERMINAL_ENV", "kubernetes")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("TERMINAL_KUBERNETES", json.dumps({
        "spec": {"containers": [{"name": "workspace", "workingDir": "/home/agent"}]},
    }))
    cfg = tt._get_env_config()
    assert cfg["cwd"] == "/home/agent"

    # TERMINAL_CWD still wins over the template, as for every other backend.
    monkeypatch.setenv("TERMINAL_CWD", "/srv/work")
    assert tt._get_env_config()["cwd"] == "/srv/work"


def test_stale_kubernetes_payload_does_not_break_other_backends(monkeypatch):
    """Bridged container-only env vars must not be parsed when another backend
    is selected — a malformed value would otherwise kill the local terminal."""
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_ensure_terminal_env_bridged", lambda: None)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_KUBERNETES", "{not json")
    cfg = tt._get_env_config()
    assert cfg["env_type"] == "local"
    assert cfg["kubernetes"] == {}


def test_kubernetes_is_a_container_backend():
    """Membership drives cwd sanitization, container_config assembly and the
    file-tool path translation."""
    import tools.terminal_tool as tt

    assert "kubernetes" in tt._CONTAINER_BACKENDS


# ---------------------------------------------------------------------------
# Exec client isolation  (CRITICAL: stream() monkeypatches api_client.request)
# ---------------------------------------------------------------------------


class _RecordingApiClient:
    """ApiClient stand-in that exposes the ``request`` attribute stream() swaps."""

    def __init__(self, configuration=None):
        self.configuration = configuration
        self.request = "REST"
        self.closed = False

    def close(self):
        self.closed = True


class _RecordingCoreV1Api:
    """Real class (not a Mock) so ``api_method.__self__`` resolves, like the SDK."""

    def __init__(self, api_client=None):
        self.api_client = api_client if api_client is not None else _RecordingApiClient()

    def connect_get_namespaced_pod_exec(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("stream() is stubbed in these tests")


def _sdk_like_stream(seen, barrier=None):
    """Reproduce what kubernetes.stream.stream() actually does.

    It is not a wrapper: it MONKEYPATCHES ``api_client.request`` with a
    websocket implementation and restores it in a ``finally`` — which is not
    reentrant.
    """

    def _stream(api_method, *args, **kwargs):
        client = api_method.__self__.api_client
        seen.append(client)
        previous = client.request
        client.request = "WEBSOCKET"
        try:
            if barrier is not None:
                barrier.wait(timeout=5)
            return _FakeWSClient(open_cycles=1)
        finally:
            client.request = previous

    return _stream


def _use_recording_client(monkeypatch):
    import kubernetes.client as kclient

    monkeypatch.setattr(kclient, "ApiClient", _RecordingApiClient)
    monkeypatch.setattr(kclient, "CoreV1Api", _RecordingCoreV1Api)


def test_every_exec_gets_its_own_api_client(monkeypatch):
    """The provisioner's CoreV1Api must never be handed to stream()."""
    shared = _RecordingCoreV1Api(_RecordingApiClient(configuration="CFG"))
    env = _make_k8s_env(monkeypatch, [("", 0)], api=shared)

    seen = []
    _use_recording_client(monkeypatch)
    monkeypatch.setattr("kubernetes.stream.stream", _sdk_like_stream(seen))

    env._open_stream(["true"]).close()
    env._open_stream(["true"]).close()

    assert len(seen) == 2
    assert seen[0] is not seen[1], "each exec needs its own ApiClient"
    assert all(client is not shared.api_client for client in seen)
    # ... but the shared client's auth/config still applies.
    assert all(client.configuration == "CFG" for client in seen)
    # ... and the throwaway clients are not leaked.
    assert all(client.closed for client in seen)
    assert shared.api_client.request == "REST"


def test_concurrent_execs_do_not_poison_the_shared_api_client(monkeypatch):
    """Two overlapping execs on ONE ApiClient interleave stream()'s
    save/restore, and the second restore installs the websocket partial
    permanently — after which every provisioner REST call (create/read/delete
    pod) tries to open a websocket against a plain HTTPS URL. Concurrency is
    routine here: gateway/TUI/desktop sessions collapse to one environment and
    the idle reaper calls cleanup() from another thread."""
    shared = _RecordingCoreV1Api(_RecordingApiClient(configuration="CFG"))
    env = _make_k8s_env(monkeypatch, [("", 0)], api=shared)

    seen = []
    barrier = threading.Barrier(2)
    _use_recording_client(monkeypatch)
    monkeypatch.setattr("kubernetes.stream.stream", _sdk_like_stream(seen, barrier))

    errors = []

    def _exec():
        try:
            env._open_stream(["true"]).close()
        except Exception as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=_exec) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert shared.api_client.request == "REST", (
        "the shared ApiClient was permanently poisoned by overlapping execs"
    )
    assert len(set(id(client) for client in seen)) == 2


# ---------------------------------------------------------------------------
# Container-name resolution: the exec target is KNOWN, not guessed
# ---------------------------------------------------------------------------


def test_container_name_points_into_the_template_it_does_not_build_one():
    """It used to name the container the default base built. With the template
    required, it is a pointer: change it alone and the manifest is unchanged,
    because Hermes posts what you wrote. A pointer at a container you did not
    write fails at exec (see test_exec_container_error_names_what_it_found)."""
    template = render_session_object(_kcfg(exec_container_name="devbox"))
    assert [c["name"] for c in template["spec"]["containers"]] == ["workspace"]


def test_pod_ensure_targets_the_configured_container():
    api = MagicMock()
    api.read_namespaced_pod.return_value = _running_pod(containers=("devbox",))
    p = _provisioner_with_api(api, exec_container_name="devbox")
    ref = p.ensure("abc")
    assert ref.container == "devbox"


def test_exec_container_prefers_the_configured_name_when_the_pod_has_it():
    p = _provisioner_with_api(MagicMock())
    pod = _running_pod(containers=("istio-proxy", "workspace"))
    assert p.exec_container(pod) == "workspace"


def test_exec_container_refuses_a_pod_whose_container_list_is_unreadable():
    """Fail-open is the wrong default for a check whose whole purpose is to
    prove the RUNNING pod matches the submitted one. "We could not establish
    that the pod carries the container we rendered" is the same failure as "it
    carries a different one"."""
    p = _provisioner_with_api(MagicMock())
    for unreadable in (SimpleNamespace(),
                       SimpleNamespace(spec=SimpleNamespace(containers=[]))):
        with pytest.raises(RuntimeError) as caught:
            p.exec_container(unreadable)
        # Still fails closed, and still says which knob to look at.
        assert "no readable container list" in str(caught.value)
        assert "exec_container_name" in str(caught.value)


def test_exec_container_refuses_a_pod_that_lacks_the_rendered_container():
    """The old code silently exec'd into names[0]. `exec_container_name` is the
    exec target SELECTOR, so a pod that lacks it is either not the pod this
    backend rendered or a pod whose container the `spec` renamed — and the very
    next thing that happens is a credential-file upload into whatever we
    exec'd into.

    The message must name the KNOB. It used to say the running pod "does not
    match the template Hermes submitted", which blames drift that did not
    happen and leaves the operator with nothing to change."""
    p = _provisioner_with_api(MagicMock())
    pod = _running_pod(containers=("istio-proxy", "somebody-elses-shell"))
    with pytest.raises(RuntimeError) as caught:
        p.exec_container(pod)
    message = str(caught.value)
    assert "exec_container_name" in message
    assert "terminal.kubernetes.spec.containers[]" in message
    # It lists what the pod actually has, so the fix is copy-pasteable.
    assert "istio-proxy" in message and "somebody-elses-shell" in message
    assert "does not match the template" not in message


# ---------------------------------------------------------------------------
# Unknown exit status is a failure, not a success
# ---------------------------------------------------------------------------


def test_unknown_exit_status_is_reported_as_failure(monkeypatch):
    """WSClient.returncode raises (empty error channel) or returns None (stream
    still open). Both mean "unknown" — reporting 0 told the model a failed or
    half-killed command had succeeded."""
    client = _FakeWSClient(stdout="partial output\n",
                           returncode=TypeError("NoneType not subscriptable"))
    env = _make_k8s_env(monkeypatch, [("", 0), client])
    result = env.execute("something")
    assert "partial output" in result["output"]
    assert result["returncode"] != 0
    assert "status unavailable" in result["output"]


def test_exec_capture_raises_instead_of_reading_a_running_returncode(monkeypatch):
    env = _make_k8s_env(monkeypatch, [("", 0)])
    stuck = _FakeWSClient(open_cycles=10_000, returncode=None, update_sleep=0.05)
    monkeypatch.setattr("kubernetes.stream.stream", lambda *a, **k: stuck)
    with pytest.raises(TimeoutError):
        env._exec_capture(["sh", "-c", "sleep 999"], timeout=1)


# ---------------------------------------------------------------------------
# File-sync transport (no payload in exec argv, failures are loud)
# ---------------------------------------------------------------------------


def _recording_sync_stream(commands, clients, returncodes):
    def _stream(*args, **kwargs):
        commands.append(list(kwargs.get("command") or ()))
        index = min(len(clients), len(returncodes) - 1)
        client = _FakeWSClient(returncode=returncodes[index])
        clients.append(client)
        return client

    return _stream


def test_bulk_upload_streams_the_payload_over_stdin_never_argv(monkeypatch, tmp_path):
    """ws_client.get_websocket_url appends every argv element to the exec
    request URL, and kube-apiserver records requestURI in the audit log at
    Metadata level and above. iter_sync_files() starts with the agent's
    credential files, so an argv transport writes them into the cluster audit
    log. The chunked-argv fallback that did this is gone."""
    secret = tmp_path / "creds.json"
    secret.write_bytes(b'{"api_key": "SUPER-SECRET-TOKEN"}' * 400)  # > one argv chunk

    env = _make_k8s_env(monkeypatch, [("", 0)])
    commands, clients = [], []
    monkeypatch.setattr(
        "kubernetes.stream.stream", _recording_sync_stream(commands, clients, [0])
    )

    env._bulk_upload([(str(secret), "/workspace/.hermes/creds.json")])

    argv = " ".join(" ".join(command) for command in commands)
    assert "SUPER-SECRET-TOKEN" not in argv
    assert all(
        len(part) < 1024 for command in commands for part in command
    ), "the payload must never travel through exec argv (apiserver audit log)"

    written = "".join(clients[-1].stdin_writes)
    assert written.rstrip().endswith("__HERMES_TAR_EOF__")
    payload = base64.b64decode(written.split("__HERMES_TAR_EOF__")[0])
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        member = tar.getmember("workspace/.hermes/creds.json")
        assert tar.extractfile(member).read() == secret.read_bytes()


def test_no_argv_chunking_fallback_remains():
    assert not hasattr(KubernetesEnvironment, "_upload_tar_base64")


def test_stdin_upload_half_closes_when_v5_is_available(monkeypatch):
    env = _make_k8s_env(monkeypatch, [("", 0)])
    client = _FakeWSClient(returncode=0, subprotocol="v5.channel.k8s.io")
    monkeypatch.setattr("kubernetes.stream.stream", lambda *a, **k: client)
    env._stdin_upload("Zm9v\n")
    assert client.channels_closed == [0]


def test_bulk_upload_treats_an_unknown_exit_status_as_failure(monkeypatch, tmp_path):
    """`if rc not in (0, None)` whitelisted None as success, so a timed-out or
    abnormally-closed tar extract reported a completed upload."""
    payload = tmp_path / "a.txt"
    payload.write_text("x")
    env = _make_k8s_env(monkeypatch, [("", 0)])
    commands, clients = [], []
    monkeypatch.setattr(
        "kubernetes.stream.stream",
        _recording_sync_stream(commands, clients,
                               [0, TypeError("empty error channel")]),
    )
    with pytest.raises(RuntimeError, match="unknown"):
        env._bulk_upload([(str(payload), "/workspace/a.txt")])


def test_stdin_upload_raises_when_the_deadline_expires(monkeypatch):
    env = _make_k8s_env(monkeypatch, [("", 0)])
    stuck = _FakeWSClient(open_cycles=10_000, returncode=None, update_sleep=0.05)
    monkeypatch.setattr("kubernetes.stream.stream", lambda *a, **k: stuck)
    with pytest.raises(TimeoutError):
        env._stdin_upload("Zm9v\n", timeout=1)


def test_failed_upload_does_not_mark_files_as_synced(monkeypatch, tmp_path):
    """FileSyncManager commits its state whenever the transport returns without
    raising. A silent failure therefore marked the credential files as synced
    and never retried them — the session pod runs without them, with nothing in
    the logs."""
    from tools.environments.file_sync import FileSyncManager

    creds = tmp_path / "creds.json"
    creds.write_text("secret")
    env = _make_k8s_env(monkeypatch, [("", 0)])
    commands, clients = [], []
    monkeypatch.setattr(
        "kubernetes.stream.stream",
        _recording_sync_stream(commands, clients,
                               [0, TypeError("empty error channel")]),
    )
    manager = FileSyncManager(
        get_files_fn=lambda: [(str(creds), "/workspace/.hermes/creds.json")],
        upload_fn=env._upload_file,
        delete_fn=env._delete_files,
        bulk_upload_fn=env._bulk_upload,
    )
    manager.sync(force=True)
    assert manager._synced_files == {}, (
        "a failed upload must not be recorded as synced"
    )


# ---------------------------------------------------------------------------
# Approval trust: same as every other containerized backend
# ---------------------------------------------------------------------------


def test_kubernetes_skips_the_dangerous_command_guards_when_trusted():
    """trusted_sandbox (default) treats the pod as disposable and skips the
    approval layer; setting it false keeps the prompts on."""
    from tools.approval import _should_skip_container_guards
    from tools.environments.kubernetes import (
        merge_kubernetes_config, trusted_sandbox,
    )

    assert trusted_sandbox(merge_kubernetes_config({})) is True
    assert trusted_sandbox(merge_kubernetes_config({"trusted_sandbox": False})) is False
    # Skip is docker-shaped: gated on has_host_access, which the terminal tool
    # sets to `not trusted_sandbox`.
    assert _should_skip_container_guards("kubernetes", has_host_access=False) is True
    assert _should_skip_container_guards("kubernetes", has_host_access=True) is False


def test_trusted_sandbox_drives_the_host_access_signal():
    """The terminal tool routes kubernetes through the same host-access input
    as docker, derived from trusted_sandbox."""
    from tools.terminal_tool import _docker_has_host_access

    assert _docker_has_host_access(
        {"env_type": "kubernetes", "kubernetes": _kcfg()}
    ) is False
    untrusted = _kcfg(trusted_sandbox=False)
    assert _docker_has_host_access(
        {"env_type": "kubernetes", "kubernetes": untrusted}
    ) is True


def test_the_hardline_floor_still_applies_to_host_backends_only():
    """Skipping the guards must not be confused with skipping the hardline
    blocklist, which exists for environments that can damage the host."""
    from tools.approval import _should_skip_container_guards

    for host_backend in ("local", "ssh"):
        assert _should_skip_container_guards(host_backend) is False


# ---------------------------------------------------------------------------
# The managed-by label survives the merge (adoption + the admin's NetworkPolicy)
# ---------------------------------------------------------------------------


def test_the_managed_by_label_is_on_the_rendered_template():
    template = render_session_object(
        _kcfg(pod_template={"metadata": {"labels": {"team": "hermes"}}}),
    )
    assert template["metadata"]["labels"][MANAGED_BY] == "hermes-agent"
    assert template["metadata"]["labels"]["team"] == "hermes"


def test_the_managed_by_label_is_a_default_like_everything_else():
    """It used to be stamped AFTER the merge, which made these two keys the
    only un-overridable fields in the whole template — in a backend whose
    stated contract is "nothing is reserved". Nothing in this backend selects
    on the label (pods are found by name), so an override costs the operator
    only what reads it from outside: the shipped NetworkPolicy and
    ValidatingAdmissionPolicy."""
    template = render_session_object(
        _kcfg(pod_template={"metadata": {"labels": {MANAGED_BY: "Helm"}}}),
    )
    assert template["metadata"]["labels"][MANAGED_BY] == "Helm"


def test_deleting_the_label_node_falls_back_to_the_default():
    """RFC 7386 null-deletion of metadata (or of labels) removes the base's
    labels — the renderer must still produce a labels dict rather than a
    template the API server rejects, and re-seeds the default because the
    user expressed no opinion about the KEY, only about the node."""
    for overlay in ({"metadata": None},
                    {"metadata": {"labels": None}},
                    {"metadata": {"labels": 7}}):
        template = render_session_object(_kcfg(pod_template=overlay))
        assert template["metadata"]["labels"][MANAGED_BY] == "hermes-agent", overlay


# ---------------------------------------------------------------------------
# Adoption requires ownership proof
# ---------------------------------------------------------------------------


def test_adoption_fails_closed_without_an_owner_reference():
    """With no agent identity the ownership check used to `return True`.

    A 30-scenario live run proved the consequence: a pod created by ANOTHER
    Hermes was adopted with no log line, credential files uploaded into it,
    and the object deleted at teardown. The pod name is not evidence — it is
    derived from a hostname and a pid, and anyone with `get pods` can read it.
    The instance label is, and a foreign object carries a different one."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    p = PodProvisioner(_kcfg(), "hermes", api=api, owner_reference=None)

    api.read_namespaced_pod.return_value = _running_pod(
        labels={MANAGED_BY: "hermes-agent",
                "hermes.nousresearch.com/instance": "someone-else"})
    assert p._is_ours("hermes-ws-x-default") is False

    api.read_namespaced_pod.return_value = _running_pod(
        labels={MANAGED_BY: "hermes-agent",
                "hermes.nousresearch.com/instance": p._instance})
    assert p._is_ours("hermes-ws-x-default") is True


def test_yaml_boolean_off_disables_owner_references():
    """The Kubernetes docs page tells operators to write `owner_reference: off`, and YAML
    1.1 parses an unquoted `off` as the boolean False. Comparing only against
    the string made the documented syntax a silent no-op."""
    import yaml

    from tools.environments.kubernetes import owner_reference_disabled

    parsed = yaml.safe_load("owner_reference: off")
    assert parsed["owner_reference"] is False
    assert owner_reference_disabled(merge_kubernetes_config(parsed)) is True
    assert owner_reference_disabled(merge_kubernetes_config(
        {"owner_reference": "off"})) is True
    assert owner_reference_disabled(merge_kubernetes_config({})) is False


def test_a_sidecar_with_a_shared_pid_namespace_is_warned_about():
    """shareProcessNamespace is required for background completion AND makes
    /proc shared: verified live that the agent shell reads a sidecar's
    /proc/<pid>/environ and a Secret mounted only into it. The template
    comment called that "harmless here: there is one container" — true until
    an operator adds the mesh proxy the code elsewhere anticipates."""
    from tools.environments.kubernetes import preflight_spec

    two = _starter_object()
    two["spec"]["containers"].append({"name": "istio-proxy", "image": "proxy:1"})
    errors, warnings = preflight_spec(_kcfg(pod_template_raw=two))
    assert not errors
    assert any("istio-proxy" in w and "/proc" in w for w in warnings), warnings
    # The shipped single-container template must stay clean.
    assert preflight_spec(_kcfg()) == ([], [])


def test_ownership_check_follows_a_relabelled_template():
    """`_is_ours` compared against a hardcoded "hermes-agent". Once
    metadata.labels became overridable, an operator who relabels their session
    pods would fail every adoption check against their OWN pods: the 409-resume
    path would refuse to reuse or delete them, so each one leaks. The
    ownerReference UID is the ownership proof; the label is a cheap filter and
    has to track the config."""
    from kubernetes.client.exceptions import ApiException

    relabelled = {"metadata": {"labels": {MANAGED_BY: "platform-team"}}}
    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    api.read_namespaced_pod.return_value = _running_pod(
        labels={MANAGED_BY: "platform-team"}, owners=[],
    )
    p = _provisioner_with_api(api, pod_template=relabelled)
    # Reaches the ownerReference check (which rejects an unowned pod) instead
    # of bailing out at the label.
    with pytest.raises(RuntimeError, match="not created by this Hermes instance"):
        p.ensure("abc")

    # And a pod carrying the DEFAULT label is not ours once we have relabelled.
    api.read_namespaced_pod.return_value = _running_pod(
        labels={MANAGED_BY: "hermes-agent"}, owners=[],
    )
    p2 = _provisioner_with_api(api, pod_template=relabelled)
    assert p2._is_ours("hermes-ws-x-abc") is False


def test_ensure_refuses_an_unowned_pod_that_carries_our_label():
    """`or not owners` accepted any labelled pod with no ownerReferences —
    which is exactly what another agent's workspace looks like."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    api.read_namespaced_pod.return_value = _running_pod(
        labels={MANAGED_BY: "hermes-agent"}, owners=[],
    )
    p = _provisioner_with_api(api)
    with pytest.raises(RuntimeError, match="not created by this Hermes instance"):
        p.ensure("abc")


# ---------------------------------------------------------------------------
# `spec` is all-or-nothing: the shipped default when unset, posted verbatim when set
# ---------------------------------------------------------------------------
#
# There is deliberately no partial merge over a base: an operator could not
# predict their own pod without knowing an invisible base and simulating a
# merge in their head. These tests pin the rule: what you write is what
# is posted, and the only additions are metadata Hermes must compute.


def test_your_metadata_passes_through_untouched():
    """`metadata` is as much yours as `spec`. Annotations and extra labels are
    posted as written — only name/namespace are always Hermes'."""
    pod = PodProvisioner(
        _kcfg(pod_template={"metadata": {
            "annotations": {"kubectl.kubernetes.io/default-container": "workspace",
                            "cost-center": "ml-platform"},
            "labels": {"app.kubernetes.io/part-of": "hermes"},
            "finalizers": ["example.com/cleanup"],
        }}),
        "hermes", api=None, owner_reference=OWNER_REF,
    ).pod_manifest("abc")
    md = pod["metadata"]
    assert md["annotations"] == {"kubectl.kubernetes.io/default-container": "workspace",
                                 "cost-center": "ml-platform"}
    assert md["labels"]["app.kubernetes.io/part-of"] == "hermes"
    assert md["finalizers"] == ["example.com/cleanup"]
    # ...and the two that cannot be yours are still Hermes'.
    assert md["namespace"] == "hermes"
    assert md["name"].startswith("hermes-ws-")


def test_a_template_owner_reference_is_kept_and_hermes_appends_its_own():
    """Honouring a user-supplied ownerReference by ITSELF was a blocker.

    `_is_ours` proves ownership by finding the agent pod's UID among the
    pod's owners. Letting a user list replace ours produced a pod Hermes had
    just created and could then neither adopt nor delete: the 409 path refused
    it, the terminal was dead for the life of the process, and the pod leaked
    until its deadline. Kubernetes allows several owners, so both intents
    survive by appending."""
    mine = [{"apiVersion": "example.com/v1", "kind": "Widget",
             "name": "w1", "uid": "9999", "controller": True}]
    provisioner = PodProvisioner(
        _kcfg(pod_template={"metadata": {"ownerReferences": mine}}),
        "hermes", api=None, owner_reference=OWNER_REF,
    )
    pod = provisioner.pod_manifest("abc")
    assert pod["metadata"]["ownerReferences"] == mine + [OWNER_REF]

    # The whole point: our own pod is recognisable as ours.
    provisioner._api = MagicMock()
    # The client returns attribute-style objects, not dicts.
    provisioner._api.read_namespaced_pod.return_value = _running_pod(
        labels=pod["metadata"]["labels"],
        owners=[SimpleNamespace(uid=o["uid"])
                for o in pod["metadata"]["ownerReferences"]],
    )
    assert provisioner._is_ours("hermes-ws-x-default") is True

    # Idempotent: rendering twice must not accumulate duplicate owners.
    assert provisioner.pod_manifest("abc")["metadata"]["ownerReferences"] == \
        mine + [OWNER_REF]

    # Absent -> Hermes supplies the agent pod, as before.
    default = PodProvisioner(
        _kcfg(), "hermes", api=None, owner_reference=OWNER_REF,
    ).pod_manifest("abc")
    assert default["metadata"]["ownerReferences"] == [OWNER_REF]


def test_the_pod_name_is_never_taken_from_the_template():
    """It is how the pod is found again on the next command; honouring a
    template-supplied name would strand every pod after the first."""
    pod = PodProvisioner(
        _kcfg(pod_template={"metadata": {"name": "my-own-pod",
                                         "namespace": "elsewhere"}}),
        "hermes", api=None, owner_reference=OWNER_REF,
    ).pod_manifest("abc")
    assert pod["metadata"]["name"] != "my-own-pod"
    assert pod["metadata"]["namespace"] == "hermes"


def test_a_stuck_terminating_pod_is_named_not_blamed_on_the_image_pull(monkeypatch):
    """`_wait_pod_gone` used to fall off the end silently, handing a pod stuck
    Terminating to wait_pod_ready — which burned the whole ready timeout and
    then suggested RAISING it, the one knob that makes the hang longer.
    Observed live: two session pods Terminating for 12h on kata nodes."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    terminating = _running_pod(labels={MANAGED_BY: "hermes-agent"},
                               owners=[SimpleNamespace(uid=OWNER_REF["uid"])])
    terminating.metadata.deletion_timestamp = "2026-08-09T00:00:00Z"
    terminating.metadata.uid = "corpse-uid"
    api.read_namespaced_pod.return_value = terminating

    # Fake clock: the real path waits 30s per attempt, which is not something
    # a unit test should spend twice.
    import tools.environments.kubernetes as k8s_mod

    ticks = iter([0.0] + [i * 10.0 for i in range(1, 200)])
    monkeypatch.setattr(k8s_mod.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(k8s_mod.time, "sleep", lambda _s: None)

    p = _provisioner_with_api(api)
    with pytest.raises(RuntimeError, match="stuck Terminating") as caught:
        p.ensure("abc")
    # It must not send the operator to the knob that lengthens the hang.
    assert "ready_timeout_seconds will not help" in str(caught.value)
    assert "kubectl describe pod" in str(caught.value)


def test_preflight_warns_about_secrets_in_the_session_pod():
    """This is the ONE rule the shipped ValidatingAdmissionPolicy caught that
    nothing else did. The policy is gone: it hooked on a label the pod creator
    chooses (`owned_selector`), so rebranding skipped it silently, and an agent
    holding `pods create` could omit the label anyway — it defended against
    your own config, not a compromised agent. Checked here instead, where it
    fires at config time and names the field."""
    from tools.environments.kubernetes import preflight_spec

    obj = _starter_object()
    obj["spec"]["volumes"].append(
        {"name": "creds", "secret": {"secretName": "provider-keys"}})
    obj["spec"]["containers"][0]["envFrom"] = [
        {"secretRef": {"name": "provider-keys"}}]
    errors, warnings = preflight_spec(_kcfg(pod_template_raw=obj))
    assert not errors
    assert any("Secret" in w and "creds" in w for w in warnings), warnings
    # The shipped object stays clean.
    assert preflight_spec(_kcfg()) == ([], [])


def test_preflight_catches_what_the_api_server_cannot():
    """Five realistic omissions all passed fieldValidation=Strict AND produced
    a green `hermes doctor`, then misbehaved with no operator-facing signal.
    Every check here is a question about how Hermes USES the pod, which no
    admission controller has an opinion on — so nobody asks it unless we do."""
    from tools.environments.kubernetes import preflight_spec

    # A container the exec target does not name: valid Kubernetes, useless pod.
    errors, _ = preflight_spec(_kcfg(
        exec_container_name="workspace",
        pod_template_raw={"spec": {"containers": [
            {"name": "sidecar", "workingDir": "/w"}]}},
    ))
    assert errors and "exec_container_name" in errors[0]
    assert "'sidecar'" in errors[0] or "sidecar" in errors[0]

    # readOnlyRootFilesystem with no /tmp mount: env tracking dies silently.
    errors, _ = preflight_spec(_kcfg(pod_template_raw={"spec": {
        "containers": [{"name": "workspace", "workingDir": "/w",
                        "securityContext": {"readOnlyRootFilesystem": True}}],
    }}))
    assert any("/tmp" in e for e in errors)

    # Warnings, not errors: these work until they do not.
    _, warnings = preflight_spec(_kcfg(pod_template_raw={"spec": {
        "containers": [{"name": "workspace"}]}}))
    assert any("workingDir" in w for w in warnings)
    assert any("shareProcessNamespace" in w for w in warnings)

    # The shipped starter template must be clean on both counts, or the thing
    # we tell operators to copy trips our own checks.
    errors, warnings = preflight_spec(_kcfg())
    assert not errors and not warnings, (errors, warnings)


def test_the_factory_refuses_a_spec_that_cannot_serve_a_session(monkeypatch):
    """Hard errors are caught BEFORE a pod is created and an image pulled.
    The container-name mismatch was previously discovered only after create +
    pull + wait_pod_ready, with a message that blamed template drift."""
    import tools.terminal_tool as tt

    _install_fake_backend(monkeypatch)
    with pytest.raises(ValueError, match="cannot serve a session"):
        tt._create_environment(
            env_type="kubernetes", image="img:1", cwd="/workspace", timeout=30,
            container_config={"kubernetes": {
                "namespace": "hermes", "exec_container_name": "workspace",
                "spec": {"containers": [{"name": "devbox"}]},
            }},
            task_id="abc",
        )


def test_a_missing_spec_falls_back_to_the_shipped_default():
    """An unset spec means the documented default pod — the exact
    STARTER_SESSION_OBJECT spec, not an approximation — and it must pass
    preflight clean. metadata alone is not a pod; the fallback covers it too."""
    from tools.environments.kubernetes import (
        STARTER_SESSION_OBJECT, preflight_spec,
    )

    for absent in ({}, {"metadata": {"labels": {"a": "b"}}}, None, "", []):
        kcfg = _kcfg(pod_template_raw=absent)
        rendered = render_session_object(kcfg)
        assert rendered["spec"] == STARTER_SESSION_OBJECT["spec"], absent
        assert preflight_spec(kcfg) == ([], []), absent


def test_the_default_spec_is_not_shared_between_renders():
    """The fallback must be a fresh copy each time: a caller mutating one
    rendered object must not poison the next pod or the shipped constant."""
    from tools.environments.kubernetes import STARTER_SESSION_OBJECT

    first = render_session_object(_kcfg(pod_template_raw={}))
    first["spec"]["containers"][0]["image"] = "mutated:1"
    second = render_session_object(_kcfg(pod_template_raw={}))
    assert second["spec"]["containers"][0]["image"] == "ubuntu:26.04"
    assert STARTER_SESSION_OBJECT["spec"]["containers"][0]["image"] == "ubuntu:26.04"


def test_the_spec_is_posted_verbatim():
    """Byte-for-byte: no key added to spec, none removed."""
    spec = {
        "containers": [{"name": "workspace", "image": "alpine:3.20",
                        "command": ["sleep", "infinity"]}],
        "restartPolicy": "Never",
        "activeDeadlineSeconds": 60,
    }
    rendered = render_session_object(_kcfg(pod_template_raw={"spec": deepcopy(spec)}))
    assert rendered["spec"] == spec


def test_nothing_is_supplied_that_you_did_not_write():
    """The old base contributed shareProcessNamespace, terminationGracePeriod,
    enableServiceLinks, host* flags, securityContext, volumes and a whole
    container. A template that omits them now gets a pod without them — which
    is the point: no invisible layer."""
    rendered = render_session_object(_kcfg(pod_template_raw={
        "spec": {"containers": [{"name": "workspace", "image": "alpine:3.20"}]},
    }))
    spec = rendered["spec"]
    for absent in ("shareProcessNamespace", "terminationGracePeriodSeconds",
                   "enableServiceLinks", "hostNetwork", "hostPID", "hostIPC",
                   "securityContext", "volumes", "restartPolicy",
                   "automountServiceAccountToken", "activeDeadlineSeconds"):
        assert absent not in spec, absent
    assert spec["containers"] == [{"name": "workspace", "image": "alpine:3.20"}]


def test_render_does_not_mutate_the_configured_template():
    """render_session_object runs once per create; mutating the config would make
    the second pod differ from the first."""
    template = {"spec": {"containers": [{"name": "workspace",
                                         "image": "alpine:3.20"}]}}
    kcfg = _kcfg(pod_template_raw=template)
    before = deepcopy(template)
    render_session_object(kcfg, "inst1")
    render_session_object(kcfg, "inst2")
    assert template == before
    assert "metadata" not in template


def test_the_shipped_starter_template_renders_a_usable_pod():
    """The Kubernetes docs page is what the error message tells operators
    to copy, so it has to satisfy every REQUIRED-BY-HERMES field it documents.
    A drift here means the starting point does not start."""
    template = _starter_object()
    spec = render_session_object(_kcfg(pod_template_raw=template))["spec"]

    container = next(c for c in spec["containers"] if c["name"] == "workspace")
    assert container["command"] == ["sleep", "infinity"]
    assert container["workingDir"] == "/workspace"
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert mounts == {"workspace": "/workspace", "tmp": "/tmp"}
    assert {v["name"] for v in spec["volumes"]} == {"workspace", "tmp"}
    # Background completion detection depends on a reaping PID 1.
    assert spec["shareProcessNamespace"] is True
    assert spec["restartPolicy"] == "Never"

# ---------------------------------------------------------------------------
# Misc correctness fixes
# ---------------------------------------------------------------------------


def test_namespace_is_not_resolved_from_an_environment_variable(monkeypatch):
    """HERMES_POD_NAMESPACE was a redundant third source: in-cluster the kubelet
    projects the namespace and out-of-cluster the kubeconfig context or the
    config key covers it."""
    import tools.environments.kubernetes as k8s_mod

    monkeypatch.setenv("HERMES_POD_NAMESPACE", "from-env")
    monkeypatch.setenv("KUBECONFIG", "/nonexistent/kubeconfig")
    monkeypatch.setattr(k8s_mod, "_SA_NAMESPACE_FILE", "/nonexistent/namespace")
    with pytest.raises(ValueError):
        k8s_mod.resolve_namespace(_kcfg())
    assert k8s_mod.resolve_namespace(_kcfg(namespace="explicit")) == "explicit"


_KUBECONFIG_WITH_NAMESPACES = """\
apiVersion: v1
kind: Config
current-context: dev
clusters:
  - name: c
    cluster:
      server: https://kubeconfig.invalid
contexts:
  - name: dev
    context:
      cluster: c
      namespace: from-active-context
  - name: prod
    context:
      cluster: c
      namespace: from-prod-context
  - name: bare
    context:
      cluster: c
users: []
"""


def test_namespace_resolution_prefers_kubeconfig_context_over_sa_file(
        monkeypatch, tmp_path):
    """Empty `namespace` resolves in this order: the kubeconfig context's
    default namespace, then the projected ServiceAccount namespace file.
    An operator driving a remote cluster from inside another cluster's pod
    should land where their kubeconfig points, not in the pod's namespace."""
    import tools.environments.kubernetes as k8s_mod

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text(_KUBECONFIG_WITH_NAMESPACES, encoding="utf-8")
    sa_file = tmp_path / "sa-namespace"
    sa_file.write_text("from-sa-file", encoding="utf-8")
    monkeypatch.setattr(k8s_mod, "_SA_NAMESPACE_FILE", str(sa_file))

    # The autouse stub has no list_kube_config_contexts; give it one with
    # the SDK's documented (all_contexts, active_context) return shape.
    def _list_contexts(config_file=None):
        data = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
        contexts = data.get("contexts") or []
        current = data.get("current-context")
        active = next((c for c in contexts if c.get("name") == current), None)
        return contexts, active

    monkeypatch.setattr(
        sys.modules["kubernetes"].config, "list_kube_config_contexts",
        _list_contexts, raising=False)

    # Active context's namespace beats the SA file.
    assert k8s_mod.resolve_namespace(
        _kcfg(kubeconfig=str(kubeconfig))) == "from-active-context"
    # `context` selects a non-active context's namespace.
    assert k8s_mod.resolve_namespace(
        _kcfg(kubeconfig=str(kubeconfig), context="prod")) == "from-prod-context"
    # The explicit key still beats everything.
    assert k8s_mod.resolve_namespace(
        _kcfg(kubeconfig=str(kubeconfig), namespace="explicit")) == "explicit"
    # kubectl semantics: a context with no namespace field means "default"
    # (a fresh kind/minikube kubeconfig sets none) — NOT the SA file.
    assert k8s_mod.resolve_namespace(
        _kcfg(kubeconfig=str(kubeconfig), context="bare")) == "default"
    # No matching context at all -> the SA file.
    assert k8s_mod.resolve_namespace(
        _kcfg(kubeconfig=str(kubeconfig), context="nonexistent")) == "from-sa-file"


def test_kubernetes_blob_is_only_bridged_for_the_kubernetes_backend():
    """The ~1.2KB JSON payload used to land in the environment of every child
    process the agent spawns, for every backend."""
    from hermes_cli.config import apply_terminal_config_to_env

    local_env = apply_terminal_config_to_env(
        env={}, config={"terminal": {"backend": "local", "kubernetes": {"namespace": "x"}}},
        override=True,
    )
    assert "TERMINAL_KUBERNETES" not in local_env

    k8s_env = apply_terminal_config_to_env(
        env={},
        config={"terminal": {"backend": "kubernetes", "kubernetes": {"namespace": "x"}}},
        override=True,
    )
    assert json.loads(k8s_env["TERMINAL_KUBERNETES"])["namespace"] == "x"


# ---------------------------------------------------------------------------
# The shipped cluster manifests say what they actually do
#
# The controls are documented in website/docs/user-guide/kubernetes.md now,
# failure mode with real consequences: an operator who believes Hermes is
# enforcing something stops enforcing it themselves.
# ---------------------------------------------------------------------------

def test_sanitize_name_hashes_case_only_normalisation():
    """The collision guard compared the slug against a LOWERCASED copy of the
    input, so case-only normalisation never got the hash suffix and two task ids
    differing only in case shared one pod."""
    assert sanitize_name("Default") != sanitize_name("default")
    assert sanitize_name("Foo-Bar") != sanitize_name("foo-bar")
    assert sanitize_name("default") == "default"


def test_kill_cancels_only_its_own_exec(monkeypatch):
    """_active_stream/_cancelled were single-slot per environment. Because
    _wait_for_process calls _kill_process() on an ORDINARY TIMEOUT (base.py),
    command A timing out closed command B's websocket: B returned rc=130
    'interrupted' with truncated output while A ran on untouched."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    # update_sleep keeps both execs genuinely in flight while kill() lands.
    clients = {"a": _FakeWSClient(open_cycles=10_000, update_sleep=0.01),
               "b": _FakeWSClient(open_cycles=10_000, update_sleep=0.01)}
    handed: list[str] = []
    opened = {"a": threading.Event(), "b": threading.Event()}
    lock = threading.Lock()

    def fake_stream(*args, **kwargs):
        with lock:
            if "a" not in handed:
                key = "a"
            elif "b" not in handed:
                key = "b"
            else:
                # cancel() opens a THIRD stream of its own to kill the remote
                # process tree. It must get its own client, not be handed a
                # sibling exec's — that would make this test's own harness the
                # thing that closes B.
                return _FakeWSClient(open_cycles=2)
            handed.append(key)
        client = clients[key]
        opened[key].set()
        return client

    monkeypatch.setattr("kubernetes.stream.stream", fake_stream)
    handle_a = env._run_bash("sleep 3600")
    assert opened["a"].wait(5)
    handle_b = env._run_bash("sleep 3600")
    assert opened["b"].wait(5)
    time.sleep(0.1)  # let B's worker publish its stream

    handle_a.kill()
    assert handle_a.wait(timeout=5) == 130
    assert clients["a"].closed is True
    assert clients["b"].closed is False, "kill() closed an unrelated exec"
    assert handle_b.poll() is None, "an unrelated exec was reported interrupted"
    handle_b.kill()


def test_kubernetes_blob_survives_a_backend_selected_by_env_var():
    """terminal.backend ALWAYS defaults to 'local' in the merged config, so the
    round-2 gate's TERMINAL_ENV fallback was dead code: selecting the kubernetes
    backend by environment variable silently dropped the whole
    terminal.kubernetes.* block and the backend ran on DEFAULT_KUBERNETES_CONFIG
    with nothing logged."""
    from hermes_cli.config import apply_terminal_config_to_env

    merged_like = {"terminal": {"backend": "local",
                                "kubernetes": {"namespace": "hermes-agents"}}}
    out = apply_terminal_config_to_env(
        env={"TERMINAL_ENV": "kubernetes"}, config=merged_like, override=False,
    )
    assert json.loads(out["TERMINAL_KUBERNETES"])["namespace"] == "hermes-agents"

    # An EXPLICIT config backend still wins over the ambient env var, so the
    # gate keeps doing its job.
    gated = apply_terminal_config_to_env(
        env={"TERMINAL_ENV": "kubernetes"}, config=merged_like, override=True,
    )
    assert "TERMINAL_KUBERNETES" not in gated


def test_doctor_probes_the_exec_verb_the_client_actually_issues():
    """connect_get_namespaced_pod_exec is a websocket-upgrading GET, authorized
    as verb `get`. Probing `create` failed a truly minimal Role (pushing
    operators to widen it) and passed the kubectl-shaped create-only Role that
    then 403s on the first command.

    Asserted behaviourally, by recording the SelfSubjectAccessReviews doctor
    actually issues — not by reading doctor's source."""
    reviews = _doctor_rbac_reviews(_kcfg(namespace="hermes"))
    # BOTH are required. Verified against Kubernetes 1.36: a Role granting
    # only `get` on pods/exec is refused with 403 and every command fails,
    # even though `kubectl auth can-i get pods/exec` says yes. Probing only
    # `get` green-lit exactly that broken Role.
    assert ("", "pods/exec", "get") in reviews
    assert ("", "pods/exec", "create") in reviews
    assert ("", "pods", "create") in reviews
    # `get pods` backs readiness polling, 409 ownership and the ownerRef
    # lookup — omitting it green-lit a Role that 403s at the first session.
    assert ("", "pods", "get") in reviews
    docs = (Path(__file__).resolve().parents[2] / "website" / "docs"
            / "user-guide" / "kubernetes.md").read_text(encoding="utf-8")
    # Both verbs must stay documented: granting only `get` produces a healthy
    # startup on which every command then 403s.
    assert 'verbs: ["get", "create"]' in docs
    assert "pods/exec" in docs


def _doctor_rbac_reviews(kcfg):
    """Drive _check_kubernetes_backend and record every SSAR it submits."""
    import importlib.util

    import kubernetes.client as kclient
    from hermes_cli import doctor

    seen: list = []

    class _Attrs:
        def __init__(self, namespace=None, group=None, resource=None, verb=None):
            self.group, self.resource, self.verb = group, resource, verb

    class _Spec:
        def __init__(self, resource_attributes=None):
            self.resource_attributes = resource_attributes

    class _Review:
        def __init__(self, spec=None):
            self.spec = spec

    class _AuthApi:
        def __init__(self, api_client=None):
            pass

        def create_self_subject_access_review(self, review, **kwargs):
            attrs = review.spec.resource_attributes
            seen.append((attrs.group, attrs.resource, attrs.verb))
            return SimpleNamespace(status=SimpleNamespace(allowed=True))

    core = MagicMock()
    original = {
        "V1ResourceAttributes": getattr(kclient, "V1ResourceAttributes", None),
        "V1SelfSubjectAccessReviewSpec": getattr(
            kclient, "V1SelfSubjectAccessReviewSpec", None),
        "V1SelfSubjectAccessReview": getattr(
            kclient, "V1SelfSubjectAccessReview", None),
        "AuthorizationV1Api": getattr(kclient, "AuthorizationV1Api", None),
    }
    kclient.V1ResourceAttributes = _Attrs
    kclient.V1SelfSubjectAccessReviewSpec = _Spec
    kclient.V1SelfSubjectAccessReview = _Review
    kclient.AuthorizationV1Api = _AuthApi

    import tools.environments.kubernetes as k8s_mod
    import tools.terminal_tool as tt

    saved = (importlib.util.find_spec, k8s_mod.load_core_api,
             tt._get_env_config, doctor._dry_run_pod_template)
    try:
        importlib.util.find_spec = lambda name, *a, **k: (
            object() if name == "kubernetes" else saved[0](name, *a, **k)
        )
        k8s_mod.load_core_api = lambda cfg: core
        tt._get_env_config = lambda: {"kubernetes": kcfg}
        doctor._dry_run_pod_template = lambda *a, **kw: None
        doctor._check_kubernetes_backend([])
    finally:
        (importlib.util.find_spec, k8s_mod.load_core_api,
         tt._get_env_config, doctor._dry_run_pod_template) = saved
        for name, value in original.items():
            if value is None:
                delattr(kclient, name)
            else:
                setattr(kclient, name, value)
    return seen


def test_doctor_dry_runs_the_rendered_pod_with_strict_field_validation():
    """Without Strict, an unknown field in pod_template is accepted with 201 and
    silently dropped — the python client throws away the API server's
    "Warning: 299 - unknown field" header, so Warn is indistinguishable from
    success. doctor submits the real rendered pod as a dry-run create so the
    server names the offending path here, not at the first session."""
    from hermes_cli import doctor

    core = MagicMock()
    doctor._dry_run_pod_template(_kcfg(), "hermes", core, [])

    kwargs = core.create_namespaced_pod.call_args.kwargs
    assert kwargs["dry_run"] == "All"
    assert kwargs["field_validation"] == "Strict"
    assert kwargs["body"]["kind"] == "Pod"
    assert kwargs["body"]["spec"]["containers"][0]["command"] == ["sleep", "infinity"]


def test_doctor_reports_the_api_servers_rejection_as_a_failure():
    """The API server IS the validator, so its rejection is what doctor
    reports — with the server's own message. 400 = malformed field, 422 = a
    known field with an invalid value, 403 = ADMISSION refused the pod (SCC /
    PSA / quota / the VAP this repo ships). The SSAR probes already proved
    `create pods` is allowed, so a 403 here is a verdict about the pod, not a
    permission problem — reporting it as "skipped" hid the failures doctor
    exists to surface. Only a transient error (429, 5xx, network) is a warning."""
    from hermes_cli import doctor
    from kubernetes.client.exceptions import ApiException

    for status, reason in ((400, "Bad Request"), (422, "Unprocessable Entity"),
                           (403, "Forbidden")):
        core = MagicMock()
        core.create_namespaced_pod.side_effect = ApiException(status=status,
                                                              reason=reason)
        issues: list[str] = []
        doctor._dry_run_pod_template(_kcfg(), "hermes", core, issues)
        assert issues, f"{status} must be reported as a failure"

    # Transient: says nothing about the template.
    core = MagicMock()
    core.create_namespaced_pod.side_effect = ApiException(status=503,
                                                          reason="Unavailable")
    issues = []
    doctor._dry_run_pod_template(_kcfg(), "hermes", core, issues)
    assert issues == []


# ---------------------------------------------------------------------------
# Refusing to adopt is refusing to delete
# ---------------------------------------------------------------------------


def test_refusing_to_adopt_a_foreign_pod_does_not_delete_it():
    """The read side was guarded and the write side was not: ensure() raised
    "not created by this Hermes instance", KubernetesEnvironment.__init__ caught
    it and ran its cleanup path, which rebuilds the SAME conventional name and
    deletes it. The refusal has to bind both directions."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    api.read_namespaced_pod.return_value = _running_pod(
        labels={MANAGED_BY: "hermes-agent"},
        owners=[SimpleNamespace(uid="SOMEONE-ELSE")],
    )
    provisioner = _provisioner_with_api(api)

    # Driven through the real environment constructor, which is where the
    # cleanup path lives.
    with pytest.raises(RuntimeError, match="not created by this Hermes instance"):
        KubernetesEnvironment(
            provisioner=provisioner, task_id="abc", api=api, sync_files=False,
        )
    api.delete_namespaced_pod.assert_not_called()


def test_an_unreadable_pod_is_not_deleted_either():
    """`except ApiException: return False` turned a 403 on `get pods` (or any
    transient API error) into "not ours" — and the cleanup path then deleted
    another agent's live workspace. Unreadable is its own answer, and it fails
    closed in BOTH directions."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = ApiException(status=409)
    api.read_namespaced_pod.side_effect = ApiException(status=403, reason="Forbidden")
    p = _provisioner_with_api(api)

    with pytest.raises(RuntimeError, match="ownership could not be read"):
        p.ensure("abc")
    p.destroy(PodRef("hermes", p.workspace_name("abc"), "workspace"))
    api.delete_namespaced_pod.assert_not_called()


# ---------------------------------------------------------------------------
# Recovery from a pod that is dead but still PRESENT (B2): deadline,
# eviction and OOMKill under restartPolicy: Never leave the pod in phase
# Failed — exec returns 400, not 404, and 404-only recovery wedged the
# session permanently.
# ---------------------------------------------------------------------------


def _pod_with(phase="Running", deletion=None, terminated_container=None):
    statuses = []
    if terminated_container:
        statuses.append(SimpleNamespace(
            name=terminated_container,
            state=SimpleNamespace(terminated=SimpleNamespace(reason="OOMKilled"),
                                  waiting=None),
        ))
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="p", uid="pod-uid",
            # Ours, so _is_ours passes and the terminal-phase branch is what
            # decides — the point of these tests.
            labels={MANAGED_BY: "hermes-agent"},
            owner_references=[SimpleNamespace(uid=OWNER_REF["uid"])],
            deletion_timestamp=deletion,
        ),
        spec=SimpleNamespace(containers=[SimpleNamespace(name="workspace")]),
        status=SimpleNamespace(phase=phase, conditions=[],
                               container_statuses=statuses),
    )


def test_exec_400_against_a_failed_pod_clears_the_ref(monkeypatch):
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.read_namespaced_pod.return_value = _pod_with(phase="Failed")
    completed = _FakeWSClient(
        raise_on_update=ApiException(status=400, reason="Bad Request")
    )
    env = _make_k8s_env(monkeypatch, [("", 0), completed], api=api)
    env.execute("ls")
    assert env._pod_ref is None, (
        "a 400 against a phase-Failed pod must clear the ref so the next "
        "command re-provisions"
    )


def test_exec_400_against_a_live_pod_keeps_the_ref(monkeypatch):
    """A transient 400 with the pod Running is NOT death — clearing the ref
    there would re-provision (and wipe) a healthy workspace."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.read_namespaced_pod.return_value = _running_pod()
    hiccup = _FakeWSClient(
        raise_on_update=ApiException(status=400, reason="Bad Request")
    )
    env = _make_k8s_env(monkeypatch, [("", 0), hiccup], api=api)
    env.execute("ls")
    assert env._pod_ref is not None


def test_ensure_deletes_and_recreates_its_own_terminal_pod():
    """409 on create + the existing pod is ours but phase Failed: handing the
    corpse to wait_pod_ready raised immediately, with no recreate branch."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    # First create 409s; after the delete, the second create succeeds.
    api.create_namespaced_pod.side_effect = [ApiException(status=409), None]
    reads = {"n": 0}

    def _read(**kwargs):
        reads["n"] += 1
        # _is_ours + _pod_is_terminal see the corpse; after deletion the
        # name is free (404) and the recreated pod comes up Running.
        if reads["n"] <= 2:
            return _pod_with(phase="Failed")
        if reads["n"] == 3:
            raise ApiException(status=404)
        return _running_pod()

    api.read_namespaced_pod.side_effect = _read
    p = _provisioner_with_api(api)
    ref = p.ensure("abc")
    api.delete_namespaced_pod.assert_called_once()
    assert api.create_namespaced_pod.call_count == 2
    assert ref.pod_name == p.workspace_name("abc")


def test_execute_surfaces_the_workspace_reset_to_the_model(monkeypatch):
    """A logger.warning is invisible to the model; the first MODEL-FACING
    result after a re-provision must say the workspace is empty (and where
    cwd went)."""
    from kubernetes.client.exceptions import ApiException

    gone = _FakeWSClient(raise_on_update=ApiException(status=404, reason="gone"))
    env = _make_k8s_env(monkeypatch, [("", 0), gone, ("", 0), ("ok\n", 0)])
    env.execute("ls", bounded_capture=True)              # kills the ref
    result = env.execute("echo ok", bounded_capture=True)  # re-provisions
    assert "re-provisioned" in result["output"]
    assert "/workspace" in result["output"]
    # Once, not forever.
    follow_up = env.execute("echo again", bounded_capture=True)
    assert "re-provisioned" not in follow_up["output"]


def test_the_first_command_after_a_reprovision_ignores_the_stale_cwd(monkeypatch):
    """`_ensure_pod` reset `self.cwd`, which settled nothing.

    base.execute() computes `effective_cwd = cwd or self.cwd`, and
    terminal_tool ALWAYS passes an explicit cwd read from the per-session
    record — a directory inside the pod that just died. So the first recovered
    command ran `builtin cd -- <gone> || exit 126` in the fresh pod, died
    without running, and was handed a note claiming the cwd had been reset.
    One wasted turn, and the model was told something untrue about a command
    that never executed."""
    from kubernetes.client.exceptions import ApiException

    gone = _FakeWSClient(raise_on_update=ApiException(status=404, reason="gone"))
    env = _make_k8s_env(monkeypatch, [("", 0), gone, ("", 0), ("ok\n", 0)])

    seen = []
    real_execute = type(env).__mro__[1].execute
    monkeypatch.setattr(
        type(env).__mro__[1], "execute",
        lambda self, command, cwd="", **kw: (
            seen.append(cwd), real_execute(self, command, cwd, **kw))[1],
    )

    env.execute("ls", cwd="/workspace/proj", bounded_capture=True)   # kills ref
    env.execute("echo ok", cwd="/workspace/proj", bounded_capture=True)
    # The caller asked for the dead pod's cwd both times; the recovered
    # command must have been redirected to the workspace root instead.
    assert seen[0] == "/workspace/proj"
    assert seen[-1] == "/workspace", seen
    # ...and only for that one command: the override does not stick.
    env.execute("echo again", cwd="/workspace/proj", bounded_capture=True)
    assert seen[-1] == "/workspace/proj", seen


def test_the_reset_note_never_reaches_internal_readers(monkeypatch):
    """`execute()` is also the full-fidelity path for `cat` reads feeding the
    patch engine, `stat` output parsed with int(), `command -v` probes and
    the code-exec JSON-RPC loop — all of which leave bounded_capture False.
    Prefixing prose there corrupts data instead of informing anyone."""
    from kubernetes.client.exceptions import ApiException

    gone = _FakeWSClient(raise_on_update=ApiException(status=404, reason="gone"))
    env = _make_k8s_env(monkeypatch, [("", 0), gone, ("", 0), ("4096\n", 0)])
    env.execute("ls", bounded_capture=True)
    internal = env.execute("stat -c %s /workspace/f")  # bounded_capture unset
    assert internal["output"] == "4096\n"
    assert int(internal["output"].strip()) == 4096
    # The note is still owed to the model, and arrives on its next turn.
    assert "re-provisioned" in env.execute("echo hi", bounded_capture=True)["output"]


def test_cleanup_is_final_for_background_pollers(monkeypatch):
    """An orphaned poller thread calling into a cleaned-up environment must
    not resurrect it into an untracked pod."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    env.cleanup()
    with pytest.raises(RuntimeError, match="cleaned up"):
        env._ensure_pod()


# ---------------------------------------------------------------------------
# Same-pod multi-process isolation (D9) and the removed image override (D11)
# ---------------------------------------------------------------------------


def test_instance_discriminator_is_process_scoped(monkeypatch):
    """Two Hermes processes in ONE agent pod share the pod UID; seeded from it
    alone they adopted and deleted each other's workspace."""
    import tools.environments.kubernetes as k8s_mod

    monkeypatch.setattr(k8s_mod.os, "getpid", lambda: 1111)
    a = k8s_mod._instance_discriminator("pod-uid")
    monkeypatch.setattr(k8s_mod.os, "getpid", lambda: 2222)
    b = k8s_mod._instance_discriminator("pod-uid")
    assert a != b


def test_registering_a_kubernetes_image_override_fails_loudly():
    """Silently dropping the per-task image made an RL sweep evaluate every
    rollout against the wrong image with no warning."""
    import tools.terminal_tool as tt

    with pytest.raises(ValueError, match="terminal.kubernetes.spec"):
        tt.register_task_env_overrides("rollout-1", {"kubernetes_image": "x:1"})
    assert "rollout-1" not in tt._task_env_overrides


def test_cd_guard_126_hint_names_the_vanished_cwd():
    """The cwd guard's own shell diagnostic (`cd: ...: No such file or
    directory`, merged into output) is the discriminator; the generic
    'chmod +x' hint sent the model the wrong way after a workspace reset."""
    from tools.terminal_hints import annotate_failure

    guard = annotate_failure(
        "make build", 126,
        "bash: line 1: cd: /workspace/proj: No such file or directory",
    )
    assert guard and "working directory" in guard
    # A real not-executable 126 keeps the chmod hint — even with its output
    # redirected away (the common `>/dev/null 2>&1` shape).
    assert "chmod" in (annotate_failure("./script.sh", 126, "permission denied") or "")
    assert "chmod" in (annotate_failure("./script.sh >/dev/null 2>&1", 126, "") or "")


# ---------------------------------------------------------------------------
# Deadness detection covers every shape that leaves the pod PRESENT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pod, dead", [
    (_pod_with(), False),
    (_pod_with(phase="Failed"), True),
    (_pod_with(phase="Succeeded"), True),
    # Phase only turns terminal once EVERY container exits, so a sidecar keeps
    # a pod Running while the container we exec into is gone.
    (_pod_with(terminated_container="workspace"), True),
    # A sidecar dying is not our problem.
    (_pod_with(terminated_container="istio-proxy"), False),
    # Stuck Terminating (node loss / finalizer): phase stays Running forever.
    (_pod_with(deletion="2026-08-07T00:00:00Z"), True),
])
def test_pod_cannot_exec_covers_present_but_dead_shapes(pod, dead):
    from tools.environments.kubernetes import pod_cannot_exec

    assert pod_cannot_exec(pod, "workspace") is dead


def test_exec_error_on_a_dead_exec_container_clears_the_ref(monkeypatch):
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.read_namespaced_pod.return_value = _pod_with(
        terminated_container="workspace")
    broken = _FakeWSClient(
        raise_on_update=ApiException(status=400, reason="Bad Request"))
    env = _make_k8s_env(monkeypatch, [("", 0), broken], api=api)
    env.execute("ls", bounded_capture=True)
    assert env._pod_ref is None


def test_terminal_pod_delete_carries_a_uid_precondition():
    """Session pod names are deterministic, so a delete racing a concurrent
    re-provision would otherwise remove the REPLACEMENT."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.create_namespaced_pod.side_effect = [ApiException(status=409), None]
    reads = {"n": 0}

    def _read(**kwargs):
        reads["n"] += 1
        if reads["n"] <= 2:
            return _pod_with(phase="Failed")
        if reads["n"] == 3:
            raise ApiException(status=404)
        return _running_pod()

    api.read_namespaced_pod.side_effect = _read
    p = _provisioner_with_api(api)
    p.ensure("abc")
    body = api.delete_namespaced_pod.call_args.kwargs.get("body")
    assert body is not None, "delete must carry preconditions"
    assert getattr(body.preconditions, "uid", None) == "pod-uid"


def test_a_replaced_pod_is_not_deleted_out_from_under_the_replacement(caplog):
    """409 on delete = the name now belongs to a different object. That is the
    precondition doing its job, not an error."""
    from kubernetes.client.exceptions import ApiException

    api = MagicMock()
    api.delete_namespaced_pod.side_effect = ApiException(status=409)
    p = _provisioner_with_api(api)
    with caplog.at_level("WARNING"):
        p._delete_pod("hermes", "hermes-ws-abc", uid="stale-uid")
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_concurrent_re_provision_creates_one_pod(monkeypatch):
    """Up to 8 tool workers share one environment; with the ref check and the
    (now destructive) create unlocked, two workers each deleted the dead pod
    and then the other's fresh replacement."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    with env._lock:
        env._pod_ref = None

    calls = {"n": 0}

    def _slow_ensure(task_id):
        calls["n"] += 1
        time.sleep(0.2)
        return PodRef("hermes", "hermes-ws-abc", "workspace")

    env._provisioner.ensure.side_effect = _slow_ensure
    threads = [threading.Thread(target=env._ensure_pod) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert calls["n"] == 1, "each worker provisioned its own session pod"


def test_cleanup_during_provisioning_destroys_the_new_pod(monkeypatch):
    """cleanup() returns early when _pod_ref is None — exactly the state a
    re-provision is in — so the in-flight object must be destroyed by the
    provisioner itself or nothing ever will."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    with env._lock:
        env._pod_ref = None

    def _ensure_then_teardown(task_id):
        env.cleanup()  # races in while we "provision"
        return PodRef("hermes", "hermes-ws-abc", "workspace")

    env._provisioner.ensure.side_effect = _ensure_then_teardown
    env._provisioner.destroy.reset_mock()
    with pytest.raises(RuntimeError, match="cleaned up while provisioning"):
        env._ensure_pod()
    env._provisioner.destroy.assert_called_once()


# ---------------------------------------------------------------------------
# Client resilience: timeouts, transient retries, 401 refresh
# ---------------------------------------------------------------------------


def test_api_calls_carry_a_request_timeout():
    """Unset, urllib3 builds no Timeout at all, so a blackholed apiserver
    pins the calling thread forever and ready_timeout_seconds — checked only
    BETWEEN polls — can never expire."""
    from tools.environments.kubernetes import API_TIMEOUT, api_call

    seen = {}
    api_call(lambda **kw: seen.update(kw), namespace="hermes")
    assert seen["_request_timeout"] == API_TIMEOUT


@pytest.mark.parametrize("status, attempts", [
    (503, 2),   # control-plane rollout: retried
    (429, 2),   # priority-and-fairness shedding: retried
    (403, 1),   # RBAC: our request is wrong, retrying repeats the mistake
    (404, 1),
])
def test_transient_statuses_retry_and_terminal_ones_do_not(monkeypatch, status,
                                                           attempts):
    from kubernetes.client.exceptions import ApiException
    from tools.environments.kubernetes import api_call

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def _flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ApiException(status=status)
        return "ok"

    if attempts == 1:
        with pytest.raises(ApiException):
            api_call(_flaky)
    else:
        assert api_call(_flaky) == "ok"
    assert calls["n"] == attempts


def test_a_401_reloads_credentials_once_then_surfaces(monkeypatch):
    """A long-lived agent outlives its projected token; nothing retried the
    call that raced the rotation."""
    from kubernetes.client.exceptions import ApiException
    import tools.environments.kubernetes as k8s_mod

    reloads = {"n": 0}
    monkeypatch.setattr(k8s_mod, "_reload_kubernetes_auth",
                        lambda: reloads.__setitem__("n", reloads["n"] + 1))
    calls = {"n": 0}

    def _expired(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ApiException(status=401)
        return "ok"

    assert k8s_mod.api_call(_expired) == "ok"
    assert reloads["n"] == 1

    # A persistent 401 is surfaced, not retried forever.
    def _always(**kwargs):
        raise ApiException(status=401)

    with pytest.raises(ApiException):
        k8s_mod.api_call(_always)


def test_stdin_upload_bounds_the_write_phase(monkeypatch, tmp_path):
    """The deadline used to be established AFTER the write loop, so the phase
    most likely to block (a remote tar that stopped draining) was unbounded."""
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    env = _make_k8s_env(monkeypatch, [("", 0)])

    class _WedgedStdin(_FakeWSClient):
        def write_stdin(self, data):
            time.sleep(0.2)   # never drains

    monkeypatch.setattr("kubernetes.stream.stream",
                        lambda *a, **kw: _WedgedStdin(open_cycles=10_000))
    with pytest.raises(TimeoutError):
        env._stdin_upload("x" * (64 * 1024 * 6), timeout=1)


# ---------------------------------------------------------------------------
# Session pods are labelled for MANUAL reclaim — Hermes never sweeps
# ---------------------------------------------------------------------------


def test_session_pods_carry_the_instance_label_for_manual_reclaim():
    """A process cannot tell a dead predecessor from a LIVE SIBLING by label
    alone, so Hermes deletes nothing it did not create. The label exists so an
    operator can find leftovers: kubectl get pods -l <label>=<instance>."""
    p = _provisioner_with_api(MagicMock())
    labels = p.pod_manifest("abc")["metadata"]["labels"]
    assert labels[MANAGED_BY] == "hermes-agent"
    assert labels[INSTANCE_LABEL] == p._instance


def test_the_backend_never_lists_pods():
    """`list pods` would let a compromised agent enumerate every pod in the
    namespace; nothing in the backend needs it, and the shipped Role does not
    grant it. This pins that no code path reintroduces one."""
    import inspect
    import tools.environments.kubernetes as k8s_mod

    assert "list_namespaced_pod" not in inspect.getsource(k8s_mod)
    assert not hasattr(_provisioner_with_api(MagicMock()), "reap_orphans")


def test_a_dead_pod_reports_something_the_model_can_act_on(monkeypatch):
    """Exec against a completed pod fails deep inside the websocket client
    ("'NoneType' object has no attribute 'decode'" — observed on a real
    DeadlineExceeded pod). Handing that to the model tells it nothing; the
    failure has to name the cause and the remedy."""
    api = MagicMock()
    api.read_namespaced_pod.return_value = _pod_with(phase="Failed")
    broken = _FakeWSClient(
        raise_on_update=AttributeError("'NoneType' object has no attribute 'decode'"))
    env = _make_k8s_env(monkeypatch, [("", 0), broken], api=api)
    out = env.execute("ls", bounded_capture=True)["output"]
    assert "workspace stopped" in out and "next command" in out
    assert "NoneType" not in out


def test_an_unexplained_exec_error_still_shows_the_error(monkeypatch):
    """Only a CONFIRMED dead pod gets the friendly message; anything else
    must keep the diagnostic, or a real bug becomes invisible."""
    api = MagicMock()
    api.read_namespaced_pod.return_value = _running_pod()
    env = _make_k8s_env(
        monkeypatch, [("", 0), _FakeWSClient(raise_on_update=RuntimeError("boom"))],
        api=api)
    assert "boom" in env.execute("ls", bounded_capture=True)["output"]


def test_cleanup_does_nothing_once_the_ref_has_been_dropped(monkeypatch):
    """The pod is gone the moment the ref is (either destroy() ran, or
    _forget_pod_if_dead nulled it because the pod died on its own). Calling
    destroy on a synthesised ref would delete whatever now holds the name."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    with env._lock:
        env._pod_ref = None
    env.cleanup()
    env._provisioner.destroy.assert_not_called()


def test_a_replacement_pod_gets_the_files_again(monkeypatch):
    """sync(force=True) only bypasses the rate limit — the per-file mtime
    cache still short-circuits every upload, so without dropping that state a
    re-provisioned (empty) pod silently received NOTHING."""
    from kubernetes.client.exceptions import ApiException

    gone = _FakeWSClient(raise_on_update=ApiException(status=404, reason="gone"))
    env = _make_k8s_env(monkeypatch, [("", 0), gone, ("", 0), ("ok\n", 0)])
    env._sync_manager = MagicMock()
    env.execute("ls", bounded_capture=True)      # kills the ref
    env.execute("echo ok", bounded_capture=True)  # re-provisions
    env._sync_manager.forget_remote_state.assert_called_once()
    env._sync_manager.sync.assert_any_call(force=True)


def test_exec_connect_is_bounded(monkeypatch):
    """kubernetes.stream.stream() ignores _request_timeout, and every other
    deadline in the module is computed AFTER the connect returns — so a peer
    that accepts and then goes silent used to pin the caller's thread."""
    import tools.environments.kubernetes as k8s_mod

    monkeypatch.setattr(k8s_mod, "API_TIMEOUT", (0.2, 0.3))
    env = _make_k8s_env(monkeypatch, [("", 0)])
    monkeypatch.setattr("kubernetes.stream.stream",
                        lambda *a, **kw: time.sleep(30))
    with pytest.raises(TimeoutError, match="no websocket"):
        env._open_stream(["sh", "-c", "true"])


def test_cancel_kills_the_remote_process_tree(monkeypatch):
    """Closing the websocket frees OUR side only — verified against Kubernetes
    1.36, a loop kept writing for at least 8s after the stream was closed. So
    cancel() must also kill the remote tree, or a timed-out command runs on to
    completion while the model is told it stopped and re-runs it."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    execs: list[list[str]] = []
    monkeypatch.setattr(env, "_exec_capture",
                        lambda cmd, timeout=60: (execs.append(cmd) or ("", 0)))
    client = _FakeWSClient(open_cycles=10_000)
    monkeypatch.setattr("kubernetes.stream.stream", lambda *a, **kw: client)

    handle = env._run_bash("sleep 3600")
    handle.kill()
    handle.wait(timeout=5)

    assert client.closed is True, "the stream must still be closed"
    kill_cmds = [c for c in execs if any("kill" in part for part in c)]
    assert kill_cmds, "cancel() issued no kill — the remote process survives"
    assert any("HERMES_EXEC_" in part for c in kill_cmds for part in c), (
        "the kill must target THIS exec's marker, not every process in the pod"
    )


def test_each_exec_is_tagged_uniquely(monkeypatch):
    """Two concurrent commands must not cancel each other."""
    env = _make_k8s_env(monkeypatch, [("", 0)])
    seen: list[str] = []

    def _capture(*args, **kwargs):
        seen.append(kwargs.get("command") or args[3])
        return _FakeWSClient(open_cycles=2)

    monkeypatch.setattr("kubernetes.stream.stream", _capture)
    env._run_bash("echo a").wait(timeout=5)
    env._run_bash("echo b").wait(timeout=5)
    markers = [p for cmd in seen for p in cmd if "HERMES_EXEC_" in str(p)]
    assert len(markers) >= 2 and markers[0] != markers[1]
