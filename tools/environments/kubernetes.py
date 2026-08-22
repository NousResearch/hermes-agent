"""Kubernetes session-pod execution environment: agent commands exec into a
stateless pod, one per Hermes process. See website/docs/user-guide/kubernetes.md."""

import base64
import hashlib
import io
import logging
import os
import posixpath
import re
import shlex
import socket
import tarfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)

# Namespace the kubelet projects into every pod.
_SA_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
_SA_TOKEN_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/token"

WORKSPACE_CONTAINER_NAME = "workspace"
# Grace beyond the caller's timeout; _wait_for_process is expected to act first.
_EXEC_GRACE_SECONDS = 15
# Terminator for the stdin file-sync payload (see _stdin_upload).
_SYNC_SENTINEL = "__HERMES_TAR_EOF__"
_STDIN_CHUNK_BYTES = 64 * 1024
MANAGED_BY_LABEL = {"app.kubernetes.io/managed-by": "hermes-agent"}
#: Marks which process created a pod; also the ownership tiebreaker in
#: _is_ours() when no ownerReference identity exists. Leftovers go to GC.
INSTANCE_LABEL = "hermes.nousresearch.com/instance"


#: The default session object: used when spec is unset and written by `hermes
#: setup`. A user spec replaces it wholesale. Pinned to the docs page by test.
STARTER_SESSION_OBJECT: dict[str, Any] = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {"labels": dict(MANAGED_BY_LABEL)},
    "spec": {
        "containers": [{
            "name": "workspace",
            "image": "ubuntu:26.04",
            "command": ["sleep", "infinity"],
            "workingDir": "/workspace",
            "volumeMounts": [
                {"name": "workspace", "mountPath": "/workspace"},
                {"name": "tmp", "mountPath": "/tmp"},
            ],
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "capabilities": {"drop": ["ALL"]},
            },
        }],
        "volumes": [
            {"name": "workspace", "emptyDir": {}},
            {"name": "tmp", "emptyDir": {}},
        ],
        "shareProcessNamespace": True,
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 1,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "hostNetwork": False,
        "hostPID": False,
        "hostIPC": False,
        "activeDeadlineSeconds": 14400,
        # No runAsNonRoot / runAsUser: the default must admit and start on any
        # cluster. An explicit UID violates OpenShift's restricted-v2 range,
        # and runAsNonRoot without a UID fails at start on any cluster where
        # no admission controller assigns one (kubeadm, k3s, kind, EKS, GKE,
        # AKS defaults). Where nothing assigns a UID the session runs as the
        # image's user inside the pod boundary — the docker backend's exact
        # posture. Operators who want non-root set it in `spec`.
        "securityContext": {
            "seccompProfile": {"type": "RuntimeDefault"},
        },
    },
}


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------
#
# Source of truth for terminal.kubernetes.* defaults, mirrored literally in
# hermes_cli/config_defaults.py (pinned by test_kubernetes_config_schema).
DEFAULT_KUBERNETES_CONFIG: dict[str, Any] = {
    # --- connection ----------------------------------------------------
    "namespace": "",                  # "" -> kubeconfig context, else SA file
    "kubeconfig": "",                 # out-of-cluster dev only (a path, not a secret)
    "context": "",                    # kubeconfig context; ignored in-cluster
    # --- the object ----------------------------------------------------
    # apiVersion/kind say which object Hermes creates and knows how to drive;
    # unknown pairs fail in-process (see PROVISIONERS_BY_KIND).
    "apiVersion": "v1",
    "kind": "Pod",
    # Labels/annotations for created objects. `name` and `namespace` are
    # computed per pod (they are how Hermes finds the object again).
    "metadata": {},
    # The whole PodSpec, posted verbatim. Empty falls back to
    # STARTER_SESSION_OBJECT; non-empty replaces the default with no merge.
    "spec": {},
    # --- backend behaviour ----------------------------------------------
    # Which container in `spec` to exec into. A pointer, not a creator.
    "exec_container_name": "workspace",
    # Ownership labels: stamped into metadata.labels when absent, matched in
    # _is_ours before a 409 becomes "resume". {} means MANAGED_BY_LABEL.
    "owned_selector": {},
    # --- lifetime -------------------------------------------------------
    # How long to wait for Ready. Pod lifetime caps belong in the spec
    # (activeDeadlineSeconds), not here.
    "ready_timeout_seconds": 120,
    "owner_reference": "auto",        # auto | off
    # Treat session pods as disposable sandboxes and skip the approval layer,
    # like the peer container backends. False keeps the dangerous-command
    # prompts on (approvals.mode / approvals.deny then govern).
    "trusted_sandbox": True,
}

#: The apiVersion/kind pairs this backend knows how to drive. New object
#: support (an agent-sandbox API, say) is a new entry here, not a config key.
PROVISIONERS_BY_KIND: dict[tuple, str] = {
    ("v1", "Pod"): "pod",
}

# Strict makes unknown/duplicate fields a 400 naming the JSON path; the
# python client drops the "Warning: 299" header, so Warn looks like success.
STRICT_FIELD_VALIDATION: dict[str, str] = {"field_validation": "Strict"}

# (connect, read) ceiling for every API call. The client defaults to no
# timeout at all, and a blackholed apiserver would pin the thread forever.
API_TIMEOUT: tuple[float, float] = (5.0, 30.0)
# Transient statuses worth retrying (taxonomy shared with
# tools/microsoft_graph_client.py). Other 4xx means our request is wrong.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_ATTEMPTS = 4


def _retry_after_seconds(exc: Any, attempt: int) -> float:
    """Honour the apiserver's Retry-After, else exponential backoff."""
    headers = getattr(exc, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("Retry-After")
            if raw is not None:
                return max(0.0, min(30.0, float(raw)))
        except (TypeError, ValueError, AttributeError):
            pass
    return min(8.0, 0.5 * (2 ** attempt))


def _reload_kubernetes_auth() -> None:
    """Re-read the rotated in-cluster token after a 401. Out-of-cluster,
    kubeconfig refresh plugins already handle expiry, so do nothing."""
    if not in_cluster():
        return
    from kubernetes import config as k8s_config

    k8s_config.load_incluster_config()


def api_call(fn, *args, **kwargs):
    """Invoke a client method with a timeout and transient retries. Exec stays
    outside: it owns its own deadline, and a retried exec re-runs a command."""
    import time as _time

    from kubernetes.client.exceptions import ApiException

    kwargs.setdefault("_request_timeout", API_TIMEOUT)
    last: Exception
    refreshed = False
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except ApiException as exc:
            if exc.status == 401 and not refreshed:
                # One reload + one retry for a raced token rotation. 403 is
                # RBAC, never retried.
                refreshed = True
                logger.info("got 401; reloading credentials and retrying")
                try:
                    _reload_kubernetes_auth()
                except Exception as reload_exc:
                    logger.debug("credential reload failed: %s", reload_exc)
                    raise
                continue
            if exc.status not in _RETRY_STATUSES:
                raise
            last = exc
        except TypeError:
            # Mocked/older client without _request_timeout: retry this one
            # call plainly.
            kwargs.pop("_request_timeout", None)
            return fn(*args, **kwargs)
        except (OSError, ConnectionError) as exc:
            # Connection reset / DNS blip / apiserver LB failover.
            last = exc
        if attempt < _RETRY_ATTEMPTS - 1:
            delay = _retry_after_seconds(last, attempt)
            logger.info(
                "retrying %s after transient error (%s); attempt %d/%d "
                "in %.1fs", getattr(fn, "__name__", fn), last, attempt + 2,
                _RETRY_ATTEMPTS, delay,
            )
            _time.sleep(delay)
    raise last


def pod_cannot_exec(pod: Any, exec_container: str) -> bool:
    """True for a pod that exists but can never serve another exec: terminal
    phase, exec container terminated in a Running pod, or Terminating."""
    status = getattr(pod, "status", None)
    if getattr(status, "phase", "") in ("Failed", "Succeeded"):
        return True
    metadata = getattr(pod, "metadata", None)
    if getattr(metadata, "deletion_timestamp", None) is not None:
        return True
    for entry in (getattr(status, "container_statuses", None) or []):
        if getattr(entry, "name", None) != exec_container:
            continue
        state = getattr(entry, "state", None)
        if getattr(state, "terminated", None) is not None:
            return True
    return False


def merge_kubernetes_config(user_config: Any) -> dict:
    """Shallow-merge a partial terminal.kubernetes block over the defaults.
    Shallow on purpose: user dicts replace, they never accumulate."""
    merged = deepcopy(DEFAULT_KUBERNETES_CONFIG)
    if isinstance(user_config, dict):
        merged.update(deepcopy(user_config))
    return merged


# No validate_kubernetes_config() on purpose: the API server owns the schema
# (fieldValidation=Strict) and the cluster admission stack owns the rest.


# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PodRef:
    """Coordinates for exec-ing into a session pod. ``uid`` preconditions
    deletes so a retried DELETE cannot land on a same-name replacement."""

    namespace: str
    pod_name: str
    container: str
    uid: str = ""


def sanitize_name(raw: str, *, max_len: int = 40) -> str:
    """Slugify *raw* into a DNS-1123 label fragment; anything normalised or
    truncated gets a hash suffix so distinct ids cannot collide."""
    slug = re.sub(r"[^a-z0-9-]+", "-", str(raw or "default").lower()).strip("-")
    if not slug:
        slug = "default"
    # Compare against the RAW input so case-only ids ("Default" vs "default")
    # still get distinct hashes.
    if len(slug) > max_len or slug != str(raw or ""):
        digest = hashlib.sha1(str(raw or "default").encode("utf-8")).hexdigest()[:6]
        slug = f"{slug[: max_len - 7].strip('-') or 'task'}-{digest}"
    return slug


def _instance_discriminator(owner_pod_uid: str = "") -> str:
    """Per-process pod-name suffix (agent pod UID or hostname, plus pid), so
    two Hermes processes never target the same session pod."""
    seed = f"{owner_pod_uid or socket.gethostname() or 'hermes'}:{os.getpid()}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Client / namespace resolution
# ---------------------------------------------------------------------------


def _ensure_sdk() -> None:
    """Lazily install the kubernetes client, mirroring the Daytona pattern."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("terminal.kubernetes", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # FeatureUnavailable etc.
        # Convert: terminal_tool turns ImportError into a clean "terminal tool
        # disabled" payload instead of a traceback.
        raise ImportError(str(exc))


def in_cluster() -> bool:
    """True when the projected ServiceAccount token is present."""
    return os.path.exists(_SA_TOKEN_FILE) and os.path.exists(_SA_NAMESPACE_FILE)


def load_core_api(kcfg: dict):
    """Return a CoreV1Api. Precedence: explicit kubeconfig, then the
    in-cluster ServiceAccount, then ambient KUBECONFIG/~/.kube/config."""
    _ensure_sdk()
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    kubeconfig = str(kcfg.get("kubeconfig") or "").strip()
    context = str(kcfg.get("context") or "").strip() or None

    loaded = False
    if not kubeconfig:
        try:
            k8s_config.load_incluster_config()
            loaded = True
        except Exception:
            loaded = False
    if not loaded:
        try:
            k8s_config.load_kube_config(
                config_file=os.path.expanduser(kubeconfig) or None,
                context=context,
            )
        except Exception as exc:
            raise RuntimeError(
                "kubernetes backend: could not authenticate to a cluster. "
                + (f"Tried terminal.kubernetes.kubeconfig={kubeconfig} "
                   if kubeconfig else
                   "Tried the in-cluster ServiceAccount, then ")
                + f"KUBECONFIG/~/.kube/config. Underlying error: {exc}"
            ) from exc

    return k8s_client.CoreV1Api()


def _kubeconfig_context_namespace(kcfg: dict) -> str:
    """The default namespace of the selected (or active) kubeconfig context.

    kubectl semantics: a resolvable context with no ``namespace`` field means
    ``default`` (a fresh kind/minikube kubeconfig sets none). Empty string only
    when there is no reachable kubeconfig or no matching context — never raises.
    """
    try:
        # Plain import, not _ensure_sdk(): this helper must not trigger an
        # install, and every caller that needs the client runs _ensure_sdk()
        # itself. No SDK importable -> fall through to the SA namespace file.
        from kubernetes import config as k8s_config

        kubeconfig = str(kcfg.get("kubeconfig") or "").strip()
        wanted = str(kcfg.get("context") or "").strip()
        contexts, active = k8s_config.list_kube_config_contexts(
            config_file=os.path.expanduser(kubeconfig) or None,
        )
        if wanted:
            chosen = next(
                (c for c in contexts or [] if c.get("name") == wanted), None)
        else:
            chosen = active
        if not chosen:
            return ""
        return str((chosen.get("context") or {}).get("namespace")
                   or "").strip() or "default"
    except Exception:
        return ""


def resolve_namespace(kcfg: dict) -> str:
    """Resolve the session-pod namespace: terminal.kubernetes.namespace,
    else the kubeconfig context's default namespace, else the projected
    ServiceAccount namespace file."""
    namespace = str(kcfg.get("namespace") or "").strip()
    if namespace:
        return namespace
    namespace = _kubeconfig_context_namespace(kcfg)
    if namespace:
        return namespace
    try:
        with open(_SA_NAMESPACE_FILE, encoding="utf-8") as handle:
            namespace = handle.read().strip()
    except OSError:
        namespace = ""
    if not namespace:
        raise ValueError(
            "kubernetes backend: could not resolve a namespace. Set "
            "terminal.kubernetes.namespace in config.yaml, set a namespace "
            "on your kubeconfig context, or run Hermes in-cluster so the "
            "ServiceAccount namespace is projected."
        )
    return namespace


def resolve_owner_reference(core_api, namespace: str, kcfg: dict) -> Optional[dict]:
    """Build the ownerReference that GCs session pods with the agent pod.
    Identity: downward API env vars, else a self-lookup on the hostname."""
    if owner_reference_disabled(kcfg):
        return None

    name = (os.getenv("HERMES_POD_NAME") or "").strip()
    uid = (os.getenv("HERMES_POD_UID") or "").strip()
    if not (name and uid):
        if not in_cluster() or core_api is None:
            return None
        try:
            pod = api_call(
                core_api.read_namespaced_pod,
                name=socket.gethostname(), namespace=namespace,
            )
            name = getattr(pod.metadata, "name", "") or ""
            uid = getattr(pod.metadata, "uid", "") or ""
        except Exception as exc:
            # Without an ownerReference the pod loses its only GC path, so
            # warn loudly.
            logger.warning(
                "could not resolve the agent pod identity (%s: %s); session "
                "pods will carry no ownerReference and will NOT be garbage "
                "collected when this agent dies. Set HERMES_POD_NAME/"
                "HERMES_POD_UID from the downward API, or grant 'get pods'.",
                type(exc).__name__, exc,
            )
            return None
    if not (name and uid):
        logger.warning(
            "agent pod identity incomplete (name=%r uid=%r); session pods "
            "will carry no ownerReference.", name, uid,
        )
        return None
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "name": name,
        "uid": uid,
        # Never block deletion of the agent pod on a session pod.
        "controller": False,
        "blockOwnerDeletion": False,
    }


# ---------------------------------------------------------------------------
# Pod template construction
# ---------------------------------------------------------------------------


def exec_container_name(kcfg: dict) -> str:
    """Name of the container this backend builds and execs into."""
    return str(kcfg.get("exec_container_name") or "").strip() or WORKSPACE_CONTAINER_NAME


#: Session cwd when the exec container declares no workingDir.
FALLBACK_SESSION_CWD = "/workspace"


def effective_spec(kcfg: dict) -> dict:
    """The PodSpec to post: the operator's, or STARTER_SESSION_OBJECT's when
    unset. A non-empty spec replaces the default wholesale; nothing merges."""
    spec = kcfg.get("spec")
    if isinstance(spec, dict) and spec:
        return spec
    return deepcopy(STARTER_SESSION_OBJECT["spec"])


def session_cwd(kcfg: dict) -> str:
    """The session's default cwd: the exec container's workingDir, else
    FALLBACK_SESSION_CWD. TERMINAL_CWD still overrides both."""
    expected = exec_container_name(kcfg)
    for container in effective_spec(kcfg).get("containers") or []:
        if container.get("name") == expected:
            return str(container.get("workingDir") or "").strip() or FALLBACK_SESSION_CWD
    return FALLBACK_SESSION_CWD


def object_kind(kcfg: dict) -> tuple:
    """The normalised (apiVersion, kind), used by dispatch and the manifest
    builder alike so a stray "kind: |" newline cannot split them."""
    api_version = str(kcfg.get("apiVersion") or "").strip() or "v1"
    kind = str(kcfg.get("kind") or "").strip() or "Pod"
    return api_version, kind


def resolve_provisioner_kind(kcfg: dict) -> str:
    """Map the configured apiVersion/kind to an implementation, failing
    in-process with the supported list instead of a downstream 404."""
    api_version, kind = object_kind(kcfg)
    try:
        return PROVISIONERS_BY_KIND[(api_version, kind)]
    except KeyError:
        supported = ", ".join(
            f"{a}/{k}" for a, k in sorted(PROVISIONERS_BY_KIND)
        )
        raise ValueError(
            f"kubernetes backend: unsupported apiVersion/kind "
            f"{api_version}/{kind}. Supported: {supported}. Set "
            "terminal.kubernetes.apiVersion and terminal.kubernetes.kind."
        ) from None


def owner_reference_disabled(kcfg: dict) -> bool:
    """True when owner_reference is off. Boolean False counts too: YAML 1.1
    parses an unquoted `off` as False."""
    value = kcfg.get("owner_reference", "")
    if value is False:
        return True
    return str(value).strip().lower() == "off"


def trusted_sandbox(kcfg: dict) -> bool:
    """True when session pods are treated as disposable sandboxes and skip the
    approval layer. Default True; set false to keep the prompts on."""
    return bool(kcfg.get("trusted_sandbox", True))


def owned_selector(kcfg: dict) -> dict:
    """Labels that mark an object as this backend's: stamped at create,
    matched in _is_ours. One key drives both, so relabelling stays coherent."""
    selector = kcfg.get("owned_selector")
    if not isinstance(selector, dict) or not selector:
        return dict(MANAGED_BY_LABEL)
    return {str(k): str(v) for k, v in selector.items()}


def preflight_spec(kcfg: dict) -> tuple:
    """Return (errors, warnings) for how Hermes will USE the pod, questions no
    admission controller asks. Errors cannot work; warnings name symptoms."""
    errors: list = []
    warnings: list = []
    spec = effective_spec(kcfg)

    wanted = exec_container_name(kcfg)
    containers = spec.get("containers")
    names = [c.get("name") for c in containers or [] if isinstance(c, dict)]
    container = next(
        (c for c in containers or []
         if isinstance(c, dict) and c.get("name") == wanted),
        None,
    )
    if container is None:
        errors.append(
            f"terminal.kubernetes.exec_container_name is {wanted!r} but "
            f"spec.containers declares {names or 'none'}. Every command would "
            f"fail with \"session pod has no container {wanted!r}\", after "
            "the pod is created and pulled, which is a slow way to learn it."
        )
        return errors, warnings

    if not str(container.get("workingDir") or "").strip():
        warnings.append(
            f"spec.containers[{wanted}].workingDir is unset, so the session "
            f"falls back to {FALLBACK_SESSION_CWD}. If your workspace volume "
            "is mounted somewhere else the agent silently works in the image's "
            "WORKDIR and never touches the volume."
        )
    if not spec.get("shareProcessNamespace"):
        warnings.append(
            "spec.shareProcessNamespace is not true. `sleep` as PID 1 never "
            "reaps, so a backgrounded command's wrapper zombifies and Hermes "
            "never detects it finished."
        )
    elif len(containers or []) > 1:
        others = [n for n in names if n != wanted]
        warnings.append(
            "spec.shareProcessNamespace is true AND this pod has more than one "
            f"container ({', '.join(others)}). A shared PID namespace is a "
            "shared /proc: the agent's shell can read those containers' "
            "/proc/<pid>/environ and /proc/<pid>/root (including a Secret "
            "volume mounted only into them) and can signal their processes. "
            "Hermes needs the shared namespace to detect background "
            "completion, so this is a trade you make deliberately: keep "
            "credential-holding sidecars out of the session pod."
        )
    mounts = {
        m.get("mountPath") for m in container.get("volumeMounts") or []
        if isinstance(m, dict)
    }
    read_only_root = (container.get("securityContext") or {}).get(
        "readOnlyRootFilesystem")
    if read_only_root and "/tmp" not in mounts:
        errors.append(
            "readOnlyRootFilesystem is true and nothing is mounted at /tmp. "
            "init_session() writes its environment snapshot there, so cwd and "
            "environment stop persisting between commands — silently."
        )
    if spec.get("automountServiceAccountToken") is not False:
        warnings.append(
            "spec.automountServiceAccountToken is not false, so the "
            "ServiceAccount's token is projected into the session pod and any "
            "command the agent runs can read it. Nothing rejects this: it "
            "passes fieldValidation=Strict, and only you know whether that SA "
            "is harmless."
        )
    # Secrets in the session pod are readable by every session; warn here at
    # config time, where the message can name the field.
    secret_sources: list = []
    for volume in spec.get("volumes") or []:
        if volume.get("secret"):
            secret_sources.append(f"volume {volume.get('name')!r}")
    for entry in container.get("envFrom") or []:
        if entry.get("secretRef"):
            secret_sources.append(
                f"envFrom.secretRef {entry['secretRef'].get('name')!r}")
    for entry in container.get("env") or []:
        if (entry.get("valueFrom") or {}).get("secretKeyRef"):
            secret_sources.append(f"env {entry.get('name')!r}")
    if secret_sources:
        warnings.append(
            "the session pod mounts Secrets (" + ", ".join(secret_sources) +
            "). Everything in the session pod is readable by any command the "
            "agent runs, and by every session this Hermes process serves — "
            "put provider credentials in the AGENT's pod, not this one."
        )

    selector = owned_selector(kcfg)
    if selector.get("app.kubernetes.io/managed-by") != "hermes-agent":
        warnings.append(
            "owned_selector no longer carries "
            "app.kubernetes.io/managed-by=hermes-agent, which is what the "
            "NetworkPolicies on the Kubernetes docs page select on. Session "
            "pods will not be covered until you update your policies' "
            "podSelectors to match — until then their egress is unrestricted."
        )
    if not spec.get("activeDeadlineSeconds") and \
            owner_reference_disabled(kcfg):
        warnings.append(
            "spec.activeDeadlineSeconds is unset and owner_reference is off, "
            "so nothing bounds a session pod whose agent died. That is a "
            "`sleep infinity` pod running until someone notices."
        )
    return errors, warnings


def render_session_object(kcfg: dict, instance: str = "") -> dict:
    """Render metadata + spec as configured. The spec posts verbatim; Hermes
    only adds the ownership and instance labels when absent."""
    spec = effective_spec(kcfg)

    metadata = deepcopy(kcfg.get("metadata"))
    if not isinstance(metadata, dict):
        metadata = {}
    labels = metadata.get("labels")
    if not isinstance(labels, dict):
        labels = {}
    metadata["labels"] = labels
    for key, value in owned_selector(kcfg).items():
        labels.setdefault(key, value)
    if instance:
        labels.setdefault(INSTANCE_LABEL, instance)
    return {"metadata": metadata, "spec": deepcopy(spec)}


# ---------------------------------------------------------------------------
# Provisioners
# ---------------------------------------------------------------------------


class WorkspaceProvisioner(ABC):
    """Creates and destroys the session pod. Injected into
    KubernetesEnvironment, so tests drive it with fakes."""

    #: Namespace the session objects live in.
    namespace: str

    @abstractmethod
    def ensure(self, task_id: str) -> PodRef:
        """Create (or resume) the session workspace; return a Ready PodRef."""
        ...

    @abstractmethod
    def destroy(self, pod_ref: PodRef) -> None:
        """Tear down the session workspace."""
        ...

    @abstractmethod
    def workspace_name(self, task_id: str) -> str:
        """Object name for this task's workspace, derivable from task_id
        alone (teardown recomputes it when ensure() raised)."""
        ...


class PodProvisioner(WorkspaceProvisioner):
    """Creates session pods via the core API. Supplies no pod content, only
    metadata: name, namespace, ownership labels, and the ownerReference."""

    def __init__(self, kcfg: dict, namespace: str, api=None, owner_reference=None):
        self.kcfg = kcfg
        self.namespace = namespace
        self._api = api  # kubernetes.client.CoreV1Api (None in manifest tests)
        self._owner_reference = owner_reference
        self._instance = _instance_discriminator(
            (owner_reference or {}).get("uid", "")
        )
        self.ready_timeout = int(kcfg.get("ready_timeout_seconds") or 120)
        # Objects we refused to adopt. A refusal to reuse must also be a
        # refusal to delete, or teardown destroys a foreign workspace.
        self._foreign_names: set[str] = set()

    def refresh_api(self) -> None:
        """Swap in a freshly-built client.

        At interpreter exit the kubernetes client's own atexit hook may have
        already removed the temp cert files it materialized from a kubeconfig
        with embedded cert data (kind, minikube), which kills every request on
        the cached client with an SSL FileNotFoundError. Reloading the
        kubeconfig re-materializes them.
        """
        self._api = load_core_api(self.kcfg)

    def _refuse(self, name: str, message: str) -> "RuntimeError":
        """Record *name* as not-ours and build the error to raise."""
        self._foreign_names.add(name)
        return RuntimeError(message)

    def _may_delete(self, name: str) -> bool:
        if name in self._foreign_names:
            logger.warning(
                "refusing to delete %s: this Hermes instance declined to "
                "adopt it (it is not ours, or ownership could not be read), so "
                "deleting it would destroy another agent's workspace.", name,
            )
            return False
        return True

    # -- naming ---------------------------------------------------------
    def workspace_name(self, task_id: str) -> str:
        return f"hermes-ws-{self._instance}-{sanitize_name(task_id)}"

    def exec_container_name(self) -> str:
        return exec_container_name(self.kcfg)

    def exec_container(self, pod: Any) -> str:
        """Assert the running pod carries the container we exec into; a
        fallback would hide drift right before a credential upload."""
        expected = self.exec_container_name()
        names: list[str] = []
        for entry in (getattr(getattr(pod, "spec", None), "containers", None) or []):
            name = getattr(entry, "name", None)
            if name:
                names.append(str(name))
        if expected in names:
            return expected
        # An unreadable container list fails the same way as a wrong one:
        # this check exists to prove a match, so it cannot fail open.
        raise RuntimeError(
            f"session pod has no container {expected!r} (found "
            f"{', '.join(names) or 'no readable container list'}). "
            "terminal.kubernetes.exec_container_name names the container to "
            "exec into, and it must match a name in "
            "terminal.kubernetes.spec.containers[]. Either point "
            f"exec_container_name at one of {', '.join(names) or 'them'}, or "
            f"rename a container in your spec to {expected!r}. "
            "(`hermes doctor` catches this before a pod is ever created.)"
        )

    # -- readiness ------------------------------------------------------
    def _pod_failure_detail(self, pod) -> str:
        """Surface *why* a pod isn't Ready instead of a bare timeout."""
        details: list[str] = []
        for status in (getattr(pod.status, "container_statuses", None) or []):
            state = getattr(status, "state", None)
            waiting = getattr(state, "waiting", None) if state else None
            if waiting is not None and getattr(waiting, "reason", None):
                details.append(
                    f"{status.name}: {waiting.reason}"
                    f"{': ' + waiting.message if getattr(waiting, 'message', None) else ''}"
                )
            terminated = getattr(state, "terminated", None) if state else None
            if terminated is not None and getattr(terminated, "reason", None):
                details.append(f"{status.name}: {terminated.reason}")
        for cond in (getattr(pod.status, "conditions", None) or []):
            if cond.status != "True" and getattr(cond, "message", None):
                details.append(f"{cond.type}: {cond.message}")
        return "; ".join(details)

    _FATAL_WAITING_REASONS = frozenset({
        "ImagePullBackOff", "ErrImagePull", "InvalidImageName",
        "CreateContainerConfigError", "CreateContainerError",
    })

    def wait_pod_ready(self, pod_name: str):
        """Block until the pod is Ready, then return it (for container pick)."""
        from kubernetes.client.exceptions import ApiException

        deadline = time.monotonic() + self.ready_timeout
        last_detail = ""
        while time.monotonic() < deadline:
            try:
                pod = api_call(
                    self._api.read_namespaced_pod,
                    name=pod_name, namespace=self.namespace,
                )
            except ApiException as exc:
                if exc.status != 404:
                    raise
                time.sleep(0.5)
                continue
            conditions = getattr(pod.status, "conditions", None) or []
            ready = any(
                c.type == "Ready" and c.status == "True" for c in conditions
            )
            if pod.status.phase == "Running" and ready:
                return pod
            if pod.status.phase in ("Failed", "Succeeded"):
                raise RuntimeError(
                    f"session pod {pod_name} entered phase {pod.status.phase}. "
                    f"{self._pod_failure_detail(pod)}"
                )
            last_detail = self._pod_failure_detail(pod)
            for status in (getattr(pod.status, "container_statuses", None) or []):
                waiting = getattr(getattr(status, "state", None), "waiting", None)
                reason = getattr(waiting, "reason", None) if waiting else None
                if reason in self._FATAL_WAITING_REASONS:
                    # Fail fast: these never resolve on their own, and burning
                    # the full ready timeout hides the real cause.
                    raise RuntimeError(
                        f"session pod {pod_name} cannot start ({reason}). {last_detail}"
                    )
            time.sleep(0.5)
        raise TimeoutError(
            f"session pod {pod_name} not Ready after {self.ready_timeout}s. "
            f"{last_detail or 'No pod conditions reported.'} "
            "Raise terminal.kubernetes.ready_timeout_seconds if this is a slow "
            "image pull or a kata/sandboxed runtime."
        )

    def _delete_pod(self, namespace: str, pod_name: str, uid: str = "") -> None:
        """Delete a session pod. *uid* preconditions the delete so a race
        with re-provisioning cannot remove a same-name replacement."""
        from kubernetes.client.exceptions import ApiException

        kwargs: dict[str, Any] = {"grace_period_seconds": 0}
        if uid:
            from kubernetes import client as k8s_client

            kwargs["body"] = k8s_client.V1DeleteOptions(
                preconditions=k8s_client.V1Preconditions(uid=uid),
                grace_period_seconds=0,
            )
        try:
            api_call(
                self._api.delete_namespaced_pod,
                name=pod_name, namespace=namespace, **kwargs
            )
        except ApiException as exc:
            if exc.status == 409:
                # Precondition failed: the name now belongs to a different
                # object. Not ours to delete — that is the point.
                logger.info(
                    "pod %s was replaced before our delete landed; "
                    "leaving the new one alone.", pod_name,
                )
            elif exc.status != 404:
                logger.warning("failed to delete pod %s: %s", pod_name, exc)

    def _terminal_pod_uid(self, pod_name: str) -> str:
        """Uid of *pod_name* if it exists but can never serve exec again,
        else "". Feeds the precise delete in _delete_pod()."""
        try:
            pod = api_call(
                self._api.read_namespaced_pod,
                name=pod_name, namespace=self.namespace,
            )
        except Exception:
            return ""
        if not pod_cannot_exec(pod, self.exec_container_name()):
            return ""
        return str(getattr(getattr(pod, "metadata", None), "uid", "") or "")

    def _wait_pod_gone(self, pod_name: str, timeout: int = 30) -> bool:
        """Block until a deleted pod's name frees up; False (with a loud log)
        when something like a finalizer or lost node holds it Terminating."""
        from kubernetes.client.exceptions import ApiException

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                pod = api_call(
                    self._api.read_namespaced_pod,
                    name=pod_name, namespace=self.namespace,
                )
            except ApiException as exc:
                if exc.status == 404:
                    return True
                raise
            time.sleep(0.5)
        stamp = getattr(
            getattr(pod, "metadata", None), "deletion_timestamp", None,
        )
        logger.warning(
            "session pod %s is still Terminating after %ss (deletionTimestamp "
            "%s). It is not a slow image pull: something is holding the pod "
            "(a finalizer, a lost node, or a sandbox runtime that will not "
            "tear down). `kubectl describe pod %s` names it.",
            pod_name, timeout, stamp, pod_name,
        )
        return False

    def pod_manifest(self, task_id: str) -> dict:
        """The Pod body. User metadata passes through except name/namespace
        (Hermes'); Hermes' ownerReference is appended, never substituted."""
        api_version, kind = object_kind(self.kcfg)
        obj = render_session_object(self.kcfg, self._instance)
        metadata = dict(obj["metadata"])
        metadata["name"] = self.workspace_name(task_id)
        metadata["namespace"] = self.namespace
        if self._owner_reference is not None:
            # Appended to any operator-declared owners; this reference is also
            # the ownership proof _is_ours checks.
            existing = list(metadata.get("ownerReferences") or [])
            our_uid = self._owner_reference.get("uid")
            # Dicts only: operator-written metadata, never client model objects.
            if not any(o.get("uid") == our_uid for o in existing):
                existing.append(dict(self._owner_reference))
            metadata["ownerReferences"] = existing
        return {
            # Through the same normaliser dispatch used, so what is posted is
            # what was dispatched (see object_kind).
            "apiVersion": api_version,
            "kind": kind,
            "metadata": metadata,
            "spec": obj["spec"],
        }

    def _is_ours(self, pod_name: str) -> bool:
        """True when an existing pod was created by THIS agent instance; a 409
        is only "resume" for our own pod. Unreadable ownership fails closed."""
        from kubernetes.client.exceptions import ApiException

        try:
            pod = api_call(
                self._api.read_namespaced_pod,
                name=pod_name, namespace=self.namespace,
            )
        except ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return False
            raise self._refuse(pod_name, (
                f"session pod {pod_name} already exists but its ownership could "
                f"not be read ({exc}); refusing to reuse or delete it. Grant "
                "'get pods' in this namespace, or use a distinct "
                "terminal.kubernetes.namespace."
            )) from exc
        # Cheap first filter: the same owned_selector labels stamped at
        # create time. The ownerReference UID below is the actual proof.
        expected = owned_selector(self.kcfg)
        labels = getattr(pod.metadata, "labels", None) or {}
        for key, value in expected.items():
            if labels.get(key) != value:
                return False
        if self._owner_reference is None:
            # No agent identity (out-of-cluster dev, owner_reference off):
            # fall back to the instance label. Names alone are not evidence.
            return labels.get(INSTANCE_LABEL) == self._instance
        owners = getattr(pod.metadata, "owner_references", None) or []
        our_uid = self._owner_reference.get("uid")
        # Ownership is proved, never assumed: an unowned pod with our label is
        # what another agent's workspace looks like.
        return any(getattr(o, "uid", None) == our_uid for o in owners)

    def ensure(self, task_id: str) -> PodRef:
        from kubernetes.client.exceptions import ApiException

        pod_name = self.workspace_name(task_id)
        for attempt in (1, 2):
            try:
                api_call(
                    self._api.create_namespaced_pod,
                    namespace=self.namespace,
                    body=self.pod_manifest(task_id),
                    **STRICT_FIELD_VALIDATION,
                )
            except ApiException as exc:
                if exc.status != 409:
                    raise
                # 409: the pod already exists (racing session or leftover).
                if not self._is_ours(pod_name):
                    raise self._refuse(pod_name, (
                        f"session pod {pod_name} already exists and was not "
                        "created by this Hermes instance; refusing to reuse it."
                    ))
                # Our pod, but dead (deadline/eviction/OOM leave it present
                # in phase Failed): delete the corpse and recreate.
                dead_uid = self._terminal_pod_uid(pod_name) if attempt == 1 else ""
                if dead_uid:
                    logger.warning(
                        "session pod %s can no longer serve exec; "
                        "deleting and re-provisioning.", pod_name,
                    )
                    self._delete_pod(self.namespace, pod_name, uid=dead_uid)
                    if not self._wait_pod_gone(pod_name):
                        # Name the real cause instead of letting
                        # wait_pod_ready blame the image pull.
                        raise RuntimeError(
                            f"session pod {pod_name} is stuck Terminating and "
                            "the name cannot be reused. This is not a slow "
                            "image pull — raising "
                            "terminal.kubernetes.ready_timeout_seconds will "
                            f"not help. Run `kubectl describe pod {pod_name} "
                            f"-n {self.namespace}` to see what is holding it "
                            "(a finalizer, a lost node, or a sandbox runtime "
                            "that will not tear down)."
                        )
                    continue
            break

        pod = self.wait_pod_ready(pod_name)
        return PodRef(
            self.namespace, pod_name, self.exec_container(pod),
            uid=str(getattr(getattr(pod, "metadata", None), "uid", "") or ""),
        )

    def destroy(self, pod_ref: PodRef) -> None:
        # A pod we refused to adopt is a pod we must not delete.
        if not self._may_delete(pod_ref.pod_name):
            return
        self._delete_pod(pod_ref.namespace, pod_ref.pod_name, uid=pod_ref.uid)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class _ExecState:
    """Cancellation state for one exec, so kill() addresses exactly the
    command whose handle was killed while others run concurrently."""

    __slots__ = ("stream", "cancelled")

    def __init__(self) -> None:
        self.stream = None
        self.cancelled = False


class KubernetesEnvironment(BaseEnvironment):
    """Exec-into-session-pod backend. Lifecycle delegated to a provisioner."""

    _stdin_mode = "heredoc"  # no real stdin pipe over the exec channel
    _snapshot_timeout = 60  # pod cold-start can be slow

    def __init__(
        self,
        provisioner: WorkspaceProvisioner,
        task_id: str,
        cwd: str = "/workspace",
        timeout: int = 60,
        api=None,
        sync_files: bool = True,
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self._provisioner = provisioner
        self._task_id = task_id
        self._exec_api = api  # configured CoreV1Api; falls back to a fresh one
        self._lock = threading.Lock()
        # Held across a whole re-provision; _lock itself must never be held
        # over an API call.
        self._provision_lock = threading.Lock()
        self._sync_manager = None
        # Captured once: the synced ~/.hermes tree stays where the first sync
        # put it, even after the agent cds away.
        self._hermes_base = posixpath.join(cwd or "/workspace", ".hermes")
        # Where a re-provisioned session restarts; the old pod's cwd is gone.
        self._initial_cwd = cwd or "/workspace"
        # After a re-provision the next execute() result carries a one-line
        # note, because a logger.warning is invisible to the model.
        self._reset_note_pending = False
        # Set on re-provision, cleared by the very next execute().
        self._cwd_invalidated = False
        # cleanup() is final: a background poller must not resurrect the pod.
        self._torn_down = False

        try:
            self._pod_ref = provisioner.ensure(task_id=task_id)
        except BaseException:
            # A pod created but never Ready would linger until its deadline.
            self._pod_ref = None
            self._best_effort_destroy(task_id)
            raise

        if sync_files:
            # No bind mounts, so skills/credentials/caches get pushed in.
            # One-way on purpose: the synced set is host-authored.
            self._sync_manager = FileSyncManager(
                get_files_fn=lambda: iter_sync_files(self._hermes_base),
                upload_fn=self._upload_file,
                delete_fn=self._delete_files,
                bulk_upload_fn=self._bulk_upload,
            )
            try:
                self._sync_manager.sync(force=True)
            except Exception as exc:
                logger.warning("initial file sync failed: %s", exc)

        self.init_session()

    # -- provisioning ---------------------------------------------------
    def _best_effort_destroy(self, task_id: str) -> None:
        try:
            ref = PodRef(
                self._provisioner.namespace,
                self._provisioner.workspace_name(task_id),
                WORKSPACE_CONTAINER_NAME,
            )
            self._provisioner.destroy(ref)
        except Exception:
            pass

    def _ensure_pod(self) -> PodRef:
        """Re-provision after the session pod died (deadline, eviction, OOM);
        without this one dead pod would brick the session."""
        # Serialise the whole re-provision: concurrent workers would otherwise
        # each delete the corpse and then each other's replacement.
        with self._provision_lock:
            with self._lock:
                if self._torn_down:
                    raise RuntimeError(
                        "kubernetes environment has been cleaned up; refusing "
                        "to provision a new session pod for it"
                    )
                if self._pod_ref is not None:
                    return self._pod_ref
            logger.warning(
                "session pod for task %s is gone; provisioning a new one "
                "— the workspace starts empty again.", self._task_id,
            )
            pod_ref = self._provisioner.ensure(task_id=self._task_id)
            with self._lock:
                torn_down = self._torn_down
                if not torn_down:
                    self._pod_ref = pod_ref
            if torn_down:
                # cleanup() ran mid-provision and saw nothing to destroy, so
                # this pod is ours to destroy or nothing ever will.
                logger.warning(
                    "environment was cleaned up mid-provision; "
                    "destroying the session pod it created.",
                )
                try:
                    self._provisioner.destroy(pod_ref)
                except Exception as exc:
                    logger.warning("mid-provision cleanup failed: %s", exc)
                raise RuntimeError(
                    "kubernetes environment was cleaned up while provisioning"
                )
        # The tracked cwd pointed inside the dead pod's workspace.
        self.cwd = self._initial_cwd
        self._reset_note_pending = True
        # Callers pass an explicit cwd that also names the dead workspace;
        # this flag makes the first recovered command use the fresh root.
        self._cwd_invalidated = True
        # The sync cache still describes the dead pod; drop it or the
        # replacement receives nothing (force= only bypasses the rate limit).
        if self._sync_manager is not None:
            try:
                self._sync_manager.forget_remote_state()
                self._sync_manager.sync(force=True)
            except Exception as exc:
                logger.warning("file re-sync failed: %s", exc)
        self._snapshot_ready = False
        # Bare, as __init__ does: init_session() swallows and logs its own
        # failures.
        self.init_session()
        return pod_ref

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        # Provision before base computes effective_cwd, so the cwd override
        # below reaches the recovered command, not the one after it.
        self._ensure_pod()
        if self._cwd_invalidated:
            # The caller-supplied cwd is stale for every consumer after a
            # re-provision, so clear it here, not on the note path.
            cwd = self._initial_cwd
            self._cwd_invalidated = False
        result = super().execute(command, cwd, **kwargs)
        # Model-facing foreground path only: internal readers (cat/stat/
        # command -v/RPC) parse output and a prose prefix corrupts them.
        if self._reset_note_pending and kwargs.get("bounded_capture"):
            # Once, on the first model-visible result: the model is who has
            # to react to a vanished workspace.
            self._reset_note_pending = False
            note = (
                "[note: the session workspace was re-provisioned and starts "
                "empty — files, installed packages and previously saved tool "
                f"outputs are gone; cwd reset to {self._initial_cwd}]\n"
            )
            result = dict(result)
            result["output"] = note + (result.get("output") or "")
        return result

    def _before_execute(self) -> None:
        self._ensure_pod()
        if self._sync_manager is not None:
            try:
                self._sync_manager.sync()
            except Exception as exc:
                logger.debug("file sync skipped: %s", exc)

    # -- raw exec -------------------------------------------------------
    def _exec_client(self):
        """Private ApiClient per exec: kubernetes.stream monkeypatches
        api_client.request non-reentrantly, so overlapping execs cannot share."""
        from kubernetes.client import ApiClient, CoreV1Api

        configuration = getattr(
            getattr(self._exec_api, "api_client", None), "configuration", None
        )
        # No try/except fallback: ApiClient(None) would re-run the identical
        # failing expression. _run_bash surfaces constructor errors.
        api_client = (
            ApiClient(configuration) if configuration is not None else ApiClient()
        )
        return CoreV1Api(api_client), api_client

    def _open_stream(self, command: list[str], *, stdin: bool = False):
        """Open an exec websocket with the connect in a bounded worker;
        stream() itself has no connect timeout and would pin the thread."""
        from kubernetes.stream import stream as k8s_stream

        ref = self._pod_ref
        if ref is None:
            raise RuntimeError("kubernetes session pod is not provisioned")
        api, api_client = self._exec_client()

        result: dict[str, Any] = {}

        def _connect() -> None:
            try:
                result["stream"] = k8s_stream(
                    api.connect_get_namespaced_pod_exec,
                    ref.pod_name,
                    ref.namespace,
                    container=ref.container,
                    command=command,
                    stderr=True,
                    stdin=stdin,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
            except BaseException as exc:  # re-raised on the caller's thread
                result["error"] = exc

        worker = threading.Thread(
            target=_connect, name="k8s-exec-connect", daemon=True
        )
        worker.start()
        worker.join(API_TIMEOUT[0] + API_TIMEOUT[1])
        try:
            if worker.is_alive():
                raise TimeoutError(
                    "kubernetes exec: the API server accepted no websocket "
                    f"within {API_TIMEOUT[0] + API_TIMEOUT[1]:.0f}s "
                    f"(pod {ref.pod_name})"
                )
            if "error" in result:
                raise self._explain_exec_failure(result["error"])
            return result["stream"]
        finally:
            # The WSClient owns its websocket; the ApiClient only built the
            # request and can go now.
            try:
                api_client.close()
            except Exception:
                pass

    def _explain_exec_failure(self, exc: BaseException) -> BaseException:
        """Turn the client's opaque handshake AttributeError into a message
        naming the usual cause (RBAC, checked via SelfSubjectAccessReview)."""
        if not isinstance(exc, AttributeError) or "decode" not in str(exc):
            return exc
        missing = []
        try:
            from kubernetes import client as k8s_client

            api_client = getattr(self._exec_api, "api_client", None)
            auth = k8s_client.AuthorizationV1Api(api_client)
            for verb in ("get", "create"):
                review = k8s_client.V1SelfSubjectAccessReview(
                    spec=k8s_client.V1SelfSubjectAccessReviewSpec(
                        resource_attributes=k8s_client.V1ResourceAttributes(
                            namespace=self._pod_ref.namespace if self._pod_ref else "",
                            group="", resource="pods/exec", verb=verb,
                        )
                    )
                )
                allowed = getattr(
                    api_call(auth.create_self_subject_access_review, review).status,
                    "allowed", False,
                )
                if not allowed:
                    missing.append(verb)
        except Exception:
            return RuntimeError(
                "kubernetes exec: the API server refused the websocket upgrade "
                f"({exc}). Usual causes: the ServiceAccount lacks 'get' and "
                "'create' on pods/exec, the session pod is not Running, or an "
                "admission policy refused the connection."
            )
        if missing:
            return RuntimeError(
                "kubernetes exec: refused — this ServiceAccount is missing "
                f"{', '.join(missing)} on pods/exec in "
                f"{self._pod_ref.namespace if self._pod_ref else 'the namespace'}. "
                "BOTH verbs are required (see the RBAC on the Kubernetes docs page); `kubectl auth "
                "can-i get pods/exec` answering yes is not sufficient."
            )
        return RuntimeError(
            "kubernetes exec: the API server refused the websocket upgrade "
            f"({exc}), and RBAC on pods/exec looks correct. Check that the "
            "session pod is Running and that no admission policy or proxy is "
            "blocking the connection upgrade."
        )

    @staticmethod
    def _drain(resp, chunks: list[str]) -> None:
        if resp.peek_stdout():
            chunks.append(resp.read_stdout())
        if resp.peek_stderr():
            chunks.append(resp.read_stderr())

    @staticmethod
    def _safe_returncode(resp) -> "int | None":
        """WSClient.returncode, or None when unknown (abnormal disconnect or
        still-open stream). Callers must treat None as failure, not success."""
        try:
            rc = resp.returncode
        except Exception:
            return None
        return rc if isinstance(rc, int) else None

    def _exec_capture(
        self, command: list[str], *, timeout: int = 60
    ) -> "tuple[str, int | None]":
        """Blocking exec for the file-sync transport: (output, returncode),
        raising TimeoutError instead of mistaking still-running for success."""
        resp = self._open_stream(command)
        chunks: list[str] = []
        deadline = time.monotonic() + max(1, timeout)
        timed_out = False
        try:
            while True:
                if not resp.is_open():
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                resp.update(timeout=1)
                self._drain(resp, chunks)
            self._drain(resp, chunks)
        finally:
            try:
                resp.close()
            except Exception:
                pass
        if timed_out:
            raise TimeoutError(
                f"kubernetes exec did not finish within {timeout}s: "
                f"{' '.join(command[:2])}"
            )
        return "".join(chunks), self._safe_returncode(resp)

    def _forget_pod_if_dead(self, exc: Exception) -> bool:
        """Drop the pod ref when the pod is gone or terminal-but-present
        (phase Failed serves exec 400 forever, and nothing deletes it)."""
        dead = (
            getattr(exc, "status", None) == 404
            or "not found" in str(exc).lower()
        )
        if not dead:
            with self._lock:
                ref = self._pod_ref
            api = self._exec_api
            if ref is None or api is None:
                return False
            try:
                pod = api_call(
                    api.read_namespaced_pod,
                    name=ref.pod_name, namespace=ref.namespace,
                )
                dead = pod_cannot_exec(pod, ref.container)
            except Exception as read_exc:
                dead = getattr(read_exc, "status", None) == 404
        if not dead:
            return False
        with self._lock:
            self._pod_ref = None
        self._snapshot_ready = False
        return True

    def _run_bash(
        self, cmd_string: str, *, login: bool = False, timeout: int = 120,
        stdin_data: str | None = None,
    ):
        shell = "bash -l -c" if login else "bash -c"
        # Closing the websocket does not stop the remote process, so cancel()
        # finds this command's tree via a unique env marker from a second exec.
        marker = f"HERMES_EXEC_{uuid.uuid4().hex[:16]}"
        tagged = f"export {marker}=1; exec {shell} {shlex.quote(cmd_string)}"
        command = ["bash", "-c", tagged]
        # stdin_data is always None (heredoc mode). The deadline is a backstop
        # under _wait_for_process so a wedged websocket cannot pin the worker.
        deadline = time.monotonic() + max(1, int(timeout or 0)) + _EXEC_GRACE_SECONDS
        # Created before the worker starts, or a kill() landing mid-connect
        # would be erased.
        state = _ExecState()

        def exec_fn() -> tuple[str, int]:
            chunks: list[str] = []
            resp = None
            timed_out = False
            try:
                resp = self._open_stream(command)
                with self._lock:
                    state.stream = resp
                    already_cancelled = state.cancelled
                if already_cancelled:
                    # kill() landed while the stream was opening.
                    return "", 130
                while resp.is_open():
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    resp.update(timeout=1)
                    self._drain(resp, chunks)
                # Drain any tail buffered by the final update().
                self._drain(resp, chunks)
            except Exception as exc:
                # Return whatever the command produced instead of losing it.
                with self._lock:
                    cancelled = state.cancelled
                if not cancelled:
                    logger.warning("exec stream error: %s", exc)
                    # Check for a dead pod first; the raw websocket error
                    # tells the model nothing it can act on.
                    died = self._forget_pod_if_dead(exc)
                    chunks.append(
                        "\n[the session workspace stopped (deadline, eviction "
                        "or OOM) and its files are gone; the next command "
                        "provisions a fresh one]"
                        if died else f"\n[kubernetes exec error: {exc}]"
                    )
                return "".join(chunks), (130 if cancelled else 1)
            finally:
                with self._lock:
                    state.stream = None
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            with self._lock:
                cancelled = state.cancelled
            if cancelled:
                return "".join(chunks), 130
            if timed_out:
                chunks.append(f"\n[kubernetes: exec exceeded {timeout}s]")
                return "".join(chunks), 124
            rc = self._safe_returncode(resp)
            if rc is None:
                # Unknown is not success; reporting 0 would lie to the model.
                chunks.append("\n[kubernetes: exec status unavailable]")
                return "".join(chunks), 1
            return "".join(chunks), rc

        def cancel() -> None:
            with self._lock:
                state.cancelled = True
                stream = state.stream
            # Closing the websocket frees our side only; without the kill
            # below the remote command keeps running after "cancellation".
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            # Kill the tagged tree from a second exec. Best effort: a failure
            # here must never mask the interrupt the caller asked for.
            try:
                self._exec_capture(
                    ["sh", "-c",
                     "for p in $(grep -lz " + marker + "=1 /proc/*/environ "
                     "2>/dev/null | sed 's#/proc/##;s#/environ##'); do "
                     "kill -TERM -$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' ') "
                     "2>/dev/null || kill -TERM \"$p\" 2>/dev/null; done; "
                     "sleep 0.3; "
                     "for p in $(grep -lz " + marker + "=1 /proc/*/environ "
                     "2>/dev/null | sed 's#/proc/##;s#/environ##'); do "
                     "kill -KILL \"$p\" 2>/dev/null; done; true"],
                    timeout=15,
                )
            except Exception as exc:
                logger.debug("exec cancel: kill pass failed: %s", exc)

            # The pod survives cancellation on purpose: ordinary timeouts
            # also land here, and destroying it would wipe /workspace.

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    # -- file sync transport --------------------------------------------
    def agent_visible_cache_base(self) -> str:
        """Where the agent sees ~/.hermes inside the pod (extension hook used
        by tools/image_generation_tool.py to surface generated artifacts)."""
        return self._hermes_base

    def _bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Push many files into the pod in one tar, over the exec stdin
        channel and never over argv (see _stdin_upload)."""
        if not files:
            return
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for host_path, remote_path in files:
                try:
                    tar.add(host_path, arcname=remote_path.lstrip("/"))
                except OSError as exc:
                    logger.debug("skipping %s: %s", host_path, exc)
        payload = buf.getvalue()
        if not payload:
            return

        dirs = unique_parent_dirs(files)
        if dirs:
            out, rc = self._exec_capture(["sh", "-c", quoted_mkdir_command(dirs)])
            if rc != 0:
                raise RuntimeError(
                    "kubernetes file sync: could not create target directories "
                    f"(exit {rc if rc is not None else 'unknown'}): {out.strip()}"
                )

        # encodebytes() wraps at 76 columns, so the remote sed reads short
        # lines instead of one multi-MB line.
        self._stdin_upload(base64.encodebytes(payload).decode("ascii"))

    def _stdin_upload(self, encoded: str, *, timeout: int = 120) -> None:
        """Stream a base64 tar over exec stdin, never argv: exec argv lands in
        the apiserver audit log, and this payload includes credential files."""
        from kubernetes.stream.ws_client import STDIN_CHANNEL, V5_CHANNEL_PROTOCOL

        remote = (
            f"sed -n '/^{_SYNC_SENTINEL}$/q;p' | base64 -d | tar xf - -C /"
        )
        resp = self._open_stream(["sh", "-c", remote], stdin=True)
        chunks: list[str] = []
        timed_out = False
        # Set before the write loop: writes are the phase most likely to hang.
        deadline = time.monotonic() + max(1, timeout)
        try:
            if not encoded.endswith("\n"):
                encoded += "\n"
            for offset in range(0, len(encoded), _STDIN_CHUNK_BYTES):
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                resp.write_stdin(encoded[offset:offset + _STDIN_CHUNK_BYTES])
                # Pump the read side so remote output cannot deadlock writes.
                self._drain(resp, chunks)
            if not timed_out:
                resp.write_stdin(f"{_SYNC_SENTINEL}\n")
            if getattr(resp, "subprotocol", None) == V5_CHANNEL_PROTOCOL:
                try:
                    resp.close_channel(STDIN_CHANNEL)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("stdin half-close unavailable: %s", exc)

            while not timed_out:
                if not resp.is_open():
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                resp.update(timeout=1)
                self._drain(resp, chunks)
            self._drain(resp, chunks)
        finally:
            try:
                resp.close()
            except Exception:
                pass

        if timed_out:
            raise TimeoutError(
                f"kubernetes file sync: tar extract did not finish within "
                f"{timeout}s ({''.join(chunks).strip()})"
            )
        rc = self._safe_returncode(resp)
        if rc != 0:
            # rc None lands here too: an unknown status must never commit the
            # sync state as if it succeeded.
            raise RuntimeError(
                "kubernetes file sync: tar extract failed (exit "
                f"{rc if rc is not None else 'unknown'}): {''.join(chunks).strip()}"
            )

    def _upload_file(self, host_path: str, remote_path: str) -> None:
        self._bulk_upload([(host_path, remote_path)])

    def _delete_files(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        out, rc = self._exec_capture(["sh", "-c", quoted_rm_command(remote_paths)])
        if rc != 0:
            raise RuntimeError(
                "kubernetes file sync: delete failed (exit "
                f"{rc if rc is not None else 'unknown'}): {out.strip()}"
            )

    # -- teardown -------------------------------------------------------
    def cleanup(self):
        with self._lock:
            ref = getattr(self, "_pod_ref", None)
            self._pod_ref = None
            # Final: a late _ensure_pod must fail, not resurrect the pod.
            self._torn_down = True
        if ref is None:
            return
        for attempt in (1, 2):
            try:
                self._provisioner.destroy(ref)
                return
            except Exception as exc:
                if attempt == 1:
                    logger.warning("cleanup: destroy failed (%s); retrying once", exc)
                    # Retry on a fresh client: at exit the cached one may
                    # hold dead temp cert paths (see refresh_api).
                    try:
                        self._provisioner.refresh_api()
                    except Exception:
                        pass
                else:
                    logger.error(
                        "cleanup: could not delete session pod %s (%s); it is "
                        "left to ownerReference GC or "
                        "spec.activeDeadlineSeconds to collect.",
                        ref.pod_name, exc,
                    )
