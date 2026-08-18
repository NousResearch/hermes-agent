"""browser_exec egress interposer — stdlib-only child-side socket guard.

Region B of the browser_exec network-boundary class closure (PR #84999).
Loaded by ``sitecustomize`` (via PYTHONPATH injection) at interpreter boot,
BEFORE any harness or model code. It interposes the Python ``socket`` layer
in the CLI subprocess so direct ``connect``/``sendto``/``sendmsg``/
``create_connection`` egress is checked at connect time:

* **Loopback is allowed unconditionally** — every browser helper
  (``new_tab``, ``js``, ``page_info``, …) is a loopback IPC connect from this
  same interpreter to the local browser daemon, so a loopback block would
  break the tool.
* **Public egress is allowed.**
* **Private EXTERNAL destinations are blocked** (RFC1918, CGNAT,
  link-local/IMDS, multicast, unspecified, reserved) unless the operator's
  ``allow_private_urls`` opt-out is active — and the IMDS/metadata floor is
  armed unconditionally.
* Resolution happens ONCE with the captured ``getaddrinfo``; the caller
  dials the checked literal, never re-resolves (DNS-rebinding safe).

Correct Python-wrapper subclassing (probe-verified, Region B consensus):

* The stdlib ``socket.socket`` is a PYTHON wrapper around the C type
  ``_socket.socket``. Subclassing the C type directly drops ``makefile``/
  ``accept``/``dup``/``sendfile``; subclassing the wrapper keeps them.
* The wrapper methods are patched IN PLACE first, then a visible
  ``_GuardedSocket`` subclass is introduced and reassigned across
  ``socket.socket``, ``_socket.socket``, and ``socket._socket.socket`` so
  every named path — including ``socket.socket.__bases__[0]`` — is guarded.
* ``__init__`` routes to the captured C init and replicates the wrapper's
  ``_io_refs``/``_closed`` initialization (no recursion on the reassignment,
  no close() crash).

Captured-primitive hygiene (defect-4 fix + closure-closure): the raw C
callables (``_socket.socket.connect``, ``_sock.getaddrinfo``, …) live ONLY
as attributes of the private module-global holder ``_PRIMS``. Every
exported callable (``install``, ``_check_and_resolve``,
``_assert_allowed_ip``, ``_g_create_connection``, ``_g_getaddrinfo``, and
the ``_GuardedSocket`` methods) is a plain module-level function whose
``__closure__`` is None: the primitives are reached by name through the
module globals at call time. A ``f.__closure__`` walk of any exported
callable therefore cannot recover ``<method 'connect' of '_socket.socket'
objects>`` — the one-line ``g._C_CONNECT(...)`` bypass is gone, and the
~6-line closure-introspection bypass is gone with it. The module's
``__getattribute__`` additionally raises AttributeError for every
non-exported single-underscore name, so ``g._PRIMS`` (and any other
internal name) is not reachable by attribute access either. The accepted
residual is the deliberate deep-introspection walk — MRO/ctypes
(``socket.socket.__bases__[0].__bases__[0]``) or ``__dict__`` — NOT a
closure or attribute intro.

Threat model — child-process egress coverage boundary (accepted residual):
the interposer covers ONLY the CLI process that imports this module via
sitecustomize, and the proxy pinning covers only spawned processes that
honor the proxy env vars. Model-spawned env-stripped or native children —
``subprocess`` from harness code with a scrubbed environment (no
PYTHONPATH, so no sitecustomize and no socket guard; proxy vars dropped),
or native binaries such as a statically-linked curl — bypass both the
interposer and the proxy pinning and emit no guard markers, so the parent
cannot observe or block their egress. This is an accepted residual for this
PR: OS-level egress enforcement (firewall, seccomp, network namespaces,
sandbox) is out of scope.

Markers go to an install-time ``os.dup(2)`` fd so ``sys.stderr`` rebinding
and ``dup2(NUL, 2)`` cannot swallow them; the parent parses stderr for
``__HERMES_EGRESS_GUARD__:installed:<nonce>`` / ``:block:`` / ``:tamper:`` /
``:disabled:`` and withholds output on any block/tamper/missing-nonce.
"""

import atexit
import errno
import ipaddress
import json
import os
import socket as _sock
import sys
import types
from urllib.parse import urlsplit

import _socket

__all__ = [
    "install",
    "_check_and_resolve",
    "_assert_allowed_ip",
    "EgressBlocked",
    "_GuardedSocket",
    "_g_create_connection",
]


class EgressBlocked(OSError):
    """Raised when a connect targets a private/internal destination."""


# ── Captured C primitives (hidden indirection — closure-closure fix) ──────
# The raw C callables live ONLY as attributes of this private holder object.
# No exported callable closes over them (every exported callable below is a
# plain module-level function with __closure__ = None); the guarded methods
# reach them by name through the module globals at call time. The module
# __getattribute__ at the bottom of this file hides every non-exported
# single-underscore name (including ``_PRIMS`` itself) from attribute access.
class _Prims:
    __slots__ = (
        "socket_type", "init", "connect", "connect_ex", "sendto",
        "sendmsg", "create", "getaddrinfo",
    )


_PRIMS = _Prims()
_PRIMS.socket_type = _socket.socket           # C type
_PRIMS.init = _socket.socket.__init__         # C-level tp_init
_PRIMS.connect = _socket.socket.connect
_PRIMS.connect_ex = _socket.socket.connect_ex
_PRIMS.sendto = _socket.socket.sendto
_PRIMS.sendmsg = _socket.socket.sendmsg if hasattr(_socket.socket, "sendmsg") else None
_PRIMS.create = _sock.create_connection
_PRIMS.getaddrinfo = _sock.getaddrinfo        # resolve-once primitive; NOT patched

# Mutable guard state (module globals — hidden from attribute access).
_MARKER_FD = None
_INSTALLED = False

_POLICY = {
    "nonce": "",
    "allow_private": False,
    "blocked_hostnames": ("metadata.google.internal", "metadata.goog"),
    "always_blocked_ips": (),
    "always_blocked_networks": (),
    "cgnat_network": "100.64.0.0/10",
    "allow_hosts": (),
}
_ALWAYS_BLOCKED_IP_SET = frozenset()
_ALWAYS_BLOCKED_NET_LIST: tuple = ()
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_ALLOWLIST: list = []      # list of (host_lower, port_or_None)
_ALLOWLIST_IPS: set = set()


def _marker(kind: str, payload: str = "") -> None:
    fd = _MARKER_FD if _MARKER_FD is not None else 2
    try:
        os.write(fd, f"__HERMES_EGRESS_GUARD__:{kind}:{payload}\n".encode("utf-8", "replace"))
    except Exception:
        pass


def _load_policy() -> None:
    """Read the per-spawn policy snapshot from env (rendered by the parent)."""
    global _POLICY, _ALWAYS_BLOCKED_IP_SET, _ALWAYS_BLOCKED_NET_LIST, _CGNAT_NETWORK
    global _ALLOWLIST, _ALLOWLIST_IPS
    raw = os.environ.get("HERMES_BROWSER_EXEC_EGRESS_POLICY", "")
    if raw:
        try:
            _POLICY.update(json.loads(raw))
        except Exception:
            pass
    _ALWAYS_BLOCKED_IP_SET = frozenset(
        ipaddress.ip_address(s) for s in _POLICY.get("always_blocked_ips", ()) if s
    )
    nets = []
    for s in _POLICY.get("always_blocked_networks", ()):
        try:
            nets.append(ipaddress.ip_network(s))
        except ValueError:
            pass
    _ALWAYS_BLOCKED_NET_LIST = tuple(nets)
    try:
        _CGNAT_NETWORK = ipaddress.ip_network(_POLICY.get("cgnat_network") or "100.64.0.0/10")
    except ValueError:
        pass

    # Allowlist: operator env + CDP endpoints from the spawn env.
    allow_entries: list[str] = []
    for key in ("HERMES_BROWSER_EXEC_EGRESS_ALLOW",):
        val = os.environ.get(key, "")
        if val:
            allow_entries.extend(v.strip() for v in val.split(",") if v.strip())
    for key in ("BU_CDP_WS", "BU_CDP_URL"):
        val = os.environ.get(key, "")
        if val:
            try:
                parsed = urlsplit(val)
                if parsed.hostname:
                    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
                    allow_entries.append(f"{parsed.hostname}:{port}")
            except ValueError:
                pass
    for entry in allow_entries:
        host, sep, port_s = entry.rpartition(":")
        if not sep:
            host, port_s = entry, ""
        host = host.strip().lower().rstrip(".")
        if not host:
            continue
        port = None
        if port_s.isdigit():
            port = int(port_s)
        _ALLOWLIST.append((host, port))
        # Resolve allowlist hostnames to (ip, port) pairs (best-effort) so
        # operator/CDP endpoints that resolve privately are reachable — the
        # port stays scoped: 10.0.0.9:8443 allowed does NOT allow
        # 10.0.0.9:8444.
        try:
            for _fam, _typ, _proto, _canon, sockaddr in _PRIMS.getaddrinfo(
                host, port or None, _sock.AF_UNSPEC, _sock.SOCK_STREAM
            ):
                _ALLOWLIST_IPS.add((sockaddr[0].split("%", 1)[0], port))
        except Exception:
            pass


# ── Policy helpers ─────────────────────────────────────────────────────────

def _ip_blocked(ip: ipaddress._BaseAddress) -> bool:
    """Floor + private-external check for one IP (toggle-aware)."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip in _ALWAYS_BLOCKED_IP_SET:
        return True
    if ip.is_link_local:
        return True  # 169.254.0.0/16 + fe80::/10 — floor, always armed
    if ip.is_loopback:
        return False  # loopback allowed unconditionally (checked earlier anyway)
    if ip in _CGNAT_NETWORK:
        return not _POLICY.get("allow_private", False)
    if ip.is_private or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return not _POLICY.get("allow_private", False)
    if isinstance(ip, ipaddress.IPv6Address) and ip in ipaddress.ip_network("::/96"):
        return True  # IPv4-compatible — floor class
    return False


def _assert_allowed_ip(ip: ipaddress._BaseAddress) -> None:
    """Raise EgressBlocked when *ip* must not be dialed (no markers here)."""
    if _ip_blocked(ip):
        raise EgressBlocked(f"egress guard blocked private/internal address {ip}")


def _check_and_resolve(self_or_none, address):
    """Validate one connect target; return the literal ``(ip, port)`` to dial.

    Raises ``EgressBlocked`` on any refusal. Callers dial exactly the
    returned literal and never re-resolve.
    """
    if not isinstance(address, (tuple, list)) or len(address) < 2:
        raise EgressBlocked(f"egress guard: malformed address {address!r}")
    host = address[0]
    if isinstance(host, bytes):
        try:
            host = host.decode("utf-8")
        except UnicodeDecodeError:
            raise EgressBlocked("egress guard: undecodable host bytes")
    host = str(host or "").strip().lower().rstrip(".")
    port = address[1]

    # 1. Non-IP socket families (AF_UNIX/AF_PACKET/local IPC): no network
    #    egress — allow.
    if self_or_none is not None:
        try:
            if getattr(self_or_none, "family", None) not in (_sock.AF_INET, _sock.AF_INET6):
                return host, port
        except Exception:
            pass

    # 2. Allowlist first — exact (host, port) or resolved-IP membership
    #    (port-scoped: an allowlisted 10.0.0.9:8443 does not open 8444).
    for allow_host, allow_port in _ALLOWLIST:
        if allow_port is not None and allow_port != port:
            continue
        if host == allow_host:
            return host, port
    if (host, port) in _ALLOWLIST_IPS or (host, None) in _ALLOWLIST_IPS:
        return host, port

    # 3. Loopback allow (the tool's working channel — daemon IPC + CDP).
    if host in ("localhost", "localhost.localdomain"):
        return host, port
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if literal.is_loopback:
            return host, port
        if isinstance(literal, ipaddress.IPv6Address) and literal.ipv4_mapped is not None \
                and literal.ipv4_mapped.is_loopback:
            return host, port
        _assert_allowed_ip(literal)
        return host, port

    # 4. Blocked-hostname floor (no DNS).
    if host in _POLICY.get("blocked_hostnames", ()):
        raise EgressBlocked(f"egress guard blocked internal hostname {host}")
    # 5. Hostname heuristics — private DNS names (toggle-armed).
    if host.endswith(".internal") or host.endswith(".local") or host.endswith(".lan"):
        if not _POLICY.get("allow_private", False):
            raise EgressBlocked(f"egress guard blocked private hostname {host}")

    # 7. Resolve ONCE with the captured primitive; dial the first checked
    #    literal. Any private answer blocks; gaierror fails closed.
    try:
        addr_info = _PRIMS.getaddrinfo(host, port, _sock.AF_UNSPEC, _sock.SOCK_STREAM)
    except Exception:
        raise EgressBlocked(f"egress guard: DNS failure for {host} (fail closed)")
    checked_ip = None
    for _fam, _typ, _proto, _canon, sockaddr in addr_info:
        ip_str = sockaddr[0].split("%", 1)[0]
        try:
            resolved = ipaddress.ip_address(ip_str)
        except ValueError:
            raise EgressBlocked(f"egress guard: unparseable address {sockaddr[0]!r}")
        if _ip_blocked(resolved):
            raise EgressBlocked(f"egress guard blocked private/internal address {ip_str}")
        if checked_ip is None:
            checked_ip = ip_str
    if checked_ip is None:
        raise EgressBlocked(f"egress guard: no usable addresses for {host}")
    return checked_ip, port


# ── Guarded method implementations (replicate stdlib wrapper mechanics) ────

def _g_init(self, family=-1, type=-1, proto=-1, fileno=None):
    if fileno is None:
        if family == -1:
            family = _sock.AF_INET
        if type == -1:
            type = _sock.SOCK_STREAM
        if proto == -1:
            proto = 0
    _PRIMS.init(self, family, type, proto, fileno)
    self._io_refs = 0
    self._closed = False


def _g_connect(self, address):
    try:
        ip, port = _check_and_resolve(self, address)
    except EgressBlocked as exc:
        _marker("block", f"{address[0]}:{address[1]}:connect")
        raise
    return _PRIMS.connect(self, (ip, port))


def _g_connect_ex(self, address):
    try:
        ip, port = _check_and_resolve(self, address)
    except EgressBlocked as exc:
        _marker("block", f"{address[0]}:{address[1]}:connect_ex")
        return errno.EACCES
    return _PRIMS.connect_ex(self, (ip, port))


def _g_sendto(self, data, *args):
    address = args[-1] if args and isinstance(args[-1], tuple) else None
    if address is not None:
        try:
            _check_and_resolve(self, address)
        except EgressBlocked as exc:
            _marker("block", f"{address[0]}:{address[1]}:sendto")
            raise
    return _PRIMS.sendto(self, data, *args)


def _g_sendmsg(self, buffers, ancdata=(), flags=0, address=None):
    if address is not None:
        try:
            _check_and_resolve(self, address)
        except EgressBlocked as exc:
            _marker("block", f"{address[0]}:{address[1]}:sendmsg")
            raise
    if _PRIMS.sendmsg is None:
        raise AttributeError("sendmsg not available on this platform")
    return _PRIMS.sendmsg(self, buffers, ancdata, flags, address)


def _g_create_connection(address, timeout=_sock._GLOBAL_DEFAULT_TIMEOUT,
                         source_address=None, *, all_errors=False):
    try:
        ip, port = _check_and_resolve(None, address)
    except EgressBlocked as exc:
        _marker("block", f"{address[0]}:{address[1]}:create_connection")
        raise
    return _PRIMS.create((ip, port), timeout, source_address, all_errors=all_errors)


def _g_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Guarded ``getaddrinfo``: hostname floor before any DNS; private-external
    answers are filtered out (loopback allowed — the tool's working channel).
    """
    h = str(host or "").strip().lower().rstrip(".")
    if h in _POLICY.get("blocked_hostnames", ()):
        raise EgressBlocked(f"egress guard blocked internal hostname {h}")
    if h.endswith(".internal") or h.endswith(".local") or h.endswith(".lan"):
        if not _POLICY.get("allow_private", False):
            raise EgressBlocked(f"egress guard blocked private hostname {h}")
    infos = _PRIMS.getaddrinfo(host, port, family, type, proto, flags)
    kept = []
    for info in infos:
        try:
            ip_str = info[4][0].split("%", 1)[0]
            ip = ipaddress.ip_address(ip_str)
        except (ValueError, IndexError, TypeError):
            kept.append(info)  # non-IP sockaddr (AF_UNIX etc.) — keep
            continue
        if ip.is_loopback:
            kept.append(info)
            continue
        if _ip_blocked(ip):
            continue  # filter private-external answers
        kept.append(info)
    if not kept and infos:
        raise EgressBlocked(f"egress guard: no public answers for {h}")
    return kept


# ── Installation ───────────────────────────────────────────────────────────

def install() -> None:
    """Interpose the socket layer (idempotent). Must run before model code."""
    global _MARKER_FD, _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    try:
        _MARKER_FD = os.dup(2)
    except Exception:
        _MARKER_FD = None
    _load_policy()
    nonce = str(_POLICY.get("nonce") or "")
    _marker("installed", nonce)

    # STEP 1 — patch the ORIGINAL wrapper class IN PLACE so any reference to
    # it (incl. socket.socket.__bases__[0] after step 2) is guarded.
    _sock.socket.__init__ = _g_init
    _sock.socket.connect = _g_connect
    _sock.socket.connect_ex = _g_connect_ex
    _sock.socket.sendto = _g_sendto
    if _PRIMS.sendmsg is not None:
        _sock.socket.sendmsg = _g_sendmsg

    # STEP 2 — the visible guarded class (base = the wrapper, patched in
    # STEP 1 above; defined at import time so the subclass observes the
    # in-place patches through inheritance).
    # STEP 3 — reassign EVERY named path (no recursion: __init__ routes to
    # the captured C init, never to `_socket.socket.__init__` by name).
    _sock.socket = _GuardedSocket
    _socket.socket = _GuardedSocket
    _socket.__dict__["socket"] = _GuardedSocket
    _sock.create_connection = _g_create_connection
    _sock.getaddrinfo = _g_getaddrinfo

    def _exit_check():
        if (_sock.socket is not _GuardedSocket
                or _socket.socket is not _GuardedSocket
                or _sock.create_connection is not _g_create_connection
                or _sock.getaddrinfo is not _g_getaddrinfo):
            _marker("tamper", "binding")

    atexit.register(_exit_check)
    sys.modules.setdefault("socket", _sock)


# The guarded class subclasses the stdlib WRAPPER (not the C type), so the
# wrapper's Python machinery (makefile/accept/dup/sendfile) survives. It is
# defined at import time against the still-unpatched wrapper; install()
# patches the wrapper in place BEFORE any socket is constructed, so all
# instances observe the guarded methods through inheritance. The class is
# exported (``__all__``) and deliberately named ``socket`` for repr parity.
class _GuardedSocket(_sock.socket):
    __module__ = "socket"
    __qualname__ = "socket"


# ── Module hygiene: hide every non-exported single-underscore name ─────────
# `g._PRIMS`, `g._POLICY`, `g._C_CONNECT`-style misses, etc. all raise
# AttributeError; only `__all__` names and dunders remain reachable by
# attribute access. Internal code references these globals through function
# __globals__ dicts, so hiding them from attribute access breaks nothing.

def __getattr__(name: str):
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _EgressGuardModule(types.ModuleType):
    def __getattribute__(self, name: str):
        if name.startswith("_") and not name.startswith("__") and name not in __all__:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _EgressGuardModule
