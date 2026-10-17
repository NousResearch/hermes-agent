"""Child-enforced membership leases survive a lost parent release or private writer."""
from __future__ import annotations

import threading
from time import monotonic
import uuid

MEMBERSHIP_SECONDS = 15.0
MEMBERSHIP_POLL_SECONDS = 2.0


class HostBoundPeer:
    """Logical subscriber, live only within a child-clock parent-confirmation lease."""
    def __init__(self, owner):
        provider, _, user = owner.partition(":")
        self.auth_identity = {"provider": provider, "user_id": user}
        self._lease_lock = threading.Lock()  # leaf: never acquire protocol/transport locks from here
        self._revoked = False
        self._expires_at = monotonic() + MEMBERSHIP_SECONDS
        self._confirmed = False

    def _closed_locked(self):
        if monotonic() >= self._expires_at:
            self._revoked = True
        return self._revoked

    @property
    def _closed(self):
        # Fail closed even if cleanup is waiting on an engine/transport lock.
        with self._lease_lock:
            return self._closed_locked()

    @property
    def confirmed(self):
        with self._lease_lock:
            return self._confirmed

    def renew(self, deadline):
        if self._closed:
            return
        with self._lease_lock:
            # The earlier observation grants nothing: expiry or close may have won since it.
            if not self._closed_locked():
                self._expires_at = max(self._expires_at, deadline)
                self._confirmed = True

    def write(self, _obj):
        return not self._closed

    def close(self):
        with self._lease_lock:
            self._revoked = True


class HostMemberships:
    def __init__(self, protocol):
        self.protocol = protocol
        self.stop = threading.Event()
        self._challenge = None

    def start(self):
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(self._run, name="host-membership-lease").start()

    def _run(self):
        while not self.stop.wait(MEMBERSHIP_POLL_SECONDS):
            self.poll()

    def release(self, subscription):
        protocol = self.protocol
        with protocol._lock:
            member = protocol._members.pop(subscription, None)
            for pending in protocol._pending.values():
                if pending.get("subscription") == subscription:
                    pending["peer"].close()
                    pending["event"].set()
            if member is not None:
                member[0].close()
        if member is not None:
            from . import server
            protocol._executor.submit(server._detach_session_transport, member[1], member[0])

    def reap(self):
        with self.protocol._lock:
            expired = [key for key, (peer, _session) in self.protocol._members.items() if peer._closed]
        for subscription in expired:
            self.release(subscription)

    def poll(self):
        self.reap()
        protocol = self.protocol
        with protocol._lock:
            snapshot = dict(protocol._members)
            if not snapshot or protocol.host._closed.is_set():
                self._challenge = None
                return
            nonce = uuid.uuid4().hex
            # A delayed positive reply cannot extend authority from its delivery time.
            self._challenge = (nonce, monotonic() + MEMBERSHIP_SECONDS, snapshot)
        protocol.host.emit({"type": "conditional.membership", "boot_id": protocol.host._boot_id,
            "challenge": nonce, "members": [{"subscription": key, "sid": session["creation_binding"].session_id}
                                             for key, (_peer, session) in snapshot.items()]})

    def answer(self, frame):
        protocol = self.protocol
        revoked = []
        with protocol._lock:
            challenge = self._challenge
            if challenge is None or challenge[0] != frame.get("challenge"):
                return
            self._challenge = None
            _nonce, deadline, snapshot = challenge
            allowed = set(frame.get("members") or [])
            for key, member in snapshot.items():
                if protocol._members.get(key) is not member:
                    continue
                peer = member[0]
                if key in allowed:
                    peer.renew(deadline)
                elif peer.confirmed:
                    revoked.append(key)
                # First commit may precede parent publication. No confirmation gives
                # only the original bounded lease, never a fresh grant or extension.
        for key in revoked:
            self.release(key)
        self.reap()

    def close(self):
        self.stop.set()
        with self.protocol._lock:
            self._challenge = None
            for peer, _session in self.protocol._members.values():
                peer.close()
            self.protocol._members.clear()
