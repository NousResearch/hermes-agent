"""``FanoutTransport.auth_identity``: the login shared by every attached peer, else None (fail closed).

Session slots hold a FanoutTransport once a second client attaches. Auth-bound lookups (browser-control
principal) read ``transport.auth_identity``; a fanout may only name a login when ALL current peers agree.
"""
from tui_gateway.transport import FanoutTransport

ALICE = {"user_id": "alice", "provider": "basic"}


class Peer:
    def __init__(self, identity=None, *, has_attr=True):
        self.frames = []
        if has_attr:
            self.auth_identity = identity

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        return None


def fanout(*peers):
    return FanoutTransport(*peers)


def test_single_authenticated_peer_returns_its_identity():
    assert fanout(Peer(dict(ALICE))).auth_identity == ALICE


def test_all_peers_with_same_identity_return_it():
    assert fanout(Peer(dict(ALICE)), Peer(dict(ALICE)), Peer(dict(ALICE))).auth_identity == ALICE


def test_peers_with_different_user_fail_closed():
    assert fanout(Peer(dict(ALICE)), Peer({"user_id": "bob", "provider": "basic"})).auth_identity is None


def test_peers_with_same_user_but_different_provider_fail_closed():
    assert fanout(Peer(dict(ALICE)), Peer({"user_id": "alice", "provider": "oidc"})).auth_identity is None


def test_authenticated_and_identityless_peer_fail_closed_in_either_order():
    assert fanout(Peer(dict(ALICE)), Peer(None)).auth_identity is None
    assert fanout(Peer(None), Peer(dict(ALICE))).auth_identity is None


def test_peer_without_the_attribute_counts_as_identityless():
    assert fanout(Peer(dict(ALICE)), Peer(has_attr=False)).auth_identity is None


def test_no_peers_means_no_identity():
    assert fanout().auth_identity is None


def test_all_peers_identityless_means_no_identity():
    assert fanout(Peer(None), Peer(None)).auth_identity is None


def test_malformed_identities_fail_closed():
    for bad in ({}, {"user_id": "alice"}, {"provider": "basic"}, {"user_id": "", "provider": "basic"},
                {"user_id": "alice", "provider": "  "}, {"user_id": 7, "provider": "basic"}, "alice", ["alice"]):
        assert fanout(Peer(bad)).auth_identity is None, bad


def test_identity_is_reevaluated_from_current_peers_after_attach_and_detach():
    a, b, c = Peer(dict(ALICE)), Peer({"user_id": "bob", "provider": "basic"}), Peer(dict(ALICE))
    transport = fanout(a)
    assert transport.auth_identity == ALICE

    transport.attach(b)  # a second, different login joins
    assert transport.auth_identity is None

    transport.detach(b)  # ...and leaves: the stale "no identity" must not stick
    assert transport.auth_identity == ALICE

    transport.attach(c)  # same login on a second window
    assert transport.auth_identity == ALICE

    transport.detach(a)
    transport.detach(c)
    assert transport.auth_identity is None  # nobody left


def test_identity_of_a_departed_peer_is_never_cached():
    a = Peer(dict(ALICE))
    transport = fanout(a)
    assert transport.auth_identity == ALICE
    transport.detach(a)
    replacement = Peer({"user_id": "bob", "provider": "basic"})
    transport.attach(replacement)
    assert transport.auth_identity == {"user_id": "bob", "provider": "basic"}


def test_reconnect_of_the_same_peer_is_reevaluated():
    a, other = Peer(dict(ALICE)), Peer(None)
    transport = fanout(a)
    transport.attach(other)
    assert transport.auth_identity is None
    transport.detach(other)
    transport.detach(a)
    transport.attach(a)  # reconnect
    assert transport.auth_identity == ALICE


def test_identity_change_on_a_live_peer_is_seen_immediately():
    a, b = Peer(dict(ALICE)), Peer(dict(ALICE))
    transport = fanout(a, b)
    assert transport.auth_identity == ALICE
    b.auth_identity = {"user_id": "mallory", "provider": "basic"}
    assert transport.auth_identity is None


def test_returned_identity_is_a_copy():
    a = Peer(dict(ALICE))
    got = fanout(a).auth_identity
    got["user_id"] = "mallory"
    assert a.auth_identity == ALICE


def test_nested_fanout_peers_follow_the_same_rule():
    inner_same = fanout(Peer(dict(ALICE)), Peer(dict(ALICE)))
    inner_mixed = fanout(Peer(dict(ALICE)), Peer(None))
    assert fanout(inner_same, Peer(dict(ALICE))).auth_identity == ALICE
    assert fanout(inner_mixed, Peer(dict(ALICE))).auth_identity is None
