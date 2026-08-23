"""Property-style regression suite: Bot Mode canonical-chat resolution algebra.

Bug class (10 incidents in 2 weeks): canonical "Bot Chat" resolution forking
or minting duplicates —
  #92040  newest-session preference made the pinned canonical chat unreachable
  #90705  null canonical pointer minted a duplicate empty Bot Chat per click
  #92692  fail-open minting forked fresh chats after a desktop update
  #90005  same-name group recreation reused old member sessions (PR #91082)
  #90732  infinite fork loop (set_session_title silently dropping conflicts)
  PR #92129  name-based resolution replaced session-id pins for good

Standing maintainer rule: canonical-chat identity must FAIL CLOSED — never
mint on ambiguity. Post-#92129 the contract is name-as-identity:

  One bot = ONE canonical forever-chat = (profile, session titled exactly
  "Bot Chat"). SessionDB title uniqueness (compare-and-swap in
  ``_set_session_title``, ValueError on conflict) makes that pair a registry
  of at most one row. Resolution (``session.list {title}`` and
  ``profiles.list``'s ``canonical_session``) is a PURE READ: absent registry
  → empty result / None, and creation is an explicit, separate,
  adopt-before-mint path (desktop plugin.js). No stored session-id pointer
  exists anywhere in the resolution path.

What is pinned here is the Python layer the desktop relies on. The desktop
click/create flow itself (plugin.js), and group ROOM member-session lifecycle
(#90005's original surface), live in Electron JS and are covered by
tests/canonical-chat-registry.test.mjs et al.; here we pin the DB/RPC algebra
those flows depend on, including a Python re-enactment of the #90005 and
#92692 shapes at the registry layer.
"""

from __future__ import annotations

import threading

import pytest

import tui_gateway.server as srv

CANON = "Bot Chat"


# ---------------------------------------------------------------------------
# Fixtures (style shared with tests/tui_gateway/test_profiles_list_canonical_session.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Temp HERMES_HOME so the gateway resolvers read a throwaway state.db.

    ``session.list`` reaches the DB through the gateway's shared launch
    handle (``srv._get_db``), which is bound at process launch — so, like
    tests/tui_gateway/test_session_hidden_rpc.py, we point it at the temp
    DB directly. ``profiles.list`` reads each profile dir's state.db itself
    and only needs the env override.
    """
    h = tmp_path / ".hermes"
    h.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))

    from hermes_state import SessionDB

    handles = []

    def _shared_db():
        database = SessionDB(db_path=h / "state.db")
        handles.append(database)
        return database

    monkeypatch.setattr(srv, "_get_db", _shared_db)
    yield h
    for database in handles:
        try:
            database.close()
        except Exception:
            pass


def _db(home):
    from hermes_state import SessionDB

    return SessionDB(db_path=home / "state.db")


def _add_session(db, sid, *, title="", ts=1000, source="cli", hidden=False, text="hi"):
    db.create_session(sid, source)
    db.append_message(sid, "user", text, timestamp=ts)
    if title:
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET title = ?, title_source = 'user' WHERE id = ?",
                (title, sid),
            )
    if hidden:
        db.set_session_hidden(sid, True)


def _count_sessions(home):
    import sqlite3

    conn = sqlite3.connect(home / "state.db")
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


def _canonical_rows(home):
    import sqlite3

    conn = sqlite3.connect(home / "state.db")
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT id FROM sessions WHERE title = ?", (CANON,)
            ).fetchall()
        ]
    finally:
        conn.close()


def _resolve(params=None):
    """The REAL registry resolver the desktop bot row uses: session.list {title}."""
    p = {"title": CANON, "include_hidden": True}
    p.update(params or {})
    envelope = srv._methods["session.list"](1, p)
    return envelope["result"]["sessions"]


def _canonical_via_profiles():
    """The REAL roster resolver: profiles.list → canonical_session."""
    envelope = srv._methods["profiles.list"](1, {})
    row = next(p for p in envelope["result"]["profiles"] if p["name"] == "default")
    return row["canonical_session"]


# ---------------------------------------------------------------------------
# State builders — every registry state named in the incident history
# ---------------------------------------------------------------------------


def _state_empty(home):
    _db(home).close()  # create the DB file with zero sessions


def _state_one_canonical(home):
    db = _db(home)
    _add_session(db, "forever", title=CANON, ts=1000, hidden=True,
                 text="forever chat content")
    db.close()


def _state_canonical_plus_newer(home):
    """#92040 shape: newer ordinary sessions must never outrank the registry."""
    db = _db(home)
    _add_session(db, "forever", title=CANON, ts=1000, hidden=True,
                 text="forever chat content")
    _add_session(db, "draft1", title="Scratch", ts=5000, text="stray draft")
    _add_session(db, "draft2", title="", ts=9000, text="newest untitled")
    db.close()


def _state_rotated(home):
    """Identity rotated: old row renamed away, a NEW session id holds the name.

    Name-as-identity (PR #92129) means the registry follows the NAME, not any
    remembered session id — a stale pointer must never be consulted.
    """
    db = _db(home)
    _add_session(db, "old-forever", title="Bot Chat (retired)", ts=1000,
                 text="old generation")
    _add_session(db, "new-forever", title=CANON, ts=2000, hidden=True,
                 text="new generation")
    db.close()


STATES = {
    "no_sessions": (_state_empty, None),
    "one_canonical": (_state_one_canonical, "forever"),
    "canonical_plus_newer": (_state_canonical_plus_newer, "forever"),
    "rotated_identity": (_state_rotated, "new-forever"),
}


# ---------------------------------------------------------------------------
# Property 1 — IDEMPOTENCE: resolve twice, same answer, in every state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", list(STATES))
def test_resolution_is_idempotent(home, state):
    """Resolving the canonical chat twice returns the identical row both times.

    #90705 / #90732: any resolver that mutates on read (minting, adopting,
    re-pinning) breaks idempotence and forks the forever-chat.
    """
    build, expected = STATES[state]
    build(home)

    first = _resolve()
    second = _resolve()

    assert first == second
    if expected is None:
        assert first == []  # fail closed: absent registry resolves to nothing
    else:
        assert [r["id"] for r in first] == [expected]


@pytest.mark.parametrize("state", list(STATES))
def test_profiles_list_agrees_with_registry(home, state):
    """Roster preview identity == click identity (#92040, PR #92129).

    profiles.list's canonical_session and session.list's title lookup must
    name the same registry row — never the newest session.
    """
    build, expected = STATES[state]
    build(home)

    canonical = _canonical_via_profiles()
    if expected is None:
        assert canonical is None
    else:
        assert canonical["id"] == expected
        assert canonical["root_title"] == CANON


# ---------------------------------------------------------------------------
# Property 2 — NEVER-MINT: resolution is a pure read in every state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", list(STATES))
def test_resolution_never_mints(home, state):
    """Session count is unchanged by any number of resolutions (#90705, #92692).

    Fail-closed contract pinned from code (methods_session.py title-lookup
    path and methods_profiles._canonical_session_row): when NO canonical row
    exists the resolvers return []/None — they REFUSE rather than create.
    Creation is an explicit separate path (desktop adopt-before-mint), never
    a side effect of lookup.
    """
    build, _ = STATES[state]
    build(home)

    before = _count_sessions(home)
    for _ in range(3):
        _resolve()
        _canonical_via_profiles()
    assert _count_sessions(home) == before


def test_absent_registry_resolves_empty_not_minted(home):
    """The #90705 shape exactly: null canonical → resolver must NOT mint."""
    _state_empty(home)
    assert _resolve() == []
    assert _canonical_via_profiles() is None
    assert _count_sessions(home) == 0


# ---------------------------------------------------------------------------
# Property 3 — UNIQUENESS: at most one canonical row per profile, always
# ---------------------------------------------------------------------------


def test_title_registry_rejects_second_canonical(home):
    """SessionDB refuses a second "Bot Chat" title (ValueError, not silence).

    #90732's fork loop started because set_session_title silently dropped
    conflicting titles; the current CAS raises, which is what lets creators
    adopt instead of minting.
    """
    db = _db(home)
    _add_session(db, "forever", title=CANON, ts=1000)
    db.create_session("pretender", "cli")
    with pytest.raises(ValueError):
        db.set_session_title("pretender", CANON)
    db.close()
    assert _canonical_rows(home) == ["forever"]


def _adopt_or_mint(home, sid, results, idx):
    """The adopt-before-mint creation algebra plugin.js runs, re-enacted in
    Python against the real SessionDB: lookup → create+claim → on conflict
    re-lookup and ADOPT. This is the #92692 racing-resolvers shape."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        row = db.get_session_by_title(CANON)
        if row:
            results[idx] = row["id"]
            return
        db.create_session(sid, "cli")
        try:
            db.set_session_title(sid, CANON)
            results[idx] = sid
        except ValueError:
            # Lost the race: adopt the winner, never fork.
            row = db.get_session_by_title(CANON)
            results[idx] = row["id"] if row else None
    finally:
        db.close()


@pytest.mark.parametrize("seq", [
    ["resolve", "resolve", "resolve"],
    ["create", "resolve", "create", "resolve"],
    ["resolve", "create", "resolve", "create", "resolve"],
])
def test_any_operation_sequence_leaves_at_most_one_canonical(home, seq):
    """After ANY sequence of resolve/create ops: ≤1 canonical row (#92692)."""
    _state_empty(home)
    results = {}
    for i, op in enumerate(seq):
        if op == "resolve":
            _resolve()
        else:
            _adopt_or_mint(home, f"mint-{i}", results, i)
    assert len(_canonical_rows(home)) <= 1
    creators = [v for k, v in results.items()]
    assert len(set(creators)) <= 1  # every creator converged on one identity


def test_two_resolvers_racing_cold_profile_yield_one_canonical(home):
    """#92692: two resolvers racing on a cold profile mint at most ONE chat.

    The DB-level guarantee pinned: the title-uniqueness CAS serializes the
    claim; the loser gets ValueError and adopts. Post-state: exactly one
    canonical row, and both racers hold the SAME id.
    """
    _state_empty(home)
    results = {}
    threads = [
        threading.Thread(target=_adopt_or_mint, args=(home, f"racer-{i}", results, i))
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    rows = _canonical_rows(home)
    assert len(rows) == 1
    assert set(results.values()) == {rows[0]}


# ---------------------------------------------------------------------------
# Property 4 — RECREATION: a recreated same-name identity is FRESH (#90005)
# ---------------------------------------------------------------------------


def test_recreated_same_name_identity_gets_fresh_session(home):
    """Disband + recreate under the same name must NOT resurrect old rows.

    #90005's surface (group ROOM member sessions) lives desktop-side in
    plugin.js ($groupChats) and is not importable here; what the fix
    (PR #91082) relies on at this layer is that identity follows the CURRENT
    registry row: once the old row's title is released and a new session
    claims the name, resolution returns ONLY the new session — the retired
    row is never adopted again.
    """
    db = _db(home)
    _add_session(db, "gen1", title=CANON, ts=1000, text="first life")
    db.close()
    assert [r["id"] for r in _resolve()] == ["gen1"]

    # Disband: release the name (rename away), then recreate under same name.
    db = _db(home)
    db.set_session_title("gen1", "Bot Chat (disbanded)")
    _add_session(db, "gen2", title=CANON, ts=2000, text="second life")
    db.close()

    resolved = _resolve()
    assert [r["id"] for r in resolved] == ["gen2"]
    assert _canonical_via_profiles()["id"] == "gen2"
    assert len(_canonical_rows(home)) == 1
