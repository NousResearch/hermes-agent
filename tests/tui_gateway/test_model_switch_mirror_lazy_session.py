"""Regression for #122678: a typed /model on a pre-first-turn (lazy) session must
update the live session's route and repaint the Desktop picker.

Desktop's default pane state is a live session whose agent has not been built yet
(fresh chat, or a watch-window resume). ``slash.exec`` applies the switch in the
slash worker and reports success — but the in-process mirror was gated on a built
agent, so the live record never learned the new route: no pinned ``model_override``
and no ``session.info`` repaint. The UI kept billing-time truth invisible until an
app restart (restart "fixed" it because the cold resume restores the persisted row).

The mirror must therefore run for a lazy session too: pin the route the next build
consumes and emit ``session.info`` from the same canonical precedence
(``_live_session_identity``) that resume reports.
"""

import threading
import time
import types

import tui_gateway.server as server


def _lazy_session(**extra):
    """A live session with no built agent — Desktop's pre-first-turn record."""
    session = {
        "agent": None,
        "session_key": "lazy-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }
    return session


def _install_fake_switch(monkeypatch, *, model, provider):
    """Stand in for ``_apply_model_switch`` exactly as the agentless path behaves:
    resolve+pin, no live commit, no announce of its own."""
    calls = []

    def fake_apply(sid, session, arg, **kwargs):
        calls.append((sid, arg, kwargs))
        session["model_override"] = {"model": model, "provider": provider}
        return {"value": model, "warning": "", "scope": "session"}

    monkeypatch.setattr(server, "_apply_model_switch", fake_apply)
    return calls


def _capture_emits(monkeypatch):
    events = []
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload=None: events.append((kind, sid, payload)) or True)
    return events


def test_lazy_model_mirror_pins_route_and_announces(monkeypatch):
    """The core gap: with no built agent, the mirror used to skip entirely — the
    switch was reported as done while the live session kept the old route and no
    session.info reached the Desktop picker."""
    session = _lazy_session()
    server._sessions["lazy-mirror-sid"] = session
    calls = _install_fake_switch(monkeypatch, model="stealth/space-bunny-alpha", provider="openrouter")
    events = _capture_emits(monkeypatch)
    try:
        warning = server._mirror_slash_side_effects(
            "lazy-mirror-sid", session, "/model stealth/space-bunny-alpha --provider openrouter")
    finally:
        server._sessions.pop("lazy-mirror-sid", None)

    assert warning == ""
    assert calls, "typed /model must run the switch mirror even without a built agent"
    assert session["model_override"] == {"model": "stealth/space-bunny-alpha", "provider": "openrouter"}

    infos = [e for e in events if e[0] == "session.info"]
    assert infos, "the switch must announce session.info so the Desktop picker repaints without restart"
    _, sid, payload = infos[-1]
    assert sid == "lazy-mirror-sid"
    assert payload["model"] == "stealth/space-bunny-alpha"
    assert payload["provider"] == "openrouter"


def test_session_info_without_agent_reports_pinned_override():
    """_session_info's route precedence mirrors _live_session_identity: pending >
    mirror > agent > override. With no agent and no mirror, the override must be
    the answer — reporting "" here would repaint the picker blank instead of switched."""
    session = {
        "session_key": "lazy-session-key",
        "history": [],
        "model_override": {"model": "stealth/space-bunny-alpha", "provider": "openrouter"},
    }

    info = server._session_info(None, session)

    assert info["model"] == "stealth/space-bunny-alpha"
    assert info["provider"] == "openrouter"

    # No override at all: the old empty report stands (a truly undecided draft).
    session.pop("model_override")
    assert server._session_info(None, session)["model"] == ""


def test_warm_model_mirror_does_not_double_announce(monkeypatch):
    """With a built agent the commit announces through _commit_agent_switch; the
    mirror must not emit a second session.info for the same switch."""
    session = _lazy_session(agent=types.SimpleNamespace(model="old/model", provider="openrouter", tools=[]))
    server._sessions["warm-mirror-sid"] = session
    calls = _install_fake_switch(monkeypatch, model="new/model", provider="openrouter")
    events = _capture_emits(monkeypatch)
    try:
        warning = server._mirror_slash_side_effects("warm-mirror-sid", session, "/model new/model")
    finally:
        server._sessions.pop("warm-mirror-sid", None)

    assert warning == ""
    assert calls
    assert not [e for e in events if e[0] == "session.info"], (
        "the agent path's own commit emit is the announcement; a second one churns the client")


def test_bare_model_never_reaches_the_switch_mirror(monkeypatch):
    """/model with no argument is the read-only display path; it must not pin or
    announce anything (the live formatter answers it before the mirror runs)."""
    session = _lazy_session()
    calls = _install_fake_switch(monkeypatch, model="surprise", provider="none")
    events = _capture_emits(monkeypatch)

    warning = server._mirror_slash_side_effects("bare-sid", session, "/model ")

    assert warning == ""
    assert not calls
    assert not events


def test_prewarm_build_in_flight_switch_wins_the_race(monkeypatch):
    """``session.create`` prewarms a build that announces its own model on
    completion. A /model landing mid-build must not pin under the announce — the
    late announce repaints the picker back to the old route after the switch
    response said success (an inverted #122678). The mirror waits for the in-flight
    build and commits through the live-agent path, whose announce is the switch's
    own and lands last. Without the wait the last event is the build's old model."""
    session = _lazy_session()
    ready = session["agent_ready"] = threading.Event()
    session["agent_build_started"] = True  # the create prewarm is underway
    server._sessions["prewarm-sid"] = session

    def fake_build():
        time.sleep(0.4)  # the build resolves its route BEFORE the switch arrives
        agent = types.SimpleNamespace(model="model-one", provider="custom:laby", tools=[])
        session["agent"] = agent
        ready.set()
        server._emit("session.info", "prewarm-sid", server._session_info(agent, session))

    threading.Thread(target=fake_build, daemon=True).start()

    def apply(sid, sess, arg, **kwargs):
        agent = sess.get("agent")
        if agent is not None:  # warm commit: swap in place, announce (commit's job)
            agent.model, agent.provider = "model-two", "custom:laby"
            server._emit("session.info", sid, server._session_info(agent, sess))
        else:                  # lazy path: pin for the next build
            sess["model_override"] = {"model": "model-two", "provider": "custom:laby"}
        return {"value": "model-two", "warning": "", "scope": "session"}

    monkeypatch.setattr(server, "_apply_model_switch", apply)
    events = _capture_emits(monkeypatch)
    try:
        warning = server._mirror_slash_side_effects("prewarm-sid", session, "/model model-two")
    finally:
        server._sessions.pop("prewarm-sid", None)

    assert warning == ""
    assert session["agent"].model == "model-two", "the switch must land on the built agent"
    announced = [(p or {}).get("model") for kind, _sid, p in events if kind == "session.info"]
    assert announced[-1] == "model-two", (
        f"the LAST announce must be the switched route, saw {announced} — "
        "a build landing after the pin re-paints the old model")


def test_lazy_watch_window_never_parks_on_an_unset_build(monkeypatch):
    """A lazy watch-window resume never prewarms (``_start_agent_build`` bails while
    the child runs): its record holds an unset ``agent_ready`` FOREVER. The wait must
    key on the build actually having started, or every /model on a spectated session
    parks for the full 30s ceiling."""
    session = _lazy_session()
    session["agent_ready"] = threading.Event()  # set at record creation, build never starts
    server._sessions["watch-sid"] = session
    calls = _install_fake_switch(monkeypatch, model="model-two", provider="laby")
    events = _capture_emits(monkeypatch)
    t0 = time.monotonic()
    try:
        warning = server._mirror_slash_side_effects("watch-sid", session, "/model model-two")
    finally:
        server._sessions.pop("watch-sid", None)

    assert warning == ""
    assert calls and session["model_override"], "pins the route for the mirror"
    assert [e for e in events if e[0] == "session.info"], "and announces it"
    # Without the started-guard this parks the full 30s ceiling; the budget only
    # has to tell that apart from the first-call import cost of _session_info (~1s).
    assert time.monotonic() - t0 < 10.0, "must not wait on a build that never starts"
