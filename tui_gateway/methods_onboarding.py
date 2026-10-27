import threading
from pathlib import Path

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

# onboarding.state and onboarding.set_run are unscoped on purpose: onboarding.run and the first-chat
# flag live in the root profile's config.yaml whichever profile this backend was launched under.
@method("onboarding.state")
def _(rid, params: dict) -> dict:
    return _ok(rid, _run_state(assume_fresh=bool(params.get("assume_fresh"))))


@method("onboarding.set_run")
def _(rid, params: dict) -> dict:
    from hermes_cli.onboarding_run import set_run
    if not isinstance(params.get("run"), bool):
        return _err(rid, 4002, "onboarding.set_run requires a boolean 'run'")
    try:
        set_run(params["run"], mark_profile_offered=bool(params.get("mark_profile_offered")))
    except Exception as e:
        logger.exception("onboarding.set_run failed")
        return _err(rid, 5077, str(e))
    return _ok(rid, _run_state())


def _run_state(*, assume_fresh: bool = False) -> dict:
    """``run``: the root config says so, else this OS user never used Hermes (or the unpackaged
    desktop's ``--fresh-install`` says to assume that). ``eligible``: this is the desktop's local
    backend and something can carry the first chat (the welcome trial or a configured provider);
    remote backends answer at once and run onboarding only from Settings. On the local backend the
    answer comes from the boot computation (``onboarding_due``); ``--fresh-install`` asks live."""
    from hermes_cli import onboarding_due
    from hermes_cli.anon_auth import on_local_desktop_backend
    from hermes_cli.onboarding_run import should_run

    if not on_local_desktop_backend():
        return {"run": should_run(assume_fresh=assume_fresh), "eligible": False, **_active_run()}
    due = _due_now(assume_fresh=True) if assume_fresh else onboarding_due.read(_due_now, stamp=_due_stamp)
    return {**due, **_active_run()}


def _due_now(*, assume_fresh: bool = False) -> dict[str, bool]:
    from hermes_cli.onboarding_run import should_run
    from hermes_cli.welcome_trial import inference_available

    # Read in the default profile, where onboarding runs, not the launch profile.
    home = _profile_home("default")
    with _session_profile_runtime_scope({"profile_home": str(home) if home else None}):
        eligible = inference_available(assume_fresh=assume_fresh)
    return {"run": should_run(assume_fresh=assume_fresh), "eligible": eligible}


def _due_stamp() -> tuple:
    """The files the due answer reads, by their config-cache signature: the root config
    (``onboarding.run``), the default profile's config, ``.env`` and ``auth.json`` (what carries a
    first chat), the shared Nous store (another profile's sign-in or connector guest) and the managed
    layer's ``config.yaml`` and ``.env`` (an admin's onboarding or provider policy)."""
    from hermes_cli import managed_scope
    from hermes_constants import get_default_hermes_root
    from utils import file_signature

    home = Path(_profile_home("default") or get_default_hermes_root())
    managed = managed_scope.get_managed_dir()
    paths = (get_default_hermes_root() / "config.yaml", home / "config.yaml", home / ".env", home / "auth.json",
             *_shared_nous_store(), *((managed / "config.yaml", managed / ".env") if managed else ()))
    return tuple(file_signature(st) if (st := _stat(path)) else None for path in paths)


def _shared_nous_store() -> tuple[Path, ...]:
    """The shared Nous store as ``_read_shared_nous_state`` finds it: none when the test seat belt trips."""
    from hermes_cli.auth_nous import _nous_shared_store_path
    try:
        return (_nous_shared_store_path(),)
    except RuntimeError:
        return ()


def _stat(path):
    try:
        return path.stat()
    except OSError:
        return None


def start_due_boot() -> None:
    """Work out the local Desktop's due answer off the RPC path, after the setup bootstrap has recorded
    a fresh install (``begin_local_desktop_boot``)."""
    from agent.memory_provider import spawn_context_thread
    from hermes_cli import free_tier_bootstrap, onboarding_due

    finish = onboarding_due.begin_boot(_due_stamp)

    def compute() -> dict[str, bool]:
        free_tier_bootstrap.wait_for_record()
        return _due_now()

    # finish() marks the boot done even when compute raises, so no reader waits on a dead thread.
    spawn_context_thread(finish, name="onboarding-due", args=(compute,)).start()


_local_desktop_boot_lock = threading.Lock()
_local_desktop_boot_started = False


def begin_local_desktop_boot() -> None:
    """An accepted local Desktop connection to this process: record a fresh install, settle an
    interrupted claim, the setup record and the due answer, once, as work for that Desktop. This
    process outlives app launches, so a later connection retries the install steps a failed boot
    left unsettled."""
    global _local_desktop_boot_started
    from agent.memory_provider import spawn_context_thread
    from hermes_cli import free_tier_bootstrap
    from hermes_cli.process_identity import serving_local_desktop
    from tui_gateway.launch_profile_policy import launch_profile_scope_if_multiplexed

    with _local_desktop_boot_lock:
        first, _local_desktop_boot_started = not _local_desktop_boot_started, True
    # The record and the latch are the launch profile's; the connecting profile may be another one.
    with launch_profile_scope_if_multiplexed(), serving_local_desktop(True):  # threads copy this context
        if not first:
            spawn_context_thread(free_tier_bootstrap.resettle_after_failed_boot, name="free-tier-resettle").start()
            return
        free_tier_bootstrap.start_background_bootstrap()
        start_due_boot()


# ── first-run onboarding: one chat in the default profile, driven by setup_beat ───────────────

# Onboarding always runs in the default profile, whichever profile this backend was launched under.
_ONBOARDING_PROFILE = {"profile": "default"}
# Two windows pressing start at once must not both find no run and open two onboarding chats.
_onboarding_start_lock = threading.Lock()


def _active_run() -> dict:
    """The onboarding chat in progress and its run id, both None when no run is open."""
    from agent.onboarding_record import active_session_id, load
    with _profile_db(_ONBOARDING_PROFILE) as db:
        session_id = None if db is None else active_session_id(db)
        return {"session_id": session_id, "run_id": (load(db, session_id) or {}).get("run_id") if session_id else None}


class _NoModelForTheRun(Exception):
    """The claim was blocked and nothing else can carry the run's first chat."""


def _claim_before_run(scope: dict, *, assume_fresh: bool):
    """Claim the welcome trial with ``onboarding.run: true`` already on disk, so a crash after the claim
    makes this machine read as used still opens onboarding next launch. The run's chat comes after:
    a refused claim (error 5079) leaves no guest and no chat, and the install still due. A blocked claim
    (another install took the trial since the due check) with no trial or provider to run on is error
    5080, with no chat either: the desktop lands on provider setup and sign-in, not a chat that fails."""
    from hermes_cli import anon_auth
    from hermes_cli.onboarding_run import set_run
    from hermes_cli.welcome_trial import claim_desktop_free_tier, provider_configured
    if not anon_auth.on_local_desktop_backend():  # gateway/session_onboarding.py::_claim_before_run
        raise _NoModelForTheRun("First-run onboarding starts only in the Desktop app on this machine.")
    set_run(True)
    with _session_profile_runtime_scope(scope):
        claimed = claim_desktop_free_tier(assume_fresh=assume_fresh)
        if claimed is None and not (anon_auth.free_inference_active() or provider_configured()):
            raise _NoModelForTheRun("The welcome trial is not available here, and no provider is set up.")
    return claimed


@method("onboarding.start")
def _(rid, params: dict) -> dict:
    from agent.onboarding_facts import read_facts
    from agent.onboarding_flow import End, Facts
    from agent.onboarding_record import advance, load, open_session_ids, start
    from agent.onboarding_text import CHAT_TITLE, opening_prompt
    from hermes_cli.auth_constants import AuthError
    from hermes_cli import free_tier_bootstrap
    # The app's once-per-launch start: closing the app mid-run does not count, so that run's chat goes
    # and the run starts over. A second window or a reconnect sends no flag and gets the run back.
    fresh_launch = bool(params.get("fresh_launch"))
    # Held through the opening submit: two windows pressing start at once get one chat and one opening.
    with _onboarding_start_lock:
        claimed = None
        try:
            with _profile_db(_ONBOARDING_PROFILE, writer=True) as db:
                if db is None:
                    return _db_unavailable_error(rid, code=5000)
                open_ids = open_session_ids(db)
                stored = None if params.get("rerun") or fresh_launch else next(iter(open_ids), None)
                created = stored is None
                if created:
                    # The tours setting is the default profile's, whichever profile launched the backend.
                    home = _profile_home("default")
                    scope = {"profile_home": str(home) if home else None}
                    with _session_profile_runtime_scope(scope, hydrate_secrets=False):
                        facts = read_facts(facts_source=params.get("facts_source") or "local", locale=params.get("locale"))
                    if not params.get("rerun"):  # a rerun never claims: it runs on the configured model
                        claimed = _claim_before_run(scope, assume_fresh=bool(params.get("assume_fresh")))
                    stored = db.create_session(new_session_id(), "desktop")
                    # The record exists before the agent is built, so the build folds the onboarding toolset in.
                    start(db, stored, facts)
                else:
                    facts = Facts.from_stored(load(db, stored)["facts"])
                run_id = load(db, stored).get("run_id") or ""
                # One open run at a time. The kept run is open before the others end, so the install
                # flag never closes in between; finishing it closes the flag.
                replaced = [sid for sid in open_ids if sid != stored]
                if fresh_launch:
                    _discard_unfinished_runs(db, replaced)
                else:
                    for old in replaced:
                        advance(db, old, End("replaced"))
                if created:
                    # After the discard, so a run started over keeps the title its deleted chat had.
                    db.set_session_title(stored, db.get_next_title_in_lineage(CHAT_TITLE))
                # A chat with no messages never got its opening (a restart between start and the first turn).
                opening = None if (db.get_session(stored) or {}).get("message_count") else opening_prompt(facts)
        except AuthError as e:
            # The trial claim failed (portal down, rate limited): no guest and no chat, so the next
            # launch is still a first install and tries again.
            logger.info("onboarding.start: welcome trial not claimed: %s", e)
            return _err(rid, 5079, str(e))
        except _NoModelForTheRun as e:
            logger.info("onboarding.start: no chat opened: %s", e)
            return _err(rid, 5080, str(e))
        except Exception as e:
            logger.exception("onboarding.start failed")
            return _err(rid, 5078, str(e))
        if claimed is not None:
            # Outside the default-profile scope: setup.status serves the launch profile's record.
            free_tier_bootstrap.refresh_record()
        _withdraw_setup_cards([] if fresh_launch else replaced)
        resumed = _methods["session.resume"](rid, {**_ONBOARDING_PROFILE, "session_id": stored, "source": "desktop"})
        if "error" in resumed:
            return resumed
        sid = resumed["result"]["session_id"]
        if opening is not None:
            # The opening rules ride a hidden user turn; its prelude plays the greeting and the name card.
            submitted = _methods["prompt.submit"](rid, {"session_id": sid, "text": opening, "display_kind": "hidden"})
            if "error" in submitted:
                return submitted
    return _ok(rid, {"session_id": sid, "stored_session_id": stored, "resumed": not created, "run_id": run_id})


@method("onboarding.skip")
def _(rid, params: dict) -> dict:
    from agent.onboarding_flow import End
    from agent.onboarding_record import advance, load
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    stored = getattr(agent, "session_id", None) or session["session_key"]
    with _session_db(session) as db:
        if load(db, stored) is None:
            return _ok(rid, {"skipped": False})
        step = advance(db, stored, End("skipped"))
    # The card that waits returns to setup_beat, which reads the skipped record.
    _cancel_session_requests(params["session_id"], "resolved", method="setup_beat")
    return _ok(rid, {"skipped": bool(step.result.get("skipped_all"))})


def _live_session_ids(stored_ids: list[str]) -> list[str]:
    """The live sessions here that show one of *stored_ids*."""
    if not stored_ids:
        return []
    with _sessions_lock:
        return [sid for sid, sess in _sessions.items()
                if (getattr(sess.get("agent"), "session_id", None) or sess.get("session_key")) in stored_ids]


def _withdraw_setup_cards(stored_ids: list[str]) -> None:
    """Close the card still waiting in a chat whose run just ended; its answer would be refused."""
    for sid in _live_session_ids(stored_ids):
        _cancel_session_requests(sid, "resolved", method="setup_beat")


def _discard_unfinished_runs(db, stored_ids: list[str]) -> None:
    """A new launch's start: the runs the app was closed on never count as done. Each chat is closed if
    still live here (a backend that outlived the app), then deleted the way ``session.delete`` deletes
    it, which ends its run (``deleted``) and tells every window. A chat the delete refuses keeps its
    messages, and its run ends as ``replaced`` so it is never resumed."""
    from agent.onboarding_flow import End
    from agent.onboarding_record import advance, delete_chat
    for sid in _live_session_ids(stored_ids):
        _close_session_by_id(sid, end_reason="onboarding_restart")
    sessions_dir = Path(_profile_home("default") or get_hermes_home()) / "sessions"
    for old in stored_ids:
        try:
            deleted = delete_chat(db, old, lambda old=old: db.delete_session(
                old, sessions_dir=sessions_dir, exclude_active_write_guards=True))
        except Exception:
            logger.exception("onboarding.start could not delete unfinished chat %s", old)
            deleted = False
        if not deleted:
            advance(db, old, End("replaced"))


def _setup_beat_request(sid: str, payload: dict) -> dict | None:
    from agent.onboarding_flow import CARD_TIMEOUT_S
    from tui_gateway import server_requests
    return server_requests.send("setup_beat", sid, dict(payload), timeout=CARD_TIMEOUT_S)


def _stage_onboarding_turn_note(sid: str, agent, history_empty: bool) -> None:
    """Before a user turn in the onboarding chat (``agent.onboarding_session.onboarding_turn_note``)."""
    from agent.onboarding_session import append_turn_note, onboarding_turn_note
    from tui_gateway import server_requests
    append_turn_note(agent, onboarding_turn_note(
        agent, history_empty=history_empty, card_waiting=server_requests.pending_kind(sid) == "setup_beat"))


def register(server) -> None:
    from agent.onboarding_record import on_ended

    bind_module(globals(), server, skip=("_",))
    # Every window, not one session: the window showing the run may be subscribed to another chat.
    on_ended(lambda ended: server._broadcast_global_event("onboarding.ended", ended))
