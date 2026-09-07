"""Internal Wisdom activity for native, Desktop and embedded TUI sessions."""

from __future__ import annotations

import time
import logging

from hermes_wisdom.consent import ConsentActor
from hermes_wisdom.delivery import DeliveryReceipt
from hermes_wisdom.mediation import WisdomMediation, session_runtime
from hermes_wisdom.mediation_view import advice_view


def note_activity(session: dict, *, profile_scope) -> None:
    from hermes_wisdom.mediation import delivery_mode
    from hermes_wisdom.store import WisdomStore
    from hermes_wisdom.mediation_store import MediationStore

    with profile_scope(session):
        if delivery_mode() != "agent":
            return
        store = WisdomStore()
        org = store.active_org_id()
        key = str(session.get("session_key") or "")
        if not org or not key:
            return
        MediationStore(store).register_session(
            org,
            session_key=key,
            session_id=key,
            platform="local",
            actor_id="local-user",
            private=True,
            available=False,
            user_activity=True,
            activity_at=session["_wisdom_user_activity"],
            address=ConsentActor(key, "local", "local-user", f"local:{key}").address,
        )


def poll(session: dict, *, emit, profile_scope, connected=lambda: True) -> None:
    from tools.approval import get_pending_gateway_approval
    from tools.clarify_gateway import has_pending
    from hermes_wisdom.service import WisdomService

    key = str(session.get("session_key") or "")
    if (
        not key
        or not session.get("_wisdom_user_activity")
        or session.get("agent") is None
    ):
        return
    if time.time() - session["_wisdom_user_activity"] > 600:
        return
    if not connected() or get_pending_gateway_approval(key) or has_pending(key):
        return
    with session["history_lock"]:
        if (
            session.get("running")
            or session.get("_closing")
            or session.get("_finalized")
        ):
            return
        session["running"] = True
        generation = int(session.get("_queued_prompt_generation", 0))
        runtime = session_runtime(session["agent"])
        history = list(session.get("history") or [])
    try:
        with profile_scope(session):
            service = WisdomService()
            service.require_setup()
            mediation = WisdomMediation(service)
            org = service.store.active_org_id()
            actor = ConsentActor(key, "local", "local-user", f"local:{key}")
            activity = session.get("_wisdom_user_activity")
            mediation.queue.register_session(
                org,
                session_key=key,
                session_id=key,
                platform="local",
                actor_id=actor.actor_id,
                private=True,
                available=True,
                user_activity=activity != session.get("_wisdom_registered_activity"),
                address=actor.address,
                activity_at=activity,
            )
            session["_wisdom_registered_activity"] = activity
            refreshed = False
            if mediation.queue.claim_refresh(org):
                try:
                    service.check(apply_automatic=False)
                    mediation.ingest()
                    refreshed = True
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Wisdom feed refresh deferred (%s)", type(exc).__name__
                    )
            items = mediation.prepare(org, actor, runtime=runtime, history=history)
            if not items:
                if (
                    refreshed
                    and not service.store.feed_events(unseen_only=True)
                    and not service.store.local_events(
                        kind="wisdom.candidate", session_id=key
                    )
                    and not any(
                        item["state"] in {"pending", "applying"}
                        for item in mediation.consent.pending(org)
                    )
                ):
                    emit("notification.clear", {"key": "wisdom.advice"})
                return
            if (
                not connected()
                or session.get("_closing")
                or session.get("_finalized")
                or generation != int(session.get("_queued_prompt_generation", 0))
            ):
                return
            introduction = not mediation.queue.introduced(org)
            selected = mediation.begin_delivery(org, items)
            if not selected:
                return
            if (
                not connected()
                or session.get("_closing")
                or session.get("_finalized")
                or generation != int(session.get("_queued_prompt_generation", 0))
                or not mediation.delivery_ready(org, selected)
                or get_pending_gateway_approval(key)
                or has_pending(key)
            ):
                mediation.cancel_delivery(org, selected)
                return
            try:
                view = advice_view(selected, introduction=introduction)
                text = (
                    view.to_text()
                    + "\n\nOpen /wisdom inbox to review and use consent controls."
                )
            except Exception:
                mediation.cancel_delivery(org, selected)
                raise
            try:
                accepted = emit(
                    "notification.show",
                    {
                        "key": "wisdom.advice",
                        "id": "wisdom.advice",
                        "kind": "wisdom",
                        "level": "info",
                        "ttl_ms": None,
                        "text": text,
                    },
                )
                for item in selected:
                    job = item["assessment"]
                    if accepted is True:
                        mediation.queue.complete_delivery(
                            org,
                            job["id"],
                            job["lease_token"],
                            introduced=introduction,
                            receipt=DeliveryReceipt(
                                platform="local",
                                destination=actor.chat_id,
                                message_id="wisdom.advice",
                                acknowledgement="transport_accepted",
                            ),
                        )
                    else:
                        mediation.queue.uncertain_delivery(
                            org, job["id"], job["lease_token"]
                        )
            except BaseException:
                for item in selected:
                    job = item["assessment"]
                    try:
                        mediation.queue.uncertain_delivery(
                            org, job["id"], job["lease_token"]
                        )
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "Wisdom uncertain delivery awaits lease recovery"
                        )
                raise
    finally:
        with session["history_lock"]:
            if generation == int(session.get("_queued_prompt_generation", 0)):
                session["running"] = False
