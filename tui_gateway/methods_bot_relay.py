"""Bot-relay JSON-RPC handlers — the gateway side of cross-connection A2A.

Connections ARE the peer set: every gateway the Desktop holds a socket to
(local, remote URL, SSH, Hermes Cloud, docker) must be able to find every
other connection's agents and message them. The Desktop is the relay — it
owns every socket — and these four methods are the door it uses on EACH
connected gateway:

- ``bot_relay.roster.sync``  — Desktop pushes the union roster of agents on
  the OTHER connections into this gateway's ``bot_relay/roster.json``, so
  ``message_agent`` can resolve cross-connection targets and Bot Chat
  prompts list them (capability-epoch refresh picks up changes).
- ``bot_relay.outbox.drain`` — Desktop collects envelopes queued here by
  ``message_agent`` for targets on other connections.
- ``bot_relay.deliver``      — Desktop hands an envelope to the TARGET
  gateway; this method runs the same one-turn Bot Chat delivery local DMs
  use and returns the reply text.
- ``bot_relay.reply``        — Desktop writes the reply (or a delivery
  error) back on the SENDER gateway; the waiter spawned at send time picks
  it up and wakes the sending agent via the standard completion path.

Storage/validation plumbing lives in ``tools/bot_relay.py``. Handlers are
rebound onto server.py's globals at install time (see method_ctx.py) and may
reference server module globals (``_ok``, ``_err``) not imported here.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("bot_relay.roster.sync")
def _(rid, params: dict) -> dict:
    """Replace this gateway's view of agents on OTHER connections.

    Params: ``agents`` — list of rows ``{profile, handle, connection_id,
    connection_label?, title?, description?}``. Rows failing validation are
    dropped, not fatal. Result: ``{count}`` (accepted rows).
    """
    try:
        import os
        from pathlib import Path

        from tools.bot_relay import write_remote_roster

        home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
        root = home.parent.parent if home.parent.name == "profiles" else home
        count = write_remote_roster(root, params.get("agents"))
        return _ok(rid, {"count": count})
    except Exception as e:
        return _err(rid, 5090, str(e))


@method("bot_relay.outbox.drain")
def _(rid, params: dict) -> dict:
    """Claim every pending cross-connection envelope queued on this gateway.

    Claimed envelopes move to ``claimed/`` atomically, so concurrent drains
    (two Desktop windows) can't double-deliver. Result: ``{envelopes}``.
    """
    try:
        import os
        from pathlib import Path

        from tools.bot_relay import claim_pending_envelopes

        home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
        root = home.parent.parent if home.parent.name == "profiles" else home
        return _ok(rid, {"envelopes": claim_pending_envelopes(root)})
    except Exception as e:
        return _err(rid, 5091, str(e))


@method("bot_relay.deliver")
def _(rid, params: dict) -> dict:
    """Deliver a relayed DM into a profile's Bot Chat ON THIS GATEWAY.

    Params: ``profile`` (target on this install), ``message`` (already
    attribution-prefixed by the sender gateway). Runs the same one-turn
    ``hermes -p <profile> chat -c "Bot Chat"`` transport local DMs use and
    returns ``{reply}`` — the target agent's response text. Blocking by
    design (the Desktop calls it from its relay worker, off any UI path;
    the RPC pool keeps it off the WS reader thread).
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    profile = str(params.get("profile") or "").strip()
    message = str(params.get("message") or "").strip()
    if not profile or not message:
        return _err(rid, 4090, "profile and message required")
    try:
        from tools.bot_mode_dm import MESSAGE_MAX_CHARS
        from tools.bot_relay import acquire_turn_lock, local_delivery_command

        if len(message) > MESSAGE_MAX_CHARS + 200:  # + attribution headroom
            return _err(rid, 4091, "message too long")

        home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
        root = home.parent.parent if home.parent.name == "profiles" else home
        known = {"default"}
        profiles_dir = root / "profiles"
        if profiles_dir.is_dir():
            known.update(c.name for c in profiles_dir.iterdir() if c.is_dir())
        resolved = "default" if profile.lower() == "hermes" else profile
        if resolved not in known:
            return _err(rid, 4092, f"no profile '{profile}' on this gateway")

        fd, tmp = tempfile.mkstemp(prefix="hermes-relay-dm-", suffix=".txt", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(message)
            # Per-profile turn lock (#93091): serialize with any other
            # delivery turn into this profile (relay or local message_agent).
            # The lock covers only the turn execution window. Worst-case
            # handler hold is lock wait (bot_mode.turn_wait_seconds, default
            # 120s) + the 600s turn timeout below — doubled when the retry
            # policy grants one bounded re-run — so clients calling
            # bot_relay.deliver must tolerate ~1320s before assuming failure.
            with acquire_turn_lock(root, resolved):
                proc = subprocess.run(
                    local_delivery_command(resolved, tmp),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    # Retry session policy (#93091 item 5): transient classes
                    # re-run the SAME session once; context_overflow also
                    # re-runs the same session — the retried turn's pre-API
                    # compaction pass (agent/conversation_loop.py) compacts
                    # the over-threshold Bot Chat transcript first, which is
                    # the sanctioned compression lever (no fresh session is
                    # ever minted). Auth/quota/config classes never retry.
                    from tools.bot_failure_reasons import (
                        RETRY_NONE,
                        classify_agent_error,
                        retry_action,
                    )

                    first_detail = (proc.stderr or proc.stdout or "").strip()[-500:]
                    if retry_action(classify_agent_error(first_detail)) != RETRY_NONE:
                        proc = subprocess.run(
                            local_delivery_command(resolved, tmp),
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if proc.returncode != 0:
            from tools.bot_failure_reasons import classify_agent_error

            detail = (proc.stderr or proc.stdout or "").strip()[-500:]
            return _err(
                rid,
                5092,
                f"delivery turn failed: {detail or proc.returncode}",
                data={"reason": classify_agent_error(detail)},
            )
        return _ok(rid, {"reply": (proc.stdout or "").strip()})
    except subprocess.TimeoutExpired:
        return _err(rid, 5093, "delivery turn timed out")
    except Exception as e:
        # 'target_busy' extends the #93091 item-1 structured refusal enum.
        if getattr(e, "reason", "") == "target_busy":
            return _err(rid, 5096, str(e))
        return _err(rid, 5094, str(e))


@method("bot_relay.reply")
def _(rid, params: dict) -> dict:
    """Write a relayed reply (or delivery error) for a sender-side waiter.

    Params: ``id`` (envelope id), ``reply`` and/or ``error``, optional
    ``reason`` (typed failure code, see ``tools.bot_failure_reasons``).
    """
    envelope_id = str(params.get("id") or "").strip()
    if not envelope_id:
        return _err(rid, 4093, "id required")
    try:
        import os
        from pathlib import Path

        from tools.bot_relay import write_reply

        home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
        root = home.parent.parent if home.parent.name == "profiles" else home
        write_reply(
            root,
            envelope_id,
            reply=str(params.get("reply") or ""),
            error=str(params.get("error") or ""),
            reason=str(params.get("reason") or ""),
        )
        return _ok(rid, {"ok": True})
    except ValueError as e:
        return _err(rid, 4094, str(e))
    except Exception as e:
        return _err(rid, 5095, str(e))


def register(server) -> None:
    _registry.install(server)
