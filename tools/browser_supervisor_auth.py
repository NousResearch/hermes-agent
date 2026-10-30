"""Vault-only native HTTP Basic authentication over the existing supervised socket.

No secrets are retained on the supervisor. A separate flattened session scopes Fetch
interception to one selected page and is always disabled/detached before returning.
"""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit


def _origin(url):
    from agent.vault_store import normalize_origin
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username is not None or parsed.password is not None:
            return None
        return normalize_origin(url)
    except (ValueError, TypeError):
        return None


def _challenge_allowed(params, origin, frame_id, requests):
    challenge = params.get("authChallenge") or {}
    return (challenge.get("source") == "Server"
            and str(challenge.get("scheme", "")).lower() == "basic"
            and _origin(challenge.get("origin", "")) == origin
            and _origin((params.get("request") or {}).get("url", "")) == origin
            and params.get("frameId") == frame_id
            and params.get("resourceType") == "Document"
            and params.get("requestId") in requests)


def _refused(reason):
    return {"success": False, "error_type": reason,
            "error": "Native HTTP authentication refused or did not complete; no automatic retry."}

class _NativeCleanupFailed(RuntimeError):
    """Interception could not be removed; this supervisor is poisoned."""



class NativeAuthMixin:
    def authenticate_vault_basic(self, allowed_origins, resolve_credentials, authorize, *, timeout=10.0):
        """Internal vault broker entry; resolver runs only after a validated challenge.

        The resolver captures the caller's profile scope via the scheduled coroutine,
        not a reader callback running in the supervisor thread's launch context.
        """
        from tools.browser_supervisor import _schedule
        from hermes_constants import hermes_home_key
        if self._vault_owner_home != hermes_home_key():
            return _refused("native_auth_profile_mismatch")
        if self._loop is None or not self._loop.is_running():
            return _refused("supervisor_required")
        try:
            return _schedule(self._vault_basic(allowed_origins, resolve_credentials, timeout, authorize),
                             self._loop, timeout=timeout + 8)
        except _NativeCleanupFailed:
            return _refused("native_auth_cleanup_failed")
        except Exception:
            # Transport errors can contain peer-supplied data. Never return/log them.
            return _refused("native_auth_failed")

    async def _vault_basic(self, allowed, resolve, timeout, authorize):
        # Serialize per supervisor: concurrent fills must not race interception/navigation.
        if getattr(self, "_vault_auth_poisoned", False):
            return _refused("native_auth_cleanup_failed")
        if getattr(self, "_vault_auth_busy", False):
            return _refused("native_auth_busy")
        self._vault_auth_busy = True
        sid = None
        try:
            async with asyncio.timeout(timeout):
                targets = (await self._cdp("Target.getTargets")).get("result", {}).get("targetInfos", [])
                candidates = [t for t in targets if t.get("type") == "page" and _origin(t.get("url", "")) in allowed]
                if len(candidates) != 1:
                    return _refused("native_auth_target_ambiguous" if candidates else "native_auth_target_missing")
                target = candidates[0]
                origin = _origin(target["url"])
                attached = await self._cdp("Target.attachToTarget", {"targetId": target["targetId"], "flatten": True})
                sid = attached["result"]["sessionId"]
                queue = asyncio.Queue()
                self._vault_auth_events[sid] = queue
                await self._cdp("Page.enable", session_id=sid)
                await self._cdp("Network.enable", session_id=sid)
                tree = (await self._cdp("Page.getFrameTree", session_id=sid))["result"]["frameTree"]
                frame_id = tree["frame"]["id"]
                await self._cdp("Fetch.enable", {"handleAuthRequests": True,
                                "patterns": [{"resourceType": "Document", "requestStage": "Request"}]}, session_id=sid)
                # Target URL selects only where to navigate. The new browser challenge,
                # not that URL or a chrome-error renderer, authorizes secret resolution.
                async def navigate():
                    authorize()
                    return await self._cdp("Page.navigate", {"url": target["url"]}, session_id=sid)

                navigation = asyncio.create_task(navigate())
                try:
                    result = await self._vault_auth_events_loop(queue, sid, origin, frame_id, resolve, authorize)
                    if result.get("success"):
                        nav = (await navigation).get("result", {})
                        if (nav.get("errorText") or nav.get("frameId") != frame_id
                                or not nav.get("loaderId") or nav["loaderId"] != result.pop("_loader_id", None)):
                            return _refused("native_auth_navigation_failed")
                        result["target_id"] = target["targetId"]
                    return result
                finally:
                    if not navigation.done():
                        navigation.cancel()
                    await asyncio.gather(navigation, return_exceptions=True)
        except TimeoutError:
            return _refused("native_auth_timeout")
        except Exception:
            return _refused("native_auth_failed")
        finally:
            clean = True
            if sid:
                clean = False
                for method, params, session in [
                    ("Fetch.disable", None, sid),
                    ("Target.detachFromTarget", {"sessionId": sid}, None),
                ]:
                    try:
                        # Bound send backpressure as well as the CDP response.
                        await asyncio.wait_for(self._cdp(method, params, session_id=session, timeout=2), timeout=2)
                        clean = True  # either disables interception or destroys its session
                    except Exception:
                        pass
                if clean:
                    self._vault_auth_events.pop(sid, None)
                else:
                    # Retain event ownership; never fall through to dialog passthrough.
                    self._vault_auth_poisoned = True
            self._vault_auth_busy = False
            if not clean:
                raise _NativeCleanupFailed("native auth interception remains owned but unavailable")

    async def _vault_auth_events_loop(self, queue, sid, origin, frame_id, resolve, authorize):
        requests = {}
        challenged_network = None
        attempted = False
        while True:
            method, params = await queue.get()
            request_id = params.get("requestId")
            if method == "Fetch.requestPaused":
                valid = (params.get("frameId") == frame_id and params.get("resourceType") == "Document"
                         and _origin((params.get("request") or {}).get("url", "")) == origin)
                if not valid:
                    await self._cdp("Fetch.failRequest", {"requestId": request_id, "errorReason": "BlockedByClient"}, session_id=sid)
                    return _refused("native_auth_origin_changed")
                requests[request_id] = params.get("networkId")
                await self._cdp("Fetch.continueRequest", {"requestId": request_id}, session_id=sid)
            elif method == "Fetch.authRequired":
                if attempted or not _challenge_allowed(params, origin, frame_id, requests):
                    await self._cdp("Fetch.continueWithAuth", {"requestId": request_id,
                                    "authChallengeResponse": {"response": "CancelAuth"}}, session_id=sid)
                    return _refused("native_auth_rejected")
                attempted = True
                challenged_network = requests[request_id]
                if not challenged_network:
                    return _refused("native_auth_request_unbound")
                from tools.browser_supervisor import _schedule
                active = True
                loop = asyncio.get_running_loop()

                async def send(username, password):
                    if not active or sid not in self._vault_auth_events:
                        raise TimeoutError("native authentication expired")
                    authorize()
                    await self._cdp("Fetch.continueWithAuth", {"requestId": request_id,
                                    "authChallengeResponse": {"response": "ProvideCredentials",
                                                              "username": username, "password": password}}, session_id=sid)

                def transmit(username, password):
                    return _schedule(send(username, password), loop, timeout=3)

                try:
                    # The broker's backend context (if any) spans resolution AND
                    # transmission on this one worker thread, never the reader loop.
                    await asyncio.to_thread(resolve, origin, transmit)
                finally:
                    active = False  # a timed-out resolver must never send late
            elif method == "Network.responseReceived":
                response = params.get("response") or {}
                if (attempted and params.get("frameId") == frame_id and params.get("type") == "Document"
                        and request_id == challenged_network and _origin(response.get("url", "")) == origin):
                    status = response.get("status", 0)
                    if 200 <= status < 300:
                        return {"success": True, "authentication": "http_basic", "kind": "login",
                                "origin": origin, "_loader_id": params.get("loaderId")}
                    if status >= 400:
                        return _refused("native_auth_rejected")
            elif method == "Network.loadingFailed" and request_id in requests.values():
                return _refused("native_auth_navigation_failed")
