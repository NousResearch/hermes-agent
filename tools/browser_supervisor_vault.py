"""Private, single-use native vault widget objects bound to a CDP session."""

from dataclasses import dataclass, field
import secrets
import time


@dataclass(frozen=True)
class MaskedPasswordProof:
    owner: object = field(repr=False)
    session_id: str = field(repr=False)
    context_id: int = field(repr=False)
    object_id: str = field(repr=False)


class VaultWidgetMixin:
    def prepare_masked_password(self, expected_origin, timeout=10):
        """Prove nonsecret behavior in a fresh isolated world; retain the object privately."""
        from agent.vault_masked_password import build_masked_password_probe
        from tools.browser_supervisor import _schedule

        with self._state_lock:
            session = self._page_session_id if self._active else None
        if not session or not self._loop:
            return {"ok": False, "error_type": "supervisor_required"}
        request = {"abandoned": False, "proof": None}
        deadline = int((time.time() + timeout) * 1000)

        async def prepare():
            with self._state_lock:
                if request["abandoned"]:
                    return {"ok": False, "error_type": "widget_preparation_abandoned"}
            tree = await self._cdp(
                "Page.getFrameTree", session_id=session, timeout=timeout
            )
            frame = tree["result"]["frameTree"]["frame"]["id"]
            world = await self._cdp(
                "Page.createIsolatedWorld",
                {
                    "frameId": frame,
                    "worldName": "hermes-vault-" + secrets.token_hex(12),
                },
                session_id=session,
                timeout=timeout,
            )
            context = world["result"]["executionContextId"]
            response = await self._cdp(
                "Runtime.evaluate",
                {
                    "expression": build_masked_password_probe(
                        expected_origin, deadline
                    ),
                    "contextId": context,
                    "returnByValue": False,
                    "awaitPromise": True,
                    "userGesture": True,
                },
                session_id=session,
                timeout=timeout,
            )
            payload = response.get("result", {})
            result = payload.get("result", {})
            if payload.get("exceptionDetails") or not result.get("objectId"):
                return {"ok": False, "error_type": "unsupported_password_widget"}
            proof = MaskedPasswordProof(self, session, context, result["objectId"])
            with self._state_lock:
                abandoned = request["abandoned"]
                if not abandoned:
                    self._vault_widget_proofs.add(proof)
                    request["proof"] = proof
            if abandoned:
                await self._dispose_vault_object(proof, timeout)
                return {"ok": False, "error_type": "widget_preparation_abandoned"}
            return {"ok": True, "proof": proof}

        try:
            return _schedule(prepare(), self._loop, timeout=timeout + 1)
        except Exception:
            with self._state_lock:
                request["abandoned"] = True
                proof = request["proof"]
            if proof is not None:
                self.dispose_masked_password(proof, timeout=timeout)
            return {"ok": False, "error_type": "widget_context_unavailable"}

    def check_masked_password(self, proof, timeout=10):
        return self._call_masked_password(proof, "check", timeout=timeout)

    def fill_masked_password(self, proof, password, timeout=10):
        """Private CDP argument delivery, never an expression/argv or model argument."""
        return self._call_masked_password(
            proof, "fill", password=password, timeout=timeout
        )

    def dispose_masked_password(self, proof, timeout=10):
        return self._call_masked_password(proof, "dispose", timeout=timeout)

    async def _dispose_vault_object(self, proof, timeout):
        try:
            await self._cdp(
                "Runtime.callFunctionOn",
                {
                    "objectId": proof.object_id,
                    "functionDeclaration": "function(){return this.dispose();}",
                    "returnByValue": True,
                },
                session_id=proof.session_id,
                timeout=timeout,
            )
        finally:
            await self._cdp(
                "Runtime.releaseObject",
                {"objectId": proof.object_id},
                session_id=proof.session_id,
                timeout=timeout,
            )

    def _call_masked_password(self, proof, operation, password=None, timeout=10):
        from tools.browser_supervisor import _schedule

        with self._state_lock:
            valid = (
                isinstance(proof, MaskedPasswordProof)
                and proof.owner is self
                and proof in self._vault_widget_proofs
                and (
                    operation == "dispose"
                    or (self._active and self._page_session_id == proof.session_id)
                )
            )
            if valid and operation in {"fill", "dispose"}:
                self._vault_widget_proofs.remove(proof)
        if not valid or not self._loop:
            return {"ok": False, "error_type": "widget_context_changed"}

        async def call():
            # focus_page uses the same admission lock. Recheck on the loop,
            # not only before scheduling across threads.
            async with self._page_action_lock:
                with self._state_lock:
                    if operation != "dispose" and (
                        not self._active or self._page_session_id != proof.session_id
                    ):
                        return {"ok": False, "error_type": "widget_context_changed"}
                arguments = [{"value": password}] if operation == "fill" else []
                response = await self._cdp(
                    "Runtime.callFunctionOn",
                    {
                        "objectId": proof.object_id,
                        "functionDeclaration": {
                            "check": "function(){return this.check();}",
                            "fill": "function(value){return this.fill(value);}",
                            "dispose": "function(){return this.dispose();}",
                        }[operation],
                        "arguments": arguments,
                        "returnByValue": True,
                        "userGesture": True,
                    },
                    session_id=proof.session_id,
                    timeout=timeout,
                )
            payload = response.get("result", {})
            if payload.get("exceptionDetails"):
                return {"ok": False, "error_type": "widget_execution_failed"}
            return {"ok": True, "result": payload.get("result", {}).get("value", {})}

        try:
            return _schedule(call(), self._loop, timeout=timeout + 1)
        except Exception:
            # Exception descriptions from site handlers may echo the password.
            return {"ok": False, "error_type": "widget_context_lost"}
        finally:
            if operation in {"fill", "dispose"}:
                try:
                    _schedule(
                        self._dispose_vault_object(proof, timeout),
                        self._loop,
                        timeout=timeout + 1,
                    )
                except Exception:
                    pass  # A destroyed context already releases its remote objects.
