"""Private exact-wire document capture for native Vault operations.

Never discovers targets outside the owned page's fresh frame tree. No secret
values are read here. Frame origin is live document origin, not parent origin.
"""
from __future__ import annotations

from tools.browser_supervisor import _schedule

import concurrent.futures
from agent.async_utils import safe_schedule_threadsafe


class DocumentRefused(RuntimeError):
    pass


class OwnedWire:
    def __init__(self, supervisor):
        from tools.browser_vault_tool import _owned_vault, _owned_vault_lease
        self.supervisor = supervisor
        pinned = _owned_vault.get()
        self.owns_binding = pinned is None
        if pinned is not None and pinned[0] is not supervisor:
            raise DocumentRefused('owned supervisor changed')
        from hermes_constants import get_hermes_home
        from tools.browser_vault_tool import _bot_desktop_browser_session
        admission = _owned_vault_lease.get()
        self.profile_home, self.lease_epoch = admission or (str(get_hermes_home()), None)
        if admission is None and _bot_desktop_browser_session(supervisor.task_id):
            from tools.bot_desktop import lease
            self.lease_epoch = lease.assert_agent_may_act(self.profile_home).epoch
        self.binding = pinned[1] if pinned is not None else supervisor.bind_owned_vault()
        _, self.browser, self.target, self.root_session, self.ws = self.binding
        self.attached = []
        try:
            self.check_profile()
            self.check()
        except Exception:
            self.close()
            raise

    def check_profile(self):
        # Called on the native tool thread (not the supervisor event-loop
        # thread, whose contextvars do not inherit the calling profile).
        from hermes_constants import get_hermes_home
        from pathlib import Path
        if Path(self.binding[0][0]).resolve().parent != get_hermes_home().resolve():
            raise DocumentRefused('calling profile changed')

    def check(self):
        if self.lease_epoch is not None:
            from tools.bot_desktop import lease
            admitted = lease.assert_agent_may_act(self.profile_home)
            if admitted.epoch != self.lease_epoch:
                raise DocumentRefused('desktop lease changed')
        try:
            self.supervisor.validate_owned_vault(self.binding)
        except Exception:
            raise DocumentRefused('owned target changed or unavailable') from None

    def request(self, method, params=None, *, session=None):
        self.check_profile()
        self.check()
        s = self.supervisor
        try:
            result = _schedule(s._cdp(method, params or {}, session_id=session or self.root_session,
                                     timeout=8, expected_ws=self.ws, before_send=self.check),
                               s._loop, timeout=9)
        except Exception:
            raise DocumentRefused('document command refused') from None
        if 'error' in result:
            raise DocumentRefused('document command refused')
        return result.get('result', {})

    def frames(self):
        tree = self.request('Page.getFrameTree')['frameTree']
        out = []
        def visit(node, depth=0):
            if depth > 8 or len(out) >= 32:
                raise DocumentRefused('frame discovery limit')
            out.append(node['frame']['id'])
            for child in node.get('childFrames', []):
                visit(child, depth + 1)
        visit(tree)
        # Chromium can omit out-of-process children from Page.getFrameTree.
        # Read frame IDs from actual frame-owner nodes in THIS captured root
        # document; never substitute a browser-wide Target.getTargets scan.
        root = self.request('DOM.getDocument', {'depth': 0})['root']['nodeId']
        owners = self.request('DOM.querySelectorAll', {'nodeId': root, 'selector': 'iframe,frame'})['nodeIds']
        if len(owners) + len(out) > 32:
            raise DocumentRefused('frame discovery limit')
        for owner in owners:
            node = self.request('DOM.describeNode', {'nodeId': owner, 'depth': 0})['node']
            frame = node.get('frameId')
            if frame and frame not in out:
                out.append(frame)
        return out

    def attach_frame(self, frame):
        # Retain acquisition after the caller's deadline. A late attach reply
        # belongs to us and must be detached on the same wire, never guessed.
        self.check()
        s = self.supervisor
        future = safe_schedule_threadsafe(s._cdp('Target.attachToTarget',
            {'targetId': frame, 'flatten': True}, session_id=self.root_session,
            timeout=16, expected_ws=self.ws, before_send=self.check), s._loop)
        if future is None:
            raise DocumentRefused('frame acquisition unavailable')
        try:
            response = future.result(timeout=9)
        except concurrent.futures.TimeoutError:
            def late(reply):
                try:
                    sid = reply.result()['result']['sessionId']
                    self.release(sid)
                except Exception:
                    pass
            future.add_done_callback(late)
            raise DocumentRefused('frame acquisition pending') from None
        sid = response.get('result', {}).get('sessionId')
        if not sid:
            raise DocumentRefused('frame acquisition refused')
        self.attached.append(sid)
        return sid

    def document(self, frame):
        if frame not in self.frames():
            raise DocumentRefused('foreign frame')
        sid = self.root_session
        try:
            world = self.request('Page.createIsolatedWorld', {
                'frameId': frame, 'worldName': 'hermes-vault-private'})
            if not isinstance(world.get('executionContextId'), int) or world['executionContextId'] <= 0:
                raise DocumentRefused('frame requires its own transport')
        except DocumentRefused:
            # OOPIF target IDs are frame IDs. Attach only a frame proved under
            # this exact owned root, never scan arbitrary iframe targets.
            info = self.request('Target.getTargetInfo', {'targetId': frame})['targetInfo']
            if info.get('targetId') != frame or info.get('type') != 'iframe':
                raise DocumentRefused('frame transport unavailable')
            sid = self.attach_frame(frame)
            world = self.request('Page.createIsolatedWorld', {
                'frameId': frame, 'worldName': 'hermes-vault-private'}, session=sid)
        return OwnedDocument(self, frame, sid, world['executionContextId'])

    def release(self, sid):
        def same_wire():
            if self.supervisor._ws is not self.ws:
                raise DocumentRefused('captured wire changed')
        s = self.supervisor
        future = safe_schedule_threadsafe(s._cdp('Target.detachFromTarget',
            {'sessionId': sid}, timeout=3, expected_ws=self.ws, before_send=same_wire), s._loop)
        if future is not None:
            def consumed(done):
                try:
                    done.result()
                except Exception:
                    pass
            future.add_done_callback(consumed)

    def close(self):
        # Only release acquired sessions on their original connection.
        for sid in self.attached:
            self.release(sid)
        self.attached.clear()
        if self.owns_binding:
            self.supervisor.release_owned_vault(self.binding)
            self.owns_binding = False


class OwnedDocument:
    def __init__(self, wire, frame, session, context):
        self.wire, self.frame, self.session, self.context = wire, frame, session, context
        self.origin = self.evaluate('window.location.origin')

    def evaluate(self, expression):
        if self.frame not in self.wire.frames():
            raise DocumentRefused('document detached')
        result = self.wire.request('Runtime.evaluate', {
            'expression': expression, 'contextId': self.context,
            'returnByValue': True, 'awaitPromise': False}, session=self.session)
        if result.get('exceptionDetails'):
            # A script exception can embed values. Only a fixed reason escapes.
            raise DocumentRefused('document evaluation failed')
        return result.get('result', {}).get('value')

    def live_origin(self):
        from agent.vault_store import normalize_origin
        origin = self.evaluate('window.location.origin')
        if not isinstance(origin, str) or normalize_origin(origin) != origin:
            raise DocumentRefused('unsupported document origin')
        if origin != self.origin:
            raise DocumentRefused('document origin changed')
        return origin


def password_documents(wire):
    docs = []
    for frame in wire.frames():
        doc = wire.document(frame)
        if doc.evaluate("!!document.querySelector('input[type=password]')"):
            doc.live_origin()
            docs.append(doc)
    return docs
