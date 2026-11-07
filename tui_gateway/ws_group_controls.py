"""Bounded, ordered controls for each canonical group chat on a shared socket."""

import asyncio
import collections
import contextlib
from functools import partial


def _group_key(connection, request):
    if connection is None or not isinstance(request, dict):
        return None
    from gateway.hosted_rooms import _room_id
    from gateway.session_group_controls import GROUP_METHODS
    from hermes_state_runtime import RuntimeStoreError

    if request.get("method") not in GROUP_METHODS:
        return None
    params = request.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("room_id"), str):
        return None
    try:
        room_id = _room_id(params["room_id"])
        profile = params.get("profile")
        if isinstance(profile, str) and profile:
            connection = connection._sibling_for(profile) or connection
    except (ValueError, RuntimeStoreError):
        return None  # dispatch supplies the normal wire refusal
    return connection.actor.profile_id, room_id


class GroupControlLanes:
    def __init__(self, dispatch, send_failed, client_gone, *, limit):
        self.dispatch = dispatch
        self.send_failed = send_failed
        self.client_gone = client_gone
        self.room = asyncio.Semaphore(limit)
        self.frames = {}
        self.tasks = set()

    async def queue(self, connection, request, wait):
        key = _group_key(connection, request)
        if key is None:
            return False
        # Running and waiting frames share one connection-wide bound, including distinct rooms.
        await wait(self.room.acquire())
        if self.client_gone.is_set():
            self.room.release()
            return True
        if key in self.frames:
            self.frames[key].append(request)
        else:
            self.frames[key] = frames = collections.deque([request])
            task = asyncio.create_task(self._drain(key, frames))
            self.tasks.add(task)
            task.add_done_callback(partial(self._finished, key, frames))
        return True

    async def _drain(self, key, frames):
        from tui_gateway.ws import _SendFailed

        try:
            while frames and not self.client_gone.is_set() and not self.send_failed.done():
                await self.dispatch(frames[0])
                frames.popleft()
                self.room.release()
        except _SendFailed:
            if not self.send_failed.done():
                self.send_failed.set_result(None)
        finally:
            # Remove the queue before yielding back: a new frame must not join a finished task.
            self._drop(key, frames)

    def _drop(self, key, frames):
        if self.frames.get(key) is not frames:
            return
        del self.frames[key]
        for _ in frames:
            self.room.release()

    def _finished(self, key, frames, task):
        self.tasks.discard(task)
        # Cancellation before the first step never enters _drain's finally.
        if task.cancelled():
            self._drop(key, frames)
        with contextlib.suppress(asyncio.CancelledError):
            task.result()

    async def close(self):
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            for pending in self.tasks:
                pending.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
