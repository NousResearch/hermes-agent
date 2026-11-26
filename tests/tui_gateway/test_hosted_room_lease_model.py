from __future__ import annotations

import itertools
import threading

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    FakeSessionRPC,
    RecordingTurnLocks,
    store_fixture,
)

store = store_fixture


def _schedules():
    actors = range(2)
    actions = tuple((kind, actor) for actor in actors for kind in ("commit", "publish"))
    for schedule in itertools.permutations(actions):
        positions = {action: index for index, action in enumerate(schedule)}
        if all(
            positions[("commit", actor)] < positions[("publish", actor)]
            for actor in actors
        ):
            yield schedule


def test_renewal_cache_is_monotonic_for_all_two_renewal_interleavings(
    store, monkeypatch
):
    real_renew = state.renew_lease
    schedules = list(_schedules())
    assert len(schedules) == 6
    for number, schedule in enumerate(schedules):
        room = f"lease-model-{number}"
        binding = HostedRoomBinding(
            room_id=room, gateway_id="gateway-a", authority_epoch=1
        )
        hosted_rooms.create_room(
            store,
            placement="profile",
            room_id=room,
            name="room",
            members=[{"profile": "ops", "handle": "ops"}],
            authority_gateway_id="gateway-a",
            now=0.0,
        )
        clock = [1.0]
        runtime = HostedRoomRuntime(
            store=store,
            placement="profile",
            rooms=[binding],
            rpc=FakeSessionRPC(),
            turn_lock=RecordingTurnLocks(),
            clock=lambda: clock[0],
            lease_ttl_seconds=1000,
            process_generation="model-process",
        )
        initial = runtime._ensure_lease(binding)
        starts = [threading.Event() for _ in range(2)]
        committed = [threading.Event() for _ in range(2)]
        resumes = [threading.Event() for _ in range(2)]
        published = [threading.Event() for _ in range(2)]
        results = [None] * 2
        errors = []
        actor_for_thread = {}

        def delayed_renew(*args, **kwargs):
            actor = actor_for_thread[threading.current_thread()]
            if not starts[actor].wait(5):
                raise TimeoutError("commit not scheduled")
            result = real_renew(*args, **kwargs)
            results[actor] = result
            committed[actor].set()
            if not resumes[actor].wait(5):
                raise TimeoutError("publication not scheduled")
            return result

        def renew(actor):
            try:
                actor_for_thread[threading.current_thread()] = actor
                runtime._renew_lease_if_needed(initial, force=True)
            except BaseException as error:
                errors.append(error)
            finally:
                published[actor].set()

        monkeypatch.setattr(state, "renew_lease", delayed_renew)
        threads = [threading.Thread(target=renew, args=(actor,)) for actor in range(2)]
        for thread in threads:
            thread.start()
        expected = initial.expires_at
        try:
            for kind, actor in schedule:
                if kind == "commit":
                    clock[0] += 1
                    starts[actor].set()
                    assert committed[actor].wait(5), (schedule, errors)
                else:
                    resumes[actor].set()
                    assert published[actor].wait(5), (schedule, errors)
                    assert not errors, (schedule, errors)
                    expected = max(expected, results[actor].expires_at)
                    observed = runtime._renew_lease_if_needed(initial)
                    assert observed.expires_at == expected, schedule
        finally:
            for event in starts + resumes:
                event.set()
            for thread in threads:
                thread.join(5)
            monkeypatch.setattr(state, "renew_lease", real_renew)
        assert not errors, (schedule, errors)
        assert all(not thread.is_alive() for thread in threads)
