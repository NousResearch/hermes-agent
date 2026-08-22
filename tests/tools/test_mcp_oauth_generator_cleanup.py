"""Focused safety tests for stranded MCP OAuth lock recovery."""

from __future__ import annotations

import asyncio

import anyio
import pytest

from tools.mcp_oauth_manager import _reclaim_stranded_auth_lock


@pytest.mark.asyncio
async def test_reclaim_frees_lock_owned_by_abandoned_flow():
    lock = anyio.Lock()
    owner_ready = asyncio.Event()
    owner_done = asyncio.Event()

    async def abandoned_owner() -> None:
        await lock.acquire()
        owner_ready.set()
        await owner_done.wait()

    owner_task = asyncio.create_task(abandoned_owner())
    with anyio.fail_after(3):
        await owner_ready.wait()
    prior_owner = getattr(lock, "_owner_task", None)

    assert prior_owner is owner_task
    assert _reclaim_stranded_auth_lock(lock, prior_owner, server_name="fixture")
    assert not lock.locked()

    owner_done.set()
    with anyio.fail_after(3):
        await owner_task


@pytest.mark.asyncio
async def test_reclaim_ignores_lock_owned_by_different_flow():
    lock = anyio.Lock()
    await lock.acquire()

    assert not _reclaim_stranded_auth_lock(lock, object(), server_name="fixture")
    assert lock.locked()
    lock.release()


@pytest.mark.asyncio
async def test_reclaim_is_noop_without_recorded_owner():
    lock = anyio.Lock()

    assert not _reclaim_stranded_auth_lock(lock, None, server_name="fixture")
    assert not lock.locked()


@pytest.mark.asyncio
async def test_reclaim_hands_lock_to_waiting_flow():
    lock = anyio.Lock()
    owner_ready = asyncio.Event()
    owner_done = asyncio.Event()
    waiter_acquired = asyncio.Event()

    async def abandoned_owner() -> None:
        await lock.acquire()
        owner_ready.set()
        await owner_done.wait()

    async def waiter() -> None:
        await lock.acquire()
        waiter_acquired.set()
        lock.release()

    owner_task = asyncio.create_task(abandoned_owner())
    with anyio.fail_after(3):
        await owner_ready.wait()
    prior_owner = getattr(lock, "_owner_task", None)
    waiter_task = asyncio.create_task(waiter())

    with anyio.fail_after(3):
        while not getattr(lock, "_waiters", ()):
            await anyio.lowlevel.checkpoint()

    assert _reclaim_stranded_auth_lock(lock, prior_owner, server_name="fixture")
    with anyio.fail_after(3):
        await waiter_acquired.wait()
        await waiter_task
    assert not lock.locked()

    owner_done.set()
    with anyio.fail_after(3):
        await owner_task
