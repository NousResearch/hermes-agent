"""Exercise real thread ownership through the synchronous async bridge and keyless entry points."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import threading

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path):
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(tmp_path)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(tmp_path / 'config.yaml', {'lsp': {'enabled': False}})
    try:
        yield
    finally:
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


@pytest.mark.parametrize('running_loop', [False, True])
def test_native_bridge_keeps_origin_and_restores_context(running_loop):
    from model_tools import _run_async
    from tools.interrupt import acting_for_tid, is_interrupted, set_interrupt

    owner = threading.get_ident()
    previous = acting_for_tid.get()

    async def observe():
        return await asyncio.to_thread(lambda: (acting_for_tid.get(), is_interrupted()))

    async def from_running_loop():
        return _run_async(observe())

    def invoke():
        return asyncio.run(from_running_loop()) if running_loop else _run_async(observe())

    assert invoke() == (owner, False)
    assert acting_for_tid.get() == previous
    try:
        set_interrupt(True, thread_id=owner)
        assert invoke() == (owner, True)

        def independent_session():
            independent = threading.get_ident()
            return independent, _run_async(observe())

        with ThreadPoolExecutor(max_workers=1) as pool:
            independent, observed = pool.submit(independent_session).result(timeout=30)
        assert independent != owner and observed == (independent, False)
        assert acting_for_tid.get() == previous
    finally:
        set_interrupt(False, thread_id=owner)


@pytest.mark.parametrize('operation', ['search', 'extract'])
def test_pre_cancelled_keyless_entry_returns_without_network_dispatch(operation):
    from model_tools import _run_async
    from plugins.web.keyless_mcp import extract_with_failover, search_with_failover
    from tools.interrupt import set_interrupt

    owner = threading.get_ident()

    async def request():
        if operation == 'search':
            return await asyncio.to_thread(search_with_failover, 'exa', 'NousResearch Hermes Agent official GitHub', 3)
        return await asyncio.to_thread(extract_with_failover, 'exa', ['https://github.com/NousResearch/hermes-agent'])

    async def from_running_loop():
        return _run_async(request())

    try:
        set_interrupt(True, thread_id=owner)
        result = asyncio.run(from_running_loop())
        if operation == 'search':
            assert result['success'] is False and result['error'] == 'Interrupted'
        else:
            assert len(result) == 1 and result[0]['error'] == 'Interrupted'
    finally:
        set_interrupt(False, thread_id=owner)


def test_cancellation_during_real_thread_handoff_is_observed():
    from model_tools import _run_async
    from tools.interrupt import acting_for_tid, is_interrupted, set_interrupt

    ready, resume = threading.Event(), threading.Event()
    owners = []

    def origin():
        owner = threading.get_ident()
        owners.append(owner)

        def awaited_work():
            ready.set()
            assert resume.wait(timeout=30)
            return acting_for_tid.get(), is_interrupted()

        async def tool():
            return await asyncio.to_thread(awaited_work)

        async def running_loop():
            return _run_async(tool())

        return asyncio.run(running_loop())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(origin)
        try:
            assert ready.wait(timeout=30)
            set_interrupt(True, thread_id=owners[0])
            resume.set()
            assert future.result(timeout=30) == (owners[0], True)
        finally:
            resume.set()
            if owners:
                set_interrupt(False, thread_id=owners[0])
