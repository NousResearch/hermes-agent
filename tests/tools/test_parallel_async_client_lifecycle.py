"""Regression coverage for Parallel's loop-affine async client lifecycle."""

import asyncio
import logging
import sys
import threading
import types
from types import SimpleNamespace

from tools.daemon_pool import DaemonThreadPoolExecutor


def test_concurrent_extract_closes_each_client_on_its_owner_loop(monkeypatch):
    """Concurrent workers must never publish an async client process-wide.

    A constructor barrier forces the old cache implementation's three workers
    to observe an empty slot before any client can be published. Two clients
    then lose the publication race and become cyclic garbage after their
    worker threads exit, reproducing the lifecycle that surfaced as
    ``RuntimeError('Event loop is closed')`` in prompt_toolkit.
    """
    from model_tools import _run_async
    from plugins.web.parallel import provider as parallel_provider
    import tools.web_tools as web_tools

    constructor_barrier = threading.Barrier(3, timeout=15)
    instances = []

    class FakeBeta:
        def __init__(self, client):
            self.client = client

        async def extract(self, *, urls, full_content):
            assert full_content is True
            self.client.use_loop_id = id(asyncio.get_running_loop())
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        url=urls[0],
                        title="Example",
                        full_content="content",
                        excerpts=[],
                    )
                ],
                errors=[],
            )

    class FakeAsyncParallel:
        def __init__(self, *, api_key):
            assert api_key == "parallel-test-key"
            self.beta = FakeBeta(self)
            self.closed = False
            self.use_loop_id = None
            self.close_loop_id = None
            instances.append(self)
            constructor_barrier.wait()

        async def close(self):
            self.close_loop_id = id(asyncio.get_running_loop())
            self.closed = True

    fake_parallel = types.ModuleType("parallel")
    fake_parallel.AsyncParallel = FakeAsyncParallel
    monkeypatch.setitem(sys.modules, "parallel", fake_parallel)
    monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
    monkeypatch.setattr(
        parallel_provider,
        "_ensure_parallel_sdk_installed",
        lambda: None,
    )
    # Preserve compatibility with callers/tests that still create this legacy
    # attribute dynamically. The provider must neither read nor populate it.
    monkeypatch.setattr(web_tools, "_async_parallel_client", None, raising=False)

    provider = parallel_provider.ParallelWebSearchProvider()

    def extract(index):
        return _run_async(provider.extract([f"https://example.test/{index}"]))

    with DaemonThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(extract, range(3)))

    assert len(instances) == 3
    assert all(client.closed for client in instances)
    assert all(client.close_loop_id == client.use_loop_id for client in instances)
    assert web_tools._async_parallel_client is None
    assert [result[0]["content"] for result in results] == ["content"] * 3


def test_close_failure_does_not_discard_a_successful_extraction(monkeypatch, caplog):
    """A teardown error must not rewrite fetched content into an error result.

    ``AsyncParallel.close()`` funnels into ``httpx.aclose()`` ->
    ``transport.aclose()``, which can raise while the pool is being drained.
    That happens *after* the content is already in hand, so the extraction
    must still succeed; the cleanup failure is masked from the caller but
    logged at warning level so a regressed ownership fix stays visible.
    """
    from plugins.web.parallel import provider as parallel_provider

    class FakeBeta:
        def __init__(self, client):
            self.client = client

        async def extract(self, *, urls, full_content):
            assert full_content is True
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        url=urls[0],
                        title="Example",
                        full_content="content",
                        excerpts=[],
                    )
                ],
                errors=[],
            )

    class ExplodingOnCloseParallel:
        def __init__(self, *, api_key):
            self.beta = FakeBeta(self)
            self.close_calls = 0
            instances.append(self)

        async def close(self):
            self.close_calls += 1
            raise RuntimeError("Event loop is closed")

    instances = []
    fake_parallel = types.ModuleType("parallel")
    fake_parallel.AsyncParallel = ExplodingOnCloseParallel
    monkeypatch.setitem(sys.modules, "parallel", fake_parallel)
    monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
    monkeypatch.setattr(
        parallel_provider,
        "_ensure_parallel_sdk_installed",
        lambda: None,
    )

    provider = parallel_provider.ParallelWebSearchProvider()
    with caplog.at_level(logging.WARNING, logger=parallel_provider.__name__):
        results = asyncio.run(provider.extract(["https://example.test/1"]))

    assert results[0]["content"] == "content"
    assert "error" not in results[0]
    # Pin this test's own trigger: deleting the close entirely must not pass.
    assert [client.close_calls for client in instances] == [1]
    close_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "close failed" in record.message
    ]
    assert len(close_warnings) == 1


def test_extraction_failure_outranks_a_close_failure(monkeypatch):
    """When both the request and the cleanup fail, report the request error.

    The primary exception must win: a secondary cleanup failure must not
    overwrite the reason the extraction actually failed.
    """
    from plugins.web.parallel import provider as parallel_provider

    class FakeBeta:
        async def extract(self, *, urls, full_content):
            raise RuntimeError("extract failed")

    class ExplodingBothWaysParallel:
        def __init__(self, *, api_key):
            self.beta = FakeBeta()

        async def close(self):
            raise RuntimeError("close failed")

    fake_parallel = types.ModuleType("parallel")
    fake_parallel.AsyncParallel = ExplodingBothWaysParallel
    monkeypatch.setitem(sys.modules, "parallel", fake_parallel)
    monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
    monkeypatch.setattr(
        parallel_provider,
        "_ensure_parallel_sdk_installed",
        lambda: None,
    )

    provider = parallel_provider.ParallelWebSearchProvider()
    results = asyncio.run(provider.extract(["https://example.test/1"]))

    assert "extract failed" in results[0]["error"]
    assert "close failed" not in results[0]["error"]
