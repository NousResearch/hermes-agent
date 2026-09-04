"""Real SDK flows and token files, with deterministic local token responses."""
import asyncio
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs

import pytest

pytest.importorskip("mcp.client.auth.oauth2")


async def _provider(home):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    storage = HermesTokenStorage("shared", hermes_home=home)
    metadata = OAuthClientMetadata(redirect_uris=["http://localhost/callback"])
    if await storage.get_client_info() is None:
        await storage.set_client_info(OAuthClientInformationFull(
            client_id="test-client", redirect_uris=metadata.redirect_uris,
            token_endpoint_auth_method="none",
        ))

    async def unused(*args):
        raise AssertionError("Interactive authorization must not run")

    provider = HermesMCPOAuthProvider(
        server_name="shared", server_url="https://resource.example/mcp",
        client_metadata=metadata, storage=storage,
        redirect_handler=unused, callback_handler=unused,
    )
    provider.context.oauth_metadata = OAuthMetadata(
        issuer="https://auth.example", authorization_endpoint="https://auth.example/authorize",
        token_endpoint="https://auth.example/token", response_types_supported=["code"],
    )
    await provider._initialize()
    return provider, storage


async def _worker(home, label):
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    provider, _ = await _provider(home)
    print("ready", flush=True)
    await asyncio.to_thread(sys.stdin.readline)
    flow = provider.async_auth_flow(httpx.Request("POST", "https://resource.example/mcp"))
    try:
        request = await flow.__anext__()
        print(parse_qs(request.content.decode())["refresh_token"][0], flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        resource = await flow.asend(httpx.Response(200, request=request, json={
            "access_token": f"access-{label}", "token_type": "Bearer",
            "refresh_token": f"refresh-{label}", "expires_in": 3600,
        }))
        assert resource.headers["Authorization"] == f"Bearer access-{label}"
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_processes_refresh_serially_using_the_latest_persisted_token(tmp_path):
    from mcp.shared.auth import OAuthToken
    _, storage = await _provider(tmp_path)
    await storage.set_tokens(OAuthToken(
        access_token="expired", refresh_token="original", token_type="Bearer", expires_in=0,
    ))
    workers = []
    try:
        for label in ("first", "second"):
            worker = await asyncio.create_subprocess_exec(
                sys.executable, __file__, str(tmp_path), label,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": str(Path.cwd()), "HERMES_HOME": str(tmp_path)},
            )
            workers.append(worker)
            assert await asyncio.wait_for(worker.stdout.readline(), 15) == b"ready\n"
        first, second = workers
        first.stdin.write(b"go\n")
        await first.stdin.drain()
        assert await asyncio.wait_for(first.stdout.readline(), 15) == b"original\n"
        second.stdin.write(b"go\n")
        await second.stdin.drain()
        # Keep the first token response pending. No second refresh may be built.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(second.stdout.readline(), 2)
        first.stdin.write(b"finish\n")
        await first.stdin.drain()
        assert await asyncio.wait_for(second.stdout.readline(), 15) == b"refresh-first\n"
        second.stdin.write(b"finish\n")
        await second.stdin.drain()
        for worker in workers:
            assert await asyncio.wait_for(worker.wait(), 15) == 0, (await worker.stderr.read()).decode()
        assert (await storage.get_tokens()).refresh_token == "refresh-second"
    finally:
        for worker in workers:
            if worker.returncode is None:
                worker.kill()
            await worker.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["close", "cancel-waiter", "invalid-response"])
async def test_failed_or_cancelled_refresh_releases_the_file_lock(tmp_path, outcome):
    from mcp.shared.auth import OAuthToken
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    _, storage = await _provider(tmp_path)
    await storage.set_tokens(OAuthToken(
        access_token="expired", refresh_token="original", token_type="Bearer", expires_in=0,
    ))
    provider, _ = await _provider(tmp_path)
    flow = provider.async_auth_flow(httpx.Request("POST", "https://resource.example/mcp"))
    request = await flow.__anext__()
    try:
        if outcome == "cancel-waiter":
            waiter = asyncio.create_task(storage.acquire_refresh_lock())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        elif outcome == "invalid-response":
            await flow.asend(httpx.Response(200, request=request, content=b"invalid-json"))
    finally:
        await flow.aclose()
    handle = await storage.acquire_refresh_lock(timeout=2)
    storage.release_refresh_lock(handle)
    assert (await storage.get_tokens()).refresh_token == "original"


if __name__ == "__main__":
    asyncio.run(_worker(sys.argv[1], sys.argv[2]))
