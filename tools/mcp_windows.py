"""Reach Windows loopback MCP servers from WSL without exposing a LAN port.

The client still uses the configured URL (Host, TLS SNI and OAuth origin are
unchanged). Its exact-origin HTTP transport connects to a private Unix socket;
a Windows PowerShell child carries the TCP bytes over WSL interop. There is no
HTTP proxy, IP rewrite, firewall change, or third-party bridge installation.
"""

import asyncio
import base64
import contextlib
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from hermes_constants import is_wsl

logger = logging.getLogger("tools.mcp_tool")
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class InvalidMcpNetworkError(ValueError):
    """An explicit network target cannot be honored on this backend."""


def _loopback_endpoint(url: str) -> tuple[str, int, str] | None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOOPBACK:
        return None
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    if not 0 < port < 65536:
        raise ValueError("MCP loopback port must be between 1 and 65535")
    host = parsed.hostname
    authority = f"[{host}]" if ":" in host else host
    return host, port, f"{parsed.scheme}://{authority}:{port}"


def _windows_tunnel_command(host: str, port: int) -> list[str]:
    # Only validated loopback literals/names reach the script. Never interpolate
    # URLs, credentials, request bodies, or arbitrary config commands into it.
    if host not in _LOOPBACK or not 0 < port < 65536:
        raise ValueError("Windows MCP tunnel only accepts loopback endpoints")
    executable = shutil.which("powershell.exe")
    if not executable:
        fallback = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        if fallback.is_file():
            executable = str(fallback)
    if not executable:
        raise ConnectionError("Windows MCP bridge requires WSL Windows interop and powershell.exe")
    script = f"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$client = $null
try {{
    # Windows PowerShell's .NET Framework default TcpClient is IPv4-only.
    # Resolve localhost to fixed loopback literals, never an arbitrary DNS IP.
    $addresses = if ('{host}' -eq 'localhost') {{ @('127.0.0.1', '::1') }} else {{ @('{host}') }}
    foreach ($address in $addresses) {{
        $family = if ($address -eq '::1') {{ [System.Net.Sockets.AddressFamily]::InterNetworkV6 }} else {{ [System.Net.Sockets.AddressFamily]::InterNetwork }}
        $candidate = [System.Net.Sockets.TcpClient]::new($family)
        try {{
            $connect = $candidate.ConnectAsync($address, {port})
            if (-not $connect.Wait(10000)) {{ throw 'Windows loopback connection timed out' }}
            # PowerShell must not emit the Task's VoidTaskResult to stdout.
            [void]$connect.GetAwaiter().GetResult()
            $client = $candidate
            break
        }} catch {{
            $candidate.Dispose()
        }}
    }}
    if ($null -eq $client) {{ throw 'Windows loopback connection failed' }}
    $client.NoDelay = $true
    $stream = $client.GetStream()
    $inputStream = [Console]::OpenStandardInput()
    $outputStream = [Console]::OpenStandardOutput()
    $outputStream.WriteByte(0)
    $outputStream.Flush()
    $send = $inputStream.CopyToAsync($stream)
    $receive = $stream.CopyToAsync($outputStream)
    [void][System.Threading.Tasks.Task]::WhenAny($send, $receive).GetAwaiter().GetResult()
}} catch {{
    [Console]::Error.WriteLine('Could not connect to the Windows MCP loopback server')
    exit 1
}} finally {{
    if ($null -ne $client) {{ $client.Dispose() }}
}}
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


async def _copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while chunk := await reader.read(65536):
        writer.write(chunk)
        await writer.drain()


async def _close_child(process: asyncio.subprocess.Process) -> None:
    # EOF on stdin makes the Windows-side WhenAny finish and dispose its socket.
    # Prefer this to killing only WSL's interop shim and orphaning powershell.exe.
    if process.stdin is not None:
        process.stdin.close()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await asyncio.wait_for(process.wait(), timeout=2.0)


@dataclass
class WindowsMcpRoute:
    origin: str
    socket_path: str
    error: str | None = None

    def client_options(self, httpx, *, verify=True, cert=None) -> dict:
        # An exact-origin mount, NOT a global uds transport: redirects to other
        # origins must never send their credentials to the original local server.
        expected = _loopback_endpoint(self.origin)
        tunnel = httpx.AsyncHTTPTransport(uds=self.socket_path, verify=verify, cert=cert, trust_env=False)
        direct = httpx.AsyncHTTPTransport(verify=verify, cert=cert, trust_env=False)

        class ExactOriginTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                # HTTPX normalizes :80/:443 mount patterns to a wildcard port.
                # Guard the actual origin too, or a same-host redirect to a
                # DIFFERENT port would wrongly travel through this tunnel.
                target = tunnel if _loopback_endpoint(str(request.url)) == expected else direct
                return await target.handle_async_request(request)

            async def aclose(self):
                try:
                    await tunnel.aclose()
                finally:
                    await direct.aclose()

        return {"mounts": {self.origin: ExactOriginTransport()}}


@contextlib.asynccontextmanager
async def windows_loopback_route(url: str, *, connect_timeout: float = 10.0):
    """Own the private socket and all bridge children for one MCP connection."""
    endpoint = _loopback_endpoint(url)
    if endpoint is None:
        raise InvalidMcpNetworkError("network: windows requires an HTTP(S) loopback MCP URL")
    host, port, origin = endpoint
    command = _windows_tunnel_command(host, port)
    tasks: set[asyncio.Task] = set()
    children: set[asyncio.subprocess.Process] = set()
    closing = False
    # A byte-only helper does not need any provider credentials. WSL_INTEROP
    # is needed by the interop launcher even with appendWindowsPath disabled.
    from tools.mcp_tool_config import _SAFE_ENV_KEYS, _SAFE_ENV_KEYS_CASE_INSENSITIVE
    child_env = {key: value for key, value in os.environ.items()
                 if key in _SAFE_ENV_KEYS or key.upper() in _SAFE_ENV_KEYS_CASE_INSENSITIVE
                 or key in {"WSL_INTEROP", "WSL_DISTRO_NAME"}}
    # A short path avoids AF_UNIX path length limits with long profile homes.
    # TemporaryDirectory creates mode 0700; another local user cannot connect.
    with tempfile.TemporaryDirectory(prefix="hmcp-") as directory:
        route = WindowsMcpRoute(origin, os.path.join(directory, "bridge.sock"))

        async def forward(reader, writer):
            process = None
            connected = False
            pumps = []
            try:
                spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, env=child_env))
                try:
                    process = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    # Cancellation during exec must not lose ownership of the
                    # child that the OS may already have created.
                    process = await spawn
                    children.add(process)
                    raise
                children.add(process)
                # The child sends one readiness byte only after TCP connected.
                ready = await asyncio.wait_for(process.stdout.readexactly(1), timeout=connect_timeout)
                if ready != b"\0":
                    raise ConnectionError("Invalid Windows MCP bridge handshake")
                connected = True
                route.error = None
                pumps = [asyncio.create_task(_copy_stream(reader, process.stdin)),
                         asyncio.create_task(_copy_stream(process.stdout, writer))]
                done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError) as exc:
                if not connected:
                    route.error = (f"Cannot reach Windows MCP at {host}:{port}; start its HTTP/SSE server "
                                   "and check WSL Windows interop. No LAN port or firewall rule was changed.")
                logger.debug("Windows MCP bridge connection failed: %s", type(exc).__name__)
            finally:
                for task in pumps:
                    task.cancel()
                await asyncio.gather(*pumps, return_exceptions=True)
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()
                if process is not None:
                    await _close_child(process)
                    children.discard(process)

        def accept(reader, writer):
            if closing:
                writer.close()
                return
            task = asyncio.create_task(forward(reader, writer))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        server = await asyncio.start_unix_server(accept, path=route.socket_path)
        try:
            yield route
        except Exception as exc:
            if route.error:
                raise ConnectionError(route.error) from exc
            raise
        finally:
            closing = True
            server.close()
            await server.wait_closed()
            active = list(tasks)
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            # A handler can already be cleaning up when context exit cancels
            # it. Reap those children here too, after all spawning has stopped.
            await asyncio.gather(*(_close_child(child) for child in children))


async def _local_endpoint_available(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=1.0)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return True


def effective_mcp_network(config: Mapping[str, object]) -> str:
    """Return the authorized network namespace for one MCP config.

    ``network`` did not exist before the WSL bridge. Its omission therefore
    preserves the historical backend-local meaning of ``localhost``. Crossing
    into Windows requires an explicitly persisted ``auto`` or ``windows``. New
    setup surfaces that choose automatic routing write ``network: auto`` rather
    than retroactively widening legacy entries.
    """
    network = config.get("network", "local")
    if not isinstance(network, str) or network not in {"auto", "local", "windows"}:
        raise InvalidMcpNetworkError("MCP network must be auto, local, or windows")
    return network


def validate_mcp_network_config(config: dict) -> None:
    """Validate explicit routing intent without opening sockets or loading the SDK."""
    if "network" not in config:
        return
    network = effective_mcp_network(config)
    if not config.get("url"):
        raise InvalidMcpNetworkError("MCP network is only supported for HTTP/SSE servers, not stdio")
    # Environment placeholders are resolved by the runtime loader, which also
    # checks the final origin. Do not reject portable ${MCP_URL} configuration.
    url = config["url"]
    if network == "windows" and isinstance(url, str) and "${" not in url:
        try:
            endpoint = _loopback_endpoint(url)
        except ValueError:
            endpoint = None
        if endpoint is None:
            raise InvalidMcpNetworkError("network: windows requires an HTTP(S) loopback MCP URL")


@contextlib.asynccontextmanager
async def mcp_http_client(httpx, server_name: str, config: dict, **client_kwargs):
    """Route standalone MCP OAuth clients exactly like the resource transport.

    Each client owns its route. A cached OAuth provider must never retain a
    closed Unix socket or a route bound to another connection's event loop.
    Only the configured MCP origin is tunneled; external authorization servers
    keep their normal destination. This does not forward arbitrary redirects.
    """
    from tools.mcp_tool_errors import _resolve_client_cert

    verify = config.get("ssl_verify", True)
    cert = _resolve_client_cert(server_name, config)
    async with mcp_http_route(config["url"], network=effective_mcp_network(config)) as route:
        options = route.client_options(httpx, verify=verify, cert=cert) if route else {}
        async with httpx.AsyncClient(
            **client_kwargs, verify=verify, **({"cert": cert} if cert is not None else {}), **options
        ) as client:
            yield client


def validate_mcp_network(url: str, network: str = "local") -> None:
    """Reject unfulfillable explicit targets before any optional preflight."""
    if not isinstance(network, str) or network not in {"auto", "local", "windows"}:
        raise InvalidMcpNetworkError("MCP network must be auto, local, or windows")
    endpoint = _loopback_endpoint(url)
    if network == "windows" and endpoint is None:
        raise InvalidMcpNetworkError("network: windows requires an HTTP(S) loopback MCP URL")
    wsl = is_wsl()
    if network == "windows" and not wsl and sys.platform != "win32":
        raise InvalidMcpNetworkError("network: windows requires a WSL or native Windows backend")


@contextlib.asynccontextmanager
async def mcp_http_route(url: str, *, network: str = "local", connect_timeout: float = 10.0):
    """Route one explicitly qualified MCP endpoint.

    Omitted routing is ``local`` for legacy/fail-closed compatibility. Explicit
    ``auto`` tries WSL-local first and bridges only when it is unreachable;
    ``windows`` selects the Windows listener even
    when a WSL process uses the same port. Non-WSL auto/local routes are no-ops;
    an explicit Windows target is also a no-op on native Windows, never on an
    unrelated Linux/macOS/remote machine. Non-loopback URLs are never bridged.
    """
    validate_mcp_network(url, network)
    endpoint = _loopback_endpoint(url)
    wsl = is_wsl()
    if not wsl or endpoint is None or network == "local":
        yield None
        return
    host, port, _ = endpoint
    if network == "auto" and await _local_endpoint_available(host, port):
        yield None
        return
    logger.info("MCP %s:%s: using Windows loopback via WSL interop", host, port)
    async with windows_loopback_route(url, connect_timeout=connect_timeout) as route:
        yield route
