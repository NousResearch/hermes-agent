"""In-process SSRF-filtering forward proxy for shelled-out fetchers (yt-dlp).

Why this exists
---------------
``video_analyze`` fetches non-direct video URLs (YouTube/X/Vimeo/TikTok) by
shelling out to ``yt-dlp``. A pre-flight ``is_safe_url`` check validates only the
*initial* URL; yt-dlp then performs its own fetches (page, player JS, manifest,
media segments) and follows redirects, any of which can target an internal /
cloud-metadata address that is never re-validated. That is a real SSRF hole.

This module stands up a tiny **in-process asyncio forward proxy** bound to
loopback. yt-dlp is pointed at it via ``--proxy http://127.0.0.1:<port>`` (plus
``HTTP_PROXY``/``HTTPS_PROXY`` env). Every connection yt-dlp makes — the initial
fetch, **every redirect**, and every sub-resource — arrives at the proxy, which:

  1. parses ``CONNECT host:port`` (HTTPS) or an absolute-form request (HTTP),
  2. resolves the host (once) via the event loop's non-blocking resolver,
  3. validates **every** resolved address (A + AAAA) against the single SSRF
     policy source ``url_safety.ip_is_blocked`` — if ANY is blocked, refuse,
  4. connects to a **validated IP literal** (never re-resolving — TOCTOU-safe),
  5. for CONNECT, pipes bytes without terminating TLS (no MITM, no certs).

The proxy blocks the always-metadata floor **and** private/RFC1918 addresses for
this egress (``block_private=True``), independent of the global
``allow_private_urls`` toggle: an attacker-controlled page URL has no legitimate
reason to redirect to the LAN.

It binds ``127.0.0.1`` on an ephemeral port, lives only for one ``async with``
block, and tracks + cancels every per-connection handler task on exit so no
sockets/fds leak across the fleet's high ``video_analyze`` call volume.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Optional, Set

from tools.url_safety import ip_is_blocked

logger = logging.getLogger(__name__)

# Bytes we shovel per pipe read. 64KiB balances throughput vs loop-time fairness.
_PIPE_CHUNK = 65536
# Cap the request line + headers we read before a CONNECT/absolute request so a
# malformed client can't make us buffer unboundedly.
_MAX_HEADER_BYTES = 16384


class SsrfFilteringProxy:
    """An async context manager yielding a loopback proxy URL.

    Usage::

        async with SsrfFilteringProxy(block_private=True) as proxy_url:
            # proxy_url == "http://127.0.0.1:<ephemeral>"
            ... run yt-dlp --proxy proxy_url ...
        # on exit: server closed, all handler tasks cancelled, no leaks

    ``connection_count`` records how many outbound connections were validated
    and opened (the traffic-observing signal the acceptance gate asserts).
    """

    def __init__(self, *, block_private: bool = True) -> None:
        self._block_private = block_private
        self._server: Optional[asyncio.AbstractServer] = None
        self._host = "127.0.0.1"
        self._port: Optional[int] = None
        self._tasks: Set[asyncio.Task] = set()
        # Number of successfully-validated outbound connections opened. The
        # traffic gate asserts this is > (initial + redirects) for a real
        # segmented download, i.e. segments actually traversed the proxy.
        self.connection_count = 0
        # Number of refused connections (blocked by policy or malformed).
        self.blocked_count = 0

    @property
    def proxy_url(self) -> str:
        if self._port is None:
            raise RuntimeError("proxy not started")
        return f"http://{self._host}:{self._port}"

    async def __aenter__(self) -> str:
        # start_server can raise (port exhaustion, permission) — let it
        # propagate so the caller fails CLOSED rather than running un-proxied.
        self._server = await asyncio.start_server(
            self._handle_client, self._host, 0
        )
        sock = self._server.sockets[0]
        self._port = sock.getsockname()[1]
        logger.debug("SSRF proxy listening on %s:%s", self._host, self._port)
        return self.proxy_url

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Close the listener, then cancel + await EVERY per-connection handler
        # task. asyncio.start_server does NOT cancel handler tasks on close, so
        # without this, orphaned CONNECT pipes leak sockets/fds every call.
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # pragma: no cover - defensive
                pass
        tasks = list(self._tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            # Give cancelled handlers a chance to unwind (close sockets) before
            # the loop tears down, so no "Task was destroyed but pending" warning
            # and no leaked fds.
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # asyncio.start_server invokes this coroutine as its own task; register
        # THAT task so __aexit__ can cancel any handler still piping when the
        # context manager tears down (the fd-leak / orphaned-pipe guard). On
        # Python <3.12, Server.wait_closed() does NOT drain handler tasks.
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        try:
            request_line = await self._read_request_line(reader)
            if request_line is None:
                await self._refuse(writer, "400 Bad Request", "malformed request")
                return

            method, target, _version = request_line

            if method == "CONNECT":
                # target == "host:port"
                host, port = self._split_hostport(target)
                if host is None or port is None:
                    await self._refuse(writer, "400 Bad Request", "bad CONNECT target")
                    return
                # Drain the remaining request headers (up to blank line).
                await self._drain_headers(reader)
                await self._handle_connect(host, port, reader, writer)
            else:
                # Absolute-form HTTP request:  GET http://host/path HTTP/1.1
                host, port, rest = self._split_absolute_url(target)
                if host is None or port is None or rest is None:
                    await self._refuse(writer, "400 Bad Request", "non-absolute HTTP request")
                    return
                await self._handle_http(
                    method, host, port, rest, request_line, reader, writer
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("SSRF proxy handler error: %s", exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _read_request_line(self, reader: asyncio.StreamReader):
        try:
            line = await reader.readuntil(b"\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            return None
        text = line.decode("latin-1", "replace").strip()
        parts = text.split(" ")
        if len(parts) != 3:
            return None
        return parts[0].upper(), parts[1], parts[2]

    async def _drain_headers(self, reader: asyncio.StreamReader) -> None:
        total = 0
        while True:
            try:
                line = await reader.readuntil(b"\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                return
            total += len(line)
            if line in (b"\r\n", b"\n", b""):
                return
            if total > _MAX_HEADER_BYTES:
                return

    def _split_hostport(self, target: str):
        if ":" not in target:
            return None, None
        # rsplit for IPv6-literal safety (…]:443).
        host, _, port_s = target.rpartition(":")
        host = host.strip("[]")
        try:
            port = int(port_s)
        except ValueError:
            return None, None
        if not host or not (0 < port < 65536):
            return None, None
        return host, port

    def _split_absolute_url(self, url: str):
        # Only proxy absolute http:// forms; https always arrives via CONNECT.
        if not url.lower().startswith("http://"):
            return None, None, None
        rest = url[len("http://"):]
        slash = rest.find("/")
        if slash == -1:
            authority, path = rest, "/"
        else:
            authority, path = rest[:slash], rest[slash:]
        if "@" in authority:
            authority = authority.split("@", 1)[1]
        if ":" in authority:
            host, _, port_s = authority.rpartition(":")
            host = host.strip("[]")
            try:
                port = int(port_s)
            except ValueError:
                return None, None, None
        else:
            host, port = authority.strip("[]"), 80
        if not host or not (0 < port < 65536):
            return None, None, None
        return host, port, path

    async def _validate_and_connect(self, host: str, port: int):
        """Resolve host, validate EVERY address, connect to a validated literal.

        Returns (reader, writer) on success or None if refused. Resolves once and
        connects to the validated IP literal — no re-resolution between check and
        connect (TOCTOU-safe).
        """
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            )
        except (socket.gaierror, OSError) as exc:
            logger.warning("SSRF proxy blocked %s:%s (resolve failed: %s)", host, port, exc)
            return None
        if not infos:
            logger.warning("SSRF proxy blocked %s:%s (no addresses)", host, port)
            return None

        # ANY blocked address → refuse the whole connection (fail closed).
        addrs = []
        for family, _type, _proto, _canon, sockaddr in infos:
            ip = str(sockaddr[0])
            if "%" in ip:
                ip = ip.split("%", 1)[0]
            if ip_is_blocked(ip, block_private=self._block_private):
                logger.warning(
                    "SSRF proxy blocked %s:%s -> %s (policy)", host, port, ip
                )
                return None
            addrs.append((family, ip))

        # Connect to a validated IP literal (first that connects).
        last_exc: Optional[Exception] = None
        for family, ip in addrs:
            try:
                r, w = await asyncio.open_connection(ip, port)
                self.connection_count += 1
                return r, w
            except (OSError, asyncio.TimeoutError) as exc:  # pragma: no cover - net
                last_exc = exc
                continue
        logger.warning("SSRF proxy could not connect %s:%s (%s)", host, port, last_exc)
        return None

    async def _handle_connect(
        self, host: str, port: int,
        creader: asyncio.StreamReader, cwriter: asyncio.StreamWriter,
    ) -> None:
        conn = await self._validate_and_connect(host, port)
        if conn is None:
            self.blocked_count += 1
            await self._refuse(cwriter, "403 Forbidden", "blocked by SSRF policy")
            return
        rreader, rwriter = conn
        try:
            cwriter.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await cwriter.drain()
            await asyncio.gather(
                self._pipe(creader, rwriter),
                self._pipe(rreader, cwriter),
            )
        finally:
            for w in (rwriter, cwriter):
                try:
                    w.close()
                except Exception:
                    pass

    async def _handle_http(
        self, method: str, host: str, port: int, path: str,
        request_line, creader: asyncio.StreamReader, cwriter: asyncio.StreamWriter,
    ) -> None:
        conn = await self._validate_and_connect(host, port)
        if conn is None:
            self.blocked_count += 1
            await self._refuse(cwriter, "403 Forbidden", "blocked by SSRF policy")
            return
        rreader, rwriter = conn
        try:
            # Rewrite the absolute-form request line to origin-form and forward.
            rwriter.write(f"{method} {path} {request_line[2]}\r\n".encode("latin-1"))
            await rwriter.drain()
            await asyncio.gather(
                self._pipe(creader, rwriter),
                self._pipe(rreader, cwriter),
            )
        finally:
            for w in (rwriter, cwriter):
                try:
                    w.close()
                except Exception:
                    pass

    async def _pipe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Copy reader→writer with backpressure (drain) + half-close on EOF."""
        try:
            while True:
                chunk = await reader.read(_PIPE_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()  # backpressure — bound memory on big segments
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            raise
        except Exception:  # pragma: no cover - defensive
            pass
        finally:
            # Half-close: signal EOF downstream so the peer's read loop ends.
            try:
                if writer.can_write_eof():
                    writer.write_eof()
            except Exception:
                pass

    async def _refuse(
        self, writer: asyncio.StreamWriter, status: str, reason: str
    ) -> None:
        try:
            body = f"SSRF proxy refused: {reason}".encode("latin-1", "replace")
            writer.write(
                f"HTTP/1.1 {status}\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n".encode("latin-1")
                + body
            )
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass
