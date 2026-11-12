"""iroh -> TCP bridge (PoC): accepts iroh connections, pipes each
bidirectional stream to a local TCP target (the hermes dashboard).

Run beside `hermes dashboard`. v0 of the iroh transport experiment;
will fold into web_server config once proven.
"""
import asyncio, pathlib, sys

import iroh

TARGET_HOST, TARGET_PORT = "127.0.0.1", 9119
ALPN = b"hermes-http/0"
KEYFILE = pathlib.Path.home() / ".hermes" / "iroh-http.key"
READ_CHUNK = 64 * 1024


def load_secret_key() -> bytes:
    if KEYFILE.exists():
        return KEYFILE.read_bytes()
    sk = iroh.SecretKey.generate().to_bytes()
    KEYFILE.parent.mkdir(parents=True, exist_ok=True)
    KEYFILE.write_bytes(sk)
    KEYFILE.chmod(0o600)
    return sk


async def pipe_stream(stream):
    target_reader, target_writer = await asyncio.open_connection(TARGET_HOST, TARGET_PORT)
    rs, ss = stream.recv(), stream.send()

    async def iroh_to_tcp():
        try:
            while True:
                data = await rs.read(READ_CHUNK)
                if not data:
                    break
                target_writer.write(data)
                await target_writer.drain()
        finally:
            target_writer.close()

    async def tcp_to_iroh():
        try:
            while True:
                data = await target_reader.read(READ_CHUNK)
                if not data:
                    break
                await ss.write(data)
        finally:
            await ss.finish()

    await asyncio.gather(iroh_to_tcp(), tcp_to_iroh(), return_exceptions=True)


async def main():
    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
    b = iroh.EndpointBuilder()
    b.apply_n0()
    b.alpns([ALPN])
    b.secret_key(load_secret_key())
    ep = await b.bind()
    addr = ep.addr()
    print("nodeid:", ep.id(), flush=True)
    print("ticket:", iroh.EndpointTicket.from_addr(addr), flush=True)
    print(f"bridging iroh -> {TARGET_HOST}:{TARGET_PORT}", flush=True)
    while True:
        incoming = await ep.accept_next()
        if incoming is None:
            break
        acc = await incoming.accept()
        conn = await acc.connect()

        async def serve(conn=conn):
            while True:
                stream = await conn.accept_bi()
                asyncio.create_task(pipe_stream(stream))

        asyncio.create_task(serve())


if __name__ == "__main__":
    asyncio.run(main())
