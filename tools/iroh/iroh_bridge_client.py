"""iroh client bridge (PoC): listens on a local TCP port and dials the
remote iroh bridge for every connection, piping bytes both ways.

Run on the machine with the browser: `python iroh_bridge_client.py <ticket> [port]`
then open http://127.0.0.1:<port>
"""
import asyncio, sys

import iroh

ALPN = b"hermes-http/0"
LOCAL_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9119
TICKET = sys.argv[1]
READ_CHUNK = 64 * 1024


async def handle(reader, writer, ep):
    try:
        conn = await ep.connect(iroh.EndpointTicket.from_string(TICKET).endpoint_addr(), ALPN)
        stream = await conn.open_bi()
    except Exception as e:
        print("dial failed:", e, flush=True)
        writer.close()
        return
    rs, ss = stream.recv(), stream.send()

    async def client_to_iroh():
        try:
            while True:
                data = await reader.read(READ_CHUNK)
                if not data:
                    break
                await ss.write(data)
        finally:
            await ss.finish()

    async def iroh_to_client():
        try:
            while True:
                data = await rs.read(READ_CHUNK)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    await asyncio.gather(client_to_iroh(), iroh_to_client(), return_exceptions=True)


async def main():
    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
    b = iroh.EndpointBuilder()
    b.apply_n0()
    b.alpns([ALPN])
    ep = await b.bind()
    print("client nodeid:", ep.id(), flush=True)
    server = await asyncio.start_server(lambda r, w: handle(r, w, ep), "127.0.0.1", LOCAL_PORT)
    print(f"listening http://127.0.0.1:{LOCAL_PORT} -> iroh", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
