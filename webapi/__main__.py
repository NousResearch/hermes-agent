import os
import socket

import uvicorn


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def main() -> None:
    host = os.getenv("HERMES_WEBAPI_HOST", "127.0.0.1")
    env_port = os.getenv("HERMES_WEBAPI_PORT")
    if env_port:
        port = int(env_port)
    elif not is_port_in_use(8642):
        port = 8642
    else:
        port = 8643
    print(f"Starting Hermes WebAPI on {host}:{port}")
    uvicorn.run("webapi.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
