import os, sys, logging
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TOKEN = os.environ.get("OMARCHY_MCP_TOKEN", "")
if not TOKEN:
    print("ERROR: OMARCHY_MCP_TOKEN is required", file=sys.stderr)
    sys.exit(1)

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from auth import verify_token
from tools import ALL_TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omarchy-mcp")

mcp = MCPServer("omarchy-mcp", instructions="Omarchy VM remote execution server.")
for fn in ALL_TOOLS:
    fn(mcp)
    logger.info("Registered tool from %s", fn.__module__)

import starlette.middleware.base as base
from starlette.responses import JSONResponse

class BearerAuthMiddleware(base.BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"error": "Unauthorized: missing Bearer token"}, status_code=401)
        token_val = auth[len("Bearer "):]
        if not verify_token(token_val):
            return JSONResponse({"error": "Unauthorized: invalid token"}, status_code=401)
        return await call_next(request)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8911"))
    bind = os.environ.get("BIND", "127.0.0.1")
    ssl_cert = os.environ.get("TLS_CERT", "")
    ssl_key = os.environ.get("TLS_KEY", "")

    kwargs = {}
    scheme = "http"
    if ssl_cert and ssl_key:
        kwargs["ssl_certfile"] = ssl_cert
        kwargs["ssl_keyfile"] = ssl_key
        scheme = "https"

    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    app = BearerAuthMiddleware(app)
    logger.info("Starting omarchy-mcp on %s://%s:%s/mcp", scheme, bind, port)
    uvicorn.run(app, host=bind, port=port, log_level="info", **kwargs)
