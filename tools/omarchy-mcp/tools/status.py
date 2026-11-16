import json
import os
import time
import psutil


def register(mcp):
    @mcp.tool()
    async def status() -> str:
        """Get the health status of the Omarchy MCP server and VM."""
        claude = os.system("which claude >/dev/null 2>&1") == 0
        codex = os.system("which codex >/dev/null 2>&1") == 0

        mem = psutil.virtual_memory()
        try:
            load = psutil.getloadavg()
        except AttributeError:
            load = (0, 0, 0)

        return json.dumps({
            "ok": True,
            "hostname": os.uname().nodename,
            "platform": "linux",
            "python_version": os.popen("python3 --version 2>&1").read().strip(),
            "uptime_s": round(time.time() - psutil.boot_time()),
            "load_avg": [round(x, 2) for x in load],
            "memory_total_mb": round(mem.total / 1024 / 1024),
            "memory_available_mb": round(mem.available / 1024 / 1024),
            "claude_available": claude,
            "codex_available": codex,
            "version": "1.0.0",
        })
