"""Recording stdio MCP peer (2026-07-28 stateless) that runs ``slow_job`` as an MCP Task.

Usage: mcp_tasks_server.py <wire log path>. ``slow_job`` answers with a task handle when the
request declares the Tasks extension, else synchronously; ``tasks/get`` reports ``working`` twice,
then the result. ``broken_job`` fails as a task;
``ask_job`` stops at ``input_required``; ``lost_job``'s status reads fail as unauthorized. Every inbound message is appended to the log.
"""

import json
import sys
from pathlib import Path

EXT = "io.modelcontextprotocol/tasks"
TOOLS = [{"name": name, "description": desc, "inputSchema": {"type": "object", "properties": {}}}
         for name, desc in (("slow_job", "Run the nightly report job and return its summary."),
                            ("broken_job", "Run a job that always fails."),
                            ("ask_job", "Run a job that stops to ask a question."),
                            ("lost_job", "Run a job whose status can no longer be read."))]
polls: dict[str, int] = {}


def _declares_tasks(params: dict) -> bool:
    caps = (params.get("_meta") or {}).get("io.modelcontextprotocol/clientCapabilities") or {}
    return EXT in (caps.get("extensions") or {})


def _task_state(task_id: str) -> dict:
    polls[task_id] = polls.get(task_id, 0) + 1
    if polls[task_id] < 3:
        return {"taskId": task_id, "status": "working", "ttlMs": 60000, "pollIntervalMs": 200,
                "statusMessage": f"step {polls[task_id]} of 3"}
    if task_id.startswith("ask"):
        return {"taskId": task_id, "status": "input_required", "ttlMs": 60000,
                "inputRequests": {"q1": {"method": "elicitation/create", "params": {"message": "Which branch?"}}}}
    if task_id.startswith("broken"):
        return {"taskId": task_id, "status": "failed", "ttlMs": 60000,
                "error": {"code": -32000, "message": "upstream build server rejected the job"}}
    return {"taskId": task_id, "status": "completed", "ttlMs": 60000,
            "result": {"content": [{"type": "text", "text": "Nightly report: 1,284 rows, checksum ORCHID-5521"}]}}


def response(method: str, params: dict) -> dict:
    base = {"resultType": "complete"}
    if method == "server/discover":
        return {**base, "supportedVersions": ["2026-07-28"],
                "capabilities": {"tools": {}, "extensions": {EXT: {}}},
                "serverInfo": {"name": "tasks-peer", "version": "1.0"}}
    if method == "tools/list":
        return {**base, "tools": TOOLS, "ttlMs": 0, "cacheScope": "private"}
    if method == "tools/call":
        name = params["name"]
        if _declares_tasks(params):
            # Long task ids are legal; hosts must not cap them at 128 characters.
            task_id = f"{name.split('_')[0]}-" + "x" * 300
            return {"resultType": "task", "taskId": task_id, "status": "working", "ttlMs": 60000, "pollIntervalMs": 200}
        return {**base, "content": [{"type": "text", "text": "sync fallback: job not started"}], "isError": True}
    if method == "tasks/get" and params["taskId"].startswith("lost"):
        raise PermissionError("token expired")
    if method == "tasks/get":
        return {**base, **_task_state(params["taskId"])}
    if method == "tasks/cancel":
        return base
    if method == "ping":
        return {}
    raise KeyError(method)


def main(log: Path) -> None:
    for line in sys.stdin:
        message = json.loads(line)
        with log.open("a", encoding="utf-8") as output:
            output.write(json.dumps(message) + "\n")
        if "id" not in message:
            continue
        try:
            payload = {"result": response(message["method"], message.get("params", {}))}
        except KeyError:
            payload = {"error": {"code": -32601, "message": "Method not found"}}
        except PermissionError as exc:
            payload = {"error": {"code": -32001, "message": f"Unauthorized: {exc}"}}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], **payload}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
