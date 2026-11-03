"""Recording stdio peer for the Skills extension; all state belongs to the test directory."""

import hashlib
import json
from pathlib import Path
import sys

URI = "skill://review/SKILL.md"
REFERENCE = "skill://review/references/check.md"
FRONTMATTER = {"name": "review", "description": "Review changes."}
FILES = {URI: "---\nname: review\ndescription: Review changes.\n---\n\nRead references/check.md.\n",
         REFERENCE: "Check correctness.\n"}


def response(method: str, params: dict) -> dict:
    entry = {"uri": URI, "frontmatter": FRONTMATTER,
             "resources": [{"uri": uri, "size": len(text.encode()),
                            "digest": "sha256:" + hashlib.sha256(text.encode()).hexdigest()}
                           for uri, text in FILES.items()]}
    methods = {
        "server/discover": lambda: {"supportedVersions": ["2026-07-28"], "capabilities": {
            "resources": {}, "extensions": {"io.modelcontextprotocol/skills": {}}}},
        "skills/list": lambda: {"skills": [entry]},
        "skills/get": lambda: {"skill": entry},
        "resources/read": lambda: {"contents": [{"uri": params["uri"], "mimeType": "text/markdown",
                                                 "text": FILES[params["uri"]]}]},
        "ping": dict,
    }
    return {"resultType": "complete", "ttlMs": 0, "cacheScope": "private", **methods[method]()}


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
            payload = {"error": {"code": -32601, "message": "Method or resource not found"}}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], **payload}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
