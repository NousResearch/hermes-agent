import subprocess
import json
import time

# Using python -m to ensure we use the local editable install
proc = subprocess.Popen(
    ["python3", "-m", "hermes_cli.main", "--tui"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

def send_rpc(method, params):
    req = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()

send_rpc("session.new", {})
send_rpc("chat.send", {"message": "Hi", "session_id": "default"})

found = False
start = time.time()
while time.time() - start < 10:
    line = proc.stdout.readline()
    if not line: break
    if "usage.delta" in line:
        print(f"FOUND: {line.strip()}")
        found = True

proc.terminate()
print(f"RESULT: {'SUCCESS' if found else 'FAILURE'}")
