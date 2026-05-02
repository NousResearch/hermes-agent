import subprocess
import json
import time
import os

# Start the hermes TUI
proc = subprocess.Popen(
    ["hermes", "--tui"],
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

# 1. Create a session
send_rpc("session.new", {})

# 2. Send a message to trigger streaming
send_rpc("chat.send", {"message": "Tell me a short story about a robot learning to paint.", "session_id": "default"})

# 3. Capture output for a few seconds
found_usage_delta = False
start_time = time.time()
while time.time() - start_time < 15:
    line = proc.stdout.readline()
    if not line:
        break
    if "usage.delta" in line:
        print(f"FOUND: {line.strip()}")
        found_usage_delta = True
    
proc.terminate()

if found_usage_delta:
    print("\nRESULT: SUCCESS - usage.delta events detected in the TUI stream.")
else:
    print("\nRESULT: FAILURE - No usage.delta events detected.")
