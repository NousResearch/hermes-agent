"""Completed work remains retrievable when its finite CLI owner exits."""

import http.server
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_headless_terminal_result_survives_cli_exit(tmp_path):
    """Real CLI, tool dispatch, shell child and fresh reader; only the LLM is local."""
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  api_mode: chat_completions\n"
        "terminal:\n  env: local\n  oneshot_completion_wait_seconds: 10\n"
        "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n",
        encoding="utf-8",
    )
    release = tmp_path / "release"
    child = tmp_path / "review.py"
    child.write_text(textwrap.dedent('''
        import pathlib, sys, time
        deadline = time.monotonic() + 15
        while not pathlib.Path(sys.argv[1]).exists():
            if time.monotonic() > deadline:
                sys.exit(91)
            time.sleep(0.02)
        print("SYNTHETIC_REVIEW_COMPLETE")
        print("review stderr", file=sys.stderr)
        sys.exit(7)
    '''), encoding="utf-8")
    argv = [sys.executable, str(child), str(release)]
    command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    observed = []

    class Provider(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(404)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "messages" not in request:
                self.send_error(404)
                return
            tool_results = [m for m in request["messages"] if m["role"] == "tool"]
            has_terminal = any(t.get("function", {}).get("name") == "terminal"
                               for t in request.get("tools", []))
            message = {"role": "assistant", "content": "Coordinator finished."}
            if has_terminal and not tool_results:
                message.update(content=None, tool_calls=[{
                    "id": "call_review", "type": "function", "function": {
                        "name": "terminal", "arguments": json.dumps({
                            "command": command, "background": True, "notify": True,
                        }),
                    },
                }])
            elif tool_results:
                observed.extend(json.loads(m["content"]) for m in tool_results)
                release.touch()
            response = {
                "id": "chatcmpl-local", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{
                    "index": 0, "message": message,
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                }], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            content_type = "application/json"
            if request.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                for index, tool in enumerate(message.get("tool_calls", [])):
                    tool["index"] = index
                raw = ("data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    env = {**os.environ, "HERMES_HOME": str(home), "HOME": str(tmp_path),
           "USERPROFILE": str(tmp_path), "TERMINAL_CWD": str(tmp_path),
           "OPENAI_BASE_URL": url, "OPENAI_API_KEY": "local-test-only",
           "PYTHONPATH": str(REPO_ROOT)}
    try:
        producer = subprocess.run([
            sys.executable, "-c",
            "import cli; cli.main(query='Run the background review', quiet=True, "
            "oneshot=True, provider='custom', model='test-model', api_key='local-test-only', "
            f"base_url={url!r}, toolsets='terminal', max_turns=3, ignore_rules=True)",
        ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    finally:
        release.touch()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert producer.returncode == 0, producer.stdout + producer.stderr
    assert "Coordinator finished." in producer.stdout
    assert len(observed) == 1, (observed, producer.stdout, producer.stderr)
    process_id = observed[0]["session_id"]
    assert observed[0].get("notify_on_complete") is True, observed

    consumer = textwrap.dedent('''
        import json, sys
        import tools.process_registry as pr
        from tools.registry import registry
        result = registry.get_entry("process_manage").handler(
            {"action": "log", "session_id": sys.argv[1]})
        status = registry.get_entry("process_manage").handler(
            {"action": "poll", "session_id": sys.argv[1]})
        print(json.dumps({"result": json.loads(result), "status": json.loads(status),
                          "replayed": not pr.process_registry.completion_queue.empty()}))
    ''')
    def read_result(profile):
        result = subprocess.run([sys.executable, "-c", consumer, process_id],
                                cwd=tmp_path, env={**env, "HERMES_HOME": str(profile)},
                                check=True, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)

    recovered = read_result(home)
    assert recovered["result"]["status"] == "exited", recovered
    assert recovered["status"]["exit_code"] == 7
    assert "SYNTHETIC_REVIEW_COMPLETE" in recovered["result"]["output"]
    assert "review stderr" in recovered["result"]["output"]
    assert recovered["replayed"] is False
    assert read_result(tmp_path / "other-profile")["result"]["status"] == "not_found"


def test_receipts_are_bounded_redacted_and_session_scoped(tmp_path, monkeypatch):
    import time
    from tools import process_registry_results as receipts
    from tools.process_registry import MAX_OUTPUT_CHARS, ProcessRegistry, ProcessSession

    monkeypatch.setattr(receipts, "MAX_RETAINED_RESULTS", 2)
    secret = "sk-" + "aB2cD3eF4gH5iJ6kL7mN8pQ9rS0tU1vW2xY3zA4bC5dE6fG7"
    sessions = []
    registry = ProcessRegistry()
    for index in range(3):
        session = ProcessSession(
            id=f"proc_{index:012x}", command=f"echo {secret}", task_id=f"owner-{index}",
            owner_task_id=f"owner-{index}", session_key=f"chat-{index}",
            started_at=time.time() - receipts.RESULT_RETENTION_SECONDS * 2,
            output_buffer="x" * MAX_OUTPUT_CHARS + "\n" + secret,
            exited=True, exit_code=index,
        )
        registry._running[session.id] = session
        registry._move_to_finished(session)
        sessions.append(session)
    from hermes_constants import get_hermes_home
    paths = list((get_hermes_home() / "logs" / "process-results").glob("*.json"))
    assert len(paths) == 2
    assert all(secret not in path.read_text(encoding="utf-8") for path in paths)
    fresh = ProcessRegistry()
    assert fresh.get(sessions[0].id) is None
    recovered = fresh.get(sessions[-1].id)
    assert recovered.owner_task_id == sessions[-1].owner_task_id
    assert len(recovered.output_buffer) <= MAX_OUTPUT_CHARS
    assert [s["session_id"] for s in fresh.list_sessions(task_id="owner-2")] == [recovered.id]
    assert fresh.list_sessions(task_id="unrelated", session_key="unrelated") == []
    assert fresh.get("proc_0000") is None  # Ambiguous across durable results.
    assert fresh.completion_queue.empty()
    for path in paths:
        expired = time.time() - receipts.RESULT_RETENTION_SECONDS - 1
        os.utime(path, (expired, expired))
    assert fresh.get(recovered.id) is None
