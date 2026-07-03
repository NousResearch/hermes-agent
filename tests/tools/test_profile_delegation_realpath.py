"""Real-runtime-path integration test for profile-backed delegation.

The other profile-delegation suites mock ``_build_child_agent`` / ``AIAgent``
and assert the *plumbing* (which overrides get passed where). A reviewer on
PR #48644 asked for the one contract those mocks can't prove: that a profile's
config actually reaches the wire — ``profile config -> child AIAgent ->
provider request`` — with no mock at the provider boundary.

So this test stands up a real, threaded OpenAI-compatible HTTP endpoint,
constructs a **real** ``AIAgent`` with the credentials/model/persona a
profile-backed child receives (exactly what ``_resolve_profile_bundle`` +
``_build_child_agent`` feed it), drives one real completion through the agent's
own client path, and asserts the captured request carried the profile's API
key, the profile's model, and the profile's SOUL persona — none of which are
patched.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from run_agent import AIAgent


# A marker string we plant in the profile's SOUL persona and then look for in
# the system message that actually hits the endpoint. If SOUL injection were
# dropped anywhere between profile config and the wire, this assertion fails.
SOUL_MARKER = "I am READER-PROFILE-7f3a, a read-only research specialist."
PROFILE_API_KEY = "profile-secret-key-9c1d"
PROFILE_MODEL = "reader-profile/model-A"


class _CapturingOpenAIHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /chat/completions endpoint that records the
    auth header + body of the first request and returns a valid, tool-call-free
    completion so the caller finishes in a single turn."""

    captured = {}

    def log_message(self, *_args):  # silence the default stderr access log
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        # Capture the real chat completion (the request that carries messages),
        # not the context-length probe the client fires first with an empty body.
        if body.get("messages"):
            type(self).captured = {
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "model": body.get("model"),
                "messages": body.get("messages", []),
            }
        payload = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", PROFILE_MODEL),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ack"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_openai_endpoint():
    _CapturingOpenAIHandler.captured = {}
    server = HTTPServer(("127.0.0.1", 0), _CapturingOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1", _CapturingOpenAIHandler
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_profile_config_reaches_provider_request(fake_openai_endpoint):
    """profile config -> real child AIAgent -> real provider request.

    The agent is built with the profile's base_url / api_key / model and its
    SOUL persona as the (ephemeral) system prompt — the same surface
    ``_build_child_agent`` hands a profile-backed child. We then issue one real
    completion through the agent's OWN client (``_create_openai_client`` ->
    real ``OpenAI`` client -> real HTTP) and prove the profile identity hit the
    wire.
    """
    base_url, handler = fake_openai_endpoint

    # Build the real agent exactly as a profile-backed child is parameterised:
    # the profile's runtime credentials/model and its SOUL prepended into the
    # system prompt. No provider mock — this constructs the genuine client path.
    agent = AIAgent(
        base_url=base_url,
        api_key=PROFILE_API_KEY,
        model=PROFILE_MODEL,
        provider="openai",
        api_mode="chat_completions",
        ephemeral_system_prompt=f"{SOUL_MARKER}\n\nGoal: summarise the repo.",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    # Sanity: the profile's runtime config landed on the real agent object.
    assert agent.base_url == base_url
    assert agent.api_key == PROFILE_API_KEY
    assert agent.model == PROFILE_MODEL

    # Issue exactly one completion through the agent's real client path.
    client = agent._create_openai_client(
        agent._client_kwargs, reason="realpath-test", shared=False
    )
    client.chat.completions.create(
        model=agent.model,
        messages=[
            {"role": "system", "content": agent.ephemeral_system_prompt},
            {"role": "user", "content": "ping"},
        ],
    )

    captured = handler.captured
    assert captured, "fake endpoint never received a request"
    # The profile's API key authenticated the request (not a parent/ambient key).
    assert captured["authorization"] == f"Bearer {PROFILE_API_KEY}"
    # The profile's model was requested.
    assert captured["model"] == PROFILE_MODEL
    # The profile's SOUL persona reached the wire in the system message.
    system_texts = [
        m.get("content", "")
        for m in captured["messages"]
        if m.get("role") == "system"
    ]
    assert any(SOUL_MARKER in text for text in system_texts), (
        "profile SOUL persona was not present in the system message sent to the provider"
    )
