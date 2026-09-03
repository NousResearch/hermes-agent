"""Unit tests for browser supervisor network-response tracking."""

import asyncio

from tools.browser_supervisor import CDPSupervisor, NETWORK_RESPONSE_HISTORY_MAX


def test_network_response_records_browser_reported_remote_ip():
    supervisor = CDPSupervisor(task_id="net-test", cdp_url="ws://127.0.0.1/devtools/browser/test")

    supervisor._on_network_response_received(
        {
            "type": "Document",
            "response": {
                "url": "https://rebind.example/",
                "remoteIPAddress": "127.0.0.1",
                "status": 200,
            },
        }
    )

    responses = supervisor.snapshot().network_responses
    assert len(responses) == 1
    assert responses[0].url == "https://rebind.example/"
    assert responses[0].remote_ip == "127.0.0.1"
    assert responses[0].status == 200
    assert responses[0].resource_type == "Document"


def test_network_response_without_remote_ip_is_ignored():
    supervisor = CDPSupervisor(task_id="net-test", cdp_url="ws://127.0.0.1/devtools/browser/test")

    supervisor._on_network_response_received(
        {
            "type": "Document",
            "response": {
                "url": "https://cached.example/",
                "status": 200,
            },
        }
    )

    assert supervisor.snapshot().network_responses == ()


def test_clear_network_responses_drops_recorded_history():
    supervisor = CDPSupervisor(task_id="net-test", cdp_url="ws://127.0.0.1/devtools/browser/test")

    supervisor._on_network_response_received(
        {
            "type": "Document",
            "response": {
                "url": "https://rebind.example/",
                "remoteIPAddress": "127.0.0.1",
                "status": 200,
            },
        }
    )
    assert supervisor.snapshot().network_responses

    supervisor.clear_network_responses()

    assert supervisor.snapshot().network_responses == ()


def test_private_peer_survives_network_history_trimming():
    supervisor = CDPSupervisor(
        task_id="net-test",
        cdp_url="ws://127.0.0.1/devtools/browser/test",
    )

    supervisor._on_network_response_received(
        {
            "type": "Document",
            "response": {
                "url": "https://rebind.example/internal",
                "remoteIPAddress": "127.0.0.1",
                "status": 200,
            },
        }
    )
    supervisor._on_network_response_received(
        {
            "type": "Fetch",
            "response": {
                "url": "https://rebind.example/metadata",
                "remoteIPAddress": "169.254.169.254",
                "status": 200,
            },
        }
    )
    for index in range(NETWORK_RESPONSE_HISTORY_MAX * 2 + 1):
        supervisor._on_network_response_received(
            {
                "type": "Image",
                "response": {
                    "url": f"https://cdn.example/image-{index}.png",
                    "remoteIPAddress": "93.184.216.34",
                    "status": 200,
                },
            }
        )

    responses = supervisor.snapshot().network_responses

    assert len(responses) == NETWORK_RESPONSE_HISTORY_MAX
    assert any(record.remote_ip == "127.0.0.1" for record in responses)
    assert any(record.remote_ip == "169.254.169.254" for record in responses)


def test_start_network_response_window_atomically_returns_and_clears_history():
    supervisor = CDPSupervisor(
        task_id="net-test",
        cdp_url="ws://127.0.0.1/devtools/browser/test",
    )
    supervisor._on_network_response_received(
        {
            "type": "Document",
            "response": {
                "url": "https://rebind.example/internal",
                "remoteIPAddress": "127.0.0.1",
                "status": 200,
            },
        }
    )

    prior_responses = supervisor.start_network_response_window()

    assert [record.remote_ip for record in prior_responses] == ["127.0.0.1"]
    assert supervisor.snapshot().network_responses == ()


def test_retain_network_violation_restores_security_latch():
    supervisor = CDPSupervisor(
        task_id="net-test",
        cdp_url="ws://127.0.0.1/devtools/browser/test",
    )

    supervisor.retain_network_violation(
        "127.0.0.1",
        "https://rebind.example/internal",
    )

    responses = supervisor.snapshot().network_responses
    assert [record.remote_ip for record in responses] == ["127.0.0.1"]
    assert responses[0].resource_type == "SecurityViolation"


def test_enable_network_tracking_sends_network_enable():
    supervisor = CDPSupervisor(task_id="net-test", cdp_url="ws://127.0.0.1/devtools/browser/test")
    calls = []

    async def fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
        calls.append((method, params, session_id, timeout))
        return {"result": {}}

    supervisor._cdp = fake_cdp

    asyncio.run(supervisor._enable_network_tracking("sid-1"))

    assert calls == [("Network.enable", None, "sid-1", 3.0)]
