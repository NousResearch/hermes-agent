"""Regression for https://github.com/NousResearch/hermes-agent/issues/120330.

``hermes -p <profile> cron run <job>`` for a profile routed through another
profile's bot via ``gateway.profile_routes`` failed delivery with
``platform 'telegram' not configured/enabled`` and overwrote ``last_status``
with ``delivery_failed`` — the CLI process has no native adapter for the
target platform. The multiplex gateway's ``SharedRouteAdapters`` owns the
transport for routed profiles (#89302 covers the in-gateway form); the CLI
must forward ``POST /api/jobs/{id}/run`` to the gateway rather than fail
in-process (#120330).
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


_JOB_NO_PLATFORMS = {
    "id": "keeper-daily",
    "name": "Daily brief",
    "deliver": "telegram:12345:topic=42",
    "next_run_at": "2026-09-23T15:00:00Z",
}

_JOB_OWN_BOT = {
    "id": "owner-daily",
    "name": "Owner brief",
    "deliver": "discord:67890",
    "next_run_at": "2026-09-23T15:00:00Z",
}

_ARGS = {
    "action": "run",
    "job_id": "keeper-daily",
    "name": None,
    "deliver": None,
    "skill": None,
    "skills": None,
    "failure_deliver": None,
    "prompt": None,
    "session_id": None,
}


@pytest.fixture(autouse=True)
def _no_async_session():
    """Background dispatch must report no live ``HERMES_SESSION_KEY`` so
    the synchronous path is exercised (the routed-profile forward only
    runs after the background path returns None — see ``_action_run``)."""
    saved = os.environ.pop("HERMES_SESSION_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["HERMES_SESSION_KEY"] = saved


def test_routed_profile_run_forwards_to_gateway():
    """``platforms:`` absent on this profile + a delivery target → forward."""
    import tools.cronjob_tools as ct

    mock_config = {"platforms": {}}
    mock_resp = MagicMock(status_code=200)

    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config), \
         patch("agent.secret_scope.get_secret", return_value="test-key"), \
         patch("tools.cronjob_tools._api_server_base_url",
               return_value="http://127.0.0.1:8642"), \
         patch("httpx.post", return_value=mock_resp) as mock_post, \
         patch("tools.cronjob_tools._try_dispatch_background_run", return_value=None), \
         patch("tools.cronjob_tools._forward_relay_fronted_run", return_value=None), \
         patch("tools.cronjob_tools._execute_job_now") as mock_exec:
        mock_exec.return_value = {"claimed": True, "success": True}
        result = json.loads(ct._action_run(dict(_JOB_NO_PLATFORMS), dict(_ARGS)))

    assert result.get("forwarded_to_gateway") is True
    assert mock_post.call_count == 1
    url = mock_post.call_args[0][0]
    assert url.endswith("/api/jobs/keeper-daily/run")
    assert mock_exec.call_count == 0, "in-process execution must NOT run when forwarded"


def test_own_bot_profile_runs_in_process():
    """This profile has its own adapter for the target platform → no forward."""
    import tools.cronjob_tools as ct

    mock_config = {"platforms": {"discord": {"enabled": True}}}
    mock_resp = MagicMock(status_code=200)

    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config), \
         patch("agent.secret_scope.get_secret", return_value="test-key"), \
         patch("tools.cronjob_tools._api_server_base_url",
               return_value="http://127.0.0.1:8642"), \
         patch("httpx.post", return_value=mock_resp) as mock_post, \
         patch("tools.cronjob_tools._try_dispatch_background_run", return_value=None), \
         patch("tools.cronjob_tools._forward_relay_fronted_run", return_value=None), \
         patch("tools.cronjob_tools._execute_job_now") as mock_exec:
        mock_exec.return_value = {"claimed": True, "success": True}
        job_args = dict(_ARGS)
        job_args["job_id"] = "owner-daily"
        result = json.loads(ct._action_run(dict(_JOB_OWN_BOT), job_args))

    assert result.get("forwarded_to_gateway") is None
    assert mock_post.call_count == 0, "no POST expected — own bot delivers natively"
    assert mock_exec.call_count == 1, "in-process execution runs for own-bot profiles"


def test_routed_delivery_platforms_helper():
    """The detection helper returns only platforms missing from this profile."""
    import tools.cronjob_tools as ct

    mock_config = {"platforms": {"discord": {"enabled": True}}}

    # All missing
    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
        assert ct._routed_delivery_platforms({"deliver": "telegram:12345"}) == {"telegram"}

    # All present
    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
        assert ct._routed_delivery_platforms({"deliver": "discord:67890"}) == set()

    # Partial
    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
        assert ct._routed_delivery_platforms(
            {"deliver": "discord:67890,telegram:12345"}
        ) == {"telegram"}

    # Local job — never routed
    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
        assert ct._routed_delivery_platforms({"deliver": ""}) == set()