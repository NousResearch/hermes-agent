import asyncio

import pytest

from hermes_cli.web_routers import wisdom as wisdom_routes
from hermes_cli.web_models import WisdomMuteChooseRequest, WisdomMutePrepareRequest
from tests.wisdom.test_mute_controls import controls  # noqa: F401
from tests.wisdom.test_preferences import preferences  # noqa: F401


def test_native_mute_routes_keep_profile_scope_and_exact_choice(controls, monkeypatch):
    p, service, _ = controls
    scopes = []

    async def run(profile, fn):
        scopes.append(profile)
        service.require_setup()
        return fn(service)

    monkeypatch.setattr(wisdom_routes, "_run_wisdom", run)
    monkeypatch.setattr("hermes_wisdom.preferences.WisdomPreferences", lambda service: p)
    initial = asyncio.run(wisdom_routes.get_wisdom_mute(profile="demo"))
    assert initial["mute"]["muted"] is False
    control = asyncio.run(wisdom_routes.post_wisdom_mute_prepare(WisdomMutePrepareRequest(profile="demo")))
    service.client.set_recommendation_mute.assert_not_called()
    assert control["organization_id"] == "org"
    body = WisdomMuteChooseRequest(profile="demo", control_id=control["id"], duration="1_day")
    first = asyncio.run(wisdom_routes.post_wisdom_mute_choose(body))
    second = asyncio.run(wisdom_routes.post_wisdom_mute_choose(body))
    assert first["sync"] == second["sync"]
    assert first["sync"]["preference_sync"] == "synced"
    service.client.set_recommendation_mute.assert_called_once()
    assert service.client.set_recommendation_mute.call_args.kwargs["expected_revision"] == 3
    assert scopes == ["demo"] * 4


@pytest.mark.parametrize("patch", [
    {"duration": "tomorrow"}, {"duration": False}, {"control_id": ""},
    {"expected_revision": 0}, {"user_id": "someone"}, {"organization_id": "other"},
])
def test_browser_cannot_supply_preference_authority(patch):
    with pytest.raises(ValueError):
        WisdomMuteChooseRequest(**{"control_id": "a" * 32, "duration": None, **patch})


def test_unmute_requires_explicit_null_not_missing_duration():
    assert WisdomMuteChooseRequest(control_id="a" * 32, duration=None).duration is None
    with pytest.raises(ValueError):
        WisdomMuteChooseRequest(control_id="a" * 32)
    with pytest.raises(ValueError):
        WisdomMutePrepareRequest(profile="demo", user_id="other")
