from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_wisdom.client import WisdomError, WisdomMuteResponse
from hermes_wisdom.preferences import WisdomPreferences
from hermes_wisdom.store import WisdomStore
from tests.wisdom.test_mute_preferences import configure
from tests.wisdom.test_preferences import preferences  # noqa: F401


@pytest.fixture
def controls(preferences):
    p, service, now = preferences
    service.require_setup = Mock()
    service.client.recommendation_mute.return_value = WisdomMuteResponse(
        org_id="org",
        muted=False,
        duration=None,
        muted_until=None,
        forever=False,
        revision=3,
    )
    configure(service)
    return p, service, now


def test_opening_menu_does_not_change_preference(controls):
    p, service, now = controls
    control = p.prepare_mute_control("org")
    assert control["expires_at"] == now[0] + 600
    assert control["mute"]["revision"] == 3
    assert p.mute_status("org") is None
    service.client.set_recommendation_mute.assert_not_called()


def test_same_click_is_durable_and_never_extends_mute(controls):
    p, service, now = controls
    control = p.prepare_mute_control("org")
    first = p.choose_mute_control("org", control["id"], "1_week")
    p.flush_mute("org")
    original = service.client.set_recommendation_mute.call_args
    now[0] += 30
    resumed = WisdomPreferences(
        SimpleNamespace(
            store=WisdomStore(p.store.path.parent),
            client=service.client,
            require_setup=Mock(),
        ),
        clock=lambda: now[0],
    )
    repeat = resumed.choose_mute_control("org", control["id"], "1_week")
    resumed.flush_mute("org")
    assert repeat["mutation_id"] == first["mutation_id"]
    assert repeat["requested_until"] == first["requested_until"]
    assert repeat["preference_sync"] == "synced"
    service.client.set_recommendation_mute.assert_called_once_with(
        *original.args, **original.kwargs
    )


def test_menu_is_one_choice_across_simultaneous_connections(controls):
    p, service, now = controls
    control = p.prepare_mute_control("org")
    other = WisdomPreferences(
        SimpleNamespace(
            store=WisdomStore(p.store.path.parent),
            client=service.client,
            require_setup=Mock(),
        ),
        clock=lambda: now[0],
    )

    def choose(pair):
        pref, duration = pair
        try:
            return pref.choose_mute_control("org", control["id"], duration)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(choose, [(p, "1_day"), (other, "forever")]))
    assert sum(result is not None for result in results) == 1
    assert p.mute_status("org")["requested_duration"] in {"1_day", "forever"}


@pytest.mark.parametrize("selected", [False, True])
def test_old_menu_cannot_overwrite_newer_local_choice(controls, selected):
    p, _, _ = controls
    control = p.prepare_mute_control("org")
    if selected:
        p.choose_mute_control("org", control["id"], "forever")
    fresh = p.stage_mute("org", None, expected_revision=4)
    with pytest.raises(ValueError, match="changed|superseded"):
        p.choose_mute_control("org", control["id"], "forever")
    assert p.mute_status("org") == fresh


def test_menu_keeps_rendered_server_revision_on_click(controls):
    p, service, _ = controls
    control = p.prepare_mute_control("org")
    service.client.recommendation_mute.return_value.revision = 4
    p.choose_mute_control("org", control["id"], "forever")
    service.client.set_recommendation_mute.side_effect = WisdomError(
        "conflict", status=409
    )
    p.flush_mute("org")
    assert (
        service.client.set_recommendation_mute.call_args.kwargs["expected_revision"]
        == 3
    )
    assert p.mute_status("org")["preference_sync"] == "conflict"


def test_menu_rejects_expiry_and_wrong_account(controls):
    p, service, now = controls
    control = p.prepare_mute_control("org")
    service.client.identity = {"owner": "another-account"}
    with pytest.raises(ValueError, match="expired"):
        p.choose_mute_control("org", control["id"], None)
    service.client.identity = {"owner": "account"}
    now[0] += 600
    with pytest.raises(ValueError, match="expired"):
        p.choose_mute_control("org", control["id"], None)
    assert p.mute_status("org") is None


def test_menu_rejects_organization_change(controls):
    p, _, _ = controls
    control = p.prepare_mute_control("org")
    p.store.activate_installation_identity("installation", "other-org")
    with pytest.raises(ValueError, match="organization"):
        p.choose_mute_control("org", control["id"], "forever")


def test_failed_refresh_cannot_issue_menu(controls):
    p, service, _ = controls
    service.client.recommendation_mute.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        p.prepare_mute_control("org")
    with p.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_mute_control").fetchone()[0] == 0


def test_native_messaging_callbacks_use_same_durable_menu(controls, monkeypatch):
    from gateway.wisdom_command import (
        WisdomCommandContext,
        WisdomCommandController,
        bind_view_callbacks,
    )

    p, service, _ = controls
    monkeypatch.setattr(
        "hermes_wisdom.preferences.WisdomPreferences", lambda service: p
    )
    controller = WisdomCommandController()
    context = WisdomCommandContext(
        user_id="user",
        chat_id="dm",
        profile="demo",
        organization_id="org",
        is_group=False,
    )
    view = controller.execute("mute", service, context)
    bind_view_callbacks(view, context)
    actions = [*(a for item in view.items for a in item.actions), *view.actions]
    assert all(action.callback_data.startswith("wi:cmd:") for action in actions)
    assert view.actions[-1].label == "Turn on"
    assert view.actions[-1].primary
    control_ids = {action.arguments["control_id"] for action in actions}
    assert len(control_ids) == 1
    target = actions[1].callback_data.removeprefix("wi:cmd:")
    controller.execute_token(target, service, context)
    first = p.mute_status("org")
    controller.execute_token(target, service, context)
    assert p.mute_status("org") == first
    service.client.set_recommendation_mute.assert_called_once()
    with pytest.raises(PermissionError):
        controller.execute_token(
            target,
            service,
            WisdomCommandContext(
                user_id="stranger",
                chat_id="dm",
                profile="demo",
                organization_id="org",
                is_group=False,
            ),
        )


def test_local_confirmation_uses_control_not_conversational_yes(controls, monkeypatch):
    from gateway.wisdom_command import (
        WisdomCommandContext,
        WisdomCommandController,
        render_local_view,
    )

    p, service, _ = controls
    monkeypatch.setattr(
        "hermes_wisdom.preferences.WisdomPreferences", lambda service: p
    )
    context = WisdomCommandContext(
        user_id="user",
        chat_id="local:session",
        profile="demo",
        organization_id="org",
        is_group=False,
    )
    controller = WisdomCommandController()
    view = controller.execute("mute", service, context)
    output = render_local_view(view, context)
    assert "/wisdom action " in output
    service.client.set_recommendation_mute.assert_not_called()
    token = view.items[0].actions[0].callback_data.removeprefix("wi:cmd:")
    controller.execute(f"action {token}", service, context)
    assert p.mute_status("org")["requested_duration"] == "1_day"
