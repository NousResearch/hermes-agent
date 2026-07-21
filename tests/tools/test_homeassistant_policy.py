import pytest

from tools.homeassistant_policy import classify_service_action


@pytest.mark.parametrize("domain", ["light", "fan", "media_player"])
def test_safe_domain_with_exact_target_is_allowed(domain):
    decision = classify_service_action(domain, f"{domain}.office", {}, mode="safe")
    assert decision.action == "allow"


@pytest.mark.parametrize("entity_id", ["switch.kettle", "scene.goodnight"])
def test_switches_and_scenes_require_approval_until_trusted(entity_id):
    domain = entity_id.split(".", 1)[0]
    decision = classify_service_action(domain, entity_id, {}, mode="safe")
    assert decision.action == "approve"

    trusted = classify_service_action(
        domain, entity_id, {}, mode="safe", trusted_entities={entity_id}
    )
    assert trusted.action == "allow"


@pytest.mark.parametrize("domain", ["lock", "cover", "climate", "alarm_control_panel"])
def test_sensitive_domains_always_require_approval(domain):
    entity_id = f"{domain}.primary"
    decision = classify_service_action(
        domain, entity_id, {}, mode="safe", trusted_entities={entity_id}
    )
    assert decision.action == "approve"


@pytest.mark.parametrize(
    "entity_id,data",
    [
        (None, {}),
        (None, {"entity_id": "all"}),
        (None, {"entity_id": ["light.one", "light.two"]}),
        (None, {"area_id": "kitchen"}),
        (None, {"device_id": "abc"}),
        ("light.one", {"target": {"area_id": "kitchen"}}),
    ],
)
def test_broad_or_ambiguous_targets_require_approval(entity_id, data):
    decision = classify_service_action("light", entity_id, data, mode="safe")
    assert decision.action == "approve"


def test_data_entity_id_is_normalized_for_safe_decision():
    decision = classify_service_action(
        "light", None, {"entity_id": "light.office"}, mode="safe"
    )
    assert decision.action == "allow"
    assert decision.targets == ("light.office",)


def test_legacy_mode_preserves_existing_nonblocked_behavior():
    decision = classify_service_action("lock", "lock.front_door", {}, mode="legacy")
    assert decision.action == "allow"


@pytest.mark.parametrize(
    "domain", ["shell_command", "command_line", "python_script", "pyscript", "hassio", "rest_command"]
)
def test_command_execution_domains_are_blocked_in_every_mode(domain):
    assert classify_service_action(domain, None, {}, mode="legacy").action == "block"
    assert classify_service_action(domain, None, {}, mode="safe").action == "block"

