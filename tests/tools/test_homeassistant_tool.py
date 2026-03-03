"""Tests for the Home Assistant tool module.

Tests real logic: entity filtering, payload building, response parsing,
handler validation, and availability gating.
"""

import json

import pytest

from tools.homeassistant_tool import (
    _check_ha_available,
    _filter_and_summarize,
    _build_service_payload,
    _parse_service_response,
    _get_headers,
    _handle_get_state,
    _handle_call_service,
    _BLOCKED_DOMAINS,
    _ENTITY_ID_RE,
)


# ---------------------------------------------------------------------------
# Sample HA state data (matches real HA /api/states response shape)
# ---------------------------------------------------------------------------

SAMPLE_STATES = [
    {"entity_id": "light.bedroom", "state": "on", "attributes": {"friendly_name": "Bedroom Light", "brightness": 200}},
    {"entity_id": "light.kitchen", "state": "off", "attributes": {"friendly_name": "Kitchen Light"}},
    {"entity_id": "switch.fan", "state": "on", "attributes": {"friendly_name": "Living Room Fan"}},
    {"entity_id": "sensor.temperature", "state": "22.5", "attributes": {"friendly_name": "Kitchen Temperature", "unit_of_measurement": "C"}},
    {"entity_id": "climate.thermostat", "state": "heat", "attributes": {"friendly_name": "Main Thermostat", "current_temperature": 21}},
    {"entity_id": "binary_sensor.motion", "state": "off", "attributes": {"friendly_name": "Hallway Motion"}},
    {"entity_id": "sensor.humidity", "state": "55", "attributes": {"friendly_name": "Bedroom Humidity", "area": "bedroom"}},
]


# ---------------------------------------------------------------------------
# Entity filtering and summarization
# ---------------------------------------------------------------------------


class TestFilterAndSummarize:
    def test_no_filters_returns_all(self):
        result = _filter_and_summarize(SAMPLE_STATES)
        assert result["count"] == 7
        ids = {e["entity_id"] for e in result["entities"]}
        assert "light.bedroom" in ids
        assert "climate.thermostat" in ids

    def test_domain_filter_lights(self):
        result = _filter_and_summarize(SAMPLE_STATES, domain="light")
        assert result["count"] == 2
        for e in result["entities"]:
            assert e["entity_id"].startswith("light.")

    def test_domain_filter_sensor(self):
        result = _filter_and_summarize(SAMPLE_STATES, domain="sensor")
        assert result["count"] == 2
        ids = {e["entity_id"] for e in result["entities"]}
        assert ids == {"sensor.temperature", "sensor.humidity"}

    def test_domain_filter_no_matches(self):
        result = _filter_and_summarize(SAMPLE_STATES, domain="media_player")
        assert result["count"] == 0
        assert result["entities"] == []

    def test_area_filter_by_friendly_name(self):
        result = _filter_and_summarize(SAMPLE_STATES, area="kitchen")
        assert result["count"] == 2
        ids = {e["entity_id"] for e in result["entities"]}
        assert "light.kitchen" in ids
        assert "sensor.temperature" in ids

    def test_area_filter_by_area_attribute(self):
        result = _filter_and_summarize(SAMPLE_STATES, area="bedroom")
        ids = {e["entity_id"] for e in result["entities"]}
        # "Bedroom Light" matches via friendly_name, "Bedroom Humidity" matches via area attr
        assert "light.bedroom" in ids
        assert "sensor.humidity" in ids

    def test_area_filter_case_insensitive(self):
        result = _filter_and_summarize(SAMPLE_STATES, area="KITCHEN")
        assert result["count"] == 2

    def test_combined_domain_and_area(self):
        result = _filter_and_summarize(SAMPLE_STATES, domain="sensor", area="kitchen")
        assert result["count"] == 1
        assert result["entities"][0]["entity_id"] == "sensor.temperature"

    def test_summary_includes_friendly_name(self):
        result = _filter_and_summarize(SAMPLE_STATES, domain="climate")
        assert result["entities"][0]["friendly_name"] == "Main Thermostat"
        assert result["entities"][0]["state"] == "heat"

    def test_empty_states_list(self):
        result = _filter_and_summarize([])
        assert result["count"] == 0

    def test_missing_attributes_handled(self):
        states = [{"entity_id": "light.x", "state": "on"}]
        result = _filter_and_summarize(states)
        assert result["count"] == 1
        assert result["entities"][0]["friendly_name"] == ""


# ---------------------------------------------------------------------------
# Service payload building
# ---------------------------------------------------------------------------


class TestBuildServicePayload:
    def test_entity_id_only(self):
        payload = _build_service_payload(entity_id="light.bedroom")
        assert payload == {"entity_id": "light.bedroom"}

    def test_data_only(self):
        payload = _build_service_payload(data={"brightness": 255})
        assert payload == {"brightness": 255}

    def test_entity_id_and_data(self):
        payload = _build_service_payload(
            entity_id="light.bedroom",
            data={"brightness": 200, "color_name": "blue"},
        )
        assert payload["entity_id"] == "light.bedroom"
        assert payload["brightness"] == 200
        assert payload["color_name"] == "blue"

    def test_no_args_returns_empty(self):
        payload = _build_service_payload()
        assert payload == {}

    def test_entity_id_param_takes_precedence_over_data(self):
        payload = _build_service_payload(
            entity_id="light.a",
            data={"entity_id": "light.b"},
        )
        # explicit entity_id parameter wins over data["entity_id"]
        assert payload["entity_id"] == "light.a"


# ---------------------------------------------------------------------------
# Service response parsing
# ---------------------------------------------------------------------------


class TestParseServiceResponse:
    def test_list_response_extracts_entities(self):
        ha_response = [
            {"entity_id": "light.bedroom", "state": "on", "attributes": {}},
            {"entity_id": "light.kitchen", "state": "on", "attributes": {}},
        ]
        result = _parse_service_response("light", "turn_on", ha_response)
        assert result["success"] is True
        assert result["service"] == "light.turn_on"
        assert len(result["affected_entities"]) == 2
        assert result["affected_entities"][0]["entity_id"] == "light.bedroom"

    def test_empty_list_response(self):
        result = _parse_service_response("scene", "turn_on", [])
        assert result["success"] is True
        assert result["affected_entities"] == []

    def test_non_list_response(self):
        # Some HA services return a dict instead of a list
        result = _parse_service_response("script", "run", {"result": "ok"})
        assert result["success"] is True
        assert result["affected_entities"] == []

    def test_none_response(self):
        result = _parse_service_response("automation", "trigger", None)
        assert result["success"] is True
        assert result["affected_entities"] == []

    def test_service_name_format(self):
        result = _parse_service_response("climate", "set_temperature", [])
        assert result["service"] == "climate.set_temperature"


# ---------------------------------------------------------------------------
# Handler validation (no mocks - these paths don't reach the network)
# ---------------------------------------------------------------------------


class TestHandlerValidation:
    def test_get_state_missing_entity_id(self):
        result = json.loads(_handle_get_state({}))
        assert "error" in result
        assert "entity_id" in result["error"]

    def test_get_state_empty_entity_id(self):
        result = json.loads(_handle_get_state({"entity_id": ""}))
        assert "error" in result

    def test_call_service_missing_domain(self):
        result = json.loads(_handle_call_service({"service": "turn_on"}))
        assert "error" in result
        assert "domain" in result["error"]

    def test_call_service_missing_service(self):
        result = json.loads(_handle_call_service({"domain": "light"}))
        assert "error" in result
        assert "service" in result["error"]

    def test_call_service_missing_both(self):
        result = json.loads(_handle_call_service({}))
        assert "error" in result

    def test_call_service_empty_strings(self):
        result = json.loads(_handle_call_service({"domain": "", "service": ""}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Security: domain blocklist
# ---------------------------------------------------------------------------


class TestDomainBlocklist:
    """Verify dangerous HA service domains are blocked."""

    @pytest.mark.parametrize("domain", sorted(_BLOCKED_DOMAINS))
    def test_blocked_domain_rejected(self, domain):
        result = json.loads(_handle_call_service({
            "domain": domain, "service": "any_service"
        }))
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_safe_domain_not_blocked(self):
        """Safe domains like 'light' should not be blocked (will fail on network, not blocklist)."""
        # This will try to make a real HTTP call and fail, but the important thing
        # is it does NOT return a "blocked" error
        result = json.loads(_handle_call_service({
            "domain": "light", "service": "turn_on", "entity_id": "light.test"
        }))
        # Should fail with a network/connection error, not a "blocked" error
        if "error" in result:
            assert "blocked" not in result["error"].lower()

    def test_blocked_domains_include_shell_command(self):
        assert "shell_command" in _BLOCKED_DOMAINS

    def test_blocked_domains_include_hassio(self):
        assert "hassio" in _BLOCKED_DOMAINS

    def test_blocked_domains_include_rest_command(self):
        assert "rest_command" in _BLOCKED_DOMAINS


# ---------------------------------------------------------------------------
# Security: entity_id validation
# ---------------------------------------------------------------------------


class TestEntityIdValidation:
    """Verify entity_id format validation prevents path traversal."""

    def test_valid_entity_id_accepted(self):
        assert _ENTITY_ID_RE.match("light.bedroom")
        assert _ENTITY_ID_RE.match("sensor.temperature_1")
        assert _ENTITY_ID_RE.match("binary_sensor.motion")
        assert _ENTITY_ID_RE.match("climate.main_thermostat")

    def test_path_traversal_rejected(self):
        assert _ENTITY_ID_RE.match("../../config") is None
        assert _ENTITY_ID_RE.match("light/../../../etc/passwd") is None
        assert _ENTITY_ID_RE.match("../api/config") is None

    def test_special_chars_rejected(self):
        assert _ENTITY_ID_RE.match("light.bed room") is None  # space
        assert _ENTITY_ID_RE.match("light.bed;rm -rf") is None  # semicolon
        assert _ENTITY_ID_RE.match("light.bed/room") is None  # slash
        assert _ENTITY_ID_RE.match("LIGHT.BEDROOM") is None  # uppercase

    def test_missing_domain_rejected(self):
        assert _ENTITY_ID_RE.match(".bedroom") is None
        assert _ENTITY_ID_RE.match("bedroom") is None

    def test_get_state_rejects_invalid_entity_id(self):
        result = json.loads(_handle_get_state({"entity_id": "../../config"}))
        assert "error" in result
        assert "Invalid entity_id" in result["error"]

    def test_call_service_rejects_invalid_entity_id(self):
        result = json.loads(_handle_call_service({
            "domain": "light",
            "service": "turn_on",
            "entity_id": "../../../etc/passwd",
        }))
        assert "error" in result
        assert "Invalid entity_id" in result["error"]

    def test_call_service_allows_no_entity_id(self):
        """Some services (like scene.turn_on) don't need entity_id."""
        # Will fail on network, but should NOT fail on entity_id validation
        result = json.loads(_handle_call_service({
            "domain": "scene", "service": "turn_on"
        }))
        if "error" in result:
            assert "Invalid entity_id" not in result["error"]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


class TestCheckAvailable:
    def test_unavailable_without_token(self, monkeypatch):
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        assert _check_ha_available() is False

    def test_available_with_token(self, monkeypatch):
        monkeypatch.setenv("HASS_TOKEN", "eyJ0eXAiOiJKV1Q")
        assert _check_ha_available() is True

    def test_empty_token_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("HASS_TOKEN", "")
        assert _check_ha_available() is False


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestGetHeaders:
    def test_bearer_token_format(self, monkeypatch):
        monkeypatch.setattr("tools.homeassistant_tool._HASS_TOKEN", "my-secret-token")
        headers = _get_headers()
        assert headers["Authorization"] == "Bearer my-secret-token"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_tools_registered_in_registry(self):
        from tools.registry import registry

        names = registry.get_all_tool_names()
        assert "ha_list_entities" in names
        assert "ha_get_state" in names
        assert "ha_call_service" in names

    def test_tools_in_homeassistant_toolset(self):
        from tools.registry import registry

        toolset_map = registry.get_tool_to_toolset_map()
        for tool in ("ha_list_entities", "ha_get_state", "ha_call_service"):
            assert toolset_map[tool] == "homeassistant"

    def test_check_fn_gates_availability(self, monkeypatch):
        """Registry should exclude HA tools when HASS_TOKEN is not set."""
        from tools.registry import registry

        monkeypatch.delenv("HASS_TOKEN", raising=False)
        defs = registry.get_definitions({"ha_list_entities", "ha_get_state", "ha_call_service"})
        assert len(defs) == 0

    def test_check_fn_includes_when_token_set(self, monkeypatch):
        """Registry should include HA tools when HASS_TOKEN is set."""
        from tools.registry import registry

        monkeypatch.setenv("HASS_TOKEN", "test-token")
        defs = registry.get_definitions({"ha_list_entities", "ha_get_state", "ha_call_service"})
        assert len(defs) == 3
