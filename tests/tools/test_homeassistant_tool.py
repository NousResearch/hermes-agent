"""Tests for the Home Assistant tool module.

Tests real logic: entity filtering, payload building, response parsing,
handler validation, and availability gating.
"""

import json
from unittest.mock import AsyncMock, patch

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
    _SERVICE_NAME_RE,
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

    @patch("tools.homeassistant_tool._async_call_service", new_callable=AsyncMock)
    def test_safe_domain_not_blocked(self, mock_call_service):
        """Safe domains like ``light`` reach the service-call layer."""
        mock_call_service.return_value = {"success": True}
        result = json.loads(_handle_call_service({
            "domain": "light", "service": "turn_on", "entity_id": "light.test"
        }))
        assert result["result"]["success"] is True
        mock_call_service.assert_awaited_once_with(
            "light",
            "turn_on",
            "light.test",
            None,
        )

    def test_blocked_domains_include_shell_command(self):
        assert "shell_command" in _BLOCKED_DOMAINS

    def test_blocked_domains_include_hassio(self):
        assert "hassio" in _BLOCKED_DOMAINS


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


    @patch("tools.homeassistant_tool._async_call_service", new_callable=AsyncMock)
    def test_call_service_allows_no_entity_id(self, mock_call_service):
        """Some services (like scene.turn_on) don't need entity_id."""
        mock_call_service.return_value = {"success": True}
        result = json.loads(_handle_call_service({
            "domain": "scene", "service": "turn_on"
        }))
        assert result["result"]["success"] is True
        mock_call_service.assert_awaited_once_with(
            "scene",
            "turn_on",
            None,
            None,
        )


# ---------------------------------------------------------------------------
# String-data deserialization (XML tool calling workaround)
# ---------------------------------------------------------------------------


class TestCallServiceStringData:
    """data param may arrive as a JSON string (XML tool calling mode)."""

    @patch("tools.homeassistant_tool._run_async", return_value={"success": True})
    def test_string_data_deserialized(self, mock_run):
        """JSON string data is parsed into a dict before dispatch."""
        _handle_call_service({
            "domain": "climate",
            "service": "set_hvac_mode",
            "entity_id": "climate.living_room",
            "data": '{"hvac_mode": "heat"}',
        })
        call_args = mock_run.call_args[0][0]  # the coroutine arg
        # _run_async was called, meaning we got past validation


    @patch("tools.homeassistant_tool._run_async", return_value={"success": True})
    def test_empty_string_data_becomes_none(self, mock_run):
        """Empty/whitespace string data is treated as None."""
        _handle_call_service({
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.bedroom",
            "data": "   ",
        })
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Security: domain/service name format validation
# ---------------------------------------------------------------------------


class TestServiceNameValidation:
    """Verify domain/service format validation prevents path traversal in URL.

    The domain and service parameters are interpolated into
    /api/services/{domain}/{service}, so allowing arbitrary strings would
    enable SSRF via path traversal or blocked-domain bypass.
    """

    def test_valid_domain_names(self):
        assert _SERVICE_NAME_RE.match("light")
        assert _SERVICE_NAME_RE.match("switch")
        assert _SERVICE_NAME_RE.match("climate")
        assert _SERVICE_NAME_RE.match("shell_command")
        assert _SERVICE_NAME_RE.match("media_player")


    def test_path_traversal_in_domain_rejected(self):
        assert _SERVICE_NAME_RE.match("../../api/config") is None
        assert _SERVICE_NAME_RE.match("light/../../../etc") is None
        assert _SERVICE_NAME_RE.match("../config") is None

    def test_path_traversal_in_service_rejected(self):
        assert _SERVICE_NAME_RE.match("../../api/config") is None
        assert _SERVICE_NAME_RE.match("turn_on/../../config") is None

    def test_blocked_domain_bypass_via_traversal_rejected(self):
        """Ensure shell_command/../light is rejected, not just checked against blocklist."""
        assert _SERVICE_NAME_RE.match("shell_command/../light") is None
        assert _SERVICE_NAME_RE.match("python_script/../scene") is None
        assert _SERVICE_NAME_RE.match("hassio/../automation") is None


    def test_special_chars_rejected(self):
        assert _SERVICE_NAME_RE.match("light;rm") is None
        assert _SERVICE_NAME_RE.match("light&cmd") is None
        assert _SERVICE_NAME_RE.match("light cmd") is None

    def test_handler_rejects_traversal_domain(self):
        """_handle_call_service must reject domain with path traversal."""
        result = json.loads(_handle_call_service({
            "domain": "../../api/config",
            "service": "turn_on",
        }))
        assert "error" in result
        assert "Invalid domain" in result["error"]

    def test_handler_rejects_traversal_service(self):
        """_handle_call_service must reject service with path traversal."""
        result = json.loads(_handle_call_service({
            "domain": "light",
            "service": "../../api/config",
        }))
        assert "error" in result
        assert "Invalid service" in result["error"]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


class TestCheckAvailable:
    def test_unavailable_without_token(self, monkeypatch):
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        assert _check_ha_available() is False


    def test_empty_token_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("HASS_TOKEN", "")
        assert _check_ha_available() is False

    def test_multiplex_scope_does_not_fall_back_to_another_profile(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("HASS_TOKEN", "default-profile-token")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({})
        try:
            assert _check_ha_available() is False
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

    def test_multiplex_scope_supplies_profile_url_and_token(self, monkeypatch):
        from agent import secret_scope
        from tools.homeassistant_tool import _get_config

        monkeypatch.setattr("tools.homeassistant_tool._HASS_URL", "")
        monkeypatch.setattr("tools.homeassistant_tool._HASS_TOKEN", "")
        monkeypatch.setenv("HASS_URL", "http://default-profile:8123")
        monkeypatch.setenv("HASS_TOKEN", "default-profile-token")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({
            "HASS_URL": "http://secondary-profile:8123/",
            "HASS_TOKEN": "secondary-profile-token",
        })
        try:
            assert _get_config() == (
                "http://secondary-profile:8123",
                "secondary-profile-token",
            )
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)


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


    def test_check_fn_includes_when_token_set(self, monkeypatch):
        """Registry should include HA tools when HASS_TOKEN is set."""
        from tools.registry import invalidate_check_fn_cache, registry

        monkeypatch.setenv("HASS_TOKEN", "test-token")
        invalidate_check_fn_cache()
        defs = registry.get_definitions({"ha_list_entities", "ha_get_state", "ha_call_service"})
        assert len(defs) == 3
