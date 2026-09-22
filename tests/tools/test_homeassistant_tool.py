"""Tests for the Home Assistant tool module.

Tests real logic: entity filtering, payload building, response parsing,
handler validation, and availability gating.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

import tools.homeassistant_tool as ha_tool
from tools.homeassistant_tool import (
    _check_ha_available,
    _filter_and_summarize,
    _build_service_payload,
    _parse_service_response,
    _get_headers,
    _get_entity_filter_config,
    _parse_entity_filter_patterns,
    _entity_matches,
    _apply_operator_filters,
    _operator_filter_violation,
    _handle_get_state,
    _handle_call_service,
    _handle_list_entities,
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
        monkeypatch.setattr("tools.homeassistant_tool._get_config",
                            lambda: ("http://ha.local:8123", "my-secret-token"))
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


# ---------------------------------------------------------------------------
# Tool schema: the description must steer the model toward targeted queries
# ---------------------------------------------------------------------------


class TestListEntitiesSchema:
    def test_schema_advertises_filters(self):
        schema = ha_tool.HA_LIST_ENTITIES_SCHEMA
        props = schema["parameters"]["properties"]
        for key in ("domain", "area", "name", "entity_ids", "max"):
            assert key in props, f"schema missing {key!r}"

    def test_description_steers_to_filters_without_asserting_a_fixed_cap(self):
        """The old 'Omit to list all entities' framing was the footgun; but the
        cap is operator-configurable, so the description must not present a
        hardcoded number as a guarantee either."""
        desc = ha_tool.HA_LIST_ENTITIES_SCHEMA["description"].lower()
        assert "filter" in desc
        assert "truncated" in desc
        assert "omit to list all" not in desc
        assert "returns only the first 100" not in desc  # 100 is no longer a fact
        assert "hass_max_entities" in desc  # the cap is mentioned as operator config

    def test_entity_ids_schema_is_a_string_array(self):
        assert ha_tool.HA_LIST_ENTITIES_SCHEMA["parameters"]["properties"]["entity_ids"]["type"] == "array"
        assert ha_tool.HA_LIST_ENTITIES_SCHEMA["parameters"]["properties"]["entity_ids"]["items"] == {"type": "string"}


# ---------------------------------------------------------------------------
# Targeted queries: entity-id list, name/area match, cap, operator lists
# ---------------------------------------------------------------------------

def _reset_filter_mirrors(monkeypatch):
    """Clear the module-level allow/deny/cap mirrors + env so each test is isolated."""
    monkeypatch.setattr(ha_tool, "_HASS_ENTITY_ALLOWLIST", "")
    monkeypatch.setattr(ha_tool, "_HASS_ENTITY_DENYLIST", "")
    monkeypatch.setattr(ha_tool, "_HASS_MAX_ENTITIES", "")
    monkeypatch.delenv("HASS_ENTITY_ALLOWLIST", raising=False)
    monkeypatch.delenv("HASS_ENTITY_DENYLIST", raising=False)
    monkeypatch.delenv("HASS_MAX_ENTITIES", raising=False)


# A realistic-ish payload with a zombie aux-entity cluster left behind by a
# device migration, plus a live duplicate of the same physical room.
ZOMBY_STATES = [
    {"entity_id": "climate.office", "state": "heat",
     "attributes": {"friendly_name": "Office Thermostat", "current_temperature": 28.0, "temperature": 28.0}},
    {"entity_id": "office_thermostat.0_external_temperature", "state": "0.0",
     "attributes": {"friendly_name": "Office Thermostat External Temperature"}},
    {"entity_id": "office_thermostat.0_valve_position", "state": "0",
     "attributes": {"friendly_name": "Office Thermostat Valve Position"}},
    {"entity_id": "climate.office_zombie", "state": "off",
     "attributes": {"friendly_name": "Office Thermostat (duplicate)", "current_temperature": 22.0}},
    {"entity_id": "sensor.kitchen_temp", "state": "24.0",
     "attributes": {"friendly_name": "Kitchen Temperature"}},
]


class TestTargetedEntityIdFilter:
    def test_entity_ids_returns_only_requested(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        result = _filter_and_summarize(ZOMBY_STATES, entity_ids=["climate.office", "sensor.kitchen_temp"])
        assert result["count"] == 2
        ids = {e["entity_id"] for e in result["entities"]}
        assert ids == {"climate.office", "sensor.kitchen_temp"}

    def test_entity_ids_accepts_comma_string(self, monkeypatch):
        """The XML/LLM tool-calling path may deliver the list as a string."""
        _reset_filter_mirrors(monkeypatch)
        result = _filter_and_summarize(ZOMBY_STATES, entity_ids="climate.office, sensor.kitchen_temp")
        assert result["count"] == 2
        assert {e["entity_id"] for e in result["entities"]} == {"climate.office", "sensor.kitchen_temp"}

    def test_entity_ids_supersedes_domain(self, monkeypatch):
        """An explicit id list wins over a broader co-passed domain filter."""
        _reset_filter_mirrors(monkeypatch)
        result = _filter_and_summarize(ZOMBY_STATES, domain="light", entity_ids=["sensor.kitchen_temp"])
        assert [e["entity_id"] for e in result["entities"]] == ["sensor.kitchen_temp"]

    def test_handler_passes_entity_ids_through(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        with patch.object(ha_tool, "_run_async", return_value={"count": 1, "entities": []}) as run:
            _handle_list_entities({"entity_ids": ["climate.office"]})
            coro = run.call_args[0][0]
            try:
                assert coro.cr_frame.f_locals["entity_ids"] == ["climate.office"]
            finally:
                coro.close()


class TestNameAreaFilter:
    def test_name_substring_matches_friendly_name_and_entity_id(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        result = _filter_and_summarize(ZOMBY_STATES, name="kitchen")
        assert [e["entity_id"] for e in result["entities"]] == ["sensor.kitchen_temp"]

    def test_name_matches_entity_id_when_no_friendly_name(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        states = [{"entity_id": "sensor.office_co2", "state": "600", "attributes": {}}]
        result = _filter_and_summarize(states, name="office")
        assert [e["entity_id"] for e in result["entities"]] == ["sensor.office_co2"]

    def test_area_filter_unaffected_by_name_param(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        states = [
            {"entity_id": "switch.a", "state": "on", "attributes": {"area": "kitchen"}},
            {"entity_id": "switch.b", "state": "on", "attributes": {"area": "garage"}},
        ]
        result = _filter_and_summarize(states, area="kitchen")
        assert [e["entity_id"] for e in result["entities"]] == ["switch.a"]


class TestMaxEntitiesConfig:
    def test_unset_env_means_no_cap_returns_everything(self, monkeypatch):
        """Compatibility guarantee: unset = current upstream behaviour."""
        _reset_filter_mirrors(monkeypatch)
        big = [
            {"entity_id": f"sensor.idx_{i}", "state": "0", "attributes": {}}
            for i in range(2500)
        ]
        result = _filter_and_summarize(big)
        assert result["count"] == 2500
        assert "truncated" not in result

    def test_env_set_caps_and_marks_truncated(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_MAX_ENTITIES", "100")
        big = [
            {"entity_id": f"sensor.idx_{i}", "state": "0", "attributes": {}}
            for i in range(2500)
        ]
        result = _filter_and_summarize(big)
        assert result["count"] == 100
        assert result["truncated"] is True
        assert "2500" in result["truncated_note"]

    def test_env_set_below_count_no_truncation_flag(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_MAX_ENTITIES", "10")
        result = _filter_and_summarize(ZOMBY_STATES)
        assert result["count"] == len(ZOMBY_STATES)
        assert "truncated" not in result

    def test_env_set_still_caps_a_filtered_query(self, monkeypatch):
        """The cap applies even when domain/name filters are present."""
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_MAX_ENTITIES", "200")
        big = [
            {"entity_id": f"sensor.idx_{i}", "state": "0", "attributes": {}}
            for i in range(250)
        ]
        result = _filter_and_summarize(big, domain="sensor")
        assert result["count"] == 200
        assert result["truncated"] is True
        assert "250" in result["truncated_note"]

    def test_explicit_max_overrides_env(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_MAX_ENTITIES", "100")
        big = [
            {"entity_id": f"sensor.idx_{i}", "state": "0", "attributes": {}}
            for i in range(300)
        ]
        result = _filter_and_summarize(big, max_entities=50)
        assert result["count"] == 50
        assert result["truncated"] is True

    def test_handler_coerces_string_max(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        with patch.object(ha_tool, "_run_async", return_value={"count": 1, "entities": []}) as run:
            _handle_list_entities({"max": "50"})
            coro = run.call_args[0][0]
            try:
                assert coro.cr_frame.f_locals["max_entities"] == 50
            finally:
                coro.close()

    def test_handler_rejects_non_integer_max(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        result = json.loads(_handle_list_entities({"max": "lots"}))
        assert "error" in result
        assert "max" in result["error"].lower()

    def test_get_max_entities_config_reads_env(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        assert ha_tool._get_max_entities_config() is None
        monkeypatch.setenv("HASS_MAX_ENTITIES", "250")
        assert ha_tool._get_max_entities_config() == 250

    def test_get_max_entities_config_rejects_bad_values(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_MAX_ENTITIES", "not-a-number")
        assert ha_tool._get_max_entities_config() is None
        monkeypatch.setenv("HASS_MAX_ENTITIES", "0")
        assert ha_tool._get_max_entities_config() is None
        monkeypatch.setenv("HASS_MAX_ENTITIES", "-5")
        assert ha_tool._get_max_entities_config() is None


class TestOperatorAllowDeny:
    def test_denylist_excludes_zombie_cluster(self, monkeypatch):
        """Zombie entities left behind by a device migration are hidden by config."""
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_DENYLIST", "office_thermostat_*")
        result = _filter_and_summarize(ZOMBY_STATES)
        ids = {e["entity_id"] for e in result["entities"]}
        assert "office_thermostat.0_external_temperature" not in ids
        assert "office_thermostat.0_valve_position" not in ids
        assert "climate.office" in ids  # live entity with different id survives

    def test_denylist_vetoes_even_when_allowlisted(self, monkeypatch):
        """Denylist is a final veto: an entity in both lists is excluded."""
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_ALLOWLIST", "climate.*")
        monkeypatch.setenv("HASS_ENTITY_DENYLIST", "climate.office_zombie")
        result = _filter_and_summarize(ZOMBY_STATES)
        ids = {e["entity_id"] for e in result["entities"]}
        assert ids == {"climate.office"}  # zombie in both lists dropped; non-allowlisted dropped

    def test_allowlist_only_keeps_matched(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_ALLOWLIST", "sensor.kitchen_temp")
        result = _filter_and_summarize(ZOMBY_STATES)
        assert [e["entity_id"] for e in result["entities"]] == ["sensor.kitchen_temp"]

    def test_patterns_parsed_from_env(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_DENYLIST", "  a.b , c.d  ")
        allow, deny = _get_entity_filter_config()
        assert allow == []
        assert deny == ["a.b", "c.d"]

    def test_apply_operator_filters_noops_when_empty(self):
        out = _apply_operator_filters(ZOMBY_STATES, [], [])
        assert out is ZOMBY_STATES

    def test_entity_matches_prefix_exact_and_glob(self):
        assert _entity_matches("office_thermostat.0_external_temperature", ["office_thermostat_*"])
        assert _entity_matches("office_thermostat.0", ["office_thermostat.0"])
        assert _entity_matches("office_thermostat.0", ["office_thermostat."])
        assert not _entity_matches("sensor.other", ["office_thermostat.*"])


# ---------------------------------------------------------------------------
# Single-entity read path (ha_get_state) honors the operator allow/deny lists
# ---------------------------------------------------------------------------


class TestGetStateOperatorFilters:
    """A denylisted zombie must not be readable via ha_get_state either."""

    def test_denylisted_entity_is_refused_with_clear_error(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_DENYLIST", "office_thermostat_*")
        # Refused BEFORE any network call — _run_async must not be reached.
        with patch.object(ha_tool, "_run_async") as run:
            result = json.loads(_handle_get_state(
                {"entity_id": "office_thermostat.0_external_temperature"}
            ))
        assert "error" in result
        assert "operator config" in result["error"]
        assert "HASS_ENTITY_DENYLIST" in result["error"]
        run.assert_not_called()

    def test_allowlist_blocks_non_listed_entity(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_ALLOWLIST", "climate.*")
        with patch.object(ha_tool, "_run_async") as run:
            result = json.loads(_handle_get_state({"entity_id": "sensor.kitchen_temp"}))
        assert "error" in result
        assert "HASS_ENTITY_ALLOWLIST" in result["error"]
        run.assert_not_called()

    def test_denylist_vetoes_allowlisted_entity_on_read(self, monkeypatch):
        """Consistent with the list path: deny is a final veto."""
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_ALLOWLIST", "climate.*")
        monkeypatch.setenv("HASS_ENTITY_DENYLIST", "climate.office_zombie")
        with patch.object(ha_tool, "_run_async") as run:
            result = json.loads(_handle_get_state({"entity_id": "climate.office_zombie"}))
        assert "error" in result
        assert "HASS_ENTITY_DENYLIST" in result["error"]
        run.assert_not_called()

    def test_normal_entity_still_reads(self, monkeypatch):
        """No operator config: a valid entity reaches the fetch layer."""
        _reset_filter_mirrors(monkeypatch)
        with patch.object(ha_tool, "_async_get_state", new_callable=AsyncMock) as fetch:
            fetch.return_value = {"entity_id": "climate.office", "state": "heat"}
            result = json.loads(_handle_get_state({"entity_id": "climate.office"}))
        assert result["result"]["entity_id"] == "climate.office"
        fetch.assert_awaited_once_with("climate.office")

    def test_operator_configured_but_matched_entity_reads(self, monkeypatch):
        """With a denylist set, an entity that does NOT match it still reads."""
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_DENYLIST", "office_thermostat_*")
        with patch.object(ha_tool, "_async_get_state", new_callable=AsyncMock) as fetch:
            fetch.return_value = {"entity_id": "climate.office", "state": "heat",
                                  "current_temperature": 28.0}
            result = json.loads(_handle_get_state({"entity_id": "climate.office"}))
        assert result["result"]["entity_id"] == "climate.office"
        fetch.assert_awaited_once_with("climate.office")

    def test_violation_helper_returns_none_when_unset(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        assert _operator_filter_violation("sensor.anything") is None

    def test_violation_helper_reports_deny_then_allow(self, monkeypatch):
        _reset_filter_mirrors(monkeypatch)
        monkeypatch.setenv("HASS_ENTITY_ALLOWLIST", "climate.*")
        assert _operator_filter_violation("sensor.x") == "not in HASS_ENTITY_ALLOWLIST"
        monkeypatch.setenv("HASS_ENTITY_DENYLIST", "office_thermostat_*")
        assert _operator_filter_violation("office_thermostat.0_valve_position") == (
            "excluded by HASS_ENTITY_DENYLIST"
        )
