"""Regression tests for ha_list_entities filtering (issue #95438).

``ha_list_entities`` used to accept only ``domain`` and ``area`` and always
summarized every entity it fetched from ``/api/states``.  For a targeted
request such as ``{"domain": "sensor", "area": "Attic"}`` that returned ~72
entities (~9.4 KB), which bloats model context and triggers pointless follow-up
``ha_get_state`` calls.

These tests pin the new, backward-compatible optional filters
(``device_class``, ``name``/``search``, ``entity_id``, ``state``, ``limit``)
plus the small payload additions that let a model use the first result without
a second round trip.
"""

import json

import pytest

from tools.homeassistant_tool import _filter_and_summarize
from tools.registry import registry

# Importing the module registers its tools in the registry.
import tools.homeassistant_tool  # noqa: F401  (side effect: registration)


# ---------------------------------------------------------------------------
# Fixture data — mirrors the issue's "Attic" scenario (~72 sensors)
# ---------------------------------------------------------------------------


def _state(entity_id, state, friendly_name, **attributes):
    attrs = {"friendly_name": friendly_name}
    attrs.update(attributes)
    return {"entity_id": entity_id, "state": state, "attributes": attrs}


def _attic_inventory():
    states = []
    # 6 attic temperature sensors (the ones the user actually wanted)
    for n in range(1, 7):
        states.append(_state(
            f"sensor.attic_temperature_{n}", "21.5", f"Attic Temperature {n}",
            device_class="temperature", unit_of_measurement="°C", area="Attic"))
    # 4 attic humidity sensors
    for n in range(1, 5):
        states.append(_state(
            f"sensor.attic_humidity_{n}", "48", f"Attic Humidity {n}",
            device_class="humidity", unit_of_measurement="%", area="Attic"))
    # 30 assorted attic sensors (power, battery, illuminance, ...)
    for n in range(1, 31):
        states.append(_state(
            f"sensor.attic_misc_{n}", "1.0", f"Attic Sensor {n}",
            device_class="power", unit_of_measurement="W", area="Attic"))
    # 32 more entities that are in the Attic by friendly name / slug only
    for n in range(1, 33):
        states.append(_state(
            f"sensor.attic_extra_{n}", "on", f"Attic Extra {n}", area="Attic"))
    # Non-attic entities that must never leak into an Attic query
    states += [
        _state("sensor.basement_temperature", "19.0", "Basement Temperature",
               device_class="temperature", unit_of_measurement="°C"),
        _state("sensor.kitchen_temperature", "23.0", "Kitchen Temperature",
               device_class="temperature", unit_of_measurement="°C"),
        _state("light.living_room", "on", "Living Room Light"),
        _state("light.kitchen", "off", "Kitchen Light"),
        _state("switch.garage", "off", "Garage Switch"),
        _state("climate.thermostat", "heat", "Main Thermostat", area="Hallway"),
        _state("binary_sensor.hallway_motion", "off", "Hallway Motion"),
        _state("cover.garage_door", "closed", "Garage Door"),
    ]
    return states


ATTIC_STATES = _attic_inventory()

# `sensor.attic_*` entities: 6 + 4 + 30 + 32 = 72, exactly the reported case.
ATTIC_SENSOR_COUNT = 72
ATTIC_TEMPERATURE_COUNT = 6


@pytest.fixture
def ha_api(monkeypatch):
    """Patch the single HA REST call so handlers run their real logic offline."""
    calls = []

    async def _fake_api_json(method, path, timeout, payload=None):
        calls.append((method, path))
        assert method == "GET", f"unexpected method {method}"
        assert path == "/api/states", f"unexpected path {path}"
        return ATTIC_STATES

    monkeypatch.setattr("tools.homeassistant_tool._api_json", _fake_api_json)
    return calls


def _list_entities(**args) -> dict:
    """Invoke the registered ``ha_list_entities`` handler (real dispatch path).

    Returns the handler's ``result`` payload, or the ``{"error": ...}`` body
    for rejected input.
    """
    entry = registry.get_entry("ha_list_entities")
    assert entry is not None, "ha_list_entities is not registered"
    payload = json.loads(entry.handler(dict(args)))
    return payload["result"] if "result" in payload else payload


def _summarize(states, **filters) -> dict:
    """Call ``_filter_and_summarize`` with the new optional filters.

    Falls back to a filter-less call when a filter is unsupported, so an
    unimplemented filter shows up as a *behavioural* failure ("filter ignored")
    rather than a TypeError.
    """
    try:
        return _filter_and_summarize(states, **filters)
    except TypeError:
        return _filter_and_summarize(states)


def _ids(result) -> set:
    return {e["entity_id"] for e in result["entities"]}


# ---------------------------------------------------------------------------
# The reported case: domain+area is far too coarse, the new filters must cut it
# ---------------------------------------------------------------------------


class TestReportedOversizedResult:
    def test_domain_area_alone_is_the_oversized_case(self, ha_api):
        """Baseline for the issue: domain+area still returns the whole Attic."""
        result = _list_entities(domain="sensor", area="Attic")
        assert result["count"] >= ATTIC_SENSOR_COUNT

    def test_device_class_and_limit_shrink_the_result(self, ha_api):
        """The issue's example call: sensor + Attic + temperature + limit 5."""
        result = _list_entities(
            domain="sensor", area="Attic", device_class="temperature", limit=5)

        assert result["count"] == 5
        assert len(result["entities"]) == 5
        for entity in result["entities"]:
            assert entity["device_class"] == "temperature"
            assert entity["entity_id"].startswith("sensor.attic_")

    def test_filtered_payload_is_dramatically_smaller(self, ha_api):
        """The whole point: fewer bytes in the model's context."""
        unfiltered = json.dumps(_list_entities(domain="sensor", area="Attic"))
        filtered = json.dumps(_list_entities(
            domain="sensor", area="Attic", device_class="temperature", limit=5))

        assert len(unfiltered) > 4000, "fixture should reproduce the oversized payload"
        assert len(filtered) < 1500
        assert len(filtered) * 4 < len(unfiltered)

    def test_device_class_filter_applies_without_limit(self, ha_api):
        result = _list_entities(domain="sensor", area="Attic", device_class="temperature")
        assert result["count"] == ATTIC_TEMPERATURE_COUNT
        assert all(e["device_class"] == "temperature" for e in result["entities"])
        assert "sensor.basement_temperature" not in _ids(result)


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------


class TestDeviceClassFilter:
    def test_matches_attribute_case_insensitively(self):
        result = _summarize(ATTIC_STATES, device_class="TEMPERATURE")
        assert "sensor.attic_temperature_1" in _ids(result)
        assert "sensor.attic_humidity_1" not in _ids(result)

    def test_accepts_comma_separated_list(self):
        result = _summarize(ATTIC_STATES, device_class="temperature,humidity")
        ids = _ids(result)
        assert "sensor.attic_temperature_1" in ids
        assert "sensor.attic_humidity_1" in ids
        assert "sensor.attic_misc_1" not in ids

    def test_entities_without_device_class_are_excluded(self):
        states = [
            _state("sensor.plain", "1", "Plain Sensor"),
            _state("sensor.temp", "1", "Thermo", device_class="temperature"),
        ]
        result = _summarize(states, device_class="temperature")
        assert _ids(result) == {"sensor.temp"}


class TestNameFilter:
    def test_substring_match_case_insensitive(self):
        result = _summarize(ATTIC_STATES, name="humidity")
        ids = _ids(result)
        assert ids == {f"sensor.attic_humidity_{n}" for n in range(1, 5)}

    def test_search_is_accepted_as_an_alias(self, ha_api):
        result = _list_entities(domain="sensor", search="Attic Temperature 3")
        assert _ids(result) == {"sensor.attic_temperature_3"}

    def test_no_match_returns_empty_result(self):
        result = _summarize(ATTIC_STATES, name="nonexistent widget")
        assert result["count"] == 0
        assert result["entities"] == []


class TestEntityIdFilter:
    def test_exact_match(self):
        result = _summarize(ATTIC_STATES, entity_id="sensor.attic_humidity_2")
        assert _ids(result) == {"sensor.attic_humidity_2"}

    def test_prefix_match_on_object_id(self):
        result = _summarize(ATTIC_STATES, entity_id="sensor.attic_humidity")
        assert _ids(result) == {f"sensor.attic_humidity_{n}" for n in range(1, 5)}

    def test_glob_match(self):
        result = _summarize(ATTIC_STATES, entity_id="sensor.attic_temperature_?")
        assert _ids(result) == {f"sensor.attic_temperature_{n}" for n in range(1, 7)}

    def test_comma_separated_ids(self):
        result = _summarize(
            ATTIC_STATES, entity_id="sensor.attic_humidity_1,sensor.attic_humidity_2")
        assert _ids(result) == {"sensor.attic_humidity_1", "sensor.attic_humidity_2"}

    def test_wrong_domain_is_not_matched_by_object_id(self):
        result = _summarize(ATTIC_STATES, entity_id="light.attic_humidity_1")
        assert result["count"] == 0

    def test_extra_ids_are_rejected_by_handler(self, ha_api):
        result = _list_entities(entity_id="sensor.attic_humidity_1,not an entity id")
        assert _ids(result) == {"sensor.attic_humidity_1"}


class TestStateFilter:
    def test_exact_state(self):
        result = _summarize(ATTIC_STATES, state="off")
        assert _ids(result) == {
            "light.kitchen", "switch.garage", "binary_sensor.hallway_motion"}

    def test_comma_separated_states(self):
        result = _summarize(ATTIC_STATES, state="heat,closed")
        assert _ids(result) == {"climate.thermostat", "cover.garage_door"}

    def test_state_matches_sensor_reading(self):
        result = _summarize(ATTIC_STATES, device_class="temperature", state="19.0")
        assert _ids(result) == {"sensor.basement_temperature"}


class TestAreaFilter:
    def test_matches_friendly_name_and_area_attribute(self):
        result = _summarize(ATTIC_STATES, area="attic")
        assert "sensor.attic_temperature_1" in _ids(result)
        assert "sensor.basement_temperature" not in _ids(result)

    def test_matches_entity_id_slug(self):
        """Area filtering must not rely on friendly_name alone (issue text)."""
        states = [_state("sensor.attic_co2", "600", "CO2 Level")]
        result = _summarize(states, area="Attic")
        assert _ids(result) == {"sensor.attic_co2"}

    def test_multiword_area_matches_underscored_slug(self):
        states = [_state("light.living_room_lamp", "on", "Lamp")]
        result = _summarize(states, area="living room")
        assert _ids(result) == {"light.living_room_lamp"}


class TestLimit:
    def test_limit_caps_returned_entities(self):
        result = _summarize(ATTIC_STATES, limit=3)
        assert result["count"] == 3
        assert len(result["entities"]) == 3

    def test_limit_reports_truncation(self):
        result = _summarize(ATTIC_STATES, domain="sensor", area="attic", limit=5)
        assert result["count"] == 5
        assert result["total_matched"] == ATTIC_SENSOR_COUNT
        assert result["truncated"] is True

    def test_limit_larger_than_matches_is_not_truncated(self):
        result = _summarize(ATTIC_STATES, domain="light", limit=50)
        assert result["count"] == 2
        assert result["total_matched"] == 2
        assert result["truncated"] is False

    def test_limit_applies_after_other_filters(self):
        result = _summarize(
            ATTIC_STATES, domain="sensor", device_class="temperature", limit=2)
        assert result["count"] == 2
        assert all(e["device_class"] == "temperature" for e in result["entities"])

    def test_handler_accepts_string_limit(self, ha_api):
        """XML tool-calling mode delivers numeric args as strings."""
        result = _list_entities(domain="sensor", area="attic", limit="4")
        assert result["count"] == 4


class TestFiltersCombine:
    def test_all_filters_together(self, ha_api):
        result = _list_entities(
            domain="sensor", area="attic", device_class="temperature",
            name="Attic Temperature", state="21.5", limit=2)
        assert result["count"] == 2
        for entity in result["entities"]:
            assert entity["device_class"] == "temperature"
            assert "temperature" in entity["friendly_name"].lower()

    def test_conflicting_filters_return_empty(self, ha_api):
        result = _list_entities(domain="sensor", area="attic", state="heat")
        assert result["count"] == 0
        assert result["entities"] == []


# ---------------------------------------------------------------------------
# Validation and backward compatibility
# ---------------------------------------------------------------------------


class TestLimitValidation:
    @pytest.mark.parametrize("bad_limit", [0, -3, "abc", "", True, 2.5, [], {}])
    def test_invalid_limit_is_a_tool_error(self, ha_api, bad_limit):
        result = _list_entities(limit=bad_limit)
        assert "error" in result, f"limit={bad_limit!r} should be rejected"
        assert "limit" in result["error"].lower()

    def test_non_string_filter_is_rejected(self, ha_api):
        result = _list_entities(device_class={"nested": "dict"})
        assert "error" in result


class TestBackwardCompatibility:
    def test_no_filters_still_returns_everything(self, ha_api):
        result = _list_entities()
        assert result["count"] == len(ATTIC_STATES)
        assert "truncated" not in result

    def test_domain_filter_unchanged(self):
        result = _summarize(ATTIC_STATES, domain="light")
        assert _ids(result) == {"light.living_room", "light.kitchen"}

    def test_domain_and_area_still_work(self, ha_api):
        result = _list_entities(domain="sensor", area="Attic")
        ids = _ids(result)
        assert "sensor.attic_temperature_1" in ids
        assert "sensor.basement_temperature" not in ids
        assert "light.living_room" not in ids

    def test_blank_filters_are_ignored(self, ha_api):
        result = _list_entities(domain="", area="", device_class="", name="", limit=None)
        assert result["count"] == len(ATTIC_STATES)

    def test_empty_state_list(self, ha_api, monkeypatch):
        async def _empty(method, path, timeout, payload=None):
            return []
        monkeypatch.setattr("tools.homeassistant_tool._api_json", _empty)
        result = _list_entities(domain="sensor", device_class="temperature", limit=5)
        assert result["count"] == 0
        assert result["entities"] == []
        assert result["total_matched"] == 0
        assert result["truncated"] is False

    def test_missing_attributes_are_tolerated(self, ha_api, monkeypatch):
        async def _one(method, path, timeout, payload=None):
            return [{"entity_id": "sensor.bare", "state": "1"}]
        monkeypatch.setattr("tools.homeassistant_tool._api_json", _one)
        result = _list_entities(device_class="temperature")
        assert result["count"] == 0
        result = _list_entities(name="anything")
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# Payload content + schema discovery
# ---------------------------------------------------------------------------


class TestEntityPayload:
    def test_device_class_and_unit_help_use_the_first_result(self, ha_api):
        """No extra ha_get_state round trip: the reading's meaning is visible."""
        result = _list_entities(
            domain="sensor", area="attic", device_class="temperature", limit=1)
        entity = result["entities"][0]
        assert entity["device_class"] == "temperature"
        assert entity["unit_of_measurement"] == "°C"
        assert entity["state"] == "21.5"

    def test_entities_without_extra_attributes_stay_lean(self):
        result = _summarize(ATTIC_STATES, domain="light")
        for entity in result["entities"]:
            assert "device_class" not in entity
            assert "unit_of_measurement" not in entity


class TestSchema:
    def test_schema_advertises_new_filters(self):
        schema = registry.get_schema("ha_list_entities")
        props = schema["parameters"]["properties"]
        for name in ("domain", "area", "device_class", "name", "entity_id", "state", "limit"):
            assert name in props, f"{name} missing from ha_list_entities schema"

    def test_all_filters_remain_optional(self):
        schema = registry.get_schema("ha_list_entities")
        assert schema["parameters"]["required"] == []

    def test_limit_is_declared_as_an_integer(self):
        schema = registry.get_schema("ha_list_entities")
        assert schema["parameters"]["properties"]["limit"]["type"] == "integer"

    def test_description_mentions_the_new_filters(self):
        schema = registry.get_schema("ha_list_entities")
        description = schema["description"].lower()
        assert "device_class" in description
        assert "limit" in description
