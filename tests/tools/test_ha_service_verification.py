"""Tests for HA service call state verification."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class FakeResponse:
    """Minimal async context manager response for aiohttp."""
    def __init__(self, status, json_data):
        self.status = status
        self._json = json_data

    async def json(self):
        return self._json

    def raise_for_status(self):
        if self.status >= 400:
            raise Exception(f"HTTP {self.status}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeSession:
    """Minimal async context manager session for aiohttp."""
    def __init__(self, responses):
        self._responses = responses
        self._call_idx = 0

    def post(self, url, **kwargs):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp

    def get(self, url, **kwargs):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture(autouse=True)
def mock_ha_config(monkeypatch):
    monkeypatch.setenv("HASS_URL", "http://ha.test:8123")
    monkeypatch.setenv("HASS_TOKEN", "test-token")


class TestServiceCallVerification:
    """Test that service calls verify entity state afterward."""

    @pytest.mark.asyncio
    async def test_successful_call_includes_current_state(self):
        from tools.homeassistant_tool import _async_call_service

        service_resp = FakeResponse(200, [])
        state_resp = FakeResponse(200, {
            "state": "on",
            "attributes": {
                "friendly_name": "Living Room",
                "brightness": 255,
                "color_temp_kelvin": 4000,
            },
        })

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = FakeSession([service_resp, state_resp])
            result = await _async_call_service("light", "turn_on", entity_id="light.living_room")

        assert result["success"] is True
        assert result["current_state"] == "on"
        assert result["friendly_name"] == "Living Room"
        assert result["brightness"] == 255
        assert result["color_temp_kelvin"] == 4000

    @pytest.mark.asyncio
    async def test_nonexistent_entity_returns_error(self):
        from tools.homeassistant_tool import _async_call_service

        service_resp = FakeResponse(200, [])
        state_resp = FakeResponse(404, {})

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = FakeSession([service_resp, state_resp])
            result = await _async_call_service("light", "turn_on", entity_id="light.does_not_exist")

        assert result["success"] is False
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_call_without_entity_skips_verification(self):
        from tools.homeassistant_tool import _async_call_service

        service_resp = FakeResponse(200, [])

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = FakeSession([service_resp])
            result = await _async_call_service("homeassistant", "restart")

        assert result["success"] is True
        assert "current_state" not in result

    @pytest.mark.asyncio
    async def test_verification_timeout_gracefully_handled(self):
        from tools.homeassistant_tool import _async_call_service

        service_resp = FakeResponse(200, [])

        class TimeoutSession(FakeSession):
            def get(self, url, **kwargs):
                raise TimeoutError("verify timed out")

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = TimeoutSession([service_resp])
            result = await _async_call_service("light", "turn_on", entity_id="light.test")

        assert result["success"] is True
        assert result["current_state"] == "unknown (verification failed)"

    @pytest.mark.asyncio
    async def test_climate_attributes_included(self):
        from tools.homeassistant_tool import _async_call_service

        service_resp = FakeResponse(200, [])
        state_resp = FakeResponse(200, {
            "state": "heat",
            "attributes": {
                "friendly_name": "Thermostat",
                "temperature": 21.5,
                "current_temperature": 19.8,
                "hvac_action": "heating",
            },
        })

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = FakeSession([service_resp, state_resp])
            result = await _async_call_service("climate", "set_temperature", entity_id="climate.living_room")

        assert result["current_state"] == "heat"
        assert result["temperature"] == 21.5
        assert result["current_temperature"] == 19.8
        assert result["hvac_action"] == "heating"

    @pytest.mark.asyncio
    async def test_media_player_attributes_included(self):
        from tools.homeassistant_tool import _async_call_service

        service_resp = FakeResponse(200, [])
        state_resp = FakeResponse(200, {
            "state": "playing",
            "attributes": {
                "friendly_name": "Sonos",
                "media_title": "Bohemian Rhapsody",
                "media_artist": "Queen",
                "volume_level": 0.45,
            },
        })

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = FakeSession([service_resp, state_resp])
            result = await _async_call_service("media_player", "play_media", entity_id="media_player.sonos")

        assert result["current_state"] == "playing"
        assert result["media_title"] == "Bohemian Rhapsody"
        assert result["volume_level"] == 0.45
