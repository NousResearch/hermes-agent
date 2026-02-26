"""
Tests for weather tools.
Run with: pytest tests/test_weather.py -v
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock

MOCK_CURRENT = {
    "name": "London", "sys": {"country": "GB", "sunrise": 1708930000, "sunset": 1708970000},
    "main": {"temp": 12.3, "feels_like": 10.1, "temp_min": 9.0, "temp_max": 14.5, "humidity": 78},
    "wind": {"speed": 4.2, "deg": 220}, "weather": [{"description": "overcast clouds"}],
    "visibility": 10000, "clouds": {"all": 90},
}

MOCK_FORECAST = {
    "city": {"name": "Paris", "country": "FR"},
    "list": [
        {"dt_txt": "2026-02-26 12:00:00", "main": {"temp": 8.5, "humidity": 70},
         "weather": [{"description": "light rain"}], "wind": {"speed": 3.0}, "pop": 0.6},
        {"dt_txt": "2026-02-27 12:00:00", "main": {"temp": 11.0, "humidity": 60},
         "weather": [{"description": "partly cloudy"}], "wind": {"speed": 2.5}, "pop": 0.1},
    ],
}

MOCK_GEO = [{"name": "Tokyo", "country": "JP", "lat": 35.6762, "lon": 139.6503}]
MOCK_AQI = {"list": [{"main": {"aqi": 2}, "components": {"co": 201.94, "no2": 0.82, "o3": 68.66, "pm2_5": 0.49, "pm10": 0.54}}]}

def _mock(data):
    m = MagicMock()
    m.read.return_value = json.dumps(data).encode()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    return m

@patch("urllib.request.urlopen")
@patch.dict(os.environ, {"OPENWEATHERMAP_API_KEY": "test_key"})
def test_weather_current(mock_urlopen):
    mock_urlopen.return_value = _mock(MOCK_CURRENT)
    import sys; sys.path.insert(0, "tools")
    exec(open("tools/weather.py").read(), globals())
    result = weather_current("London")
    assert result["location"] == "London"
    assert result["country"] == "GB"
    assert "C" in result["temperature"]
    assert "%" in result["humidity"]
    print("PASS: weather_current")

@patch("urllib.request.urlopen")
@patch.dict(os.environ, {"OPENWEATHERMAP_API_KEY": "test_key"})
def test_weather_forecast(mock_urlopen):
    mock_urlopen.return_value = _mock(MOCK_FORECAST)
    import sys; sys.path.insert(0, "tools")
    exec(open("tools/weather.py").read(), globals())
    result = weather_forecast("Paris", days=2)
    assert result["location"] == "Paris"
    assert len(result["forecast"]) >= 1
    print("PASS: weather_forecast")

def test_missing_api_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENWEATHERMAP_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        import sys; sys.path.insert(0, "tools")
        exec(open("tools/weather.py").read(), globals())
        try:
            _get_api_key()
            assert False, "Should have raised"
        except (ValueError, NameError):
            print("PASS: missing api key raises error")

if __name__ == "__main__":
    test_missing_api_key()
    print("Basic tests passed!")
