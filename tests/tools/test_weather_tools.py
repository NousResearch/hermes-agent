"""Tests for the restricted Open-Meteo weather and PNG tools."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools import weather_tools


def _sample_payload() -> dict:
    return {
        "source": "Open-Meteo public API (no API key)",
        "resolved_location": {
            "name": "Волгоград",
            "admin1": "Волгоградская область",
            "country": "Россия",
            "timezone": "Europe/Volgograd",
        },
        "current": {
            "time": "2026-07-13T14:00",
            "temperature_2m": 29.4,
            "apparent_temperature": 31.1,
            "relative_humidity_2m": 43,
            "pressure_msl": 1012.7,
            "cloud_cover": 27,
            "wind_speed_10m": 15.8,
            "weather_code": 2,
        },
        "daily": {
            "time": ["2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17"],
            "weather_code": [2, 1, 61, 3, 95],
            "temperature_2m_max": [31.4, 33.2, 27.8, 29.6, 26.1],
            "temperature_2m_min": [20.2, 21.1, 18.6, 19.5, 17.9],
            "precipitation_probability_max": [8, 3, 72, 24, 81],
            "wind_speed_10m_max": [19.4, 16.2, 25.7, 13.9, 31.3],
        },
        "hourly": {
            "time": [f"2026-07-13T{hour:02d}:00" for hour in range(24)],
            "weather_code": [2] * 24,
            "temperature_2m": [20 + hour / 2 for hour in range(24)],
            "apparent_temperature": [19 + hour / 2 for hour in range(24)],
            "precipitation_probability": list(range(24)),
            "relative_humidity_2m": [60 - hour for hour in range(24)],
            "cloud_cover": [hour * 3 for hour in range(24)],
            "wind_speed_10m": [5 + hour / 3 for hour in range(24)],
            "pressure_msl": [1013 - hour / 10 for hour in range(24)],
        },
    }


def test_render_weather_png_creates_valid_image(tmp_path: Path) -> None:
    path = weather_tools._render_weather_png(_sample_payload(), tmp_path, 5)

    assert path.is_absolute()
    assert path.parent == tmp_path
    assert path.suffix == ".png"
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (1200, 900)


def test_get_weather_image_returns_media_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(weather_tools, "_weather_payload", lambda *args, **kwargs: _sample_payload())
    monkeypatch.setattr(weather_tools, "get_hermes_home", lambda: tmp_path)

    result = json.loads(weather_tools.get_weather_image_tool("Волгоград", forecast_days=5))

    assert result["success"] is True
    path = Path(result["path"])
    assert path.exists()
    assert path.parent == tmp_path / "generated" / "weather"
    assert result["media"] == f"MEDIA:{path}"
    assert f"MEDIA:{path}" in result["instruction"]


def test_weather_image_schema_explicitly_matches_picture_requests() -> None:
    description = weather_tools.GET_WEATHER_IMAGE_SCHEMA["description"]
    assert "PNG" in description
    assert "hourly weather table" in description
    assert "forecast_days" not in weather_tools.GET_WEATHER_IMAGE_SCHEMA["parameters"]["properties"]


def test_hourly_table_rows_preserve_july_12_schedule_columns_and_units() -> None:
    rows = weather_tools._hourly_table_rows(_sample_payload())

    assert [row["time"] for row in rows] == [f"{hour:02d}" for hour in range(7, 24)]
    assert set(rows[0]) == {
        "time", "code", "temperature", "apparent", "rain",
        "humidity", "wind", "pressure",
    }
    assert rows[0]["wind"] == 2  # 7.3 km/h rounds to 2 m/s
    assert rows[0]["pressure"] == 759  # hPa converted to mm Hg


def test_render_weather_png_uses_a_light_background(tmp_path: Path) -> None:
    path = weather_tools._render_weather_png(_sample_payload(), tmp_path, 1)

    with Image.open(path) as image:
        red, green, blue = image.getpixel((10, 10))
    assert red + green + blue > 650
