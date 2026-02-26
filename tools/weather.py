"""
weather — Get current weather for any city using Open-Meteo (no API key required).
Generated autonomously by Hermes Self Tool Builder skill.
"""

import httpx
from typing import Any

TOOL_NAME = "weather"
TOOL_DESCRIPTION = (
    "Get current weather conditions for any city in the world. "
    "Returns temperature, wind speed, weather condition, and humidity. "
    "No API key required."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "City name (e.g. 'Istanbul', 'London', 'Tokyo')"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature units. Default: celsius",
            "default": "celsius"
        }
    },
    "required": ["city"]
}

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


async def _geocode(city: str) -> tuple[float, float, str]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"}
        )
        resp.raise_for_status()
        data = resp.json()

    if not data.get("results"):
        raise ValueError(f"City not found: '{city}'")

    result = data["results"][0]
    return result["latitude"], result["longitude"], result.get("country", "")


async def run(city: str, units: str = "celsius", **kwargs) -> dict[str, Any]:
    try:
        lat, lon, country = await _geocode(city)

        temp_unit = "celsius" if units != "fahrenheit" else "fahrenheit"
        temp_symbol = "°C" if temp_unit == "celsius" else "°F"

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "wind_speed_10m",
                        "weather_code",
                        "apparent_temperature",
                    ],
                    "temperature_unit": temp_unit,
                    "wind_speed_unit": "kmh",
                    "timezone": "auto",
                }
            )
            resp.raise_for_status()
            data = resp.json()

        current = data["current"]
        weather_code = current.get("weather_code", 0)
        condition = WMO_CODES.get(weather_code, f"Unknown (code {weather_code})")

        return {
            "success": True,
            "result": {
                "city": city,
                "country": country,
                "condition": condition,
                "temperature": f"{current['temperature_2m']}{temp_symbol}",
                "feels_like": f"{current['apparent_temperature']}{temp_symbol}",
                "humidity": f"{current['relative_humidity_2m']}%",
                "wind_speed": f"{current['wind_speed_10m']} km/h",
                "coordinates": {"lat": lat, "lon": lon},
            }
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {type(e).__name__}: {e}"}
