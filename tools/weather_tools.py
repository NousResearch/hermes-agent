#!/usr/bin/env python3
"""No-key weather tools backed by public Open-Meteo APIs.

The tools are intentionally narrow: they accept a human location string,
resolve it through Open-Meteo geocoding, and either return weather JSON or
render a forecast PNG into the active Hermes profile's generated-media cache.
They do not expose arbitrary URL fetching, shell, filesystem, or browser access,
which makes them suitable for restricted group profiles.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from hermes_constants import get_hermes_home
from tools.registry import registry


_OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
_OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_SECONDS = 20
_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_WMO_DESCRIPTIONS_RU = {
    0: "Ясно",
    1: "Преимущественно ясно",
    2: "Переменная облачность",
    3: "Пасмурно",
    45: "Туман",
    48: "Изморозь и туман",
    51: "Лёгкая морось",
    53: "Морось",
    55: "Сильная морось",
    56: "Лёгкая ледяная морось",
    57: "Ледяная морось",
    61: "Небольшой дождь",
    63: "Дождь",
    65: "Сильный дождь",
    66: "Лёгкий ледяной дождь",
    67: "Сильный ледяной дождь",
    71: "Небольшой снег",
    73: "Снег",
    75: "Сильный снег",
    77: "Снежная крупа",
    80: "Небольшой ливень",
    81: "Ливень",
    82: "Сильный ливень",
    85: "Небольшой снегопад",
    86: "Сильный снегопад",
    95: "Гроза",
    96: "Гроза с градом",
    99: "Сильная гроза с градом",
}

_WEEKDAYS_RU = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
_MONTHS_RU = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)


def _fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HermesAgentWeatherTool/2.0 (+https://github.com/NousResearch/hermes-agent)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        payload = resp.read().decode(charset, errors="replace")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Weather API returned a non-object JSON payload")
    return data


def _first_result(results: Any) -> dict[str, Any] | None:
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    return first if isinstance(first, dict) else None


def _weather_code_description(code: Any, language: str = "en") -> str | None:
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return None
    descriptions = _WMO_DESCRIPTIONS_RU if language == "ru" else _WMO_DESCRIPTIONS
    return descriptions.get(code_int)


def _weather_payload(location: str, language: str, forecast_days: int) -> dict[str, Any]:
    location = (location or "").strip()
    if not location:
        raise ValueError("location is required")

    language = (language or "ru").strip().lower()[:2] or "ru"
    forecast_days = max(1, min(int(forecast_days), 7))
    geocode_query = urllib.parse.urlencode(
        {"name": location, "count": 1, "language": language, "format": "json"}
    )
    place = _first_result(_fetch_json(f"{_OPEN_METEO_GEOCODE}?{geocode_query}").get("results"))
    if not place:
        raise LookupError(f"Could not geocode location: {location}")

    latitude = place.get("latitude")
    longitude = place.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError("Geocoding result did not include latitude/longitude")

    current_fields = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature",
        "is_day", "precipitation", "rain", "showers", "snowfall",
        "weather_code", "cloud_cover", "pressure_msl", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    ]
    daily_fields = [
        "weather_code", "temperature_2m_max", "temperature_2m_min",
        "precipitation_probability_max", "precipitation_sum",
        "wind_speed_10m_max", "wind_gusts_10m_max", "sunrise", "sunset",
    ]
    hourly_fields = [
        "temperature_2m", "apparent_temperature", "precipitation_probability",
        "relative_humidity_2m", "cloud_cover", "wind_speed_10m",
        "pressure_msl", "weather_code",
    ]
    forecast_query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(current_fields),
            "daily": ",".join(daily_fields),
            "hourly": ",".join(hourly_fields),
            "timezone": "auto",
            "forecast_days": forecast_days,
        }
    )
    forecast = _fetch_json(f"{_OPEN_METEO_FORECAST}?{forecast_query}")
    raw_current = forecast.get("current")
    current: dict[str, Any] = raw_current if isinstance(raw_current, dict) else {}
    return {
        "source": "Open-Meteo public API (no API key)",
        "location_query": location,
        "resolved_location": {
            "name": place.get("name"),
            "admin1": place.get("admin1"),
            "country": place.get("country"),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": forecast.get("timezone") or place.get("timezone"),
        },
        "current": current,
        "current_units": forecast.get("current_units") if isinstance(forecast.get("current_units"), dict) else {},
        "daily": forecast.get("daily") if isinstance(forecast.get("daily"), dict) else {},
        "daily_units": forecast.get("daily_units") if isinstance(forecast.get("daily_units"), dict) else {},
        "hourly": forecast.get("hourly") if isinstance(forecast.get("hourly"), dict) else {},
        "hourly_units": forecast.get("hourly_units") if isinstance(forecast.get("hourly_units"), dict) else {},
        "weather_description": _weather_code_description(current.get("weather_code"), language),
    }


def get_weather_tool(location: str, *, language: str = "ru") -> str:
    """Return current weather for *location* as a JSON string."""
    try:
        result = _weather_payload(location, language, 1)
        return json.dumps(result, ensure_ascii=False)
    except urllib.error.URLError as exc:
        return json.dumps({"error": f"Network error while fetching weather: {exc}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Weather lookup failed: {type(exc).__name__}: {exc}"}, ensure_ascii=False)


def _int(value: Any, default: int = 0) -> int:
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return default


def _daily_value(daily: dict[str, Any], field: str, index: int, default: Any = None) -> Any:
    values = daily.get(field)
    if not isinstance(values, list) or index >= len(values):
        return default
    return values[index]


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    font_path = _FONT_BOLD if bold else _FONT_REGULAR
    return ImageFont.truetype(font_path, size=size)


def _rounded_panel(draw: Any, box: tuple[int, int, int, int], *, fill: tuple[int, ...], radius: int = 28) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def _draw_weather_icon(draw: Any, center: tuple[int, int], code: int, scale: float = 1.0) -> None:
    """Draw a compact weather glyph without relying on emoji fonts."""
    x, y = center
    sun = code in (0, 1, 2)
    cloud = code not in (0,)
    rain = code in set(range(51, 68)) | {80, 81, 82, 95, 96, 99}
    snow = code in {71, 73, 75, 77, 85, 86}
    fog = code in {45, 48}

    if sun:
        radius = int(23 * scale)
        sx, sy = x - int(17 * scale), y - int(10 * scale)
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=(255, 199, 74, 255))
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            r1, r2 = int(32 * scale), int(43 * scale)
            draw.line(
                (sx + math.cos(rad) * r1, sy + math.sin(rad) * r1,
                 sx + math.cos(rad) * r2, sy + math.sin(rad) * r2),
                fill=(255, 214, 111, 255), width=max(2, int(4 * scale)),
            )

    if cloud:
        cy = y + int(4 * scale)
        color = (225, 235, 246, 255) if code != 3 else (182, 198, 218, 255)
        draw.ellipse((x - int(45 * scale), cy - int(13 * scale), x + int(42 * scale), cy + int(27 * scale)), fill=color)
        draw.ellipse((x - int(32 * scale), cy - int(37 * scale), x + int(10 * scale), cy + int(18 * scale)), fill=color)
        draw.ellipse((x - int(5 * scale), cy - int(28 * scale), x + int(35 * scale), cy + int(20 * scale)), fill=color)

    if rain:
        for dx in (-28, 0, 28):
            draw.line((x + int(dx * scale), y + int(37 * scale), x + int((dx - 7) * scale), y + int(54 * scale)), fill=(77, 174, 255, 255), width=max(3, int(5 * scale)))
    if snow:
        for dx in (-25, 4, 31):
            px, py = x + int(dx * scale), y + int(47 * scale)
            r = max(2, int(4 * scale))
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(235, 247, 255, 255))
    if fog:
        for dy in (-14, 4, 22):
            draw.line((x - int(45 * scale), y + int(dy * scale), x + int(45 * scale), y + int(dy * scale)), fill=(206, 219, 232, 255), width=max(2, int(4 * scale)))


def _place_label(place: dict[str, Any]) -> str:
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    clean: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in clean:
            clean.append(text)
    return ", ".join(clean) or "Прогноз погоды"


def _render_weather_card_png(payload: dict[str, Any], output_dir: Path, forecast_days: int) -> Path:
    from PIL import Image, ImageDraw

    width, height = 1200, 900
    image = Image.new("RGBA", (width, height), (16, 28, 48, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = (
            round(20 + 16 * ratio),
            round(43 + 10 * ratio),
            round(73 + 22 * ratio),
            255,
        )
        draw.line((0, y, width, y), fill=color)

    # Soft ambient highlights.
    draw.ellipse((760, -360, 1370, 260), fill=(65, 150, 230, 35))
    draw.ellipse((-250, 560, 430, 1160), fill=(35, 204, 170, 22))

    raw_place = payload.get("resolved_location")
    raw_current = payload.get("current")
    raw_daily = payload.get("daily")
    place: dict[str, Any] = raw_place if isinstance(raw_place, dict) else {}
    current: dict[str, Any] = raw_current if isinstance(raw_current, dict) else {}
    daily: dict[str, Any] = raw_daily if isinstance(raw_daily, dict) else {}
    code = _int(current.get("weather_code"), 0)

    place_label = _place_label(place)
    title_size = 46
    title_font = _font(title_size, bold=True)
    while title_size > 28 and draw.textbbox((0, 0), place_label, font=title_font)[2] > width - 124:
        title_size -= 2
        title_font = _font(title_size, bold=True)
    draw.text((62, 48), place_label, font=title_font, fill=(247, 250, 255, 255))
    updated = str(current.get("time") or "").replace("T", " · ")
    timezone = str(place.get("timezone") or "")
    subtitle = "Прогноз Open‑Meteo"
    if updated:
        subtitle += f"  ·  {updated}"
    if timezone:
        subtitle += f"  ·  {timezone}"
    draw.text((64, 108), subtitle, font=_font(20), fill=(170, 194, 220, 255))

    _rounded_panel(draw, (52, 158, 1148, 377), fill=(8, 21, 39, 150), radius=32)
    _draw_weather_icon(draw, (170, 264), code, 1.35)
    temperature = _int(current.get("temperature_2m"))
    apparent = _int(current.get("apparent_temperature"))
    draw.text((270, 190), f"{temperature:+d}°", font=_font(82, bold=True), fill=(255, 255, 255, 255))
    description = _weather_code_description(code, "ru") or "Нет данных"
    draw.text((274, 292), description, font=_font(27, bold=True), fill=(211, 227, 244, 255))
    draw.text((274, 333), f"Ощущается как {apparent:+d}°", font=_font(19), fill=(156, 181, 208, 255))

    metrics = [
        ("Влажность, %", _int(current.get("relative_humidity_2m"))),
        ("Давление, гПа", f"{_int(current.get('pressure_msl')):,}".replace(",", " ")),
        ("Облачность, %", _int(current.get("cloud_cover"))),
        ("Ветер, км/ч", _int(current.get("wind_speed_10m"))),
    ]
    mx = 660
    for idx, (label, value) in enumerate(metrics):
        col, row = idx % 2, idx // 2
        x, y = mx + col * 245, 190 + row * 87
        draw.text((x, y), label, font=_font(16), fill=(143, 169, 196, 255))
        draw.text((x, y + 28), str(value), font=_font(31, bold=True), fill=(242, 247, 253, 255))

    raw_times = daily.get("time")
    times: list[Any] = raw_times if isinstance(raw_times, list) else []
    count = min(max(1, forecast_days), len(times), 7)
    if count == 0:
        raise ValueError("Open-Meteo response did not include daily forecast data")
    draw.text((58, 410), f"Прогноз на {count} дн.", font=_font(28, bold=True), fill=(242, 247, 253, 255))

    cols = min(3, count)
    rows = math.ceil(count / cols)
    gap = 18
    left, top, right, bottom = 52, 458, 1148, 844
    card_w = (right - left - gap * (cols - 1)) // cols
    card_h = (bottom - top - gap * (rows - 1)) // rows
    for index in range(count):
        row, col = divmod(index, cols)
        x1 = left + col * (card_w + gap)
        y1 = top + row * (card_h + gap)
        x2, y2 = x1 + card_w, y1 + card_h
        _rounded_panel(draw, (x1, y1, x2, y2), fill=(24, 48, 80, 255), radius=24)

        try:
            date = datetime.strptime(str(times[index]), "%Y-%m-%d")
            date_label = f"{_WEEKDAYS_RU[date.weekday()]}, {date.day} {_MONTHS_RU[date.month - 1]}"
        except ValueError:
            date_label = str(times[index])
        daily_code = _int(_daily_value(daily, "weather_code", index), 0)
        t_min = _int(_daily_value(daily, "temperature_2m_min", index))
        t_max = _int(_daily_value(daily, "temperature_2m_max", index))
        precip = _int(_daily_value(daily, "precipitation_probability_max", index))
        wind = _int(_daily_value(daily, "wind_speed_10m_max", index))

        draw.text((x1 + 20, y1 + 16), date_label, font=_font(19, bold=True), fill=(224, 236, 249, 255))
        _draw_weather_icon(draw, (x1 + 65, y1 + 102), daily_code, 0.72)
        draw.text((x1 + 128, y1 + 61), f"{t_max:+d}°  /  {t_min:+d}°", font=_font(30, bold=True), fill=(255, 255, 255, 255))
        condition = _weather_code_description(daily_code, "ru") or "Нет данных"
        if len(condition) > 23:
            condition = condition[:22].rstrip() + "…"
        draw.text((x1 + 128, y1 + 105), condition, font=_font(16), fill=(170, 194, 220, 255))
        draw.text((x1 + 128, y1 + 137), f"Осадки {precip}%   ·   Ветер {wind} км/ч", font=_font(14), fill=(137, 166, 195, 255))

    draw.text((58, 863), "Источник: Open‑Meteo · значения округлены", font=_font(15), fill=(174, 198, 222, 255))

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_place = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]+", "-", str(place.get("name") or "weather")).strip("-")[:40] or "weather"
    path = output_dir / f"weather-{safe_place}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}.png"
    image.convert("RGB").save(path, format="PNG", optimize=True)
    return path.resolve()


def _hourly_table_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_hourly = payload.get("hourly")
    hourly: dict[str, Any] = raw_hourly if isinstance(raw_hourly, dict) else {}
    raw_times = hourly.get("time")
    times: list[Any] = raw_times if isinstance(raw_times, list) else []
    index_by_time = {str(value)[-5:]: index for index, value in enumerate(times)}

    def value(field: str, index: int) -> Any:
        values = hourly.get(field)
        return values[index] if isinstance(values, list) and index < len(values) else None

    rows: list[dict[str, Any]] = []
    for hour in range(7, 24):
        time_text = f"{hour:02d}:00"
        if time_text not in index_by_time:
            continue
        index = index_by_time[time_text]
        rows.append(
            {
                "time": f"{hour:02d}",
                "code": _int(value("weather_code", index), -1),
                "temperature": _int(value("temperature_2m", index)),
                "apparent": _int(value("apparent_temperature", index)),
                "rain": _int(value("precipitation_probability", index)),
                "humidity": _int(value("relative_humidity_2m", index)),
                # Open-Meteo defaults to km/h and hPa. The approved July 12
                # digest used m/s and millimetres of mercury.
                "wind": _int(float(value("wind_speed_10m", index) or 0) / 3.6),
                "pressure": _int(float(value("pressure_msl", index) or 0) * 0.750061683),
            }
        )
    if not rows:
        raise ValueError("Open-Meteo response did not include hourly data for 07:00–23:00")
    return rows


def _render_weather_png(payload: dict[str, Any], output_dir: Path, forecast_days: int) -> Path:
    """Render the approved July 12 hourly digest as a light PNG table."""
    from PIL import Image, ImageDraw

    del forecast_days  # The established digest is today's 07:00–23:00 table.
    width, height = 1200, 900
    image = Image.new("RGBA", (width, height), (246, 249, 252, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(height):
        ratio = y / max(1, height - 1)
        draw.line(
            (0, y, width, y),
            fill=(round(250 - 12 * ratio), round(252 - 9 * ratio), round(255 - 5 * ratio), 255),
        )

    # Quiet daylight accents; the table remains high-contrast and printable.
    draw.ellipse((820, -310, 1390, 240), fill=(255, 224, 158, 42))
    draw.ellipse((-260, 610, 390, 1160), fill=(133, 201, 228, 30))

    raw_place = payload.get("resolved_location")
    raw_hourly = payload.get("hourly")
    place: dict[str, Any] = raw_place if isinstance(raw_place, dict) else {}
    hourly: dict[str, Any] = raw_hourly if isinstance(raw_hourly, dict) else {}
    rows = _hourly_table_rows(payload)

    raw_times = hourly.get("time")
    times: list[Any] = raw_times if isinstance(raw_times, list) else []
    try:
        date = datetime.fromisoformat(str(times[0]))
        date_text = date.strftime("%d.%m.%Y")
    except (ValueError, IndexError):
        date_text = datetime.now().strftime("%d.%m.%Y")

    location_name = str(place.get("name") or "Прогноз погоды")
    if location_name.casefold() == "волгоград":
        title = f"Погода в Волгограде на {date_text}"
    else:
        title = f"Погода: {location_name} · {date_text}"
    title_font = _font(34, bold=True)
    while draw.textbbox((0, 0), title, font=title_font)[2] > width - 190:
        title_font = _font(max(25, getattr(title_font, "size", 34) - 2), bold=True)
        if getattr(title_font, "size", 25) <= 25:
            break
    draw.text((95, 34), title, font=title_font, fill=(30, 54, 78, 255))
    draw.text(
        (97, 82),
        "07:00–23:00 · каждый час · значения округлены",
        font=_font(17),
        fill=(88, 113, 136, 255),
    )

    table_left, table_top = 95, 116
    widths = [110, 80, 130, 120, 135, 130, 145, 160]
    headers = ["Время", "П", "Темп", "Ощ", "Дождь", "Влаж", "Ветер", "Давл"]
    units = ["", "", "°C", "°C", "%", "%", "м/с", "мм"]
    header_height, row_height = 68, 38
    table_right = table_left + sum(widths)
    table_bottom = table_top + header_height + row_height * len(rows)

    draw.rounded_rectangle(
        (table_left, table_top, table_right, table_bottom),
        radius=20,
        fill=(255, 255, 255, 238),
        outline=(151, 177, 199, 200),
        width=2,
    )
    draw.rounded_rectangle(
        (table_left, table_top, table_right, table_top + header_height),
        radius=20,
        fill=(208, 228, 243, 255),
    )
    draw.rectangle(
        (table_left, table_top + header_height - 20, table_right, table_top + header_height),
        fill=(208, 228, 243, 255),
    )

    x_positions = [table_left]
    for col_width in widths:
        x_positions.append(x_positions[-1] + col_width)

    def draw_cell_text(text: str, col: int, y: int, *, font: Any, fill: tuple[int, ...], align: str) -> None:
        x1, x2 = x_positions[col], x_positions[col + 1]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        if align == "left":
            x = x1 + 13
        elif align == "center":
            x = x1 + (x2 - x1 - text_width) / 2
        else:
            x = x2 - 13 - text_width
        draw.text((x, y), text, font=font, fill=fill)

    for col, header in enumerate(headers):
        align = "left" if col == 0 else ("center" if col == 1 else "right")
        draw_cell_text(header, col, table_top + 8, font=_font(17, bold=True), fill=(31, 58, 82, 255), align=align)
        if units[col]:
            draw_cell_text(units[col], col, table_top + 35, font=_font(14), fill=(75, 108, 136, 255), align=align)

    for col_x in x_positions[1:-1]:
        draw.line((col_x, table_top + 7, col_x, table_bottom - 7), fill=(112, 145, 172, 80), width=1)
    draw.line((table_left, table_top + header_height, table_right, table_top + header_height), fill=(112, 151, 181, 150), width=2)

    for row_index, row in enumerate(rows):
        y1 = table_top + header_height + row_index * row_height
        if row_index % 2:
            draw.rectangle((table_left + 2, y1, table_right - 2, y1 + row_height), fill=(232, 242, 249, 210))
        if row_index:
            draw.line((table_left + 10, y1, table_right - 10, y1), fill=(128, 158, 183, 55), width=1)

        center_y = y1 + row_height // 2
        draw_cell_text(str(row["time"]), 0, y1 + 8, font=_font(17, bold=True), fill=(38, 63, 86, 255), align="left")
        _draw_weather_icon(draw, ((x_positions[1] + x_positions[2]) // 2, center_y), int(row["code"]), 0.31)
        values = [
            row["temperature"], row["apparent"], row["rain"], row["humidity"],
            row["wind"], row["pressure"],
        ]
        for offset, value in enumerate(values, start=2):
            draw_cell_text(str(value), offset, y1 + 8, font=_font(17, bold=True), fill=(34, 58, 80, 255), align="right")

    draw.text((97, 871), "Источник: Open‑Meteo", font=_font(14), fill=(101, 126, 148, 255))

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_place = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]+", "-", location_name).strip("-")[:40] or "weather"
    path = output_dir / f"weather-table-{safe_place}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}.png"
    image.convert("RGB").save(path, format="PNG", optimize=True)
    return path.resolve()


def get_weather_image_tool(location: str, *, language: str = "ru", forecast_days: int = 1) -> str:
    """Create a forecast PNG and return an absolute MEDIA path as JSON."""
    try:
        days = max(1, min(int(forecast_days), 7))
        payload = _weather_payload(location, language, days)
        output_dir = Path(get_hermes_home()) / "generated" / "weather"
        path = _render_weather_png(payload, output_dir, days)
        return json.dumps(
            {
                "success": True,
                "path": str(path),
                "media": f"MEDIA:{path}",
                "instruction": f"Attach the generated image by including exactly MEDIA:{path} in the final response.",
                "source": payload["source"],
                "resolved_location": payload["resolved_location"],
            },
            ensure_ascii=False,
        )
    except urllib.error.URLError as exc:
        return json.dumps({"success": False, "error": f"Network error while fetching weather: {exc}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Weather image generation failed: {type(exc).__name__}: {exc}"}, ensure_ascii=False)


GET_WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": (
        "Get current weather for a city or place using Open-Meteo public APIs. "
        "No API key is required. Use this for ordinary weather questions like "
        "'weather in Volgograd' or 'погода в Москве'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City or place name, e.g. 'Волгоград'."},
            "language": {"type": "string", "description": "Two-letter geocoding language code.", "default": "ru"},
        },
        "required": ["location"],
    },
}

GET_WEATHER_IMAGE_SCHEMA = {
    "name": "get_weather_image",
    "description": (
        "Create an accurate PNG of the approved hourly weather table for today, 07:00–23:00 with an "
        "hourly step, using live Open-Meteo data. The table preserves the columns Time, weather, "
        "temperature, feels-like, rain probability, humidity, wind in m/s, and pressure in mm Hg. "
        "Use whenever the user asks for the weather forecast as a picture, table image, or PNG."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City or place name, e.g. 'Волгоград'."},
            "language": {"type": "string", "description": "Two-letter geocoding language code.", "default": "ru"},
        },
        "required": ["location"],
    },
}


def _handle_get_weather(args: dict[str, Any] | None, **kw: Any) -> str:
    args = args or {}
    return get_weather_tool(str(args.get("location", "")), language=str(args.get("language", "ru")))


def _handle_get_weather_image(args: dict[str, Any] | None, **kw: Any) -> str:
    args = args or {}
    return get_weather_image_tool(
        str(args.get("location", "")),
        language=str(args.get("language", "ru")),
        forecast_days=int(args.get("forecast_days", 1)),
    )


registry.register(
    name="get_weather",
    toolset="web",
    schema=GET_WEATHER_SCHEMA,
    handler=_handle_get_weather,
    emoji="🌦️",
    max_result_size_chars=20_000,
)

registry.register(
    name="get_weather_image",
    toolset="web",
    schema=GET_WEATHER_IMAGE_SCHEMA,
    handler=_handle_get_weather_image,
    emoji="🖼️",
    max_result_size_chars=10_000,
)
