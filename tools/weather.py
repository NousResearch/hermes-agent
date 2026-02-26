"""
OpenWeatherMap tools for Hermes Agent.
Requires OPENWEATHERMAP_API_KEY in ~/.hermes/.env
"""
import os, json, urllib.request, urllib.parse

def _get_api_key():
    key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
    if not key:
        raise ValueError("OPENWEATHERMAP_API_KEY not set. Add it to ~/.hermes/.env")
    return key

def _owm_request(endpoint, params):
    params["appid"] = _get_api_key()
    params["units"] = params.get("units", "metric")
    url = f"https://api.openweathermap.org/data/2.5/{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())

def weather_current(location: str, units: str = "metric") -> dict:
    """Get current weather for a city. Args: location: e.g. London, Tokyo,JP. units: metric/imperial/standard"""
    data = _owm_request("weather", {"q": location, "units": units})
    u = {"metric": "C", "imperial": "F", "standard": "K"}.get(units, "C")
    s = "mph" if units == "imperial" else "m/s"
    return {"location": data["name"], "country": data["sys"]["country"],
            "temperature": f"{data['main']['temp']:.1f}{u}", "feels_like": f"{data['main']['feels_like']:.1f}{u}",
            "temp_min": f"{data['main']['temp_min']:.1f}{u}", "temp_max": f"{data['main']['temp_max']:.1f}{u}",
            "humidity": f"{data['main']['humidity']}%", "wind_speed": f"{data['wind']['speed']} {s}",
            "description": data["weather"][0]["description"].capitalize(),
            "visibility": f"{data.get('visibility', 'N/A')} m", "clouds": f"{data['clouds']['all']}%"}

def weather_forecast(location: str, days: int = 5, units: str = "metric") -> dict:
    """Get multi-day forecast (up to 5 days). Args: location: city name. days: 1-5. units: metric/imperial"""
    days = max(1, min(days, 5))
    data = _owm_request("forecast", {"q": location, "units": units, "cnt": days * 8})
    u = {"metric": "C", "imperial": "F", "standard": "K"}.get(units, "C")
    s = "mph" if units == "imperial" else "m/s"
    dm = {}
    for e in data["list"]:
        d = e["dt_txt"].split(" ")[0]
        if d not in dm:
            dm[d] = {"temps": [], "descs": [], "hum": [], "wind": [], "pop": []}
        dm[d]["temps"].append(e["main"]["temp"])
        dm[d]["descs"].append(e["weather"][0]["description"])
        dm[d]["hum"].append(e["main"]["humidity"])
        dm[d]["wind"].append(e["wind"]["speed"])
        dm[d]["pop"].append(e.get("pop", 0))
    fc = []
    for date, d in list(dm.items())[:days]:
        desc = max(set(d["descs"]), key=d["descs"].count)
        fc.append({"date": date, "temp_min": f"{min(d['temps']):.1f}{u}", "temp_max": f"{max(d['temps']):.1f}{u}",
                   "description": desc.capitalize(), "humidity_avg": f"{sum(d['hum'])//len(d['hum'])}%",
                   "wind_speed_avg": f"{sum(d['wind'])/len(d['wind']):.1f} {s}",
                   "precipitation_chance": f"{max(d['pop'])*100:.0f}%"})
    return {"location": data["city"]["name"], "country": data["city"]["country"], "forecast": fc}

def weather_alerts(location: str) -> dict:
    """Get weather alerts and Air Quality Index. Args: location: city name e.g. Miami, London,GB"""
    geo_url = "https://api.openweathermap.org/geo/1.0/direct?" + urllib.parse.urlencode({"q": location, "limit": 1, "appid": _get_api_key()})
    with urllib.request.urlopen(geo_url, timeout=10) as r:
        geo = json.loads(r.read().decode())
    if not geo:
        return {"error": f"Location not found: {location}"}
    lat, lon = geo[0]["lat"], geo[0]["lon"]
    alerts = []
    try:
        oc_url = "https://api.openweathermap.org/data/3.0/onecall?" + urllib.parse.urlencode({"lat": lat, "lon": lon, "exclude": "minutely,hourly,daily", "appid": _get_api_key()})
        with urllib.request.urlopen(oc_url, timeout=10) as r:
            oc = json.loads(r.read().decode())
        alerts = [{"event": a["event"], "sender": a.get("sender_name","NWS"), "start": a["start"], "end": a["end"], "description": a["description"][:300]} for a in oc.get("alerts",[])]
    except Exception:
        pass
    aqi_url = "https://api.openweathermap.org/data/2.5/air_pollution?" + urllib.parse.urlencode({"lat": lat, "lon": lon, "appid": _get_api_key()})
    with urllib.request.urlopen(aqi_url, timeout=10) as r:
        aqi = json.loads(r.read().decode())
    score = aqi["list"][0]["main"]["aqi"]
    comp = aqi["list"][0]["components"]
    return {"location": geo[0].get("name", location), "country": geo[0].get("country",""),
            "alerts": alerts, "alert_count": len(alerts),
            "air_quality": {"aqi": score, "label": {1:"Good",2:"Fair",3:"Moderate",4:"Poor",5:"Very Poor"}.get(score,"Unknown"),
                           "co": f"{comp.get('co',0):.1f} ug/m3", "no2": f"{comp.get('no2',0):.1f} ug/m3",
                           "o3": f"{comp.get('o3',0):.1f} ug/m3", "pm2_5": f"{comp.get('pm2_5',0):.1f} ug/m3",
                           "pm10": f"{comp.get('pm10',0):.1f} ug/m3"}}
