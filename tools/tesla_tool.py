"""
Tesla vehicle control tool for Hermes.

Controls Tesla vehicles via the Fleet API.
Supports: status, wake, lock, unlock, climate on/off, set temp, horn, flash lights, charge start/stop.
API audience and token/key paths are loaded from config file.
"""

import json
import logging
import os
import time
from pathlib import Path

import yaml

from tools.registry import registry

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".hermes" / "smart_home"
_CONFIG_FILE = _CONFIG_DIR / "tesla.yaml"


def _load_config() -> dict:
    """Load tool config from YAML file."""
    if not _CONFIG_FILE.exists():
        logger.warning("Tesla config not found: %s", _CONFIG_FILE)
        return {}
    try:
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_audience() -> str:
    return _load_config().get("api_audience", "https://fleet-api.prd.na.vn.cloud.tesla.com")


def _get_token_path() -> Path:
    config = _load_config()
    return Path(config.get("token_file", "~/.hermes/smart_home/tesla_token.json")).expanduser()


def _check_tesla_reqs() -> bool:
    return _get_token_path().exists()


def _load_token() -> dict:
    return json.loads(_get_token_path().read_text())


def _get_headers() -> dict:
    import requests
    tokens = _load_token()
    # Refresh if expiring within 60s
    if tokens.get("expires_at", 0) < time.time() + 60:
        client_id     = os.getenv("TESLA_CLIENT_ID", "")
        client_secret = os.getenv("TESLA_CLIENT_SECRET", "")
        resp = requests.post("https://auth.tesla.com/oauth2/v3/token", json={
            "grant_type":    "refresh_token",
            "client_id":     client_id,
            "client_secret": client_secret,
            "refresh_token": tokens["refresh_token"],
        })
        if resp.status_code == 200:
            new = resp.json()
            tokens.update({
                "access_token": new["access_token"],
                "expires_at":   time.time() + new.get("expires_in", 3600) - 60,
            })
            if "refresh_token" in new:
                tokens["refresh_token"] = new["refresh_token"]
            _get_token_path().write_text(json.dumps(tokens, indent=2))
    return {
        "Authorization": f"Bearer {tokens['access_token']}",
        "Content-Type":  "application/json",
    }


def _get_vin(vin: str = None) -> str:
    import requests
    audience = _get_audience()
    resp = requests.get(f"{audience}/api/1/vehicles", headers=_get_headers())
    vehicles = resp.json().get("response", [])
    if not vehicles:
        raise RuntimeError("No Tesla vehicles found")
    if vin:
        for v in vehicles:
            if v["vin"] == vin:
                return vin
        raise RuntimeError(f"VIN {vin} not found")
    return vehicles[0]["vin"]


def _wake(vin: str) -> bool:
    import requests
    audience = _get_audience()
    requests.post(f"{audience}/api/1/vehicles/{vin}/wake_up", headers=_get_headers())
    for _ in range(12):
        time.sleep(5)
        r = requests.get(f"{audience}/api/1/vehicles/{vin}", headers=_get_headers())
        if r.json().get("response", {}).get("state") == "online":
            return True
    return False


def tesla_control(action: str, vin: str = None, param: str = None) -> str:
    import requests
    # pylint: disable=possibly-used-before-assignment
    AUDIENCE = _get_audience()

    try:
        v = _get_vin(vin)

        if action == "status":
            _wake(v)
            r = requests.get(f"{AUDIENCE}/api/1/vehicles/{v}/vehicle_data", headers=_get_headers())
            d = r.json().get("response", {})
            charge  = d.get("charge_state", {})
            climate = d.get("climate_state", {})
            vehicle = d.get("vehicle_state", {})
            drive   = d.get("drive_state", {})

            battery_range_mi = charge.get("battery_range", 0)
            locked = vehicle.get("locked", True)
            result = {
                "name":            d.get("display_name", "Tesla"),
                "battery_level":   charge.get("battery_level"),
                "range_miles":     round(battery_range_mi, 1),
                "charging_state":  charge.get("charging_state"),
                "charge_limit_pct": charge.get("charge_limit_soc"),
                "charge_amps":     charge.get("charge_current_request"),
                "locked":          locked,
                "sentry_mode":     vehicle.get("sentry_mode"),
                "inside_temp_c":   climate.get("inside_temp"),
                "outside_temp_c":  climate.get("outside_temp"),
                "climate_on":      climate.get("is_climate_on"),
                "climate_keeper":  climate.get("climate_keeper_mode"),
                "latitude":        drive.get("latitude"),
                "longitude":       drive.get("longitude"),
                "odometer_mi":     round(vehicle.get("odometer", 0), 1),
                "sw_version":      vehicle.get("car_version", "").split(" ")[0],
                "sw_update":       d.get("vehicle_state", {}).get("software_update", {}).get("status"),
            }
            return json.dumps(result)

        elif action == "wake":
            ok = _wake(v)
            return json.dumps({"success": ok, "message": "Vehicle is online" if ok else "Wake timed out"})

        elif action == "lock":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/door_lock", headers=_get_headers())
            return json.dumps({"success": True, "message": "Car locked"})

        elif action == "unlock":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/door_unlock", headers=_get_headers())
            return json.dumps({"success": True, "message": "Car unlocked"})

        elif action == "climate_on":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/auto_conditioning_start", headers=_get_headers())
            return json.dumps({"success": True, "message": "Climate started"})

        elif action == "climate_off":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/auto_conditioning_stop", headers=_get_headers())
            return json.dumps({"success": True, "message": "Climate stopped"})

        elif action == "set_temp":
            if not param:
                return json.dumps({"error": "param (temperature in °C) is required"})
            _wake(v)
            temp = float(param)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/set_temps",
                          headers=_get_headers(),
                          json={"driver_temp": temp, "passenger_temp": temp})
            return json.dumps({"success": True, "message": f"Temperature set to {temp}°C"})

        elif action == "horn":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/honk_horn", headers=_get_headers())
            return json.dumps({"success": True, "message": "Horn honked"})

        elif action == "flash":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/flash_lights", headers=_get_headers())
            return json.dumps({"success": True, "message": "Lights flashed"})

        elif action == "set_charge_limit":
            if not param:
                return json.dumps({"error": "param (percent 50-100) is required"})
            _wake(v)
            pct = int(param)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/set_charge_limit",
                              headers=_get_headers(),
                              json={"percent": pct})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": f"Charge limit set to {pct}%"})

        elif action == "charge_max_range":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/charge_max_range", headers=_get_headers())
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Charge limit set to max range (100%)"})

        elif action == "set_charging_amps":
            if not param:
                return json.dumps({"error": "param (amps as integer) is required"})
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/set_charging_amps",
                              headers=_get_headers(),
                              json={"charging_amps": int(param)})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": f"Charging amps set to {param}A"})

        elif action == "charge_port_open":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/charge_port_door_open", headers=_get_headers())
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Charge port opened"})

        elif action == "charge_port_close":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/charge_port_door_close", headers=_get_headers())
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Charge port closed"})

        elif action == "trunk_open":
            _wake(v)
            which = param if param in ("front", "rear") else "rear"
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/actuate_trunk",
                              headers=_get_headers(),
                              json={"which_trunk": which})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": f"{which.title()} trunk actuated"})

        elif action == "sentry_on":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/set_sentry_mode",
                              headers=_get_headers(),
                              json={"on": True})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Sentry mode enabled"})

        elif action == "sentry_off":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/set_sentry_mode",
                              headers=_get_headers(),
                              json={"on": False})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Sentry mode disabled"})

        elif action == "window_vent":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/window_control",
                              headers=_get_headers(),
                              json={"command": "vent", "lat": 0, "lon": 0})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Windows vented"})

        elif action == "window_close":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/window_control",
                              headers=_get_headers(),
                              json={"command": "close", "lat": 0, "lon": 0})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Windows closed"})

        elif action == "trigger_homelink":
            _wake(v)
            # Need current vehicle location for homelink
            r = requests.get(f"{AUDIENCE}/api/1/vehicles/{v}/vehicle_data",
                             headers=_get_headers(),
                             params={"endpoints": "drive_state"})
            d = r.json().get("response", {})
            lat = d.get("drive_state", {}).get("latitude", 0)
            lon = d.get("drive_state", {}).get("longitude", 0)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/trigger_homelink",
                              headers=_get_headers(),
                              json={"lat": lat, "lon": lon})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "HomeLink triggered"})

        elif action == "navigate":
            if not param:
                return json.dumps({"error": "param (address or 'lat,lon') is required"})
            _wake(v)
            # Check if param is lat,lon format
            if "," in param and all(p.strip().replace("-", "").replace(".", "").isdigit() for p in param.split(",")):
                lat, lon = [float(x.strip()) for x in param.split(",")]
                r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/navigation_gps_request",
                                  headers=_get_headers(),
                                  json={"lat": lat, "lon": lon, "order": 0})
            else:
                r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/navigation_request",
                                  headers=_get_headers(),
                                  json={"type": "share_ext_content_raw",
                                        "value": {"android.intent.extra.TEXT": param},
                                        "locale": "en-US",
                                        "timestamp_ms": str(int(time.time() * 1000))})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": f"Navigation sent: {param}"})

        elif action == "climate_keeper":
            if not param:
                return json.dumps({"error": "param required: off, keep, dog, camp"})
            mode_map = {"off": 0, "keep": 1, "dog": 2, "camp": 3}
            mode = mode_map.get(param.lower(), 0)
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/set_climate_keeper_mode",
                              headers=_get_headers(),
                              json={"climate_keeper_mode": mode})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": f"Climate keeper set to {param}"})

        elif action == "remote_start":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/remote_start_drive",
                              headers=_get_headers())
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Remote start activated (2 min to shift into gear)"})

        elif action == "seat_heater":
            if not param:
                return json.dumps({"error": "param required: 'seat,level' e.g. '0,3' (seat 0=driver,1=passenger,2=rear-left,4=rear-center,5=rear-right; level 0=off,1=low,2=med,3=high)"})
            parts = param.split(",")
            seat = int(parts[0])
            level = int(parts[1])
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/remote_seat_heater_request",
                              headers=_get_headers(),
                              json={"heater": seat, "level": level})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            seat_names = {0: "Driver", 1: "Passenger", 2: "Rear-left", 4: "Rear-center", 5: "Rear-right"}
            level_names = {0: "Off", 1: "Low", 2: "Medium", 3: "High"}
            return json.dumps({"success": ok, "message": f"{seat_names.get(seat, f'Seat {seat}')} heater set to {level_names.get(level, f'level {level}')}"})

        elif action == "schedule_sw_update":
            _wake(v)
            offset = int(param) if param else 0  # seconds from now
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/schedule_software_update",
                              headers=_get_headers(),
                              json={"offset_sec": offset})
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": f"Software update scheduled in {offset}s"})

        elif action == "cancel_sw_update":
            _wake(v)
            r = requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/cancel_software_update", headers=_get_headers())
            resp_data = r.json()
            ok = resp_data.get("response", {}).get("result", False)
            return json.dumps({"success": ok, "message": "Software update cancelled"})

        elif action == "charge_start":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/charge_start", headers=_get_headers())
            return json.dumps({"success": True, "message": "Charging started"})

        elif action == "charge_stop":
            _wake(v)
            requests.post(f"{AUDIENCE}/api/1/vehicles/{v}/command/charge_stop", headers=_get_headers())
            return json.dumps({"success": True, "message": "Charging stopped"})

        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    except Exception as e:
        logger.error("Tesla tool error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


TESLA_SCHEMA = {
    "name": "tesla",
    "description": (
        "Control and query a Tesla vehicle via the Fleet API. "
        "Actions: status, wake, lock, unlock, climate_on, climate_off, "
        "set_temp (param=°C), horn, flash, "
        "set_charge_limit (param=percent 50-100), charge_max_range, "
        "set_charging_amps (param=amps), charge_start, charge_stop, "
        "charge_port_open, charge_port_close, "
        "trunk_open (param=front|rear, default rear), "
        "sentry_on, sentry_off, "
        "window_vent, window_close, "
        "trigger_homelink (opens/closes garage door), "
        "navigate (param=address or 'lat,lon'), "
        "climate_keeper (param=off|keep|dog|camp), "
        "remote_start (2 min to shift into gear), "
        "seat_heater (param='seat,level' e.g. '0,3'), "
        "schedule_sw_update (param=seconds delay), cancel_sw_update."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "wake", "lock", "unlock",
                    "climate_on", "climate_off", "set_temp",
                    "horn", "flash",
                    "set_charge_limit", "charge_max_range", "set_charging_amps",
                    "charge_start", "charge_stop", "charge_port_open", "charge_port_close",
                    "trunk_open", "sentry_on", "sentry_off",
                    "window_vent", "window_close", "trigger_homelink",
                    "navigate", "climate_keeper", "remote_start",
                    "seat_heater", "schedule_sw_update", "cancel_sw_update"
                ],
                "description": "Action to perform on the vehicle",
            },
            "vin": {
                "type": "string",
                "description": "Vehicle VIN (optional, defaults to the only/first vehicle)",
            },
            "param": {
                "type": "string",
                "description": "Extra parameter — required for set_temp (temperature in °C as string)",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="tesla",
    toolset="tesla",
    schema=TESLA_SCHEMA,
    handler=lambda args, **kw: tesla_control(
        action=args["action"],
        vin=args.get("vin"),
        param=args.get("param"),
    ),
    check_fn=_check_tesla_reqs,
    emoji="🚗",
)
