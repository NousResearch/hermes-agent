---
title: Home Assistant
description: Control your smart home with Hermes Agent via Home Assistant integration.
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant Integration

Hermes Agent integrates with [Home Assistant](https://www.home-assistant.io/) in two ways:

1. **Gateway platform** — subscribes to real-time state changes via WebSocket and responds to events
2. **Smart home tools** — four LLM-callable tools for querying and controlling devices via the REST API

## Setup

### 1. Create a Long-Lived Access Token

1. Open your Home Assistant instance
2. Go to your **Profile** (click your name in the sidebar)
3. Scroll to **Long-Lived Access Tokens**
4. Click **Create Token**, give it a name like "Hermes Agent"
5. Copy the token

### 2. Configure Environment Variables

```bash
# Add to ~/.hermes/.env

# Required: your Long-Lived Access Token
HASS_TOKEN=your-long-lived-access-token

# Optional: HA URL (default: http://homeassistant.local:8123)
HASS_URL=http://192.168.1.100:8123
```

:::info
The `homeassistant` toolset is automatically enabled when `HASS_TOKEN` is set. Both the gateway platform and the device control tools activate from this single token.
:::

### 3. Start the Gateway

```bash
hermes gateway
```

Home Assistant will appear as a connected platform alongside any other messaging platforms (Telegram, Discord, etc.).

## Available Tools

Hermes Agent registers four tools for smart home control:

### `ha_list_entities`

List Home Assistant entities, optionally filtered by domain or area.

**Parameters:**
- `domain` *(optional)* — Filter by entity domain: `light`, `switch`, `climate`, `sensor`, `binary_sensor`, `cover`, `fan`, `media_player`, etc.
- `area` *(optional)* — Filter by area/room name (matches against friendly names): `living room`, `kitchen`, `bedroom`, etc.

**Example:**
```
List all lights in the living room
```

Returns entity IDs, states, and friendly names.

### `ha_get_state`

Get detailed state of a single entity, including all attributes (brightness, color, temperature setpoint, sensor readings, etc.).

**Parameters:**
- `entity_id` *(required)* — The entity to query, e.g., `light.living_room`, `climate.thermostat`, `sensor.temperature`

**Example:**
```
What's the current state of climate.thermostat?
```

Returns: state, all attributes, last changed/updated timestamps.

### `ha_list_services`

List available services (actions) for device control. Shows what actions can be performed on each device type and what parameters they accept.

**Parameters:**
- `domain` *(optional)* — Filter by domain, e.g., `light`, `climate`, `switch`

**Example:**
```
What services are available for climate devices?
```

### `ha_call_service`

Call a Home Assistant service to control a device.

**Parameters:**
- `domain` *(required)* — Service domain: `light`, `switch`, `climate`, `cover`, `media_player`, `fan`, `scene`, `script`
- `service` *(required)* — Service name: `turn_on`, `turn_off`, `toggle`, `set_temperature`, `set_hvac_mode`, `open_cover`, `close_cover`, `set_volume_level`
- `entity_id` *(optional)* — Target entity, e.g., `light.living_room`
- `data` *(optional)* — Additional parameters as a JSON object

**Examples:**

```
Turn on the living room lights
→ ha_call_service(domain="light", service="turn_on", entity_id="light.living_room")
```

```
Set the thermostat to 22 degrees in heat mode
→ ha_call_service(domain="climate", service="set_temperature",
    entity_id="climate.thermostat", data={"temperature": 22, "hvac_mode": "heat"})
```

```
Set living room lights to blue at 50% brightness
→ ha_call_service(domain="light", service="turn_on",
    entity_id="light.living_room", data={"brightness": 128, "color_name": "blue"})
```

## Gateway Platform: Real-Time Events

The Home Assistant gateway adapter connects via WebSocket and subscribes to `state_changed` events. When a device state changes, it's forwarded to the agent as a message.

### Event Filtering

Configure which events the agent sees via platform config in the gateway:

```python
# In platform extra config
{
    "watch_domains": ["climate", "binary_sensor", "alarm_control_panel"],
    "watch_entities": ["sensor.front_door"],
    "ignore_entities": ["sensor.uptime", "sensor.cpu_usage"],
    "cooldown_seconds": 30
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `watch_domains` | *(all)* | Only watch these entity domains |
| `watch_entities` | *(all)* | Only watch these specific entities |
| `ignore_entities` | *(none)* | Always ignore these entities |
| `cooldown_seconds` | `30` | Minimum seconds between events for the same entity |

:::tip
Without any filters, the agent receives **all** state changes, which can be noisy. For practical use, set `watch_domains` to the domains you care about (e.g., `climate`, `binary_sensor`, `alarm_control_panel`).
:::

### Event Formatting

State changes are formatted as human-readable messages based on domain:

| Domain | Format |
|--------|--------|
| `climate` | "HVAC mode changed from 'off' to 'heat' (current: 21, target: 23)" |
| `sensor` | "changed from 21°C to 22°C" |
| `binary_sensor` | "triggered" / "cleared" |
| `light`, `switch`, `fan` | "turned on" / "turned off" |
| `alarm_control_panel` | "alarm state changed from 'armed_away' to 'triggered'" |
| *(other)* | "changed from 'old' to 'new'" |

### Agent Responses

Outbound messages from the agent are delivered as **Home Assistant persistent notifications** (via `persistent_notification.create`). These appear in the HA notification panel with the title "Hermes Agent".

### Connection Management

- **WebSocket** with 30-second heartbeat for real-time events
- **Automatic reconnection** with backoff: 5s → 10s → 30s → 60s
- **REST API** for outbound notifications (separate session to avoid WebSocket conflicts)
- **Authorization** — HA events are always authorized (no user allowlist needed, since the `HASS_TOKEN` authenticates the connection)

## Security

The Home Assistant tools enforce security restrictions:

:::warning Blocked Domains
The following service domains are **blocked** to prevent arbitrary code execution on the HA host:

- `shell_command` — arbitrary shell commands
- `command_line` — sensors/switches that execute commands
- `python_script` — scripted Python execution
- `pyscript` — broader scripting integration
- `hassio` — addon control, host shutdown/reboot
- `rest_command` — HTTP requests from HA server (SSRF vector)

Attempting to call services in these domains returns an error.
:::

Entity IDs are validated against the pattern `^[a-z_][a-z0-9_]*\.[a-z0-9_]+$` to prevent injection attacks.

## Example Automations

### Morning Routine

```
User: Start my morning routine

Agent:
1. ha_call_service(domain="light", service="turn_on",
     entity_id="light.bedroom", data={"brightness": 128})
2. ha_call_service(domain="climate", service="set_temperature",
     entity_id="climate.thermostat", data={"temperature": 22})
3. ha_call_service(domain="media_player", service="turn_on",
     entity_id="media_player.kitchen_speaker")
```

### Security Check

```
User: Is the house secure?

Agent:
1. ha_list_entities(domain="binary_sensor")
     → checks door/window sensors
2. ha_get_state(entity_id="alarm_control_panel.home")
     → checks alarm status
3. ha_list_entities(domain="lock")
     → checks lock states
4. Reports: "All doors closed, alarm is armed_away, all locks engaged."
```

### Reactive Automation (via Gateway Events)

When connected as a gateway platform, the agent can react to events:

```
[Home Assistant] Front Door: triggered (was cleared)

Agent automatically:
1. ha_get_state(entity_id="binary_sensor.front_door")
2. ha_call_service(domain="light", service="turn_on",
     entity_id="light.hallway")
3. Sends notification: "Front door opened. Hallway lights turned on."
```
