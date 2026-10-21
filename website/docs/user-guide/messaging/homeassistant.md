---
title: Home Assistant
description: Control your smart home with Hermes Agent via the Home Assistant plugin from the plugin catalog.
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant Integration

Hermes Agent integrates with [Home Assistant](https://www.home-assistant.io/) through the official **`homeassistant` plugin** from the [plugin catalog](../features/plugins.md). The plugin is maintained by Nous Research in [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/homeassistant) and is not part of Hermes core. It provides two things:

1. **Gateway platform** — subscribes to real-time state changes via WebSocket and responds to events
2. **Smart home tools** — four LLM-callable tools (the `homeassistant` toolset) for querying and controlling devices via the REST API

## Install

```bash
hermes plugins install homeassistant
```

Plugins are installed per profile. To use Home Assistant in another profile, install it there too:

```bash
hermes -p <profile> plugins install homeassistant
```

The plugin declares its own Python dependency (`aiohttp`), so there is no pip extra to install. The old `hermes-agent[homeassistant]` extra has been removed.

:::info Upgrading from a release that bundled Home Assistant
Nothing to do. Every profile that was already using Home Assistant — `HASS_TOKEN` in its `.env`, `platforms.homeassistant` enabled (or given a `token`) in `config.yaml`, or the `homeassistant` toolset listed in `platform_toolsets` — gets the plugin installed automatically from the catalog by `hermes update` (for all profiles sharing the install). If that step could not run, Hermes installs it the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). After a failed attempt (offline, catalog unreachable) starts retry at most once an hour; `hermes update` always retries. The outcome is reported in the terminal, the Desktop app and chat. This happens once per profile: if you later remove the plugin (`hermes plugins remove homeassistant`), it stays removed.

Your configuration carries over unchanged: the same `HASS_TOKEN` / `HASS_URL` variables, the same `homeassistant` platform name and `platforms.homeassistant` keys, the same `homeassistant` toolset and tool names, and the same cron `deliver: homeassistant:<notify target>` syntax. One difference: like every plugin tool, the `ha_*` tools sit behind [Tool Search](../features/tools.md) (`tool_search` / `tool_call`) when it is on, instead of being listed directly.
:::

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

# Optional: default notify target for a bare `deliver: homeassistant`
HASS_HOME_CHANNEL=mobile_app_my_phone
```

:::info
With the plugin installed, the `homeassistant` toolset is enabled automatically when `HASS_TOKEN` is set. Both the gateway platform and the device control tools activate from this single token.
:::

### 3. Start the Gateway

```bash
hermes gateway
```

Home Assistant will appear as a connected platform alongside any other messaging platforms (Telegram, Discord, etc.).

## Available Tools

The plugin registers four tools for smart home control in the `homeassistant` toolset:

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

The Home Assistant gateway adapter connects via WebSocket and subscribes to `state_changed` events. When a device state changes and matches your filters, it's forwarded to the agent as a message.

### Event Filtering

:::warning Required Configuration
By default, **no events are forwarded**. You must configure at least one of `watch_domains`, `watch_entities`, or `watch_all` to receive events. Without filters, a warning is logged at startup and all state changes are silently dropped.
:::

Configure which events the agent sees in `~/.hermes/config.yaml` under the Home Assistant platform's `extra` section:

```yaml
platforms:
  homeassistant:
    enabled: true
    extra:
      url: http://192.168.1.100:8123   # optional; same as HASS_URL
      watch_domains:
        - climate
        - binary_sensor
        - alarm_control_panel
        - light
      watch_entities:
        - sensor.front_door_battery
      ignore_entities:
        - sensor.uptime
        - sensor.cpu_usage
        - sensor.memory_usage
      cooldown_seconds: 30
```

| Setting | Default | Description |
|---------|---------|-------------|
| `url` | `HASS_URL`, else `http://homeassistant.local:8123` | Home Assistant base URL |
| `watch_domains` | *(none)* | Only watch these entity domains (e.g., `climate`, `light`, `binary_sensor`) |
| `watch_entities` | *(none)* | Only watch these specific entity IDs |
| `watch_all` | `false` | Set to `true` to receive **all** state changes (not recommended for most setups) |
| `ignore_entities` | *(none)* | Always ignore these entities (applied before domain/entity filters) |
| `cooldown_seconds` | `30` | Minimum seconds between events for the same entity |

:::tip
Start with a focused set of domains — `climate`, `binary_sensor`, and `alarm_control_panel` cover the most useful automations. Add more as needed. Use `ignore_entities` to suppress noisy sensors like CPU temperature or uptime counters.
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

The platform uses the `minimal` display defaults (no tool-progress or streaming chatter in notifications). Override them under `display.platforms.homeassistant` in `config.yaml` if you want more.

### Cron and Webhook Delivery

Scheduled jobs and webhook routes can deliver to Home Assistant:

```yaml
deliver: homeassistant:mobile_app_my_phone   # explicit notify target
deliver: homeassistant                       # uses HASS_HOME_CHANNEL
```

The bare `homeassistant` form needs `HASS_HOME_CHANNEL` set to a default notify target. See [Scheduled Tasks](../features/cron.md) and [Webhooks](webhooks.md).

### Connection Management

- **WebSocket** with 30-second heartbeat for real-time events
- **Automatic reconnection** with backoff: 5s → 10s → 30s → 60s
- **REST API** for outbound notifications (separate session to avoid WebSocket conflicts)
- **Authorization** — HA events are always authorized (no user allowlist or pairing needed: `HASS_TOKEN` authenticates the connection and there is no human sender)

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

## Troubleshooting

**Platform or tools missing.**
Check that the plugin is installed and enabled in the active profile with
`hermes plugins list`. If it is missing, run `hermes plugins install homeassistant`
(or `hermes -p <profile> plugins install homeassistant`) and restart the gateway.
With `security.allow_lazy_installs` turned off, the automatic first-start install
is skipped and you must install the plugin yourself.

**Environment variables not picked up.**
The adapter reads credentials from `~/.hermes/.env` (auto-merged at startup) or
from `config.yaml`. Double-check the file lives under the active Hermes profile
home and that there's no stray quoting around the URL/token. Restart the gateway
after editing — env changes are only applied on process start.

**REST auth failing (`401 Unauthorized`).**
The token must be a *Long-Lived Access Token* created from your HA user profile
page (**Profile → Security → Long-lived access tokens**). Short-lived UI
session tokens won't work. Also verify the base URL includes the scheme and
port (e.g. `http://homeassistant.local:8123`) and is reachable from the host
running Hermes — `curl -H "Authorization: Bearer <token>" <url>/api/` should
return `{"message": "API running."}`.
