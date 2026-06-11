---
title: Home Assistant
description: 通过 Home Assistant 集成，使用 Hermes Agent 控制您的智能家居。
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant 集成

Hermes Agent 通过以下两种方式与 [Home Assistant](https://www.home-assistant.io/) 集成：

1. **Gateway 平台** — 通过 WebSocket 订阅实时状态变更并响应事件
2. **智能家居工具** — 四个可供 LLM 调用的工具，通过 REST API 查询和控制设备

## 配置

### 1. 创建长期访问令牌

1. 打开您的 Home Assistant 实例
2. 进入**个人资料**（点击侧边栏中的用户名）
3. 滚动至**长期访问令牌**
4. 点击**创建令牌**，命名为"Hermes Agent"
5. 复制令牌

### 2. 配置环境变量

```bash
# Add to ~/.hermes/.env

# Required: your Long-Lived Access Token
HASS_TOKEN=your-long-lived-access-token

# Optional: HA URL (default: http://homeassistant.local:8123)
HASS_URL=http://192.168.1.100:8123
```

:::info
设置 `HASS_TOKEN` 后，`homeassistant` 工具集将自动启用。Gateway 平台和设备控制工具均通过这一个令牌激活。
:::

### 3. 启动 Gateway

```bash
hermes gateway
```

Home Assistant 将作为已连接平台出现，与其他消息平台（Telegram、Discord 等）并列显示。

## 可用工具

Hermes Agent 注册了四个智能家居控制工具：

### `ha_list_entities`

列出 Home Assistant 实体，可按域（domain）或区域（area）过滤。

**参数：**
- `domain` *（可选）* — 按实体域过滤：`light`、`switch`、`climate`、`sensor`、`binary_sensor`、`cover`、`fan`、`media_player` 等。
- `area` *（可选）* — 按区域/房间名称过滤（与友好名称匹配）：`living room`、`kitchen`、`bedroom` 等。

**示例：**
```
List all lights in the living room
```

返回实体 ID、状态及友好名称。

### `ha_get_state`

获取单个实体的详细状态，包括所有属性（亮度、颜色、温度设定值、传感器读数等）。

**参数：**
- `entity_id` *（必填）* — 要查询的实体，例如 `light.living_room`、`climate.thermostat`、`sensor.temperature`

**示例：**
```
What's the current state of climate.thermostat?
```

返回：状态、所有属性、最后变更/更新时间戳。

### `ha_list_services`

列出可用于设备控制的服务（操作）。显示每种设备类型可执行的操作及其接受的参数。

**参数：**
- `domain` *（可选）* — 按域过滤，例如 `light`、`climate`、`switch`

**示例：**
```
What services are available for climate devices?
```

### `ha_call_service`

调用 Home Assistant 服务以控制设备。

**参数：**
- `domain` *（必填）* — 服务域：`light`、`switch`、`climate`、`cover`、`media_player`、`fan`、`scene`、`script`
- `service` *（必填）* — 服务名称：`turn_on`、`turn_off`、`toggle`、`set_temperature`、`set_hvac_mode`、`open_cover`、`close_cover`、`set_volume_level`
- `entity_id` *（可选）* — 目标实体，例如 `light.living_room`
- `data` *（可选）* — 以 JSON 对象形式传入的附加参数

**示例：**

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

## Gateway 平台：实时事件

Home Assistant gateway 适配器通过 WebSocket 连接并订阅 `state_changed` 事件。当设备状态发生变更且符合过滤条件时，该事件将作为消息转发给 agent。

### 事件过滤

:::warning 必要配置
默认情况下，**不转发任何事件**。您必须配置 `watch_domains`、`watch_entities` 或 `watch_all` 中的至少一项才能接收事件。若未设置过滤器，启动时将记录警告日志，所有状态变更将被静默丢弃。
:::

在 `~/.hermes/config.yaml` 中，于 Home Assistant 平台的 `extra` 部分配置 agent 接收的事件：

```yaml
platforms:
  homeassistant:
    enabled: true
    extra:
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

| 设置 | 默认值 | 说明 |
|---------|---------|-------------|
| `watch_domains` | *（无）* | 仅监听这些实体域（例如 `climate`、`light`、`binary_sensor`） |
| `watch_entities` | *（无）* | 仅监听这些特定实体 ID |
| `watch_all` | `false` | 设为 `true` 以接收**所有**状态变更（不推荐用于大多数场景） |
| `ignore_entities` | *（无）* | 始终忽略这些实体（在域/实体过滤器之前应用） |
| `cooldown_seconds` | `30` | 同一实体两次事件之间的最小间隔秒数 |

:::tip
从一组精简的域开始 — `climate`、`binary_sensor` 和 `alarm_control_panel` 已覆盖最常用的自动化场景。按需添加更多域。使用 `ignore_entities` 屏蔽 CPU 温度或运行时间计数器等噪声传感器。
:::

### 事件格式化

状态变更将根据域格式化为人类可读的消息：

| 域 | 格式 |
|--------|--------|
| `climate` | "HVAC mode changed from 'off' to 'heat' (current: 21, target: 23)" |
| `sensor` | "changed from 21°C to 22°C" |
| `binary_sensor` | "triggered" / "cleared" |
| `light`、`switch`、`fan` | "turned on" / "turned off" |
| `alarm_control_panel` | "alarm state changed from 'armed_away' to 'triggered'" |
| *（其他）* | "changed from 'old' to 'new'" |

### Agent 响应

Agent 发出的消息将以 **Home Assistant 持久通知**的形式推送（通过 `persistent_notification.create`），标题为"Hermes Agent"，显示在 HA 通知面板中。

### 连接管理

- **WebSocket** 每 30 秒发送一次心跳，用于实时事件
- **自动重连**，退避策略：5s → 10s → 30s → 60s
- **REST API** 用于出站通知（独立会话，避免与 WebSocket 冲突）
- **鉴权** — HA 事件始终已授权（无需用户白名单，`HASS_TOKEN` 负责验证连接）

## 安全性

Home Assistant 工具强制执行安全限制：

:::warning 已屏蔽的域
以下服务域已被**屏蔽**，以防止在 HA 主机上执行任意代码：

- `shell_command` — 任意 shell 命令
- `command_line` — 执行命令的传感器/开关
- `python_script` — 脚本化 Python 执行
- `pyscript` — 更广泛的脚本集成
- `hassio` — 插件控制、主机关机/重启
- `rest_command` — 来自 HA 服务器的 HTTP 请求（SSRF 向量）

尝试调用这些域中的服务将返回错误。
:::

实体 ID 将通过正则表达式 `^[a-z_][a-z0-9_]*\.[a-z0-9_]+$` 进行验证，以防止注入攻击。

## 自动化示例

### 晨间例程

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

### 安全检查

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

### 响应式自动化（通过 Gateway 事件）

作为 gateway 平台连接后，agent 可对事件作出响应：

```
[Home Assistant] Front Door: triggered (was cleared)

Agent automatically:
1. ha_get_state(entity_id="binary_sensor.front_door")
2. ha_call_service(domain="light", service="turn_on",
     entity_id="light.hallway")
3. Sends notification: "Front door opened. Hallway lights turned on."
```