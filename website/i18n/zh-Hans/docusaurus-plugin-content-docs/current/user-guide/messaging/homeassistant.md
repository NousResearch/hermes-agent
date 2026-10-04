---
title: Home Assistant
description: 通过插件目录中的 Home Assistant 插件，使用 Hermes Agent 控制您的智能家居。
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant 集成

Hermes Agent 通过[插件目录](../features/plugins.md)中的官方 **`homeassistant` 插件**与 [Home Assistant](https://www.home-assistant.io/) 集成。该插件由 Nous Research 在 [NousResearch/hermes-homeassistant](https://github.com/NousResearch/hermes-homeassistant) 中维护，不再属于 Hermes 核心。它提供两部分功能：

1. **Gateway 平台** — 通过 WebSocket 订阅实时状态变更并响应事件
2. **智能家居工具** — 四个可供 LLM 调用的工具（`homeassistant` 工具集），通过 REST API 查询和控制设备

## 安装

```bash
hermes plugins install homeassistant
```

插件按 profile 安装。如需在其他 profile 中使用 Home Assistant，请在该 profile 中同样安装：

```bash
hermes -p <profile> plugins install homeassistant
```

插件自行声明其 Python 依赖（`aiohttp`），因此无需安装 pip extra。旧的 `hermes-agent[homeassistant]` extra 已被移除。

:::info 从内置 Home Assistant 的版本升级
无需任何操作。每个已在使用 Home Assistant 的 profile（`.env` 中有 `HASS_TOKEN`、`config.yaml` 中启用了 `platforms.homeassistant`（或为其设置了 `token`），或 `platform_toolsets` 中列出了 `homeassistant` 工具集）都会在 `hermes update` 时自动从插件目录安装该插件（覆盖共享同一安装的所有 profile）。若该步骤未能执行，Hermes 会在该 profile 首次启动时（agent 启动或 gateway 启动）安装插件，此行为遵循 `security.allow_lazy_installs`。安装结果会显示在终端、Desktop 应用和聊天中。每个 profile 只会自动安装一次：之后若你移除插件（`hermes plugins remove homeassistant`），它将保持移除状态。

您的配置保持不变：相同的 `HASS_TOKEN` / `HASS_URL` 变量、相同的 `homeassistant` 平台名称和 `platforms.homeassistant` 配置键、相同的 `homeassistant` 工具集和工具名称，以及相同的 cron `deliver: homeassistant:<notify target>` 语法。唯一的区别：与所有插件工具一样，启用 [Tool Search](../features/tools.md) 时，`ha_*` 工具通过 `tool_search` / `tool_call` 调用，而不是直接列出。
:::

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

# Optional: default notify target for a bare `deliver: homeassistant`
HASS_HOME_CHANNEL=mobile_app_my_phone
```

:::info
安装插件后，设置 `HASS_TOKEN` 即会自动启用 `homeassistant` 工具集。Gateway 平台和设备控制工具均通过这一个令牌激活。
:::

### 3. 启动 Gateway

```bash
hermes gateway
```

Home Assistant 将作为已连接平台出现，与其他消息平台（Telegram、Discord 等）并列显示。

## 可用工具

插件在 `homeassistant` 工具集中注册了四个智能家居控制工具：

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

| 设置 | 默认值 | 说明 |
|---------|---------|-------------|
| `url` | `HASS_URL`，否则为 `http://homeassistant.local:8123` | Home Assistant 基础 URL |
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

该平台使用 `minimal` 显示默认值（通知中不包含工具进度或流式输出）。如需更多输出，可在 `config.yaml` 的 `display.platforms.homeassistant` 下覆盖。

### Cron 与 Webhook 投递

定时任务和 webhook 路由可以投递到 Home Assistant：

```yaml
deliver: homeassistant:mobile_app_my_phone   # explicit notify target
deliver: homeassistant                       # uses HASS_HOME_CHANNEL
```

裸名 `homeassistant` 需要将 `HASS_HOME_CHANNEL` 设置为默认通知目标。参见[定时任务](../features/cron.md)和 [Webhooks](webhooks.md)。

### 连接管理

- **WebSocket** 每 30 秒发送一次心跳，用于实时事件
- **自动重连**，退避策略：5s → 10s → 30s → 60s
- **REST API** 用于出站通知（独立会话，避免与 WebSocket 冲突）
- **鉴权** — HA 事件始终已授权（无需用户白名单或配对：`HASS_TOKEN` 负责验证连接，且没有人类发送者）

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

## 故障排查

**平台或工具缺失。**
使用 `hermes plugins list` 检查插件是否已在当前 profile 中安装并启用。若未安装，
运行 `hermes plugins install homeassistant`（或 `hermes -p <profile> plugins install homeassistant`），
然后重启 gateway。若关闭了 `security.allow_lazy_installs`，首次启动时的自动安装会被跳过，
需要您手动安装插件。

**环境变量未生效。**
适配器从 `~/.hermes/.env`（启动时自动合并）或 `config.yaml` 读取凭据。请确认该文件位于
当前 Hermes profile 主目录下，且 URL/令牌两侧没有多余的引号。编辑后请重启 gateway —
环境变量的变更只在进程启动时生效。

**REST 鉴权失败（`401 Unauthorized`）。**
令牌必须是在 HA 用户个人资料页面（**个人资料 → 安全 → 长期访问令牌**）创建的*长期访问令牌*，
短期的 UI 会话令牌无效。另请确认基础 URL 包含协议和端口（例如 `http://homeassistant.local:8123`），
并且运行 Hermes 的主机可以访问 — `curl -H "Authorization: Bearer <token>" <url>/api/`
应返回 `{"message": "API running."}`。
