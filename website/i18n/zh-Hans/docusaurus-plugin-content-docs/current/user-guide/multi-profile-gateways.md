---
sidebar_position: 4
---

# 同时运行多个网关

在一台机器上以托管服务形式运行多个[配置文件](./profiles.md) —— 每个配置文件有自己的机器人令牌、会话和记忆。本页涵盖运维注意事项：一起启动、跨配置文件查看日志、防止主机休眠，以及从常见的 launchd/systemd 问题中恢复。

如果你只运行一个 Hermes Agent，不需要本页 —— 请参阅[配置文件](./profiles.md)了解基础知识。

## 适用场景

当你有两个或多个需要同时在线的 Hermes Agent 时，就需要这种设置。常见原因：

- 一个 Telegram 机器人做个人助理，另一个做编程 Agent
- 每个家庭成员一个 Agent，或每个 Slack 工作区一个
- 同一配置的沙箱 + 生产实例
- 一个研究 Agent + 一个写作 Agent + 一个定时任务驱动的机器人 —— 各自有独立的记忆和技能

每个配置文件已经有自己的按平台 LaunchAgent（`ai.hermes.gateway-<name>.plist`）或 systemd 用户服务（`hermes-gateway-<name>.service`）。本指南增加了集体管理的模式。

## 快速开始

```bash
# Create profiles (once)
hermes profile create coder
hermes profile create personal-bot
hermes profile create research

# Configure each
coder setup
personal-bot setup
research setup

# Install each gateway as a managed service
coder gateway install
personal-bot gateway install
research gateway install

# Start them all
coder gateway start
personal-bot gateway start
research gateway start
```

就这样 —— 三个独立的 Agent，各自运行在独立进程中，崩溃和用户登录时自动重启。

## 替代方案：所有配置文件共用一个网关（多路复用）

上面的模型运行**每个配置文件一个进程**。这是默认值，也是大多数设置的正确选择。但在有许多配置文件的主机上 —— 或者每个配置文件一个进程在运维上过于沉重的容器部署中 —— 你可以改为运行**单个多路复用网关**：默认配置文件的网关成为唯一的入站进程，为机器上的*每个*配置文件服务。

这是**可选的**且**默认关闭**。关闭时，本页中的任何内容都不会改变 —— 以下所有行为都不生效。

### 何时优先使用多路复用

- 容器/VPS 部署中，N 个管理单元、N 个端口和 N 个 PID 文件是负担。
- 许多低流量配置文件，每个不值得一个完整进程。
- 你想要一个可以启动、监控和重启的单一对象。

当你想要配置文件之间的硬进程级隔离时（独立的内存占用、独立的崩溃域、能够重启一个配置文件而不影响其他），请坚持一个配置文件一个进程。

### 如何启用

在**默认配置文件**上设置标志并重启其网关：

```bash
hermes config set gateway.multiplex_profiles true
hermes gateway restart
```

等效地，在默认配置文件的 `~/.hermes/config.yaml` 中：

```yaml
gateway:
  multiplex_profiles: true
```

（为了方便，该标志也接受顶层 `multiplex_profiles: true`。）下次启动时，默认网关会枚举每个配置文件，在该配置文件自己的凭据下启动其启用的平台，并将每条入站消息路由到它所属的配置文件。每次轮次解析路由配置文件的配置、技能、记忆、SOUL 和**Provider 密钥** —— 凭据永远不会跨配置文件共享。

你**不需要**为次要配置文件运行 `hermes gateway start` —— 默认网关为它们服务。参见下方的契约变更。

### 多路复用启用后变更的内容

启用标志会改变一些东西的行为。这些都在标志关闭时立即恢复。

#### 1. 次级配置文件不能启动自己的网关

多路复用器运行时，命名配置文件的 `hermes gateway start` / `run` 会触发**硬错误**，指向多路复用器：

```
The default gateway is running as a profile multiplexer and already serves
profile 'coder'. ...
```

多路复用器是唯一的入站进程；第二个配置文件网关会双重绑定该配置文件的平台。仅当你确定想为该配置文件单独运行进程时才传递 `--force`（多路复用器运行时不推荐）。因此本页前面的跨配置文件生命周期包装脚本在多路复用模式下**不使用** —— 你只管理默认网关。

#### 2. HTTP 入站平台通过 `/p/<profile>/` URL 前缀访问

次级配置文件的 Webhook（和其他 HTTP 入站）流量到达默认监听器上的配置文件前缀，**而非第二个端口**：

```
# default profile
POST http://host:8644/webhooks/<route>
# the "coder" profile, same listener
POST http://host:8644/p/coder/webhooks/<route>
```

前缀中未知或未配置的配置文件返回 `404`。因为一个共享监听器已经以这种方式服务所有配置文件，**次级配置文件不得自行启用端口绑定平台** —— 这是配置错误，网关会拒绝启动并指明配置文件和平台：

```
Profile 'coder' enables the port-binding platform 'webhook', but
gateway.multiplex_profiles is on. ... Remove platforms.webhook from profile
'coder's config.yaml (configure it only on the default profile).
```

受此规则覆盖的端口绑定平台：`webhook`、`api_server`、`msgraph_webhook`、`feishu`、`wecom_callback`、`bluebubbles`、`sms`。**仅在默认配置文件**上配置其中任何一个；每个配置文件通过其 `/p/<profile>/` 前缀可达。

#### 3. 按凭据的平台仍需每个配置文件自己的令牌

轮询/连接平台（Telegram、Discord、Slack、Matrix、Signal……）在多路复用下工作正常，但启用它们的每个配置文件必须提供**自己的**机器人令牌 —— 同一个令牌不能被两个配置文件同时轮询。如果两个配置文件配置了相同的 `(platform, token)`，启动会快速失败并指明两个配置文件（参见[令牌冲突安全](#令牌冲突安全) —— 规则不变，只是现在在同一个进程内执行）。

#### 4. 会话键按配置文件命名空间化

每个配置文件的会话位于 `agent:<profile>:…` 命名空间下，因此同一平台/聊天上的两个配置文件永远不会在共享会话存储中冲突。**默认**配置文件保持历史的 `agent:main:…` 命名空间字节级不变，因此现有的默认配置文件会话不受影响 —— 无需迁移、无孤立历史。

#### 5. 一个 PID/锁和一个状态界面

有一个进程级的 PID 和锁（多路复用器，在默认 home 下）。`hermes status` 报告多路复用器及其服务的配置文件；`hermes status -p <name>` 切片到单个配置文件。每个配置文件仍在其自己的 home 下写入自己的 `runtime_status.json`，因此现有的按配置文件读取器继续工作。

#### **不变的内容**

按配置文件的 `.env` 凭据隔离被保持，甚至更严格：配置文件的密钥从其自己的作用域解析，永远不会联合到共享环境中（这也意味着 MCP 服务器和看板工作者等子进程只能看到自己配置文件的密钥）。看板、配置文件级的技能/记忆/SOUL 和模型路由都按配置文件运行，与独立网关时完全一样。

### 将共享机器人聊天路由到配置文件（`profile_routes`）

多路复用按**凭据**（每个配置文件自己的机器人令牌）或按 **URL 前缀**（HTTP 平台的 `/p/<profile>/`）选择配置文件。当多个社区共享**一个**机器人令牌时 —— 例如一个 Discord 机器人服务多个公会 —— 你可以通过 `gateway.profile_routes` 额外将特定公会/频道/线程路由到不同配置文件：

```yaml
gateway:
  multiplex_profiles: true
  profile_routes:
    # An entire Discord server → one profile
    - name: acme-server
      platform: discord
      guild_id: "1234567890"
      profile: acme

    # One channel in that server → a different profile
    - name: acme-support
      platform: discord
      guild_id: "1234567890"
      chat_id: "9876543210"
      profile: acme-support

    # A Telegram group (no guild concept — chat_id only)
    - name: tg-group
      platform: telegram
      chat_id: "-1001234567890"
      profile: tg-profile
```

路由按最具体优先匹配（`thread_id` > `chat_id` > `guild_id`），所有声明的字段必须全部满足（AND），按频道键的路由也匹配父频道为该频道的线程/论坛帖子。匹配不到路由的消息留在默认/活动配置文件上。路由的配置文件获得上面描述的完整按配置文件隔离（配置、技能、记忆、凭据、会话命名空间）。路由在每个平台适配器上工作，不仅限于 Discord。

`profile_routes` 需要 `gateway.multiplex_profiles: true`；多路复用关闭时路由被忽略。如果路由指定了磁盘上不存在的配置文件，网关会记录一条警告指明配置文件和来源，并回退到默认 home。

## 一次启动、停止或重启所有网关

CLI 提供单配置文件生命周期命令。要在每个配置文件上操作，用 shell 循环包装它们。将以下片段放入 `~/.local/bin/hermes-gateways` 并 `chmod +x`：

```sh
#!/bin/sh
set -eu

# Add or remove profile names here as you create / delete profiles.
profiles="default coder personal-bot research"

usage() {
  echo "Usage: hermes-gateways {start|stop|restart|status|list}"
}

run_for_profile() {
  profile="$1"
  action="$2"
  if [ "$profile" = "default" ]; then
    hermes gateway "$action"
  else
    hermes -p "$profile" gateway "$action"
  fi
}

action="${1:-}"
case "$action" in
  start|stop|restart|status)
    for profile in $profiles; do
      echo "==> $action $profile"
      run_for_profile "$profile" "$action"
    done
    ;;
  list)
    hermes gateway list
    ;;
  *)
    usage
    exit 2
    ;;
esac
```

然后：

```bash
hermes-gateways start      # start every configured profile
hermes-gateways stop       # stop every configured profile
hermes-gateways restart    # restart all
hermes-gateways status     # status across all
hermes-gateways list       # delegates to `hermes gateway list`
```

:::tip
`default` 配置文件使用 `hermes gateway <action>`（不带 `-p`）定位，而不是 `hermes -p default gateway <action>`。上面的包装器处理两种形式。
:::

## 管理单个配置文件

每个配置文件安装的快捷命令：

```bash
coder gateway run        # foreground (Ctrl-C to stop)
coder gateway start      # start the managed service
coder gateway stop       # stop the managed service
coder gateway restart    # restart
coder gateway status     # status
coder gateway install    # create the LaunchAgent / systemd unit
coder gateway uninstall  # remove the service file
```

等同于 `hermes -p coder gateway <action>` —— 在配置文件别名不在 `PATH` 上或你从脚本动态定位配置文件时很有用。

## 服务文件

每个配置文件安装自己的服务，使用唯一名称，因此安装永远不会冲突：

| 平台 | 路径 |
| --- | --- |
| macOS | `~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist` |
| Linux | `~/.config/systemd/user/hermes-gateway-<profile>.service` |

默认配置文件保持历史名称：`ai.hermes.gateway.plist` / `hermes-gateway.service`。

## 查看日志

每个配置文件写入自己的日志文件：

```bash
# Default profile
tail -f ~/.hermes/logs/gateway.log
tail -f ~/.hermes/logs/gateway.error.log

# Named profile
tail -f ~/.hermes/profiles/<name>/logs/gateway.log
tail -f ~/.hermes/profiles/<name>/logs/gateway.error.log
```

同时流式传输所有配置文件的日志：

```bash
tail -f ~/.hermes/logs/gateway.log ~/.hermes/profiles/*/logs/gateway.log
```

CLI 还有结构化日志查看器：

```bash
hermes logs -f                  # follow default profile
hermes -p coder logs -f         # follow one profile
hermes logs --help              # filters, levels, JSON output
```

## 识别实际运行的内容

```bash
hermes profile list             # profiles + model + gateway state
hermes-gateways status          # full status across every profile
launchctl list | grep hermes    # macOS — PIDs and labels
systemctl --user list-units 'hermes-gateway-*'   # Linux — units
```

## 编辑配置

每个配置文件在其自己的目录内保持配置：

```
~/.hermes/profiles/<name>/
├── .env              # API keys, bot tokens (chmod 600)
├── config.yaml       # model, provider, toolsets, gateway settings
└── SOUL.md           # personality / system prompt
```

默认配置文件直接使用 `~/.hermes/`，同样的三个文件。

使用任何编辑器或通过 CLI 编辑：

```bash
hermes config set model.model anthropic/claude-sonnet-4    # default profile
coder config set model.model openai/gpt-5                  # named profile
```

编辑 `.env` 或 `config.yaml` 后，重启受影响的网关：

```bash
coder gateway restart
# or, for everything:
hermes-gateways restart
```

## 保持主机唤醒

网关进程可以全天运行，但操作系统在空闲时仍会尝试休眠。两种模式：

### macOS —— `caffeinate`

`caffeinate` 内置于 macOS，在运行时阻止休眠。无需安装。

```bash
caffeinate -dis                    # block display, idle, and system sleep
caffeinate -dis -t 28800           # same, auto-exit after 8 hours
caffeinate -i -w $(cat ~/.hermes/gateway.pid) &   # awake while default gateway runs

# Persistent: run in background and forget
nohup caffeinate -dis >/dev/null 2>&1 &
disown

# Inspect / stop
pmset -g assertions | grep -iE 'caffeinate|prevent|user is active'
pkill caffeinate
```

| 标志 | 效果 |
| --- | --- |
| `-d` | 阻止显示休眠 |
| `-i` | 阻止空闲系统休眠（默认） |
| `-m` | 阻止磁盘休眠 |
| `-s` | 阻止系统休眠（仅限交流电供电的 Mac） |
| `-u` | 模拟用户活动（阻止屏幕锁定） |
| `-t N` | N 秒后自动退出 |
| `-w P` | 当 PID `P` 退出时退出 |

:::warning 合盖仍然会让 Mac 休眠
`caffeinate` 无法覆盖 MacBook 上硬件驱动的合盖休眠。合盖运行请更改节能/电池偏好设置或使用第三方工具。
:::

### Linux —— `systemd-inhibit` 或 `loginctl`

```bash
# Inhibit suspend while a command runs
systemd-inhibit --what=idle:sleep --who=hermes --why="gateways running" \
  sleep infinity &

# Allow user services to keep running after logout (recommended)
sudo loginctl enable-linger "$USER"
```

启用 lingering 后，你的 systemd 用户单元（包括 `hermes-gateway-<profile>.service`）在 SSH 断开和重启后继续运行。

## 令牌冲突安全

每个配置文件必须为每个平台使用唯一的机器人令牌。如果两个配置文件共享 Telegram、Discord、Slack、WhatsApp 或 Signal 令牌，第二个网关会拒绝启动并报错指明冲突的配置文件。

审计：

```bash
grep -H 'TELEGRAM_BOT_TOKEN\|DISCORD_BOT_TOKEN' \
     ~/.hermes/.env ~/.hermes/profiles/*/.env
```

## 更新代码

`hermes update` 拉取最新代码一次，并将新的内置技能同步到每个配置文件：

```bash
hermes update
hermes-gateways restart
```

用户修改的技能永远不会被覆盖。

## 故障排除

### "Could not find service in domain for user gui: 501"

你在之前 `hermes gateway stop` 后运行了 `hermes gateway start`。CLI 的 `stop` 执行完整的 `launchctl unload`，从 launchd 注册表中移除服务。CLI 在 `start` 时捕获此特定错误并自动重新加载 plist（`↻ launchd job was unloaded; reloading service definition`）。服务正常启动。无需修复。

### 崩溃后的残留 PID

如果配置文件的网关显示 `not running` 但进程仍在运行：

```bash
ps -ef | grep "hermes_cli.*-p <profile>"
cat ~/.hermes/profiles/<profile>/gateway.pid
kill -TERM <pid>          # graceful
kill -KILL <pid>          # if that fails after a few seconds
<profile> gateway start
```

### 强制重置一个服务

```bash
# macOS
launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist
launchctl load   ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist

# Linux
systemctl --user restart hermes-gateway-<profile>.service
```

### 健康检查

```bash
hermes doctor                  # default profile
hermes -p <profile> doctor     # one profile
```
