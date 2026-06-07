---
title: 浏览器自动化
description: 通过多种提供商控制浏览器，支持通过 CDP 连接本地 Chromium 系浏览器或云端浏览器，用于网页交互、表单填写、数据抓取等场景。
sidebar_label: Browser
sidebar_position: 5
---

# 浏览器自动化

Hermes Agent 内置完整的浏览器自动化工具集，支持多种后端选项：

- **Browserbase 云端模式** — 通过 [Browserbase](https://browserbase.com) 使用托管云端浏览器及反机器人工具
- **Browser Use 云端模式** — 通过 [Browser Use](https://browser-use.com) 作为备选云端浏览器提供商
- **Firecrawl 云端模式** — 通过 [Firecrawl](https://firecrawl.dev) 使用内置抓取功能的云端浏览器
- **Camofox 本地模式** — 通过 [Camofox](https://github.com/jo-inc/camofox-browser) 实现本地反检测浏览（基于 Firefox 的指纹伪装）
- **本地 Chromium 系 CDP** — 使用 `/browser connect` 将浏览器工具连接到本地运行的 Chrome、Brave、Chromium 或 Edge 实例
- **本地浏览器模式** — 通过 `agent-browser` CLI 和本地 Chromium 安装运行

所有模式下，Agent 均可导航网站、与页面元素交互、填写表单并提取信息。

## 概述

页面以**无障碍树**（accessibility tree，基于文本的快照）表示，非常适合 LLM Agent 使用。交互元素会获得引用 ID（如 `@e1`、`@e2`），Agent 通过这些 ID 执行点击和输入操作。

核心能力：

- **多提供商云端执行** — Browserbase、Browser Use 或 Firecrawl — 无需本地浏览器
- **本地 Chromium 系集成** — 通过 CDP 连接正在运行的 Chrome、Brave、Chromium 或 Edge 浏览器，实现实时操控
- **内置隐身功能** — 随机指纹、CAPTCHA 解决、住宅代理（Browserbase）
- **会话隔离** — 每个任务拥有独立的浏览器会话
- **自动清理** — 非活跃会话在超时后自动关闭
- **视觉分析** — 截图 + AI 分析，实现视觉理解

## 配置

:::tip Nous 订阅用户
如果您拥有付费 [Nous Portal](https://portal.nousresearch.com) 订阅，可通过 **[Tool Gateway](tool-gateway.md)** 使用浏览器自动化功能，无需单独的 API 密钥。新安装可运行 `hermes setup --portal` 登录并一次性开启所有 gateway 工具；已有安装可通过 `hermes model` 或 `hermes tools` 选择 **Nous Subscription** 作为浏览器提供商。
:::

### Browserbase 云端模式

要使用 Browserbase 托管的云端浏览器，请添加：

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

在 [browserbase.com](https://browserbase.com) 获取您的凭据。

### Browser Use 云端模式

要使用 Browser Use 作为云端浏览器提供商，请添加：

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

在 [browser-use.com](https://browser-use.com) 获取 API 密钥。Browser Use 通过 REST API 提供云端浏览器。若同时设置了 Browserbase 和 Browser Use 凭据，Browserbase 优先。

### Firecrawl 云端模式

要使用 Firecrawl 作为云端浏览器提供商，请添加：

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***
```

在 [firecrawl.dev](https://firecrawl.dev) 获取 API 密钥，然后选择 Firecrawl 作为浏览器提供商：

```bash
hermes setup tools
# → Browser Automation → Firecrawl
```

可选配置：

```bash
# Self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

### 混合路由：公网 URL 使用云端，LAN/localhost 使用本地

配置云端提供商后，Hermes 会为解析到私有/回环/LAN 地址的 URL（`localhost`、`127.0.0.1`、`192.168.x.x`、`10.x.x.x`、`172.16-31.x.x`、`*.local`、`*.lan`、`*.internal`、IPv6 回环 `::1`、链路本地 `169.254.x.x`）自动启动一个**本地 Chromium 辅助进程**。公网 URL 在同一对话中继续使用云端提供商。

这解决了常见的"本地开发但使用 Browserbase"场景 — Agent 可以截取 `http://localhost:3000` 上的仪表盘，同时抓取 `https://github.com`，无需切换提供商或禁用 SSRF 防护。云端提供商永远不会看到私有 URL。

该功能**默认开启**。如需禁用（所有 URL 均走已配置的云端提供商，与之前行为一致）：

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

禁用自动路由后，私有 URL 将被拒绝并返回 `"Blocked: URL targets a private or internal address"`，除非同时设置 `browser.allow_private_urls: true`（允许云端提供商尝试访问，但通常无法成功，因为 Browserbase 等无法访问您的 LAN）。

要求：本地辅助进程使用与纯本地模式相同的 `agent-browser` CLI，因此需要先安装（`hermes setup tools → Browser Automation` 会自动安装）。从公网 URL 导航后重定向到私有地址的情况仍会被阻止（无法通过公网路径的重定向访问 LAN）。

### Camofox 本地模式

[Camofox](https://github.com/jo-inc/camofox-browser) 是一个自托管的 Node.js 服务器，封装了 Camoufox（一个带有 C++ 指纹伪装的 Firefox 分支）。它无需云端依赖即可提供本地反检测浏览。

```bash
# Clone the Camofox browser server first
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser

# Build and start with Docker using the default container settings
# (auto-detects arch: aarch64 on M1/M2, x86_64 on Intel)
make up

# Stop and remove the default container
make down

# Force a clean rebuild (for example, after upgrading VERSION/RELEASE)
make reset

# Just download binaries without building
make fetch

# Override arch or version explicitly
make up ARCH=x86_64
make up VERSION=135.0.1 RELEASE=beta.24
```

`make up` 会立即启动默认容器。如需自定义运行时设置（如更大的 Node 堆内存、VNC 或持久化 profile 目录），请先构建镜像再手动运行：

```bash
# Build the image without starting the default container
make build

# Start with persistence, VNC live view, and a larger Node heap
mkdir -p ~/.camofox-docker
docker run -d \
  --name camofox-browser \
  --restart unless-stopped \
  -p 9377:9377 \
  -p 6080:6080 \
  -p 5901:5900 \
  -e CAMOFOX_PORT=9377 \
  -e ENABLE_VNC=1 \
  -e VNC_BIND=0.0.0.0 \
  -e VNC_RESOLUTION=1920x1080 \
  -e MAX_OLD_SPACE_SIZE=2048 \
  -v ~/.camofox-docker:/root/.camofox \
  camofox-browser:135.0.1-aarch64
```

启用 VNC 后，浏览器以有头模式运行，可在浏览器中通过 `http://localhost:6080`（noVNC）实时查看。也可使用原生 VNC 客户端连接 `localhost:5901`。

如果已运行过 `make up`，请在启动自定义容器前先停止并删除默认容器：

```bash
make down
# then run the custom docker run command above
```

然后在 `~/.hermes/.env` 中设置：

```bash
CAMOFOX_URL=http://localhost:9377
```

或通过 `hermes tools` → Browser Automation → Camofox 进行配置。

设置 `CAMOFOX_URL` 后，所有浏览器工具将自动通过 Camofox 路由，而非 Browserbase 或 agent-browser。

#### 持久化浏览器会话

默认情况下，每个 Camofox 会话使用随机身份 — Cookie 和登录状态不会在 Agent 重启后保留。要启用持久化浏览器会话，请在 `~/.hermes/config.yaml` 中添加：

```yaml
browser:
  camofox:
    managed_persistence: true
```

然后完全重启 Hermes 以使新配置生效。

:::warning 嵌套路径很重要
Hermes 读取的是 `browser.camofox.managed_persistence`，**而非**顶层的 `managed_persistence`。常见错误写法：

```yaml
# ❌ Wrong — Hermes ignores this
managed_persistence: true
```

如果该标志放在错误的路径下，Hermes 会静默回退到随机临时 `userId`，您的登录状态将在每次会话后丢失。
:::

##### Hermes 的行为
- 向 Camofox 发送确定性的 profile 范围 `userId`，使服务器能够跨会话复用同一 Firefox profile。
- 在清理时跳过服务端 context 销毁，使 Cookie 和登录状态在 Agent 任务间保留。
- 将 `userId` 限定在当前 Hermes profile 范围内，不同 Hermes profile 对应不同浏览器 profile（profile 隔离）。

##### Hermes 不做的事
- 不会强制 Camofox 服务器持久化。Hermes 只发送稳定的 `userId`；服务器必须通过将该 `userId` 映射到持久化 Firefox profile 目录来支持它。
- 如果您的 Camofox 服务器构建将每个请求视为临时的（例如始终调用 `browser.newContext()` 而不加载已存储的 profile），Hermes 无法使这些会话持久化。请确保运行的 Camofox 版本实现了基于 userId 的 profile 持久化。

##### 验证是否正常工作

1. 启动 Hermes 和 Camofox 服务器。
2. 在浏览器任务中打开 Google（或任意登录网站）并手动登录。
3. 正常结束浏览器任务。
4. 开始新的浏览器任务。
5. 再次打开同一网站 — 应仍处于登录状态。

如果第 5 步退出了登录，说明 Camofox 服务器未遵守稳定的 `userId`。请检查配置路径，确认编辑 `config.yaml` 后已完全重启 Hermes，并验证您的 Camofox 服务器版本是否支持基于用户的持久化 profile。

##### 状态存储位置

Hermes 从 profile 范围目录 `~/.hermes/browser_auth/camofox/`（非默认 profile 则在 `$HERMES_HOME` 下的对应位置）派生稳定的 `userId`。实际浏览器 profile 数据存储在 Camofox 服务器端，以该 `userId` 为键。要完全重置持久化 profile，请在 Camofox 服务器端清除对应数据，并删除相应 Hermes profile 的状态目录。

#### 外部管理的 Camofox 会话

当另一个应用驱动可见的 Camofox 浏览器（桌面助手、自定义集成、另一个 Agent）时，可配置 Hermes 在同一身份下运行，而非启动独立的隔离 profile。

三个参数控制行为：

| 设置 | 环境变量 | 效果 |
|---------|---------|--------|
| `browser.camofox.user_id` | `CAMOFOX_USER_ID` | Hermes 创建标签页时使用的 Camofox `userId`。设置此项即进入"外部管理"模式。 |
| `browser.camofox.session_key` | `CAMOFOX_SESSION_KEY` | 创建标签页时发送的 `sessionKey`（即 `listItemId`）。用于接管时匹配已有标签页。未设置时默认为每任务值。 |
| `browser.camofox.adopt_existing_tab` | `CAMOFOX_ADOPT_EXISTING_TAB` | 为 true 时，Hermes 在首次使用时调用 `GET /tabs?userId=<user_id>` 并优先复用已有标签页，而非新建。 |

环境变量优先于 `config.yaml`。两种形式均可：

```yaml
browser:
  camofox:
    user_id: shared-camofox
    session_key: visible-tab
    adopt_existing_tab: true
```

```bash
CAMOFOX_USER_ID=shared-camofox
CAMOFOX_SESSION_KEY=visible-tab
CAMOFOX_ADOPT_EXISTING_TAB=true
```

**设置 `user_id` 后的变化：**

- Hermes 在任务结束时跳过破坏性清理（与 `managed_persistence: true` 相同）。其他应用的标签页/Cookie/profile 得以保留。
- Hermes **不会**调用 `DELETE /sessions/<user_id>` — 该端点会清除所有用户数据，若触发将销毁外部应用的会话。

**标签页接管的工作方式（当 `adopt_existing_tab: true` 时）：**

1. 进程启动后首次调用浏览器工具时，Hermes 发出 `GET /tabs?userId=<user_id>`（5 秒超时）。
2. 若响应中有标签页的 `listItemId == session_key`，Hermes 接管该组中最近创建的一个。
3. 否则，Hermes 接管该用户最近创建的标签页（任意 `listItemId`）。
4. 若无标签页或请求失败，Hermes 在下次操作时回退到新建标签页。

接管仅在会话的 `tab_id` 填充之前触发一次。若外部应用在运行中关闭了被接管的标签页，下次浏览器工具调用将返回 Camofox 错误 — Hermes 不会在每次调用时重新轮询新标签页。

**选择 `session_key`：** 若要 Hermes 可靠地附加到*特定*已有标签页，请将 `session_key` 设置为外部应用创建该标签页时使用的 `listItemId`。若只设置 `user_id` 而不设置 `session_key`，Hermes 会生成每任务的 `session_key`（`task_<id>`）— Hermes 将与外部应用共享 Cookie 和 profile，但会并排打开自己的标签页而非复用已有标签页。

**并发说明：** 外部应用和 Hermes 可同时驱动同一 Camofox `userId`，但 Camofox 不会在客户端之间协调每个标签页的焦点。请在应用层协调所有权（例如，Hermes 运行时外部应用暂停）。

#### VNC 实时查看

当 Camofox 以有头模式运行（带可见浏览器窗口）时，其健康检查响应中会暴露 VNC 端口。Hermes 自动发现此信息，并在导航响应中包含 VNC URL，Agent 可分享链接供您实时查看浏览器。

### 通过 CDP 连接本地 Chromium 系浏览器（`/browser connect`）

除云端提供商外，您还可以通过 Chrome DevTools Protocol（CDP）将 Hermes 浏览器工具连接到本地运行的 Chrome、Brave、Chromium 或 Edge 实例。当您希望实时查看 Agent 操作、与需要自身 Cookie/会话的页面交互，或避免云端浏览器费用时，此方式非常有用。

:::note
`/browser connect` 是**交互式 CLI 斜杠命令** — 不由 gateway 分发。若在 WebUI、Telegram、Discord 或其他 gateway 聊天中尝试运行，消息将作为纯文本发送给 Agent，命令不会执行。请从终端启动 Hermes（`hermes` 或 `hermes chat`）并在那里执行 `/browser connect`。
:::

在 CLI 中使用：

```
/browser connect                 # Auto-launch/connect to a local Chromium-family browser at http://127.0.0.1:9222
/browser connect ws://host:port  # Connect to a specific CDP endpoint
/browser status                  # Check current connection
/browser disconnect              # Detach and return to cloud/local mode
```

若浏览器尚未以远程调试模式运行，Hermes 将尝试自动启动支持的 Chromium 系浏览器并使用 `--remote-debugging-port=9222`。检测范围包括 Brave、Brave Origin/Nightly、Google Chrome、Chromium 和 Microsoft Edge，以及常见 Linux 安装路径和二进制名称（如 `brave-origin`、`brave-origin-nightly`、`/opt/brave.com/brave-origin/brave-origin`、`/opt/brave.com/brave-origin-nightly/brave-origin`、`/opt/brave-bin/brave` 和 `/snap/bin/brave`）。

:::tip
要手动启动带 CDP 的 Chromium 系浏览器，请使用专用的 user-data-dir，确保即使浏览器已以普通 profile 运行，调试端口也能正常开启：

```bash
# Linux — Brave
brave-browser \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# Linux — Google Chrome
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# macOS — Brave
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &

# macOS — Google Chrome
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &
```

然后启动 Hermes CLI 并运行 `/browser connect`。

**为什么需要 `--user-data-dir`？** 若不指定，在普通实例已运行时启动 Chromium 系浏览器通常只会在现有进程上打开新窗口 — 而该进程启动时未带 `--remote-debugging-port`，因此端口 9222 永远不会开启。专用的 user-data-dir 会强制启动新的浏览器进程，使调试端口正常监听。`--no-first-run --no-default-browser-check` 跳过新 profile 的首次启动向导。
:::

通过 CDP 连接后，所有浏览器工具（`browser_navigate`、`browser_click` 等）将在您的实时浏览器实例上运行，而非启动云端会话。

### WSL2 + Windows Chrome：优先使用 MCP 而非 `/browser connect`

若 Hermes 在 WSL2 内运行，但您想控制的 Chrome 窗口在 Windows 宿主机上，`/browser connect` 通常不是最佳方案。

原因：

- `/browser connect` 要求 Hermes 本身能访问可用的 CDP 端点
- 现代 Chrome 实时调试会话通常暴露仅宿主机本地可访问的端点，WSL 无法像访问经典 `9222` 端口那样直接访问
- 即使 Windows Chrome 可调试，最简洁的集成方式通常是让 Windows 侧的浏览器 MCP 服务器连接 Chrome，再让 Hermes 与该 MCP 服务器通信

对于此场景，建议通过 Hermes MCP 支持使用 `chrome-devtools-mcp`。

具体配置请参阅 MCP 指南：

- [在 Hermes 中使用 MCP](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

### 本地浏览器模式

若**未**设置任何云端凭据且未使用 `/browser connect`，Hermes 仍可通过由 `agent-browser` 驱动的本地 Chromium 安装使用浏览器工具。

### 可选环境变量

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in milliseconds (default: project default)
# Examples: 600000 (10min), 1800000 (30min)
BROWSERBASE_SESSION_TIMEOUT=600000

# Inactivity timeout before auto-cleanup in seconds (default: 120)
BROWSER_INACTIVITY_TIMEOUT=120

# Extra Chromium launch flags (comma- or newline-separated). Hermes auto-injects
# `--no-sandbox,--disable-dev-shm-usage` when it detects root or AppArmor-restricted
# unprivileged user namespaces (Ubuntu 23.10+, DGX Spark, many container images),
# so most users don't need to set this. Set it manually only if you need a flag
# Hermes doesn't add automatically; setting it disables the auto-injection.
AGENT_BROWSER_ARGS=--no-sandbox
```

### 安装 agent-browser CLI

```bash
npm install -g agent-browser
# Or install locally in the repo:
npm install
```

:::info
`browser` 工具集必须包含在配置的 `toolsets` 列表中，或通过 `hermes config set toolsets '["hermes-cli", "browser"]'` 启用。
:::

## 可用工具

### `browser_navigate`

导航到指定 URL。必须在其他任何浏览器工具之前调用。初始化 Browserbase 会话。

```
Navigate to https://github.com/NousResearch
```

:::tip
对于简单的信息检索，优先使用 `web_search` 或 `web_extract` — 它们更快且成本更低。仅在需要**与页面交互**（点击按钮、填写表单、处理动态内容）时使用浏览器工具。
:::

### `browser_snapshot`

获取当前页面无障碍树的文本快照。返回带有引用 ID（如 `@e1`、`@e2`）的交互元素，供 `browser_click` 和 `browser_type` 使用。

- **`full=false`**（默认）：仅显示交互元素的紧凑视图
- **`full=true`**：完整页面内容

超过 8000 字符的快照将由 LLM 自动摘要。

### `browser_click`

点击快照中由引用 ID 标识的元素。

```
Click @e5 to press the "Sign In" button
```

### `browser_type`

向输入框输入文本。先清空字段，再输入新文本。

```
Type "hermes agent" into the search field @e3
```

### `browser_scroll`

向上或向下滚动页面以显示更多内容。

```
Scroll down to see more results
```

### `browser_press`

按下键盘按键。适用于提交表单或导航。

```
Press Enter to submit the form
```

支持的按键：`Enter`、`Tab`、`Escape`、`ArrowDown`、`ArrowUp` 等。

### `browser_back`

在浏览器历史记录中返回上一页。

### `browser_get_images`

列出当前页面上所有图片及其 URL 和 alt 文本。适用于查找需要分析的图片。

### `browser_vision`

截图并使用视觉 AI 进行分析。当文本快照无法捕获重要视觉信息时使用 — 尤其适用于 CAPTCHA、复杂布局或视觉验证挑战。

截图会持久保存，文件路径与 AI 分析结果一并返回。在消息平台（Telegram、Discord、Slack、WhatsApp）上，您可以要求 Agent 分享截图 — 它将通过 `MEDIA:` 机制作为原生图片附件发送。

```
What does the chart on this page show?
```

截图存储在 `~/.hermes/cache/screenshots/`，24 小时后自动清理。

### `browser_console`

获取当前页面的浏览器控制台输出（log/warn/error 消息）及未捕获的 JavaScript 异常。对于检测无障碍树中不可见的静默 JS 错误至关重要。

```
Check the browser console for any JavaScript errors
```

使用 `clear=True` 可在读取后清空控制台，使后续调用只显示新消息。

`browser_console` 在带有 `expression` 参数调用时也可执行 JavaScript — 与 DevTools 控制台形式相同，结果以解析后的形式返回（JSON 序列化的对象变为 dict；原始值保持原始类型）。

```
browser_console(expression="document.querySelector('h1').textContent")
browser_console(expression="JSON.stringify(performance.timing)")
```

当当前会话存在活跃的 CDP 监督器时（通常适用于任何对 CDP 兼容后端运行过 `browser_navigate` 的会话），执行通过监督器的持久 WebSocket 进行 — 无子进程启动开销。否则回退到标准 agent-browser CLI 路径。两种方式行为完全相同，仅延迟有差异。

### `browser_cdp`

原始 Chrome DevTools Protocol 直通 — 用于其他工具未覆盖的浏览器操作的逃生舱口。适用于原生对话框处理、iframe 范围内的执行、Cookie/网络控制，或 Agent 需要的任何 CDP 命令。

**仅在会话启动时 CDP 端点可访问的情况下可用** — 即 `/browser connect` 已连接到运行中的 Chrome、Brave、Chromium 或 Edge 浏览器，或 `config.yaml` 中设置了 `browser.cdp_url`。默认本地 agent-browser 模式、Camofox 和云端提供商（Browserbase、Browser Use、Firecrawl）目前不向此工具暴露 CDP — 云端提供商有每会话 CDP URL，但实时会话路由是后续功能。

**CDP 方法参考：** https://chromedevtools.github.io/devtools-protocol/ — Agent 可通过 `web_extract` 访问特定方法页面以查阅参数和返回结构。

常见用法：

```
# List tabs (browser-level, no target_id)
browser_cdp(method="Target.getTargets")

# Handle a native JS dialog on a tab
browser_cdp(method="Page.handleJavaScriptDialog",
            params={"accept": true, "promptText": ""},
            target_id="<tabId>")

# Evaluate JS in a specific tab
browser_cdp(method="Runtime.evaluate",
            params={"expression": "document.title", "returnByValue": true},
            target_id="<tabId>")

# Get all cookies
browser_cdp(method="Network.getAllCookies")
```

浏览器级方法（`Target.*`、`Browser.*`、`Storage.*`）省略 `target_id`。页面级方法（`Page.*`、`Runtime.*`、`DOM.*`、`Emulation.*`）需要来自 `Target.getTargets` 的 `target_id`。每次无状态调用相互独立 — 调用间不保留会话状态。

**跨域 iframe：** 传入 `frame_id`（来自 `browser_snapshot.frame_tree.children[]` 中 `is_oopif=true` 的条目）可通过监督器的实时会话路由该 iframe 的 CDP 调用。这是在 Browserbase 上对跨域 iframe 执行 `Runtime.evaluate` 的方式，避免无状态 CDP 连接遭遇签名 URL 过期问题。示例：

```
browser_cdp(
  method="Runtime.evaluate",
  params={"expression": "document.title", "returnByValue": True},
  frame_id="<frame_id from browser_snapshot>",
)
```

同域 iframe 无需 `frame_id` — 在顶层 `Runtime.evaluate` 中使用 `document.querySelector('iframe').contentDocument` 即可。

### `browser_dialog`

响应原生 JS 对话框（`alert` / `confirm` / `prompt` / `beforeunload`）。在此工具出现之前，对话框会静默阻塞页面的 JavaScript 线程，后续 `browser_*` 调用会挂起或抛出异常；现在 Agent 可在 `browser_snapshot` 输出中看到待处理对话框并显式响应。

**工作流程：**
1. 调用 `browser_snapshot`。若对话框正在阻塞页面，将显示为 `pending_dialogs: [{"id": "d-1", "type": "alert", "message": "..."}]`。
2. 调用 `browser_dialog(action="accept")` 或 `browser_dialog(action="dismiss")`。对于 `prompt()` 对话框，传入 `prompt_text="..."` 提供响应内容。
3. 重新快照 — `pending_dialogs` 为空；页面 JS 线程已恢复。

**检测通过持久 CDP 监督器自动进行** — 每个任务一个 WebSocket，订阅 Page/Runtime/Target 事件。监督器还会在快照中填充 `frame_tree` 字段，使 Agent 可查看当前页面的 iframe 结构，包括跨域（OOPIF）iframe。

**可用性矩阵：**

| 后端 | 通过 `pending_dialogs` 检测 | 响应（`browser_dialog` 工具） |
|---|---|---|
| 通过 `/browser connect` 或 `browser.cdp_url` 连接的本地 Chrome | ✓ | ✓ 完整工作流 |
| Browserbase | ✓ | ✓ 完整工作流（通过注入的 XHR 桥接） |
| Camofox / 默认本地 agent-browser | ✗ | ✗（无 CDP 端点） |

**在 Browserbase 上的工作原理。** Browserbase 的 CDP 代理会在约 10ms 内在服务端自动关闭真实的原生对话框，因此无法使用 `Page.handleJavaScriptDialog`。监督器通过 `Page.addScriptToEvaluateOnNewDocument` 注入一段小脚本，将 `window.alert`/`confirm`/`prompt` 替换为同步 XHR。我们通过 `Fetch.enable` 拦截这些 XHR — 页面 JS 线程在 XHR 上保持阻塞，直到我们用 Agent 的响应调用 `Fetch.fulfillRequest`。`prompt()` 的返回值原样传回页面 JS。

**对话框策略**在 `config.yaml` 的 `browser.dialog_policy` 下配置：

| 策略 | 行为 |
|--------|----------|
| `must_respond`（默认） | 捕获，在快照中显示，等待显式 `browser_dialog()` 调用。在 `browser.dialog_timeout_s`（默认 300 秒）后安全自动关闭，防止有问题的 Agent 永久阻塞。 |
| `auto_dismiss` | 捕获，立即关闭。Agent 仍可在 `browser_state` 历史中看到对话框，但无需操作。 |
| `auto_accept` | 捕获，立即接受。适用于导航带有频繁 `beforeunload` 提示的页面。 |

`browser_snapshot.frame_tree` 中的**帧树**上限为 30 帧、OOPIF 深度 2，以控制广告密集页面的负载大小。达到限制时会显示 `truncated: true` 标志；需要完整帧树的 Agent 可使用 `browser_cdp` 配合 `Page.getFrameTree`。

## 实际示例

### 填写网页表单

```
User: Sign up for an account on example.com with my email john@example.com

Agent workflow:
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → sees form fields with refs
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → clicks "Create Account"
6. browser_snapshot()  → confirms success
```

### 研究动态内容

```
User: What are the top trending repos on GitHub right now?

Agent workflow:
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → reads trending repo list
3. Returns formatted results
```

## 会话录制

自动将浏览器会话录制为 WebM 视频文件：

```yaml
browser:
  record_sessions: true  # default: false
```

启用后，录制在首次 `browser_navigate` 时自动开始，会话关闭时保存到 `~/.hermes/browser_recordings/`。本地模式和云端模式（Browserbase）均支持。超过 72 小时的录制文件自动清理。

## 隐身功能

Browserbase 提供自动隐身能力：

| 功能 | 默认状态 | 说明 |
|---------|---------|-------|
| 基础隐身 | 始终开启 | 随机指纹、视口随机化、CAPTCHA 解决 |
| 住宅代理 | 开启 | 通过住宅 IP 路由以提高访问成功率 |
| 高级隐身 | 关闭 | 自定义 Chromium 构建，需要 Scale 计划 |
| Keep Alive | 开启 | 网络中断后的会话重连 |

:::note
若付费功能在您的计划中不可用，Hermes 会自动降级 — 先禁用 `keepAlive`，再禁用代理 — 确保免费计划也能正常浏览。
:::

## 会话管理

- 每个任务通过 Browserbase 获得独立的浏览器会话
- 非活跃会话在超时后自动清理（默认：2 分钟）
- 后台线程每 30 秒检查一次过期会话
- 进程退出时执行紧急清理，防止孤立会话
- 通过 Browserbase API 释放会话（`REQUEST_RELEASE` 状态）

## 限制

- **基于文本的交互** — 依赖无障碍树，而非像素坐标
- **快照大小** — 大型页面可能在 8000 字符处被截断或由 LLM 摘要
- **会话超时** — 云端会话根据提供商计划设置过期
- **费用** — 云端会话消耗提供商额度；对话结束或非活跃后会话自动清理。使用 `/browser connect` 可免费本地浏览。
- **不支持文件下载** — 无法从浏览器下载文件