---
title: 浏览器自动化
description: 使用 browser_exec 驱动真实浏览器——Hermes 启动的 Chromium、云端浏览器、Lightpanda，或通过 CDP 连接您自己的 Chrome——用于网页交互、表单填写、数据抓取等场景。
sidebar_label: Browser
sidebar_position: 5
---

# 浏览器自动化

`browser` 工具集为 Agent 提供一个真实的网页浏览器。真正负责驱动的只有一个工具——**`browser_exec`**，它通过 Chrome DevTools Protocol（CDP）在浏览器上运行模型编写的 Python；另有三个配套工具（`browser_cdp` 和 `browser_dialog` 仅在配置了 CDP 端点时提供，见[配套工具](#配套工具)）：

| 工具 | 用途 |
|------|------|
| `browser_exec` | 通过运行预导入浏览器辅助函数的 Python 来导航、读取、点击、输入、抓取和截图。 |
| `browser_cdp` | 发送单条原始 CDP 命令（Cookie、网络控制、iframe 范围内执行，以及辅助函数未覆盖的任何操作）。 |
| `browser_dialog` | 接受或关闭阻塞页面的原生 JavaScript 对话框（`alert` / `confirm` / `prompt` / `beforeunload`）。 |
| `browser_vault` | 将已保存的登录信息、银行卡、地址和 2FA 验证码填入当前页面，机密信息不会进入对话。 |

浏览器来源是另一项独立选择：

- **本地 Chromium（默认）**——Hermes 启动自己打包的 Chromium，无头模式，每个会话一个浏览器。无需 API 密钥，无需配置。
- **云端浏览器**——[Nous 订阅](tool-gateway.md)、[Browserbase](https://browserbase.com)、[Browser Use](https://browser-use.com) 云端、[Firecrawl](https://firecrawl.dev)，或浏览器提供商插件。
- **[Lightpanda](https://lightpanda.io)**——轻量级无头引擎，适合在小型机器上以文本为主的浏览。
- **您自己的浏览器**——用 `/browser connect` 连接正在运行的 Chrome、Brave、Chromium 或 Edge。
- **您的真实 profile**——您日常浏览器 profile 的托管副本，包含登录状态（需主动开启）。

:::tip
对于简单的信息检索，优先使用 `web_search` 或 `web_extract`——它们更快且成本更低。仅在 Agent 需要**与页面交互**（点击按钮、填写表单、处理动态内容）时使用浏览器。
:::

## `browser_exec` 的工作方式

每次调用发送一段 Python（`code`），运行时浏览器辅助函数已预先导入；代码打印的内容即为返回结果。浏览器会话在调用之间保持（已打开的标签页、Cookie、当前页面），Python 变量则不保留——每次调用都从全新的解释器开始。每个会话有一个工作目录（`$BH_AGENT_WORKSPACE`，也会作为 `workspace` 返回），在调用之间保留，因此长时间的抓取可以把每批结果追加到文件中，最后再汇总。

辅助函数：

| 辅助函数 | 作用 |
|----------|------|
| `new_tab(url)` | 打开 URL（用于第一次导航）。 |
| `goto_url(url)` | 在当前标签页导航（后续导航）。 |
| `wait_for_load()` | 等待页面加载完成。 |
| `page_info()` | 概述当前页面状态。 |
| `js(expr)` | 执行 JavaScript 并返回值（函数体需写成 `js('(() => {...})()')`）。 |
| `fill_input(selector, text)` | 向输入框输入文本。 |
| `click_at_xy(x, y)` | 点击视口坐标。 |
| `capture_screenshot()` | 保存截图并打印其路径。 |
| `cdp('Domain.method', **kwargs)` | 原始 CDP——例如 `cdp('Accessibility.getFullAXTree')` 获取无障碍树，`cdp('DOM.getBoxModel', backendNodeId=n)` 获取点击坐标。 |
| `ensure_real_tab()` | 从失效或内部标签页中恢复。 |

模型会把一整段子流程（导航、等待、提取、操作）合并到一次调用中，而不是每个动作调用一次：

```python
# Searching GitHub trending for Python repos
new_tab("https://github.com/trending/python")
wait_for_load()
print(js("[...document.querySelectorAll('article h2 a')].slice(0, 10).map(a => a.textContent.trim())"))
```

**截图。** 在具备原生视觉能力的模型上，`capture_screenshot()` 保存的图片会附加到工具结果中，模型直接查看。纯文本模型会被告知以文本为主（`page_info()`、`js()`、`fill_input()`），仍可把保存的路径交给 `vision_analyze`。在消息平台上，您可以让 Agent 把截图发给您——它会以原生图片附件送达。

**参数。** 除 `code` 外，该工具还接受 `timeout_s`（默认 300，最大 1800）和 `session`。每个 `session` 名称拥有自己的驱动进程和自己的浏览器（本地是独立的 Chromium，云端提供商则是独立的云端浏览器），因此并行的子 Agent 或同时进行的聊天不会互相干扰同一个浏览器。省略 `session` 则使用共享的默认会话，一次只做一件浏览任务时完全够用。开启[真实 profile 浏览](#real-profile-browsing-use-your-own-logins)后，该工具还会提供一个 `local` 参数（见该小节）。

:::note
`browser_exec` 会在您的机器上执行模型编写的 Python，因此只提供给同时拥有 `terminal` 工具集的会话。未配置终端访问的平台（例如受限的消息平台）不会获得浏览器。
:::

驱动程序是 [browser-harness](https://github.com/browser-use/browser-use)，即 Browser Use CLI 的引擎。它是 Hermes 的常规 Python 依赖，因此每个安装（包括桌面应用）都自带它，无需单独下载。

## 本地 Chromium（默认） {#local-chromium-default}

在未配置云端提供商、`/browser connect` 端点或真实 profile 开关时，Hermes 会自己为会话启动一个 Chromium，并让 `browser_exec` 指向它。您安装的 Chrome 永远不会被触碰——无需开启 `chrome://inspect` 开关，也不会弹出"允许远程调试？"对话框——在完全没有桌面浏览器的无头服务器上也能工作。

- **每个会话一个浏览器**，默认使用全新的一次性 profile：Agent 起始时未登录任何网站，会话结束后不留下任何内容。
- **默认无头。** 设置 `browser.headed: true` 可显示窗口（见[下文](#headed-mode-visible-browser-window)）。
- **自动清理。** 浏览器空闲 `browser.inactivity_timeout` 秒（默认 120）后、Hermes 退出时，以及进程崩溃后的孤儿清理中都会被关闭。

**使用哪个 Chromium。** Hermes 按顺序选择第一个存在的：

1. `browser.executable_path`（如已设置）。
2. Hermes 包管理器安装的 Chromium（Chrome for Testing）——安装脚本和 `hermes update` 默认会安装它。
3. 官方 Docker 镜像中的 `/etc/hermes/chromium-executable-path`。
4. `PATH` 上的系统 `chromium` / `chromium-browser`（Android/Termux，以及包管理器没有 Chromium 构建的平台）。

如果都找不到，浏览器工具会给出明确提示，而不是静默失败。如果您安装时使用了 `--skip-browser` / `-SkipBrowser`，或下载失败，可以这样安装：

```bash
hermes pm install chromium
```

这也会撤销之前的 `--skip-browser` 选择，使后续更新继续保持浏览器为最新。在 Linux 上，Chromium 可能还需要发行版的系统库（在 Debian/Ubuntu 上可用 `npx playwright install-deps chromium` 安装）。在 Android/Termux 上，如果您的软件源提供，请运行 `pkg install chromium`；没有系统 Chromium 时，该设备上无法使用浏览器。

**本地选项**（`~/.hermes/config.yaml`）：

```yaml
browser:
  executable_path: ""     # Chromium binary to launch instead of the packaged one
  user_data_dir: ""       # persistent profile directory; "" = fresh profile per session
  chromium_args: []       # extra Chromium command-line flags, e.g. ["--lang=de-DE"]
  headed: false           # true = visible window
  command_timeout: 30     # Chromium launch timeout floor (seconds)
  inactivity_timeout: 120 # close an idle browser after N seconds
```

`browser.user_data_dir` 可在会话之间保留 Cookie 和登录状态：此后所有会话共享这一个 profile 目录。留空则为隔离的一次性浏览。

### 有头模式（可见浏览器窗口） {#headed-mode-visible-browser-window}

```yaml
browser:
  headed: true  # default: false
```

有头模式做两件事：

1. **以可见窗口启动 Chromium**，您可以观看 Agent 操作，并在需要时手动介入（登录验证、CAPTCHA）。
2. **在轮次之间保持浏览器打开。** 通常每次 Agent 回复后浏览器都会被清理；有头模式跳过每轮清理，使登录状态在整个对话中保持。

空闲浏览器仍会在 `browser.inactivity_timeout` 后被回收，所有浏览器在关闭时都会退出。在无显示器的主机（服务器、CI）上，浏览器始终以无头模式运行。有头模式只影响本地浏览器——云端会话不受影响。

## 云端浏览器

:::tip Nous 订阅用户
如果您拥有付费 [Nous Portal](https://portal.nousresearch.com) 订阅，可通过 **[Tool Gateway](tool-gateway.md)** 使用云端浏览器，无需单独的 API 密钥。新安装可运行 `hermes setup --portal` 登录并一次性开启所有 gateway 工具；已有安装可通过 `hermes model` 或 `hermes tools` 选择 **Nous Subscription** 作为浏览器提供商。
:::

在 `hermes tools` → Browser Automation 中选择提供商。之后 `browser_exec` 驱动该提供商的浏览器，方式与驱动本地 Chromium 完全相同。

### Browser Use 云端

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

在 [browser-use.com](https://browser-use.com) 获取 API 密钥。Browser Use Cloud 运行托管 Chromium，默认启用[隐身](https://docs.browser-use.com/cloud/browser/stealth)和[住宅代理](https://docs.browser-use.com/cloud/browser/proxies)，包含 CAPTCHA 解决，并支持用于 Cookie、localStorage 和已保存密码的[持久 profile](https://docs.browser-use.com/cloud/guides/authentication)。

### Browserbase

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

在 [browserbase.com](https://browserbase.com) 获取凭据。可选配置：

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in seconds (max 21600 = 6 hours) (default: project default)
BROWSERBASE_SESSION_TIMEOUT=1800
```

Browserbase 隐身功能：

| 功能 | 默认状态 | 说明 |
|---------|---------|-------|
| 基础隐身 | 始终开启 | 随机指纹、视口随机化、CAPTCHA 解决 |
| 住宅代理 | 开启 | 通过住宅 IP 路由以提高访问成功率 |
| 高级隐身 | 关闭 | 自定义 Chromium 构建，需要 Scale 计划 |
| Keep Alive | 开启 | 网络中断后的会话重连 |

若付费功能在您的计划中不可用，Hermes 会自动降级——先禁用 `keepAlive`，再禁用代理——确保免费计划也能正常浏览。

### Firecrawl

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***

# Optional: self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Optional: session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

在 [firecrawl.dev](https://firecrawl.dev) 获取 API 密钥，然后在 `hermes tools` → Browser Automation 中选择 **Firecrawl**。

:::note 选择提供商
上述 `.env` 键只提供**凭据**。当前使用的浏览器由 `hermes tools` → Browser Automation 写入的 `browser.cloud_provider` 决定（`local`、`browserbase`、`browser-use`、`firecrawl`，或代表 Nous 订阅的 `nous`；提供商插件会添加自己的名称）。一旦存在选择，增删密钥不会切换提供商——已选择但缺少密钥的提供商会报错并提示运行 `hermes tools`，而不是静默改道。从未配置过的环境仍会根据可用凭据自动检测。
:::

云端会话消耗提供商额度。Hermes 会在对话结束、`browser.inactivity_timeout` 到期以及进程退出时释放它们。

### 混合路由：公网 URL 使用云端，LAN/localhost 使用本地

配置云端提供商后，Hermes 会为解析到私有/回环/LAN 地址的 URL（`localhost`、`127.0.0.1`、`192.168.x.x`、`10.x.x.x`、`172.16-31.x.x`、`*.local`、`*.lan`、`*.internal`、IPv6 回环 `::1`、链路本地 `169.254.x.x`）自动启动一个**本地 Chromium 辅助进程**。公网 URL 在同一对话中继续使用云端提供商。

这解决了常见的"本地开发但使用 Browserbase"场景——Agent 可以截取 `http://localhost:3000` 上的仪表盘，同时抓取 `https://github.com`，无需切换提供商或禁用 SSRF 防护。云端提供商永远不会看到私有 URL。

该功能**默认开启**。如需禁用（所有 URL 均走已配置的云端提供商）：

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

禁用自动路由后，私有 URL 将被拒绝并返回 `"Blocked: URL targets a private or internal address"`，除非同时设置 `browser.allow_private_urls: true`（允许云端提供商尝试访问，但通常无法成功，因为 Browserbase 等无法访问您的 LAN）。

辅助进程就是[上文](#local-chromium-default)所述的本地 Chromium，因此需要安装该 Chromium。

## 真实 profile 浏览（使用您自己的登录状态） {#real-profile-browsing-use-your-own-logins}

默认情况下，本地浏览在干净的一次性 profile 中运行——Agent 未登录任何网站。开启**真实 profile 浏览**后，Agent 可以以*您*的身份浏览，使用您现有的登录状态和 Cookie：

```yaml
# ~/.hermes/config.yaml
browser:
  use_real_profile: true
```

开启后，Hermes 会把您默认浏览器的**当前活跃** profile（`Local State → profile.last_used`，包括 Cookie、已保存登录和偏好设置）复制到 `~/.hermes/browser-profile/<browser>/` 下的托管快照中，然后用您**真实的浏览器程序**在该快照上启动并连接。您正在使用的 profile **从不被直接打开**：快照是独立目录，因此不会与运行中的浏览器争抢 profile 锁，也绕开了 Chrome 136+ 对默认 profile 目录远程调试的封锁。每次启动新会话时，认证文件都会从真实 profile 重新同步。只复制活跃 profile。

快照浏览器以**无头**方式在后台运行，不抢占焦点；设置 `browser.headed: true` 可显示窗口。在无显示器的主机上始终为无头模式。如需固定快照来源，可设置 `browser.real_profile_pin: "Profile 2"`（浏览器 user-data 目录下的 profile 目录名）；指向不存在目录的固定值不会回退到其他 profile。关闭开关后，Hermes 会在下一次使用浏览器时删除快照存储（`~/.hermes/browser-profile/`）。

如果无法复制您的 profile（例如 Windows 上运行中的浏览器以独占锁持有 Cookie 数据库、默认浏览器不是受支持的 Chromium，或 Cookie 数据库损坏），Agent 会获得一个**不带登录状态**的全新浏览器，且第一个浏览器结果会说明原因（`real_profile_unavailable`）。设置 `browser.real_profile_autoclose: true` 后，Hermes 会在需要时**询问**是否替您关闭浏览器，获得批准后才运行 `hermes browser close-profile`。

- **支持的浏览器：** Chrome、Edge、Brave、Brave Origin、Chromium（以系统默认为准）。
- **适用于任何浏览器来源。** 本地浏览时开启即生效。使用**云端**提供商时，Agent 仍可通过 `browser_exec` 工具的 `local` 参数按需打开真实 profile 的本地会话（仅在开启此开关时提供该参数）。
- **安全说明：** 这是基于授权的便利功能，而非隔离边界。默认关闭。
- **桌面应用：** 在 **Capabilities → Tools → Browser → Use My Real Browser Profile** 中切换。

### 定时与无人值守运行 {#scheduled-and-unattended-runs}

真实 profile 会话可以让 `cronjob` 操作您已登录的网站。开启开关，为任务提供 browser 工具集（`cronjob(action="create", enabled_toolsets=["browser", ...], ...)`），并预先在[凭据保险库](./credential-vault.md#headless-sessions)中保存无法仅靠同步 Cookie 使用的网站凭据（包括验证器密钥）——无人值守的会话无法回答任何提示。

## Lightpanda 本地引擎

[Lightpanda](https://lightpanda.io) 是一个从零编写的开源无头浏览器。它瞬间启动，速度比 Chrome 快 9 倍，内存占用低 16 倍，这对长时间运行在小型 VM 上的 Agent 很重要。

Lightpanda 是**本地引擎**（与本地 Chromium 一样是浏览器来源），而非云端提供商。安装该程序并将其放到 `PATH` 上（参见 [Lightpanda 安装指南](https://lightpanda.io/docs/run-locally/installation/one-liner)），然后在 `hermes tools` → Browser Automation 中选择 **Lightpanda**，或设置：

```yaml
# Add to ~/.hermes/config.yaml
browser:
  cloud_provider: local
  engine: lightpanda   # auto (default, Chromium) | lightpanda
```

Hermes 会自己启动 `lightpanda serve --host 127.0.0.1 --port <free>`——每个 `browser_exec` 会话名（或每个任务）一个进程——并让 `browser_exec` 指向它。无需 Chromium 或 Node.js。进程会在 `browser.inactivity_timeout` 后、退出时以及 Hermes 崩溃后的孤儿清理中被回收。所有这些进程共享位于 `$HERMES_HOME/cache/browser-use/lightpanda/http-cache` 的磁盘 HTTP 缓存。

Lightpanda 没有图形渲染器，因此 `capture_screenshot()` 不可用，工具描述会让模型以文本为主；它每个会话只保持一个页面，因此模型会被告知只调用一次 `new_tab()`，之后使用 `goto_url()`（上游跟踪：[lightpanda-io/browser#1962](https://github.com/lightpanda-io/browser/issues/1962)）。

**引擎何时被忽略。** `browser.engine` 是优先级最低的浏览器设置：云端提供商（包括 Nous 订阅浏览器）、`browser.cdp_url` / `/browser connect` 覆盖，或 `browser.use_real_profile` 都优先于它。`/browser status` 和 `hermes doctor` 会报告引擎已配置但被覆盖的情况，以及被什么覆盖。

## 通过 CDP 连接你自己的浏览器（`/browser connect`）

除了 Hermes 自己启动的 Chromium 或云端提供商，您还可以通过 Chrome DevTools Protocol（CDP）将浏览器工具连接到本地运行的 Chrome、Brave、Chromium 或 Edge 实例。当您希望实时查看 Agent 操作、与需要自身 Cookie/会话的页面交互，或避免云端浏览器费用时，此方式非常有用。

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

连接后，`browser_exec`、`browser_cdp`、`browser_dialog` 和 `browser_vault` 都会驱动您的实时浏览器，而不再自行启动浏览器。也可以在 `config.yaml` 中设置 `browser.cdp_url`，或使用 `BROWSER_CDP_URL` 环境变量，持久指定该端点。

### WSL2 + Windows Chrome：优先使用 MCP 而非 `/browser connect`

若 Hermes 在 WSL2 内运行，但您想控制的 Chrome 窗口在 Windows 宿主机上，`/browser connect` 通常不是最佳方案。

原因：

- `/browser connect` 要求 Hermes 本身能访问可用的 CDP 端点
- 现代 Chrome 实时调试会话通常暴露仅宿主机本地可访问的端点，WSL 无法像访问经典 `9222` 端口那样直接访问
- 即使 Windows Chrome 可调试，最简洁的集成方式通常是让 Windows 侧的浏览器 MCP 服务器连接 Chrome，再让 Hermes 与该 MCP 服务器通信

对于此场景，建议通过 Hermes MCP 支持使用 `chrome-devtools-mcp`。

具体配置请参阅 MCP 指南：

- [在 Hermes 中使用 MCP](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

## Bot Desktop 屏幕

当 gateway 运行 [Bot Desktop 屏幕](bot-screen.md)时，机器人的浏览器与 Dock 上的 **Browser** 图标是同一个有头 Chromium，每个机器人一个持久 profile，因此您可以接管、登录某个网站，再把会话交还给 Agent。profile 位置以及 `browser.executable_path` / `browser.user_data_dir` 在那里的作用，请参阅该页面。

## 配套工具

`browser_vault` 作用于 `browser_exec` 正在驱动的同一个浏览器会话——本地 Chromium、云端浏览器，或通过 `/browser connect` 连接的浏览器。`browser_cdp` 和 `browser_dialog` 仅在配置了 CDP 端点（`/browser connect`、`BROWSER_CDP_URL` 或 `browser.cdp_url`）时提供，并作用于该浏览器；Hermes 启动的本地会话或云端会话不会带来这两个工具。

### `browser_cdp`

原始 Chrome DevTools Protocol 直通——用于一次性操作的逃生舱口：原生对话框处理、iframe 范围内执行、Cookie/网络控制，或 Agent 需要的任何 CDP 命令。（在 `browser_exec` 中，`cdp()` 辅助函数可完成相同的多步工作。）

**CDP 方法参考：** https://chromedevtools.github.io/devtools-protocol/ ——Agent 可通过 `web_extract` 访问特定方法页面以查阅参数和返回结构。

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

浏览器级方法（`Target.*`、`Browser.*`、`Storage.*`）省略 `target_id`。页面级方法（`Page.*`、`Runtime.*`、`DOM.*`、`Emulation.*`）需要来自 `Target.getTargets` 的 `target_id`。每次无状态调用相互独立——调用间不保留会话状态。

**跨域 iframe：** 传入进程外（跨域）iframe 的 `frame_id`，可通过 CDP 监督器为该 frame 维护的实时会话路由调用。这是在 Browserbase 上对跨域 iframe 执行 `Runtime.evaluate` 的方式，可避免无状态 CDP 连接遭遇签名 URL 过期。同域 iframe 无需 `frame_id`——在顶层 `Runtime.evaluate` 中使用 `document.querySelector('iframe').contentDocument` 即可。

### `browser_dialog` {#browser_dialog}

响应原生 JS 对话框（`alert` / `confirm` / `prompt` / `beforeunload`）。打开的对话框会阻塞页面的 JavaScript 线程，因此在得到响应之前，该页面上的浏览器操作都会停滞。

持久的 **CDP 监督器**——每个任务一个 WebSocket，订阅 Page/Runtime/Target 事件——会在对话框出现时检测到它。Agent 用 `browser_dialog(action="accept")` 或 `browser_dialog(action="dismiss")` 响应；对于 `prompt()` 对话框传入 `prompt_text="..."`，有多个对话框排队时传入 `dialog_id` 指定其中一个。随后页面 JS 线程恢复运行。

**在 Browserbase 上的工作原理。** Browserbase 的 CDP 代理会在约 10ms 内在服务端自动关闭真实的原生对话框，因此无法使用 `Page.handleJavaScriptDialog`。监督器通过 `Page.addScriptToEvaluateOnNewDocument` 注入一段小脚本，将 `window.alert`/`confirm`/`prompt` 替换为同步 XHR，并通过 `Fetch.enable` 拦截这些 XHR——页面 JS 线程在 XHR 上保持阻塞，直到监督器用 Agent 的响应调用 `Fetch.fulfillRequest`。`prompt()` 的返回值原样传回页面 JS。

**对话框策略**在 `config.yaml` 的 `browser.dialog_policy` 下配置：

| 策略 | 行为 |
|--------|----------|
| `must_respond`（默认） | 捕获对话框，等待显式 `browser_dialog()` 调用。在 `browser.dialog_timeout_s`（默认 300 秒）后安全自动关闭，防止有问题的 Agent 永久阻塞。 |
| `auto_dismiss` | 捕获，立即关闭。Agent 无需操作。 |
| `auto_accept` | 捕获，立即接受。适用于导航带有频繁 `beforeunload` 提示的页面。 |

### `browser_vault`

在当前页面使用已保存的登录信息、密码、银行卡、地址和一次性验证码，机密信息永远不会进入对话。Agent 调用 `browser_vault(action="list")` 以句柄形式查看已保存条目，然后使用 `fill`、`save_login`、`enter_code` 或 `unlock`（1Password / Bitwarden）。页面来源与已保存条目不匹配时拒绝填充。设置、密码管理器后端以及无头使用方式请参阅[凭据保险库](credential-vault.md)。

## 关闭浏览功能

浏览器没有配置开关——请改为禁用 `browser` 工具集。运行 `hermes tools`，为不应浏览的平台取消勾选 **Browser Automation**，或在脚本中：

```bash
hermes tools disable browser                      # CLI
hermes tools disable browser --platform telegram  # one gateway platform
```

没有 `terminal` 工具集的会话也不会出现 `browser_exec`。

## Camofox

Camofox 已移至插件：`hermes plugins install camofox`。现有的 Camofox 配置会在更新时自动迁移。

## 实际示例

### 填写网页表单

```
User: Sign up for an account on example.com with my email john@example.com

Agent: browser_exec(code="""
# Opening the signup form
new_tab("https://example.com/signup")
wait_for_load()
fill_input("input[name=email]", "john@example.com")
js("document.querySelector('button[type=submit]').click()")
wait_for_load()
print(page_info())
""")
```

对于密码，Agent 使用 `browser_vault`（`save_login` 会在掩码输入框中提示您），而不是自己输入机密信息。

### 研究动态内容

```
User: What are the top trending repos on GitHub right now?

Agent: browser_exec(code="""
# Reading GitHub trending
new_tab("https://github.com/trending")
wait_for_load()
print(js("[...document.querySelectorAll('article h2 a')].map(a => a.textContent.replace(/\\s+/g, ' ').trim())"))
""")
```

## 限制

- **会话超时**——云端会话根据提供商计划设置过期。
- **费用**——云端会话消耗提供商额度；对话结束或非活跃后会话自动清理。本地 Chromium 和 `/browser connect` 免费。
- **Lightpanda**——不支持截图，每个会话只有一个页面。
- **没有终端就没有浏览器**——`browser_exec` 运行 Python，因此没有 `terminal` 工具集的会话无法使用它。
