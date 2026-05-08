---
title: 浏览器自动化
description: "通过多个提供商、本地 Chrome via CDP 或云浏览器控制浏览器，用于网页交互、表单填写、抓取等。"
sidebar_label: 浏览器
sidebar_position: 5
---

# 浏览器自动化

Hermes Agent 包含完整的浏览器自动化工具集，支持多种后端选项：

- **Browserbase 云模式**——通过 [Browserbase](https://browserbase.com) 提供托管云浏览器和反机器人工具
- **Browser Use 云模式**——通过 [Browser Use](https://browser-use.com) 作为替代云浏览器提供商
- **Firecrawl 云模式**——通过 [Firecrawl](https://firecrawl.dev) 提供内置抓取的云浏览器
- **Camofox 本地模式**——通过 [Camofox](https://github.com/jo-inc/camofox-browser) 进行本地反检测浏览（基于 Firefox 的指纹欺骗）
- **通过 CDP 连接本地 Chrome**——使用 `/browser connect` 将浏览器工具附加到你自己的 Chrome 实例
- **本地浏览器模式**——通过 `agent-browser` CLI 和本地 Chromium 安装

在所有模式下，agent 都可以导航网站、与页面元素交互、填写表单和提取信息。

## 概述

页面表示为**无障碍树**（基于文本的快照），非常适合 LLM agent。交互元素获取 ref ID（如 `@e1`、`@e2`），agent 使用它们进行点击和输入。

主要功能：

- **多提供商云执行**——Browserbase、Browser Use 或 Firecrawl——无需本地浏览器
- **本地 Chrome 集成**——通过 CDP 附加到你正在运行的 Chrome 以进行手动浏览
- **内置隐身**——随机指纹、CAPTCHA 解决、住宅代理（Browserbase）
- **会话隔离**——每个任务获得自己的浏览器会话
- **自动清理**——非活动会话在超时后关闭
- **视觉分析**——截图 + AI 分析用于视觉理解

## 设置

:::tip Nous 订阅者
如果你有付费的 [Nous Portal](https://portal.nousresearch.com) 订阅，你可以通过 **[Tool Gateway](tool-gateway.md)** 使用浏览器自动化，无需单独的 API 密钥。运行 `hermes model` 或 `hermes tools` 来启用它。
:::

### Browserbase 云模式

要使用 Browserbase 托管的云浏览器，请添加：

```bash
# 添加到 ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

在 [browserbase.com](https://browserbase.com) 获取你的凭据。

### Browser Use 云模式

要使用 Browser Use 作为你的云浏览器提供商，请添加：

```bash
# 添加到 ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

在 [browser-use.com](https://browser-use.com) 获取你的 API 密钥。Browser Use 通过其 REST API 提供云浏览器。如果同时设置了 Browserbase 和 Browser Use 凭据，Browserbase 优先。

### Firecrawl 云模式

要使用 Firecrawl 作为你的云浏览器提供商，请添加：

```bash
# 添加到 ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***
```

在 [firecrawl.dev](https://firecrawl.dev) 获取你的 API 密钥。然后选择 Firecrawl 作为你的浏览器提供商：

```bash
hermes setup tools
# → Browser Automation → Firecrawl
```

可选设置：

```bash
# 自托管 Firecrawl 实例（默认：https://api.firecrawl.dev）
FIRECRAWL_API_URL=http://localhost:3002

# 会话 TTL 秒数（默认：300）
FIRECRAWL_BROWSER_TTL=600
```

### 混合路由：公共 URL 用云，本地/LAN 用本地

当配置了云提供商时，Hermes 自动为解析为私有/回环/LAN 地址的 URL 生成**本地 Chromium 辅助进程**
（`localhost`、`127.0.0.1`、
`192.168.x.x`、`10.x.x.x`、`172.16-31.x.x`、`*.local`、`*.lan`、`*.internal`、
IPv6 回环 `::1`、链路本地 `169.254.x.x`）。公共 URL 在同一对话中继续使用
云提供商。

这解决了常见的"我在本地开发但使用 Browserbase"工作流——
agent 可以截取 `http://localhost:3000` 上的仪表板截图 AND 抓取
`https://github.com`，而无需你切换提供商或禁用 SSRF 防护。
云提供商永远不会看到私有 URL。

该功能**默认开启**。要禁用它（所有 URL 都使用配置的
云提供商，与以前一样）：

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

禁用自动路由后，私有 URL 被拒绝并显示
`"Blocked: URL targets a private or internal address"`，除非你还设置了
`browser.allow_private_urls: true`（这让云提供商尝试它们——
通常不会起作用，因为 Browserbase 等无法访问你的 LAN）。

要求：本地辅助进程使用与纯本地模式相同的 `agent-browser` CLI，因此你需要安装它（`hermes setup tools → Browser Automation`
自动安装）。从公共 URL 到私有地址的导航后重定向仍然被阻止（你不能使用重定向到内部的技术来通过公共路径访问你的 LAN）。

### Camofox 本地模式

[Camofox](https://github.com/jo-inc/camofox-browser) 是一个自托管的 Node.js 服务器，包装了 Camoufox（带有 C++ 指纹欺骗的 Firefox 分支）。它提供本地反检测浏览，不依赖云。

```bash
# 首先克隆 Camofox 浏览器服务器
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser

# 使用默认容器设置构建并启动
#（自动检测架构：M1/M2 上为 aarch64，Intel 上为 x86_64）
make up

# 停止并移除默认容器
make down

# 强制干净重建（例如，升级 VERSION/RELEASE 后）
make reset

# 仅下载二进制文件而不构建
make fetch

# 显式覆盖架构或版本
make up ARCH=x86_64
make up VERSION=135.0.1 RELEASE=beta.24
```

`make up` 立即启动默认容器。如果你想使用自定义运行时设置（如更大的 Node 堆、VNC 或持久配置文件目录），请先构建镜像然后自己运行：

```bash
# 构建镜像而不启动默认容器
make build

# 启动并设置持久化、VNC 实时查看和更大的 Node 堆
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

启用 VNC 后，浏览器以有头模式运行，可以在 `http://localhost:6080` 的浏览器中实时观看（noVNC）。你也可以将原生 VNC 客户端连接到 `localhost:5901`。

如果你已经运行了 `make up`，在启动自定义容器之前停止并移除该默认容器：

```bash
make down
# 然后运行上面的自定义 docker run 命令
```

然后在 `~/.hermes/.env` 中设置：

```bash
CAMOFOX_URL=http://localhost:9377
```

或者通过 `hermes tools` → Browser Automation → Camofox 配置。

当设置了 `CAMOFOX_URL` 时，所有浏览器工具会自动通过 Camofox 路由，而不是 Browserbase 或 agent-browser。

#### 持久浏览器会话

默认情况下，每个 Camofox 会话都会获得随机身份——cookies 和登录信息不会在 agent 重启后保留。要启用持久浏览器会话，请在 `~/.hermes/config.yaml` 中添加以下内容：

```yaml
browser:
  camofox:
    managed_persistence: true
```

然后完全重启 Hermes 以使新配置生效。

:::warning 嵌套路径很重要
Hermes 读取 `browser.camofox.managed_persistence`，**不是**顶级 `managed_persistence`。一个常见错误是写：

```yaml
# ❌ 错误——Hermes 会忽略这个
managed_persistence: true
```

如果标志放在错误的路径上，Hermes 会静默回退到随机临时 `userId`，你的登录状态将在每次会话时丢失。
:::

##### Hermes 做什么
- 向 Camofox 发送确定性配置文件作用域的 `userId`，以便服务器可以在会话之间重用相同的 Firefox 配置文件。
- 在清理时跳过服务器端上下文销毁，因此 cookies 和登录信息在 agent 任务之间保留。
- 将 `userId` 作用域限定为活动的 Hermes 配置文件，因此不同的 Hermes 配置文件获得不同的浏览器配置文件（配置文件隔离）。

##### Hermes 不做什么
- 它不会在 Camofox 服务器上强制持久化。Hermes 只发送一个稳定的 `userId`；服务器必须通过将该 `userId` 映射到持久 Firefox 配置文件目录来遵守它。
- 如果你的 Camofox 服务器版本将每个请求视为临时的（例如，始终调用 `browser.newContext()` 而不加载存储的配置文件），Hermes 无法使那些会话持久化。确保你运行的是实现基于 userId 的配置文件持久化的 Camofox 构建。

##### 验证它是否正常工作

1. 启动 Hermes 和你的 Camofox 服务器。
2. 在浏览器任务中打开 Google（或其他登录站点）并手动登录。
3. 正常结束浏览器任务。
4. 启动新的浏览器任务。
5. 再次打开同一个站点——你应该仍然登录。

如果步骤 5 让你退出，Camofox 服务器没有遵守稳定的 `userId`。仔细检查你的配置路径，确认你在编辑 `config.yaml` 后完全重启了 Hermes，并验证你的 Camofox 服务器版本支持每个用户的持久化配置文件。

##### 状态存储在哪里

Hermes 从配置文件作用域目录 `~/.hermes/browser_auth/camofox/` 派生稳定的 `userId`（或非默认配置文件下的等价目录）。实际的浏览器配置文件数据存储在 Camofox 服务器端，由该 `userId` 键控。要完全重置持久配置文件，请在 Camofox 服务器上清除它，并移除相应 Hermes 配置文件的状态目录。

#### VNC 实时查看

当 Camofox 以有头模式运行时（带有可见浏览器窗口），它会在健康检查响应中暴露一个 VNC 端口。Hermes 自动发现这一点，并在导航响应中包含 VNC URL，以便 agent 可以分享链接供你实时观看浏览器。

### 通过 CDP 连接本地 Chrome（`/browser connect`）

你可以通过 Chrome DevTools 协议（CDP）将 Hermes 浏览器工具附加到你自己的正在运行的 Chrome 实例，而不是使用云提供商。当你想实时查看 agent 正在做什么、与需要你自己的 cookies/sessions 的页面交互或避免云浏览器成本时，这很有用。

:::note
`/browser connect` 是一个**交互式 CLI 斜杠命令**——它不由 gateway 分发。如果你尝试在 WebUI、Telegram、Discord 或其他 gateway 聊天中运行它，消息将作为纯文本发送给 agent，命令不会执行。从终端启动 Hermes（`hermes` 或 `hermes chat`）并在那里发出 `/browser connect`。
:::

在 CLI 中使用：

```
/browser connect              # 连接到 ws://localhost:9222 的 Chrome
/browser connect ws://host:port  # 连接到特定的 CDP 端点
/browser status               # 检查当前连接
/browser disconnect            # 分离并返回云/本地模式
```

如果 Chrome 尚未使用远程调试运行，Hermes 会尝试使用 `--remote-debugging-port=9222` 自动启动它。

:::tip
要手动启动启用 CDP 的 Chrome，请使用专用 user-data-dir，以便调试端口在 Chrome 已经使用你的正常配置文件运行时也能正常启动：

```bash
# Linux
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# macOS
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &
```

然后启动 Hermes CLI 并运行 `/browser connect`。

**为什么使用 `--user-data-dir`？** 如果在已经运行普通 Chrome 实例时启动 Chrome，通常会在现有进程上打开一个新窗口——而那个现有进程没有以 `--remote-debugging-port` 启动，所以端口 9222 永远不会打开。专用 user-data-dir 强制创建新的 Chrome 进程，调试端口会在那里监听。`--no-first-run --no-default-browser-check` 跳过新配置文件的首次启动向导。
:::

通过 CDP 连接时，所有浏览器工具（`browser_navigate`、`browser_click` 等）都在你的实时 Chrome 实例上操作，而不是启动云会话。

### WSL2 + Windows Chrome：优先使用 MCP 而不是 `/browser connect`

如果 Hermes 在 WSL2 内运行，但你想要控制的 Chrome 窗口在 Windows 主机上运行，`/browser connect` 通常不是最佳路径。

原因：

- `/browser connect` 期望 Hermes 本身能到达可用的 CDP 端点
- 现代 Chrome 实时调试会话通常暴露主机本地的端点，无法以经典 `9222` 端口的方式从 WSL 直接访问
- 即使 Windows Chrome 可调试，最干净的集成通常是让 Windows 端的浏览器 MCP 服务器附加到 Chrome，让 Hermes 与该 MCP 服务器通信

对于这种设置，优先通过 Hermes MCP 支持使用 `chrome-devtools-mcp`。

有关实际设置，请参阅 MCP 指南：

- [将 MCP 与 Hermes 结合使用](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

### 本地浏览器模式

如果你**没有**设置任何云凭据且不使用 `/browser connect`，Hermes 仍然可以通过 `agent-browser` 驱动的本地 Chromium 安装使用浏览器工具。

### 可选环境变量

```bash
# 用于更好 CAPTCHA 解决的住宅代理（默认："true"）
BROWSERBASE_PROXIES=true

# 使用自定义 Chromium 的高级隐身——需要 Scale 计划（默认："false"）
BROWSERBASE_ADVANCED_STEALTH=false

# 断开连接后重新连接会话——需要付费计划（默认："true"）
BROWSERBASE_KEEP_ALIVE=true

# 自定义会话超时毫秒数（默认：项目默认值）
# 示例：600000（10分钟），1800000（30分钟）
BROWSERBASE_SESSION_TIMEOUT=600000

# 自动清理前的非活动超时秒数（默认：120）
BROWSER_INACTIVITY_TIMEOUT=120
```

### 安装 agent-browser CLI

```bash
npm install -g agent-browser
# 或者在仓库中本地安装：
npm install
```

:::info
`browser` 工具集必须包含在配置的 `toolsets` 列表中，或者通过 `hermes config set toolsets '["hermes-cli", "browser"]'` 启用。
:::

## 可用工具

### `browser_navigate`

导航到 URL。必须在任何其他浏览器工具之前调用。初始化 Browserbase 会话。

```
Navigate to https://github.com/NousResearch
```

:::tip
对于简单的信息检索，优先使用 `web_search` 或 `web_extract`——它们更快更便宜。当你需要**交互**页面时（点击按钮、填写表单、处理动态内容）使用浏览器工具。
:::

### `browser_snapshot`

获取当前页面无障碍树的基于文本的快照。返回带有 ref ID 的交互元素（如 `@e1`、`@e2`），用于 `browser_click` 和 `browser_type`。

- **`full=false`**（默认）：仅显示交互元素的紧凑视图
- **`full=true`**：完整页面内容

超过 8000 字符的快照会自动由 LLM 总结。

### `browser_click`

点击快照中通过 ref ID 识别的元素。

```
Click @e5 to press the "Sign In" button
```

### `browser_type`

在输入字段中键入文本。首先清除字段，然后输入新文本。

```
Type "hermes agent" into the search field @e3
```

### `browser_scroll`

向上或向下滚动页面以显示更多内容。

```
Scroll down to see more results
```

### `browser_press`

按下键盘键。用于提交表单或导航。

```
Press Enter to submit the form
```

支持的键：`Enter`、`Tab`、`Escape`、`ArrowDown`、`ArrowUp` 等。

### `browser_back`

在浏览器历史记录中导航回上一页。

### `browser_get_images`

列出当前页面上的所有图像及其 URL 和 alt 文本。用于查找要分析的图像。

### `browser_vision`

截取屏幕截图并使用视觉 AI 进行分析。当文本快照不能捕获重要的视觉信息时使用此功能——对于 CAPTCHA、复杂布局或视觉验证挑战特别有用。

屏幕截图被持久保存，文件路径与 AI 分析一起返回。在消息平台（Telegram、Discord、Slack、WhatsApp）上，你可以要求 agent 分享屏幕截图——它将通过 `MEDIA:` 机制作为原生照片附件发送。

```
What does the chart on this page show?
```

屏幕截图存储在 `~/.hermes/cache/screenshots/` 中，24 小时后自动清理。

### `browser_console`

获取当前页面的浏览器控制台输出（log/warn/error 消息）和未捕获的 JavaScript 异常。对于检测不会出现在无障碍树中的静默 JS 错误至关重要。

```
Check the browser console for any JavaScript errors
```

使用 `clear=True` 在读取后清除控制台，以便后续调用只显示新消息。

### `browser_cdp`

原始 Chrome DevTools 协议直通——用于其他工具未涵盖的浏览器操作的逃生通道。用于原生对话框处理、iframe 作用域评估、cookie/网络控制或 agent 需要的任何 CDP 动词。

**仅在会话开始时可访问 CDP 端点时可用**——这意味着 `/browser connect` 已附加到正在运行的 Chrome，或 `browser.cdp_url` 在 `config.yaml` 中设置。默认本地 agent-browser 模式、Camofox 和云提供商（Browserbase、Browser Use、Firecrawl）当前不向此工具暴露 CDP——云提供商有按会话的 CDP URL，但实时会话路由是后续功能。

**CDP 方法参考：** https://chromedevtools.github.io/devtools-protocol/ —— agent 可以 `web_extract` 特定方法页面来查找参数和返回形状。

常见模式：

```
# 列出标签页（浏览器级别，无 target_id）
browser_cdp(method="Target.getTargets")

# 处理标签页上的原生 JS 对话框
browser_cdp(method="Page.handleJavaScriptDialog",
            params={"accept": true, "promptText": ""},
            target_id="<tabId>")

# 在特定标签页中评估 JS
browser_cdp(method="Runtime.evaluate",
            params={"expression": "document.title", "returnByValue": true},
            target_id="<tabId>")

# 获取所有 cookies
browser_cdp(method="Network.getAllCookies")
```

浏览器级别的方法（`Target.*`、`Browser.*`、`Storage.*`）省略 `target_id`。页面级别的方法（`Page.*`、`Runtime.*`、`DOM.*`、`Emulation.*`）需要 `target_id` 来自 `Target.getTargets`。每个无状态调用都是独立的——会话不会在调用之间保留。

**跨源 iframe：** 传递 `frame_id`（来自 `browser_snapshot.frame_tree.children[]`，其中 `is_oopif=true`）以通过该 iframe 的实时会话路由 CDP 调用。这是跨源 iframe 内部 `Runtime.evaluate` 在 Browserbase 上的工作方式，因为在无状态 CDP 连接上会出现签名 URL 过期。示例：

```
browser_cdp(
  method="Runtime.evaluate",
  params={"expression": "document.title", "returnByValue": True},
  frame_id="<frame_id from browser_snapshot>",
)
```

同源 iframe 不需要 `frame_id`——改用顶层 `Runtime.evaluate` 中的 `document.querySelector('iframe').contentDocument`（这会返回 iframe 的 document 对象）来替代。

### `browser_dialog`

响应原生 JS 对话框（`alert` / `confirm` / `prompt` / `beforeunload`）。在这个工具存在之前，对话框会静默阻止页面的 JavaScript 线程，后续的 `browser_*` 调用会挂起或抛出；现在 agent 在 `browser_snapshot` 输出中看到待处理的对话框并明确响应。

**工作流：**
1. 调用 `browser_snapshot`。如果对话框阻止了页面，它显示为 `pending_dialogs: [{"id": "d-1", "type": "alert", "message": "..."}]`。
2. 调用 `browser_dialog(action="accept")` 或 `browser_dialog(action="dismiss")`。对于 `prompt()` 对话框，传递 `prompt_text="..."` 来提供响应。
3. 重新获取快照——`pending_dialogs` 为空；页面的 JS 线程已恢复。

**检测自动发生**通过持久 CDP 监督器——每个任务一个 WebSocket，订阅 Page/Runtime/Target 事件。监督器还在快照中填充一个 `frame_tree` 字段，以便 agent 看到当前页面的 iframe 结构，包括跨源（OOPIF）iframe。

**可用性矩阵：**

| 后端 | 通过 `pending_dialogs` 检测 | 响应（`browser_dialog` 工具） |
|---|---|---|
| 通过 `/browser connect` 或 `browser.cdp_url` 的本地 Chrome | ✓ | ✓ 完整工作流 |
| Browserbase | ✓ | ✓ 完整工作流（通过注入的 XHR 桥接） |
| Camofox / 默认本地 agent-browser | ✗ | ✗（无 CDP 端点） |

**Browserbase 上的工作原理。** Browserbase 的 CDP 代理在大约 10ms 内自动在服务器端关闭真正的原生对话框，所以我们不能使用 `Page.handleJavaScriptDialog`。监督器通过 `Page.addScriptToEvaluateOnNewDocument` 注入一个小脚本，覆盖 `window.alert`/`confirm`/`prompt` 为同步 XHR。我们通过 `Fetch.enable` 拦截这些 XHR——页面的 JS 线程在 XHR 上被阻塞，直到我们调用 `Fetch.fulfillRequest` 与 agent 的响应。`prompt()` 返回值完整往返回页面 JS。

**对话框策略**在 `config.yaml` 下的 `browser.dialog_policy` 中配置：

| 策略 | 行为 |
|--------|----------|
| `must_respond`（默认） | 捕获，在快照中显示，等待显式 `browser_dialog()` 调用。安全自动关闭在 `browser.dialog_timeout_s`（默认 300 秒）后生效，因此有缺陷的 agent 不能永远停滞。 |
| `auto_dismiss` | 捕获，立即关闭。Agent 仍在 `browser_state` 历史中看到对话框，但不需要采取行动。 |
| `auto_accept` | 捕获，立即接受。当导航带有激进 `beforeunload` 提示的页面时很有用。 |

**frame_tree** 在 `browser_snapshot.frame_tree` 中被限制为 30 帧和 OOPIF 深度 2，以在广告密集的页面上保持有效载荷边界。当达到限制时，`truncated: true` 标志会显示；需要完整树的 agent 可以使用 `browser_cdp` 和 `Page.getFrameTree`。

## 实用示例

### 填写 Web 表单

```
User: 在 example.com 上注册一个账户，使用我的邮箱 john@example.com

Agent 工作流：
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → 看到带有 refs 的表单字段
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → 点击"创建账户"
6. browser_snapshot()  → 确认成功
```

### 研究动态内容

```
User: 现在 GitHub 上最热门的仓库是什么？

Agent 工作流：
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → 读取热门仓库列表
3. 返回格式化的结果
```

## 会话录制

自动将浏览器会话录制为 WebM 视频文件：

```yaml
browser:
  record_sessions: true  # 默认：false
```

启用后，录制在第一次 `browser_navigate` 时自动开始，并在会话关闭时保存到 `~/.hermes/browser_recordings/`。在本地和云（Browserbase）模式下都有效。超过 72 小时的录制会自动清理。

## 隐身功能

Browserbase 提供自动隐身功能：

| 功能 | 默认 | 说明 |
|---------|---------|-------|
| 基本隐身 | 始终开启 | 随机指纹、视口随机化、CAPTCHA 解决 |
| 住宅代理 | 开启 | 通过住宅 IP 路由以获得更好的访问 |
| 高级隐身 | 关闭 | 自定义 Chromium 构建，需要 Scale 计划 |
| 保持活跃 | 开启 | 网络抖动后会话重新连接 |

:::note
如果你的计划没有付费功能，Hermes 会自动回退——首先禁用 `keepAlive`，然后是代理——以便免费计划上浏览仍然有效。
:::

## 会话管理

- 每个任务通过 Browserbase 获得隔离的浏览器会话
- 会话在非活动后自动清理（默认：2 分钟）
- 后台线程每 30 秒检查一次陈旧会话
- 进程退出时运行紧急清理以防止孤立会话
- 会话通过 Browserbase API（`REQUEST_RELEASE` 状态）释放

## 限制

- **基于文本的交互**——依赖无障碍树，而非像素坐标
- **快照大小**——大页面可能在 8000 字符处被截断或由 LLM 总结
- **会话超时**——云会话基于提供商的计划设置过期
- **成本**——云会话消耗提供商积分；对话结束时或非活动后会话自动清理。使用 `/browser connect` 进行免费本地浏览。
- **无法下载文件**——不能从浏览器下载文件
