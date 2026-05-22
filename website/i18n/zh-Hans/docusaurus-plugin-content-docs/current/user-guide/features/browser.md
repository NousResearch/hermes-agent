---
title: 浏览器自动化
description: 通过多个提供商控制浏览器，通过 CDP 连接本地 Chrome，或使用云浏览器进行网页交互、表单填写、抓取等。
sidebar_label: 浏览器
sidebar_position: 5
---

# 浏览器自动化

Hermes Agent 包含一个完整的浏览器自动化工具集，具有多个后端选项：

- **Browserbase 云模式** 通过 [Browserbase](https://browserbase.com) 提供托管云浏览器和反机器人工具
- **Browser Use 云模式** 通过 [Browser Use](https://browser-use.com) 作为替代云浏览器提供商
- **Firecrawl 云模式** 通过 [Firecrawl](https://firecrawl.dev) 提供内置抓取的云浏览器
- **Camofox 本地模式** 通过 [Camofox](https://github.com/jo-inc/camofox-browser) 提供本地反检测浏览（基于 Firefox 的指纹欺骗）
- **通过 CDP 连接本地 Chrome** —— 使用 `/browser connect` 将浏览器工具连接到您自己的 Chrome 实例
- **本地浏览器模式** 通过 `agent-browser` CLI 和本地 Chromium 安装

在所有模式下，智能体都可以导航网站、与页面元素交互、填写表单和提取信息。

## 概述

页面以**可访问性树**（基于文本的快照）表示，非常适合 LLM 智能体。交互元素获得引用 ID（如 `@e1`、`@e2`），智能体使用它们进行点击和输入。

核心能力：

- **多云提供商执行** —— Browserbase、Browser Use 或 Firecrawl —— 无需本地浏览器
- **本地 Chrome 集成** —— 通过 CDP 附加到您正在运行的 Chrome，进行实际操作浏览
- **内置隐身** —— 随机指纹、CAPTCHA 解决、住宅代理（Browserbase）
- **会话隔离** —— 每个任务获得自己的浏览器会话
- **自动清理** —— 不活动会话在超时后关闭
- **视觉分析** —— 截图 + AI 分析用于视觉理解

## 设置

:::tip Nous 订阅者
如果您有付费的 [Nous Portal](https://portal.nousresearch.com) 订阅，您可以通过 **[Tool Gateway](tool-gateway.md)** 使用浏览器自动化，无需任何单独的 API 密钥。运行 `hermes model` 或 `hermes tools` 来启用它。
:::

### Browserbase 云模式

要使用 Browserbase 托管的云浏览器，请添加：

```bash
# 添加到 ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

在 [browserbase.com](https://browserbase.com) 获取您的凭证。

### Browser Use 云模式

要使用 Browser Use 作为您的云浏览器提供商，请添加：

```bash
# 添加到 ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

在 [browser-use.com](https://browser-use.com) 获取您的 API 密钥。Browser Use 通过其 REST API 提供云浏览器。如果同时设置了 Browserbase 和 Browser Use 凭证，Browserbase 优先。

### Firecrawl 云模式

要使用 Firecrawl 作为您的云浏览器提供商，请添加：

```bash
# 添加到 ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***
```

在 [firecrawl.dev](https://firecrawl.dev) 获取您的 API 密钥。然后选择 Firecrawl 作为您的浏览器提供商：

```bash
hermes setup tools
# → 浏览器自动化 → Firecrawl
```

可选设置：

```bash
# 自托管 Firecrawl 实例（默认：https://api.firecrawl.dev）
FIRECRAWL_API_URL=http://localhost:3002

# 会话 TTL（秒）（默认：300）
FIRECRAWL_BROWSER_TTL=600
```

### 混合路由：公共 URL 用云，LAN/localhost 用本地

当配置了云提供商时，Hermes 会自动为解析到私有/回环/LAN 地址的 URL 生成一个**本地 Chromium 副进程**（`localhost`、`127.0.0.1`、`192.168.x.x`、`10.x.x.x`、`172.16-31.x.x`、`*.local`、`*.lan`、`*.internal`、IPv6 回环 `::1`、链路本地 `169.254.x.x`）。公共 URL 在同一会话中继续使用云提供商。

这解决了常见的"我在本地开发但使用 Browserbase"工作流 —— 智能体可以截图您的 `http://localhost:3000` 仪表板**并**抓取 `https://github.com`，无需您切换提供商或禁用 SSRF 防护。云提供商永远不会看到私有 URL。

该功能**默认开启**。要禁用它（所有 URL 都转到配置的云提供商，与以前一样）：

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

禁用自动路由后，私有 URL 会被拒绝并显示 `"Blocked: URL targets a private or internal address"`，除非您同时设置 `browser.allow_private_urls: true`（这会让云提供商尝试访问它们 —— 通常无法工作，因为 Browserbase 等无法访问您的 LAN）。

要求：本地副进程使用与纯本地模式相同的 `agent-browser` CLI，因此您需要安装它（`hermes setup tools → 浏览器自动化` 会自动安装）。从公共 URL 到私有地址的导航后重定向仍被阻止（您不能使用重定向到内部的技巧通过公共路径访问您的 LAN）。

### Camofox 本地模式

[Camofox](https://github.com/jo-inc/camofox-browser) 是一个自托管的 Node.js 服务器，包装了 Camoufox（一个带有 C++ 指纹欺骗的 Firefox 分支）。它提供本地反检测浏览，无需云依赖。

```bash
# 首先克隆 Camofox 浏览器服务器
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser

# 使用默认容器设置通过 Docker 构建和启动
#（自动检测架构：M1/M2 上的 aarch64，Intel 上的 x86_64）
make up

# 停止并移除默认容器
make down

# 强制干净重建（例如，在升级 VERSION/RELEASE 后）
make reset

# 只下载二进制文件而不构建
make fetch

# 显式覆盖架构或版本
make up ARCH=x86_64
make up VERSION=135.0.1 RELEASE=beta.24
```

`make up` 立即启动默认容器。如果您想要自定义运行时设置，例如更大的 Node 堆、VNC 或持久化配置文件目录，请先构建镜像，然后自己运行：

```bash
# 构建镜像而不启动默认容器
make build

# 使用持久化、VNC 实时视图和更大的 Node 堆启动
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

启用 VNC 后，浏览器以 headed 模式运行，可以在浏览器中通过 `http://localhost:6080`（noVNC）实时观看。您也可以将原生 VNC 客户端连接到 `localhost:5901`。

如果您已经运行了 `make up`，在启动自定义容器之前先停止并移除该默认容器：

```bash
make down
# 然后运行上面的自定义 docker run 命令
```

然后在 `~/.hermes/.env` 中设置：

```bash
CAMOFOX_URL=http://localhost:9377
```

或通过 `hermes tools` → 浏览器自动化 → Camofox 配置。

当设置了 `CAMOFOX_URL` 时，所有浏览器工具都会自动通过 Camofox 路由，而不是 Browserbase 或 agent-browser。

#### 持久化浏览器会话

默认情况下，每个 Camofox 会话获得一个随机身份 —— Cookie 和登录不会在智能体重启后保留。要启用持久化浏览器会话，请将以下内容添加到 `~/.hermes/config.yaml`：

```yaml
browser:
  camofox:
    managed_persistence: true
```

然后完全重启 Hermes 以便新配置被加载。

:::warning 嵌套路径很重要
Hermes 读取 `browser.camofox.managed_persistence`，**而不是**顶级的 `managed_persistence`。常见错误是写成：

```yaml
# ❌ 错误 —— Hermes 会忽略这个
managed_persistence: true
```

如果标志放在了错误的路径，Hermes 会静默回退到随机的临时 `userId`，您的登录状态会在每次会话时丢失。
:::

##### Hermes 做什么
- 向 Camofox 发送一个确定性的、基于配置文件的 `userId`，以便服务器可以在会话间重用相同的 Firefox 配置文件。
- 跳过清理时的服务端上下文销毁，因此 Cookie 和登录在智能体任务之间保留。
- 将 `userId` 限定到活动的 Hermes 配置文件，因此不同的 Hermes 配置文件获得不同的浏览器配置文件（配置文件隔离）。

##### Hermes 不做什么
- 它不会强制 Camofox 服务器持久化。Hermes 只发送一个稳定的 `userId`；服务器必须通过将该 `userId` 映射到持久的 Firefox 配置文件目录来遵守它。
- 如果您的 Camofox 服务器构建将每个请求视为临时的（例如，总是调用 `browser.newContext()` 而不加载存储的配置文件），Hermes 无法使这些会话持久化。确保您运行的 Camofox 构建实现了基于 userId 的配置文件持久化。

##### 验证它是否工作

1. 启动 Hermes 和您的 Camofox 服务器。
2. 在浏览器任务中打开 Google（或任何登录站点）并手动登录。
3. 正常结束浏览器任务。
4. 启动一个新的浏览器任务。
5. 再次打开同一站点 —— 您应该仍然处于登录状态。

如果第 5 步让您登出，Camofox 服务器没有遵守稳定的 `userId`。仔细检查您的配置路径，确认在编辑 `config.yaml` 后完全重启了 Hermes，并验证您的 Camofox 服务器版本是否支持持久的每用户配置文件。

##### 状态存储位置

Hermes 从基于配置文件的目录 `~/.hermes/browser_auth/camofox/`（或非默认配置文件的 `$HERMES_HOME` 等效目录）派生稳定的 `userId`。实际的浏览器配置文件数据存在于 Camofox 服务器端，以该 `userId` 为键。要完全重置持久化配置文件，请在 Camofox 服务器上清除它并移除相应 Hermes 配置文件的状态目录。

#### VNC 实时视图

当 Camofox 以 headed 模式运行（带有可见浏览器窗口）时，它会在健康检查响应中暴露一个 VNC 端口。Hermes 自动发现这一点，并在导航响应中包含 VNC URL，因此智能体可以分享一个链接供您实时观看浏览器。

### 通过 CDP 连接本地 Chrome（`/browser connect`）

除了云提供商，您还可以通过 Chrome DevTools Protocol (CDP) 将 Hermes 浏览器工具附加到您自己正在运行的 Chrome 实例。当您想要实时查看智能体在做什么、与需要您自己 Cookie/会话的页面交互，或避免云浏览器成本时，这很有用。

:::note
`/browser connect` 是一个**交互式 CLI 斜杠命令** —— 它不由网关调度。如果您尝试在 WebUI、Telegram、Discord 或其他网关聊天中运行它，消息会作为纯文本发送给智能体，命令不会执行。从终端启动 Hermes（`hermes` 或 `hermes chat`）并在那里发出 `/browser connect`。
:::

在 CLI 中，使用：

```
/browser connect              # 连接到 ws://localhost:9222 的 Chrome
/browser connect ws://host:port  # 连接到特定的 CDP 端点
/browser status               # 检查当前连接
/browser disconnect            # 断开并返回云/本地模式
```

如果 Chrome 尚未以远程调试模式运行，Hermes 会尝试自动启动它并带上 `--remote-debugging-port=9222`。

:::tip
要手动启动启用了 CDP 的 Chrome，请使用专用的 user-data-dir，这样即使 Chrome 已经以您的正常配置文件运行，调试端口也能实际启动：

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

**为什么使用 `--user-data-dir`？** 如果没有它，当普通 Chrome 实例已经在运行时启动 Chrome 通常会在现有进程上打开一个新窗口 —— 而那个现有进程没有以 `--remote-debugging-port` 启动，所以端口 9222 永远不会打开。专用的 user-data-dir 会强制启动一个新的 Chrome 进程，其中调试端口实际监听。`--no-first-run --no-default-browser-check` 会跳过新配置文件的首次启动向导。
:::

通过 CDP 连接时，所有浏览器工具（`browser_navigate`、`browser_click` 等）都在您的实时 Chrome 实例上操作，而不是启动云会话。

### WSL2 + Windows Chrome：优先使用 MCP 而非 `/browser connect` {#wsl2--windows-chrome-prefer-mcp-over-browser-connect}

如果 Hermes 在 WSL2 内运行，但您想要控制的 Chrome 窗口在 Windows 主机上运行，`/browser connect` 通常不是最佳路径。

原因：

- `/browser connect` 期望 Hermes 本身能够访问可用的 CDP 端点
- 现代 Chrome 实时调试会话通常暴露一个主机本地端点，无法像经典的 `9222` 端口那样直接从 WSL 访问
- 即使 Windows Chrome 可调试，最干净的集成通常也是让 Windows 侧的浏览器 MCP 服务器附加到 Chrome，然后让 Hermes 与该 MCP 服务器通信

对于该设置，优先通过 Hermes MCP 支持使用 `chrome-devtools-mcp`。

查看 MCP 指南了解实际设置：

- [将 MCP 与 Hermes 一起使用](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

### 本地浏览器模式

如果您**没有**设置任何云凭证且不使用 `/browser connect`，Hermes 仍然可以通过由 `agent-browser` 驱动的本地 Chromium 安装使用浏览器工具。

### 可选环境变量

```bash
# 住宅代理以获得更好的 CAPTCHA 解决（默认："true"）
BROWSERBASE_PROXIES=true

# 使用自定义 Chromium 的高级隐身 —— 需要 Scale Plan（默认："false"）
BROWSERBASE_ADVANCED_STEALTH=false

# 断开连接后的会话重连 —— 需要付费计划（默认："true"）
BROWSERBASE_KEEP_ALIVE=true

# 自定义会话超时（毫秒）（默认：项目默认）
# 示例：600000（10分钟），1800000（30分钟）
BROWSERBASE_SESSION_TIMEOUT=600000

# 自动清理前的不活动超时（秒）（默认：120）
BROWSER_INACTIVITY_TIMEOUT=120
```

### 安装 agent-browser CLI

```bash
npm install -g agent-browser
# 或在仓库中本地安装：
npm install
```

:::info
`browser` 工具集必须包含在您配置的 `toolsets` 列表中，或通过 `hermes config set toolsets '["hermes-cli", "browser"]'` 启用。
:::

## 可用工具

### `browser_navigate`

导航到 URL。必须在任何其他浏览器工具之前调用。初始化 Browserbase 会话。

```
导航到 https://github.com/NousResearch
```

:::tip
对于简单的信息检索，优先使用 `web_search` 或 `web_extract` —— 它们更快更便宜。当您需要**交互**页面（点击按钮、填写表单、处理动态内容）时使用浏览器工具。
:::

### `browser_snapshot`

获取当前页面可访问性树的基于文本的快照。返回带有引用 ID 的交互元素，如 `@e1`、`@e2`，供 `browser_click` 和 `browser_type` 使用。

- **`full=false`**（默认）：紧凑视图，仅显示交互元素
- **`full=true`**：完整页面内容

超过 8000 个字符的快照会自动由 LLM 总结。

### `browser_click`

点击由快照中的引用 ID 标识的元素。

```
点击 @e5 按下"登录"按钮
```

### `browser_type`

在输入字段中输入文本。先清除字段，然后输入新文本。

```
在搜索字段 @e3 中输入 "hermes agent"
```

### `browser_scroll`

向上或向下滚动页面以显示更多内容。

```
向下滚动查看更多结果
```

### `browser_press`

按下键盘键。用于提交表单或导航。

```
按 Enter 提交表单
```

支持的键：`Enter`、`Tab`、`Escape`、`ArrowDown`、`ArrowUp` 等。

### `browser_back`

导航回浏览器历史中的上一页。

### `browser_get_images`

列出当前页面上的所有图像及其 URL 和替代文本。用于查找要分析的图像。

### `browser_vision`

截图并使用视觉 AI 分析。当文本快照无法捕获重要的视觉信息时特别有用 —— 尤其是 CAPTCHA、复杂布局或视觉验证挑战。

截图被持久保存，文件路径与 AI 分析一起返回。在消息平台（Telegram、Discord、Slack、WhatsApp）上，您可以要求智能体分享截图 —— 它将通过 `MEDIA:` 机制作为原生照片附件发送。

```
这个页面上的图表显示了什么？
```

截图存储在 `~/.hermes/cache/screenshots/` 中，24 小时后自动清理。

### `browser_console`

获取浏览器控制台输出（日志/警告/错误消息）和当前页面的未捕获 JavaScript 异常。对于检测不出现在可访问性树中的静默 JS 错误至关重要。

```
检查浏览器控制台是否有 JavaScript 错误
```

使用 `clear=True` 在阅读后清除控制台，以便后续调用只显示新消息。

`browser_console` 在调用带有 `expression` 参数时还会评估 JavaScript —— 与 DevTools 控制台相同的形状，结果被解析返回（JSON 序列化的对象变为字典；原始值保持原始）。

```
browser_console(expression="document.querySelector('h1').textContent")
browser_console(expression="JSON.stringify(performance.timing)")
```

当当前会话有活跃的 CDP 主管时（对于任何针对支持 CDP 的后端运行过 `browser_navigate` 的会话都是典型的），评估通过主管的持久 WebSocket 运行 —— 没有子进程启动开销。否则回退到标准的 agent-browser CLI 路径。行为在两种方式下完全相同；只有延迟不同。

### `browser_cdp`

原始 Chrome DevTools Protocol 透传 —— 用于其他工具未涵盖的浏览器操作的逃生舱口。用于原生对话框处理、iframe 范围内的评估、Cookie/网络控制，或智能体需要的任何 CDP 动词。

**仅在会话启动时可到达 CDP 端点时可用** —— 意味着 `/browser connect` 已附加到正在运行的 Chrome，或 `browser.cdp_url` 已在 `config.yaml` 中设置。默认的本地 agent-browser 模式、Camofox 和云提供商（Browserbase、Browser Use、Firecrawl）目前不为此工具暴露 CDP —— 云提供商有每会话 CDP URL，但实时会话路由是后续功能。

**CDP 方法参考：** https://chromedevtools.github.io/devtools-protocol/ —— 智能体可以 `web_extract` 特定方法的页面来查找参数和返回形状。

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

# 获取所有 Cookie
browser_cdp(method="Network.getAllCookies")
```

浏览器级别的方法（`Target.*`、`Browser.*`、`Storage.*`）省略 `target_id`。页面级别的方法（`Page.*`、`Runtime.*`、`DOM.*`、`Emulation.*`）需要来自 `Target.getTargets` 的 `target_id`。每个无状态调用是独立的 —— 会话在调用之间不持久化。

**跨域 iframe：** 传递 `frame_id`（来自 `browser_snapshot.frame_tree.children[]` 其中 `is_oopif=true`）以通过主管的实时会话路由该 iframe 的 CDP 调用。这就是在 Browserbase 上跨域 iframe 内 `Runtime.evaluate` 的工作方式，无状态 CDP 连接会遇到签名 URL 过期。示例：

```
browser_cdp(
  method="Runtime.evaluate",
  params={"expression": "document.title", "returnByValue": True},
  frame_id="<来自 browser_snapshot 的 frame_id>",
)
```

同源 iframe 不需要 `frame_id` —— 改用顶级 `Runtime.evaluate` 的 `document.querySelector('iframe').contentDocument`。

### `browser_dialog`

响应原生 JS 对话框（`alert` / `confirm` / `prompt` / `beforeunload`）。在此工具存在之前，对话框会静默阻塞页面的 JavaScript 线程，后续的 `browser_*` 调用会挂起或抛出；现在智能体在 `browser_snapshot` 输出中看到待处理的对话框并明确响应。

**工作流：**
1. 调用 `browser_snapshot`。如果对话框阻塞了页面，它会显示为 `pending_dialogs: [{"id": "d-1", "type": "alert", "message": "..."}]`。
2. 调用 `browser_dialog(action="accept")` 或 `browser_dialog(action="dismiss")`。对于 `prompt()` 对话框，传递 `prompt_text="..."` 来提供响应。
3. 重新快照 —— `pending_dialogs` 为空；页面的 JS 线程已恢复。

**检测自动发生** 通过持久 CDP 主管 —— 每个任务一个 WebSocket，订阅 Page/Runtime/Target 事件。主管还在快照中填充 `frame_tree` 字段，以便智能体可以看到当前页面的 iframe 结构，包括跨域 (OOPIF) iframe。

**可用性矩阵：**

| 后端 | 通过 `pending_dialogs` 检测 | 响应 (`browser_dialog` 工具) |
|---|---|---|
| 通过 `/browser connect` 或 `browser.cdp_url` 的本地 Chrome | ✓ | ✓ 完整工作流 |
| Browserbase | ✓ | ✓ 完整工作流（通过注入的 XHR 桥接） |
| Camofox / 默认本地 agent-browser | ✗ | ✗（无 CDP 端点） |

**在 Browserbase 上如何工作。** Browserbase 的 CDP 代理在服务器端约 10ms 内自动关闭真正的原生对话框，因此我们无法使用 `Page.handleJavaScriptDialog`。主管通过 `Page.addScriptToEvaluateOnNewDocument` 注入一个小脚本，用同步 XHR 覆盖 `window.alert`/`confirm`/`prompt`。我们通过 `Fetch.enable` 拦截这些 XHR —— 页面的 JS 线程在 XHR 上保持阻塞，直到我们用智能体的响应调用 `Fetch.fulfillRequest`。`prompt()` 返回值原样返回到页面 JS。

**对话框策略** 在 `config.yaml` 的 `browser.dialog_policy` 下配置：

| 策略 | 行为 |
|--------|----------|
| `must_respond`（默认） | 捕获，在快照中显示，等待显式 `browser_dialog()` 调用。安全自动关闭在 `browser.dialog_timeout_s`（默认 300s）后，因此有缺陷的智能体不能永远停滞。 |
| `auto_dismiss` | 捕获，立即关闭。智能体仍在 `browser_state` 历史中看到对话框，但不必操作。 |
| `auto_accept` | 捕获，立即接受。用于导航带有激进 `beforeunload` 提示的页面时很有用。 |

`browser_snapshot.frame_tree` 中的 **Frame tree** 限制为 30 个帧和 OOPIF 深度 2，以在广告重的页面上保持负载有界。当达到限制时，会显示 `truncated: true` 标志；需要完整树的智能体可以使用 `browser_cdp` 和 `Page.getFrameTree`。

## 实际示例

### 填写网页表单

```
用户：在 example.com 上使用我的邮箱 john@example.com 注册账户

智能体工作流：
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → 看到带有引用的表单字段
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → 点击"创建账户"
6. browser_snapshot()  → 确认成功
```

### 研究动态内容

```
用户：GitHub 上目前最热门的仓库是什么？

智能体工作流：
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → 读取热门仓库列表
3. 返回格式化结果
```

## 会话录制

自动将浏览器会话录制为 WebM 视频文件：

```yaml
browser:
  record_sessions: true  # 默认：false
```

启用后，录制在第一次 `browser_navigate` 时自动开始，并在会话关闭时保存到 `~/.hermes/browser_recordings/`。在本地和云（Browserbase）模式下都有效。超过 72 小时的录制会自动清理。

## 隐身功能

Browserbase 提供自动隐身能力：

| 功能 | 默认 | 说明 |
|---------|---------|-------|
| 基础隐身 | 始终开启 | 随机指纹、视口随机化、CAPTCHA 解决 |
| 住宅代理 | 开启 | 通过住宅 IP 路由以获得更好的访问 |
| 高级隐身 | 关闭 | 自定义 Chromium 构建，需要 Scale Plan |
| 保持连接 | 开启 | 网络中断后的会话重连 |

:::note
如果付费功能在您的计划上不可用，Hermes 会自动回退 —— 首先禁用 `keepAlive`，然后禁用代理 —— 因此免费计划上的浏览仍然有效。
:::

## 会话管理

- 每个任务通过 Browserbase 获得隔离的浏览器会话
- 不活动后自动清理会话（默认：2 分钟）
- 后台线程每 30 秒检查一次过期会话
- 进程退出时运行紧急清理以防止孤立会话
- 会话通过 Browserbase API 释放（`REQUEST_RELEASE` 状态）

## 限制

- **基于文本的交互** —— 依赖可访问性树，而非像素坐标
- **快照大小** —— 大页面可能被截断或在 8000 字符时由 LLM 总结
- **会话超时** —— 云会话根据您提供商的计划设置过期
- **成本** —— 云会话消耗提供商积分；会话在对话结束或不活动后自动清理。使用 `/browser connect` 进行免费本地浏览。
- **无文件下载** —— 无法从浏览器下载文件
