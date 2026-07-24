---
sidebar_position: 3
title: "桌面应用"
description: "Hermes 原生桌面应用 —— 精心打造的 Hermes 聊天体验，支持流式工具输出、并排预览、文件浏览器、语音、定时任务、配置文件、技能和设置。支持 macOS、Windows 和 Linux。"
---

# 桌面应用

Hermes 桌面应用是一个原生应用，基于你通过 CLI 和网关使用的**同一个** Agent —— 同样的配置、同样的 API 密钥、同样的会话、同样的技能、同样的记忆。它不是独立的产品或精简的克隆；它使用相同的 Hermes Agent 核心和设置，通过一个现代且精心设计的 UI 来驱动。如果你在终端中使用过 `hermes`，你在那边设置的一切都已经在这里了，你在这里做的任何事情也会在那里显示。

它运行在 **macOS、Windows 和 Linux** 上。

:::tip 哪个界面是哪个？
Hermes 有多个前端，都与同一个 Agent 通信：

- **桌面应用**（本页） —— 一个带有专用 UI 的原生应用，用于聊天、配置和管理。
- **CLI**（`hermes`）和 **[TUI](./tui.md)**（`hermes --tui`） —— 终端界面。
- **[Web Dashboard](./features/web-dashboard.md)**（`hermes dashboard`） —— 浏览器管理面板；其可选的 **Chat** 标签页通过伪终端嵌入了 TUI。

根据当前需要选择合适的界面。它们共享状态，所以你可以在一个界面开始会话，在另一个界面继续。
:::

## 安装

请参阅 [Hermes Desktop 安装说明](../getting-started/installation.md)。

如果你已经安装了 Hermes，只需运行

```bash
hermes desktop
```

它会使用你当前的配置、密钥、会话和技能。

## 应用功能

桌面应用以聊天为核心窗口，左侧边栏用于导航。它支持管理多个同时进行的 Agent 对话、配置消息 Provider、创建制品、浏览项目文件夹结构，以及同时处理多个项目。

### 聊天

应用的核心。你可以获得：

- **流式响应**，在 Agent 工作时实时显示工具活动和结构化的工具调用摘要。
- **与其他 Hermes 界面相同的对话历史** —— 这里开始的会话可以在 CLI/TUI 中继续，反之亦然。
- **拖放文件**到聊天区域的任何位置来附加到你的下一条消息。
- **右侧预览栏** —— 在你继续聊天的同时，渲染网页、文件和工具输出。
- **编辑器历史和队列编辑** —— 在空编辑器中按上/下箭头键可以召回和重用之前的提示，并在发送前编辑已排队的消息。

#### 状态栏

聊天底部的状态栏显示实时会话状态并提供快速控制，无需打开设置：

- **按会话的 YOLO 切换** —— 仅为当前会话开关 YOLO（与 TUI 一致）。YOLO 会绕过危险命令的审批提示，所以请了解你在关闭什么 —— 参见[安全 → YOLO 模式](./security.md#yolo-mode)。

不是在连接另一台机器上的 Hermes 实例，而是连接捆绑的本地后端？请参阅下方的[连接远程后端](#连接远程后端) —— 有关远程托管 Dashboard 连接的完整说明（认证网关、`/api/ws` 聊天套接字和 WebSocket 关闭码诊断），请参阅 [Web Dashboard → 将 Hermes Desktop 连接到远程后端](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend)。

#### 选择模型

模型选择器位于**编辑器**中，麦克风的左侧。点击它可以切换模型、推理力度和快速模式。

- **编辑器选择器是粘性 UI 状态，不会修改你的默认设置。** 它在本地记住（按设备），并且**跨新聊天和重启保持**，而不是回滚到默认值 —— 选择一次模型，下次 `Cmd/Ctrl+N` 就会打开该模型。在活跃聊天中切换模型会将变更限定到**当前聊天**；无论哪种方式，选择都会随会话创建/切换而保持，**绝不会**写入配置文件默认值。（切换[配置文件](#会话--配置文件)会重新设置为该配置文件自己的默认值。）
- **在 设置 → 模型 中设置默认值。** 那个"主"模型是你的**每个配置文件的全局默认值** —— 它是新聊天、定时任务、子代理和辅助任务的起始模型，也是唯一写入默认值的地方。每个[配置文件](#会话--配置文件)保持自己的默认值。
- **按模型的推理力度/快速预设。** 桌面应用中每个模型都记住自己的推理力度和快速模式选择，每次选择该模型时重新应用到会话。这些预设仅影响桌面应用的便利性，不影响定时任务或子代理。
- **会话中切换模型会重置提示缓存。** 在活跃聊天中切换模型意味着下一条消息会以完整输入价格重新读取整个对话（Provider 的提示缓存按模型标识）。偶尔切换没问题；在长聊天中，在新模型上开一个新聊天通常比来回切换更划算。

### 文件浏览器

在不离开应用的情况下探索和预览工作目录 —— 在 Agent 读取、写入和编辑文件时方便跟踪。使用 `hermes desktop --cwd <path>`（或 `HERMES_DESKTOP_CWD` 环境变量）设置初始项目目录。

### 语音

与 Hermes 语音对话并听到回复，与在其他地方可用的[语音模式](./features/voice-mode.md)相同。在 macOS 上，系统会提示一次麦克风访问权限。

### 设置和引导

从真正的 UI 中管理 Provider、模型、工具和凭据，而不是编辑 YAML。首次运行引导让你在几秒内就能发送第一条消息。设置面板涵盖 Provider/密钥、模型选择、工具集配置、MCP 服务器、网关和会话管理。

- **Provider 设置面板** —— 一个专门管理推理 Provider 的地方，提供账户/API 密钥交互体验，用于登录和存储每个 Provider 的凭据。
- **菜单中的每个 Provider 和模型** —— GUI 展示了完整的 Provider 列表和 `hermes model` 所知的每个模型，因此你从 CLI 看到的同一个目录中选择，而不是精选的子集。
- **xAI Grok OAuth** —— Grok 在启动器中是一等 OAuth Provider；通过浏览器流程登录，与其他 OAuth Provider 相同。
- **GUI 中的工具后端安装** —— 直接从应用运行工具后端的后安装步骤，无需切换到终端。
- **辅助模型警告** —— 如果你将主模型切换到新 Provider，而辅助任务（标题生成、摘要等）仍固定在另一个 Provider 上，应用会提醒你，避免在两个 Provider 之间无形地分散工作。

首次运行引导已重新设计为统一的覆盖层设计系统，你可以选择 **Choose provider later** 跳过 Provider 设置，先进入应用。

### 管理面板

应用还展示了更广泛的 Hermes 管理界面，无需切换到终端：

- **技能（Skills）** —— 浏览、安装和管理[技能](./features/skills.md)。
- **定时任务（Cron）** —— 查看和管理[定时任务](../reference/cli-commands.md#hermes-cron)。
- **配置文件（Profiles）** —— 在 [Hermes 配置文件](./profiles.md)之间切换（隔离的配置/技能/会话）。
- **消息（Messaging）** —— 设置网关频道。
- **代理（Agents）** 和 **指挥中心（Command Center）** —— 多 Agent 工作的编排界面。

### 键盘和导航

- **命令面板** —— 按 **Cmd+K**（Windows/Linux 上为 Ctrl+K）跳转到操作并使用键盘导航应用。
- **可重新绑定的快捷键** —— 设置中的快捷键面板允许你将应用的键盘快捷键重新映射为自定义按键。
- **自定义缩放快捷键** —— 以半步增量缩放界面，精细控制文字大小。
- **UI 语言切换器** —— 在应用内切换界面语言，包括简体中文（zh-Hans）。

### 会话和配置文件

- **会话列表改版** —— 重写的会话列表，支持归档和通用会话管理，让列表在增长时保持可控。
- **按 ID 搜索会话** —— 通过 ID 直接查找特定会话。
- **并发多配置文件会话** —— 同时运行多个[配置文件](./profiles.md)的会话，并通过跨配置文件的 `@session` 链接引用另一个配置文件中的会话。

## 更新

应用会在后台检查更新，并在有更新时提供一键更新。

[手动更新流程](https://hermes-agent.nousresearch.com/docs/getting-started/updating)也可以通过 GUI 完成。

## 卸载

打开 **设置 → 关于 → 危险区域**，选择要移除的范围：

- **仅卸载 Chat GUI** —— 移除桌面应用及其数据；Hermes Agent、你的配置和聊天记录保留。（等同于 `hermes uninstall --gui`。）
- **卸载 GUI + Agent，保留数据** —— 移除应用和 Agent，但保留配置、聊天记录和密钥以便将来重新安装。（等同于 `hermes uninstall`。）
- **卸载所有内容** —— 移除应用、Agent 和所有用户数据。（等同于 `hermes uninstall --full`。）

应用关闭以完成操作（清理在退出后运行，以便移除正在运行的应用包及其 venv）。当没有本地 Agent 安装时（例如，仅 GUI 的"精简"客户端连接到远程后端），Agent 移除选项会自动隐藏。

你也可以在终端完成相同操作 —— `hermes uninstall --gui` 仅卸载 GUI，或 `hermes uninstall` / `hermes uninstall --full` 同时卸载 Agent。

:::note
从**源码检出**（`hermes desktop` 开发构建）运行 `hermes uninstall --gui` 还会移除工作区 `node_modules` 和 `apps/desktop/{dist,release}` 构建输出，因为这些是 GUI 构建制品。它们可以通过 `hermes desktop`（或 `npm install` + 重新构建）恢复 —— 但如果你正在开发桌面应用，预期之后需要重新安装依赖。
:::

## CLI 参考：`hermes desktop`

要通过 CLI 启动，只需运行 `hermes desktop`。默认情况下，它会安装工作区 Node 依赖、构建当前操作系统的未打包 Electron 应用，然后启动该打包产物。

| 标志 | 说明 |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `--skip-build` | 跳过 npm install/package，直接从 `apps/desktop/release` 启动现有的未打包应用 |
| `--force-build` | 即使内容戳记匹配也强制完全重建 |
| `--build-only` | 构建桌面应用但不启动（由 `hermes update` 使用） |
| `--source` | 通过 `electron .` 启动，指向 `apps/desktop/dist` 而非打包应用 |
| `--cwd PATH` | 桌面聊天会话的初始项目目录（设置 `HERMES_DESKTOP_CWD`） |
| `--hermes-root PATH` | 覆盖应用使用的 Hermes 源码根目录（设置 `HERMES_DESKTOP_HERMES_ROOT`） |
| `--ignore-existing` | 强制应用在后端解析时忽略 `PATH` 上已有的 `hermes` CLI |
| `--fake-boot` | 启用确定性启动延迟，用于验证启动 UI |

## 工作原理

打包应用包含 Electron shell 和原生 React 聊天界面。首次启动时可以将 Hermes Agent 运行时安装到 `HERMES_HOME`（`~/.hermes`，或 Windows 上的 `%LOCALAPPDATA%\hermes`） —— **与 CLI 安装使用相同的布局**，这就是为什么两者可以互换。后端解析首先遵循 `HERMES_DESKTOP_HERMES_ROOT`，然后是已完成的托管安装，接着是 `PATH` 上探测到的 `hermes`（除非设置了 `--ignore-existing` / `HERMES_DESKTOP_IGNORE_EXISTING=1`），最后是 Nix 等打包器的显式 `HERMES_DESKTOP_HERMES` 命令覆盖。React 渲染器与应用为你启动的无头后端通信 —— 一个运行 `tui_gateway` JSON-RPC/WebSocket API 的 `hermes serve` 进程 —— 并复用 Agent 运行时而不是嵌入 `hermes --tui`。桌面应用是**自包含的**：它运行自己的 `hermes serve` 后端，从不打开或需要 [Web Dashboard](./features/web-dashboard.md)。（比 `serve` 命令更早的运行时会自动回退到无头的 `dashboard --no-open`，因此应用更新永远不会超前后端。）安装、后端解析和自更新逻辑都在 Electron 主进程中。

## 连接远程后端

默认情况下，应用启动并管理自己的**本地**后端。你也可以将其指向运行在另一台机器上的 Hermes 后端 —— VPS、家庭服务器或 Tailscale 后面的 Mini。

:::info 远程后端是一个运行中的 `hermes serve` 进程
"远程后端"是指在远程机器上运行的 **`hermes serve`** 服务器 —— 这是桌面应用连接的进程。本节中的任何操作在该后端实际运行并可达之前都不起作用。桌面应用不会为你启动它；你（或 `systemd` 服务）在远程主机上保持 `hermes serve` 运行，应用连接到它。如果你还使用消息频道（Telegram、Discord 等），**网关**是一个*独立的*长时间运行进程，你需要单独启动 —— 请参阅设置步骤之后的说明。
:::

连接分为两半：在后端你通过 **Auth Provider** 保护它，在应用中你输入后端的 URL 并登录。将后端绑定到非回环地址会自动启用其认证网关，你配置的 Provider 是允许桌面应用通过的钥匙。

**根据后端所在位置选择 Provider：**

- **OAuth（Nous Portal） —— 推荐用于任何超出你自己机器的可达环境。** 登录凭据通过你的 Nous 账户验证，因此适合 VPS、公共主机或任何远程后端。通过 `hermes dashboard register`（或 Portal [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) 页面）注册 Dashboard 以配置其 OAuth 客户端，然后在应用中使用 **Sign in with Nous Research** 登录。如果你运行自己的身份提供者，自托管 OIDC Provider 的工作方式相同。
- **用户名/密码 —— 仅用于本地/可信网络。** 当后端在同一个可信 LAN 上或仅通过 VPN 可达时（例如 Tailscale），这是最简单的选项。它使用单个共享凭据保护，没有外部身份提供者，因此**不要用于暴露在公共互联网上的 Dashboard** —— 此时请使用 OAuth。

本节其余部分展示用户名/密码路径，因为它在可信网络上最快搭建；OAuth 路径请参阅 [Web Dashboard → 默认 Provider：Nous Research](./features/web-dashboard.md#default-provider-nous-research)。

### 在后端（远程机器）上

设置用户名和密码，然后启动后端绑定到可达地址。凭据存放在 `~/.hermes/.env` 中（密钥文件，权限 0600）：

```bash
# 1. Set the dashboard login credentials.
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# Recommended: a stable signing secret so sessions survive restarts.
# Without it a random key is generated per boot and you'll be logged out
# on every restart.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. Run the backend bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes serve --host 0.0.0.0 --port 9119
```

保持 `hermes serve` 进程运行的时间就是你希望桌面应用能连接的时间 —— 如果它停止了，应用就无法再访问后端。在 `systemd`、`tmux` 或你选择的进程管理器下运行它，使其在注销和重启后保持存活。

另外，如果你依赖消息频道，请确保**网关正在远程主机上运行** —— `hermes serve` 后端是桌面应用通信的对象，但你的 Telegram/Discord/Slack 网关会话是一个独立的进程，需要你单独启动和保持运行。请参阅[消息](./messaging/index.md)了解网关设置。

不想以明文保存密码？将 `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` 设置为 scrypt 哈希值 —— 使用 `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"` 计算。完整的配置界面（config.yaml 键、所有环境变量、速率限制器）：[Web Dashboard → 用户名/密码 Provider](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp)。

作为 systemd 服务运行后端？给单元文件添加 `EnvironmentFile=%h/.hermes/.env`，使凭据在启动时就在环境中。

:::warning
后端读写你的 `.env`（API 密钥、密钥），并可以运行 Agent 命令。上面展示的**用户名/密码**设置适用于可信网络 —— 永远不要将密码保护的后端直接暴露在公共互联网上；将其放在 VPN 后面。[Tailscale](https://tailscale.com/) 是最佳选择：绑定到机器的 Tailscale IP（`--host <tailscale-ip>`），使用 `http://<tailscale-ip>:9119` 作为远程 URL，这样只有你的 tailnet 能访问它。要通过公共互联网访问后端，请使用 **OAuth（Nous Portal）** Provider。
:::

### 在应用中

**设置 → 网关 → 远程网关：**

1. **远程 URL** —— `http://<后端主机>:9119`（如果前面有反向代理，`/hermes` 等路径前缀也可以）
2. **登录** —— 应用会检测后端公告的 Provider 并适配按钮。对于用户名/密码后端，它会显示一个 **Sign in** 按钮，打开凭据表单（输入步骤 1 的凭据）。对于 OAuth 后端，它会显示 **Sign in with `<provider>`**（例如 *Sign in with Nous Research*），执行 Provider 的浏览器登录流程。无论哪种方式，应用都会获得一个对后端的已认证会话。
3. **保存并重新连接** —— 将桌面 shell 切换到远程后端。会话会自动刷新；当设置了 `HERMES_DASHBOARD_BASIC_AUTH_SECRET` 时，你跨重启保持登录状态。

你也可以在启动应用前通过 `HERMES_DESKTOP_REMOTE_URL` 环境变量设置后端 URL（它覆盖应用内设置）；你仍然从网关设置面板登录。

:::note 按配置文件的远程主机
远程网关主机按[配置文件](./profiles.md)配置，因此每个配置文件可以指向自己的远程后端（或保持本地）。切换配置文件会切换应用连接的远程主机。
:::

### 故障排除

- **登录失败 401 / "Invalid credentials"** —— 用户名或密码与后端的 `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` 不匹配。后端对未知用户和错误密码返回相同的通用错误（没有枚举漏洞），所以请仔细检查两者。用 `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` 确认网关已开启 —— 应该报告 `true` 并包含 `"basic"`。
- **没有 "Sign in" 按钮 —— 它要求输入会话令牌** —— 后端的用户名/密码 Provider 未激活。`/api/status` 不会在 `auth_providers` 中列出 `"basic"`。确保在 `~/.hermes/.env` 中设置了用户名和密码（或密码哈希），并且 Dashboard 进程实际加载了它们。
- **每次重启都退出登录** —— 将 `HERMES_DASHBOARD_BASIC_AUTH_SECRET` 设置为稳定的值。没有它的话，令牌签名密钥每次启动都会重新生成，使所有会话失效。
- **连接被拒绝 / 超时** —— 后端绑定到 `127.0.0.1`（默认），或防火墙/VPN 阻止了端口。绑定到 `0.0.0.0` 或 Tailscale IP 并将端口开放给你的可信网络。

相同的 Web Dashboard 角度设置请参阅 [Web Dashboard → 将 Hermes Desktop 连接到远程后端](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend)；环境变量列在[环境变量 → Web Dashboard & Hermes Desktop](../reference/environment-variables.md#web-dashboard--hermes-desktop) 下。

## 故障排除

启动日志位于 `HERMES_HOME/logs/desktop.log`（包含后端输出和最近的 Python 回溯） —— 如果应用报告启动失败，首先查看它。你也可以从 CLI 跟踪它：

```bash
hermes logs gui -f
```

常用重置操作：

```bash
# 强制干净的首次启动设置（macOS/Linux）
rm "$HOME/.hermes/hermes-agent/.hermes-bootstrap-complete"

# 重建损坏的 Python venv（macOS/Linux）
rm -rf "$HOME/.hermes/hermes-agent/venv"

# 重置卡住的 macOS 麦克风提示
tccutil reset Microphone com.nousresearch.hermes
```

### "Build desktop app" 卡在 Electron 下载

构建过程从 `github.com/electron/electron/releases` 下载 Electron 运行时（约 114 MB）。如果安装程序在 **Build desktop app** 步骤挂起，实时输出重复 `retrying attempt=…`，说明你的网络上 GitHub 被阻止或限速（防火墙、代理或区域限制）。

安装程序会自动修复此问题：构建失败时它会 (1) 清除损坏的缓存 Electron zip 并重试，然后 (2) 如果仍然失败且你没有设置 `ELECTRON_MIRROR`，会通过 `npmmirror.com`（事实上的 Electron 社区镜像）再重试一次。`@electron/get` 会进行 SHASUM 校验，但校验和来自同一镜像 —— 这能捕获损坏或不完整的下载，而非被篡改的镜像。如果你不想信任第三方主机，请设置你自己的 `ELECTRON_MIRROR`（如下）；构建不会覆盖你已设置的值。

要**选择自己的镜像**（例如企业/可信镜像），在安装前设置 `ELECTRON_MIRROR` 或手动重新构建 —— 构建会遵循它且不会覆盖：

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
  bash -c 'cd "$HOME/.hermes/hermes-agent/apps/desktop" && CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack'
```

手动清除损坏的缓存 zip：

```bash
rm -f "$HOME/Library/Caches/electron"/electron-*.zip   # macOS
rm -f "$HOME/.cache/electron"/electron-*.zip            # Linux
```

## 从源码构建

如果你想修改应用本身，从仓库根目录安装一次工作区依赖，然后从 `apps/desktop` 运行开发服务器：

```bash
npm install          # from repo root — links apps/desktop, web, apps/shared
cd apps/desktop
npm run dev          # Vite renderer + Electron, which boots the Python backend
```

将应用指向特定的检出目录，或将其与真实配置隔离：

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev
HERMES_HOME=/tmp/throwaway npm run dev
npm run dev:fake-boot   # exercise the startup overlay with deterministic delays
```

构建安装程序：

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # unpacked app under release/ (no installer)
```

macOS/Windows 签名和公证在环境中有相关凭据时自动运行（macOS 使用 `CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*`，Windows 使用 `WIN_CSC_*`）。

## 另请参阅

- [CLI 指南](./cli.md) —— 终端界面
- [TUI](./tui.md) —— `hermes --tui` 和 Dashboard 聊天标签页使用的现代终端 UI
- [Web Dashboard](./features/web-dashboard.md) —— 带有嵌入式聊天标签页的浏览器管理面板
- [配置](./configuration.md) —— 桌面应用读写的配置
- [Windows（原生）](./windows-native.md) —— Windows 原生安装路径
