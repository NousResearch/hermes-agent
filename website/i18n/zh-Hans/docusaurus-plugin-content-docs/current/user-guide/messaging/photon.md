---
sidebar_position: 18
---

# Photon iMessage

通过 [Photon][photon] 将 Hermes 连接到 **iMessage**，这是一个托管服务，处理 Apple 线路分配和滥用预防层，因此你不需要运行自己的 Mac 中继。

免费层使用 Photon 的共享 iMessage 线路池 —— 不同收件人可能看到不同的发送号码，但每个对话保持稳定。付费 Business 层为每个用户提供相同的专用号码；插件同时支持两者，推荐从免费层开始。

:::info 免费开始
Photon 的共享线路池是免费的。从 Hermes 发送你的第一条 iMessage 无需订阅 —— 只需要一个可以绑定到你账户的手机号码。
:::

## 架构

Photon 是一个**持久连接**频道，与 Discord 或 Slack 类似 —— **无需 webhook、无需公共 URL、无需管理签名密钥。**

`spectrum-ts` SDK 通过双向 gRPC 流维护与 Photon 的长连接。因为 SDK 仅支持 TypeScript，Hermes 在一个小型受监督的 **Node sidecar** 中运行它，并通过回环地址通信：

- **入站** —— sidecar 消费 SDK 的 `app.messages` gRPC 流，通过回环 `GET /inbound`（NDJSON）将每条消息转发给 Python 适配器。适配器去重并分派给 Agent，如果流断开会自动重连。
- **出站** —— 回复是发给 sidecar 的回环 POST，它调用 SDK 的 `space.send(...)`。

Python 插件自动启动、监督和关闭 sidecar。

## 前置条件

- 一个 Photon 账户 —— 在 [app.photon.codes][app] 注册
- **Node.js 18.17 或更新版本**在 PATH 中（`node --version`）
- 一个可以接收 iMessage 的手机号码（用于绑定你的账户）

就这样 —— 无需设置公共 URL 或隧道。

## 首次设置

运行统一网关向导并选择 **Photon iMessage**：

```bash
hermes gateway setup
```

……或直接运行 Photon 设置（向导调用相同的流程）：

```bash
# Device-code login + project + user + sidecar deps, all in one
hermes photon setup --phone +155****4567
```

设置按以下顺序执行：

1. **设备登录**（`client_id=photon-cli`）—— 打开 `https://app.photon.codes/` 进行授权并存储 bearer 令牌。
2. 在你的账户上**查找或创建** `Hermes Agent` 项目。
3. **启用 Spectrum**，读取项目的 Spectrum id，并轮换项目密钥。
4. **注册你的手机号码**为 Spectrum 用户 —— 如果该号码的用户已存在则跳过，所以重新运行是安全的。
5. **打印分配给你的 iMessage 线路** —— 你用来联系 Agent 的号码。
6. **运行 `npm install`**，在插件的 sidecar 目录内。

运行时凭据写入 `~/.hermes/.env`（`PHOTON_PROJECT_ID` = Spectrum 项目 id，`PHOTON_PROJECT_SECRET`），与其他频道存放令牌的位置相同。管理元数据（设备令牌、Dashboard 项目 id）存放在 `~/.hermes/auth.json` 的 `credential_pool.photon` / `credential_pool.photon_project` 下。

## 授权用户

Photon 使用与其他 Hermes 频道相同的授权模型。选择一种方式：

**DM 配对（默认）。** 当一个未知号码向你的 Photon 线路发消息时，Hermes 回复一个配对码。使用以下命令批准：

```bash
hermes pairing approve photon <CODE>
```

使用 `hermes pairing list` 查看待处理的配对码和已批准的用户。

**预授权特定号码**（在 `~/.hermes/.env` 中）：

```bash
PHOTON_ALLOWED_USERS=+155****4567,+155****6543
```

**开放访问**（仅限开发，在 `~/.hermes/.env` 中）：

```bash
PHOTON_ALLOW_ALL_USERS=true
```

当设置了 `PHOTON_ALLOWED_USERS` 时，未知发送者会被静默忽略，而不是提供配对码（允许列表表明你有意限制了访问）。

### 群聊中要求 @提及

默认情况下 Hermes 响应所有已授权的私信和群组消息。要使群聊成为选择性参与，请启用提及门控（私信仍然始终有效）：

```yaml
gateway:
  platforms:
    photon:
      enabled: true
      require_mention: true
```

设置 `require_mention: true` 后，群聊消息在不匹配唤醒词模式时会被忽略。默认匹配 `Hermes` 和 `@Hermes agent` 变体。对于自定义 Agent 名称，设置正则模式：

```yaml
gateway:
  platforms:
    photon:
      require_mention: true
      mention_patterns:
        - '(?<![\\w@])@?amos\\b[,:\\-]?'
```

两个键也接受环境变量（`PHOTON_REQUIRE_MENTION`、`PHOTON_MENTION_PATTERNS`）。这与 BlueBubbles iMessage 频道使用的提及门控模型相同。

## 启动网关

```bash
hermes gateway start
```

你会看到类似这样的输出：

```
[photon] connected — sidecar on 127.0.0.1:8789, streaming inbound over gRPC
```

向你分配的号码发送一条 iMessage，Hermes 就会回复。

## 状态和故障排除

```bash
hermes photon status
```

打印已保存的凭据、sidecar 健康状态、你注册的号码和 Hermes 使用的分配 iMessage 线路。当 Photon 令牌和 Dashboard 项目可用时，`status` 会从 Dashboard 刷新缺失的号码行，而不会配置新线路。

```
Photon iMessage status
──────────────────────
  device token        : ✓ stored
  dashboard project   : 3c90c3cc-0d44-4b50-...
  spectrum project id : sp-...
  project secret      : ✓ stored
  my number           : +155****4567
  assigned number     : +162****9185
  node binary         : /usr/bin/node
  sidecar deps        : ✓ installed
```

常见问题：

- **`sidecar deps : ✗ run hermes photon install-sidecar`** —— Node 已安装但 `spectrum-ts` 未安装。运行建议的命令。
- **`device token : ✗ missing`** —— 运行 `hermes photon setup` 进行登录。
- **`No iMessage line assigned yet`** —— Spectrum 已启用但未配置线路；重新运行 `hermes photon setup` 或检查 [Dashboard][app]。
- **Sidecar 无法启动** —— 确认 `node --version` 是 18.17+，并且 `hermes photon install-sidecar` 完成且无错误。

## 当前限制

- **入站附件仅包含元数据。** 入站事件携带文件名 + MIME 类型；Agent 看到标记但还不能读取字节。SDK 通过 `content.read()` 暴露附件字节，这是 sidecar 的后续功能。
- **出站附件已支持。** Hermes 通过 sidecar 的 `/send-attachment` 端点使用 spectrum-ts 的 `attachment()` / `voice()` 内容构建器发送图片、语音笔记、视频和文档。说明文字作为单独的 iMessage 气泡在媒体之后发送。
- **Photon 免费配额：** 每服务器每天 5,000 条消息，每共享线路每天 50 次新对话发起。如需提升请联系 `help@photon.codes`。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PHOTON_PROJECT_ID` | 从 `.env` | Spectrum 项目 id（SDK 的 `projectId`）；由 setup 设置 |
| `PHOTON_PROJECT_SECRET` | 从 `.env` | 项目密钥；由 setup 设置 |
| `PHOTON_SIDECAR_PORT` | `8789` | Sidecar 控制 + 入站通道的回环端口 |
| `PHOTON_SIDECAR_AUTOSTART` | `true` | 适配器是否启动 sidecar |
| `PHOTON_NODE_BIN` | `which node` | 覆盖 Node 二进制路径 |
| `PHOTON_HOME_CHANNEL` | （未设置） | 定时任务/通知的默认空间 id |
| `PHOTON_HOME_CHANNEL_NAME` | （未设置） | 主频道的人类可读标签 |
| `PHOTON_ALLOWED_USERS` | （未设置） | 逗号分隔的 E.164 允许列表 |
| `PHOTON_ALLOW_ALL_USERS` | `false` | 仅限开发 —— 接受任何发送者 |
| `PHOTON_REQUIRE_MENTION` | `false` | 在群组中响应前需要唤醒词 |
| `PHOTON_MENTION_PATTERNS` | Hermes 唤醒词 | JSON 列表/逗号/换行正则模式，用于群组提及 |
| `PHOTON_DASHBOARD_HOST` | `app.photon.codes` | 覆盖 Dashboard / 设备登录主机 |
| `PHOTON_SPECTRUM_HOST` | `spectrum.photon.codes` | 覆盖 Spectrum API 主机 |

[photon]: https://photon.codes/
[app]: https://app.photon.codes/
