---
sidebar_position: 17
title: "LINE"
description: "将 Hermes Agent 设置为 LINE Messaging API 机器人"
---

# LINE 配置

通过官方 LINE Messaging API 将 Hermes Agent 作为 [LINE](https://line.me/) 机器人运行。适配器以捆绑平台插件的形式存放于 `plugins/platforms/line/` — 无需修改核心代码，像其他平台一样启用即可。

LINE 是日本、台湾和泰国的主流即时通讯应用。如果你的用户在这些地区，这就是他们与你沟通的方式。

## 机器人响应方式

| 场景 | 行为 |
|---------|----------|
| **1:1 聊天**（`U` 开头 ID） | 响应每条消息 |
| **群聊**（`C` 开头 ID） | 仅当群组在白名单中时响应 |
| **多人房间**（`R` 开头 ID） | 仅当房间在白名单中时响应 |

入站的文本、图片、音频、视频、文件、贴纸和位置信息均可处理。出站文本优先使用**免费 reply token**（单次使用，有效期约 60 秒），token 过期后回退至计费的 Push API。

---

## 第一步：创建 LINE Messaging API 频道

1. 前往 [LINE Developers Console](https://developers.line.biz/console/)。
2. 创建一个 Provider，然后在其下创建一个 **Messaging API** 频道。
3. 在频道的 **Basic settings** 标签页中，复制 **Channel secret**。
4. 在 **Messaging API** 标签页中，滚动至 **Channel access token (long-lived)** 并点击 **Issue**，复制该 token。
5. 在 **Messaging API** 标签页中，同时禁用 **Auto-reply messages** 和 **Greeting messages**，避免与机器人回复冲突。

---

## 第二步：暴露 webhook 端口

LINE 通过公网 HTTPS 推送 webhook。默认端口为 `8646` — 如需修改，可通过 `LINE_PORT` 覆盖。

```bash
# Cloudflare Tunnel（推荐用于生产环境 — 固定主机名）
cloudflared tunnel --url http://localhost:8646

# ngrok（适合开发环境）
ngrok http 8646

# devtunnel
devtunnel create hermes-line --allow-anonymous
devtunnel port create hermes-line -p 8646 --protocol https
devtunnel host hermes-line
```

复制 `https://...` URL — 稍后将其设置为 webhook URL。**保持隧道运行**以便测试。生产环境请配置固定的 Cloudflare 命名隧道，避免重启后 webhook URL 变更。

---

## 第三步：配置 Hermes

在 `~/.hermes/.env` 中添加：

```env
LINE_CHANNEL_ACCESS_TOKEN=YOUR_LONG_LIVED_TOKEN
LINE_CHANNEL_SECRET=YOUR_CHANNEL_SECRET

# 白名单 — 至少填写其中一项（开发环境可使用 LINE_ALLOW_ALL_USERS=true）
LINE_ALLOWED_USERS=U1234567890abcdef...           # 逗号分隔的 U 开头 ID
LINE_ALLOWED_GROUPS=C1234567890abcdef...          # 可选的群组 ID
LINE_ALLOWED_ROOMS=R1234567890abcdef...           # 可选的房间 ID

# 发送图片 / 音频 / 视频时必填 — 隧道解析到的公网 HTTPS 基础 URL
# 未设置时，send_image/voice/video 将拒绝执行
LINE_PUBLIC_URL=https://my-tunnel.example.com
```

然后在 `~/.hermes/config.yaml` 中：

```yaml
gateway:
  platforms:
    line:
      enabled: true
```

这就够了 — `gateway/config.py` 中的捆绑插件扫描会自动识别 `plugins/platforms/line/`。无需编辑 `Platform.LINE` 枚举，无需注册 `_create_adapter`。

---

## 第四步：设置 webhook URL

回到 LINE 控制台：

1. 打开你的频道 → **Messaging API** 标签页。
2. 在 **Webhook settings** → **Webhook URL** 下，粘贴 `https://<your-tunnel>/line/webhook`（注意 `/line/webhook` 路径 — 适配器在此监听）。
3. 点击 **Verify**。LINE 会 ping 该 URL，你应看到 200 响应。
4. 将 **Use webhook** 切换为 **On**。

---

## 第五步：运行 gateway

```bash
hermes gateway
```

Agent 日志显示：

```
LINE: webhook listening on 0.0.0.0:8646/line/webhook (public: https://my-tunnel.example.com)
```

从 LINE 应用将机器人添加为好友（扫描频道 **Messaging API** 标签页中的二维码），然后发送一条消息。

---

## LLM 响应缓慢

LINE 的 reply token 为单次使用，在入站事件发生后约 60 秒过期。LLM 响应过慢时将无法及时回复，通常会被迫调用付费的 Push API。

当 LLM 运行时间超过 `LINE_SLOW_RESPONSE_THRESHOLD` 秒（默认 `45`）时，适配器会消耗原始 reply token，发送一个 **Template Buttons** 气泡：

> 🤔 Still thinking. Tap below to fetch the answer when it's ready.
>
> [ Get answer ]

用户在方便时点击 **Get answer** — 该 postback 会带来一个*新的* reply token，适配器用它发送缓存的答案（仍然免费）。

状态机：`PENDING → READY → DELIVERED`，以及 `ERROR`（用于已取消的运行 — 执行 `/stop` 后，孤立的 PENDING 状态会解析为"Run was interrupted before completion."，避免持久按钮循环触发）。

如需禁用 postback 按钮并始终回退至 Push API：

```env
LINE_SLOW_RESPONSE_THRESHOLD=0
```

为使 postback 流程可靠触发，请抑制可能在阈值前消耗 reply token 的冗余输出：

```yaml
# ~/.hermes/config.yaml
display:
  interim_assistant_messages: false
  platforms:
    line:
      tool_progress: off
```

---

## Cron / 通知推送

```env
LINE_HOME_CHANNEL=Uxxxxxxxxxxxxxxxxxxxx     # 默认推送目标
```

设置了 `deliver: line` 的 Cron 任务会路由至 `LINE_HOME_CHANNEL`。适配器内置独立的仅 Push 发送器，因此即使 cron 在独立进程中运行，也能正常工作。

---

## 环境变量参考

| 变量 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|
| `LINE_CHANNEL_ACCESS_TOKEN` | 是 | — | 长期有效的频道访问 token |
| `LINE_CHANNEL_SECRET` | 是 | — | Channel secret（用于 HMAC-SHA256 webhook 验证） |
| `LINE_HOST` | 否 | `0.0.0.0` | Webhook 绑定主机 |
| `LINE_PORT` | 否 | `8646` | Webhook 绑定端口 |
| `LINE_PUBLIC_URL` | 媒体发送时必填 | — | 公网 HTTPS 基础 URL；发送图片/音频/视频时必须设置 |
| `LINE_ALLOWED_USERS` | 三选一 | — | 逗号分隔的用户 ID（U 开头） |
| `LINE_ALLOWED_GROUPS` | 三选一 | — | 逗号分隔的群组 ID（C 开头） |
| `LINE_ALLOWED_ROOMS` | 三选一 | — | 逗号分隔的房间 ID（R 开头） |
| `LINE_ALLOW_ALL_USERS` | 仅开发环境 | `false` | 完全跳过白名单验证 |
| `LINE_HOME_CHANNEL` | 否 | — | 默认 cron / 通知推送目标 |
| `LINE_SLOW_RESPONSE_THRESHOLD` | 否 | `45` | 触发 postback 按钮的等待秒数（`0` = 禁用） |
| `LINE_PENDING_TEXT` | 否 | "🤔 Still thinking…" | postback 按钮旁显示的气泡文本 |
| `LINE_BUTTON_LABEL` | 否 | "Get answer" | 按钮标签 |
| `LINE_DELIVERED_TEXT` | 否 | "Already replied ✅" | 再次点击已送达按钮时的回复 |
| `LINE_INTERRUPTED_TEXT` | 否 | "Run was interrupted before completion." | 点击 `/stop` 孤立按钮时的回复 |

---

## 故障排查

**webhook 验证时提示"invalid signature"。** `Channel secret` 复制有误，或隧道重写了请求体。请先用 `curl -i https://<tunnel>/line/webhook/health` 验证 — 应返回 `{"status":"ok","platform":"line"}`。

**机器人在群组中收不到消息。** 检查 `LINE_ALLOWED_GROUPS` 是否包含对应的 `C...` 群组 ID。如需查找群组 ID，发送一条测试消息后在 `~/.hermes/logs/gateway.log` 中搜索 `LINE: rejecting unauthorized source` — 被拒绝的 source 字典中包含相关 ID。

**`send_image` 报错"LINE_PUBLIC_URL must be set"。** LINE Messaging API 不接受二进制上传 — 图片、音频和视频必须是可访问的 HTTPS URL。将 `LINE_PUBLIC_URL` 设置为隧道的公网主机名，适配器会自动从 `/line/media/<token>/<filename>` 提供文件服务。

**postback 按钮始终不出现。** 要么 LLM 的响应速度快于 `LINE_SLOW_RESPONSE_THRESHOLD`，要么其他气泡（工具进度、流式输出）已提前消耗了 reply token。参见"LLM 响应缓慢"中的抑制配置。

**"already in use by another profile"。** 同一个频道访问 token 已被另一个运行中的 Hermes profile 占用。请停止另一个 gateway，或使用独立的频道。

---

## 限制

* **气泡与长度上限。** 每个 LINE 文本气泡最多 5000 个字符。超长响应会在每次 Reply/Push 调用中按约 4500 个字符智能分块（最多 5 个气泡），并尽可能在自然边界处切分。
* **不支持原生消息编辑。** LINE 没有编辑消息的 API — 流式响应始终发送新气泡，不会编辑已有气泡。
* **不支持 Markdown 渲染。** 粗体（`**`）、斜体（`*`）、代码块和标题均以字面字符显示。适配器在发送前会将其剥离；URL 会被保留（`[label](url)` 转换为 `label (url)`）。
* **加载指示器仅限私聊。** LINE 对群组和房间拒绝 chat/loading API，因此输入指示器仅在 1:1 聊天中显示。