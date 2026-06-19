---
sidebar_position: 5
title: "WhatsApp"
description: "通过内置 Baileys 桥接将 Hermes Agent 设置为 WhatsApp 机器人"
---

# WhatsApp 配置

Hermes 通过基于 **Baileys** 的内置桥接连接到 WhatsApp。其工作原理是模拟 WhatsApp Web 会话——**而非**通过官方 WhatsApp Business API。无需 Meta 开发者账号或 Business 认证。

:::warning 非官方 API — 封号风险
WhatsApp **不**官方支持 Business API 以外的第三方机器人。使用第三方桥接存在账号受限的小概率风险。为降低风险：
- **为机器人使用专用手机号**（而非个人号码）
- **不要发送批量/垃圾消息**——保持对话式使用
- **不要向未主动发消息的用户自动发送外发消息**
:::

:::warning WhatsApp Web 协议更新
WhatsApp 会定期更新其 Web 协议，这可能导致第三方桥接暂时失效。
发生这种情况时，Hermes 会更新桥接依赖。如果机器人在 WhatsApp 更新后停止工作，
请拉取最新版 Hermes 并重新配对。
:::

## 两种模式

| 模式 | 工作方式 | 适用场景 |
|------|---------|---------|
| **独立机器人号码**（推荐） | 为机器人专用一个手机号，用户直接向该号码发消息。 | 体验简洁、多用户、封号风险低 |
| **个人自聊** | 使用你自己的 WhatsApp，向自己发消息与 Agent 对话。 | 快速配置、单用户、测试用途 |

---

## 前置条件

- **Node.js v18+** 和 **npm**——WhatsApp 桥接作为 Node.js 进程运行
- **已安装 WhatsApp 的手机**（用于扫描二维码）

与旧版浏览器驱动的桥接不同，当前基于 Baileys 的桥接**不**需要本地 Chromium 或 Puppeteer 依赖栈。

---

## 第一步：运行配置向导

```bash
hermes whatsapp
```

向导将：

1. 询问你想要哪种模式（**bot** 或 **self-chat**）
2. 如有需要，安装桥接依赖
3. 在终端中显示**二维码**
4. 等待你扫描

**扫描二维码的步骤：**

1. 在手机上打开 WhatsApp
2. 进入**设置 → 已关联设备**
3. 点击**关联设备**
4. 将摄像头对准终端中的二维码

配对成功后，向导确认连接并退出。你的会话将自动保存。

:::tip
如果二维码显示乱码，请确保终端宽度至少为 60 列且支持 Unicode。
也可以尝试换用其他终端模拟器。
:::

---

## 第二步：获取第二个手机号（机器人模式）

机器人模式需要一个尚未注册 WhatsApp 的手机号。有三种选择：

| 选项 | 费用 | 说明 |
|------|------|------|
| **Google Voice** | 免费 | 仅限美国。在 [voice.google.com](https://voice.google.com) 获取号码，通过 Google Voice 应用以短信验证 WhatsApp。 |
| **预付费 SIM 卡** | 一次性 $5–15 | 任意运营商。激活后验证 WhatsApp，SIM 卡可放置不用。号码需保持有效（每 90 天拨打一次电话）。 |
| **VoIP 服务** | 免费–$5/月 | TextNow、TextFree 等。部分 VoIP 号码被 WhatsApp 屏蔽——如第一个不可用，可多试几个。 |

获取号码后：

1. 在手机上安装 WhatsApp（或使用支持双 SIM 的 WhatsApp Business 应用）
2. 用新号码注册 WhatsApp
3. 运行 `hermes whatsapp` 并从该 WhatsApp 账号扫描二维码

---

## 第三步：配置 Hermes

在 `~/.hermes/.env` 文件中添加以下内容：

```bash
# 必填
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" 或 "self-chat"

# 访问控制——选择以下其中一项：
WHATSAPP_ALLOWED_USERS=15551234567         # 逗号分隔的手机号（含国家代码，不含 +）
# WHATSAPP_ALLOWED_USERS=*                 # 或使用 * 允许所有人
# WHATSAPP_ALLOW_ALL_USERS=true            # 或设置此标志（效果等同于 *）
```

:::tip 允许所有人的简写
将 `WHATSAPP_ALLOWED_USERS=*` 设置为允许**所有**发送者（等同于 `WHATSAPP_ALLOW_ALL_USERS=true`）。
这与 [Signal 群组白名单](/reference/environment-variables) 保持一致。
如需使用配对流程，请移除这两个变量，改用
[私信配对系统](/user-guide/security#dm-pairing-system)。
:::

在 `~/.hermes/config.yaml` 中可选的行为设置：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `unauthorized_dm_behavior: pair` 是全局默认值。未知私信发送者将收到配对码。
- `whatsapp.unauthorized_dm_behavior: ignore` 使 WhatsApp 对未授权私信保持静默，通常更适合私人号码。

然后启动 gateway（网关）：

```bash
hermes gateway              # 前台运行
hermes gateway install      # 安装为用户服务
sudo hermes gateway install --system   # 仅 Linux：开机启动系统服务
```

Gateway 会使用已保存的会话自动启动 WhatsApp 桥接。

---

## 会话持久化

Baileys 桥接将会话保存在 `~/.hermes/platforms/whatsapp/session` 目录下。这意味着：

- **会话在重启后仍然有效**——无需每次重新扫描二维码
- 会话数据包含加密密钥和设备凭证
- **请勿共享或提交此会话目录**——它可授予对 WhatsApp 账号的完整访问权限

---

## 重新配对

如果会话中断（手机重置、WhatsApp 更新、手动取消关联），你将在 gateway 日志中看到连接错误。修复方法：

```bash
hermes whatsapp
```

这将生成新的二维码。重新扫描后会话即恢复。Gateway 会通过重连逻辑自动处理**临时**断线（网络抖动、手机短暂离线）。

---

## 语音消息

Hermes 支持 WhatsApp 上的语音功能：

- **接收：** 语音消息（`.ogg` opus 格式）会使用已配置的 STT 提供商自动转录：本地 `faster-whisper`、Groq Whisper（`GROQ_API_KEY`）或 OpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`）
- **发送：** TTS 响应以 MP3 音频文件附件形式发送
- Agent 响应默认以"⚕ **Hermes Agent**"为前缀。可在 `config.yaml` 中自定义或禁用：

```yaml
# ~/.hermes/config.yaml
whatsapp:
  reply_prefix: ""                          # 空字符串禁用标题
  # reply_prefix: "🤖 *My Bot*\n──────\n"  # 自定义前缀（支持 \n 换行）
```

---

## 消息格式与投递

WhatsApp 支持**流式（渐进式）响应**——机器人在 AI 生成文本时实时编辑消息，与 Discord 和 Telegram 一样。在内部，WhatsApp 被归类为 TIER_MEDIUM 平台（投递能力中等）。

### 分块

长响应会自动按每块 **4,096 个字符**拆分为多条消息（WhatsApp 的实际显示上限）。无需任何配置——gateway 会自动处理拆分并按顺序发送各块。

### WhatsApp 兼容 Markdown

AI 响应中的标准 Markdown 会自动转换为 WhatsApp 的原生格式：

| Markdown | WhatsApp | 渲染效果 |
|----------|----------|---------|
| `**bold**` | `*bold*` | **粗体** |
| `~~strikethrough~~` | `~strikethrough~` | ~~删除线~~ |
| `# Heading` | `*Heading*` | 粗体文本（无原生标题） |
| `[link text](url)` | `link text (url)` | 内联 URL |

代码块和内联代码保持原样，因为 WhatsApp 原生支持三反引号格式。

### 工具进度

当 Agent 调用工具（网页搜索、文件操作等）时，WhatsApp 会显示实时进度指示器，显示正在运行的工具。此功能默认启用，无需配置。

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| **二维码无法扫描** | 确保终端宽度足够（60 列以上）。尝试换用其他终端。确保从正确的 WhatsApp 账号（机器人号码，而非个人号码）扫描。 |
| **二维码过期** | 二维码约每 20 秒刷新一次。如果超时，重新运行 `hermes whatsapp`。 |
| **会话未持久化** | 检查 `~/.hermes/platforms/whatsapp/session` 是否存在且可写。如在容器中运行，请将其挂载为持久卷。 |
| **意外退出登录** | WhatsApp 会在长时间不活跃后取消关联设备。保持手机开机并连接网络，如有需要使用 `hermes whatsapp` 重新配对。 |
| **桥接崩溃或重连循环** | 重启 gateway，更新 Hermes，如会话因 WhatsApp 协议变更而失效则重新配对。 |
| **WhatsApp 更新后机器人停止工作** | 更新 Hermes 以获取最新桥接版本，然后重新配对。 |
| **macOS："Node.js not installed"但终端中 node 可用** | launchd 服务不继承你的 shell PATH。运行 `hermes gateway install` 将当前 PATH 重新快照到 plist 中，然后运行 `hermes gateway start`。详见 [Gateway 服务文档](./index.md#macos-launchd)。 |
| **未收到消息** | 确认 `WHATSAPP_ALLOWED_USERS` 包含发送者号码（含国家代码，不含 `+` 或空格），或将其设为 `*` 允许所有人。在 `.env` 中设置 `WHATSAPP_DEBUG=true` 并重启 gateway，可在 `bridge.log` 中查看原始消息事件。 |
| **机器人向陌生人回复配对码** | 如需对未授权私信静默处理，在 `~/.hermes/config.yaml` 中设置 `whatsapp.unauthorized_dm_behavior: ignore`。 |

---

## 安全

:::warning
**上线前请配置访问控制。** 在 `WHATSAPP_ALLOWED_USERS` 中填写具体手机号（含国家代码，不含 `+`），
使用 `*` 允许所有人，或设置 `WHATSAPP_ALLOW_ALL_USERS=true`。
若未配置上述任何一项，gateway 将**拒绝所有传入消息**作为安全措施。
:::

默认情况下，未授权私信仍会收到配对码回复。如果你希望私人 WhatsApp 号码对陌生人完全静默，请设置：

```yaml
whatsapp:
  unauthorized_dm_behavior: ignore
```

- `~/.hermes/platforms/whatsapp/session` 目录包含完整会话凭证——请像保护密码一样保护它
- 设置文件权限：`chmod 700 ~/.hermes/platforms/whatsapp/session`
- 为机器人使用**专用手机号**，将风险与个人账号隔离
- 如怀疑账号被入侵，在 WhatsApp → 设置 → 已关联设备中取消关联该设备
- 日志中的手机号已部分脱敏，但请审查你的日志保留策略