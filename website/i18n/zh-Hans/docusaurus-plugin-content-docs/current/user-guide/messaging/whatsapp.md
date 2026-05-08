---
sidebar_position: 5
title: "WhatsApp"
description: "通过内置 Baileys 桥接将 Hermes Agent 设置为 WhatsApp 机器人"
---

# WhatsApp 设置

Hermes 通过基于 **Baileys** 的内置桥接连接到 WhatsApp。这通过模拟 WhatsApp Web 会话工作 — **不是**通过官方 WhatsApp Business API。无需 Meta 开发者账户或 Business 验证。

:::warning 非官方 API — 封号风险
WhatsApp **不**官方支持 Business API 之外的第三方机器人。使用第三方桥接有账户限制的小风险。最小化风险：
- **使用专用电话号码** 用于机器人（不是你的个人号码）
- **不要发送批量/垃圾消息** — 保持对话使用
- **不要自动化向未先发消息的人发送出站消息**
:::

:::warning WhatsApp Web 协议更新
WhatsApp 定期更新其 Web 协议，这可能暂时破坏第三方桥接的兼容性。当这发生时，Hermes 会更新桥接依赖。如果机器人 WhatsApp 更新后停止工作，拉取最新 Hermes 版本并重新配对。
:::

## 两种模式

| 模式 | 工作方式 | 最适合 |
|------|-------------|---------|
| **独立机器人号码**（推荐） | 将电话号码专用给机器人。人们直接向该号码发消息。 | 干净 UX、多个用户、较低封号风险 |
| **个人自聊** | 使用你自己的 WhatsApp。你向自己发消息与智能体对话。 | 快速设置、单用户、测试 |

## 前置条件

- **Node.js v18+** 和 **npm** — WhatsApp 桥接作为 Node.js 进程运行
- **装有 WhatsApp 的手机**（用于扫描二维码）

## 步骤 1：运行设置向导

```bash
hermes whatsapp
```

向导会：
1. 询问你想要的模式（**bot** 或 **self-chat**）
2. 如需要安装桥接依赖
3. 在终端显示 **二维码**
4. 等待你扫描

**扫描二维码：**
1. 在手机上打开 WhatsApp
2. 进入 **Settings → Linked Devices**
3. 点击 **Link a Device**
4. 将相机对准终端二维码

配对后，向导确认连接并退出。会话自动保存。

## 步骤 2：获取第二个电话号码（机器人模式）

对于机器人模式，你需要一个未在 WhatsApp 注册的电话号码。三种选择：

| 选项 | 成本 | 说明 |
|--------|------|-------|
| **Google Voice** | 免费 | 仅美国。在 [voice.google.com](https://voice.google.com) 获取号码。通过 Google Voice 应用通过 SMS 验证 WhatsApp。 |
| **预付费 SIM** | $5-15 一次性 | 任何运营商。激活，验证 WhatsApp，然后 SIM 可以放在抽屉里。号码必须保持活跃（每 90 天打一次电话）。 |
| **VoIP 服务** | 免费-$5/月 | TextNow、TextFree 或类似。一些 VoIP 号码被 WhatsApp 阻止 — 如果第一个不行，试试几个。 |

## 步骤 3：配置 Hermes

添加到 `~/.hermes/.env`：

```bash
# 必需
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" 或 "self-chat"

# 访问控制 — 选择其一：
WHATSAPP_ALLOWED_USERS=15551234567         # 逗号分隔的电话号码（带国家代码，无 +）
# WHATSAPP_ALLOWED_USERS=*                 # 或使用 * 允许所有人
# WHATSAPP_ALLOW_ALL_USERS=true            # 或设置此标志（与 * 相同效果）
```

:::tip 允许所有人简写
设置 `WHATSAPP_ALLOWED_USERS=*` 允许**所有**发送者（相当于 `WHATSAPP_ALLOW_ALL_USERS=true`）。
:::

然后启动 gateway：

```bash
hermes gateway
```

---

## 会话持久化

Baileys 桥接将会话保存在 `~/.hermes/platforms/whatsapp/session`。这意味着：

- **会话在重启后保留** — 你不需要每次重新扫描二维码
- 会话数据包含加密密钥和设备凭证
- **不要分享或提交此会话目录** — 它授予 WhatsApp 账户的完全访问权限

## 重新配对

如果会话损坏（手机重置、WhatsApp 更新、手动取消链接），你会在 gateway 日志中看到连接错误。修复：

```bash
hermes whatsapp
```

这会生成新的二维码。再次扫描，会话重新建立。Gateway 自动处理**临时**断开（网络抖动、手机暂时离线）。

## 语音消息

Hermes 支持 WhatsApp 语音：

- **传入：** 语音消息（`.ogg` opus）使用配置的 STT 提供商自动转录：本地 `faster-whisper`、Groq Whisper（`GROQ_API_KEY`）或 OpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`）
- **传出：** TTS 响应作为 MP3 音频文件附件发送

---

## 消息格式和投递

WhatsApp 支持**流式（渐进）响应** — 机器人在 AI 生成文本时实时编辑其消息，类似 Discord 和 Telegram。

### 分块

长响应自动分割为每块 **4,096 字符** 的多条消息（WhatsApp 的实际显示限制）。无需配置任何内容 — gateway 处理分割并顺序发送块。

### WhatsApp 兼容 Markdown

AI 响应中的标准 Markdown 自动转换为 WhatsApp 原生格式：

| Markdown | WhatsApp | 渲染为 |
|----------|----------|------------|
| `**bold**` | `*bold*` | **粗体** |
| `~~strikethrough~~` | `~strikethrough~` | ~~删除线~~ |
| `# Heading` | `*Heading*` | 粗体文本 |
| `[link text](url)` | `link text (url)` | 内联 URL |

代码块和内联代码保持原样，因为 WhatsApp 原生支持三个反引号格式。

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| **二维码不扫描** | 确保终端足够宽（60+ 列）。尝试不同的终端。确保从正确的 WhatsApp 账户扫描（机器人号码，不是个人）。 |
| **二维码过期** | 二维码每 ~20 秒刷新。如果超时，重启 `hermes whatsapp`。 |
| **会话不持久化** | 检查 `~/.hermes/platforms/whatsapp/session` 存在且可写。如果是容器化的，挂载为持久卷。 |
| **意外登出** | WhatsApp 在长期不活动后取消链接设备。保持手机开机并连接网络，然后在需要时用 `hermes whatsapp` 重新配对。 |
| **WhatsApp 更新后机器人停止工作** | 更新 Hermes 获取最新桥接版本，然后重新配对。 |
| **消息未收到** | 验证 `WHATSAPP_ALLOWED_USERS` 包含发送者号码（带国家代码，无 `+` 或空格），或设为 `*` 允许所有人。 |

---

## 安全

:::warning
**上线前配置访问控制。** 设置 `WHATSAPP_ALLOWED_USERS` 包含特定电话号码（带国家代码，无 `+`），使用 `*` 允许所有人，或设置 `WHATSAPP_ALLOW_ALL_USERS=true`。没有这些，gateway **拒绝所有传入消息** 作为安全措施。
:::

- `~/.hermes/platforms/whatsapp/session` 目录包含完整会话凭证 — 像密码一样保护它
- 设置文件权限：`chmod 700 ~/.hermes/platforms/whatsapp/session`
- 使用**专用电话号码**用于机器人以将风险与你的个人账户隔离
- 如果怀疑被入侵，从 WhatsApp → Settings → Linked Devices 取消链接设备
