---
sidebar_position: 5
title: "WhatsApp"
description: "通过内置 Baileys 桥将 Hermes Agent 作为 WhatsApp 机器人连接"
---

# WhatsApp 设置

Hermes 通过内置的 Baileys 桥连接 WhatsApp，原理是模拟 WhatsApp Web 会话——**不是**使用官方 WhatsApp Business API。无需 Meta 开发者账号或 Business 验证。

:::warning 非官方 API — 存在封号风险
WhatsApp 并不官方支持除 Business API 之外的第三方机器人。使用第三方桥接存在小概率被限制账号的风险。为降低风险：
- **使用专用电话号码**（不要用个人号码）
- **勿发送批量/垃圾消息**——保持会话式使用
- **不要向未先发送消息的人自动发起外发消息**
:::

## 两种模式

| 模式 | 工作方式 | 适用场景 |
|------|----------|----------|
| **独立机器人号码**（推荐） | 为机器人专设号码，用户直接私信该号码 | 干净 UX、多用户、较低封号风险 |
| **个人自聊模式** | 使用你的个人 WhatsApp 号码，与自己对话 | 适合快速测试、单用户 |

## 前提条件

- **Node.js v18+** 与 **npm** —— WhatsApp 桥为 Node.js 进程运行
- **一部已装 WhatsApp 的手机**（用于扫码配对）

## 第 1 步：运行设置向导

```bash
hermes whatsapp
```

向导会：

1. 询问使用的模式（bot 或 self-chat）
2. 如需则安装桥接依赖
3. 在终端显示二维码
4. 等待扫码

扫码步骤：

1. 打开手机 WhatsApp → 设置 → 已链接设备（Linked Devices）
2. 点击“链接设备”并扫码终端二维码

扫码成功后向导确认并退出，会话会自动保存。

:::tip
如二维码显示乱码，请确保终端至少 60 列宽并支持 Unicode，或换个终端试试。
:::

## 第 2 步：为机器人模式获取第二个号码（仅 bot 模式）

列出几种常见方案（Google Voice、预付费 SIM、VoIP）。完成后在手机上注册该号码并用该号码扫码配对。

## 第 3 步：配置 Hermes

将以下写入 `~/.hermes/.env`：

```bash
# 必需
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" 或 "self-chat"

# 访问控制 —— 三选一
WHATSAPP_ALLOWED_USERS=15551234567         # 逗号分隔的电话号码（含国家码，不带 +）
# WHATSAPP_ALLOWED_USERS=*                 # 或使用 * 允许所有人
# WHATSAPP_ALLOW_ALL_USERS=true            # 或设置此标志
```

可选的 `config.yaml` 设置示例：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

然后启动网关：

```bash
hermes gateway
```

网关会自动使用已保存的会话启动 WhatsApp 桥。

## 会话持久化与重新配对等说明

（文档与英文一致，已按需翻译关键配置与安全提示）

## 安全

:::warning
在投入生产前请先配置访问控制。将 `WHATSAPP_ALLOWED_USERS` 设置为明确号码（含国家码、不带 `+`）或 `*` 允许所有人，或者设置 `WHATSAPP_ALLOW_ALL_USERS=true`。否则，网关会默认**拒绝**所有来信以保证安全。
:::

请务必保护 `~/.hermes/platforms/whatsapp/session` 中的会话凭证（`chmod 700`），并在怀疑泄露时重新配对。
