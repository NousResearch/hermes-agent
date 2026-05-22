---
sidebar_position: 8
sidebar_label: "SMS (Twilio)"
title: "SMS (Twilio)"
description: "通过 Twilio 把 Hermes 接到短信。"
---

# SMS 设置（Twilio）

Hermes 通过 Twilio API 提供 SMS 聊天机器人能力。用户发短信到你的 Twilio 号码后，Hermes 会回短信。

## 快速开始

1. 准备 Twilio 账号和可收发短信的号码
2. 在 `~/.hermes/.env` 中配置 `TWILIO_ACCOUNT_SID`、`TWILIO_AUTH_TOKEN`、`TWILIO_PHONE_NUMBER`
3. 配置 `SMS_WEBHOOK_URL`
4. 启动 `hermes gateway`

## 安全与限制

- 默认只允许白名单号码
- SMS 不支持 Markdown
- Webhook 需要签名校验

## 相关文档

- [技能目录](/reference/skills-catalog)
- [安全](/user-guide/security)
- [消息网关总览](/user-guide/messaging)
