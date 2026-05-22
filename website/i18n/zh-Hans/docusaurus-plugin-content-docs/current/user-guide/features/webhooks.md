---
title: "Webhook"
sidebar_label: "Webhook"
sidebar_position: 2
---

# Webhook

Hermes 可以通过 Webhook 接收外部事件，例如 GitHub 更新、CI 状态变化或监控告警，并把这些事件转交给代理会话处理。

常见用途包括：

- 触发自动化工作流。
- 生成审查、摘要或通知。
- 把外部系统事件接到 Hermes 的任务流里。

配置时通常需要注意：

- 在 `config.yaml` 或对应平台配置里启用 Webhook 接入。
- 使用签名校验来源，避免未授权请求。
- 把事件过滤和速率限制放在网关侧处理。

如果你要，我可以继续把这页扩展成完整的中文操作指南。
