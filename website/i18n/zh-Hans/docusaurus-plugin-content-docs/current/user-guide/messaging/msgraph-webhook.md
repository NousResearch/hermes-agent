---
sidebar_position: 6
title: "MS Graph Webhook"
description: "Microsoft Graph webhook 与 Teams 会议事件入口。"
---

# MS Graph Webhook

Graph webhook 负责接收 Teams 会议事件、转成后续处理任务，并把结果交给 Hermes 的会议流水线。

## 关键配置

- 公网可访问的 webhook URL
- Graph tenant / client / secret
- `MSGRAPH_WEBHOOK_CLIENT_STATE`
- 订阅续期计划

## 相关文档

- [Microsoft Graph 应用注册](/guides/microsoft-graph-app-registration)
- [运维 Teams 会议流水线](/guides/operate-teams-meeting-pipeline)
- [环境变量参考](/reference/environment-variables)
