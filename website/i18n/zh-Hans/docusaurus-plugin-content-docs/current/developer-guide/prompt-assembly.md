---
title: "Prompt 组装流程"
description: "Hermes 如何拼装系统提示、用户消息、记忆与工具上下文。"
---

# Prompt 组装流程

Hermes 在每轮调用前会按固定顺序组装提示内容，以兼顾稳定性、可缓存性与成本。

## 组装顺序

- 系统提示与角色约束
- 会话历史与压缩上下文
- 记忆/技能注入
- 工具定义与可用能力说明

相关文档：

- [上下文压缩与缓存](./context-compression-and-caching.md)
- [网关内部机制](./gateway-internals.md)
