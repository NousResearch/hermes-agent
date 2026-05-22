---
title: "Agent 循环机制"
description: "Hermes Agent 的主对话循环、工具调用迭代与中断处理。"
---

# Agent 循环机制

本页说明 Hermes 的核心运行循环：模型响应、工具调用、结果回写、预算与中断控制。

## 核心流程

- 接收用户消息并组装上下文。
- 调用模型获得回答或工具调用指令。
- 执行工具并将结果回写到消息流。
- 在预算或停止条件触发时结束本轮。

相关文档：

- [上下文压缩与缓存](./context-compression-and-caching.md)
- [工具运行时](./tools-runtime.md)
- [系统架构](./architecture.md)
