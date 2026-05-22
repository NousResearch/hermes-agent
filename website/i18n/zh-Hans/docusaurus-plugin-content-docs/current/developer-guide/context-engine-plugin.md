---
sidebar_position: 9
title: "上下文引擎插件"
description: "用插件替换内置 ContextCompressor 的简版指南。"
---

# 上下文引擎插件

上下文引擎插件可以替换 Hermes 的默认压缩器，实现不同的上下文管理策略。

## 关键点

- 只有一个上下文引擎可激活
- 由 `config.yaml` 的 `context.engine` 选择
- 插件需要实现 `ContextEngine` ABC

## 典型能力

- 更新 token 计数
- 决定何时压缩
- 执行压缩并返回有效消息列表
- 可选地提供工具

## 相关文档

- [Context Compression and Caching](/developer-guide/context-compression-and-caching)
- [Memory Provider Plugins](/developer-guide/memory-provider-plugin)
- [Plugins](/user-guide/features/plugins)
