---
sidebar_position: 8
title: "记忆提供商插件"
description: "为 Hermes 添加持久化记忆后端的简版指南。"
---

# 记忆提供商插件

记忆提供商插件为 Hermes 提供跨会话持久化知识，作为内置 `MEMORY.md` / `USER.md` 的补充。

## 要点

- 实现 `MemoryProvider` ABC
- 通过 `ctx.register_memory_provider()` 注册
- 存储路径必须使用 profile-scoped 的 `hermes_home`

## 常见方法

- `is_available()`
- `initialize()`
- `get_tool_schemas()`
- `handle_tool_call()`
- `get_config_schema()`
- `save_config()`

## 相关文档

- [Context Engine Plugins](/developer-guide/context-engine-plugin)
- [Plugins](/user-guide/features/plugins)
