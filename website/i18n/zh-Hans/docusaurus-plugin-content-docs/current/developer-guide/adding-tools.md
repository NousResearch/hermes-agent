---
sidebar_position: 1
title: "添加工具"
description: "为 Hermes 添加新工具的简版说明。"
---

# 添加工具

Hermes 工具通过 `tools/registry.py` 注册，并由 `model_tools.py` 自动发现。

## 要点

- 新工具通常放在 `tools/<name>.py`
- 顶层调用 `registry.register()` 才会被发现
- 工具必须返回 JSON 字符串

## 相关文档

- [插件系统](/user-guide/features/plugins)
- [构建 Hermes 插件](/guides/build-a-hermes-plugin)
- [工具与工具集](/user-guide/features/tools)
