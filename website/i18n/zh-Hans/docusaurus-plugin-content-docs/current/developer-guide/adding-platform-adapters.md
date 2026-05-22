---
sidebar_position: 9
title: "添加平台适配器"
description: "为 Hermes 网关新增消息平台适配器的简版指南。"
---

# 添加平台适配器

Hermes 的消息平台通过 `BasePlatformAdapter` 接入。对于社区或第三方平台，优先考虑插件方式；如果要做内置支持，则需要同时更新代码、配置、文档和测试。

## 两种路径

- 插件路径: 推荐用于社区平台，减少核心改动
- 内置路径: 适用于需要深度整合到 gateway 的官方平台

## <a id="step-by-step-checklist"></a>分步检查清单

1. 继承 `gateway/platforms/base.py` 中的 `BasePlatformAdapter`
2. 实现 `connect()`、`disconnect()`、`send()` 等核心方法
3. 处理入站消息并交给 gateway runner
4. 为平台补充配置与环境变量
5. 在插件或内置注册路径中暴露该平台
6. 补充 `hermes gateway setup`、`hermes status`、cron 投递等集成点

## 插件路径（推荐）

插件可放入 `~/.hermes/plugins/`，通过 `ctx.register_platform()` 注册。这样通常不需要改 Hermes 核心文件。

## 内置路径

如果你在做内置平台适配器，通常还要更新:

- gateway 平台枚举
- 配置解析
- 状态/健康检查
- 文档与测试

## 相关文档

- [网关内部](/developer-guide/gateway-internals)
- [添加提供商](/developer-guide/adding-providers)
- [插件系统](/user-guide/features/plugins)
