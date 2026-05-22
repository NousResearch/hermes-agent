---
sidebar_position: 2
title: "架构概览"
description: "Hermes 代码库的高层架构简版。"
---

# 架构概览

Hermes 由 CLI、gateway、工具系统、插件系统和各类提供商/适配器组成。大体上，模型请求、工具调用和消息投递都通过统一的运行时协调。

## 相关文档

- [构建 Hermes 插件](/guides/build-a-hermes-plugin)
- [网关内部](/developer-guide/gateway-internals)
- [工具运行时](/developer-guide/tools-runtime)
