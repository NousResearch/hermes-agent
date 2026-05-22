---
sidebar_position: 3
title: "上下文压缩与缓存"
description: "解释 Hermes 的上下文压缩、缓存与相关插件接口。"
---

# 上下文压缩与缓存

Hermes 会在会话变长时压缩上下文，并配合 prompt caching 降低 token 成本。上下文引擎插件可以替换默认压缩器。

## 相关文档

- [上下文引擎插件](/developer-guide/context-engine-plugin)
- [安全](/user-guide/security)
- [配置](/user-guide/configuration)
