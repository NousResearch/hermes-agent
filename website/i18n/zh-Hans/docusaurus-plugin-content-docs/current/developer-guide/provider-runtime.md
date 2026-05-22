---
title: "Provider 运行时"
description: "模型提供商解析、路由、回退与运行时行为说明。"
---

# Provider 运行时

本页解释 Hermes 在运行时如何选择与调用模型提供商，并处理回退与异常。

## 关键点

- 提供商解析与模型路由
- 自定义提供商与插件提供商
- 失败重试、超时与回退链

相关文档：

- [添加提供商](./adding-providers.md)
- [模型提供商插件](./model-provider-plugin.md)
- [Cron 内部机制](./cron-internals.md)
- [上下文压缩与缓存](./context-compression-and-caching.md)
