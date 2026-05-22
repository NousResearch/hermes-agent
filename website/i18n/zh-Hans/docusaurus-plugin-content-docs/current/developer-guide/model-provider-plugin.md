---
sidebar_position: 10
title: "模型提供商插件"
description: "为 Hermes 添加模型提供商插件的简版指南。"
---

# 模型提供商插件

模型提供商插件会把一个 provider 注册到 Hermes 的提供商注册表中。它适合 OpenAI-compatible 或 OAuth 型第三方模型源。

## 关键点

- 用 `register_provider()` 注册
- 在 `plugins/model-providers/<name>/` 下放置插件
- 用户插件可以覆盖内置插件

## 你通常需要补的内容

- 认证元数据
- 模型目录与别名
- runtime 解析
- 测试与文档

## 相关文档

- [Provider Runtime Resolution](/developer-guide/provider-runtime)
- [Adding Providers](/developer-guide/adding-providers)
- [AI Providers](/integrations/providers)
