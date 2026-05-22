---
sidebar_position: 5
title: "添加提供商"
description: "为 Hermes 添加新的推理提供商的简版指南。"
---

# 添加提供商

Hermes 已经支持很多 OpenAI-compatible endpoint。只有当你需要第一方体验、专用认证或非标准 API 形状时，才值得新增内置 provider。

## 先选实现路径

- OpenAI-compatible provider: 通常只需补 auth、模型目录和 runtime 解析
- Native provider: 需要额外的 adapter 和 `api_mode` 分支

## 必做文件

- `hermes_cli/auth.py`
- `hermes_cli/models.py`
- `hermes_cli/runtime_provider.py`
- `hermes_cli/main.py`
- `agent/auxiliary_client.py`
- `agent/model_metadata.py`
- 相关测试与文档

## 快速路径

如果只是标准 API key + OpenAI-compatible endpoint，优先考虑 provider plugin，而不是改核心代码。

## 相关文档

- [Provider Runtime Resolution](/developer-guide/provider-runtime)
- [Model Provider Plugin](/developer-guide/model-provider-plugin)
- [AI Providers](/integrations/providers)
