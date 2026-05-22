---
sidebar_position: 11
title: "图像生成提供商插件"
description: "为 Hermes 图像生成工具添加后端提供商的简版指南。"
---

# 图像生成提供商插件

图像生成插件为 `image_generate` 工具提供后端。你可以接入云服务、本地 ComfyUI，或替换已有实现。

## 核心概念

- 插件通过 `ctx.register_image_gen_provider()` 注册
- 激活 provider 由 `image_gen.provider` 决定
- `generate()` 返回成功或错误结果

## 目录结构

```text
plugins/image_gen/my-backend/
├── __init__.py
└── plugin.yaml
```

## 相关文档

- [Model Provider Plugin](/developer-guide/model-provider-plugin)
- [Memory Provider Plugin](/developer-guide/memory-provider-plugin)
- [Build a Hermes Plugin](/guides/build-a-hermes-plugin)
