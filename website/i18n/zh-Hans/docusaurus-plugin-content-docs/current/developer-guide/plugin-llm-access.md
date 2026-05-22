---
sidebar_position: 11
title: "插件 LLM 访问"
description: "在插件内通过 ctx.llm 调用模型的简版说明。"
---

# 插件 LLM 访问

`ctx.llm` 让插件可以在不进入主对话流的情况下发起一次模型调用。适合 hook 重写、摘要、分类和预筛选。

## 典型接口

- `complete()`
- `complete_structured()`
- `acomplete()`
- `acomplete_structured()`

## 适用场景

- 重写错误信息
- 摘要长文本
- 结构化抽取
- 轻量分类

## 相关文档

- [Build a Hermes Plugin](/guides/build-a-hermes-plugin)
- [Provider Runtime Resolution](/developer-guide/provider-runtime)
