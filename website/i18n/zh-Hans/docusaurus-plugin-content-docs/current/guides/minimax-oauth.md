---
sidebar_position: 15
title: "MiniMax OAuth"
description: "通过浏览器 OAuth 登录 MiniMax，在 Hermes 中使用 MiniMax-M2.7 模型，无需 API Key。"
---

# MiniMax OAuth

Hermes 支持通过浏览器 OAuth 登录 MiniMax，登录后可自动刷新会话，无需手动管理 API Key。

## 快速开始

```bash
hermes model
# 选择 MiniMax (OAuth)
```

或手动触发登录:

```bash
hermes auth add minimax-oauth
```

无图形环境可用:

```bash
hermes auth add minimax-oauth --no-browser
```

## 常用模型

- `MiniMax-M2.7`
- `MiniMax-M2.7-highspeed`

## 常见问题

- 登录超时: 重新执行 `hermes auth add minimax-oauth`
- token 失效: 重新登录即可
- 运行时报“未登录”: 先完成 OAuth 登录，再切模型

## 相关文档

- [AI 提供商](/integrations/providers)
- [环境变量参考](/reference/environment-variables)
- [配置](/user-guide/configuration)
