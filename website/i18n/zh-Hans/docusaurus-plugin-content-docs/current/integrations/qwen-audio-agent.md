---
sidebar_position: 5
title: "qwen-audio-agent 集成"
description: "通过 ACP 用实时全双工语音免提驱动 Hermes —— 插嘴打断、本地唤醒词、后台任务结果读回"
---

# qwen-audio-agent 集成

[qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent) 是面向 ACP Agent 的开源实时全双工语音前台。它通过 `hermes acp` 以 stdio 方式启动 Hermes，把会话变成免提语音对话 —— 提供 macOS 桌面悬浮球、终端 TUI 与浏览器三种形态。

## 带来的能力

| 能力 | 行为 |
|---|---|
| **全双工语音 + 插嘴打断** | Hermes 说话时随时开口，打断会立即停止生成 |
| **本地唤醒词** | 唤醒词检测完全在本地运行（sherpa-onnx），激活不出本机 |
| **后台任务结果读回** | 长任务异步执行，完成后结果自然回到当前对话 |
| **语音权限确认** | 用语音批准或拒绝 Hermes 的权限请求 |

## 配置步骤

1. 安装 Hermes 并完成认证：

   ```bash
   hermes setup --portal
   ```

2. 安装 qwen-audio-agent，自动补齐缺失组件：

   ```bash
   npm install -g qwen-audio-agent
   qwenaudio install hermes
   ```

3. 检查并启动：

   ```bash
   qwenaudio setup --backend hermes
   qwenaudio --backend hermes
   ```

Hermes 是 qwen-audio-agent 的原生支持后台，你的 Hermes 配置 —— 模型、工具、MCP 服务器、技能 —— 原样复用。
