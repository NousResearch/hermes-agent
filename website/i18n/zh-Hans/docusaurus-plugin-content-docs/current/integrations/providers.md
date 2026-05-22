---
title: "AI 提供商"
sidebar_label: "AI 提供商"
sidebar_position: 1
---

# AI 提供商

本页汇总 Hermes 的推理提供商接入方式。你至少需要配置一个 provider 才能使用 Hermes。

## 推荐入口

```bash
hermes model
```

`hermes model` 会引导你完成 provider 认证、模型选择和必要配置。

## 常见提供商

- Nous Portal（OAuth）
- OpenAI Codex（OAuth）
- GitHub Copilot（OAuth / token）
- Anthropic（OAuth 或 API Key）
- OpenRouter（`OPENROUTER_API_KEY`）
- Gemini（`GOOGLE_API_KEY`）
- Google Gemini OAuth（`google-gemini-cli`）
- DeepSeek（`DEEPSEEK_API_KEY`）
- MiniMax / MiniMax CN
- AWS Bedrock
- 自定义 OpenAI-compatible endpoint

## 通过环境变量配置（示例）

在 `~/.hermes/.env` 中设置:

```bash
OPENROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

## 通过 config.yaml 固定 provider（示例）

```yaml
model:
  provider: openrouter
  default: openai/gpt-5
```

## 相关专题

- [Google Gemini 指南](/guides/google-gemini)
- [MiniMax OAuth 指南](/guides/minimax-oauth)
- [AWS Bedrock 指南](/guides/aws-bedrock)
- [提供商路由](/user-guide/features/provider-routing)
- [备用提供商](/user-guide/features/fallback-providers)
- [配置](/user-guide/configuration)

## 备用提供商 {#fallback-providers}

当你希望在主模型不可用时自动切换到其他提供商，可以在这里配置回退链、默认模型和路由优先级。

### OpenRouter Pareto Code Router {#openrouter-pareto-code-router}

OpenRouter 的 Pareto Code Router 可用于在代码任务中选择更合适的模型。该锚点供配置页与示例文档引用。

<a id="fallback-model"></a>

<!-- fallback-model: 这个锚点用于引用备用模型/提供商配置说明 -->
