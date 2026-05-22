---
sidebar_position: 2
title: "环境变量"
description: "Hermes Agent 常用环境变量参考"
---

# 环境变量参考

所有变量通常写入 `~/.hermes/.env`。也可以用 `hermes config set VAR value` 写入。

本页先提供高频键位速查，完整清单会继续补齐。

## LLM Provider

| 变量 | 说明 |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key（通用推荐） |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI-compatible endpoint 的密钥 |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint 基础地址 |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `XAI_API_KEY` | xAI API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `AWS_REGION` | Bedrock 区域（如 `us-east-1`） |
| `AWS_PROFILE` | Bedrock 使用的 AWS profile |

## 网页与工具 API

| 变量 | 说明 |
|---|---|
| `FIRECRAWL_API_KEY` | Firecrawl 搜索/提取 |
| `FIRECRAWL_API_URL` | 自托管 Firecrawl 地址 |
| `PARALLEL_API_KEY` | Parallel 搜索后端 |
| `TAVILY_API_KEY` | Tavily 搜索后端 |
| `EXA_API_KEY` | Exa 搜索后端 |
| `FAL_KEY` | 图像生成（FAL.ai） |
| `BROWSERBASE_API_KEY` | Browserbase 浏览器自动化 |
| `BROWSERBASE_PROJECT_ID` | Browserbase 项目 ID |
| `BROWSER_CDP_URL` | 本地 Chrome CDP 地址 |

## 消息平台

| 变量 | 说明 |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram 机器人 token |
| `DISCORD_BOT_TOKEN` | Discord 机器人 token |
| `SLACK_BOT_TOKEN` | Slack 机器人 token |
| `WHATSAPP_ACCESS_TOKEN` | WhatsApp token |

## 终端与运行环境

| 变量 | 说明 |
|---|---|
| `HERMES_HOME` | Hermes 数据目录（默认 `~/.hermes`） |
| `TERMINAL_ENV` | 终端后端（`local`/`docker`/`ssh` 等） |
| `TERMINAL_DOCKER_IMAGE` | Docker 后端镜像 |
| `TERMINAL_TIMEOUT` | 终端命令超时（秒） |
| `TERMINAL_CWD` | 终端工作目录 |

## Microsoft Graph Teams Meetings

以下变量用于 Teams 会议流水线与 Microsoft Graph 集成：

| 变量 | 说明 |
|---|---|
| `MSGRAPH_TENANT_ID` | Azure 租户 ID |
| `MSGRAPH_CLIENT_ID` | 应用 Client ID |
| `MSGRAPH_CLIENT_SECRET` | 应用 Client Secret |
| `MSGRAPH_SUBSCRIPTION_SECRET` | Webhook 校验密钥 |

结合指南：

- [Microsoft Graph 应用注册](/guides/microsoft-graph-app-registration)
- [Teams 适配器](/user-guide/messaging/teams)
- [MS Graph Webhook](/user-guide/messaging/msgraph-webhook)
