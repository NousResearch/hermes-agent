---
sidebar_position: 1
title: "AI Providers"
description: "Как Hermes Agent подключается к провайдерам моделей и API"
---

# AI Providers

Hermes умеет работать с несколькими типами провайдеров:
- OAuth-провайдерами с готовыми входами
- API-key провайдерами
- self-hosted и совместимыми OpenAI-стилем endpoint'ами

Эта страница даёт практический обзор основных вариантов настройки. Полный каталог провайдеров и все тонкие детали остаются в английской версии документации.

## Две команды, с которых всё начинается

```bash
hermes model
hermes config set
```

`hermes model` — самый удобный способ выбрать провайдера и модель интерактивно.  
`hermes config set` полезен, когда вы хотите зафиксировать значения явно.

## Основные типы провайдеров

### OAuth / account-based

Это провайдеры, где Hermes авторизуется через браузер или существующий credential store:
- Nous Portal
- OpenAI Codex
- Anthropic через Claude Code credential store
- GitHub Copilot
- Google Gemini
- Qwen Portal
- MiniMax OAuth

### API key providers

Для этих провайдеров обычно достаточно задать ключ в `~/.hermes/.env`:
- OpenRouter
- DeepSeek
- Z.AI / GLM
- Kimi / Moonshot
- MiniMax
- Alibaba Cloud / DashScope
- NVIDIA NIM
- Hugging Face
- Arcee AI
- GMI Cloud

### Self-hosted / custom endpoints

Hermes также поддерживает:
- Ollama
- vLLM
- SGLang
- любой совместимый OpenAI-style endpoint

Здесь важнее всего три вещи:
- base URL
- API key, если он нужен
- точное имя модели

## Что обычно проверяют первым

1. Что выбран правильный provider
2. Что модель существует и доступна
3. Что контекст модели достаточен для Hermes workflows
4. Что ключи и env vars лежат в правильном месте

## Context Length Detection

`context_length` — это общий размер контекстного окна модели. Hermes использует его, чтобы:
- вовремя включать сжатие истории
- не отправлять запросы, которые превышают лимит провайдера
- корректно сравнивать длину контекста между custom provider и model config

Если автоопределение ошибается, задайте `model.context_length` явно в `config.yaml`.

## Если что-то не работает

- сначала запустите `hermes doctor`
- затем перепроверьте `hermes model`
- после этого убедитесь, что нужные ключи заданы в `~/.hermes/.env`

## Короткий совет

Если вы только начинаете, не усложняйте конфиг раньше времени. Сначала добейтесь одного стабильного провайдера и одного стабильного чата, а уже потом включайте routing, fallback и self-hosted сценарии.
