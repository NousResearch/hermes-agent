---
slug: /
sidebar_position: 0
title: "Документация Hermes Agent"
description: "Самоулучшающийся AI-агент от Nous Research. Встроенный цикл обучения, который создаёт навыки на основе опыта, улучшает их в процессе работы и запоминает знания между сеансами."
hide_table_of_contents: true
displayed_sidebar: docs
---

# Hermes Agent

Самоулучшающийся AI-агент от [Nous Research](https://nousresearch.com). Это единственный агент со встроенным циклом обучения: он создаёт навыки на основе опыта, улучшает их в процессе работы, мягко подталкивает сам себя к сохранению знаний и постепенно формирует более глубокую модель того, кто вы есть, между сеансами.

<div style={{display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap'}}>
  <a href="/docs/getting-started/installation" style={{display: 'inline-block', padding: '0.6rem 1.2rem', backgroundColor: '#FFD700', color: '#07070d', borderRadius: '8px', fontWeight: 600, textDecoration: 'none'}}>Начать работу →</a>
  <a href="https://github.com/NousResearch/hermes-agent" style={{display: 'inline-block', padding: '0.6rem 1.2rem', border: '1px solid rgba(255,215,0,0.2)', borderRadius: '8px', textDecoration: 'none'}}>Смотреть на GitHub</a>
</div>

## Что такое Hermes Agent?

Это не кодовый copilot, привязанный к IDE, и не чат-обёртка вокруг одного API. Это **автономный агент**, который становится сильнее, чем дольше работает. Он живёт там, где вы его разместите: на VPS за $5, в GPU-кластере или в serverless-инфраструктуре вроде Daytona и Modal, которая почти ничего не стоит в простое. Вы можете общаться с ним в Telegram, пока он работает на облачной VM, к которой вам даже не нужно подключаться по SSH. Он не привязан к вашему ноутбуку.

## Быстрые ссылки

| | |
|---|---|
| 🚀 **[Установка](/docs/getting-started/installation)** | Установка за 60 секунд на Linux, macOS или WSL2 |
| 📖 **[Quickstart](/docs/getting-started/quickstart)** | Ваш первый диалог и ключевые возможности для проверки |
| 🗺️ **[Путь обучения](/docs/getting-started/learning-path)** | Подберите нужные документы под ваш уровень |
| ⚙️ **[Конфигурация](/docs/user-guide/configuration)** | config-файл, провайдеры, модели и параметры |
| 💬 **[Шлюз сообщений](/docs/user-guide/messaging)** | Настройка Telegram, Discord, Slack, WhatsApp, Teams и других каналов |
| 🔧 **[Инструменты и toolsets](/docs/user-guide/features/tools)** | 68 встроенных инструментов и способы их настройки |
| 🧠 **[Система памяти](/docs/user-guide/features/memory)** | Постоянная память, которая растёт вместе с сеансами |
| 📚 **[Система навыков](/docs/user-guide/features/skills)** | Процедурная память, которую агент создаёт и переиспользует |
| 🔌 **[Интеграция MCP](/docs/user-guide/features/mcp)** | Подключение к MCP-серверам, фильтрация инструментов и безопасное расширение Hermes |
| 🧭 **[Использование MCP с Hermes](/docs/guides/use-mcp-with-hermes)** | Практические сценарии, примеры и пошаговые инструкции |
| 🎙️ **[Режим голоса](/docs/user-guide/features/voice-mode)** | Голосовое взаимодействие в CLI, Telegram, Discord и Discord VC в реальном времени |
| 🗣️ **[Использование голосового режима с Hermes](/docs/guides/use-voice-mode-with-hermes)** | Практическая настройка и сценарии использования голосовых рабочих процессов |
| 🎭 **[Личность и SOUL.md](/docs/user-guide/features/personality)** | Задайте базовый голос Hermes через глобальный SOUL.md |
| 📄 **[Контекстные файлы](/docs/user-guide/features/context-files)** | Файлы контекста проекта, которые влияют на каждый диалог |
| 🔒 **[Безопасность](/docs/user-guide/security)** | Подтверждение команд, авторизация, изоляция контейнеров |
| 💡 **[Советы и лучшие практики](/docs/guides/tips)** | Быстрые способы получить от Hermes больше пользы |
| 🏗️ **[Архитектура](/docs/developer-guide/architecture)** | Как всё устроено под капотом |
| ❓ **[FAQ и устранение неполадок](/docs/reference/faq)** | Частые вопросы и решения |

## Ключевые возможности

- **Замкнутый цикл обучения** — память под управлением агента, периодические подсказки, автономное создание навыков, самоулучшение навыков во время работы, FTS5-поиск между сеансами с LLM-суммаризацией и [Honcho](https://github.com/plastic-labs/honcho) для диалектического моделирования пользователя
- **Работа не только на ноутбуке** — 6 терминальных бэкендов: локальный, Docker, SSH, Daytona, Singularity и Modal. Daytona и Modal дают serverless-персистентность: среда “спит” в простое и почти ничего не стоит
- **Живёт там, где удобно вам** — CLI, Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, Email, SMS, DingTalk, Feishu, WeCom, BlueBubbles, Home Assistant, Microsoft Teams: больше 15 платформ через один gateway
- **Создан разработчиками моделей** — Hermes делают [Nous Research](https://nousresearch.com), лаборатория, стоящая за Hermes, Nomos и Psyche. Поддерживает [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai), OpenAI и любые совместимые endpoint'ы
- **Плановые автоматизации** — встроенный cron с доставкой на любую платформу
- **Делегирование и параллелизм** — отдельные subagents для параллельных потоков работы. Programmatic Tool Calling через `execute_code` сводит многошаговые пайплайны к одному вызову inference
- **Открытые навыки** — совместимость с [agentskills.io](https://agentskills.io). Навыки переносимы, ими можно делиться, и они пополняются сообществом через Skills Hub
- **Полный контроль над web** — поиск, извлечение, браузинг, vision, генерация изображений, TTS
- **Поддержка MCP** — подключайте любой MCP-сервер для расширения возможностей инструментов
- **Готовность к research-задачам** — пакетная обработка, экспорт trajectory, RL-обучение с Atropos. Создан [Nous Research](https://nousresearch.com) — лабораторией, стоящей за моделями Hermes, Nomos и Psyche

## Для LLM и кодовых агентов

Машиночитаемые точки входа в эту документацию:

- **[`/llms.txt`](/llms.txt)** — curated index всех страниц документации с короткими описаниями. Около 17 KB, безопасно для загрузки в контекст LLM.
- **[`/llms-full.txt`](/llms-full.txt)** — все страницы документации, объединённые в один markdown-файл для разовой загрузки. Около 1.8 MB.

Оба файла также доступны по адресам `/docs/llms.txt` и `/docs/llms-full.txt`. Они генерируются заново при каждом деплое.
