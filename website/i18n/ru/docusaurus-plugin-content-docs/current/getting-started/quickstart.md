---
sidebar_position: 1
title: "Быстрый старт"
description: "Ваш первый разговор с Hermes Agent — от установки до чата меньше чем за 5 минут"
---

# Быстрый старт

Этот гид проведёт вас от нуля до рабочей установки Hermes, которая выдерживает реальную нагрузку. Мы поставим агент, выберем провайдера, проверим живой чат и разберёмся, что делать, если что-то сломалось.

## Хотите посмотреть видео?

**Onchain AI Garage** подготовили мастер-класс по установке, настройке и базовым командам — удобный спутник к этой странице, если вам проще следовать по видео. Полный плейлист с разбором Hermes Agent Tutorials & Use Cases доступен [здесь](https://www.youtube.com/channel/UCqB1bhMwGsW-yefBxYwFCCg).

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '1.5rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube-nocookie.com/embed/R3YOGfTBcQg"
    title="Hermes Agent Masterclass: Installation, Setup, Basic Commands"
    frameBorder="0"
    allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
  ></iframe>
</div>

## Для кого эта страница

- Для тех, кто только что начал и хочет самый короткий путь к рабочей настройке
- Для тех, кто переключает провайдера и не хочет терять время на ошибки в конфиге
- Для тех, кто настраивает Hermes для команды, бота или постоянно работающего workflow
- Для тех, кто устал от сценария «установилось, но всё равно ничего не делает»

## Самый быстрый путь

Выберите строку, которая лучше всего подходит под вашу задачу:

| Цель | Что сделать сначала | Что сделать потом |
|---|---|---|
| Хочу, чтобы Hermes просто работал на моей машине | `hermes setup` | Запустите реальный чат и проверьте, что он отвечает |
| Провайдер уже выбран | `hermes model` | Сохраните конфиг, затем начинайте чат |
| Нужен бот или режим «всегда онлайн» | `hermes gateway setup` после того, как CLI уже работает | Подключите Telegram, Discord, Slack или другую платформу |
| Нужна локальная или self-hosted модель | `hermes model` → custom endpoint | Проверьте endpoint, имя модели и длину контекста |
| Нужен fallback между несколькими провайдерами | Сначала `hermes model` | Добавляйте routing и fallback только после того, как базовый чат уже работает |

**Правило большого пальца:** если Hermes не может пройти обычный чат, не добавляйте пока новые возможности. Сначала добейтесь одного чистого разговора, а потом уже подключайте gateway, cron, skills, voice или routing.

---

## 1. Установка Hermes Agent

**Вариант A — pip (самый простой):**

```bash
pip install hermes-agent
hermes postinstall     # optional: installs Node.js, browser, ripgrep, ffmpeg + runs setup
```

Релизы PyPI следуют за тегированными версиями (major/minor releases), а не за каждым коммитом в `main`. Если нужен bleeding-edge, используйте Вариант B.

**Вариант B — git installer (следит за веткой main):**

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Android / Termux
Если ставите на телефон, откройте отдельное [руководство по Termux](/getting-started/termux) — там описан проверенный ручной путь, поддерживаемые extras и текущие ограничения Android.
:::

:::tip Пользователям Windows
Сначала установите [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install), а затем выполните команду выше внутри WSL2-терминала.
:::

Когда установка завершится, обновите shell:

```bash
source ~/.bashrc   # or source ~/.zshrc
```

За подробностями по способам установки, требованиям и troubleshooting см. [руководство по установке](/getting-started/installation).

## 2. Выберите провайдера

Самый важный шаг настройки. Используйте `hermes model`, чтобы пройти выбор интерактивно:

```bash
hermes model
```

Хорошие варианты по умолчанию:

| Провайдер | Что это такое | Как настроить |
|----------|-----------|---------------|
| **Nous Portal** | Подписка без лишней настройки | OAuth-логин через `hermes model` |
| **OpenAI Codex** | ChatGPT OAuth, использует модели Codex | Device code auth через `hermes model` |
| **Anthropic** | Модели Claude напрямую — либо Max plan + extra usage credits (OAuth), либо API key для pay-per-token | `hermes model` → OAuth-логин (нужны Max + extra credits) или Anthropic API key |
| **OpenRouter** | Маршрутизация между множеством моделей | Введите API key |
| **Z.AI** | Модели GLM / Zhipu-hosted | Укажите `GLM_API_KEY` / `ZAI_API_KEY` |
| **Kimi / Moonshot** | coding и chat-модели Moonshot | Укажите `KIMI_API_KEY` (или `KIMI_CODING_API_KEY` для Kimi-Coding) |
| **Kimi / Moonshot China** | Moonshot endpoint для китайского региона | Укажите `KIMI_CN_API_KEY` |
| **Arcee AI** | Trinity models | Укажите `ARCEEAI_API_KEY` |
| **GMI Cloud** | Direct API с несколькими моделями | Укажите `GMI_API_KEY` |
| **MiniMax (OAuth)** | MiniMax-M2.7 через browser OAuth — без API key | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | Международный endpoint MiniMax | Укажите `MINIMAX_API_KEY` |
| **MiniMax China** | Region-specific endpoint MiniMax | Укажите `MINIMAX_CN_API_KEY` |
| **Alibaba Cloud** | Qwen через DashScope | Укажите `DASHSCOPE_API_KEY` |
| **Hugging Face** | Более 20 open models через unified router (Qwen, DeepSeek, Kimi и др.) | Укажите `HF_TOKEN` |
| **AWS Bedrock** | Claude, Nova, Llama, DeepSeek через нативный Converse API | IAM role или `aws configure` ([руководство](/integrations/providers)) |
| **Kilo Code** | Модели, хостимые KiloCode | Укажите `KILOCODE_API_KEY` |
| **OpenCode Zen** | Pay-as-you-go доступ к curated моделям | Укажите `OPENCODE_ZEN_API_KEY` |
| **OpenCode Go** | Подписка за $10/месяц на open models | Укажите `OPENCODE_GO_API_KEY` |
| **DeepSeek** | Прямой доступ к DeepSeek API | Укажите `DEEPSEEK_API_KEY` |
| **NVIDIA NIM** | Nemotron через build.nvidia.com или локальный NIM | Укажите `NVIDIA_API_KEY` (опционально `NVIDIA_BASE_URL`) |
| **GitHub Copilot** | Подписка GitHub Copilot (GPT-5.x, Claude, Gemini и др.) | OAuth через `hermes model`, либо `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Backend Copilot ACP (запускает локальный `copilot` CLI) | `hermes model` (нужны `copilot` CLI + `copilot login`) |
| **Vercel AI Gateway** | Маршрутизация через Vercel AI Gateway | Укажите `AI_GATEWAY_API_KEY` |
| **Custom Endpoint** | VLLM, SGLang, Ollama или любой OpenAI-compatible API | Укажите base URL и API key |

Для большинства новичков лучший путь простой: выберите провайдера и принимайте дефолты, если только не знаете, зачем вам их менять. Полный каталог провайдеров с env vars и шагами настройки лежит на странице [Providers](/integrations/providers).

:::caution Минимальный контекст: 64K tokens
Hermes Agent требует модель минимум с **64,000 tokens** контекста. Модели с меньшим окном не смогут удерживать достаточно рабочей памяти для многошаговых tool-calling workflows и будут отклонены при старте. Большинство hosted-моделей (Claude, GPT, Gemini, Qwen, DeepSeek) это требование покрывают. Если используете локальную модель, выставьте context size не меньше 64K (например, `--ctx-size 65536` для llama.cpp или `-c 65536` для Ollama).
:::

:::tip
Переключить провайдера можно в любой момент через `hermes model` — никакой привязки нет. Полный список поддерживаемых провайдеров и подробности настройки смотрите на [Провайдеры ИИ](/integrations/providers).
:::

### Где хранятся настройки

Hermes разделяет секреты и обычный конфиг:

- **Secrets и tokens** → `~/.hermes/.env`
- **Не секретные настройки** → `~/.hermes/config.yaml`

Проще всего задавать значения через CLI:

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

Правильное значение само попадёт в нужный файл.

## 3. Запустите первый чат

```bash
hermes            # classic CLI
hermes --tui      # modern TUI (recommended)
```

Вы увидите welcome banner с выбранной моделью, доступными инструментами и skills. Используйте запрос, который легко проверить:

:::tip Выберите интерфейс
Hermes поставляется с двумя терминальными интерфейсами: классическим `prompt_toolkit` CLI и более новым [TUI](/user-guide/tui) с модальными окнами, выбором мышью и неблокирующим вводом. У них общие sessions, slash commands и config — попробуйте оба варианта: `hermes` и `hermes --tui`.
:::

```
Summarize this repo in 5 bullets and tell me what the main entrypoint is.
```

```
Check my current directory and tell me what looks like the main project file.
```

```
Help me set up a clean GitHub PR workflow for this codebase.
```

**Как выглядит успех:**

- banner показывает выбранную модель / провайдера
- Hermes отвечает без ошибок
- при необходимости он может использовать tool (terminal, file read, web search)
- разговор нормально продолжается больше чем на один turn

Если это работает, вы уже прошли самую сложную часть.

## 4. Проверьте, что sessions работают

Перед следующим шагом убедитесь, что resume работает:

```bash
hermes --continue    # Resume the most recent session
hermes -c            # Short form
```

Это должно вернуть вас в только что созданную сессию. Если не возвращает, проверьте, не сменили ли вы профиль и действительно ли session сохранилась. Это важно, когда потом появятся несколько конфигураций или устройств.

## 5. Попробуйте ключевые возможности

### Используйте терминал

```
❯ What's my disk usage? Show the top 5 largest directories.
```

Агент выполняет команды в терминале от вашего имени и показывает результат.

### Slash commands

Введите `/`, чтобы увидеть автодополнение со всеми командами:

| Команда | Что она делает |
|---------|----------------|
| `/help` | Показать все доступные команды |
| `/tools` | Показать список доступных инструментов |
| `/model` | Переключить модель интерактивно |
| `/personality pirate` | Попробовать забавную personality |
| `/save` | Сохранить conversation |

### Многострочный ввод

Нажмите `Alt+Enter`, `Ctrl+J` или `Shift+Enter`, чтобы добавить новую строку. `Shift+Enter` работает только в терминалах, которые отправляют его как отдельную последовательность (по умолчанию Kitty / foot / WezTerm / Ghostty; в iTerm2 / Alacritty / VS Code terminal — после включения Kitty keyboard protocol). `Alt+Enter` и `Ctrl+J` работают в любом терминале.

### Прервать агента

Если агент слишком долго думает, введите новое сообщение и нажмите Enter — это прервёт текущую задачу и переключит Hermes на ваш новый запрос. `Ctrl+C` тоже работает.

## 6. Добавьте следующий слой

Только после того, как базовый чат уже работает. Выберите то, что вам нужно:

### Бот или shared assistant

```bash
hermes gateway setup    # Interactive platform configuration
```

Подключите [Telegram](/user-guide/messaging/telegram), [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack), [WhatsApp](/user-guide/messaging/whatsapp), [Signal](/user-guide/messaging/signal), [Email](/user-guide/messaging/email), [Home Assistant](/user-guide/messaging/homeassistant) или [Microsoft Teams](/user-guide/messaging/teams).

### Автоматизация и инструменты

- `hermes tools` — настройте доступ к инструментам для каждой платформы
- `hermes skills` — просматривайте и устанавливайте reusable workflows
- Cron — только после того, как бот или CLI уже стабильно настроены

### Безопасный terminal

Для безопасности запускайте агента в Docker-контейнере или на удалённом сервере:

```bash
hermes config set terminal.backend docker    # Docker isolation
hermes config set terminal.backend ssh       # Remote server
```

### Voice mode

```bash
# From the Hermes install directory (the curl installer placed it at
# ~/.hermes/hermes-agent on Linux/macOS or %LOCALAPPDATA%\hermes\hermes-agent on Windows):
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# Includes faster-whisper for free local speech-to-text
```

Затем в CLI включите `/voice on`. Для записи нажмите `Ctrl+B`. Подробнее см. [Голосовой режим](/user-guide/features/voice-mode).

### Skills

```bash
hermes skills search kubernetes
hermes skills install openai/skills/k8s
```

Или используйте `/skills` внутри чата.

### MCP servers

```yaml
# Add to ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### Интеграция с editor (ACP)

Поддержка ACP входит в стандартный extra `[all]`, так что curl-установщик уже всё включает. Просто выполните:

```bash
hermes acp
```

(Если ставили без `[all]`, сначала выполните `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"`.)

См. [интеграцию ACP в редактор](/user-guide/features/acp).

---

## Типовые ошибки

Вот проблемы, на которые чаще всего уходит время:

| Симптом | Вероятная причина | Что делать |
|---|---|---|
| Hermes открывается, но отвечает пусто или с поломанным текстом | Ошибка в аутентификации провайдера или в выборе модели | Снова запустите `hermes model` и проверьте провайдера, модель и auth |
| Custom endpoint «работает», но выдаёт мусор | Неверный base URL, имя модели или endpoint вообще не OpenAI-compatible | Сначала проверьте endpoint в отдельном клиенте |
| Gateway стартует, но никто не может ему писать | Не завершена настройка bot token, allowlist или платформы | Снова запустите `hermes gateway setup` и проверьте `hermes gateway status` |
| `hermes --continue` не находит старую сессию | Вы сменили профиль или session не сохранилась | Проверьте `hermes sessions list` и убедитесь, что вы в правильном профиле |
| Модель недоступна или fallback ведёт себя странно | Routing или fallback слишком агрессивны | Оставьте routing выключенным, пока не стабилизируете базового провайдера |
| `hermes doctor` ругается на проблемы в конфиге | Часть значений отсутствует или устарела | Исправьте конфиг, затем ещё раз проверьте обычный чат, прежде чем подключать новые возможности |

## Набор для восстановления

Если что-то пошло не так, используйте такой порядок:

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

Этот порядок быстро возвращает вас из состояния «что-то не так» в понятную и проверенную конфигурацию.

---

## Краткая справка

| Команда | Что делает |
|---------|------------|
| `hermes` | Начать чат |
| `hermes model` | Выбрать LLM-провайдера и модель |
| `hermes tools` | Настроить, какие инструменты включены для каждой платформы |
| `hermes setup` | Полный мастер настройки (настраивает всё сразу) |
| `hermes doctor` | Диагностика проблем |
| `hermes update` | Обновить до последней версии |
| `hermes gateway` | Запустить messaging gateway |
| `hermes --continue` | Возобновить последнюю сессию |

## Что делать дальше

- **[Руководство по CLI](/user-guide/cli)** — Освойте terminal interface
- **[Конфигурация](/user-guide/configuration)** — Настройте Hermes под себя
- **[Шлюз сообщений](/user-guide/messaging)** — Подключите Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant, Teams и другие платформы
- **[Инструменты и toolsets](/user-guide/features/tools)** — Изучите доступные возможности
- **[Провайдеры ИИ](/integrations/providers)** — Полный список провайдеров и способы настройки
- **[Система навыков](/user-guide/features/skills)** — Reusable workflows и знания
- **[Советы и лучшие практики](/guides/tips)** — Советы для опытных пользователей
