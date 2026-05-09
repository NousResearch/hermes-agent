---
sidebar_position: 1
title: "Быстрый старт"
description: "Ваш первый диалог с Hermes Agent — от установки до общения меньше чем за 5 минут"
---

# Быстрый старт

Этот гид проведёт вас от нуля до рабочей установки Hermes, которая выдерживает реальное использование. Мы установим агент, выберем провайдера, проверим рабочий чат и заранее поймём, что делать, если что-то пойдёт не так.

## Для кого этот раздел

- Для тех, кто только начинает и хочет самый короткий путь к рабочей настройке
- Для тех, кто меняет провайдера и не хочет терять время на ошибки в конфиге
- Для тех, кто настраивает Hermes для команды, бота или постоянно работающего сценария
- Для тех, кому надоело, что “оно установилось, но ничего не делает”

## Самый быстрый путь

Выберите строку, которая соответствует вашей цели:

| Цель | Сначала сделайте это | Потом сделайте это |
|---|---|---|
| Хочу просто, чтобы Hermes работал на моём компьютере | `hermes setup` | Запустите реальный чат и проверьте ответ |
| Уже знаю своего провайдера | `hermes model` | Сохраните конфиг, затем начните общение |
| Хочу бота или always-on настройку | `hermes gateway setup` после того, как CLI заработал | Подключите Telegram, Discord, Slack или другую платформу |
| Хочу локальную или self-hosted модель | `hermes model` → custom endpoint | Проверьте endpoint, имя модели и длину контекста |
| Хочу fallback между несколькими провайдерами | Сначала `hermes model` | Добавляйте routing и fallback только после того, как базовый чат уже работает |

**Правило большого пальца:** если Hermes не может завершить обычный диалог, не добавляйте новые возможности. Сначала добейтесь одного чистого рабочего разговора, а потом подключайте gateway, cron, навыки, голос или routing.

---

## 1. Установите Hermes Agent

Запустите установщик в одну строку:

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Android / Termux
Если вы устанавливаете Hermes на телефон, сначала загляните в отдельный [гид по Termux](./termux.md): там описан проверенный ручной путь, поддерживаемые дополнительные пакеты и текущие ограничения Android.
:::

:::tip Пользователям Windows
Сначала установите [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install), а затем выполните команду выше внутри терминала WSL2.
:::

После завершения обновите shell:

```bash
source ~/.bashrc   # или source ~/.zshrc
```

Подробные варианты установки, требования и устранение неполадок см. в [руководстве по установке](./installation.md).

## 2. Выберите провайдера

Это самый важный шаг настройки. Используйте `hermes model`, чтобы пройти выбор интерактивно:

```bash
hermes model
```

Хорошие варианты по умолчанию:

| Провайдер | Что это | Как настроить |
|----------|---------|---------------|
| **Nous Portal** | Подписка без ручной настройки | OAuth-вход через `hermes model` |
| **OpenAI Codex** | OAuth ChatGPT, использует модели Codex | Авторизация по device code через `hermes model` |
| **Anthropic** | Модели Claude напрямую — через план Max и дополнительные кредиты (OAuth) либо по API-ключу с оплатой за токены | `hermes model` → OAuth-вход (нужен Max + дополнительные кредиты) либо Anthropic API key |
| **OpenRouter** | Маршрутизация между многими моделями | Введите свой API key |
| **Z.AI** | Модели GLM / Zhipu-hosted | Установите `GLM_API_KEY` / `ZAI_API_KEY` |
| **Kimi / Moonshot** | Coding и chat-модели от Moonshot | Установите `KIMI_API_KEY` |
| **Kimi / Moonshot China** | Региональный endpoint Moonshot для Китая | Установите `KIMI_CN_API_KEY` |
| **Arcee AI** | Модели Trinity | Установите `ARCEEAI_API_KEY` |
| **GMI Cloud** | Прямой multi-model API | Установите `GMI_API_KEY` |
| **MiniMax (OAuth)** | MiniMax-M2.7 через browser OAuth — API key не нужен | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | Международный endpoint MiniMax | Установите `MINIMAX_API_KEY` |
| **MiniMax China** | Китайский endpoint MiniMax | Установите `MINIMAX_CN_API_KEY` |
| **Alibaba Cloud** | Модели Qwen через DashScope | Установите `DASHSCOPE_API_KEY` |
| **Hugging Face** | Более 20 open-моделей через unified router (Qwen, DeepSeek, Kimi и др.) | Установите `HF_TOKEN` |
| **Kilo Code** | Модели от KiloCode | Установите `KILOCODE_API_KEY` |
| **OpenCode Zen** | Оплата по факту за curated models | Установите `OPENCODE_ZEN_API_KEY` |
| **OpenCode Go** | Подписка за $10/месяц на открытые модели | Установите `OPENCODE_GO_API_KEY` |
| **DeepSeek** | Прямой доступ к API DeepSeek | Установите `DEEPSEEK_API_KEY` |
| **NVIDIA NIM** | Модели Nemotron через build.nvidia.com или локальный NIM | Установите `NVIDIA_API_KEY` (опционально: `NVIDIA_BASE_URL`) |
| **GitHub Copilot** | Подписка GitHub Copilot (GPT-5.x, Claude, Gemini и др.) | OAuth через `hermes model` либо `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | ACP-backend Copilot (запускает локальный `copilot` CLI) | `hermes model` (нужны `copilot` CLI и `copilot login`) |
| **Vercel AI Gateway** | Маршрутизация через Vercel AI Gateway | Установите `AI_GATEWAY_API_KEY` |
| **Custom Endpoint** | VLLM, SGLang, Ollama или любой совместимый с OpenAI API endpoint | Укажите base URL + API key |

Для большинства новичков: выберите провайдера и принимайте значения по умолчанию, если только вы точно не понимаете, зачем их меняете. Полный каталог провайдеров с переменными окружения и шагами настройки есть на странице [Providers](../integrations/providers.md).

:::caution Minimum context: 64K tokens
Hermes Agent требует модель с контекстом не меньше **64,000 токенов**. Модели с меньшим окном не могут удерживать достаточный объём рабочей памяти для многошаговых tool-calling workflows и будут отклонены при запуске. Большинство hosted-моделей (Claude, GPT, Gemini, Qwen, DeepSeek) легко проходят это требование. Если вы используете локальную модель, выставьте размер контекста не меньше 64K (например, `--ctx-size 65536` для llama.cpp или `-c 65536` для Ollama).
:::

:::tip
Вы всегда можете переключить провайдера через `hermes model` — никакой жёсткой привязки нет. Полный список поддерживаемых провайдеров и детали настройки см. на странице [AI Providers](../integrations/providers.md).
:::

### Как хранятся настройки

Hermes разделяет секреты и обычный конфиг:

- **Секреты и токены** → `~/.hermes/.env`
- **Неконфиденциальные настройки** → `~/.hermes/config.yaml`

Самый удобный способ задать значения корректно — через CLI:

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

Правильное значение автоматически попадает в правильный файл.

## 3. Запустите первый чат

```bash
hermes            # классический CLI
hermes --tui      # современный TUI (рекомендуется)
```

Вы увидите welcome-banner с вашей моделью, доступными инструментами и навыками. Начните с запроса, который легко проверить:

:::tip Выберите интерфейс
В Hermes есть два терминальных интерфейса: классический `prompt_toolkit` CLI и новый [TUI](../user-guide/tui.md) с модальными overlay, выделением мышью и неблокирующим вводом. Оба используют одни и те же sessions, slash-команды и конфиг — попробуйте `hermes` и `hermes --tui`.
:::

```
Суммируйте этот репозиторий в 5 пунктах и скажите, где находится основной entrypoint.
```

```
Проверьте мой текущий каталог и скажите, какой файл выглядит основным проектным файлом.
```

```
Помогите настроить чистый workflow для GitHub PR в этом кодовой базе.
```

**Как выглядит успех:**

- Баннер показывает выбранную модель и провайдера
- Hermes отвечает без ошибок
- При необходимости он использует инструмент (terminal, file read, web search)
- Разговор продолжается нормально больше одного хода

Если это сработало, вы прошли самую сложную часть.

## 4. Проверьте, что сессии работают

Прежде чем идти дальше, убедитесь, что resume работает:

```bash
hermes --continue    # Возобновить последнюю сессию
hermes -c            # Краткая форма
```

Это должно вернуть вас в ту сессию, которую вы только что открыли. Если нет, проверьте, что вы находитесь в том же profile и что сессия действительно сохранилась. Это особенно важно, когда вы работаете с несколькими настройками или разными машинами.

## 5. Попробуйте ключевые возможности

### Используйте терминал

```
❯ Сколько у меня занято диска? Покажи 5 самых больших каталогов.
```

Агент выполняет terminal-команды от вашего имени и показывает результат.

### Slash-команды

Введите `/`, чтобы увидеть autocomplete со всеми командами:

| Команда | Что делает |
|---------|------------|
| `/help` | Показывает все доступные команды |
| `/tools` | Показывает доступные инструменты |
| `/model` | Интерактивно переключает модель |
| `/personality pirate` | Пробует забавную персону |
| `/save` | Сохраняет беседу |

### Многострочный ввод

Нажмите `Alt+Enter` или `Ctrl+J`, чтобы добавить новую строку. Это удобно, если вы вставляете код или пишете подробный запрос.

### Прервите агента

Если агент отвечает слишком долго, отправьте новое сообщение и нажмите Enter — это прервёт текущую задачу и переключит его на новые инструкции. `Ctrl+C` тоже работает.

## 6. Добавьте следующий слой

Только после того, как базовый чат заработал. Выберите то, что нужно:

### Бот или общий ассистент

```bash
hermes gateway setup    # Интерактивная настройка платформы
```

Подключите [Telegram](/docs/user-guide/messaging/telegram), [Discord](/docs/user-guide/messaging/discord), [Slack](/docs/user-guide/messaging/slack), [WhatsApp](/docs/user-guide/messaging/whatsapp), [Signal](/docs/user-guide/messaging/signal), [Email](/docs/user-guide/messaging/email) или [Home Assistant](/docs/user-guide/messaging/homeassistant), либо [Microsoft Teams](/docs/user-guide/messaging/teams).

### Автоматизация и инструменты

- `hermes tools` — настройка доступа к инструментам по платформам
- `hermes skills` — просмотр и установка reusable workflows
- Cron — только после того, как CLI или бот уже стабильно работают

### Изолированный терминал

Для безопасности запускайте агента в Docker-контейнере или на удалённом сервере:

```bash
hermes config set terminal.backend docker    # Изоляция в Docker
hermes config set terminal.backend ssh       # Удалённый сервер
```

### Голосовой режим

```bash
pip install "hermes-agent[voice]"
# Включает faster-whisper для локального speech-to-text
```

Затем в CLI: `/voice on`. Нажмите `Ctrl+B`, чтобы записать голос. См. [Voice Mode](../user-guide/features/voice-mode.md).

### Навыки

```bash
hermes skills search kubernetes
hermes skills install openai/skills/k8s
```

Или используйте `/skills` внутри сессии.

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

### Интеграция с редактором (ACP)

```bash
pip install -e '.[acp]'
hermes acp
```
