---
sidebar_position: 6
title: "Хуки событий"
description: "Запускайте собственный код в ключевых точках жизненного цикла — логируйте активность, отправляйте оповещения, публикуйте вебхуки"
---

# Хуки событий

У Hermes есть три системы хуков, которые запускают собственный код в ключевых точках жизненного цикла:

| Система | Регистрируется через | Работает в | Сценарий использования |
|--------|---------------------|-----------|------------------------|
| **[Хуки gateway](#gateway-event-hooks)** | `HOOK.yaml` + `handler.py` в `~/.hermes/hooks/` | Только gateway | Логирование, оповещения, вебхуки |
| **[Хуки плагинов](#plugin-hooks)** | `ctx.register_hook()` в [плагине](/docs/user-guide/features/plugins) | CLI + gateway | Перехват инструментов, метрики, защитные барьеры |
| **[Shell-хуки](#shell-hooks)** | Блок `hooks:` в `~/.hermes/config.yaml`, указывающий на shell-скрипты | CLI + gateway | Готовые скрипты для блокировки, автоформатирования, инъекции контекста |

Все три системы работают неблокирующим образом — ошибки в любом хуке ловятся и логируются, агент из-за них не падает.

## Хуки gateway {#gateway-event-hooks}

Хуки gateway автоматически срабатывают во время работы gateway (Telegram, Discord, Slack, WhatsApp, Teams), не блокируя основной pipeline агента.

### Создание хука

Каждый хук — это каталог в `~/.hermes/hooks/`, содержащий два файла:

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # Определяет, какие события слушать
    └── handler.py     # Функция-обработчик на Python
```

#### HOOK.yaml

```yaml
name: my-hook
description: Log all agent activity to a file
events:
  - agent:start
  - agent:end
  - agent:step
```

Список `events` определяет, какие события вызывают ваш обработчик. Можно подписываться на любую комбинацию событий, включая wildcard-шаблоны вроде `command:*`.

#### handler.py

```python
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".hermes" / "hooks" / "my-hook" / "activity.log"

async def handle(event_type: str, context: dict):
    """Called for each subscribed event. Must be named 'handle'."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **context,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Правила для обработчика:**
- функция должна называться `handle`
- она получает `event_type` (строка) и `context` (dict)
- может быть как `async def`, так и обычной `def` — работают оба варианта
- ошибки перехватываются и логируются, агент из-за них не падает

### Доступные события

| Событие | Когда срабатывает | Ключи контекста |
|-------|------------------|----------------|
| `gateway:startup` | Запускается процесс gateway | `platforms` (список активных платформ) |
| `session:start` | Создаётся новый messaging-сеанс | `platform`, `user_id`, `session_id`, `session_key` |
| `session:end` | Сеанс завершается (до сброса) | `platform`, `user_id`, `session_key` |
| `session:reset` | Пользователь выполнил `/new` или `/reset` | `platform`, `user_id`, `session_key` |
| `agent:start` | Агент начинает обрабатывать сообщение | `platform`, `user_id`, `session_id`, `message` |
| `agent:step` | Каждый проход цикла вызова инструментов | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | Агент завершает обработку | `platform`, `user_id`, `session_id`, `message`, `response` |
| `command:*` | Выполняется любая slash-команда | `platform`, `user_id`, `command`, `args` |

#### Сопоставление по wildcard

Обработчики, зарегистрированные на `command:*`, срабатывают для любого события вида `command:` (`command:model`, `command:reset` и т. д.). Так можно отслеживать все slash-команды одним подписочным правилом.

### Примеры

#### Уведомление в Telegram о долгих задачах

Отправляйте себе сообщение, когда агент работает больше 10 шагов:

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Alert when agent is taking many steps
events:
  - agent:step
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
import os
import httpx

THRESHOLD = 10
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_HOME_CHANNEL")

async def handle(event_type: str, context: dict):
    iteration = context.get("iteration", 0)
    if iteration == THRESHOLD and BOT_TOKEN and CHAT_ID:
        tools = ", ".join(context.get("tool_names", []))
        text = f"⚠️ Agent has been running for {iteration} steps. Last tools: {tools}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
            )
```

#### Логгер использования команд

Отслеживайте, какие slash-команды используются:

```yaml
# ~/.hermes/hooks/command-logger/HOOK.yaml
name: command-logger
description: Log slash command usage
events:
  - command:*
```

```python
# ~/.hermes/hooks/command-logger/handler.py
import json
from datetime import datetime
from pathlib import Path

LOG = Path.home() / ".hermes" / "logs" / "command_usage.jsonl"

def handle(event_type: str, context: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "command": context.get("command"),
        "args": context.get("args"),
        "platform": context.get("platform"),
        "user": context.get("user_id"),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

#### Вебхук при старте сеанса

Отправляйте POST-запрос внешнему сервису при создании нового сеанса:

```yaml
# ~/.hermes/hooks/session-webhook/HOOK.yaml
name: session-webhook
description: Notify external service on new sessions
events:
  - session:start
  - session:reset
```

```python
# ~/.hermes/hooks/session-webhook/handler.py
import httpx

WEBHOOK_URL = "https://your-service.example.com/hermes-events"

async def handle(event_type: str, context: dict):
    async with httpx.AsyncClient() as client:
        await client.post(WEBHOOK_URL, json={
            "event": event_type,
            **context,
        }, timeout=5)
```

### Учебный пример: BOOT.md — запуск стартового чеклиста при каждом старте gateway

Популярный сценарий из сообщества: положить Markdown-чеклист в `~/.hermes/BOOT.md` и заставить агента выполнять его каждый раз при старте gateway. Это удобно для сценариев вроде «при каждом старте проверь ночные падения cron-задач и, если что-то сломалось, напиши мне в Discord» или «суммируй последние 24 часа `deploy.log` и отправляй в Slack #ops».

Этот пример показывает, как собрать такой механизм самостоятельно через пользовательский хук. Hermes не поставляет встроенный хук BOOT.md — вы сами определяете ровно то поведение, которое вам нужно.

#### Что мы строим

1. Файл `~/.hermes/BOOT.md` с инструкциями на естественном языке.
2. Хук gateway, который срабатывает на `gateway:startup`, запускает одноразового агента с моделью и учётными данными вашего gateway и выполняет инструкции из BOOT.md.
3. Соглашение `[SILENT]`, позволяющее агенту не отправлять сообщение, если сообщать нечего.

#### Шаг 1: напишите чеклист

Создайте `~/.hermes/BOOT.md`. Пишите его так, как будто даёте инструкции человеку-ассистенту:

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, send a summary to Discord #ops using the `send_message` tool.
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same Discord message.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

Агент видит это как часть своего промпта, так что можно описывать всё, что он умеет делать: tool calls, shell-команды, отправку сообщений, суммирование файлов.

#### Шаг 2: создайте хук

```text
~/.hermes/hooks/boot-md/
├── HOOK.yaml
└── handler.py
```

**`~/.hermes/hooks/boot-md/HOOK.yaml`**

```yaml
name: boot-md
description: Run ~/.hermes/BOOT.md on gateway startup
events:
  - gateway:startup
```

**`~/.hermes/hooks/boot-md/handler.py`**

```python
"""Run ~/.hermes/BOOT.md on every gateway startup."""

import logging
import threading
from pathlib import Path

logger = logging.getLogger("hooks.boot-md")

BOOT_FILE = Path.home() / ".hermes" / "BOOT.md"


def _build_prompt(content: str) -> str:
    return (
        "You are running a startup boot checklist. Follow the instructions "
        "below exactly.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Execute each instruction. Use the send_message tool to deliver any "
        "messages to platforms like Discord or Slack.\n"
        "If nothing needs attention and there is nothing to report, reply "
        "with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str) -> None:
    """Spawn a one-shot agent and execute the checklist.

    Uses the gateway's resolved model and runtime credentials so this works
    against custom endpoints, aggregators, and OAuth-based providers alike.
    """
    try:
        from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs
        from run_agent import AIAgent

        agent = AIAgent(
            model=_resolve_gateway_model(),
            **_resolve_runtime_agent_kwargs(),
            platform="gateway",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
        )
        result = agent.run_conversation(_build_prompt(content))
        response = result.get("final_response", "")
        if response and "[SILENT]" not in response:
            logger.info("boot-md completed: %s", response[:200])
        else:
            logger.info("boot-md completed (nothing to report)")
    except Exception as e:
        logger.error("boot-md agent failed: %s", e)


async def handle(event_type: str, context: dict) -> None:
    if not BOOT_FILE.exists():
        return
    content = BOOT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return

    logger.info("Running BOOT.md (%d chars)", len(content))

    # Background thread so gateway startup isn't blocked on a full agent turn.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

Две ключевые строки:

- `_resolve_gateway_model()` берёт модель, настроенную для gateway
- `_resolve_runtime_agent_kwargs()` получает runtime-учётные данные так же, как обычный ход gateway — включая API-ключи, base URLs, OAuth-токены и credential pools

Без этого обычный `AIAgent()` откатится к дефолтным значениям и начнёт получать 401 на любом нестандартном endpoint'е.

#### Шаг 3: протестируйте

Перезапустите gateway:

```bash
hermes gateway restart
```

Посмотрите логи:

```bash
hermes logs --follow --level INFO | grep boot-md
```

Вы должны увидеть `Running BOOT.md (N chars)`, а затем либо `boot-md completed: ...` (краткое резюме того, что сделал агент), либо `boot-md completed (nothing to report)`, если агент ответил `[SILENT]`.

Удалите `~/.hermes/BOOT.md`, чтобы отключить чеклист — хук останется загруженным, но при отсутствии файла будет молча пропускать выполнение.

#### Как расширить этот паттерн

- **Чеклисты с учётом расписания:** проверяйте `datetime.now().weekday()` внутри инструкций BOOT.md («если сегодня понедельник, ещё проверь weekly deploy log»). Инструкции задаются свободным текстом, так что подойдёт всё, что агент умеет понять.
- **Несколько чеклистов:** укажите другой файл (`STARTUP.md`, `MORNING.md` и т. п.) и зарегистрируйте отдельные каталоги хуков для каждого.
- **Вариант без агента:** если полный агентный цикл не нужен, можно вообще не использовать `AIAgent`, а просто отправлять фиксированное уведомление через `httpx`. Это дешевле, быстрее и не требует провайдера.

#### Почему это не встроенная функция

Ранние версии Hermes поставляли такой сценарий как встроенный хук и незаметно запускали агента с дефолтными настройками при каждом старте gateway. Это удивляло пользователей с собственными endpoints и делало функцию невидимой для тех, кто не знал, что она вообще работает. Хранить это как документированный паттерн — у вас в каталоге hooks — значит, вы точно видите, что происходит, и явно соглашаетесь на это, создавая файлы.

### Как это работает

1. При старте gateway `HookRegistry.discover_and_load()` сканирует `~/.hermes/hooks/`
2. Каждый подкаталог с `HOOK.yaml` + `handler.py` загружается динамически
3. Обработчики регистрируются на указанные события
4. На каждом жизненном этапе `hooks.emit()` вызывает все совпавшие обработчики
5. Ошибки в любом обработчике перехватываются и логируются — сломанный хук никогда не валит агент

:::info
Gateway-хуки срабатывают только в **gateway** (Telegram, Discord, Slack, WhatsApp, Teams). CLI такие хуки не загружает. Для хуков, которые должны работать везде, используйте [хуки плагинов](#plugin-hooks).
:::

## Хуки плагинов {#plugin-hooks}

[Плагины](/docs/user-guide/features/plugins) могут регистрировать хуки, которые срабатывают и в CLI, и в gateway-сеансах. Они регистрируются программно через `ctx.register_hook()` внутри функции `register()` вашего плагина.

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
```

**Общие правила для всех хуков:**

- Колбэки получают **keyword arguments**. Всегда принимайте `**kwargs` ради обратной совместимости — новые параметры могут появиться в будущих версиях, не ломая ваш плагин
- Если колбэк **падает**, это логируется и пропускается. Остальные хуки и сам агент продолжают работу. Плохой плагин никогда не должен ломать агента
- На поведение влияют только два хука: [`pre_tool_call`](#pre_tool_call) может **заблокировать** инструмент, а [`pre_llm_call`](#pre_llm_call) может **вставить контекст** в LLM-вызов. Все остальные хуки — это только наблюдатели

### Краткий справочник

| Хук | Когда срабатывает | Возвращает |
|-----|------------------|-----------|
| [`pre_tool_call`](#pre_tool_call) | Перед выполнением любого инструмента | `{"action": "block", "message": str}`, чтобы отклонить вызов |
| [`post_tool_call`](#post_tool_call) | После того как инструмент вернулся | игнорируется |
| [`pre_llm_call`](#pre_llm_call) | Один раз за ход, до цикла вызова инструментов | `{"context": str}`, чтобы добавить контекст к сообщению пользователя |
| [`post_llm_call`](#post_llm_call) | Один раз за ход, после цикла вызова инструментов | игнорируется |
| [`on_session_start`](#on_session_start) | Создан новый сеанс (только первый ход) | игнорируется |
| [`on_session_end`](#on_session_end) | Сеанс завершается | игнорируется |
| [`on_session_finalize`](#on_session_finalize) | CLI/gateway завершает активный сеанс (flush, save, stats) | игнорируется |
| [`on_session_reset`](#on_session_reset) | Gateway подменяет ключ активного сеанса (например, `/new`, `/reset`) | игнорируется |
| [`subagent_stop`](#subagent_stop) | Дочерний `delegate_task` завершился | игнорируется |
| [`pre_gateway_dispatch`](#pre_gateway_dispatch) | Gateway получил сообщение пользователя до auth + dispatch | `{"action": "skip" \| "rewrite" \| "allow", ...}`, чтобы повлиять на поток |
| [`pre_approval_request`](#pre_approval_request) | Опасная команда нуждается в подтверждении, до отправки запроса/уведомления | игнорируется |
| [`post_approval_response`](#post_approval_response) | Пользователь ответил на запрос подтверждения (или вышел по таймауту) | игнорируется |
| [`transform_tool_result`](#transform_tool_result) | После возвращения любого инструмента, до передачи результата модели | `str`, чтобы заменить результат, `None`, чтобы оставить как есть |
| [`transform_terminal_output`](#transform_terminal_output) | Внутри инструмента `terminal`, до усечения/strip ANSI/redact | `str`, чтобы заменить сырой вывод, `None`, чтобы оставить как есть |
| [`transform_llm_output`](#transform_llm_output) | После завершения цикла вызова инструментов, до отдачи финального ответа | `str`, чтобы заменить текст ответа, `None`/пустое значение, чтобы оставить без изменений |

---

### `pre_tool_call`

Срабатывает **непосредственно перед** выполнением каждого инструмента — и встроенного, и плагинного.

**Сигнатура колбэка:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| Параметр | Тип | Описание |
|---------|-----|----------|
| `tool_name` | `str` | Имя инструмента, который вот-вот будет вызван (например, `"terminal"`, `"web_search"`, `"read_file"`) |
| `args` | `dict` | Аргументы, которые модель передала инструменту |
| `task_id` | `str` | Идентификатор сеанса/задачи. Пустая строка, если его нет |

**Когда срабатывает:** в `model_tools.py`, внутри `handle_function_call()`, перед запуском обработчика инструмента. Срабатывает один раз на каждый вызов инструмента — если модель вызывает 3 инструмента параллельно, хук сработает 3 раза.

**Возвращаемое значение — отклонить вызов:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
```

Агент обрывает вызов, а `message` возвращается модели как ошибка. Первое подходящее directive на блокировку выигрывает (сначала Python-плагины, затем shell-хуки). Любое другое значение игнорируется, так что существующие observer-only колбэки продолжают работать как раньше.

**Сценарии использования:** логирование, audit trail, счётчики вызовов инструментов, блокировка опасных операций, rate limiting, политики на уровне пользователя.

**Пример — лог audit для вызовов инструментов:**

```python
import json, logging
from datetime import datetime

logger = logging.getLogger(__name__)

def audit_tool_call(tool_name, args, task_id, **kwargs):
    logger.info("TOOL_CALL session=%s tool=%s args=%s",
                task_id, tool_name, json.dumps(args)[:200])

def register(ctx):
    ctx.register_hook("pre_tool_call", audit_tool_call)
```

**Пример — предупреждение о опасных инструментах:**

```python
DANGEROUS = {"terminal", "write_file", "patch"}

def warn_dangerous(tool_name, **kwargs):
    if tool_name in DANGEROUS:
        print(f"⚠ Executing potentially dangerous tool: {tool_name}")

def register(ctx):
    ctx.register_hook("pre_tool_call", warn_dangerous)
```

---

### `post_tool_call`

Срабатывает **сразу после** выполнения инструмента.

**Сигнатура колбэка:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `tool_name` | `str` | Имя инструмента, который только что выполнился |
| `args` | `dict` | Аргументы, которые модель передала инструменту |
| `result` | `str` | Возвращаемое значение инструмента (всегда JSON-строка) |
| `task_id` | `str` | Идентификатор сеанса/задачи. Пустая строка, если его нет |
| `duration_ms` | `int` | Сколько миллисекунд заняла доставка этого инструмента (измеряется через `time.monotonic()` вокруг `registry.dispatch()`) |

**Когда срабатывает:** в `model_tools.py`, внутри `handle_function_call()`, после того как обработчик инструмента вернулся. Срабатывает один раз на каждый вызов инструмента. **Не** срабатывает, если инструмент выбросил необработанное исключение (ошибка перехватывается и возвращается как JSON-строка ошибки, а `post_tool_call` всё равно срабатывает с этой строкой в `result`).

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** логирование результатов, сбор метрик, учёт частоты успешных/неудачных вызовов, latency-дашборды, оповещения по бюджету на инструмент, уведомления о завершении конкретных инструментов.

**Пример — метрики использования инструментов:**

```python
from collections import Counter, defaultdict
import json

_tool_counts = Counter()
_error_counts = Counter()
_latency_ms = defaultdict(list)

def track_metrics(tool_name, result, duration_ms=0, **kwargs):
    _tool_counts[tool_name] += 1
    _latency_ms[tool_name].append(duration_ms)
    try:
        parsed = json.loads(result)
        if "error" in parsed:
            _error_counts[tool_name] += 1
    except (json.JSONDecodeError, TypeError):
        pass

def register(ctx):
    ctx.register_hook("post_tool_call", track_metrics)
```

---

### `pre_llm_call`

Срабатывает **один раз за ход**, до начала цикла вызова инструментов. Это **единственный хук, чей результат используется** — он может внедрить контекст в текущее пользовательское сообщение.

**Сигнатура колбэка:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `session_id` | `str` | Уникальный идентификатор текущего сеанса |
| `user_message` | `str` | Исходное сообщение пользователя в этом ходе (до подстановки навыков) |
| `conversation_history` | `list` | Копия полного списка сообщений (формат OpenAI: `[{"role": "user", "content": "..."}]`) |
| `is_first_turn` | `bool` | `True`, если это первый ход нового сеанса, `False` на последующих ходах |
| `model` | `str` | Идентификатор модели (например, `"anthropic/claude-sonnet-4.6"`) |
| `platform` | `str` | Где работает сеанс: `"cli"`, `"telegram"`, `"discord"` и т. п. |

**Когда срабатывает:** в `run_agent.py`, внутри `run_conversation()`, после сжатия контекста, но до входа в основной цикл `while`. Срабатывает один раз на каждый вызов `run_conversation()` (то есть один раз на пользовательский ход), а не на каждый API-вызов внутри цикла инструментов.

**Возвращаемое значение:** если колбэк возвращает dict с ключом `"context"` или просто непустую строку, этот текст добавляется к текущему пользовательскому сообщению. Возвращайте `None`, если ничего внедрять не нужно.

```python
# Inject context
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# Plain string (equivalent)
return "Recalled memories:\n- User likes Python"

# No injection
return None
```

**Куда внедряется контекст:** всегда в **пользовательское сообщение**, никогда в системную инструкцию. Так сохраняется prompt cache — системная инструкция остаётся идентичной между ходами, поэтому кешированные токены переиспользуются. Системная инструкция — это территория Hermes (guidance модели, enforcement инструментов, personality, skills). Плагины подают контекст рядом с пользовательским вводом.

Весь внедрённый контекст **эпемерный** — он добавляется только на время API-вызова. Исходное пользовательское сообщение в истории беседы не изменяется, и ничего не записывается в базу сеансов.

Когда контекст возвращают **несколько плагинов**, их выводы объединяются через двойной перевод строки в порядке обнаружения плагинов (по алфавиту каталогов).

**Сценарии использования:** вспомогательная память, RAG-внедрение контекста, guardrails, аналитика на уровне хода.

**Пример — вспомогательная память:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall(session_id, user_message, is_first_turn, **kwargs):
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None
        text = "Recalled context:\n" + "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None

def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
```

**Пример — guardrails:**

```python
POLICY = "Never execute commands that delete files without explicit user confirmation."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

Срабатывает **один раз за ход**, после завершения цикла вызова инструментов и после того, как агент сформировал финальный ответ. Срабатывает только на **успешных** ходах — не срабатывает, если ход был прерван.

**Сигнатура колбэка:**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `session_id` | `str` | Уникальный идентификатор текущего сеанса |
| `user_message` | `str` | Исходное сообщение пользователя в этом ходе |
| `assistant_response` | `str` | Финальный текст ответа агента в этом ходе |
| `conversation_history` | `list` | Копия полного списка сообщений после завершения хода |
| `model` | `str` | Идентификатор модели |
| `platform` | `str` | Где работает сеанс |

**Когда срабатывает:** в `run_agent.py`, внутри `run_conversation()`, после выхода из цикла инструментов с финальным ответом. Защищён условием `if final_response and not interrupted` — то есть не срабатывает, когда пользователь прерывает ход или когда агент достигает лимита итераций, так и не выдав ответ.

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** синхронизация данных о разговоре во внешнюю систему памяти, расчёт метрик качества ответов, логирование сводок ходов, запуск follow-up действий.

**Пример — синхронизация во внешнюю память:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def sync_memory(session_id, user_message, assistant_response, **kwargs):
    try:
        httpx.post(f"{MEMORY_API}/store", json={
            "session_id": session_id,
            "user": user_message,
            "assistant": assistant_response,
        }, timeout=5)
    except Exception:
        pass  # best-effort

def register(ctx):
    ctx.register_hook("post_llm_call", sync_memory)
```

**Пример — учёт длины ответов:**

```python
import logging
logger = logging.getLogger(__name__)

def log_response_length(session_id, assistant_response, model, **kwargs):
    logger.info("RESPONSE session=%s model=%s chars=%d",
                session_id, model, len(assistant_response or ""))

def register(ctx):
    ctx.register_hook("post_llm_call", log_response_length)
```

---

### `on_session_start`

Срабатывает **один раз**, когда создаётся совершенно новый сеанс. Не срабатывает при продолжении сеанса (когда пользователь отправляет второе сообщение в уже существующий сеанс).

**Сигнатура колбэка:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `session_id` | `str` | Уникальный идентификатор нового сеанса |
| `model` | `str` | Идентификатор модели |
| `platform` | `str` | Где работает сеанс |

**Когда срабатывает:** в `run_agent.py`, внутри `run_conversation()`, на первом ходе нового сеанса — конкретно после сборки системной инструкции, но до старта цикла инструментов. Проверка выглядит как `if not conversation_history` (нет предыдущих сообщений = новый сеанс).

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** инициализация состояния, привязанного к сеансу, прогрев кешей, регистрация сеанса во внешнем сервисе, логирование старта сеанса.

**Пример — инициализация кеша сеанса:**

```python
_session_caches = {}

def init_session(session_id, model, platform, **kwargs):
    _session_caches[session_id] = {
        "model": model,
        "platform": platform,
        "tool_calls": 0,
        "started": __import__("datetime").datetime.now().isoformat(),
    }

def register(ctx):
    ctx.register_hook("on_session_start", init_session)
```

---

### `on_session_end`

Срабатывает в **самом конце** каждого вызова `run_conversation()`, независимо от результата. Также срабатывает из обработчика выхода CLI, если агент был в середине хода, когда пользователь вышел.

**Сигнатура колбэка:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `session_id` | `str` | Уникальный идентификатор сеанса |
| `completed` | `bool` | `True`, если агент выдал финальный ответ, иначе `False` |
| `interrupted` | `bool` | `True`, если ход был прерван (новое сообщение пользователя, `/stop` или выход) |
| `model` | `str` | Идентификатор модели |
| `platform` | `str` | Где работает сеанс |

**Когда срабатывает:** в двух местах:
1. **`run_agent.py`** — в конце каждого `run_conversation()` после всей очистки. Срабатывает всегда, даже если ход завершился ошибкой
2. **`cli.py`** — в atexit-обработчике CLI, но **только** если агент был в середине хода (`_agent_running=True`) в момент выхода. Это ловит Ctrl+C и `/exit` во время обработки. В этом случае `completed=False`, `interrupted=True`

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** сброс буферов, закрытие соединений, сохранение состояния сеанса, логирование длительности сеанса, очистка ресурсов, инициализированных в `on_session_start`.

**Пример — сброс и очистка:**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # Flush accumulated data to disk or external service
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**Пример — отслеживание длительности сеанса:**

```python
import time, logging
logger = logging.getLogger(__name__)

_start_times = {}

def on_start(session_id, **kwargs):
    _start_times[session_id] = time.time()

def on_end(session_id, completed, interrupted, **kwargs):
    start = _start_times.pop(session_id, None)
    if start:
        duration = time.time() - start
        logger.info("SESSION_DURATION session=%s seconds=%.1f completed=%s interrupted=%s",
                     session_id, duration, completed, interrupted)

def register(ctx):
    ctx.register_hook("on_session_start", on_start)
    ctx.register_hook("on_session_end", on_end)
```

---

### `on_session_finalize`

Срабатывает, когда CLI или gateway **сбрасывает** активный сеанс — например, когда пользователь выполняет `/new`, когда gateway удаляет неактивный сеанс по GC, или когда CLI завершается при активном агенте. Это последняя возможность сбросить состояние, привязанное к уходящему сеансу, до того как его идентичность исчезнет.

**Сигнатура колбэка:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `session_id` | `str` or `None` | Исходящий идентификатор сеанса. Может быть `None`, если активного сеанса не было |
| `platform` | `str` | `"cli"` или название платформы мессенджера (`"telegram"`, `"discord"` и т. д.) |

**Когда срабатывает:** в `cli.py` (при `/new` / выходе CLI) и `gateway/run.py` (когда сеанс сбрасывается или собирается GC). На стороне gateway всегда парно с `on_session_reset`.

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** сохранение финальных метрик сеанса до того, как ID будет отброшен, закрытие ресурсов, привязанных к сеансу, отправка финального телеметрического события, сброс отложенных записей.

---

### `on_session_reset`

Срабатывает, когда gateway **подменяет новый ключ сеанса** для активного чата — пользователь вызвал `/new`, `/reset`, `/clear`, либо адаптер выбрал новый сеанс после окна простоя. Это позволяет плагинам отреагировать на сброс состояния разговора, не дожидаясь следующего `on_session_start`.

**Сигнатура колбэка:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `session_id` | `str` | ID нового сеанса (уже переведён на новое значение) |
| `platform` | `str` | Название платформы мессенджера |

**Когда срабатывает:** в `gateway/run.py`, сразу после того, как выделен новый ключ сеанса, но до обработки следующего входящего сообщения. На gateway порядок такой: `on_session_finalize(old_id)` → swap → `on_session_reset(new_id)` → `on_session_start(new_id)` на первом входящем ходе.

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** сброс кешей, привязанных к `session_id`, выпуск аналитики о «переключении сеанса», прогрев нового bucket'а состояния.

См. полное руководство **[Build a Plugin guide](/docs/guides/build-a-hermes-plugin)** — там показаны схемы инструментов, обработчики и продвинутые паттерны хуков.

---

### `subagent_stop`

Срабатывает **один раз на каждого дочернего агента** после завершения `delegate_task`. Даже если вы делегировали одну задачу или пачку из трёх, этот хук сработает по одному разу для каждого ребёнка, последовательно на родительском потоке.

**Сигнатура колбэка:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `parent_session_id` | `str` | ID сессии родительского агента, который делегировал задачу |
| `child_role` | `str \| None` | Тег роли оркестратора, заданный для ребёнка (`None`, если функция не включена) |
| `child_summary` | `str \| None` | Финальный ответ, который ребёнок вернул родителю |
| `child_status` | `str` | `"completed"`, `"failed"`, `"interrupted"` или `"error"` |
| `duration_ms` | `int` | Wall-clock время выполнения ребёнка в миллисекундах |

**Когда срабатывает:** в `tools/delegate_tool.py`, после того как `ThreadPoolExecutor.as_completed()` завершает обработку всех дочерних future. Вызов маршалится в родительский поток, чтобы авторам хуков не приходилось думать о конкурентном выполнении колбэков.

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** логирование активности оркестрации, суммирование длительности дочерних задач для биллинга, запись audit-логов после делегирования.

**Пример — лог активности оркестратора:**

```python
import logging
logger = logging.getLogger(__name__)

def log_subagent(parent_session_id, child_role, child_status, duration_ms, **kwargs):
    logger.info(
        "SUBAGENT parent=%s role=%s status=%s duration_ms=%d",
        parent_session_id, child_role, child_status, duration_ms,
    )

def register(ctx):
    ctx.register_hook("subagent_stop", log_subagent)
```

:::info
При тяжёлом делегировании (например, роли оркестратора × 5 листьев × вложенная глубина) `subagent_stop` срабатывает очень много раз за ход. Делайте колбэк быстрым; дорогую работу выносите в фоновую очередь.
:::

---

### `pre_gateway_dispatch`

Срабатывает **один раз на каждое входящее `MessageEvent`** в gateway, после проверки internal-события, но **до** auth/pairing и dispatch в агент. Это точка перехвата для политик на уровне gateway, связанных с потоком сообщений (окна только для чтения, handover человеку, маршрутизация по чату и т. п.), которые не укладываются в логику одного адаптера платформы.

**Сигнатура колбэка:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `event` | `MessageEvent` | Нормализованное входящее сообщение (с полями `.text`, `.source`, `.message_id`, `.internal` и т. д.) |
| `gateway` | `GatewayRunner` | Активный gateway-runner, чтобы плагин мог вызывать `gateway.adapters[platform].send(...)` для side-channel ответов (уведомления владельцу и т. п.) |
| `session_store` | `SessionStore` | Для тихого инжеста транскрипта через `session_store.append_to_transcript(...)` |

**Когда срабатывает:** в `gateway/run.py`, внутри `GatewayRunner._handle_message()`, сразу после вычисления `is_internal`. **Внутренние события обходят хук полностью** (это системные события — завершение фоновых процессов и т. п. — и их нельзя gate-keep'ить политиками, видимыми пользователю).

**Возвращаемое значение:** `None` или dict. Побеждает первое распознанное action-значение; остальные результаты плагинов игнорируются. Исключения в колбэках перехватываются и логируются; gateway всегда продолжает обычную обработку при ошибке.

| Возврат | Эффект |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | Отбросить сообщение — без ответа агента, без pairing-flow, без auth. Предполагается, что плагин уже обработал сообщение (например, тихо записал его в транскрипт) |
| `{"action": "rewrite", "text": "new text"}` | Заменить `event.text`, затем продолжить обычную обработку с изменённым событием. Полезно для сворачивания буферизованных ambient-сообщений в один prompt |
| `{"action": "allow"}` / `None` | Обычная обработка — полный chain auth / pairing / agent-loop |

**Сценарии использования:** групповые чаты только на прослушивание (отвечать только при упоминании; буферизовать ambient-сообщения в контекст); human handover (тихо инжестить сообщения клиента, пока владелец ведёт чат вручную); rate limiting на уровне профиля; маршрутизация на основе политики.

**Пример — тихо отбрасывать неавторизованные DM, не вызывая pairing-код:**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**Пример — переписывать буфер ambient-сообщений в один prompt при упоминании:**

```python
_buffers = {}

def buffer_or_rewrite(event, **kwargs):
    key = (event.source.platform, event.source.chat_id)
    buf = _buffers.setdefault(key, [])
    if _bot_mentioned(event.text):
        combined = "\n".join(buf + [event.text])
        buf.clear()
        return {"action": "rewrite", "text": combined}
    buf.append(event.text)
    return {"action": "skip", "reason": "ambient-buffered"}

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", buffer_or_rewrite)
```

---

### `pre_approval_request`

Срабатывает **непосредственно перед** показом запроса подтверждения пользователю — охватывает все поверхности: интерактивный CLI, Ink TUI, gateway-платформы (Telegram, Discord, Slack, WhatsApp, Matrix и т. д.), а также ACP-клиенты (VS Code, Zed, JetBrains).

Это правильное место для подключения собственного notifier'а — например, macOS menu-bar app, которая показывает уведомление «разрешить / запретить», или audit-лога, фиксирующего каждый запрос подтверждения с контекстом.

**Сигнатура колбэка:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    **kwargs,
):
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `command` | `str` | Shell-команда, ожидающая подтверждения |
| `description` | `str` | Понятная человеку причина, по которой команда помечена (объединяется, если совпало несколько шаблонов) |
| `pattern_key` | `str` | Основной ключ шаблона, который вызвал подтверждение (например, `"rm_rf"`, `"sudo"`) |
| `pattern_keys` | `list[str]` | Все ключи шаблонов, которые совпали |
| `session_key` | `str` | Идентификатор сеанса, полезен для уведомлений, scoped по чату |
| `surface` | `str` | `"cli"` для интерактивных запросов CLI/TUI, `"gateway"` для асинхронных подтверждений на платформах |

**Возвращаемое значение:** игнорируется. Хуки здесь только наблюдают; они не могут ни отменить, ни предварительно подтвердить запрос. Используйте [`pre_tool_call`](#pre_tool_call), чтобы заблокировать инструмент до того, как он дойдёт до системы подтверждения.

**Сценарии использования:** desktop notifications, push-уведомления, audit logging, Slack webhooks, маршрутизация эскалаций, метрики.

**Пример — desktop notification на macOS:**

```python
import subprocess

def notify_approval(command, description, session_key, **kwargs):
    title = "Hermes needs approval"
    body = f"{description}: {command[:80]}"
    subprocess.Popen([
        "osascript", "-e",
        f'display notification "{body}" with title "{title}"',
    ])

def register(ctx):
    ctx.register_hook("pre_approval_request", notify_approval)
```

---

### `post_approval_response`

Срабатывает **после** ответа пользователя на запрос подтверждения (или после таймаута).

**Сигнатура колбэка:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    choice: str,
    **kwargs,
):
```

Те же kwargs, что и у `pre_approval_request`, плюс:

| Параметр | Тип | Описание |
|----------|-----|----------|
| `choice` | `str` | Одно из `"once"`, `"session"`, `"always"`, `"deny"` или `"timeout"` |

**Возвращаемое значение:** игнорируется.

**Сценарии использования:** закрыть соответствующее desktop-уведомление, записать финальное решение в audit-лог, обновить метрики, продвинуть rate limiter.

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `transform_tool_result`

Срабатывает **после** возврата инструмента и **до** добавления результата в беседу. Позволяет плагину переписать строку результата ЛЮБОГО инструмента — не только terminal — до того, как её увидит модель.

**Сигнатура колбэка:**

```python
def my_callback(
    tool_name: str,
    arguments: dict,
    result: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `tool_name` | `str` | Инструмент, который вернул результат (`read_file`, `web_extract`, `delegate_task`, …) |
| `arguments` | `dict` | Аргументы, с которыми модель вызвала инструмент |
| `result` | `str` | Сырой результат инструмента, после усечения и удаления ANSI |
| `task_id` | `str \| None` | ID задачи/сеанса в RL/benchmark-окружениях |

**Возвращаемое значение:** `str`, чтобы заменить результат (возвращаемая строка и есть то, что увидит модель), `None`, чтобы оставить как есть.

**Сценарии использования:** удаление PII из вывода `web_extract`, оборачивание длинных JSON-результатов в summary header, инъекция retrieval-augmented hints в результаты `read_file`, переписывание отчётов `delegate_task` в схему проекта.

```python
import re
SECRET = re.compile(r"sk-[A-Za-z0-9]{32,}")

def redact_secrets(tool_name, result, **kwargs):
    if SECRET.search(result):
        return SECRET.sub("[REDACTED]", result)
    return None

def register(ctx):
    ctx.register_hook("transform_tool_result", redact_secrets)
```

Этот хук применяется ко всем инструментам. Для переписывания только terminal см. `transform_terminal_output` ниже — он уже и срабатывает раньше в пайплайне (до усечения и до redaction).

---

### `transform_terminal_output`

Срабатывает внутри pipeline вывода инструмента `terminal`, **до** стандартного усечения на 50 КБ, удаления ANSI и redaction секретов. Позволяет плагинам переписать сырой stdout/stderr shell-команды до того, как любой downstream-процесс его коснётся.

**Сигнатура колбэка:**

```python
def my_callback(
    command: str,
    output: str,
    exit_code: int,
    cwd: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `command` | `str` | Shell-команда, которая породила вывод |
| `output` | `str` | Сырой объединённый stdout/stderr (может быть очень большим — усечение происходит после хука) |
| `exit_code` | `int` | Код выхода процесса |
| `cwd` | `str` | Рабочий каталог, в котором выполнялась команда |

**Возвращаемое значение:** `str`, чтобы заменить вывод, `None`, чтобы оставить как есть.

**Сценарии использования:** суммирование команд, которые печатают гигантский вывод (`du -ah`, `find`, `tree`), маркировка вывода проектным тегом, чтобы downstream-хуки понимали, как его обрабатывать, удаление шумов по времени, которые прыгают между запусками и ломают prompt cache.

```python
def summarize_find(command, output, **kwargs):
    if command.startswith("find ") and len(output) > 50_000:
        lines = output.count("\n")
        head = "\n".join(output.splitlines()[:40])
        return f"{head}\n\n[summary: {lines} paths total, showing first 40]"
    return None

def register(ctx):
    ctx.register_hook("transform_terminal_output", summarize_find)
```

Хорошо работает вместе с `transform_tool_result` (который покрывает все остальные инструменты).

---

### `transform_llm_output`

Срабатывает **один раз за ход** после завершения цикла вызова инструментов, когда модель уже сформировала финальный ответ, **до** передачи этого ответа пользователю (CLI, gateway или programmatic caller). Позволяет плагину переписать финальный текст ответа классическими средствами программирования — без дополнительных инференс-токенов на SOUL-flavor или skill-driven transform.

**Сигнатура колбэка:**

```python
def my_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs,
) -> str | None:
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `response_text` | `str` | Финальный текст ответа агента в этом ходе |
| `session_id` | `str` | ID сеанса для этого разговора (может быть пустым для разовых запусков) |
| `model` | `str` | Имя модели, которая сформировала ответ (например, `anthropic/claude-sonnet-4.6`) |
| `platform` | `str` | Платформа доставки (`cli`, `telegram`, `discord` и т. д.; пусто, если не задано) |

**Возвращаемое значение:** непустая `str`, чтобы заменить текст ответа, `None` или пустая строка, чтобы оставить его как есть. **Побеждает первая непустая строка**, если несколько плагинов зарегистрировали этот хук — по аналогии с `transform_tool_result`.

**Сценарии использования:** преобразование личности/лексики (пиратская речь, Spongebob), удаление пользовательских идентификаторов из финального текста, добавление проектного footer'а подписи, enforcement house style guide без траты токенов на инструкции SOUL.

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # pass through unchanged
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

Хук срабатывает только на непустой и непрерванный ответ — он не вызовется при остановке пользователем или на пустых ходах. Исключения логируются как предупреждения и не ломают выполнение агента.

---

## Shell-хуки {#shell-hooks}

Объявляйте shell-хуки в `cli-config.yaml`, и Hermes будет запускать их как подпроцессы при каждом срабатывании соответствующего события плагин-хука — и в CLI, и в gateway-сеансах. Писать Python-плагин не нужно.

Используйте shell-хуки, когда нужен простой однофайловый скрипт (Bash, Python, любой язык с shebang), чтобы:

- **Блокировать вызов инструмента** — отклонять опасные команды `terminal`, навязывать политики по каталогам, требовать подтверждения для разрушительных операций `write_file` / `patch`
- **Запускаться после вызова инструмента** — автоматически форматировать Python- или TypeScript-файлы, которые только что записал агент, логировать API-вызовы, запускать CI workflow
- **Внедрять контекст в следующий LLM-ход** — подмешивать `git status`, текущий день недели или полученные документы в сообщение пользователя (см. [`pre_llm_call`](#pre_llm_call))
- **Наблюдать за жизненным циклом** — писать лог-строку, когда субагент завершается (`subagent_stop`) или когда сеанс начинается (`on_session_start`)

Shell-хуки регистрируются вызовом `agent.shell_hooks.register_from_config(cfg)` как при запуске CLI (`hermes_cli/main.py`), так и при запуске gateway (`gateway/run.py`). Они естественно сочетаются с Python-плагинами — и те и другие проходят через один и тот же диспетчер.

### Сравнение с одного взгляда

| Параметр | Shell-хуки | [Хуки плагинов](#plugin-hooks) | [Хуки gateway](#gateway-event-hooks) |
|----------|------------|---------------------------------|--------------------------------------|
| Объявляются в | Блок `hooks:` в `~/.hermes/config.yaml` | `register()` в плагине `plugin.yaml` | Каталог `HOOK.yaml` + `handler.py` |
| Хранятся в | `~/.hermes/agent-hooks/` (по соглашению) | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| Язык | Любой (Bash, Python, Go binary и т. д.) | Только Python | Только Python |
| Работают в | CLI + gateway | CLI + gateway | Только gateway |
| Сценарии | `VALID_HOOKS` (включая `subagent_stop`) | `VALID_HOOKS` | Gateway lifecycle (`gateway:startup`, `agent:*`, `command:*`) |
| Могут блокировать вызов инструмента | Да (`pre_tool_call`) | Да (`pre_tool_call`) | Нет |
| Могут внедрять LLM-контекст | Да (`pre_llm_call`) | Да (`pre_llm_call`) | Нет |
| Согласие | Запрос при первом использовании для пары `(event, command)` | Имплицитное доверие к Python-плагину | Имплицитное доверие к каталогу |
| Изоляция между процессами | Да (subprocess) | Нет (in-process) | Нет (in-process) |

### Схема конфигурации

```yaml
hooks:
  <event_name>:                  # Должно быть в VALID_HOOKS
    - matcher: "<regex>"         # Необязательно; используется только для pre/post_tool_call
      command: "<shell command>" # Обязательно; запускается через shlex.split, shell=False
      timeout: <seconds>         # Необязательно; по умолчанию 60, максимум 300

hooks_auto_accept: false         # См. модель согласия ниже
```

Имена событий должны быть одним из событий [plugin hooks](#plugin-hooks); опечатки вызывают предупреждение «Did you mean X?» и пропускаются. Неизвестные ключи внутри одной записи игнорируются; отсутствие `command` даёт предупреждение и пропуск. `timeout > 300` усекается до 300 с предупреждением.

### JSON wire protocol

Каждый раз, когда событие срабатывает, Hermes запускает подпроцесс для каждого совпавшего хука (с учётом правила сопоставления), передаёт JSON-данные в **stdin** и читает **stdout** обратно как JSON.

**stdin — данные, которые получает скрипт:**

```json
{
  "hook_event_name": "pre_tool_call",
  "tool_name":       "terminal",
  "tool_input":      {"command": "rm -rf /"},
  "session_id":      "sess_abc123",
  "cwd":             "/home/user/project",
  "extra":           {"task_id": "...", "tool_call_id": "..."}
}
```

`tool_name` и `tool_input` равны `null` для неинструментальных событий (`pre_llm_call`, `subagent_stop`, session lifecycle). Словарь `extra` содержит все event-specific kwargs (`user_message`, `conversation_history`, `child_role`, `duration_ms` и т. д.). Не сериализуемые значения приводятся к строке, а не отбрасываются.

**stdout — необязательный ответ:**

```jsonc
// Заблокировать pre_tool_call (поддерживаются оба формата; внутри нормализуются):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code style
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical

// Внедрить контекст для pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Тихий no-op — любое пустое / несовпадающее содержимое нормально:
```

Сломанный JSON, ненулевые коды выхода и таймауты пишут предупреждение, но никогда не прерывают цикл агента.

### Рабочие примеры

#### 1. Автоформатирование Python-файлов после каждой записи

```yaml
# ~/.hermes/config.yaml
hooks:
  post_tool_call:
    - matcher: "write_file|patch"
      command: "~/.hermes/agent-hooks/auto-format.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/auto-format.sh
payload="$(cat -)"
path=$(echo "$payload" | jq -r '.tool_input.path // empty')
[[ "$path" == *.py ]] && command -v black >/dev/null && black "$path" 2>/dev/null
printf '{}\n'
```

Внутренняя, уже загруженная в контекст версия файла **не перечитывается автоматически** — форматирование влияет только на файл на диске. Последующие вызовы `read_file` увидят уже отформатированную версию.

#### 2. Блокировка опасных команд `terminal`

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal"
      command: "~/.hermes/agent-hooks/block-rm-rf.sh"
      timeout: 5
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/block-rm-rf.sh
payload="$(cat -)"
cmd=$(echo "$payload" | jq -r '.tool_input.command // empty')
if echo "$cmd" | grep -qE 'rm[[:space:]]+-rf?[[:space:]]+/'; then
  printf '{"decision": "block", "reason": "blocked: rm -rf / is not permitted"}\n'
else
  printf '{}\n'
fi
```

#### 3. Внедрение `git status` в каждый ход (аналог Claude-Code `UserPromptSubmit`)

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # discard stdin payload
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

Событие Claude Code `UserPromptSubmit` специально не выделяется как отдельное событие Hermes — `pre_llm_call` срабатывает в том же месте и уже поддерживает внедрение контекста. Используйте его здесь.

#### 4. Логирование завершения каждого субагента

```yaml
hooks:
  subagent_stop:
    - command: "~/.hermes/agent-hooks/log-orchestration.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/log-orchestration.sh
log=~/.hermes/logs/orchestration.log
jq -c '{ts: now, parent: .session_id, extra: .extra}' < /dev/stdin >> "$log"
printf '{}\n'
```

### Модель согласия

Каждая уникальная пара `(event, command)` запрашивает у пользователя разрешение при первом обнаружении Hermes, а затем решение сохраняется в `~/.hermes/shell-hooks-allowlist.json`. Последующие запуски (CLI или шлюз) пропускают запрос.

Три аварийных обхода обходят интерактивный запрос — достаточно любого одного:

1. флаг `--accept-hooks` у CLI (например, `hermes --accept-hooks chat`)
2. переменная окружения `HERMES_ACCEPT_HOOKS=1`
3. `hooks_auto_accept: true` в `cli-config.yaml`

Неработающие в TTY-сеансе (gateway, cron, CI) должны использовать один из этих трёх вариантов — иначе любой новый хук просто не зарегистрируется и запишет предупреждение.

**Правки скрипта не сбрасывают согласие.** Список разрешений привязан к точной строке команды, а не к хешу скрипта, так что редактирование скрипта на диске не отменяет согласие. `hermes hooks doctor` показывает расхождения по mtime, чтобы можно было заметить изменения и решить, надо ли повторно подтверждать.

### CLI `hermes hooks`

| Команда | Что делает |
|--------|-----------|
| `hermes hooks list` | Показывает настроенные хуки с правилом сопоставления, таймаутом и статусом согласия |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | Запускает все совпавшие хуки на синтетическом payload и печатает разобранный ответ |
| `hermes hooks revoke <command>` | Удаляет все записи списка разрешений, совпадающие с `<command>` (действует после следующего перезапуска) |
| `hermes hooks doctor` | Для каждого настроенного хука проверяет бит исполняемости, статус списка разрешений, расхождение по mtime, корректность JSON-ответа и примерное время исполнения |

### Безопасность

Shell-хуки работают с **вашими полными пользовательскими учётными данными** — это тот же уровень доверия, что и cron-запись или shell alias. Считайте блок `hooks:` в `config.yaml` привилегированной конфигурацией:

- Используйте только те скрипты, которые написали сами или полностью проверили
- Держите скрипты в `~/.hermes/agent-hooks/`, чтобы путь было удобно аудитить
- После pull'а общего конфига повторно запускайте `hermes hooks doctor`, чтобы заметить новые хуки до их регистрации
- Если `config.yaml` версионируется в команде, PR, которые меняют секцию `hooks:`, ревьюьте так же строго, как CI-конфиг

### Порядок и приоритеты

И Python plugin hooks, и shell hooks проходят через один и тот же диспетчер `invoke_hook()`. Python-плагины регистрируются первыми (`discover_and_load()`), shell-hooks — вторыми (`register_from_config()`), поэтому решения Python `pre_tool_call` о блокировке имеют приоритет в спорных случаях. Первое валидное block-решение выигрывает — агрегатор возвращает результат, как только любой колбэк отдаёт `{"action": "block", "message": str}` с непустым сообщением.
