---
sidebar_position: 11
sidebar_label: "Плагины"
title: "Плагины"
description: "Расширяйте Hermes собственными инструментами, хуками и интеграциями через систему плагинов"
---

# Плагины

У Hermes есть система плагинов, которая позволяет добавлять собственные инструменты, хуки и интеграции без правки ядра.

Если вы хотите создать собственный инструмент для себя, своей команды или одного проекта, это, как правило, правильный путь. Страница [Добавление инструментов](/docs/developer-guide/adding-tools) в руководстве разработчика предназначена для встроенных инструментов Hermes, которые живут в `tools/` и `toolsets.py`.

**→ [Соберите плагин Hermes](/docs/guides/build-a-hermes-plugin)** — пошаговое руководство с полностью рабочим примером.

## Краткий обзор

Поместите каталог в `~/.hermes/plugins/` с `plugin.yaml` и Python-кодом:

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # манифест
├── __init__.py      # register() — связывает схемы с обработчиками
├── schemas.py       # схемы инструментов (что видит LLM)
└── tools.py         # обработчики инструментов (что запускается при вызове)
```

Запустите Hermes — и ваши инструменты появятся рядом со встроенными. Модель сможет вызывать их сразу же.

### Минимальный рабочий пример

Ниже — полноценный плагин, который добавляет инструмент `hello_world` и логирует каждый вызов инструмента через хук.

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: Минимальный пример плагина
```

**`~/.hermes/plugins/hello-world/__init__.py`**

```python
"""Минимальный плагин Hermes — регистрирует инструмент и хук."""

import json


def register(ctx):
    # --- Инструмент: hello_world ---
    schema = {
        "name": "hello_world",
        "description": "Возвращает дружелюбное приветствие для указанного имени.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Имя для приветствия",
                }
            },
            "required": ["name"],
        },
    }

    def handle_hello(params, **kwargs):
        del kwargs
        name = params.get("name", "World")
        return json.dumps({"success": True, "greeting": f"Hello, {name}!"})

    ctx.register_tool(
        name="hello_world",
        toolset="hello_world",
        schema=schema,
        handler=handle_hello,
        description="Возвращает дружелюбное приветствие для указанного имени.",
    )

    # --- Хук: логируем каждый вызов инструмента ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

Положите оба файла в `~/.hermes/plugins/hello-world/`, перезапустите Hermes — и модель сразу сможет вызывать `hello_world`. Хук будет печатать строку в лог после каждого вызова инструмента.

Проектные плагины внутри `./.hermes/plugins/` по умолчанию отключены. Включайте их только для доверенных репозиториев, установив `HERMES_ENABLE_PROJECT_PLUGINS=true` перед запуском Hermes.

## Что умеют плагины

Каждый API `ctx.*`, перечисленный ниже, доступен внутри функции `register(ctx)` вашего плагина.

| Возможность | Как |
|-----------|-----|
| Добавить инструменты | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| Добавить хуки | `ctx.register_hook("post_tool_call", callback)` |
| Добавить слэш-команды | `ctx.register_command(name, handler, description)` — добавляет `/name` в CLI и сеансах шлюза |
| Вызывать инструменты из команд | `ctx.dispatch_tool(name, args)` — вызывает зарегистрированный инструмент с автоматическим привязыванием контекста родительского агента |
| Добавить CLI-команды | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — добавляет `hermes <plugin> <subcommand>` |
| Вставлять сообщения | `ctx.inject_message(content, role="user")` — см. [Вставка сообщений](#вставка-сообщений) |
| Включать файлы данных | `Path(__file__).parent / "data" / "file.yaml"` |
| Встраивать навыки | `ctx.register_skill(name, path)` — с пространством имён `plugin:skill`, загружается через `skill_view("plugin:skill")` |
| Проверять переменные окружения | `requires_env: [API_KEY]` в `plugin.yaml` — запрашивается во время `hermes plugins install` |
| Распространять через pip | `[project.entry-points."hermes_agent.plugins"]` |
| Регистрировать платформу шлюза (Discord, Telegram, IRC, …) | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — см. [Добавление адаптеров платформ](/docs/developer-guide/adding-platform-adapters) |
| Регистрировать бэкенд для генерации изображений | `ctx.register_image_gen_provider(provider)` — см. [Плагины провайдера генерации изображений](/docs/developer-guide/image-gen-provider-plugin) |
| Регистрировать бэкенд для генерации видео | `ctx.register_video_gen_provider(provider)` — см. [Плагины провайдера генерации видео](/docs/developer-guide/video-gen-provider-plugin) |
| Регистрировать движок сжатия контекста | `ctx.register_context_engine(engine)` — см. [Плагины движка контекста](/docs/developer-guide/context-engine-plugin) |
| Регистрировать бэкенд памяти | Подкласс `MemoryProvider` в `plugins/memory/<name>/__init__.py` — см. [Плагины провайдера памяти](/docs/developer-guide/memory-provider-plugin) (использует отдельную систему обнаружения) |
| Выполнять вызов LLM от имени хоста | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — берёт активную модель и авторизацию пользователя для разового completion-запроса с необязательной проверкой JSON-схемы. См. [Доступ плагина к LLM](/docs/developer-guide/plugin-llm-access) |
| Регистрировать бэкенд инференса (LLM-провайдер) | `register_provider(ProviderProfile(...))` в `plugins/model-providers/<name>/__init__.py` — см. [Плагины провайдера модели](/docs/developer-guide/model-provider-plugin) (использует отдельную систему обнаружения) |

## Обнаружение плагинов

| Источник | Путь | Сценарий использования |
|--------|------|----------|
| Встроенные | `<repo>/plugins/` | Поставляется вместе с Hermes — см. [Встроенные плагины](/docs/user-guide/features/built-in-plugins) |
| Пользовательские | `~/.hermes/plugins/` | Личные плагины |
| Проектные | `.hermes/plugins/` | Плагины для конкретного проекта (требуется `HERMES_ENABLE_PROJECT_PLUGINS=true`) |
| pip | `hermes_agent.plugins` entry_points | Распространяемые пакеты |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | Декларативная установка в NixOS — см. [руководство по Nix Setup](/docs/getting-started/nix-setup#plugins) |

При совпадении имён более поздние источники перекрывают более ранние, поэтому пользовательский плагин с тем же именем, что и встроенный, заменит его.

### Подкатегории плагинов

Внутри каждого источника Hermes также распознаёт каталоги подкатегорий, которые направляют плагины в специализированные системы обнаружения:

| Подкаталог | Что в нём лежит | Система обнаружения |
|---|---|---|
| `plugins/` (корень) | Общие плагины — инструменты, хуки, слэш-команды, CLI-команды, встроенные навыки | `PluginManager` (тип: `standalone` или `backend`) |
| `plugins/platforms/<name>/` | Адаптеры каналов шлюза (`ctx.register_platform()`) | `PluginManager` (тип: `platform`, на один уровень глубже) |
| `plugins/image_gen/<name>/` | Бэкенды генерации изображений (`ctx.register_image_gen_provider()`) | `PluginManager` (тип: `backend`, на один уровень глубже) |
| `plugins/memory/<name>/` | Провайдеры памяти (подкласс `MemoryProvider`) | **Собственный загрузчик** в `plugins/memory/__init__.py` (тип: `exclusive` — активен только один) |
| `plugins/context_engine/<name>/` | Движки сжатия контекста (`ctx.register_context_engine()`) | **Собственный загрузчик** в `plugins/context_engine/__init__.py` (активен только один) |
| `plugins/model-providers/<name>/` | Профили LLM-провайдеров (`register_provider(ProviderProfile(...))`) | **Собственный загрузчик** в `providers/__init__.py` (лениво сканируется при первом вызове `get_provider_profile()`) |

Пользовательские плагины в `~/.hermes/plugins/model-providers/<name>/` и `~/.hermes/plugins/memory/<name>/` перекрывают встроенные плагины с тем же именем — действует правило «последний записавший побеждает» в `register_provider()` / `register_memory_provider()`. Просто положите каталог в нужное место, и он заменит встроенную реализацию без правок репозитория.

Плагины подкатегорий отображаются в `hermes plugins list` и в интерактивном интерфейсе `hermes plugins` под ключом, полученным из пути — например `observability/langfuse`, `image_gen/openai`, `platforms/teams`. Именно этот ключ, а не простое значение `name:` из манифеста, используется в `hermes plugins enable …` / `disable …` и в поле `plugins.enabled` в `config.yaml`.

## Плагины по умолчанию отключены, с несколькими исключениями

**Общие плагины и пользовательские бэкенды по умолчанию отключены** — система обнаруживает их (поэтому они видны в `hermes plugins` и `/plugins`), но никакой код с хуками или инструментами не загружается, пока вы не добавите имя плагина в `plugins.enabled` в `~/.hermes/config.yaml`. Это защищает от запуска стороннего кода без вашего явного согласия.

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # необязательный список запрета — он всегда побеждает, если имя есть в обоих списках
    - noisy-plugin
```

Три способа переключить состояние:

```bash
hermes plugins                    # единый интерактивный интерфейс (space — отметить/снять отметку)
hermes plugins list               # таблица: enabled / disabled / not enabled
hermes plugins install user/repo   # установка из Git, затем запрос Enable? [y/N]
hermes plugins install user/repo --enable    # установить И включить (без запроса)
hermes plugins install user/repo --no-enable # установить, но оставить отключённым (без запроса)
hermes plugins update my-plugin    # подтянуть последнюю версию
hermes plugins remove my-plugin    # удалить
hermes plugins enable my-plugin    # добавить в список разрешений (обычный плагин)
hermes plugins enable observability/langfuse # добавить в список разрешений (плагин подкатегории)
hermes plugins disable my-plugin   # убрать из списка разрешений + добавить в disabled
```

Для плагинов внутри подкатегорий (например, `plugins/observability/langfuse/`, `plugins/image_gen/openai/`) используйте полный ключ `<category>/<plugin>` — именно его показывает `hermes plugins list` в колонке **Name**.

### Интерактивный интерфейс

Если запустить `hermes plugins` без аргументов, открывается общий интерактивный экран:

```
Плагины
  ↑↓ навигация  SPACE переключить  ENTER настроить/подтвердить  ESC готово

  Общие плагины
 → [✓] my-tool-plugin — Пользовательский инструмент поиска
   [ ] webhook-notifier — Событийные хуки
   [ ] disk-cleanup — Автоочистка временных файлов [bundled]
   [ ] observability/langfuse — Трассировка ходов / LLM-вызовов / инструментов в Langfuse [bundled]

  Плагины-провайдеры
     Memory Provider          ▸ honcho
     Context Engine           ▸ compressor
```

- **Раздел общих плагинов** — это чекбоксы, переключение делается пробелом. Отмеченный пункт = в `plugins.enabled`, снятый = в `plugins.disabled` (явно отключён).
- **Раздел плагинов-провайдеров** — показывает текущий выбор. Нажмите ENTER, чтобы открыть выбор одного активного провайдера.
- Встроенные плагины в списке помечаются тегом `[bundled]`.

Выбор плагинов-провайдеров сохраняется в `config.yaml`:

```yaml
memory:
  provider: "honcho"      # пустая строка = только встроенный вариант

context:
  engine: "compressor"    # встроенный компрессор по умолчанию
```

### enabled / disabled / не включён

Плагины могут находиться в одном из трёх состояний:

| Состояние | Значение | В `plugins.enabled`? | В `plugins.disabled`? |
|---|---|---|---|
| `enabled` | Загружается при следующем сеансе | Да | Нет |
| `disabled` | Явно выключен — не загрузится, даже если есть в `enabled` | (неважно) | Да |
| `not enabled` | Обнаружен, но вы его ещё не включали | Нет | Нет |

По умолчанию любой новый или встроенный плагин имеет состояние `not enabled`. `hermes plugins list` показывает все три состояния отдельно, чтобы было видно, что именно было выключено явно, а что просто ждёт включения.

В уже запущенном сеансе `/plugins` показывает, какие плагины сейчас загружены.

## Вставка сообщений

Плагины могут вставлять сообщения в активный разговор через `ctx.inject_message()`:

```python
ctx.inject_message("От вебхука пришли новые данные", role="user")
```

**Сигнатура:** `ctx.inject_message(content: str, role: str = "user") -> bool`

Как это работает:

- Если агент **ничего не делает** и ждёт ввода пользователя, сообщение ставится в очередь как следующий ввод и начинает новый ход.
- Если агент **в середине хода** (уже выполняет операцию), сообщение прерывает текущую работу — так же, как если бы пользователь начал печатать новое сообщение и нажал Enter.
- Для ролей, отличных от `"user"`, содержимое получает префикс `[role]` (например, `[system] ...`).
- Возвращает `True`, если сообщение было успешно поставлено в очередь, и `False`, если недоступна ссылка на CLI (например, в режиме шлюза).

Это позволяет таким плагинам, как удалённые панели управления, мосты для мессенджеров или обработчики вебхуков, передавать сообщения в разговор из внешних источников.

:::note
`inject_message` доступен только в CLI-режиме. В режиме шлюза ссылки на CLI нет, поэтому метод возвращает `False`.
:::

См. **[полное руководство](/docs/guides/build-a-hermes-plugin)**, если нужны контракты обработчиков, формат схем, поведение хуков, обработка ошибок и типичные ошибки.
