---
sidebar_position: 1
title: "Инструменты и наборы инструментов"
description: "Обзор инструментов Hermes Agent — что доступно, как работают toolsets и терминальные бэкенды"
---

# Инструменты и наборы инструментов

Инструменты — это функции, которые расширяют возможности агента. Они сгруппированы в логические **наборы инструментов** (toolsets), которые можно включать и отключать для каждой платформы.

## Доступные инструменты

В Hermes встроен большой реестр инструментов: веб-поиск, автоматизация браузера, выполнение команд в терминале, редактирование файлов, память, делегирование, обучение с подкреплением, доставка сообщений, Home Assistant и многое другое.

:::note
**Межсеансовая память Honcho** доступна как плагин провайдера памяти (`plugins/memory/honcho/`), а не как встроенный набор инструментов. См. [Плагины](./plugins.md) для установки.
:::

Основные категории:

| Категория | Примеры | Описание |
|----------|---------|-----------|
| **Веб** | `web_search`, `web_extract` | Поиск по вебу и извлечение содержимого страниц. |
| **X Search** | `x_search` | Поиск постов и тредов в X (Twitter) через встроенный инструмент Responses `x_search` от xAI — доступен только при наличии учётных данных xAI (SuperGrok OAuth или `XAI_API_KEY`); по умолчанию выключен, включается вручную через `hermes tools` → 🐦 X (Twitter) Search. |
| **Терминал и файлы** | `terminal`, `process`, `read_file`, `patch` | Выполнение команд и работа с файлами. |
| **Браузер** | `browser_navigate`, `browser_snapshot`, `browser_vision` | Интерактивная автоматизация браузера с поддержкой текста и vision. |
| **Медиа** | `vision_analyze`, `image_generate`, `video_generate`, `video_analyze`, `text_to_speech` | Мультимодальный анализ и генерация. `video_generate` и `video_analyze` доступны по явному включению (добавьте toolsets `video_gen` / `video` через `hermes tools` или `--toolsets`). |
| **Оркестрация агентов** | `todo`, `clarify`, `execute_code`, `delegate_task` | Планирование, уточнение, выполнение кода и делегирование субагентам. |
| **Память и поиск по истории** | `memory`, `session_search` | Постоянная память и поиск по сеансам. |
| **Автоматизация и доставка** | `cronjob`, `send_message` | Запланированные задачи с действиями create/list/update/pause/resume/run/remove, а также доставка сообщений наружу. |
| **Интеграции** | `ha_*`, инструменты MCP-серверов, `rl_*` | Home Assistant, MCP, обучение с подкреплением и другие интеграции. |

Авторитетный реестр, сформированный из кода, см. в разделах [Справочник встроенных инструментов](/docs/reference/tools-reference) и [Справочник наборов инструментов](/docs/reference/toolsets-reference).

:::tip Tool Gateway Nous
Платные подписчики [Nous Portal](https://portal.nousresearch.com) могут пользоваться веб-поиском, генерацией изображений, TTS и автоматизацией браузера через **[Tool Gateway](tool-gateway.md)** — без отдельных API-ключей. Включите его через `hermes model` или настройте отдельные инструменты через `hermes tools`.
:::

## Использование наборов инструментов

```bash
# Использовать конкретные наборы инструментов
hermes chat --toolsets "web,terminal"

# Показать все доступные инструменты
hermes tools

# Настраивать инструменты для платформы (интерактивно)
hermes tools
```

Часто используемые наборы инструментов: `web`, `search`, `terminal`, `file`, `browser`, `vision`, `image_gen`, `moa`, `skills`, `tts`, `todo`, `memory`, `session_search`, `cronjob`, `code_execution`, `delegation`, `clarify`, `homeassistant`, `messaging`, `spotify`, `discord`, `discord_admin`, `debugging`, `safe` и `rl`.

Полный список, включая платформенные пресеты вроде `hermes-cli`, `hermes-telegram` и динамические MCP-наборы вида `mcp-<server>`, см. в [Справочнике наборов инструментов](/docs/reference/toolsets-reference).

## Терминальные бэкенды

Терминальный инструмент может выполнять команды в разных окружениях:

| Бэкенд | Описание | Сценарий использования |
|--------|---------|------------------------|
| `local` | Выполняет команды на вашей машине (по умолчанию) | Разработка, доверенные задачи |
| `docker` | Изолированные контейнеры | Безопасность, воспроизводимость |
| `ssh` | Удалённый сервер | Песочница, изоляция агента от собственного кода |
| `singularity` | Контейнеры HPC | Кластерные вычисления, запуск без root |
| `modal` | Облачное выполнение | Serverless, масштабирование |
| `daytona` | Облачное рабочее пространство-сандбокс | Постоянные удалённые среды разработки |
| `vercel_sandbox` | Облачная microVM Vercel Sandbox | Облачное выполнение с сохранением файловой системы через snapshots |

### Конфигурация

```yaml
# В ~/.hermes/config.yaml
terminal:
  backend: local    # или: docker, ssh, singularity, modal, daytona, vercel_sandbox
  cwd: "."          # Рабочий каталог
  timeout: 180      # Таймаут команды в секундах
```

### Бэкенд Docker

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**Один постоянный контейнер, общий для всего процесса.** Hermes при первом использовании поднимает один долгоживущий контейнер (`docker run -d ... sleep 2h`) и направляет все вызовы terminal, file и `execute_code` через `docker exec` в этот же контейнер. Изменения рабочего каталога, установленные пакеты, правки окружения и файлы, записанные в `/workspace`, сохраняются от одного вызова инструмента к другому, включая `/new`, `/reset` и субагентов `delegate_task`, на всё время жизни процесса Hermes. При завершении работы контейнер останавливается и удаляется.

Это значит, что Docker-бэкенд ведёт себя как постоянная sandbox VM, а не как новый контейнер на каждую команду. Если однажды выполнить `pip install foo`, пакет останется доступным до конца сеанса. Если перейти в `/workspace/project`, последующие `ls` увидят этот каталог. См. [Конфигурация → Бэкенд Docker](../configuration.md#docker-backend) для полного описания жизненного цикла и флага `container_persistent`, который определяет, сохраняются ли `/workspace` и `/root` между перезапусками Hermes.

### Бэкенд SSH

Рекомендуется для безопасности — агент не может изменить собственный код:

```yaml
terminal:
  backend: ssh
```
```bash
# Задайте учётные данные в ~/.hermes/.env
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer

```bash
# Соберите SIF для параллельных исполнителей заранее
apptainer build ~/python.sif docker://python:3.11-slim

# Настройка
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Modal (serverless cloud)

```bash
uv pip install modal
modal setup
hermes config set terminal.backend modal
```

### Vercel Sandbox

```bash
pip install 'hermes-agent[vercel]'
hermes config set terminal.backend vercel_sandbox
hermes config set terminal.vercel_runtime node24
```

Аутентифицируйтесь с помощью всех трёх переменных: `VERCEL_TOKEN`, `VERCEL_PROJECT_ID` и `VERCEL_TEAM_ID`. Такая схема с access token — поддерживаемый путь для деплоя и обычных долгоживущих процессов Hermes на Render, Railway, Docker и подобных хостах. Поддерживаются runtime `node24`, `node22` и `python3.13`; по умолчанию Hermes использует `/vercel/sandbox` как корневой каталог удалённого workspace.

Для разового локального запуска Hermes также принимает короткоживущие Vercel OIDC-токены:

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

Из каталога связанного Vercel-проекта:

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

При `container_persistent: true` Hermes использует Vercel snapshots, чтобы сохранять состояние файловой системы между пересозданиями sandbox для одной и той же задачи. В snapshot могут попадать синхронизированные Hermes учётные данные, навыки и кэш-файлы внутри sandbox. Snapshots не сохраняют живые процессы, пространство PID и саму identity запущенного sandbox.

Фоновые terminal-команды используют общий для Hermes не локальный поток выполнения: spawn, poll, wait, log и kill проходят через стандартный process tool, пока sandbox жив, но Hermes не умеет нативно восстанавливать отложенные процессы Vercel после очистки или перезапуска.

Оставьте `container_disk` пустым или равным общему значению по умолчанию `51200`; произвольная настройка диска для Vercel Sandbox не поддерживается и завершится ошибкой диагностики или создания бэкенда.

### Ресурсы контейнера

Настройте CPU, память, диск и постоянство для всех контейнерных бэкендов:

```yaml
terminal:
  backend: docker  # или singularity, modal, daytona, vercel_sandbox
  container_cpu: 1              # Ядра CPU (по умолчанию: 1)
  container_memory: 5120        # Память в МБ (по умолчанию: 5 ГБ)
  container_disk: 51200         # Диск в МБ (по умолчанию: 50 ГБ)
  container_persistent: true    # Сохранять файловую систему между сеансами (по умолчанию: true)
```

Когда `container_persistent: true`, установленные пакеты, файлы и конфигурация сохраняются между сеансами.

### Безопасность контейнеров

Все контейнерные бэкенды запускаются с усилением безопасности:

- root filesystem только для чтения (Docker)
- все Linux capabilities отключены
- повышение привилегий запрещено
- лимит PID (256 процессов)
- полная изоляция пространств имён
- постоянное рабочее пространство через volumes, а не через writable root layer

Docker может дополнительно получить явный список разрешений переменных окружения через `terminal.docker_forward_env`, но перенаправленные переменные видны командам внутри контейнера и должны считаться доступными этому сеансу.

## Управление фоновыми процессами

Запускайте фоновые процессы и управляйте ими:

```python
terminal(command="pytest -v tests/", background=true)
# Возвращает: {"session_id": "proc_abc123", "pid": 12345}

# Затем управляйте через process tool:
process(action="list")       # Показать все запущенные процессы
process(action="poll", session_id="proc_abc123")   # Проверить состояние
process(action="wait", session_id="proc_abc123")   # Блокировать до завершения
process(action="log", session_id="proc_abc123")    # Полный вывод
process(action="kill", session_id="proc_abc123")   # Завершить
process(action="write", session_id="proc_abc123", data="y")  # Отправить ввод
```

Режим `pty=true` включает интерактивные CLI-инструменты вроде Codex и Claude Code.

## Поддержка sudo

Если команде нужен sudo, вам предложат ввести пароль (он кешируется на сеанс). Либо можно задать `SUDO_PASSWORD` в `~/.hermes/.env`.

:::warning
На платформах обмена сообщениями, если sudo не проходит, в выводе появляется подсказка добавить `SUDO_PASSWORD` в `~/.hermes/.env`.
:::
