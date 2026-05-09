---
sidebar_position: 2
title: "Установка"
description: "Установка Hermes Agent на Linux, macOS, WSL2 и Android через Termux"
---

# Установка

Hermes Agent можно поднять буквально за пару минут через один установочный скрипт.

## Быстрая установка

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Android / Termux

Для Termux используется тот же установщик:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Установщик сам распознаёт Termux и переключается на проверенный сценарий для Android:
- ставит системные зависимости через `pkg`
- создаёт виртуальное окружение через `python -m venv`
- выставляет `ANDROID_API_LEVEL` для сборки Android-совместимых пакетов
- ставит curated-extra `.[termux]`
- пропускает неподтверждённую bootstrap-логику для браузера и WhatsApp

Если нужен полностью ручной путь, откройте отдельный [гид по Termux](./termux.md).

:::tip Пользователям Windows
Нативный Windows не поддерживается. Установите [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) и выполните команду выше внутри WSL2.
:::

## Что делает установщик

Установщик берёт на себя всё основное:
- клонирует репозиторий
- настраивает Python, Node.js, ripgrep и ffmpeg
- создаёт виртуальное окружение
- добавляет команду `hermes` в PATH
- подготавливает базовую конфигурацию провайдера

После завершения установки просто обновите shell и начните работать:

```bash
source ~/.bashrc   # или source ~/.zshrc
hermes
```

Для дальнейшей настройки пригодятся команды:

```bash
hermes model
hermes tools
hermes gateway setup
hermes config set
hermes setup
```

## Требования

Минимальное требование одно: `git`.

Остальное установщик подтянет сам:
- `uv`
- Python 3.11
- Node.js v22
- `ripgrep`
- `ffmpeg`

:::info
Вручную ставить Python, Node.js, ripgrep или ffmpeg не нужно. Достаточно, чтобы `git` был доступен в `PATH`.
:::

## Ручная установка

Если вы хотите развернуть Hermes из исходников для разработки или работы с конкретной веткой, используйте раздел Development Setup в английской документации проекта.

## Поиск проблем

| Проблема | Что сделать |
|---|---|
| `hermes: command not found` | Перезапустите shell или проверьте `PATH` |
| Не задан API key | Запустите `hermes model` или задайте ключ через `hermes config set` |
| Конфиг пропал после обновления | Выполните `hermes config check`, затем `hermes config migrate` |

Если нужна дополнительная диагностика, запустите `hermes doctor`.
