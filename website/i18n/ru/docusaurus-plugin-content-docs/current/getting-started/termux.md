---
sidebar_position: 3
title: "Android / Termux"
description: "Запуск Hermes Agent на Android через Termux"
---

# Hermes на Android через Termux

Это проверенный путь для запуска Hermes Agent прямо на Android-телефоне через [Termux](https://termux.dev/).

Он даёт полноценный локальный CLI на телефоне и при этом ставит только те зависимости, которые действительно собираются и работают на Android.

## Что поддерживается

Проверенный Termux-бандл включает:
- Hermes CLI
- поддержку cron
- работу с PTY и фоновым терминалом
- Telegram gateway
- MCP
- Honcho memory
- ACP

Эквивалентная команда установки:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## Что пока не входит в проверенный путь

Некоторые функции пока зависят от desktop/server-окружения или не проверены на Android:
- `.[all]` на Android не поддерживается
- `voice` упирается в `faster-whisper` и `ctranslate2`, а Android wheels для `ctranslate2` не публикуются
- автоматическая bootstrap-установка браузерных инструментов пропускается
- Docker-изоляция терминала в Termux недоступна
- Android может прерывать фоновые задачи, поэтому gateway-процессы на телефоне всегда best-effort

Это не мешает Hermes работать как удобному телефонному CLI-агенту. Просто мобильный install-путь сознательно уже, чем desktop/server-путь.

## Вариант 1: установка в одну строку

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

На Termux установщик сам:
- ставит системные пакеты через `pkg`
- создаёт виртуальное окружение
- ставит `.[termux]`
- кладёт `hermes` в `$PREFIX/bin`
- пропускает неподтверждённые браузерные и WhatsApp-части

## Вариант 2: ручная установка

### 1. Обновите Termux и поставьте системные пакеты

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

### 2. Склонируйте Hermes

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

Если репозиторий уже склонирован без submodules:

```bash
git submodule update --init --recursive
```

### 3. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
```

`ANDROID_API_LEVEL` нужен для пакетов на базе `maturin`, например `jiter`.

### 4. Установите Termux-бандл

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

Если нужен только минимальный core-агент:

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. Добавьте `hermes` в PATH Termux

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

### 6. Проверьте установку

```bash
hermes version
hermes doctor
```

### 7. Запустите Hermes

```bash
hermes
```

## Что обычно настраивают дальше

```bash
hermes model
hermes setup
```

После этого можно уже подключать gateway, модели и навыки.

## Типичные проблемы

### `No solution found` при установке `.[all]`

Используйте проверенный Termux-бандл:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` жалуется на `ANDROID_API_LEVEL`

Перед установкой задайте API level явно:

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor` говорит, что не хватает ripgrep или Node

Поставьте их через Termux:

```bash
pkg install ripgrep nodejs
```
