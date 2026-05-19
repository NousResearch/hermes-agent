---
title: "Windows (нативный режим) — ранняя бета"
description: "Ранняя бета: нативный запуск Hermes Agent на Windows 10 / 11 — установка, матрица возможностей, консоль UTF-8, Git Bash, gateway как запланированная задача, работа с редактором, PATH, удаление и типичные ловушки"
sidebar_label: "Windows (нативный режим) — бета"
sidebar_position: 3
---

# Windows (нативный режим) — ранняя бета

:::warning РАННЯЯ БЕТА
Нативная поддержка Windows находится на стадии **ранней беты**. Установка работает, Hermes запускается и проходит наши проверки на типичные Windows-ошибки, но в боевых условиях он пока не обкатан так же широко, как пути Linux/macOS/WSL2. Ждите шероховатостей — особенно в subprocess-поверхности, в нюансах путей и в выводе консоли с не-ASCII символами. Если столкнётесь с проблемой, пожалуйста, [сообщите о ней](https://github.com/NousResearch/hermes-agent/issues) с шагами воспроизведения. Если нужен наиболее проверенный сценарий уже сейчас, используйте [установщик Linux/macOS внутри WSL2](/user-guide/windows-wsl-quickstart).
:::

Hermes нативно работает на Windows 10 и Windows 11 — без WSL, без Cygwin, без Docker. Эта страница подробно объясняет, что именно работает нативно, что остаётся только для WSL, что делает установщик, и какие Windows-специфичные настройки могут понадобиться.

Если вам нужно просто установить Hermes, достаточно одной команды на [главной странице](/) или на [странице установки](/getting-started/installation). Возвращайтесь сюда, когда захотите разобраться в деталях.

:::tip Нужен WSL вместо этого?
Если вам нужна полноценная POSIX-среда для встроенного терминала dashboard, семантики `fork`, Linux-стиля файловых наблюдателей и похожих вещей, смотрите **[руководство по Windows (WSL2)](/user-guide/windows-wsl-quickstart)**. Оба варианта спокойно сосуществуют: нативные данные лежат в `%LOCALAPPDATA%\hermes`, а данные WSL — в `~/.hermes`.
:::

## Быстрая установка

Откройте **PowerShell** (или Windows Terminal) и выполните:

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

Админские права не нужны. Установщик развернётся в `%LOCALAPPDATA%\hermes\` и добавит `hermes` в ваш **User PATH** — после завершения откройте новый терминал.

**Параметры установщика** (для передачи аргументов нужен вариант со scriptblock):

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1))) -NoVenv -SkipSetup -Branch main
```

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `-Branch` | `main` | Клонировать конкретную ветку (полезно для проверки PR) |
| `-Commit` | не задано | Привязать установку к конкретному SHA коммита (имеет приоритет над `-Branch`) |
| `-Tag` | не задано | Привязать установку к конкретному git-тегу (например, `v0.14.0`) |
| `-NoVenv` | выключено | Пропустить создание venv (для продвинутых сценариев, где Python вы управляете сами) |
| `-SkipSetup` | выключено | Пропустить мастер первого запуска `hermes setup` |
| `-HermesHome` | `%LOCALAPPDATA%\hermes` | Переопределить каталог данных |
| `-InstallDir` | `%LOCALAPPDATA%\hermes\hermes-agent` | Переопределить каталог с кодом |

Установщик автоматически повторяет неудачные `git fetch` и удаляет BOM из любого скачанного `install.ps1`, так что UTF-8 BOM, попавший в payload при HTTP-передаче, больше не ломает форму `[scriptblock]::Create((irm ...))`.

### Графический установщик (альтернатива) {#desktop-installer-alternative}

Существует и тонкий графический установщик — он удобен, если вам проще дважды щёлкнуть `.exe`, чем открывать PowerShell. Скачайте Hermes Desktop, запустите установщик, и при первом запуске GUI сам вызовет `install.ps1`, чтобы подготовить Python (через `uv`), Node, PortableGit и остальные зависимости из описанного ниже bootstrap-процесса. После первого запуска настольное приложение и CLI, установленный через PowerShell, используют одну и ту же установку в `%LOCALAPPDATA%\hermes\hermes-agent` и общий каталог данных `%USERPROFILE%\.hermes` — можно свободно переключаться между GUI и CLI.

Используйте графический установщик, когда хотите привычный Windows-сценарий установки или отдаёте Hermes пользователю без опыта работы в терминале; используйте однострочную команду PowerShell, если вы уже работаете из shell.

### Автоматическая подготовка зависимостей (`dep_ensure`)

При первом запуске (и по требованию, когда Hermes обнаруживает, что чего-то не хватает) он запускает небольшой Python-bootstrapper — `hermes_cli/dep_ensure.py`. Он проверяет и при необходимости поднимает не-Python зависимости. На Windows к ним относятся:

| Зависимость | Зачем она нужна Hermes |
|---|---|
| **PortableGit** | Даёт `bash.exe` для работы в терминале и `git` для клонирования внутри сессии. Устанавливается на этапе инсталляции, а не через `dep_ensure`. |
| **Node.js 22** | Нужен для браузерного инструмента (`agent-browser`), веб-моста TUI и моста WhatsApp. |
| **ffmpeg** | Конвертация аудиоформатов для TTS / голосовых сообщений. |
| **ripgrep** | Быстрый поиск по файлам — при отсутствии откатывается на `grep`. |
| **npm packages** | `agent-browser`, Playwright Chromium и любые Node-зависимости для наборов инструментов ставятся один раз при первом использовании браузерного инструмента. |

У каждой зависимости есть проверка в стиле `shutil.which(...)`; если бинарника нет и запуск интерактивный, `dep_ensure` предложит установить его, а саму установку делегирует `scripts\install.ps1 -ensure <dep>`. Неинтерактивные запуски (gateway, cron, headless desktop launches) пропускают диалог и вместо этого показывают понятную ошибку `this feature needs <dep>`.

## Что делает установщик на самом деле

По порядку, сверху вниз:

1. **Поднимает `uv`** — быстрый менеджер Python от Astral. Устанавливается в `%USERPROFILE%\.local\bin`.
2. **Ставит Python 3.11** через `uv`. Существующий Python не нужен.
3. **Устанавливает Node.js 22** (через winget, если он доступен, иначе распаковывает портативный tarball Node в `%LOCALAPPDATA%\hermes\node`). Он нужен для браузерного инструмента и моста WhatsApp.
4. **Ставит портативный Git** — если `git` уже есть в PATH, установщик использует его; иначе он скачивает урезанный самодостаточный **PortableGit** (~45 МБ, из официального релиза `git-for-windows`) в `%LOCALAPPDATA%\hermes\git`. Без админских прав, без реестра Windows Installer, без какого-либо конфликта с уже установленным Git.
5. **Клонирует репозиторий** в `%LOCALAPPDATA%\hermes\hermes-agent` и создаёт внутри него виртуальное окружение.
6. **Tiered `uv pip install`** — сначала пробует `.[all]`, а если `git+https`-зависимость ловит сбой на rate-limited GitHub, откатывается к всё более компактным наборам (`[messaging,dashboard,ext]` → `[messaging]` → `.`). Это убирает режим, когда единичный сбой оставляет вас с урезанной установкой.
7. **Автоустанавливает messaging SDKs** по `.env` — если присутствуют `TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` / `WHATSAPP_ENABLED`, он запускает `python -m ensurepip --upgrade` и точечные `pip install`, чтобы SDK каждой платформы действительно можно было импортировать.
8. **Задаёт `HERMES_GIT_BASH_PATH`** на найденный `bash.exe`, чтобы Hermes детерминированно находил его в свежих shell-сессиях.
9. **Добавляет `%LOCALAPPDATA%\hermes\bin` в User PATH** — так команда `hermes` становится доступна после открытия нового терминала.
10. **Запускает `hermes setup`** — обычный мастер первого запуска (модель, провайдер, toolsets). Его можно пропустить через `-SkipSetup`.

## Матрица возможностей

Всё, кроме встроенной terminal-панели dashboard, работает на Windows нативно.

| Возможность | Native Windows | WSL2 |
|---|---|---|
| CLI (`hermes chat`, `hermes setup`, `hermes gateway`, …) | ✓ | ✓ |
| Интерактивный TUI (`hermes --tui`) | ✓ | ✓ |
| Messaging gateway (Telegram, Discord, Slack, WhatsApp, 15+ платформ) | ✓ | ✓ |
| Cron scheduler | ✓ | ✓ |
| Browser tool (Chromium через Node) | ✓ | ✓ |
| MCP servers (stdio и HTTP) | ✓ | ✓ |
| Локальные Ollama / LM Studio / llama-server | ✓ | ✓ (через WSL networking) |
| Web dashboard (sessions, jobs, metrics, config) | ✓ | ✓ |
| Встроенная terminal-панель `/chat` в dashboard | ✗ (нужен POSIX PTY) | ✓ |
| Автозапуск при входе | ✓ (schtasks) | ✓ (systemd) |

Вкладка `/chat` в dashboard встраивает настоящий терминал через POSIX PTY (`ptyprocess`). В нативном режиме Windows такого примитива нет; в теории подошли бы `pywinpty` / Windows ConPTY, но это отдельная реализация — считайте это будущей работой. **Весь остальной dashboard работает нативно** — только одна эта вкладка показывает баннер вида "используйте WSL2 для этого".

## Как Hermes запускает shell-команды в Windows {#how-hermes-runs-shell-commands-on-windows}

Terminal tool Hermes запускает команды через **Git Bash**, как это делает Claude Code. Это позволяет обойти разрыв между POSIX и Windows, не переписывая каждый инструмент.

Порядок поиска `bash.exe`:

1. Переменная окружения `HERMES_GIT_BASH_PATH`, если она задана.
2. `%LOCALAPPDATA%\hermes\git\usr\bin\bash.exe` (PortableGit, которым управляет установщик).
3. `%LOCALAPPDATA%\hermes\git\bin\bash.exe` (старая раскладка Git for Windows).
4. Системная установка Git for Windows (`%ProgramFiles%\Git\bin\bash.exe` и т. п.).
5. MSYS2, Cygwin или любой другой `bash.exe` в PATH — как самый последний вариант.

Установщик задаёт `HERMES_GIT_BASH_PATH` явно, чтобы свежим PowerShell-сессиям не приходилось заново искать Bash. Переопределите этот путь, если хотите, чтобы Hermes использовал конкретный bash — например, системный Git Bash или bash, проброшенный из WSL через symlink.

**Подводный камень:** у MinGit другая раскладка, чем у полного Git for Windows — `bash` лежит в `usr\bin\bash.exe`, а не в `bin\bash.exe`. Hermes проверяет оба варианта. Если вы вручную распаковываете архив MinGit, берите **не busybox-вариант** (`MinGit-*-64-bit.zip`, а не `MinGit-*-busybox*.zip`) — busybox-сборки поставляют `ash` вместо `bash`, и большинство coreutils там отсутствует.

## Консоль UTF-8 в Windows

По умолчанию Python stdio в Windows использует активную кодовую страницу консоли — обычно `cp1252` или `cp437`. Баннер Hermes, список slash-команд, вывод инструментов, панели Rich и описания навыков содержат Unicode. Без дополнительной настройки это быстро приводит к `UnicodeEncodeError: 'charmap' codec can't encode character…`.

Исправление находится в `hermes_cli/stdio.py::configure_windows_stdio()`, которая вызывается рано в каждой точке входа (`cli.py::main`, `hermes_cli/main.py::main`, `gateway/run.py::main`). Она:

1. Переключает кодовую страницу консоли на CP_UTF8 (65001) через `kernel32.SetConsoleCP` / `SetConsoleOutputCP`.
2. Перенастраивает `sys.stdout` / `sys.stderr` / `sys.stdin` на UTF-8 с `errors='replace'`.
3. Выставляет `PYTHONIOENCODING=utf-8` и `PYTHONUTF8=1` (через `setdefault`, поэтому явные пользовательские значения имеют приоритет), чтобы дочерние Python-процессы тоже наследовали UTF-8.
4. Задаёт `EDITOR=notepad`, если не определены ни `EDITOR`, ни `VISUAL` (см. раздел про редактор ниже).

Идемпотентно. На не-Windows платформах ничего не делает.

**Как отключить:** `HERMES_DISABLE_WINDOWS_UTF8=1` в окружении возвращает старый путь stdio с кодовой страницей `cp1252`. Это полезно для бисекта бага кодировки, но в обычной работе почти наверняка не нужно.

## Редактор (`Ctrl-X Ctrl-E`, `/edit`)

До PR #21561 на Windows нажатие `Ctrl-X Ctrl-E` или команда `/edit` тихо ничего не делали. prompt_toolkit использует жёстко заданный POSIX-список fallback-редакторов (`/usr/bin/nano`, `/usr/bin/pico`, `/usr/bin/vi`, …), и на Windows он не срабатывает даже при полном Git for Windows.

Windows stdio shim в Hermes теперь задаёт `EDITOR=notepad` по умолчанию. Notepad есть в каждой Windows-сборке и работает как блокирующий редактор — `subprocess.call(["notepad", file])` ждёт, пока окно не закроется.

**Переопределения пользователя по-прежнему имеют приоритет** (они проверяются раньше `setdefault`):

| Редактор | Команда PowerShell |
|---|---|
| VS Code | `$env:EDITOR = "code --wait"` |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim | `$env:EDITOR = "nvim"` |
| Helix | `$env:EDITOR = "hx"` |

Флаг `--wait` у VS Code критически важен — без него редактор сразу возвращает управление, и Hermes получает пустой буфер.

Чтобы задать это постоянно, добавьте переменную в профиль PowerShell:

```powershell
# В $PROFILE
$env:EDITOR = "code --wait"
```

Или задайте её как пользовательскую переменную окружения в System Settings, чтобы каждый новый shell её подхватывал.

## `Ctrl+Enter` для новой строки в CLI

Windows Terminal передаёт `Ctrl+Enter` как отдельную последовательность клавиш. Hermes привязывает её к действию "вставить новую строку", чтобы можно было писать многострочные промпты в CLI, не переходя на схему `Esc` → `Enter`. Это работает в Windows Terminal, во встроенном терминале VS Code и в любом современном Windows console host, который понимает VT escape-sequences.

В старом `cmd.exe` `Ctrl+Enter` схлопывается в обычный `Enter` — тогда используйте `Esc Enter` или обновитесь до Windows Terminal (он бесплатен и по умолчанию установлен в Windows 11).

## Запуск gateway при входе в Windows

`hermes gateway install` в Windows использует **Scheduled Tasks** с запасным вариантом через Startup folder — без админских прав.

### Установка

```powershell
hermes gateway install
```

Что происходит под капотом:

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN HermesGateway` — регистрирует задачу, которая стартует при входе пользователя со стандартными, не повышенными правами. Без UAC-подтверждения.
2. Если schtasks блокирует group policy, Hermes откатывается к созданию ярлыка `start /min cmd.exe /d /c <wrapper>` в `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`. Эффект тот же, реализация чуть грубее.
3. Запускает gateway **через `pythonw.exe` и в detached-режиме** — не через `python.exe`. У `pythonw.exe` нет прикреплённой консоли, поэтому его не задевают `CTRL_C_EVENT`, которые распространяют соседние процессы в той же группе (это реальная проблема, из-за которой gateway раньше падал, когда вы жали Ctrl+C где-нибудь рядом).

Флаги запуска: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`.

### Управление

```powershell
hermes gateway status      # Сводный вид: schtasks + Startup folder + running PID
hermes gateway start       # Запускает задачу сейчас
hermes gateway stop        # Мягкий эквивалент SIGTERM (TerminateProcess через psutil)
hermes gateway restart
hermes gateway uninstall   # Удаляет запись schtasks, ярлык в Startup и pid file
```

`hermes gateway status` идемпотентен — вызывайте его хоть тысячу раз подряд, gateway он случайно не убьёт. (До PR #21561 он этого тихо и делал из-за `os.kill(pid, 0)`, который на уровне C конфликтовал с `CTRL_C_EVENT` — если интересна история, см. раздел про внутренности управления процессами ниже.)

### Почему не Windows Service?

Службы требуют админских прав на установку и привязывают жизненный цикл gateway к загрузке машины, а не к входу пользователя. Типичный пользователь Hermes хочет другое: вошёл в систему → gateway доступен, вышел из системы → gateway завершился. Scheduled Tasks делают именно это и не требуют повышения прав. Если вам действительно нужна служба, можно использовать `nssm` или `sc create` вручную — но, скорее всего, в этом нет необходимости.

## Схема данных

| Путь | Содержимое |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Git checkout + venv. Можно безболезненно удалить через `Remove-Item -Recurse` и установить заново. |
| `%LOCALAPPDATA%\hermes\git\` | PortableGit (только если его поставил сам установщик). |
| `%LOCALAPPDATA%\hermes\node\` | Portable Node.js (только если его поставил сам установщик). |
| `%LOCALAPPDATA%\hermes\bin\` | `hermes.cmd` shim, добавленный в User PATH. |
| `%USERPROFILE%\.hermes\` | Конфиг, auth, skills, sessions, logs. **Переживает переустановки.** |

Разделение намеренное: `%LOCALAPPDATA%\hermes` — это расходуемая инфраструктура (её можно снести, и one-liner восстановит всё заново). `%USERPROFILE%\.hermes` — это ваши данные: конфиг, память, навыки, история сессий. По структуре она совпадает с Linux-установкой. Синхронизируйте её между машинами, и Hermes будет следовать за вами.

**Переопределить `HERMES_HOME`:** задайте эту переменную окружения, чтобы указать другой каталог данных. Работает так же, как на Linux.

## Браузерный инструмент

Browser tool использует `agent-browser` — Node-helper для управления Chromium. На Windows:

- Установщик ставит `agent-browser` в PATH через npm.
- `shutil.which("agent-browser", path=...)` автоматически находит `.cmd` shim — `CreateProcessW` не умеет запускать shebang без расширения, поэтому Hermes всегда резолвит `.CMD`-обёртку. Не запускайте shebang-файл напрямую; всегда идите через `.cmd`.
- Playwright Chromium автоматически ставится при первом запуске (`npx playwright install chromium`). Если установка не удалась, `hermes doctor` покажет ошибку и подскажет, что именно нужно исправить.

## Практические заметки по запуску Hermes в Windows

### PATH после установки

Установщик добавляет `%LOCALAPPDATA%\hermes\bin` в ваш **User PATH** через `[Environment]::SetEnvironmentVariable`. Уже открытые терминалы это не подхватят — после установки откройте новое окно PowerShell или новую вкладку Windows Terminal. Закрыть и открыть заново, а не править `$env:PATH += …` вручную, если только вы не понимаете, зачем это делаете.

Проверка:

```powershell
Get-Command hermes        # should print C:\Users\<you>\AppData\Local\hermes\bin\hermes.cmd
hermes --version
```

### Переменные окружения

Hermes понимает и `$env:X` (область процесса), и пользовательские переменные окружения (постоянные, задаваемые в System Properties → Environment Variables). Обычно API-ключи кладут в `%USERPROFILE%\.hermes\.env` — как и на Linux:

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

Не кладите секреты в User environment variables, если только вы сознательно не хотите, чтобы их видели вообще все Windows-процессы. Обычно этого не нужно.

### Специфические для Windows переменные окружения

Они влияют только на нативные установки Windows:

| Переменная | Эффект |
|---|---|
| `HERMES_GIT_BASH_PATH` | Переопределяет поиск `bash.exe`. Можно указать любой bash — полный Git for Windows, bash из WSL через symlink, MSYS2, Cygwin. Установщик задаёт её автоматически. |
| `HERMES_DISABLE_WINDOWS_UTF8` | Если поставить `1`, отключает UTF-8 stdio shim и возвращает кодовую страницу локали. Полезно для бисектов бага кодировки. |
| `EDITOR` / `VISUAL` | Ваш редактор для `/edit` и `Ctrl-X Ctrl-E`. Если обе переменные пусты, Hermes по умолчанию использует `notepad`. |

## Удаление

Из PowerShell:

```powershell
hermes uninstall
```

Это "чистый" путь — он удаляет запись schtasks, ярлык в Startup, shim `hermes.cmd`, стирает `%LOCALAPPDATA%\hermes\hermes-agent\` и сокращает User PATH. При этом `%USERPROFILE%\.hermes\` остаётся нетронутым (ваш конфиг, auth, skills, sessions, logs) на случай, если вы потом захотите переустановить Hermes.

Если нужно снести вообще всё:

```powershell
hermes uninstall
Remove-Item -Recurse -Force "$env:USERPROFILE\.hermes"
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes"
```

Подкоманда `hermes uninstall` умеет и такой случай, когда запись schtasks была создана под другим именем задачи (старые установки) — она ищет задачу по install path, а не по жёстко зашитому имени.

## Внутренности управления процессами

Это фоновой материал — пропустите его, если не отлаживаете странность уровня "оно само себя убивает".

В Linux и macOS идиома `os.kill(pid, 0)` — это безвредная проверка прав: "жив ли этот PID и могу ли я его сигналить?" В Windows Python `os.kill` сопоставляет `sig=0` с `CTRL_C_EVENT` — у них одинаковое числовое значение 0 — и прокидывает вызов через `GenerateConsoleCtrlEvent(0, pid)`, который рассылает Ctrl+C **всю консольную группу процессов**, где живёт целевой PID. Это [bpo-14484](https://bugs.python.org/issue14484), открытый с 2012 года. Исправлять это не будут, потому что изменение сломает скрипты, которые полагаются на нынешнее поведение.

Следствие: любой код, который использовал `os.kill(pid, 0)` на Windows как "проверку, жив ли PID", тихо убивал цель. Hermes перевёл все такие места (14 в 11 файлах) на `gateway.status._pid_exists()`, а тот использует `psutil.pid_exists()` (который, в свою очередь, применяет `OpenProcess + GetExitCodeProcess` на Windows — без сигналов). Если вы пишете плагин или патч, используйте либо `psutil.pid_exists()`, либо `gateway.status._pid_exists()` — никогда не используйте `os.kill(pid, 0)`.

`scripts/check-windows-footguns.py` закрепляет это в CI: любой новый вызов `os.kill(pid, 0)` проваливает проверку `Windows footguns (blocking)`, если только рядом нет маркера `# windows-footgun: ok — <reason>`.

## Типичные ловушки

**`hermes: command not found` сразу после установки.**
Откройте новое окно PowerShell. Установщик уже добавил `%LOCALAPPDATA%\hermes\bin` в User PATH, но существующим shell-сессиям нужен перезапуск, чтобы его увидеть. Пока этого не сделали, можно запускать `& "$env:LOCALAPPDATA\hermes\bin\hermes.cmd"`.

**`WinError 193: %1 is not a valid Win32 application` при запуске инструмента.**
Вы наткнулись на shebang-скрипт, минуя `.cmd` shim. Hermes резолвит команды через `shutil.which(cmd, path=local_bin)`, чтобы PATHEXT подхватывал `.CMD`. Если вы вызываете инструмент по жёстко заданному пути, переключитесь на `.cmd`-вариант (например, `npx.cmd`, а не `npx`).

**`[scriptblock]::Create(...)` падает с `The assignment expression is not valid`.**
Ваш `install.ps1` был скачан с UTF-8 BOM. Форма `irm | iex` удаляет BOM автоматически, а `[scriptblock]::Create((irm ...))` — нет. Повторите установку в простой форме `irm | iex` или скачайте скрипт вручную и сохраните его без BOM через `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))`.

**Gateway не остаётся запущенным после перезапуска.**
Проверьте `hermes gateway status` — он объединяет запись schtasks, ярлык в Startup (если он использовался) и живой PID. Если schtasks зарегистрирован, но gateway не стартует, group policy может блокировать триггеры `ONLOGON`. Выполните `schtasks /Query /TN HermesGateway /V /FO LIST`, чтобы увидеть причину сбоя, или откатитесь к сценарию со Startup folder, удалив и установив Hermes заново с `HERMES_GATEWAY_FORCE_STARTUP=1`.

**`/edit` по-прежнему ничего не делает после задания `$env:EDITOR`.**
Вы задали переменную только в текущем процессе; закройте и откройте shell заново либо задайте её на уровне User в System Properties → Environment Variables. Проверить можно через `echo $env:EDITOR` в новом окне PowerShell.

**Browser tool запускается, но инструменты тайм-аутятся.**
Chromium ставится автоматически при первом запуске. Если установка сорвалась (rate-limited GitHub, сбой на Playwright CDN), запустите `hermes doctor` — он покажет, что именно не хватает, и выведет точную команду `npx playwright install chromium` для исправления.

**`agent-browser` падает с какой-то странной ошибкой версии Node.**
Установщик разворачивает Node 22 в `%LOCALAPPDATA%\hermes\node`, но в PATH раньше может стоять более старый системный Node 18. Либо поднимите Hermes node dir выше в PATH, либо удалите системную установку, если Node вам больше нигде не нужен.

**Китайские / японские / арабские символы показываются как `?` в CLI.**
UTF-8 stdio shim не активировался. Проверьте, что `HERMES_DISABLE_WINDOWS_UTF8` НЕ задан (`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`). Если переменная пуста, а `?` всё равно остаются, возможно, сам console host (очень старый `cmd.exe`) вообще не поддерживает UTF-8 — тогда переходите на Windows Terminal.

**Gateway не может отправить Telegram-фото — "`BadRequest: payload contains invalid characters`".**
Это не связано с Windows, но иногда первым проявляется именно там. Обычно причина в том, что путь к файлу содержит неэкранированные backslashes в JSON-body. Telegram должен получать пути, которые уже нормализовал Hermes, а не сырой Windows path — если это всплывает внутри кастомного плагина, убедитесь, что вы передаёте путь от Hermes, а не `str(Path(...))` из пользовательского ввода.

**Странности с кодировкой в духе "Works on my other machine" после `git pull`.**
Если вы редактировали конфиг Hermes или skill на Windows в редакторе без UTF-8 (старый Notepad, некоторые китайские IME), файл мог сохраниться с BOM. Hermes обычно терпит `utf-8-sig` при чтении config, но BOM внутри folded YAML scalar (`description: >`) может тихо сломать YAML-парсер. Сохраните файл как plain UTF-8 без BOM.

## Куда дальше

- **[Установка](/getting-started/installation)** — полная страница установки, включая Linux/macOS/WSL2/Termux.
- **[Руководство по Windows (WSL2)](/user-guide/windows-wsl-quickstart)** — если вам нужна POSIX-среда или dashboard с терминальной панелью.
- **[Справочник CLI](/reference/cli-commands)** — все подкоманды `hermes`.
- **[FAQ](/reference/faq)** — частые вопросы, не завязанные на Windows.
- **[Messaging gateway](/user-guide/messaging)** — запуск Telegram/Discord/Slack на Windows.
