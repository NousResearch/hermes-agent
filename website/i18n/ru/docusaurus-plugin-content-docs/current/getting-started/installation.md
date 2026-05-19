---
sidebar_position: 2
title: "Установка"
description: "Установите Hermes Agent на Linux, macOS, WSL2, нативный Windows (ранняя бета) или Android через Termux"
---

# Установка

Запустите Hermes Agent меньше чем за две минуты с помощью установщика одной командой.

## Быстрая установка

### Установка одной командой (Linux / macOS / WSL2)

Если вам нужна Git-установка, которая идёт за `main` и сразу подхватывает свежие изменения:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows (нативный PowerShell) — ранняя бета

:::warning РАННЯЯ БЕТА
Поддержка нативного Windows пока находится в **ранней бете**. Установка работает для типовых сценариев, но её пока не гоняли так же широко, как POSIX-установщики. Если наткнётесь на шероховатости, пожалуйста, [сообщайте о проблемах](https://github.com/NousResearch/hermes-agent/issues). Для самого проверенного варианта на Windows сегодня используйте линейную установку Linux/macOS внутри **WSL2**.
:::

Откройте PowerShell и выполните:

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

Установщик берёт на себя **всё**: `uv`, Python 3.11, Node.js 22, `ripgrep`, `ffmpeg` и **портативный Git Bash** (PortableGit — самодостаточный дистрибутив Git for Windows, который поставляет `bash.exe` и весь POSIX toolchain, используемый Hermes для shell-команд; на 32-битном Windows установщик переключается на MinGit, где bash нет и terminal-tool / agent-browser функции отключаются). Он клонирует репозиторий в `%LOCALAPPDATA%\hermes\hermes-agent`, создаёт виртуальное окружение и добавляет `hermes` в **User PATH**. После установки перезапустите терминал (или откройте новое окно PowerShell), чтобы PATH подхватился.

**Как обрабатывается Git:**
1. Если `git` уже есть в PATH, установщик использует его.
2. Иначе он скачивает портативный **PortableGit** (~50MB, из официального GitHub-релиза `git-for-windows`) и распаковывает его в `%LOCALAPPDATA%\hermes\git`. Админские права не нужны. Всё изолировано и не конфликтует с системным Git, каким бы он ни был. (На 32-битном Windows установщик откатывается на MinGit, потому что PortableGit публикуется только для 64-bit и ARM64; зависящие от bash возможности Hermes на 32-битных хостах работать не будут.)

**Почему не winget?** Раньше Git ставили через `winget install Git.Git`, но winget часто ломается, если системный Git уже частично повреждён именно в тот момент, когда пользователю нужен просто рабочий установщик. Портативный Git обходит winget, реестр установщика Windows и любой уже установленный system Git. Если когда-нибудь сломается сам Git-установщик Hermes, удалите `%LOCALAPPDATA%\hermes\git` и запустите установку заново — без последствий для системы и без сценариев с деинсталляцией.

Установщик также выставляет `HERMES_GIT_BASH_PATH` на найденный `bash.exe`, чтобы Hermes в свежих shell-сессиях детерминированно находил Bash.

Если вы предпочитаете WSL2, Linux-установщик выше работает и там; native и WSL-установки могут сосуществовать без конфликтов (нативные данные живут в `%LOCALAPPDATA%\hermes`, данные WSL — в `~/.hermes`).

**Альтернатива: графический установщик.** Для Hermes Desktop также доступен тонкий GUI-установщик: скачайте Hermes Desktop, запустите `.exe`, и при первом старте он сам вызовет `install.ps1`, чтобы подготовить Python (через `uv`), Node, PortableGit и остальные зависимости. Desktop-приложение и CLI, установленный через PowerShell, используют одни и те же папки установки и данных, так что можно пользоваться любым вариантом или обоими сразу. Подробности см. в [руководстве по Windows (нативный режим)](/user-guide/windows-native).

### Android / Termux

Hermes теперь поддерживает и путь установки, учитывающий Termux:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Установщик автоматически распознаёт Termux и переключается на проверенный Android-flow:
- использует `pkg` для системных зависимостей (`git`, `python`, `nodejs`, `ripgrep`, `ffmpeg`, build tools)
- создаёт виртуальное окружение через `python -m venv`
- автоматически экспортирует `ANDROID_API_LEVEL` для сборки Android wheels
- сначала пробует широкий extra `.[termux-all]`, затем откатывается на более компактный `.[termux]` (и, если нужно, на базовую установку)
- по умолчанию пропускает непроверенный bootstrap для браузера и WhatsApp

Если нужен полностью явный сценарий, откройте отдельное [руководство по Termux](/getting-started/termux).

:::note Паритет возможностей Windows (ранняя бета)

Нативный Windows сейчас в **ранней бете**. Всё, кроме встроенного терминала чата в dashboard, запускается нативно на Windows:
- **CLI (`hermes chat`, `hermes setup`, `hermes gateway`, …)** — native, использует ваш терминал по умолчанию
- **Gateway (Telegram, Discord, Slack, …)** — native, работает как фоновый PowerShell-процесс
- **Cron scheduler** — native
- **Browser tool** — native (Chromium через Node.js)
- **MCP servers** — native (поддерживаются и stdio, и HTTP transport)
- **Dashboard `/chat` terminal pane** — **только WSL2** (использует POSIX PTY; у native Windows нет эквивалента). Остальная часть dashboard (sessions, jobs, metrics) работает нативно — ограничение касается только встроенной PTY-вкладки.

Если столкнётесь с проблемой кодировки и захотите вернуться к старому cp1252 stdio path, установите `HERMES_DISABLE_WINDOWS_UTF8=1` в окружении. Это удобно для бисектов.
:::

### Что делает установщик

Установщик автоматически закрывает все базовые шаги: зависимости (Python, Node.js, ripgrep, ffmpeg), клон репозитория, виртуальное окружение, глобальный `hermes` и настройку LLM-провайдера. После этого можно сразу начинать работу.

#### Схема установки

Где именно окажутся файлы, зависит от того, ставите ли вы Hermes как обычный пользователь или от root:

| Установщик | Где лежит код | `hermes` binary | Папка данных |
|---|---|---|---|
| pip install | Python site-packages | `~/.local/bin/hermes` (console_scripts) | `~/.hermes/` |
| Per-user (git installer) | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` (symlink) | `~/.hermes/` |
| Root-mode (`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/` (или `$HERMES_HOME`) |

Root-mode **FHS layout** (`/usr/local/lib/…`, `/usr/local/bin/hermes`) совпадает с тем, как обычно раскладывают системные developer tools в Linux. Это удобно для shared-machine deployments, когда одна системная установка должна обслуживать всех пользователей. Персональные настройки (auth, skills, sessions) по-прежнему живут в `~/.hermes/` каждого пользователя или в явном `HERMES_HOME`.

### После установки

Перезагрузите shell и начните чат:

```bash
source ~/.bashrc   # or: source ~/.zshrc
hermes             # Start chatting!
```

Чтобы позже изменить отдельные настройки, используйте специальные команды:

```bash
hermes model          # Choose your LLM provider and model
hermes tools          # Configure which tools are enabled
hermes gateway setup  # Set up messaging platforms
hermes config set     # Set individual config values
hermes setup          # Or run the full setup wizard to configure everything at once
```

---

## Требования

**pip install:** кроме Python 3.11+ ничего не требуется. Всё остальное установщик возьмёт на себя.

**Git installer:** единственное обязательное условие — **Git**. Дальше установщик автоматизирует всё сам:

- **uv** (быстрый пакетный менеджер для Python)
- **Python 3.11** (через uv, без sudo)
- **Node.js v22** (для браузерной автоматизации и WhatsApp bridge)
- **ripgrep** (быстрый поиск по файлам)
- **ffmpeg** (конвертация аудио для TTS)

:::info
Вручную ставить Python, Node.js, ripgrep или ffmpeg **не нужно**. Установщик сам поймёт, чего не хватает, и поставит это. Просто убедитесь, что `git` доступен (`git --version`).
:::

:::tip Для пользователей Nix
Если вы используете Nix (на NixOS, macOS или Linux), у проекта есть отдельный путь установки с Nix flake, декларативным NixOS module и опциональным контейнерным режимом. Смотрите **[настройку Nix и NixOS](/getting-started/nix-setup)**.
:::

---

## Ручная / developer-установка

Если хотите клонировать репозиторий и ставить из исходников — для контрибьюта, запуска с конкретной ветки или полного контроля над virtualenv — см. раздел [настройки для разработки](/developer-guide/contributing#development-setup) в руководстве для участников.

---

## Установка без sudo / для service user

Запуск Hermes от отдельного непривилегированного пользователя (например, systemd service account `hermes` или любого пользователя без `sudo`) поддерживается. Единственное место, где реально нужен root, — это шаг `--with-deps` у Playwright: он ставит системные библиотеки (`libnss3`, `libxkbcommon` и т. п.), которые использует Chromium. Установщик проверяет наличие sudo и аккуратно деградирует, если его нет: он установит Chromium в локальный cache Playwright для service user и выведет точную команду, которую должен выполнить администратор отдельно.

**Рекомендуемая схема (Debian/Ubuntu):**

1. **Один раз, от admin-пользователя с sudo**, поставьте системные библиотеки для Chromium:
   ```bash
   sudo npx playwright install-deps chromium
   ```
   (Команду можно запускать из любой директории — `npx` сам подтянет Playwright.)

2. **От имени непривилегированного service user** запустите обычный установщик. Он увидит отсутствие sudo, пропустит `--with-deps` и поставит Chromium в локальный cache Playwright:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```

   Если хотите вообще пропустить шаг Playwright — например, когда работаете headless и браузерная автоматизация не нужна, — добавьте `--skip-browser`:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-browser
   ```

3. **Сделайте `hermes` доступным в shell service user.** Установщик пишет launcher в `~/.local/bin/hermes`. У service account часто урезанный PATH без `~/.local/bin`. Добавьте его в профиль пользователя или создайте системный symlink:
   ```bash
   # Option A — add to the service user's profile
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Option B — symlink system-wide (run as an admin)
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **Проверка:** `hermes doctor` должен пройти без ошибок. Если видите `ModuleNotFoundError: No module named 'dotenv'`, значит вы запускаете source-версию `hermes` (`~/.hermes/hermes-agent/hermes`) системным Python вместо launcher из venv (`~/.hermes/hermes-agent/venv/bin/hermes`) — вернитесь к шагу 3.

Та же схема работает на Arch (установщик использует pacman с той же логикой определения sudo), Fedora/RHEL и openSUSE — там `--with-deps` вообще не поддерживается, поэтому системные библиотеки всегда ставит администратор отдельно. Соответствующие `dnf`/`zypper` команды установщик печатает сам.

---

## Устранение неполадок

| Проблема | Решение |
|---------|----------|
| `hermes: command not found` | Перезагрузите shell (`source ~/.bashrc`) или проверьте PATH |
| `API key not set` | Запустите `hermes model`, чтобы настроить провайдера, или `hermes config set OPENROUTER_API_KEY your_key` |
| После обновления пропала конфигурация | Выполните `hermes config check`, затем `hermes config migrate` |

Для более глубокой диагностики запустите `hermes doctor` — он покажет, чего именно не хватает и как это исправить.

## Автоопределение способа установки

Hermes сам определяет, был ли он установлен через `pip`, Git installer, Homebrew или NixOS, и `hermes update` выводит соответствующую команду обновления для этого способа. Отдельный env var не нужен — определение строится по схеме установки (Python site-packages, `~/.hermes/hermes-agent/`, Homebrew prefix или путь Nix store). `hermes doctor` тоже показывает распознанный способ в сводке окружения.
