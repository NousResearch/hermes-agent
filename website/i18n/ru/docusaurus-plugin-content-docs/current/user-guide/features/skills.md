---
sidebar_position: 2
title: "Система навыков"
description: "Документы знаний по запросу — поэтапное раскрытие, навыки под управлением агента и Skills Hub"
---

# Система навыков

Навыки — это документы знаний, которые агент подгружает по мере необходимости. Они следуют принципу **поэтапного раскрытия**, чтобы расходовать как можно меньше токенов, и совместимы с открытым стандартом [agentskills.io](https://agentskills.io/specification).

Все навыки хранятся в **`~/.hermes/skills/`** — это основной каталог и единственный источник истины. При новой установке встроенные навыки копируются из репозитория. Навыки, установленные через хаб, и навыки, созданные самим агентом, тоже попадают сюда. Hermes может изменять или удалять любой навык.

Hermes также может работать с **внешними каталогами навыков** — дополнительными папками, которые сканируются вместе с локальным каталогом. См. раздел [Внешние каталоги навыков](#external-skill-directories) ниже.

См. также:

- [Каталог встроенных навыков](/docs/reference/skills-catalog)
- [Официальный каталог optional-skills](/docs/reference/optional-skills-catalog)

## Внешние каталоги навыков {#external-skill-directories}

Помимо `~/.hermes/skills/`, Hermes может сканировать дополнительные каталоги с навыками. Это удобно, если вы храните собственную коллекцию навыков отдельно от основного хаба.

## Использование навыков

Каждый установленный навык автоматически доступен как слэш-команда:

```bash
# В CLI или на любой платформе обмена сообщениями:
/gif-search смешные коты
/axolotl помоги дообучить Llama 3 на моём датасете
/github-pr-workflow создай PR для рефакторинга auth
/plan составь план перехода на новый провайдер аутентификации

# Если ввести только имя навыка, он загрузится и позволит агенту уточнить запрос:
/excalidraw
```

Встроенный навык `plan` — хороший пример. Если выполнить `/plan [request]`, Hermes загрузит инструкции навыка, при необходимости запросит контекст, составит план внедрения в Markdown вместо немедленного выполнения задачи и сохранит результат в `.hermes/plans/` внутри активного рабочего каталога `workspace/backend`.

С навыками можно работать и в обычном разговоре:

```bash
hermes chat --toolsets skills -q "Какие навыки у тебя есть?"
hermes chat --toolsets skills -q "Покажи навык axolotl"
```

## Поэтапное раскрытие

Навыки используют токенно-экономичную схему загрузки:

```
Уровень 0: skills_list()         → [{name, description, category}, ...]   (~3k токенов)
Уровень 1: skill_view(name)      → полное содержимое + метаданные         (зависит)
Уровень 2: skill_view(name, path) → конкретный справочный файл            (зависит)
```

Агент загружает полный текст навыка только тогда, когда он действительно нужен.

## Формат SKILL.md {#skillmd-format}

```markdown
---
name: my-skill
description: Краткое описание того, что делает этот навык
version: 1.0.0
platforms: [macos, linux]     # Необязательно — ограничение по конкретным ОС
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    fallback_for_toolsets: [web]    # Необязательно — условная активация (см. ниже)
    requires_toolsets: [terminal]   # Необязательно — условная активация (см. ниже)
    config:                          # Необязательно — настройки config.yaml
      - key: my.setting
        description: "За что отвечает этот параметр"
        default: "value"
        prompt: "Подсказка для настройки"
---

# Название навыка

## Когда использовать
Условия, при которых этот навык нужен.

## Процедура
1. Шаг первый
2. Шаг второй

## Подводные камни
- Известные сбои и способы их устранения

## Проверка
Как убедиться, что всё сработало.
```

### Навыки, привязанные к платформам

Навыки могут ограничивать себя конкретными операционными системами с помощью поля `platforms`:

| Значение | Для чего подходит |
|-------|---|
| `macos` | macOS (Darwin) |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # Только macOS (например, iMessage, Apple Reminders, FindMy)
platforms: [macos, linux]     # macOS и Linux
```

Если поле задано, навык автоматически скрывается из системной инструкции, `skills_list()` и слэш-команд на несовместимых платформах. Если поле опущено, навык доступен на всех платформах.

## Передача вывода навыков и медиа {#skill-output-and-media-delivery}

Если ответ навыка (или любой ответ агента) содержит отдельный абсолютный путь к медиафайлу — например `/home/user/screenshots/diagram.png`, — шлюз автоматически распознаёт его, убирает из видимого текста и доставляет файл нативным способом в чат пользователя (Telegram photo, Discord attachment и т. д.), а не оставляет в сообщении сырой путь.

Для аудио директива `[[audio_as_voice]]` превращает аудиофайл в нативное голосовое сообщение на платформах, где это поддерживается (Telegram, WhatsApp).

### Форсировать доставку как документ: `[[as_document]]`

Иногда нужен **противоположный** режим по сравнению со встроенным предпросмотром: файл должен уйти как скачиваемое вложение, а не как сжатая картинка. Классический пример — скриншот или график высокого разрешения. Telegram `sendPhoto` сжимает такие файлы примерно до 200 КБ при ширине 1280 px и ломает читаемость. PNG размером 1-2 МБ, отправленный через `sendDocument`, сохраняет исходные байты.

Если ответ (или любой текст внутри него — обычно последняя строка) содержит буквальную директиву `[[as_document]]`, все медиафайлы, найденные в этом ответе, будут доставлены как документ или файл, а не как изображение:

```
Вот ваш отрендеренный график:

/home/user/.hermes/cache/chart-q4-2025.png

[[as_document]]
```

Директива удаляется перед доставкой, поэтому пользователь её не видит. Гранулярность намеренно сделана по принципу "всё или ничего" для всего ответа: если один раз указать `[[as_document]]`, каждый путь к изображению в этом ответе будет доставлен как документ. Это соответствует масштабу `[[audio_as_voice]]`.

Используйте это из навыка, если:

- вы создаёте скриншоты или графики, которые пользователю нужно сохранить как файлы (для редактирования в другом инструменте, архивации, передачи без потерь);
- при стандартном сжатии теряется важная детализация: мелкий текст, пиксельно точные схемы, цветочувствительные рендеры.

На платформах без отдельного пути для документов (например, SMS) используется доступный им механизм вложений.

### Условная активация (резервные навыки)

Навыки могут автоматически показываться или скрываться в зависимости от того, какие инструменты доступны в текущем сеансе. Это особенно полезно для **резервных навыков** — бесплатных или локальных альтернатив, которые должны появляться только тогда, когда премиальный инструмент недоступен.

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Показывать ТОЛЬКО когда эти toolsets недоступны
    requires_toolsets: [terminal]     # Показывать ТОЛЬКО когда эти toolsets доступны
    fallback_for_tools: [web_search]  # Показывать ТОЛЬКО когда эти конкретные tools недоступны
    requires_tools: [terminal]        # Показывать ТОЛЬКО когда эти конкретные tools доступны
```

| Поле | Поведение |
|-------|---|
| `fallback_for_toolsets` | Навык **скрывается**, когда перечисленные наборы инструментов доступны. Показывается, когда их нет. |
| `fallback_for_tools` | То же самое, но проверяются отдельные инструменты, а не наборы. |
| `requires_toolsets` | Навык **скрывается**, когда перечисленные наборы инструментов недоступны. Показывается, когда они есть. |
| `requires_tools` | То же самое, но проверяются отдельные инструменты. |

**Пример:** встроенный навык `duckduckgo-search` использует `fallback_for_toolsets: [web]`. Когда задан `FIRECRAWL_API_KEY`, toolset `web` доступен, и агент использует `web_search` — навык DuckDuckGo остаётся скрытым. Если ключ отсутствует, toolset `web` недоступен, и навык DuckDuckGo автоматически появляется как резервный вариант.

Навыки без условных полей ведут себя как и раньше — они всегда показываются.

## Безопасная настройка при загрузке

Навыки могут объявлять обязательные переменные окружения, не исчезая из обнаружения:

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

Если значение отсутствует, Hermes запрашивает его безопасно только тогда, когда навык действительно загружается в локальном CLI. Можно пропустить настройку и продолжать пользоваться навыком. В чатовых интерфейсах секреты никогда не запрашиваются — вместо этого они предлагают использовать `hermes setup` или `~/.hermes/.env` локально.

После настройки объявленные переменные окружения **автоматически передаются** в песочницы `execute_code` и `terminal` — скрипты навыка могут напрямую обращаться к `$TENOR_API_KEY`. Для не относящихся к навыкам переменных используйте параметр конфигурации `terminal.env_passthrough`. Подробности см. в разделе [Передача переменных окружения](/docs/user-guide/security#environment-variable-passthrough).

### Параметры конфигурации навыков

Навыки могут также объявлять не секретные параметры конфигурации (пути, предпочтения), которые хранятся в `config.yaml`:

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: "Путь к каталогу данных плагина"
        default: "~/myplugin-data"
        prompt: "Путь к каталогу данных плагина"
```

Параметры хранятся в `skills.config` внутри `config.yaml`. `hermes config migrate` запрашивает отсутствующие настройки, а `hermes config show` показывает их. Когда навык загружается, его разрешённые значения конфигурации автоматически подмешиваются в контекст, чтобы агент сразу знал настроенные значения.

См. [настройки навыков](/docs/user-guide/configuration#skill-settings) и [создание навыков — параметры конфигурации](/docs/developer-guide/creating-skills#config-settings-configyaml) для подробностей.

## Структура каталога навыков

```text
~/.hermes/skills/                  # Единственный источник истины
├── mlops/                         # Каталог по категориям
│   ├── axolotl/
│   │   ├── SKILL.md               # Основные инструкции (обязательно)
│   │   ├── references/            # Дополнительные документы
│   │   ├── templates/             # Форматы вывода
│   │   ├── scripts/               # Вспомогательные скрипты, вызываемые из навыка
│   │   └── assets/                # Дополнительные файлы
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # Навык, созданный агентом
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Состояние Skills Hub
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # Отслеживает предустановленные встроенные навыки
```

## Внешние каталоги навыков

Если вы храните навыки вне Hermes — например, в общем каталоге `~/.agents/skills/`, которым пользуются несколько ИИ-инструментов, — Hermes можно настроить на сканирование и этих каталогов.

Добавьте `external_dirs` в раздел `skills` в `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
    - ${SKILLS_REPO}/skills
```

Пути поддерживают разворачивание `~` и подстановку переменных окружения `${VAR}`.

### Как это работает

- **Только чтение**: внешние каталоги используются только для обнаружения навыков. Когда агент создаёт или редактирует навык, он всегда записывает его в `~/.hermes/skills/`.
- **Локальный приоритет**: если один и тот же навык существует и в локальном, и во внешнем каталоге, побеждает локальная версия.
- **Полная интеграция**: внешние навыки появляются в индексе системной инструкции, `skills_list`, `skill_view` и как слэш-команды `/skill-name` — ничем не отличаясь от локальных.
- **Несуществующие пути игнорируются без ошибок**: если настроенного каталога нет, Hermes просто пропускает его. Это удобно для общих каталогов, которых может не быть на каждой машине.

### Пример

```text
~/.hermes/skills/               # Локальный (основной, с чтением и записью)
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # Внешний (только чтение, общий)
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

Все четыре навыка отображаются в индексе навыков. Если вы создадите локальный навык с именем `my-custom-workflow`, он перекроет внешнюю версию.

## Наборы навыков

Наборы навыков — это небольшие YAML-файлы, которые группируют несколько навыков под одной слэш-командой. Когда вы запускаете `/<bundle-name>`, все указанные в наборе навыки загружаются сразу. Это удобно, если для одной и той же задачи всегда нужен один и тот же комплект навыков.

### Краткий пример

```bash
# Создаём набор для работы над backend-функциями
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work — review, test, PR workflow"
```

Затем в CLI или на любой платформе шлюза:

```
/backend-dev refactor the auth middleware
```

Агент получает все три навыка, загруженные в одно пользовательское сообщение, а любой текст после слэш-команды прикрепляется как инструкция пользователя.

### YAML-схема

Наборы хранятся в **`~/.hermes/skill-bundles/<slug>.yaml`** и выглядят так:

```yaml
name: backend-dev
description: Backend feature work — review, test, PR workflow.
skills:
  - github-code-review
  - test-driven-development
  - github-pr-workflow
instruction: |
  Always start by writing failing tests, then implement.
  Open the PR through the standard workflow with co-author tags.
```

Поля:
- `name` (необязательно — по умолчанию берётся имя файла без расширения) — отображаемое имя набора. Оно нормализуется в slug через дефисы для слэш-команды (`Backend Dev` → `/backend-dev`).
- `description` (необязательно) — короткий текст, который показывается в `/bundles` и `hermes bundles list`.
- `skills` (обязательно, непустой список) — имена навыков или пути относительно каталога навыков. Используйте тот же идентификатор, который вы бы передали в `/<skill-name>`.
- `instruction` (необязательно) — дополнительная инструкция, которая добавляется перед загруженным содержимым навыков. Удобно для фиксации того, как именно вы обычно используете эти навыки вместе.

### Управление наборами

```bash
# Список всех установленных наборов
hermes bundles list

# Показать один набор
hermes bundles show backend-dev

# Создать набор интерактивно (опустите --skill, чтобы вводить их по одному)
hermes bundles create research

# Перезаписать существующий набор
hermes bundles create backend-dev --skill ... --force

# Удалить набор
hermes bundles delete backend-dev

# Пересканировать ~/.hermes/skill-bundles/ и показать изменения
hermes bundles reload
```

Внутри сеанса чата `/bundles` показывает все установленные наборы и их навыки.

### Поведение

- **Наборы имеют приоритет над отдельными навыками**, если совпадает slug. Если вы назвали набор `research` и у вас есть навык `research`, `/research` вызовет именно набор. Это сделано намеренно — вы сами выбрали это имя.
- **Отсутствующие навыки пропускаются, а не приводят к ошибке.** Если в наборе указан `skill-foo`, а он не установлен, Hermes всё равно загрузит все найденные навыки и добавит заметку о том, что именно было пропущено.
- **Наборы работают во всех поверхностях** — в интерактивном CLI, TUI, чатовой панели и на любой платформе шлюза (Telegram, Discord, Slack и т. д.), потому что диспетчеризация централизована там же, где и отдельные команды навыков.
- **Наборы не сбрасывают кэш промпта.** При вызове они создают свежее пользовательское сообщение так же, как `/<skill-name>`, без изменения системной инструкции.

### Когда наборы лучше, чем ручная установка каждого навыка

Используйте набор, когда:
- вы всегда сочетаете одни и те же навыки для повторяющейся задачи (`/backend-dev`, `/release-prep`, `/incident-response`);
- вам нужен более короткий мысленный шаблон, чем последовательный ввод нескольких вызовов `/skill`;
- вы хотите развернуть командный "профиль задачи", положив YAML набора в общий репозиторий dotfiles и подключив его к `~/.hermes/skill-bundles/`.

Набор — это просто YAML-алиас. Он не устанавливает навыки за вас. Сами навыки должны уже существовать (в `~/.hermes/skills/` или во внешнем каталоге навыков). Иначе при вызове набора Hermes просто пропустит отсутствующие элементы.

## Управляемые агентом навыки (инструмент skill_manage)

Агент может создавать, обновлять и удалять собственные навыки с помощью инструмента `skill_manage`. Это его **процедурная память**: когда он находит нетривиальный рабочий процесс, он сохраняет этот подход как навык для повторного использования.

### Когда агент создаёт навыки

- После успешного выполнения сложной задачи (5+ вызовов инструментов)
- Когда он упёрся в ошибки или тупики и нашёл рабочий путь
- Когда пользователь поправил его подход
- Когда он обнаружил нетривиальный рабочий процесс

### Действия

| Действие | Для чего | Ключевые параметры |
|--------|----------|--------------------|
| `create` | Новый навык с нуля | `name`, `content` (полный `SKILL.md`), необязательно `category` |
| `patch` | Точечные правки (предпочтительно) | `name`, `old_string`, `new_string` |
| `edit` | Крупная структурная переработка | `name`, `content` (полная замена `SKILL.md`) |
| `delete` | Полностью удалить навык | `name` |
| `write_file` | Добавить или обновить вспомогательный файл | `name`, `file_path`, `file_content` |
| `remove_file` | Удалить вспомогательный файл | `name`, `file_path` |

:::tip
Для обновлений предпочтителен `patch` — он экономнее по токенам, чем `edit`, потому что в вызов попадает только изменившийся текст.
:::

## Skills Hub

Просматривайте, ищите, устанавливайте и управляйте навыками из онлайн-реестров, `skills.sh`, прямых well-known конечных точек и официальных дополнительных навыков.

### Частые команды

```bash
hermes skills browse                              # Просмотреть все навыки хаба (сначала official)
hermes skills browse --source official            # Просмотреть только официальные дополнительные навыки
hermes skills search kubernetes                   # Искать по всем источникам
hermes skills search react --source skills-sh     # Искать в каталоге skills.sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # Предпросмотр перед установкой
hermes skills install openai/skills/k8s           # Установка с проверкой безопасности
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # Прямой URL (один файл SKILL.md)
hermes skills install https://example.com/SKILL.md --name my-skill # Переопределить имя, если в frontmatter его нет
hermes skills list --source hub                   # Список навыков, установленных из хаба
hermes skills check                               # Проверить, какие навыки хаба изменились upstream
hermes skills update                              # Переустановить навыки хаба с учётом upstream-изменений
hermes skills audit                               # Повторно просканировать все навыки хаба на безопасность
hermes skills uninstall k8s                       # Удалить навык из хаба
hermes skills reset google-workspace              # Снять статус "user-modified" со встроенного навыка
hermes skills reset google-workspace --restore    # Заодно восстановить встроенную версию, удалив локальные правки
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # Экспортировать конфигурацию навыков
hermes skills tap add myorg/skills-repo           # Добавить собственный GitHub-источник
```

### Поддерживаемые источники хаба

| Источник | Пример | Примечание |
|--------|---------|------------|
| `official` | `official/security/1password` | Дополнительные навыки, поставляемые вместе с Hermes. |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | Доступно через `hermes skills search <query> --source skills-sh`. Hermes умеет сопоставлять псевдонимы slug'ов, если slug в skills.sh отличается от имени папки в репозитории. |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | Навыки, отдаваемые напрямую из `/.well-known/skills/index.json` на сайте. Ищите через сам сайт или URL документации. |
| `url` | `https://sharethis.chat/SKILL.md` | Прямой HTTP(S)-URL на одностраничный `SKILL.md`. Разрешение имени: frontmatter → slug URL → интерактивный запрос → флаг `--name`. |
| `github` | `openai/skills/k8s` | Прямая установка из GitHub-репозиториев и пользовательских tap'ов. |
| `clawhub`, `lobehub`, `claude-marketplace` | Идентификаторы конкретных источников | Интеграции с комьюнити- и marketplace-источниками. |

### Интегрированные хабы и реестры

Сейчас Hermes интегрируется со следующими экосистемами и источниками обнаружения навыков:

#### 1. Официальные дополнительные навыки (`official`)

Они поддерживаются в самом репозитории Hermes и устанавливаются с доверенной политикой по умолчанию.

- Каталог: [Official Optional Skills Catalog](../../reference/optional-skills-catalog)
- Источник в репозитории: `optional-skills/`
- Пример:

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh (`skills-sh`)

Это публичный каталог навыков от Vercel. Hermes может искать по нему напрямую, просматривать страницы навыков, сопоставлять псевдонимы slug'ов и устанавливать навыки из исходного репозитория.

- Каталог: [skills.sh](https://skills.sh/)
- CLI/tooling repo: [vercel-labs/skills](https://github.com/vercel-labs/skills)
- Официальный репозиторий навыков Vercel: [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- Пример:

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. Конечные точки навыков по механизму well-known (`well-known`)

Это обнаружение по URL через сайты, которые публикуют `/.well-known/skills/index.json`. Это не единый центральный хаб, а веб-конвенция для обнаружения навыков.

- Живой пример конечной точки: [Mintlify docs skills index](https://mintlify.com/docs/.well-known/skills/index.json)
- Реализация reference-сервера: [vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- Пример:

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. Прямые GitHub-навыки (`github`)

Hermes умеет устанавливать навыки прямо из GitHub-репозиториев и GitHub-based tap'ов. Это удобно, если вы уже знаете repo/path или хотите добавить свой собственный источник.

Встроенные taps по умолчанию:
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [huggingface/skills](https://github.com/huggingface/skills)
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- Пример:

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

#### 5. ClawHub (`clawhub`)

Сторонний marketplace навыков, интегрированный как комьюнити-источник.

- Сайт: [clawhub.ai](https://clawhub.ai/)
- Идентификатор источника Hermes: `clawhub`

#### 6. Репозитории в стиле Claude marketplace (`claude-marketplace`)

Hermes поддерживает marketplace-репозитории, которые публикуют совместимые с Claude plugin/marketplace manifest'ы.

Известные интегрированные источники:
- [anthropics/skills](https://github.com/anthropics/skills)
- [aiskillstore/marketplace](https://github.com/aiskillstore/marketplace)

Идентификатор источника Hermes: `claude-marketplace`

#### 7. LobeHub (`lobehub`)

Hermes умеет искать и конвертировать записи агентов из публичного каталога LobeHub в устанавливаемые навыки Hermes.

- Сайт: [LobeHub](https://lobehub.com/)
- Публичный индекс агентов: [chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- Базовый репозиторий: [lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- Идентификатор источника Hermes: `lobehub`

#### 8. Прямой URL (`url`)

Установить один файл `SKILL.md` можно прямо по HTTP(S)-URL. Это полезно, когда автор размещает навык на собственном сайте, без хаб-листа и без указания GitHub path. Hermes скачивает URL, парсит YAML frontmatter, проверяет безопасность и устанавливает навык.

- Идентификатор источника Hermes: `url`
- Идентификатор: сам URL (без префикса)
- Область действия: только одностраничный `SKILL.md`. Многофайловые навыки с `references/` или `scripts/` требуют манифеста и должны публиковаться через один из других источников выше.

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

Разрешение имени, по порядку:
1. Поле `name:` в YAML frontmatter файла SKILL.md (рекомендуется — у каждого корректно оформленного навыка оно есть).
2. Имя родительского каталога из URL path (например, `.../my-skill/SKILL.md` → `my-skill`, или `.../my-skill.md` → `my-skill`), если это допустимый идентификатор (`^[a-z][a-z0-9_-]*$`).
3. Интерактивный запрос в терминале с TTY.
4. На неинтерактивных поверхностях (`/skills install` в TUI, платформах шлюза, скриптах) — понятная ошибка с подсказкой использовать `--name`.

```bash
# В frontmatter нет name, а URL slug неудачен — укажите имя вручную:
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# Или внутри сеанса чата:
/skills install https://example.com/SKILL.md --name sharethis-chat
```

Уровень доверия всегда `community` — та же проверка безопасности выполняется, как и для любого другого источника. URL хранится как идентификатор установки, поэтому `hermes skills update` при обновлении автоматически повторно загружает данные с того же адреса.

### Проверка безопасности и `--force`

Все навыки, установленные через хаб, проходят **сканер безопасности**, который проверяет утечку данных, инъекцию промпта, разрушительные команды, признаки атак на цепочку поставок и другие угрозы.

`hermes skills inspect ...` теперь также показывает upstream-метаданные, если они доступны:
- URL репозитория
- URL страницы навыка в skills.sh
- команду установки
- еженедельные установки
- статусы upstream security audit
- URL well-known index / endpoint

Используйте `--force`, когда вы уже проверили сторонний навык и хотите переопределить неопасную политику блокировки:

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

Важное поведение:
- `--force` может переопределить policy block для findings уровня caution/warn.
- `--force` **не** переопределяет verdict `dangerous`.
- Официальные дополнительные навыки (`official/...`) считаются доверенными встроенными и не показывают панель предупреждения о стороннем источнике.

### Уровни доверия

| Уровень | Источник | Политика |
|-------|--------|--------|
| `builtin` | Поставляется вместе с Hermes | Всегда доверенный |
| `official` | `optional-skills/` в репозитории | Доверие уровня builtin, без предупреждения о стороннем источнике |
| `trusted` | Доверенные реестры и репозитории, такие как `openai/skills`, `anthropics/skills`, `huggingface/skills` | Более мягкая политика, чем у community-источников |
| `community` | Всё остальное (`skills.sh`, well-known endpoints, пользовательские GitHub-репозитории, большинство marketplace) | Неопасные findings можно переопределить через `--force`; verdict `dangerous` остаётся заблокированным |

### Жизненный цикл обновлений

Теперь хаб хранит достаточно provenance-данных, чтобы перепроверять upstream-копии установленных навыков:

```bash
hermes skills check          # Показать, какие установленные навыки хаба изменились upstream
hermes skills update         # Переустановить только те навыки, для которых есть обновления
hermes skills update react   # Обновить один конкретный установленный навык хаба
```

Это использует сохранённый source identifier вместе с текущим upstream bundle content hash, чтобы обнаруживать расхождения.

:::tip Ограничения GitHub API
Операции Skills Hub используют GitHub API, у которого лимит 60 запросов в час для неаутентифицированных пользователей. Если во время установки или поиска вы видите ошибки лимита, добавьте `GITHUB_TOKEN` в файл `.env`, чтобы поднять лимит до 5,000 запросов в час. Сообщение об ошибке при этом даёт конкретную подсказку.
:::

### Публикация собственного skill tap

Если вы хотите делиться отобранным набором навыков — для команды, организации или публично, — вы можете опубликовать их как **tap**: репозиторий GitHub, который другие пользователи Hermes подключают через `hermes skills tap add <owner/repo>`. Никакого сервера, регистрации в реестре или release pipeline. Просто каталог файлов `SKILL.md`.

#### Структура репозитория

Tap — это любой GitHub-репозиторий (публичный или приватный — для приватного нужен `GITHUB_TOKEN`), оформленный так:

```
owner/repo
├── skills/                       # путь по умолчанию; можно настроить для каждого tap
│   ├── my-workflow/
│   │   ├── SKILL.md              # обязательно
│   │   ├── references/           # дополнительные файлы
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # необязательно, но полезно
```

Правила:
- Каждый навык живёт в своей директории под корнем tap (по умолчанию `skills/`).
- Имя директории становится install slug навыка.
- В каждой директории навыка должен быть `SKILL.md` со стандартным [SKILL.md frontmatter](#skillmd-format) (`name`, `description`, а также необязательные `metadata.hermes.tags`, `version`, `author`, `platforms`, `metadata.hermes.config`).
- Подкаталоги вроде `references/`, `templates/`, `scripts/`, `assets/` скачиваются вместе с `SKILL.md` при установке.
- Директории навыков, имя которых начинается с `.` или `_`, игнорируются.

Hermes обнаруживает навыки, перечисляя все подкаталоги в пути tap и проверяя каждый на наличие `SKILL.md`.

#### Минимальный пример tap

```
my-org/hermes-skills
└── skills/
    └── deploy-runbook/
        └── SKILL.md
```

`skills/deploy-runbook/SKILL.md`:

```markdown
---
name: deploy-runbook
description: Our deployment runbook — services, rollback, Slack channels
version: 1.0.0
author: My Org Platform Team
metadata:
  hermes:
    tags: [deployment, runbook, internal]
---

# Deploy Runbook

Step 1: ...
```

После публикации этого репозитория в GitHub любой пользователь Hermes может подписаться и установить:

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### Нестандартные пути

Если ваши навыки лежат не в `skills/` (это часто бывает, когда вы добавляете поддерево `skills/` в уже существующий проект), отредактируйте запись tap в `~/.hermes/.hub/taps.json`:

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

CLI `hermes skills tap add` по умолчанию создаёт новый tap с `path: "skills/"`; при необходимости измените файл вручную, если нужен другой путь. `hermes skills tap list` показывает эффективный путь для каждого tap.

#### Установка отдельных навыков без добавления tap

Пользователи могут установить и один конкретный навык из любого публичного GitHub-репозитория, не добавляя весь репозиторий как tap:

```bash
hermes skills install owner/repo/skills/my-workflow
```

Это полезно, если вы хотите поделиться одним навыком, не заставляя пользователя подписываться на весь ваш реестр.

#### Уровни доверия для tap'ов

Новые tap'ы по умолчанию получают уровень доверия `community`. Навыки, установленные из них, проходят стандартный security scan и при первой установке показывают панель предупреждения о стороннем источнике. Если вашей организации или широко доверенному источнику нужен более высокий уровень доверия, добавьте репозиторий в `TRUSTED_REPOS` в `tools/skills_hub.py` (требуется PR в ядро Hermes).

#### Управление tap'ами

```bash
hermes skills tap list                                # показать все настроенные tap'ы
hermes skills tap add myorg/skills-repo               # добавить (путь по умолчанию: skills/)
hermes skills tap remove myorg/skills-repo            # удалить
```

Внутри уже запущенного сеанса:

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

Tap'ы хранятся в `~/.hermes/.hub/taps.json` (создаётся по мере необходимости).

## Обновление встроенных навыков (`hermes skills reset`)

Hermes поставляется с набором встроенных навыков в `skills/` внутри репозитория. При установке и при каждом `hermes update` выполняется синхронизация, которая копирует их в `~/.hermes/skills/` и записывает манифест в `~/.hermes/skills/.bundled_manifest`, сопоставляющий каждый навык с хешем содержимого на момент синхронизации (это **origin hash**).

При каждой синхронизации Hermes пересчитывает хеш вашей локальной копии и сравнивает его с origin hash:

- **Без изменений** → можно безопасно подтянуть upstream-изменения, скопировать новую встроенную версию и записать новый origin hash.
- **Изменён** → считается **user-modified** и навсегда пропускается, чтобы ваши правки не перетирались.

Защита полезная, но у неё есть одна тонкость. Если вы правите встроенный навык, а потом хотите отказаться от изменений и вернуться к встроенной версии просто копированием из `~/.hermes/hermes-agent/skills/`, манифест всё ещё хранит старый origin hash от последней успешной синхронизации. Ваш свежий скопированный вариант (текущий bundled hash) не совпадёт с этим устаревшим origin hash, и синхронизация продолжит считать его user-modified.

`hermes skills reset` — это аварийный выход:

```bash
# Безопасный вариант: очищает запись манифеста для этого навыка. Текущая копия сохраняется,
# но следующая синхронизация заново базируется на ней, чтобы будущие обновления работали нормально.
hermes skills reset google-workspace

# Полное восстановление: также удаляет локальную копию и снова копирует текущую
# встроенную версию. Используйте это, когда нужна чистая upstream-версия навыка.
hermes skills reset google-workspace --restore

# Неинтерактивный режим (например, в скриптах или TUI) — пропустить подтверждение --restore.
hermes skills reset google-workspace --restore --yes
```

Та же команда работает в чате как слэш-команда:

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note Профили
У каждого профиля свой `.bundled_manifest` внутри собственного `HERMES_HOME`, поэтому `hermes -p coder skills reset <name>` влияет только на этот профиль.
:::

### Слэш-команды (внутри чата)

Все те же команды работают и с `/skills`:

```text
/skills browse
/skills search react --source skills-sh
/skills search https://mintlify.com/docs --source well-known
/skills inspect skills-sh/vercel-labs/json-render/json-render-react
/skills install openai/skills/skill-creator --force
/skills check
/skills update
/skills reset google-workspace
/skills list
```

Официальные дополнительные навыки по-прежнему используют идентификаторы вроде `official/security/1password` и `official/migration/openclaw-migration`.
