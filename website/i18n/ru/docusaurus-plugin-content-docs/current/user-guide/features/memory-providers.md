---
sidebar_position: 4
title: "Провайдеры памяти"
description: "Внешние плагины провайдеров памяти — Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory"
---

# Провайдеры памяти

Hermes Agent поставляется с восемью внешними плагинами-провайдерами памяти, которые добавляют постоянную межсеансовую память поверх встроенных `MEMORY.md` и `USER.md`. Активным может быть только **один** внешний провайдер одновременно — при этом встроенная память всегда работает параллельно.

## Быстрый старт

```bash
hermes memory setup      # интерактивный выбор + настройка
hermes memory status     # проверить, что активно
hermes memory off        # отключить внешний провайдер
```

Активный провайдер памяти можно также выбрать через `hermes plugins` → `Плагины провайдеров` → `Провайдер памяти`.

Или задать вручную в `~/.hermes/config.yaml`:

```yaml
memory:
  provider: openviking   # или honcho, mem0, hindsight, holographic, retaindb, byterover, supermemory
```

## Как это работает

Когда активен провайдер памяти, Hermes автоматически:

1. **Подмешивает контекст провайдера** в системную инструкцию (то, что знает провайдер)
2. **Заранее подгружает релевантные воспоминания** перед каждым ходом (в фоне, неблокирующе)
3. **Синхронизирует ходы разговора** с провайдером после каждого ответа
4. **Извлекает воспоминания по завершении сеанса** (для провайдеров, которые это поддерживают)
5. **Дублирует записи встроенной памяти** во внешний провайдер
6. **Добавляет инструменты, специфичные для провайдера**, чтобы агент мог искать, сохранять и управлять воспоминаниями

Встроенная память (`MEMORY.md` / `USER.md`) продолжает работать как и раньше. Внешний провайдер — это дополнительный слой, а не замена.

## Доступные провайдеры

### Honcho

Ориентированная на ИИ система межсеансового моделирования пользователя с диалектическим рассуждением, межсеансовой подстановкой контекста, семантическим поиском и устойчивыми выводами. Базовый контекст теперь включает сводку сеанса, пользовательское представление и карточки участников, так что агент помнит, о чём уже говорили.

| | |
|---|---|
| **Лучше всего подходит** | Многоагентные системы с межсеансовым контекстом и согласованием между пользователем и агентом |
| **Требует** | `pip install honcho-ai` + [API-ключ](https://app.honcho.dev) или самостоятельно размещённый инстанс |
| **Хранение данных** | Honcho Cloud или собственный хост |
| **Стоимость** | Тарифы Honcho (cloud) / бесплатно (самостоятельно размещаемый вариант) |

**Инструменты (5):** `honcho_profile` (чтение/обновление карточки участника), `honcho_search` (семантический поиск), `honcho_context` (контекст сеанса — сводка, представление, карточка, сообщения), `honcho_reasoning` (выводы, синтезированные LLM), `honcho_conclude` (создавать и удалять заключения)

**Архитектура:** Двухслойная подстановка контекста — базовый слой (сводка сеанса + представление + карточка участника, обновляется по `contextCadence`) плюс диалектическая надстройка (LLM-рассуждение, обновляется по `dialecticCadence`). Диалектика автоматически выбирает промпты для холодного старта (общие факты о пользователе) или прогретые промпты (контекст текущего сеанса) в зависимости от наличия базового контекста.

**Три независимых параметра конфигурации** отдельно управляют стоимостью и глубиной:

- `contextCadence` — как часто обновляется базовый слой (частота API-вызовов)
- `dialecticCadence` — как часто срабатывает диалектическая LLM (частота LLM-вызовов)
- `dialecticDepth` — сколько проходов `.chat()` выполняется за одну диалектическую итерацию (1–3, глубина рассуждения)

**Мастер настройки:**
```bash
hermes memory setup        # выбрать "honcho" — запустится post-setup именно для Honcho
```

Старая команда `hermes honcho setup` всё ещё работает (она теперь перенаправляется на `hermes memory setup`), но регистрируется только после того, как Honcho выбран активным провайдером памяти.

**Конфиг:** `$HERMES_HOME/honcho.json` (локальная для профиля) или `~/.honcho/config.json` (глобальная). Порядок поиска: `$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`. См. [справочник по конфигу](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md) и [руководство по интеграции Honcho](https://docs.honcho.dev/v3/guides/integrations/hermes).

<details>
<summary>Полный справочник по конфигу</summary>

| Ключ | Значение по умолчанию | Описание |
|-----|-----------------------|----------|
| `apiKey` | -- | API-ключ из [app.honcho.dev](https://app.honcho.dev) |
| `baseUrl` | -- | Базовый URL для самостоятельно размещаемого Honcho |
| `peerName` | -- | Идентификатор пользовательского участника |
| `aiPeer` | ключ хоста | Идентификатор ИИ-участника (один на профиль) |
| `workspace` | ключ хоста | Общий идентификатор workspace |
| `contextTokens` | `null` (без лимита) | Бюджет токенов для автоматически подставляемого контекста на ход. Усечение идёт по границам слов |
| `contextCadence` | `1` | Минимум ходов между вызовами `context()` API (обновление базового слоя) |
| `dialecticCadence` | `2` | Минимум ходов между LLM-вызовами `peer.chat()`. Рекомендуется 1–5. Применяется только в режимах `hybrid`/`context` |
| `dialecticDepth` | `1` | Количество проходов `.chat()` за одну диалектическую итерацию. Ограничивается диапазоном 1–3. Проход 0: промпт холодного/прогретого старта, проход 1: самопроверка, проход 2: согласование |
| `dialecticDepthLevels` | `null` | Необязательный массив уровней рассуждения на каждый проход, например `["minimal", "low", "medium"]`. Переопределяет пропорциональные значения по умолчанию |
| `dialecticReasoningLevel` | `'low'` | Базовый уровень рассуждения: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | Если `true`, модель может переопределять уровень рассуждения для каждого вызова через параметр инструмента |
| `dialecticMaxChars` | `600` | Максимум символов диалектического результата, которые подставляются в системную инструкцию |
| `recallMode` | `'hybrid'` | `hybrid` (автоподстановка + инструменты), `context` (только подстановка), `tools` (только инструменты) |
| `writeFrequency` | `'async'` | Когда сбрасывать сообщения: `async` (фоновой поток), `turn` (синхронно), `session` (пакетно в конце) или целое N |
| `saveMessages` | `true` | Сохранять ли сообщения в API Honcho |
| `observationMode` | `'directional'` | `directional` (всё включено) или `unified` (общий пул). Переопределяется объектом `observation` |
| `messageMaxChars` | `25000` | Максимум символов на сообщение (при превышении сообщение режется на части) |
| `dialecticMaxInputChars` | `10000` | Максимум символов входа для запроса `peer.chat()` |
| `sessionStrategy` | `'per-directory'` | `per-directory`, `per-repo`, `per-session`, `global` |

</details>

<details>
<summary>Минимальный honcho.json (cloud)</summary>

```json
{
  "apiKey": "your-key-from-app.honcho.dev",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

<details>
<summary>Минимальный honcho.json (самостоятельно размещаемый вариант)</summary>

```json
{
  "baseUrl": "http://localhost:8000",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

:::tip Переезд с `hermes honcho`
Если вы раньше использовали `hermes honcho setup`, ваш конфиг и все серверные данные сохранены. Просто включите систему снова через мастер настройки или вручную задайте `memory.provider: honcho`, чтобы активировать её через новую систему.
:::

**Многопользовательская схема:**

Honcho моделирует разговоры как обмен сообщениями между участниками — один пользовательский участник и один ИИ-участник на каждый профиль Hermes, все в одном workspace. Workspace — это общее окружение: пользовательский участник общий для всех профилей, а каждый ИИ-участник — отдельная идентичность. Каждый ИИ-участник строит независимое представление / карточку на основе собственных наблюдений, так что профиль `coder` остаётся ориентированным на код, а `writer` — на редактуру, даже если пользователь один и тот же.

Сопоставление:

| Понятие | Что это такое |
|---------|---------------|
| **Workspace** | Общее окружение. Все профили Hermes в одном workspace видят одну и ту же пользовательскую идентичность |
| **Пользовательский участник** (`peerName`) | Человек. Общий для всех профилей в workspace |
| **ИИ-участник** (`aiPeer`) | Один на каждый профиль Hermes. Host key `hermes` → по умолчанию; `hermes.<profile>` для остальных |
| **Observation** | Переключатели на уровне участника, управляющие тем, чьи сообщения моделирует Honcho. `directional` (по умолчанию, все четыре включены) или `unified` (пул с одним наблюдателем) |

### Новый профиль, новый участник Honcho

```bash
hermes profile create coder --clone
```

`--clone` создаёт блок `hermes.coder` в `honcho.json` с `aiPeer: "coder"`, общим `workspace`, унаследованным `peerName`, `recallMode`, `writeFrequency`, `observation` и т. д. ИИ-участник создаётся в Honcho заранее, чтобы он существовал до первого сообщения.

### Существующие профили: синхронизация участников Honcho

```bash
hermes honcho sync
```

Сканирует все профили Hermes, создаёт host-блоки для тех профилей, у которых их нет, наследует настройки из блока `hermes` по умолчанию и заранее создаёт новых ИИ-участников. Команда идемпотентна — пропускает профили, у которых host-блок уже есть.

### Наблюдение на уровне профиля

Каждый блок host может независимо переопределять конфигурацию наблюдения. Например, профиль, ориентированный на код, где ИИ-участник наблюдает за пользователем, но не моделирует себя сам:

```json
"hermes.coder": {
  "aiPeer": "coder",
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": false, "observeOthers": true }
  }
}
```

**Переключатели наблюдения (набор на каждого участника):**

| Переключатель | Эффект |
|--------------|--------|
| `observeMe` | Honcho строит представление этого участника на основе его собственных сообщений |
| `observeOthers` | Этот участник наблюдает сообщения другого участника (питает межучастническое рассуждение) |

Пресеты через `observationMode`:

- **`"directional"`** (по умолчанию) — все четыре флага включены. Полное взаимное наблюдение; включает диалектику между участниками
- **`"unified"`** — пользователь `observeMe: true`, AI `observeOthers: true`, остальные false. Пул с одним наблюдателем; ИИ моделирует пользователя, но не себя, а пользовательский участник моделирует только себя

Серверные переключатели, заданные в [панели Honcho](https://app.honcho.dev), имеют приоритет над локальными значениями по умолчанию — они синхронизируются обратно при инициализации сеанса.

Полный справочник по наблюдению см. на [странице Honcho](./honcho.md#observation-directional-vs-unified).

<details>
<summary>Полный пример honcho.json (мультипрофиль)</summary>

```json
{
  "apiKey": "your-key",
  "workspace": "hermes",
  "peerName": "eri",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "hybrid",
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "dialecticReasoningLevel": "low",
      "dialecticDynamic": true,
      "dialecticCadence": 2,
      "dialecticDepth": 1,
      "dialecticMaxChars": 600,
      "contextCadence": 1,
      "messageMaxChars": 25000,
      "saveMessages": true
    },
    "hermes.coder": {
      "enabled": true,
      "aiPeer": "coder",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "tools",
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    },
    "hermes.writer": {
      "enabled": true,
      "aiPeer": "writer",
      "workspace": "hermes",
      "peerName": "eri"
    }
  },
  "sessions": {
    "/home/user/myproject": "myproject-main"
  }
}
```

</details>

См. [справочник по конфигу](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md) и [руководство по интеграции Honcho](https://docs.honcho.dev/v3/guides/integrations/hermes).

---

### OpenViking

База контекста от Volcengine (ByteDance) с иерархией знаний в стиле файловой системы, многоуровневым поиском и автоматическим извлечением памяти по шести категориям.

| | |
|---|---|
| **Лучше всего подходит** | Самостоятельно размещаемая система управления знаниями со структурированным просмотром |
| **Требует** | `pip install openviking` + запущенный сервер |
| **Хранение данных** | Self-hosted (локально или в облаке) |
| **Стоимость** | Бесплатно (open-source, AGPL-3.0) |

**Инструменты:** `viking_search` (семантический поиск), `viking_read` (tiered: abstract/overview/full), `viking_browse` (навигация по файловой системе), `viking_remember` (сохранение фактов), `viking_add_resource` (ингест URL/документов)

**Настройка:**
```bash
# Сначала запустите сервер OpenViking
pip install openviking
openviking-server

# Затем настройте Hermes
hermes memory setup    # выберите "openviking"
# Или вручную:
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

**Ключевые возможности:**
- Многоуровневая загрузка контекста: L0 (~100 токенов) → L1 (~2k) → L2 (полный)
- Автоматическое извлечение памяти по завершении сеанса (profile, preferences, entities, events, cases, patterns)
- URI-схема `viking://` для иерархического просмотра знаний

---

### Mem0

Серверное извлечение фактов с помощью LLM, семантический поиск, reranking и автоматическая дедупликация.

| | |
|---|---|
| **Лучше всего подходит** | Управление памятью без ручного участия: Mem0 сам извлекает факты |
| **Требует** | `pip install mem0ai` + API-ключ |
| **Хранение данных** | Mem0 Cloud |
| **Стоимость** | Тарифы Mem0 |

**Инструменты:** `mem0_profile` (все сохранённые воспоминания), `mem0_search` (семантический поиск + reranking), `mem0_conclude` (сохранение фактов дословно)

**Настройка:**
```bash
hermes memory setup    # выберите "mem0"
# Или вручную:
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**Конфиг:** `$HERMES_HOME/mem0.json`

| Ключ | По умолчанию | Описание |
|-----|-------------|----------|
| `user_id` | `hermes-user` | Идентификатор пользователя |
| `agent_id` | `hermes` | Идентификатор агента |

---

### Hindsight

Долговременная память с графом знаний, разрешением сущностей и многостратегическим поиском. Инструмент `hindsight_reflect` обеспечивает синтез между разными областями памяти, которого нет ни у одного другого провайдера. Автоматически сохраняет полные ходы разговора (включая вызовы инструментов) и ведёт отслеживание документов на уровне сеанса.

| | |
|---|---|
| **Лучше всего подходит** | Вспоминание на основе графа знаний с отношениями между сущностями |
| **Требует** | Cloud: API-ключ с [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io). Local: LLM API-ключ (OpenAI, Groq, OpenRouter и т. д.) |
| **Хранение данных** | Hindsight Cloud или локальный встроенный PostgreSQL |
| **Стоимость** | Тарифы Hindsight (cloud) или бесплатно (local) |

**Инструменты:** `hindsight_retain` (сохранение с извлечением сущностей), `hindsight_recall` (мультистратегический поиск), `hindsight_reflect` (синтез между разными областями памяти)

**Настройка:**
```bash
hermes memory setup    # выберите "hindsight"
# Или вручную:
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

Мастер настройки автоматически ставит зависимости и устанавливает только то, что нужно для выбранного режима (`hindsight-client` для cloud, `hindsight-all` для local). Требуется `hindsight-client >= 0.4.22` (при старте сеанса автоматически обновляется, если версия устарела).

**UI локального режима:** `hindsight-embed -p hermes ui start`

**Конфиг:** `$HERMES_HOME/hindsight/config.json`

| Ключ | По умолчанию | Описание |
|-----|-------------|----------|
| `mode` | `cloud` | `cloud` или `local` |
| `bank_id` | `hermes` | Идентификатор банка памяти |
| `recall_budget` | `mid` | Глубина recall: `low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid` (context + tools), `context` (только автоподстановка), `tools` (только инструменты) |
| `auto_retain` | `true` | Автоматически сохранять ходы разговора |
| `auto_recall` | `true` | Автоматически вспоминать память перед каждым ходом |
| `retain_async` | `true` | Обрабатывать retain асинхронно на сервере |
| `retain_context` | `conversation between Hermes Agent and the User` | Метка контекста для сохраняемых воспоминаний |
| `retain_tags` | — | Теги по умолчанию для сохраняемых воспоминаний; объединяются с тегами инструментов на каждый вызов |
| `retain_source` | — | Необязательный `metadata.source`, прикрепляемый к сохранённым воспоминаниям |
| `retain_user_prefix` | `User` | Метка, используемая перед репликами пользователя в авто-сохраняемых транскриптах |
| `retain_assistant_prefix` | `Assistant` | Метка, используемая перед репликами ассистента в авто-сохраняемых транскриптах |
| `recall_tags` | — | Теги для фильтрации recall |

См. [README плагина](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md) для полного справочника по конфигу.

---

### Holographic

Локальное хранилище фактов на SQLite с полнотекстовым поиском FTS5, оценкой доверия и HRR (Holographic Reduced Representations) для составных алгебраических запросов.

| | |
|---|---|
| **Лучше всего подходит** | Локальная память без внешних зависимостей, но с продвинутым поиском |
| **Требует** | Ничего (SQLite всегда доступен). NumPy необязателен для HRR-алгебры. |
| **Хранение данных** | Локальный SQLite |
| **Стоимость** | Бесплатно |

**Инструменты:** `fact_store` (9 действий: add, search, probe, related, reason, contradict, update, remove, list), `fact_feedback` (оценка helpful/unhelpful, которая обучает trust scores)

**Настройка:**
```bash
hermes memory setup    # выберите "holographic"
# Или вручную:
hermes config set memory.provider holographic
```

**Конфиг:** `config.yaml` в `plugins.hermes-memory-store`

| Ключ | По умолчанию | Описание |
|-----|-------------|----------|
| `db_path` | `$HERMES_HOME/memory_store.db` | Путь к базе SQLite |
| `auto_extract` | `false` | Автоматически извлекать факты при завершении сеанса |
| `default_trust` | `0.5` | Базовый trust score (0.0–1.0) |

**Уникальные возможности:**
- `probe` — алгебраический recall для конкретной сущности (все факты о человеке или вещи)
- `reason` — композиционные AND-запросы по нескольким сущностям
- `contradict` — автоматическое обнаружение конфликтующих фактов
- Оценка доверия с асимметричной обратной связью (+0.05 helpful / -0.10 unhelpful)

---

### RetainDB

Облачный API памяти с гибридным поиском (Vector + BM25 + Reranking), 7 типами памяти и delta compression.

| | |
|---|---|
| **Лучше всего подходит** | Команды, которые уже используют инфраструктуру RetainDB |
| **Требует** | Аккаунт RetainDB + API-ключ |
| **Хранение данных** | RetainDB Cloud |
| **Стоимость** | $20/month |

**Инструменты:** `retaindb_profile` (user profile), `retaindb_search` (семантический поиск), `retaindb_context` (контекст, релевантный задаче), `retaindb_remember` (сохранение с типом + важностью), `retaindb_forget` (удаление воспоминаний)

**Настройка:**
```bash
hermes memory setup    # выберите "retaindb"
# Или вручную:
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

Постоянная память через CLI `brv`: иерархическое дерево знаний с многоуровневым поиском (fuzzy text → LLM-driven search). С приоритетом локальной работы и опциональной облачной синхронизацией.

| | |
|---|---|
| **Лучше всего подходит** | Разработчики, которым нужна переносимая, local-first память с CLI |
| **Требует** | ByteRover CLI (`npm install -g byterover-cli` или [install script](https://byterover.dev)) |
| **Хранение данных** | Локально (по умолчанию) или ByteRover Cloud (опциональная синхронизация) |
| **Стоимость** | Бесплатно (локально) или тарифы ByteRover (cloud) |

**Инструменты:** `brv_query` (поиск по дереву знаний), `brv_curate` (сохранение фактов/решений/паттернов), `brv_status` (версия CLI + статистика дерева)

**Настройка:**
```bash
# Сначала установите CLI
curl -fsSL https://byterover.dev/install.sh | sh

# Затем настройте Hermes
hermes memory setup    # выберите "byterover"
# Или вручную:
hermes config set memory.provider byterover
```

**Ключевые возможности:**
- Автоматическое извлечение перед сжатием контекста (сохраняет инсайты до того, как сжатие их сотрёт)
- Дерево знаний хранится в `$HERMES_HOME/byterover/` (привязано к профилю)
- Облачная синхронизация с сертификатом SOC2 Type II (опционально)

---

### Supermemory

Семантическая долговременная память с профильным семантическим поиском, явными инструментами памяти и импортом разговора по завершении сеанса через графовый API Supermemory.

| | |
|---|---|
| **Лучше всего подходит** | Профильный семантический поиск с построением графа на уровне сеанса |
| **Требует** | `pip install supermemory` + [API-ключ](https://supermemory.ai) |
| **Хранение данных** | Supermemory Cloud |
| **Стоимость** | Тарифы Supermemory |

**Инструменты:** `supermemory_store` (сохранение явных воспоминаний), `supermemory_search` (поиск по семантической близости), `supermemory_forget` (забыть по ID или по лучшему совпадению), `supermemory_profile` (постоянный профиль + свежий контекст)

**Настройка:**
```bash
hermes memory setup    # выберите "supermemory"
# Или вручную:
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

**Конфиг:** `$HERMES_HOME/supermemory.json`

| Ключ | По умолчанию | Описание |
|-----|-------------|----------|
| `container_tag` | `hermes` | Тег контейнера для поиска и записи. Поддерживает шаблон `{identity}` для тегов, привязанных к профилю |
| `auto_recall` | `true` | Подставлять релевантный контекст памяти перед ходами |
| `auto_capture` | `true` | Сохранять очищенные ходы пользователь-ассистент после каждого ответа |
| `max_recall_results` | `10` | Максимум найденных элементов, которые нужно отформатировать в контекст |
| `profile_frequency` | `50` | Включать факты профиля на первом ходе и каждые N ходов |
| `capture_mode` | `all` | По умолчанию пропускать слишком короткие или тривиальные ходы |
| `search_mode` | `hybrid` | Режим поиска: `hybrid`, `memories` или `documents` |
| `api_timeout` | `5.0` | Таймаут для SDK и запросов на импорт |

**Переменные окружения:** `SUPERMEMORY_API_KEY` (обязательно), `SUPERMEMORY_CONTAINER_TAG` (переопределяет конфиг).

**Ключевые возможности:**
- Автоматическая изоляция контекста — удаляет извлекаемые элементы из захваченных ходов, чтобы не возникало рекурсивного загрязнения памяти
- Импорт разговора по завершении сеанса для более богатого графового знания
- Факты профиля подставляются на первом ходе и через настраиваемые интервалы
- Фильтрация тривиальных сообщений (пропускает «ok», «thanks» и т. п.)
- **Контейнеры, привязанные к профилю** — используйте `{identity}` в `container_tag` (например, `hermes-{identity}` → `hermes-coder`), чтобы изолировать память для каждого профиля Hermes
- **Многоконтейнерный режим** — включите `enable_custom_container_tags` со списком `custom_containers`, чтобы агент мог читать и писать в нескольких именованных контейнерах. Автоматические операции (sync, prefetch) остаются на основном контейнере

<details>
<summary>Пример многоконтейнерной конфигурации</summary>

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding context."
}
```

</details>

**Поддержка:** [Discord](https://supermemory.link/discord) · [support@supermemory.com](mailto:support@supermemory.com)

---

## Сравнение провайдеров

| Провайдер | Хранение | Стоимость | Инструменты | Зависимости | Уникальная особенность |
|----------|----------|-----------|------------|-------------|-----------------------|
| **Honcho** | Cloud | Платно | 5 | `honcho-ai` | Диалектическое моделирование пользователя + контекст на уровне сеанса |
| **OpenViking** | Self-hosted | Бесплатно | 5 | `openviking` + сервер | Иерархия в стиле файловой системы + многоуровневая загрузка |
| **Mem0** | Cloud | Платно | 3 | `mem0ai` | Серверное LLM-извлечение |
| **Hindsight** | Cloud/Local | Бесплатно/платно | 3 | `hindsight-client` | Граф знаний + синтез reflect |
| **Holographic** | Local | Бесплатно | 2 | Нет | HRR-алгебра + оценка доверия |
| **RetainDB** | Cloud | $20/мес | 5 | `requests` | Delta compression |
| **ByteRover** | Local/Cloud | Бесплатно/платно | 3 | CLI `brv` | Извлечение до сжатия контекста |
| **Supermemory** | Cloud | Платно | 4 | `supermemory` | Контекстное fencing + ingest графа по завершении сеанса + многоконтейнерность |

## Изоляция профилей

Данные каждого провайдера изолированы по [профилям](/docs/user-guide/profiles):

- **Провайдеры с локальным хранением** (Holographic, ByteRover) используют пути `$HERMES_HOME/`, которые различаются между профилями
- **Провайдеры с конфиг-файлами** (Honcho, Mem0, Hindsight, Supermemory) хранят конфиги в `$HERMES_HOME/`, поэтому у каждого профиля свои учётные данные
- **Облачные провайдеры** (RetainDB) автоматически выводят имена проектов, привязанные к профилю
- **Провайдеры, зависящие от env vars** (OpenViking) настраиваются через `.env` каждого профиля

## Создание собственного провайдера памяти

См. [Руководство разработчика: плагины провайдеров памяти](/docs/developer-guide/memory-provider-plugin) — там описано, как создать собственный провайдер.
