---
title: "Siyuan — API SiYuan Note для поиска, чтения, создания и управления блоками и документами"
sidebar_label: "Siyuan"
description: "API SiYuan Note для поиска, чтения, создания и управления блоками и документами в самохостируемой базе знаний через curl"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Siyuan

API SiYuan Note для поиска, чтения, создания и управления блоками и документами в самохостируемой базе знаний через curl.

## Метаданные навыка

| | |
|---|---|
| Источник | Опционально — `hermes skills install official/productivity/siyuan` |
| Путь | `optional-skills/productivity/siyuan` |
| Версия | `1.0.0` |
| Автор | FEUAZUR |
| Лицензия | MIT |
| Платформы | linux, macos, windows |
| Теги | `SiYuan`, `Notes`, `Knowledge Base`, `PKM`, `API` |
| Связанные навыки | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) |

## Справка: полный `SKILL.md`

:::info
Ниже приведено полное определение навыка, которое Hermes загружает при его активации. Именно это видит агент как набор инструкций, когда навык включён.
:::

# API SiYuan Note

Используйте kernel API [SiYuan](https://github.com/siyuan-note/siyuan) через curl, чтобы искать, читать, создавать, обновлять и удалять блоки и документы в самохостируемой базе знаний. Дополнительные инструменты не нужны — достаточно curl и API-токена.

## Предварительные требования

1. Установите и запустите SiYuan (desktop или Docker).
2. Получите API-токен: **Settings > About > API token**.
3. Сохраните его в `~/.hermes/.env`:
   ```
   SIYUAN_TOKEN=your_token_here
   SIYUAN_URL=http://127.0.0.1:6806
   ```
   Если `SIYUAN_URL` не задан, по умолчанию используется `http://127.0.0.1:6806`.

## Основы API

Все вызовы API SiYuan выполняются как **POST с JSON-телом**. Каждый запрос строится по одному и тому же шаблону:

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/..." \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"param": "value"}'
```

Ответы приходят в JSON с такой структурой:
```json
{"code": 0, "msg": "", "data": { ... }}
```
Значение `code: 0` означает успех. Любое другое значение — ошибка; подробности ищите в `msg`.

**Формат ID:** ID в SiYuan выглядят как `20210808180117-6v0mkxr` (14-значная временная метка + 7 буквенно-цифровых символов).

## Быстрая справка

| Операция | Конечная точка |
|-----------|----------|
| Полнотекстовый поиск | `/api/search/fullTextSearchBlock` |
| SQL-запрос | `/api/query/sql` |
| Чтение блока | `/api/block/getBlockKramdown` |
| Чтение дочерних блоков | `/api/block/getChildBlocks` |
| Получение пути | `/api/filetree/getHPathByID` |
| Получение атрибутов | `/api/attr/getBlockAttrs` |
| Список блокнотов | `/api/notebook/lsNotebooks` |
| Список документов | `/api/filetree/listDocsByPath` |
| Создать блокнот | `/api/notebook/createNotebook` |
| Создать документ | `/api/filetree/createDocWithMd` |
| Добавить блок | `/api/block/appendBlock` |
| Обновить блок | `/api/block/updateBlock` |
| Переименовать документ | `/api/filetree/renameDocByID` |
| Задать атрибуты | `/api/attr/setBlockAttrs` |
| Удалить блок | `/api/block/deleteBlock` |
| Удалить документ | `/api/filetree/removeDocByID` |
| Экспорт в Markdown | `/api/export/exportMdContent` |

## Частые операции

### Поиск (полнотекстовый)

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/search/fullTextSearchBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "meeting notes", "page": 0}' | jq '.data.blocks[:5]'
```

### Поиск (SQL)

Запрашивает блоки напрямую через SQL. Безопасны только операторы SELECT.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/query/sql" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stmt": "SELECT id, content, type, box FROM blocks WHERE content LIKE '\''%keyword%'\'' AND type='\''p'\'' LIMIT 20"}' | jq '.data'
```

Полезные столбцы: `id`, `parent_id`, `root_id`, `box` (ID блокнота), `path`, `content`, `type`, `subtype`, `created`, `updated`.

### Чтение содержимого блока

Возвращает содержимое блока в формате Kramdown (Markdown-подобный формат).

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/getBlockKramdown" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data.kramdown'
```

### Чтение дочерних блоков

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/getChildBlocks" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### Получение читаемого пути

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/getHPathByID" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### Получение атрибутов блока

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/attr/getBlockAttrs" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### Список блокнотов

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/notebook/lsNotebooks" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' | jq '.data.notebooks[] | {id, name, closed}'
```

### Список документов в блокноте

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/listDocsByPath" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"notebook": "NOTEBOOK_ID", "path": "/"}' | jq '.data.files[] | {id, name}'
```

### Создание документа

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/createDocWithMd" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "notebook": "NOTEBOOK_ID",
    "path": "/Meeting Notes/2026-03-22",
    "markdown": "# Meeting Notes\n\n- Discussed project timeline\n- Assigned tasks"
  }' | jq '.data'
```

### Создание блокнота

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/notebook/createNotebook" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My New Notebook"}' | jq '.data.notebook.id'
```

### Добавление блока в документ

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/appendBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "parentID": "DOCUMENT_OR_BLOCK_ID",
    "data": "New paragraph added at the end.",
    "dataType": "markdown"
  }' | jq '.data'
```

Также доступны `/api/block/prependBlock` (те же параметры, вставка в начало) и `/api/block/insertBlock` (использует `previousID` вместо `parentID`, чтобы вставить после конкретного блока).

### Обновление содержимого блока

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/updateBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "BLOCK_ID",
    "data": "Updated content here.",
    "dataType": "markdown"
  }' | jq '.data'
```

### Переименование документа

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/renameDocByID" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "DOCUMENT_ID", "title": "New Title"}'
```

### Задание атрибутов блока

Пользовательские атрибуты должны начинаться с `custom-`:

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/attr/setBlockAttrs" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "BLOCK_ID",
    "attrs": {
      "custom-status": "reviewed",
      "custom-priority": "high"
    }
  }'
```

### Удаление блока

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/deleteBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "BLOCK_ID"}'
```

Чтобы удалить весь документ, используйте `/api/filetree/removeDocByID` с `{"id": "DOC_ID"}`. Чтобы удалить блокнот, используйте `/api/notebook/removeNotebook` с `{"notebook": "NOTEBOOK_ID"}`.

### Экспорт документа в Markdown

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/export/exportMdContent" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "DOCUMENT_ID"}' | jq -r '.data.content'
```

## Типы блоков

Обычно в SQL-запросах используются такие значения `type`:

| Тип | Описание |
|------|-------------|
| `d` | Документ (корневой блок) |
| `p` | Абзац |
| `h` | Заголовок |
| `l` | Список |
| `i` | Элемент списка |
| `c` | Блок кода |
| `m` | Математический блок |
| `t` | Таблица |
| `b` | Цитата |
| `s` | Суперблок |
| `html` | HTML-блок |

## Подводные камни

- **Все endpoints используют POST** — даже операции только для чтения. Не используйте GET.
- **Безопасность SQL**: используйте только SELECT-запросы. INSERT/UPDATE/DELETE/DROP опасны и никогда не должны отправляться.
- **Проверка ID**: ID должны соответствовать шаблону `YYYYMMDDHHmmss-xxxxxxx`. Отбрасывайте всё, что не подходит.
- **Ответы с ошибкой**: всегда проверяйте `code != 0` до обработки `data`.
- **Большие документы**: содержимое блоков и результаты экспорта могут быть очень большими. Используйте `LIMIT` в SQL и `jq`, чтобы извлечь только нужное.
- **ID блокнотов**: при работе с конкретным блокнотом сначала получите его ID через `lsNotebooks`.

## Альтернатива: MCP Server

Если вместо curl вы предпочитаете нативную интеграцию, установите SiYuan MCP server:

```yaml
# В ~/.hermes/config.yaml в разделе mcp_servers:
mcp_servers:
  siyuan:
    command: npx
    args: ["-y", "@porkll/siyuan-mcp"]
    env:
      SIYUAN_TOKEN: "your_token"
      SIYUAN_URL: "http://127.0.0.1:6806"
```
