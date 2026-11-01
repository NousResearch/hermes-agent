---
sidebar_position: 8
title: "コンテキストファイル"
description: "プロジェクトコンテキストファイル — .hermes.md、AGENTS.md、CLAUDE.md、グローバルな SOUL.md、.cursorrules — はすべての会話に自動的に注入されます"
---

# コンテキストファイル

Hermes Agent は、その挙動を形作るコンテキストファイルを自動的に発見して読み込みます。一部はプロジェクトローカルで、作業ディレクトリから発見されます。`SOUL.md` は現在、Hermes インスタンスに対してグローバルであり、`HERMES_HOME` からのみ読み込まれます。

## サポートされるコンテキストファイル

| ファイル | 目的 | 発見方法 |
|------|------|----------| 
| **.hermes.md** / **HERMES.md** | プロジェクト指示（最優先） | git ルートまで遡る |
| **AGENTS.md** | プロジェクト指示、規約、アーキテクチャ | 起動時の CWD + サブディレクトリを漸進的に |
| **CLAUDE.md** | Claude Code のコンテキストファイル（こちらも検出） | 起動時の CWD + サブディレクトリを漸進的に |
| **SOUL.md** | この Hermes インスタンスのグローバルなパーソナリティとトーンのカスタマイズ | `HERMES_HOME/SOUL.md` のみ |
| **.cursorrules** | Cursor IDE のコーディング規約 | CWD のみ |
| **.cursor/rules/*.mdc** | Cursor IDE のルールモジュール | CWD のみ |

:::info 優先システム
セッションごとに読み込まれるプロジェクトコンテキストタイプは**1 つ**だけです（最初にマッチしたものが優先）: `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。**SOUL.md** は常にエージェントのアイデンティティとして独立して読み込まれます（スロット #1）。
:::

## AGENTS.md

`AGENTS.md` は主要なプロジェクトコンテキストファイルです。プロジェクトがどのように構成されているか、どの規約に従うべきか、そして特別な指示をエージェントに伝えます。

### 漸進的なサブディレクトリ発見

セッション開始時、Hermes は作業ディレクトリの `AGENTS.md` をシステムプロンプトに読み込みます。セッション中にエージェントがサブディレクトリに移動すると（`read_file`、`terminal`、`search_files` などを介して）、それらのディレクトリ内のコンテキストファイルを**漸進的に発見**し、関連性が生じた時点で会話に注入します。

```
my-project/
├── AGENTS.md              ← 起動時に読み込み（システムプロンプト）
├── frontend/
│   └── AGENTS.md          ← エージェントが frontend/ のファイルを読むときに発見
├── backend/
│   └── AGENTS.md          ← エージェントが backend/ のファイルを読むときに発見
└── shared/
    └── AGENTS.md          ← エージェントが shared/ のファイルを読むときに発見
```

このアプローチには、起動時にすべてを読み込むことに対して 2 つの利点があります:
- **システムプロンプトの肥大化なし** — サブディレクトリのヒントは必要なときにのみ現れます
- **プロンプトキャッシュの保持** — システムプロンプトはターンをまたいで安定したままです

各サブディレクトリは、セッションごとに最大 1 回チェックされます。発見は親ディレクトリも遡るため、`backend/src/main.py` を読むと、`backend/src/` 自体にコンテキストファイルがなくても `backend/AGENTS.md` を発見します。

:::info
サブディレクトリのコンテキストファイルは、起動時のコンテキストファイルと同じ[セキュリティスキャン](#security-prompt-injection-protection)を通過します。悪意のあるファイルはブロックされます。
:::

### AGENTS.md の例

```markdown
# Project Context

This is a Next.js 14 web application with a Python FastAPI backend.

## Architecture
- Frontend: Next.js 14 with App Router in `/frontend`
- Backend: FastAPI in `/backend`, uses SQLAlchemy ORM
- Database: PostgreSQL 16
- Deployment: Docker Compose on a Hetzner VPS

## Conventions
- Use TypeScript strict mode for all frontend code
- Python code follows PEP 8, use type hints everywhere
- All API endpoints return JSON with `{data, error, meta}` shape
- Tests go in `__tests__/` directories (frontend) or `tests/` (backend)

## Important Notes
- Never modify migration files directly — use Alembic commands
- The `.env.local` file has real API keys, don't commit it
- Frontend port is 3000, backend is 8000, DB is 5432
```

## SOUL.md

`SOUL.md` はエージェントのパーソナリティ、トーン、コミュニケーションスタイルを制御します。詳細は [Personality](/docs/user-guide/features/personality) ページを参照してください。

**場所:**

- `~/.hermes/SOUL.md`
- またはカスタムホームディレクトリで Hermes を実行している場合は `$HERMES_HOME/SOUL.md`

重要な詳細:

- `SOUL.md` がまだ存在しない場合、Hermes はデフォルトの `SOUL.md` を自動的に生成します
- Hermes は `SOUL.md` を `HERMES_HOME` からのみ読み込みます
- Hermes は作業ディレクトリで `SOUL.md` を探しません
- ファイルが空の場合、`SOUL.md` からはプロンプトに何も追加されません
- ファイルに内容がある場合、その内容はスキャンと切り詰めの後、そのまま注入されます

## .cursorrules

Hermes は Cursor IDE の `.cursorrules` ファイルと `.cursor/rules/*.mdc` ルールモジュールに対応しています。これらのファイルがプロジェクトルートに存在し、より優先度の高いコンテキストファイル（`.hermes.md`、`AGENTS.md`、`CLAUDE.md`）が見つからない場合、それらがプロジェクトコンテキストとして読み込まれます。

これは、Hermes を使用するときに既存の Cursor の規約が自動的に適用されることを意味します。

## コンテキストファイルの読み込み方法

### 起動時（システムプロンプト）

コンテキストファイルは `agent/prompt_builder.py` の `build_context_files_prompt()` によって読み込まれます:

1. **作業ディレクトリのスキャン** — `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules` をチェック（最初にマッチしたものが優先）
2. **内容の読み取り** — 各ファイルは UTF-8 テキストとして読み取られます
3. **セキュリティスキャン** — 内容がプロンプトインジェクションパターンについてチェックされます
4. **切り詰め** — 20,000 文字を超えるファイルは先頭/末尾で切り詰められます（先頭 70%、末尾 20%、中央にマーカー付き）
5. **アセンブル** — すべてのセクションが `# Project Context` ヘッダーの下にまとめられます
6. **注入** — まとめられた内容がシステムプロンプトに追加されます

### セッション中（漸進的な発見）

`agent/subdirectory_hints.py` の `SubdirectoryHintTracker` は、ツールコールの引数からファイルパスを監視します:

1. **パス抽出** — 各ツールコールの後、引数（`path`、`workdir`、シェルコマンド）からファイルパスが抽出されます
2. **祖先の遡り** — そのディレクトリと最大 5 つの親ディレクトリがチェックされます（既に訪問済みのディレクトリで停止）
3. **ヒントの読み込み** — `AGENTS.md`、`CLAUDE.md`、`.cursorrules` が見つかれば、それが読み込まれます（ディレクトリごとに最初にマッチしたもの）
4. **セキュリティスキャン** — 起動時ファイルと同じプロンプトインジェクションスキャン
5. **切り詰め** — ファイルごとに 8,000 文字でキャップ
6. **注入** — ツール結果に追加され、モデルがコンテキスト内で自然に見られるようにします

最終的なプロンプトセクションはおおよそ次のようになります:

```text
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[Your AGENTS.md content here]

## .cursorrules

[Your .cursorrules content here]

[Your SOUL.md content here]
```

SOUL の内容は、余分なラッパーテキストなしで直接挿入される点に注意してください。

## セキュリティ: プロンプトインジェクション保護 {#security-prompt-injection-protection}

すべてのコンテキストファイルは、含める前に潜在的なプロンプトインジェクションについてスキャンされます。スキャナーは次をチェックします:

- **指示の上書きの試み**: "ignore previous instructions"、"disregard your rules"
- **欺瞞パターン**: "do not tell the user"
- **システムプロンプトの上書き**: "system prompt override"
- **隠された HTML コメント**: `<!-- ignore instructions -->`
- **隠された div 要素**: `<div style="display:none">`
- **クレデンシャルの抜き取り**: `curl ... $API_KEY`
- **秘密ファイルへのアクセス**: `cat .env`、`cat credentials`
- **不可視文字**: ゼロ幅スペース、双方向オーバーライド、ワードジョイナー

脅威パターンが検出された場合、ファイルはブロックされます:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
このスキャナーは一般的なインジェクションパターンから保護しますが、共有リポジトリ内のコンテキストファイルをレビューする代わりにはなりません。自分が作成していないプロジェクトでは、必ず AGENTS.md の内容を検証してください。
:::

## サイズ制限

| 制限 | 値 |
|------|-----|
| ファイルごとの最大文字数 | 20,000（約 7,000 トークン） |
| 先頭の切り詰め比率 | 70% |
| 末尾の切り詰め比率 | 20% |
| 切り詰めマーカー | 10%（文字数を表示し、ファイルツールの使用を提案） |

ファイルが 20,000 文字を超えると、切り詰めメッセージは次のように表示されます:

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## 効果的なコンテキストファイルのためのヒント

:::tip AGENTS.md のベストプラクティス
1. **簡潔に保つ** — 20K 文字を十分に下回るようにする。エージェントは毎ターンこれを読みます
2. **ヘッダーで構造化する** — アーキテクチャ、規約、重要な注記に `##` セクションを使う
3. **具体的な例を含める** — 好ましいコードパターン、API の形、命名規約を示す
4. **やってはいけないことを記載する** — 「マイグレーションファイルを直接変更しない」
5. **主要なパスとポートを列挙する** — エージェントはこれらをターミナルコマンドに使います
6. **プロジェクトの進化に合わせて更新する** — 古いコンテキストはコンテキストがないより悪いです
:::

### サブディレクトリごとのコンテキスト

モノレポでは、サブディレクトリ固有の指示をネストした AGENTS.md ファイルに入れます:

```markdown
<!-- frontend/AGENTS.md -->
# Frontend Context

- Use `pnpm` not `npm` for package management
- Components go in `src/components/`, pages in `src/app/`
- Use Tailwind CSS, never inline styles
- Run tests with `pnpm test`
```

```markdown
<!-- backend/AGENTS.md -->
# Backend Context

- Use `poetry` for dependency management
- Run the dev server with `poetry run uvicorn main:app --reload`
- All endpoints need OpenAPI docstrings
- Database models are in `models/`, schemas in `schemas/`
```
