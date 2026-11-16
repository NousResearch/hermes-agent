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
| **AGENTS.override.md** | AGENTS.md の個人用・ディレクトリごとの上書き（通常は gitignore 対象） | 起動時の CWD + サブディレクトリを漸進的に |
| **AGENTS.md** | プロジェクト指示、規約、アーキテクチャ | 起動時の CWD + サブディレクトリを漸進的に |
| **CLAUDE.md** | Claude Code のコンテキストファイル（こちらも検出） | 起動時の CWD + サブディレクトリを漸進的に |
| **SOUL.md** | この Hermes インスタンスのグローバルなパーソナリティとトーンのカスタマイズ | `HERMES_HOME/SOUL.md` のみ |
| **.cursorrules** | Cursor IDE のコーディング規約 | CWD のみ |
| **.cursor/rules/*.mdc** | Cursor IDE のルールモジュール | CWD のみ |

:::info 優先システム
セッションごとに読み込まれるプロジェクトコンテキストタイプは**1 つ**だけです（最初にマッチしたものが優先）: `.hermes.md` → `AGENTS.override.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。**SOUL.md** は常にエージェントのアイデンティティとして独立して読み込まれます（スロット #1）。

`AGENTS.md` の隣に `AGENTS.override.md` が存在する場合、コミットされたファイルの**代わりに**上書きファイルが読み込まれます — リポジトリにチェックインされた指示とは異なる指示を使いたいときは、追跡対象の `AGENTS.md` を編集せずに、個人用の（通常は gitignore 対象の）`AGENTS.override.md` を置いてください。
:::

## AGENTS.md

`AGENTS.md` は主要なプロジェクトコンテキストファイルです。プロジェクトがどのように構成されているか、どの規約に従うべきか、そして特別な指示をエージェントに伝えます。

### ディレクトリチェーン（git ルート → 作業ディレクトリ）

作業ディレクトリが git リポジトリ内にある場合、Hermes はセッション開始時に `AGENTS.md` ファイルの**マージされたチェーン**を読み込みます: まず git ルートの `AGENTS.md`、次に作業ディレクトリまでのすべての中間ディレクトリにある `AGENTS.md` です。より深い階層のファイルはプロンプト内で後に現れるため、より具体的なガイダンスが優先されます。各ファイルには独自の出所ヘッダー（例: `## ../../AGENTS.md`）が付き、チェーン上の同一のコピーは重複排除されます。

```
monorepo/                   (git root, cwd = packages/webapp/)
├── AGENTS.md              ← Loaded first (repo-wide conventions)
└── packages/
    ├── AGENTS.md          ← Loaded second
    └── webapp/
        └── AGENTS.md      ← Loaded last (most specific, takes precedence)
```

<!-- no-tmp: ok — explains why /tmp is never trusted -->
git リポジトリの外では、作業ディレクトリ自体のみがチェックされます — 親ディレクトリが参照されることはないため、`/tmp` や `$HOME` に仕込まれた `AGENTS.md` が無関係なセッションに漏れ込むことはありません。

### 漸進的なサブディレクトリ発見

セッション開始時、Hermes は作業ディレクトリの `AGENTS.md` をシステムプロンプトに読み込みます。セッション中にエージェントがサブディレクトリに移動すると（`read_file`、`terminal`、`search_files` などを介して）、それらのディレクトリ内のコンテキストファイルを**漸進的に発見**し、関連性が生じた時点で会話に注入します。

```
my-project/
├── AGENTS.md              ← Loaded at startup (system prompt)
├── frontend/
│   └── AGENTS.md          ← Discovered when agent reads frontend/ files
├── backend/
│   └── AGENTS.md          ← Discovered when agent reads backend/ files
└── shared/
    └── AGENTS.md          ← Discovered when agent reads shared/ files
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

`SOUL.md` はエージェントのパーソナリティ、トーン、コミュニケーションスタイルを制御します。詳細は [Personality](./personality.md) ページを参照してください。

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
4. **切り詰め** — 文字数の上限を超えるファイルは先頭/末尾で切り詰められます（先頭 70%、末尾 20%、中央にマーカー付き）。上限は、config.yaml で明示的に `context_file_max_chars` が設定されていればその値、そうでなければモデルのコンテキストウィンドウに応じて動的に決まります（下限 20,000 文字、上限 500,000 文字）
5. **アセンブル** — すべてのセクションが `# Project Context` ヘッダーの下にまとめられます
6. **注入** — まとめられた内容がシステムプロンプトに追加されます

### セッション中（漸進的な発見）

`agent/subdirectory_hints.py` の `SubdirectoryHintTracker` は、ツールコールの引数からファイルパスを監視します:

1. **パス抽出** — 各ツールコールの後、引数（`path`、`workdir`、シェルコマンド）からファイルパスが抽出されます
2. **祖先の遡り** — そのディレクトリと最大 5 つの親ディレクトリがチェックされます（既に訪問済みのディレクトリで停止）
3. **ヒントの読み込み** — `AGENTS.md`、`CLAUDE.md`、`.cursorrules` が見つかれば、それが読み込まれます（ディレクトリごとに最初にマッチしたもの）
4. **セキュリティスキャン** — 起動時ファイルと同じプロンプトインジェクションスキャン
5. **切り詰め** — ファイルごとに 32,000 文字でキャップ（固定のプレビュー上限で、`context_file_max_chars` やモデルのコンテキストウィンドウによって変わりません）。サイズ超過のヒントは、完全なファイルを指す先頭/末尾のマーカーを保持してログに記録されますが、起動時のコンテキストファイルのようなチャットでの切り詰め警告は出ません
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

プロジェクトコンテキストファイル（`.hermes.md`、`AGENTS.md`、`CLAUDE.md`、
`.cursorrules`）で脅威パターンが検出された場合、ファイルはブロックされます:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

`HERMES_HOME` にある自分自身の `SOUL.md` は扱いが異なります: これはあなたが書いたファイルであり（ファイルツールによる書き込みにはあなたの承認が必要で、
プロジェクトのチェックアウトがこれを提供することはありません）、そこでスキャナーがヒットしても**ファイルはブロックされません**。
Hermes はマッチしたパターンを示す警告をログに記録し、通常どおりファイルを読み込み、`/context` はそれを
`⚠ SOUL.md … loaded — matched prompt-injection pattern(s); review the file` として一覧表示します。これにより、攻撃フレーズを
*説明する*アイデンティティファイル（「以前の指示を無視するよう伝えるコンテンツ」といったセキュリティガイダンスなど）も
動作し続けます。フラグが立ったテキストを自分で書いていない場合は、その警告を、何か別のものがファイルを編集した兆候として扱ってください。
この例外は、プロファイルディストリビューションによって提供される `SOUL.md` には適用されません: `hermes profile
install <git-url>` と `hermes profile update` は、サードパーティの `SOUL.md` をスキャンや承認プロンプトなしでプロファイルホームにコピーするため、
`distribution.yaml` がファイルを所有している場合は、スキャナーのヒットによって引き続きブロックされます。

:::warning
このスキャナーは一般的なインジェクションパターンから保護しますが、共有リポジトリ内のコンテキストファイルをレビューする代わりにはなりません。自分が作成していないプロジェクトでは、必ず AGENTS.md の内容を検証してください。
:::

## サイズ制限

| 制限 | 値 |
|------|-----|
| ファイルごとの最大文字数 | 設定されている場合は `context_file_max_chars`。それ以外は動的（モデルのコンテキストウィンドウに応じて変化、下限 20,000、上限 500,000） |
| ファイルごとの読み取りタイムアウト | `context_file_read_timeout`（デフォルト 5 秒）。読み取りにそれ以上かかるファイル — 例: iCloud Drive、OneDrive、NFS 上のもの — は警告付きでスキップされます |
| 先頭の切り詰め比率 | 70% |
| 末尾の切り詰め比率 | 20% |
| 切り詰めマーカー | 10%（文字数を表示し、ファイルツールの使用を提案） |

ファイルが設定された上限を超えると、切り詰めメッセージは次のように表示されます:

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## 効果的なコンテキストファイルのためのヒント

:::tip AGENTS.md のベストプラクティス
1. **簡潔に保つ** — 設定された `context_file_max_chars` を下回るようにする。エージェントは毎ターンこれを読みます
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
