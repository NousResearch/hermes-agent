---
sidebar_position: 2
title: "スキルシステム"
description: "オンデマンドの知識ドキュメント — 漸進的開示、エージェント管理スキル、Skills Hub"
---

# スキルシステム

スキルは、エージェントが必要なときにロードできるオンデマンドの知識ドキュメントです。トークン使用量を最小化するために**漸進的開示（progressive disclosure）**パターンに従っており、[agentskills.io](https://agentskills.io/specification) オープン標準と互換性があります。

すべてのスキルは **`~/.hermes/skills/`** に存在します — これがプライマリディレクトリであり信頼できる情報源です。新規インストール時には、同梱スキルがリポジトリからコピーされます。Hubからインストールされたスキルやエージェントが作成したスキルもここに置かれます。エージェントは任意のスキルを変更または削除できます。

Hermesに**外部スキルディレクトリ**を指定することもできます — ローカルのものと並んでスキャンされる追加のフォルダです。下記の[外部スキルディレクトリ](#external-skill-directories)を参照してください。

関連項目:

- [同梱スキルカタログ](/docs/reference/skills-catalog)
- [公式オプションスキルカタログ](/docs/reference/optional-skills-catalog)

## スキルの使用

インストールされたすべてのスキルは、自動的にスラッシュコマンドとして利用できます:

```bash
# CLIまたは任意のメッセージングプラットフォームで:
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor
/plan design a rollout for migrating our auth provider

# スキル名だけでロードされ、エージェントが必要なものを尋ねてくる:
/excalidraw
```

同梱の `plan` スキルが良い例です。`/plan [request]` を実行するとスキルの指示がロードされ、Hermesに対して、必要に応じてコンテキストを確認し、タスクを実行する代わりにMarkdownの実装プランを書き、結果をアクティブなワークスペース/バックエンドの作業ディレクトリを基準とした `.hermes/plans/` 配下に保存するよう指示します。

自然な会話を通じてスキルとやり取りすることもできます:

```bash
hermes chat --toolsets skills -q "What skills do you have?"
hermes chat --toolsets skills -q "Show me the axolotl skill"
```

## 漸進的開示

スキルは、トークン効率の良いロードパターンを使用します:

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

エージェントは、実際に必要になったときにのみスキルの完全な内容をロードします。

## SKILL.md 形式 {#skillmd-format}

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
platforms: [macos, linux]     # 任意 — 特定のOSプラットフォームに制限する
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    fallback_for_toolsets: [web]    # 任意 — 条件付きアクティベーション（後述）
    requires_toolsets: [terminal]   # 任意 — 条件付きアクティベーション（後述）
    config:                          # 任意 — config.yaml の設定
      - key: my.setting
        description: "What this controls"
        default: "value"
        prompt: "Prompt for setup"
---

# Skill Title

## When to Use
Trigger conditions for this skill.

## Procedure
1. Step one
2. Step two

## Pitfalls
- Known failure modes and fixes

## Verification
How to confirm it worked.
```

### プラットフォーム固有のスキル

スキルは、`platforms` フィールドを使って特定のオペレーティングシステムに自身を制限できます:

| 値 | 一致する対象 |
|-------|---------|
| `macos` | macOS（Darwin） |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # macOSのみ（例: iMessage、Apple Reminders、FindMy）
platforms: [macos, linux]     # macOSとLinux
```

設定すると、互換性のないプラットフォームでは、スキルはシステムプロンプト、`skills_list()`、スラッシュコマンドから自動的に非表示になります。省略した場合、スキルはすべてのプラットフォームでロードされます。

### 条件付きアクティベーション（フォールバックスキル）

スキルは、現在のセッションでどのツールが利用可能かに基づいて、自身を自動的に表示または非表示にできます。これは**フォールバックスキル** — プレミアムツールが利用できないときにのみ表示されるべき無料またはローカルの代替手段 — に最も役立ちます。

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # これらのツールセットが利用不可のときのみ表示
    requires_toolsets: [terminal]     # これらのツールセットが利用可能のときのみ表示
    fallback_for_tools: [web_search]  # これらの特定ツールが利用不可のときのみ表示
    requires_tools: [terminal]        # これらの特定ツールが利用可能のときのみ表示
```

| フィールド | 動作 |
|-------|----------|
| `fallback_for_toolsets` | 列挙されたツールセットが利用可能なときスキルは**非表示**になる。それらが欠けているときに表示される。 |
| `fallback_for_tools` | 同様だが、ツールセットの代わりに個々のツールをチェックする。 |
| `requires_toolsets` | 列挙されたツールセットが利用不可のときスキルは**非表示**になる。それらが存在するときに表示される。 |
| `requires_tools` | 同様だが、個々のツールをチェックする。 |

**例:** 組み込みの `duckduckgo-search` スキルは `fallback_for_toolsets: [web]` を使用します。`FIRECRAWL_API_KEY` が設定されている場合、webツールセットが利用可能になりエージェントは `web_search` を使用します — DuckDuckGoスキルは非表示のままです。APIキーが欠けている場合、webツールセットは利用不可となり、DuckDuckGoスキルが自動的にフォールバックとして表示されます。

条件付きフィールドを持たないスキルは、これまでとまったく同じように振る舞います — 常に表示されます。

## ロード時の安全なセットアップ

スキルは、検出から消えることなく、必要な環境変数を宣言できます:

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

欠けている値に遭遇すると、Hermesはスキルが実際にローカルCLIでロードされたときにのみ、それを安全に尋ねます。セットアップをスキップしてスキルを使い続けることもできます。メッセージングサーフェスがチャットで秘密情報を尋ねることは決してありません — 代わりにローカルで `hermes setup` または `~/.hermes/.env` を使うように伝えます。

設定されると、宣言された環境変数は `execute_code` と `terminal` のサンドボックスに**自動的に渡されます** — スキルのスクリプトは `$TENOR_API_KEY` を直接使用できます。スキル以外の環境変数については、`terminal.env_passthrough` 設定オプションを使用してください。詳細は[環境変数のパススルー](/docs/user-guide/security#environment-variable-passthrough)を参照してください。

### スキル設定の項目

スキルは、`config.yaml` に保存される非秘密の設定項目（パス、設定値）を宣言することもできます:

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
```

設定は config.yaml の `skills.config` 配下に保存されます。`hermes config migrate` は未設定の項目についてプロンプトを表示し、`hermes config show` はそれらを表示します。スキルがロードされると、その解決済みの設定値がコンテキストに注入されるため、エージェントは設定された値を自動的に把握します。

詳細は[スキル設定](/docs/user-guide/configuration#skill-settings)および[スキルの作成 — 設定項目](/docs/developer-guide/creating-skills#config-settings-configyaml)を参照してください。

## スキルディレクトリ構造

```text
~/.hermes/skills/                  # 唯一の信頼できる情報源
├── mlops/                         # カテゴリディレクトリ
│   ├── axolotl/
│   │   ├── SKILL.md               # メインの指示（必須）
│   │   ├── references/            # 追加のドキュメント
│   │   ├── templates/             # 出力形式
│   │   ├── scripts/               # スキルから呼び出せるヘルパースクリプト
│   │   └── assets/                # 補足ファイル
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # エージェントが作成したスキル
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Skills Hub の状態
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # シードされた同梱スキルを追跡
```

## 外部スキルディレクトリ {#external-skill-directories}

Hermesの外でスキルを管理している場合 — 例えば、複数のAIツールで使われる共有の `~/.agents/skills/` ディレクトリなど — Hermesにそれらのディレクトリもスキャンするよう指示できます。

`~/.hermes/config.yaml` の `skills` セクション配下に `external_dirs` を追加します:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
    - ${SKILLS_REPO}/skills
```

パスは `~` の展開と `${VAR}` 環境変数の置換をサポートします。

### 仕組み

- **読み取り専用**: 外部ディレクトリはスキルの検出のためにのみスキャンされます。エージェントがスキルを作成または編集するときは、常に `~/.hermes/skills/` に書き込みます。
- **ローカルの優先**: 同じスキル名がローカルディレクトリと外部ディレクトリの両方に存在する場合、ローカルバージョンが優先されます。
- **完全な統合**: 外部スキルは、システムプロンプトのインデックス、`skills_list`、`skill_view`、`/skill-name` スラッシュコマンドに表示されます — ローカルスキルと変わりません。
- **存在しないパスは静かにスキップされる**: 設定されたディレクトリが存在しない場合、Hermesはエラーなしでそれを無視します。すべてのマシンに存在するとは限らないオプションの共有ディレクトリに便利です。

### 例

```text
~/.hermes/skills/               # ローカル（プライマリ、読み書き可）
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # 外部（読み取り専用、共有）
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

4つすべてのスキルがスキルインデックスに表示されます。`my-custom-workflow` という名前の新しいスキルをローカルで作成すると、外部バージョンを隠します。

## エージェント管理スキル（skill_manage ツール）

エージェントは、`skill_manage` ツールを介して自身のスキルを作成、更新、削除できます。これはエージェントの**手続き的メモリ**です — 自明でないワークフローを見つけ出したとき、将来再利用するためにそのアプローチをスキルとして保存します。

### エージェントがスキルを作成するとき

- 複雑なタスク（5回以上のツール呼び出し）を成功裏に完了した後
- エラーや行き詰まりに遭遇し、うまくいく道筋を見つけたとき
- ユーザーがそのアプローチを修正したとき
- 自明でないワークフローを発見したとき

### アクション

| アクション | 用途 | 主なパラメーター |
|--------|---------|------------|
| `create` | ゼロからの新しいスキル | `name`、`content`（完全なSKILL.md）、任意で `category` |
| `patch` | ピンポイントの修正（推奨） | `name`、`old_string`、`new_string` |
| `edit` | 大規模な構造の書き換え | `name`、`content`（完全なSKILL.mdの置き換え） |
| `delete` | スキルを完全に削除する | `name` |
| `write_file` | 補助ファイルの追加/更新 | `name`、`file_path`、`file_content` |
| `remove_file` | 補助ファイルの削除 | `name`、`file_path` |

:::tip
更新には `patch` アクションが推奨されます — 変更されたテキストのみがツール呼び出しに現れるため、`edit` よりもトークン効率が良いです。
:::

## Skills Hub

オンラインレジストリ、`skills.sh`、直接のwell-knownスキルエンドポイント、公式オプションスキルから、スキルを閲覧、検索、インストール、管理します。

### よく使うコマンド

```bash
hermes skills browse                              # すべてのhubスキルを閲覧（公式が先）
hermes skills browse --source official            # 公式オプションスキルのみを閲覧
hermes skills search kubernetes                   # すべてのソースを検索
hermes skills search react --source skills-sh     # skills.sh ディレクトリを検索
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # インストール前にプレビュー
hermes skills install openai/skills/k8s           # セキュリティスキャン付きでインストール
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # 直接URL（単一ファイルのSKILL.md）
hermes skills install https://example.com/SKILL.md --name my-skill # frontmatterに名前がないとき名前を上書き
hermes skills list --source hub                   # hubからインストールしたスキルを一覧表示
hermes skills check                               # インストール済みhubスキルの上流更新を確認
hermes skills update                              # 必要に応じて上流の変更があるhubスキルを再インストール
hermes skills audit                               # すべてのhubスキルをセキュリティ再スキャン
hermes skills uninstall k8s                       # hubスキルを削除
hermes skills reset google-workspace              # 同梱スキルを「ユーザー変更済み」状態から解除（後述）
hermes skills reset google-workspace --restore    # ローカルの編集を削除し、同梱バージョンも復元
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # スキル設定をエクスポート
hermes skills tap add myorg/skills-repo           # カスタムのGitHubソースを追加
```

### サポートされているhubソース

| ソース | 例 | 備考 |
|--------|---------|-------|
| `official` | `official/security/1password` | Hermesに同梱されるオプションスキル。 |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | `hermes skills search <query> --source skills-sh` で検索可能。skills.shのスラッグがリポジトリフォルダと異なる場合、Hermesはエイリアス形式のスキルを解決します。 |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | ウェブサイトの `/.well-known/skills/index.json` から直接提供されるスキル。サイトまたはドキュメントのURLを使って検索します。 |
| `url` | `https://sharethis.chat/SKILL.md` | 単一ファイルの `SKILL.md` への直接HTTP(S) URL。名前解決: frontmatter → URLスラッグ → インタラクティブプロンプト → `--name` フラグ。 |
| `github` | `openai/skills/k8s` | 直接のGitHubリポジトリ/パスのインストールとカスタムタップ。 |
| `clawhub`、`lobehub`、`claude-marketplace` | ソース固有の識別子 | コミュニティまたはマーケットプレイスの統合。 |

### 統合されたhubとレジストリ

Hermesは現在、以下のスキルエコシステムと検出ソースと統合しています:

#### 1. 公式オプションスキル（`official`）

これらはHermesリポジトリ自体で保守されており、組み込みの信頼でインストールされます。

- カタログ: [公式オプションスキルカタログ](../../reference/optional-skills-catalog)
- リポジトリ内のソース: `optional-skills/`
- 例:

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh（`skills-sh`）

これはVercelの公開スキルディレクトリです。Hermesはこれを直接検索し、スキルの詳細ページを確認し、エイリアス形式のスラッグを解決し、基盤となるソースリポジトリからインストールできます。

- ディレクトリ: [skills.sh](https://skills.sh/)
- CLI/ツールのリポジトリ: [vercel-labs/skills](https://github.com/vercel-labs/skills)
- 公式Vercelスキルリポジトリ: [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- 例:

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. well-knownスキルエンドポイント（`well-known`）

これは、`/.well-known/skills/index.json` を公開するサイトからのURLベースの検出です。これは単一の集中型hubではなく、ウェブ検出の慣例です。

- ライブエンドポイントの例: [Mintlify docs skills index](https://mintlify.com/docs/.well-known/skills/index.json)
- 参照サーバー実装: [vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- 例:

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. 直接のGitHubスキル（`github`）

Hermesは、GitHubリポジトリおよびGitHubベースのタップから直接インストールできます。リポジトリ/パスをすでに知っている場合や、独自のカスタムソースリポジトリを追加したい場合に便利です。

デフォルトのタップ（セットアップなしで閲覧可能）:
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- 例:

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

#### 5. ClawHub（`clawhub`）

コミュニティソースとして統合されたサードパーティのスキルマーケットプレイスです。

- サイト: [clawhub.ai](https://clawhub.ai/)
- HermesソースID: `clawhub`

#### 6. Claudeマーケットプレイス形式のリポジトリ（`claude-marketplace`）

Hermesは、Claude互換のプラグイン/マーケットプレイスのマニフェストを公開するマーケットプレイスリポジトリをサポートします。

既知の統合ソースには次のものが含まれます:
- [anthropics/skills](https://github.com/anthropics/skills)
- [aiskillstore/marketplace](https://github.com/aiskillstore/marketplace)

HermesソースID: `claude-marketplace`

#### 7. LobeHub（`lobehub`）

Hermesは、LobeHubの公開カタログからエージェントエントリを検索し、インストール可能なHermesスキルに変換できます。

- サイト: [LobeHub](https://lobehub.com/)
- 公開エージェントインデックス: [chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- バッキングリポジトリ: [lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- HermesソースID: `lobehub`

#### 8. 直接URL（`url`）

任意のHTTP(S) URLから単一ファイルの `SKILL.md` を直接インストールします — 作者が自分のサイトにスキルをホストしている場合（hubのリスティングなし、入力するGitHubパスなし）に便利です。HermesはURLを取得し、YAML frontmatterをパースし、セキュリティスキャンを行い、インストールします。

- HermesソースID: `url`
- 識別子: URL自体（プレフィックス不要）
- 範囲: **単一ファイルの `SKILL.md`** のみ。`references/` や `scripts/` を含む複数ファイルのスキルにはマニフェストが必要で、上記の他のソースのいずれかを介して公開すべきです。

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

名前解決の順序:
1. SKILL.mdのYAML frontmatter内の `name:` フィールド（推奨 — 適切に作られたスキルにはすべて存在する）。
2. URLパスからの親ディレクトリ名（例: `.../my-skill/SKILL.md` → `my-skill`、または `.../my-skill.md` → `my-skill`）。有効な識別子（`^[a-z][a-z0-9_-]*$`）である場合。
3. TTYを持つターミナルでのインタラクティブプロンプト。
4. 非インタラクティブなサーフェス（TUI内の `/skills install` スラッシュコマンド、Gatewayプラットフォーム、スクリプト）では、`--name` での上書きを指し示す明確なエラー。

```bash
# frontmatterに名前がなく、URLスラッグも役に立たない — 名前を指定する:
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# またはチャットセッション内で:
/skills install https://example.com/SKILL.md --name sharethis-chat
```

信頼レベルは常に `community` です — 他のすべてのソースと同じセキュリティスキャンが実行されます。URLがインストール識別子として保存されるため、リフレッシュしたいときには `hermes skills update` が同じURLから自動的に再取得します。

### セキュリティスキャンと `--force`

hubからインストールされたすべてのスキルは、データ漏洩、プロンプトインジェクション、破壊的なコマンド、サプライチェーンのシグナル、その他の脅威をチェックする**セキュリティスキャナー**を通過します。

`hermes skills inspect ...` は、利用可能な場合に上流のメタデータも表示するようになりました:
- リポジトリURL
- skills.shの詳細ページURL
- インストールコマンド
- 週間インストール数
- 上流のセキュリティ監査ステータス
- well-knownのインデックス/エンドポイントURL

サードパーティのスキルをレビューした上で、危険でないポリシーブロックを上書きしたい場合は `--force` を使用してください:

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

重要な動作:
- `--force` は、caution/warn形式の検出結果に対するポリシーブロックを上書きできます。
- `--force` は `dangerous` のスキャン判定を上書き**しません**。
- 公式オプションスキル（`official/...`）は組み込みの信頼として扱われ、サードパーティ警告パネルを表示しません。

### 信頼レベル

| レベル | ソース | ポリシー |
|-------|--------|--------|
| `builtin` | Hermesに同梱 | 常に信頼される |
| `official` | リポジトリ内の `optional-skills/` | 組み込みの信頼、サードパーティ警告なし |
| `trusted` | `openai/skills`、`anthropics/skills` などの信頼されたレジストリ/リポジトリ | コミュニティソースよりも寛容なポリシー |
| `community` | それ以外すべて（`skills.sh`、well-knownエンドポイント、カスタムGitHubリポジトリ、ほとんどのマーケットプレイス） | 危険でない検出結果は `--force` で上書き可能。`dangerous` の判定はブロックされたままになる |

### 更新のライフサイクル

hubは、インストール済みスキルの上流のコピーを再チェックできるだけの来歴を追跡するようになりました:

```bash
hermes skills check          # どのインストール済みhubスキルが上流で変更されたかを報告
hermes skills update         # 更新が利用可能なスキルのみを再インストール
hermes skills update react   # 特定の1つのインストール済みhubスキルを更新
```

これは、保存されたソース識別子と現在の上流バンドルのコンテンツハッシュを使って、差異を検出します。

:::tip GitHubのレート制限
Skills hubの操作はGitHub APIを使用しており、これは未認証ユーザーに対して60リクエスト/時のレート制限があります。インストールや検索の際にレート制限エラーが表示される場合は、`.env` ファイルに `GITHUB_TOKEN` を設定して制限を5,000リクエスト/時に引き上げてください。この場合、エラーメッセージには実行可能なヒントが含まれます。
:::

### カスタムスキルタップの公開

スキルのキュレーションされたセットを共有したい場合 — チーム、組織、または一般向けに — それらを**タップ**として公開できます: 他のHermesユーザーが `hermes skills tap add <owner/repo>` で追加するGitHubリポジトリです。サーバー不要、レジストリへのサインアップ不要、リリースパイプライン不要。`SKILL.md` ファイルのディレクトリがあるだけです。

#### リポジトリのレイアウト

タップは、次のようにレイアウトされた任意のGitHubリポジトリ（パブリックまたはプライベート — プライベートには `GITHUB_TOKEN` が必要）です:

```
owner/repo
├── skills/                       # デフォルトのパス。タップごとに設定可能
│   ├── my-workflow/
│   │   ├── SKILL.md              # 必須
│   │   ├── references/           # 任意の補助ファイル
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # 任意だが役立つ
```

ルール:
- 各スキルは、タップのルートパス（デフォルト `skills/`）配下の独自のディレクトリに存在します。
- ディレクトリ名がスキルのインストールスラッグになります。
- 各スキルディレクトリには、標準の [SKILL.md frontmatter](#skillmd-format)（`name`、`description`、加えて任意の `metadata.hermes.tags`、`version`、`author`、`platforms`、`metadata.hermes.config`）を持つ `SKILL.md` が含まれている必要があります。
- `references/`、`templates/`、`scripts/`、`assets/` のようなサブディレクトリは、インストール時に `SKILL.md` と一緒にダウンロードされます。
- ディレクトリ名が `.` または `_` で始まるスキルは無視されます。

Hermesは、タップパスのすべてのサブディレクトリを列挙し、それぞれに `SKILL.md` があるかを調べることでスキルを検出します。

#### 最小限のタップの例

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

これをGitHubにプッシュすると、任意のHermesユーザーが購読してインストールできます:

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### デフォルト以外のパス

スキルが `skills/` 配下に存在しない場合（既存のプロジェクトに `skills/` サブツリーを追加するときによくあります）、`~/.hermes/.hub/taps.json` のタップエントリを編集します:

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

`hermes skills tap add` CLIは、新しいタップのデフォルトを `path: "skills/"` にします。別のパスが必要な場合はファイルを直接編集してください。`hermes skills tap list` は、タップごとの有効なパスを表示します。

#### 個々のスキルを直接インストールする（タップを追加せずに）

ユーザーは、リポジトリ全体をタップとして追加することなく、任意の公開GitHubリポジトリから単一のスキルをインストールすることもできます:

```bash
hermes skills install owner/repo/skills/my-workflow
```

ユーザーにレジストリ全体への購読を求めることなく、1つのスキルを共有したいときに便利です。

#### タップの信頼レベル

新しいタップにはデフォルトで `community` の信頼が割り当てられます。そこからインストールされたスキルは標準のセキュリティスキャンを通過し、初回インストール時にサードパーティ警告パネルを表示します。あなたの組織や広く信頼されたソースがより高い信頼を得るべき場合は、`tools/skills_hub.py` の `TRUSTED_REPOS` にそのリポジトリを追加してください（Hermesコアへのプルリクエストが必要です）。

#### タップの管理

```bash
hermes skills tap list                                # 設定されたすべてのタップを表示
hermes skills tap add myorg/skills-repo               # 追加（デフォルトパス: skills/）
hermes skills tap remove myorg/skills-repo            # 削除
```

実行中のセッション内で:

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

タップは `~/.hermes/.hub/taps.json` に保存されます（必要に応じて作成されます）。

## 同梱スキルの更新（`hermes skills reset`）

Hermesは、リポジトリ内の `skills/` に同梱スキルのセットを同梱しています。インストール時およびすべての `hermes update` 時に、同期パスがそれらを `~/.hermes/skills/` にコピーし、各スキル名を同期時のコンテンツハッシュ（**起点ハッシュ**）にマッピングするマニフェストを `~/.hermes/skills/.bundled_manifest` に記録します。

各同期で、Hermesはローカルコピーのハッシュを再計算し、起点ハッシュと比較します:

- **未変更** → 上流の変更を取り込んで安全。新しい同梱バージョンをコピーし、新しい起点ハッシュを記録する。
- **変更あり** → **ユーザー変更済み**として扱われ、以後永久にスキップされる。これによりあなたの編集が踏みつぶされることはない。

この保護は優れていますが、1つの鋭い落とし穴があります。同梱スキルを編集した後で、変更を放棄して `~/.hermes/hermes-agent/skills/` からコピー&ペーストするだけで同梱バージョンに戻したい場合、マニフェストには最後に同期が成功したときの*古い*起点ハッシュが依然として保持されています。あなたの新しくコピー&ペーストした内容（現在の同梱ハッシュ）はその古い起点ハッシュと一致しないため、同期はそれをユーザー変更済みとしてフラグ付けし続けます。

`hermes skills reset` がその抜け道です:

```bash
# 安全: このスキルのマニフェストエントリをクリアする。現在のコピーは保持されるが、
# 次の同期がそれを基準に再ベースライン化するため、将来の更新は通常どおり機能する。
hermes skills reset google-workspace

# 完全な復元: ローカルコピーも削除し、現在の同梱バージョンを再コピーする。
# 元の上流スキルを取り戻したいときに使う。
hermes skills reset google-workspace --restore

# 非インタラクティブ（例: スクリプトやTUIモード内）— --restore の確認をスキップする。
hermes skills reset google-workspace --restore --yes
```

同じコマンドが、チャット内でスラッシュコマンドとして機能します:

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note プロファイル
各プロファイルは、独自の `HERMES_HOME` の下に独自の `.bundled_manifest` を持つため、`hermes -p coder skills reset <name>` はそのプロファイルにのみ影響します。
:::

### スラッシュコマンド（チャット内）

同じコマンドはすべて `/skills` で機能します:

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

公式オプションスキルは、依然として `official/security/1password` や `official/migration/openclaw-migration` のような識別子を使用します。
