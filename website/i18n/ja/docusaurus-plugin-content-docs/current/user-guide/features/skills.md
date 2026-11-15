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

- [同梱スキルカタログ](../../reference/skills-catalog.md)
- [公式オプションスキルカタログ](../../reference/optional-skills-catalog.md)

## Webサイトからインストールする {#install-from-the-website}

公開されている[Skills Hub](/skills)のインストール可能な各カードには、**Install in Hermes** ボタンがあります。これは、URLエンコードされたソース修飾付きのスキルターゲットでHermes Desktopを開きます。オプションスキルには `official/...`、ClawHubには `clawhub/...`、同梱スキルには曖昧な素の名前ではなく明示的なリポジトリパスが使われます。カードのCLIコマンドも同じターゲットを使います:

```text
hermes://skill/install?identifier=official%2Fsecurity%2F1password
```

Desktopは、**Source** と **Install to** の行を分けて **Install “skill-name”?** を表示します。キャンセルすると何も変更されません。確認後、ダイアログには **Installing…**、続いて **Installed** と完了通知が表示されます。エラーはダイアログ内に残るので再試行できます。インストールは通常のSkills Hubのパイプライン（セキュリティスキャン、アクションログ、インストール済みリストの更新）を通ります。確認ダイアログが開いている間にプロファイルや接続を切り替えた場合は、新しいインストール先に対してリンクを開き直してください。リンクでスキャンを回避したり、別のプロファイルを選んだりすることはできません。開いている会話には、次のメッセージで新しいスキルが通知されます。

`skill/install` ルートには更新されたDesktopビルドが必要です。アプリがない場合やリンクを認識しない場合は、カードを展開してCLIのインストールコマンドをコピーしてください。

## 白紙の状態から始める

デフォルトでは、すべてのプロファイルに同梱スキルカタログがシードされ、`hermes update` のたびに新しく同梱されたスキルが追加されます。**同梱スキルなし**のプロファイル — しかもアップデート後も空のまま — が欲しい場合は、2つの方法があります:

**インストール時**（デフォルトの `~/.hermes` プロファイルに適用）: インストーラーには `--no-skills` フラグはありません。Blank Slateセットアップを選ぶと、セットアップステージで同梱カタログをシードするかどうかを尋ねられます。noと答えると、後述のオプトアウトマーカーが書き込まれます。非インタラクティブなインストールではカタログがシードされるため、プロファイルを空にしたい場合は後から `hermes skills opt-out` を実行してください。

**プロファイル作成時**（名前付きプロファイル）:

```bash
hermes profile create research --no-skills
```

**インストール済みのプロファイル**（デフォルトまたは名前付き）では、実行時に切り替えます:

```bash
hermes skills opt-out            # stop future seeding — nothing on disk is touched
hermes skills opt-out --remove   # also delete UNMODIFIED bundled skills (confirms first)
hermes skills opt-in --sync      # undo: remove the marker and re-seed now
```

これらの方法はすべて、プロファイルディレクトリに `.no-bundled-skills` マーカーを書き込みます。マーカーが存在する間、インストーラー、`hermes update`、およびあらゆるスキル同期は、そのプロファイルへの同梱スキルのシードをスキップします。再び有効にするには、マーカーを削除してください（または `hermes skills opt-in` を実行してください）。

:::note デフォルトで安全
`hermes skills opt-out` は*今後の*シードを止めるだけで、すでにディスク上にあるものを削除することはありません。オプションの `--remove` フラグは、同梱スキルが未変更（Hermesがインストールしたバージョンとバイト単位で同一）の場合に**のみ**それらを削除します。あなたが編集したスキル、hubからインストールしたスキル、自分で書いたスキルは常に保持されます。
:::

## スキルの使用

インストールされたすべてのスキルは、自動的にスラッシュコマンドとして利用できます:

```bash
# In the CLI or any messaging platform:
/excalidraw sketch the request flow for our API
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor
/youtube-content summarize https://youtu.be/dQw4w9WgXcQ

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

### 1つのコマンドで複数のスキルを重ねる

先頭でスラッシュコマンドを連ねることで、1つのメッセージで複数のスキルを呼び出せます。先頭にあるすべての `/skill` トークン（最大5つ）がロードされ、残りがあなたの指示になります:

```bash
/github-pr-workflow /test-driven-development fix issue #123 and open a PR
```

パースはインストール済みのスキルではない最初のトークンで停止するため、たまたま `/` で始まる引数（ファイルパスなど）が飲み込まれることはありません:

```bash
/ocr-and-documents ~/.hermes/cache/scratch/scan.pdf extract the tables   # loads one skill; ~/.hermes/cache/scratch/scan.pdf is the argument
```

繰り返し使う組み合わせには、[スキルバンドル](#skill-bundles)を優先してください — 1つの短いコマンドで同じ効果が得られます。

（プランモードも同じように動作しますが、現在は組み込みコマンドです: `/plan [request]` はHermesに対して、必要に応じてコンテキストを確認し、タスクを実行する代わりにMarkdownの実装プランを書き、結果をアクティブなワークスペース/バックエンドの作業ディレクトリを基準とした `.hermes/plans/` 配下に保存するよう指示します。）

自然な会話を通じてスキルとやり取りすることもできます:

```bash
hermes chat --toolsets skills -q "What skills do you have?"
hermes chat --toolsets skills -q "Show me the axolotl skill"
```

## ソースからスキルを学習する（`/learn`）

`/learn` は、すでに知っていること — あるいは参考資料の山 — を、`SKILL.md` を手書きすることなく再利用可能なスキルに変える手早い方法です。これは自由形式です: *説明できるものなら何でも*指定すれば、エージェントがすでに持っているツールで資料を集め、[社内の作成基準](#skillmd-format)（60文字以内の説明、標準のセクション順序、Hermesツールに即した表現、でっち上げたコマンドを含まない）に従ったスキルを作成します。

```bash
# A local SDK or doc directory — read with read_file / search_files
/learn the REST client in ~/projects/acme-sdk, focus on auth + pagination

# An online doc page — fetched with web_extract
/learn https://docs.example.com/api/quickstart

# The workflow you just walked the agent through in this conversation
/learn how I just deployed the staging server

# Pasted notes / a described procedure
/learn filing an expense: open the portal, New > Expense, attach the receipt, submit

# A whole book, paper stack, or large docs corpus — becomes a knowledge-base skill
/learn ~/books/designing-data-intensive-applications.pdf
```

### 大きなソースはナレッジベーススキルになる

ソースが書籍、論文の束、仕様書、大きなドキュメントフォルダの場合、エージェントはそれを1つのファイルに詰め込んだり、情報の欠落した要約に縮めたりはしません。代わりに、**拡張可能なナレッジベーススキル**を作成します: ソースの中核となるメンタルモデルとインデックスを持つ簡潔な `SKILL.md` に加え、`references/` 配下に章やトピックごとに1つの要約ファイル（ソースに見合う場合は用語集やチートシートも）を置きます。参照ファイルは、質問がそれを必要とするまでコストがかかりません — エージェントは `skill_view` でオンデマンドにロードするため、問い合わせのコストはソースではなく回答に比例したままです。同じトピックについて新しい資料で `/learn` を再実行すると、重複を作るのではなく既存のスキルに取り込まれます。

この要約は構造 — フレームワーク、定義、判断ルール、アンチパターン — を統合するもので、ソーステキストの文章を再現することは決してありません。

ライブのエージェントが資料集めを行うため、`/learn` はCLI、メッセージングゲートウェイ、TUI、ダッシュボードで同じように動作し、別個の取り込みエンジンがないため、どのターミナルバックエンド（ローカル、Docker、リモート）でも動作します。**ダッシュボード**では、Skillsページに **Learn a skill** ボタンがあり、ディレクトリ欄、URL欄、自由形式のテキストボックスを備えたパネルが開きます。これは `/learn` リクエストを組み立ててチャットで実行します。

モデルツールのフットプリントはありません: `/learn` は基準に沿ったプロンプトを組み立て、通常のターンとしてエージェントに渡します。エージェントは結果を `skill_manage` ツールで保存するため、有効にしている場合は[書き込み承認ゲート](#gating-agent-skill-writes-skillswrite_approval)が適用されます。

## 漸進的開示

スキルは、トークン効率の良いロードパターンを使用します:

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

エージェントは、実際に必要になったときにのみスキルの完全な内容をロードします。

### 会話中に追加されたスキル

エージェントが選択するスキルのリストは会話のシステムプロンプトの一部であり、プロバイダーのプロンプトキャッシュが機能し続けるよう固定されています。会話が開いている間にスキルがインストール、作成、または削除された場合（Skills Hub、別のターミナルでの `hermes skills install`、プラグイン、キュレーター、またはエージェント自身によって）、エージェントへの次のメッセージには、新しいスキルの名前とその説明、および削除されたスキルを示す短い注記が付きます。この注記は変更ごとに1回送信されます。システムプロンプト自体は次の圧縮時に変更を取り込みます。このために `/reload-skills` は必要ありませんが、`/skill-name` スラッシュコマンドの更新には引き続き使えます。

## SKILL.md 形式 {#skillmd-format}

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
platforms: [macos, linux]     # Optional — restrict to specific OS platforms
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    fallback_for_toolsets: [web]    # Optional — conditional activation (see below)
    requires_toolsets: [terminal]   # Optional — conditional activation (see below)
    config:                          # Optional — config.yaml settings
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
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders, FindMy)
platforms: [macos, linux]     # macOS and Linux
```

設定すると、互換性のないプラットフォームでは、スキルはシステムプロンプト、`skills_list()`、スラッシュコマンドから自動的に非表示になります。省略した場合、スキルはすべてのプラットフォームでロードされます。

## スキルの出力とメディアの配信 {#skill-output-and-media-delivery}

スキルの応答（またはあらゆるエージェントの応答）にメディアファイルへの素の絶対パス — 例えば `/home/user/screenshots/diagram.png` — が含まれている場合、ゲートウェイはそれを自動検出して表示テキストから取り除き、生のパスをメッセージに残す代わりに、ユーザーのチャットにファイルをネイティブに配信します（Telegramの写真、Discordの添付ファイルなど）。

特に音声については、`[[audio_as_voice]]` ディレクティブにより、対応するプラットフォーム（Telegram、WhatsApp）で音声ファイルがネイティブのボイスメッセージのバブルに昇格されます。

### ドキュメント形式での配信を強制する: `[[as_document]]`

インラインプレビューの**逆**が欲しいこともあります: 再圧縮された画像バブルではなく、ダウンロード可能な添付ファイルとしてファイルを配信したい場合です。典型的な例は高解像度のスクリーンショットやチャートです — Telegramの `sendPhoto` はそれを1280 pxで約200 KBに再圧縮し、可読性を損ないます。`sendDocument` で送られた1〜2 MBのPNGは、元のバイトがそのまま保たれます。

応答（またはその中の任意のテキスト — 通常は最後の行）にリテラルのディレクティブ `[[as_document]]` が含まれている場合、その応答から抽出されたすべてのメディアパスは、画像バブルではなくドキュメント/ファイルの添付として配信されます:

```
Here is your rendered chart:

/home/user/.hermes/cache/chart-q4-2025.png

[[as_document]]
```

ディレクティブは配信前に取り除かれるため、ユーザーがそれを目にすることはありません。粒度は意図的に応答ごとのオール・オア・ナッシングです: `[[as_document]]` を1回出力すると、同じ応答内のすべての画像パスがドキュメントとして配信されます。これは `[[audio_as_voice]]` のスコープと同じです。

スキルからこれを使うのは次のような場合です:

- ユーザーがファイルとして必要とするスクリーンショットやチャートを生成する場合（別のツールでの編集、アーカイブ、そのままの状態での共有のため）。
- デフォルトの非可逆プレビューでは細部が見えにくくなる場合（小さな文字、ピクセル精度の図、色に敏感なレンダリング）。

個別のドキュメント経路がないプラットフォーム（例: SMS）は、持っている添付の仕組みにフォールバックします。

### 条件付きアクティベーション（フォールバックスキル）

スキルは、現在のセッションでどのツールが利用可能かに基づいて、自身を自動的に表示または非表示にできます。これは**フォールバックスキル** — プレミアムツールが利用できないときにのみ表示されるべき無料またはローカルの代替手段 — に最も役立ちます。

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Show ONLY when these toolsets are unavailable
    requires_toolsets: [terminal]     # Show ONLY when these toolsets are available
    fallback_for_tools: [web_search]  # Show ONLY when these specific tools are unavailable
    requires_tools: [terminal]        # Show ONLY when these specific tools are available
```

| フィールド | 動作 |
|-------|----------|
| `fallback_for_toolsets` | 列挙されたツールセットが利用可能なときスキルは**非表示**になる。それらが欠けているときに表示される。 |
| `fallback_for_tools` | 同様だが、ツールセットの代わりに個々のツールをチェックする。 |
| `requires_toolsets` | 列挙されたツールセットが利用不可のときスキルは**非表示**になる。それらが存在するときに表示される。 |
| `requires_tools` | 同様だが、個々のツールをチェックする。 |

**例:** `fallback_for_toolsets: [web]` を宣言したcurlベースの検索スキルは、webツールセットが利用可能な間は非表示のままで、それが無効化されたときにのみ表示されます。これにより、エージェントは引き続き検索手段を持てます。

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

設定されると、宣言された環境変数は `execute_code` と `terminal` のサンドボックスに**自動的に渡されます** — スキルのスクリプトは `$TENOR_API_KEY` を直接使用できます。スキル以外の環境変数については、`terminal.env_passthrough` 設定オプションを使用してください。詳細は[環境変数のパススルー](../security.md#environment-variable-passthrough)を参照してください。

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

詳細は[スキル設定](../configuration.md#skill-settings)および[スキルの作成 — 設定項目](/docs/developer-guide/creating-skills#config-settings-configyaml)を参照してください。

## スキルディレクトリ構造

```text
~/.hermes/skills/                  # Single source of truth
├── mlops/                         # Category directory
│   ├── axolotl/
│   │   ├── SKILL.md               # Main instructions (required)
│   │   ├── references/            # Additional docs
│   │   ├── templates/             # Output formats
│   │   ├── scripts/               # Helper scripts callable from the skill
│   │   ├── examples/              # Referenced example outputs
│   │   └── assets/                # Supplementary files
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # Agent-created skill
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Skills Hub state
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # Tracks seeded bundled skills
```

サードパーティのURLおよびGitHubからのインストールには、`SKILL.md` に加えて、それが `references/`、`templates/`、`scripts/`、`assets/`、`examples/` 配下で参照している正確なローカルファイルが含まれます。参照されていないリポジトリのファイルはコピーされません。Hermesは隔離されたバンドル全体をスキャンし、ソースURL、正確なコンテンツハッシュ、スキャナーのバージョン、検出結果、タイムスタンプ、新規かキャッシュかの状態を `skills/.hub/lock.json` に記録します。

### 助言的なSkillEvaluatorスキャン

組み込みのセキュリティスキャナー（上記のインストールポリシーを強制するもの）に加えて、Hermesはセカンドオピニオンとして、すべてのhubインストールで[NVIDIA SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator)のTier 1チェックを実行できます。Tier 1は決定論的でキー不要です — PII検出（漏洩したメールアドレス、個人のパス、接続文字列）、Unicode密輸の検出、スクリプトのlint、ライセンスの準拠、および[NVIDIA SkillSpector](https://github.com/NVIDIA/SkillSpector)による静的セキュリティスキャンです。

このスキャンは**助言のみ**です: 検出結果はインストールの確認前にファイルと行とともに表示され、インストールはそのまま続行されます。本物の認証情報のように見える検出結果（秘密鍵、クラウドのアクセスキー、トークン、認証情報付きの接続文字列）は赤で強調表示されるため、判断する前にフラグの付いた行を確認できます。PII系の検出結果は情報提供のみです — 上流のスキャナーには既知の誤検出のクラス（例: `git@github.com` のSSH構文、ドキュメントの例のメールアドレス）があるため、それらが何かをブロックすることはありません。

有効にするには、オプションのスキャナーバイナリをインストールします（2つ目は `security` チェックを動かすもので、ない場合そのチェックは単に「not run」と報告されます）:

```bash
uv tool install --python 3.13 \
  "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.1.0"
uv tool install "git+https://github.com/NVIDIA/SkillSpector.git@v2.9.5"
```

バイナリがPATH上にない場合、スキャンは黙ってスキップされます。完全に無効にするには:

```yaml
skills:
  tier1_advisory: false
```

ダッシュボードのBrowse-hubのスキャンボタンは、組み込みスキャナーの判定と並べて、同じ助言データを応答（`tier1` フィールド）で返します。

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

- **作成はローカル、更新はその場で**: エージェントが作成する新しいスキルは `~/.hermes/skills/`（設定されている場合は `skills.create_dir` — 後述）に書き込まれます。既存のスキルは、エージェントが `patch`（ピンポイントまたは全体の書き換え）、`write_file`、`remove_file`、`delete` などの `skill_manage` アクションを使うと、`external_dirs` 配下のスキルも含めて、見つかった場所で変更されます。
- **外部ディレクトリは書き込み保護の境界ではない**: 外部スキルディレクトリがHermesのプロセスから書き込み可能な場合、エージェント管理のスキル更新によってそのディレクトリ内のファイルが変更される可能性があります。共有の外部スキルを読み取り専用に保つ必要がある場合は、ファイルシステムの権限や、別のプロファイル/ツールセットの構成を使ってください。
- **優先順位**: 同じスキル名が複数のディレクトリに存在する場合、優先度の高いディレクトリがあらゆる場所 — スキルインデックス、`skills_list`、`hermes skills list`、`skill_view`、スラッシュコマンド、`-s` のプリロード、cron — で優先されます: `project → local (~/.hermes/skills/) → skills.create_dir → external_dirs`。隠されたコピーはシャドウされたものとしてログに記録されます。[重複したスキル名](#duplicate-skill-names)を参照してください。
- **完全な統合**: 外部スキルは、システムプロンプトのインデックス、`skills_list`、`skill_view`、`/skill-name` スラッシュコマンドに表示されます — ローカルスキルと変わりません。
- **存在しないパスは静かにスキップされる**: 設定されたディレクトリが存在しない場合、Hermesはエラーなしでそれを無視します。すべてのマシンに存在するとは限らないオプションの共有ディレクトリに便利です。

### 例

```text
~/.hermes/skills/               # Local (primary, read-write)
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # External (shared, mutable if writable)
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

4つすべてのスキルがスキルインデックスに表示されます。`my-custom-workflow` という名前の新しいスキルをローカルで作成すると、外部バージョンを隠します。

## スキル作成先のリダイレクト（`skills.create_dir`）

デフォルトでは、エージェントは新しいスキルをプロファイルローカルの `~/.hermes/skills/` に書き込みます。エージェントが作成したスキルを別の場所 — 共有の「brain」ディレクトリ、gitで追跡されたリポジトリ、フリート全体のスキルボリューム — に置きたい場合は、`skills` セクション配下に `create_dir` を設定します:

```yaml
skills:
  create_dir: /opt/brain/skills
```

これによって変わること:

- **`skill_manage` の作成はそこに書き込む。** 新しいスキル（カテゴリのサブディレクトリを含む）は、ローカルのスキルディレクトリではなく `create_dir` 配下に作成されます。ディレクトリが存在しない場合は、最初の書き込み時に作成されます。
- **エージェントへの指示が設定に従う。** スキルの作成パスに言及するエージェント向けのすべての指示 — `skill_manage` ツールの説明と関連するプロンプトテキスト — は設定されたディレクトリを動的にレンダリングするため、エージェントはそこにスキルを作成するよう指示されます。システムプロンプトの上書きやファイルシステムの小細工は必要ありません。
- **ディレクトリは完全に統合される。** `create_dir` 配下のスキルはローカルディレクトリと並んでスキャンされます: スキルインデックス、`skills_list`、`skill_view`、スラッシュコマンドに表示され、ローカルスキルと同様にパッチや削除ができます。
- **それ以外はすべてローカルのまま。** 既存のスキルは引き続き、どこにあってもその場で変更されます。同梱スキルの同期、hub、キュレーターは、プロファイルローカルのディレクトリで動作し続けます。

パスは `~` の展開と `${VAR}` の置換をサポートします。相対パスはHermesホームを基準に解決されます。`create_dir` をローカルのスキルディレクトリに設定するのは、未設定のままにするのと同じです。


## プロジェクトローカルのスキル

リポジトリは独自のスキルを持つことができ、それらはそのプロジェクト内で開始されたセッションでのみ有効になります — 他のエージェントハーネスがリポジトリローカルの設定に使うのと同じパターンです。gitチェックアウト内でHermesを起動すると、次の場所でスキルを探します:

```text
<project-root>/.hermes/skills/    # Hermes-native location
<project-root>/.agents/skills/    # cross-tool convention (shared with other agent CLIs)
```

プロジェクトルートは、`.git` を含む最も近い祖先ディレクトリです（worktreeとサブモジュールも対象になります）。

### プロジェクトを信頼する

スキルはエージェントが従う手順書であるため、Hermesは任意にクローンされたリポジトリからそれらを自動ロード**しません**。プロジェクトスキルのあるリポジトリで初めてHermesを実行すると、バナーに通知が表示されます:

```text
◆ 3 project skill(s) found in /home/you/myproject but not loaded — run `hermes skills trust` to enable them.
```

リポジトリを一度信頼します（リポジトリ内から、またはパスを渡して）:

```bash
hermes skills trust             # trust the current repo
hermes skills trust ~/myproject # or explicitly
hermes skills untrust           # revoke
```

信頼されたルートは、`~/.hermes/config.yaml` の `skills.trusted_project_dirs` に保存されます。機能を完全にオフにする（スキャンも通知もなし）には、`skills.project_discovery: false` を設定します。

### 優先順位

プロジェクトスキルは**最も優先度の高い階層**です: `project → local (~/.hermes/skills/) → skills.create_dir → external_dirs`。`deploy` という名前のプロジェクトスキルは、そのリポジトリ内のセッションでは、同名のプロファイルスキルや同梱スキルを上書きします — それが狙いです: リポジトリに同梱されたスキルは、グローバルなプロファイルに触れることなく、自分の本拠地で優先されます。プロジェクトスキルにはエージェントのスキルインデックスで `[project]` タグが付くため、出所が見えるようになっています。

外部ディレクトリと同様に、プロジェクトのスキルディレクトリはリポジトリ所有として扱われます: 自律的なスキルのメンテナンス（キュレーター）がそれらを変更することはなく、エージェントが新しく作成するスキルは常に `~/.hermes/skills/` に置かれます。

### 重複したスキル名 {#duplicate-skill-names}

すべてのサーフェスは、スキル名を同じ方法で解決します:

- **ディレクトリをまたぐ場合**、優先度の高いディレクトリが優先されます（`project → local → skills.create_dir → external_dirs`）。同じ相対パスの場合も同様です（`productivity/xdup` がローカルと外部にあれば、ローカルのコピーがロードされます）。シャドウされたコピーは非表示になり、警告がログに記録されます。
- **1つのディレクトリ内では**、名前を共有する2つの*異なる*スキルの間で推測が行われることはありません。どちらも正確な相対パス（例えば `a/one` と `b/two`）で一覧表示され、素の名前でロードすると `Ambiguous skill name dup-demo: use one of a/one, b/two` で失敗します。このメッセージは `hermes -s dup-demo` やcronジョブが報告するものでもあります。1つのスキルの同一のコピー（シンボリックリンクのビューやバイト単位で同一のコピー）は、最も浅いパスに解決されます。

プラグインのスキルは独自の `plugin:skill` という名前を使うため、これらと衝突することはありません。

### スキャン時の隔離

信頼はリポジトリ単位の判断ですが、リポジトリのスキルの内容は `git pull` のたびに変わります。このギャップを埋めるため、すべてのプロジェクトスキルは、インデックスに入る前に、Skills Hubのインストールで使われるのと同じセキュリティスキャナーでスキャンされます。スキャンの判定が **dangerous**（プロンプトインジェクションの指示、認証情報の持ち出しコマンド、隠しテキストの小細工）のスキルは隔離されます: スキルインデックス、`skills_list`、スラッシュコマンドに表示されず、名前によるロードも説明付きのエラーで拒否されます。スキャン結果は `~/.hermes/cache/project_skill_scans/`（リポジトリ内には決して置かれない）にコンテンツハッシュでキャッシュされ、スキルの内容が変わると自動的に再実行されます。

### 非インタラクティブなサーフェス（cron、API、ACP）

cronジョブやその他の非インタラクティブなサーフェスは、インタラクティブに行った信頼の判断を引き継ぎます — プロンプトを表示することも、自動的に信頼することもありません。プロジェクトルートは、サーフェスの作業ディレクトリ（cronジョブの `workdir`。ターミナルツールが使うのと同じ仕組み）から解決されます。`workdir` が以前に信頼したリポジトリ内にあるcronジョブは、そのリポジトリのプロジェクトスキルをロードします。信頼されていない、または未決定のリポジトリ内のジョブは何もロードしません。

TUIとDesktopでは、プロジェクトルートは各**セッションのワークスペース**（サイドバーに表示される、またはワークスペースピッカーで設定されるディレクトリ）に従います。そのため、信頼されたリポジトリ内で `hermes --tui` を起動すると、`terminal.cwd` がデフォルトのプレースホルダー `.` のままでも、そのプロジェクトスキルがスラッシュコマンドとして登録されます。2つのリポジトリで開いた2つのセッションは、それぞれ自身のものを参照します。

## スキルバンドル {#skill-bundles}

スキルバンドルは、複数のスキルを1つのスラッシュコマンドにまとめる小さなYAMLファイルです。`/<bundle-name>` を実行すると、バンドルに列挙されたすべてのスキルが一度にロードされます — 特定のタスクで常に同じスキルのセットが役立つ場合に便利です。

### 簡単な例

```bash
# Create a bundle for backend feature work
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work — review, test, PR workflow"
```

その後、CLIまたは任意のゲートウェイプラットフォームで:

```
/backend-dev refactor the auth middleware
```

エージェントは3つのスキルすべてが1つのユーザーメッセージにロードされた状態で受け取り、スラッシュコマンドの後のテキストはユーザーの指示として添付されます。

### YAMLスキーマ

バンドルは **`~/.hermes/skill-bundles/<slug>.yaml`** に置かれ、次のような形をしています:

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

フィールド:
- `name`（任意 — デフォルトはファイル名の語幹）— バンドルの表示名。スラッシュコマンド用にハイフン区切りのスラッグに正規化されます（`Backend Dev` → `/backend-dev`）。
- `description`（任意）— `/bundles` と `hermes bundles list` に表示される短いテキスト。
- `skills`（必須、空でないリスト）— スキル名、またはスキルディレクトリからの相対パス。`/<skill-name>` に渡すのと同じ識別子を使います。
- `instruction`（任意）— ロードされたスキルの内容の前に付け加えられる追加のガイダンス。「これらを常にどう組み合わせて使うか」を明文化するのに便利です。

### バンドルの管理

```bash
# List all installed bundles
hermes bundles list

# Inspect one bundle
hermes bundles show backend-dev

# Create a bundle interactively (omit --skill flags to enter them one per line)
hermes bundles create research

# Overwrite an existing bundle
hermes bundles create backend-dev --skill ... --force

# Delete a bundle
hermes bundles delete backend-dev

# Re-scan ~/.hermes/skill-bundles/ and report changes
hermes bundles reload
```

チャットセッション内では、`/bundles` がインストール済みのすべてのバンドルとそのスキルを一覧表示します。

### 動作

- **スラッグが衝突した場合、バンドルは個々のスキルより優先されます。** `research` という名前のバンドルを作り、`research` というスキルもある場合、`/research` はバンドルを呼び出します。これは意図的です — バンドルに名前を付けることで、あなたがそれを選んだからです。
- **欠けているスキルはスキップされ、致命的にはなりません。** バンドルが `skill-foo` を列挙していてそれをインストールしていない場合でも、バンドルは解決できたスキルをロードし、エージェントにはスキップされたものを列挙した注記が渡されます。
- **バンドルはあらゆるサーフェスで動作します** — インタラクティブCLI、TUI、ダッシュボードのチャット、すべてのゲートウェイプラットフォーム（Telegram、Discord、Slackなど）— ディスパッチが個々のスキルコマンドと同じ場所に集約されているためです。
- **バンドルはプロンプトキャッシュを無効化しません。** `/<skill-name>` と同じように、呼び出し時に新しいユーザーメッセージを生成します — システムプロンプトの変更はありません。

### 各スキルを手動でインストールするよりバンドルが優れている場合

次のような場合にバンドルを使います:
- 繰り返し行うタスクで、常に同じスキルを組み合わせる場合（`/backend-dev`、`/release-prep`、`/incident-response`）。
- 複数の `/skill` 呼び出しを続けて入力するより、短く済むメンタルモデルが欲しい場合。
- バンドルのYAMLを共有のdotfilesリポジトリにチェックインし、`~/.hermes/skill-bundles/` にシンボリックリンクすることで、チーム全体の「タスクプロファイル」を配布したい場合。

バンドルは単なるYAMLのエイリアスであり、スキルをインストールしてくれるわけではありません。スキル自体はすでに（`~/.hermes/skills/` または外部スキルディレクトリに）存在している必要があります。そうでない場合、バンドルの呼び出しは欠けているものを単にスキップします。

## エージェント管理スキル（skill_manage ツール） {#agent-managed-skills-skill_manage-tool}

エージェントは、`skill_manage` ツールを介して自身のスキルを作成、更新、削除できます。これはエージェントの**手続き的メモリ**です — 自明でないワークフローを見つけ出したとき、将来再利用するためにそのアプローチをスキルとして保存します。

スキルとメモリは、自己改善ループの中で連携して機能します: メモリは常にコンテキストにあるべき小さな永続的な事実を保存し、スキルは関連するときにのみロードされるべき長い手順を保存します。バックグラウンドレビューはセッション後にスキルの変更を提案またはステージングできますが、後述の書き込み承認ゲートを使えば、それらの変更が反映される前に人間によるレビューを必須にできます。

### エージェントがスキルを作成するとき

システムプロンプトは、自明でないワークフローを将来の再利用のために `skill_manage` で記録するようエージェントに求めています。実際には次のような場合が該当します:

- 繰り返す価値のある複数ステップのワークフローを見つけ出したとき
- エラーや行き詰まりに遭遇し、うまくいく道筋を見つけたとき
- ユーザーがそのアプローチを修正したとき

### スキルのエントリはどのようなものか

スキルとは、ある種類のタスクを、あなたの仕様に沿って最も効率的かつ正しく行うための指示です: 順序立てた手順、うまくいくコマンドやツール呼び出し、結果をどのような形にしたいか、そして時間を浪費させる落とし穴です。フォアグラウンドのターン、バックグラウンドレビュー、キュレーターの統合パスのいずれで書かれても、それが捉えるのは**ログではなく教訓**です: 落とし穴とは、一般化可能なルールに、*なぜ*（仕組み）を説明する一節を加えたものであり、影響するステップに付けられ、一度だけ述べられます。インシデントの経緯、PRやissueの番号、日付、引用されたチャットはスキルの内容ではありません。ルールは、その背後にある経緯なしに成り立たなければなりません。常に適用されるルールは `SKILL.md` 自体に置き、`references/` にはトピックごとに名付けられた少数のファイル（判断表、レシピ、プロバイダーの癖）を置き、セッションごとに1ファイルずつ積み上げるのではなく、その場で拡張します。スキルはまた、毎ターンすでにロードされているもの（リポジトリの `AGENTS.md`、ツールのスキーマ）を繰り返し述べることもしません。

`skill_manage` は、`create`、`SKILL.md` へのパッチ、`references/` への書き込みに対して助言的なlinterを実行し、その検出結果をツールの結果で返します。この形のために特に3つのルールがあります: `incident-log-shape`（PR/issue番号が密集した本文）、`references-sprawl`（60を超える参照ファイル）、`oversized-body`（約24k文字を超える `SKILL.md` の本文 — `skill_view` はファイル全体をロードし、それはセッションの残りの間コンテキストに残ります）。これらは警告するだけで、書き込みをブロックすることはありません。

### アクション

| アクション | 用途 | 主なパラメーター |
|--------|---------|------------|
| `create` | ゼロからの新しいスキル | `name`、`content`（完全なSKILL.md）、任意で `category` |
| `patch` | ピンポイントの修正（推奨） | `name`、`old_string`、`new_string` |
| `content` 付きの `patch` | 大規模な構造の書き換え（SKILL.md全体を置き換える。`edit` は従来のエイリアス） | `name`、`content` |
| `delete` | スキルを完全に削除する | `name` |
| `write_file` | 補助ファイルの追加/更新 | `name`、`file_path`、`file_content` |
| `remove_file` | 補助ファイルの削除 | `name`、`file_path` |

各アクションは独自の形として提示されます: テキストのスロットは1つのアクションにのみ属します（`content` → create / 全体の書き換え、`new_string` → ピンポイントのパッチ、`file_content` → write_file）。別のアクションのスロットを持つ操作 — 例えば `create` に `file_content` を付けたもの — はツールスキーマに対して無効であり（文法制約付きのローカルバックエンドはそれを出力しません）、それでも届いた場合は、テキストが入っているキーと移動先を示すエラーとともに、**バッチ内のどの操作も適用される前に**拒否されます。

:::tip
更新にはピンポイントの `patch` が推奨されます — 変更されたテキストのみがツール呼び出しに現れるため、全体の書き換えよりもトークン効率が良いです。
:::

### エージェントのスキル書き込みをゲートする（`skills.write_approval`） {#gating-agent-skill-writes-skillswrite_approval}

デフォルトでは、エージェントは自由にスキルを書き込みます — ターンの後に実行される[バックグラウンドの自己改善レビュー](./memory.md#controlling-memory-writes-write_approval)からの書き込みも含みます。すべてのスキル書き込みを事前に承認したい場合（学んだことを誤って判断する小さなモデル、セキュアな環境、あるいは単に自己改善ループを目で確認したい場合）は、書き込み承認ゲートをオンにします:

```yaml
skills:
  write_approval: false     # false = write freely (default) | true = require approval
```

`write_approval: true` の場合、すべての `skill_manage` の書き込み（create / edit / patch / delete / write_file / remove_file）はコミットされる代わりに**ステージング**されます — SKILL.mdはインラインでレビューするには大きすぎるため、書き込みがフォアグラウンドのターンから来たかバックグラウンドレビューから来たかにかかわらず、ステージングが適用されます。ステージングされた書き込みは `~/.hermes/pending/skills/` 配下で再起動後も保持され、危険なコマンドと同じおなじみの承認/拒否フローでレビューされます:

```
/skills pending             # list staged skill writes + a one-line gist each
/skills diff <id>           # full unified diff (best viewed in CLI or dashboard)
/skills approve <id>        # apply it (or 'all')
/skills reject <id>         # drop it (or 'all')
/skills approval on         # turn the gate on (or 'off') and persist it
```

レビューのサーフェスは、インタラクティブCLIとメッセージングプラットフォームで動作します（チャットのバブル用にdiffの出力は切り詰められます — 完全なdiffはCLIか、保留中のJSONファイルで読んでください）。メモリの書き込みにも `memory.write_approval` で同じゲートがあります — [メモリ書き込みの制御](./memory.md#controlling-memory-writes-write_approval)を参照してください。

エージェントが*特定の*スキルを一切変更できないようにするには、それをロックします: `hermes curator lock <skill>`（またはDesktopのロックスイッチ）。ロックされたスキルは引き続き読み取りや自動ロードが可能ですが、`skill_manage`、`write_file`、`patch`、`terminal` はすべて、あらゆる種類のセッションでその変更を拒否します。あなた自身の編集は引き続き機能します。[スキルのロック](./curator.md#locking-a-skill)を参照してください。

> 別の設定である `skills.guard_agent_created` はコンテンツスキャナー
> （危険なパターンのヒューリスティック）であり、承認ゲートではありません — 両者は
> 独立しています。[エージェントが作成したスキル書き込みのガード](../configuration.md#guard-on-agent-created-skill-writes)を参照してください。

## Skills Hub

オンラインレジストリ、`skills.sh`、直接のwell-knownスキルエンドポイント、公式オプションスキルから、スキルを閲覧、検索、インストール、管理します。

フィルターなしの検索（CLI、TUI、ダッシュボード）には、外部レジストリをカバーするキャッシュされた集中インデックスから応答します。このインデックスは定期的に再構築されるため、クエリに一致するものがない場合、Hermesは `skills.sh`、ClawHub、LobeHub、well-knownエンドポイントにも直接問い合わせます — 数分前に公開されたスキルでも表示されます。この追加パスには検索の予算のうち最大8秒しか割り当てられないため、遅いレジストリのせいで見つからない場合に長く待たされることはありません。カスタムのGitHubタップはこのフォールバックの対象外であり（`--source github` で検索するか、インデックスが追いつくのを待ってください）、`--source nvidia` のようなプロバイダーフィルターはこれをトリガーしません（それらのレジストリはプロバイダーのデータを持っていません）。

### よく使うコマンド

```bash
hermes skills browse                              # Browse all hub skills (official first)
hermes skills browse --source official            # Browse only official optional skills
hermes skills search kubernetes                   # Search all sources
hermes skills search react --source skills-sh     # Search the skills.sh directory
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # Preview before installing
hermes skills install openai/skills/k8s           # Install with security scan
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # Direct URL (+ referenced support files)
hermes skills install https://example.com/SKILL.md --name my-skill # Override name when frontmatter has none
hermes skills list --source hub                   # List hub-installed skills
hermes skills check                               # Check installed hub skills for upstream updates
hermes skills update                              # Reinstall hub skills with upstream changes when needed
hermes skills audit                               # Re-scan all hub skills for security
hermes skills uninstall k8s                       # Remove a hub skill
hermes skills reset google-workspace              # Un-stick a bundled skill from "user-modified" (see below)
hermes skills reset google-workspace --restore    # Also restore the bundled version, deleting your local edits
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # Export skill config
hermes skills tap add myorg/skills-repo           # Add a custom GitHub source
```

### サポートされているhubソース

| ソース | 例 | 備考 |
|--------|---------|-------|
| `official` | `official/security/1password` | Hermesに同梱されるオプションスキル。 |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | `hermes skills search <query> --source skills-sh` で検索可能。skills.shのスラッグがリポジトリフォルダと異なる場合、Hermesはエイリアス形式のスキルを解決します。 |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | ウェブサイトの `/.well-known/skills/index.json` から直接提供されるスキル。サイトまたはドキュメントのURLを使って検索します。 |
| `url` | `https://sharethis.chat/SKILL.md` | `SKILL.md` と明示的に参照される補助ファイルへの直接HTTP(S) URL。名前解決: frontmatter → URLスラッグ → インタラクティブプロンプト → `--name` フラグ。 |
| `github` | `openai/skills/k8s` | 直接のGitHubリポジトリ/パスのインストールとカスタムタップ。 |
| `clawhub`、`lobehub`、`browse-sh` | ソース固有の識別子 | コミュニティまたはマーケットプレイスの統合。 |

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
- [huggingface/skills](https://github.com/huggingface/skills)
- [NVIDIA/skills](https://github.com/NVIDIA/skills) — NVIDIAが検証したスキル（署名済みの `skill.oms.sig` + ガバナンス用の `skill-card.md`）
- [wandb/skills](https://github.com/wandb/skills)、[qdrant/skills](https://github.com/qdrant/skills)、[pinecone-io/skills](https://github.com/pinecone-io/skills)、[makenotion/skills](https://github.com/makenotion/skills)、[Airtable/skills](https://github.com/Airtable/skills)、[Shopify/Shopify-AI-Toolkit](https://github.com/Shopify/Shopify-AI-Toolkit) — 各ベンダーが自社製品向けに提供するスキル
- [garrytan/gstack](https://github.com/garrytan/gstack)
- [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) と [synthetic-sciences/openscience](https://github.com/synthetic-sciences/openscience) — 約480の科学研究スキル（バイオインフォマティクス、化学、物理学、MLトレーニング、学術ツール）で、1つの `science` カテゴリにまとめられています。コミュニティの信頼: すべてのインストールでセキュリティスキャンが行われます。多くは独自のライセンスを持つサードパーティのツールをラップしています（一部はGPL。KEGGは学術以外の利用に商用ライセンスが必要）— 各スキルの前提条件を確認してください。

- 例:

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

**カテゴリのグループ分け（`skills.sh.json`）。** GitHubタップは、[skills.shのスキーマ](https://skills.sh/schemas/skills.sh.schema.json)に従った `skills.sh.json` ファイルをリポジトリのルートに含めることができます。その `groupings`（それぞれ `title` とスキル名のリストを持つ）はインデックス作成時に読み込まれ、タグから推測したものではなく、[Skills Hub](https://hermes-agent.nousresearch.com/docs)ページに表示されるカテゴリのラベルになります。これは汎用的です: このファイルを含むタップはどれも、Hermes側の変更なしで本物のカテゴリ分けが得られます。

```json
{
  "$schema": "https://skills.sh/schemas/skills.sh.schema.json",
  "groupings": [
    { "title": "Inference AI", "skills": ["dynamo-recipe-runner", "dynamo-router-sla"] },
    { "title": "Decision Optimization", "skills": ["cuopt-developer", "cuopt-install"] }
  ]
}
```

#### 5. ClawHub（`clawhub`）

コミュニティソースとして統合されたサードパーティのスキルマーケットプレイスです。

- サイト: [clawhub.ai](https://clawhub.ai/)
- HermesソースID: `clawhub`

#### 6. LobeHub（`lobehub`）

Hermesは、LobeHubの公開カタログからエージェントエントリを検索し、インストール可能なHermesスキルに変換できます。

- サイト: [LobeHub](https://lobehub.com/)
- 公開エージェントインデックス: [chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- バッキングリポジトリ: [lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- HermesソースID: `lobehub`

#### 7. browse.sh（`browse-sh`）

Hermesは、Browserbaseが提供する200以上のサイト固有のブラウザ自動化SKILL.mdファイル（Airbnb、Amazon、arXiv、12306.cn、Etsy、Xeroなど多数）のカタログである[browse.sh](https://browse.sh)と統合しています。各スキルは1つのWebサイトをエンドツーエンドで操作する方法を記述しており、Hermesのブラウザツールや、すでにインストールしているブラウザ自動化スキルと組み合わせて使うのに適しています。

- サイト: [browse.sh](https://browse.sh/)
- カタログAPI: `https://browse.sh/api/skills`
- HermesソースID: `browse-sh`
- 信頼レベル: `community`

```bash
hermes skills search airbnb --source browse-sh
hermes skills inspect browse-sh/airbnb.com/search-listings-ddgioa
hermes skills install browse-sh/airbnb.com/search-listings-ddgioa
```

識別子は `browse-sh/<hostname>/<task-id>` の形式を使い、browse.shのカタログが公開するスラッグと一致します。コンテンツは、カタログのGitHubの `sourceUrl` ではなく、スキルごとの詳細エンドポイント（`/api/skills/<slug>` → `skillMdUrl`）を通じて解決されます。

#### 8. 直接URL（`url`）

任意のHTTP(S) URLから `SKILL.md` を直接インストールします — 作者が自分のサイトにスキルをホストしている場合（hubのリスティングなし、入力するGitHubパスなし）に便利です。Hermesは `references/`、`templates/`、`scripts/`、`assets/`、`examples/` 配下で明示的に参照されているファイルも取得し、バンドル全体をスキャンしてインストールします。

- HermesソースID: `url`
- 識別子: URL自体（プレフィックス不要）
- 範囲: `SKILL.md` と、許可リストにあるディレクトリ内の正確に参照された補助ファイル。Hermesがホスト上の無関係なファイルを列挙したりコピーしたりすることはありません。

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
# Frontmatter has no name and the URL slug is unhelpful — supply one:
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# Or inside a chat session:
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
| `trusted` | `openai/skills`、`anthropics/skills`、`huggingface/skills`、`NVIDIA/skills`、`wandb/skills`、`qdrant/skills`、`pinecone-io/skills`、`makenotion/skills`、`Airtable/skills`、`Shopify/Shopify-AI-Toolkit` などの信頼されたレジストリ/リポジトリ | コミュニティソースよりも寛容なポリシー |
| `community` | それ以外すべて（`skills.sh`、well-knownエンドポイント、カスタムGitHubリポジトリ、ほとんどのマーケットプレイス） | 危険でない検出結果は `--force` で上書き可能。`dangerous` の判定はブロックされたままになる |

### 更新のライフサイクル

hubは、インストール済みスキルの上流のコピーを再チェックできるだけの来歴を追跡するようになりました:

```bash
hermes skills check          # Report which installed hub skills changed upstream
hermes skills update         # Reinstall only the skills with updates available
hermes skills update react   # Update one specific installed hub skill
hermes skills update react --force   # Overwrite a skill you've edited locally
```

これは、保存されたソース識別子と現在の上流バンドルのコンテンツハッシュを使って、差異を検出します。

チェックでは、欠けている、またはディレクトリでないインストール（`orphaned`）と、安全でない、または解決できない記録済みのパス（`invalid_install`）に対するネットワークリクエストはスキップされます。ディレクトリが欠けているエントリは `hermes skills uninstall <name>` で削除できます。無効なパスについては、再試行する前にアクティブなプロファイルの `skills/.hub/lock.json` を確認して修復する必要があります。エントリが自動的に削除されることはありません。

有効なインストールは、引き続きソースアダプターの既存の同期フェッチとトランスポートのタイムアウトを使用します。更新チェックには厳密な合計期限はありません: 既存のインストールのソースが到達不能または遅い場合、後続のエントリが遅れる可能性があります。

ローカルで編集したスキル（ディスク上の内容がインストール時に記録されたハッシュと一致しなくなったもの）は `hermes skills update` で**スキップ**されるため、変更が黙って上書きされることはありません。それでも上流のバージョンで置き換えるには `--force` を渡してください。

:::tip GitHubのレート制限
Skills hubの操作はGitHub APIを使用しており、これは未認証ユーザーに対して60リクエスト/時のレート制限があります。1回のインストールにかかるAPI呼び出しは約2回（リポジトリとそのファイルツリー）です。スキルのファイル自体は `raw.githubusercontent.com` からダウンロードされ、これは制限にカウントされません。インストールや検索の際にレート制限エラーが表示される場合は、`.env` ファイルに `GITHUB_TOKEN` を設定して制限を5,000リクエスト/時に引き上げてください。この場合、エラーメッセージには実行可能なヒントが含まれます。
:::

### カスタムスキルタップの公開 {#publishing-a-custom-skill-tap}

スキルのキュレーションされたセットを共有したい場合 — チーム、組織、または一般向けに — それらを**タップ**として公開できます: 他のHermesユーザーが `hermes skills tap add <owner/repo>` で追加するGitHubリポジトリです。サーバー不要、レジストリへのサインアップ不要、リリースパイプライン不要。`SKILL.md` ファイルのディレクトリがあるだけです。

#### リポジトリのレイアウト

タップは、次のようにレイアウトされた任意のGitHubリポジトリ（パブリックまたはプライベート — プライベートには `GITHUB_TOKEN` が必要）です:

```
owner/repo
├── skills/                       # default path; configurable per-tap
│   ├── my-workflow/
│   │   ├── SKILL.md              # required
│   │   ├── references/           # optional supporting files
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # optional but helpful
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

スキルが `skills/` 配下に存在しない場合（既存のプロジェクトに `skills/` サブツリーを追加するときによくあります）、`~/.hermes/skills/.hub/taps.json` のタップエントリを編集します:

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

新しいタップにはデフォルトで `community` の信頼が割り当てられます。そこからインストールされたスキルは標準のセキュリティスキャンを通過し、初回インストール時にサードパーティ警告パネルを表示します。あなたの組織や広く信頼されたソースがより高い信頼を得るべき場合は、`tools/skills_guard.py` の `TRUSTED_REPOS` にそのリポジトリを追加してください（Hermesコアへのプルリクエストが必要です）。

#### タップの管理

```bash
hermes skills tap list                                # show all configured taps
hermes skills tap add myorg/skills-repo               # add (default path: skills/)
hermes skills tap remove myorg/skills-repo            # remove
```

実行中のセッション内で:

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

タップは `~/.hermes/skills/.hub/taps.json` に保存されます（必要に応じて作成されます）。

## 同梱スキルの更新（`hermes skills reset`）

Hermesは、リポジトリ内の `skills/` に同梱スキルのセットを同梱しています。インストール時およびすべての `hermes update` 時に、同期パスがそれらを `~/.hermes/skills/` にコピーし、各スキル名を同期時のコンテンツハッシュ（**起点ハッシュ**）にマッピングするマニフェストを `~/.hermes/skills/.bundled_manifest` に記録します。

各同期で、Hermesはローカルコピーのハッシュを再計算し、起点ハッシュと比較します:

- **未変更** → 上流の変更を取り込んで安全。新しい同梱バージョンをコピーし、新しい起点ハッシュを記録する。
- **変更あり** → **ユーザー変更済み**として扱われ、以後永久にスキップされる。これによりあなたの編集が踏みつぶされることはない。

スキル内で生成されるランタイムキャッシュ（`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、および `.py` の隣にある `.pyc`）はハッシュに含まれないため、スキルのヘルパースクリプトを実行しても、それがユーザー変更済みとしてマークされたり、`hermes skills list-modified` / `diff` から隠れたりすることはありません。

この保護は優れていますが、1つの鋭い落とし穴があります。同梱スキルを編集した後で、変更を放棄して `~/.hermes/hermes-agent/skills/` からコピー&ペーストするだけで同梱バージョンに戻したい場合、マニフェストには最後に同期が成功したときの*古い*起点ハッシュが依然として保持されています。あなたの新しくコピー&ペーストした内容（現在の同梱ハッシュ）はその古い起点ハッシュと一致しないため、同期はそれをユーザー変更済みとしてフラグ付けし続けます。

`hermes skills reset` がその抜け道です:

```bash
# Safe: clears the manifest entry for this skill. Your current copy is preserved,
# but the next sync re-baselines against it so future updates work normally.
hermes skills reset google-workspace

# Full restore: also deletes your local copy and re-copies the current bundled
# version. Use this when you want the pristine upstream skill back.
hermes skills reset google-workspace --restore

# Non-interactive (e.g. in scripts or TUI mode) — skip the --restore confirmation.
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

公式オプションスキルは、依然として `official/security/1password` や `official/mcp/fastmcp` のような識別子を使用します。
