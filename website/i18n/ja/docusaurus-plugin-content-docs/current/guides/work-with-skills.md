---
sidebar_position: 12
title: "スキルを使う"
description: "スキルを見つけ、インストールし、使い、作成する — Hermesに新しいワークフローを教えるオンデマンドの知識"
---

# スキルを使う

スキルは、ASCIIアートの生成からGitHub PRの管理まで、特定のタスクをHermesがどう処理するかを教えるオンデマンドの知識ドキュメントです。このガイドでは、日々の使い方を一通り説明します。

完全な技術リファレンスについては、[スキルシステム](/docs/user-guide/features/skills)を参照してください。

---

## スキルを見つける

すべてのHermesインストールには、同梱のスキルが付属しています。利用可能なものを確認しましょう。

```bash
# 任意のチャットセッション内で:
/skills

# またはCLIから:
hermes skills list
```

これにより、名前と説明を含むコンパクトなリストが表示されます。

```
ascii-art         Generate ASCII art using pyfiglet, cowsay, boxes...
arxiv             Search and retrieve academic papers from arXiv...
github-pr-workflow Full PR lifecycle — create branches, commit...
plan              Plan mode — inspect context, write a markdown...
excalidraw        Create hand-drawn style diagrams using Excalidraw...
```

### スキルを検索する

```bash
# キーワードで検索
/skills search docker
/skills search music
```

### スキルハブ

公式のオプションスキル（デフォルトでは有効化されていない、より重い、またはニッチなスキル）は、ハブ経由で利用できます。

```bash
# 公式のオプションスキルを閲覧する
/skills browse

# ハブを検索する
/skills search blockchain
```

---

## スキルを使う

インストールされたすべてのスキルは、自動的にスラッシュコマンドになります。その名前を入力するだけです。

```bash
# スキルを読み込んでタスクを与える
/ascii-art Make a banner that says "HELLO WORLD"
/plan Design a REST API for a todo app
/github-pr-workflow Create a PR for the auth refactor

# スキル名だけ（タスクなし）を入力すると、読み込んだうえで必要なことを記述できる
/excalidraw
```

自然な会話を通じてスキルをトリガーすることもできます。特定のスキルを使うようHermesに依頼すると、`skill_view` ツール経由でそれを読み込みます。

### 漸進的開示（progressive disclosure）

スキルは、トークン効率の良い読み込みパターンを使用します。エージェントはすべてを一度に読み込むわけではありません。

1. **`skills_list()`** — すべてのスキルのコンパクトなリスト（約3kトークン）。セッション開始時に読み込まれます。
2. **`skill_view(name)`** — 1つのスキルの完全なSKILL.mdコンテンツ。エージェントがそのスキルが必要だと判断したときに読み込まれます。
3. **`skill_view(name, file_path)`** — スキル内の特定の参照ファイル。必要な場合にのみ読み込まれます。

これは、スキルが実際に使われるまでトークンを消費しないことを意味します。

---

## ハブからインストールする

公式のオプションスキルはHermesに同梱されていますが、デフォルトでは有効化されていません。明示的にインストールしてください。

```bash
# 公式のオプションスキルをインストールする
hermes skills install official/research/arxiv

# チャットセッション内でハブからインストールする
/skills install official/creative/songwriting-and-ai-music

# 任意のHTTP(S) URLから単一ファイルのSKILL.mdを直接インストールする
hermes skills install https://sharethis.chat/SKILL.md
/skills install https://example.com/SKILL.md --name my-skill
```

何が起こるか:
1. スキルのディレクトリが `~/.hermes/skills/` にコピーされます
2. `skills_list` の出力に表示されます
3. スラッシュコマンドとして利用可能になります

:::tip
インストールされたスキルは新しいセッションで有効になります。現在のセッションで利用可能にしたい場合は、`/reset` を使って新しく始めるか、`--now` を追加してプロンプトキャッシュを即座に無効化してください（次のターンでより多くのトークンを消費します）。
:::

### インストールの確認

```bash
# そこにあることを確認する
hermes skills list | grep arxiv

# またはチャット内で
/skills search arxiv
```

---

## プラグイン提供のスキル

プラグインは、名前空間付きの名前（`plugin:skill`）を使って独自のスキルを同梱できます。これにより、組み込みスキルとの名前の衝突を防ぎます。

```bash
# 完全修飾名でプラグインスキルを読み込む
skill_view("superpowers:writing-plans")

# 同じベース名の組み込みスキルは影響を受けない
skill_view("writing-plans")
```

プラグインスキルはシステムプロンプトに**列挙されず**、`skills_list` にも表示されません。これらはオプトインで、プラグインが提供していると分かっているときに明示的に読み込みます。読み込まれると、エージェントには同じプラグインの兄弟スキルを列挙したバナーが表示されます。

独自のプラグインでスキルを同梱する方法については、[Hermesプラグインを構築する → スキルを同梱する](/docs/guides/build-a-hermes-plugin#bundle-skills)を参照してください。

---

## スキル設定を構成する

一部のスキルは、frontmatterで必要とする設定を宣言します。

```yaml
metadata:
  hermes:
    config:
      - key: tenor.api_key
        description: "Tenor API key for GIF search"
        prompt: "Enter your Tenor API key"
        url: "https://developers.google.com/tenor/guides/quickstart"
```

設定を持つスキルが初めて読み込まれると、Hermesは値の入力を求めます。それらは `config.yaml` の `skills.config.*` の下に保存されます。

CLIからスキル設定を管理します。

```bash
# 特定のスキルの対話的な設定
hermes skills config gif-search

# すべてのスキル設定を表示する
hermes config get skills.config
```

---

## 独自のスキルを作成する

スキルは、YAML frontmatterを持つ単なるmarkdownファイルです。作成には5分もかかりません。

### 1. ディレクトリを作成する

```bash
mkdir -p ~/.hermes/skills/my-category/my-skill
```

### 2. SKILL.mdを書く

```markdown title="~/.hermes/skills/my-category/my-skill/SKILL.md"
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
metadata:
  hermes:
    tags: [my-tag, automation]
    category: my-category
---

# My Skill

## When to Use
Use this skill when the user asks about [specific topic] or needs to [specific task].

## Procedure
1. First, check if [prerequisite] is available
2. Run `command --with-flags`
3. Parse the output and present results

## Pitfalls
- Common failure: [description]. Fix: [solution]
- Watch out for [edge case]

## Verification
Run `check-command` to confirm the result is correct.
```

### 3. 参照ファイルを追加する（任意）

スキルには、エージェントがオンデマンドで読み込むサポートファイルを含めることができます。

```
my-skill/
├── SKILL.md                    # メインのスキルドキュメント
├── references/
│   ├── api-docs.md             # エージェントが参照できるAPIリファレンス
│   └── examples.md             # 入出力の例
├── templates/
│   └── config.yaml             # エージェントが使えるテンプレートファイル
└── scripts/
    └── setup.sh                # エージェントが実行できるスクリプト
```

これらをSKILL.md内で参照します。

```markdown
For API details, load the reference: `skill_view("my-skill", "references/api-docs.md")`
```

### 4. テストする

新しいセッションを開始し、スキルを試します。

```bash
hermes chat -q "/my-skill help me with the thing"
```

スキルは自動的に表示されます。登録は不要です。`~/.hermes/skills/` に配置するだけで有効になります。

:::info
エージェントは `skill_manage` を使って、自分自身でスキルを作成・更新することもできます。複雑な問題を解決した後、Hermesは次回のためにそのアプローチをスキルとして保存することを提案する場合があります。
:::

---

## プラットフォーム単位のスキル管理

どのスキルをどのプラットフォームで利用可能にするかを制御します。

```bash
hermes skills
```

これにより、プラットフォーム（CLI、Telegram、Discordなど）ごとにスキルを有効化または無効化できる対話的なTUIが開きます。特定のコンテキストでのみスキルを利用可能にしたい場合に便利です。たとえば、開発用スキルをTelegramでは利用できないようにする、といった使い方です。

---

## スキルとメモリの比較

どちらもセッションをまたいで永続化されますが、異なる目的に役立ちます。

| | スキル | メモリ |
|---|---|---|
| **何を** | 手続き的知識 — 物事のやり方 | 事実的知識 — 物事が何であるか |
| **いつ** | オンデマンドで、関連する場合にのみ読み込まれる | すべてのセッションに自動的に注入される |
| **サイズ** | 大きくなりうる（数百行） | コンパクトであるべき（重要な事実のみ） |
| **コスト** | 読み込まれるまでトークンゼロ | 小さいが一定のトークンコスト |
| **例** | 「Kubernetesへのデプロイ方法」 | 「ユーザーはダークモードを好み、PSTに住んでいる」 |
| **誰が作成するか** | あなた、エージェント、またはハブからインストール | 会話に基づいてエージェントが作成 |

**経験則:** 参照ドキュメントに書くようなものならスキルです。付箋に書くようなものならメモリです。

---

## ヒント

**スキルは焦点を絞る。** 「DevOpsのすべて」をカバーしようとするスキルは、長すぎて漠然としすぎます。「PythonアプリをFly.ioにデプロイする」をカバーするスキルは、本当に役立つほど具体的です。

**エージェントにスキルを作成させる。** 複雑な複数ステップのタスクの後、Hermesはしばしばそのアプローチをスキルとして保存することを提案します。「はい」と答えてください。これらのエージェントが作成したスキルは、途中で発見された落とし穴を含む正確なワークフローを捉えています。

**カテゴリを使う。** スキルをサブディレクトリ（`~/.hermes/skills/devops/`、`~/.hermes/skills/research/` など）に整理してください。これによりリストが管理しやすくなり、エージェントが関連するスキルをより速く見つけられるようになります。

**スキルが古くなったら更新する。** スキルを使って、それでカバーされていない問題に遭遇した場合は、学んだ内容でスキルを更新するようHermesに伝えてください。メンテナンスされないスキルは負債になります。

---

*完全なスキルリファレンス（frontmatterフィールド、条件付き有効化、外部ディレクトリなど）については、[スキルシステム](/docs/user-guide/features/skills)を参照してください。*
