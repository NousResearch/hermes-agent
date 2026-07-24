---
sidebar_position: 9
title: "パーソナリティとSOUL.md"
description: "グローバルなSOUL.md、組み込みパーソナリティ、カスタムペルソナ定義でHermes Agentのパーソナリティをカスタマイズする"
---

# パーソナリティとSOUL.md

Hermes Agentのパーソナリティは完全にカスタマイズ可能です。`SOUL.md` は**主たるアイデンティティ**であり、システムプロンプトの先頭に置かれ、エージェントが何者であるかを定義します。

- `SOUL.md` — `HERMES_HOME` に置かれ、エージェントのアイデンティティとして機能する永続的なペルソナファイル（システムプロンプトのスロット#1）
- 組み込みまたはカスタムの `/personality` プリセット — セッションレベルのシステムプロンプトオーバーレイ

Hermesが何者であるかを変えたい場合、あるいはまったく別のエージェントペルソナに置き換えたい場合は、`SOUL.md` を編集してください。

## SOUL.mdの現在の動作

Hermesは現在、デフォルトの `SOUL.md` を以下に自動でシードします。

```text
~/.hermes/SOUL.md
```

正確には、現在のインスタンスの `HERMES_HOME` を使用するため、カスタムのホームディレクトリでHermesを実行している場合は以下を使用します。

```text
$HERMES_HOME/SOUL.md
```

### 重要な動作

- **SOUL.mdはエージェントの主たるアイデンティティです。** システムプロンプトのスロット#1を占め、ハードコードされたデフォルトのアイデンティティを置き換えます。
- まだ存在しない場合、Hermesは自動的にスターター用の `SOUL.md` を作成します
- 既存のユーザーの `SOUL.md` ファイルが上書きされることはありません
- Hermesは `HERMES_HOME` からのみ `SOUL.md` を読み込みます
- Hermesは現在の作業ディレクトリで `SOUL.md` を探しません
- `SOUL.md` が存在しても空の場合、または読み込めない場合、Hermesは組み込みのデフォルトのアイデンティティにフォールバックします
- `SOUL.md` に内容がある場合、その内容はセキュリティスキャンと切り詰めを経て、そのまま注入されます
- SOUL.mdはコンテキストファイルのセクションで重複して**現れることはなく**、アイデンティティとして一度だけ現れます

これにより、`SOUL.md` は単なる追加レイヤーではなく、真のユーザーごと・インスタンスごとのアイデンティティになります。

## この設計の理由

これによってパーソナリティが予測可能になります。

もしHermesが起動したディレクトリから `SOUL.md` を読み込んでいたら、プロジェクト間でパーソナリティが予期せず変わってしまう可能性があります。`HERMES_HOME` からのみ読み込むことで、パーソナリティはHermesインスタンス自体に属します。

これはユーザーに教える際にも分かりやすくなります。
- 「Hermesのデフォルトのパーソナリティを変えるには `~/.hermes/SOUL.md` を編集してください。」

## 編集する場所

ほとんどのユーザーの場合:

```bash
~/.hermes/SOUL.md
```

カスタムのホームを使用している場合:

```bash
$HERMES_HOME/SOUL.md
```

## SOUL.mdには何を書くべきか？

以下のような、永続的な語り口とパーソナリティのガイダンスに使用します。
- トーン
- コミュニケーションスタイル
- 率直さの度合い
- デフォルトのやり取りのスタイル
- スタイル上で避けるべきこと
- Hermesが不確実性・意見の相違・曖昧さをどう扱うべきか

以下にはあまり使用しません。
- 一回限りのプロジェクト指示
- ファイルパス
- リポジトリの規約
- 一時的なワークフローの詳細

これらは `SOUL.md` ではなく `AGENTS.md` に属します。

## 良いSOUL.mdの内容

良いSOULファイルは次のようなものです。
- コンテキストをまたいで安定している
- 多くの会話に適用できるほど幅広い
- 語り口を実質的に形作るほど具体的
- タスク固有の指示ではなく、コミュニケーションとアイデンティティに焦点を当てている

### 例

```markdown
# Personality

You are a pragmatic senior engineer with strong taste.
You optimize for truth, clarity, and usefulness over politeness theater.

## Style
- Be direct without being cold
- Prefer substance over filler
- Push back when something is a bad idea
- Admit uncertainty plainly
- Keep explanations compact unless depth is useful

## What to avoid
- Sycophancy
- Hype language
- Repeating the user's framing if it's wrong
- Overexplaining obvious things

## Technical posture
- Prefer simple systems over clever systems
- Care about operational reality, not idealized architecture
- Treat edge cases as part of the design, not cleanup
```

## Hermesがプロンプトに注入するもの

`SOUL.md` の内容はシステムプロンプトのスロット#1、つまりエージェントのアイデンティティの位置に直接入ります。その周囲にラッパー的な文言は追加されません。

内容は以下を経由します。
- プロンプトインジェクションのスキャン
- 大きすぎる場合の切り詰め

ファイルが空、空白のみ、または読み込めない場合、Hermesは組み込みのデフォルトのアイデンティティ（"You are Hermes Agent, an intelligent AI assistant created by Nous Research..."）にフォールバックします。このフォールバックは `skip_context_files` が設定されている場合（例: サブエージェント/委譲のコンテキスト）にも適用されます。

## セキュリティスキャン

`SOUL.md` は、他のコンテキストを持つファイルと同様に、組み込み前にプロンプトインジェクションのパターンがないかスキャンされます。

つまり、奇妙なメタ指示をこっそり忍び込ませようとするのではなく、ペルソナ・語り口に焦点を当て続けるべきだということです。

## SOUL.md と AGENTS.md

これが最も重要な区別です。

### SOUL.md
以下に使用します。
- アイデンティティ
- トーン
- スタイル
- コミュニケーションのデフォルト
- パーソナリティレベルの振る舞い

### AGENTS.md
以下に使用します。
- プロジェクトのアーキテクチャ
- コーディング規約
- ツールの好み
- リポジトリ固有のワークフロー
- コマンド、ポート、パス、デプロイのメモ

役立つルール:
- どこへでもあなたについて回るべきものは `SOUL.md` に属します
- 特定のプロジェクトに属するものは `AGENTS.md` に属します

## SOUL.md と `/personality`

`SOUL.md` はあなたの永続的なデフォルトのパーソナリティです。

`/personality` は、現在のシステムプロンプトを変更または補完するセッションレベルのオーバーレイです。

つまり:
- `SOUL.md` = ベースラインの語り口
- `/personality` = 一時的なモード切り替え

例:
- 実用的なデフォルトのSOULを保ちつつ、チュータリングの会話には `/personality teacher` を使う
- 簡潔なSOULを保ちつつ、ブレインストーミングには `/personality creative` を使う

## 組み込みパーソナリティ

Hermesには `/personality` で切り替えられる組み込みパーソナリティが付属しています。

| 名前 | 説明 |
|------|-------------|
| **helpful** | フレンドリーで汎用的なアシスタント |
| **concise** | 簡潔で要点を押さえた応答 |
| **technical** | 詳細で正確な技術エキスパート |
| **creative** | 革新的で型にはまらない発想 |
| **teacher** | 明確な例を示す忍耐強い教育者 |
| **kawaii** | かわいい表現、きらめき、熱意 ★ |
| **catgirl** | 猫っぽい表現のねこちゃん、にゃ〜 |
| **pirate** | テクノロジーに精通した海賊、キャプテンHermes |
| **shakespeare** | ドラマチックな趣のある吟遊詩人風の散文 |
| **surfer** | とことくつろいだブラザーな雰囲気 |
| **noir** | ハードボイルドな探偵のナレーション |
| **uwu** | uwu語による最大級のかわいさ |
| **philosopher** | あらゆる問いに対する深い思索 |
| **hype** | 最大級のエネルギーと熱意!!! |

## コマンドでパーソナリティを切り替える

### CLI

```text
/personality
/personality concise
/personality technical
```

### メッセージングプラットフォーム

```text
/personality teacher
```

これらは便利なオーバーレイですが、オーバーレイが意味のある形で変更しない限り、グローバルな `SOUL.md` が引き続きHermesに永続的なデフォルトのパーソナリティを与えます。

## 設定でのカスタムパーソナリティ

`~/.hermes/config.yaml` の `agent.personalities` 配下で、名前付きのカスタムパーソナリティを定義することもできます。

```yaml
agent:
  personalities:
    codereviewer: >
      You are a meticulous code reviewer. Identify bugs, security issues,
      performance concerns, and unclear design choices. Be precise and constructive.
```

その後、以下で切り替えます。

```text
/personality codereviewer
```

## 推奨ワークフロー

強力なデフォルト構成は次のとおりです。

1. `~/.hermes/SOUL.md` に練り込んだグローバルな `SOUL.md` を保つ
2. プロジェクトの指示は `AGENTS.md` に入れる
3. 一時的にモードを切り替えたいときだけ `/personality` を使う

これにより以下が得られます。
- 安定した語り口
- 然るべき場所にあるプロジェクト固有の振る舞い
- 必要なときの一時的なコントロール

## パーソナリティはプロンプト全体とどう相互作用するか

大まかには、プロンプトスタックには以下が含まれます。
1. **SOUL.md**（エージェントのアイデンティティ — SOUL.mdが利用できない場合は組み込みのフォールバック）
2. ツールを意識した振る舞いのガイダンス
3. メモリ/ユーザーコンテキスト
4. スキルのガイダンス
5. コンテキストファイル（`AGENTS.md`、`.cursorrules`）
6. タイムスタンプ
7. プラットフォーム固有のフォーマットのヒント
8. `/personality` などのオプションのシステムプロンプトオーバーレイ

`SOUL.md` は土台であり、他のすべてはその上に積み重なります。

## 関連ドキュメント

- [コンテキストファイル](/docs/user-guide/features/context-files)
- [設定](/docs/user-guide/configuration)
- [ヒントとベストプラクティス](/docs/guides/tips)
- [SOUL.mdガイド](/docs/guides/use-soul-with-hermes)

## CLIの外観と会話のパーソナリティ

会話のパーソナリティとCLIの外観は別物です。

- `SOUL.md`、`agent.system_prompt`、`/personality` はHermesの話し方に影響します
- `display.skin` と `/skin` はターミナルでのHermesの見た目に影響します

ターミナルの外観については、[スキンとテーマ](./skins.md)を参照してください。
