---
sidebar_position: 7
title: "HermesでSOUL.mdを使う"
description: "SOUL.mdを使ってHermes Agentのデフォルトの語り口を形作る方法、何をそこに書くべきか、AGENTS.mdや/personalityとどう違うか"
---

# HermesでSOUL.mdを使う

`SOUL.md` は、あなたのHermesインスタンスの**主たるアイデンティティ**です。これはシステムプロンプトの先頭に置かれ — エージェントが何者で、どう話し、何を避けるかを定義します。

Hermesに、話しかけるたびに同じアシスタントだと感じさせたい場合 — あるいはHermesのペルソナを完全に自分のものに置き換えたい場合 — 使うべきはこのファイルです。

## SOUL.mdは何のためか

`SOUL.md` は以下に使用します。
- トーン
- パーソナリティ
- コミュニケーションスタイル
- Hermesがどれくらい率直、または温かくあるべきか
- Hermesがスタイル上で避けるべきこと
- Hermesが不確実性、意見の相違、曖昧さにどう向き合うべきか

要するに:
- `SOUL.md` はHermesが何者で、どう話すかについてのものです

## SOUL.mdは何のためではないか

以下には使用しないでください。
- リポジトリ固有のコーディング規約
- ファイルパス
- コマンド
- サービスのポート
- アーキテクチャのメモ
- プロジェクトのワークフロー指示

これらは `AGENTS.md` に属します。

良いルール:
- どこでも適用すべきものは `SOUL.md` に入れる
- 1つのプロジェクトにだけ属するものは `AGENTS.md` に入れる

## どこに置かれるか

Hermesは現在、現在のインスタンスのグローバルなSOULファイルのみを使用します。

```text
~/.hermes/SOUL.md
```

カスタムのホームディレクトリでHermesを実行している場合は、次のようになります。

```text
$HERMES_HOME/SOUL.md
```

## 初回実行時の動作

Hermesは、まだ存在しない場合、スターター用の `SOUL.md` を自動でシードします。

つまり、ほとんどのユーザーは今、すぐに読んで編集できる実際のファイルから始められます。

重要:
- すでに `SOUL.md` を持っている場合、Hermesはそれを上書きしません
- ファイルが存在しても空の場合、Hermesはそこから何もプロンプトに追加しません

## Hermesがそれをどう使うか

Hermesはセッションを開始するとき、`HERMES_HOME` から `SOUL.md` を読み込み、プロンプトインジェクションのパターンがないかスキャンし、必要なら切り詰め、それを**エージェントのアイデンティティ**、つまりシステムプロンプトのスロット#1として使用します。これは、SOUL.mdが組み込みのデフォルトのアイデンティティテキストを完全に置き換えることを意味します。

SOUL.mdが存在しない、空、または読み込めない場合、Hermesは組み込みのデフォルトのアイデンティティにフォールバックします。

ファイルの周囲にラッパー的な文言は追加されません。内容そのものが重要です — あなたのエージェントに考え、話してほしいとおりに書きましょう。

## 良い最初の編集

他に何もしないとしても、ファイルを開いて数行だけ変え、あなたらしく感じられるようにしましょう。

例えば:

```markdown
You are direct, calm, and technically precise.
Prefer substance over politeness theater.
Push back clearly when an idea is weak.
Keep answers compact unless deeper detail is useful.
```

これだけでも、Hermesの感じ方をはっきりと変えられます。

## スタイルの例

### 1. 実用的なエンジニア

```markdown
You are a pragmatic senior engineer.
You care more about correctness and operational reality than sounding impressive.

## Style
- Be direct
- Be concise unless complexity requires depth
- Say when something is a bad idea
- Prefer practical tradeoffs over idealized abstractions

## Avoid
- Sycophancy
- Hype language
- Overexplaining obvious things
```

### 2. リサーチパートナー

```markdown
You are a thoughtful research collaborator.
You are curious, honest about uncertainty, and excited by unusual ideas.

## Style
- Explore possibilities without pretending certainty
- Distinguish speculation from evidence
- Ask clarifying questions when the idea space is underspecified
- Prefer conceptual depth over shallow completeness
```

### 3. 教師 / 解説者

```markdown
You are a patient technical teacher.
You care about understanding, not performance.

## Style
- Explain clearly
- Use examples when they help
- Do not assume prior knowledge unless the user signals it
- Build from intuition to details
```

### 4. 厳しいレビュアー

```markdown
You are a rigorous reviewer.
You are fair, but you do not soften important criticism.

## Style
- Point out weak assumptions directly
- Prioritize correctness over harmony
- Be explicit about risks and tradeoffs
- Prefer blunt clarity to vague diplomacy
```

## 強いSOUL.mdとは？

強い `SOUL.md` は次のようなものです。
- 安定している
- 広く適用できる
- 語り口が具体的
- 一時的な指示で過負荷になっていない

弱い `SOUL.md` は次のようなものです。
- プロジェクトの詳細でいっぱい
- 矛盾している
- すべての応答の形を細かく管理しようとしている
- ほとんどが「役に立て」「明確であれ」のような一般的な埋め草

Hermesはすでに役に立ち、明確であろうとします。`SOUL.md` は、当たり前のデフォルトを言い直すのではなく、本物のパーソナリティとスタイルを加えるべきです。

## 推奨される構造

見出しは必須ではありませんが、役に立ちます。

うまく機能するシンプルな構造:

```markdown
# Identity
Who Hermes is.

# Style
How Hermes should sound.

# Avoid
What Hermes should not do.

# Defaults
How Hermes should behave when ambiguity appears.
```

## SOUL.md と /personality

これらは補完的です。

`SOUL.md` は永続的なベースラインに使います。
`/personality` は一時的なモード切り替えに使います。

例:
- あなたのデフォルトのSOULは実用的で率直
- そして、あるセッションだけ `/personality teacher` を使う
- その後、ベースの語り口のファイルを変えずに元に戻す

## SOUL.md と AGENTS.md

これは最もよくある間違いです。

### これをSOUL.mdに入れる
- 「率直であれ。」
- 「誇張表現を避けよ。」
- 「深さが役立つ場合を除き、短い回答を好め。」
- 「ユーザーが間違っているときは反論せよ。」

### これをAGENTS.mdに入れる
- 「unittestではなくpytestを使え。」
- 「フロントエンドは `frontend/` にある。」
- 「マイグレーションを直接編集するな。」
- 「APIはポート8000で動く。」

## 編集の仕方

```bash
nano ~/.hermes/SOUL.md
```

または

```bash
vim ~/.hermes/SOUL.md
```

その後、Hermesを再起動するか、新しいセッションを開始します。

## 実践的なワークフロー

1. シードされたデフォルトのファイルから始める
2. 望む語り口だと感じられないものを削る
3. トーンとデフォルトを明確に定義する4〜8行を加える
4. しばらくHermesと話す
5. まだしっくりこない点に基づいて調整する

この反復的なアプローチは、完璧なパーソナリティを一発で設計しようとするよりうまくいきます。

## トラブルシューティング

### SOUL.mdを編集したのにHermesの話し方が変わらない

確認すること:
- `~/.hermes/SOUL.md` または `$HERMES_HOME/SOUL.md` を編集したか
- リポジトリローカルの `SOUL.md` ではないか
- ファイルが空ではないか
- 編集後にセッションを再起動したか
- `/personality` オーバーレイが結果を支配していないか

### HermesがSOUL.mdの一部を無視する

考えられる原因:
- より優先度の高い指示がそれを上書きしている
- ファイルに矛盾するガイダンスが含まれている
- ファイルが長すぎて切り詰められた
- 一部のテキストがプロンプトインジェクションの内容に似ており、スキャナーによってブロックまたは変更された可能性がある

### SOUL.mdがプロジェクト固有になりすぎた

プロジェクトの指示を `AGENTS.md` に移し、`SOUL.md` はアイデンティティとスタイルに焦点を当て続けましょう。

## 関連ドキュメント

- [パーソナリティとSOUL.md](/docs/user-guide/features/personality)
- [コンテキストファイル](/docs/user-guide/features/context-files)
- [設定](/docs/user-guide/configuration)
- [ヒントとベストプラクティス](/docs/guides/tips)
