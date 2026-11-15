---
sidebar_position: 3
title: '学習パス'
description: 'あなたの経験レベルと目標に合わせて、Hermes Agent ドキュメントの学習パスを選びましょう。'
---

# 学習パス

Hermes Agent は多くのことができます。CLI アシスタント、Telegram/Discord ボット、タスク自動化、RL トレーニングなどです。このページでは、あなたの経験レベルと達成したいことに応じて、どこから始め、何を読めばよいかを見つける手助けをします。

:::tip Start Here
まだ Hermes Agent をインストールしていない場合は、[インストールガイド](./installation.md) から始め、続いて [クイックスタート](./quickstart.md) を実行してください。以下の内容はすべて、動作するインストール環境があることを前提としています。
:::

:::tip 初めてのプロバイダー設定
初めてのユーザーには、ほぼ必ず `hermes setup --portal` がおすすめです — 1 回の OAuth で、モデルと 4 つの Tool Gateway ツール（検索/画像/TTS/ブラウザ）をまとめて利用できるようになります。[Nous Portal](/docs/integrations/nous-portal) を参照してください。
:::

## このページの使い方

- **自分のレベルがわかっている場合は？** [経験レベル別の表](#by-experience-level) に進み、あなたの段階に応じた読む順序に従ってください。
- **具体的な目標がある場合は？** [ユースケース別](#by-use-case) に進み、当てはまるシナリオを見つけてください。
- **ざっと眺めたいだけの場合は？** [主な機能](#key-features-at-a-glance) の表で、Hermes Agent ができることの概要を素早く確認してください。

## 経験レベル別 {#by-experience-level}

| レベル | 目標 | 推奨される読む順序 | 所要時間の目安 |
|---|---|---|---|
| **初級** | 環境を整えて動かし、基本的な会話を行い、組み込みのツールを使う | [インストール](./installation.md) → [クイックスタート](./quickstart.md) → [CLI の使い方](../user-guide/cli.md) → [設定](../user-guide/configuration.md) | 約1時間 |
| **中級** | メッセージングボットをセットアップし、メモリ、cron ジョブ、スキルなどの高度な機能を使う | [セッション](../user-guide/sessions.md) → [メッセージング](../user-guide/messaging/index.md) → [ツール](../user-guide/features/tools.md) → [スキル](../user-guide/features/skills.md) → [メモリ](../user-guide/features/memory.md) → [Cron](../user-guide/features/cron.md) | 約2〜3時間 |
| **上級** | カスタムツールを構築し、スキルを作成し、RL でモデルをトレーニングし、プロジェクトに貢献する | [アーキテクチャ](/docs/developer-guide/architecture) → [ツールの追加](/docs/developer-guide/adding-tools) → [スキルの作成](/docs/developer-guide/creating-skills) → [コントリビュート](/docs/developer-guide/contributing) | 約4〜6時間 |

## ユースケース別 {#by-use-case}

やりたいことに当てはまるシナリオを選んでください。それぞれ、読むべき順序で関連ドキュメントへのリンクを示しています。

### 「CLI のコーディングアシスタントが欲しい」

Hermes Agent を、コードの記述、レビュー、実行を行うインタラクティブなターミナルアシスタントとして使います。

1. [インストール](./installation.md)
2. [クイックスタート](./quickstart.md)
3. [CLI の使い方](../user-guide/cli.md)
4. [コード実行](../user-guide/features/code-execution.md)
5. [コンテキストファイル](../user-guide/features/context-files.md)
6. [ヒントとコツ](../guides/tips.md)

:::tip
コンテキストファイルを使って、ファイルを直接会話に渡せます。Hermes Agent はあなたのプロジェクト内のコードを読み取り、編集し、実行できます。
:::

### 「Telegram/Discord ボットが欲しい」

Hermes Agent を、お気に入りのメッセージングプラットフォーム上のボットとしてデプロイします。

1. [インストール](./installation.md)
2. [設定](../user-guide/configuration.md)
3. [メッセージングの概要](../user-guide/messaging/index.md)
4. [Telegram のセットアップ](../user-guide/messaging/telegram.md)
5. [Discord のセットアップ](../user-guide/messaging/discord.md)
6. [ボイスモード](../user-guide/features/voice-mode.md)
7. [Hermes でボイスモードを使う](../guides/use-voice-mode-with-hermes.md)
8. [セキュリティ](../user-guide/security.md)

完全なプロジェクトの例については、以下を参照してください。
- [デイリーブリーフィングボット](../guides/daily-briefing-bot.md)
- [チーム向け Telegram アシスタント](../guides/team-telegram-assistant.md)

### 「タスクを自動化したい」

定期的なタスクをスケジュールしたり、バッチジョブを実行したり、エージェントのアクションを連鎖させたりします。

1. [クイックスタート](./quickstart.md)
2. [Cron スケジューリング](../user-guide/features/cron.md)
3. [バッチ処理](../user-guide/features/batch-processing.md)
4. [委譲](../user-guide/features/delegation.md)
5. [フック](../user-guide/features/hooks.md)

:::tip
Cron ジョブを使うと、あなたがその場にいなくても、Hermes Agent がスケジュールに従ってタスク（毎日のサマリー、定期的なチェック、自動レポートなど）を実行できます。
:::

### 「専門ボットのチームが欲しい」

独自のモデル、メモリ、スキル、ルーティン、チャットを持つ名前付きのボットを作成し、グループチャットや `@mentions` で連携させます。

1. [デスクトップ](/docs/user-guide/desktop)
2. [プロファイル](../user-guide/profiles.md)
3. [ボットモード](/docs/user-guide/bot-mode)
4. [Cron スケジューリング](../user-guide/features/cron.md)
5. [マルチ接続デスクトップ](/docs/user-guide/multi-connection-desktop)

### 「カスタムツール/スキルを構築したい」

あなた自身のツールや再利用可能なスキルパッケージで Hermes Agent を拡張します。

1. [プラグイン](../user-guide/features/plugins.md)
2. [Hermes プラグインを構築する](../developer-guide/plugins/index.md)
3. [ツールの概要](../user-guide/features/tools.md)
4. [スキルの概要](../user-guide/features/skills.md)
5. [MCP（Model Context Protocol）](../user-guide/features/mcp.md)
6. [アーキテクチャ](/docs/developer-guide/architecture)
7. [ツールの追加](/docs/developer-guide/adding-tools)
8. [スキルの作成](/docs/developer-guide/creating-skills)

:::tip
ほとんどのカスタムツール作成では、プラグインから始めてください。[ツールの追加](/docs/developer-guide/adding-tools)
のページは Hermes コア本体の開発向けであり、通常のユーザー/カスタムツールのパスではありません。
:::

### 「モデルをトレーニングしたい」

Hermes Agent の RL トレーニングパイプライン（[Atropos](https://github.com/NousResearch/atropos) を利用）を使い、強化学習でモデルの振る舞いをファインチューニングします。

1. [クイックスタート](./quickstart.md)
2. [設定](../user-guide/configuration.md)
3. [Atropos RL 環境](https://github.com/NousResearch/atropos)（外部）
4. [プロバイダールーティング](../user-guide/features/provider-routing.md)
5. [アーキテクチャ](/docs/developer-guide/architecture)

:::tip
RL トレーニングは、Hermes Agent が会話やツール呼び出しをどのように扱うかという基本をすでに理解している場合に最も効果的です。初めての方は、まず初級パスを進めてください。
:::

### 「Python ライブラリとして使いたい」

Hermes Agent をプログラムから、あなた自身の Python アプリケーションに統合します。

1. [インストール](./installation.md)
2. [クイックスタート](./quickstart.md)
3. [Python ライブラリガイド](../guides/python-library.md)
4. [アーキテクチャ](/docs/developer-guide/architecture)
5. [ツール](../user-guide/features/tools.md)
6. [セッション](../user-guide/sessions.md)

## 主な機能の早見表 {#key-features-at-a-glance}

何が使えるのかわからない場合は？ こちらが主要な機能の簡単な一覧です。

| 機能 | 何ができるか | リンク |
|---|---|---|
| **ツール** | エージェントが呼び出せる組み込みツール（ファイル I/O、検索、シェルなど） | [ツール](../user-guide/features/tools.md) |
| **スキル** | 新しい機能を追加するインストール可能なプラグインパッケージ | [スキル](../user-guide/features/skills.md) |
| **メモリ** | セッションをまたいで永続化されるメモリ | [メモリ](../user-guide/features/memory.md) |
| **ボットモード** | 永続的なチャット、ルーティン、グループチャット、`@mentions` を備えた名前付きの専門ボット | [ボットモード](/docs/user-guide/bot-mode) |
| **コンテキストファイル** | ファイルやディレクトリを会話に取り込む | [コンテキストファイル](../user-guide/features/context-files.md) |
| **MCP** | Model Context Protocol を介して外部のツールサーバーに接続する | [MCP](../user-guide/features/mcp.md) |
| **Cron** | 定期的なエージェントタスクをスケジュールする | [Cron](../user-guide/features/cron.md) |
| **委譲** | 並列作業のためにサブエージェントを起動する | [委譲](../user-guide/features/delegation.md) |
| **コード実行** | Hermes のツールをプログラムから呼び出す Python スクリプトを実行する | [コード実行](../user-guide/features/code-execution.md) |
| **ブラウザ** | Web ブラウジングとスクレイピング | [ブラウザ](../user-guide/features/browser.md) |
| **フック** | イベント駆動のコールバックとミドルウェア | [フック](../user-guide/features/hooks.md) |
| **バッチ処理** | 複数の入力をまとめて処理する | [バッチ処理](../user-guide/features/batch-processing.md) |
| **プロバイダールーティング** | 複数の LLM プロバイダーにリクエストを振り分ける | [プロバイダールーティング](../user-guide/features/provider-routing.md) |

## 次に読むべきもの

今あなたがいる段階に応じて、以下を参照してください。

- **インストールが終わったばかりの場合は？** → [クイックスタート](./quickstart.md) に進み、最初の会話を実行しましょう。
- **クイックスタートを完了した場合は？** → [CLI の使い方](../user-guide/cli.md) と [設定](../user-guide/configuration.md) を読み、セットアップをカスタマイズしましょう。
- **基本に慣れてきた場合は？** → [ツール](../user-guide/features/tools.md)、[スキル](../user-guide/features/skills.md)、[メモリ](../user-guide/features/memory.md) を探求し、エージェントの本領を引き出しましょう。
- **チーム向けにセットアップする場合は？** → [セキュリティ](../user-guide/security.md) と [セッション](../user-guide/sessions.md) を読み、アクセス制御と会話の管理について理解しましょう。
- **構築する準備ができた場合は？** → [開発者ガイド](/docs/developer-guide/architecture) に飛び込み、内部構造を理解してコントリビュートを始めましょう。
- **実践的な例が欲しい場合は？** → [ガイド](../guides/tips.md) のセクションで、実際のプロジェクトやヒントを確認しましょう。

:::tip
すべてを読む必要はありません。あなたの目標に合ったパスを選び、順番にリンクをたどれば、すぐに使いこなせるようになります。次のステップを見つけるために、いつでもこのページに戻ってこられます。
:::
