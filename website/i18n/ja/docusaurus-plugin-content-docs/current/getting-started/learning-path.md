---
sidebar_position: 3
title: '学習パス'
description: 'あなたの経験レベルと目標に合わせて、Hermes Agent ドキュメントの学習パスを選びましょう。'
---

# 学習パス

Hermes Agent は多くのことができます。CLI アシスタント、Telegram/Discord ボット、タスク自動化、RL トレーニングなどです。このページでは、あなたの経験レベルと達成したいことに応じて、どこから始め、何を読めばよいかを見つける手助けをします。

:::tip Start Here
まだ Hermes Agent をインストールしていない場合は、[インストールガイド](/docs/getting-started/installation) から始め、続いて [クイックスタート](/docs/getting-started/quickstart) を実行してください。以下の内容はすべて、動作するインストール環境があることを前提としています。
:::

## このページの使い方

- **自分のレベルがわかっている場合は？** [経験レベル別の表](#by-experience-level) に進み、あなたの段階に応じた読む順序に従ってください。
- **具体的な目標がある場合は？** [ユースケース別](#by-use-case) に進み、当てはまるシナリオを見つけてください。
- **ざっと眺めたいだけの場合は？** [主な機能](#key-features-at-a-glance) の表で、Hermes Agent ができることの概要を素早く確認してください。

## 経験レベル別 {#by-experience-level}

| レベル | 目標 | 推奨される読む順序 | 所要時間の目安 |
|---|---|---|---|
| **初級** | 環境を整えて動かし、基本的な会話を行い、組み込みのツールを使う | [インストール](/docs/getting-started/installation) → [クイックスタート](/docs/getting-started/quickstart) → [CLI の使い方](/docs/user-guide/cli) → [設定](/docs/user-guide/configuration) | 約1時間 |
| **中級** | メッセージングボットをセットアップし、メモリ、cron ジョブ、スキルなどの高度な機能を使う | [セッション](/docs/user-guide/sessions) → [メッセージング](/docs/user-guide/messaging) → [ツール](/docs/user-guide/features/tools) → [スキル](/docs/user-guide/features/skills) → [メモリ](/docs/user-guide/features/memory) → [Cron](/docs/user-guide/features/cron) | 約2〜3時間 |
| **上級** | カスタムツールを構築し、スキルを作成し、RL でモデルをトレーニングし、プロジェクトに貢献する | [アーキテクチャ](/docs/developer-guide/architecture) → [ツールの追加](/docs/developer-guide/adding-tools) → [スキルの作成](/docs/developer-guide/creating-skills) → [RL トレーニング](/docs/user-guide/features/rl-training) → [コントリビュート](/docs/developer-guide/contributing) | 約4〜6時間 |

## ユースケース別 {#by-use-case}

やりたいことに当てはまるシナリオを選んでください。それぞれ、読むべき順序で関連ドキュメントへのリンクを示しています。

### 「CLI のコーディングアシスタントが欲しい」

Hermes Agent を、コードの記述、レビュー、実行を行うインタラクティブなターミナルアシスタントとして使います。

1. [インストール](/docs/getting-started/installation)
2. [クイックスタート](/docs/getting-started/quickstart)
3. [CLI の使い方](/docs/user-guide/cli)
4. [コード実行](/docs/user-guide/features/code-execution)
5. [コンテキストファイル](/docs/user-guide/features/context-files)
6. [ヒントとコツ](/docs/guides/tips)

:::tip
コンテキストファイルを使って、ファイルを直接会話に渡せます。Hermes Agent はあなたのプロジェクト内のコードを読み取り、編集し、実行できます。
:::

### 「Telegram/Discord ボットが欲しい」

Hermes Agent を、お気に入りのメッセージングプラットフォーム上のボットとしてデプロイします。

1. [インストール](/docs/getting-started/installation)
2. [設定](/docs/user-guide/configuration)
3. [メッセージングの概要](/docs/user-guide/messaging)
4. [Telegram のセットアップ](/docs/user-guide/messaging/telegram)
5. [Discord のセットアップ](/docs/user-guide/messaging/discord)
6. [ボイスモード](/docs/user-guide/features/voice-mode)
7. [Hermes でボイスモードを使う](/docs/guides/use-voice-mode-with-hermes)
8. [セキュリティ](/docs/user-guide/security)

完全なプロジェクトの例については、以下を参照してください。
- [デイリーブリーフィングボット](/docs/guides/daily-briefing-bot)
- [チーム向け Telegram アシスタント](/docs/guides/team-telegram-assistant)

### 「タスクを自動化したい」

定期的なタスクをスケジュールしたり、バッチジョブを実行したり、エージェントのアクションを連鎖させたりします。

1. [クイックスタート](/docs/getting-started/quickstart)
2. [Cron スケジューリング](/docs/user-guide/features/cron)
3. [バッチ処理](/docs/user-guide/features/batch-processing)
4. [委譲](/docs/user-guide/features/delegation)
5. [フック](/docs/user-guide/features/hooks)

:::tip
Cron ジョブを使うと、あなたがその場にいなくても、Hermes Agent がスケジュールに従ってタスク（毎日のサマリー、定期的なチェック、自動レポートなど）を実行できます。
:::

### 「カスタムツール/スキルを構築したい」

あなた自身のツールや再利用可能なスキルパッケージで Hermes Agent を拡張します。

1. [プラグイン](/docs/user-guide/features/plugins)
2. [Hermes プラグインを構築する](/docs/guides/build-a-hermes-plugin)
3. [ツールの概要](/docs/user-guide/features/tools)
4. [スキルの概要](/docs/user-guide/features/skills)
5. [MCP（Model Context Protocol）](/docs/user-guide/features/mcp)
6. [アーキテクチャ](/docs/developer-guide/architecture)
7. [ツールの追加](/docs/developer-guide/adding-tools)
8. [スキルの作成](/docs/developer-guide/creating-skills)

:::tip
ほとんどのカスタムツール作成では、プラグインから始めてください。[ツールの追加](/docs/developer-guide/adding-tools)
のページは Hermes コア本体の開発向けであり、通常のユーザー/カスタムツールのパスではありません。
:::

### 「モデルをトレーニングしたい」

Hermes Agent の組み込み RL トレーニングパイプラインを使い、強化学習でモデルの振る舞いをファインチューニングします。

1. [クイックスタート](/docs/getting-started/quickstart)
2. [設定](/docs/user-guide/configuration)
3. [RL トレーニング](/docs/user-guide/features/rl-training)
4. [プロバイダールーティング](/docs/user-guide/features/provider-routing)
5. [アーキテクチャ](/docs/developer-guide/architecture)

:::tip
RL トレーニングは、Hermes Agent が会話やツール呼び出しをどのように扱うかという基本をすでに理解している場合に最も効果的です。初めての方は、まず初級パスを進めてください。
:::

### 「Python ライブラリとして使いたい」

Hermes Agent をプログラムから、あなた自身の Python アプリケーションに統合します。

1. [インストール](/docs/getting-started/installation)
2. [クイックスタート](/docs/getting-started/quickstart)
3. [Python ライブラリガイド](/docs/guides/python-library)
4. [アーキテクチャ](/docs/developer-guide/architecture)
5. [ツール](/docs/user-guide/features/tools)
6. [セッション](/docs/user-guide/sessions)

## 主な機能の早見表 {#key-features-at-a-glance}

何が使えるのかわからない場合は？ こちらが主要な機能の簡単な一覧です。

| 機能 | 何ができるか | リンク |
|---|---|---|
| **ツール** | エージェントが呼び出せる組み込みツール（ファイル I/O、検索、シェルなど） | [ツール](/docs/user-guide/features/tools) |
| **スキル** | 新しい機能を追加するインストール可能なプラグインパッケージ | [スキル](/docs/user-guide/features/skills) |
| **メモリ** | セッションをまたいで永続化されるメモリ | [メモリ](/docs/user-guide/features/memory) |
| **コンテキストファイル** | ファイルやディレクトリを会話に取り込む | [コンテキストファイル](/docs/user-guide/features/context-files) |
| **MCP** | Model Context Protocol を介して外部のツールサーバーに接続する | [MCP](/docs/user-guide/features/mcp) |
| **Cron** | 定期的なエージェントタスクをスケジュールする | [Cron](/docs/user-guide/features/cron) |
| **委譲** | 並列作業のためにサブエージェントを起動する | [委譲](/docs/user-guide/features/delegation) |
| **コード実行** | Hermes のツールをプログラムから呼び出す Python スクリプトを実行する | [コード実行](/docs/user-guide/features/code-execution) |
| **ブラウザ** | Web ブラウジングとスクレイピング | [ブラウザ](/docs/user-guide/features/browser) |
| **フック** | イベント駆動のコールバックとミドルウェア | [フック](/docs/user-guide/features/hooks) |
| **バッチ処理** | 複数の入力をまとめて処理する | [バッチ処理](/docs/user-guide/features/batch-processing) |
| **RL トレーニング** | 強化学習でモデルをファインチューニングする | [RL トレーニング](/docs/user-guide/features/rl-training) |
| **プロバイダールーティング** | 複数の LLM プロバイダーにリクエストを振り分ける | [プロバイダールーティング](/docs/user-guide/features/provider-routing) |

## 次に読むべきもの

今あなたがいる段階に応じて、以下を参照してください。

- **インストールが終わったばかりの場合は？** → [クイックスタート](/docs/getting-started/quickstart) に進み、最初の会話を実行しましょう。
- **クイックスタートを完了した場合は？** → [CLI の使い方](/docs/user-guide/cli) と [設定](/docs/user-guide/configuration) を読み、セットアップをカスタマイズしましょう。
- **基本に慣れてきた場合は？** → [ツール](/docs/user-guide/features/tools)、[スキル](/docs/user-guide/features/skills)、[メモリ](/docs/user-guide/features/memory) を探求し、エージェントの本領を引き出しましょう。
- **チーム向けにセットアップする場合は？** → [セキュリティ](/docs/user-guide/security) と [セッション](/docs/user-guide/sessions) を読み、アクセス制御と会話の管理について理解しましょう。
- **構築する準備ができた場合は？** → [開発者ガイド](/docs/developer-guide/architecture) に飛び込み、内部構造を理解してコントリビュートを始めましょう。
- **実践的な例が欲しい場合は？** → [ガイド](/docs/guides/tips) のセクションで、実際のプロジェクトやヒントを確認しましょう。

:::tip
すべてを読む必要はありません。あなたの目標に合ったパスを選び、順番にリンクをたどれば、すぐに使いこなせるようになります。次のステップを見つけるために、いつでもこのページに戻ってこられます。
:::
