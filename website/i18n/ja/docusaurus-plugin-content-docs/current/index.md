---
slug: /
sidebar_position: 0
title: "Hermes Agent ドキュメント"
description: "Nous Research が開発した自己改善型 AI エージェント。経験からスキルを生み出し、利用しながら改善し、セッションをまたいで記憶する組み込みの学習ループを備えています。"
hide_table_of_contents: true
displayed_sidebar: docs
---

# Hermes Agent

[Nous Research](https://nousresearch.com) が開発した自己改善型 AI エージェント。組み込みの学習ループを備えた唯一のエージェントです。経験からスキルを生み出し、利用しながら改善し、知識を残すよう自らを促し、セッションをまたいで「あなたが何者か」を深く学習し続けます。

<div style={{display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap'}}>
  <a href="/docs/getting-started/installation" style={{display: 'inline-block', padding: '0.6rem 1.2rem', backgroundColor: '#FFD700', color: '#07070d', borderRadius: '8px', fontWeight: 600, textDecoration: 'none'}}>はじめる →</a>
  <a href="https://github.com/NousResearch/hermes-agent" style={{display: 'inline-block', padding: '0.6rem 1.2rem', border: '1px solid rgba(255,215,0,0.2)', borderRadius: '8px', textDecoration: 'none'}}>GitHub で見る</a>
</div>

## インストール

**Linux / macOS / WSL2**

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

**Windows（ネイティブ、PowerShell）** — *アーリーベータ、[詳細 →](/docs/user-guide/windows-native)*

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

**Android（Termux）** — Linux と同じ curl ワンライナーでインストールできます。インストーラーが Termux を自動検出します。

インストーラーの動作内容、ユーザー単位 vs root のレイアウト、Windows 固有の注意点については、**[インストールガイド](/docs/getting-started/installation)** を参照してください。

## Hermes Agent とは？

IDE に紐付いたコーディング用コパイロットでも、単一の API をラップしたチャットボットでもありません。**自律エージェント**であり、稼働時間が長くなるほど能力が高まります。置いた場所がそのまま住処になります。月額 5 ドルの VPS でも、GPU クラスタでも、アイドル時にほとんどコストがかからないサーバーレス基盤（Daytona、Modal）でも動作します。自分では一度も SSH しないクラウド VM 上で作業させながら、Telegram から話しかけられます。あなたのノート PC に縛られることはありません。

## クイックリンク

| | |
|---|---|
| 🚀 **[インストール](/docs/getting-started/installation)** | Linux、macOS、WSL2、または Windows ネイティブ（アーリーベータ）に 60 秒でインストール |
| 📖 **[クイックスタートチュートリアル](/docs/getting-started/quickstart)** | 最初の対話と、試すべき主要機能 |
| 🗺️ **[ラーニングパス](/docs/getting-started/learning-path)** | 自分の経験レベルに合ったドキュメントを見つける |
| ⚙️ **[設定](/docs/user-guide/configuration)** | 設定ファイル、プロバイダー、モデル、オプション |
| 💬 **[メッセージングゲートウェイ](/docs/user-guide/messaging)** | Telegram、Discord、Slack、WhatsApp、Teams などをセットアップ |
| 🔧 **[ツールとツールセット](/docs/user-guide/features/tools)** | 70 以上の組み込みツールと、その設定方法 |
| 🧠 **[メモリシステム](/docs/user-guide/features/memory)** | セッションをまたいで成長する永続メモリ |
| 📚 **[スキルシステム](/docs/user-guide/features/skills)** | エージェントが生成・再利用する手続き的記憶 |
| 🔌 **[MCP 連携](/docs/user-guide/features/mcp)** | MCP サーバーへの接続、ツールのフィルタリング、Hermes の安全な拡張 |
| 🧭 **[Hermes で MCP を使う](/docs/guides/use-mcp-with-hermes)** | 実践的な MCP セットアップのパターン、例、チュートリアル |
| 🎙️ **[ボイスモード](/docs/user-guide/features/voice-mode)** | CLI、Telegram、Discord、Discord VC でのリアルタイム音声対話 |
| 🗣️ **[Hermes でボイスモードを使う](/docs/guides/use-voice-mode-with-hermes)** | Hermes の音声ワークフローの実践的なセットアップと利用パターン |
| 🎭 **[パーソナリティと SOUL.md](/docs/user-guide/features/personality)** | グローバルな SOUL.md で Hermes の既定の人格を定義 |
| 📄 **[コンテキストファイル](/docs/user-guide/features/context-files)** | すべての対話に影響を与えるプロジェクトのコンテキストファイル |
| 🔒 **[セキュリティ](/docs/user-guide/security)** | コマンド承認、認可、コンテナ分離 |
| 💡 **[ヒントとベストプラクティス](/docs/guides/tips)** | Hermes を最大限に活用するためのちょっとしたコツ |
| 🏗️ **[アーキテクチャ](/docs/developer-guide/architecture)** | 内部の仕組み |
| ❓ **[FAQ とトラブルシューティング](/docs/reference/faq)** | よくある質問と解決策 |

## 主な機能

- **閉じた学習ループ** — 定期的なナッジを伴うエージェント主導のメモリ管理、自律的なスキル生成、利用中のスキル自己改善、LLM 要約付きの FTS5 セッション横断リコール、そして [Honcho](https://github.com/plastic-labs/honcho) による弁証法的ユーザーモデリング
- **ノート PC だけでなく、どこでも動く** — 6 つのターミナルバックエンド: local、Docker、SSH、Daytona、Singularity、Modal。Daytona と Modal はサーバーレスの永続性を提供し、環境はアイドル時に休止してほとんどコストがかかりません
- **あなたのいる場所で動く** — CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、メール、SMS、DingTalk、Feishu、WeCom、Weixin、QQ Bot、Yuanbao、BlueBubbles、Home Assistant、Microsoft Teams、Google Chat など、1 つのゲートウェイから 20 以上のプラットフォームに対応
- **モデル開発者による開発** — Hermes、Nomos、Psyche を生み出したラボ [Nous Research](https://nousresearch.com) が開発。[Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)、OpenAI、その他あらゆるエンドポイントで動作します
- **スケジュール自動化** — 任意のプラットフォームへの配信に対応した組み込み cron
- **委譲と並列化** — 並列ワークストリーム用に分離されたサブエージェントを起動。`execute_code` によるプログラム的ツール呼び出しが、複数ステップのパイプラインを単一の推論呼び出しにまとめます
- **オープン標準のスキル** — [agentskills.io](https://agentskills.io) と互換。スキルはポータブルで共有可能、Skills Hub を通じてコミュニティから提供されます
- **完全な Web 操作** — 検索、抽出、ブラウジング、ビジョン、画像生成、TTS
- **MCP サポート** — 任意の MCP サーバーに接続してツール機能を拡張
- **研究にすぐ使える** — バッチ処理、トラジェクトリのエクスポート、Atropos による RL トレーニング。Hermes、Nomos、Psyche モデルを生み出したラボ [Nous Research](https://nousresearch.com) が開発

## LLM とコーディングエージェント向け

このドキュメントへの機械可読なエントリーポイント:

- **[`/llms.txt`](/llms.txt)** — 各ドキュメントページの短い説明付きの厳選インデックス。約 17 KB で、LLM のコンテキストに安全に読み込めます。
- **[`/llms-full.txt`](/llms-full.txt)** — すべてのドキュメントページを 1 つの Markdown ファイルに連結したもの。一括取り込み用。約 1.8 MB。

どちらのファイルも `/docs/llms.txt` および `/docs/llms-full.txt` でも解決されます。デプロイのたびに新しく生成されます。
