---
title: "連携"
sidebar_label: "概要"
sidebar_position: 0
---

# 連携

Hermes Agent は、AI 推論、ツールサーバー、IDE ワークフロー、プログラムによるアクセスなどのために外部システムに接続します。これらの連携は、Hermes ができることと、Hermes が実行できる場所を拡張します。

:::tip まずはここから
連携を 1 つだけ設定する時間しかない場合は、[Nous Portal](/docs/integrations/nous-portal) を設定してください。1 回の OAuth ログインで 300 以上のモデルと、4 つの Tool Gateway ツール（Web 検索、画像生成、TTS、ブラウザ自動化）が利用できます。
:::

## AI プロバイダーとルーティング

Hermes は、複数の AI 推論プロバイダーを標準でサポートしています。インタラクティブに設定するには `hermes model` を使用するか、`config.yaml` で設定します。

- **[AI プロバイダー](./providers.md)** — OpenRouter、Anthropic、OpenAI、Google、および任意の OpenAI 互換エンドポイント。Hermes は、プロバイダーごとにビジョン、ストリーミング、ツール使用などの機能を自動検出します。
- **[プロバイダールーティング](../user-guide/features/provider-routing.md)** — OpenRouter リクエストを処理する基盤プロバイダーをきめ細かく制御します。並べ替え、ホワイトリスト、ブラックリスト、明示的な優先順位付けにより、コスト、速度、品質を最適化します。
- **[フォールバックプロバイダー](../user-guide/features/fallback-providers.md)** — プライマリモデルがエラーに遭遇したときに、バックアップ LLM プロバイダーへ自動フェイルオーバーします。プライマリモデルのフォールバックと、ビジョン、圧縮、タイトル生成などの補助タスク向けの独立したフォールバックを含みます。

## ツールサーバー（MCP）

- **[MCP サーバー](../user-guide/features/mcp.md)** — Model Context Protocol を介して Hermes を外部ツールサーバーに接続します。ネイティブの Hermes ツールを書くことなく、GitHub、データベース、ファイルシステム、ブラウザスタック、内部 API などのツールにアクセスできます。stdio と SSE の両方のトランスポート、サーバーごとのツールフィルタリング、機能を考慮したリソース/プロンプトの登録をサポートします。

## Web 検索バックエンド

`web_search` および `web_extract` ツールは 8 つのバックエンドプロバイダーをサポートし、`config.yaml` または `hermes tools` で設定します:

| バックエンド | 環境変数 | 検索 | 抽出 | クロール |
|---------|---------|--------|---------|-------|
| **Firecrawl**（デフォルト） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Brave**（無料枠） | `BRAVE_SEARCH_API_KEY` | ✔ | — | — |
| **DuckDuckGo**（ddgs） | _（なし）_ | ✔ | — | — |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **xAI** | `XAI_API_KEY` | ✔ | — | — |

クイックセットアップの例:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | brave-free | ddgs | tavily | perplexity | keenable | exa | parallel | xai
```

`web.backend` が設定されていない場合、利用可能な API キーからバックエンドが自動検出されます。セルフホストの Firecrawl も `FIRECRAWL_API_URL` 経由でサポートされています。

## ブラウザ自動化

Hermes には、`browser_exec`（CDP 経由の Python）による完全なブラウザ自動化が含まれており、Web サイトのナビゲーション、フォームの入力、情報の抽出が行えます。ブラウザの実行場所は複数から選べます:

- **Browser Use Cloud** — ステルス、住宅用プロキシ、CAPTCHA 解決、再利用可能なブラウザプロファイルを備えたマネージド Chromium
- **Browserbase** — マネージドブラウザ、アンチボットツール、CAPTCHA 解決、住宅用プロキシを備えた代替のクラウドブラウザプロバイダー
- **ローカルの Chromium 系 CDP** — `/browser connect` を使用して、実行中の Chrome、Brave、Chromium、Edge ブラウザに接続します
- **ローカル Chromium** — Hermes 自身が起動するヘッドレスのローカル Chromium（デフォルト）

Camofox アンチ検出ブラウザはカタログプラグインです: `hermes plugins install camofox`。

セットアップと使用方法については [ブラウザ自動化](../user-guide/features/browser.md) を参照してください。

## 音声と TTS プロバイダー

すべてのメッセージングプラットフォームにわたるテキスト読み上げと音声テキスト変換:

| プロバイダー | 品質 | コスト | API キー |
|----------|---------|------|---------|
| **Edge TTS**（デフォルト） | 良好 | 無料 | 不要 |
| **ElevenLabs** | 優秀 | 有料 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 良好 | 有料 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax** | 良好 | 有料 | `MINIMAX_API_KEY` |
| **xAI TTS** | 良好 | 有料 | `XAI_API_KEY` |
| **NeuTTS** | 良好 | 無料 | 不要 |

音声テキスト変換は 8 つのプロバイダーをサポートします: ローカルの faster-whisper（無料、オンデバイスで実行）、ローカルコマンドラッパー、Groq、OpenAI Whisper API、Mistral、xAI、ElevenLabs Scribe、DeepInfra。ボイスメッセージの文字起こしは、Telegram、Discord、WhatsApp、その他のメッセージングプラットフォームで動作します。詳細については [音声と TTS](../user-guide/features/tts.md) と [ボイスモード](../user-guide/features/voice-mode.md) を参照してください。

## IDE とエディタの連携

- **[IDE 連携（ACP）](../user-guide/features/acp.md)** — VS Code、Zed、JetBrains などの ACP 互換エディタ内で Hermes Agent を使用します。Hermes は ACP サーバーとして実行され、チャットメッセージ、ツールアクティビティ、ファイル差分、ターミナルコマンドをエディタ内でレンダリングします。

## プログラムによるアクセス

- **[API サーバー](../user-guide/features/api-server.md)** — Hermes を OpenAI 互換の HTTP エンドポイントとして公開します。OpenAI 形式を話す任意のフロントエンド — Open WebUI、LobeChat、LibreChat、NextChat、ChatBox — が接続し、フルツールセットを備えた Hermes をバックエンドとして使用できます。

## メモリとパーソナライゼーション

- **[組み込みメモリ](../user-guide/features/memory.md)** — `MEMORY.md` と `USER.md` ファイルによる、永続的で厳選されたメモリ。エージェントは、セッションをまたいで存続する、個人的なメモとユーザープロファイルデータの上限付きストアを維持します。
- **[メモリプロバイダー](../user-guide/features/memory-providers.md)** — より深いパーソナライゼーションのために外部メモリバックエンドをプラグインします。Hermes には 2 つのプロバイダーが同梱されています: RetainDB（ハイブリッド検索）と ByteRover（CLI ベース）。Honcho（弁証法的推論）、Hindsight（ナレッジグラフ）、Supermemory（セマンティック想起）、Mem0（クラウド抽出）、OpenViking（階層型検索）、Holographic（ローカル SQLite）は、`hermes plugins install <name>` でプラグインカタログから利用できます。

## メッセージングプラットフォーム

Hermes は、すべて同じ `gateway` サブシステムを通じて設定される、27 以上のメッセージングプラットフォームでゲートウェイボットとして実行されます:

- **[Telegram](../user-guide/messaging/telegram.md)**、**[Discord](../user-guide/messaging/discord.md)**、**[Slack](../user-guide/messaging/slack.md)**、**[WhatsApp](../user-guide/messaging/whatsapp.md)**、**[Signal](../user-guide/messaging/signal.md)**、**[Matrix](../user-guide/messaging/matrix.md)**、**[Mattermost](../user-guide/messaging/mattermost.md)**、**[Email](../user-guide/messaging/email.md)**、**[SMS](../user-guide/messaging/sms.md)**、**[DingTalk](../user-guide/messaging/dingtalk.md)**、**[Feishu/Lark](../user-guide/messaging/feishu.md)**、**[WeCom](../user-guide/messaging/wecom.md)**、**[WeCom コールバック](../user-guide/messaging/wecom-callback.md)**、**[Weixin](../user-guide/messaging/weixin.md)**、**[BlueBubbles](../user-guide/messaging/bluebubbles.md)**、**[Buzz](/docs/user-guide/messaging/buzz)**、**[QQ Bot](../user-guide/messaging/qqbot.md)**、**[Yuanbao](../user-guide/messaging/yuanbao.md)**、**[Home Assistant](../user-guide/messaging/homeassistant.md)**（プラグイン）、**[Microsoft Teams](../user-guide/messaging/teams.md)**、**[Microsoft Teams Meetings](../user-guide/messaging/teams-meetings.md)**、**[Microsoft Graph Webhook](../user-guide/messaging/msgraph-webhook.md)**、**[Google Chat](../user-guide/messaging/google_chat.md)**、**[LINE](../user-guide/messaging/line.md)**、**[ntfy](/docs/user-guide/messaging/ntfy)**、**[SimpleX](/docs/user-guide/messaging/simplex)**、**[Open WebUI](../user-guide/messaging/open-webui.md)**、**[Webhooks](../user-guide/messaging/webhooks.md)**

プラットフォーム比較表とセットアップガイドについては [メッセージングゲートウェイ概要](../user-guide/messaging/index.md) を参照してください。

### クイック接続リンク

主要なプラットフォームには「ボット/アプリを作成する」ための正規の URL があり、中には適切なフォームを最初から開くパラメーターを受け付けるものもあります。コンソール内を探し回らずに、直接アクセスしてください:

| プラットフォーム | 直接リンク | 開くもの |
|----------|-------------|---------------|
| **Telegram** | [t.me/BotFather](https://t.me/BotFather) | BotFather とのチャット — `/newbot` を送信してボットトークンを発行します |
| **Discord** | [discord.com/developers/applications?new_application=true](https://discord.com/developers/applications?new_application=true) | **New Application** ダイアログが開いた状態の Developer Portal |
| **Slack** | [api.slack.com/apps?new_app=1](https://api.slack.com/apps?new_app=1) | **Create New App** ダイアログ — *From an app manifest* を選び、`hermes slack manifest --agent-view` が生成するマニフェストを貼り付けます |
| **LINE** | [developers.line.biz/console](https://developers.line.biz/console/) | Messaging API チャネルを作成するための LINE Developers Console |
| **Feishu/Lark** | [open.feishu.cn/app](https://open.feishu.cn/app) | カスタムアプリを作成するための Feishu オープンプラットフォームコンソール |

アクセスした後の手順は、各プラットフォームのセットアップページで説明しています。

## コラボレーションワークスペース

- **[Buzz](/docs/integrations/buzz)** — Block による Nostr ベースの人間 + エージェントのワークスペース。連携方法は 3 つあります: Buzz Desktop が Hermes をマネージド ACP ランタイムとして起動する方法、`buzz-acp` リレーブリッジがサーバー側で Hermes のアイデンティティをホストする方法、ネイティブのゲートウェイプラットフォームが Hermes のメモリ/スキル/承認/cron をすべて備えた状態で Buzz チャンネルに参加する方法です。概要ページで 3 つを比較しています。

## ホームオートメーション

- **[Home Assistant](../user-guide/messaging/homeassistant.md)** — 4 つの専用ツール（`ha_list_entities`、`ha_get_state`、`ha_list_services`、`ha_call_service`）を介してスマートホームデバイスを制御し、ゲートウェイを通じてリアルタイムの状態変化に反応します。`homeassistant` カタログプラグイン（`hermes plugins install homeassistant`）として提供され、`HASS_TOKEN` が設定されるとツールセットが自動的に有効化されます。

## プラグイン

- **[プラグインシステム](../user-guide/features/plugins.md)** — コアコードを変更せずに、カスタムツール、ライフサイクルフック、CLI コマンドで Hermes を拡張します。プラグインは、`~/.hermes/plugins/`、プロジェクトローカルの `.hermes/plugins/`、pip インストールされたエントリーポイントから発見されます。
- **[プラグインの構築](../developer-guide/plugins/index.md)** — ツール、フック、CLI コマンドを備えた Hermes プラグインを作成するためのステップバイステップガイド。

## トレーニングと評価

- **[バッチ処理](../user-guide/features/batch-processing.md)** — 何百ものプロンプトにわたってエージェントを並列実行し、トレーニングデータ生成または評価のために、構造化された ShareGPT 形式の軌跡データを生成します。
