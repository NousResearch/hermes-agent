---
title: "連携"
sidebar_label: "概要"
sidebar_position: 0
---

# 連携

Hermes Agent は、AI 推論、ツールサーバー、IDE ワークフロー、プログラムによるアクセスなどのために外部システムに接続します。これらの連携は、Hermes ができることと、Hermes が実行できる場所を拡張します。

## AI プロバイダーとルーティング

Hermes は、複数の AI 推論プロバイダーを標準でサポートしています。インタラクティブに設定するには `hermes model` を使用するか、`config.yaml` で設定します。

- **[AI プロバイダー](/docs/user-guide/features/provider-routing)** — OpenRouter、Anthropic、OpenAI、Google、および任意の OpenAI 互換エンドポイント。Hermes は、プロバイダーごとにビジョン、ストリーミング、ツール使用などの機能を自動検出します。
- **[プロバイダールーティング](/docs/user-guide/features/provider-routing)** — OpenRouter リクエストを処理する基盤プロバイダーをきめ細かく制御します。並べ替え、ホワイトリスト、ブラックリスト、明示的な優先順位付けにより、コスト、速度、品質を最適化します。
- **[フォールバックプロバイダー](/docs/user-guide/features/fallback-providers)** — プライマリモデルがエラーに遭遇したときに、バックアップ LLM プロバイダーへ自動フェイルオーバーします。プライマリモデルのフォールバックと、ビジョン、圧縮、Web 抽出用の独立した補助タスクフォールバックを含みます。

## ツールサーバー（MCP）

- **[MCP サーバー](/docs/user-guide/features/mcp)** — Model Context Protocol を介して Hermes を外部ツールサーバーに接続します。ネイティブの Hermes ツールを書くことなく、GitHub、データベース、ファイルシステム、ブラウザスタック、内部 API などのツールにアクセスできます。stdio と SSE の両方のトランスポート、サーバーごとのツールフィルタリング、機能を考慮したリソース/プロンプトの登録をサポートします。

## Web 検索バックエンド

`web_search` および `web_extract` ツールは 4 つのバックエンドプロバイダーをサポートし、`config.yaml` または `hermes tools` で設定します:

| バックエンド | 環境変数 | 検索 | 抽出 | クロール |
|---------|---------|--------|---------|-------|
| **Firecrawl**（デフォルト） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

クイックセットアップの例:

```yaml
web:
  backend: firecrawl    # firecrawl | parallel | tavily | exa
```

`web.backend` が設定されていない場合、利用可能な API キーからバックエンドが自動検出されます。セルフホストの Firecrawl も `FIRECRAWL_API_URL` 経由でサポートされています。

## ブラウザ自動化

Hermes には、Web サイトのナビゲーション、フォームの入力、情報の抽出のための、複数のバックエンドオプションを備えた完全なブラウザ自動化が含まれています:

- **Browserbase** — アンチボットツール、CAPTCHA 解決、住宅用プロキシを備えたマネージドクラウドブラウザ
- **Browser Use** — 代替のクラウドブラウザプロバイダー
- **CDP 経由のローカル Chrome** — `/browser connect` を使用して実行中の Chrome インスタンスに接続します
- **ローカル Chromium** — `agent-browser` CLI 経由のヘッドレスローカルブラウザ

セットアップと使用方法については [ブラウザ自動化](/docs/user-guide/features/browser) を参照してください。

## 音声と TTS プロバイダー

すべてのメッセージングプラットフォームにわたるテキスト読み上げと音声テキスト変換:

| プロバイダー | 品質 | コスト | API キー |
||----------|---------|------|---------|
|| **Edge TTS**（デフォルト） | 良好 | 無料 | 不要 |
|| **ElevenLabs** | 優秀 | 有料 | `ELEVENLABS_API_KEY` |
|| **OpenAI TTS** | 良好 | 有料 | `VOICE_TOOLS_OPENAI_KEY` |
|| **MiniMax** | 良好 | 有料 | `MINIMAX_API_KEY` |
|| **NeuTTS** | 良好 | 無料 | 不要 |

音声テキスト変換は 6 つのプロバイダーをサポートします: ローカルの faster-whisper（無料、オンデバイスで実行）、ローカルコマンドラッパー、Groq、OpenAI Whisper API、Mistral、xAI。ボイスメッセージの文字起こしは、Telegram、Discord、WhatsApp、その他のメッセージングプラットフォームで動作します。詳細については [音声と TTS](/docs/user-guide/features/tts) と [ボイスモード](/docs/user-guide/features/voice-mode) を参照してください。

## IDE とエディタの連携

- **[IDE 連携（ACP）](/docs/user-guide/features/acp)** — VS Code、Zed、JetBrains などの ACP 互換エディタ内で Hermes Agent を使用します。Hermes は ACP サーバーとして実行され、チャットメッセージ、ツールアクティビティ、ファイル差分、ターミナルコマンドをエディタ内でレンダリングします。

## プログラムによるアクセス

- **[API サーバー](/docs/user-guide/features/api-server)** — Hermes を OpenAI 互換の HTTP エンドポイントとして公開します。OpenAI 形式を話す任意のフロントエンド — Open WebUI、LobeChat、LibreChat、NextChat、ChatBox — が接続し、フルツールセットを備えた Hermes をバックエンドとして使用できます。

## メモリとパーソナライゼーション

- **[組み込みメモリ](/docs/user-guide/features/memory)** — `MEMORY.md` と `USER.md` ファイルによる、永続的で厳選されたメモリ。エージェントは、セッションをまたいで存続する、個人的なメモとユーザープロファイルデータの上限付きストアを維持します。
- **[メモリプロバイダー](/docs/user-guide/features/memory-providers)** — より深いパーソナライゼーションのために外部メモリバックエンドをプラグインします。8 つのプロバイダーがサポートされています: Honcho（弁証法的推論）、OpenViking（階層型検索）、Mem0（クラウド抽出）、Hindsight（ナレッジグラフ）、Holographic（ローカル SQLite）、RetainDB（ハイブリッド検索）、ByteRover（CLI ベース）、Supermemory。

## メッセージングプラットフォーム

Hermes は、すべて同じ `gateway` サブシステムを通じて設定される、19 以上のメッセージングプラットフォームでゲートウェイボットとして実行されます:

- **[Telegram](/docs/user-guide/messaging/telegram)**、**[Discord](/docs/user-guide/messaging/discord)**、**[Slack](/docs/user-guide/messaging/slack)**、**[WhatsApp](/docs/user-guide/messaging/whatsapp)**、**[Signal](/docs/user-guide/messaging/signal)**、**[Matrix](/docs/user-guide/messaging/matrix)**、**[Mattermost](/docs/user-guide/messaging/mattermost)**、**[Email](/docs/user-guide/messaging/email)**、**[SMS](/docs/user-guide/messaging/sms)**、**[DingTalk](/docs/user-guide/messaging/dingtalk)**、**[Feishu/Lark](/docs/user-guide/messaging/feishu)**、**[WeCom](/docs/user-guide/messaging/wecom)**、**[WeCom コールバック](/docs/user-guide/messaging/wecom-callback)**、**[Weixin](/docs/user-guide/messaging/weixin)**、**[BlueBubbles](/docs/user-guide/messaging/bluebubbles)**、**[QQ Bot](/docs/user-guide/messaging/qqbot)**、**[Yuanbao](/docs/user-guide/messaging/yuanbao)**、**[Home Assistant](/docs/user-guide/messaging/homeassistant)**、**[Microsoft Teams](/docs/user-guide/messaging/teams)**、**[Webhooks](/docs/user-guide/messaging/webhooks)**

プラットフォーム比較表とセットアップガイドについては [メッセージングゲートウェイ概要](/docs/user-guide/messaging) を参照してください。

## ホームオートメーション

- **[Home Assistant](/docs/user-guide/messaging/homeassistant)** — 4 つの専用ツール（`ha_list_entities`、`ha_get_state`、`ha_list_services`、`ha_call_service`）を介してスマートホームデバイスを制御します。Home Assistant ツールセットは、`HASS_TOKEN` が設定されると自動的に有効化されます。

## プラグイン

- **[プラグインシステム](/docs/user-guide/features/plugins)** — コアコードを変更せずに、カスタムツール、ライフサイクルフック、CLI コマンドで Hermes を拡張します。プラグインは、`~/.hermes/plugins/`、プロジェクトローカルの `.hermes/plugins/`、pip インストールされたエントリーポイントから発見されます。
- **[プラグインの構築](/docs/guides/build-a-hermes-plugin)** — ツール、フック、CLI コマンドを備えた Hermes プラグインを作成するためのステップバイステップガイド。

## トレーニングと評価

- **[RL トレーニング](/docs/user-guide/features/rl-training)** — 強化学習とモデルのファインチューニングのために、エージェントセッションから軌跡データを生成します。カスタマイズ可能な報酬関数を備えた Atropos 環境をサポートします。
- **[バッチ処理](/docs/user-guide/features/batch-processing)** — 何百ものプロンプトにわたってエージェントを並列実行し、トレーニングデータ生成または評価のために、構造化された ShareGPT 形式の軌跡データを生成します。
