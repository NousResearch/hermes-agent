---
sidebar_position: 16
title: "Google Gemini"
description: "Hermes AgentをGoogle Geminiで使う — ネイティブAI Studio API、APIキーのセットアップ、ツール呼び出し、ストリーミング、クォータガイダンス"
---

# Google Gemini

Hermes Agentは、**Google AI Studio / Gemini API** を使ってGoogle Geminiをネイティブプロバイダーとしてサポートします（OpenAI互換エンドポイントではありません）。これにより、Hermesは内部のOpenAI形式のメッセージとツールループを、ツール呼び出し、ストリーミング、マルチモーダル入力、Gemini固有のレスポンスメタデータを保持しながら、GeminiのネイティブAPI（Gemini 3以降では[Interactions API](#interactions-api)、それより古いモデルでは `generateContent`）に変換できます。

## 前提条件

- **Google AI Studio APIキー** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)で作成します
- **課金が有効なGoogle Cloudプロジェクト** — エージェント利用には推奨されます。Geminiの無料枠は、Hermesがユーザーのターンごとに複数のモデル呼び出しを行う可能性があるため、長時間実行のエージェントセッションには小さすぎます。
- **Hermesがインストールされていること** — ネイティブGeminiプロバイダーには追加のPythonパッケージは不要です。

:::tip APIキーのパス
`GOOGLE_API_KEY` または `GEMINI_API_KEY` を設定してください。Hermesは `gemini` プロバイダーに対して両方の名前を確認します。
:::

## クイックスタート

```bash
# Add your Gemini API key
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

# Select Gemini as your provider
hermes model
# → Choose "More providers..." → "Google AI Studio"
# → Hermes checks your key tier and shows Gemini models
# → Select a model

# Start chatting
hermes chat
```

直接設定を編集したい場合は、ネイティブGemini APIのベースURLを使用します。

```yaml
model:
  default: gemini-3.7-flash
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 設定

`hermes model` を実行すると、`~/.hermes/config.yaml` には次が含まれます。

```yaml
model:
  default: gemini-3.7-flash
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

そして `~/.hermes/.env` には:

```bash
GOOGLE_API_KEY=...
```

### ネイティブGemini API

推奨されるエンドポイントは次のとおりです。

```text
https://generativelanguage.googleapis.com/v1beta
```

Hermesはこのエンドポイントを検出し、ネイティブGeminiアダプターを作成します。内部的には、Hermesはエージェントループを引き続きOpenAI形式のメッセージで保持し、各リクエストをGeminiのネイティブスキーマに変換します。

- `messages[]` → Gemini `contents[]`
- システムプロンプト → Gemini `systemInstruction`
- ツールスキーマ → Gemini `functionDeclarations`
- ツール結果 → Gemini `functionResponse` パーツ
- ストリーミング応答 → Hermesループ用のOpenAI形式のストリームチャンク

`"type": ["number", "null"]` のようなツールパラメータの型配列は、Geminiのスカラー型と `nullable` の形式に変換されます。複数の型のユニオンは、ネストされたプロパティや配列の要素も含め、`anyOf` によってすべての選択肢が保持されます。これは自動的に行われ、MCPサーバーやプロバイダーの設定を変更する必要はありません。

:::note Gemini 3のthought signature
Gemini 3のツール使用では、Hermesは関数呼び出しパーツに付随する `thoughtSignature` の値を保持し、次のツールターンで再生します。これは複数ステップのエージェントワークフローにおける検証上重要なパスをカバーします。

Gemini 3は、他のレスポンスパーツにもthought signatureを付加する場合があります。Hermesのネイティブアダプターは現在エージェントのツールループに最適化されているため、ツール呼び出し以外のすべてのシグネチャをパーツレベルで完全な忠実度で再生するわけではまだありません。
:::

### ネイティブエンドポイントを優先する

GoogleはOpenAI互換のエンドポイントも公開しています。

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

Hermesのエージェントセッションでは、上記のネイティブGeminiエンドポイントを優先してください。Hermesにはネイティブのアダプターが含まれているため、複数ターンのツール使用、ツール呼び出しの結果、ストリーミング、マルチモーダル入力、Geminiのレスポンスメタデータを、GeminiのネイティブAPIに直接マッピングできます。OpenAI互換エンドポイントは、OpenAI APIの互換性が特に必要な場合に依然として有用です。

以前 `GEMINI_BASE_URL` を `/openai` のURLに設定していた場合は、それを削除するか変更してください。

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

GoogleホストのホストルートのベースURLは自動的に正規化されます。URLがAPIバージョンのセグメント（`v1beta`、`v1alpha`、`v1` など）で終わっていない場合、Hermesが `/v1beta` を付加するため、`GEMINI_BASE_URL=https://generativelanguage.googleapis.com` は `/v1beta` サフィックスを明記した場合と同じように動作します。同じ正規化はGemini TTSのベースURL（`tts.gemini.base_url`）にも適用されます。チャットリクエストがネイティブGeminiのパスを通るのは、ベースURLが `generativelanguage.googleapis.com` または後述のVertex AI expressホストを指している場合のみです。別のホスト上のプロキシはOpenAI互換エンドポイントとして扱われるため、`/openai` 形式のURLで設定してください。

### Interactions API

Google AI Studioでは、HermesはGemini 3以降のモデル（および `gemini-*-latest` エイリアス）を、Googleがそれらに推奨するAPIである[Interactions API](https://aistudio.google.com/docs/interactions-overview)経由で送信します。Gemini 2.xとGemmaモデルは `generateContent` を使用します。

APIは、すべてのモデルに対して、またはモデルごとに自分で選択できます。

```yaml
providers:
  gemini:
    gemini_api: generate_content    # every Gemini model
    models:
      gemini-3.8-flash:
        gemini_api: interactions    # except this one
```

各設定には `interactions`、`generate_content`、または `auto`（デフォルト。上記の動作）を指定します。モデルごとの設定はプロバイダー全体の設定より優先されます。変更は次のリクエストから適用され、再起動は不要です。Gemini 2.xはトークンバジェットでthinkingを制御しますが、Interactions APIにはその設定がないため、2.xモデルを強制的に `interactions` にすると、そのバジェットは失われます。

HermesはInteractionsをステートレス（`store: false`）で呼び出します。`generateContent` と同様に、リクエストごとに会話全体を送信するため、Googleは会話のコピーを保持せず、プロンプトキャッシュも同じように機能します。例外はlong decode（`xhigh` 以上のeffortでのGemini 4）です。Googleはこうした複数リクエストにまたがる回答を、保存されたinteractionからのみ継続するため、それらのリクエストは `store: true` を設定し、Googleは標準の期間（有料ティアでは55日、無料ティアでは1日）それらを保持します。InteractionsはAI Studioのホスト（`generativelanguage.googleapis.com`）でのみ利用できます。Vertex AI expressやプロキシのベースURLでは常に `generateContent` が使用されます。

### Vertex AI Expressモードのキー

Googleは現在、Google AI StudioとVertex AI expressモードの**両方**に `AQ.…` で始まるキーを発行しています（従来の `AIza…` 形式のStudioキーは段階的に廃止されています）。そのため、キーのプレフィックスでは対象のサーフェスを判別できなくなりました。Hermesはキーの形でルートを変更することはなく、設定されたベースURLがサーフェスを決定します。デフォルトのAI Studioホストを使う場合は、`GEMINI_API_KEY` を設定し、`GEMINI_BASE_URL` は未設定のままにします。Vertex AI expressモードのキーの場合は、`GEMINI_BASE_URL` を `https://aiplatform.googleapis.com`（`/v1beta1` の有無は問いません）に設定すると、Hermesが `publishers/google` の形式に補完します。各サーフェスは自身のキーのみを受け付けます。`403 PERMISSION_DENIED` は通常、キーとホストの組み合わせが食い違っていることを意味し、Hermesはもう一方のサーフェスを示すガイダンスを付け加えます。その他のホスト（プロキシ）上のベースURLが書き換えられることはありません。ExpressキーはOAuthベースの[Vertex AIプロバイダー](/docs/guides/google-vertex)とは別物で、こちらはAPIキーを必要としません。

`global` ロケーションでのみ提供されるモデル（Gemini 4 Argon）は、`us-central1` にルーティングされる素のexpressホストからは到達できません。そうしたモデルでは、ベースURLをプロジェクトスコープの形式 `https://aiplatform.googleapis.com/v1beta1/projects/<project>/locations/global` に設定すると、Hermesが同じように `…/publishers/google` に補完します。

:::warning 既存のexpressキー利用者向けのアップグレードに関する注意
以前のHermesリリースは `AQ.` プレフィックスを検出し、そのようなキーを自動的に `aiplatform.googleapis.com` にルーティングしていたため、ドキュメント上のセットアップは「`GEMINI_API_KEY` にexpressキーを設定し、`GEMINI_BASE_URL` は未設定のままにする」というものでした。この自動ルーティングは廃止されました。`GEMINI_BASE_URL` が未設定の場合、チャット、`hermes doctor`、TTSのすべてのリクエストはAI Studioホストに送られ、Vertexのexpressキーはそこで `403 PERMISSION_DENIED` になります。`~/.hermes/.env` に `GEMINI_BASE_URL=https://aiplatform.googleapis.com` を追加する（またはプロバイダーに `base_url` を設定する）操作を一度行い、再起動してください。
:::

## 利用可能なモデル

`hermes model` のピッカーは、Hermesのプロバイダーレジストリで管理されているGeminiモデルを表示します。一般的な選択肢には次が含まれます。

| モデル | ID | 備考 |
|-------|----|-------|
| Gemini 4 Hydrogen | `gemini-4-hydrogen` | 日常的なコーディングとエージェント的な作業向けの高速なGemini 4モデル。Argonと同じウィンドウとlong decodeに加え、`xhigh` のthinkingレベルを備えます。無料ティアと有料ティア |
| Gemini 4 Carbon | `gemini-4-carbon` | 最も強力な推論モデル。Argonよりもターンごとに多くのthinkingを行い、ウィンドウ、long decode、価格はArgonと同じです。有料ティアのみ |
| Gemini 4 Argon | `gemini-4-argon` | フラッグシップの推論モデル。入力と出力で共有の1Mウィンドウ、最大1M出力トークンのlong decode |
| Gemini 3.8 Flash | `gemini-3.8-flash` | 長期的なエージェント作業やコーディング作業向けの、最も高性能なFlashモデル |
| Gemini 3.7 Flash | `gemini-3.7-flash` | 推奨されるデフォルト。速度、性能、マルチモーダル理解のバランス |
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | 推論、数学、コーディングで最も高性能なモデル |
| Gemini 3.5 Flash Lite | `gemini-3.5-flash-lite` | 軽量なタスク向けの最速かつ最低コストのオプション |
| Gemini 2.5 Flash | `gemini-2.5-flash` | thinking機能を備えた前世代の高速モデル |
| Gemini 2.5 Pro | `gemini-2.5-pro` | 前世代の複雑な推論モデル |

モデルの利用可否は時間とともに変化します。モデルが消えたり、あなたのキーで有効化されていない場合は、`hermes model` を再度実行し、現在のリストから1つ選んでください。

:::info モデルID
`provider: gemini` の場合は、`google/gemini-3.7-flash` のようなOpenRouterスタイルのIDではなく、`gemini-3.7-flash` のようなGeminiのネイティブモデルIDを使用してください。
:::

### Gemini 4 Argon: Reasoning effortとlong decode

以下のすべてについて、Gemini 4 CarbonはArgonと同じように動作します。Gemini 4 Argonは常にthinkingを行います。Hermesはreasoning effort（`/reasoning`、`agent.reasoning_effort`）をその `thinkingLevel` にマッピングします。

| Hermesのeffort | `thinkingLevel` | 合計出力バジェット |
|---------------|-----------------|---------------------|
| `none`、`minimal`、`low` | `low`（`none` ではthoughtは非表示） | リクエストごとの上限 |
| `medium`（Googleのデフォルト） | `medium` | リクエストごとの上限 |
| `high` | `high` | リクエストごとの上限 |
| `xhigh` | `high` | 262,144トークン（long decode） |
| `max`、`ultra` | `high` | 1,048,576トークンから入力を引いた値（long decode） |

Gemini 4 Hydrogenには4番目のレベル `xhigh` があります。Hydrogenでは、Hermesは `xhigh`、`max`、`ultra` のeffortに対して `xhigh` を送信し、出力バジェットは上記と同じです。Hydrogenに対するGoogle自身のデフォルトレベルは `high` です。

long decodeを使わない場合、1回のリクエストはストリーミング時に65,536出力トークン（非ストリーミング時は32,768）で停止し、これにはthinkingも含まれます。`xhigh` または `max` では、Hermesは `maxOutputTokens` を引き上げ、リクエストがバジェットを残したままリクエストごとの上限に達した場合（`finishReason: CONTINUATION`）、モデルが完了するまでGoogleの `continuationToken` を付けて同じリクエストを再送信します。これらのホップは1つの応答に結合されます。

ホップをまたぐ使用量は、Googleの課金方法に従って報告されます。各ホップはプロンプトとそれまでの出力を再度読み込むため、合計された入力数は実際のコンテキストよりもはるかに大きくなります。Hermesは合計値で料金を計算しますが、コンテキスト圧縮とコンテキストメーターには最初のホップのプロンプトサイズを渡すため、long decodeのターンが早期の圧縮を引き起こすことはありません。long decodeはネイティブエンドポイント（AI StudioとVertex AI express）でのみ実行されます。Googleの `/openai` 互換エンドポイント、OpenRouter、OAuthの[Vertex AIプロバイダー](/docs/guides/google-vertex)では、リクエストごとの上限で停止します。

### 最新エイリアス

Googleは、ProとFlashのGeminiファミリー向けに移動するエイリアスを公開しています。`gemini-pro-latest` と `gemini-flash-latest` は、Hermesの設定を変更せずにGoogleにモデルを自動的に進化させてほしい場合に便利です。新しいモデルで料金体系が異なる場合、利用料金に影響する可能性があることに注意してください。

| エイリアス | 現在の追跡対象 | 備考 |
|-------|------------------|-------|
| `gemini-pro-latest` | 最新のGemini Proモデル | Googleの現在のProデフォルトを使いたい場合に最適 |
| `gemini-flash-latest` | 最新のGemini Flashモデル | Googleの現在のFlashデフォルトを使いたい場合に最適 |

```yaml
model:
  default: gemini-pro-latest
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

厳密な再現性が必要な場合は、`gemini-3.1-pro-preview` や `gemini-3.7-flash` のような明示的なモデルIDを優先してください。

### Gemini API経由のGemma

GoogleはGemma モデルもGemini API経由で公開しています。HermesはこれらをGoogleモデルとして認識しますが、新規ユーザーが長時間実行のエージェントセッションに評価ティアのモデルを誤って選択しないように、非常にスループットの低いGemmaエントリをデフォルトのモデルピッカーから隠しています。

有用な評価用IDには次が含まれます。

| モデル | ID | 備考 |
|-------|----|-------|
| Gemma 4 31B IT | `gemma-4-31b-it` | より大きなGemmaモデル。互換性と品質の評価に有用 |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | 利用可能な場合、より小さなアクティブパラメータのバリアント |

これらのモデルは、Gemini APIキー上の評価オプションとして扱うのが最適です。GoogleのGemma APIの料金は無料枠のみで、利用上限は本番のGeminiモデルと比べて低いため、Hermesエージェントの持続的な利用は通常、有料のGeminiモデル、セルフホスト型のデプロイ、または適切なクォータを持つ別のプロバイダーに移行すべきです。

ピッカーから隠されているGemmaモデルを使用するには、直接設定します。

```yaml
model:
  default: gemma-4-31b-it
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## セッション中のモデル切り替え

会話中に `/model` コマンドを使用します。

```text
/model gemini-3.7-flash
/model gemini-flash-latest
/model gemini-3.1-pro-preview
/model gemini-pro-latest
/model gemma-4-31b-it
/model gemini-3.1-flash-lite-preview
```

まだGeminiを設定していない場合は、セッションを終了して先に `hermes model` を実行してください。`/model` は、すでに設定済みのプロバイダーとモデルの間で切り替えます。新しいAPIキーを収集することはありません。

## 診断

```bash
hermes doctor
```

doctorは次を確認します。

- `GOOGLE_API_KEY` または `GEMINI_API_KEY` が利用可能かどうか
- 設定されたプロバイダーの認証情報が解決できるかどうか

## ゲートウェイ（メッセージングプラットフォーム）

Geminiは、すべてのHermesゲートウェイプラットフォーム（Telegram、Discord、Slack、WhatsApp、LINE、Feishuなど）で機能します。Geminiをプロバイダーとして設定し、通常どおりゲートウェイを起動します。

```bash
hermes gateway setup
hermes gateway start
```

ゲートウェイは `config.yaml` を読み取り、同じGeminiプロバイダー設定を使用します。

## トラブルシューティング

### 「Gemini native client requires an API key」

Hermesが使用可能なAPIキーを見つけられませんでした。次のいずれかを `~/.hermes/.env` に追加してください。

```bash
GOOGLE_API_KEY=...
# or
GEMINI_API_KEY=...
```

その後、`hermes model` を再度実行してください。

### 「This Google API key is on the free tier」

Hermesはセットアップ中にGemini APIキーをプローブします。ツール使用、再試行、圧縮、補助タスクが複数のモデル呼び出しを必要とする可能性があるため、無料枠のクォータはわずか数ターンのエージェント操作で枯渇する場合があります。

キーに紐づくGoogle Cloudプロジェクトで課金を有効化し、必要に応じてキーを再生成してから、次を実行してください。

```bash
hermes model
```

### 「404 model not found」

選択したモデルは、あなたのアカウント、リージョン、またはキーで利用できません。`hermes model` を再度実行し、現在のリストから別のGeminiモデルを選んでください。

### `hermes model` にGemmaモデルが表示されない

Hermesは、スループットの低いGemmaモデルをデフォルトでピッカーから隠す場合があります。意図的に評価したい場合は、`~/.hermes/config.yaml` でモデルIDを直接設定してください。

### Gemmaで「429 quota exceeded」

Gemini API経由で公開されるGemmaモデルは評価に有用ですが、そのGemini API無料枠の上限は低いです。互換性テストに使用し、その後、持続的なエージェントセッションには有料のGeminiモデルまたは別のプロバイダーに切り替えてください。

### OpenAI互換エンドポイントが設定されている

`~/.hermes/.env` で次を確認してください。

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

ネイティブエンドポイントに変更するか、オーバーライドを削除してください。

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### ツール呼び出しがスキーマエラーで失敗する

Hermesをアップグレードし、`hermes model` を再実行してください。ネイティブGeminiアダプターは、Geminiのより厳格な関数宣言形式に合わせてツールスキーマをサニタイズします。古いビルドやカスタムエンドポイントではそうでない場合があります。

## 関連

- [AIプロバイダー](../integrations/providers.md)
- [設定](../user-guide/configuration.md)
- [フォールバックプロバイダー](../user-guide/features/fallback-providers.md)
- [AWS Bedrock](./aws-bedrock.md) — AWS認証情報を使用したネイティブなクラウドプロバイダー連携
