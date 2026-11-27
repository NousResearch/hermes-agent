---
sidebar_position: 14
title: "APIサーバー"
description: "hermes-agentをOpenAI互換APIとして公開し、任意のフロントエンドから利用する"
---

# APIサーバー

APIサーバーは、hermes-agentをOpenAI互換のHTTPエンドポイントとして公開します。OpenAI形式を話せるフロントエンド — Open WebUI、LobeChat、LibreChat、NextChat、ChatBox、その他数百種類 — であれば、hermes-agentに接続してバックエンドとして利用できます。

エージェントは、フルツールセット（ターミナル、ファイル操作、ウェブ検索、メモリ、スキル）を使ってリクエストを処理し、最終的な応答を返します。ストリーミング時には、ツールの進捗インジケーターがインラインで表示されるため、フロントエンドはエージェントが何をしているかを表示できます。

:::tip 1つのバックエンドでモデルとツールをカバー
APIサーバーを役立てるには、Hermes自体にプロバイダーとツールバックエンドが設定されている必要があります。[Nous Portal](./tool-gateway.md) のサブスクリプションは両方を担います — 300以上のモデルに加え、Tool Gateway経由のweb/画像/TTS/ブラウザです。APIサーバーを起動する前に一度 `hermes setup --portal` を実行すれば、Open WebUIやLobeChatのようなフロントエンドは、ツールを完備したバックエンドを得られます。
:::

## クイックスタート

### 1. APIサーバーを有効にする

`~/.hermes/.env` に追加します:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# Optional: only if a browser must call Hermes directly
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. ゲートウェイを起動する

```bash
hermes gateway
```

次のように表示されます:

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. フロントエンドを接続する

任意のOpenAI互換クライアントを `http://localhost:8642/v1` に向けます:

```bash
# Test with curl
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

または、Open WebUI、LobeChat、その他のフロントエンドを接続します — 手順については [Open WebUI連携ガイド](../messaging/open-webui.md) を参照してください。

## 会話と終端結果のセマンティクス

APIリクエストは、ゲートウェイの永続的な受け付けキューを使います。構造化された画像コンテンツは
実行を通じて構造化されたまま保たれ、ゲートウェイのスラッシュコマンドとして解釈されることはありません。

- `/v1/runs` と `/v1/responses` は、`X-Hermes-Session-Key` で宣言された会話を
  保持します。明示的なセッションやレスポンスチェーンが優先され、その会話の
  アイデンティティが別のヘッダーに移ることはありません。
- 空でない `/v1/runs` の `conversation_history` は、明示的な `session_id` があっても
  優先されます。それ以外の場合、セッションの継続では、先にキューに入った作業が終わった後、
  ターンの実行時に履歴を読み込みます。
- Runs のポーリングとSSEは `cancelled` や `failed` について一致します。Responses も同じ
  終端ステータスと `response.cancelled` / `response.failed` イベントを使い、診断テキストが
  ある場合も同様です。Chat Completions は、中断されたターンには `finish_reason: "cancelled"`、
  失敗には `"error"` を報告し、成功の `"stop"` は報告しません。
- イテレーション予算（`agent.max_turns`）を使い切ったターンも、エージェントが見つけたことの
  まとめで応答します。Chat Completions はそのテキストを
  `finish_reason: "length"` と `hermes.error_code: "iteration_budget_exhausted"` とともに返し、
  Responses は `status: "incomplete"` を報告します。その内容は、キャンセルではなく、途中で止まった
  タスクに対する利用可能な回答として扱ってください。
- セッションチャットの返信には `status` が含まれ、そのストリームは対応する終端の実行
  イベントを報告します。アシスタントの最終化フレームそれ自体は、実行が成功した証拠ではありません。

## 冪等性キー {#idempotency-keys}

`POST /v1/chat/completions` と `POST /v1/responses` は `Idempotency-Key` ヘッダーを受け付けます。
キーが何を保証するかは実行モードによって異なります:

- **正規モード**（ゲートウェイがセッションオーソリティを実行している場合）: キーは永続的な受け付けの
  アイデンティティです。そのキーを持つ最初のリクエストが受け付けられて1回だけ実行され、以後キーは
  その受け付けに属します。キーは認証されたAPIプロファイルと認証情報にスコープされるため、別のプロファイルや
  ローテーションされた `API_SERVER_KEY` が同じクライアントキーを再利用した場合は新しいリクエストになります。
  このセクションの残りはこのモードについて説明します。
- **レガシーモード**（セッションオーソリティなし）: 非ストリーミングのリクエストは、5分間の
  インメモリキャッシュで重複排除されます。ストリーミングのリクエストは重複排除されません。

`/v1/runs` のキーは独自の台帳と24時間の保持期間を持ちます（下記のRuns APIを参照）。

**完全一致のリトライと競合。** リトライが完全一致となるのは、同じアイデンティティフィールドと
同じ宣言されたターゲットでキーを繰り返した場合です。アイデンティティフィールドを変更したり、キーを
別のセッションに送ったりすると競合になります: `admission_conflict` で拒否され、
何も実行されません。拒否がどのように届くかはリクエストによって異なります:

- **Responses**（JSONまたはストリーミング）と **非ストリーミングのChat Completions:** HTTP 409
  とエラーコード `admission_conflict`。
- **ストリーミングのChat Completions:** ストリームはすでにHTTP 200で開いているため、
  拒否はその最後のチャンクとして、`finish_reason: "error"` と `error.message`
  `"admission_conflict"` を伴って届き、続いて `data: [DONE]` が送られます。ステータスではなく、そのチャンクを確認してください。

アイデンティティフィールドはエンドポイントによって異なります:

- **Responses:** `input`、`instructions`、`previous_response_id`、`conversation`、
  `conversation_history`、`model`、`provider`、`model_options`、`tools`、および
  送信された場合の `truncation`。
- **Chat Completions:** 最後のuserメッセージ、それ以前のメッセージ（
  `X-Hermes-Session-Id` が送られた場合は、履歴がセッションから取られるため除く）、
  システムプロンプト、そして要求されたモデル、プロバイダー、
  `model_options` などのターン設定。`tools` と `tool_choice` はこれに含まれません: それらだけを
  変更したリトライは、競合ではなく元の回答をリプレイします。

配信モードはどちらのエンドポイントでもアイデンティティに含まれません: リトライでは
JSONと `stream: true` を切り替えることができ、要求された形式で同じ回答を受け取ります。

**リプレイと重複防止は別物です。** 完全一致のリトライが元の回答をリプレイするのは、
サーバーがそのキーのリクエストアイデンティティを、リプレイ記録または回答を再構築できる
終端の受け付けとともにまだ保持している間だけです。リプレイができなくなっても、受け付けの
証拠が残っている間は、そのキーが再び実行されることはありません:
リトライは競合として拒否され、何も実行されません。Responses のリトライでは、代わりに
HTTP 409 `unknown_execution` が返ることがあります: ストアがキーの主張（たとえば古いリリースで
記録されたもの）を保持しているものの回答がないため、最初の試行が実行されたかどうかが
不明な状態です。これは新しいキーで作業を再送してよいという許可ではありません。最初の試行は
すでに実行されている可能性があります。

セッションが削除された後（`DELETE /api/sessions/{id}`）、その受け付けは墓標（tombstone）として
保持されます。削除から7日後に刈り取りの対象となり、刈り取りのパスが実行されたときにのみ
削除されます。それまでは、キーは以下のように振る舞います。

- **Chat Completions:** 完全一致のリトライは、セッションを再作成せずに元の回答をリプレイし、
  他のセッションに送られたキーは競合として拒否されます。
  墓標が刈り取られた後は、そのキーで有効な新しいターゲットに対する新しい作業を指定できます。
- **Responses:** リプレイにはレスポンスストアがキーのアイデンティティを保持していることも必要なため、
  7日より前に終わることがあります: 確定したアイデンティティは上限付きの数だけ保持され、
  `DELETE /v1/responses/{id}` は1つを即座に削除します。アイデンティティが失われたキーは、
  稼働中のセッションであっても 409 `admission_conflict` を返します。アイデンティティが
  失われ、墓標が刈り取られた後は、そのキーで新しい作業を指定できます。

:::note
デフォルトでは、レスポンスストアはプロファイルごとに直近に確定した1,000件のアイデンティティを
保持します（保存されたレスポンスのスロットごとに10件）。保留中または実行中のアイデンティティは数えられず、
決して破棄されません。またレスポンスを削除すると、そのアイデンティティはより早く削除されます。これは
現在のデフォルトであり、リプレイの保証ではありません。
:::

明示的な `X-Hermes-Session-Id` が再マッピングされることはありません。削除されたセッションのIDは、
刈り取り後も退役したままです。そこでアイデンティティがまだ保持されている完全一致のキー付きリトライは
リプレイされ、変更されたものは競合として拒否されます。新しいリクエストや、アイデンティティが刈り取られた
リトライなど、そこに保持されたアイデンティティのないリクエストは、
`not_found` で拒否されます。

`X-Hermes-Session-Id` のない Chat Completions のリクエストは、システムプロンプトと最初のuserメッセージから
セッションを導出します。削除されたセッションのIDが再利用されることはないため、削除されたものと同じように
始まる新しい会話には新しいセッションが割り当てられ、その後のターンは、再起動後も含めて
その同じセッションに届きます。削除された会話のターンに対する完全一致のキー付きリトライは
移動されず、上記のとおりリプレイされます。

## エンドポイント

### POST /v1/chat/completions

標準的なOpenAI Chat Completions形式です。ステートレスで — 完全な会話が各リクエストの `messages` 配列に含まれます。

**リクエスト:**
```json
{
  "model": "hermes-agent",
  "messages": [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "Write a fibonacci function"}
  ],
  "stream": false
}
```

**レスポンス:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "hermes-agent",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Here's a fibonacci function..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

**インライン画像入力:** userメッセージは `content` を `text` パートと `image_url` パートの配列として送信できます。リモートの `http(s)` URLと `data:image/...` URLの両方がサポートされます:

```json
{
  "model": "hermes-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}}
      ]
    }
  ]
}
```

アップロードされたファイル（`file` / `input_file` / `file_id`）や画像以外の `data:` URLは `400 unsupported_content_type` を返します。

**ストリーミング**（`"stream": true`）: トークンごとの応答チャンクをServer-Sent Events（SSE）で返します。**Chat Completions** の場合、ストリームは標準の `chat.completion.chunk` イベントに加え、ツール開始のUXのためにHermes独自の `hermes.tool.progress` イベントを使用します。**Responses** の場合、ストリームは `response.created`、`response.output_text.delta`、`response.output_item.added`、`response.output_item.done`、`response.completed` などのOpenAI Responsesイベントタイプを使用します。

すべてのSSEストリーム（Chat Completions、Responses、`/api/sessions/{id}/chat/stream`、`/v1/runs/{id}/events`）は、10秒間イベントが送られなかった場合に `: keepalive` コメント行を送出するため、長いツール呼び出しでもクライアントのアイドルタイムアウトに引っかかりません。標準のSSEクライアントはコメント行を無視します。カスタムパーサーは `:` で始まる行をスキップする必要があります。

**ストリーム内のツール進捗**:
- **Chat Completions**: Hermesは、永続化されるアシスタントテキストを汚すことなくツール開始を可視化するため、`event: hermes.tool.progress` を発行します。名前付きのSSEイベントで詰まる厳格なOpenAIクライアントでは、`gateway.platforms.api_server.tool_progress_events: false`（デフォルト `true`）でこれらのフレームを無効にできます。コンテンツのチャンクには影響しません。このオプトアウトは Chat Completions にのみ適用されます — `/v1/runs/{id}/events` は常にツールイベントを発行し、`/v1/capabilities` の `tool_progress_events` 機能が表しているのはそちらです。
- **Responses**: Hermesは、SSEストリーム中に仕様準拠の `function_call` および `function_call_output` 出力アイテムを発行するため、クライアントは構造化されたツールUIをリアルタイムにレンダリングできます。

**モデルの推論**（モデルが実際に推論を生成し、解決された `reasoning` 設定がそれを許可している場合にのみ発行されます。入力側でのオプトアウトは `model_options.reasoning.enabled: false` です）:
- **Chat Completions**: 推論のデルタは `choices[0].delta.reasoning_content` のチャンク（Open WebUI、opencode、Vercel AI SDK が思考ブロックとしてレンダリングする DeepSeek 形式のフィールド）として届き、回答のテキストは `delta.content` に残ります。
- **Responses**: 思考の各バーストは、仕様準拠の `reasoning` 出力アイテムです — `response.output_item.added`（`item.type: "reasoning"`）、`response.reasoning_summary_part.added`、`response.reasoning_summary_text.delta` … `response.reasoning_summary_text.done`、`response.reasoning_summary_part.done`、`response.output_item.done` — 次のメッセージや `function_call` アイテムが開く前に閉じられ、`response.completed` の出力に `{"id": "rs_…", "type": "reasoning", "status": "completed", "summary": [{"type": "summary_text", "text": "…"}]}` としてエコーされます。`sequence_number` は推論、テキスト、ツールのイベントをまたいで単調増加のままです。
- **非ストリーミング**: `/v1/chat/completions` はターンの推論を `choices[0].message.reasoning_content` で返します。`/v1/responses` は同じ `reasoning` 出力アイテムをメッセージ（およびそのステップの `function_call` アイテム）の前に返し、`GET /v1/responses/{id}` のリプレイでも同様です。
- 前のレスポンスの `output` リストを次の `input` としてそのまま返す（Responses SDK のクライアントが行うこと）のは問題ありません: `reasoning` アイテムは、空のuserターンとして解析されるのではなく、入力では無視されます。
- サポートは `GET /v1/capabilities` で `features.reasoning_streaming: true` として広告されます。

### POST /v1/responses

OpenAI Responses API形式です。`previous_response_id` を介したサーバーサイドの会話状態をサポートします — サーバーが（ツール呼び出しと結果を含む）完全な会話履歴を保存するため、クライアントが管理しなくてもマルチターンのコンテキストが保持されます。

**リクエスト:**
```json
{
  "model": "hermes-agent",
  "input": "What files are in my project?",
  "instructions": "You are a helpful coding assistant.",
  "store": true
}
```

**レスポンス:**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "hermes-agent",
  "output": [
    {"type": "function_call", "status": "completed", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "status": "completed", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

`output` 配列内のツール呼び出しは、Hermesエージェントによってすでにサーバー側で実行されたものです — 構造化されたツールUIのために `"status": "completed"` でリプレイされるのであり、クライアントが実行すべき保留中の呼び出しとして返されることは決してありません。

`"stream": true` の場合、ターン途中のアシスタントのコメンタリー（`openai-codex` バックエンドの `phase="commentary"` の進捗前置き、またはモデルがツール呼び出しとともに書くテキスト）は、`"phase": "commentary"` を持つ独立した完了済みの `message` 出力アイテムとして届きます（`response.output_item.added` + `response.output_item.done`。`response.completed` にも列挙されます）。最終回答のアイテムにマージされることは決してないため、クライアントはそれをライブの進捗としてレンダリングし、返信を組み立てるときはスキップできます。プライベートな推論がこのアイテムに届くことはありません。`display.interim_assistant_messages: false`（または `display.platforms.api_server` の上書き）を設定すると、すべてのAPIサーバーのサーフェスでこれが抑制されます。

**インライン画像入力:** `input[].content` には `input_text` パートと `input_image` パートを含められます。リモートURLと `data:image/...` URLの両方がサポートされます:

```json
{
  "model": "hermes-agent",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this screenshot."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K..."}
      ]
    }
  ]
}
```

アップロードされたファイル（`input_file` / `file_id`）や画像以外の `data:` URLは `400 unsupported_content_type` を返します。

#### previous_response_idによるマルチターン

レスポンスをチェーンして、ターンをまたいで（ツール呼び出しを含む）完全なコンテキストを維持します:

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

サーバーは、保存されたレスポンスチェーンから完全な会話を再構築します — 過去のすべてのツール呼び出しと結果が保持されます。チェーンされたリクエストは同じセッションを共有するため、マルチターンの会話はダッシュボードやセッション履歴上で単一のエントリとして表示されます。

各レスポンスの `output` には、そのターンのアイテム（その `function_call` / `function_call_output` エントリと最終的な `message`）だけが列挙され、以前のターンのツール呼び出しは決して含まれません — Hermesが呼び出しの前に提供された履歴を修復した場合（連続する `assistant` や `user` のアイテムをマージした、孤立したツール結果を削除した）や、チェーンの途中で圧縮した場合も同様です。保存されるチェーンはその修復済みのトランスクリプトなので、ターンごとに履歴が2重にコピーされて増えることはありません。

#### 名前付き会話

レスポンスIDを追跡する代わりに、`conversation` パラメーターを使います:

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

サーバーはその会話内の最新のレスポンスへ自動的にチェーンします。ゲートウェイセッションにおける `/title` コマンドのようなものです。

### GET /v1/responses/\{id\}

保存済みのレスポンスをIDで取得します。

### DELETE /v1/responses/\{id\}

保存済みのレスポンスを削除します。

### GET /v1/models

エージェントを利用可能なモデルとして一覧表示します。広告されるモデル名は、デフォルトで [プロファイル](../profiles.md) 名（デフォルトプロファイルの場合は `hermes-agent`）になります。ほとんどのフロントエンドがモデル検出のために必要とします。

`/v1/models` は、意図的に軽量なOpenAI互換のサーフェスになっています。Hermesがルーティングできる
認証済みのプロバイダー/モデルの組み合わせをすべて列挙することは**なく**、
価格や機能の情報付加も行いません。

### GET /api/model/options

Hermesを認識するクライアントは、ダッシュボードやTUIが使うのと同じ、キュレーションされたプロバイダー/モデルの
一覧を要求できます。このルートはAPIサーバーの通常のbearer
認証を使い、OpenAI互換の `/v1/models` のレスポンスには属さない、プロバイダーの行、モデルの機能のヒント、価格の
メタデータを返します:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options"
```

このペイロードは、ダッシュボードのModelsページやTUIの
`model.options` RPCが使うのと同じ基盤です。認証済みのプロバイダー、キュレーションされたモデルの
一覧、モデルごとの価格、モデルの機能のヒントを返します。

カスタムプロバイダーについては、通常のオープンは意図的に保守的です: Hermesは
**現在選択されている**カスタムエンドポイントだけをプローブするため、古くなったりオフラインになったりした保存済みの
エンドポイントがピッカーをブロックすることはありません。明示的なリフレッシュでは完全なプローブに切り替わり、
プロバイダーのモデルキャッシュが破棄されます:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options?refresh=1"
```

OpenAI互換のクライアントが、chat/responses のリクエストで送り返すモデル名だけを必要とする場合は
`/v1/models` を使います。認証されたUIが、よりリッチなHermes固有のピッカーのメタデータを必要とする場合は
`/api/model/options` を使います。

### GET /v1/capabilities

外部UI、オーケストレーター、プラグインブリッジ向けに、APIサーバーの安定した表面を機械可読形式で記述したものを返します。

```json
{
  "object": "hermes.api_server.capabilities",
  "platform": "hermes-agent",
  "model": "hermes-agent",
  "auth": {"type": "bearer", "required": true},
  "features": {
    "chat_completions": true,
    "responses_api": true,
    "run_submission": true,
    "run_status": true,
    "run_events_sse": true,
    "run_stop": true,
    "reasoning_streaming": true
  }
}
```

ダッシュボード、ブラウザUI、コントロールプレーンを統合する際にこのエンドポイントを使うと、プライベートなPython内部に依存することなく、稼働中のHermesバージョンがruns、ストリーミング、キャンセル、セッション継続をサポートしているかどうかを検出できます。

## ブラウザ拡張機能による制御（削除済み） {#browser-extension-control-removed}

ブラウザ拡張機能のコントローラープロトコル（`POST /v1/browser-control/register`、
`GET /v1/browser-control/ws`、`/v1/artifacts/*` のルート、ダッシュボードの
`browser.controller.*` RPC、`browser.extension_control` 設定キー）は
削除されました。これは古い `browser_navigate`/`browser_click`/… ツールをルーティングしていましたが、
それらはもう存在しません: ブラウザ自動化は現在、CDP上の `browser_exec` です。すでに実行している
ブラウザをHermesに操作させるには、`config.yaml` の `browser.cdp_url` または `/browser connect` で、
そのブラウザのDevToolsエンドポイントを指定してください —
[CDP経由で自分のブラウザを使う](./browser#your-own-browser-via-cdp-browser-connect)を参照してください。
アップグレードすると、`config.yaml` に残った `browser.extension_control` ブロックは削除されます。

## リクエストごとのモデル選択

認証されたクライアントは、次を送ることで、リクエストごとにHermesのデフォルトのモデル選択を
上書きできます:

- `model` — このターンの対象モデルID
- `provider` — このターンの認証情報/ランタイムを解決するためのHermesのプロバイダースラッグ
- `model_options` — リクエストにスコープされた推論 / サービスティアの制御

同じリクエストフィールドは次で受け付けられます:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/runs`
- `POST /api/sessions/{session_id}/chat`
- `POST /api/sessions/{session_id}/chat/stream`

優先順位は決定的です:

1. セッションの `/model` による上書き（そのセッションにすでにある場合）
2. リクエストの `model` が設定済みのルートエイリアスである場合に選択される、静的な
   `gateway.platforms.api_server.model_routes` のマッピング
3. ルートエイリアスに一致しない場合は、リクエストの `model` / `provider` を直接使用
4. ゲートウェイのグローバル設定 / 環境変数のデフォルト

`model_options` は、どのモデル/プロバイダーが選ばれてもリクエストにスコープされたままです。
リクエストが設定済みの `model_routes` エイリアスと矛盾する `provider` を送った場合、
Hermesはルートの認証情報を別のプロバイダーと黙って混ぜ合わせる代わりに、そのリクエストを `400` で
拒否します。

**OpenAI互換エンドポイントでの `model` の単独指定はオプトインです。** 汎用の
OpenAIクライアントはモデル名（`gpt-4o` など）をハードコードしていることが多く、既存の
デプロイメントはそれらがゲートウェイのデフォルトにフォールバックすることに依存しています。そのため
`POST /v1/chat/completions` と `POST /v1/responses` では、`provider` なしで送られた
`model` の値は、次を有効にしない限り無視されます:

```yaml
gateway:
  platforms:
    api_server:
      direct_model_requests: true
```

明示的な `provider` を含むリクエスト — およびHermesネイティブの
`/v1/runs` とセッションチャットのエンドポイント — は、このフラグにかかわらず、常に要求されたモデルに
従います。

例:

```json
{
  "model": "MiniMax-M3",
  "provider": "minimax",
  "model_options": {
    "reasoning_effort": "high",
    "service_tier": "priority"
  },
  "messages": [
    {"role": "user", "content": "Summarize the repo status."}
  ]
}
```

### GET /health

ヘルスチェック。`{"status": "ok"}` を返します。`/v1/` プレフィックスを期待するOpenAI互換クライアント向けに **GET /v1/health** でも利用できます。

### GET /health/detailed

監視やコントロールプレーン向けの、認証付きのレディネスチェックです。アクティブなプロファイルの設定、
状態データベース、設定済みのモデル、ディスク容量、ゲートウェイ/プラットフォームの状態、アクティブなAPIの実行、保留中のプロセスの
完了、アクティブな委譲について、範囲を限定したステータスを報告します。レスポンスが公開するのはステータスとカウントであり、
設定値、認証情報、パス、コマンド、キューのペイロード、生のエラーは含まれません。

公開の `/health` ルートは引き続き軽量な死活確認のプローブであり、レディネスチェックは
実行しません。レディネスの結果が劣化していてもHTTP 200が使われます。トップレベルの
`status` と `readiness.checks` フィールドを確認してください。

## Runs API（ストリーミングに適した代替手段）

`/v1/chat/completions` と `/v1/responses` に加えて、サーバーは **runs** APIを公開します。これは、クライアントがストリーミングを自身で管理する代わりに進捗イベントを購読したい、長時間のセッション向けです。

### POST /v1/runs

新しいエージェントの実行（run）を作成します。進捗イベントの購読に使える `run_id` を返します。

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

runsは、シンプルな `input` 文字列と、任意の `session_id`、`instructions`、`conversation_history`、`previous_response_id` を受け付けます。`session_id` が指定されると、Hermesはそれを実行ステータスに表出させるため、外部UIは自身の会話IDと実行を関連付けられます。

安全にリトライできる作成のためには、`Idempotency-Key` ヘッダー（1〜255文字の可視ASCII文字）を送ってください。Hermesは作業を開始する前にキーを永続的に予約します。同一のリトライは、ゲートウェイの再起動後や、実行が完了、失敗、キャンセルされた後も含めて、HTTP 202 と `Idempotency-Replayed: true` とともに元の `run_id` を返します。同じキーを異なるJSONペイロードで再利用すると、コード `idempotency_key_conflict` とともに HTTP 409 が返ります。キーは認証されたAPIプロファイル/認証情報ごとに分離され、最後のステータス更新から24時間保持されます。クライアントは一意で推測できないキーを使い、無関係な操作に再利用してはいけません。ヘッダーのないリクエストはレガシーの挙動を維持し、常に新しい実行を作成します。

ゲートウェイが受け付けられなかった実行は予約を保持しないため、完全一致のリトライは安全です。プロファイルが
ドレイン中または退役中の場合や、状態データベースが書き込みタイムアウトを過ぎてもロックされたままの場合、`POST /v1/runs`
と実行の制御は HTTP 503（`runtime_draining` または `storage_unavailable`）と `Retry-After` を返します。
不正な形式の制御ボディには 400 `invalid_params` が返り、所有権、世代、アイデンティティによる拒否は
409 のままです。

`session_id` が既存のHermesセッションを識別し、明示的な
`conversation_history` や `previous_response_id` が指定されていない場合、実行は
そのセッションのアクティブなトランスクリプトを読み込みます。セッションのターンリースは並行する
書き手を直列化し、競合して待機した後にトランスクリプトを更新します。

**APIセッションはそれ自体のセッションです。** `session_id`（ここ）と
`X-Hermes-Session-Id` ヘッダー（`/v1/chat/completions` と `/v1/responses`）は、
APIサーバーが作成したセッションか、作成し得るセッションを指定できます。別のサーフェス — CLI、TUI、Desktop、ACP、メッセージングの
会話 — が所有するセッションをバインドすることはできません。そのようなリクエストは、受け付けの前に HTTP 409 とエラーコード
`permission_denied` で拒否されます: 実行は作成されず、
`Idempotency-Key` の予約も保持されず、完全一致のリトライには再び 409 が返ります。これは、
APIサーバーが他者のトランスクリプト上で、その稼働中の所有者と並行して独自のエージェントを実行していた以前の挙動を
置き換えるものです。別のサーフェスのチャットで作業するには、OpenAI互換のルートではなく、
ゲートウェイのWebSocket経由でアタッチしてください（共有セッション上の `prompt.submit`）。

### GET /v1/runs/\{run_id\}

現在の実行状態をポーリングします。SSE接続を開いたままにせずにステータスを取得したいダッシュボードや、ナビゲーション後に再接続するUIに役立ちます。

```json
{
  "object": "hermes.run",
  "run_id": "run_abc123",
  "status": "completed",
  "session_id": "space-session",
  "model": "hermes-agent",
  "output": "Done.",
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250, "cache_read_tokens": 40, "cache_write_tokens": 0},
  "runtime": {"provider": "openai", "model": "gpt-5", "route_source": "global"}
}
```

`model` はリクエストが要求したものをエコーします。完了した実行では、`runtime` は実際にターンを処理したプロバイダー/モデルの組です — [フォールバックプロバイダー](fallback-providers.md)への切り替えの後はフォールバックの組を示すため、コスト配分のポーラーは実行を正しいプロバイダーに計上できます。`usage.cache_read_tokens` / `usage.cache_write_tokens` はセッションのプロンプトキャッシュの読み取りと書き込みで、キャッシュされた入力が定価の入力として価格付けされないようにします。`runtime` は `/v1/chat/completions` や `/v1/responses` と同じ形です: `route_source` はランタイムがどのように選ばれたか（`global`、`raw_request`、`model_routes`）を示し、`model`/`provider` を指定したリクエストには `requested: {provider, model}` も付くため、要求された組と処理した組を比較できます。イベントストリームの `run.completed` イベントも、同じ `usage` と `runtime` フィールドを持ちます。

ステータスは、終端状態（`completed`、`failed`、`cancelled`、`interrupted`）に達した後も、ポーリングやUIの整合性確認のために短時間保持されます。実行がアクティブな間にゲートウェイがシャットダウンすると、エージェントに停止を求める前に、実行は `interrupted`（エラー `Gateway shutdown interrupted the run.`、終端イベント `run.interrupted`）として永続化されるため、永続的な実行が `running` のまま再起動を越えることはありません。中断されたターンから遅れて届いた結果がそれを上書きすることもできません。所有者がターンの途中で再起動した正規の実行は、代わりに `unknown` を報告します。これは終端ではありません: そのセッションの以降のターンは、`POST /v1/runs/{run_id}/resolve-unknown`（下記）で確認応答されるまで待機します。

ゲートウェイがまだドレイン中の間（処理中のターンがある状態での `hermes gateway stop`/`restart` や SIGTERM）、終端でないすべての実行には、新しいターンが拒否された時点からの `shutdown_requested_at`（Unix秒）も付きます。ターンはまだ処理中なので `status` は `running` のままです。このフィールドを見たポーラーは、プロセスが終了に向かっており、遅くともドレインの予算が尽きた時点で実行が `interrupted` で終わることが分かります。終端の実行にこのフィールドが付くことはありません。

### GET /v1/runs/\{run_id\}/events

実行のツール呼び出し進捗、トークンデルタ、ライフサイクルイベントのServer-Sent Eventsストリームです。状態を失わずにアタッチ・デタッチしたいダッシュボードやシッククライアント向けに設計されています。

ターン途中のアシスタントのコメンタリー — `openai-codex` バックエンドの `phase="commentary"` の進捗
前置き、またはモデルがツール呼び出しとともに書くテキスト — は `message.interim`
（`text`、`already_streamed`）として届きます。これはTUIゲートウェイが使うのと同じ契約です。`already_streamed: true`
は、そのテキストが `message.delta` としても送出されたことを意味するため、デルタをレンダリングするクライアントはそれをスキップできます。
最終回答は引き続き `run.completed` でのみ届き、プライベートな推論が
`message.interim` になることは決してありません。ゲート: `display.interim_assistant_messages`（デフォルト `true`）。

ツールのライフサイクルイベントは、`tool.started`（`tool`、引数の `preview`）と
`tool.completed`（`tool`、秒単位の `duration`、`error`、結果の `preview`）を持ちます。
`error` フラグはツール自身の結果を反映します — 非ゼロの終端 `exit_code`、構造化された
`{"error": ...}` の結果、拒否された承認 — 結果がJSON文字列として届いても、
すでに解析されたオブジェクトとして届いても同様です。完了時の `preview` は結果のテキスト（構造化された結果は
JSONエンコードされます）で、強制的なシークレットの秘匿化を経た後に500文字に切り詰められるため、
クライアントは上限のないツールのペイロードを受け取ることなく、承認の拒否（`BLOCKED: ...`）と通常の失敗を
区別できます。

エージェントがバックグラウンドのサブエージェントに作業を委譲すると、ストリームには
`subagent.start` と `subagent.complete` のライフサイクルイベントも流れるため、クライアントは
子が作業している間に実行が沈黙してしまうのではなく、委譲の結果 — タイムアウトや失敗を含む — を
観測できます。`subagent.complete` のペイロードには、
子のステータス、要約、所要時間、トークン/コストの数値、関連付けのための
`child_session_id`、そして所属するバッチの `delegation_id`
が含まれます（そのため並行またはネストしたファンアウトも区別できます）。自由記述のフィールドは、プロセスを出る前に強制的なシークレットの
秘匿化を経ます。ツールごとの子のイベント
（`subagent.tool`、進捗のティック）は意図的に転送**されません** — それらは
大量のUIノイズです。逐一の経過には、子ごとのライブトランスクリプトファイルを
使ってください。これらのイベントは親のストリームが開いている間に利用できます。
遅れて届いたデタッチされた完了が、終了した実行のSSEストリームを再び開いたり、その終端ステータスを
変更したりすることはありません。

#### デタッチされた結果とセッション履歴

バックグラウンドでの委譲には、サーバー側のセッション
履歴を読み込む継続が必要です: Chat Completions での明示的な `X-Hermes-Session-Id`、ネイティブの
`/api/sessions/{id}/chat` リクエスト、またはセッション履歴を使う Runs のリクエストです。
ヘッダーのない Chat Completions、Responses のチェーン、および
`previous_response_id` や呼び出し側が提供した履歴を伴う Runs のリクエストは、代わりに委譲を
同期的に実行し、元のターンで結果を返します。リクエストの内容からセッションIDを
導出するだけでは、デタッチされた配信は有効になりません。

再開可能なリクエストでは、完了は委譲の単位ごとに1回永続化されます。
それは `GET /api/sessions/{id}/messages` と、次の実際の
クライアントのターンのセッション履歴で利用できます。リトライで同じ結果が再び挿入されることはなく、
途中のタスク失敗の通知は別のアイデンティティを持ちます。配信は、クライアントのターンが
セッションのリースを保持している間は待機し、圧縮による継続に追従します。
Chat Completions は、あなたが指定した明示的なセッションIDをJSONと
ストリーミングの両方のレスポンスでエコーします。圧縮の後も、そのIDを送り続けてください。

完了によって **頼まれていないモデルのターンが開始されることは決してなく**、保留中の
人間による確認がバイパスされることもありません。次のターンはクライアントが所有します。独自の履歴スナップショットを
使い続けるクライアントは、サーバー側の配信行がそれらのスナップショットにマージされることを期待するのではなく、
同期的な委譲を使ってください。

消費されていないイベントバッファは5分後に期限切れになるため、デタッチされたクライアントが
メモリを無制限に増やすことはありません。期限切れになるのはトランスポートの状態だけです: まだ
実行中の実行は、エグゼキューターの作業が実際に終了するまで、ステータスのポーリング、承認、停止の制御、
並行数の計算で引き続き見えます。接続しているSSEの
購読者は通常どおりドレインを続けます。

### POST /v1/runs/\{run_id\}/resolve-unknown

オーソリティの所有者が再起動した後、すでに要求されていたターンは
実行結果が不明になり、`GET /v1/runs/{run_id}` はそれを
`"status": "unknown"`（終端ではない）として報告します。Hermesは、失われた先頭をリプレイする危険を冒すのではなく、
そのセッションの以降の作業を一時停止します。そのボディから正確な `admission_id` と
`execution_generation` を読み取り、失われた実行が終わらないことを
確認応答してください:

```json
{
  "admission_id": "adm_abc123",
  "execution_generation": 7
}
```

このリクエストは、他の実行の制御と同様に認証され、実行の所有者にスコープされます。
ホストされたルームの呼び出し元には、既存の `stop` 権限が必要です。成功したレスポンスは
正規の終端レシート（`outcome: "interrupted"`）と `run_id` です。これは
FIFOを1回解放し、不明な先頭をリプレイせずにキューに入った後続をスケジュールします。
すでにコミットされた解決を完全に繰り返すと同じレシート（200）が返るため、レスポンスが失われた後のリトライは安全です。
古いまたは整数でない世代、あるいは現在不明でない実行には `409 stale_generation` が返り、
パスの実行に属さない受け付けIDには `409 not_found` が返ります。
ボディにはちょうどこの2つのフィールドが含まれている必要があり、そうでなければサーバーは `400 invalid_params` を返します。

ホストされたルームのホームは、メンバーの `unknown` の実行を次のポーリングでこの方法で解決し、
ルームのタスクを中断（"member gateway restarted"）として確定させます。古いリリースのルームの
ホームは `unknown` ステータスを知らないため、アップグレードされるまで実行ステータスの不一致を
報告します。

この回復用の制御は、正規のセッションオーソリティがアクティブな場合にのみ、`features.run_unknown_resolution` および
`run_unknown_resolution` エンドポイントとして広告されます。レガシーのAPI実行モードでは
広告されません。通常の
`POST /v1/runs/{run_id}/stop` は意図的に別のままであり、不明な受け付けに対しては
`409 unknown_execution` を返します。

このバージョンで作成された受け付けでは、不透明な実行の所有者のスコープが
正規の受け付けとアトミックに保存され、終端状態を含め、その受け付けの
制御/ステータスの有効期間中保持されます。これはサーバーのプライベートな状態であり、
bearerの認証情報、モデルへの入力、APIレスポンスのフィールドではありません。キーのないリクエストは
冪等ではないままなので、一致するリクエストボディでも別々の実行が作成されます。
キーのない古い受け付けには永続化された所有者のスコープがなく、アダプターの再起動後は引き続き
フェイルクローズします。それらを安全に後から埋めることはできません。キー付きの古い
実行は、引き続き既存の冪等性の台帳を使います。明示的なセッションの
退役は、正規の受け付けのペイロードと、その回復可能な所有者を削除します。
リプレイの台帳と正規の受け付けは別々のデータベーストランザクションです。
この回復用の制御は、グローバルな厳密に1回の実行を約束するものではありません。

### POST /v1/runs/\{run_id\}/stop

実行中のエージェントのターンを中断します。エンドポイントは即座に `{"status": "stopping"}` を返し、その間Hermesはアクティブなエージェントに次の安全な中断ポイントで停止するよう要求します。
実行は、エグゼキューターによる作業が終了するまで `stopping` として追跡され、その後
`cancelled` として確定します。停止を要求しても、まだ実行中のワーカーが
隠れることはありません。

### POST /v1/runs/\{run_id\}/approval

人間の判断を待っている実行（たとえば、承認ポリシーの背後でゲートされたツール呼び出し）の保留中の承認を解決します。ボディには承認の判断を含めます。判断が記録されると実行が再開します。このエンドポイントは `/v1/capabilities` で `run_approval` 機能として広告されるため、外部UIは承認プロンプトを表示する前にサポートを検出できます。

正規のゲートウェイの所有者がいる場合、`GET /v1/runs/{run_id}` は
`pending_controls` を公開し、アタッチしたWebSocketの閲覧者と同じ `prompt_id` と `execution_generation` を
使います。正確な現在のアイデンティティで応答してください:

```json
{"request_id": "<prompt_id>", "execution_generation": 3, "choice": "once"}
```

正規の制御は、アイデンティティのない応答や一括の応答を受け付けません。古い世代、
他者のプロンプトID、確定済みの実行には HTTP 409 が返ります。既存の実行の認証と
ホストされたルームの承認の制限は引き続き適用されます。

### POST /v1/runs/\{run_id\}/clarify

`request_id`、`execution_generation`、`answer`（文字列）で正規の確認質問に回答します。
この認証付きのエンドポイントと WebSocket の `clarify.respond` は、
待機している同じツールを解決します。回答しても、別の推論ターンが送信されることはありません。
MCPの信頼ゲートの同意 — `trust: untrusted` に設定されたサーバー上の書き込み可能なツール — も同じように表面化します: 実行は `approval.request` イベントを発行し、このエンドポイントがそれを解決するまで `waiting_for_approval` で待機します（`once` はツールを実行し、`deny` はブロックします）。

## Jobs API（バックグラウンドのスケジュール作業）

サーバーは、リモートクライアントからスケジュール済み／バックグラウンドのエージェント実行を管理するための、軽量なジョブCRUD表面を公開します。すべてのエンドポイントは、同じbearer認証の背後でゲートされます。

### GET /api/jobs

スケジュール済みジョブをすべて一覧表示します。

### POST /api/jobs

新しいスケジュールジョブを作成します。ボディは `hermes cron` と同じ形式を受け付けます — プロンプト、スケジュール、スキル、プロバイダーのオーバーライド、配信先。

### GET /api/jobs/\{job_id\}

単一ジョブの定義と直近の実行状態を取得します。

### PATCH /api/jobs/\{job_id\}

既存ジョブのフィールド（プロンプト、スケジュールなど）を更新します。部分更新はマージされます。

### DELETE /api/jobs/\{job_id\}

ジョブを削除します。実行中の実行もキャンセルします。

### POST /api/jobs/\{job_id\}/pause

ジョブを削除せずに一時停止します。次回スケジュール実行のタイムスタンプは、再開されるまで保留されます。

### POST /api/jobs/\{job_id\}/resume

一時停止したジョブを再開します。

### POST /api/jobs/\{job_id\}/run

スケジュール外で、ジョブを即座に実行させます。

## Sessions API（REST経由のセッション制御）

外部UIは、ダッシュボードを立ち上げることなく、REST経由でHermesのセッションを管理できます。すべてのエンドポイントは `API_SERVER_KEY` でゲートされ、`/api/sessions/*` の下にあります。

| メソッド | パス | 説明 |
|--------|------|-------------|
| `GET` | `/api/sessions` | セッションを一覧表示（ページ分割 — `limit`、`offset`、`source`、`include_children`） |
| `POST` | `/api/sessions` | 空のセッションを作成 |
| `GET` | `/api/sessions/{id}` | セッションのメタデータを読み取る |
| `PATCH` | `/api/sessions/{id}` | タイトルまたは `end_reason` を更新 |
| `DELETE` | `/api/sessions/{id}` | セッションを削除 |
| `GET` | `/api/sessions/{id}/messages` | セッションのメッセージ履歴。`include_compacted=true` を指定すると、コンテキストの圧縮によってアーカイブされたターンも返します（デフォルト: ライブのトランスクリプトのみ）。`inline_images=false` を指定すると、画像の添付をインラインのデータURIではなく `[image]` のプレースホルダーとしてレンダリングします — ネットワーク越しに読むクライアント向けに、トランスクリプトがキロバイト単位で済みます |
| `POST` | `/api/sessions/{id}/fork` | `SessionDB` の系譜を使ってセッションを分岐（CLIの `/branch` のセマンティクスと一致） |
| `POST` | `/api/sessions/{id}/chat` | 同期的なエージェントのターンを1回実行 |
| `POST` | `/api/sessions/{id}/chat/stream` | 1ターンに対するSSEラッパー — `assistant.delta`、`assistant.commentary`（ターン途中のコメンタリー: `message_id`、`text`、`already_streamed`。`assistant.completed` に畳み込まれることはない）、`tool.started`、`tool.completed`、`tool.failed`（エラーで終わったツール）を発行し、最後にターンの終わり方に一致する終端の `run.completed` / `run.failed` / `run.cancelled` イベントを発行します（[終端の実行ステータス](/docs/developer-guide/programmatic-integration#terminal-run-status)を参照） |

`/v1/capabilities` は、`session_*` 機能フラグと `endpoints.session_*` エントリでサーフェス全体を広告するため、外部UIはサポートを検出して安全にフォールバックできます。インライン画像は `chat` と `chat/stream` のペイロードでサポートされます（マルチモーダル対応の経路）。

```bash
# fork a session and run one turn
curl -X POST http://localhost:8642/api/sessions/$ID/fork \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"title": "explore alt path"}'

# stream a turn over SSE
curl -N -X POST http://localhost:8642/api/sessions/$ID/chat/stream \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"input": "what files changed in the last hour?"}'
```

## スキルとツールセットの検出

`GET /v1/skills` と `GET /v1/toolsets` を使うと、外部クライアントはモデルに尋ねる代わりに、REST経由でエージェントの機能を決定的に列挙できます。どちらも読み取り専用で、`API_SERVER_KEY` でゲートされます。

```bash
curl http://localhost:8642/v1/skills \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "github-pr-workflow", "description": "...", "category": "..."}, ...]

curl http://localhost:8642/v1/toolsets \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "core", "label": "...", "description": "...", "enabled": true,
#     "configured": true, "tools": ["read_file", "write_file", ...]}, ...]
```

`/v1/skills` は、スキルハブが内部で使うのと同じメタデータを返します。`/v1/toolsets` は、`api_server` プラットフォーム向けに解決されたツールセットを、それぞれが展開される具体的な `tools` の一覧とともに返します。どちらも `/v1/capabilities` の `endpoints.*` の下で広告されます。

## 長期メモリのスコープ（`X-Hermes-Session-Key`）

Open WebUIのようなマルチユーザーのフロントエンドには、トランスクリプトにスコープされた `X-Hermes-Session-Id`（`/new` でローテーションする）とは**独立した**、長期メモリ（Honchoなど）用の安定したチャネルごとの識別子が必要です。`/v1/chat/completions`、`/v1/responses`、`/v1/runs` で `X-Hermes-Session-Key` を渡すと、Hermesはそれを `AIAgent(gateway_session_key=...)` まで通し、Honchoのメモリプロバイダーはそれを使って安定したスコープを導出します。

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer ***
X-Hermes-Session-Id: transcript-alpha
X-Hermes-Session-Key: agent:main:webui:dm:user-42
```

ルール: 最大256文字、制御文字（`\r`、`\n`、`\x00`）は拒否され、値はレスポンス（JSON + SSE）でエコーされます。`/v1/capabilities` は `"session_key_header": "X-Hermes-Session-Key"` でサポートを広告します。キーがない場合、Honchoの `per-session` 戦略は `session_id` ごとに異なるスコープを生成します — まさにHermesが以前持っていた挙動です。

自動的な想起は**トランスクリプト**に従います: メモリプロバイダーはセッションごとに1回初期化され、リクエストをまたいで保持されるため、継続されたセッション（`X-Hermes-Session-Id`、`previous_response_id`、または宣言された `X-Hermes-Session-Key` の会話）は、TelegramやDiscordのチャットとまったく同じように、プロバイダーが前のターンの後に準備した想起を受け取ります。継続のないリクエストは新しいセッションを開始し、新しいCLIセッションの最初のターンと同様に、まだ何もキューに入っていません。アイドル状態のセッションは、ゲートウェイのエージェントキャッシュと同じアイドルTTL（`agent.agent_cache.idle_ttl_secs`、デフォルトは1時間）の後にプロバイダーを解放します。

## システムプロンプトの扱い

フロントエンドが `system` メッセージ（Chat Completions）または `instructions` フィールド（Responses API）を送信すると、hermes-agentはそれをコアのシステムプロンプトの **上に重ねます**。エージェントはすべてのツール、メモリ、スキルを維持し — フロントエンドのシステムプロンプトは追加の指示を加えるだけです。

つまり、機能を失うことなく、フロントエンドごとに挙動をカスタマイズできます:
- Open WebUIのシステムプロンプト: 「You are a Python expert. Always include type hints.」
- エージェントは依然としてターミナル、ファイルツール、ウェブ検索、メモリなどを持つ

## 認証

`Authorization` ヘッダーによるBearerトークン認証:

```
Authorization: Bearer ***
```

キーは `API_SERVER_KEY` 環境変数で設定します。ブラウザからHermesを直接呼び出す必要がある場合は、`API_SERVER_CORS_ORIGINS` に明示的な許可リストも設定してください。

### マルチプロファイルのルーティング（`/p/<profile>/…`）

[マルチプロファイルのゲートウェイルーティング](/docs/user-guide/multi-profile-gateways)が
有効な場合（`gateway.multiplex_profiles`）、共有リスナーは `/p/<profile>/` のURLプレフィックスを通じて
すべてのプロファイルを提供します — そして**認証はルーティング先のプロファイルに
結び付けられます**:

- `/p/<profile>/v1/...` へのリクエストは、そのプロファイル自身の
  `API_SERVER_KEY`（`~/.hermes/profiles/<profile>/.env` から）を提示する必要があります。デフォルトの
  リスナーのキーは、名前付きプロファイルのプレフィックスでは拒否されます。
- プレフィックスのないルートと `/p/default/...` は、引き続きデフォルトプロファイルのキーを使います。
- 独自の `API_SERVER_KEY` を持たない名前付きプロファイルはフェイルクローズします — その
  プレフィックスは、キーを設定するまで到達できません。
- 実行はプロファイルごとにスコープされます: `/v1/runs/{run_id}` とその `events`、`stop`、
  `steer`、`approval` のルートは、その実行を作成したプロファイルに対してのみ応答します
  （`/api/sessions/{id}/chat/stream` で開始された実行を含む）。
  別のプロファイルの実行IDには `404` が返り、`403` が返ることはありません。

:::warning 破壊的変更（2026年7月）
この修正の前は、有効なデフォルトプロファイルのキーが、任意の
`/p/<profile>/` プレフィックスで受け付けられていました。プロファイルのプレフィックスをまたいで1つの共有キーに
依存していた場合は、各プロファイルの `.env` に別々の `API_SERVER_KEY` を設定してください — 名前付きプレフィックスで
再利用されたデフォルトのキーは、現在 `401` を返します。
:::

:::warning Security
APIサーバーは、**ターミナルコマンドを含む** hermes-agentのツールセットへのフルアクセスを与えます。`API_SERVER_KEY` は、`127.0.0.1` へのデフォルトのループバックバインドを含め、**すべてのデプロイメントで必須** です。ブラウザからの呼び出しを明示的に許可する場合は、ブラウザアクセスを制御するため `API_SERVER_CORS_ORIGINS` を狭く保ってください。
:::

## 設定

### 環境変数

| 変数 | デフォルト | 説明 |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | APIサーバーを有効にする |
| `API_SERVER_PORT` | `8642` | HTTPサーバーのポート |
| `API_SERVER_HOST` | `127.0.0.1` | バインドアドレス（デフォルトはlocalhostのみ） |
| `API_SERVER_KEY` | _(必須)_ | 認証用のBearerトークン |
| `API_SERVER_CORS_ORIGINS` | _(なし)_ | カンマ区切りの許可するブラウザオリジン |
| `API_SERVER_MODEL_NAME` | _(プロファイル名)_ | `/v1/models` に表示するモデル名。デフォルトはプロファイル名、デフォルトプロファイルの場合は `hermes-agent`。 |

### config.yaml

同じ設定は、`~/.hermes/config.yaml` のネストした `gateway.api_server:` セクションに置くこともできます:

```yaml
gateway:
  api_server:
    enabled: true
    port: 8642
    host: 127.0.0.1
    key: your-secret-key
    cors_origins: http://localhost:3000
    model_name: my-hermes
    max_concurrent_runs: 10   # concurrent-run cap; 0 disables the limit
    history_tool_output_max_chars: 0   # cap tool outputs in stored /v1/responses history; 0 = verbatim
```

`port`、`key`、`host`、`cors_origins`、`model_name` は自動的にプラットフォームの `extra` 設定に橋渡しされるため、対応する `API_SERVER_*` 環境変数とまったく同じように振る舞います。環境変数は `config.yaml` の値より優先されます。このブロックは、`gateway.platforms.api_server:` またはトップレベルの `platforms.api_server:` セクションの下でも受け付けられます。

### 並行実行数の上限

APIサーバーは、エージェントの実行を直接開始するエンドポイント — OpenAI互換のエンドポイント、Runs のエンドポイント、セッションチャットのエンドポイント（`POST /api/sessions/{id}/chat` とその `/stream` 版。マシン間のエージェントDMを運びます）— をまたいで、同時に実行できるエージェントの実行数を制限します。cron によってトリガーされる実行（`POST /api/jobs/{id}/run`、`POST /api/cron/fire`）は cron スケジューラーを経由し、この上限ではなく cron 自身の制限に従います。上限は `gateway.api_server.max_concurrent_runs` から読み込まれます（デフォルト **10**。`0` で制限を無効化し、負の値は 0 に丸められます）。上限に達すると、実行を開始する新しいリクエストは **HTTP 429** `Too many concurrent runs (max N)` で拒否されます — クライアントはバックオフしてリトライしてください。

### `previous_response_id` チェーン用に保存される履歴のサイズ

保存される各 `/v1/responses` のスナップショットには、累積した会話履歴全体（`previous_response_id` と `conversation` のチェーンがリプレイするもの）が、すべてのツール出力をそのまま含めて埋め込まれます。そのため、大きなツール出力がいくつかある会話では、1回の `response_store.db` への書き込みが数百KBになることがあります。`gateway.api_server.history_tool_output_max_chars`（デフォルト **0** = そのまま保存）を設定すると、**保存される**履歴内の各ツール出力と、各文字列のツール呼び出し引数をその文字数で上限設定します。それより長いものは、先頭部分と `...[N more chars]` マーカーに切り詰められます。ユーザーとアシスタントのテキストには決して手を加えず、`response.completed` のペイロードと逐次のSSEイベントにも影響しません。保存された履歴は次のチェーンされたターンでモデルが目にするものなので、上限を有効にするとモデルにリプレイされる内容も削られます — ワークフローがターンをまたいで完全なツール出力を必要とする場合は 0 のままにしてください。

## セキュリティヘッダー

すべてのレスポンスにセキュリティヘッダーが含まれます:
- `X-Content-Type-Options: nosniff` — MIMEタイプのスニッフィングを防ぐ
- `Referrer-Policy: no-referrer` — リファラーの漏洩を防ぐ

## CORS

APIサーバーは、デフォルトではブラウザCORSを **有効にしません**。

ブラウザからの直接アクセスには、明示的な許可リストを設定します:

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

CORSが有効な場合:
- **プリフライト応答** には `Access-Control-Max-Age: 600`（10分のキャッシュ）が含まれます
- **SSEストリーミング応答** にはCORSヘッダーが含まれるため、ブラウザのEventSourceクライアントが正しく動作します
- **`X-Hermes-Session-Id`** は許可されるリクエストヘッダーなので、許可リストにあるオリジンのブラウザはセッションの継続を要求できます。
- **`Idempotency-Key`** は許可されるリクエストヘッダーです — クライアントは重複排除のために送信できます（キーが何をカバーし、どのくらいの期間保持されるかについては[冪等性キー](#idempotency-keys)を参照してください）

Open WebUIなど、ドキュメント化されたほとんどのフロントエンドはサーバー間で接続するため、CORSはまったく必要ありません。

## 互換フロントエンド

OpenAI API形式をサポートする任意のフロントエンドが動作します。テスト済み／ドキュメント化された連携:

| フロントエンド | スター数 | 接続方法 |
|----------|-------|------------|
| [Open WebUI](../messaging/open-webui.md) | 126k | 完全ガイドあり |
| LobeChat | 73k | カスタムプロバイダーのエンドポイント |
| LibreChat | 34k | librechat.yaml内のカスタムエンドポイント |
| AnythingLLM | 56k | 汎用OpenAIプロバイダー |
| NextChat | 87k | BASE_URL環境変数 |
| ChatBox | 39k | API Host設定 |
| Jan | 26k | リモートモデル設定 |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | カスタムエンドポイント |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | 直接HTTPリクエスト |

## プロファイルによるマルチユーザー設定

複数のユーザーに、それぞれ独立したHermesインスタンス（個別の設定、メモリ、スキル）を与えるには、[プロファイル](../profiles.md) を使います:

```bash
# Create a profile per user
hermes profile create alice
hermes profile create bob

# Configure each profile's API server on a different port. API_SERVER_* are env
# vars (not config.yaml keys), so write them to each profile's .env:
cat >> ~/.hermes/profiles/alice/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
API_SERVER_KEY=alice-secret
EOF

cat >> ~/.hermes/profiles/bob/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8644
API_SERVER_KEY=bob-secret
EOF

# Start each profile's gateway
hermes -p alice gateway &
hermes -p bob gateway &
```

各プロファイルのAPIサーバーは、自動的にプロファイル名をモデルIDとして広告します:

- `http://localhost:8643/v1/models` → モデル `alice`
- `http://localhost:8644/v1/models` → モデル `bob`

Open WebUIでは、それぞれを別々の接続として追加します。モデルのドロップダウンには `alice` と `bob` が別個のモデルとして表示され、それぞれが完全に独立したHermesインスタンスで動作します。詳細は [Open WebUIガイド](../messaging/open-webui.md#multi-user-setup-with-profiles) を参照してください。

## 制限事項

- **レスポンスの保存** — 保存されたレスポンス（`previous_response_id` 用）はSQLiteに永続化され、ゲートウェイの再起動後も残ります。保存できるレスポンスは最大100件（LRU方式で退避）。
- **ファイルアップロード非対応** — インライン画像は `/v1/chat/completions` と `/v1/responses` の両方でサポートされますが、アップロードされたファイル（`file`、`input_file`、`file_id`）や画像以外のドキュメント入力はAPI経由ではサポートされません。
- **シンプルなOpenAIクライアントには引き続きエイリアスが見えます** — `/v1/models` は
  安定したHermesのエイリアス（`hermes-agent` またはアクティブなプロファイル名）を広告します。よりリッチな
  クライアントは、リクエストで明示的な `provider` / `model_options` の上書きを送れます。

## プロキシモード

APIサーバーは、**ゲートウェイのプロキシモード** のバックエンドとしても機能します。別のHermesゲートウェイインスタンスが `GATEWAY_PROXY_URL` をこのAPIサーバーに向けて設定されている場合、そのインスタンスは自身のエージェントを実行する代わりに、すべてのメッセージをここへ転送します。これにより分割デプロイが可能になります — 例えば、Matrix E2EEを処理するDockerコンテナが、ホスト側のエージェントへ中継する構成です。

完全なセットアップガイドは [Matrixプロキシモード](../messaging/matrix.md#proxy-mode-e2ee-on-macos) を参照してください。
