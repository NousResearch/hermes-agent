---
sidebar_position: 14
title: "APIサーバー"
description: "hermes-agentをOpenAI互換APIとして公開し、任意のフロントエンドから利用する"
---

# APIサーバー

APIサーバーは、hermes-agentをOpenAI互換のHTTPエンドポイントとして公開します。OpenAI形式を話せるフロントエンド — Open WebUI、LobeChat、LibreChat、NextChat、ChatBox、その他数百種類 — であれば、hermes-agentに接続してバックエンドとして利用できます。

エージェントは、フルツールセット（ターミナル、ファイル操作、ウェブ検索、メモリ、スキル）を使ってリクエストを処理し、最終的な応答を返します。ストリーミング時には、ツールの進捗インジケーターがインラインで表示されるため、フロントエンドはエージェントが何をしているかを表示できます。

## クイックスタート

### 1. APIサーバーを有効にする

`~/.hermes/.env` に追加します:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# 任意: ブラウザがHermesを直接呼び出す必要がある場合のみ
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
# curlでテスト
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

または、Open WebUI、LobeChat、その他のフロントエンドを接続します — 手順については [Open WebUI連携ガイド](/docs/user-guide/messaging/open-webui) を参照してください。

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

**ストリーム内のツール進捗**:
- **Chat Completions**: Hermesは、永続化されるアシスタントテキストを汚すことなくツール開始を可視化するため、`event: hermes.tool.progress` を発行します。
- **Responses**: Hermesは、SSEストリーム中に仕様準拠の `function_call` および `function_call_output` 出力アイテムを発行するため、クライアントは構造化されたツールUIをリアルタイムにレンダリングできます。

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
    {"type": "function_call", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

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

エージェントを利用可能なモデルとして一覧表示します。広告されるモデル名は、デフォルトで [プロファイル](/docs/user-guide/profiles) 名（デフォルトプロファイルの場合は `hermes-agent`）になります。ほとんどのフロントエンドがモデル検出のために必要とします。

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
    "run_stop": true
  }
}
```

ダッシュボード、ブラウザUI、コントロールプレーンを統合する際にこのエンドポイントを使うと、プライベートなPython内部に依存することなく、稼働中のHermesバージョンがruns、ストリーミング、キャンセル、セッション継続をサポートしているかどうかを検出できます。

### GET /health

ヘルスチェック。`{"status": "ok"}` を返します。`/v1/` プレフィックスを期待するOpenAI互換クライアント向けに **GET /v1/health** でも利用できます。

### GET /health/detailed

アクティブなセッション、実行中のエージェント、リソース使用状況も報告する拡張ヘルスチェックです。監視・可観測性ツールに役立ちます。

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
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

ステータスは、終端状態（`completed`、`failed`、`cancelled`）に達した後も、ポーリングやUIの整合性確認のために短時間保持されます。

### GET /v1/runs/\{run_id\}/events

実行のツール呼び出し進捗、トークンデルタ、ライフサイクルイベントのServer-Sent Eventsストリームです。状態を失わずにアタッチ・デタッチしたいダッシュボードやシッククライアント向けに設計されています。

### POST /v1/runs/\{run_id\}/stop

実行中のエージェントのターンを中断します。エンドポイントは即座に `{"status": "stopping"}` を返し、その間Hermesはアクティブなエージェントに次の安全な中断ポイントで停止するよう要求します。

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

:::warning Security
APIサーバーは、**ターミナルコマンドを含む** hermes-agentのツールセットへのフルアクセスを与えます。`0.0.0.0` のような非ループバックアドレスにバインドする場合、`API_SERVER_KEY` は **必須** です。また、ブラウザアクセスを制御するため `API_SERVER_CORS_ORIGINS` を狭く保ってください。

デフォルトのバインドアドレス（`127.0.0.1`）はローカル専用です。ブラウザアクセスはデフォルトで無効になっています。明示的に信頼できるオリジンに対してのみ有効にしてください。
:::

## 設定

### 環境変数

| 変数 | デフォルト | 説明 |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | APIサーバーを有効にする |
| `API_SERVER_PORT` | `8642` | HTTPサーバーのポート |
| `API_SERVER_HOST` | `127.0.0.1` | バインドアドレス（デフォルトはlocalhostのみ） |
| `API_SERVER_KEY` | _(なし)_ | 認証用のBearerトークン |
| `API_SERVER_CORS_ORIGINS` | _(なし)_ | カンマ区切りの許可するブラウザオリジン |
| `API_SERVER_MODEL_NAME` | _(プロファイル名)_ | `/v1/models` に表示するモデル名。デフォルトはプロファイル名、デフォルトプロファイルの場合は `hermes-agent`。 |

### config.yaml

```yaml
# まだ未サポート — 環境変数を使ってください。
# config.yamlのサポートは将来のリリースで対応予定です。
```

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
- **`Idempotency-Key`** は許可されるリクエストヘッダーです — クライアントは重複排除のために送信できます（応答はキーごとに5分間キャッシュされます）

Open WebUIなど、ドキュメント化されたほとんどのフロントエンドはサーバー間で接続するため、CORSはまったく必要ありません。

## 互換フロントエンド

OpenAI API形式をサポートする任意のフロントエンドが動作します。テスト済み／ドキュメント化された連携:

| フロントエンド | スター数 | 接続方法 |
|----------|-------|------------|
| [Open WebUI](/docs/user-guide/messaging/open-webui) | 126k | 完全ガイドあり |
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

複数のユーザーに、それぞれ独立したHermesインスタンス（個別の設定、メモリ、スキル）を与えるには、[プロファイル](/docs/user-guide/profiles) を使います:

```bash
# ユーザーごとにプロファイルを作成
hermes profile create alice
hermes profile create bob

# 各プロファイルのAPIサーバーを別々のポートで設定。API_SERVER_* は環境変数
# （config.yamlのキーではない）なので、各プロファイルの.envに書き込みます:
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

# 各プロファイルのゲートウェイを起動
hermes -p alice gateway &
hermes -p bob gateway &
```

各プロファイルのAPIサーバーは、自動的にプロファイル名をモデルIDとして広告します:

- `http://localhost:8643/v1/models` → モデル `alice`
- `http://localhost:8644/v1/models` → モデル `bob`

Open WebUIでは、それぞれを別々の接続として追加します。モデルのドロップダウンには `alice` と `bob` が別個のモデルとして表示され、それぞれが完全に独立したHermesインスタンスで動作します。詳細は [Open WebUIガイド](/docs/user-guide/messaging/open-webui#multi-user-setup-with-profiles) を参照してください。

## 制限事項

- **レスポンスの保存** — 保存されたレスポンス（`previous_response_id` 用）はSQLiteに永続化され、ゲートウェイの再起動後も残ります。保存できるレスポンスは最大100件（LRU方式で退避）。
- **ファイルアップロード非対応** — インライン画像は `/v1/chat/completions` と `/v1/responses` の両方でサポートされますが、アップロードされたファイル（`file`、`input_file`、`file_id`）や画像以外のドキュメント入力はAPI経由ではサポートされません。
- **modelフィールドは見た目だけ** — リクエスト内の `model` フィールドは受け付けられますが、実際に使われるLLMモデルはサーバー側のconfig.yamlで設定されます。

## プロキシモード

APIサーバーは、**ゲートウェイのプロキシモード** のバックエンドとしても機能します。別のHermesゲートウェイインスタンスが `GATEWAY_PROXY_URL` をこのAPIサーバーに向けて設定されている場合、そのインスタンスは自身のエージェントを実行する代わりに、すべてのメッセージをここへ転送します。これにより分割デプロイが可能になります — 例えば、Matrix E2EEを処理するDockerコンテナが、ホスト側のエージェントへ中継する構成です。

完全なセットアップガイドは [Matrixプロキシモード](/docs/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos) を参照してください。
