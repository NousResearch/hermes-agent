---
sidebar_position: 13
title: "Webhook"
description: "GitHub、GitLab、その他のサービスからイベントを受信して Hermes エージェントの実行をトリガーする"
---

# Webhook

外部サービス（GitHub、GitLab、JIRA、Stripe など）からイベントを受信し、Hermes エージェントの実行を自動的にトリガーします。Webhook アダプターは、POST リクエストを受け付け、HMAC 署名を検証し、ペイロードをエージェントのプロンプトに変換し、応答を送信元または設定済みの別のプラットフォームにルーティングする HTTP サーバーを実行します。

エージェントはイベントを処理し、PR にコメントを投稿したり、Telegram/Discord にメッセージを送信したり、結果をログに記録したりして応答できます。

## 動画チュートリアル

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/WNYe5mD4fY8"
    title="Hermes Agent — Webhooks Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

---

## クイックスタート

1. `hermes gateway setup` または環境変数で有効化する
2. `config.yaml` でルートを定義する **または** `hermes webhook subscribe` で動的に作成する
3. サービスを `http://your-server:8644/webhooks/<route-name>` に向ける

---

## セットアップ

Webhook アダプターを有効化する方法は 2 つあります。

### セットアップウィザード経由

```bash
hermes gateway setup
```

プロンプトに従って Webhook を有効化し、ポートを設定し、グローバルな HMAC シークレットを設定します。

### 環境変数経由

`~/.hermes/.env` に追加します:

```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644        # デフォルト
WEBHOOK_SECRET=your-global-secret
```

### サーバーの確認

ゲートウェイが実行されたら:

```bash
curl http://localhost:8644/health
```

期待される応答:

```json
{"status": "ok", "platform": "webhook"}
```

---

## ルートの設定 {#configuring-routes}

ルートは、異なる Webhook 送信元をどう処理するかを定義します。各ルートは、`config.yaml` の `platforms.webhook.extra.routes` の下にある名前付きエントリです。

### ルートのプロパティ

| プロパティ | 必須 | 説明 |
|----------|----------|-------------|
| `events` | 不要 | 受け付けるイベントタイプのリスト（例: `["pull_request"]`）。空の場合、すべてのイベントが受け付けられます。イベントタイプは `X-GitHub-Event`、`X-GitLab-Event`、またはペイロード内の `event_type` から読み取られます。 |
| `secret` | **必須** | 署名検証用の HMAC シークレット。ルートに設定されていない場合はグローバルの `secret` にフォールバックします。テスト目的のみ `"INSECURE_NO_AUTH"` に設定可能です（検証をスキップ）。 |
| `prompt` | 不要 | ドット記法によるペイロードアクセスを使うテンプレート文字列（例: `{pull_request.title}`）。省略すると、JSON ペイロード全体がプロンプトにダンプされます。 |
| `skills` | 不要 | エージェント実行時にロードするスキル名のリスト。 |
| `deliver` | 不要 | 応答の送信先: `github_comment`、`telegram`、`discord`、`slack`、`signal`、`sms`、`whatsapp`、`matrix`、`mattermost`、`homeassistant`、`email`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`、または `log`（デフォルト）。 |
| `deliver_extra` | 不要 | 追加の配信設定 — キーは `deliver` のタイプに依存します（例: `repo`、`pr_number`、`chat_id`）。値は `prompt` と同じ `{dot.notation}` テンプレートをサポートします。 |
| `deliver_only` | 不要 | `true` の場合、エージェントを完全にスキップします — レンダリングされた `prompt` テンプレートが、そのまま配信されるリテラルなメッセージになります。LLM コストゼロ、サブ秒の配信。ユースケースについては[直接配信モード](#direct-delivery-mode)を参照してください。`deliver` が実際のターゲット（`log` 以外）である必要があります。 |

### 完全な例

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-fallback-secret"
      routes:
        github-pr:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          prompt: |
            Review this pull request:
            Repository: {repository.full_name}
            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            URL: {pull_request.html_url}
            Diff URL: {pull_request.diff_url}
            Action: {action}
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
        deploy-notify:
          events: ["push"]
          secret: "deploy-secret"
          prompt: "New push to {repository.full_name} branch {ref}: {head_commit.message}"
          deliver: "telegram"
```

### プロンプトテンプレート

プロンプトはドット記法を使って、Webhook ペイロード内のネストされたフィールドにアクセスします:

- `{pull_request.title}` は `payload["pull_request"]["title"]` に解決されます
- `{repository.full_name}` は `payload["repository"]["full_name"]` に解決されます
- `{__raw__}` — **ペイロード全体**をインデント付き JSON としてダンプする特別なトークン（4000 文字で切り詰め）。エージェントが完全なコンテキストを必要とする監視アラートや汎用 Webhook に便利です。
- 存在しないキーはリテラルな `{key}` 文字列のまま残されます（エラーにはなりません）
- ネストされた dict や list は JSON シリアライズされ、2000 文字で切り詰められます

`{__raw__}` を通常のテンプレート変数と組み合わせることもできます:

```yaml
prompt: "PR #{pull_request.number} by {pull_request.user.login}: {__raw__}"
```

ルートに `prompt` テンプレートが設定されていない場合、ペイロード全体がインデント付き JSON としてダンプされます（4000 文字で切り詰め）。

同じドット記法のテンプレートは `deliver_extra` の値でも機能します。

### フォーラムトピックへの配信

Webhook の応答を Telegram に配信する際、`deliver_extra` に `message_thread_id`（または `thread_id`）を含めることで、特定のフォーラムトピックを指定できます:

```yaml
webhooks:
  routes:
    alerts:
      events: ["alert"]
      prompt: "Alert: {__raw__}"
      deliver: "telegram"
      deliver_extra:
        chat_id: "-1001234567890"
        message_thread_id: "42"
```

`deliver_extra` に `chat_id` が指定されていない場合、配信はターゲットプラットフォームに設定されたホームチャンネルにフォールバックします。

---

## GitHub PR レビュー（ステップバイステップ） {#github-pr-review}

この手順では、すべてのプルリクエストで自動コードレビューを設定します。

### 1. GitHub で Webhook を作成する

1. リポジトリ → **Settings** → **Webhooks** → **Add webhook** に移動する
2. **Payload URL** を `http://your-server:8644/webhooks/github-pr` に設定する
3. **Content type** を `application/json` に設定する
4. **Secret** をルート設定（例: `github-webhook-secret`）と一致するように設定する
5. **Which events?** で **Let me select individual events** を選び、**Pull requests** にチェックを入れる
6. **Add webhook** をクリックする

### 2. ルート設定を追加する

上記の例のとおり、`github-pr` ルートを `~/.hermes/config.yaml` に追加します。

### 3. `gh` CLI が認証済みであることを確認する

`github_comment` 配信タイプは、コメントの投稿に GitHub CLI を使います:

```bash
gh auth login
```

### 4. テストする

リポジトリでプルリクエストを開きます。Webhook が発火し、Hermes がイベントを処理し、PR にレビューコメントを投稿します。

---

## GitLab Webhook のセットアップ {#gitlab-webhook-setup}

GitLab の Webhook も同様に動作しますが、異なる認証メカニズムを使います。GitLab はシークレットをプレーンな `X-Gitlab-Token` ヘッダーとして送信します（HMAC ではなく完全一致）。

### 1. GitLab で Webhook を作成する

1. プロジェクト → **Settings** → **Webhooks** に移動する
2. **URL** を `http://your-server:8644/webhooks/gitlab-mr` に設定する
3. **Secret token** を入力する
4. **Merge request events**（および必要な他のイベント）を選択する
5. **Add webhook** をクリックする

### 2. ルート設定を追加する

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        gitlab-mr:
          events: ["merge_request"]
          secret: "your-gitlab-secret-token"
          prompt: |
            Review this merge request:
            Project: {project.path_with_namespace}
            MR !{object_attributes.iid}: {object_attributes.title}
            Author: {object_attributes.last_commit.author.name}
            URL: {object_attributes.url}
            Action: {object_attributes.action}
          deliver: "log"
```

---

## 配信オプション {#delivery-options}

`deliver` フィールドは、Webhook イベントの処理後にエージェントの応答がどこに送られるかを制御します。

| 配信タイプ | 説明 |
|-------------|-------------|
| `log` | 応答をゲートウェイのログ出力に記録します。これはデフォルトであり、テストに便利です。 |
| `github_comment` | `gh` CLI を介して応答を PR/issue のコメントとして投稿します。`deliver_extra.repo` と `deliver_extra.pr_number` が必要です。`gh` CLI がゲートウェイホストにインストールされ、認証済み（`gh auth login`）である必要があります。 |
| `telegram` | 応答を Telegram にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `discord` | 応答を Discord にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `slack` | 応答を Slack にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `signal` | 応答を Signal にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `sms` | 応答を Twilio 経由で SMS にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `whatsapp` | 応答を WhatsApp にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `matrix` | 応答を Matrix にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `mattermost` | 応答を Mattermost にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `homeassistant` | 応答を Home Assistant にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `email` | 応答を Email にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `dingtalk` | 応答を DingTalk にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `feishu` | 応答を Feishu/Lark にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `wecom` | 応答を WeCom にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `weixin` | 応答を Weixin（WeChat）にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `bluebubbles` | 応答を BlueBubbles（iMessage）にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |

クロスプラットフォーム配信の場合、ターゲットプラットフォームもゲートウェイで有効化され接続されている必要があります。`deliver_extra` に `chat_id` が指定されていない場合、応答はそのプラットフォームに設定されたホームチャンネルに送信されます。

---

## 直接配信モード {#direct-delivery-mode}

デフォルトでは、すべての Webhook POST がエージェントの実行をトリガーします — ペイロードがプロンプトになり、エージェントがそれを処理し、エージェントの応答が配信されます。これはイベントごとに LLM トークンを消費します。

**プレーンな通知をプッシュしたいだけ** のユースケース — 推論なし、エージェントループなし、メッセージを配信するだけ — の場合は、ルートに `deliver_only: true` を設定します。レンダリングされた `prompt` テンプレートがリテラルなメッセージ本文になり、アダプターがそれを設定された配信ターゲットに直接ディスパッチします。

### 直接配信を使うべきとき

- **外部サービスのプッシュ** — Supabase/Firebase の Webhook がデータベース変更で発火 → Telegram でユーザーに即座に通知
- **監視アラート** — Datadog/Grafana のアラート Webhook → Discord チャンネルにプッシュ
- **エージェント間 ping** — エージェント A がエージェント B のユーザーに、長時間タスクが完了したことを通知
- **バックグラウンドジョブの完了** — Cron ジョブが完了 → 結果を Slack に投稿

メリット:

- **LLM トークンゼロ** — エージェントは一切呼び出されません
- **サブ秒の配信** — 単一のアダプター呼び出し、推論ループなし
- **エージェントモードと同じセキュリティ** — HMAC 認証、レート制限、冪等性、ボディサイズ制限がすべて引き続き適用されます
- **同期応答** — POST は配信が成功すると `200 OK` を返し、ターゲットが拒否した場合は `502` を返すため、上流サービスがインテリジェントに再試行できます

### 例: Supabase からの Telegram プッシュ

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-secret"
      routes:
        antenna-matches:
          secret: "antenna-webhook-secret"
          deliver: "telegram"
          deliver_only: true
          prompt: "🎉 New match: {match.user_name} matched with you!"
          deliver_extra:
            chat_id: "{match.telegram_chat_id}"
```

Supabase の edge function はペイロードに HMAC-SHA256 で署名し、`https://your-server:8644/webhooks/antenna-matches` に POST します。Webhook アダプターは署名を検証し、ペイロードからテンプレートをレンダリングし、Telegram に配信し、`200 OK` を返します。

### 例: CLI による動的サブスクリプション

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 New match: {match.user_name} matched with you!" \
  --description "Antenna match notifications"
```

### レスポンスコード

| ステータス | 意味 |
|--------|---------|
| `200 OK` | 配信成功。ボディ: `{"status": "delivered", "route": "...", "target": "...", "delivery_id": "..."}` |
| `200 OK`（status=duplicate） | 冪等性 TTL（1 時間）内での `X-GitHub-Delivery` ID の重複。再配信されません。 |
| `401 Unauthorized` | HMAC 署名が無効または欠落しています。 |
| `400 Bad Request` | 不正な形式の JSON ボディ。 |
| `404 Not Found` | 不明なルート名。 |
| `413 Payload Too Large` | ボディが `max_body_bytes` を超えました。 |
| `429 Too Many Requests` | ルートのレート制限を超過しました。 |
| `502 Bad Gateway` | ターゲットアダプターがメッセージを拒否または例外を発生させました。エラーはサーバー側でログに記録されます。アダプターの内部情報の漏洩を避けるため、レスポンスボディは汎用的な `Delivery failed` になります。 |

### 設定上の注意点

- `deliver_only: true` には `deliver` が実際のターゲットである必要があります。`deliver: log`（または `deliver` の省略）は起動時に拒否されます — 設定ミスのルートが見つかると、アダプターは起動を拒否します。
- `skills` フィールドは直接配信モードでは無視されます（エージェントが実行されないため、スキルを注入する対象がありません）。
- テンプレートのレンダリングは、エージェントモードと同じ `{dot.notation}` 構文を使い、`{__raw__}` トークンも含みます。
- 冪等性は同じ `X-GitHub-Delivery` / `X-Request-ID` ヘッダーを使います — 同じ ID での再試行は `status=duplicate` を返し、再配信は **行いません**。

---

## 動的サブスクリプション（CLI） {#dynamic-subscriptions}

`config.yaml` の静的ルートに加えて、`hermes webhook` CLI コマンドを使って Webhook サブスクリプションを動的に作成できます。これは、エージェント自身がイベント駆動のトリガーをセットアップする必要がある場合に特に便利です。

### サブスクリプションを作成する

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New issue #{issue.number}: {issue.title}\nBy: {issue.user.login}\n\n{issue.body}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "Triage new GitHub issues"
```

これにより、Webhook URL と自動生成された HMAC シークレットが返されます。その URL に POST するようサービスを設定してください。

### サブスクリプションを一覧表示する

```bash
hermes webhook list
```

### サブスクリプションを削除する

```bash
hermes webhook remove github-issues
```

### サブスクリプションをテストする

```bash
hermes webhook test github-issues
hermes webhook test github-issues --payload '{"issue": {"number": 42, "title": "Test"}}'
```

### 動的サブスクリプションの仕組み

- サブスクリプションは `~/.hermes/webhook_subscriptions.json` に保存されます
- Webhook アダプターは、受信リクエストごとにこのファイルをホットリロードします（mtime ゲート付き、オーバーヘッドは無視できる程度）
- `config.yaml` の静的ルートは、同名の動的ルートよりも常に優先されます
- 動的サブスクリプションは、静的ルートと同じルート形式と機能（events、プロンプトテンプレート、スキル、配信）を使います
- ゲートウェイの再起動は不要 — サブスクライブすればすぐに有効になります

### エージェント駆動のサブスクリプション

エージェントは、`webhook-subscriptions` スキルにガイドされたときに、ターミナルツールを介してサブスクリプションを作成できます。エージェントに「GitHub issue 用の Webhook をセットアップして」と頼むと、適切な `hermes webhook subscribe` コマンドを実行します。

---

## セキュリティ {#security}

Webhook アダプターには複数層のセキュリティが含まれています:

### HMAC 署名検証

アダプターは、各送信元に適した方法で受信 Webhook の署名を検証します:

- **GitHub**: `X-Hub-Signature-256` ヘッダー — `sha256=` がプレフィックスされた HMAC-SHA256 16 進ダイジェスト
- **GitLab**: `X-Gitlab-Token` ヘッダー — プレーンなシークレット文字列の一致
- **汎用**: `X-Webhook-Signature` ヘッダー — 生の HMAC-SHA256 16 進ダイジェスト

シークレットが設定されているのに認識可能な署名ヘッダーが存在しない場合、リクエストは拒否されます。

### シークレットは必須

すべてのルートにはシークレットが必要です — ルートに直接設定するか、グローバルの `secret` から継承します。シークレットのないルートは、アダプターが起動時にエラーで失敗する原因になります。開発/テスト目的のみ、シークレットを `"INSECURE_NO_AUTH"` に設定して検証を完全にスキップできます。

`INSECURE_NO_AUTH` は、ゲートウェイがループバックホスト（`127.0.0.1`、`localhost`、`::1`）にバインドされている場合にのみ受け付けられます。`0.0.0.0` や LAN IP などの非ループバックバインドと組み合わされた場合、アダプターは起動を拒否します — これは、未認証のエンドポイントを公開インターフェースに誤って晒すことを防ぎます。

### レート制限

各ルートはデフォルトで **1 分あたり 30 リクエスト**にレート制限されます（固定ウィンドウ）。これをグローバルに設定します:

```yaml
platforms:
  webhook:
    extra:
      rate_limit: 60  # 1 分あたりのリクエスト数
```

制限を超えるリクエストは `429 Too Many Requests` 応答を受け取ります。

### 冪等性

配信 ID（`X-GitHub-Delivery`、`X-Request-ID`、またはタイムスタンプのフォールバックから取得）は **1 時間** キャッシュされます。重複配信（例: Webhook の再試行）は `200` 応答で静かにスキップされ、エージェントの重複実行を防ぎます。

### ボディサイズ制限

**1 MB** を超えるペイロードは、ボディが読み取られる前に拒否されます。これを設定します:

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### プロンプトインジェクションのリスク

:::warning
Webhook ペイロードには攻撃者が制御できるデータが含まれます — PR タイトル、コミットメッセージ、issue の説明などにはすべて悪意ある命令が含まれる可能性があります。インターネットに公開する場合は、サンドボックス環境（Docker、VM）でゲートウェイを実行してください。隔離のために Docker または SSH ターミナルバックエンドの使用を検討してください。
:::

---

## トラブルシューティング {#troubleshooting}

### Webhook が届かない

- ポートが公開され、Webhook 送信元からアクセス可能であることを確認する
- ファイアウォールルールを確認する — ポート `8644`（または設定したポート）が開いている必要があります
- URL パスが一致するか確認する: `http://your-server:8644/webhooks/<route-name>`
- `/health` エンドポイントを使ってサーバーが実行中であることを確認する

### 署名検証が失敗する

- ルート設定のシークレットが、Webhook 送信元に設定されたシークレットと完全に一致することを確認する
- GitHub の場合、シークレットは HMAC ベースです — `X-Hub-Signature-256` を確認する
- GitLab の場合、シークレットはプレーンなトークンの一致です — `X-Gitlab-Token` を確認する
- ゲートウェイログで `Invalid signature` の警告を確認する

### イベントが無視される

- イベントタイプがルートの `events` リストに含まれているか確認する
- GitHub イベントは `pull_request`、`push`、`issues`（`X-GitHub-Event` ヘッダーの値）のような値を使います
- GitLab イベントは `merge_request`、`push`（`X-GitLab-Event` ヘッダーの値）のような値を使います
- `events` が空または未設定の場合、すべてのイベントが受け付けられます

### エージェントが応答しない

- ログを見るためにゲートウェイをフォアグラウンドで実行する: `hermes gateway run`
- プロンプトテンプレートが正しくレンダリングされているか確認する
- 配信ターゲットが設定され接続されているか確認する

### 重複した応答

- 冪等性キャッシュがこれを防ぐはずです — Webhook 送信元が配信 ID ヘッダー（`X-GitHub-Delivery` または `X-Request-ID`）を送信しているか確認する
- 配信 ID は 1 時間キャッシュされます

### `gh` CLI エラー（GitHub コメント配信）

- ゲートウェイホストで `gh auth login` を実行する
- 認証された GitHub ユーザーがリポジトリへの書き込みアクセス権を持っていることを確認する
- `gh` がインストールされ、PATH 上にあることを確認する

---

## 環境変数 {#environment-variables}

| 変数 | 説明 | デフォルト |
|----------|-------------|---------|
| `WEBHOOK_ENABLED` | Webhook プラットフォームアダプターを有効化する | `false` |
| `WEBHOOK_PORT` | Webhook を受信する HTTP サーバーのポート | `8644` |
| `WEBHOOK_SECRET` | グローバル HMAC シークレット（ルートが独自に指定しない場合のフォールバックとして使用） | _(なし)_ |
