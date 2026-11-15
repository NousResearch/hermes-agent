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
WEBHOOK_PORT=8644        # default
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

ルートは、異なる Webhook 送信元をどう処理するかを定義します。各ルートは、`config.yaml` の `platforms.webhook.extra.routes` の下にある名前付きエントリです。アダプターの設定（`port`、`host`、`secret`、`routes`）は `platforms.webhook:` の直下に書くこともできます — どちらの書き方でもアダプターに届きます。同じキーが両方に現れた場合は、`extra:` の下にネストされた値が優先されます。

### ルートのプロパティ

| プロパティ | 必須 | 説明 |
|----------|----------|-------------|
| `events` | 不要 | 受け付けるイベントタイプのリスト（例: `["pull_request"]`）。空の場合、すべてのイベントが受け付けられます。イベントタイプは `X-GitHub-Event`、`X-GitLab-Event`、またはペイロード内の `event_type` から読み取られます。 |
| `secret` | **必須** | 署名検証用の HMAC シークレット。ルートに設定されていない場合はグローバルの `secret` にフォールバックします。テスト目的のみ `"INSECURE_NO_AUTH"` に設定可能です（検証をスキップ）。 |
| `signature` | 不要 | ルートを 1 つの署名方式に固定します: `generic`（`X-Webhook-Signature-V2` + `X-Webhook-Timestamp`、または Standard Webhooks / Svix ヘッダー）、`github`（`X-Hub-Signature-256`）、`gitlab`（`X-Gitlab-Token`、または Standard Webhooks ヘッダーとして送信される GitLab の署名トークン）、`linear`（`Linear-Signature`）。それ以外のヘッダーファミリーはすべて `401` になります。省略すると、サポートされるすべての方式を受け付けます（このキーが存在する以前の動作）。`hermes webhook subscribe` は新しいルートに `signature: generic` を書き込みます。推奨設定です。[署名方式の固定](#pin-the-signature-scheme)を参照してください。 |
| `profile` | 不要 | `gateway.multiplex_profiles` が有効な場合に、このルートの実行を許可されるプロファイル。省略するとデフォルトプロファイル専用のルートになります。プロファイル名（例: `coder`）を設定すると、ルートとそのシークレットが `/p/coder/webhooks/<route>` にバインドされます。動的サブスクリプションでは `hermes webhook subscribe <name> --route-profile coder` で設定します。 |
| `prompt` | 不要 | ドット記法によるペイロードアクセスを使うテンプレート文字列（例: `{pull_request.title}`）。省略すると、JSON ペイロード全体がプロンプトにダンプされます。ペイロードのフィールドは信頼できません — [認証済みは信頼できることを意味しない](#authenticated-does-not-mean-trusted)を参照してください。 |
| `filters` | 不要 | 認証/ボディ/イベントのフィルタリングの後、エージェントまたは直接配信の処理の前に評価される宣言的なペイロードフィルター。一致しない場合は HTTP 200 で `{"status":"ignored","reason":"filter"}` を返します。 |
| `script` | 不要 | `~/.hermes/scripts/` 配下のフィルター/変換スクリプト。Webhook ペイロードは JSON として stdin に渡されます。stdout が JSON オブジェクトの場合はテンプレート処理の前にペイロードを置き換えます。テキストの stdout は `script_output` として公開されます。stdout が空、`[SILENT]`、または終了コードが 0 以外の場合、Webhook は無視されます。 |
| `skills` | 不要 | エージェント実行時にロードするスキル名のリスト。 |
| `toolsets` | 不要 | このルートでトリガーされた実行に限り、プラットフォームレベルの Webhook ツールセットを **置き換える** ツールセットキーのリスト（例: `["terminal", "file", "web"]`）。設定ファイルの手動編集でのみ設定可能で、`hermes webhook subscribe` では設定できません。そのため、エージェントが作成したサブスクリプションが自身に昇格したツールを付与することはできません。名前は `platform_toolsets` のエントリと同じ方法で検証されます（不明な名前やプラットフォーム制限のある名前は除外されます）。[ルートごとのツールセット](#per-route-toolsets)を参照してください。 |
| `deliver` | 不要 | 応答の送信先: `github_comment`、`telegram`、`discord`、`slack`、`signal`、`sms`、`whatsapp`、`matrix`、`mattermost`、`homeassistant`（プラグイン）、`email`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`、または `log`（デフォルト）。 |
| `deliver_extra` | 不要 | 追加の配信設定 — キーは `deliver` のタイプに依存します（例: `repo`、`pr_number`、`chat_id`）。値は `prompt` と同じ `{dot.notation}` テンプレートをサポートします。 |
| `deliver_only` | 不要 | `true` の場合、エージェントを完全にスキップします — レンダリングされた `prompt` テンプレートが、そのまま配信されるリテラルなメッセージになります。LLM コストゼロ、サブ秒の配信。ユースケースについては[直接配信モード](#direct-delivery-mode)を参照してください。`deliver` が実際のターゲット（`log` 以外）である必要があります。 |
| `cron_job` | 不要 | 新しい Webhook エージェントセッションを開始する代わりに、イベントごとに既存の cron ジョブ（ID または名前で指定）を発火します。レンダリングされた `prompt` は実行ごとの一時的なコンテキストになり、ジョブ自身のプロンプト、スキル、モデル、配信設定が適用されます。`deliver_only` とは併用できません。[イベントトリガーの cron ジョブ](#event-triggered-cron-jobs)を参照してください。 |
| `coalesce` | 不要 | 同じ論理エンティティに対する立て続けの異なるイベントを、1 回のエージェント実行にまとめます（デバウンス）。必須の `key`（エンティティを識別するペイロードフィールドまたはテンプレート、例: `pull_request.number`）、任意の `window_seconds`（静止ウィンドウ、デフォルト 30）と `max_wait_seconds`（ディスパッチの上限、デフォルト 300）を持つブロックです。[イベントの集約](#event-coalescing)を参照してください。`deliver_only` および `cron_job` とは併用できません。 |
| `mirror_to_session` | 不要 | デフォルトは `false`。`true` の場合、チャットプラットフォームへの配信が成功した後、配信されたメッセージがそのチャットのセッションのトランスクリプトにも書き込まれます（継続可能な cron ブリーフと同様に、ラベル付きのユーザーターンとして）。そのため、そのチャットで返信すると、エージェントは直前に何を送ったかを把握しています。[配信への返信](#replying-to-a-delivery)を参照してください。 |

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
          signature: github
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
          filters:
            - field: "ref"
              equals: "refs/heads/main"
          deliver: "telegram"
```

### ペイロードフィルター

プロバイダーが広範なイベントストリームを送信するものの、一部のペイロードだけでエージェントを起動したり `deliver_only` の配信をトリガーしたりしたい場合は、`filters` を使います。フィルターは署名検証、ボディの解析、`events` の後、プロンプトのレンダリング、冪等性、エージェントのディスパッチ、直接配信の前に実行されます。

```yaml
platforms:
  webhook:
    extra:
      routes:
        todoist:
          events: ["item:updated"]
          secret: "todoist-secret"
          filters:
            - field: "payload.labels"
              contains: "hermes"
            - any:
                - field: "payload.priority"
                  equals: 4
                - field: "payload.project_id"
                  in_file: "~/.hermes/data/todoist/watchlist.json"
          prompt: "Todoist task changed: {payload.content}"
```

サポートされる演算子:

- `exists: true|false`
- `missing: true`
- `equals` / `not_equals`
- `contains`（文字列、リスト、dict のキーに対応）
- `in`（インラインのリスト）
- `in_file`（JSON 配列、JSON オブジェクト（キーが使われます）、または改行区切りのテキストファイル）
- `regex`
- `all`、`any`、`not` のグループ

フィールドパスはドット記法を使います。`payload.foo` は、トップレベルの `payload` オブジェクトが存在する場合はそこから、フラットなペイロードの場合は Webhook ボディのルートから読み取ります。`event` / `event_type` は解決されたイベントタイプに一致し、`headers.<Name>` はリクエストヘッダーを読み取ります。

### イベントの集約 {#event-coalescing}

プロバイダーは、同じ論理エンティティに対して複数の異なるイベントを立て続けに発火することがよくあります — 1 つのプルリクエストへの 5 回の連続 push、1 つのチケットへの一連の編集、フラッピングする監視アラートなどです。各イベントは新しい配信 ID を持つため、冪等性キャッシュではこれらを抑制できず、イベントごとに別々のエージェント実行が起動されます。

ルートに `coalesce` を設定すると、これらをエンティティごとに 1 回の実行にまとめられます:

```yaml
platforms:
  webhook:
    extra:
      routes:
        github-pr:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          signature: github
          coalesce:
            key: "{repository.full_name}#{pull_request.number}"
            window_seconds: 30      # quiet window (default 30)
            max_wait_seconds: 300   # dispatch cap (default 300)
          prompt: "Review PR #{pull_request.number}: {pull_request.title}"
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{pull_request.number}"
```

仕組み:

- イベントは、レンダリングされた `key` によってルートごとにグループ化されます。ドット区切りのフィールドのみ（`pull_request.number`）でも、完全なテンプレート（`{repository.full_name}#{pull_request.number}`）でも機能します。
- 新しいイベントはそれぞれ保留中のイベントを **置き換え**、静止ウィンドウのタイマーを延長します。新しいイベントがないまま `window_seconds` が経過すると、グループは最新のイベントのペイロード、プロンプト、配信テンプレートを使って **1 回** のエージェント実行をディスパッチします。
- `max_wait_seconds` はグループの最初のイベントからのバッファリング時間の合計に上限を設けるため、イベントが絶え間なく届いてもディスパッチが永遠に先延ばしされることはありません。
- 複数のイベントが集約された場合、プロンプトには、それ以前のイベントがいくつ置き換えられたかをエージェントに伝える短い注記が追加されます。
- あるイベントで `key` が解決できない場合（ペイロードにそのフィールドがない場合 — 例: `pull_request.number` をキーとするルートでの `issue_comment` イベント）、そのイベントは集約されずに **即座にディスパッチ** されるため、無関係なエンティティが 1 つのグループにまとめられることはありません。ルートが受け付けるすべてのイベントタイプに存在するキーを選んでください。
- 集約されたリクエストは `{"status": "coalesced"}` とともに HTTP 202 を返します。配信 ID による冪等性チェックは引き続き先に実行されるため、同じ配信に対するプロバイダーの再試行はカウントされずに破棄されます。
- 保留中のグループは、アダプターの切断時（ゲートウェイの再接続または `hermes gateway stop`）に破棄されずにフラッシュ（即座にディスパッチ）されます。バッファされたイベントはメモリ上にあるため、プロセスが強制終了された場合に失われるのは、最大でも現在のウィンドウでバッファされた分だけです。
- `coalesce` はエージェントモードのルートにのみ適用されます。`deliver_only` または `cron_job` との組み合わせは起動時に拒否されます。

### スクリプトによるフィルターと変換 {#script-filters-and-transforms}

宣言的なフィルターでは不十分な場合は `script` を使います。スクリプトはアクティブなプロファイルの `~/.hermes/scripts/` 配下に置く必要があります。相対パスはそこを基準に解決され、そのディレクトリ外へのパストラバーサルはブロックされます。`.sh` および `.bash` スクリプトは bash で実行され、それ以外の拡張子はすべて現在の Python インタープリターで実行されます。

ルートのペイロードは JSON として stdin に送られます:

```python
# ~/.hermes/scripts/todoist-hermes-label.py
import json
import sys

payload = json.load(sys.stdin)
labels = payload.get("payload", {}).get("labels", [])
if "hermes" not in labels:
    print("[SILENT]")
    raise SystemExit(0)

payload["body"] = payload["payload"]["content"]
print(json.dumps(payload))
```

スクリプトの結果:

- stdout が JSON オブジェクトの場合、`prompt` と `deliver_extra` で使われるペイロードを置き換えます。
- JSON 以外のテキストの stdout は、`script_output` としてペイロードに追加されます。
- stdout が空、`[SILENT]` と完全一致、`{"__hermes_ignore__": true}`、タイムアウト、スクリプトが存在しない、または終了コードが 0 以外の場合は、HTTP 200 で `{"status":"ignored","reason":"script"}` を返します。

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
          signature: gitlab
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
| `homeassistant` | 応答を Home Assistant にルーティングします（[`homeassistant` プラグイン](homeassistant.md)が必要です）。ホームチャンネル（`HASS_HOME_CHANNEL`）を使うか、`deliver_extra` で `chat_id` を指定します。 |
| `email` | 応答を Email にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `dingtalk` | 応答を DingTalk にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `feishu` | 応答を Feishu/Lark にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `wecom` | 応答を WeCom にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `weixin` | 応答を Weixin（WeChat）にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |
| `bluebubbles` | 応答を BlueBubbles（iMessage）にルーティングします。ホームチャンネルを使うか、`deliver_extra` で `chat_id` を指定します。 |

クロスプラットフォーム配信の場合、ターゲットプラットフォームもゲートウェイで有効化され接続されている必要があります。`deliver_extra` に `chat_id` が指定されていない場合、応答はそのプラットフォームに設定されたホームチャンネルに送信されます。

### 配信への返信 {#replying-to-a-delivery}

デフォルトでは、配信は送りっぱなしです。各 Webhook イベントはそれぞれ独自のセッションで実行されるため、配信されたメッセージにそのチャットで返信しても、そこにいるエージェントは何が送られたかの記録を持っていません。ルートに `mirror_to_session: true` を設定する（または `hermes webhook subscribe` に `--mirror-to-session` を渡す）と、配信されたテキストが `[Webhook delivery: <route>]` に続けてメッセージという形で、ターゲットのチャットのセッションにも追記されます。そのため、続けての質問（「じゃあ彼は欠席？」など）でもコンテキストが保たれます。

- ミラーはベストエフォートです。配信を失敗させることはなく、チャットにまだゲートウェイセッションがない場合（そこで誰もエージェントと話していない場合）はスキップされます。
- `/p/<profile>/` ルートでは、そのプロファイルのチャット用セッションに書き込まれ、他のプロファイルのセッションに書き込まれることはありません。
- ミラーされたテキストは、あなたが送信したかのようにチャットの履歴に入ります。[`deliver_only`](#direct-delivery-mode) ルートでは、レンダリングされた生のペイロードになるため、その内容が会話に入っても信頼できる送信元（公開の issue トラッカーではなく、自分自身のサービス）に対してのみ有効化してください。

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

## イベントトリガーの cron ジョブ {#event-triggered-cron-jobs}

ルートに `cron_job` を設定すると、イベントが届くたびに **既存の cron ジョブ** を発火します — 固定間隔でのポーリングや、新しい Webhook エージェントセッションの開始の代わりです。これにより、あらゆるスケジュール済みジョブをイベント駆動のタスクに変えられます。スケジュールはフォールバックの定期巡回として残し（またはめったに発火しないものにし）、何かが実際に変化した瞬間に Webhook でジョブを発火させます。

仕組み:

1. イベントは、他のルートと同じ HMAC 認証、レート制限、`events`/`filters`/`script` によるフィルタリング、冪等性を通過します。
2. ルートの `prompt` テンプレートがペイロードからレンダリングされ、**実行ごとの一時的なコンテキスト** としてジョブに注入されます（`cronjob(action='run', prompt=...)` と同じ経路で、ジョブに保存されたプロンプトが変更されることはありません）。
3. ジョブはスケジューラーと同じ最大 1 回（at-most-once）のクレームを通じて発火するため、Webhook が集中しても既に実行中のジョブが二重に発火することはなく、ジョブ自身の配信ターゲットが出力を受け取ります。

### 例: レビューのフィードバックで PR レビュージョブを発火する

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        pr-feedback:
          events: ["pull_request_review"]
          secret: "github-webhook-secret"
          signature: github
          cron_job: "pr-review-sweeper"        # existing job ID or name
          prompt: |
            PR #{number} in {repository.full_name} received new review feedback
            from {review.user.login}: {review.body}
```

### CLI 経由

```bash
hermes webhook subscribe pr-feedback \
  --events "pull_request_review" \
  --signature github \
  --cron-job "pr-review-sweeper" \
  --prompt "PR #{number} received feedback: {review.body}"
```

ジョブの参照はサブスクリプションの作成時に検証されるため、タイプミスはすぐに判明します。

### 注意事項

- `cron_job` と `deliver_only` は併用できません（ルートが両方を設定している場合、アダプターは起動を拒否します）。cron ジョブは自身の配信を処理します。
- ルートレベルの `deliver`、`deliver_extra`、`skills` フィールドは `cron_job` ルートでは無視されます — ジョブ自身の設定が適用されます。
- 一時停止/無効化されたジョブは発火されません。イベントはログに記録されて破棄されます。
- POST はすぐに `202 Accepted` を返し、ジョブはバックグラウンドで実行されます。

---

## 動的サブスクリプション（CLI） {#dynamic-subscriptions}

`config.yaml` の静的ルートに加えて、`hermes webhook` CLI コマンドを使って Webhook サブスクリプションを動的に作成できます。これは、エージェント自身がイベント駆動のトリガーをセットアップする必要がある場合に特に便利です。

### サブスクリプションを作成する

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --signature github \
  --prompt "New issue #{issue.number}: {issue.title}\nBy: {issue.user.login}\n\n{issue.body}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "Triage new GitHub issues"
```

これにより、Webhook URL と、自動生成された HMAC シークレットのマスクされた形式が返されます。完全なシークレットは `~/.hermes/webhook_subscriptions.json` 内のルート名の下に保存されます（独自のシークレットを使うには `--secret` を渡します）。その URL に POST するようサービスを設定してください。

新しいサブスクリプションは、独自の方法で署名するプロバイダー向けに `--signature github`、`gitlab`、`linear` を渡さない限り、`--signature generic`（タイムスタンプにバインドされた V2 または Standard Webhooks）に固定されます。既存のサブスクリプションを更新しても、その固定は維持されます。`signature` を省略して既存のルートをダッシュボードから再保存した場合も固定は維持され（明示的に選択した場合は置き換えられます）、MCP のイベントサブスクリプションは最初の作成時に `generic` に固定されます。

ダッシュボードの **New subscription** フォーム（Web ダッシュボードとデスクトップアプリ）にも、同じ選択肢が **Signature** セレクトとして用意されており、デフォルトは `generic` です。GitHub、GitLab、Linear から配信されるルートでは、それぞれを選択してください。API 経由では、`POST /api/webhooks` は任意の `signature`（`generic`、`github`、`gitlab`、`linear` のいずれか。それ以外は `422`）を受け取ります。これを指定しない新しいルートは `generic` に固定され、指定せずに再 POST した場合は保存済みの固定が維持されます。

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

`test` は、ルートの `signature` の固定が期待する方法で署名します: `github` では `X-Hub-Signature-256`、`gitlab` では `X-Gitlab-Token`、`linear` では `Linear-Signature`、`generic` または固定されていないルートでは `X-Webhook-Signature-V2` + `X-Webhook-Timestamp` です。

### 動的サブスクリプションの仕組み

- サブスクリプションは `~/.hermes/webhook_subscriptions.json` に保存されます
- Webhook アダプターは、受信リクエストごとにこのファイルの一貫したスナップショットをホットリロードします（stat ゲート付き（mtime/size/inode）、ロックフリー）
- `config.yaml` の静的ルートは、同名の動的ルートよりも常に優先されます
- 動的サブスクリプションは、静的ルートと同じルート形式と機能（events、プロンプトテンプレート、スキル、配信）を使います
- 作成、更新、有効化/無効化、削除の操作は直列化され、ストアをアトミックに置き換えます。古い状態に基づく並行更新は、別の操作で削除または無効化されたルートを復元するのではなく、拒否されます
- 既存のルートを `--route-profile` で再バインドすると、その HMAC シークレットは自動的にローテーションされます。古いプロファイルのシークレットを持つ呼び出し元は、再バインドされたルートで認証できません。受信側のサービスがオペレーターの選んだ値を必要とする場合にのみ、別の `--secret` を渡してください
- ゲートウェイの再起動は不要 — サブスクライブすればすぐに有効になります

### エージェント駆動のサブスクリプション

エージェントは、`webhook-subscriptions` スキルにガイドされたときに、ターミナルツールを介してサブスクリプションを作成できます。エージェントに「GitHub issue 用の Webhook をセットアップして」と頼むと、適切な `hermes webhook subscribe` コマンドを実行します。

---

## ルートごとのツールセット {#per-route-toolsets}

Webhook のペイロードには信頼できない第三者のコンテンツが含まれる可能性があるため、Webhook によるエージェント実行は、意図的に制限されたツールセット（`web_search`、`web_extract`、`vision_analyze`、`clarify`）がデフォルトになっています — 公開の PR タイトルや issue コメントが、プロンプトインジェクションによってターミナルに到達できてはなりません。

**信頼できる** ルート — システムアラートをプッシュする localhost の監視デーモンや社内の CI システムなど — では、他のすべての Webhook ルートを広げることなく、そのルートに限ってより広いツールセットを付与できます:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        oom-emergency:
          secret: "monitor-secret"
          prompt: "Memory emergency: {detail}. Diagnose with ps/free/py-spy and report."
          toolsets: ["terminal", "file", "code_execution", "web"]
          deliver: "telegram"
```

動的サブスクリプションの場合は、`~/.hermes/webhook_subscriptions.json` を直接編集して `toolsets` キーを追加します:

```json
{
  "oom-emergency": {
    "secret": "...",
    "prompt": "...",
    "toolsets": ["terminal", "file", "web"],
    "deliver": "telegram"
  }
}
```

動作と安全性に関する特性:

- ルートのリストは、そのルートの実行におけるプラットフォームレベルの Webhook ツールセットの解決結果を **置き換えます**（マージはされません）。明示的な空のリスト（`toolsets: []`）も上書きになります。そのルートにはプラットフォームのツールセットも MCP サーバーも与えられず、`platform_toolsets.webhook: []` と同じになります。
- 名前は `platform_toolsets` の設定と同じ経路で検証されます — 不明な名前やプラットフォーム制限のあるツールセットは除外されます。
- `hermes webhook subscribe` は意図的に toolsets フラグを受け付け **ません**。昇格したツールの付与は設定ファイルの手動編集であるため、実行時に自身のサブスクリプションを作成するエージェントが `terminal` を自身に付与することはできません。
- 昇格したツールセットは、送信元を完全に管理でき、実際の HMAC シークレットを持つルートにのみ付与してください。そのルートに正しく署名されたペイロードを POST できる人は誰でも、実質的にそれらのツールを持つエージェントを実行していることになります。

---

## セキュリティ {#security}

Webhook アダプターには複数層のセキュリティが含まれています:

### HMAC 署名検証

アダプターは、各送信元に適した方法で受信 Webhook の署名を検証します:

- **GitHub**: `X-Hub-Signature-256` ヘッダー — `sha256=` がプレフィックスされた HMAC-SHA256 16 進ダイジェスト
- **GitLab**: `X-Gitlab-Token` ヘッダー — プレーンなシークレット文字列の一致
- **Standard Webhooks**: `webhook-id`、`webhook-timestamp`、`webhook-signature` ヘッダー — 署名対象は `{id}.{timestamp}.{raw_body}` で、署名は `v1,<base64-hmac-sha256>` 形式
- **汎用（V2、推奨）**: `X-Webhook-Signature-V2` + `X-Webhook-Timestamp` ヘッダー — `<timestamp>.<body>` の HMAC-SHA256 16 進ダイジェスト。タイムスタンプ（Unix 秒）はサーバー時刻から ±300 秒以内である必要があり、これにより傍受されたリクエストが後でリプレイされることを防ぎます。
- **汎用（V1、拒否）**: `X-Webhook-Signature` はボディのみに署名するため、傍受されたリクエストは配信 ID のキャッシュが期限切れになった後も有効なままです。V1 のみのリクエストは、ループバックであっても `401 Unauthorized` を受け取ります。送信元を上記のタイムスタンプにバインドされた V2 形式に切り替えてください。ゲートウェイはルートごとに一度、移行手順をログに出力します。

シークレットが設定されているのに認識可能な署名ヘッダーが存在しない場合、リクエストは拒否されます。

### 署名方式の固定 {#pin-the-signature-scheme}

GitHub（`X-Hub-Signature-256`）、Linear（`Linear-Signature`）、および拒否される V1 ヘッダーは、いずれも同じ値を持ちます: ルートのシークレットによるボディの HMAC-SHA256 で、タイムスタンプはありません。そのため、`signature` のないルートでは、傍受された V1 リクエストの MAC を `Linear-Signature`（または `sha256=` プレフィックスを付けた `X-Hub-Signature-256`）に移し、新しい配信 ID を付けて送信することで、依然としてリプレイが可能です。すべてのルートに `signature` を設定し、送信元が使う方式のみを受け付けるようにしてください:

```yaml
routes:
  github-pr:
    secret: "github-webhook-secret"
    signature: github      # only X-Hub-Signature-256
  deploy-hook:
    secret: "deploy-secret"
    signature: generic     # only X-Webhook-Signature-V2 + X-Webhook-Timestamp, or Standard Webhooks
```

不明な値は設定エラーになります: 静的ルートの場合はアダプターが起動しなくなり、動的サブスクリプションの場合はスキップされます。どちらの場合も、すべての方式を受け付ける動作にフォールバックすることはありません。

送信元がボディのみの V1（`X-Webhook-Signature`）を使っていた場合は、V2 に移行する際に **そのルートのシークレットをローテーション** してください。古い傍受データがすべてのヘッダーファミリーで検証されなくなるのは、シークレットが変更された後だけです。GitHub と Linear の署名には独自のタイムスタンプがないため、`github`/`linear` に固定されたルートでも、自身のプロバイダーからのリクエストのリプレイに対しては 1 時間の配信 ID キャッシュに依存します。

### シークレットは必須

すべてのルートにはシークレットが必要です — ルートに直接設定するか、グローバルの `secret` から継承します。シークレットのないルートは、アダプターが起動時にエラーで失敗する原因になります。開発/テスト目的のみ、シークレットを `"INSECURE_NO_AUTH"` に設定して検証を完全にスキップできます。

マルチプロファイルルーティングが有効な場合、ルートの `profile` フィールドは、そのシークレットを 1 つの実行ターゲットにもバインドします。`profile` のないルートはデフォルトプロファイル専用です。有効なルート署名を持つリクエストであっても、`/p/<profile>/` プレフィックスがルートのバインドと一致しない場合は拒否されます。

`INSECURE_NO_AUTH` は、ゲートウェイがループバックホスト（`127.0.0.1`、`localhost`、`::1`）にバインドされている場合にのみ受け付けられます。`0.0.0.0` や LAN IP などの非ループバックバインドと組み合わされた場合、アダプターは起動を拒否します — これは、未認証のエンドポイントを公開インターフェースに誤って晒すことを防ぎます。

### レート制限

各ルートはデフォルトで **1 分あたり 30 リクエスト**にレート制限されます（固定ウィンドウ）。これをグローバルに設定します:

```yaml
platforms:
  webhook:
    extra:
      rate_limit: 60  # requests per minute
```

制限を超えるリクエストは `429 Too Many Requests` 応答を受け取ります。

### 冪等性

配信 ID（`X-GitHub-Delivery`、`svix-id`、`webhook-id`、`X-Request-ID`、またはリクエストごとのランダムな ID から取得）は **1 時間** キャッシュされます。重複配信（例: Webhook の再試行）は `200` 応答で静かにスキップされ、エージェントの重複実行を防ぎます。

### ボディサイズ制限

**1 MB** を超えるペイロードは、ボディが読み取られる前に拒否されます。これを設定します:

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### 認証済みは信頼できることを意味しない {#authenticated-does-not-mean-trusted}

:::warning
**HMAC 検証が認証するのは _送信者_ であり、_内容_ ではありません。** 有効な署名が証明するのは、リクエストがルートのシークレットを持つ当事者（例: GitHub）から来たということだけです。ペイロード内の _業務フィールド_ を誰が書いたかについては何も示しません — PR タイトル、コミットメッセージ、issue の説明、その他の上流のテキストは任意の第三者によって書かれたものであり、信頼できないものとして扱う必要があります。

これは、エージェントが読み取るすべてのものに適用されるのと同じ信頼モデルです: Web ページ、ファイル、ツールの出力はすべて信頼できない入力です。Hermes は信頼できないテキストをブロックリストでサニタイズしません — また、確実にサニタイズすることもできません。言い回し、エンコーディング、翻訳によって簡単に回避できるからです。**信頼境界は入力チャンネルではなく、エージェントの能力の範囲です。** そこを強化してください:

- **ランタイムをサンドボックス化する。** インターネットに公開する場合は、Docker または SSH ターミナルバックエンド（または VM）でゲートウェイを実行し、乗っ取られたターンがホストに触れられないようにします。
- **ツールセットを絞る。** ルートが読み取りと要約だけを必要とする場合は、Webhook でトリガーされたセッションで `terminal`、`file`、および外部へのアクションを行うツールを無効にします。能力が少ないほど、ペイロードのフィールドに注入された命令が含まれていた場合の被害範囲が小さくなります。
- **承認を有効にしておく。** 破壊的な操作や外部への操作に対して承認を有効にし、注入された命令が無人で実行されないようにします。
- **テンプレートを狭くする。** ペイロード全体をダンプする `{__raw__}` や空のテンプレートよりも、名前付きフィールド（`{pull_request.title}`）を使った具体的な `prompt` を優先し、意図したフィールドだけがプロンプトに届くようにします。
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
- 汎用の送信元の場合は、`X-Webhook-Signature-V2` と `X-Webhook-Timestamp` を使う。ボディのみの V1 署名は拒否されます
- `signature` が設定されたルートは、それ以外のすべての方式を `401` で拒否します（ゲートウェイログで `is pinned to signature` を探してください）。`--signature` なしで `hermes webhook subscribe` により作成されたサブスクリプションは `generic` に固定されます
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

- 冪等性キャッシュがこれを防ぐはずです — Webhook 送信元が配信 ID ヘッダー（`X-GitHub-Delivery`、`svix-id`、`webhook-id`、または `X-Request-ID`）を送信しているか確認する
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
