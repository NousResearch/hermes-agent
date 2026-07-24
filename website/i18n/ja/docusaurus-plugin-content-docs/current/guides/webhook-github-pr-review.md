---
sidebar_position: 11
sidebar_label: "Webhook経由のGitHub PRレビュー"
title: "WebhookによるGitHub PRコメントの自動投稿"
description: "HermesをGitHubに接続し、PRのdiffの取得、コード変更のレビュー、コメント投稿を自動化する — webhookでトリガーされ、手動のプロンプトは不要"
---

# WebhookによるGitHub PRコメントの自動投稿

このガイドでは、Hermes AgentをGitHubに接続し、プルリクエストのdiffの取得、コード変更の分析、コメントの投稿を自動化する手順を説明します。これらはwebhookイベントによってトリガーされ、手動のプロンプトは不要です。

PRが作成または更新されると、GitHubはあなたのHermesインスタンスにwebhook POSTを送信します。Hermesは、`gh` CLIでdiffを取得するよう指示するプロンプトでエージェントを実行し、その応答がPRスレッドに投稿されます。

:::tip 公開エンドポイントなしで、もっとシンプルにセットアップしたいですか？
公開URLがない場合や、とにかく手早く始めたい場合は、[GitHub PRレビューエージェントを構築する](./github-pr-review-agent.md)を参照してください。cronジョブでスケジュールに従ってPRをポーリングするため、NATやファイアウォールの背後でも動作します。
:::

:::info リファレンスドキュメント
webhookプラットフォームの完全なリファレンス（すべての設定オプション、配信タイプ、動的サブスクリプション、セキュリティモデル）については、[Webhooks](/docs/user-guide/messaging/webhooks)を参照してください。
:::

:::warning プロンプトインジェクションのリスク
webhookペイロードには攻撃者が制御可能なデータ（PRのタイトル、コミットメッセージ、説明文）が含まれており、悪意ある指示を含む可能性があります。webhookエンドポイントをインターネットに公開する場合は、ゲートウェイをサンドボックス環境（Docker、SSHバックエンド）で実行してください。下記の[セキュリティに関する注意](#security-notes)を参照してください。
:::

---

## 前提条件

- Hermes Agentがインストールされ実行されていること（`hermes gateway`）
- ゲートウェイホスト上で[`gh` CLI](https://cli.github.com/)がインストールされ認証済みであること（`gh auth login`）
- Hermesインスタンスへの公開到達可能なURL（ローカルで実行している場合は[ngrokによるローカルテスト](#local-testing-with-ngrok)を参照）
- GitHubリポジトリへの管理者アクセス（webhookを管理するために必要）

---

## ステップ1 — webhookプラットフォームを有効化する

`~/.hermes/config.yaml` に以下を追加します：

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644          # デフォルト。このポートを別のサービスが使っている場合は変更
      rate_limit: 30      # ルートごとの1分あたり最大リクエスト数（グローバルな上限ではない）

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"   # GitHubのwebhookシークレットと完全に一致させること
          events:
            - pull_request

          # エージェントはレビュー前に実際のdiffを取得するよう指示される。
          # {number} と {repository.full_name} はGitHubのペイロードから解決される。
          prompt: |
            A pull request event was received (action: {action}).

            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            Branch: {pull_request.head.ref} → {pull_request.base.ref}
            Description: {pull_request.body}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the code changes for correctness, security issues, and clarity.
            3. Write a concise, actionable review comment and post it.

          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

**主要なフィールド：**

| フィールド | 説明 |
|---|---|
| `secret`（ルートレベル） | このルートのHMACシークレット。省略した場合は `extra.secret` のグローバル値にフォールバックする。 |
| `events` | 受け付ける `X-GitHub-Event` ヘッダー値のリスト。空リスト = すべて受け付ける。 |
| `prompt` | テンプレート。`{field}` と `{nested.field}` はGitHubのペイロードから解決される。 |
| `deliver` | `github_comment` は `gh pr comment` 経由で投稿。`log` はゲートウェイログに書き込むだけ。 |
| `deliver_extra.repo` | ペイロードから例えば `org/repo` に解決される。 |
| `deliver_extra.pr_number` | ペイロードからPR番号に解決される。 |

:::note ペイロードにはコードが含まれない
GitHubのwebhookペイロードにはPRのメタデータ（タイトル、説明文、ブランチ名、URL）が含まれますが、**diffは含まれません**。上記のプロンプトは、実際の変更を取得するためにエージェントに `gh pr diff` を実行するよう指示しています。`terminal` ツールはデフォルトの `hermes-webhook` ツールセットに含まれているため、追加の設定は不要です。
:::

---

## ステップ2 — ゲートウェイを起動する

```bash
hermes gateway
```

次のように表示されるはずです：

```
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

実行中であることを確認します：

```bash
curl http://localhost:8644/health
# {"status": "ok", "platform": "webhook"}
```

---

## ステップ3 — GitHubにwebhookを登録する

1. リポジトリ → **Settings** → **Webhooks** → **Add webhook** に移動します
2. 入力します：
   - **Payload URL:** `https://your-public-url.example.com/webhooks/github-pr-review`
   - **Content type:** `application/json`
   - **Secret:** ルート設定の `secret` に設定したものと同じ値
   - **Which events?** → Select individual events → **Pull requests** にチェック
3. **Add webhook** をクリックします

GitHubは接続を確認するため、即座に `ping` イベントを送信します。これは安全に無視されます — `ping` はあなたの `events` リストに含まれていないため — そして `{"status": "ignored", "event": "ping"}` を返します。これはDEBUGレベルでのみログに記録されるため、デフォルトのログレベルではコンソールに表示されません。

---

## ステップ4 — テスト用PRを開く

ブランチを作成し、変更をプッシュしてPRを開きます。30〜90秒以内（PRのサイズとモデルによる）に、Hermesがレビューコメントを投稿するはずです。

エージェントの進捗をリアルタイムで追うには：

```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

---

## ngrokによるローカルテスト {#local-testing-with-ngrok}

Hermesをノートパソコンで実行している場合は、[ngrok](https://ngrok.com/)を使って公開します：

```bash
ngrok http 8644
```

`https://...ngrok-free.app` のURLをコピーし、GitHubのPayload URLとして使用します。無料のngrokプランでは、ngrokを再起動するたびにURLが変わります — セッションごとにGitHubのwebhookを更新してください。有料のngrokアカウントでは固定ドメインが使えます。

静的なルートは `curl` で直接スモークテストできます — GitHubアカウントや実際のPRは不要です。

:::tip ローカルでテストするときは `deliver: log` を使う
テスト中は設定の `deliver: github_comment` を `deliver: log` に変更してください。そうしないと、エージェントはテストペイロード内の架空の `org/repo#99` リポジトリにコメントを投稿しようとして失敗します。プロンプトの出力に満足したら `deliver: github_comment` に戻してください。
:::

```bash
SECRET="your-webhook-secret-here"
BODY='{"action":"opened","number":99,"pull_request":{"title":"Test PR","body":"Adds a feature.","user":{"login":"testuser"},"head":{"ref":"feat/x"},"base":{"ref":"main"},"html_url":"https://github.com/org/repo/pull/99"},"repository":{"full_name":"org/repo"}}'
SIG=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print "sha256="$2}')

curl -s -X POST http://localhost:8644/webhooks/github-pr-review \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-Hub-Signature-256: $SIG" \
  -d "$BODY"
# 期待される結果: {"status":"accepted","route":"github-pr-review","event":"pull_request","delivery_id":"..."}
```

その後、エージェントの実行を監視します：
```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

:::note
`hermes webhook test <name>` は、`hermes webhook subscribe` で作成された**動的サブスクリプション**でのみ機能します。`config.yaml` のルートは読み込みません。
:::

---

## 特定のアクションへのフィルタリング

GitHubは多くのアクションについて `pull_request` イベントを送信します：`opened`、`synchronize`、`reopened`、`closed`、`labeled` などです。`events` リストは `X-GitHub-Event` ヘッダー値だけでフィルタリングします — ルーティングレベルでアクションのサブタイプではフィルタリングできません。

ステップ1のプロンプトは、`closed` および `labeled` イベントについてエージェントが早期に停止するよう指示することで、これにすでに対処しています。

:::warning エージェントは依然として実行され、トークンを消費する
「stop here」の指示は意味のあるレビューを防ぎますが、アクションに関わらず、すべての `pull_request` イベントに対してエージェントは最後まで実行されます。GitHubのwebhookはイベントタイプ（`pull_request`、`push`、`issues` など）でしかフィルタリングできません — アクションのサブタイプ（`opened`、`closed`、`labeled`）ではフィルタリングできません。サブアクション用のルーティングレベルのフィルタは存在しません。トラフィックの多いリポジトリでは、このコストを受け入れるか、条件付きでwebhook URLを呼び出すGitHub Actionsワークフローで上流側でフィルタリングしてください。
:::

> Jinja2や条件分岐のテンプレート構文はありません。`{field}` と `{nested.field}` がサポートされている唯一の置換です。それ以外はすべてそのままエージェントに渡されます。

---

## 一貫したレビュースタイルのためにスキルを使う

[Hermesスキル](/docs/user-guide/features/skills)を読み込むことで、エージェントに一貫したレビューのペルソナを与えられます。`config.yaml` の `platforms.webhook.extra.routes` 内のルートに `skills` を追加します：

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"
          events: [pull_request]
          prompt: |
            A pull request event was received (action: {action}).
            PR #{number}: {pull_request.title} by {pull_request.user.login}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the diff using your review guidelines.
            3. Write a concise, actionable review comment and post it.
          skills:
            - review
          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

> **注意:** リスト内で最初に見つかったスキルのみが読み込まれます。Hermesは複数のスキルを重ねません — 後続のエントリーは無視されます。

---

## 代わりにSlackやDiscordへ応答を送信する

ルート内の `deliver` と `deliver_extra` フィールドを、対象のプラットフォームに置き換えます：

```yaml
# platforms.webhook.extra.routes.<route-name>: の中

# Slack
deliver: slack
deliver_extra:
  chat_id: "C0123456789"   # SlackチャンネルID（省略すると設定済みのホームチャンネルを使用）

# Discord
deliver: discord
deliver_extra:
  chat_id: "987654321012345678"  # DiscordチャンネルID（省略するとホームチャンネルを使用）
```

対象のプラットフォームもゲートウェイで有効化され接続されている必要があります。`chat_id` を省略すると、応答はそのプラットフォームの設定済みホームチャンネルに送信されます。

有効な `deliver` 値：`log` · `github_comment` · `telegram` · `discord` · `slack` · `signal` · `sms`

---

## GitLabのサポート

同じアダプターはGitLabでも動作します。GitLabは認証に `X-Gitlab-Token` を使用します（HMACではなく単純な文字列一致） — Hermesは両方を自動的に処理します。

イベントフィルタリングについては、GitLabは `X-GitLab-Event` を `Merge Request Hook`、`Push Hook`、`Pipeline Hook` のような値に設定します。`events` には正確なヘッダー値を使用してください：

```yaml
events:
  - Merge Request Hook
```

GitLabのペイロードフィールドはGitHubのものとは異なります — 例えばMRのタイトルは `{object_attributes.title}`、MR番号は `{object_attributes.iid}` です。完全なペイロード構造を把握する最も簡単な方法は、webhook設定の**Test**ボタンと**Recent Deliveries**ログを組み合わせることです。あるいは、ルート設定から `prompt` を省略すると、Hermesは整形されたJSONとしてペイロード全体を直接エージェントに渡し、エージェントの応答（`deliver: log` でゲートウェイログに表示される）がその構造を説明します。

---

## セキュリティに関する注意 {#security-notes}

- **本番環境では決して `INSECURE_NO_AUTH` を使わないでください** — 署名検証が完全に無効になります。これはローカル開発専用です。
- **webhookシークレットを定期的にローテーションし**、GitHub（webhook設定）と `config.yaml` の両方で更新してください。
- **レート制限**はデフォルトでルートごとに30 req/分です（`extra.rate_limit` で設定可能）。これを超えると `429` を返します。
- **重複配信**（webhookの再試行）は、1時間の冪等性キャッシュによって重複排除されます。キャッシュキーは、存在すれば `X-GitHub-Delivery`、次に `X-Request-ID`、次にミリ秒タイムスタンプです。どちらの配信IDヘッダーも設定されていない場合、再試行は重複排除**されません**。
- **プロンプトインジェクション:** PRのタイトル、説明文、コミットメッセージは攻撃者が制御可能です。悪意あるPRはエージェントのアクションを操作しようとする可能性があります。公開インターネットに公開する場合は、ゲートウェイをサンドボックス環境（Docker、VM）で実行してください。

---

## トラブルシューティング

| 症状 | 確認事項 |
|---|---|
| `401 Invalid signature` | config.yaml のシークレットがGitHubのwebhookシークレットと一致していない |
| `404 Unknown route` | URL内のルート名が `routes:` のキーと一致していない |
| `429 Rate limit exceeded` | ルートごとの30 req/分を超過 — GitHubのUIからテストイベントを再配信する際によく起こる。1分待つか `extra.rate_limit` を引き上げる |
| コメントが投稿されない | `gh` がインストールされていない、PATHにない、または認証されていない（`gh auth login`） |
| エージェントは実行されるがコメントがない | ゲートウェイログを確認 — エージェントの出力が空または単に「SKIP」だった場合でも、配信は試行される |
| ポートがすでに使用中 | config.yaml の `extra.port` を変更する |
| エージェントは実行されるがPRの説明文だけをレビューする | プロンプトに `gh pr diff` の指示が含まれていない — diffはwebhookペイロードに含まれていない |
| pingイベントが見えない | 無視されたイベントは `{"status":"ignored","event":"ping"}` を返すが、DEBUGログレベルでのみ — GitHubの配信ログを確認（リポジトリ → Settings → Webhooks → 該当のwebhook → Recent Deliveries） |

**GitHubのRecent Deliveriesタブ**（リポジトリ → Settings → Webhooks → 該当のwebhook）は、すべての配信について正確なリクエストヘッダー、ペイロード、HTTPステータス、レスポンスボディを表示します。サーバーログに触れずに障害を診断する最も速い方法です。

---

## 完全な設定リファレンス

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      host: "0.0.0.0"         # バインドアドレス（デフォルト: 0.0.0.0）
      port: 8644               # リッスンポート（デフォルト: 8644）
      secret: ""               # オプションのグローバルフォールバックシークレット
      rate_limit: 30           # ルートごとの1分あたりリクエスト数
      max_body_bytes: 1048576  # ペイロードサイズ上限（バイト単位、デフォルト: 1 MB）

      routes:
        <route-name>:
          secret: "required-per-route"
          events: []            # [] = すべて受け付ける。それ以外は X-GitHub-Event の値をリスト化
          prompt: ""            # {field} / {nested.field} がペイロードから解決される
          skills: []            # 最初に一致したスキルが読み込まれる（1つのみ）
          deliver: "log"        # log | github_comment | telegram | discord | slack | signal | sms
          deliver_extra: {}     # github_comment には repo + pr_number、それ以外には chat_id
```

---

## 次のステップ

- **[CronベースのPRレビュー](./github-pr-review-agent.md)** — スケジュールに従ってPRをポーリングする。公開エンドポイント不要
- **[Webhookリファレンス](/docs/user-guide/messaging/webhooks)** — webhookプラットフォームの完全な設定リファレンス
- **[プラグインを構築する](/docs/guides/build-a-hermes-plugin)** — レビューロジックを共有可能なプラグインにパッケージ化する
- **[プロファイル](/docs/user-guide/profiles)** — 独自のメモリと設定を持つ専用のレビュアープロファイルを実行する
