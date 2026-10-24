---
sidebar_position: 5
title: "Microsoft Teams"
description: "Hermes Agent を Microsoft Teams ボットとしてセットアップします"
---

# Microsoft Teams のセットアップ

Hermes Agent を Microsoft Teams にボットとして接続します。Slack の Socket Mode とは異なり、Teams は **公開された HTTPS Webhook** を呼び出すことでメッセージを配信するため、インスタンスには公開到達可能なエンドポイントが必要です。開発トンネル（ローカル開発）か、実際のドメイン（本番）のいずれかです。

通常のボット会話ではなく、Microsoft Graph イベントからの会議要約が必要ですか？ 専用のセットアップページをご利用ください: [Teams 会議](/docs/user-guide/messaging/teams-meetings)。

## ボットの応答の仕方

| コンテキスト | 動作 |
|---------|----------|
| **個人チャット（DM）** | ボットはすべてのメッセージに応答します。@メンションは不要です。 |
| **グループチャット** | ボットは @メンションされたときのみ応答します。 |
| **チャネル** | ボットは @メンションされたときのみ応答します。 |

Teams は @メンションを `<at>BotName</at>` タグ付きの通常のメッセージとして配信し、Hermes は処理前にこれを自動的に取り除きます。

---

## ステップ1: Teams CLI をインストールする

`@microsoft/teams.cli` はボット登録を自動化します。Azure ポータルは不要です。

```bash
npm install -g @microsoft/teams.cli@preview
teams login
```

ログインを確認し、自分の AAD オブジェクトID（`TEAMS_ALLOWED_USERS` に必要）を確認するには:

```bash
teams status --verbose
```

---

## ステップ2: Webhook ポートを公開する

Teams は `localhost` にメッセージを配信できません。ローカル開発では、任意のトンネルツールを使って公開 HTTPS URL を取得してください。デフォルトのポートは `3978` です。必要に応じて `TEAMS_PORT` で変更してください。

```bash
# devtunnel (Microsoft)
devtunnel create hermes-bot --allow-anonymous
devtunnel port create hermes-bot -p 3978 --protocol https  # 変更した場合は 3978 を TEAMS_PORT に置き換え
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # 変更した場合は 3978 を TEAMS_PORT に置き換え

# cloudflared
cloudflared tunnel --url http://localhost:3978  # 変更した場合は 3978 を TEAMS_PORT に置き換え
```

出力から `https://` URL をコピーします。次のステップで使用します。開発中はトンネルを実行したままにしてください。

本番環境では、代わりにボットのエンドポイントをサーバーの公開ドメインに向けてください（[本番デプロイ](#production-deployment) を参照）。

---

## ステップ3: ボットを作成する

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://<your-tunnel-url>/api/messages"
```

CLI は `CLIENT_ID`、`CLIENT_SECRET`、`TENANT_ID` と、ステップ6用のインストールリンクを出力します。クライアントシークレットは再表示されないので保存してください。

---

## ステップ4: 環境変数を設定する

`~/.hermes/.env` に追加します。

```bash
# 必須
TEAMS_CLIENT_ID=<your-client-id>
TEAMS_CLIENT_SECRET=<your-client-secret>
TEAMS_TENANT_ID=<your-tenant-id>

# 特定のユーザーにアクセスを制限（推奨）
# `teams status --verbose` から AAD オブジェクトID を使用
TEAMS_ALLOWED_USERS=<your-aad-object-id>
```

---

## ステップ5: ゲートウェイを起動する

```bash
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

これでゲートウェイが起動します。デフォルトの Webhook ポートは `3978` です（`TEAMS_PORT` で上書き可能）。実行されているか確認します。

```bash
curl http://localhost:3978/health   # ok を返すはずです
docker logs -f hermes
```

次の行を探します。
```
[teams] Webhook server listening on 0.0.0.0:3978/api/messages
```

---

## ステップ6: Teams にアプリをインストールする

```bash
teams app get <teamsAppId> --install-link
```

出力されたリンクをブラウザで開きます。Teams クライアントで直接開きます。インストール後、ボットにダイレクトメッセージを送信してください。準備完了です。

---

## 設定リファレンス

### 環境変数

| 変数 | 説明 |
|----------|-------------|
| `TEAMS_CLIENT_ID` | Azure AD アプリ（クライアント）ID |
| `TEAMS_CLIENT_SECRET` | Azure AD クライアントシークレット |
| `TEAMS_TENANT_ID` | Azure AD テナントID |
| `TEAMS_ALLOWED_USERS` | ボットの利用を許可する AAD オブジェクトID（カンマ区切り） |
| `TEAMS_ALLOW_ALL_USERS` | 許可リストをスキップして誰でも許可するには `true` に設定 |
| `TEAMS_HOME_CHANNEL` | cron／プロアクティブメッセージ配信用の会話ID |
| `TEAMS_HOME_CHANNEL_NAME` | ホームチャネルの表示名 |
| `TEAMS_PORT` | Webhook ポート（デフォルト: `3978`） |

### config.yaml

あるいは、`~/.hermes/config.yaml` を介して設定します。

```yaml
platforms:
  teams:
    enabled: true
    extra:
      client_id: "your-client-id"
      client_secret: "your-secret"
      tenant_id: "your-tenant-id"
      port: 3978
```

---

## 機能

### インタラクティブな承認カード

エージェントが潜在的に危険なコマンドを実行する必要がある場合、`/approve` の入力を求める代わりに、4つのボタンを持つアダプティブカードを送信します。

- **Allow Once** — この特定のコマンドを承認
- **Allow Session** — このパターンをセッションの残りの間承認
- **Always Allow** — このパターンを恒久的に承認
- **Deny** — コマンドを拒否

ボタンをクリックすると、承認がインラインで解決され、カードが決定内容に置き換わります。

### 会議要約の配信（Teams 会議パイプライン）

[Teams 会議パイプラインプラグイン](/docs/user-guide/messaging/msgraph-webhook) が有効な場合、このアダプターは会議要約の送信配信も処理します。2つではなく、1つの Teams 連携面です。会議の文字起こしが要約された後、ライターは要約を選択した Teams のターゲットに投稿します。

パイプラインの要約配信は、ボット設定と並んで `teams` プラットフォームのエントリの下で設定します。

```yaml
platforms:
  teams:
    enabled: true
    extra:
      # 既存のボット設定 (client_id, client_secret, tenant_id, port) ...

      # 会議要約の配信（teams_pipeline プラグインが有効な場合のみ使用）
      delivery_mode: "graph"       # または "incoming_webhook"
      # delivery_mode: graph の場合 — 次のうち1つを選択:
      chat_id: "19:meeting_..."    # Teams チャットに投稿
      # team_id: "..."             # またはチャネルに投稿
      # channel_id: "..."
      # access_token: "..."        # 任意; 未指定なら MSGRAPH_* アプリ認証情報にフォールバック
      # delivery_mode: incoming_webhook の場合:
      # incoming_webhook_url: "https://outlook.office.com/webhook/..."
```

| モード | 使用する場面 | トレードオフ |
|------|----------|-----------|
| `incoming_webhook` | 静的な Teams 生成 URL で「このチャネルに要約を投稿」するシンプルな用途。 | 返信スレッドなし、リアクションなし、Webhook の設定済みアイデンティティとして表示。 |
| `graph` | Microsoft Graph を介して、ボットのアイデンティティでスレッド化されたチャネル投稿や1:1／グループチャット投稿。 | `ChannelMessage.Send`（チャネル）または `Chat.ReadWrite.All`（チャット）のアプリケーション権限を持つ [Graph アプリ登録](/docs/guides/microsoft-graph-app-registration) が必要。 |

`teams_pipeline` プラグインが **有効でない** 場合、これらの設定は不活性です。パイプラインランタイムが Graph Webhook の受信にバインドされたときにのみ機能します。

---

## 本番デプロイ {#production-deployment}

恒久的なサーバーでは、devtunnel をスキップして、サーバーの公開 HTTPS エンドポイントでボットを登録します。

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

すでにボットを作成済みで、エンドポイントを更新するだけの場合:

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

設定したポート（`TEAMS_PORT`、デフォルト `3978`）がインターネットから到達可能で、TLS 証明書が有効であることを確認してください。Teams は自己署名証明書を拒否します。

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| `health` エンドポイントは動くがボットが応答しない | トンネルがまだ実行中で、ボットのメッセージングエンドポイントがトンネル URL と一致しているか確認 |
| ログに `KeyError: 'teams'` | コンテナを再起動。これは現在のバージョンで修正済み |
| ボットが認証エラーで応答する | `TEAMS_CLIENT_ID`、`TEAMS_CLIENT_SECRET`、`TEAMS_TENANT_ID` がすべて正しく設定されているか確認 |
| `No inference provider configured` | `~/.hermes/.env` に `ANTHROPIC_API_KEY`（または別のプロバイダーキー）が設定されているか確認 |
| ボットはメッセージを受信するが無視する | AAD オブジェクトID が `TEAMS_ALLOWED_USERS` にない可能性があります。`teams status --verbose` を実行して確認 |
| 再起動するとトンネル URL が変わる | devtunnel の URL は、名前付きトンネル（`devtunnel create hermes-bot`）を使えば永続的です。ngrok と cloudflared は、有料プランでない限り実行のたびに新しい URL を生成します。変わったら `teams app update` でボットのエンドポイントを更新してください |
| Teams に「This bot is not responding」と表示される | Webhook がエラーを返しました。`docker logs hermes` でトレースバックを確認 |
| ログに `[teams] Failed to connect` | SDK の認証に失敗しました。認証情報と、テナントID が `teams login` で使用したアカウントと一致しているか再確認 |

---

## セキュリティ

:::warning
**必ず `TEAMS_ALLOWED_USERS`** に認可されたユーザーの AAD オブジェクトID を設定してください。これがないと、ボットを見つけたりインストールしたりできる人なら誰でも操作できてしまいます。

`TEAMS_CLIENT_SECRET` はパスワードのように扱ってください。Azure ポータルまたは Teams CLI を介して定期的にローテーションしてください。
:::

- 認証情報は `~/.hermes/.env` にパーミッション `600`（`chmod 600 ~/.hermes/.env`）で保存してください
- ボットは `TEAMS_ALLOWED_USERS` 内のユーザーからのメッセージのみを受け付けます。認可されていないメッセージは黙って破棄されます
- 公開エンドポイント（`/api/messages`）は Teams Bot Framework によって認証されます。有効な JWT のないリクエストは拒否されます

## 関連ドキュメント

- [Teams 会議](/docs/user-guide/messaging/teams-meetings)
- [Teams 会議パイプラインを運用する](/docs/guides/operate-teams-meeting-pipeline)
