---
sidebar_position: 5
title: "Microsoft Teams"
description: "Hermes Agent を Microsoft Teams ボットとしてセットアップします"
---

# Microsoft Teams のセットアップ

このページの Python 依存関係コマンドは、
[PM で準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を前提としています。
依存関係を変更した後は、チェックアウトを再アクティベートして Hermes を再起動してください。

Hermes Agent を Microsoft Teams にボットとして接続します。Slack の Socket Mode とは異なり、Teams は **公開された HTTPS Webhook** を呼び出すことでメッセージを配信するため、インスタンスには公開到達可能なエンドポイントが必要です。開発トンネル（ローカル開発）か、実際のドメイン（本番）のいずれかです。

通常のボット会話ではなく、Microsoft Graph イベントからの会議要約が必要ですか？ 専用のセットアップページをご利用ください: [Teams 会議](./teams-meetings.md)。

> ガイド付きの手順を見るには、`hermes gateway setup` を実行して **Microsoft Teams** を選択してください。

## ボットの応答の仕方

| コンテキスト | 動作 |
|---------|----------|
| **個人チャット（DM）** | ボットはすべてのメッセージに応答します。@メンションは不要です。 |
| **グループチャット** | ボットは @メンションされたときのみ応答します。 |
| **チャネル** | ボットは @メンションされたときのみ応答します。 |

Teams は @メンションを `<at>BotName</at>` タグ付きの通常のメッセージとして配信し、Hermes は処理前にこれを自動的に取り除きます。

リソース固有の同意（RSC）がない場合、Teams はボットを @メンションしたメッセージのみを配信するため、フィルタリングは不要です。アプリのマニフェストで `ChannelMessage.Read.Group` または `ChatMessage.Read.Chat` が付与されると、Teams は会話内の**すべて**のメッセージを配信します — `require_mention: true`（または `TEAMS_REQUIRE_MENTION=true`）を設定すると、ボットはチャネル／グループチャットで自分を @メンションしたメッセージか、自分のメッセージへの返信にのみ応答します。個人チャットが制限されることはなく、制限されたメッセージは添付ファイルがダウンロードされる前に破棄されます。

---

ソースまたはローカルインストールの場合は、同梱のアダプターが Microsoft Teams SDK を
インポートできるように Teams extra を含めてください:

```bash
python -c "import pm; pm.sync_venv(['teams'], explicit=True)"
```

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
devtunnel port create hermes-bot -p 3978 --protocol http  # replace 3978 with TEAMS_PORT if changed
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # replace 3978 with TEAMS_PORT if changed

# cloudflared
cloudflared tunnel --url http://localhost:3978  # replace 3978 with TEAMS_PORT if changed
```

出力から `https://` URL をコピーします。次のステップで使用します。開発中はトンネルを実行したままにしてください。

公開トンネルの URL は HTTPS を使いますが、Hermes のローカル Webhook リスナーはプレーンな HTTP を使います。トンネルが TLS を終端し、HTTP をポート `3978` に転送します。ローカルのトンネルポートを HTTPS として設定しないでください。

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
# Required
TEAMS_CLIENT_ID=<your-client-id>
TEAMS_CLIENT_SECRET=<your-client-secret>
TEAMS_TENANT_ID=<your-tenant-id>

# Restrict access to specific users (recommended)
# Use AAD object IDs from `teams status --verbose`
TEAMS_ALLOWED_USERS=<your-aad-object-id>
```

---

## ステップ5: ゲートウェイを起動する

**Docker**（`docker-compose.yml` があるディレクトリから実行する必要があります — 通常は `~` ではなく、クローンした `hermes-agent` リポジトリです）:

```bash
cd /path/to/hermes-agent
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

**ネイティブ / systemd インストール**（一般的な `hermes` ワンライナーインストーラーによる `~/.hermes/hermes-agent` 配下のインストール）:

```bash
hermes gateway restart
# or foreground: hermes gateway run
```

Teams SDK はオプションです。ポリシーで許可されている場合、ゲートウェイは初回起動時に PM を通じて
`teams` extra を要求します。PM は実行中のインタープリターを変更するのではなく、完全な環境を
準備します。準備済みのソースチェックアウトから明示的に extra を要求するには:

```bash
python -c "import pm; pm.sync_venv(['teams'], explicit=True)"
```

デフォルトの Webhook ポートは `3978` です（`TEAMS_PORT` で上書き可能）。実行されているか確認します。

```bash
curl http://localhost:3978/health   # should return: ok
# Docker:
docker logs -f hermes
# Native:
hermes gateway status -l
```

次の行を探します。
```
[teams] Webhook server listening on * (all interfaces, IPv4+IPv6):3978/api/messages
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
| `TEAMS_REQUIRE_MENTION` | `true` に設定すると、チャネルとグループチャットでは @メンションとボットへの返信にのみ応答します（デフォルト: `false`。RSC によるメッセージ読み取りの同意を持つアプリ向け） |

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
    require_mention: false   # true once the app has RSC message-read consent
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

### 会議要約の配信（Teams 会議パイプライン） {#meeting-summary-delivery-teams-meeting-pipeline}

[Teams 会議パイプラインプラグイン](./msgraph-webhook.md) が有効な場合、このアダプターは会議要約の送信配信も処理します。2つではなく、1つの Teams 連携面です。会議の文字起こしが要約された後、ライターは要約を選択した Teams のターゲットに投稿します。

パイプラインの要約配信は、ボット設定と並んで `teams` プラットフォームのエントリの下で設定します。

```yaml
platforms:
  teams:
    enabled: true
    extra:
      # existing bot config (client_id, client_secret, tenant_id, port) ...

      # Meeting summary delivery (only used when the teams_pipeline plugin is enabled)
      delivery_mode: "graph"       # or "incoming_webhook"
      # For delivery_mode: graph — pick ONE of:
      chat_id: "19:meeting_..."    # post into a Teams chat
      # team_id: "..."             # OR post into a channel
      # channel_id: "..."
      # access_token: "..."        # optional; falls back to MSGRAPH_* app credentials
      # For delivery_mode: incoming_webhook:
      # incoming_webhook_url: "https://outlook.office.com/webhook/..."
```

| モード | 使用する場面 | トレードオフ |
|------|----------|-----------|
| `incoming_webhook` | 静的な Teams 生成 URL で「このチャネルに要約を投稿」するシンプルな用途。 | 返信スレッドなし、リアクションなし、Webhook の設定済みアイデンティティとして表示。 |
| `graph` | Microsoft Graph を介して、ボットのアイデンティティでスレッド化されたチャネル投稿や1:1／グループチャット投稿。 | `ChannelMessage.Send`（チャネル）または `Chat.ReadWrite.All`（チャット）のアプリケーション権限を持つ [Graph アプリ登録](../../guides/microsoft-graph-app-registration.md) が必要。 |

`teams_pipeline` プラグインが **有効でない** 場合、これらの設定は不活性です。パイプラインランタイムが Graph Webhook の受信にバインドされたときにのみ機能します。

---

## 本番デプロイ {#production-deployment}

恒久的なサーバーでは、リバースプロキシで TLS を終端し、リクエストをプレーンな HTTP の Hermes リスナー（通常は `http://127.0.0.1:3978`）に転送します。プロキシの公開 HTTPS エンドポイントを Teams に登録します。

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

すでにボットを作成済みで、エンドポイントを更新するだけの場合:

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

公開 HTTPS エンドポイントがインターネットから到達可能で、有効な TLS 証明書を使っていることを確認してください。Teams は自己署名証明書を拒否します。Hermes のリスナーはプロキシの背後に置いてください。ポート `3978` 自体は HTTPS を提供しません。

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| `docker compose` で `Can't find a suitable configuration file` | `docker-compose.yml` があるリポジトリにいないか、ネイティブインストールを使っています — 代わりに `hermes gateway restart` を使うか、先にクローンへ `cd` してください |
| `requirements not met` / `Teams SDK missing` / `No adapter available for teams` | 上記のとおり PM を通じて `teams` extra を要求し、ゲートウェイを再起動してください。システムの Python にインストールしたり、Hermes の環境を変更したりしないでください。 |
| `health` エンドポイントは動くがボットが応答しない | トンネルがまだ実行中で、ボットのメッセージングエンドポイントがトンネル URL と一致しているか確認 |
| Teams がメッセージを送ると、ログに `"UNKNOWN / HTTP/1.0" 400` と表示される | トンネルまたはリバースプロキシが、Hermes のプレーンな HTTP リスナーに HTTPS を転送しています。プロキシで TLS を終端し、HTTP をポート `3978` に転送してください |
| ログに `KeyError: 'teams'` | コンテナを再起動。これは現在のバージョンで修正済み |
| ボットが認証エラーで応答する | `TEAMS_CLIENT_ID`、`TEAMS_CLIENT_SECRET`、`TEAMS_TENANT_ID` がすべて正しく設定されているか確認 |
| `No inference provider configured` | `~/.hermes/.env` に `ANTHROPIC_API_KEY`（または別のプロバイダーキー）が設定されているか確認 |
| ボットはメッセージを受信するが無視する | AAD オブジェクトID が `TEAMS_ALLOWED_USERS` にない可能性があります。`teams status --verbose` を実行して確認 |
| 再起動するとトンネル URL が変わる | devtunnel の URL は、名前付きトンネル（`devtunnel create hermes-bot`）を使えば永続的です。ngrok と cloudflared は、有料プランでない限り実行のたびに新しい URL を生成します。変わったら `teams app update` でボットのエンドポイントを更新してください |
| Teams に「This bot is not responding」と表示される | Webhook がエラーを返しました。`docker logs hermes` / `hermes gateway status -l` でトレースバックを確認 |
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

- [Teams 会議](./teams-meetings.md)
- [Teams 会議パイプラインを運用する](../../guides/operate-teams-meeting-pipeline.md)
