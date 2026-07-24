---
sidebar_position: 6
title: "Teamsミーティング"
description: "Microsoft Graph の webhook を使って Microsoft Teams のミーティング要約パイプラインをセットアップする"
---

# Microsoft Teamsミーティング

HermesにMicrosoft Graphのミーティングイベントを取り込ませ、まずトランスクリプトを取得し、必要に応じて録画とSTTにフォールバックし、構造化された要約を下流のシンクに配信させたい場合は、Teamsミーティングパイプラインを使用します。

このページはセットアップと有効化に焦点を当てています。
- Graph認証情報
- webhookリスナーの設定
- Teamsの配信モード
- パイプライン設定の形

day-2運用、本番稼働前のチェック、オペレーター用ワークシートについては、専用ガイドを使用してください: [Teamsミーティングパイプラインの運用](/docs/guides/operate-teams-meeting-pipeline)。

## この機能が行うこと

このパイプラインは次を行います。
1. Microsoft Graphのwebhookイベントを受信する
2. ミーティングを解決し、まずトランスクリプトアーティファクトを優先する
3. 使用可能なトランスクリプトがない場合、録画のダウンロードとSTTにフォールバックする
4. 永続的なジョブ状態とシンクレコードをローカルに保存する
5. Notion、Linear、Microsoft Teamsに要約を書き込める

オペレーターの操作はCLIにとどまります（`teams-pipeline` サブコマンドは `teams_pipeline` プラグインによって登録されます — `hermes plugins enable teams_pipeline` で有効化するか、`config.yaml` で `plugins.enabled: [teams_pipeline]` を設定してください）。

```bash
hermes teams-pipeline validate
hermes teams-pipeline list
hermes teams-pipeline maintain-subscriptions
```

## 前提条件

ミーティングパイプラインを有効化する前に、次が揃っていることを確認してください。

- 動作するHermesインストール
- Teamsのアウトバウンド配信が必要な場合は、既存の [Microsoft Teamsボットのセットアップ](/docs/user-guide/messaging/teams)
- サブスクライブ予定のミーティングリソースに必要な権限を持つMicrosoft Graphアプリケーション認証情報
- webhook配信のためにMicrosoft Graphが呼び出せるパブリックHTTPS URL
- 録画＋STTのフォールバックが必要な場合は `ffmpeg` がインストールされていること

## ステップ1: Microsoft Graph認証情報を追加する

Graphのアプリ専用認証情報を `~/.hermes/.env` に追加します。

```bash
MSGRAPH_TENANT_ID=<tenant-id>
MSGRAPH_CLIENT_ID=<client-id>
MSGRAPH_CLIENT_SECRET=<client-secret>
```

これらの認証情報は次で使われます。
- Graphクライアントの基盤
- サブスクリプション保守コマンド
- ミーティングの解決とアーティファクトの取得
- 専用のTeamsアクセストークンを提供しない場合の、GraphベースのTeamsアウトバウンド配信

## ステップ2: Graph webhookリスナーを有効化する

webhookリスナーは、`msgraph_webhook` という名前のゲートウェイプラットフォームです。最低限、それを有効化してクライアントステート値を設定します。

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<random-shared-secret>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

リスナーは次を公開します。
- `/msgraph/webhook` — Graph通知用
- `/health` — シンプルなヘルスチェック用

パブリックHTTPSエンドポイントをそのリスナーへルーティングする必要があります。例えば、パブリックドメインが `https://ops.example.com` の場合、Graphの通知URLは通常次のようになります。

```text
https://ops.example.com/msgraph/webhook
```

## ステップ3: Teamsの配信とパイプラインの挙動を設定する

ミーティングパイプラインは、既存の `teams` プラットフォームエントリからランタイム設定を読みます。パイプライン固有のノブは `teams.extra.meeting_pipeline` 配下に置かれます。Teamsのアウトバウンド配信は、通常のTeamsプラットフォーム設定面のままです。

`~/.hermes/config.yaml` の例:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      port: 8646
      client_state: "replace-me"
      accepted_resources:
        - "communications/onlineMeetings"

  teams:
    enabled: true
    extra:
      client_id: "your-teams-client-id"
      client_secret: "your-teams-client-secret"
      tenant_id: "your-teams-tenant-id"

      # アウトバウンド要約配信
      delivery_mode: "graph" # または incoming_webhook
      team_id: "team-id"
      channel_id: "channel-id"
      # incoming_webhook_url: "https://..."

      meeting_pipeline:
        transcript_min_chars: 80
        transcript_required: false
        transcription_fallback: true
        ffmpeg_extract_audio: true
        notion:
          enabled: false
        linear:
          enabled: false
```

## Teamsの配信モード

このパイプラインは、既存のTeamsプラグイン内で2つのTeams要約配信モードをサポートします。

### `incoming_webhook`

Graph経由でチャネルメッセージを作成せずに、Teamsへシンプルなwebhookポストを行いたい場合に使用します。

必要な設定:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "incoming_webhook"
      incoming_webhook_url: "https://..."
```

### `graph`

HermesにMicrosoft Graph経由でTeamsのチャットまたはチャネルへ要約をポストさせたい場合に使用します。

サポートされるターゲット:
- `chat_id`
- `team_id` + `channel_id`
- 既存のTeamsプラットフォーム向けの `team_id` + `home_channel` フォールバック

例:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "graph"
      team_id: "team-id"
      channel_id: "channel-id"
```

## ステップ4: ゲートウェイを起動する

設定を更新した後、通常どおりHermesを起動します。

```bash
hermes gateway run
```

または、HermesをDockerで実行している場合は、デプロイで既に行っているのと同じ方法でゲートウェイを起動します。

リスナーを確認します。

```bash
curl http://localhost:8646/health
```

## ステップ5: Graphサブスクリプションを作成する

プラグインのCLIを使ってサブスクリプションを作成・確認します。

例:

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllRecordings \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"
```

:::warning Graphサブスクリプションは72時間で期限切れになります

Microsoft Graphはwebhookサブスクリプションを72時間に制限しており、自動更新しません。本番稼働の前に `hermes teams-pipeline maintain-subscriptions` をスケジュールする必要があります。さもなければ、手動でサブスクリプションを作成してから3日後に通知が黙って止まります。オペレーター用ランブックの[サブスクリプション更新の自動化](/docs/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production)を参照してください — 3つの選択肢（Hermes cron、systemdタイマー、素のcrontab）があります。

:::

サブスクリプションの保守とday-2のオペレーターフローについては、ガイドに進んでください: [Teamsミーティングパイプラインの運用](/docs/guides/operate-teams-meeting-pipeline)。

## バリデーション

組み込みのバリデーションスナップショットを実行します。

```bash
hermes teams-pipeline validate
```

役立つ補助チェック:

```bash
hermes teams-pipeline token-health
hermes teams-pipeline subscriptions
```

## トラブルシューティング

| 問題 | 確認すべきこと |
|---------|---------------|
| Graph webhookのバリデーションが失敗する | パブリックURLが正しく到達可能であること、Graphが正確な `/msgraph/webhook` パスを呼び出していることを確認する |
| `hermes teams-pipeline list` にジョブが現れない | `msgraph_webhook` が有効化されており、サブスクリプションが正しい通知URLを指していることを確認する |
| トランスクリプト優先が一度も成功しない | トランスクリプトリソースに対するGraph権限と、そのミーティングにトランスクリプトアーティファクトが存在するかを確認する |
| 録画フォールバックが失敗する | `ffmpeg` がインストールされており、Graphアプリが録画アーティファクトにアクセスできることを確認する |
| Teams要約の配信が失敗する | `delivery_mode`、ターゲットID、Teams認証設定を再確認する |

## 関連ドキュメント

- [Microsoft Teamsボットのセットアップ](/docs/user-guide/messaging/teams)
- [Teamsミーティングパイプラインの運用](/docs/guides/operate-teams-meeting-pipeline)
