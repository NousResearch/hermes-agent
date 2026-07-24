---
title: "Teamsミーティングパイプラインを運用する"
description: "Microsoft Teamsミーティングパイプラインのランブック、本番投入チェックリスト、オペレーター用ワークシート"
---

# Teamsミーティングパイプラインを運用する

このガイドは、すでに[Teamsミーティング](/docs/user-guide/messaging/teams-meetings)から機能を有効化した後で使用してください。

このページでは次の内容を扱います。
- オペレーターのCLIフロー
- 定期的なサブスクリプションのメンテナンス
- 障害の切り分け
- 本番投入チェック
- ロールアウト用ワークシート

## 主要なオペレーターコマンド

### 設定スナップショットを検証する

```bash
hermes teams-pipeline validate
```

設定変更後は、まずこれを使用してください。

### トークンの健全性を確認する

```bash
hermes teams-pipeline token-health
hermes teams-pipeline token-health --force-refresh
```

認証状態が古くなっている疑いがある場合は `--force-refresh` を使用してください。

### サブスクリプションを確認する

```bash
hermes teams-pipeline subscriptions
```

### 有効期限が近いサブスクリプションを更新する

```bash
hermes teams-pipeline maintain-subscriptions
hermes teams-pipeline maintain-subscriptions --dry-run
```

### サブスクリプション更新の自動化（本番環境では必須） {#automating-subscription-renewal-required-for-production}

**Microsoft Graphのサブスクリプションは最長でも72時間で期限切れになります。** 何も更新しないと、ミーティング通知は3日後に静かに停止し、パイプラインが「壊れた」ように見えます。これは、あらゆるGraphベースの連携で最も多い運用上の障害モードです。

`maintain-subscriptions` をスケジュールで実行する必要があります。次の3つのオプションのいずれかを選択してください。

#### オプション1: Hermes cron（すでにHermesゲートウェイを稼働している場合に推奨）

Hermesには組み込みのcronスケジューラーが付属しています。`--no-agent` モードはジョブとして（LLMを使わずに）スクリプトを実行し、`--script` は `~/.hermes/scripts/` 配下のファイルを指す必要があります。まずスクリプトを作成します。

```bash
mkdir -p ~/.hermes/scripts
cat > ~/.hermes/scripts/maintain-teams-subscriptions.sh <<'EOF'
#!/usr/bin/env bash
exec hermes teams-pipeline maintain-subscriptions
EOF
chmod +x ~/.hermes/scripts/maintain-teams-subscriptions.sh
```

次に、12時間ごとに実行されるスクリプト専用のcronジョブを登録します（72時間の有効期限に対して6倍の余裕を持たせます）。

```bash
hermes cron create "0 */12 * * *" \
  --name "teams-pipeline-maintain-subscriptions" \
  --no-agent \
  --script maintain-teams-subscriptions.sh \
  --deliver local
```

登録されたことを確認し、次回の実行時刻を確認します。

```bash
hermes cron list
hermes cron status        # スケジューラーの状態
```

#### オプション2: systemdタイマー（Linuxの本番デプロイに推奨）

`/etc/systemd/system/hermes-teams-pipeline-maintain.service` を作成します。

```ini
[Unit]
Description=Hermes Teams pipeline subscription maintenance
After=network-online.target

[Service]
Type=oneshot
User=hermes
EnvironmentFile=/etc/hermes/env
ExecStart=/usr/local/bin/hermes teams-pipeline maintain-subscriptions
```

そして `/etc/systemd/system/hermes-teams-pipeline-maintain.timer` を作成します。

```ini
[Unit]
Description=Run Hermes Teams pipeline subscription maintenance every 12 hours

[Timer]
OnBootSec=5min
OnUnitActiveSec=12h
Persistent=true

[Install]
WantedBy=timers.target
```

有効化します。

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-teams-pipeline-maintain.timer
systemctl list-timers hermes-teams-pipeline-maintain.timer
```

#### オプション3: 通常のcrontab

```cron
0 */12 * * * /usr/local/bin/hermes teams-pipeline maintain-subscriptions >> /var/log/hermes/teams-pipeline-maintain.log 2>&1
```

cron環境に `MSGRAPH_*` の認証情報があることを確認してください。最も簡単な解決策は、crontabが呼び出すラッパースクリプトの先頭で `~/.hermes/.env` を読み込むことです。

#### 更新が機能していることを確認する

スケジュールを設定したら、最初のスケジュール実行の後に更新アクティビティを確認します。

```bash
hermes teams-pipeline subscriptions   # expirationDateTime が前進しているはず
hermes teams-pipeline maintain-subscriptions --dry-run   # ほとんどの場合 "0 expiring soon" と表示されるはず
```

Graph webhookがちょうど約72時間後に不可解に「動作を停止する」場合、最初に確認すべきはこれです。更新ジョブは実際に実行されたか？

### 最近のジョブを確認する

```bash
hermes teams-pipeline list
hermes teams-pipeline list --status failed
hermes teams-pipeline show <job-id>
```

### 保存されたジョブを再実行する

```bash
hermes teams-pipeline run <job-id>
```

### ミーティングアーティファクトの取得をドライランする

```bash
hermes teams-pipeline fetch --meeting-id <meeting-id>
hermes teams-pipeline fetch --join-web-url "<join-url>"
```

## 定期ランブック

### 初回セットアップ後

次のコマンドを順に実行します。

```bash
hermes teams-pipeline validate
hermes teams-pipeline token-health --force-refresh
hermes teams-pipeline subscriptions
```

その後、実際のミーティングイベントをトリガーするか待機して、次を確認します。

```bash
hermes teams-pipeline list
hermes teams-pipeline show <job-id>
```

### 日次または定期的なチェック

- `hermes teams-pipeline maintain-subscriptions --dry-run` を実行する
- `hermes teams-pipeline list --status failed` を確認する
- Teamsの配信先が正しいチャットまたはチャンネルのままであることを確認する

### webhook URLや配信先を変更する前に

- 公開通知URLまたはTeamsの配信先設定を更新する
- `hermes teams-pipeline validate` を実行する
- 影響を受けるサブスクリプションを更新または再作成する
- 新しいイベントが期待されるシンクに届くことを確認する

## 障害の切り分け

### ジョブが作成されない

確認事項:
- `msgraph_webhook` が有効になっている
- 公開通知URLが `/msgraph/webhook` を指している
- サブスクリプションのクライアントステートが `MSGRAPH_WEBHOOK_CLIENT_STATE` と一致している
- サブスクリプションがリモートにまだ存在し、期限切れになっていない

### ジョブが要約処理の前に再試行のままになるか失敗する

確認事項:
- トランスクリプトの権限と可用性
- 録画の権限とアーティファクトの可用性
- 録画フォールバックが有効な場合の `ffmpeg` の可用性
- Graphトークンの健全性

### 要約は生成されるがTeamsに配信されない

確認事項:
- `platforms.teams.enabled: true`
- `delivery_mode`
- webhookモードの場合の `incoming_webhook_url`
- Graphモードの場合の `chat_id`、または `team_id` と `channel_id`
- Graph投稿を使用する場合のTeams認証設定

### 重複または予期しない再実行

確認事項:
- `hermes teams-pipeline run` でジョブを手動で再実行したかどうか
- そのミーティングのシンクレコードがすでに存在するかどうか
- ローカル設定で再送パスを意図的に有効化したかどうか

## 本番投入チェックリスト

- [ ] Graph認証情報が存在し、正しい
- [ ] `msgraph_webhook` が有効で、公開インターネットから到達可能
- [ ] `MSGRAPH_WEBHOOK_CLIENT_STATE` が設定され、サブスクリプションと一致している
- [ ] トランスクリプトのサブスクリプションが作成されている
- [ ] STTフォールバックが必要な場合、録画のサブスクリプションが作成されている
- [ ] 録画フォールバックが有効な場合、`ffmpeg` がインストールされている
- [ ] Teamsの送信配信先が設定され、検証済み
- [ ] NotionとLinearのシンクは、実際に必要な場合にのみ設定されている
- [ ] `hermes teams-pipeline validate` がOKのスナップショットを返す
- [ ] `hermes teams-pipeline token-health --force-refresh` が成功する
- [ ] **`maintain-subscriptions` がスケジュールされている**（Hermes cron、systemdタイマー、またはcrontab — [サブスクリプション更新の自動化](#automating-subscription-renewal-required-for-production)を参照）。これがないと、Graphのサブスクリプションは72時間以内に静かに期限切れになります。
- [ ] 実際のエンドツーエンドのミーティングイベントが、保存されたジョブを生成している
- [ ] 少なくとも1つの要約が意図した配信シンクに到達している

## 配信モードの判断ガイド

| モード | 使う場面 | トレードオフ |
|------|----------|----------|
| `incoming_webhook` | Teamsへの単純な投稿だけが必要な場合 | 最もシンプルなセットアップ、制御性は低い |
| `graph` | Graph経由でチャンネルまたはチャットに投稿する必要がある場合 | 制御性が高い、認証と配信先の設定が多い |

## オペレーター用ワークシート

ロールアウト前にこれを記入してください。

| 項目 | 値 |
|------|-------|
| 公開通知URL | |
| GraphテナントID | |
| GraphクライアントID | |
| Webhookクライアントステート | |
| トランスクリプトリソースのサブスクリプション | |
| 録画リソースのサブスクリプション | |
| Teams配信モード | |
| TeamsチャットIDまたはチーム/チャンネル | |
| NotionデータベースID | |
| LinearチームID | |
| ストアパスのオーバーライド（ある場合） | |
| 日次チェックの担当者 | |

## 変更レビュー用ワークシート

デプロイを変更する前にこれを使用してください。

| 質問 | 回答 |
|----------|--------|
| 公開webhook URLを変更しますか？ | |
| Graph認証情報をローテーションしますか？ | |
| Teams配信モードを変更しますか？ | |
| 新しいTeamsチャットまたはチャンネルに移行しますか？ | |
| サブスクリプションを再作成または更新する必要がありますか？ | |
| 新たなエンドツーエンドの検証実行が必要ですか？ | |

## 関連ドキュメント

- [Teamsミーティングのセットアップ](/docs/user-guide/messaging/teams-meetings)
- [Microsoft Teamsボットのセットアップ](/docs/user-guide/messaging/teams)
