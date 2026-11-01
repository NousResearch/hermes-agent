---
sidebar_position: 23
title: "Microsoft Graph Webhook リスナー"
description: "Microsoft Graph の変更通知（会議、カレンダー、チャットなど）を Hermes で受信する"
---

# Microsoft Graph Webhook リスナー

`msgraph_webhook` ゲートウェイプラットフォームは、インバウンドのイベントリスナーです。これは Hermes が Microsoft Graph から **変更通知** を受信する手段です — 「Teams 会議が終了した」「このチャットに新しいメッセージが届いた」「このカレンダーイベントが更新された」といった通知です。`teams` プラットフォーム（ユーザーが入力するチャットボット）とは異なり、こちらは人間ではなく M365 が Hermes に何かが起こったことを伝えるものです。

現時点で主な利用者は Teams 会議サマリーパイプラインです。Graph は会議がトランスクリプトを生成したときに通知し、パイプラインがそれを取得し、Hermes が Teams にサマリーを投稿し返します。その他の Graph リソース（`/chats/.../messages`、`/users/.../events`）も同じリスナーを使用します — パイプラインのコンシューマーはそれぞれ独自の PR で導入されます。

## 前提条件

- Microsoft Graph アプリケーションの認証情報 — [Microsoft Graph アプリケーションの登録](../../guides/microsoft-graph-app-registration.md)
- Microsoft Graph が到達できる **公開 HTTPS URL**（Graph はプライベートエンドポイントを呼び出しません）。テストには開発トンネルが使えますが、本番環境では有効な証明書を持つ実際のドメインが必要です。
- `clientState` の値として使用する強力な共有シークレット。`openssl rand -hex 32` で生成し、`~/.hermes/.env` に `MSGRAPH_WEBHOOK_CLIENT_STATE` として設定します。

## クイックスタート

最小限の `~/.hermes/config.yaml`:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8646
      client_state: "replace-with-a-strong-secret"
      accepted_resources:
        - "communications/onlineMeetings"
```

または `~/.hermes/.env` の環境変数経由（起動時に自動マージされます）:

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<generate-with-openssl-rand-hex-32>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

注: バインドホストは `config.yaml` の `extra.host` から読み取られます（上の例を参照）。`MSGRAPH_WEBHOOK_HOST` による環境変数での上書きはありません。

ゲートウェイを起動します: `hermes gateway run`。リスナーは以下を公開します:

- `POST /msgraph/webhook` — Graph からの変更通知
- `GET /msgraph/webhook?validationToken=...` — Graph サブスクリプション検証ハンドシェイク
- `GET /health` — 受理/重複カウンター付きのレディネスプローブ

リスナーを公開します（リバースプロキシ、開発トンネル、イングレス）。Graph サブスクリプション用の通知 URL は、公開 HTTPS オリジンの後ろに `/msgraph/webhook` を付けたものです:

```
https://ops.example.com/msgraph/webhook
```

## 設定

すべての設定は `platforms.msgraph_webhook.extra` の下に置きます:

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `host` | 未設定（デュアルスタック: すべてのインターフェース、IPv4+IPv6） | HTTP リスナーのバインドアドレス。ループバック以外へのバインドには `allowed_source_cidrs` が必要です。ループバック（`127.0.0.1` / `::1`）は、開発トンネル / リバースプロキシ構成で最も簡単な選択肢です。 |
| `port` | `8646` | バインドポート。 |
| `webhook_path` | `/msgraph/webhook` | Graph が POST する URL パス。 |
| `health_path` | `/health` | レディネスエンドポイント。 |
| `client_state` | — | Graph がすべての通知でエコーする共有シークレット。`hmac.compare_digest` で比較されます — `openssl rand -hex 32` で生成します。 |
| `accepted_resources` | `[]`（すべて受理） | Graph リソースパス/パターンの許可リスト。末尾の `*` は前方一致として機能します。先頭の `/` は許容されます。例: `["communications/onlineMeetings", "chats/*/messages"]`。 |
| `max_seen_receipts` | `5000` | 通知 ID の重複排除キャッシュサイズ。上限に達すると最も古いエントリが削除されます。 |
| `allowed_source_cidrs` | `[]` | ループバック以外へのバインドでは必須です。空のままにするのは、リスナーがループバックにバインドされ、ローカルのトンネル / リバースプロキシの背後にある場合のみにしてください。 |

ほとんどの設定には、ゲートウェイ起動時に設定にマージされる同等の環境変数（`MSGRAPH_WEBHOOK_*`）もあります（例外は `host` で、設定ファイルでのみ指定できます — 上の注を参照）— [環境変数リファレンス](../../reference/environment-variables.md#microsoft-graph-teams-meetings)を参照してください。

## セキュリティ強化

### clientState が主要な認証チェック

すべての Graph 通知には、サブスクリプションが登録した `clientState` 文字列が含まれています。リスナーは、タイミングセーフな比較を使用して、`clientState` が一致しない通知を拒否します。これは Microsoft が文書化しているメカニズムです — この値は強力な共有シークレットとして扱ってください。

`client_state` が設定されていない場合、リスナーは起動を拒否します。

### 送信元 IP の許可リスト化（本番デプロイメント）

本番環境では、Microsoft が公開している Graph Webhook の送信元 IP 範囲にリスナーを制限してください。Microsoft は [Office 365 IP アドレスおよび URL Web サービス](https://learn.microsoft.com/en-us/microsoft-365/enterprise/urls-and-ip-address-ranges) でエグレス範囲を文書化しています。以下のように設定します:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 0.0.0.0
      client_state: "..."
      allowed_source_cidrs:
        - "52.96.0.0/14"
        - "52.104.0.0/14"
        # ...現在の Microsoft 365 の "Common" + "Teams" カテゴリのエグレス範囲を追加します
```

または環境変数として:

```bash
MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS="52.96.0.0/14,52.104.0.0/14"
```

`0.0.0.0`、`::`、LAN IP などのループバック以外のホストに `allowed_source_cidrs` なしでバインドすると、起動時に拒否されます。同じマシン上で開発トンネルやリバースプロキシを使っている場合は、Hermes を `127.0.0.1` または `::1` にバインドし、許可リストは空のままにしてください。無効な CIDR 文字列は警告をログに記録して無視されます。**Microsoft の IP リストは四半期ごとに確認してください** — 変更されます。

### HTTPS 終端

リスナーはプレーン HTTP を話します。リバースプロキシ（Caddy、Nginx、Cloudflare Tunnel、AWS ALB）で TLS を終端し、ローカルネットワーク経由でリスナーにプロキシしてください。Graph は非 HTTPS エンドポイントへの配信を拒否するため、Graph 自体から暗号化されていないトラフィックがあなたに届く経路はありません。

### レスポンスの衛生管理

成功時、リスナーは空のボディとともに `202 Accepted` を返します — 内部カウンターはワイヤレスポンスには含まれません。オペレーターは `/health` 経由でカウントを確認できます。`/health` は Webhook パスと同じ送信元 IP ルールで保護されています。

ステータスコード表:

| 結果 | ステータス |
|---------|--------|
| 通知が受理または重複排除された | 202 |
| 検証ハンドシェイク（`validationToken` 付きの GET） | 200（トークンをエコー） |
| バッチ内のすべての項目が clientState で失敗 | 403 |
| 不正な JSON / `value` 配列の欠落 / 不明なリソース | 400 |
| 送信元 IP が許可リストにない | 403 |
| `validationToken` のない単独の GET | 400 |

## トラブルシューティング

| 問題 | 確認事項 |
|---------|---------------|
| Graph サブスクリプション検証が失敗する | 公開 URL が到達可能であること、`/msgraph/webhook` パスが一致していること、`validationToken` 付きの GET が 10 秒以内にトークンをそのまま `text/plain` としてエコーすること。 |
| 通知は POST されるが何も取り込まれない | `client_state` がサブスクリプション登録時の値と一致していること。値がずれた場合は `openssl rand -hex 32` を再実行して新しいサブスクリプションを作成してください。`accepted_resources` に Graph が送信しているリソースパスが含まれていることを確認してください。 |
| すべての通知が 403 になる | `clientState` の不一致（偽造、または異なる値で登録されたサブスクリプション）。`hermes teams-pipeline subscribe --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE" ...` でサブスクリプションを再作成してください（パイプラインランタイムの PR に付属します）。 |
| リスナーが `0.0.0.0` で起動を拒否する | `allowed_source_cidrs` に Microsoft の現在の Webhook エグレス範囲を設定するか、トンネルまたはリバースプロキシの背後で Hermes を `127.0.0.1` / `::1` にバインドしてください。 |
| リスナーは起動するが `curl http://localhost:8646/health` がハングする | ポートバインドの衝突。`ss -tlnp \| grep 8646` を確認し、必要に応じて `port:` を変更してください。 |
| Microsoft からの実際の Graph リクエストが 403 になる | 送信元 IP 許可リストが狭すぎます。現在の Microsoft エグレス範囲を含むようにリストを広げてください。まだトンネル経路を検証中の場合は、Hermes をループバックにバインドし、公開はトンネルに任せてください。 |

## 関連ドキュメント

- [Microsoft Graph アプリケーションの登録](../../guides/microsoft-graph-app-registration.md) — Azure アプリ登録の前提条件
- [環境変数 → Microsoft Graph](../../reference/environment-variables.md#microsoft-graph-teams-meetings) — 環境変数の完全なリスト
- [Microsoft Teams ボットのセットアップ](./teams.md) — ユーザーが Teams で Hermes とチャットできる別のプラットフォーム
