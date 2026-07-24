---
sidebar_position: 15
---

# WeCom コールバック（自社開発アプリ）

コールバック/Webhookモデルを使用して、Hermes を自社開発のエンタープライズアプリケーションとして WeCom（企業微信）に接続します。

:::info WeCom Bot と WeCom コールバックの違い
Hermes は 2 つの WeCom 連携モードをサポートしています:
- **[WeCom Bot](wecom.md)** — Bot 形式で、WebSocket 経由で接続します。セットアップがシンプルで、グループチャットでも動作します。
- **WeCom コールバック**（このページ） — 自社開発アプリで、暗号化された XML コールバックを受信します。ユーザーの WeCom サイドバーにファーストクラスのアプリとして表示されます。複数企業（マルチコープ）ルーティングをサポートします。
:::

## 仕組み

1. WeCom 管理コンソールで自社開発アプリケーションを登録します
2. WeCom が暗号化された XML を HTTP コールバックエンドポイントにプッシュします
3. Hermes がメッセージを復号し、エージェント用にキューに入れます
4. 即座に確認応答します（サイレント — ユーザーには何も表示されません）
5. エージェントがリクエストを処理します（通常 3〜30 分）
6. 返信は WeCom の `message/send` API 経由でプロアクティブに配信されます

## 前提条件

- 管理者アクセス権を持つ WeCom 企業アカウント
- `aiohttp` および `httpx` の Python パッケージ（デフォルトインストールに含まれています）
- コールバック URL 用の公開アクセス可能なサーバー（または ngrok のようなトンネル）

## セットアップ

### 1. WeCom で自社開発アプリを作成する

1. [WeCom 管理コンソール](https://work.weixin.qq.com/) → **アプリケーション** → **アプリを作成** に移動します
2. **Corp ID** をメモします（管理コンソールの上部に表示されています）
3. アプリ設定で **Corp Secret** を作成します
4. アプリの概要ページから **Agent ID** をメモします
5. **メッセージ受信** で、コールバック URL を設定します:
   - URL: `http://YOUR_PUBLIC_IP:8645/wecom/callback`
   - Token: ランダムなトークンを生成します（WeCom が提供します）
   - EncodingAESKey: キーを生成します（WeCom が提供します）

### 2. 環境変数を設定する

`.env` ファイルに追加します:

```bash
WECOM_CALLBACK_CORP_ID=your-corp-id
WECOM_CALLBACK_CORP_SECRET=your-corp-secret
WECOM_CALLBACK_AGENT_ID=1000002
WECOM_CALLBACK_TOKEN=your-callback-token
WECOM_CALLBACK_ENCODING_AES_KEY=your-43-char-aes-key

# オプション
WECOM_CALLBACK_HOST=0.0.0.0
WECOM_CALLBACK_PORT=8645
WECOM_CALLBACK_ALLOWED_USERS=user1,user2
```

### 3. ゲートウェイを起動する

```bash
hermes gateway
```

（`hermes gateway start` は、`hermes gateway install` で systemd/launchd サービスを登録した後にのみ使用してください。）

コールバックアダプターは、設定されたポートで HTTP サーバーを起動します。WeCom は GET リクエストでコールバック URL を検証し、その後 POST でメッセージの送信を開始します。

## 設定リファレンス

`config.yaml` の `platforms.wecom_callback.extra` の下にこれらを設定するか、環境変数を使用します:

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `corp_id` | — | WeCom 企業 Corp ID（必須） |
| `corp_secret` | — | 自社開発アプリの Corp Secret（必須） |
| `agent_id` | — | 自社開発アプリの Agent ID（必須） |
| `token` | — | コールバック検証トークン（必須） |
| `encoding_aes_key` | — | コールバック暗号化用の 43 文字の AES キー（必須） |
| `host` | `0.0.0.0` | HTTP コールバックサーバーのバインドアドレス |
| `port` | `8645` | HTTP コールバックサーバーのポート |
| `path` | `/wecom/callback` | コールバックエンドポイントの URL パス |

## マルチアプリルーティング

複数の自社開発アプリを運用するエンタープライズ（例: 異なる部門や子会社をまたぐ場合）では、`config.yaml` で `apps` リストを設定します:

```yaml
platforms:
  wecom_callback:
    enabled: true
    extra:
      host: "0.0.0.0"
      port: 8645
      apps:
        - name: "dept-a"
          corp_id: "ww_corp_a"
          corp_secret: "secret-a"
          agent_id: "1000002"
          token: "token-a"
          encoding_aes_key: "key-a-43-chars..."
        - name: "dept-b"
          corp_id: "ww_corp_b"
          corp_secret: "secret-b"
          agent_id: "1000003"
          token: "token-b"
          encoding_aes_key: "key-b-43-chars..."
```

ユーザーは企業間の衝突を防ぐために `corp_id:user_id` でスコープ化されます。ユーザーがメッセージを送信すると、アダプターはそのユーザーが属するアプリ（企業）を記録し、正しいアプリのアクセストークンを通じて返信をルーティングします。

## アクセス制御

アプリと対話できるユーザーを制限します:

```bash
# 特定のユーザーを許可リストに追加
WECOM_CALLBACK_ALLOWED_USERS=zhangsan,lisi,wangwu

# またはすべてのユーザーを許可
WECOM_CALLBACK_ALLOW_ALL_USERS=true
```

## エンドポイント

アダプターは以下を公開します:

| メソッド | パス | 用途 |
|--------|------|---------|
| GET | `/wecom/callback` | URL 検証ハンドシェイク（WeCom がセットアップ時に送信） |
| POST | `/wecom/callback` | 暗号化メッセージコールバック（WeCom がユーザーメッセージをここに送信） |
| GET | `/health` | ヘルスチェック — `{"status": "ok"}` を返します |

## 暗号化

すべてのコールバックペイロードは、EncodingAESKey を使用して AES-CBC で暗号化されます。アダプターは以下を処理します:

- **インバウンド**: XML ペイロードを復号し、SHA1 署名を検証します
- **アウトバウンド**: 返信はプロアクティブ API 経由で送信されます（暗号化されたコールバックレスポンスではありません）

暗号化の実装は、Tencent の公式 WXBizMsgCrypt SDK と互換性があります。

## 制限事項

- **ストリーミングなし** — 返信はエージェントの処理完了後に完全なメッセージとして届きます
- **タイピングインジケーターなし** — コールバックモデルはタイピングステータスをサポートしていません
- **テキストのみ** — 現在、入力はテキストメッセージをサポートしています。画像/ファイル/音声の入力はまだ実装されていません。エージェントは、WeCom プラットフォームのヒント（画像、ドキュメント、動画、音声）を通じて、アウトバウンドのメディア機能を認識しています。
- **レスポンスのレイテンシ** — エージェントセッションは 3〜30 分かかります。ユーザーは処理が完了したときに返信を確認できます
