---
sidebar_position: 8
sidebar_label: "SMS（Twilio）"
title: "SMS（Twilio）"
description: "Twilio を使って Hermes Agent を SMS チャットボットとしてセットアップする"
---

# SMS セットアップ（Twilio）

Hermes は [Twilio](https://www.twilio.com/) API を通じて SMS に接続します。人々があなたの Twilio 電話番号にテキストを送ると、AI の応答が返ってきます — Telegram や Discord と同じ会話体験を、標準のテキストメッセージ越しに得られます。

:::info 共有クレデンシャル
SMS ゲートウェイは、オプションの [telephony スキル](/docs/reference/skills-catalog) とクレデンシャルを共有します。音声通話や一度限りの SMS のために既に Twilio をセットアップしている場合、ゲートウェイは同じ `TWILIO_ACCOUNT_SID`、`TWILIO_AUTH_TOKEN`、`TWILIO_PHONE_NUMBER` で動作します。
:::

---

## 前提条件

- **Twilio アカウント** — [twilio.com でサインアップ](https://www.twilio.com/try-twilio)（無料トライアルあり）
- **SMS 機能付きの Twilio 電話番号**
- **公開アクセス可能なサーバー** — SMS が到着すると Twilio がサーバーに Webhook を送信します
- **aiohttp** — `pip install 'hermes-agent[sms]'`

---

## ステップ 1: Twilio クレデンシャルを取得

1. [Twilio Console](https://console.twilio.com/) にアクセスします
2. ダッシュボードから **Account SID** と **Auth Token** をコピーします
3. **Phone Numbers → Manage → Active Numbers** にアクセスします — E.164 形式（例: `+15551234567`）であなたの電話番号をメモします

---

## ステップ 2: Hermes を設定

### インタラクティブセットアップ（推奨）

```bash
hermes gateway setup
```

プラットフォームリストから **SMS (Twilio)** を選択します。ウィザードがクレデンシャルの入力を求めます。

### 手動セットアップ

`~/.hermes/.env` に追加します:

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# セキュリティ: 特定の電話番号に制限（推奨）
SMS_ALLOWED_USERS=+15559876543,+15551112222

# オプション: cron ジョブ配信用のホームチャンネルを設定
SMS_HOME_CHANNEL=+15559876543
```

---

## ステップ 3: Twilio Webhook を設定

Twilio は、受信メッセージをどこに送るかを知る必要があります。[Twilio Console](https://console.twilio.com/) で:

1. **Phone Numbers → Manage → Active Numbers** にアクセスします
2. 電話番号をクリックします
3. **Messaging → A MESSAGE COMES IN** の下で、次を設定します:
   - **Webhook**: `https://your-server:8080/webhooks/twilio`
   - **HTTP Method**: `POST`

:::tip Webhook の公開
Hermes をローカルで実行している場合は、トンネルを使って Webhook を公開します:

```bash
# cloudflared を使用
cloudflared tunnel --url http://localhost:8080

# ngrok を使用
ngrok http 8080
```

結果として得られた公開 URL を Twilio Webhook として設定します。
:::

**`SMS_WEBHOOK_URL` を、Twilio で設定したのと同じ URL に設定してください。** これは Twilio の署名検証に必要です — アダプターはこれがないと起動を拒否します:

```bash
# Twilio Console の Webhook URL と一致する必要がある
SMS_WEBHOOK_URL=https://your-server:8080/webhooks/twilio
```

Webhook ポートはデフォルトで `8080` です。次で上書きします:

```bash
SMS_WEBHOOK_PORT=3000
```

---

## ステップ 4: ゲートウェイを起動

```bash
hermes gateway
```

次のように表示されるはずです:

```
[sms] Twilio webhook server listening on 127.0.0.1:8080, from: +1555***4567
```

`Refusing to start: SMS_WEBHOOK_URL is required` と表示される場合は、`SMS_WEBHOOK_URL` を Twilio Console で設定した公開 URL に設定してください（ステップ 3 を参照）。

Twilio の番号にテキストを送ると — Hermes が SMS で応答します。

---

## 環境変数

| 変数 | 必須 | 説明 |
|----------|----------|------|
| `TWILIO_ACCOUNT_SID` | はい | Twilio Account SID（`AC` で始まる） |
| `TWILIO_AUTH_TOKEN` | はい | Twilio Auth Token（Webhook 署名検証にも使用） |
| `TWILIO_PHONE_NUMBER` | はい | あなたの Twilio 電話番号（E.164 形式） |
| `SMS_WEBHOOK_URL` | はい | Twilio 署名検証用の公開 URL — Twilio Console の Webhook URL と一致する必要がある |
| `SMS_WEBHOOK_PORT` | いいえ | Webhook リスナーポート（デフォルト: `8080`） |
| `SMS_WEBHOOK_HOST` | いいえ | Webhook バインドアドレス（デフォルト: `0.0.0.0`） |
| `SMS_INSECURE_NO_SIGNATURE` | いいえ | `true` に設定すると署名検証を無効化（ローカル開発のみ — **本番では使用しない**） |
| `SMS_ALLOWED_USERS` | いいえ | チャットを許可する E.164 電話番号のカンマ区切り |
| `SMS_ALLOW_ALL_USERS` | いいえ | `true` に設定すると誰でも許可（非推奨） |
| `SMS_HOME_CHANNEL` | いいえ | cron ジョブ / 通知配信用の電話番号 |
| `SMS_HOME_CHANNEL_NAME` | いいえ | ホームチャンネルの表示名（デフォルト: `Home`） |

---

## SMS 固有の挙動

- **プレーンテキストのみ** — SMS は Markdown をリテラル文字としてレンダリングするため、Markdown は自動的に除去されます
- **1600 文字の制限** — より長い応答は、自然な区切り（改行、次にスペース）で複数のメッセージに分割されます
- **エコー防止** — 自分の Twilio 番号からのメッセージは、ループを防ぐために無視されます
- **電話番号の伏せ字化** — プライバシーのため、電話番号はログ内で伏せ字化されます

---

## セキュリティ

### Webhook 署名検証

Hermes は、`X-Twilio-Signature` ヘッダー（HMAC-SHA1）を検証することで、受信 Webhook が本当に Twilio から発信されたものであることを検証します。これにより、攻撃者が偽造したメッセージを注入するのを防ぎます。

**`SMS_WEBHOOK_URL` は必須です。** Twilio Console で設定した公開 URL に設定してください。アダプターはこれがないと起動を拒否します。

公開 URL のないローカル開発では、検証を無効化できます:

```bash
# ローカル開発のみ — 本番では使用しない
SMS_INSECURE_NO_SIGNATURE=true
```

### ユーザー許可リスト

**ゲートウェイはデフォルトですべてのユーザーを拒否します。** 許可リストを設定してください:

```bash
# 推奨: 特定の電話番号に制限
SMS_ALLOWED_USERS=+15559876543,+15551112222

# またはすべてを許可（ターミナルアクセスを持つボットには非推奨）
SMS_ALLOW_ALL_USERS=true
```

:::warning
SMS には組み込みの暗号化がありません。セキュリティ上の影響を理解していない限り、機密性の高い操作に SMS を使用しないでください。機密性の高いユースケースには、Signal または Telegram を優先してください。
:::

---

## トラブルシューティング

### メッセージが届かない

1. Twilio Webhook URL が正しく、公開アクセス可能であることを確認します
2. `TWILIO_ACCOUNT_SID` と `TWILIO_AUTH_TOKEN` が正しいことを検証します
3. Twilio Console → **Monitor → Logs → Messaging** で配信エラーを確認します
4. あなたの電話番号が `SMS_ALLOWED_USERS` に含まれている（または `SMS_ALLOW_ALL_USERS=true`）ことを確認します

### 返信が送信されない

1. `TWILIO_PHONE_NUMBER` が正しく設定されている（`+` 付きの E.164 形式）ことを確認します
2. Twilio アカウントに SMS 対応の番号があることを検証します
3. Hermes ゲートウェイのログで Twilio API エラーを確認します

### Webhook ポートの競合

ポート 8080 が既に使用中の場合は、変更します:

```bash
SMS_WEBHOOK_PORT=3001
```

一致するように Twilio Console の Webhook URL を更新します。
