---
sidebar_position: 17
title: "LINE"
description: "Hermes Agent を LINE Messaging API ボットとしてセットアップ"
---

# LINE セットアップ

公式の LINE Messaging API を通じて、Hermes Agent を [LINE](https://line.me/) ボットとして実行します。アダプターは `plugins/platforms/line/` 配下のバンドルプラットフォームプラグインとして存在します — コアの編集は不要で、他のプラットフォームと同様に有効化するだけです。

LINE は日本、台湾、タイで圧倒的に普及しているメッセージングアプリです。ユーザーがそれらの地域にいるなら、これが彼らにリーチする方法です。

## ボットの応答方法

| コンテキスト | 動作 |
|---------|----------|
| **1:1 チャット**（`U` ID） | すべてのメッセージに応答 |
| **グループチャット**（`C` ID） | グループが許可リストにある場合に応答 |
| **複数人ルーム**（`R` ID） | ルームが許可リストにある場合に応答 |

受信したテキスト、画像、音声、動画、ファイル、スタンプ、位置情報はすべて処理されます。送信テキストは**まず無料の reply トークン**（単回使用、約 60 秒のウィンドウ）を使用し、トークンが期限切れになると従量課金の Push API にフォールバックします。

---

## ステップ 1: LINE Messaging API チャネルを作成する

1. [LINE Developers Console](https://developers.line.biz/console/) にアクセスします。
2. Provider を作成し、その配下に **Messaging API** チャネルを作成します。
3. チャネルの **Basic settings** タブから **Channel secret** をコピーします。
4. **Messaging API** タブで **Channel access token (long-lived)** までスクロールし、**Issue** をクリックします。トークンをコピーします。
5. **Messaging API** タブで、**Auto-reply messages** と **Greeting messages** も無効化し、ボットの返信と競合しないようにします。

---

## ステップ 2: Webhook ポートを公開する

LINE は公開 HTTPS 経由で Webhook を配信します。デフォルトポートは `8646` です — 必要に応じて `LINE_PORT` で上書きします。

```bash
# Cloudflare Tunnel（本番環境推奨 — 固定ホスト名）
cloudflared tunnel --url http://localhost:8646

# ngrok（開発に最適）
ngrok http 8646

# devtunnel
devtunnel create hermes-line --allow-anonymous
devtunnel port create hermes-line -p 8646 --protocol https
devtunnel host hermes-line
```

`https://...` の URL をコピーします — 以下で Webhook URL として設定します。テスト中は**トンネルを起動したままにしてください**。本番環境では、再起動しても Webhook URL が変わらないよう、固定の Cloudflare 名前付きトンネルを設定してください。

---

## ステップ 3: Hermes を設定する

`~/.hermes/.env` に追加します:

```env
LINE_CHANNEL_ACCESS_TOKEN=YOUR_LONG_LIVED_TOKEN
LINE_CHANNEL_SECRET=YOUR_CHANNEL_SECRET

# 許可リスト — 少なくともこれらのいずれか（または開発用に LINE_ALLOW_ALL_USERS=true）
LINE_ALLOWED_USERS=U1234567890abcdef...           # カンマ区切りの U で始まる ID
LINE_ALLOWED_GROUPS=C1234567890abcdef...          # 任意のグループ ID
LINE_ALLOWED_ROOMS=R1234567890abcdef...           # 任意のルーム ID

# 画像 / 音声 / 動画の送信に必須 — トンネルが解決する公開 HTTPS ベース URL。
# これがないと、send_image/voice/video は拒否されます。
LINE_PUBLIC_URL=https://my-tunnel.example.com
```

次に `~/.hermes/config.yaml` で:

```yaml
gateway:
  platforms:
    line:
      enabled: true
```

これで十分です — `gateway/config.py` のバンドルプラグインスキャンが `plugins/platforms/line/` を自動的に拾います。`Platform.LINE` enum の編集も、`_create_adapter` の登録も不要です。

---

## ステップ 4: Webhook URL を設定する

LINE コンソールに戻ります:

1. チャネルを開く → **Messaging API** タブ。
2. **Webhook settings** → **Webhook URL** に `https://<your-tunnel>/line/webhook` を貼り付けます（`/line/webhook` のパスに注意 — アダプターはそこでリッスンします）。
3. **Verify** をクリックします。LINE が URL に ping を送り、200 が表示されるはずです。
4. **Use webhook** を **On** に切り替えます。

---

## ステップ 5: ゲートウェイを実行する

```bash
hermes gateway
```

エージェントのログには次が表示されます:

```
LINE: webhook listening on 0.0.0.0:8646/line/webhook (public: https://my-tunnel.example.com)
```

LINE アプリからボットを友だち追加し（チャネルの **Messaging API** タブの QR をスキャン）、メッセージを送信します。

---

## 遅い LLM 応答

LINE の reply トークンは単回使用で、受信イベントからおよそ 60 秒後に期限切れになります。遅い LLM は時間内に返信できず、通常であれば有料の Push API 呼び出しを強いられます。

LLM が `LINE_SLOW_RESPONSE_THRESHOLD` 秒（デフォルト `45`）を過ぎてもまだ実行中の場合、アダプターは元の reply トークンを消費して **Template Buttons** のバブルを送信します:

> 🤔 Still thinking. Tap below to fetch the answer when it's ready.
>
> [ Get answer ]

ユーザーは都合のよいときに **Get answer** をタップします — そのポストバックが*新しい* reply トークンを届け、アダプターはそれを使ってキャッシュ済みの回答を送信します（これも無料）。

ステートマシン: `PENDING → READY → DELIVERED`、加えてキャンセルされた実行のための `ERROR`（孤立した PENDING は、`/stop` 後に「Run was interrupted before completion.」へ解決され、永続ボタンがループしないようにします）。

ポストバックボタンを無効化し、常に Push フォールバックにするには:

```env
LINE_SLOW_RESPONSE_THRESHOLD=0
```

ポストバックフローを確実に発火させるには、閾値より前に reply トークンを消費してしまうチャタを抑制します:

```yaml
# ~/.hermes/config.yaml
display:
  interim_assistant_messages: false
  platforms:
    line:
      tool_progress: off
```

---

## Cron / 通知の配信

```env
LINE_HOME_CHANNEL=Uxxxxxxxxxxxxxxxxxxxx     # デフォルトの配信先
```

`deliver: line` の Cron ジョブは `LINE_HOME_CHANNEL` にルーティングされます。アダプターは Push 専用のスタンドアロン送信機を備えているため、Cron がゲートウェイとは別プロセスで実行されても Cron ジョブが機能します。

---

## 環境変数リファレンス

| 変数 | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `LINE_CHANNEL_ACCESS_TOKEN` | はい | — | 長期チャネルアクセストークン |
| `LINE_CHANNEL_SECRET` | はい | — | チャネルシークレット（HMAC-SHA256 Webhook 検証） |
| `LINE_HOST` | いいえ | `0.0.0.0` | Webhook のバインドホスト |
| `LINE_PORT` | いいえ | `8646` | Webhook のバインドポート |
| `LINE_PUBLIC_URL` | メディア用 | — | 公開 HTTPS ベース URL；画像/音声/動画の送信に必須 |
| `LINE_ALLOWED_USERS` | いずれか | — | カンマ区切りのユーザー ID（U で始まる） |
| `LINE_ALLOWED_GROUPS` | いずれか | — | カンマ区切りのグループ ID（C で始まる） |
| `LINE_ALLOWED_ROOMS` | いずれか | — | カンマ区切りのルーム ID（R で始まる） |
| `LINE_ALLOW_ALL_USERS` | 開発のみ | `false` | 許可リストを完全にスキップ |
| `LINE_HOME_CHANNEL` | いいえ | — | デフォルトの Cron / 通知配信先 |
| `LINE_SLOW_RESPONSE_THRESHOLD` | いいえ | `45` | ポストバックボタンが発火するまでの秒数（`0` = 無効） |
| `LINE_PENDING_TEXT` | いいえ | "🤔 Still thinking…" | ポストバックボタンと併せて表示されるバブルテキスト |
| `LINE_BUTTON_LABEL` | いいえ | "Get answer" | ボタンのラベル |
| `LINE_DELIVERED_TEXT` | いいえ | "Already replied ✅" | 配信済みのボタンが再度タップされたときの返信 |
| `LINE_INTERRUPTED_TEXT` | いいえ | "Run was interrupted before completion." | `/stop` で孤立したボタンがタップされたときの返信 |

---

## トラブルシューティング

**Webhook の verify で「invalid signature」。** `Channel secret` のコピーが間違っているか、トンネルがリクエストボディを書き換えています。まず `curl -i https://<tunnel>/line/webhook/health` で確認してください — `{"status":"ok","platform":"line"}` が返るはずです。

**グループでボットが何も受信しない。** `LINE_ALLOWED_GROUPS` に `C...` グループ ID が含まれているか確認してください。グループ ID を見つけるには、テストメッセージを送信し、`~/.hermes/logs/gateway.log` を `LINE: rejecting unauthorized source` で grep してください — 拒否された source の dict に ID が含まれています。

**`send_image` が「LINE_PUBLIC_URL must be set」で失敗する。** LINE の Messaging API はバイナリアップロードを受け付けません — 画像、音声、動画は到達可能な HTTPS URL でなければなりません。`LINE_PUBLIC_URL` をトンネルの公開ホスト名に設定すると、アダプターが `/line/media/<token>/<filename>` からファイルを自動的に配信します。

**ポストバックボタンが一向に表示されない。** LLM が `LINE_SLOW_RESPONSE_THRESHOLD` より速く応答したか、別のバブル（ツール進捗、ストリーミング）が先に reply トークンを消費したかのいずれかです。「遅い LLM 応答」セクションの抑制ブロックを参照してください。

**「already in use by another profile」。** 同じチャネルアクセストークンが、実行中の別の Hermes プロファイルにバインドされています。もう一方のゲートウェイを停止するか、別のチャネルを使用してください。

---

## 制限事項

* **チャンクごとに 1 バブル。** 各 LINE テキストバブルは 5000 文字に制限され、1 回の Reply/Push 呼び出しで最大 5 バブルまで送信されます。それより長い応答は省略記号で切り詰められます。
* **ネイティブのメッセージ編集なし。** LINE には編集 API がないため、ストリーミング応答は常に新しいバブルを送信し、以前のものを編集することはありません。
* **Markdown レンダリングなし。** 太字（`**`）、斜体（`*`）、コードフェンス、見出しはそのままの文字としてレンダリングされます。アダプターは送信前にそれらを除去します。URL は保持されます（`[label](url)` は `label (url)` になります）。
* **読み込みインジケーターは DM のみ。** LINE はグループとルームに対して chat/loading API を拒否するため、入力中インジケーターは 1:1 チャットでのみ表示されます。
