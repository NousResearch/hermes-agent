---
sidebar_position: 16
title: "Yuanbao"
description: "WebSocket ゲートウェイ経由で Hermes Agent をエンタープライズメッセージングプラットフォーム Yuanbao に接続する"
---

# Yuanbao

Hermes を Tencent のエンタープライズメッセージングプラットフォーム
[Yuanbao](https://yuanbao.tencent.com/) に接続します。このアダプターはリアルタイムの
メッセージ配信に WebSocket ゲートウェイを使用し、ダイレクト（C2C）とグループの両方の会話を
サポートします。

:::info
Yuanbao は、主に Tencent 内およびエンタープライズ環境で使用されるエンタープライズメッセージング
プラットフォームです。リアルタイム通信に WebSocket、HMAC ベースの認証を使用し、画像、ファイル、
音声メッセージを含むリッチメディアをサポートします。
:::

## 前提条件

- ボット作成権限を持つ Yuanbao アカウント
- Yuanbao の APP_ID と APP_SECRET（プラットフォーム管理者から取得）
- Python パッケージ: `websockets` と `httpx`
- メディアサポート用: `aiofiles`

必要な依存関係をインストールします。

```bash
pip install websockets httpx aiofiles
```

## セットアップ

### 1. Yuanbao でボットを作成する

1. [https://yuanbao.tencent.com/](https://yuanbao.tencent.com/) から Yuanbao アプリを
   ダウンロードします
2. アプリ内で **PAI → My Bot** に移動し、新しいボットを作成します
3. ボットが作成されたら、**APP_ID** と **APP_SECRET** をコピーします

### 2. セットアップウィザードを実行する

Yuanbao を設定する最も簡単な方法は、インタラクティブなセットアップです。

```bash
hermes gateway setup
```

プロンプトが表示されたら **Yuanbao** を選択します。ウィザードは次を行います。

1. APP_ID を尋ねます
2. APP_SECRET を尋ねます
3. 設定を自動的に保存します

:::tip
WebSocket URL と API ドメインには、適切なデフォルトが組み込まれています。開始するには APP_ID と
APP_SECRET を指定するだけで済みます。
:::

### 3. 環境変数を設定する

初期セットアップの後、`~/.hermes/.env` でこれらの変数を確認します。

```bash
# Required
YUANBAO_APP_ID=your-app-id
YUANBAO_APP_SECRET=your-app-secret
YUANBAO_WS_URL=wss://api.yuanbao.example.com/ws
YUANBAO_API_DOMAIN=https://api.yuanbao.example.com

# Optional: ボットのアカウント ID（通常は sign-token から自動的に取得される）
# YUANBAO_BOT_ID=your-bot-id

# Optional: 内部ルーティング環境（例: test/staging/production）
# YUANBAO_ROUTE_ENV=production

# Optional: cron/通知用のホームチャネル（形式: direct:<account> または group:<group_code>）
YUANBAO_HOME_CHANNEL=direct:bot_account_id
YUANBAO_HOME_CHANNEL_NAME="Bot Notifications"

# Optional: アクセスを制限する（レガシー、きめ細かいポリシーは下記の「アクセス制御」を参照）
YUANBAO_ALLOWED_USERS=user_account_1,user_account_2
```

### 4. ゲートウェイを起動する

```bash
hermes gateway
```

アダプターは Yuanbao の WebSocket ゲートウェイに接続し、HMAC 署名を使って認証し、メッセージの
処理を開始します。

## 機能

- **WebSocket ゲートウェイ** — リアルタイムの双方向通信
- **HMAC 認証** — APP_ID/APP_SECRET によるセキュアなリクエスト署名
- **C2C メッセージング** — ユーザーとボットのダイレクトな会話
- **グループメッセージング** — グループチャットでの会話
- **メディアサポート** — COS（Cloud Object Storage）経由の画像、ファイル、音声メッセージ
- **Markdown フォーマット** — メッセージは Yuanbao のサイズ制限に合わせて自動的にチャンク分割される
- **メッセージの重複排除** — 同じメッセージの重複処理を防ぐ
- **ハートビート／キープアライブ** — WebSocket 接続の安定性を維持する
- **タイピングインジケーター** — エージェントが処理している間「typing…」ステータスを表示する
- **自動再接続** — WebSocket の切断を指数バックオフで処理する
- **グループ情報のクエリ** — グループの詳細とメンバー一覧を取得する
- **ステッカー／絵文字のサポート** — 会話で TIMFaceElem ステッカーと絵文字を送信する
- **自動 sethome** — 最初にボットにメッセージを送ったユーザーが自動的にホームチャネルの所有者になる
- **低速応答の通知** — エージェントが想定より時間がかかっている場合に待機メッセージを送信する

## 設定オプション

### チャット ID の形式

Yuanbao は、会話タイプに応じてプレフィックス付きの識別子を使用します。

| チャットタイプ | 形式 | 例 |
|-----------|--------|---------|
| ダイレクトメッセージ（C2C） | `direct:<account>` | `direct:user123` |
| グループメッセージ | `group:<group_code>` | `group:grp456` |

### メディアのアップロード

Yuanbao アダプターは、COS（Tencent Cloud Object Storage）経由のメディアアップロードを自動的に
処理します。

- **画像**: JPEG、PNG、GIF、WebP をサポート
- **ファイル**: 一般的なすべてのドキュメントタイプをサポート
- **音声**: WAV、MP3、OGG をサポート

メディア URL は、SSRF 攻撃を防ぐためにアップロード前に自動的に検証・ダウンロードされます。

## ホームチャネル

任意の Yuanbao チャット（DM またはグループ）で `/sethome` コマンドを使い、それを**ホームチャネル**
として指定します。スケジュールされたタスク（cron ジョブ）は、その結果をこのチャネルに配信します。

:::tip 自動 sethome
ホームチャネルが設定されていない場合、最初にボットにメッセージを送ったユーザーが自動的に
ホームチャネルの所有者になります。現在のホームチャネルがグループチャットの場合、最初の DM が
それをダイレクトチャネルに格上げします。
:::

`~/.hermes/.env` で手動で設定することもできます。

```bash
YUANBAO_HOME_CHANNEL=direct:user_account_id
# またはグループの場合:
# YUANBAO_HOME_CHANNEL=group:group_code
YUANBAO_HOME_CHANNEL_NAME="My Bot Updates"
```

### 例: ホームチャネルの設定

1. Yuanbao でボットとの会話を開始します
2. コマンドを送信します: `/sethome`
3. ボットが応答します: 「Home channel set to [chat_name] with ID [chat_id]. Cron jobs will deliver to this location.」
4. 今後の cron ジョブと通知は、このチャネルに送信されます

### 例: cron ジョブの配信

cron ジョブを作成します。

```bash
/cron "0 9 * * *" Check server status
```

スケジュールされた出力は、毎日午前 9 時に Yuanbao のホームチャネルに配信されます。

## 利用のヒント

### 会話の開始

Yuanbao でボットに任意のメッセージを送信します。

```
hello
```

ボットは同じ会話スレッドで応答します。

### 利用可能なコマンド

すべての標準的な Hermes コマンドが Yuanbao で動作します。

| コマンド | 説明 |
|---------|-------------|
| `/new` | 新しい会話を開始する |
| `/model [provider:model]` | モデルを表示または変更する |
| `/sethome` | このチャットをホームチャネルに設定する |
| `/status` | セッション情報を表示する |
| `/help` | 利用可能なコマンドを表示する |

### ファイルの送信

ボットにファイルを送るには、Yuanbao チャットで直接添付するだけです。ボットは添付ファイルを自動的に
ダウンロードして処理します。

添付ファイルにメッセージを含めることもできます。

```
Please analyze this document
```

### ファイルの受信

ボットにファイルの作成やエクスポートを依頼すると、ボットはそのファイルを Yuanbao チャットに直接
送信します。

## トラブルシューティング

### ボットはオンラインだがメッセージに応答しない

**原因**: WebSocket ハンドシェイク中に認証が失敗しました。

**対処**:
1. APP_ID と APP_SECRET が正しいことを確認します
2. WebSocket URL にアクセスできることを確認します
3. ボットアカウントが適切な権限を持っていることを確認します
4. ゲートウェイのログを確認します: `tail -f ~/.hermes/logs/gateway.log`

### 「Connection refused」エラー

**原因**: WebSocket URL に到達できないか、誤っています。

**対処**:
1. WebSocket URL の形式を確認します（`wss://` で始まるはずです）
2. Yuanbao API ドメインへのネットワーク接続を確認します
3. ファイアウォールが WebSocket 接続を許可していることを確認します
4. URL をテストします: `curl -I https://[YUANBAO_API_DOMAIN]`

### メディアのアップロードが失敗する

**原因**: COS 認証情報が無効か、メディアサーバーに到達できません。

**対処**:
1. API_DOMAIN が正しいことを確認します
2. ボットでメディアアップロード権限が有効になっていることを確認します
3. メディアファイルがアクセス可能で破損していないことを確認します
4. プラットフォーム管理者に COS バケットの設定を確認します

### メッセージがホームチャネルに配信されない

**原因**: ホームチャネル ID の形式が誤っているか、cron ジョブがトリガーされていません。

**対処**:
1. YUANBAO_HOME_CHANNEL が正しい形式であることを確認します
2. `/sethome` コマンドで正しい形式を自動検出してテストします
3. `/status` で cron ジョブのスケジュールを確認します
4. ボットが対象のチャットで送信権限を持っていることを確認します

### 頻繁な切断

**原因**: WebSocket 接続が不安定か、ネットワークが信頼できません。

**対処**:
1. ゲートウェイのログでエラーパターンを確認します
2. 接続設定でハートビートタイムアウトを増やします
3. Yuanbao API への安定したネットワーク接続を確保します
4. 詳細ログの有効化を検討します: `HERMES_LOG_LEVEL=debug`

## アクセス制御

Yuanbao は、DM とグループの両方の会話に対してきめ細かいアクセス制御をサポートします。

```bash
# DM ポリシー: open（デフォルト）| allowlist | disabled
YUANBAO_DM_POLICY=open
# ボットへの DM を許可するユーザー ID のカンマ区切り（DM_POLICY=allowlist の場合のみ使用）
YUANBAO_DM_ALLOW_FROM=user_id_1,user_id_2

# グループポリシー: open（デフォルト）| allowlist | disabled
YUANBAO_GROUP_POLICY=open
# 許可するグループコードのカンマ区切り（GROUP_POLICY=allowlist の場合のみ使用）
YUANBAO_GROUP_ALLOW_FROM=group_code_1,group_code_2
```

これらは `config.yaml` でも設定できます。

```yaml
platforms:
  yuanbao:
    extra:
      dm_policy: allowlist
      dm_allow_from: "user1,user2"
      group_policy: open
      group_allow_from: ""
```

## 高度な設定

### メッセージのチャンク分割

Yuanbao には最大メッセージサイズがあります。Hermes は、Markdown を考慮した分割で大きな応答を
自動的にチャンク分割します（コードフェンス、テーブル、段落の境界を尊重します）。

### 接続パラメータ

以下の接続パラメータは、適切なデフォルトとともにアダプターに組み込まれています。

| パラメータ | デフォルト値 | 説明 |
|-----------|---------------|-------------|
| WebSocket 接続タイムアウト | 15 秒 | WS ハンドシェイクを待つ時間 |
| ハートビート間隔 | 30 秒 | 接続を維持するための ping の頻度 |
| 最大再接続試行回数 | 100 | 再接続の試行回数の上限 |
| 再接続バックオフ | 1s → 60s（指数） | 再接続試行間の待機時間 |
| 返信ハートビート間隔 | 2 秒 | RUNNING ステータスの送信頻度 |
| 送信タイムアウト | 30 秒 | 送信 WS メッセージのタイムアウト |

:::note
これらの値は現在、環境変数では設定できません。典型的な Yuanbao デプロイ向けに最適化されています。
:::

### 詳細ログ

接続の問題をトラブルシューティングするために、デバッグログを有効にします。

```bash
HERMES_LOG_LEVEL=debug hermes gateway
```

## 他の機能との統合

### cron ジョブ

Yuanbao で実行されるタスクをスケジュールします。

```
/cron "0 */4 * * *" Report system health
```

結果はホームチャネルに配信されます。

### バックグラウンドタスク

会話をブロックせずに長時間の操作を実行します。

```
/background Analyze all files in the archive
```

### クロスプラットフォームメッセージ

CLI から Yuanbao にメッセージを送信します。

```bash
hermes chat -q "Send 'Hello from CLI' to yuanbao:group:group_code"
```

## 関連ドキュメント

- [メッセージングゲートウェイの概要](./index.md)
- [スラッシュコマンドリファレンス](/docs/reference/slash-commands.md)
- [cron ジョブ](/docs/user-guide/features/cron.md)
- [バックグラウンドセッション](/docs/user-guide/cli#background-sessions)
