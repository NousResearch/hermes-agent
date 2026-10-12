---
sidebar_position: 14
title: "WeCom（企業WeChat）"
description: "AI Bot WebSocketゲートウェイ経由でHermes AgentをWeComに接続する"
---

# WeCom（企業WeChat）

Hermesを、Tencentのエンタープライズメッセージングプラットフォームである[WeCom](https://work.weixin.qq.com/)（企业微信）に接続します。このアダプターは、リアルタイム双方向通信のためにWeComのAI Bot WebSocketゲートウェイを使用します。公開エンドポイントやWebhookは不要です。

## 前提条件

- WeCom組織アカウント
- WeCom管理コンソールで作成したAI Bot
- ボットの認証情報ページから取得したBot IDとSecret
- Pythonパッケージ: `aiohttp` および `httpx`

## セットアップ

### ステップ1: AI Botを作成する

#### 推奨: スキャンによる作成（コマンド1つ）

```bash
hermes gateway setup
```

**WeCom** を選択し、WeComモバイルアプリでQRコードをスキャンします。Hermesが適切な権限を持つボットアプリケーションを自動的に作成し、認証情報を保存します。

セットアップウィザードは次の処理を行います。
1. ターミナルにQRコードを表示する
2. WeComモバイルアプリでのスキャンを待機する
3. Bot IDとSecretを自動的に取得する
4. アクセス制御の設定をガイドする

#### 代替方法: 手動セットアップ

スキャンによる作成が利用できない場合、ウィザードは手動入力にフォールバックします。

1. [WeCom管理コンソール](https://work.weixin.qq.com/wework_admin/frame)にログインします
2. **Applications** → **Create Application** → **AI Bot** に移動します
3. ボットの名前と説明を設定します
4. 認証情報ページから **Bot ID** と **Secret** をコピーします
5. `hermes gateway setup` を実行し、**WeCom** を選択して、プロンプトに従って認証情報を入力します

:::warning
Bot Secretは非公開にしてください。これを知っている人は誰でもあなたのボットになりすますことができます。
:::

### ステップ2: Hermesを設定する

#### オプションA: 対話的セットアップ（推奨）

```bash
hermes gateway setup
```

**WeCom** を選択し、プロンプトに従います。ウィザードは次の内容をガイドします。
- ボットの認証情報（QRスキャンまたは手動入力）
- アクセス制御の設定（許可リスト、ペアリングモード、またはオープンアクセス）
- 通知用のホームチャンネル

#### オプションB: 手動設定

`~/.hermes/.env` に次の内容を追加します。

```bash
WECOM_BOT_ID=your-bot-id
WECOM_SECRET=your-secret

# 任意: アクセスを制限する
WECOM_ALLOWED_USERS=user_id_1,user_id_2

# 任意: cron/通知用のホームチャンネル
WECOM_HOME_CHANNEL=chat_id
```

### ステップ3: ゲートウェイを起動する

```bash
hermes gateway
```

## 機能

- **WebSocketトランスポート** — 永続接続。公開エンドポイント不要
- **DMおよびグループメッセージング** — 設定可能なアクセスポリシー
- **グループ単位の送信者許可リスト** — 各グループで誰が操作できるかをきめ細かく制御
- **メディアサポート** — 画像、ファイル、音声、動画のアップロードとダウンロード
- **AES暗号化メディア** — 受信添付ファイルの自動復号
- **引用コンテキスト** — 返信スレッドを保持
- **Markdownレンダリング** — リッチテキスト応答
- **返信モードストリーミング** — 応答を受信メッセージのコンテキストに対応付け
- **自動再接続** — 接続切断時の指数バックオフ

## 設定オプション

これらは `config.yaml` の `platforms.wecom.extra` の下に設定します。

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `bot_id` | — | WeCom AI Bot ID（必須） |
| `secret` | — | WeCom AI Bot Secret（必須） |
| `websocket_url` | `wss://openws.work.weixin.qq.com` | WebSocketゲートウェイURL |
| `dm_policy` | `open` | DMアクセス: `open`、`allowlist`、`disabled`、`pairing` |
| `group_policy` | `open` | グループアクセス: `open`、`allowlist`、`disabled` |
| `allow_from` | `[]` | DMを許可するユーザーID（dm_policy=allowlistの場合） |
| `group_allow_from` | `[]` | 許可するグループID（group_policy=allowlistの場合） |
| `groups` | `{}` | グループ単位の設定（下記参照） |

## アクセスポリシー

### DMポリシー

ボットにダイレクトメッセージを送信できる人を制御します。

| 値 | 動作 |
|-------|----------|
| `open` | 誰でもボットにDMを送信できる（デフォルト） |
| `allowlist` | `allow_from` に含まれるユーザーIDのみDM可能 |
| `disabled` | すべてのDMを無視する |
| `pairing` | ペアリングモード（初期セットアップ用） |

```bash
WECOM_DM_POLICY=allowlist
```

### グループポリシー

ボットが応答するグループを制御します。

| 値 | 動作 |
|-------|----------|
| `open` | ボットはすべてのグループで応答する（デフォルト） |
| `allowlist` | ボットは `group_allow_from` に列挙されたグループIDでのみ応答する |
| `disabled` | すべてのグループメッセージを無視する |

```bash
WECOM_GROUP_POLICY=allowlist
```

### グループ単位の送信者許可リスト

きめ細かく制御するために、特定のグループ内でボットと対話できるユーザーを制限できます。これは `config.yaml` で設定します。

```yaml
platforms:
  wecom:
    enabled: true
    extra:
      bot_id: "your-bot-id"
      secret: "your-secret"
      group_policy: "allowlist"
      group_allow_from:
        - "group_id_1"
        - "group_id_2"
      groups:
        group_id_1:
          allow_from:
            - "user_alice"
            - "user_bob"
        group_id_2:
          allow_from:
            - "user_charlie"
        "*":
          allow_from:
            - "user_admin"
```

**動作の仕組み:**

1. `group_policy` と `group_allow_from` の設定が、そのグループがそもそも許可されるかどうかを決定します。
2. グループがトップレベルのチェックを通過した場合、`groups.<group_id>.allow_from` リスト（存在する場合）が、そのグループ内でボットと対話できる送信者をさらに制限します。
3. ワイルドカード `"*"` のグループエントリは、明示的に列挙されていないグループのデフォルトとして機能します。
4. 許可リストのエントリは、すべてのユーザーを許可する `*` ワイルドカードをサポートし、エントリは大文字小文字を区別しません。
5. エントリには任意で `wecom:user:` または `wecom:group:` のプレフィックス形式を使用できます。プレフィックスは自動的に取り除かれます。

グループに `allow_from` が設定されていない場合、（グループ自体がトップレベルのポリシーチェックを通過することを前提に）そのグループ内のすべてのユーザーが許可されます。

## メディアサポート

### 受信（受け取り）

アダプターはユーザーからメディア添付ファイルを受信し、エージェント処理のためにローカルにキャッシュします。

| タイプ | 処理方法 |
|------|-----------------|
| **画像** | ダウンロードしてローカルにキャッシュします。URLベースの画像とbase64エンコードされた画像の両方をサポートします。 |
| **ファイル** | ダウンロードしてキャッシュします。ファイル名は元のメッセージから保持されます。 |
| **音声** | 利用可能であれば、音声メッセージのテキスト文字起こしを抽出します。 |
| **混在メッセージ** | WeComの混在タイプメッセージ（テキスト + 画像）が解析され、すべてのコンポーネントが抽出されます。 |

**引用メッセージ:** 引用された（返信先の）メッセージのメディアも抽出されるため、エージェントはユーザーが何に返信しているかのコンテキストを把握できます。

### AES暗号化メディアの復号

WeComは一部の受信メディア添付ファイルをAES-256-CBCで暗号化します。アダプターはこれを自動的に処理します。

- 受信メディア項目に `aeskey` フィールドが含まれる場合、アダプターは暗号化されたバイト列をダウンロードし、PKCS#7パディングを使用したAES-256-CBCで復号します。
- AESキーは `aeskey` フィールドのbase64デコード値です（ちょうど32バイトである必要があります）。
- IVはキーの先頭16バイトから導出されます。
- これには `cryptography` Pythonパッケージが必要です（`pip install cryptography`）。

設定は不要です。暗号化されたメディアを受信すると、復号は透過的に行われます。

### 送信（送り出し）

| メソッド | 送信内容 | サイズ制限 |
|--------|--------------|------------|
| `send` | Markdownテキストメッセージ | 4000文字 |
| `send_image` / `send_image_file` | ネイティブ画像メッセージ | 10 MB |
| `send_document` | ファイル添付 | 20 MB |
| `send_voice` | 音声メッセージ（ネイティブ音声はAMR形式のみ） | 2 MB |
| `send_video` | 動画メッセージ | 10 MB |

**チャンクアップロード:** ファイルは3段階のプロトコル（init → chunks → finish）で512 KBのチャンクごとにアップロードされます。アダプターはこれを自動的に処理します。

**自動ダウングレード:** メディアがネイティブタイプのサイズ制限を超えるが、絶対的な20 MBのファイル制限未満の場合、代わりに汎用のファイル添付として自動的に送信されます。

- 画像 > 10 MB → ファイルとして送信
- 動画 > 10 MB → ファイルとして送信
- 音声 > 2 MB → ファイルとして送信
- 非AMR音声 → ファイルとして送信（WeComはネイティブ音声でAMRのみをサポート）

絶対的な20 MB制限を超えるファイルは拒否され、その旨を伝えるメッセージがチャットに送信されます。

## 返信モードのストリーム応答

ボットがWeComのコールバック経由でメッセージを受信すると、アダプターは受信リクエストIDを記憶します。リクエストのコンテキストがまだアクティブな間に応答が送信された場合、アダプターはWeComの返信モード（`aibot_respond_msg`）とストリーミングを使用して、応答を受信メッセージに直接対応付けます。これにより、WeComクライアントでより自然な会話体験が得られます。

受信リクエストのコンテキストが期限切れになっているか利用できない場合、アダプターは `aibot_send_msg` による能動的なメッセージ送信にフォールバックします。

返信モードはメディアでも機能します。アップロードされたメディアは、発信元のメッセージへの返信として送信できます。

## 接続と再接続

アダプターは、WeComのゲートウェイ（`wss://openws.work.weixin.qq.com`）への永続的なWebSocket接続を維持します。

### 接続のライフサイクル

1. **接続:** WebSocket接続を開き、bot_idとsecretを含む `aibot_subscribe` 認証フレームを送信します。
2. **ハートビート:** 接続を維持するために、30秒ごとにアプリケーションレベルのpingフレームを送信します。
3. **リッスン:** 受信フレームを継続的に読み取り、メッセージコールバックをディスパッチします。

### 再接続の動作

接続が失われると、アダプターは指数バックオフを使用して再接続します。

| 試行 | 遅延 |
|---------|-------|
| 1回目の再試行 | 2秒 |
| 2回目の再試行 | 5秒 |
| 3回目の再試行 | 10秒 |
| 4回目の再試行 | 30秒 |
| 5回目以降の再試行 | 60秒 |

再接続が成功するたびに、バックオフカウンターはゼロにリセットされます。切断時には保留中のすべてのリクエストフューチャーが失敗扱いになるため、呼び出し元が無期限にハングすることはありません。

### 重複排除

受信メッセージは、5分間のウィンドウと最大1000エントリのキャッシュを使用して、メッセージIDで重複排除されます。これにより、再接続やネットワークの不調時にメッセージが二重処理されるのを防ぎます。

## すべての環境変数

| 変数 | 必須 | デフォルト | 説明 |
|----------|----------|---------|-------------|
| `WECOM_BOT_ID` | ✅ | — | WeCom AI Bot ID |
| `WECOM_SECRET` | ✅ | — | WeCom AI Bot Secret |
| `WECOM_ALLOWED_USERS` | — | _(空)_ | ゲートウェイレベルの許可リスト用のカンマ区切りユーザーID |
| `WECOM_HOME_CHANNEL` | — | — | cron/通知出力用のチャットID |
| `WECOM_WEBSOCKET_URL` | — | `wss://openws.work.weixin.qq.com` | WebSocketゲートウェイURL |
| `WECOM_DM_POLICY` | — | `open` | DMアクセスポリシー |
| `WECOM_GROUP_POLICY` | — | `open` | グループアクセスポリシー |

## トラブルシューティング

| 問題 | 対処法 |
|---------|-----|
| `WECOM_BOT_ID and WECOM_SECRET are required` | 両方の環境変数を設定するか、セットアップウィザードで設定する |
| `WeCom startup failed: aiohttp not installed` | aiohttpをインストールする: `pip install aiohttp` |
| `WeCom startup failed: httpx not installed` | httpxをインストールする: `pip install httpx` |
| `invalid secret (errcode=40013)` | secretがボットの認証情報と一致しているか確認する |
| `Timed out waiting for subscribe acknowledgement` | `openws.work.weixin.qq.com` へのネットワーク接続を確認する |
| ボットがグループで応答しない | `group_policy` 設定を確認し、グループIDが `group_allow_from` に含まれていることを確認する |
| ボットがグループ内の特定のユーザーを無視する | `groups` 設定セクションのグループ単位の `allow_from` リストを確認する |
| メディアの復号が失敗する | `cryptography` をインストールする: `pip install cryptography` |
| `cryptography is required for WeCom media decryption` | 受信メディアがAES暗号化されています。インストールしてください: `pip install cryptography` |
| 音声メッセージがファイルとして送信される | WeComはネイティブ音声でAMR形式のみをサポートします。他の形式はファイルに自動ダウングレードされます。 |
| `File too large` エラー | WeComはすべてのファイルアップロードに20 MBの絶対上限があります。ファイルを圧縮するか分割してください。 |
| 画像がファイルとして送信される | 10 MBを超える画像はネイティブ画像の制限を超え、ファイル添付に自動ダウングレードされます。 |
| `Timeout sending message to WeCom` | WebSocketが切断された可能性があります。ログで再接続メッセージを確認してください。 |
| `WeCom websocket closed during authentication` | ネットワークの問題または認証情報の誤りです。bot_idとsecretを確認してください。 |
