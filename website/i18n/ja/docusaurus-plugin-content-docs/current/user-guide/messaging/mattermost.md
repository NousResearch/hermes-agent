---
sidebar_position: 8
title: "Mattermost"
description: "Hermes AgentをMattermostボットとしてセットアップする"
---

# Mattermostのセットアップ

Hermes AgentはMattermostとボットとして統合され、ダイレクトメッセージやチームチャンネルを通じてAIアシスタントとチャットできます。Mattermostはセルフホスト型のオープンソースのSlack代替です — 自分のインフラ上で運用するため、データを完全に管理できます。ボットはMattermostのREST API（v4）とリアルタイムイベント用のWebSocketを介して接続し、メッセージをHermes Agentのパイプライン（ツール利用、メモリ、推論を含む）で処理し、リアルタイムで応答します。テキスト、ファイル添付、画像、スラッシュコマンドをサポートします。

外部のMattermostライブラリは不要です — アダプターは、すでにHermesの依存関係に含まれている `aiohttp` を使用します。

セットアップの前に、多くの人が知りたい部分から説明します：Mattermostインスタンスに導入したあとHermesがどう振る舞うか、です。

## Hermesの振る舞い

| コンテキスト | 振る舞い |
|---------|----------|
| **DM** | Hermesはすべてのメッセージに応答します。`@mention` は不要です。各DMは独自のセッションを持ちます。 |
| **公開/非公開チャンネル** | `@mention` するとHermesが応答します。メンションがない場合、Hermesはメッセージを無視します。 |
| **スレッド** | `MATTERMOST_REPLY_MODE=thread` の場合、Hermesはあなたのメッセージの下にスレッドで返信します。スレッドのコンテキストは親チャンネルから分離されたままになります。 |
| **複数ユーザーがいる共有チャンネル** | デフォルトでは、Hermesはチャンネル内でユーザーごとにセッション履歴を分離します。同じチャンネルで話す2人が、明示的に無効化しない限り、1つのトランスクリプトを共有することはありません。 |

:::tip
Hermesにスレッド形式の会話（元のメッセージの下にネスト）で返信させたい場合は、`MATTERMOST_REPLY_MODE=thread` を設定します。デフォルトは `off` で、チャンネルにフラットなメッセージを送信します。
:::

### Mattermostのセッションモデル

デフォルトでは：

- 各DMが独自のセッションを持つ
- 各スレッドが独自のセッション名前空間を持つ
- 共有チャンネル内の各ユーザーが、そのチャンネル内で独自のセッションを持つ

これは `config.yaml` で制御されます：

```yaml
group_sessions_per_user: true
```

チャンネル全体で1つの共有会話を明示的に望む場合にのみ、`false` に設定します：

```yaml
group_sessions_per_user: false
```

共有セッションは協働的なチャンネルでは便利ですが、次のことも意味します：

- ユーザーはコンテキストの増大とトークンコストを共有する
- ある人の長いツール多用タスクが、他の全員のコンテキストを肥大化させる可能性がある
- ある人の実行中の処理が、同じチャンネル内の別の人のフォローアップを中断する可能性がある

このガイドでは、Mattermost上でのボット作成から最初のメッセージ送信まで、セットアップの全プロセスを説明します。

## ステップ1: ボットアカウントを有効化する

ボットアカウントを作成する前に、Mattermostサーバー上でボットアカウントを有効化しておく必要があります。

1. **System Admin** としてMattermostにログインします。
2. **System Console** → **Integrations** → **Bot Accounts** に移動します。
3. **Enable Bot Account Creation** を **true** に設定します。
4. **Save** をクリックします。

:::info
System Adminアクセス権がない場合は、Mattermostの管理者にボットアカウントの有効化と作成を依頼してください。
:::

## ステップ2: ボットアカウントを作成する

1. Mattermostで、**☰** メニュー（左上）→ **Integrations** → **Bot Accounts** をクリックします。
2. **Add Bot Account** をクリックします。
3. 詳細を入力します：
   - **Username**: 例 `hermes`
   - **Display Name**: 例 `Hermes Agent`
   - **Description**: 任意
   - **Role**: `Member` で十分です
4. **Create Bot Account** をクリックします。
5. Mattermostが **ボットトークン** を表示します。**すぐにコピーしてください。**

:::warning[トークンは一度だけ表示される]
ボットトークンは、ボットアカウント作成時に一度だけ表示されます。失った場合は、ボットアカウント設定から再生成する必要があります。トークンを公開で共有したり、Gitにコミットしたりしないでください — このトークンを持つ者は誰でもボットを完全に制御できます。
:::

トークンを安全な場所（例えばパスワードマネージャー）に保管してください。ステップ5で必要になります。

:::tip
ボットアカウントの代わりに **パーソナルアクセストークン** を使うこともできます。**Profile** → **Security** → **Personal Access Tokens** → **Create Token** に移動します。これは、Hermesに別個のボットユーザーではなく自分自身のユーザーとして投稿させたい場合に便利です。
:::

## ステップ3: ボットをチャンネルに追加する

ボットは、応答させたいチャンネルのメンバーである必要があります：

1. ボットを置きたいチャンネルを開きます。
2. チャンネル名をクリック → **Add Members** をクリックします。
3. ボットのユーザー名（例 `hermes`）を検索して追加します。

DMの場合は、ボットとのダイレクトメッセージを開くだけで、すぐに応答できるようになります。

## ステップ4: あなたのMattermostユーザーIDを調べる

Hermes Agentは、誰がボットとやり取りできるかを制御するために、あなたのMattermostユーザーIDを使用します。調べるには：

1. **アバター**（左上隅）→ **Profile** をクリックします。
2. プロフィールダイアログにユーザーIDが表示されます — クリックしてコピーします。

ユーザーIDは `3uo8dkh1p7g1mfk49ear5fzs5c` のような26文字の英数字文字列です。

:::warning
ユーザーIDはユーザー名では**ありません**。ユーザー名は `@` の後に表示されるもの（例 `@alice`）です。ユーザーIDは、Mattermostが内部的に使用する長い英数字の識別子です。
:::

**代替方法**: APIを通じてユーザーIDを取得することもできます：

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-mattermost-server/api/v4/users/me | jq .id
```

:::tip
**チャンネルID** を取得するには：チャンネル名をクリック → **View Info** をクリックします。情報パネルにチャンネルIDが表示されます。ホームチャンネルを手動で設定したい場合に必要です。
:::

## ステップ5: Hermes Agentを設定する

### オプションA: 対話型セットアップ（推奨）

ガイド付きセットアップコマンドを実行します：

```bash
hermes gateway setup
```

プロンプトで **Mattermost** を選択し、尋ねられたらサーバーURL、ボットトークン、ユーザーIDを貼り付けます。

### オプションB: 手動設定

`~/.hermes/.env` ファイルに以下を追加します：

```bash
# 必須
MATTERMOST_URL=https://mm.example.com
MATTERMOST_TOKEN=***
MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c

# 複数の許可ユーザー（カンマ区切り）
# MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c,8fk2jd9s0a7bncm1xqw4tp6r3e

# オプション: 返信モード（thread または off、デフォルト: off）
# MATTERMOST_REPLY_MODE=thread

# オプション: @mention なしで応答する（デフォルト: true = メンションを要求）
# MATTERMOST_REQUIRE_MENTION=false

# オプション: ボットが @mention なしで応答するチャンネル（カンマ区切りのチャンネルID）
# MATTERMOST_FREE_RESPONSE_CHANNELS=channel_id_1,channel_id_2
```

`~/.hermes/config.yaml` の任意の振る舞い設定：

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true` は、共有チャンネルとスレッド内で各参加者のコンテキストを分離したまま保ちます

### ゲートウェイを起動する

設定が完了したら、Mattermostゲートウェイを起動します：

```bash
hermes gateway
```

ボットは数秒以内にMattermostサーバーに接続するはずです。テストのために、DMまたは追加済みのチャンネルでメッセージを送ってみてください。

:::tip
`hermes gateway` をバックグラウンドで、またはsystemdサービスとして実行して常駐運用できます。詳細はデプロイメントドキュメントを参照してください。
:::

## ホームチャンネル

ボットが能動的なメッセージ（cronジョブの出力、リマインダー、通知など）を送信する「ホームチャンネル」を指定できます。設定方法は2通りあります：

### スラッシュコマンドを使う

ボットが存在する任意のMattermostチャンネルで `/sethome` と入力します。そのチャンネルがホームチャンネルになります。

### 手動設定

`~/.hermes/.env` に以下を追加します：

```bash
MATTERMOST_HOME_CHANNEL=abc123def456ghi789jkl012mn
```

IDを実際のチャンネルID（チャンネル名をクリック → View Info → IDをコピー）に置き換えます。

## 返信モード

`MATTERMOST_REPLY_MODE` 設定は、Hermesが応答を投稿する方法を制御します：

| モード | 振る舞い |
|------|----------|
| `off`（デフォルト） | Hermesは通常のユーザーのように、チャンネルにフラットなメッセージを投稿します。 |
| `thread` | Hermesは元のメッセージの下にスレッドで返信します。やり取りが多いときにチャンネルをきれいに保ちます。 |

`~/.hermes/.env` で設定します：

```bash
MATTERMOST_REPLY_MODE=thread
```

## メンションの振る舞い

デフォルトでは、ボットは `@mention` されたときのみチャンネルで応答します。これは変更できます：

| 変数 | デフォルト | 説明 |
|----------|---------|-------------|
| `MATTERMOST_REQUIRE_MENTION` | `true` | `false` に設定すると、チャンネル内のすべてのメッセージに応答します（DMは常に機能します）。 |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | _(なし)_ | require_mention が true でも、ボットが `@mention` なしで応答するチャンネルID（カンマ区切り）。 |

MattermostでチャンネルIDを調べるには：チャンネルを開き、チャンネル名ヘッダーをクリックして、URLまたはチャンネル詳細でIDを探します。

ボットが `@mention` されると、処理前にメッセージからメンションが自動的に取り除かれます。

## トラブルシューティング

### ボットがメッセージに応答しない

**原因**: ボットがチャンネルのメンバーでない、または `MATTERMOST_ALLOWED_USERS` にあなたのユーザーIDが含まれていません。

**修正**: ボットをチャンネルに追加します（チャンネル名 → Add Members → ボットを検索）。あなたのユーザーIDが `MATTERMOST_ALLOWED_USERS` にあることを確認します。ゲートウェイを再起動します。

### 403 Forbidden エラー

**原因**: ボットトークンが無効、またはボットにそのチャンネルへの投稿権限がありません。

**修正**: `.env` ファイルの `MATTERMOST_TOKEN` が正しいことを確認します。ボットアカウントが無効化されていないことを確認します。ボットがチャンネルに追加されていることを確認します。パーソナルアクセストークンを使っている場合は、アカウントに必要な権限があることを確認します。

### WebSocketの切断 / 再接続ループ

**原因**: ネットワークの不安定さ、Mattermostサーバーの再起動、またはWebSocket接続に関するファイアウォール/プロキシの問題。

**修正**: アダプターは指数バックオフ（2秒 → 60秒）で自動的に再接続します。サーバーのWebSocket設定を確認してください — リバースプロキシ（nginx、Apache）にはWebSocketアップグレードヘッダーの設定が必要です。Mattermostサーバー上でファイアウォールがWebSocket接続をブロックしていないことを確認してください。

nginxの場合、設定に以下が含まれていることを確認します：

```nginx
location /api/v4/websocket {
    proxy_pass http://mattermost-backend;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 600s;
}
```

### 起動時に「Failed to authenticate」

**原因**: トークンまたはサーバーURLが正しくありません。

**修正**: `MATTERMOST_URL` がMattermostサーバーを指していることを確認します（`https://` を含め、末尾のスラッシュはなし）。`MATTERMOST_TOKEN` が有効であることを確認します — curlで試してください：

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/api/v4/users/me
```

これがボットのユーザー情報を返せば、トークンは有効です。エラーを返す場合は、トークンを再生成してください。

### ボットがオフライン

**原因**: Hermesゲートウェイが実行されていない、または接続に失敗しました。

**修正**: `hermes gateway` が実行されていることを確認します。ターミナル出力でエラーメッセージを確認します。よくある問題：URLの誤り、トークンの期限切れ、Mattermostサーバーに到達できない。

### 「User not allowed」 / ボットがあなたを無視する

**原因**: あなたのユーザーIDが `MATTERMOST_ALLOWED_USERS` に含まれていません。

**修正**: `~/.hermes/.env` の `MATTERMOST_ALLOWED_USERS` にあなたのユーザーIDを追加し、ゲートウェイを再起動します。注意：ユーザーIDは26文字の英数字文字列で、`@username` ではありません。

## チャンネルごとのプロンプト

特定のMattermostチャンネルにエフェメラルなシステムプロンプトを割り当てます。プロンプトはターン毎に実行時に注入され — トランスクリプト履歴には決して永続化されない — ため、変更は即座に反映されます。

```yaml
mattermost:
  channel_prompts:
    "channel_id_abc123": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "channel_id_def456": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

キーはMattermostのチャンネルID（チャンネルURLまたはAPI経由で調べられます）です。一致するチャンネル内のすべてのメッセージに、エフェメラルなシステム指示としてプロンプトが注入されます。

## セキュリティ

:::warning
ボットとやり取りできる人を制限するために、必ず `MATTERMOST_ALLOWED_USERS` を設定してください。設定しない場合、ゲートウェイは安全策としてデフォルトですべてのユーザーを拒否します。信頼できる人のユーザーIDのみを追加してください — 許可されたユーザーは、ツール利用やシステムアクセスを含むエージェントの全機能にフルアクセスできます。
:::

Hermes Agentのデプロイを保護する方法の詳細については、[セキュリティガイド](../security.md)を参照してください。

## 補足

- **セルフホストに優しい**: あらゆるセルフホスト型Mattermostインスタンスで動作します。Mattermost Cloudアカウントやサブスクリプションは不要です。
- **追加の依存関係なし**: アダプターはHTTPとWebSocketに `aiohttp` を使用し、これはすでにHermes Agentに含まれています。
- **Team Edition互換**: Mattermost Team Edition（無料）とEnterprise Editionの両方で動作します。
