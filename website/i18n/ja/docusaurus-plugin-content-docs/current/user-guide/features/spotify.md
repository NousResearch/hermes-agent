# Spotify

Hermesは、Spotifyの公式Web API（PKCE OAuth付き）を使って、Spotifyを直接制御できます — 再生、キュー、検索、プレイリスト、保存したトラック/アルバム、再生履歴など。トークンは `~/.hermes/auth.json` に保存され、401時に自動でリフレッシュされます。マシンごとに一度ログインするだけで済みます。

Hermesの組み込みOAuth連携（Google、GitHub Copilot、Codex）とは異なり、Spotifyは各ユーザーが独自の軽量な開発者アプリを登録する必要があります。Spotifyは、誰でも使えるパブリックなOAuthアプリをサードパーティが配布することを認めていません。所要時間は約2分で、`hermes auth spotify` がその手順を案内してくれます。

## 前提条件

- Spotifyアカウント。検索、プレイリスト、ライブラリ、アクティビティのツールは**無料（Free）**で動作します。再生制御（play、pause、skip、seek、volume、queue add、transfer）には**Premium**が必要です。
- Hermes Agentがインストールされ、実行されていること。
- 再生ツールの場合: **アクティブなSpotify Connectデバイス** — Web APIが制御する対象を持てるように、少なくとも1台のデバイス（スマートフォン、デスクトップ、Webプレーヤー、スピーカー）でSpotifyアプリが開いている必要があります。何もアクティブでない場合は「no active device」メッセージとともに `403 Forbidden` が返ります。任意のデバイスでSpotifyを開いて再試行してください。

## セットアップ

### 一発で: `hermes tools`

最速の経路です。実行します。

```bash
hermes tools
```

`🎵 Spotify` までスクロールし、スペースを押してオンに切り替え、次に `s` で保存します。Hermesはそのまま直接OAuthフローに進みます — まだSpotifyアプリを持っていない場合は、その場でインラインに作成手順を案内してくれます。完了すると、ツールセットの有効化と認証が一度に行われます。

手順を別々に行いたい場合（または後で再認証する場合）は、以下の2ステップのフローを使ってください。

### 2ステップのフロー

#### 1. ツールセットを有効にする

```bash
hermes tools
```

`🎵 Spotify` をオンに切り替えて保存し、インラインのウィザードが開いたら、それを閉じます（Ctrl+C）。ツールセットはオンのままで、認証ステップだけが先送りされます。

#### 2. ログインウィザードを実行する

```bash
hermes auth spotify
```

7つのSpotifyツールは、ステップ1の後にのみエージェントのツールセットに現れます — これらはデフォルトでオフになっており、使いたくないユーザーが毎回のAPI呼び出しで余分なツールスキーマを送らずに済むようにしています。

`HERMES_SPOTIFY_CLIENT_ID` が設定されていない場合、Hermesはインラインでアプリ登録の手順を案内します。

1. ブラウザで `https://developer.spotify.com/dashboard` を開く
2. Spotifyの「Create app」フォームに貼り付ける正確な値を表示する
3. 取得したClient IDの入力を促す
4. それを `~/.hermes/.env` に保存し、今後の実行ではこのステップをスキップする
5. そのままOAuthの同意フローへ進む

承認すると、トークンは `~/.hermes/auth.json` の `providers.spotify` 配下に書き込まれます。アクティブな推論プロバイダーは変更され**ません** — Spotifyの認証はLLMプロバイダーとは独立しています。

### Spotifyアプリの作成（ウィザードが尋ねる内容）

ダッシュボードが開いたら、**Create app** をクリックして次を入力します。

| フィールド | 値 |
|-------|-------|
| App name | 何でも可（例: `hermes-agent`） |
| App description | 何でも可（例: `personal Hermes integration`） |
| Website | 空欄のまま |
| Redirect URI | `http://127.0.0.1:43827/spotify/callback` |
| Which API/SDKs? | **Web API** にチェック |

利用規約に同意して **Save** をクリックします。次のページで **Settings** をクリックし、**Client ID** をコピーしてHermesのプロンプトに貼り付けます。それがHermesに必要な唯一の値です — PKCEはクライアントシークレットを使いません。

### SSH経由 / ヘッドレス環境での実行

`SSH_CLIENT` または `SSH_TTY` が設定されている場合、HermesはウィザードとOAuthステップの両方で自動的なブラウザ起動をスキップします。Hermesが表示するダッシュボードURLと認可URLをコピーし、ローカルマシンのブラウザでそれらを開いて、通常どおり進めてください — ローカルのHTTPリスナーは、リモートホストのポート43827で引き続き動作します。SSHトンネル経由でそこに到達する必要がある場合は、そのポートをフォワードします: `ssh -L 43827:127.0.0.1:43827 remote`。

## 確認

```bash
hermes auth status spotify
```

トークンが存在するかどうか、アクセストークンの有効期限がいつかを表示します。リフレッシュは自動です。SpotifyのAPI呼び出しが401を返すと、クライアントはリフレッシュトークンを交換し、一度だけ再試行します。リフレッシュトークンはHermesの再起動をまたいで保持されるため、Spotifyのアカウント設定でアプリを取り消すか、`hermes auth logout spotify` を実行しない限り、再認証は不要です。

## 使い方

ログインすると、エージェントは7つのSpotifyツールにアクセスできます。エージェントには自然に話しかけるだけで — 適切なツールとアクションを選んでくれます。最良の動作のために、エージェントは標準的な使用パターン（single-search-then-play、`get_state` を事前確認すべきでないタイミングなど）を教えるコンパニオンスキルを読み込みます。

```
> play some miles davis
> what am I listening to
> add this track to my Late Night Jazz playlist
> skip to the next song
> make a new playlist called "Focus 2026" and add the last three songs I played
> which of my saved albums are by Radiohead
> search for acoustic covers of Blackbird
> transfer playback to my kitchen speaker
```

### ツールリファレンス

再生を変更するすべてのアクションは、特定のデバイスを対象とするための任意の `device_id` を受け付けます。省略した場合、Spotifyは現在アクティブなデバイスを使用します。

#### `spotify_playback`
再生を制御・確認し、さらに最近再生した履歴を取得します。

| アクション | 目的 | Premium? |
|--------|---------|----------|
| `get_state` | 完全な再生状態（トラック、デバイス、進捗、シャッフル/リピート） | いいえ |
| `get_currently_playing` | 現在のトラックのみ（204では空を返す — 後述） | いいえ |
| `play` | 再生の開始/再開。任意: `context_uri`、`uris`、`offset`、`position_ms` | はい |
| `pause` | 再生を一時停止 | はい |
| `next` / `previous` | トラックをスキップ | はい |
| `seek` | `position_ms` にジャンプ | はい |
| `set_repeat` | `state` = `track` / `context` / `off` | はい |
| `set_shuffle` | `state` = `true` / `false` | はい |
| `set_volume` | `volume_percent` = 0-100 | はい |
| `recently_played` | 最近再生したトラック。任意の `limit`、`before`、`after`（Unixミリ秒） | いいえ |

#### `spotify_devices`
| アクション | 目的 |
|--------|---------|
| `list` | アカウントから見えるすべてのSpotify Connectデバイス |
| `transfer` | 再生を `device_id` に移す。任意の `play: true` で移行時に再生を開始 |

#### `spotify_queue`
| アクション | 目的 | Premium? |
|--------|---------|----------|
| `get` | 現在キューに入っているトラック | いいえ |
| `add` | キューに `uri` を追加 | はい |

#### `spotify_search`
カタログを検索します。`query` は必須です。任意: `types`（`track` / `album` / `artist` / `playlist` / `show` / `episode` の配列）、`limit`、`offset`、`market`。

#### `spotify_playlists`
| アクション | 目的 | 必須の引数 |
|--------|---------|---------------|
| `list` | ユーザーのプレイリスト | — |
| `get` | 1つのプレイリスト + トラック | `playlist_id` |
| `create` | 新規プレイリスト | `name`（+ 任意の `description`、`public`、`collaborative`） |
| `add_items` | トラックを追加 | `playlist_id`、`uris`（任意の `position`） |
| `remove_items` | トラックを削除 | `playlist_id`、`uris`（+ 任意の `snapshot_id`） |
| `update_details` | 名前変更 / 編集 | `playlist_id` + `name`、`description`、`public`、`collaborative` のいずれか |

#### `spotify_albums`
| アクション | 目的 | 必須の引数 |
|--------|---------|---------------|
| `get` | アルバムのメタデータ | `album_id` |
| `tracks` | アルバムのトラックリスト | `album_id` |

#### `spotify_library`
保存したトラックと保存したアルバムへの統合アクセス。`kind` 引数でコレクションを選びます。

| アクション | 目的 |
|--------|---------|
| `list` | ページ分割されたライブラリの一覧 |
| `save` | `ids` / `uris` をライブラリに追加 |
| `remove` | `ids` / `uris` をライブラリから削除 |

必須: `kind` = `tracks` または `albums`、加えて `action`。

### 機能マトリックス: Free と Premium

読み取り専用のツールはFreeアカウントで動作します。再生やキューを変更するものはすべてPremiumが必要です。

| Freeで動作 | Premiumが必要 |
|---------------|------------------|
| `spotify_search`（すべて） | `spotify_playback` — play、pause、next、previous、seek、set_repeat、set_shuffle、set_volume |
| `spotify_playback` — get_state、get_currently_playing、recently_played | `spotify_queue` — add |
| `spotify_devices` — list | `spotify_devices` — transfer |
| `spotify_queue` — get | |
| `spotify_playlists`（すべて） | |
| `spotify_albums`（すべて） | |
| `spotify_library`（すべて） | |

## スケジューリング: Spotify + cron

Spotifyツールは通常のHermesツールなので、Hermesセッション内で実行されるcronジョブは、任意のスケジュールで再生をトリガーできます。新しいコードは不要です。

### 朝の目覚めプレイリスト

```bash
hermes cron add \
  --name "morning-commute" \
  "0 7 * * 1-5" \
  "Transfer playback to my kitchen speaker and start my 'Morning Commute' playlist. Volume to 40. Shuffle on."
```

平日の毎朝7時に起こること:
1. cronがヘッドレスのHermesセッションを起動します。
2. エージェントはプロンプトを読み、`spotify_devices list` を呼んで名前で「kitchen speaker」を見つけ、次に `spotify_devices transfer` → `spotify_playback set_volume` → `spotify_playback set_shuffle` → `spotify_search` + `spotify_playback play` を呼びます。
3. 対象のスピーカーで音楽が始まります。総コスト: 1セッション、数回のツール呼び出し、人間の入力なし。

### 夜のウィンドダウン

```bash
hermes cron add \
  --name "wind-down" \
  "30 22 * * *" \
  "Pause Spotify. Then set volume to 20 so it's quiet when I start it again tomorrow."
```

### 落とし穴

- **cronが発火するときにアクティブなデバイスが存在している必要があります。** Spotifyクライアント（スマートフォン/デスクトップ/Connectスピーカー）が動作していない場合、再生アクションは `403 no active device` を返します。朝のプレイリストの場合のコツは、スマートフォンではなく、常にオンになっているデバイス（Sonos、Echo、スマートスピーカー）を対象にすることです。
- **再生を変更するものはすべてPremiumが必要** — play、pause、skip、volume、transfer。読み取り専用のcronジョブ（スケジュールした「最近再生したトラックをメールで送る」）はFreeでも問題なく動作します。
- **cronエージェントはあなたのアクティブなツールセットを継承します。** cronセッションがSpotifyツールを見られるように、Spotifyは `hermes tools` で有効になっている必要があります。
- **cronジョブは `skip_memory=True` で実行されます** ので、あなたのメモリストアに書き込みません。

完全なcronリファレンス: [Cronジョブ](./cron)。

## サインアウト

```bash
hermes auth logout spotify
```

`~/.hermes/auth.json` からトークンを削除します。アプリの設定も消去するには、`~/.hermes/.env` から `HERMES_SPOTIFY_CLIENT_ID`（および設定していれば `HERMES_SPOTIFY_REDIRECT_URI`）を削除するか、ウィザードを再実行します。

Spotify側でアプリを取り消すには、[アカウントに接続されているアプリ](https://www.spotify.com/account/apps/)にアクセスし、**REMOVE ACCESS** をクリックします。

## トラブルシューティング

**`403 Forbidden — Player command failed: No active device found`** — 少なくとも1台のデバイスでSpotifyを実行している必要があります。スマートフォン、デスクトップ、またはWebプレーヤーでSpotifyアプリを開き、任意のトラックを1秒再生して登録し、再試行してください。`spotify_devices list` で現在見えているものが表示されます。

**`403 Forbidden — Premium required`** — Freeアカウントで、再生を変更するアクションを使おうとしています。上記の機能マトリックスを参照してください。

**`get_currently_playing` での `204 No Content`** — どのデバイスでも現在何も再生されていません。これはSpotifyの通常の応答であり、エラーではありません。Hermesはこれを説明的な空の結果（`is_playing: false`）として表面化します。

**`INVALID_CLIENT: Invalid redirect URI`** — Spotifyアプリの設定にあるredirect URIが、Hermesが使用しているものと一致していません。デフォルトは `http://127.0.0.1:43827/spotify/callback` です。それをアプリの許可済みredirect URIに追加するか、`~/.hermes/.env` の `HERMES_SPOTIFY_REDIRECT_URI` を登録したものに設定してください。

**`429 Too Many Requests`** — Spotifyのレート制限です。Hermesは分かりやすいエラーを返します。1分待ってから再試行してください。これが続く場合は、おそらくスクリプトでタイトなループを実行しています — Spotifyのクォータはおよそ30秒ごとにリセットされます。

**`401 Unauthorized` が繰り返し返ってくる** — リフレッシュトークンが取り消されています（通常はアカウントからアプリを削除したか、アプリが削除されたためです）。`hermes auth spotify` をもう一度実行してください。

**ウィザードがブラウザを開かない** — SSH経由、またはディスプレイのないコンテナにいる場合、Hermesはそれを検知して自動起動をスキップします。表示されるダッシュボードURLをコピーして手動で開いてください。

## 上級: カスタムスコープ

デフォルトでは、Hermesは出荷されるすべてのツールに必要なスコープを要求します。アクセスを制限したい場合は上書きします。

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

スコープのリファレンス: [Spotify Web APIのスコープ](https://developer.spotify.com/documentation/web-api/concepts/scopes)。ツールが必要とするより少ないスコープを要求すると、そのツールの呼び出しは403で失敗します。

## 上級: カスタムのクライアントID / redirect URI

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

または `~/.hermes/.env` で恒久的に設定します。

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

redirect URIは、Spotifyアプリの設定で許可リストに登録されている必要があります。デフォルトはほぼすべての人に有効です — ポート43827が使用されている場合のみ変更してください。

## 各ファイルの場所

| ファイル | 内容 |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | アクセストークン、リフレッシュトークン、有効期限、スコープ、redirect URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID`、任意の `HERMES_SPOTIFY_REDIRECT_URI` |
| Spotifyアプリ | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) であなたが所有。Client IDとredirect URIの許可リストを含む |
