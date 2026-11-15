---
sidebar_position: 5
title: "WhatsApp"
description: "組み込みの Baileys ブリッジを介して Hermes Agent を WhatsApp ボットとしてセットアップする"
---

# WhatsAppのセットアップ

Hermesは、**Baileys** をベースとした組み込みブリッジを介してWhatsAppに接続します。これはWhatsApp Webセッションをエミュレートすることで動作し — 公式のWhatsApp Business APIを介するものでは**ありません**。Meta開発者アカウントやBusiness認証は不要です。

> ガイド付きの手順を進めるには、`hermes gateway setup` を実行して **WhatsApp** を選んでください。

:::tip 2つのWhatsApp連携
このページは **Baileysブリッジ** 向けです — セットアップが早く、個人アカウントで使え、公開URLは不要ですが、BANのリスクがあります。

実際のビジネス用ボットを運用していて安定性が欲しい場合は、代わりに **[WhatsApp Business Cloud APIガイド](/docs/user-guide/messaging/whatsapp-cloud)** を参照してください。こちらはMetaが公式にサポートする方法です: アカウントBANのリスクはありませんが、Meta Businessアカウントと公開webhook URLが必要です。

理由があれば、2つのアダプターを異なる電話番号で並行して実行することもできます。
:::

:::warning 非公式API — BANのリスク
WhatsAppは、Business API以外でのサードパーティ製ボットを公式には**サポートしていません**。サードパーティ製ブリッジの使用には、アカウント制限のわずかなリスクが伴います。リスクを最小限に抑えるには次のようにしてください。
- ボット用に**専用の電話番号を使う**（個人の番号は使わない）
- **一括/スパムメッセージを送らない** — 会話的な利用にとどめる
- 先にメッセージを送ってきていない相手への**アウトバウンドメッセージを自動化しない**
:::

:::warning WhatsApp Webプロトコルの更新
WhatsAppは定期的にWebプロトコルを更新しており、これによってサードパーティ製ブリッジとの互換性が一時的に壊れることがあります。これが起きた場合、Hermesはブリッジの依存関係を更新します。WhatsAppの更新後にボットが動作しなくなった場合は、最新のHermesバージョンをプルして再ペアリングしてください。
:::

## 複数のプロファイル

ホストのマルチプレクサーは、プロファイルごとに別々のペアリング済みWhatsAppセッションを提供できます。
セカンダリプロファイルをペアリングするには `hermes -p work whatsapp` を実行し、そのプロファイルで
WhatsAppを有効にします。`creds.json` のない有効なプロファイルは、`whatsapp_unpaired` ステータスと
そのペアリングコマンドとともにスキップされます。

明示的な `platforms.whatsapp.extra.bridge_port` が優先されます。それ以外の場合、
セカンダリは3001〜3999のうち、他のプロファイルの記録が使用していない最初の空きポートを選び、
以降の起動のために自身の `platforms/whatsapp/bridge_port` ファイルに保存します。
オペレーターはポート番号を書いたそのファイルを事前に作成できます。削除すると
新しいポートが割り当てられます。起動プロファイルは、（セカンダリとして動作したことで）そのファイルを
すでに持っている場合を除きポート3000を使います。持っている場合は、そのゲートウェイと
`hermes send --to whatsapp:<chat_id>` の両方が記録されたポートを使い続けます。

セカンダリがそのポートですでに実行中のブリッジを引き継ぐのは、自身のセッションのpidファイルが
そのプロセス（pid、カーネルの開始時刻、起動時のポート）を識別できる場合だけです。これは
ゲートウェイのクラッシュ後に残る状態です。正常でないものは同じ識別情報によって回収され、
再起動されます。そのポートにバインドしているそれ以外のプロセスは、そのプロファイルに限り
致命的なエラーになります。`platforms.whatsapp.extra.bridge_port` を別の空きポートに設定するか、
そのポートを保持しているプロセスを停止してください。他の
プロファイルは引き続き実行されます。`hermes gateway status --profile work` は、共有イングレスではなく
そのプロファイル自身のWhatsAppアダプターを報告します。

1つのマルチプレックスされたゲートウェイではなく、それぞれ独自のゲートウェイを実行するプロファイルは、
別途設定しない限りすべてポート3000を使います。各プロファイルの `config.yaml` で
それぞれに別のポートを割り当ててください:

```yaml
platforms:
  whatsapp:
    extra:
      bridge_port: 3001        # one distinct port per profile
```

ゲートウェイは、実行中のブリッジを、ブリッジが `/health` で報告するセッションディレクトリによって
識別します。別のプロファイルのセッションを提供しているブリッジが引き継がれたり停止されたりすることは
決してありません: 2つ目のプロファイルのWhatsAppは、ポートと他方のセッションを示した
`whatsapp_bridge_foreign_session` で起動に失敗します。ポートを保持しているものの時間内に
`/health` に応答しないプロセスも実行されたまま残され、WhatsAppはリトライ可能な
`whatsapp_bridge_unresponsive` で失敗します。`hermes send --to whatsapp:<chat_id>` とcronの
配信も同じフィールドを確認し、別のプロファイルのブリッジや `/health` が失敗するブリッジを通じて
何も送信しません。`session_path` を
オーバーライドする場合は、プロファイルごとに別にしてください。そうしないと、プロファイル同士が1つの
WhatsAppログインを共有してしまいます。古いHermesが起動したブリッジはセッションディレクトリを報告しないため、
ブリッジの更新後と同様に1回再起動されます。

クラッシュしたゲートウェイが残したブリッジが引き継がれるのは、このゲートウェイがブリッジを起動する際と
同じ設定で起動されていた場合だけです: DMとグループのポリシー、
許可リスト、モード、返信の接頭辞、既読通知、その他のブリッジ設定、
およびブリッジの `package.json` です。そうでない場合は、再ペアリングなしで
再起動されます。そのような記録を持たないブリッジ（古いHermesが起動したもの）は
1回再起動されます。

### ブリッジAPIの認証

ブリッジはループバックでのみリッスンしますが、ループバックは呼び出し元を識別しません:
そのままでは、同じマシン上の任意のプロセスがすべての受信メッセージを読んだり
（`GET /messages`）、あなたとして送信したり（`POST /send`）できてしまいます。そのため各セッションは
シークレット `platforms/whatsapp/session/.bridge-token`（所有者のみ。初回起動時に
ゲートウェイが作成）を持ち、ブリッジはすべてのルートでそれをbearerトークンとして要求します。
すでに実行中のブリッジを引き継ぐ前に、ゲートウェイはまず `GET /auth-proof` にチャレンジを送り、
リスナーがシークレットを保持していることを証明した場合にのみ引き継ぐため、先にポートをバインドした
なりすましがトークンを知ることはありません。証明に失敗したリスナー（独自のトークンを持つ別のプロファイルの
ブリッジ、またはなりすまし）がポートを理由に停止されることは決してありません: このセッション
自身のブリッジがpidファイルによって停止された後もポートを保持しているリスナーがあると、
WhatsAppは `whatsapp_bridge_foreign_session` で失敗します。新たに起動される
ブリッジには毎回新しいシークレットが与えられます。cronと `hermes send` の配信は同じ
ファイルを読みますが、決して作成はしません。ゲートウェイがまだ一度も起動していないセッションは
`No WhatsApp bridge token` を報告します。`HERMES_WHATSAPP_BRIDGE_TOKEN` なしで手動起動された
ブリッジは、サービスの提供を拒否します。

## 2つのモード

| モード | 仕組み | 最適な用途 |
|------|-------------|----------|
| **専用ボット番号**（推奨） | ボット専用の電話番号を割り当てます。人々はその番号に直接メッセージを送ります。 | 整ったUX、複数ユーザー、低いBANリスク |
| **個人のセルフチャット** | 自分のWhatsAppを使います。自分自身にメッセージを送ってエージェントと話します。 | 素早いセットアップ、シングルユーザー、テスト |

---

## 前提条件

- **Node.js v20以上** と **npm** — WhatsAppブリッジはNode.jsプロセスとして実行されます
- **WhatsAppがインストールされた電話**（QRコードのスキャン用）

通常のQRペアリングにブラウザは不要です。パスキー認証が必要なアカウントでは、
そのアカウントのパスキーにアクセスできる、インストール済みのChromeまたはEdgeブラウザも必要です。

---

## ステップ1: セットアップウィザードを実行する

```bash
hermes whatsapp
```

ウィザードは次を行います。

1. どのモードにするか（**bot** または **self-chat**）を尋ねる
2. 必要に応じてブリッジの依存関係をインストールする
3. ターミナルに**QRコード**を表示する
4. スキャンを待つ

**QRコードをスキャンするには:**

1. 電話でWhatsAppを開く
2. **設定 → リンク済みデバイス**に移動する
3. **デバイスをリンク**をタップする
4. カメラをターミナルのQRコードに向ける

ペアリングが完了すると、ウィザードは接続を確認して終了します。セッションは自動的に保存されます。

### パスキーが必要なアカウント

QRスキャンの後にWhatsAppがパスキーを要求した場合は、インストール済みのChromeまたはEdgeの
実行ファイルへのパスを指定して、対話型のセットアップを実行します:

```text
hermes whatsapp --passkey-browser "/absolute/path/to/browser"
```

上記のパスはプレースホルダーです。お使いのマシン上の実際のブラウザの実行ファイルに置き換えてください。
セットアップは一時的なブラウザウィンドウでWhatsApp Webを開き、そのパスキー要求を
完了させます。アカウントのパスキーを使って要求を承認してください。
セットアップが確認コードを表示した場合は、電話のコードと比較し、一致する場合にのみ
確認してください。要求が完了またはキャンセルされるとブラウザは閉じます。
Hermesは、ペアリングが成功した後にのみWhatsAppを有効にします。

一時的なウィンドウは、通常のChromeやEdgeのプロファイルや、サインイン済みの
パスワードマネージャーを引き継ぎません。パスキーは、そのウィンドウが提供する
認証オプションを通じて利用できる必要があります。電話やOSのパスキーの利用可否、
および実際のアカウントでのWhatsAppによる受け入れは、まだエンドツーエンドの検証が必要です。

この方法には、セットアップを実行するマシン上に対話型のターミナルとグラフィカルなブラウザが必要です。
ダッシュボードのオンボーディングやヘッドレスなSSHセッションでは利用できません。
ブラウザが必要なのはパスキーによるペアリングの間だけで、ペアリング後の通常の
メッセージングには不要です。

:::tip
QRコードが乱れて見える場合は、ターミナルの幅が少なくとも60カラムあり、Unicodeをサポートしていることを確認してください。別のターミナルエミュレータを試すこともできます。
:::

---

## ステップ2: 2つ目の電話番号を取得する（ボットモード）

ボットモードでは、まだWhatsAppに登録されていない電話番号が必要です。3つの選択肢があります。

| 選択肢 | コスト | 備考 |
|--------|------|-------|
| **Google Voice** | 無料 | 米国のみ。[voice.google.com](https://voice.google.com) で番号を取得します。Google VoiceアプリでSMS経由でWhatsAppを認証します。 |
| **プリペイドSIM** | $5〜15の一度きり | 任意のキャリア。アクティベートし、WhatsAppを認証すれば、あとはSIMを引き出しにしまっておけます。番号はアクティブに保つ必要があります（90日ごとに通話を1回行う）。 |
| **VoIPサービス** | 無料〜月$5 | TextNow、TextFreeなど。一部のVoIP番号はWhatsAppにブロックされます — 最初の番号がうまくいかない場合はいくつか試してください。 |

番号を取得した後は次のようにします。

1. 電話にWhatsAppをインストールする（またはデュアルSIMでWhatsApp Businessアプリを使う）
2. 新しい番号をWhatsAppに登録する
3. `hermes whatsapp` を実行し、そのWhatsAppアカウントからQRコードをスキャンする

---

## ステップ3: Hermesを設定する

`~/.hermes/.env` ファイルに次を追加します。

```bash
# Required
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" or "self-chat"

# Access control — pick ONE of these options:
WHATSAPP_ALLOWED_USERS=15551234567         # Comma-separated phone numbers (with country code, no +)
# WHATSAPP_ALLOWED_USERS=*                 # OR use * to allow everyone
# WHATSAPP_ALLOW_ALL_USERS=true            # OR set this flag instead (same effect as *)
```

:::tip 全員許可のショートハンド
`WHATSAPP_ALLOWED_USERS=*` を設定すると、**すべて**の送信者を許可します（`WHATSAPP_ALLOW_ALL_USERS=true` と同等）。
これは [Signalグループの許可リスト](../../reference/environment-variables.md)と一貫しています。
代わりにペアリングフローを使うには、両方の変数を削除して
[DMペアリングシステム](../security.md#dm-pairing-system)に頼ってください。
:::

`~/.hermes/config.yaml` のオプションの挙動設定:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `unauthorized_dm_behavior: pair` はグローバルのデフォルトです。未知のDM送信者はペアリングコードを受け取ります。
- `whatsapp.unauthorized_dm_behavior: ignore` は、未承認のDMに対してWhatsAppを沈黙させます。これはプライベートな番号には通常より良い選択です。

### グループチャット（ボットモード）

グループは、DMの許可リストではなく **グループポリシー** によって制御されます。`WHATSAPP_GROUP_POLICY` / `whatsapp.group_policy`
のデフォルトは `pairing` で、グループからは何も転送しません。`allowlist` と `WHATSAPP_GROUP_ALLOWED_USERS` /
`whatsapp.group_allow_from`（カンマ区切りの **グループJID**。例: `120363001234567890@g.us`）を組み合わせると、列挙した
グループを受け入れます。`open` はボットがメンバーであるすべてのグループを受け入れます。その後、送信者は他のゲートウェイの
プリンシパルと同様にチェックされます: `WHATSAPP_ALLOWED_USERS` が設定されている場合、参加者はそこに含まれている（またはペアリング済みである）必要があります — WhatsAppが
LIDで指定する送信者は、Baileysがそれと一緒に提供する電話番号を通じて照合されるため、まだ
`lid-mapping` ファイルのない最初の接触が破棄されることはありません。送信者の許可リストがない場合、
`allowlist` はグループJIDのリストだけを信頼し、列挙されたグループのすべての参加者を受け入れますが、`open` では依然として
参加者のペアリングまたは `WHATSAPP_ALLOW_ALL_USERS=true` が必要です。デフォルトでは、ボットは受け入れたすべてのグループ
メッセージに応答します。@メンション、ボットへの返信、
`/commands` にのみ応答させるには、`require_mention: true` / `WHATSAPP_REQUIRE_MENTION=true` を設定します（`free_response_chats` にあるグループは対象外）。正規表現のウェイクワードは `mention_patterns` に記述します。
キャプションのないボイスノートがそれらにマッチできるのは `voice_mention_patterns: true` の場合だけで、これはグループの
ボイスノートを先に文字起こしします（デフォルトはオフ: オンにすると、宛先のないすべてのボイスノートが音声認識プロバイダーに送られます。
そのターンではその文字起こしが再利用されます）。

その後、ゲートウェイを起動します。

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

ゲートウェイは、保存されたセッションを使ってWhatsAppブリッジを自動的に起動します。

---

## セッションの永続化

Baileysブリッジは、`~/.hermes/platforms/whatsapp/session` 配下にセッションを保存します。これは次を意味します。

- **セッションは再起動後も残ります** — 毎回QRコードを再スキャンする必要はありません
- セッションデータには暗号化キーとデバイス認証情報が含まれます
- **このセッションディレクトリを共有したりコミットしたりしないでください** — それはWhatsAppアカウントへの完全なアクセスを与えます
- Hermesはこのディレクトリを所有者のみ（`0700`）に保ち、ブリッジにキーファイルと `bridge.log`（QRペアリングのペイロードを含む）を `0600` で書き込ませます。マネージドな（NixOS/systemd）インストールやコンテナでは、サービスのumaskやマウントが設定するモードが維持されるため、それらは自分で制限してください

---

## 再ペアリング

セッションが壊れた場合（電話のリセット、WhatsAppの更新、手動でのリンク解除）、ゲートウェイのログに接続エラーが表示されます。修正するには次のようにします。

```bash
hermes whatsapp
```

これにより新しいQRコードが生成されます。再度スキャンすればセッションが再確立されます。ゲートウェイは**一時的な**切断（ネットワークの瞬断、電話が短時間オフラインになる等）を再接続ロジックで自動的に処理します。

---

## ボイスメッセージ

HermesはWhatsApp上でのボイスをサポートします。

- **受信:** ボイスメッセージ（`.ogg` opus）は、設定されたSTTプロバイダーを使って自動的に文字起こしされます: ローカルの `faster-whisper`、Groq Whisper（`GROQ_API_KEY`）、またはOpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`）
- **送信:** TTSによる応答は、MP3音声ファイルの添付として送信されます
- エージェントの応答には、デフォルトで「☤ **Hermes Agent**」が接頭辞として付きます。これは `config.yaml` でカスタマイズまたは無効化できます。

```yaml
# ~/.hermes/config.yaml
whatsapp:
  reply_prefix: ""                          # Empty string disables the header
  # reply_prefix: "🤖 *My Bot*\n──────\n"  # Custom prefix (supports \n for newlines)
  send_read_receipts: false                 # Mark accepted inbound messages as read (blue ticks)
```

`send_read_receipts` が `true` の場合、アダプターはDM/グループ/メンションのフィルタリングを通過した後、ポリシーで受け入れた受信メッセージを既読にします。拒否されたメッセージ（例: 許可リストにない送信者からのもの）は既読になりません。プライバシーのためデフォルトでは無効です。この設定を変更すると、他のブリッジ設定（DMとグループのポリシー、許可リスト、メンション設定、モード、`reply_prefix`、デバッグ、オーナーのメッセージの転送、メッセージ長、チャンクの遅延、送信タイムアウト、メディアキャッシュのディレクトリ）やブリッジの `package.json` を変更した場合と同様に、次の接続時に再ペアリングなしでブリッジのサブプロセスが再起動されます。

Hermesが送信したメッセージの配信状況と既読状況は、ステータスの変化ごとに1行ずつ `gateway.log` に記録されます（`server_ack`、`delivered`、`read`、`played`、またはWhatsAppのエラーコード付きの `error`）。グループでは、各受信者の通知がそれぞれ `recipient=` 付きの独自の行になります。リンクされた電話で人が入力したメッセージは記録されません。

```
[WhatsApp] WhatsApp delivery status: message=3EB0A1 chat=15551234567@s.whatsapp.net status=read at=1760000000
```

---

## メッセージのフォーマットと配信

WhatsAppは**ストリーミング（漸進的）応答**をサポートしています — ボットは、Discord や Telegram と同じように、AIがテキストを生成するにつれてメッセージをリアルタイムで編集します。内部的には、WhatsAppは配信機能の観点からTIER_MEDIUMプラットフォームに分類されます。

### チャンク分割

長い応答は、チャンクあたり**4,096文字**（WhatsAppの実用的な表示上限）で自動的に複数のメッセージに分割されます。何も設定する必要はありません — ゲートウェイが分割を処理し、チャンクを順番に送信します。

### WhatsApp互換のMarkdown

AI応答内の標準的なMarkdownは、自動的にWhatsApp独自のフォーマットに変換されます。

| Markdown | WhatsApp | 表示結果 |
|----------|----------|------------|
| `**bold**` | `*bold*` | **太字** |
| `~~strikethrough~~` | `~strikethrough~` | ~~取り消し線~~ |
| `# Heading` | `*Heading*` | 太字テキスト（ネイティブの見出しなし） |
| `[link text](url)` | `link text (url)` | インラインURL |

WhatsAppは三連バッククォートのフォーマットをネイティブでサポートしているため、コードブロックとインラインコードはそのまま保持されます。

### ツールの進行状況

エージェントがツール（Web検索、ファイル操作など）を呼び出すと、WhatsAppはどのツールが実行中かを示すリアルタイムの進行状況インジケーターを表示します。これはデフォルトで有効です — 設定は不要です。

### ネイティブの投票、投票としてのclarify、位置情報

Baileysブリッジのアダプター（ボットモード）は、いくつかのWhatsAppネイティブのメッセージタイプをサポートします:

- **投票** — エージェントは、ブリッジの `/send-poll` エンドポイントを介してWhatsAppネイティブの投票（質問 + 選択肢）を送信できます。投票結果は会話に戻ってきます。
- **投票としてのclarifyの質問** — エージェントが多肢選択のclarifyの質問をすると、ネイティブの単一選択の投票として表示され、選択肢をタップすると質問に回答できます。投票の送信に失敗した場合、アダプターはプレーンテキストの質問にフォールバックします。承認のプロンプトが投票に変換されることは**決してありません** — 投票は本当の多肢選択のclarifyにのみ使われます。
- **位置情報のピン** — エージェントは `/send-location` を介してネイティブの位置情報のピン（緯度/経度、任意の名前/住所）を送信でき、共有された位置情報（ライブ位置情報を含む）の受信は位置情報メッセージとしてエージェントに届けられます。

これらはすべてボット（Baileys）モードでそのまま動作します。設定は不要です。

### メッセージのバッチ処理（デバウンス）

WhatsAppは各メッセージを個別に配信するため、短時間の連続（転送されたまとまり、貼り付けの分割、複数行のテキスト）は、そのままでは断片ごとに別々のエージェント呼び出しを引き起こし、トークンを無駄にして、ばらばらな返信をいくつも生成してしまいます。アダプターは同じチャットからの連続したテキストメッセージをバッファリングし、短い静止期間（デフォルト **0.3秒**、非常に長い断片では **1秒** に延長。上限は2秒 / 4秒）の後に1つの結合されたリクエストとして送ります。`config.yaml` で調整します:

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    whatsapp:
      extra:
        text_batch_delay_seconds: 0.3         # quiet period before flushing a batch (max 2.0)
        text_batch_split_delay_seconds: 1.0   # extended delay near the split threshold (max 4.0)
```

各メッセージを即座に送るには `text_batch_delay_seconds: 0` を設定します（バッチ処理を無効化）。

### 引用返信

以前のメッセージに返信（引用）すると、エージェントは引用されたテキストをコンテキストとして受け取ります。画像、ボイスノート、動画、ドキュメントを引用すると、そのファイルもターンに添付されるため、引用した画像の下での「これは何?」が機能します — 添付ファイルが他の人からのものでも、ボット自身からのもの（cronで配信されたグラフ、生成された画像）でも同様です。WhatsAppは引用にはサムネイルのスタブしか付けないため、ファイルはブリッジのダウンロードキャッシュ（受信メディア。ブリッジの稼働中はメモリ内）またはボット自身の送信のローカルインデックス（直近1000メッセージ）から解決されます。それより古いものの引用は添付ファイルなしで届きます。

---

## 顧客向けのボット

デフォルトでは、許可リストに載っている人はあなたとあなたのチームであると想定しています。ボット番号が
顧客に応答する場合（`WHATSAPP_ALLOWED_USERS=*`、`WHATSAPP_ALLOW_ALL_USERS=true`、または顧客の
番号のリスト）、許可リストの全員が同じように信頼されるため、本番稼働の前にゲートウェイを
強化してください:

```yaml
# ~/.hermes/config.yaml
display:
  platforms:
    whatsapp:
      tool_progress: "off"              # or "log": no tool-progress bubbles in the chat
      interim_assistant_messages: false # no mid-turn commentary, only the final reply

platforms:
  whatsapp:
    home_channel:                       # or WHATSAPP_HOME_CHANNEL in ~/.hermes/.env
      platform: whatsapp
      chat_id: "15551234567@s.whatsapp.net"   # your operator chat, not a customer's
      name: Operator
    extra:
      allow_admin_from:                 # only these senders may run admin slash commands
        - "15551234567@s.whatsapp.net"
      group_allow_admin_from:           # same, for group chats (DM admins are not group admins)
        - "15551234567@s.whatsapp.net"
      # user_allowed_commands: [new]    # optional: commands customers may still run
```

- **`tool_progress`** はWhatsAppではデフォルトで `new` のため、顧客にはエージェントが実行する各ツールが表示されます。
- **`interim_assistant_messages`** はデフォルトで `true` のため、顧客にはエージェントのターン途中のコメントが表示されます。
- **ホームチャネル:** 設定がないと、新しいセッションでのすべての送信者の最初のメッセージ（新しい顧客、または
  `/new` の後の誰でも）に「type /sethome」というプロンプトが表示され、`/sethome` を実行した送信者が
  cronと通知の配信を乗っ取ります。自分自身のチャットに設定してください。
- **`allow_admin_from`**（DM）と **`group_allow_admin_from`**（グループ）は [スラッシュコマンドのアクセス制御](../../reference/slash-commands.md#permissions-and-adminuser-split) を有効にします:
  顧客に残るのは `/help` と `/whoami`（および `user_allowed_commands` にあるもの）だけで、
  `/sethome` は実行できず、`/sethome` のプロンプトも表示されません。列挙すべき正確なIDを得るには、自分の電話から
  `/whoami` を送信してください。

ボットモードで全員を許可していてホームチャネルが設定されていない場合、`hermes whatsapp` はここを案内する警告を表示します。

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| **QRコードがスキャンできない** | ターミナルが十分に広い（60カラム以上）ことを確認してください。別のターミナルを試してください。正しいWhatsAppアカウント（個人ではなくボット番号）からスキャンしていることを確認してください。 |
| **QRコードの有効期限切れ** | QRコードは約20秒ごとに更新されます。タイムアウトした場合は `hermes whatsapp` を再起動してください。 |
| **セッションが永続化されない** | `~/.hermes/platforms/whatsapp/session` が存在し、書き込み可能であることを確認してください。コンテナ化されている場合は、永続ボリュームとしてマウントしてください。 |
| **予期せずログアウトされる** | WhatsAppは長期間の非アクティブ後にデバイスのリンクを解除します。電話をオンにしてネットワークに接続したままにし、必要に応じて `hermes whatsapp` で再ペアリングしてください。 |
| **ブリッジがクラッシュまたは再接続ループする** | ゲートウェイを再起動し、Hermesを更新し、WhatsAppのプロトコル変更でセッションが無効化された場合は再ペアリングしてください。 |
| **WhatsApp更新後にボットが動作しなくなる** | Hermesを更新して最新のブリッジバージョンを取得し、再ペアリングしてください。 |
| **macOS: 「Node.jsがインストールされていない」だがterminalではnodeが動く** | launchdサービスはシェルのPATHを継承しません。`hermes gateway install` を実行して現在のPATHをplistに再スナップショットし、その後 `hermes gateway start` を実行してください。詳細は[ゲートウェイサービスのドキュメント](./index.md#macos-launchd)を参照してください。 |
| **メッセージが受信されない** | `WHATSAPP_ALLOWED_USERS` に送信者の番号（国番号付き、`+` やスペースなし）が含まれていることを確認するか、`*` を設定して全員を許可してください。`.env` に `WHATSAPP_DEBUG=true` を設定してゲートウェイを再起動すると、`bridge.log` で生のメッセージイベントを確認できます。 |
| **ボットが見知らぬ人にペアリングコードを返信する** | 未承認のDMを黙って無視させたい場合は、`~/.hermes/config.yaml` に `whatsapp.unauthorized_dm_behavior: ignore` を設定してください。 |

---

## セキュリティ

:::warning
本番稼働の前に**アクセス制御を設定**してください。`WHATSAPP_ALLOWED_USERS` に特定の電話番号（国番号付き、`+` なし）を設定するか、`*` で全員を許可するか、`WHATSAPP_ALLOW_ALL_USERS=true` を設定してください。これらのいずれも設定しない場合、ゲートウェイは安全策としてすべての受信メッセージを**拒否します**。
:::

デフォルトでは、未承認のDMにもペアリングコードの返信が送られます。プライベートなWhatsApp番号を見知らぬ人に対して完全に沈黙させたい場合は、次を設定します。

```yaml
whatsapp:
  unauthorized_dm_behavior: ignore
```

- `~/.hermes/platforms/whatsapp/session` ディレクトリには完全なセッション認証情報が含まれます — パスワードのように保護してください
- ファイルパーミッションを設定してください: `chmod 700 ~/.hermes/platforms/whatsapp/session`
- 個人アカウントからリスクを隔離するため、ボットには**専用の電話番号**を使ってください
- 侵害が疑われる場合は、WhatsApp → 設定 → リンク済みデバイス からデバイスのリンクを解除してください
- ログ内の電話番号は部分的にマスクされますが、ログの保持ポリシーを見直してください
