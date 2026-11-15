---
sidebar_position: 1
title: "Telegram"
description: "Hermes AgentをTelegramボットとしてセットアップする"
---

# Telegramのセットアップ

Hermes Agentは、フル機能を備えた対話型ボットとしてTelegramと統合できます。接続すると、任意のデバイスからエージェントとチャットしたり、自動で文字起こしされるボイスメモを送信したり、スケジュールされたタスクの結果を受け取ったり、グループチャットでエージェントを利用したりできます。この統合は [python-telegram-bot](https://python-telegram-bot.org/) を基盤としており、テキスト、音声、画像、ファイル添付に対応しています。

## クイックセットアップ（ダッシュボードとデスクトップアプリ）

[ダッシュボード](../features/web-dashboard.md) と [デスクトップアプリ](/docs/user-guide/desktop) の **Messaging → Telegram** ページには **Create with QR** ボタンがあります。Telegramでコードをスキャンする（またはリンクを開く）と、Hermesがボットを作成し、あなたのTelegramユーザーIDを検出し、プロファイルの `.env` に `TELEGRAM_BOT_TOKEN` と `TELEGRAM_ALLOWED_USERS` を書き込み、ゲートウェイを再起動します。ボットを自分で作成したい場合は、以下の手動の手順に従ってください。

## Team Gatewayを使うHermes Cloud

Team GatewayがオンになっているHermes Cloudインスタンスでは、Telegram（およびDiscordとSlack）は共有のHermesボットを通じてエージェントに到達します。自分のボットはそこでは実行されません: インスタンスがアイドル時にスリープできるよう、ゲートウェイは起動のたびに直接接続をオフにします。Telegramカードには **Create with QR** の代わりに **Via Team Gateway** が表示されます。

Nous Portalがエージェントに到達できる人を管理している場合（Portalで作成されたエージェント）は、Portalからアカウントをリンクします: カードにはリンク済みのアカウントが一覧表示され、その **Link accounts in Nous Portal** ボタンでPortal上のエージェントが開きます。リンクコードを取得するには、エージェントの **Message platforms** を使用します。

それ以外の場合は、カード自体がアカウントをリンクします:

1. **Link Telegram** をクリックします。Hermesは `/link AB12CD` のようなワンタイムメッセージ、QRコード、**Open Telegram** ボタンを表示します。
2. スマートフォンでQRコードをスキャンして（またはボタンをクリックして）共有ボットを開き、使用したいTelegramアカウントからそのメッセージを送信します。
3. ボットがメッセージを確認すると、カードにそのアカウントが表示されます。以降は、共有ボットにメッセージを送ってエージェントと会話します。

ボタンを機能させるには、ゲートウェイが実行中である必要があります。Team Gatewayの接続を保持しているのはゲートウェイだからです。代わりに共有ボットと並べて自分のボットを実行するには、インスタンスの環境で `GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true` を設定します。これによりTelegramへのライブ接続が開いたままになるため、インスタンスはスリープしなくなります。

## ステップ1: BotFatherでボットを作成する

すべてのTelegramボットには、Telegram公式のボット管理ツールである [@BotFather](https://t.me/BotFather) が発行するAPIトークンが必要です。

1. Telegramを開いて **@BotFather** を検索するか、[t.me/BotFather](https://t.me/BotFather) にアクセスします
2. `/newbot` を送信します
3. **表示名**（例: 「Hermes Agent」）を選びます — これは何でも構いません
4. **ユーザー名** を選びます — これは一意である必要があり、`bot` で終わる必要があります（例: `my_hermes_bot`）
5. BotFatherが **APIトークン** を返信します。次のような形式です:

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
ボットトークンは秘密にしてください。このトークンを持つ人は誰でもボットを操作できます。万が一漏洩した場合は、BotFatherで `/revoke` を使ってすぐに無効化してください。
:::

## ステップ2: ボットをカスタマイズする（任意）

以下のBotFatherコマンドはユーザー体験を向上させます。@BotFatherにメッセージを送り、次を使用します:

| コマンド | 用途 |
|---------|------|
| `/setdescription` | ユーザーがチャットを開始する前に表示される「このボットは何ができますか?」のテキスト |
| `/setabouttext` | ボットのプロフィールページに表示される短いテキスト |
| `/setuserpic` | ボットのアバターをアップロードする |
| `/setcommands` | コマンドメニュー（チャット内の `/` ボタン）を定義する |
| `/setprivacy` | ボットがグループ内のすべてのメッセージを見られるかどうかを制御する（ステップ3を参照） |

:::tip
`/setcommands` には、次のような便利な初期セットがあります:

```
help - Show help information
new - Start a new conversation
sethome - Set this chat as the home channel
```
:::

### オンライン/オフラインのステータス表示（任意）

Telegramボットには本当のオンライン/オフラインのプレゼンスドットはありません — あの緑のドットは *ユーザーアカウント* の機能であり、Bot APIがボットに公開しているものではありません。最も近いのはボットの **短い説明**（ボットのプロフィールで名前の下に表示される行）です。

`status_indicator` を有効にすると、Hermesはゲートウェイの接続時にその短い説明を **Online** に、正常なシャットダウン時に **Offline** に設定します:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        status_indicator: true
        # Optional custom strings (defaults: "Online" / "Offline"):
        status_online: "🟢 Online"
        status_offline: "🔴 Offline"
```

注意:

- 短い説明はチャット単位ではなく、ボットに対して **グローバル**（すべてのユーザーに表示）です。ユーザーはこれを開いているチャット内のライブバッジとしてではなく、ボットのプロフィールページで確認します。
- **正常な** ゲートウェイのシャットダウン（`/stop`、`disconnect`）のみが「Offline」を書き込みます。ハードクラッシュでは最後に知られたステータスが残ります — これはプロフィールテキストによる表示に固有の制限です。
- ボットのグローバルプロフィールを変更するため、デフォルトではオフです。
- Hermesは最初に **Online** を書き込む前に、ボット自身の短い説明（@BotFatherで設定したもの）を保存し、そのコピーを再起動やクラッシュをまたいで保持します。`status_indicator` をオフにすると、次の接続時にあなたのテキストが元に戻ります。ただし、その後@BotFatherで新しい説明を設定していた場合は除きます（あなたのものが保持されます）。ゲートウェイが停止している間に説明を編集すると、その新しいテキストがHermesの保存するものになります。接続時にHermesがあなたのテキストを読み取って保存できない場合、その実行ではステータスを書き込みません。

### コールドブート時の保留キュー（任意）

ゲートウェイがオフの間に送信されたメッセージ（夜間にデスクトップをシャットダウンした、再起動、アップデートなど）は、TelegramのBot APIキューで最大24時間待機します。デフォルトでは、アダプターはコールドブート時にそのキューを保持し、起動後にたまったメッセージに順番に応答します。再起動前にHermesがすでに処理した更新は、永続化された更新レシートによってスキップされるため、二重に応答されることはありません。

代わりに起動時にたまったメッセージを破棄するには（再起動が「クリーンな状態から始める」ことを意味する常時稼働サーバーなど）、`drop_pending_on_cold_boot: true` を設定します:

```yaml
platforms:
  telegram:
    extra:
      drop_pending_on_cold_boot: true
```

注意:

- デフォルトは `false` です: たまったメッセージには応答します。設定で明示的に指定した値は、どちらの場合も尊重されます。
- ウォッチャーの再接続（プロセスが生きたままの短いネットワーク障害）では、この設定に関わらず常にキューが保持されます。
- 競合からの回復（別のポーラーがボットの `getUpdates` セッションを保持している場合）でもキューは保持されます: 409競合中に送信されたメッセージは、ポーリングが再開されると配信されます。
- クラッシュ後、保持されたキューは、クラッシュしたインスタンスが部分的に処理した更新を再配信することがあります。通常はTelegramのオフセットがこれを防ぎますが、長時間の障害中に送信された時間依存のコマンドは起動時に実行されます。

### 更新の並行処理

異なるチャットからの更新は並行して処理されるため、1つの遅いターン（プロバイダーの長いリトライ、大きなダウンロード）が他のすべてのチャットを停滞させることはなくなりました。同じチャットからの更新は、引き続き到着順に1つずつ実行されます。チャット間のプールのデフォルトは32です。無効な値の場合は警告をログに出力し、デフォルトを使用します:

```yaml
platforms:
  telegram:
    extra:
      max_concurrent_updates: 32   # 1 restores fully sequential processing
```

### アルバム（任意）

Telegramはアルバム（まとめて送信された複数の写真やファイル）を、アイテムごとに1つの更新として配信します。Hermesは到着したすべてのアイテムのダウンロードが終わるまで待ち、アルバム全体を1つのメッセージとしてエージェントに送ります。遅いダウンロードについては最大8秒待ち、その後は手元にあるものを送ります。最後のアイテムの後、後から送られてくるアイテムのために短いデバウンス（デフォルト1.5秒、最大10秒）を待ちます。スマートフォンがアルバムのアイテムを大きな間隔を空けてアップロードする場合は、デバウンスを長くします:

```yaml
platforms:
  telegram:
    extra:
      media_group_wait_seconds: 3.0
```

短いテキストの直後に写真が続く場合（質問と、その対象のスクリーンショットなど）も、1つのメッセージとしてエージェントに届きます。テキストが送られる前に写真の到着が始まった場合、テキストは写真のダウンロードを最大8秒待ち、写真と結合されます。

### 繰り返し届く受信更新

Hermesは、メッセージのバッチ処理、コマンド/メディアの処理、観測されたグループ履歴の書き込み、プラグインのオブザーバーの前に、繰り返されたTelegramの `update_id` 値を抑制します。このチェックは受信アダプターと数値のボットIDによってスコープされます。テキストや `message_id` による重複排除は行いません。新しい更新IDを持つ本物の編集は引き続き処理されることがあります。

これは上限のある保護であり、厳密に1回だけ処理されることの保証ではありません:

- アダプターは、ボットごとに直近4096件の完了した受け入れを記憶します。アクティブな更新は、ディスパッチとそれによってスケジュールされたPTBハンドラーのタスク（ノンブロッキングのネイティブプラグインや登録済みのエラーコールバックを含む）が完了するまで、要求済みのまま保持されます。
- 完了した更新IDは、ボットごとにHermesホームの下にも保存され、24時間（Telegramが未配信の更新を保持する最長期間）保持されます。そのため、再接続、アダプターの再構築、ゲートウェイの再起動によって、すでに処理された更新（コールドブート後に応答したたまったメッセージを含む）に再び応答することはありません。24時間より古いレシートや、読み取れないレシートファイルは、その更新をもはや保護しません。
- 準備が失敗またはキャンセルされた場合、何も引き渡されていなければ要求は解放されます。更新がバッチ/保留キュー、ゲートウェイのディスパッチ、オブザーバー、またはネイティブプラグインに入ると、その後のエラーで再び開かれることはありません。ネイティブプラグインは自身の部分的な効果を自分で管理するため、その更新または登録済みのエラーコールバックに入ることは、保守的に引き渡しとして扱われます。PTB自身の例外ログは引き渡しではありません。キャッシュされていない静的ステッカーのビジョン解析も引き渡しです: awaitをキャンセルしても、すでに送信された補助モデルへのリクエストは取り消せません。引き渡しの前に捕捉された準備エラーは再試行可能なままです。意図的な拒否は終端です。
- 要求の解放は後の配信を許可するだけであり、Telegramに配信を要求するものではありません。ポーリングの確認応答はエージェントの完了とは独立しています。このチェックは、失敗した返信を再試行したり、下流のコンポーネントが独自に作業を重複させることを防いだりはしません。

遅れたリプレイが疑われる場合は、両方の発生について、ボット/プロファイル、チャット/トピック、`update_id`、更新の種類、`message_id`、実際の受信時刻を比較してください。編集は `message_id` を再利用することがあり、メッセージの送信タイムスタンプは受信時刻ではありません。

### コマンドメニューの優先順位と上限（任意）

HermesはTelegramゲートウェイの起動時にコマンドメニューを自動的に登録します。メニューは中央のスラッシュコマンドレジストリに、対象となるプラグイン/スキルのコマンドを加えて構築され、Telegramがペイロードを確実に受け付けるよう上限が設けられます。デフォルトの上限は60コマンドです — すべての組み込みコマンドに加えて、よく使うスキルコマンドを表示しておくのに十分な数です。

Telegramの `/` ピッカーに表示し続けたいスキル、プラグイン、組み込みのコマンドがある場合は、`~/.hermes/config.yaml` で優先順位を付けます:

```yaml
platforms:
  telegram:
    extra:
      command_menu:
        max_commands: 60
        priority_mode: prepend  # prepend | append | replace
        priority:
          - my_plugin_command
          - excalidraw       # skill commands work here too
```

`priority_mode` は、あなたのリストをHermesの組み込みの優先リストとどのように組み合わせるかを制御します:

- `prepend`: あなたのコマンドを先に置き、その後にHermesのデフォルト
- `append`: Hermesのデフォルトを先に保ち、その後にあなたのコマンド
- `replace`: 優先順位にはあなたのリストのみを使用

優先順位は、上限が適用される前に **結合された** 候補リスト（コアコマンド、プラグインコマンド、スキルコマンド）に適用されます — そのため、コアコマンドだけでメニューが埋まってしまう場合でも、優先順位を付けたスキルコマンドにはメニューの枠が保証されます。以前はスキルが常に最初にアルファベット順で削られていたため、アルファベット順で後ろのスキルは `priority` に関わらず表示されることがありませんでした。

Telegramは最大100個のBotCommandsを許可しますが、大きなコマンドペイロードは失敗することがあります。Hermesは信頼性のためにデフォルトを60とし、設定値を `1..100` に制限します。コマンドの完全なリストには `/commands` を使用してください。

### インラインコマンドピッカー: すべてのコマンドを検索（上限なし）

`/` メニューには上限がありますが、Telegramの **インラインモード** にはありません。有効にすると、任意のチャットで `@yourbotname` に続けて検索語を入力するだけで、Hermesの **すべての** コマンドとインストール済みのスキルを対象にした、ライブで検索可能なピッカーが表示されます — 結果はキー入力ごとに計算されページ分割されるため、何も削られることはありません:

```
@yourbotname plan            → tap the /plan result to send it
@yourbotname plan migrate auth to OIDC   → sends /plan migrate auth to OIDC
@yourbotname pdf             → finds skills matching "pdf" by name or description
```

最初の単語がカタログを絞り込み、それ以降はすべて送信されるコマンドの引数として引き継がれます。結果をタップすると、あなたからの通常のメッセージとしてコマンドが送信されるため、標準のコマンド経路を通じてディスパッチされます（コマンドで始まるメッセージは、プライバシーモードがオンでもボットに届きます）。

**一度だけのセットアップ:** インラインモードはすべてのTelegramボットでデフォルトでオフです。[@BotFather](https://t.me/BotFather) で `/setinline` を使って有効にします（ボットを選び、`Search commands and skills...` などの任意のプレースホルダーテキストを設定します）。それまではTelegramがインラインクエリを配信しないため、ピッカーは動作しません。

結果はゲートウェイの許可リストを通過したユーザーにのみ提供されます — 認可されていないユーザーには空のリストが返されるため、インストール済みのスキルカタログが見知らぬ人に公開されることはありません（インラインクエリは、ボットが参加していないチャットを含め、任意のチャットから送信できます）。

### フォーラムトピック

トピックが有効なTelegramスーパーグループでは、コマンドメニューはグローバルな `AllGroupChats` スコープから継承 **されません** — フォーラムスーパーグループごとに、そのチャットにスコープされたコマンド登録が必要です。Hermesはこれを自動的に処理します: フォーラムスーパーグループに初めてメッセージが届いたとき、ゲートウェイは同じ上限付きのメニューをそのチャットにスコープして（`BotCommandScopeChat`）遅延登録します。これはゲートウェイの実行ごと、フォーラムチャットごとに1回行われ、ユーザーには透過的です — 設定なしで、通常のグループと同じようにフォーラムトピックにも `/` メニューが表示されます。

## ステップ3: プライバシーモード（グループでは重要） {#step-3-privacy-mode-critical-for-groups}

Telegramボットには **プライバシーモード** があり、**デフォルトで有効** になっています。これは、グループでボットを使う際に最もよくある混乱の原因です。

**プライバシーモードがオンの場合**、ボットは以下のみを見ることができます:
- `/` コマンドで始まるメッセージ
- ボット自身のメッセージへの直接の返信
- サービスメッセージ（メンバーの参加/退出、ピン留めメッセージなど）
- ボットが管理者であるチャンネル内のメッセージ

**プライバシーモードがオフの場合**、ボットはグループ内のすべてのメッセージを受信します。

### プライバシーモードを無効にする方法

1. **@BotFather** にメッセージを送ります
2. `/mybots` を送信します
3. ボットを選択します
4. **Bot Settings → Group Privacy → Turn off** に進みます

:::warning
プライバシー設定を変更した後は、**ボットを各グループから削除して再度追加する必要があります**。Telegramはボットがグループに参加したときにプライバシー状態をキャッシュするため、ボットを削除して再追加するまで更新されません。
:::

:::tip
プライバシーモードを無効にする代わりの方法: ボットを **グループ管理者** に昇格させます。管理者ボットはプライバシー設定に関わらず常にすべてのメッセージを受信するため、グローバルなプライバシーモードを切り替える必要がなくなります。
:::

### 自動返信せずにグループの会話を観測する

OpenClaw/Yuanbaoスタイルのグループ動作にするには、ボットが通常のグループメッセージを **見る** ことはできるものの、直接トリガーされたときにのみ **応答する** ようにTelegramを設定します:

```yaml
telegram:
  allowed_chats:
    - "-1001234567890"
  group_allowed_chats:
    - "-1001234567890"
  require_mention: true
  observe_unmentioned_group_messages: true
```

このモードを有効にすると、明示的に許可リストに登録されたチャット/トピックからのメンションなしのグループメッセージは、観測されたコンテキストとして共有のチャット/トピックのセッショントランスクリプトに追加されますが、エージェントをディスパッチすることはありません。`allowed_chats` はボットがどこで応答するかをゲートし、`group_allowed_chats` は観測されたコンテキストに使われる共有のグループセッションを認可するため、このモードでは同じチャットIDを使用してください。同じ許可リスト登録済みのチャット/トピック内での後の `@botname` メンション、ボットへの返信、または設定したメンションパターンは、その観測されたコンテキストを利用できます。トリガーとなったメッセージには `[nickname|user_id]` のタグも付けられ、ターンごとの安全プロンプトが付与されるため、モデルは以前に観測された行を、ボットに向けた指示ではなくコンテキストとして扱います。そのチャット/トピックで送信されたスラッシュコマンド（`/new`、`/model`、`/undo`、スキルコマンド）は同じ共有セッションに作用しますが、管理者専用のコマンドは引き続き送信者に対してチェックされます。

観測されたコンテキストはテキストのみです。トリガーなしで送信された写真、動画、ファイルはキャッシュされ、観測された行にはその保存パスが記載されます。次に宛てられたターンでは、それがインラインで添付されることはありません。後のメッセージでそれについて尋ねられると、エージェントはキャッシュされたファイルを自分で開くため（画像には `vision_analyze`、オプトインの `video` ツールセットが有効な場合は動画に `video_analyze`、それ以外はファイルツール）、画像トークンは誰かが尋ねたときにのみ消費されます。

同等の環境変数:

```bash
TELEGRAM_ALLOWED_CHATS=-1001234567890
TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890
TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES=true
```

これにはTelegramが通常のグループメッセージをゲートウェイに配信する必要があるため、上記のとおりBotFatherのプライバシーモードを無効にするか、ボットをグループ管理者に昇格させてください。

## ステップ4: 自分のユーザーIDを確認する

Hermes Agentは、アクセスを制御するために数値のTelegramユーザーIDを使用します。あなたのユーザーIDはユーザー名 **ではなく**、`123456789` のような数値です。

**方法1（推奨）:** [@userinfobot](https://t.me/userinfobot) にメッセージを送ると、すぐにあなたのユーザーIDを返信します。

**方法2:** [@get_id_bot](https://t.me/get_id_bot) にメッセージを送ります — こちらも信頼できる選択肢です。

この数値を保存しておいてください。次のステップで必要になります。

## ステップ5: Hermesを設定する

### オプションA: 対話型セットアップ（推奨）

```bash
hermes gateway setup
```

プロンプトで **Telegram** を選択します。ウィザードがボットトークンと許可するユーザーIDを尋ね、設定を書き込んでくれます。

### オプションB: 手動設定

`~/.hermes/.env` に以下を追加します:

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # Comma-separated for multiple users
```

### ゲートウェイを起動する

```bash
hermes gateway
```

ボットは数秒以内にオンラインになるはずです。Telegramでメッセージを送って確認してください。

## Dockerベースのターミナルから生成ファイルを送信する

ターミナルバックエンドが `docker` の場合、Telegramの添付ファイルはコンテナ内部からではなく **ゲートウェイプロセス** によって送信されることに注意してください。つまり、最終的な `MEDIA:/...` のパスは、ゲートウェイが実行されているホスト上で読み取り可能でなければなりません。

よくある落とし穴:

- エージェントがDocker内のファイルを `/workspace/report.txt` に書き込む
- モデルが `MEDIA:/workspace/report.txt` を出力する
- `/workspace/report.txt` はコンテナ内部にのみ存在しホストには存在しないため、Telegramへの配信が失敗する

推奨パターン:

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

その上で:

- Docker内のファイルを `/output/...` に書き込む
- `MEDIA:` には **ホストから見える** パスを出力する。例:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`

すでに `docker_volumes:` セクションがある場合は、新しいマウントを同じリストに追加してください。YAMLでキーが重複すると、後のものが先のものを黙って上書きします。

### サポートされている `MEDIA:` のファイル拡張子

ゲートウェイはエージェントの返信から `MEDIA:/path/to/file` タグを抽出し、参照されたファイルをプラットフォームネイティブの添付ファイルとして送信します。すべてのゲートウェイプラットフォームでサポートされている拡張子:

| カテゴリ | 拡張子 |
|---|---|
| 画像 | `png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `tiff`, `svg` |
| 音声 | `mp3`, `wav`, `ogg`, `m4a`, `opus`, `flac`, `aac` |
| 動画 | `mp4`, `mov`, `webm`, `mkv`, `avi` |
| **ドキュメント** | `pdf`, `txt`, `md`, `csv`, `json`, `xml`, `html`, `yaml`, `yml`, `log` |
| **Office** | `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp` |
| **アーカイブ** | `zip`, `rar`, `7z`, `tar`, `gz`, `bz2` |
| **書籍 / パッケージ** | `epub`, `apk`, `ipa` |

このリストにあるものは、対応しているプラットフォーム（Telegram、Discord、Signal、Slack、WhatsApp、Feishu、Matrixなど）ではネイティブの添付ファイルとして配信されます。ネイティブにサポートしていないプラットフォームでは、リンクやプレーンテキストの表示にフォールバックします。**太字** のカテゴリは直近の数リリースで追加されたものです — これまでモデルに `here is the file: /path/to/report.docx` と言わせていた場合は、ネイティブ配信のために `MEDIA:/path/to/report.docx` に切り替えてください。

## Webhookモード

デフォルトでは、Hermesは **ロングポーリング** を使ってTelegramに接続します — ゲートウェイがTelegramのサーバーへ外向きのリクエストを行い、新しい更新を取得します。これはローカルや常時稼働のデプロイに適しています。

**クラウドデプロイ**（Fly.io、Railway、Renderなど）では、**Webhookモード** の方がコスト効率に優れます。これらのプラットフォームは、外向きの接続ではなく、内向きのHTTPトラフィックで一時停止中のマシンを自動的に起動できます。ポーリングは外向きなので、ポーリングするボットは決してスリープできません。Webhookモードはこの方向を逆転させます — TelegramがボットのhttpsのURLに更新をプッシュするため、アイドル時にスリープするデプロイが可能になります。

| | ポーリング（デフォルト） | Webhook |
|---|---|---|
| 方向 | ゲートウェイ → Telegram（外向き） | Telegram → ゲートウェイ（内向き） |
| 適した用途 | ローカル、常時稼働サーバー | 自動起動できるクラウドプラットフォーム |
| セットアップ | 追加設定なし | `TELEGRAM_WEBHOOK_URL` を設定 |
| アイドル時のコスト | マシンを稼働し続ける必要あり | メッセージ間でマシンをスリープ可能 |

### 設定

`~/.hermes/.env` に以下を追加します:

```bash
TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"  # required
# TELEGRAM_WEBHOOK_PORT=8443        # optional, default 8443
# TELEGRAM_WEBHOOK_HOST=::          # optional, bind host (e.g. IPv6-only)
```

| 変数 | 必須 | 説明 |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | はい | Telegramが更新を送信する公開httpsのURL。URLのパスは自動抽出されます（例: 上記の例では `/telegram`）。 |
| `TELEGRAM_WEBHOOK_SECRET` | **はい**（`TELEGRAM_WEBHOOK_URL` が設定されている場合） | 検証のためにTelegramがすべてのwebhookリクエストでエコーバックするシークレットトークン。これがないとゲートウェイは起動を拒否します — [GHSA-3vpc-7q5r-276h](https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h) を参照してください。`openssl rand -hex 32` で生成します。 |
| `TELEGRAM_WEBHOOK_PORT` | いいえ | webhookサーバーがリッスンするローカルポート（デフォルト: `8443`）。 |
| `TELEGRAM_WEBHOOK_HOST` | いいえ | webhookサーバーのローカルのバインドホスト。デフォルト: 未設定 → すべてのインターフェースでデュアルスタック（IPv4+IPv6）にバインド。IPv6のみのデプロイ（例: Fly.ioの6PN）や、リスナーを特定のインターフェースに固定したい場合に設定します。`config.yaml` の `platforms.telegram.extra.webhook_host` でも設定できます。 |

`TELEGRAM_WEBHOOK_URL` が設定されている場合、ゲートウェイはポーリングの代わりにHTTP webhookサーバーを起動します。設定されていない場合はポーリングモードが使われ、以前のバージョンから動作の変更はありません。

### クラウドデプロイの例（Fly.io）

1. Fly.ioアプリのシークレットに環境変数を追加します:

```bash
fly secrets set TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
fly secrets set TELEGRAM_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

2. `fly.toml` でwebhookポートを公開します:

```toml
[[services]]
  internal_port = 8443
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

3. デプロイします:

```bash
fly deploy
```

ゲートウェイのログに次が表示されるはずです: `[telegram] Connected to Telegram (webhook mode)`。

## プロキシのサポート

TelegramのAPIがブロックされている場合や、トラフィックをプロキシ経由でルーティングする必要がある場合は、Telegram専用のプロキシURLを設定します。これは汎用の `HTTPS_PROXY` / `HTTP_PROXY` 環境変数よりも優先されます。

**オプション1: config.yaml（推奨）**

```yaml
telegram:
  proxy_url: "socks5://127.0.0.1:1080"
```

**オプション2: 環境変数**

```bash
TELEGRAM_PROXY=socks5://127.0.0.1:1080
```

サポートされているスキーム: `http://`、`https://`、`socks5://`。

プロキシは、メインのTelegram接続とフォールバックIPトランスポートの両方に適用されます。Telegram専用のプロキシが設定されていない場合、ゲートウェイは `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`（またはmacOSのシステムプロキシ自動検出）にフォールバックします。

ホスト上でフォールバックIPの検出経路が正常に機能しない場合は、`HERMES_TELEGRAM_DISABLE_FALLBACK_IPS=true` を設定すると、コールド接続を通常の `api.telegram.org` の経路のままにできます。また、`HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT`（秒単位）でDNS-over-HTTPSによるフォールバック検出の時間に上限を設けることもできます。デフォルトは `5` です。

## ホームチャンネル

任意のTelegramチャット（DMまたはグループ）で `/sethome` コマンドを使うと、そのチャットを **ホームチャンネル** に指定できます。スケジュールされたタスク（cronジョブ）は、このチャンネルに結果を配信します。

`~/.hermes/.env` で手動で設定することもできます:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
グループチャットのIDは負の数値です（例: `-1001234567890`）。個人のDMチャットIDはあなたのユーザーIDと同じです。
`@username` が使えるのは公開チャンネルまたはスーパーグループのみです: 個人のハンドルに対してはTelegramが「Chat not found」と応答するため、DMをホームにする場合は数値のIDを使用してください。
:::

### トピックモードでのcron配信

ボットのDMでトピックモードを有効にしている場合、ルートチャットに配信されたcronメッセージはシステム専用のロビーに届きます — そこで返信してもセッションは開かれず、「main chat is reserved for system commands」という通知が表示されます。専用のフォーラムトピック（例: `Cron`）を作成し、次を設定します:

```bash
TELEGRAM_CRON_THREAD_ID=<topic_thread_id>
```

`TELEGRAM_CRON_THREAD_ID` は、cron配信に限って `TELEGRAM_HOME_CHANNEL_THREAD_ID` を上書きします。そのトピックでの返信は、トピックの既存のセッションを継続します。

## 共有された連絡先と位置情報

ボットに共有した連絡先カードや位置情報のピンは、短いテキストブロック（連絡先の場合は名前、電話番号、vCard。位置情報の場合は座標と施設）としてエージェントに届くため、エージェントはそれに基づいて行動できます。

**ライブ位置情報** は、共有を開始したときに一度だけエージェントに届きます。その後、共有が続いている間、Telegramは数秒から1分ごとに位置を更新します。更新がエージェントへの新しいメッセージになるのは、最後にエージェントに伝えられた位置から約75m移動した後、または同じ場所に15分間とどまった後だけです。静止している間のGPSのぶれでターンが始まることはありません。静的なピン（または施設）は、同じ場所を2回送った場合でも常にエージェントに届きます。

## ステッカー

静的なステッカーは、短いテキストのメモとしてエージェントに届きます: その絵文字、パック名、そしてビジョンモデル（`auxiliary.vision`、デフォルトはメインモデル）が書いた1〜2文の説明です。説明はステッカーごとに（Telegramの `file_unique_id` によって）プロファイルのHermesホームの下にある `sticker_cache.json` にキャッシュされるため、同じステッカーを再度送ってもビジョンの呼び出しは発生しません。ビジョンの呼び出しが失敗した場合、メモは絵文字のみにフォールバックします。アニメーションステッカーと動画ステッカーは解析されません: エージェントには絵文字のみのメモが届きます。

## ボイスメッセージ

### 受信ボイス（音声からテキストへ）

Telegramで送信したボイスメッセージは、Hermesに設定されたSTTプロバイダーによって自動的に文字起こしされ、テキストとして会話に挿入されます。

- `local` はHermesを実行しているマシン上で `faster-whisper` を使用します — APIキー不要
- `groq` はGroq Whisperを使用し、`GROQ_API_KEY` が必要です
- `openai` はOpenAI Whisperを使用し、`VOICE_TOOLS_OPENAI_KEY` が必要です

#### STTをスキップする: 生の音声ファイルをエージェントに渡す

音声を **エージェント自身** に処理させたい場合 — 話者分離（ダイアライゼーション）、カスタムの文字起こしツール、または単に録音のアーカイブのため — は、`~/.hermes/config.yaml` で `stt.enabled: false` を設定します:

```yaml
stt:
  enabled: false
```

STTが無効の場合でも、ゲートウェイはボイス/音声の添付ファイルをHermesの音声キャッシュにダウンロードしますが、**文字起こしは行いません**。エージェントは次のようなマーカー付きでメッセージを受け取ります:

```
[The user sent a voice message: /home/<user>/.hermes/cache/audio/<hash>.ogg]
```

その後、ツールやスキルがそのパスを直接読み取れます（例: ローカルの話者分離パイプラインや、より高性能な文字起こしモデルに渡す、または長期ストレージにアップロードする）。ファイルの拡張子は、Telegramが配信した元の形式を反映します（ボイスノートは `.ogg`、音声の添付ファイルは `.mp3`/`.m4a` など）。

これは、Telegramの20MBのgetFile上限を2GBに引き上げる、以下の [ローカルBot APIサーバー](#large-files-20mb-via-local-bot-api-server) のセクションと自然に組み合わせられます — 処理したい録音が数分より長い場合に便利です。

### 送信ボイス（テキストから音声へ）

エージェントがTTSで音声を生成すると、Telegramネイティブの **ボイスバブル**（丸い、インラインで再生できるタイプ）として配信されます。

- **OpenAIとElevenLabs** はOpusをネイティブに生成します — 追加のセットアップは不要です
- **Edge TTS**（デフォルトの無料プロバイダー）はMP3を出力するため、Opusへの変換に **ffmpeg** が必要です:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

ffmpegがない場合、Edge TTSの音声は通常の音声ファイルとして送信されます（再生は可能ですが、ボイスバブルではなく四角形のプレーヤーが使われます）。

MP3とM4AのファイルはTelegramの音声トラックとして送信されます。そのキャプションには、ボイスノートと同じmarkdownの書式が適用されます。ffmpegがインストールされている場合は、ファイル自体のアーティストとタイトルのタグ、埋め込まれたカバーアートによって、Telegramの音楽プレーヤーの演奏者、トラック名、アルバムアートも埋められます。

TTSプロバイダーは `config.yaml` の `tts.provider` キーで設定します。

## ローカルBot APIサーバーによる大きなファイル（20MB超） {#large-files-20mb-via-local-bot-api-server}

Telegramの **公開** Bot APIは `getFile` のダウンロードを **20 MB** に制限しているため、それより大きなボイスノート、音声ファイル、動画、ドキュメントは、Hermesによって「too large」という返信とともに黙って拒否されます。これを回避する文書化された方法は、**ローカル** の [telegram-bot-api](https://github.com/tdlib/telegram-bot-api) デーモンを実行することです — Telegramが使用しているのと同じサーバーソフトウェアを、あなたのネットワーク上で実行します。ローカルサーバーはファイルの上限を **2 GB** に引き上げ、Hermesはカスタムの `base_url` が設定されていることを検出すると、自身の内部上限を自動的に引き上げます。

これにより、次のようなワークフローが可能になります:

- 長いボイスメモ（45分の会議、ポッドキャスト）をボットに送信する
- ビジョンツールで処理するために大きな動画をアップロードする
- 話者分離、アライメント、学習データなどのオフラインパイプラインのために生の音声をアーカイブする

### ステップ1: TelegramのAPI認証情報を取得する

ローカルサーバーは（公開Bot APIではなく）TelegramのMTProto層と直接通信するため、**MTProtoの認証情報** が必要です:

1. [my.telegram.org/apps](https://my.telegram.org/apps) にアクセスし、Telegramアカウントでサインインします。
2. 新しいアプリケーションを作成します（名前と短い説明は何でも構いません）。
3. `api_id` と `api_hash` をコピーします — どちらも必要です。

### ステップ2: telegram-bot-apiサーバーを実行する

コミュニティがメンテナンスしている [`aiogram/telegram-bot-api`](https://hub.docker.com/r/aiogram/telegram-bot-api) のDockerイメージを使うのが最も簡単です。最小限の `docker-compose.yaml`（より高い上限を有効にするには `--local` モードを使用します）:

```yaml
services:
  tg-bot-api:
    image: aiogram/telegram-bot-api:latest
    container_name: tg-bot-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:8081"   # bind to loopback only; see security note
    environment:
      TELEGRAM_API_ID: "12345"           # your api_id from Step 1
      TELEGRAM_API_HASH: "abcdef..."     # your api_hash from Step 1
      TELEGRAM_LOCAL: "1"                # enable --local mode (raises 20MB → 2GB)
    volumes:
      - ./tg-bot-api-data:/var/lib/telegram-bot-api
```

起動します:

```bash
docker compose up -d tg-bot-api
docker logs --tail 20 tg-bot-api
```

:::warning セキュリティ
ローカルBot APIサーバーは、URLパス（例: `/bot<TOKEN>/getMe`）でボットトークンを受け取り、**追加の認証はありません**。ポートに到達できる人は誰でもボットを完全に操作できます — ボットが見られるすべてのメッセージを読む、ボットとしてメッセージを送信する、などです。コンテナを `127.0.0.1` にバインドするか、プライベートネットワーク上のリバースプロキシの背後に置くか、またはその両方を行ってください。**ポート8081を決して公開インターネットに公開しないでください。**
:::

### ステップ3: 公開APIからボットをログアウトする（一度だけ）

ボットは一度に **1つ** のBot APIサーバーでのみアクティブにできます。ボットがすでに `api.telegram.org` に対して実行されていた場合（ほぼ確実にそうです）、ローカルサーバーがボットを受け入れる前に、そちらで明示的にログアウトする必要があります:

```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/logOut"
# expected response: {"ok":true,"result":true}
```

これは一度きりの移行ステップです — 再起動のたびに繰り返す必要はありません。`logOut` の後に受信したメッセージは、Telegramが新しいサーバー経由で配信します。

ローカルサーバーがボットに代わってTelegramと通信できることを確認します:

```bash
curl "http://127.0.0.1:8081/bot<YOUR_BOT_TOKEN>/getMe"
# expected response: {"ok":true,"result":{"id":...,"is_bot":true,...}}
```

### ステップ4: Hermesをローカルサーバーに向ける

`~/.hermes/config.yaml` の `platforms.telegram.extra` の下にURLを追加します:

```yaml
platforms:
  telegram:
    extra:
      base_url: "http://127.0.0.1:8081/bot"
      base_file_url: "http://127.0.0.1:8081/file/bot"
      local_mode: true        # see Step 5 below — only set this if the bot's data
                              # directory is readable by the Hermes process
```

:::caution `telegram.extra` ではなく `platforms.telegram.extra` を使用してください
現時点では、`platforms.<name>.extra` の形式のみがプラットフォーム設定にディープマージされます。トップレベルの `telegram.extra` ブロックの直下に置いたキーは黙って破棄されます。
:::

`base_url` が設定されている場合、Hermesは次のことを行います:

- ローカルサーバーに対してpython-telegram-botのクライアントを構築する
- 内部のドキュメント/音声のサイズ上限を20 MB → 2 GBに自動的に引き上げる
- 「too large」エラーメッセージに有効な上限（`Maximum: 2048 MB.`）を表示し、どのモードにいるかが明確にわかるようにする
- ゲートウェイの外でも同じサーバーを通じて送信する: `hermes send`、`send_message` ツール、スタンドアロンの送信機能にフォールバックするcron配信

ゲートウェイを再起動し、確認のログ行を探します:

```bash
hermes gateway restart
grep -E "Using custom Telegram base_url|Using Telegram local_mode" ~/.hermes/logs/gateway.log | tail
```

### ステップ5: `local_mode` — ディスク上のファイルアクセス

ローカルサーバーには、ファイルを配信する **2つの方法** があります:

1. **`--local` なし**（デフォルト）: ファイルは公開Bot APIと同じく、`/file/bot<TOKEN>/<path>` でHTTP経由で提供されます。20MBの上限は有効なままです。ネットワークの問題を解決する目的（例: `api.telegram.org` に到達できないがセルフホストはできる場合）にのみ有用で、サイズ上限の引き上げには向きません。
2. **`--local` あり**（上記の `TELEGRAM_LOCAL=1` で設定）: ファイルはサーバーのファイルシステムに書き込まれ、`getFile` のレスポンスはHTTPのURLではなく **絶対パス** を返します。20MBの上限は解除されます。その場合、HermesはHTTP経由ではなく **ディスクから** バイトを読み取る必要があります。

ディスクからの読み取り経路を機能させるには、上記の設定で `local_mode: true` を設定し、**かつ** サーバーが返すパスをHermesのプロセスが読み取れるようにしてください。2つのシナリオがあります:

- **同じマシン** — telegram-bot-apiとHermesが同じホストで実行されている場合。データボリュームをHermesが読み取れるディレクトリ（例: `/var/lib/telegram-bot-api`）にバインドマウントし、ファイルの所有者が一致するようにします。コンテナは内部の `telegram-bot-api` ユーザー（uidはイメージによって異なります）に権限を落とします。最も簡単な解決策は、composeのサービスに `user: "<UID>:<GID>"` を追加し、Hermesがすでに実行されているuidでファイルが所有されるようにすることです。
- **異なるマシン** — ボットサーバーが1つのホスト（例: NAS、別のVM）で、Hermesが別のホストで実行されている場合。サーバーのデータディレクトリは、サーバーが報告するのと **同じ絶対パス**（通常は `/var/lib/telegram-bot-api`）でHermesのマシンと共有されている必要があります。これにはNFSがうまく機能します。ファイルシステムレベルでuidの不一致に対処したくない場合は、`uid=` によるマウントの再マッピングを備えたCIFS/SMBの方が扱いやすいでしょう。

`local_mode: true` が設定されているのに、Hermesが返されたファイルパスを `stat` できない場合（権限やマウントの誤り）、python-telegram-botは黙ってローカルサーバーに対するHTTPの `getFile` にフォールバックします — これは `--local` モードでは `404 Not Found` を返します。この症状は `gateway.log` に次のように表示されます:

```
[Telegram] Failed to cache voice: Not Found
telegram.error.InvalidToken: Not Found
```

これが表示される場合、上限の引き上げは機能していますが、ファイル共有が機能していません。Hermesのホストから、ゲートウェイを実行しているユーザーとして `ls -la /var/lib/telegram-bot-api/<TOKEN>/voice/` を確認し、1つのファイルが権限エラーなしで `cat` できることを確認してください。

**別のパスにマウントされたストレージ。** Hermesが別のコンテナや別のホストで実行されていて、サーバーのデータディレクトリが別の場所（例: composeファイルの `./tg-bot-api-data`）にマウントされている場合は、その場所をHermesに伝えます:

```yaml
platforms:
  telegram:
    extra:
      local_mode: true
      local_bot_api_root: "/home/me/tg-bot-api-data"   # the host path of the server's /var/lib/telegram-bot-api
```

サーバーは自身のパス（`/var/lib/telegram-bot-api/<TOKEN>/videos/file_7.MOV`、`base_file_url` の後ろにある場合もあります）を報告します。Hermesはトークンディレクトリより後の部分を取り出し、`<local_bot_api_root>/<TOKEN>/videos/file_7.MOV` を読み取ります。トークンディレクトリ自体をマウントした場合は `<local_bot_api_root>/videos/file_7.MOV` を読み取ります。これはディレクトリのスキャンを伴わない2回のファイルチェックであり、`..` を含むパスは決してたどられません。どちらのファイルも存在しない場合は、通常のダウンロードが以前と同様に実行されます。

### ステップ6: テストする

20 MBより大きいボイスノートまたは音声ファイルをボットに送信します。ゲートウェイのログを追跡します:

```bash
tail -f ~/.hermes/logs/gateway.log | grep -iE "telegram|cache"
```

`[Telegram] Cached user voice at /home/<user>/.hermes/cache/audio/...` の行が表示され、「too large」による拒否が **ない** はずです。（上記の）`stt.enabled: false` と組み合わせると、元の音声ファイルへのパスが、下流の処理のためにエージェントの受信メッセージに含まれます。

## グループチャットでの利用

Hermes Agentはいくつかの注意点を踏まえつつTelegramのグループチャットで動作します:

- **プライバシーモード** がボットの見られるメッセージを決定します（[ステップ3](#step-3-privacy-mode-critical-for-groups)を参照）
- `TELEGRAM_ALLOWED_USERS` は引き続き適用されます — グループ内であっても、認可されたユーザーのみがボットを起動できます
- `telegram.require_mention: true` を使うと、通常のグループの雑談にボットが反応しないようにできます
- `telegram.require_mention: true` の場合、グループメッセージは次のときに受け付けられます:
  - ボットのメッセージへの返信
  - `@botusername` のメンション
  - `/command@botusername`（ボット名を含むTelegramのボットメニューのコマンド形式）
  - `telegram.mention_patterns` に設定した正規表現のウェイクワードのいずれかに一致するもの
- 複数のHermesボットがいるグループでは、`telegram.exclusive_bot_mentions` によってルーティングが決定的に保たれます。メッセージが1つ以上のTelegramボットのユーザー名に宛てられている場合、メンションされたボットのプロファイルのみがそれを処理します。他のHermesボットは、返信やウェイクワードによるフォールバックが実行される前にそれを無視します（そして `Skipping message ... addressed to other bot(s)` の行を1行ログに出力します）。メッセージが別のボットに宛てられているとみなされるのは、そのボットのハンドルで始まる場合のみです（`@OtherBot do this`、`@BotA @BotB: compare`、`/status@OtherBot`）。`the shop has its own bot now (@ShopBot)` のようにテキストの後半にあるハンドルは、そのボットについて話しているだけであり、このボットを沈黙させることはありません。このボット自身のハンドルがどこかにあれば、冒頭のコマンド（任意でこのボットへの呼びかけの後）が明示的に別のボットを対象としていない限り、引き続きこのボットが優先されます。デフォルトで有効です。
- BotFatherでボットの `@username` を変更すると自動的に反映されます — Hermesはゲートウェイを再起動することなく、メンションのルーティングに新しいハンドルを使用します。`bot` で終わらないコレクティブル（Fragment）ユーザー名もサポートされています。
- `telegram.ignored_threads` を使うと、グループが本来は自由な応答やメンションによる返信を許可する場合でも、特定のTelegramフォーラムトピックでHermesを沈黙させることができます
- `telegram.allowed_topics` はフォーラムトピックのホワイトリストとして使用します — 設定すると、Hermesはそれ以外のグループ/フォーラムトピックからのメッセージを無視し、チャット/トピックごとに最初の破棄をINFOでログに出力します（DMはフィルタリングされません）。TelegramのGeneralトピックと、`message_thread_id` のないグループメッセージは、トピックID `1` に正規化されます。ホワイトリストの評価は `ignored_threads` より前に実行されます。
- `telegram.free_response_chats` / `telegram.free_response_topics` を使うと、チャット全体または単一のフォーラムトピックに限って `require_mention` をバイパスできます。
- `telegram.require_mention` が未設定またはfalseの場合、Hermesは以前のオープングループの動作を維持し、見られる通常のグループメッセージに応答します

### グループでボットを沈黙させる {#letting-the-bot-stay-silent-in-a-group}

すべてのグループメッセージを読むボットは、メッセージに自分が必要ない場合、`[SILENT]`（または別の [沈黙トークン](index.md#intentional-silence-tokens)）だけを答えることができます。その沈黙がそのまま通るかどうかは、メッセージが誰に宛てられているかによります:

| グループメッセージ | 沈黙トークンのみの返答 |
|---|---|
| ボットへのメンション、ボットへの返信、`mention_patterns` への一致、または `/command` | 短い通知に置き換えられるため、ボットへの質問が消えてしまうことはありません |
| 別のメンバーへの返信、または他の人への@メンションで始まるもの | 沈黙のまま |
| `?` を含まないその他のメッセージ | 沈黙のまま |
| `?` を含むその他のメッセージ（ボットへの質問である可能性がある） | 通知に置き換えられる |

ボット自身のIDが不明な間や、作成者が不明なメッセージ、またはチャットとして投稿されたメッセージ（リンクされたチャンネルの投稿、匿名の管理者）への返信の場合も、メッセージには通知が付きます。キャプションのないボイスノート、ビデオノート、音声ファイルも、その言葉がメッセージのテキストに含まれていないため通知が付きます。フォーラムトピックのルートメッセージは、別のメンバーへの返信ではありません。

### 1つのグループに複数のHermesボット

同じTelegramグループで複数のHermesプロファイルを実行する場合は、プロファイルごとに1つのTelegramボットトークンを作成し、プロファイルごとに1つのゲートウェイを起動します。同じボットトークンを複数の実行中のゲートウェイで再利用しないでください。Telegramは同じトークンに対する同時ポーリングを拒否します。

推奨されるグループ設定:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns: []
```

この設定では、`@research_bot @ops_bot summarize this` のようなグループメッセージは `research_bot` と `ops_bot` のみによって処理されます。グループ内の他のHermesボットは、そのメッセージが自分の以前のメッセージへの返信である場合や、共有のウェイクワードに一致する場合でも、沈黙を保ちます。

互いの引用返信に答え合う2つのHermesボットは、`TELEGRAM_ALLOW_BOTS=all` の場合、依然として無限にループする可能性があります。ボットへの返信は常に `require_mention` のゲートを通過するためです。`telegram.bots_require_mention: true`（環境変数 `TELEGRAM_BOTS_REQUIRE_MENTION`）を設定するとその経路が閉じられます: 別のボットからのメッセージは、明示的にこのボットを `@mentions` した場合にのみ応答をトリガーし、人間の返信はこれまでどおり動作します。これは `free_response_chats` / `free_response_topics` でも同様です: これらは *人間* がメンションなしでHermesをトリガーできるようにしますが、別のボットは引き続き明示的に `@mention` する必要があります。

ボット間ループガードも、ボットが作成したメッセージが受け入れられるすべてのチャット（`TELEGRAM_ALLOW_BOTS` が `mentions` または `all` に設定されている場合）を計測します。1つのチャットで5分以内に20件のボットメッセージが届くと、そのチャットでのそれ以降のボットメッセージは10分間破棄され、警告が1回ログに出力されます。人間のメッセージがカウントされたり破棄されたりすることはありません。設定は `config.yaml` にあります:

```yaml
gateway:
  bot_loop_guard:
    enabled: true        # false turns the guard off
    max_events: 20       # bot messages per chat per window
    window_seconds: 300
    cooldown_seconds: 600
```

1つのチャットに5分間で20件を超えるメッセージを投稿する正当な大量送信ボットもガードに引っかかります。そのゲートウェイでは `max_events` を引き上げてください。

同じブロックは、その予算よりもずっと前にボット間のやり取りを終わらせることもあります。あなたのメッセージがやり取りを開始します。その後、各ボットメッセージが1ホップとなり、`max_hops`（デフォルト4）を超えると、それ以降のボットメッセージはターンを開始しません。1つのやり取りの中で同じテキストを繰り返すボットは1回としてカウントされます。`bot_replies_expect_reply: false`（デフォルト）の場合、Hermesは、Hermes自身の返信に答えているボットには応答しません。また、Hermesはランタイムの通知をボットに送ることもありません: ビジー時や割り込み時の確認応答、進捗やステータスの行、「not processed」や再試行のヒント、モデルのフォールバック通知などです。ボットが受け取るのはモデルの返信だけです。

```yaml
gateway:
  bot_loop_guard:
    max_hops: 4
    bot_replies_expect_reply: false
```

メッセージが他の参加者にも言及している場合、グループの会話テキストとメディアのキャプションはすべてのメンションを保持します（`@research_bot , @ops_bot are you both listening?` はそのまま `research_bot` に届きます）。このボットだけが宛先の場合は、`@hermes_bot 2` のような短い回答が機能し続けるよう、ボット自身のハンドルは引き続き取り除かれます。グループのターンでは、チャンネルごとのコンテキストにボット自身のTelegramユーザー名も含まれるため、モデルは保持されたメンションのうちどれが自分宛てかを判断できます。スラッシュコマンドでは、引き続き通常のコマンドトリガーのクリーンアップが使われます。

`exclusive_bot_mentions: false` は、明示的なメンションが返信やウェイクワードによるトリガーを上書きすべきでない従来のグループでのみ設定してください。

複数のプロファイルを運用するには、プロファイルごとに1回ずつゲートウェイコマンドを実行します。例:

```bash
# default profile
hermes gateway start
hermes gateway status
hermes gateway stop

# named profiles
hermes -p research gateway start
hermes -p research gateway status
hermes -p research gateway stop
```

小規模な固定のフリートでは、デフォルトプロファイルには `hermes gateway <action>` を、名前付きプロファイルごとに `hermes -p <profile> gateway <action>` を呼び出すシェルループやスクリプトを使用してください。これは、単一のプロセスレベルのコマンドがすべてのサービスマネージャー上ですべての名前付きプロファイルを制御すると想定するよりも信頼性があります。

### トラブルシューティング: DMでは動くがグループでは動かない

ボットがプライベートチャットでは応答するのにグループでは沈黙する場合は、次のゲートを順番に確認してください:

1. **Telegramの配信:** BotFatherのプライバシーモードをオフにする、ボットを管理者に昇格させる、またはボットを直接メンションします。HermesはTelegramがボットに配信しないグループメッセージには応答できません。
2. **プライバシー変更後の再参加:** BotFatherのプライバシー設定を変更した後は、ボットをグループから削除して再度追加します。Telegramは既存のメンバーシップに対して古い配信動作を保持することがあります。
3. **Hermesの認可:** 送信者が `TELEGRAM_ALLOWED_USERS` または `TELEGRAM_GROUP_ALLOWED_USERS` に登録されているか、もしくは `TELEGRAM_GROUP_ALLOWED_CHATS` でグループチャットが許可されていることを確認します。
4. **メンションフィルター:** `telegram.require_mention: true` が設定されている場合、メッセージがスラッシュコマンド、ボットへの返信、`@botusername` のメンション、設定した `mention_patterns` への一致のいずれかでない限り、通常のグループの雑談は無視されます。
5. **複数ボットのルーティング:** グループに複数のボットがいる場合は、各Hermesプロファイルが一意のボットトークンを使用していることを確認し、意図的に従来の共有トリガーの動作を望む場合を除き、`exclusive_bot_mentions` を有効のままにしてください。

負のチャットIDはTelegramのグループやスーパーグループでは正常です。チャット単位の認可を使う場合は、それらのIDを送信者ユーザーの許可リストではなく `TELEGRAM_GROUP_ALLOWED_CHATS` に入れてください。

### グループトリガー設定の例

`~/.hermes/config.yaml` に次を追加します:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns:
    - "^\\s*chompy\\b"
  ignored_threads:
    - 31
    - "42"
```

この例では、通常の直接トリガーすべてに加えて、`@mention` を使わない場合でも `chompy` で始まるメッセージを許可します。
Telegramトピック `31` と `42` のメッセージは、メンションチェックと自由応答チェックが実行される前に常に無視されます。

### `mention_patterns` に関する注意

- パターンにはPythonの正規表現を使用します
- マッチは大文字小文字を区別しません
- パターンはテキストメッセージとメディアのキャプションの両方に対してチェックされます
- キャプションのないボイスノートにはテキストがないため、返信または@メンションによってのみボットを起動できます。`voice_mention_patterns: true` を設定すると、そのようなグループのボイスノートをまず文字起こしし、話された内容に対してこれらのパターンを照合します。グループ内の宛先のないすべてのボイスノートが音声認識プロバイダーに送られることになるため、デフォルトではオフです。ターンではその文字起こし結果が再利用されます（STTの呼び出しは1回）
- 無効な正規表現パターンは、ボットをクラッシュさせる代わりにゲートウェイのログに警告を出して無視されます
- メッセージの先頭でのみパターンを一致させたい場合は、`^` でアンカーします

## プライベートチャットトピック（Bot API 9.4） {#private-chat-topics-bot-api-94}

Telegram Bot API 9.4（2026年2月）は **プライベートチャットトピック** を導入しました — ボットはスーパーグループを使わずに、1対1のDMチャット内に直接フォーラム形式のトピックスレッドを作成できます。これにより、Hermesとの既存のDM内で複数の独立したワークスペースを運用できます。

### ユースケース

長期間にわたるプロジェクトを複数抱えている場合、トピックはそれぞれのコンテキストを分離して保ちます:

- **トピック「Website」** — 本番のWebサービスでの作業
- **トピック「Research」** — 文献レビューと論文の探索
- **トピック「General」** — その他のタスクや簡単な質問

各トピックは独自の会話セッション、履歴、コンテキストを持ち、他とは完全に分離されています。

### 設定

:::caution 前提条件
設定にトピックを追加する前に、ボットのオーナーが **@BotFather** でボットの **Threaded Modeを有効にする** 必要があります:

1. BotFatherの **Mini App** を開きます（Telegramで `botfather` を検索し、検索結果の **Open** をタップします — 従来の `/mybots` テキストメニューではこの設定は表示されません）
2. **My bots → your bot → Bot Settings → Threads Settings** に進みます
3. **Threaded Mode** をオンにします

DMチャット自体には「Topics」のトグルはありません — ボットのDMはグループではないため、一部の古いガイドで説明されているグループフォーラムのトグルはここでは当てはまりません。Threaded Modeがないと、Hermesは起動時に `The chat is not a forum` をログに出力し、トピックの作成をスキップします。同じ手順のより詳しい説明は、下の [前提条件](#prerequisites) を参照してください。
:::

`~/.hermes/config.yaml` の `platforms.telegram.extra.dm_topics` の下にトピックを追加します:

```yaml
platforms:
  telegram:
    extra:
      dm_topics:
      - chat_id: 123456789        # Your Telegram user ID
        topics:
        - name: General
          icon_color: 7322096
        - name: Website
          icon_color: 9367192
        - name: Research
          icon_color: 16766590
          skill: arxiv              # Auto-load a skill in this topic
```

**フィールド:**

| フィールド | 必須 | 説明 |
|-------|----------|-------------|
| `name` | はい | トピックの表示名 |
| `icon_color` | いいえ | Telegramのアイコンカラーコード（整数） |
| `icon_custom_emoji_id` | いいえ | トピックアイコン用のカスタム絵文字ID |
| `skill` | いいえ | このトピックで新規セッション時に自動ロードするスキル |
| `thread_id` | いいえ | トピック作成後に自動入力される — 手動で設定しないでください |

### 仕組み

1. ゲートウェイ起動時に、Hermesはまだ `thread_id` を持たない各トピックに対して `createForumTopic` を呼び出します
2. `thread_id` は自動的に `config.yaml` に書き戻されます。再起動、再接続、ネットワーク障害後のアダプターの再構築では、トピックを再作成する代わりにそれが再利用されます
3. 各トピックは独立したセッションキーにマッピングされます: `agent:main:telegram:dm:{chat_id}:{thread_id}`
4. 各トピックのメッセージは独自の会話履歴、メモリフラッシュ、コンテキストウィンドウを持ちます
5. `thread_id` を持つすべてのトピック（`dm_topics` と `group_topics` 内）は `channel_directory.json` に一覧表示されるため、最近のセッションがない場合でも `send_message` で名前を指定して送信できます。複数のチャットで使われているトピック名は `<chat_id> / <name>` として一覧表示されます。別のディレクトリエントリ（同じ名前のDMやグループ）と同じ名前のトピックは `<chat_id>:<thread_id> / <name>` として、そのエントリは `<its id> / <name>` として一覧表示されるため、名前だけではどちらにも解決されません

:::caution 既読状態の制限
TelegramのプライベートDMトピックには、Bot APIの既読状態に関する制限があります。Hermesはユーザーが作成したトピックからメッセージを受信し、そのトピックに返信をルーティングできますが、通常のボットアカウントには、受信したユーザーメッセージのトピック側のコピーを既読にするための信頼できるAPIがありません。Telegramでは、ルートの「All Messages」チャットとトピックで、未読/既読の表示が異なることがあります。これはTelegramの想定どおりの動作であり、必ずしもHermesの処理の失敗ではありません。
:::

### ルートDMの扱い

デフォルトでは、ルートDM（どのトピックにも属さない場所）に送られたメッセージは通常どおり処理されます。`ignore_root_dm: true` を設定するとルートDMがロビーになります — DMトピックが設定されているユーザーについては通常のメッセージが黙って無視されますが、システムコマンド（`/start`、`/help`、`/status` など）は引き続き動作します。

```yaml
platforms:
  telegram:
    extra:
      ignore_root_dm: true
      dm_topics:
        - chat_id: 123456789
          topics:
            - name: General
```

このチェックは **チャット単位** です: `dm_topics` に少なくとも1つのエントリを持つユーザーのみ、ルートDMが影響を受けます。トピックが設定されていないユーザーは影響を受けません。

### スキルのバインディング

`skill` フィールドを持つトピックは、そのトピックで新規セッションが始まると自動的にそのスキルをロードします。これは会話の最初に `/skill-name` と入力するのとまったく同じように動作します — スキルの内容が最初のメッセージに挿入され、以降のメッセージは会話履歴の中でそれを参照します。

たとえば、`skill: arxiv` を持つトピックは、（明示的な `/new` または `/reset` により）セッションがリセットされるたびにarxivスキルが事前ロードされます。

:::tip
設定の外で作成されたトピック（例: TelegramのAPIを手動で呼び出した場合）は、`forum_topic_created` のサービスメッセージが届くと自動的に検出されます。ゲートウェイの実行中に設定へトピックを追加することもできます — 次のキャッシュミス時に取り込まれます。
:::

## マルチセッションDMモード（`/topic`） {#multi-session-dm-mode-topic}

ChatGPTスタイルのマルチセッションDM — 1つのボットで多数の並列会話を行います。上記のオペレーターが用意する `extra.dm_topics` とは異なり、このモードは **ユーザー主導** です: 設定不要で、事前にトピック名を宣言する必要もありません。エンドユーザーが `/topic` でオンにし、Telegramの **+** ボタンをタップして好きなだけトピックを作成できます。各トピックは完全に独立したHermesセッションになります。

### `/topic` サブコマンド

| 形式 | コンテキスト | 効果 |
|------|---------|--------|
| `/topic` | ルートDM、未有効化 | BotFatherの機能を確認し、マルチセッションモードを有効化 |
| `/topic` | ルートDM、有効化済み | ステータスを表示: 復元可能なリンクされていないセッション |
| `/topic` | トピック内 | 現在のトピックのセッションバインディングを表示 |
| `/topic help` | 任意 | インラインの使い方 |
| `/topic off` | ルートDM | マルチセッションモードを無効化し、このチャットのすべてのトピックバインディングをクリア |
| `/topic <session-id>` | トピック内 | 以前のTelegramセッションを現在のトピックに復元 |

`/topic` は認可されたユーザー（`TELEGRAM_ALLOWED_USERS` / プラットフォーム認証設定による許可リスト）のみが実行できます。認可されていない送信者は、有効化ではなく拒否を受け取ります。

### DMトピック vs マルチセッションDMモード

| | `extra.dm_topics`（設定主導） | `/topic`（ユーザー主導） |
|---|---|---|
| 誰が有効化するか | オペレーターが `config.yaml` で | エンドユーザーが `/topic` を送って |
| トピックリスト | 設定で宣言された固定セット | ユーザーが自由にトピックを作成/削除 |
| トピック名 | オペレーターが選択 | ユーザーが選択。Hermesのセッションタイトルに合わせて自動リネーム |
| ルートDMの動作 | 通常のチャット（`ignore_root_dm: true` の場合はロビー） | システムロビーになる（コマンド以外のメッセージは拒否される） |
| 主なユースケース | 任意のスキルバインディングを伴う恒久的なワークスペース | アドホックな並列セッション |
| 永続化 | 設定内の `extra.dm_topics` | `telegram_dm_topic_mode` + `telegram_dm_topic_bindings` のSQLiteテーブル |

両方の機能は同じボット上で共存できます — ユーザーのDMから `/topic` を実行しつつ、`extra.dm_topics` は他のチャット向けにオペレーターが宣言したトピックを引き続き管理します。

### 前提条件 {#prerequisites}

**@BotFather** で、ボットを開き **Bot Settings → Threads Settings** に進みます:

1. **Threaded Mode** をオンにします（`has_topics_enabled` を有効化）
2. ユーザーによるトピック作成を無効に **しない** でください（`allows_users_to_create_topics` をオンに保つ）

ユーザーが初めて `/topic` を実行すると、Hermesは `getMe` を呼び出して両方のフラグを検証します。どちらかがオフの場合、HermesはBotFatherのThreads Settingsページのスクリーンショットを送り、何を切り替えるべきかを説明します — 前提条件が満たされるまで有効化は行われません。

### 有効化の流れ

ルートDMから次を送信します:

```
/topic
```

Hermesは以下を行います:

1. `getMe().has_topics_enabled` と `allows_users_to_create_topics` を確認
2. 両方がtrueなら、このDMでマルチセッショントピックモードを有効化
3. ユーザーが復元できる、以前のリンクされていないTelegramセッションのリストを返信

有効化後、**ルートDMはロビー** になります: 通常のプロンプトは拒否され、**All Messages** を指すガイダンスが表示されます。システムコマンド（`/status`、`/sessions`、`/usage`、`/help` など）はルートでも引き続き動作します。

### 新しいトピックの作成（エンドユーザーの流れ）

1. TelegramでボットのDMを開きます
2. ボットインターフェースの上部にある **All Messages** をタップし、任意のメッセージを送信します
3. Telegramがそのメッセージ用の新しいトピックを作成します
4. Hermesがそのトピック内で応答します — そのトピックが独立したセッションになります

各トピックは独自の会話履歴、モデル状態、ツール実行、セッションIDを持ちます。分離キーは `agent:main:telegram:dm:{chat_id}:{thread_id}` です — 設定主導のDMトピックの分離と同一です。

### ルートDMで送信されたメッセージ

一部のTelegramクライアントは、ルートDM（General / **All Messages**）で入力されたメッセージを `message_thread_id` なしで配信します。デフォルトでは、Hermesはそのようなメッセージを続きとして扱い、**最も最近アクティブだったトピック** で応答します（ロビーからの回復）。

代わりにルートDMのメッセージごとに新しいセッションを得るには、`dm_topic_auto_transfer` をオンにします:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        dm_topic_auto_transfer: true   # default: false
```

これをオンにすると、ルートDMでの通常のメッセージによってHermesが新しいトピック（自動タイトルによってリネームされるまでは「Hermes Chat」という名前）を作成し、そこで新しいセッションを開始し、そのトピック内に返信を投稿します。ルートDMで入力されたスラッシュコマンドはロビーのコマンドのままで、トピックを作成することはありません。トピックの作成に失敗した場合（例: BotFatherでThreaded Modeがオフにされた）、Hermesはデフォルトの動作にフォールバックし、最も最近アクティブだったトピックで応答します。このキーは `/topic` モードがオンの間のみ意味を持ちます。トップレベルの `telegram.extra.dm_topic_auto_transfer` キー（Desktopの設定に表示されるもの）も機能します。

### 自動リネームされるトピック

Hermesがトピックのセッションタイトルを（最初のやり取りの後、自動タイトルパイプラインで）生成すると、Telegramのトピック自体もそれに合わせてリネームされます — 例: 「New Topic」が「Database migration plan」になります。リネームはベストエフォートです: 失敗はログに記録されますが、セッションを壊すことはありません。

これを無効にして、手動で選んだトピック名をそのまま保つには、次を設定します:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_topic_auto_rename: true
```

このフラグがオンの場合でも、Hermesは内部のセッションタイトル（`hermes sessions` やTUIなどで使用）を生成しますが、Telegramのトピック名を編集することはありません。BotFatherのThreaded Modeでトピックを手作業で整理していて、最初の返信のたびにタイトルが上書きされるのを避けたい場合に便利です。

### トピック内での `/new`

他のトピックに触れることなく、現在のトピックのセッションをリセットします（新しいセッションID、新しい履歴）。Hermesは、並列作業のためには（**All Messages** から）別のトピックを作成するのが通常望ましい、というリマインダーを返信します。

### 以前のセッションの復元

トピック内で次を送信します:

```
/topic <session-id>
```

これにより、新規に開始する代わりに、現在のトピックを既存のHermesセッションにバインドします。トピックモードが有効になる前に始まった会話を続ける場合に便利です。制約:

- 対象のセッションは同じTelegramユーザーに属している必要があります
- 対象のセッションは、すでに別のトピックにバインドされていてはいけません

Hermesはセッションタイトルで確認し、コンテキストのために最後のアシスタントメッセージを再表示します。

セッションIDを見つけるには、ルートDMで `/topic`（引数なし）を送信します — HermesはユーザーのリンクされていないTelegramセッションを一覧表示します。

### トピック内での `/topic`（引数なし）

現在のトピックのバインディングを表示します: セッションタイトル、セッションID、そして `/new` か別のトピックの作成かについてのヒント。

### 内部の仕組み

- 有効化は `state.db` の `telegram_dm_topic_mode(profile_name, chat_id, user_id, enabled, ...)` に永続化されます。主キーは `(profile_name, chat_id)` であるため、1つの `state.db` を共有する多重化された/プロファイルでルーティングされるボットが、同じTelegramユーザーが複数のボットにDMを送ったときに互いを上書きすることはありません（プライベートの `chat_id` はユーザーIDであり、ボット間で同一です）。
- 各トピックバインディングは、PK `(profile_name, chat_id, thread_id)` と `session_id` に対する `ON DELETE CASCADE` 付きで `telegram_dm_topic_bindings(profile_name, chat_id, thread_id, session_id, ...)` に永続化されます — セッションをプルーニングすると、そのトピックバインディングも自動的にクリアされます
- トピックモードのSQLiteマイグレーションは **オプトイン** です: 最初の `/topic` 呼び出し時に実行され、ゲートウェイ起動時には決して実行されません。このプロファイルでユーザーが `/topic` を実行するまで、`state.db` は変更されません。スキーマv3で `profile_name` が追加されました。従来の行は `default` 名前空間にのみ移行されます
- 受信した各DMメッセージは、**ルーティングされた** プロファイル（プロセス全体のアクティブプロファイルではなく `source.profile`）を使って、その `(profile_name, chat_id, thread_id)` バインディングを参照します。存在する場合、参照は `SessionStore.switch_session()` を介してメッセージをバインドされたセッションへルーティングするため、セッションキーからセッションIDへのマッピングがディスク上で一貫した状態に保たれます
- トピック内での `/new` はバインディング行を新しいセッションIDを指すように書き換えるため、次のメッセージは新しいセッションに留まります
- `extra.dm_topics` で宣言されたトピックは **決して自動リネームされません** — マルチセッションモードが有効でも、オペレーターが選んだ名前が保持されます
- `extra.disable_topic_auto_rename: true` を設定すると、チャット内の **すべての** トピック（Threaded Modeで作成されたアドホックなトピックを含む）の自動リネームがオフになります
- フォーラム有効化されたDMのGeneral（最上部にピン留め）トピックは、Telegramがそのメッセージを `message_thread_id=1` で配信するか、thread_idなしで配信するかに関わらず、ルートロビーとして扱われます
- ルートロビーのメッセージは、ユーザーが最も最近バインドしたトピックにルーティングされます。`extra.dm_topic_auto_transfer: true` の場合は、代わりに `createForumTopic` で新しいトピックを開きます（その呼び出しが失敗した場合は最も最近のトピックにフォールバックします）
- ルートロビーのリマインダーは、**（プロファイル、チャット）** ごとに30秒に1メッセージへレート制限されています — トピックモードがオンであることを忘れてルートで10個のプロンプトを入力したユーザーが10件の返信を受け取ることはなく、チャットIDを共有する2つの多重化されたプロファイルが互いのリマインダーを抑制することもありません
- BotFatherのセットアップ用スクリーンショットは、**（プロファイル、チャット）** ごとに5分に1回の送信へレート制限されています — Threads Settingsがまだ無効な状態で `/topic` を繰り返し試しても、同じ画像が再アップロードされることはありません
- トピック内で開始された `/bg <prompt>` は、その結果を同じトピックに配信します。バックグラウンドセッションは、所属するトピックの自動リネームをトリガーしません
- `/topic` 自体はボットのユーザー認可チェックでゲートされています — 認可されていないDMは、有効化ではなく拒否を受け取ります
- Telegramが `message thread not found` でトピックを拒否した場合（トピックが削除された、またはTelegramクライアントでトピックがオフにされた）、Hermesは同じトピックに一度だけ再試行し、その後、返信のアンカーなしでルートチャットに返信を配信し、そのトピックのバインディングを破棄します。アンカーとなるメッセージが削除された返信は、トピックを保ったままアンカーだけを失います。Hermesが名前を指定して作成したばかりのトピックへの送信は代わりに失敗するため、返信がルートに届くのではなくトピックが再作成されます。チャットの最後のバインディングを削除するとトピックモードがオフになり、ルートDMは再び通常のチャットとして機能します

### マルチセッションモードの無効化

ルートDMで `/topic off` を送信します。Hermesは **このプロファイルの** 名前空間の行をオフにし、そのチャットに対するそのプロファイルの `(thread_id → session_id)` バインディングをクリアし、ルートDMは通常のHermesチャットに戻ります。Telegram内の既存のトピックは削除されません — 単に独立したセッションとしてゲートされなくなるだけです。後で再び `/topic` を実行すればオンに戻せます。

手動でクリーンアップする必要がある場合（例: 多数のチャットにまたがる一括リセット）は、`profile_name` で行を絞り込みます（単一プロファイルのインストールでは `default` を使用します）:

```bash
sqlite3 ~/.hermes/state.db \
  "UPDATE telegram_dm_topic_mode SET enabled = 0
     WHERE profile_name = 'default' AND chat_id = '<your_chat_id>';
   DELETE FROM telegram_dm_topic_bindings
     WHERE profile_name = 'default' AND chat_id = '<your_chat_id>';"
```

### Hermesのダウングレード

`/topic` より前のバージョンのHermesにダウングレードすると、この機能は単に動作しなくなります — `telegram_dm_topic_mode` と `telegram_dm_topic_bindings` のテーブルは `state.db` に残りますが、古いコードでは無視されます。DMはネイティブのスレッド単位の分離に戻ります（各 `message_thread_id` は引き続き `build_session_key` を介して独自のセッションを取得します）。そのため、既存のTelegramトピックは並列セッションとして引き続き動作します。ルートDMはもはやロビーではなくなり、そこのメッセージは以前と同じようにエージェントに渡されます。再アップグレードすると、マルチセッションモードは中断したところからそのまま再有効化されます。

## グループフォーラムトピックのスキルバインディング

**トピックモード** が有効なスーパーグループ（「フォーラムトピック」とも呼ばれます）は、すでにトピックごとのセッション分離を備えています — 各 `thread_id` が独自の会話にマッピングされます。しかし、DMトピックのスキルバインディングと同じように、特定のグループトピックにメッセージが届いたときに **スキルを自動ロード** したい場合もあるでしょう。

### ユースケース

異なるワークストリーム向けのフォーラムトピックを持つチームのスーパーグループ:

- **Engineering** トピック → `software-development` スキルを自動ロード
- **Research** トピック → `arxiv` スキルを自動ロード
- **General** トピック → スキルなし、汎用アシスタント

### 設定

`~/.hermes/config.yaml` の `platforms.telegram.extra.group_topics` の下にトピックバインディングを追加します:

```yaml
platforms:
  telegram:
    extra:
      group_topics:
      - chat_id: -1001234567890       # Supergroup ID
        topics:
        - name: Engineering
          thread_id: 5
          skill: software-development
        - name: Research
          thread_id: 12
          skill: arxiv
        - name: General
          thread_id: 1
          # No skill — general purpose
```

**フィールド:**

| フィールド | 必須 | 説明 |
|-------|----------|-------------|
| `chat_id` | はい | スーパーグループの数値ID（`-100` で始まる負の数値） |
| `name` | いいえ | トピックの人間向けラベル（情報目的のみ） |
| `thread_id` | はい | TelegramフォーラムトピックID — `t.me/c/<group_id>/<thread_id>` のリンクで確認できます |
| `skill` | いいえ | このトピックで新規セッション時に自動ロードするスキル |

### 仕組み

1. マッピングされたグループトピックにメッセージが届くと、Hermesは `group_topics` 設定内で `chat_id` と `thread_id` を参照します
2. 一致するエントリに `skill` フィールドがあれば、そのスキルがセッションに自動ロードされます — DMトピックのスキルバインディングと同一です
3. `skill` キーを持たないトピックはセッション分離のみを取得します（既存の動作、変更なし）
4. マッピングされていない `thread_id` 値や `chat_id` 値は黙ってスルーされます — エラーもスキルロードもありません

### DMトピックとの違い

| | DMトピック | グループトピック |
|---|---|---|
| 設定キー | `extra.dm_topics` | `extra.group_topics` |
| トピックの作成 | `thread_id` が欠けている場合、HermesがAPIでトピックを作成 | 管理者がTelegram UIでトピックを作成 |
| `thread_id` | 作成後に自動入力 | 手動で設定が必要 |
| `icon_color` / `icon_custom_emoji_id` | サポート | 該当なし（外観は管理者が制御） |
| スキルバインディング | ✓ | ✓ |
| セッション分離 | ✓ | ✓（フォーラムトピックには既に組み込み済み） |

:::tip
トピックの `thread_id` を見つけるには、Telegram WebまたはDesktopでトピックを開き、URLを確認します: `https://t.me/c/1234567890/5` — 最後の数字（`5`）が `thread_id` です。スーパーグループの `chat_id` は、グループIDの先頭に `-100` を付けたものです（例: グループ `1234567890` は `-1001234567890` になります）。
:::

## 最近のBot APIの機能

- **Bot API 9.4（2026年2月）:** プライベートチャットトピック — ボットは `createForumTopic` を介して1対1のDMチャット内にフォーラムトピックを作成できます。Hermesはこれを2つの異なる機能に使用します: オペレーターが用意する [プライベートチャットトピック](#private-chat-topics-bot-api-94)（設定主導、固定のトピックリスト）と、ユーザー主導の [マルチセッションDMモード](#multi-session-dm-mode-topic)（`/topic` で有効化、ユーザーが無制限にトピックを作成）。
- **プライバシーポリシー:** Telegramは現在、ボットにプライバシーポリシーを要求しています。BotFatherの `/setprivacy_policy` で設定するか、設定しない場合はTelegramがプレースホルダーを自動生成することがあります。ボットが一般公開されている場合は特に重要です。
- **Bot API 9.5（2026年3月）: `sendMessageDraft` によるネイティブストリーミング。** Hermesは、デフォルトの `auto` トランスポートのもとで、プライベートチャットにTelegramのネイティブなストリーミング下書きAPIを使用します（グループとグループのフォーラムトピックは常に `editMessageText` の経路を使用します。プライベートチャットのトピックも下書きを試し、Telegramに拒否された場合は編集にフォールバックします）。Telegramクライアント上で下書きプレビューが目に見えて崩れて再描画される場合は、`transport: edit` を設定してください（下記参照）。

### ストリーミングトランスポート（`gateway.streaming.transport`）

ストリーミングが有効な場合（`gateway.streaming.enabled: true`）、Hermesは4つのトランスポートのいずれかを選択します:

| 値 | 動作 |
|---|---|
| `auto`（デフォルト） | サポートされているチャット（現在はTelegram DM）ではネイティブの下書きストリーミング、それ以外では従来の編集ベースの経路。下書きフレームが失敗した場合は適切にフォールバックします。 |
| `draft` | ネイティブの下書きを強制します。チャットが下書きをサポートしていない場合（例: グループやグループのフォーラムトピック）、ダウングレードをログに記録して編集にフォールバックします。 |
| `edit` | すべてのチャットタイプで従来の漸進的な `editMessageText` ポーリング。 |
| `off` | ストリーミングを完全に無効化します（最終的な返信のみ、漸進的な更新なし）。 |

`~/.hermes/config.yaml` で:

```yaml
gateway:
  streaming:
    enabled: true
    transport: auto    # auto | draft | edit | off
```

**`auto`（デフォルト）または `draft` でDMに表示されるもの** — Telegramはトークンごとに更新されるアニメーション下書きプレビューを表示します。返信が完了すると、通常のメッセージとして配信され、下書きプレビューはクライアント側で自然に消えます。下書きにはメッセージIDがないため、最終的な回答がチャット履歴に残るものになります。

**`edit` でDMに表示されるもの** — ゲートウェイは通常のプレビューメッセージを送信し、`editMessageText` で漸進的に更新します。クライアントで、下書きプレビューが伸びていく間に崩れたりちらついたりする場合は、これを選択してください。

**グループ、スーパーグループ、グループのフォーラムトピックはどうなりますか?** Telegramは `sendMessageDraft` をプライベートチャット（DM）に制限しています。ゲートウェイはそれらに対して透過的に編集ベースの経路にフォールバックします — 以前と同じUXです。プライベートチャット内のトピックはDMとして扱われます: 下書きを試し、Telegramに拒否された場合は編集にフォールバックします。

**プレビューはどのくらいの頻度で更新されますか?** 最大で `streaming.edit_interval` ごとに1回です: その間に届いた差分はまとめられるため、ほぼ同一のフレームが連続して送られるのではなく、次のフレームに最新のテキストが載ります。Telegramは下書きフレームと入力中インジケーターをチャットごとに合わせて計測しており（5秒あたり20回、30秒あたり40回。[Telegramのストリーミングに関する注記](https://core.telegram.org/api/bots/ai)を参照）、Hermesは両方をその予算内に収めるため、`edit_interval` を下げてもTelegramが許可する以上の更新を送ることはできません。

**下書きフレームが失敗したらどうなりますか?** Telegramの `RetryAfter`（フラッド制御）は、待機時間が終わるまでプレビューの更新を一時停止し、その後最新のテキストで再開します。その応答はネイティブの下書きのままです。それ以外のあらゆる失敗（一時的なネットワークエラー、サーバー側の拒否、古いpython-telegram-botのインストール）により、その応答は残りのストリームについて編集ベースの経路に切り替わります。次の応答では改めて試行されます。完全な回答は、常に通常の最終送信によって配信されます。

**その場で伸びていくプレビューテキスト（`plain_drafts`）。** デフォルトでは、下書きプレビューは最終的な返信と同じMarkdownV2の書式でレンダリングされます。そのため、コードブロックや太字の範囲が閉じたときに、すでに画面上にあるテキストが再レンダリングされることがあります。`telegram.extra.plain_drafts: true`（ネストした形式では `gateway.platforms.telegram.extra.plain_drafts`）を設定すると、代わりにプレビューを生のテキストとしてストリーミングします: 各フレームは前のフレームに文字を追加するだけになるため、Telegramは新しいテキストだけをアニメーション表示します。最終的な返信はどちらの場合も通常の書式を保ち、プレビューについては `plain_drafts` が `rich_drafts` より優先されます。

```yaml
gateway:
  platforms:
    telegram:
      extra:
        plain_drafts: true   # default false
```

## レンダリング: リッチメッセージ、テーブル、リンクプレビュー

**リッチメッセージ（Bot API 10.1）。** 従来のMarkdownV2の経路では劣化してしまう構成要素 — テーブル、タスクリスト、折りたたみ可能な `<details>`、ブロック数式 — を含む最終的な返信は、エージェントの **生のmarkdown** を使ってTelegramネイティブの [`sendRichMessage`](https://core.telegram.org/bots/api#sendrichmessage) で送信されるため、クライアント側でフラット化されることなくネイティブにレンダリングされます。DMでは、デフォルトの `rich_drafts: false` によりストリーミングプレビューはプレーンのままになります — 従来のレンダリングでTelegramの一時的な下書きトランスポートを使用し（テーブルなどのリッチ専用の構成要素はプレビューでは生のmarkdownのまま）、その後、完成した応答を `sendRichMessage` で永続化します。`rich_drafts: true` を設定すると、ライブプレビューでも `sendRichMessageDraft` が使われます。編集ベースのストリームでは、`editMessageText` の `rich_message` パラメータを通じて既存のプレビューをその場で確定できます。通常の返信（平文、太字/斜体、単純なリスト）は、クライアント間でフォントの太さや間隔を一貫させるため、MarkdownV2の経路のままです。

リッチテキストの上限である32,768文字を超えるリッチ対象のメッセージは、ブロック間で分割された複数のリッチメッセージとして送信されます（`<details>` セクション、テーブル、コードブロックは、それ単体で上限を超えない限り切られることはありません。超える場合、テーブルはヘッダーを繰り返し、コードブロックは次のパートで再び開かれます）。Telegramからのあらゆる拒否（古い `python-telegram-bot` でサポートされていないエンドポイント、パーサーエラー、大きすぎるブロック/列）は、**透過的に** MarkdownV2の経路に **フォールバック** します — メッセージが失われることはありません。一時的なエラーやネットワークエラーは黙って再送信 *されません*（最終メッセージが重複することはありません）。

**長い返信はファイルとして届きます。** Telegramのメッセージが8件を超えて必要になる返信（平文で約32,000文字）は、Telegramのフラッド制限に引っかかりかねない大量のメッセージの代わりに、完全なテキストを含む1つの `reply-<id>.md` ドキュメントとして、短いキャプション付きで送信されます。長い返信をストリーミングしている間、Hermesは上限に達すると新しいメッセージを開くのをやめ、完成した返信をファイルとしてアップロードし、その返信のためにすでに投稿していた部分的なメッセージを削除します。アップロードが失敗した場合は何も削除されず、返信の残りは通常のメッセージとして届くため、テキストが失われることはありません。Telegramがボットに削除を許可しないメッセージ（48時間より古い、または権限がない）はそのまま残ります。`hermes send` と `send_message` ツールも、長すぎるメッセージに対して同じことを行います。Telegramの50 MBのドキュメント上限を超える返信は、番号付きの複数のファイルに分割されます。

**MarkdownV2へのフォールバック。** あるメッセージでリッチの経路が利用できない場合、HermesはmarkdownをMarkdownV2に変換します。MarkdownV2にはネイティブのテーブル構文がないため、パイプテーブルは正規化されます:

- **小さなテーブル** は **行グループの箇条書き** にフラット化されます — 各行が列見出しの下の読みやすい箇条書きリストになります。2〜4列で短いセルに適しています。
- **より大きい、または幅の広いテーブル** は、列が揃った **フェンス付きコードブロック** にフォールバックし、何も崩れないようにします。

文章中のLaTeXの記号マクロ（`\rightarrow`、`\leq`、`\alpha` など）は、それが表す記号（→、≤、α）として表示され、インラインの `$x \leq y$` は、MarkdownV2では数式を組版できないため、プレーンテキストになった時点でドル記号が取り除かれます。コードスパン、コードブロック、ファイルパス、未知のマクロはバックスラッシュを保持します。リッチの経路では、同じマッピングは数式の範囲の外でのみ適用されます。数式の範囲はTelegram自身が組版します。

リッチメッセージは **オプトイン** です。現在のTelegramクライアントでは、Bot APIのリッチメッセージをプレーンテキストとしてコピーするのが難しい場合があり、特にコマンドのスニペットやモバイルへの受け渡しで不便なため、デフォルトは従来のMarkdownV2の経路のままです。テーブル/タスクリスト/details/数式のネイティブレンダリングを有効にするには:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        rich_messages: true
        rich_drafts: false
        allow_cjk_rich_messages: false
```

この設定はクライアントのレンダリング/コピーの互換性のためのものです。TelegramがリッチAPIの呼び出しを拒否した場合、Hermesはすでに自動的にフォールバックします。`rich_drafts` はDMのストリーミングプレビューをリッチで *レンダリング* するか（`sendRichMessageDraft`）を制御し、Telegram Desktop/macOSではチャットが再描画されるまでリッチな下書きフレームが視覚的に重なって表示されることがあるため、デフォルトではオフです。オフの場合、プレビューはプレーンでストリーミングされ、最終的なものは引き続きネイティブのリッチメッセージとして届きます。

同じ設定は、ゲートウェイのプロセス外で行われる送信にも適用されます: `hermes send`、`send_message` ツール、スタンドアロンの送信機能にフォールバックするcron配信は、同じルールのもとでリッチの経路を使用します。これらは長いメッセージの `(1/2)` のチャンクマーカーもエスケープし、Telegram自身のHTMLタグ（`<b>`、`<a>`、`<code>` など）のみをHTMLとして扱います。そのため、`<details>` ブロックや山括弧のプレースホルダーはMarkdownの書式を保ちます。Bot APIがHTMLとして依然として拒否するメッセージは、プレーンテキストにフォールバックする前にMarkdownV2として再試行されます。まとめて送信された2枚以上の写真は、ネイティブのアルバム（`sendMediaGroup`、1アルバムあたり10枚）として届きます。

CJKテキスト（中国語、日本語、韓国語、およびまれな漢字の拡張）は、影響を受けるTelegram Desktop/macOSクライアントでBot APIのリッチメッセージがCJKのグリフが重なるアーティファクトとともにレンダリングされたことがあるため、デフォルトでは従来のMarkdownV2の経路のままです。影響を受けないクライアントを使っていて、CJKのペイロードでもネイティブのリッチなテーブル/タスクリスト/details/数式を使いたい場合は、`rich_messages: true` とともに `allow_cjk_rich_messages: true` を設定して、そのクライアント側のリスクを受け入れてオプトインしてください。

**リンクプレビュー。** Telegramはボットメッセージ内のURLに対してリンクプレビューを自動生成します。それらを抑制したい場合（長い `/tools` の出力、10個のリンクに言及するエージェントの返信など）:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

有効にすると、Hermesは送信する各メッセージ（ストリーミングプレビューの編集やストリーミングされた返信の最終編集を含む。ただし承認カードのようなボタンのプロンプトを解決する編集にはまだ付与されません）にTelegramの `LinkPreviewOptions(is_disabled=True)` を付与し、古いバージョンの `python-telegram-bot` では従来の `disable_web_page_preview` パラメータにフォールバックします。

**長い返信とフラッド制御。** Telegramの4,096文字の上限より長い返信は、番号付きのパート（`(1/3)`、`(2/3)`、…）として送信されます。太字、斜体、下線、取り消し線、スポイラーのテキストの途中にパートの境界がある場合、あるパートの最後でスタイルを閉じ、次のパートの最初で再び開くため、両方のパートが書式を保ちます。1つのチャットへの送信は1返信ずつ配信されるため、スケジュールされたレポートとDMの回答が同時に届いてもパートが交互に混ざることはなく、ファイルのアップロードが、それに付随するテキストの2つのパートの間に入り込むこともありません。Telegramのフラッド制御が途中でパートを拒否した場合、Hermesはペナルティが過ぎた後、すでに画面に表示されているパートを再送信するのではなく、拒否されたパートから再開します。また、チャットが既知のペナルティ期間内にある間は、そのチャットへのさらなる送信はローカルでフェイルクローズします（ペナルティを長引かせる余分なリクエストは送りません）。ゲートウェイのインライン待機の上限より長いペナルティは配信台帳に引き渡され、「part of it may already have arrived above」という注記付きで返信が再配信されます。ファイルの添付（ドキュメント、写真、動画、ボイス）も同じルールに従います: 短いペナルティは待ってからアップロードを再試行し、長いペナルティの場合は期間が過ぎた後にネイティブの再アップロードを1回スケジュールします。「Couldn't deliver the file attachment」の通知が表示されるのは、その再アップロードも拒否された場合のみです。Telegramに届かなかったアップロード（接続またはプールの失敗）や、Telegramが自身の5xxエラーで応答したアップロードは、短いバックオフで最大3回再試行されます。3回とも失敗すると、「Couldn't deliver the file attachment」の通知が届きます。すでにチャットに届いている可能性があるアップロード（ファイルの送信後に接続が切れた、またはタイムアウトした）は決して再送信されず、失敗の通知も投稿されません。未確認として報告されるため、`send_message` ツールとcron配信もそれを再送信しません。

## グループの許可リスト

Telegramのグループとフォーラムチャットには、設定できる2つの直交するゲートがあります:

- **送信者ユーザーID**（`group_allow_from` / `TELEGRAM_GROUP_ALLOWED_USERS`） — グループ/フォーラムメッセージにのみ適用される送信者単位の許可リスト。特定のユーザーを `TELEGRAM_ALLOWED_USERS`（これはDMアクセスも付与してしまいます）に追加せずに、グループ内でボットを起動できるようにしたい場合に使用します。
- **チャットID**（`group_allowed_chats` / `TELEGRAM_GROUP_ALLOWED_CHATS`） — チャット単位の許可リスト。これらのグループ/フォーラムのメンバーは誰でもボットとやり取りできます。グループのメンバーシップ自体がアクセスのシグナルとなる、チーム/サポートボットに便利です。

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Global access (DMs + groups). Users here can always invoke the bot.
        allow_from:
          - "123456789"
        # Sender IDs allowed in groups/forums only. Does NOT grant DM access.
        group_allow_from:
          - "987654321"
        # Entire groups/forums — any member is authorized.
        group_allowed_chats:
          - "-1001234567890"
```

同等の環境変数:

```bash
TELEGRAM_ALLOWED_USERS="123456789"
TELEGRAM_GROUP_ALLOWED_USERS="987654321"
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

動作:

- `TELEGRAM_ALLOWED_USERS` はすべてのチャットタイプ（DM、グループ、フォーラム）を対象とします。
- `TELEGRAM_GROUP_ALLOWED_USERS` は、リストされた送信者をグループ/フォーラム内でのみ認可します。`TELEGRAM_ALLOWED_USERS` にリストされていない限り、彼らは依然としてボットにDMを送れません。
- `TELEGRAM_GROUP_ALLOWED_CHATS` に含まれるチャットは、送信者に関わらずそのチャットのすべてのメンバーを認可します。
- これらのいずれかで `*` を使うと、任意の送信者/チャットを許可します。
- `config.yaml` でのこれらのリストの編集は、ゲートウェイを再起動しなくても次のメッセージから適用されます: 削除したユーザーは、次のメッセージやボタンのタップから拒否されます。これは、多重化されたゲートウェイ（`gateway.multiplex_profiles`）が提供するすべてのプロファイル（起動時のホストプロファイルを含む）に当てはまります。`.env`、外部のシークレットソース、またはゲートウェイ自身の環境で設定された値は、引き続き `config.yaml` より優先されます。
- これは既存のメンション/パターントリガーの上、および `group_topics` + `ignored_threads` の上に重ねられます。

### PR #17686 以前からの移行

この分割以前は、`TELEGRAM_GROUP_ALLOWED_USERS` が唯一の設定項目であり、ユーザーはそこに **チャットID** を入れていました。後方互換性のため、`TELEGRAM_GROUP_ALLOWED_USERS` 内のチャットID形式の値（`-` で始まるもの）は引き続きチャットIDとして尊重され、非推奨警告が一度ログに記録されます。移行:

```bash
# Old (still works, but deprecated)
TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890"

# New
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

### ゲストの@メンションによるバイパス（`guest_mode`）

一般的なセットアップでは、`group_allowed_chats` は厳格なゲートです: リストにないグループからのメッセージは、メンバーが明示的にボットを@メンションした場合でも黙って破棄されます。これはサポート/チームボットにとって正しいデフォルトです。

よりカジュアルなセットアップ — ボットを **ほとんど沈黙させつつ**、**明示的に呼ばれたときだけ時々使える** ようにしたい友人とのグループチャットなど — では、`guest_mode` を有効にします:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        group_allowed_chats:
          - "-1001234567890"   # your main allowlisted group
        guest_mode: true       # non-allowlisted groups: allow on @mention only
```

環境変数での同等の設定:

```bash
TELEGRAM_GUEST_MODE=true
```

デフォルト: `false`。

`guest_mode: true` の場合、許可リストにないグループからのメッセージは、明示的にボットを@メンションした場合に **のみ** 処理されます。メンションは毎ターン必要です — ゲストとのやり取りにはセッションの持続性がないため、呼ばれていない友人グループのスレッドにボットが自動的に参加することはありません。

DMと許可リスト登録済みのグループは、以前とまったく同じように動作します。

ブロードキャスト **チャンネル** の投稿には、グループと同じ `allowed_chats`、`require_mention`、トピックのゲートが適用されますが、ゲストのバイパスは決して適用されません: そうでなければ、チャンネルの投稿権限を持つ誰もが許可リストを越えられてしまうためです。チャンネルのIDを `allowed_chats` に追加し、投稿内でボットを@メンションして呼びかけてください。

## スラッシュコマンドのアクセス制御 {#slash-command-access-control}

デフォルトでは、すべての許可されたユーザーがすべてのスラッシュコマンドを実行できます。許可リストを **管理者**（すべてのスラッシュコマンドへのアクセス）と **一般ユーザー**（明示的に有効にしたコマンドのみ）に分けるには、プラットフォームの `extra` ブロックに `allow_admin_from` と `user_allowed_commands` を追加します:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Existing allowlists (unchanged)
        allow_from:
          - "123456789"     # admin
          - "555555555"     # regular user
          - "777777777"     # regular user

        # NEW — admins get all slash commands (built-in + plugin)
        allow_admin_from:
          - "123456789"

        # NEW — non-admin allowed users can only run these slash commands.
        # /help and /whoami are always allowed so users can see their access.
        user_allowed_commands:
          - status
          - model
          - history

        # Optional: separate admin/command lists for groups
        group_allow_admin_from:
          - "123456789"
        group_user_allowed_commands:
          - status
```

**動作:**

- あるスコープ（DMまたはグループ）について `allow_admin_from` にリストされたユーザーは、ライブのレジストリを通じて **すべての** 登録済みスラッシュコマンド — 組み込みコマンドおよびプラグインが登録したもの — を実行できます。
- `allow_from` には含まれるが `allow_admin_from` には **含まれない** ユーザーは、`user_allowed_commands` にリストされたコマンドと、常に許可される最低限のもの（`/help` と `/whoami`）のみを実行できます。
- 通常のチャット（スラッシュ以外のメッセージ）は影響を受けません。管理者でないユーザーも引き続き通常どおりエージェントと会話できます。ただ任意のコマンドをトリガーできないだけです。
- **後方互換性:** あるスコープについて `allow_admin_from` が設定されていない場合、そのスコープではスラッシュコマンドのゲートが無効になります。既存のインストールは変更なしで動作し続けます。
- DMの管理者ステータスはグループの管理者ステータスを意味しません。各スコープは独自の管理者リストを持ちます。
- `group_allow_admin_from` のみが設定されている場合、DMスコープは無制限（後方互換）モードのままになります。

`/whoami` を使うと、アクティブなスコープ、自分の階層（管理者 / ユーザー / 無制限）、および実行できるスラッシュコマンドを確認できます。

## 対話型モデルピッカー

Telegramチャットで引数なしの `/model` を送信すると、Hermesはモデルを切り替えるための対話型インラインキーボードを表示します:

1. **プロバイダー選択** — 利用可能な各プロバイダーをモデル数とともに表示するボタン（例: 「OpenAI (15)」、現在のプロバイダーには「✓ Anthropic (12)」）。
2. **モデル選択** — **Prev**/**Next** ナビゲーション、プロバイダーに戻る **Back** ボタン、**Cancel** を備えたページ分割されたモデルリスト。

現在のモデルとプロバイダーが上部に表示されます。すべてのナビゲーションは同じメッセージをその場で編集して行われます（チャットが散らかりません）。

:::tip
正確なモデル名がわかっている場合は、`/model <name>` を直接入力してピッカーをスキップできます。`/model <name> --global` と入力すると、変更をセッションをまたいで永続化できます。
:::

## DNS-over-HTTPS フォールバックIP

一部の制限されたネットワークでは、`api.telegram.org` が到達不能なIPに解決されることがあります。Telegramアダプターには **フォールバックIP** の仕組みが含まれており、正しいTLSホスト名とSNIを保ちつつ、代替IPに対して接続を透過的に再試行します。

### 仕組み

1. `TELEGRAM_FALLBACK_IPS` が設定されている場合、それらのIPが直接使用されます。
2. そうでない場合、アダプターはDNS-over-HTTPS（DoH）を介して **Google DNS** と **Cloudflare DNS** に自動的に問い合わせ、`api.telegram.org` の代替IPを検出します。
3. 既知のIPv4のTelegram API IPは、デュアルスタックの `api.telegram.org` ホスト名 **より先に** 試されます。ブラックホール化されたIPv6の経路は、エラーにならないまま `connect()` で止まり続けることがあり、以前はそれがイベントループを固定してしまい、30秒の初期化期限が発火しませんでした。
4. ハードコードされたIPv4のシードリスト（`149.154.166.110`、`149.154.167.220`）は常にDoHの回答の後に続き（DoHは単一のエッジしか返さないことが多く、そうでなければそれが唯一のフォールバックになってしまうため）、DoHがブロックされているかタイムアウトした場合はこれがリスト全体になります。ホスト名は最後の手段として残ります。
5. ある経路が一度成功すると、それは「スティッキー」になります — 以降のリクエストは直接それを使用します。ホスト名は、IPv6のみのネットワーク向けの最後の手段として保持されます。
6. 1つのリクエストがこれらの経路をたどる際には、30秒の送信期限の半分である15秒の接続予算を共有します。最初の経路は完全な接続タイムアウトを保ち、後続の経路は残りを分け合います。すべての経路がハングした場合、送信は期限の前に接続タイムアウトで失敗します。メッセージは送信されていないため、ゲートウェイはそれを再試行します。

ゲートウェイのプロセス外で行われる送信（`hermes send`、`send_message` ツール、スタンドアロンのcron配信）は、設定されたIPまたはシードリストで同じトランスポートを使用します。単発の送信を高速に保つため、DoHによる検出はスキップします。また、ゲートウェイの `HERMES_TELEGRAM_HTTP_*` タイムアウトとメディアアップロードの予算も使用します。接続またはプールのタイムアウトは、リクエストが送信されていないため、そこで再試行されます。それ以外のタイムアウトは再試行されません: Telegramがすでにメッセージを配信している可能性があり、`hermes send` は再送信を促す代わりに警告でその旨を伝えます。

ディスク上のファイルからのメディアアップロード（ゲートウェイまたはスタンドアロン）には、60秒に、ファイルを128 KB/sで送るのにかかる時間を加えた読み取りおよび書き込みのタイムアウトが設定されます。ゲートウェイのアップロードには、さらに300秒にその時間の1.5倍を加えた実時間の上限（最大1時間）も設定されます。`hermes send` と `send_message` ツールは、これらのリクエストごとのタイムアウトによってのみ制限されます。スタンドアロンのcron配信は `cron.standalone_send_timeout_seconds` の上限（デフォルト60秒）を保ち、配信にファイルが含まれる場合はアップロード時間の1.5倍だけ広げられます。また、ライブのゲートウェイを通じたcron配信も同じように `cron.media_send_timeout_seconds` を広げます。遅いアップリンク上の大きなファイルや、ファイルを転送し終えてから応答するセルフホストのBot APIサーバーを経由する場合でも、数分後に届くファイルについて失敗を報告するのではなく、Telegramの応答を待つようになります。

### 設定

```bash
# Explicit fallback IPs (comma-separated)
TELEGRAM_FALLBACK_IPS=149.154.167.220,149.154.167.221
```

または `~/.hermes/config.yaml` で:

```yaml
platforms:
  telegram:
    extra:
      fallback_ips:
        - "149.154.167.220"
```

:::tip
通常、これを手動で設定する必要はありません。DoHによる自動検出が、ほとんどの制限されたネットワークのシナリオを処理します。`TELEGRAM_FALLBACK_IPS` 環境変数は、ネットワーク上でDoHもブロックされている場合にのみ必要です。ホスト上でIPv6が壊れている場合は、`config.yaml` で `network.force_ipv4: true` を設定して、プロセス全体でAAAAの名前解決をスキップすることもできます。
:::

## プロキシのサポート

ネットワークがインターネットに到達するためにHTTPプロキシを必要とする場合（企業環境では一般的）、Telegramアダプターは標準的なプロキシ環境変数を自動的に読み取り、すべての接続をプロキシ経由でルーティングします。

### サポートされている変数

アダプターは次の環境変数を順番にチェックし、最初に設定されているものを使用します:

1. `HTTPS_PROXY`
2. `HTTP_PROXY`
3. `ALL_PROXY`
4. `https_proxy` / `http_proxy` / `all_proxy`（小文字のバリアント）

### 設定

ゲートウェイを起動する前に、環境にプロキシを設定します:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
hermes gateway
```

または `~/.hermes/.env` に追加します:

```bash
HTTPS_PROXY=http://proxy.example.com:8080
```

プロキシは、主トランスポートとすべてのフォールバックIPトランスポートの両方に適用されます。追加のHermes設定は不要です — 環境変数が設定されていれば、自動的に使用されます。

:::note
これは、HermesがTelegram接続に使用するカスタムフォールバックトランスポート層を対象としています。他の場所で使われる標準の `httpx` クライアントは、すでにプロキシ環境変数をネイティブに尊重します。
:::

### OpenSSL 3.5でのTLSハンドシェイクの失敗

同じプロキシを通じてcurlは動作するのに、Telegramが空の `httpx.ConnectError` や `SSL: UNEXPECTED_EOF_WHILE_READING` で繰り返し再接続する場合、ネットワーク経路がOpenSSL 3.5のより大きなポスト量子TLSのClientHelloを拒否している可能性があります。この症状だけでは証拠になりません: 回避策を有効にする前に、デフォルトのハンドシェイクとX25519でのハンドシェイクを比較してください。

Telegramに対してのみ従来のX25519鍵交換を使用するには、`config.yaml` に次を追加してゲートウェイを再起動します:

```yaml
platforms:
  telegram:
    extra:
      tls_classic_key_exchange: true
```

これは、Bot APIのリクエストとロングポーリングの両方を対象とし、プロキシやフォールバックIPの接続、`https://` プロキシ自体とのTLSハンドシェイク、DNS-over-HTTPSによるフォールバックIPの検出、単発の送信（cron、`send_message`）も含みます。TLS 1.3、ホスト名のチェック、証明書の検証は有効なままで、プラットフォームのトラストストアと `SSL_CERT_FILE` も引き続き使用されます。トレードオフは、Telegramに対してポスト量子鍵交換をオプトアウトすることです。他のプラットフォームやモデルプロバイダーのクライアントは、既存のTLS設定を保ちます。デフォルトは `false` です。OpenSSLのデフォルトのグループに戻すには、設定を削除するか `false` に設定します。true/false以外の値は、ゲートウェイのログに警告を出して無視されます。[関連するOpenSSL 3.5の報告](https://github.com/NousResearch/hermes-agent/issues/106384)も参照してください。

## メッセージリアクション

ボットは、処理状況の視覚的なフィードバックとして、メッセージに絵文字リアクションを追加できます:

- 👀 ボットがメッセージの処理を開始したとき
- 👍 応答が正常に配信されたとき
- 👎 処理中にエラーが発生したとき

リアクションは **デフォルトで無効** です。`config.yaml` で有効にします:

```yaml
telegram:
  reactions: true
```

または環境変数で:

```bash
TELEGRAM_REACTIONS=true
```

:::note
Discord（リアクションが追加的）とは異なり、TelegramのBot APIは1回の呼び出しですべてのボットリアクションを置き換えます。👀 から 👍/👎 への遷移はアトミックに行われます — 両方が同時に表示されることはありません。
:::

:::tip
ボットにグループ内でリアクションを追加する権限がない場合、リアクション呼び出しは黙って失敗し、メッセージ処理は通常どおり続行されます。
:::

## チャンネルごとのプロンプト

特定のTelegramグループやフォーラムトピックに、一時的なシステムプロンプトを割り当てます。プロンプトは各ターンで実行時に挿入され、トランスクリプト履歴には決して永続化されません — そのため変更は即座に反映されます。

```yaml
telegram:
  channel_prompts:
    "-1001234567890": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "42":  |
      This topic is for creative writing feedback. Be warm and
      constructive.
```

キーはチャットID（グループ/スーパーグループ）またはフォーラムトピックIDです。フォーラムグループでは、トピックレベルのプロンプトがグループレベルのプロンプトを上書きします:

- グループ `-1001234567890` 内のトピック `42` のメッセージ → トピック `42` のプロンプトを使用
- トピック `99`（明示的なエントリなし）のメッセージ → グループ `-1001234567890` のプロンプトにフォールバック
- エントリのないグループのメッセージ → チャンネルプロンプトは適用されない

数値のYAMLキーは自動的に文字列に正規化されます。

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| ボットがまったく応答しない | `TELEGRAM_BOT_TOKEN` が正しいか確認してください。`hermes gateway` のログにエラーがないか確認してください。 |
| ボットが「unauthorized」と応答する | あなたのユーザーIDが `TELEGRAM_ALLOWED_USERS` にありません。@userinfobot で再確認してください。 |
| ボットがグループメッセージを無視する | おそらくプライバシーモードがオンです。無効にする（ステップ3）か、ボットをグループ管理者にしてください。**プライバシー変更後はボットを削除して再追加することを忘れないでください。** |
| ボイスメッセージが文字起こしされない | STTが利用可能か確認してください: ローカル文字起こしには `faster-whisper` をインストールするか、`~/.hermes/.env` に `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY` を設定してください。 |
| ボイス返信がバブルではなくファイルになる | `ffmpeg` をインストールしてください（Edge TTSのOpus変換に必要）。 |
| ログに `Telegram polling conflict`（409）が表示される | 別のポーラーがボットの `getUpdates` セッションを保持しています。これは通常、ゲートウェイ自身の以前のロングポールであり、Telegramが自動的に期限切れにします: アダプターは約200秒間再試行し、その後ゲートウェイが約2分後、5分後、10分後に再び再接続します（多重化されたゲートウェイ内のすべてのプロファイルがこれを行います）。それでもトークンが使用中の場合、`hermes status` はプラットフォームを `token_in_use_elsewhere` 付きの `fatal` かつ要対応として表示し、ゲートウェイは再試行を停止します。別のポーラー（別のマシン、残っているプロセス、またはこのトークンを使っている別のボットフレームワーク）を停止してから、`hermes gateway restart` を実行してください。同じトークンで設定された自分の2つのプロファイルがここまで進むことはありません: 2つ目は起動時に `duplicate_credential`（もう一方が別のローカルゲートウェイの場合は `telegram-bot-token_lock`）で拒否されます。 |
| ボットトークンが無効化された/無効 | BotFatherで `/revoke` の後に `/newbot` または `/token` で新しいトークンを生成してください。`.env` ファイルを更新してください。 |
| Webhookが更新を受信しない | `TELEGRAM_WEBHOOK_URL` が公開されており到達可能か確認してください（`curl` でテスト）。プラットフォーム/リバースプロキシが、URLのポートからの内向きhttpsトラフィックを、`TELEGRAM_WEBHOOK_PORT` で設定したローカルのリッスンポートにルーティングすることを確認してください（両者は同じ番号である必要はありません）。SSL/TLSが有効であることを確認してください — TelegramはhttpsのURLにしか送信しません。ファイアウォールのルールを確認してください。 |

## 実行の承認

エージェントが潜在的に危険なコマンドを実行しようとすると、チャットで承認を求めます:

> ⚠️ This command is potentially dangerous (recursive delete). Reply "yes" to approve.

承認するには「yes」/「y」、拒否するには「no」/「n」と返信します。

## 対話型プロンプト（clarify）

エージェントが `clarify` ツールを呼び出すと — どのアプローチを好むかを尋ねる、タスク後のフィードバックを得る、または重要な判断の前に確認するため — Telegramは質問を **インラインキーボードのボタン** 付きでレンダリングします:

> ❓ Which framework should I use for the dashboard?
>
> [1. Next.js] [2. Remix] [3. Astro]
> [✏️ Other (type answer)]

ボタンをタップして回答するか、**Other** をタップして自由形式の回答を入力します（次に送信するメッセージが回答になります）。自由回答形式の `clarify` 呼び出し（事前に用意された選択肢がないもの）はボタンを省略し、次のメッセージをそのまま受け取ります。

`clarify` がセッションのツールに含まれている場合（cronジョブや、それを無効にしているプロファイルを除く）、Telegramのプラットフォームヒントは、小さな固定選択式の判断にはそれを使うようエージェントに伝えます。そのため、エージェントが番号付きリストや、回答を送り返すことのないクリップボードへのコピーボタンを即興で作ることはありません。エージェントが作成する永続的なアクションボタンは、まだHermesの汎用機能ではありません。

応答のタイムアウトは `~/.hermes/config.yaml` の `agent.clarify_timeout` で設定します（デフォルト `90` 秒）。タイムアウト内に応答しない場合、エージェントはハングする代わりに `"outcome": "timed_out"` でブロックを解除し、状況に適応します。質問をスキップするには `skip` と返信します。

Telegramがボタンカードをレンダリングできない場合（Bot APIが拒否した、または15秒の確認応答ウィンドウの後に送信が失敗した）、Hermesは同じ質問をプレーンな番号付きリストのメッセージとして尋ね直し、入力された返信（番号または選択肢のテキスト）が回答として受け取られます。それすら配信できない場合、エージェントはタイムアウトまで待って沈黙を未回答と取り違えるのではなく、`[clarify prompt could not be delivered]` ですぐに解放されます。

## プッシュ通知の量

Telegramは、ボットが送信するすべてのメッセージに対してプッシュ通知を発します。ツールの進捗バブル、ストリーミングの更新、ステータスのコールバックを出力する長いエージェントのターンでは、これがすぐにうるさくなります。Telegramアダプターには2つの通知モードがあります:

| モード | 動作 |
|------|----------|
| `important`（デフォルト） | **最終的な応答**、**承認プロンプト**、**スラッシュコマンドの確認** のみが通知を鳴らします。ツールの進捗、ストリーミングのチャンク、ステータスメッセージは `disable_notification=true` で配信されます。 |
| `all` | すべての送信メッセージがプッシュ通知を発します。従来の動作です。本当にすべてのツール呼び出しについて通知を受けたい場合にオプトインしてください。 |

`~/.hermes/config.yaml` で設定します:

```yaml
display:
  platforms:
    telegram:
      notifications: important   # or "all"
```

環境変数による上書き（手早いA/Bテストに便利）:

```bash
HERMES_TELEGRAM_NOTIFICATIONS=all
```

不明な値の場合は警告をログに出力し、`important` にフォールバックします。

`hermes send`、`send_message` ツール、スタンドアロンのcron配信も同じモードに従います。`important` のもとではこれらのメッセージは無音で届きますが、cronのブリーフは例外で、デフォルトで通知が鳴ります（ゲートウェイのレーンと同様に `cron.delivery.notify`）。

## その場で編集されるステータスメッセージ

Telegramアダプターは、繰り返し発生するエージェントのステータスコールバック（例: 「Compressing context…」、「Calling tool…」）を `send_or_update_status()` を通じてルーティングします。これは `{(chat_id, status_key) → message_id}` のキャッシュを保持し、以降の発行時には毎回新しいバブルを追加する代わりに **既存のバブルを編集** します。異なる `status_key` 値はそれぞれ独自のメッセージを持ち、異なるチャットが衝突することはありません。編集に失敗した場合（例: ユーザーがメッセージを削除した、またはTelegramが編集を許可する期間より古い）、キャッシュのエントリは破棄され、次の発行で新しいメッセージを投稿してそのIDを再キャッシュします。設定は不要です — これがTelegramのデフォルトの動作です。`send_or_update_status` を実装していない他のアダプターは、変更なしで通常の `send()` にフォールスルーします。

## エージェントのターン中に受信したユーザーメッセージをピン留めする

ユーザーがエージェントのターンをトリガーするメッセージを送信すると、Telegramアダプターはターンの間その受信メッセージをピン留めし、応答が完了するとピン留めを解除します — ボットがメッセージを無視しているのではなく、積極的に処理していることを示す軽量な視覚的インジケーターです。ピン留めは余分な通知を避けるために `disable_notification=true` を使用します。設定は不要です。

## セキュリティ

:::warning
ボットとやり取りできる人を制限するために、必ず `TELEGRAM_ALLOWED_USERS` を設定してください。設定しない場合、ゲートウェイは安全のためデフォルトですべてのユーザーを拒否します。
:::

ボットトークンを公に共有しないでください。万が一漏洩した場合は、BotFatherの `/revoke` コマンドですぐに無効化してください。

詳細については、[セキュリティのドキュメント](../security.md)を参照してください。ユーザー認可へのより動的なアプローチとして、[DMペアリング](./index.md#dm-pairing-alternative-to-allowlists)を使うこともできます。
