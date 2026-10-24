---
sidebar_position: 7
title: "セッション"
description: "セッションの永続化、再開、検索、管理、およびプラットフォームごとのセッション追跡"
---

# セッション

Hermes Agentは、すべての会話を自動的にセッションとして保存します。セッションにより、会話の再開、セッションをまたいだ検索、完全な会話履歴の管理が可能になります。

## セッションの仕組み

CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Teams、その他のメッセージングプラットフォームのいずれからの会話も、完全なメッセージ履歴を持つセッションとして保存されます。セッションは、相互に補完する2つのシステムで追跡されます。

1. **SQLiteデータベース**（`~/.hermes/state.db`） — FTS5全文検索を備えた構造化されたセッションメタデータ
2. **JSONLトランスクリプト**（`~/.hermes/sessions/`） — ツール呼び出しを含む生の会話トランスクリプト（ゲートウェイ）

SQLiteデータベースには以下が保存されます。
- セッションID、ソースプラットフォーム、ユーザーID
- **セッションタイトル**（一意で人間が読める名前）
- モデル名と設定
- システムプロンプトのスナップショット
- 完全なメッセージ履歴（ロール、コンテンツ、ツール呼び出し、ツール結果）
- トークン数（入力/出力）
- タイムスタンプ（started_at、ended_at）
- 親セッションID（圧縮によってトリガーされたセッション分割用）

### セッションソース

各セッションには、そのソースプラットフォームのタグが付けられます。

| ソース | 説明 |
|--------|-------------|
| `cli` | 対話型CLI（`hermes` または `hermes chat`） |
| `telegram` | Telegramメッセンジャー |
| `discord` | Discordサーバー/DM |
| `slack` | Slackワークスペース |
| `whatsapp` | WhatsAppメッセンジャー |
| `signal` | Signalメッセンジャー |
| `matrix` | MatrixのルームとDM |
| `mattermost` | Mattermostチャンネル |
| `email` | Email（IMAP/SMTP） |
| `sms` | Twilio経由のSMS |
| `dingtalk` | DingTalkメッセンジャー |
| `feishu` | Feishu/Larkメッセンジャー |
| `wecom` | WeCom（WeChat Work） |
| `weixin` | Weixin（個人用WeChat） |
| `bluebubbles` | BlueBubbles macOSサーバー経由のApple iMessage |
| `qqbot` | 公式API v2経由のQQ Bot（Tencent QQ） |
| `homeassistant` | Home Assistantの会話 |
| `webhook` | 着信Webhook |
| `api-server` | APIサーバーのリクエスト |
| `acp` | ACPエディター連携 |
| `cron` | スケジュールされたcronジョブ |
| `batch` | バッチ処理の実行 |

## CLIでのセッション再開

`--continue` または `--resume` を使用して、CLIから以前の会話を再開します。

### 直近のセッションを継続

```bash
# 最も新しいCLIセッションを再開
hermes --continue
hermes -c

# またはchatサブコマンドで
hermes chat --continue
hermes chat -c
```

これはSQLiteデータベースから最も新しい `cli` セッションを検索し、その完全な会話履歴を読み込みます。

### 名前で再開

セッションにタイトルを付けている場合（下記の[セッションの命名](#session-naming)を参照）、名前で再開できます。

```bash
# 名前を付けたセッションを再開
hermes -c "my project"

# 系譜のバリアント（my project、my project #2、my project #3）がある場合、
# 自動的に最も新しいものを再開します
hermes -c "my project"   # → "my project #3" を再開
```

### 特定のセッションを再開

```bash
# IDで特定のセッションを再開
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# タイトルで再開
hermes --resume "refactoring auth"

# またはchatサブコマンドで
hermes chat --resume 20250305_091523_a1b2c3d4
```

セッションIDはCLIセッションを終了するときに表示され、`hermes sessions list` で確認できます。

### 再開時の会話リキャップ {#conversation-recap-on-resume}

セッションを再開すると、Hermesは入力プロンプトの前に、スタイル付きのパネルで以前の会話のコンパクトなリキャップを表示します。

<img className="docs-terminal-figure" src="/img/docs/session-recap.svg" alt="Stylized preview of the Previous Conversation recap panel shown when resuming a Hermes session." />
<p className="docs-figure-caption">再開モードでは、ライブプロンプトに戻る前に、最近のユーザーとアシスタントのターンを含むコンパクトなリキャップパネルが表示されます。</p>

リキャップでは:
- **ユーザーメッセージ**（金色の `●`）と**アシスタントの応答**（緑色の `◆`）を表示します
- 長いメッセージを**切り詰めます**（ユーザーは300文字、アシスタントは200文字 / 3行）
- ツール呼び出しを、ツール名を含む件数に**折りたたみます**（例: `[3 tool calls: terminal, web_search]`）
- システムメッセージ、ツール結果、内部の推論を**非表示**にします
- 直近の10回のやり取りで**上限**を設け、「... N earlier messages ...」のインジケーターを付けます
- アクティブな会話と区別するため、**淡いスタイル**を使用します

リキャップを無効にして最小限の1行表示の動作を保つには、`~/.hermes/config.yaml` で次のように設定します。

```yaml
display:
  resume_display: minimal   # デフォルト: full
```

:::tip
セッションIDは `YYYYMMDD_HHMMSS_<hex>` という形式に従います。CLI/TUIセッションは6文字の16進数サフィックス（例: `20250305_091523_a1b2c3`）を使用し、ゲートウェイセッションは8文字のサフィックス（例: `20250305_091523_a1b2c3d4`）を使用します。ID（完全な、または一意のプレフィックス）またはタイトルで再開でき、どちらも `-c` と `-r` で機能します。
:::

## クロスプラットフォームハンドオフ

CLIセッションから `/handoff <platform>` を使用して、ライブの会話をメッセージングプラットフォームのホームチャンネルに転送します。エージェントはCLIが中断した正確な地点から再開します。同じセッションID、完全なロール対応のトランスクリプト、ツール呼び出しなどすべてです。

```bash
# CLIセッション内で
/handoff telegram
```

何が起こるか:

1. CLIは `<platform>` が有効になっており、ホームチャンネルが設定されていることを検証します（転送先のチャットから `/sethome` を一度実行して設定します）。
2. CLIはセッションを保留中としてマークし、**ゲートウェイをブロックポーリング**します。エージェントがターンの途中であれば拒否します。まず現在の応答が終わるのを待ってください。
3. ゲートウェイのウォッチャーがハンドオフを引き受け、転送先のアダプターに新しいスレッドを要求します。
   - **Telegram** — 新しいフォーラムトピックを開きます（チャットでBot API 9.4+のTopicsモードが有効な場合はDMトピック、またはフォーラムスーパーグループのトピック）。
   - **Discord** — ホームのテキストチャンネルの下に、1440分の自動アーカイブスレッドを作成します。
   - **Slack** — シードメッセージを投稿し、その `ts` をスレッドのアンカーとして使用します。
   - **WhatsApp / Signal / Matrix / SMS** — ネイティブなスレッドがないため、直接ホームチャンネルにフォールバックします。
4. ゲートウェイは転送先のキーを既存のCLIセッションIDに再バインドし、エージェントに確認と要約を求める合成ユーザーターンを生成します。応答は新しいスレッドに届きます。
5. ゲートウェイが成功を確認すると、CLIは `/resume` のヒントを表示してクリーンに終了します。

   ```
   ↻ Handoff complete. The session is now active on telegram.
     Resume it on this CLI later with: /resume my-session-title
   ```

6. それ以降、会話はプラットフォーム上に存在します。新しいスレッドで返信してください。そのチャンネルで認証された人は誰でも同じセッションを共有し、スレッド内での後続の実際のユーザーメッセージはシームレスに参加します。スレッドセッションは `user_id` なしでキー付けされるためです。

**CLIへの再開:** デスクトップに戻りたいときは、`/resume <title>`（またはシェルから `hermes -r "<title>"`）を実行するだけで、プラットフォームが中断した地点から再開できます。

**失敗モード:**
- ホームチャンネルが未設定 → CLIは `/sethome` のヒントとともに拒否します。
- プラットフォームが無効 / ゲートウェイが実行されていない → CLIは60秒で明確なメッセージとともにタイムアウトし、CLIセッションはそのまま維持されます。
- スレッド作成に失敗（権限、Topicsモードがオフ） → 直接ホームチャンネルにフォールバックし、それでも完了します。スレッドの分離はありませんが、ハンドオフ自体は機能します。
- `adapter.send` が失敗（レート制限、一時的なAPIエラー） → ハンドオフは理由とともに失敗とマークされます。行はクリアされるので、再試行できます。

**知っておくべき制限:** スレッド非対応のプラットフォームで、複数ユーザーのグループホームチャンネルの場合、合成ターンはDMスタイルのセッションとしてキー付けされます。これはセルフDMのホームチャンネル（典型的なセットアップ）では機能しますが、真に共有されるグループチャットには理想的ではありません。スレッド化はTelegram / Discord / Slackをカバーしており（圧倒的に一般的なケース）、ほとんどのセットアップではこの問題に遭遇しません。

## セッションの命名 {#session-naming}

セッションに人間が読めるタイトルを付けて、簡単に見つけて再開できるようにします。

### 自動生成されるタイトル

Hermesは、最初のやり取りの後、各セッションに対して短い説明的なタイトル（3〜7語）を自動的に生成します。これは高速な補助モデルを使用してバックグラウンドスレッドで実行されるため、レイテンシは追加されません。`hermes sessions list` や `hermes sessions browse` でセッションを閲覧するときに、自動生成されたタイトルが表示されます。

自動タイトル付けはセッションごとに1回だけ発火し、すでに手動でタイトルを設定している場合はスキップされます。

### タイトルを手動で設定する

任意のチャットセッション（CLIまたはゲートウェイ）内で `/title` スラッシュコマンドを使用します。

```
/title my research project
```

タイトルは即座に適用されます。セッションがまだデータベースに作成されていない場合（例: 最初のメッセージを送る前に `/title` を実行した場合）、キューに入れられ、セッションが開始されると適用されます。

コマンドラインから既存のセッションの名前を変更することもできます。

```bash
hermes sessions rename 20250305_091523_a1b2c3d4 "refactoring auth module"
```

### タイトルのルール

- **一意** — 2つのセッションが同じタイトルを共有することはできません
- **最大100文字** — 一覧表示の出力をすっきりと保ちます
- **サニタイズ** — 制御文字、ゼロ幅文字、RTLオーバーライドは自動的に取り除かれます
- **通常のUnicodeは問題ありません** — 絵文字、CJK、アクセント付き文字はすべて機能します

### 圧縮時の自動系譜

セッションのコンテキストが圧縮されると（`/compress` で手動、または自動で）、Hermesは新しい継続セッションを作成します。元のセッションにタイトルがあった場合、新しいセッションには自動的に番号付きのタイトルが付けられます。

```
"my project" → "my project #2" → "my project #3"
```

名前で再開すると（`hermes -c "my project"`）、系譜の中で最も新しいセッションが自動的に選択されます。

### メッセージングプラットフォームでの /title

`/title` コマンドはすべてのゲートウェイプラットフォーム（Telegram、Discord、Slack、WhatsApp）で機能します。

- `/title My Research` — セッションタイトルを設定
- `/title` — 現在のタイトルを表示

## セッション管理コマンド

Hermesは `hermes sessions` を介して、完全なセッション管理コマンドのセットを提供します。

### セッションの一覧表示

```bash
# 最近のセッションを一覧表示（デフォルト: 直近20件）
hermes sessions list

# プラットフォームでフィルタ
hermes sessions list --source telegram

# より多くのセッションを表示
hermes sessions list --limit 50
```

セッションにタイトルがある場合、出力にはタイトル、プレビュー、相対タイムスタンプが表示されます。

```
Title                  Preview                                  Last Active   ID
────────────────────────────────────────────────────────────────────────────────────────────────
refactoring auth       Help me refactor the auth module please   2h ago        20250305_091523_a
my project #3          Can you check the test failures?          yesterday     20250304_143022_e
—                      What's the weather in Las Vegas?          3d ago        20250303_101500_f
```

タイトルを持つセッションがない場合は、よりシンプルな形式が使用されます。

```
Preview                                            Last Active   Src    ID
──────────────────────────────────────────────────────────────────────────────────────
Help me refactor the auth module please             2h ago        cli    20250305_091523_a
What's the weather in Las Vegas?                    3d ago        tele   20250303_101500_f
```

### セッションのエクスポート

```bash
# すべてのセッションをJSONLファイルにエクスポート
hermes sessions export backup.jsonl

# 特定のプラットフォームのセッションをエクスポート
hermes sessions export telegram-history.jsonl --source telegram

# 単一のセッションをエクスポート
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4
```

エクスポートされたファイルには、完全なセッションメタデータとすべてのメッセージを含むJSONオブジェクトが1行に1つ含まれます。

### セッションの削除

```bash
# 特定のセッションを削除（確認あり）
hermes sessions delete 20250305_091523_a1b2c3d4

# 確認なしで削除
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### セッションの名前変更

```bash
# セッションのタイトルを設定または変更
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"

# 複数語のタイトルはCLIで引用符が不要です
hermes sessions rename 20250305_091523_a1b2c3d4 debugging auth flow
```

タイトルがすでに別のセッションで使用されている場合は、エラーが表示されます。

### 古いセッションの整理（prune）

```bash
# 90日より古い終了済みセッションを削除（デフォルト）
hermes sessions prune

# カスタムの経過日数しきい値
hermes sessions prune --older-than 30

# 特定のプラットフォームのセッションのみを整理
hermes sessions prune --source telegram --older-than 60

# 確認をスキップ
hermes sessions prune --older-than 30 --yes
```

:::info
整理（prune）は**終了済み**のセッション（明示的に終了された、または自動リセットされたセッション）のみを削除します。アクティブなセッションは決して整理されません。
:::

### セッションの統計

```bash
hermes sessions stats
```

出力:

```
Total sessions: 142
Total messages: 3847
  cli: 89 sessions
  telegram: 38 sessions
  discord: 15 sessions
Database size: 12.4 MB
```

より詳細な分析 — トークン使用量、コスト見積もり、ツールの内訳、アクティビティパターン — については、[`hermes insights`](/docs/reference/cli-commands#hermes-insights)を使用してください。

## セッション検索ツール

エージェントには、SQLiteのFTS5エンジンを使用して過去のすべての会話を全文検索する組み込みの `session_search` ツールがあります。

### 仕組み

1. FTS5が関連性でランク付けされた一致するメッセージを検索します
2. 結果をセッションごとにグループ化し、上位N件の一意のセッション（デフォルト3件）を取得します
3. 各セッションの会話を読み込み、一致箇所を中心に約10万文字に切り詰めます
4. 焦点を絞った要約のために高速な要約モデルに送信します
5. メタデータと周辺コンテキストを含むセッションごとの要約を返します

### FTS5クエリ構文

検索は標準のFTS5クエリ構文をサポートします。

- 単純なキーワード: `docker deployment`
- フレーズ: `"exact phrase"`
- ブール演算: `docker OR kubernetes`、`python NOT java`
- プレフィックス: `deploy*`

### 使用されるタイミング

エージェントはセッション検索を自動的に使用するよう促されます。

> *「ユーザーが過去の会話の何かに言及した場合、または関連する以前のコンテキストが存在すると思われる場合は、繰り返しを求める前に session_search を使用してそれを思い出してください。」*

## プラットフォームごとのセッション追跡

### ゲートウェイセッション

メッセージングプラットフォームでは、セッションはメッセージソースから構築された決定論的なセッションキーでキー付けされます。

| チャットタイプ | デフォルトのキー形式 | 動作 |
|-----------|--------------------|----------|
| Telegram DM | `agent:main:telegram:dm:<chat_id>` | DMチャットごとに1つのセッション |
| Discord DM | `agent:main:discord:dm:<chat_id>` | DMチャットごとに1つのセッション |
| WhatsApp DM | `agent:main:whatsapp:dm:<canonical_identifier>` | DMユーザーごとに1つのセッション（マッピングが存在する場合、LID/電話番号のエイリアスは1つのアイデンティティに集約されます） |
| グループチャット | `agent:main:<platform>:group:<chat_id>:<user_id>` | プラットフォームがユーザーIDを公開する場合、グループ内のユーザーごと |
| グループスレッド/トピック | `agent:main:<platform>:group:<chat_id>:<thread_id>` | すべてのスレッド参加者で共有されるセッション（デフォルト）。`thread_sessions_per_user: true` でユーザーごと。 |
| チャンネル | `agent:main:<platform>:channel:<chat_id>:<user_id>` | プラットフォームがユーザーIDを公開する場合、チャンネル内のユーザーごと |

Hermesが共有チャットの参加者識別子を取得できない場合は、そのルームに対して1つの共有セッションにフォールバックします。

### 共有グループセッション対分離グループセッション

デフォルトでは、Hermesは `config.yaml` で `group_sessions_per_user: true` を使用します。これは次のことを意味します。

- AliceとBobは、同じDiscordチャンネルで、トランスクリプト履歴を共有せずに両方ともHermesと話すことができます
- あるユーザーのツールを多用する長いタスクが、別のユーザーのコンテキストウィンドウを汚染しません
- 実行中エージェントのキーが分離されたセッションキーと一致するため、中断処理もユーザーごとに保たれます

代わりに1つの共有された「ルームの頭脳」が欲しい場合は、次のように設定します。

```yaml
group_sessions_per_user: false
```

これにより、グループ/チャンネルはルームごとに単一の共有セッションに戻り、共有された会話コンテキストは保持されますが、トークンコスト、中断状態、コンテキストの増加も共有されます。

### セッションリセットポリシー

ゲートウェイセッションは、設定可能なポリシーに基づいて自動的にリセットされます。

- **idle** — N分間の無活動後にリセット
- **daily** — 毎日特定の時刻にリセット
- **both** — いずれか先に来た方でリセット（idleまたはdaily）
- **none** — 自動リセットしない

セッションが自動リセットされる前に、エージェントには会話から重要なメモリやスキルを保存するためのターンが与えられます。

**アクティブなバックグラウンドプロセス**を持つセッションは、ポリシーに関係なく決して自動リセットされません。

## ストレージの場所

| 内容 | パス | 説明 |
|------|------|-------------|
| SQLiteデータベース | `~/.hermes/state.db` | FTS5を備えたすべてのセッションメタデータ + メッセージ |
| ゲートウェイトランスクリプト | `~/.hermes/sessions/` | セッションごとのJSONLトランスクリプト + sessions.jsonインデックス |
| ゲートウェイインデックス | `~/.hermes/sessions/sessions.json` | セッションキーをアクティブなセッションIDにマッピング |

SQLiteデータベースは、並行リーダーと単一のライターのためにWALモードを使用しており、ゲートウェイのマルチプラットフォームアーキテクチャに適しています。

### データベーススキーマ

`state.db` の主要なテーブル:

- **sessions** — セッションメタデータ（id、source、user_id、model、title、タイムスタンプ、トークン数）。タイトルには一意のインデックスがあります（NULLタイトルは許可され、非NULLのみ一意である必要があります）。
- **messages** — 完全なメッセージ履歴（role、content、tool_calls、tool_name、token_count）
- **messages_fts** — メッセージコンテンツ全体の全文検索のためのFTS5仮想テーブル

## セッションの期限切れとクリーンアップ

### 自動クリーンアップ

- ゲートウェイセッションは、設定されたリセットポリシーに基づいて自動リセットされます
- リセット前に、エージェントは期限切れになるセッションからメモリとスキルを保存します
- オプトインの自動整理: `sessions.auto_prune` が `true` の場合、`sessions.retention_days`（デフォルト90）より古い終了済みセッションが、CLI/ゲートウェイの起動時に整理されます
- 実際に行が削除された整理の後、`state.db` は `VACUUM` されてディスク領域が回収されます（SQLiteは通常のDELETEではファイルを縮小しません）
- 整理は `sessions.min_interval_hours`（デフォルト24）ごとに最大1回実行されます。最終実行のタイムスタンプは `state.db` 自体の内部で追跡されるため、同じ `HERMES_HOME` 内のすべてのHermesプロセスで共有されます

デフォルトは**オフ**です。セッション履歴は `session_search` での想起に貴重であり、それを黙って削除するとユーザーを驚かせる可能性があるためです。`~/.hermes/config.yaml` で有効にします。

```yaml
sessions:
  auto_prune: true          # オプトイン — デフォルトは false
  retention_days: 90        # 終了済みセッションをこの日数だけ保持
  vacuum_after_prune: true  # 整理スイープ後にディスク領域を回収
  min_interval_hours: 24    # これより頻繁にスイープを再実行しない
```

アクティブなセッションは、経過時間に関係なく決して自動整理されません。

### 手動クリーンアップ

```bash
# 90日より古いセッションを整理
hermes sessions prune

# 特定のセッションを削除
hermes sessions delete <session_id>

# 整理前にエクスポート（バックアップ）
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
データベースはゆっくりと成長し（典型的には数百のセッションで10〜15 MB）、セッション履歴は過去の会話にわたる `session_search` の想起を支えるため、自動整理は無効の状態で出荷されます。`state.db` がパフォーマンスに意味のある影響を与えている重いゲートウェイ/cronワークロードを実行している場合は有効にしてください（観測された失敗モード: 約1000セッションの384 MBのstate.dbがFTS5の挿入や `/resume` の一覧表示を遅くする）。自動スイープをオンにせずに1回限りのクリーンアップを行うには、`hermes sessions prune` を使用してください。
:::
