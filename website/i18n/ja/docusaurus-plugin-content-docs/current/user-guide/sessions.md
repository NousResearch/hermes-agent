---
sidebar_position: 7
title: "セッション"
description: "セッションの永続化、再開、検索、管理、およびプラットフォームごとのセッション追跡"
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# セッション

Hermes Agentは、すべての会話を自動的にセッションとして保存します。セッションにより、会話の再開、セッションをまたいだ検索、完全な会話履歴の管理が可能になります。

## セッションの仕組み

CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Teams、その他のメッセージングプラットフォームのいずれからの会話も、完全なメッセージ履歴を持つセッションとして保存されます。セッションは次の場所で追跡されます。

1. **SQLiteデータベース**（`~/.hermes/state.db`） — FTS5全文検索を備えた構造化されたセッションメタデータと、完全なメッセージ履歴

SQLiteデータベースには以下が保存されます。
- セッションID、ソースプラットフォーム、ユーザーID
- **セッションタイトル**（一意で人間が読める名前）
- モデル名と設定
- システムプロンプトのスナップショット
- 完全なメッセージ履歴（ロール、コンテンツ、ツール呼び出し、ツール結果）
- トークン数（入力/出力）
- タイムスタンプ（started_at、ended_at）
- 親セッションID（圧縮によってトリガーされたセッション分割用）

### 認証済みゲートウェイセッションへのアクセスと帰属

実行を所有するのはゲートウェイであり、クライアントは実行中のエージェントを所有しません。同じゲートウェイプロファイルの検証済みオペレーターは、別のオペレーター認証方式で作成された正規のローカルCLI、TUI、またはDesktopセッションにアタッチできます。たとえば、ネイティブのローカルオペレーターと認証済みのダッシュボードオペレーターは、別のランタイムを作成することなく同じ会話にアクセスできます。クライアント接続を閉じるとそのクライアントはデタッチされますが、受け付けられた作業はゲートウェイに残ります。

これは検証済みオペレーターに対する作成者限定のアクセスを意図的に置き換えるもので、ダッシュボードの既存のオペレーター権限を維持します。パスワードとOAuthのアイデンティティは区別されたままですが、両方の接続がプロファイルへのオペレーターアクセスを持つ場合、正規のローカルセッションに対して別々のアクセス障壁は作りません。信頼されるサーバーパスには、ネイティブの対話型ブートストラップ、認証済みダッシュボードチケット、サーバーが起動したPTYクライアントが使用する内部資格情報、検証済みのレガシーダッシュボードトークンが含まれます。HTTPのセッション変更操作は、検証済みのネイティブHTTPオーナーグラントも受け付けます。これは、制限されたワーカー、メッセージング、ルーム、サービスの資格情報にオペレーター権限を付与するものではなく、プロファイルや操作固有のチェックを取り除くものでもありません。

帰属はアイデンティティごとに維持されます。ダッシュボードのアイデンティティには、プロバイダー、検証済みのOAuth発行者（存在する場合）、サブジェクトが含まれます。作成レシートは元のアイデンティティを保持し、送信と変更操作のリトライは操作を行ったアイデンティティを保持します。オペレーターがセッションに参加しても、記録されたオーナーが置き換えられたり、リトライのアイデンティティがマージされたりすることはありません。ネイティブブートストラップのオーナーキーとセッションIDは変わりません。

レシートにサブジェクトしか含まれていない古いリモートセッションは、どのプロバイダーが所有していたかを確定できません。それらは保存されたままですが、最初にリクエストした認証済みアカウントに自動的に割り当てられることはありません。オペレーターアクセスでは、所有者のいない過去のトランスクリプトをリモートから先着で引き取ることはできません。

デバイス間のアタッチには、同じゲートウェイへのサポートされた認証済みルートも必要です。共有されたオペレーターアクセス自体がリモートリスナーを公開したり、ローカルセッションのOpenAI互換API経由での継続を追加したりすることはありません。ゲートウェイのクラッシュはクライアントの切断とは異なります。中断された作業は自動的に再実行されるのではなく `unknown` になり、キューに入った後続の処理が進む前に解決が必要になる場合があります。

### コンテキストにカウントされるもの

Hermesは会話を再開できるようにセッション履歴を保存しますが、これまでに扱ったすべてのバイトを送り直し続けるわけではありません。各ターンでモデルが見るのは、選択されたシステムプロンプト、現在の会話ウィンドウ、そしてそのターンのためにHermesが明示的に注入したコンテンツです。

メディアの添付ファイルは、ターン単位の入力として扱われます。

- 画像は次のモデル呼び出しにネイティブに添付されるか、アクティブなモデルがネイティブのビジョンをサポートしていない場合は事前に解析されてテキストの説明になります。
- 音声は、音声認識（speech-to-text）が設定されている場合にテキストへ文字起こしされます。
- テキストドキュメントは抽出されたテキストが含まれることがあります。その他のドキュメントタイプは通常、保存されたローカルパスと短いメモで表現されます。
- 添付ファイルのパスや抽出/派生したテキストはトランスクリプトに表示されることがありますが、生の画像、音声、バイナリファイルのバイトが将来のプロンプトに繰り返しコピーされることはありません。

たとえば、ユーザーが画像を送ってミームを作るよう依頼した場合、Hermesはその画像をビジョンで一度調べ、画像処理スクリプトを実行することがあります。以降のターンが元のJPEGを自動的にコンテキストに持ち込むことはありません。持ち込まれるのは、ユーザーのリクエスト、短い画像の説明、ローカルキャッシュのパス、最終的なアシスタントの応答など、会話に書き込まれた内容だけです。

コンテキストが増える最も一般的な原因は、メディアファイルそのものではありません。冗長なテキストです。貼り付けられたトランスクリプト、完全なログ、大きなツール出力、長いdiff、繰り返されるステータスレポート、詳細な証跡のダンプなどです。大きな成果物をチャットにコピーするよりも、要約、ファイルパス、焦点を絞った抜粋、ツールによる参照を優先してください。

:::tip
セッションが長くなったら `/compress` を、新しいスレッドには `/new` を使用し、古い終了済みセッションをストレージから削除したいときにのみ `hermes sessions prune` を使用してください。`state.db` が単に大きくなっただけなら、まず非破壊的な選択肢から始めてください。`hermes sessions optimize` は、セッションデータに一切触れずにFTS5インデックスのセグメントをマージし、データベースを `VACUUM` します。ゲートウェイが実行中の場合、`prune` は選択したセッションの削除をゲートウェイに依頼します。`optimize`、およびゲートウェイが実行されていないときの `prune` は、別のHermesプロセス（Desktop、ダッシュボード、cron）が `state.db` を保持している間は拒否されます。先にそれを停止するか、`--force` を渡してください。[セッションストレージの復旧](/docs/user-guide/session-storage-recovery)を参照してください。
圧縮はアクティブなコンテキストを縮小するものであり、プライバシーのための削除ではありません。
`/new` に名前を渡すと（例: `/new payments-refactor`）、新しいセッションの初期タイトルを最初から設定できます。後で `/resume <name>` や `/sessions` ピッカーで見つけるのに便利です。
:::

### セッションソース

各セッションには、そのソースプラットフォームのタグが付けられます。

| ソース | 説明 |
|--------|-------------|
| `cli` | 対話型CLI（`hermes` または `hermes chat`） |
| `oneshot` | 有限の非対話型実行: `hermes chat --oneshot -q`、`-Q`、`hermes -z`、および非TTYのstdioでの `-q`。TUI、Desktop、ダッシュボードのセッションピッカーには表示されません（`kanban` や `tool` と同様）。TUIやDesktopのセッション内から起動された場合でも同様です。実行はそのトランスポートの環境を引き継ぎますが、その会話そのものではありません。それでもCLI履歴としてカウントされます。`hermes -c` / `--resume latest` は直近のワンショットを継続し、`hermes sessions list` にも表示されます。明示的な `--source <tag>` が常に優先されます（`hermes chat -q --source tui` は `tui` として保存されます）。 |
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
| `homeassistant` | Home Assistantのイベント（プラグイン） |
| `webhook` | 着信Webhook |
| `api-server` | APIサーバーのリクエスト |
| `acp` | ACPエディター連携 |
| `cron` | スケジュールされたcronジョブ |
| `batch` | バッチ処理の実行 |
| `kanban` | Kanbanディスパッチャーのワーカー（ボード上で閲覧でき、セッションピッカーには表示されません） |
| `tool` | サードパーティ連携（`--source tool`）。セッションピッカーには表示されません |

会話の途中で圧縮されたセッションは、同じソースのまま継続します。`--source tool` や `oneshot` の実行の圧縮による子セッションにも同じタグが付けられるため、同じピッカーでの表示/非表示を引き継ぎます。

## CLIでのセッション再開

`--continue` または `--resume` を使用して、CLIから以前の会話を再開します。

### 直近のセッションを継続

```bash
# Resume the most recent CLI session
hermes --continue
hermes -c

# Or with the chat subcommand
hermes chat --continue
hermes chat -c
```

これはSQLiteデータベースから最も新しい `cli` セッションを検索し、その完全な会話履歴を読み込みます。

#### ターミナルごとの継続 {#per-terminal-continue}

引数なしの `-c` はターミナルを認識します。各CLIセッションは、実行中のターミナル（ttyデバイス、tmuxペイン、kittyウィンドウ、weztermペイン、Zellijペイン、Windows Terminalセッションなど）をキーとして、`~/.hermes/terminal-sessions/` の下に小さなブレッドクラムファイルを残します。*同じ*ターミナルで再び `hermes -c` を実行すると、Hermesはそのターミナル自身のセッションを再開します。そのため、横に並んだ2つのペインは、両方とも全体で最も新しいセッションを掴むのではなく、それぞれ自分の会話を継続します。そのターミナルにブレッドクラムがない場合（初回使用、削除されたセッション、または30日より古い古いブレッドクラム）、`-c` は最も新しいセッションを使う動作にフォールバックします。`-c "name"` と `--resume` は影響を受けません。`config.yaml` で `session.terminal_continue: false` を設定すると無効になります。

### 名前で再開

セッションにタイトルを付けている場合（下記の[セッションの命名](#session-naming)を参照）、名前で再開できます。

```bash
# Resume a named session
hermes -c "my project"

# If there are lineage variants (my project, my project #2, my project #3),
# this automatically resumes the most recent one
hermes -c "my project"   # → resumes "my project #3"
```

### 特定のセッションを再開

```bash
# Resume a specific session by ID
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# Resume by title
hermes --resume "refactoring auth"

# Resume the most recent session — same lookup as -c
hermes --resume latest

# Or with the chat subcommand
hermes chat --resume 20250305_091523_a1b2c3d4
```

セッションIDはCLIセッションを終了するときに表示され、`hermes sessions list` で確認できます。

:::note
`latest` は `--resume` の予約キーワードです。文字どおり「latest」というタイトルのセッションには、IDまたは `-c latest`（タイトル一致）で引き続きアクセスできます。
:::

### 特定のディレクトリで再開

`--in <dir>` を渡すと、開始または再開の前にそのディレクトリへ移動します。`--resume latest`（または `-c`）と組み合わせると、そのディレクトリのワークスペースで最も新しいセッションが選ばれます。先に `cd` したり、セッションIDを覚えておいたりする必要はありません。

```bash
# Resume the latest session that belongs to ./my-project
hermes --resume latest --in ./my-project

# Works with the TUI too
hermes --tui --resume latest --in ./my-project
```

`--in` はセッションをそのディレクトリに固定もします。再開したセッションに記録された作業ディレクトリは復元されません（`--no-restore-cwd` を渡したのと同じです）。

### 再開時に作業ディレクトリを復元

CLIセッションを再開すると、セッションに記録された作業ディレクトリ（gitリポジトリのルートまたはプロジェクトディレクトリ）にも `cd` で戻るため、会話は元のワークスペースで再開されます。現在の場所に留まりたい場合は、`--no-restore-cwd` を渡してください。

```bash
hermes --resume 20250305_091523_a1b2c3 --no-restore-cwd
```

`↪ restored workspace dir: …` という行が切り替えを確認します。復元に失敗しても、再開自体が壊れることはありません。

アイドル状態のセッションを再開すると、プロファイルの現在の設定が反映されます。ツールセット、MCPサーバー、`terminal` の設定、`max_turns`、チェックポイント、デフォルトの推論努力（reasoning effort）です。セッションは、そのモデル、起動フラグ、そして行った `/reasoning` やワークスペースの変更を保持します。サイドタスク用のモデル（`auxiliary:` セクション: 圧縮、ビジョン、タイトル生成、その他の補助呼び出し）は、それらのキーおよびそれらが指定する `providers:` エントリとともに、常にプロファイルの現在の `config.yaml` から取得されます。そのため、編集は既存セッションの次のサイドタスク呼び出しから反映されます。`--ignore-user-config` または `--safe-mode` で開始されたセッションは、それらを決して読み込みません。何らかの差分があると新しいプロンプトキャッシュのプレフィックスが始まるため、再開後の最初のターンはコストが高くなります。`terminal` の設定の変更は、`terminal.timeout` のような無関係なものであっても、以前の `cd` をセッションのディレクトリにリセットします。

### ワークスペースでセッションをフィルタ

`hermes sessions list` は `--workspace <needle>` を受け付け、ワークスペースキー（gitリポジトリのルート、なければcwd）がパスの部分文字列またはディレクトリのベース名の完全一致でマッチするセッションのみを表示します。

```bash
hermes sessions list --workspace my-project
hermes sessions list --workspace ~/code/hermes-agent
```

### 再開時の会話リキャップ {#conversation-recap-on-resume}

セッションを再開すると、Hermesは入力プロンプトの前に、スタイル付きのパネルで以前の会話のコンパクトなリキャップを表示します。

<img className="docs-terminal-figure" src={useBaseUrl('/img/docs/session-recap.svg')} alt="Stylized preview of the Previous Conversation recap panel shown when resuming a Hermes session." />
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
  resume_display: minimal   # default: full
```

:::tip
セッションIDは `YYYYMMDD_HHMMSS_<hex>` という形式に従います。CLI/TUIセッションは6文字の16進数サフィックス（例: `20250305_091523_a1b2c3`）を使用し、ゲートウェイセッションは8文字のサフィックス（例: `20250305_091523_a1b2c3d4`）を使用します。ID（完全な、または一意のプレフィックス）またはタイトルで再開でき、どちらも `-c` と `-r` で機能します。
:::

## クロスプラットフォームハンドオフ {#cross-platform-handoff}

CLIセッションから `/handoff <platform>` を使用して、ライブの会話をメッセージングプラットフォームのホームチャンネルに転送します。エージェントはCLIが中断した正確な地点から再開します。同じセッションID、完全なロール対応のトランスクリプト、ツール呼び出しなどすべてです。

```bash
# Inside a CLI session
/handoff telegram
```

何が起こるか:

1. CLIは `<platform>` が有効になっており、ホームチャンネルが設定されていることを検証します（転送先のチャットから `/sethome` を一度実行して設定します）。
2. CLIはセッションを保留中としてマークし、**ゲートウェイをブロックポーリング**します。エージェントがターンの途中であれば拒否します。まず現在の応答が終わるのを待ってください。
3. ゲートウェイのウォッチャーがハンドオフを引き受け、転送先のアダプターに新しいスレッドを要求します。
   - **Telegram** — 新しいフォーラムトピックを開きます（ボットのオーナーがBotFatherでThreaded Modeを有効にしている場合はDMトピック、またはフォーラムスーパーグループのトピック）。
   - **Discord** — ホームのテキストチャンネルの下に、1440分の自動アーカイブスレッドを作成します。
   - **Slack** — シードメッセージを投稿し、その `ts` をスレッドのアンカーとして使用します。
   - **Matrix** — シードメッセージを投稿し、そのイベントIDをスレッドのルート（`m.thread` リレーション）として使用します。
   - **WhatsApp / Signal / SMS** — ネイティブなスレッドがないため、直接ホームチャンネルにフォールバックします。
4. ゲートウェイは転送先のキーを既存のCLIセッションIDに再バインドし、エージェントに確認と要約を求める合成ユーザーターンを生成します。応答は新しいスレッドに届きます。
5. ゲートウェイが成功を確認すると、CLIは `/resume` のヒントを表示してクリーンに終了します。

   ```
   ↻ Handoff complete. The session is now active on telegram.
     Resume it on this CLI later with: /resume my-session-title
   ```

6. それ以降、会話はプラットフォーム上に存在します。新しいスレッドで返信してください。そのチャンネルで認証された人は誰でも同じセッションを共有し、スレッド内での後続の実際のユーザーメッセージはシームレスに参加します。スレッドセッションは `user_id` なしでキー付けされるためです。

**CLIへの再開:** デスクトップに戻りたいときは、`/resume <title>`（またはシェルから `hermes -r "<title>"`）を実行するだけで、プラットフォームが中断した地点から再開できます。再度ハンドオフすると会話は新しいTelegramトピックに移動し、以前のトピックは次のメッセージで新しい会話を開始します。

**失敗モード:**
- ホームチャンネルが未設定 → CLIは `/sethome` のヒントとともに拒否します。
- ゲートウェイが実行されていない（誰もリクエストを引き受けない） → CLIは60秒で明確なメッセージとともにタイムアウトし、CLIセッションはそのまま維持されます。
- 転送が遅い: ゲートウェイがハンドオフを引き受けると、実際のエージェントターンを通じてセッション全体を再生するため、長いセッションでは数分かかることがあります。CLIは「Still transferring...」のハートビートを表示して最大15分待ちます。遅い転送を「ゲートウェイが実行されていない」と誤って報告することはありません。
- スレッド作成に失敗（権限、Topicsモードがオフ） → 直接ホームチャンネルにフォールバックし、それでも完了します。スレッドの分離はありませんが、ハンドオフ自体は機能します。
- `adapter.send` が失敗（レート制限、一時的なAPIエラー） → ハンドオフは理由とともに失敗とマークされます。行はクリアされるので、再試行できます。

**知っておくべき制限:** スレッド非対応のプラットフォームで、複数ユーザーのグループホームチャンネルの場合、合成ターンはDMスタイルのセッションとしてキー付けされます。これはセルフDMのホームチャンネル（典型的なセットアップ）では機能しますが、真に共有されるグループチャットには理想的ではありません。スレッド化はTelegram / Discord / Slack / Matrixをカバーしており（圧倒的に一般的なケース）、ほとんどのセットアップではこの問題に遭遇しません。

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

### 圧縮時の自動系譜 {#auto-lineage-on-compression}

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
# List recent sessions (default: last 20)
hermes sessions list

# Filter by platform
hermes sessions list --source telegram

# Show more sessions
hermes sessions list --limit 50
```

`--limit` で許可される数より多くのセッションが存在する場合、一覧の最後に `… more not shown (use --limit N to see more)` というフッターが表示されるため、上限で切られたページが完全な一覧と誤解されることはありません。

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

### セッションのエクスポート {#export-sessions}

`hermes sessions export` は、すべてのエクスポート形式を扱う単一の窓口で、`--format` で形式を選択します。

| 形式 | 出力 | 用途 |
|--------|--------|------------|
| `jsonl`（デフォルト） | セッションごとに1つのJSONオブジェクト | バックアップ、機械的な往復変換 |
| `md` / `qmd` | セッションごとに1つのMarkdown/Quartoファイル + マニフェスト | 読みやすいアーカイブ、メモ |
| `html` | 自己完結した単一のページ（複数セッションの場合はサイドバー付き） | 共有、閲覧 |
| `trace` | Claude Code JSONL | HF Agent Trace Viewer、`--upload` |

加えて、プロンプトのみのビュー用に `--only user-prompts`（jsonlまたはmd）があります。

すべての形式は同じ選択オプションを共有します。1つのセッションには `--session-id`、一括処理には `prune`/`archive` の完全なフィルタセットを使用します。`--older-than` / `--newer-than` / `--before` / `--after`（`5h`/`2d`/`1w` のような期間、日数のみの数値、またはISOタイムスタンプ）、`--source`、`--title`、`--model`、`--provider`、`--cwd`、`--min/--max-messages`、`--min/--max-tokens`、`--min/--max-cost`、`--min/--max-tool-calls`、`--user`、`--chat-id`、`--chat-type`、`--branch`、`--end-reason` です。`--dry-run` は書き込まずに一致するセットをプレビューします。`--redact` はどの形式でも、エクスポートされたコンテンツからシークレット（APIキー、トークン、資格情報）を除去します。共有する予定のものには推奨されます。注意: 一括フィルタは*終了済み*のセッションに一致します。フィルタなしの `export` は、アクティブなものを含めてすべてをダンプします。

#### JSONL（デフォルト）

```bash
# Export all sessions to a JSONL file
hermes sessions export backup.jsonl

# Export sessions from a specific platform
hermes sessions export telegram-history.jsonl --source telegram

# Export a single session
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4

# Point at a directory (existing, or ending in /) and the file is named for you:
# ~/exports/hermes_session_20250305_091523_a1b2c3d4.jsonl
hermes sessions export ~/exports/ --session-id 20250305_091523_a1b2c3d4

# Redact API keys/tokens/credentials from the exported content
hermes sessions export backup.jsonl --redact
```

エクスポートされたファイルには、完全なセッションメタデータと保存されたすべてのメッセージを含むJSONオブジェクトが1行に1つ含まれ、各メッセージには `active`/`compacted` フラグが付きます。これには、インプレース圧縮によってアーカイブされたターンや、巻き戻しや編集によって削除されたメッセージも含まれます。このファイルをインポートすると（ダッシュボードのセッションインポート）、それらの行はライブのモデルコンテキストとしてではなく、アーカイブされた履歴として復元されます。`/save json` のスナップショット（CLI、メッセージング、TUI、Desktop）も同じ行を保持します。バックアップには、セッションがこれまでに含んだすべてが入っているものとして扱ってください。会話を共有するには、表示用の形式（`--format md` または `html`。セッションが表示する履歴のみを保持します）を `--redact` とともにエクスポートしてください。各セッションのバックアップはメモリ上で構築されるため、保存された行数が `sessions.max_export_messages` を超えるセッションは拒否されます（[過大なトランスクリプトのガード](#oversized-transcript-guards)を参照）。

フィルタ付きのエクスポートには、一致するピン留めされたセッションとアーカイブされたセッションも含まれます。ピン留めはセッションを整理（prune）から保護するものであり、バックアップから除外するものではありません。明示的な `--session-id` を解決できない場合、エクスポートは0以外のステータスで終了し、出力ファイルは作成されません。

各レコードには、メッセージのタイムスタンプから導出された `timings` ブロックも含まれます。そのため、バグレポートに添付されたエクスポートを読む人は、手作業で再構成しなくても、1回の長いモデルの待ち時間と多数の小さなツールの往復とを区別できます。このブロックには、ID、ロール、件数、所要時間 — `wall_clock_ms`、`largest_gap_ms`、`role_counts`、`tool_calls_emitted`、メッセージごとの `intervals` — のみが含まれ、プロンプトのテキスト、ツールの引数や結果は決して含まれないため、`--redact` を経ても変わりません。Hermesはモデル/ツールのストップウォッチを永続化しないため、`complete` は常に `false` です。タイムスタンプ付きのメッセージがないセッションでは、`available` が `false` になり、`unavailable_reason` がその理由を示します。このブロックはエクスポートのたびに再構築され、インポート時には無視されます（サイズ制限にもカウントされません）。

#### HTML

`--format html` は、リモートの依存関係を持たない自己完結した単一のHTMLファイルを書き出します。スタイル付きのメッセージバブル、折りたたみ可能なツール出力、そして（複数セッションのエクスポートの場合は）セッションを切り替えるサイドバーを備えています。

```bash
# One session as a standalone HTML page
hermes sessions export --format html --session-id 20250305_091523_a1b2c3d4 transcript.html

# All Telegram sessions from the last week in one file, secrets redacted
hermes sessions export --format html --newer-than 1w --source telegram --redact archive.html
```

#### プロンプトのみ

`--only user-prompts` は、あなたが書いたプロンプトだけをエクスポートします。アシスタントの返信、ツール出力、システムコンテキストは含まれません。プロンプトライブラリの構築や、何を尋ねたかの振り返りに便利です。

```bash
# One JSONL record per prompt (session id, index, timestamp, text)
hermes sessions export prompts.jsonl --session-id 20250305_091523_a1b2c3d4 --only user-prompts

# Markdown, straight to stdout
hermes sessions export - --session-id 20250305_091523_a1b2c3d4 --only user-prompts --format md
```

`--format jsonl`（デフォルト）または `md` で機能し、一括エクスポートでも同じフィルタに従い、`--redact` と組み合わせられます。

#### トレース（HF Agent Trace Viewer）

`--format trace` はClaude Code JSONLを出力します。これはHugging Face Hubが[Agent Trace Viewer](https://huggingface.co/docs/hub/agent-traces)用に自動検出するトランスクリプトの形式です。ローカルに書き出すか、`--upload` を追加して自分のプライベートな `hermes-traces` データセットにプッシュします（`HF_TOKEN` を読み込みます）。

```bash
# Trace of the most recent session, to stdout
hermes sessions export --format trace

# One session to a local trace file
hermes sessions export --format trace --session-id 20250305_091523_a1b2c3d4 trace.jsonl

# Upload straight to your private HF traces dataset
hermes sessions export --format trace --session-id 20250305_091523_a1b2c3d4 --upload
```

トレースのエクスポートは、デフォルトでシークレットが除去されます（マシンの外に出ることを前提としているためです）。手動で確認した後に `--no-redact` でオプトアウトできます。`--upload` は `--public` を指定しない限りプライベートです。フィルタ付きの一括トレースエクスポートは、セッションごとに1つの `<id>.trace.jsonl` を書き出します。

#### Markdown / QMD

古いセッションを非表示にしたり削除したりする前に、読みやすいファイルベースのアーカイブが欲しい場合は、`--format md` または `--format qmd` を渡します。Markdown/QMDのエクスポートは、セッションごとに1つのファイルをディレクトリ（デフォルト: `~/.hermes/session-exports`）に書き出します。

```bash
# Export one session to Markdown
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4

# Export a compression lineage as one logical document
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4 --lineage logical

# Preview ended sessions older than 90 days without writing files
hermes sessions export --format md --older-than 90 --dry-run

# Export ended Telegram sessions older than 2 weeks to QMD files
hermes sessions export --format qmd --older-than 2w --source telegram

# Export long Claude sessions, secrets redacted
hermes sessions export --format md --model sonnet --min-messages 50 --redact

# Only after verification, export and delete one explicitly named session
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4 --delete-after-verified --yes
```

Markdown/QMDのエクスポートは、エクスポートされたセッションごとに1つの `.md` または `.qmd` ファイルと、ファイルパス、メッセージ数、系譜のID、SHA-256を含む `manifest.jsonl` を書き出します。一括エクスポートには少なくとも1つのフィルタが必要で、フィルタなしの一括エクスポートは拒否されます。`--delete-after-verified` は意図的に `--session-id` に限定されており、`--yes` が必要です。親セッションを削除するとそのデリゲート/サブエージェントのセッションも削除されるため、このモードでは何かを削除する前に、各デリゲートを別々のファイルにエクスポートして検証します。Markdown/QMDファイルには、インプレース圧縮によってアーカイブされたターンを含め、セッションが表示する完全な履歴が含まれます。削除は、削除を実行するのと同じデータベーストランザクション内で、その正確な表示用トランスクリプトとデリゲートのセットを再度比較します。間に追記、書き換え、巻き戻し、圧縮、デリゲートの変更があれば、削除は拒否されます。同じ表示履歴のルールが、`--format html`、`--only user-prompts`（MarkdownとJSONLのどちらの出力でも）、および `/save md|html` にも適用されます。セッション全体のJSON/JSONLエクスポート（`--only` なしの `hermes sessions export`）と `/save json`（CLI、メッセージング、TUI、Desktop）は、アーカイブされた行を含む保存されたすべての行のバックアップです（[セッションのエクスポート](#export-sessions)を参照）。`--redact` は、書き込み前にメッセージ内容とツール出力からシークレット（APIキー、トークン、資格情報）を除去します。共有する予定のエクスポートには推奨されます。

### セッションの削除

```bash
# Delete a specific session (with confirmation)
hermes sessions delete 20250305_091523_a1b2c3d4

# Delete without confirmation
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

実行中のチャットでまだ開いているセッションを削除しても、そのチャットは停止しません。次の保存時に、メモリ上の完全なトランスクリプトを使って同じIDでセッションが再作成されます。セッションを消したい場合は、先にチャットを閉じてください。

ターンが実行中または圧縮中のセッションの削除は、ライブのエージェントの下でトランスクリプトが失われるのを防ぐために拒否されます（終了コード1）。削除する前に、アクティブなターンまたは圧縮が完了するのを待ってください。

### セッションの名前変更

```bash
# Set or change a session's title
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"

# Multi-word titles don't need quotes in the CLI
hermes sessions rename 20250305_091523_a1b2c3d4 debugging auth flow
```

タイトルがすでに別のセッションで使用されている場合は、エラーが表示されます。

### ゲートウェイのクラッシュで失われたターンを破棄する

ターンの途中でゲートウェイが停止した場合、そのターンの結果は不明となり、決して再実行されません。確認するまで、そのターンはセッションをブロックします。対話型の `hermes chat --resume <id>` はそれを一覧表示し、`/discard <admission_id>` を受け付けます。スクリプトからの `hermes chat --resume <id> -q` は終了コード3で拒否し、何も送信しません。スクリプトからクリアするには、次のようにします。

```bash
# Acknowledge every unknown (lost) turn on the session, then continue it
hermes sessions discard 20250305_091523_a1b2c3d4 --yes
hermes chat --resume 20250305_091523_a1b2c3d4 -q "next question"

# Only one specific admission (repeatable); the session may also be named by title
hermes sessions discard "debugging auth flow" --admission a9b59ad3ee914d75967def2a33435c88 --yes
```

`--yes` なしではコマンドは確認を求め、stdinがターミナルでない場合は拒否します（終了コード2）。失われた入力はトランスクリプトに残ります。そのアクションの一部はすでに実行されている可能性があるため、再送信する前にその影響を確認してください。その後ろにキューされた入力は、破棄されると実行されます。

### セッションのピン留め

ピン留めは永続的な「保持」フラグを設定します。ピン留めされたセッションは `sessions.auto_archive` の古いセッションのスイープから除外され、常に一覧に表示されます。これはDesktopサイドバーのPinnedセクションが使用するのと同じフラグです。どちらからピン留めしても、両方に反映されます。

セッションのエクスポートを復元すると（ダッシュボードのインポート、またはプロファイルが取り残されたセッションを引き取る場合）、ピン留め、アーカイブ、非表示のフラグと、アーカイブが `sessions.auto_archive` のスイープによるものかどうかが保持されます。復元されたピン留めセッションは引き続き保持期間のクリーンアップから除外され、引き取られたBot Modeのチャットは非表示のままとなり、復元されたスイープによるアーカイブは再開すると引き続き戻ってきます。これらのフラグより前のエクスポートは、通常の、ピン留めされていない、表示されるセッションとして復元されます。

```bash
# Pin one or more sessions (unique ID prefixes work)
hermes sessions pin 20250305_091523_a1b2c3d4
hermes sessions pin 20250305 20250306

# Remove the pin
hermes sessions unpin 20250305_091523_a1b2c3d4

# List pinned sessions
hermes sessions pinned

# Machine-readable output, e.g. for a nightly backup of your pin set
hermes sessions pinned --json > pinned-sessions.json
```

### 古いセッションの整理（prune）

```bash
# Delete ended sessions inactive for 90 days (default)
hermes sessions prune

# Custom age threshold — bare numbers are days
hermes sessions prune --older-than 30

# Durations work too: 5h, 30m, 2d, 1w
hermes sessions prune --older-than 12h

# Delete only a specific time window (e.g. a batch of test sessions
# created in the last 5 hours)
hermes sessions prune --newer-than 5h

# Explicit window with absolute timestamps
hermes sessions prune --after "2026-07-05 09:00" --before "2026-07-05 14:30"

# Only prune sessions from a specific platform (all ages — any filter
# disables the implicit 90-day default)
hermes sessions prune --source telegram
hermes sessions prune --source cron --older-than 60   # add a time flag to narrow

# More filters — all AND together
hermes sessions prune --newer-than 5h --title "smoke test"   # title substring
hermes sessions prune --older-than 30 --max-messages 3        # tiny sessions
hermes sessions prune --cwd ~/scratch --end-reason done       # by cwd / end reason
hermes sessions prune --model gpt-5 --older-than 1w           # by model (substring)
hermes sessions prune --provider openrouter --older-than 60   # by billing provider
hermes sessions prune --branch feature/old-experiment         # by git branch
hermes sessions prune --user 12345678 --chat-type group       # by messaging origin
hermes sessions prune --max-tokens 500 --older-than 7         # by token usage
hermes sessions prune --max-cost 0.01 --max-tool-calls 0      # cheap, tool-less runs

# Preview what would be deleted, without deleting anything
hermes sessions prune --newer-than 5h --dry-run

# Skip confirmation
hermes sessions prune --older-than 30 --yes
```

時間の値（`--older-than`、`--newer-than`、`--before`、`--after`）には、期間（`5h`、`30m`、`2d`、`1w`）、日数のみの数値、またはISOタイムスタンプ（`2026-07-05`、`2026-07-05 14:30`）を指定できます。`--older-than`/`--before` は上限を、`--newer-than`/`--after` は下限を設定します。`--older-than`/`--newer-than` のペアは最終アクティビティ（ライブアクティビティ、最新メッセージ、セッション開始のうち最も新しいもの）を使用し、`--before`/`--after` は明示的にセッション開始時刻を使用します。どちらのペアも組み合わせて期間を指定できます。

属性フィルタ: `--source`（プラットフォーム、完全一致）、`--title` / `--model` / `--branch`（大文字小文字を区別しない部分文字列）、`--provider`（課金プロバイダー、完全一致）、`--end-reason`、`--user`、`--chat-id`、`--chat-type`（完全一致）、`--cwd`（パスのプレフィックス）、さらに数値の範囲指定 `--min/--max-messages`、`--min/--max-tokens`（入力+出力）、`--min/--max-cost`（USD。実績値、なければ見積もり値）、`--min/--max-tool-calls` があります。いずれかのフィルタを使用すると暗黙の90日デフォルトが無効になるため、`hermes sessions prune --source cron` や `--model gpt-4o` はすべての期間に一致します。絞り込むには時間フラグを追加してください。完全に引数なしの `hermes sessions prune` だけが90日のカットオフを維持します。`--yes` なしの実行では、確認を求める前に、一致件数と、一致する最も古いセッションと最も新しいセッションが表示されます。

アーカイブされたセッションはデフォルトでスキップされます。それらも削除するには `--include-archived` を渡してください。

:::info
整理（prune）は**終了済み**のセッション（明示的に終了された、または自動リセットされたセッション）のみを削除します。アクティブなセッションは決して整理されません。
圧縮によって複数のセッションに分割された会話は、1つの単位として整理されます。後続のセグメントが残っている間は、古いセグメントも残ります。
:::

### セッションの一括アーカイブ

何も削除せずにセッションを一覧から外したい場合、`hermes sessions archive` は `prune` と同じフィルタを受け取り、一致するセッションを削除する代わりにソフト非表示にします（Desktop/DashboardのUIで単一のセッションをアーカイブするのと同じアーカイブフラグを設定します。メッセージと検索はそのまま残ります）。

```bash
# Archive everything from the last 5 hours (e.g. 75 CI smoke-test sessions)
hermes sessions archive --newer-than 5h

# Archive by title substring, preview first
hermes sessions archive --title "dry run" --dry-run
hermes sessions archive --title "dry run" --yes
```

アーカイブはデフォルトで終了済みのセッションに一致します。Desktopやその他のゲートウェイが所有するローカルセッションは決して「終了」しないため、コマンドは一致したセッションのうち、まだ開いているためにスキップした数を表示します。それらもアーカイブするには `--include-open` を追加してください（`prune` は開いているセッションを決して削除しません）。

```bash
hermes sessions archive --title "Review Hermes" --include-open --yes
```

少なくとも1つのフィルタが必要です。引数なしの `hermes sessions archive` は、履歴全体のアーカイブを拒否します。圧縮された会話は、そのライブの先端（tip）を通じて1つの単位としてアーカイブされます。古い圧縮セグメントが自身の経過時間だけで一致することはないため、まだアクティブなチャットが履歴が長いという理由で非表示になることはありません。アーカイブされたセッションは `hermes sessions list` と `/resume` から非表示になりますが、データベースには残り、Desktop/Dashboardのセッション一覧からアーカイブを解除できます。

`sessions.auto_archive` のアイドルスイープによって非表示になったチャットは、再びライブになると自動的に戻ってきます。再開されたとき、または新しいアクティビティによって新しい継続セッションに圧縮されたときです。自分でアーカイブしたチャット（サイドバー、API、または `hermes sessions archive`）は、アーカイブを解除するまでアーカイブされたままです。

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

より詳細な分析 — トークン使用量、コスト見積もり、ツールの内訳、アクティビティパターン — については、[`hermes insights`](../reference/cli-commands.md#hermes-insights)を使用してください。

### プロファイル更新後にプロンプトを更新する {#refresh-prompts-after-a-profile-update}

継続中のセッションは、毎ターン同じシステムプロンプト、つまり最初のターンで構築したバイト列を送信します。このバイト列を固定しておくことで、プロバイダーは長い会話を通じてプロンプトキャッシュを再利用できます。その副作用として、`SOUL.md`、インストールまたは編集されたスキル、プロジェクトのコンテキストファイル（`AGENTS.md` など）への変更は、すでに実行中のセッションには反映されません。ゲートウェイを再起動しても変わりません。ゲートウェイは各セッションの保存されたプロンプトを復元するためです。これまでは反映させる唯一の方法が `/new` で、会話も失われていました。

プロファイルのプロンプトソースを更新した後、そのセッションに新しいソースを採用するよう依頼します。

```bash
# Every session in the active profile (add -p <profile> for another one)
hermes sessions refresh-prompts

# One session (id or unique prefix)
hermes sessions refresh-prompts --session SESSION_ID
```

このコマンドはプロファイルの `state.db` に更新リクエストを記録し、保存されたプロンプトを持つセッションの数を表示します。各セッションの次のターンで、Hermesは現在のソースからシステムプロンプトを再構築してからメッセージを送信します。会話履歴、セッションID、セッションのツール一覧はそのままです。再構築されたプロンプトは保存されるため、以降のすべてのターンはそれをバイト単位でそのまま再利用します。

- **コスト:** 更新された各セッションは、再構築するターンで1回のプロンプトキャッシュミスを払います。以降のターンは再びキャッシュされます。
- **再起動は不要:** リクエストは、ゲートウェイがメモリに保持しているエージェントでも、データベースから復元するエージェントでも、毎ターンの開始時に読み込まれます。再起動後も残ります。新しいファイルを配置した後に一度実行してください。更新スクリプトは、ゲートウェイを置き換える前後のどちらで呼び出しても構いません。
- **再構築されるもの:** システムプロンプト全体が通常のビルダーを通るため、メモリと `USER.md` のブロックもそのターンで最新の保存状態を反映します。ツール定義は変わりません。リクエスト後に開始されたセッションはすでに新しいプロンプトを持っているため、再度再構築することはありません。
- **コマンドを使わない場合**は何も変わりません。セッションは以前どおり保存されたプロンプトを再生し続けます。

### 取り残されたゲートウェイセッションの修復 {#repair-stranded-gateway-sessions}

再起動後にゲートウェイの会話が「過去に戻る」 — 最近のメッセージがなかったかのように数日前の話題を再開する — ことがあれば、ライブの会話がルーティングアイデンティティを失ったセッション行に取り残されている可能性があります（v0.21のセッション継続性の作業で修正された種類の破損です。現在のバージョンは構造的にこれを防ぎ、実行時に自己修復します）。

`hermes sessions repair-routing` は、ルーティングアイデンティティを持たないメッセージを含むセッション行を見つけ、それぞれを継続元の会話に再アタッチします。ただし、証拠が明確な場合に限ります。

```bash
# Report only — shows each orphan, the proposed adoption, and the evidence
hermes sessions repair-routing

# Perform the adoptions (stop the gateway first — a running gateway holds
# the old routing in memory and would write it back over the repair)
hermes sessions repair-routing --apply

# Widen/narrow the contiguity window (default 900 seconds)
hermes sessions repair-routing --max-gap-seconds 300
```

証拠のルール:

- **lineage** — 孤立行の `parent_session_id` が、同じプラットフォームのキー付きの行を指している（記録された事実であり、時間枠は適用されません）
- **contiguity** — 孤立行の開始からウィンドウ内に、同じプラットフォームのキー付きの行がちょうど1つだけ静かになった

曖昧なもの（前任候補が2つある、2つの孤立行が同じ前任を主張する）は理由とともに報告され、そのまま残されます。誤った引き取りは、ある会話を別のチャットに継ぎ合わせてしまうためです。置き換えられた行は `superseded_by_repair` として退役させられるため、再起動時の復旧でそれが復活することはありません。

修復は意図的に**自動ではありません**。チャットがその後に2つ目の履歴を積み上げている場合、どのスレッドを継続するかはあなたが判断することです。取り残された会話は、どちらの場合でも `/resume` とセッション検索で読めるままです。修復が変更するのはルーティングだけです。まずバックアップを取ってください（`cp ~/.hermes/state.db ~/.hermes/state.db.bak`）。

### 劣化した保存済みプロンプトの修復 {#repair-degraded-stored-prompts}

古いビルドでは、#122822の影響により、ゲートウェイのハイジーンやゲートウェイの `/compress` が、切り離されたメンテナンスエージェントの縮小されたツールセットのシステムプロンプトを、ライブセッションの上に永続化してしまうことがありました。PR #122825の根本修正をインストールした後、`hermes sessions repair-prompts` を使用してすでに劣化した行を見つけてください。

スキャンは保守的です。保存されたプロンプトに `skill_manage` のガイダンス行が欠けており、**かつ**永続化された `tools[]` のピンに `skill_manage`（常にそのガイダンスを出力します）が含まれている場合にのみ修復を提案します。読み取り可能なピンがない行や、`memory` のみのピン（正当な `toolsets: [memory]` のセットアップでもあり得ます）の行は**検証不能**として報告され、このスキャンでは変更されません。必要に応じて `SESSION_ID` を指定して明示的にクリアしてください。memoryのみの行も自然に修復可能になります。セッションが再開されると、その `tools[]` のピンが完全なツールサーフェスで再ピン留めされ、その後のスキャンでガイダンス行のない `skill_manage` が検出されてクリアされます。

```bash
# Report verified candidates and unverifiable rows; writes nothing
hermes sessions repair-prompts

# Clear verified degraded prompts after confirmation
hermes sessions repair-prompts --apply

# Machine-readable report
hermes sessions repair-prompts --json

# Non-interactive automation: apply and report the ids actually cleared
hermes sessions repair-prompts --apply --json

# Explicit destructive override for one session (id or unique prefix).
# This clears the stored prompt even when it is healthy.
hermes sessions repair-prompts SESSION_ID --apply
```

プロンプトのクリアは意図的にNULLを保存します。次のターンで再構築されて健全なバイト列が永続化され、修復されたセッションごとに想定どおり1回 `Stored system prompt ... is null; rebuilding from scratch` という警告が出ます。この警告はこの明示的な修復の結果であり、新たな破損の証拠ではありません。

修復は#122822の根本修正が入っている場合にのみ実行してください。そうでないと、後のメンテナンス圧縮によって行が再び劣化する可能性があります。

実行中のゲートウェイはキャッシュされた各セッションの古いプロンプトをメモリに保持しているため、修復した行を反映させるには `--apply` の後にゲートウェイを再起動してください（`hermes gateway restart`）。

### プロファイル間で交差した状態の修復 {#repair-state-crossed-between-profiles}

各プロファイルは1つの `state.db` を所有し、各ゲートウェイのセッションキーは会話を所有するプロファイルを示します（デフォルトプロファイルは `agent:main:…`、名前付きプロファイルは `agent:<name>:…`）。古いリリースでは、この2つが食い違ったままになることがありました。名前付きプロファイルの行がデフォルトのストアに書き込まれる、子セッションが別のプロファイルの行から継承する、ルーティング行が間違ったストアにコピーされる、Telegramのトピックや `/voice` 設定がボットのプロファイルなしで保存される、などです。現在のバージョンは新しい状態を正しい場所に置きます。`hermes sessions repair-profiles` は、すでに交差してしまったものを整理します。

```bash
# Report only — every store is scanned, nothing is written
hermes sessions repair-profiles

# Perform the repairs (stop the gateway first; a snapshot of every store is taken)
hermes sessions repair-profiles --apply

# Machine-readable report
hermes sessions repair-profiles --json
```

検出するものと行うこと:

| 検出内容 | 修復 |
|---|---|
| `profile_name` が行自身のセッションキーと食い違っている | キーからラベルを付け直す |
| 別のプロファイルのストアにある行 | （すべてのメッセージとともに）所有するプロファイルのストアへ移動する |
| `parent_session_id` が別のプロファイルの行を指している | リンクを切断する。行自身のアイデンティティは保持される |
| デフォルトのストア以外にあるルーティング行（マルチプレクシング時） | デフォルトのストアへ移動する。そこに既存の行があればそちらが優先される |
| 存在しなくなったプロファイルのルーティング行 / `sessions.json` エントリ | 削除する |
| ボットのプロファイルが欠けたTelegramトピックのバインディングとボイスモードのエントリ | そのチャットを保持するセッションからラベルを付け直す |

2つのケースは報告されますが、それが何であるかを指示されない限り修復されません。存在しないプロファイルにキー付けされた行（プロファイルを作成するか、`hermes profile migrate-identity <old> <new>` を実行してください）と、名前付きプロファイルのストア内にある `agent:main:…` の行です。後者は、以前そのプロファイル用にスタンドアロンで実行されていたゲートウェイの履歴（`--legacy-main rekey` でプロファイルの名前空間を付与します）か、スコープ付きの書き込みで紛れ込んだデフォルトプロファイルのチャット（`--legacy-main move` でデフォルトのストアに送ります）のいずれかです。行自体からはこの2つを区別できません。

`--apply` は、ゲートウェイがいずれかのストアを所有している間は拒否されます（ゲートウェイはルーティングインデックスをメモリに保持しており、それを書き戻してしまうためです）。また、再実行しても安全です。2回目の実行では何も見つかりません。


### ストアのWALジャーナルモードとDELETEジャーナルモードの変換

`database.journal_mode: delete` は、Hermesが作成するデータベースにのみ適用されます。すでにWALモードになっている既存の `state.db` は、オープン時に**決して**ライブでダウングレードされません。他のゲートウェイ、ダッシュボード、cronのプロセスがチェックポイントされていないWALのコミットを保持している可能性があり、その下でダウングレードするとそれらのコミットが破壊されるためです。そのため、HermesはWALを維持し、設定された `delete` が適用されなかったことを伝える `ERROR` をプロセスごとに1回ログに記録します。セルフサービスでの変換方法は次のとおりです。

```bash
# stop every process using the profile's store first (gateway, dashboard, CLIs, cron)
hermes sessions set-journal-mode delete     # WAL -> rollback journal
hermes sessions set-journal-mode wal        # back to WAL
hermes sessions set-journal-mode delete --db ~/.hermes/kanban.db   # another Hermes store
```

このコマンドは、いずれかのプロセスがまだファイルやその `-wal`/`-shm` サイドカーを保持している間は、各PIDとコマンドを示して拒否します。オープンしているプロセスが終わるのを待たずにモードを切り替え（途中で現れた保持プロセスがあれば、競合する代わりにSQLiteが拒否します）、ファイルヘッダーが新しいモードを報告していることを検証します。次のオープン時に設定されたモードが再適用されるため、設定が食い違っている場合は `database.journal_mode` を同じ値に設定するよう促します。保持プロセスのスキャンはローカルで行われる（Linux/macOSではオープンファイルテーブル、WindowsではRestart Manager）ため、ボリュームを共有する別のコンテナやVM内のプロセスは検出できません。スキャン自体が失敗した場合、ストアが静止していることを証明できないためコマンドは拒否します。`--force` は、すべてのHermesプロセスを自分で停止した後に、そのケースだけを免除します。スキャンが検出したプロセスは常に拒否されます。ストアがVM間のファイルシステム（virtiofs/9p）上にある場合、WALの共有メモリが黙って破損するため、WALの有効化も拒否されます。


## Claude CodeとCodex CLIからのセッションのインポート

別のエージェントCLIで会話を始めましたか？ それをHermesに取り込んで、ここで続けることができます。HermesはClaude Codeのセッションログ（`~/.claude/projects/`、またはClaude Codeの設定ディレクトリが移動されている場合は `$CLAUDE_CONFIG_DIR/projects/`）と、Codex CLIのロールアウト（`~/.codex/sessions/`、または `$CODEX_HOME/sessions/`）を読み込みます。外部のファイルは読み取られるだけで、決して変更されません。

```bash
# Interactive picker across both tools, newest first
hermes sessions import

# Limit to one tool, or point at a specific file
hermes sessions import --from claude
hermes sessions import --from codex ~/.codex/sessions/2026/08/15/rollout-....jsonl

# Import-and-resume in one step
hermes --resume @claude
hermes --resume @codex
```

`hermes sessions import` は、`Imported from Claude Code: <first user message>`（またはCodex CLI）というタイトルの新しいHermesセッションを作成し、IDと、そのまま貼り付けられる `hermes --resume <id>` コマンドを表示します。`--resume @claude` / `--resume @codex` は同じピッカーを表示し、インポートした会話にそのまま入ります。

**Hermes Desktop** には、コマンドパレットに同じインポーター（**Import session**）があります。アプリを実行しているコンピューターではなく、接続されたバックエンドが実行されているマシン上のログを一覧表示し、読み取り専用のプレビューを表示します。**Continue in Hermes** で会話を選択したプロファイルにコピーします。閲覧がセッションストアに書き込むことはなく、インポートがソースファイルに触れることもなく、同じログを2回インポートすると、別のコピーを作る代わりに既存のコピーが開かれます。

引き継がれるもの: 順序付けられたユーザー/アシスタントの会話で、ツールのアクティビティはアシスタントのターン内の短い `[ran tool: …]` メモに要約されます。システムプロンプト、注入されたコンテキスト、推論のトレース、生のツール出力は引き継がれません。インポートはクリーンなトランスクリプトであり、バイト単位の再生ではありません。


## セッション検索ツール {#session-search-tool}

エージェントには、SQLiteのFTS5エンジンを使用して過去のすべての会話を全文検索し、見つけた任意のセッションをスクロールできる組み込みの `session_search` ツールがあります。LLM呼び出しは行わず、要約を生成するのではなく、DBの実際のメッセージのビューを返します。

### 4つの呼び出し形式

ツールは、どの引数を設定したかから意図を推測します。`mode` パラメーターはありません。

**1. 探索 — `query` を渡す:**

```python
session_search(query="auth refactor", limit=3)
```

FTS5を実行し、ヒットをセッションの系譜ごとに重複排除して、上位N件のセッションを返します。探索はデフォルトで適応的な詳細度を使用します。最上位の結果には完全なコンテキストウィンドウとブックエンドが含まれ、下位の結果はコンパクトなままです。すべての結果を完全に展開するには `detail="full"` を渡してください。

各結果には以下が含まれます。

- `session_id`、`title`、`when`、`source`
- `snippet` — FTS5でハイライトされた一致箇所の抜粋
- `detail` — `full` または `compact`
- `bookend_start` / `bookend_end` — 完全な結果では最初/最後の3つのユーザー+アシスタントのメッセージ。コンパクトな結果では空のリスト
- `messages` — 完全な結果ではFTS5の一致箇所の前後±5メッセージ。コンパクトな結果ではフラグ付きのアンカーメッセージのみ
- `match_message_id`、`messages_before`、`messages_after`

最上位の結果は、目的 → 一致 → 解決をすぐに再構成します。別のコンパクトな結果のほうが有望に見える場合は、そのセッションIDとメッセージIDをスクロール形式で使用してください。実際のセッションDBでの典型的な所要時間は数十ミリ秒です。

**2. スクロール — `session_id` + `around_message_id` を渡す:**

```python
session_search(session_id="20260510_174648_805cc2", around_message_id=590803, window=10)
```

アンカーを中心とした±`window` メッセージのウィンドウを返します。FTS5もブックエンドもなく、スライスだけです。探索呼び出しの後、デフォルトの±5ウィンドウより多くのコンテキストが必要なときに使用します。

- **前方**にスクロールするには: `messages[-1].id` を `around_message_id` として渡します
- **後方**にスクロールするには: `messages[0].id` を `around_message_id` として渡します
- 境界のメッセージは、位置を把握するための目印として両方のウィンドウに表示されます
- `messages_before` または `messages_after` が `window` より小さい場合は、セッションの先頭または末尾にいます

典型的な所要時間: スクロール呼び出し1回あたり1〜2ms。

**3. 読み取り — アンカーなしで `session_id` を渡す:**

```python
session_search(session_id="20260510_174648_805cc2")
```

セッション全体を返すか、大きなセッションの場合は範囲を限定した先頭/末尾のビューを返します。この形式は `@session:<profile>/<id>` リンクの解決にも使用されます。

**4. ブラウズ — 引数なし:**

```python
session_search()
```

最近のセッションを時系列で返します（タイトル、プレビュー、タイムスタンプ）。ユーザーがトピックを挙げずに「何に取り組んでいたっけ」と尋ねたときに便利です。

### FTS5クエリ構文

キーワードモードは標準のFTS5クエリ構文をサポートします。

- 単純なキーワード: `docker deployment`（FTS5のデフォルトはAND）
- フレーズ: `"exact phrase"`
- ブール演算: `docker OR kubernetes`、`python NOT java`
- プレフィックス: `deploy*`

### オプションのパラメーター

- `sort` — FTS5のランク付けに加えて、`newest` または `oldest`。関連性のみの順序にするには省略します（デフォルト。探索的な想起に適しています）。「Xはどこまで進んだか」という質問には `newest` を、「Xはどう始まったか」という質問には `oldest` を使用します。
- `detail` — `adaptive`（デフォルト）は最上位の探索結果のみを完全に展開し、`full` はすべての探索結果を展開します。
- `role_filter` — 含めるロールのカンマ区切りリスト。探索のデフォルトは `user,assistant` です（ツール出力は通常ノイズになるため）。ツール出力を含めるには `user,assistant,tool` を（ツールの動作のデバッグ）、ツール出力のみを検索するには `tool` を渡します。

### 誰が何を検索できるか

`session_search` は、それが実行されるプロファイルのセッションストア全体を読み込みます。そのプロファイルの許可リストにあるすべてのプラットフォーム、すべてのユーザーからのすべてのチャットです。共有ボットや家族用ボットでは、それと話すことを許可された人なら誰でも、他の人のボットとの会話を見つけて読むようエージェントに依頼できます。Hermesは1つのプロファイル内でユーザーを分離しません（`SECURITY.md` §2.6を参照）。また、1つのホスト上のプロファイル同士も互いに分離されていません。デフォルトプロファイルはすべてのプロファイルのセッションを読み込み（下記参照）、ターミナルやコード実行のツールを持つエージェントは任意のプロファイルのファイルを読むことができます。プライベートな履歴が必要な人には、別のOSユーザーまたはコンテナの下で、それぞれ専用のインスタンスを用意してください（`SECURITY.md` §2.2）。

`profile=`（および `@session:<profile>/<id>` リンク）は、別のプロファイルのストアを読み込みます。1つのプロセスが複数のプロファイルを提供する場合（マルチプレクスされたゲートウェイ、複数のプロファイルがあるホスト上のダッシュボードや `hermes serve` バックエンド）、それができるのは**デフォルト**プロファイルだけです。ただし、自分自身のDesktop、TUI、CLI、エディター（ACP）のセッションは例外です。他のプロファイルから別のプロファイルを指定した呼び出しは、エラーとともに拒否されます。単一のプロファイルを単独で実行しているプロセスは、プロファイル間の読み込みを維持します。

### 使用されるタイミング

エージェントはセッション検索を自動的に使用するよう促されます。

> *「ユーザーが過去の会話の何かに言及した場合、または関連する以前のコンテキストが存在すると思われる場合は、繰り返しを求める前に session_search を使用してそれを思い出してください。」*

典型的なトリガー: 「前にこれをやった」「覚えてる？」「前回」「前に言ったように」、または現在のウィンドウにないプロジェクト/人物/概念への言及。

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
- 実行中のターンはそれを開始した送信者にキー付けされますが、`/stop` はそれにも届きます（下記参照）

`/stop` は「このチャットで実行中のものを止める」という意味です。まず呼び出し元自身のセッションキーを試し、次にこのチャット内の任意のライブのターン（呼び出し元自身のスレッド内の他の参加者の実行を含む）を対象にします。これは認可によって制限され、別のルーム、ワークスペース、プロファイルには決して及びません。そのため、アイドル状態のAliceの `/stop` で、彼女がいるルームでBob（またはボット）が開始したターンを終了できます。スレッドの*内側*から送信された `/stop` はより範囲が狭くなります。そのスレッドに属する実行と、スレッドのスロットを持たないルーム全体の実行（ローリングDMの形）には届きますが、同じチャンネルの別のスレッドや、他の参加者の送信者ごとのトップレベルの実行には決して届きません。

代わりに1つの共有された「ルームの頭脳」が欲しい場合は、次のように設定します。

```yaml
group_sessions_per_user: false
```

これにより、グループ/チャンネルはルームごとに単一の共有セッションに戻り、共有された会話コンテキストは保持されますが、トークンコスト、中断状態、コンテキストの増加も共有されます。

### セッションの継続性 {#session-continuity}

ゲートウェイの会話は、非アクティブ期間の後や日の区切りでリセットされることはありません。明示的に新しい会話を始めるには `/new` または `/reset` を使用してください。コンテキストの圧縮は引き続き自動で行われます。コアは、レガシーの `session_reset` 設定、リセットポリシーのオーバーライド、リセットタイマーの環境変数を無視します。設定でまだ `session_reset.mode` が `idle`、`daily`、`both` になっている場合、ゲートウェイの起動時と `hermes doctor` で警告が出ます。時間ベースのリセットを維持するには、同じブロックをそのまま読み込むカタログプラグインをインストールしてください。`hermes plugins install hermes-session-reset-policy`。キャッシュされたエージェントは、永続的な会話を置き換えることなく、リソースを回収するために解放されることがあります。再起動時の復旧における鮮度の制限は自動的な継続を制限するものであり、メッセージを送信したときに読み込まれる履歴を制限するものではありません。

**`/new` は、それを実行したビューだけを移動します。** 1つのセッションは、同時に複数の場所で開くことができます。Desktopのウィンドウ、TUI、`hermes chat --resume` のターミナル、ACPエディター、Telegramのトピックなどで、すべてがゲートウェイ上の同じライブの会話にアタッチされています。そのうちの1つで `/new`（または `/reset`）を実行すると、新しいセッションが作成され、*その*ウィンドウまたはチャットのルートがそこに再バインドされます。他のビューアーは元のセッションに留まり、その履歴は変更されず、そこですでに実行中のターンは実行を続けます。共有された会話を終了したりクリアしたりするのは、別の明示的な操作です（Stopしてから `/new`、または削除。削除はターンがライブの間は拒否されます）。

### セッションのハイジーン: それでも `/new` を実行すべき理由

ゲートウェイの会話は自然に期限切れになることがないため、1つのセッションを何週間も続けてしまいがちです。それでも動作しますが、学習ループを静かに無効にし、コストを膨らませます。

- **メモリが効果を発揮するのは境界においてのみです。** `MEMORY.md` / `USER.md` はセッション開始時に注入され、`session_search` はコンテキストから外れたものを思い出すために存在します。終わりのないセッションではすべてがまだコンテキスト*内*にあるため、エージェントにはメモリを参照する理由がなく、「自己学習」の仕組みはほとんど動きません。メモリの抽出（リセット前の保存）も、セッションが実際に終了したときにしか行われません。
- **コストは履歴とともに増えます。** 圧縮によって長いセッションも機能し続けますが、すべてのターンが依然として大きな（圧縮された）プレフィックスを運びます。抽出されたメモリを持つ新しいセッションは、ほぼ常に1か月前のスレッドよりも安価です。

実践的なルール: タスクやトピックが終わったらセッションを終了してください。自然な区切りで `/new`（任意で名前付き。例: `/new payments-refactor`）を実行します。毎日でもプロジェクトごとでも構いません。作業で永続的な好みや手順が明らかになった場合は、リセット前にエージェントに「残しておく価値のあることを覚えておいて」と頼んでください。期限切れになるセッションからメモリとスキルは自動的に保存されますが、明示的に促すと効果があります。マシンやゲートウェイの再起動は境界では**ありません**。同じセッションが再開されます。

境界をまたいで何が引き継がれるかについては、[メモリ](features/memory.md)を参照してください。


### クラッシュや再起動後の継続性

ゲートウェイのチャットは、明示的に `/new`（または `/reset`）を実行するまで、**1つの継続的なセッション**であるように設計されています。成長に応じて繰り返し圧縮されます。これは、ゲートウェイのクラッシュ、再起動、アップデートをまたいでも維持されます。

- セッションのアイデンティティ（ルーティングキー、チャット、オリジン）は、すべての作成パス（`/new`、最初のメッセージ、`/branch` の子）において、セッション行の作成時に**アトミックに**書き込まれます。その書き込みが失敗した場合でも、次のターンのルーティング更新が自動的に行を修復します。
- 再起動後、ゲートウェイは各チャットを、最も新しい**実際のアクティビティ**を持つセッションに再解決します。古く陳腐化した行が、実際に行っていた会話に勝つことは決してありません。
- 復旧は **`/new` の境界を尊重します**。チャットの最新のイベントが意図的なリセットである場合、復旧はリセットの向こう側に手を伸ばして古いセッションを復活させるのではなく、新しく開始します。経過時間だけで永続的な会話の復旧が妨げられることはありません。


## ストレージの場所

| 内容 | パス | 説明 |
|------|------|-------------|
| SQLiteデータベース | `~/.hermes/state.db` | FTS5を備えたすべてのセッションメタデータ + メッセージ |
| ゲートウェイのメッセージ | `~/.hermes/state.db` | SQLite — すべてのセッションメッセージの正規ストア |
| ゲートウェイのルーティングインデックス | `~/.hermes/state.db` 内の `gateway_routing` テーブル | セッションキーをアクティブなセッションIDにマッピング（オリジンのメタデータ、期限切れフラグ） |
| レガシーのルーティングミラー | `~/.hermes/sessions/sessions.json` | ルーティングインデックスの後方互換ミラー。`gateway.write_sessions_json: true`（デフォルト）のときに書き込まれます |

SQLiteデータベースは、並行リーダーと単一のライターのためにWALモードを使用しており、ゲートウェイのマルチプラットフォームアーキテクチャに適しています。

:::warning `sessions.json` はセッション一覧ではありません
ゲートウェイのルーティングインデックスは `state.db` 内の `gateway_routing` テーブルにあります。`~/.hermes/sessions/sessions.json` はその**レガシーミラー**で、後方互換性のために維持されています（`gateway.write_sessions_json: false` で無効化できます）。これはメッセージングのセッションキー（`agent:main:<platform>:...`）をアクティブなセッションIDにマッピングします。
ゲートウェイ/メッセージングのエントリしか含まれないため、メッセージングプラットフォームを実行している場合は、それらだけが表示されます（例: `agent:main:whatsapp:dm:...`）。

これは**想定どおり**であり、CLIセッションが失われていることを意味するもの**ではありません**。`hermes sessions list`、`/sessions`、ダッシュボードはすべて、**すべての**セッション（CLI、TUI、ゲートウェイ）を保持する `state.db` を読み込みます。`~/.hermes/sessions/saved/*.json` の下にある `/save` のスナップショットは、便宜上のエクスポートであり、インデックスではありません。

CLIセッションが本当に `hermes sessions list` に表示されない場合、原因は `state.db` がそれらを受け取っていないことです。`hermes sessions repair` を実行し、CLI起動時の `⚠ Session store unavailable` という警告に注意してください。これは、その実行でSQLiteへの永続化が失敗したことを意味します。
:::

:::note レガシーのJSONLトランスクリプト
state.dbが正規になる前に作成されたセッションには、`~/.hermes/sessions/` に `*.jsonl` ファイルが残っている場合があります。これらはHermesによって書き込まれることも読み込まれることもなくなりました。対応するセッションがstate.dbに存在することを確認した後、安全に削除できます。
:::

### データベーススキーマ

`state.db` の主要なテーブル:

- **sessions** — セッションメタデータ（id、source、user_id、model、title、タイムスタンプ、トークン数）。タイトルには一意のインデックスがあります（NULLタイトルは許可され、非NULLのみ一意である必要があります）。
- **messages** — 完全なメッセージ履歴（role、content、tool_calls、tool_name、token_count）
- **messages_fts** — メッセージコンテンツ全体の全文検索のためのFTS5仮想テーブル

## セッションの期限切れとクリーンアップ

### 自動クリーンアップ {#automatic-cleanup}

- ゲートウェイの会話は非アクティブ期間をまたいで維持されます。明示的な境界には `/new` または `/reset` を使用してください
- リセット前に、エージェントは期限切れになるセッションからメモリとスキルを保存します
- 自動整理（**デフォルトはオフ**。オプトイン）: `sessions.auto_prune` が `true` の場合、`sessions.retention_days`（デフォルト90）の間活動のない終了済みセッションが、CLI/ゲートウェイ/cronの起動時に整理されます
- **ディスク容量不足時のプロンプト。** `config.yaml` に `sessions.auto_prune` キーがない間、Hermesはセッション開始時に `state.db` を保持するディスクをチェックします（ファイルシステム呼び出し2回で、測定可能なコストはありません）。そのディスクの空きが少ない場合（空きが2 GB未満、または残りが20 GB未満で空きが10%未満）、一度だけ尋ねます。クラシックCLIは起動時に尋ねます（最初の容量不足時の起動ではバックグラウンドで測定するだけで、質問は次回の起動から表示されるため、起動がデータベースを待つことはありません）。TUIとDesktopは通知を表示し、ゲートウェイはホームチャンネルに質問を投稿します。プロンプトは、整理によってどれだけの履歴が削除されるかを見積もり（保存されたバイト数で重み付けされるため、多数の小さな古いメッセージで過大に見積もられることはありません）、整理してもディスクの問題が*解決しない*場合（`retention_days` より古いセッションがない、または履歴がディスク全体のごく一部にすぎない）ははっきりとそう伝えます。削除された履歴は `state.db` 内の領域を解放します。下記の自動 `VACUUM` は1回の整理の後にはめったにトリガーされないため、ディスクに領域を返すには `hermes sessions optimize` を実行してください。`/autoprune on`、`/autoprune off`（すべて保持。二度と尋ねられません）、または `/autoprune later`（1週間後にリマインド）で回答します。`/autoprune` のみで現在の状態を表示します
- `sessions.retention_days` は `>= 0` の整数の日数である必要があります。負の値（または値がない場合）は拒否されます。起動時のメンテナンスは、許容範囲を示す警告をログに記録し、未来のカットオフを「すべて」として扱う代わりにスイープをスキップします。整理を無効にするスイッチは `sessions.auto_prune: false` です
- 実際に行が削除された整理の後、**両方**の条件を満たす場合にのみ、`state.db` は `VACUUM` されてディスク領域が回収されます。前回成功した `VACUUM` から少なくとも `sessions.min_vacuum_interval_days`（デフォルト30）が経過していること、**かつ**ファイルのページの25%超が回収可能であること（`PRAGMA freelist_count / page_count`）です。密なデータベースが数MBを回収するために完全な書き直しのコストを払うことはありません（SQLiteは通常のDELETEではファイルを縮小しません）
- 整理は `sessions.min_interval_hours`（デフォルト24）ごとに最大1回実行されます。最終実行のタイムスタンプは `state.db` 自体の内部で追跡されるため、同じ `HERMES_HOME` 内のすべてのHermesプロセスで共有されます

整理しないと、`state.db` は際限なく大きくなります。ゲートウェイ + cronのインストールでは、数週間で数GBのファイルになったという報告があります。上限を設けるには、`~/.hermes/config.yaml` で有効にします（または `/autoprune on` と回答します）。

```yaml
sessions:
  auto_prune: true          # default is false — true deletes ended sessions past retention_days
  retention_days: 90        # keep ended sessions active within this window
  vacuum_after_prune: true  # reclaim disk space after a pruning sweep
  min_vacuum_interval_days: 30 # don't rewrite the DB more often than this
  min_interval_hours: 24    # don't re-run the sweep more often than this
```

これらのキーのいずれかを明示的に設定しているインストールは、その値を維持します。未設定のキーのみがデフォルトに従います。

削除されるのは**終了済み**のセッションだけです。アクティブなセッションは、経過時間に関係なく決して自動整理されません。終了済みのセッションは最終アクティビティ（ライブアクティビティ、最新メッセージ、セッション開始のうち最も新しいもの）から経過時間が計算されるため、最近使用された長期間の会話が、保持期間より前に始まったというだけで削除されることはありません。圧縮によって複数のセッションに分割された会話も同様です。後続のセグメントが残っている間は古いセグメントも残り、会話全体が条件を満たした時点でまとめて整理されます。

**自動化による古い開いたままのセッション。** 一部の生成元 — cronジョブ、kanbanワーカー、サブエージェント、ワンショットCLIの実行 — は、セッションを終了済みとしてマークしないまま終了することがあり、整理は*終了済み*の行しか削除しません。それらが永遠に蓄積しないように、各自動整理のパスでは、状態を所有するソース（`cli`、`cron`、`kanban`、`acp`、`api_server`、`subagent`、`tool`、および `hermes sessions recover` が孤立したメッセージのために合成する `recovered` プレースホルダー）からの開いたセッションのうち、最終アクティビティが `retention_days` より古いものも*クローズ*します（`end_reason: startup_orphan_reap`）。クローズは非破壊的です。セッションは再開可能なままで、行はクローズ時点から経過時間が計算されるため、さらに保持期間が丸ごと経過した後の*後続の*パスでのみ削除されます。メッセージングプラットフォームのセッション（Telegram、Discord、…）、TUI/Desktopのセッション、ピン留めされたセッション、ライブのターンや圧縮が進行中のセッションは、このスイープで決してクローズされません。

**誰も閉じなかったローカルセッション。** 最後のビューアーが閉じないまま消えたCLI、TUI、Desktopのセッション（強制終了されたターミナル、切断された接続、開いている間のゲートウェイの停止や再起動）は、`sessions.orphan_end_grace_s` 秒後（デフォルト `300`。`0` の場合は終了しません）にゲートウェイによって終了されます。`end_reason: ws_orphan_reap`（再起動後は `startup_orphan_reap`。アイドル状態のクラシックCLIは次に使用するときにしか再接続しないため、最終アクティビティから6時間以上経過してから）です。`orphan_end_grace_s` が未設定の間は、レガシーの `dashboard.ws_orphan_reap_grace_s: 0` でこれをオフにでき、`dashboard.startup_orphan_sweep: false` で再起動後の終了をオフにできます。まだ実行中のターンは実行を続け、それが落ち着いた時点でセッションが終了します。中断されたターンで一時停止しているセッションは、Discardするまで開いたままです。猶予期間内に再接続したビューアーはセッションを開いたままにします。終了は非破壊的です。セッションは再開可能なままで、次のメッセージで同じIDのまま再び開かれます。

### 過大なトランスクリプトのガード {#oversized-transcript-guards}

2つの制限により、暴走したトランスクリプトが一度にすべてメモリに読み込まれるのを防ぎます（どちらもデフォルトは `20000` メッセージ。`0` でガードを無効化します）。

```yaml
sessions:
  max_resume_messages: 20000   # interactive resume (CLI / TUI / Desktop)
  max_export_messages: 20000   # one-shot in-memory export of a single session
```

`max_export_messages` は、JSON/JSONLのバックアップ（`hermes sessions export`、`hermes console` 内の `sessions export`、およびCLI、メッセージング、TUI、Desktopでの `/save json`）にセッションごとに適用されます。バックアップにはすべての行が含まれるため、アーカイブされた行を含め、保存されたすべての行をカウントします。ライブの末尾が小さい、大幅に圧縮されたセッションでも、これを超えることがあります。ダッシュボードのSessionsページのExportアクションは、代わりに行をストリーミングするため、上限はありません。

`max_resume_messages` は、会話の履歴全体ではなく、**再開時に実際に読み込まれるもの**を制限します。

- 通常の対話型の再開（CLIの `--resume`、TUI）は、圧縮の系譜全体 — すべての圧縮されたセグメントとライブの先端 — を具体化するため、系譜全体で制限されます。
- Desktopのコールドレジュームは、REST経由でトランスクリプトをページングし、ライブの先端セグメントだけをメモリに保持するため、先端のみで制限されます。何度も圧縮された長期間のチャット（数十のセグメント、小さな先端の後ろに数万のアーカイブされた行）は、まさに圧縮が生み出すことを意図したものであり、通常どおり開きます。そのフッターのメッセージ数は、ライブのプロンプトではなく、保存された系譜を反映します。

再開が拒否されると、クライアントはエラーコード `4130` を、件数と、それが測定された範囲（`across its lineage` または `in its tip segment`）とともに受け取ります。`max_export_messages` によって拒否されたTUI/Desktopの `/save` は、エラーコード `4131` を返します。そのようなセッションでも `hermes sessions export` は引き続き機能します。そのJSON/JSONLバックアップでは、各セッションが `max_export_messages` 未満に収まっている必要があります。

### 手動クリーンアップ

```bash
# Prune sessions older than 90 days
hermes sessions prune

# Delete a specific session
hermes sessions delete <session_id>

# Export before pruning (backup)
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
自動整理は**デフォルトでオフ**で、`state.db` を保持するディスクの空きが少なくなると、Hermesが一度だけ尋ねます（上記の[自動クリーンアップ](#automatic-cleanup)を参照）。オンの場合、`sessions.retention_days`（デフォルト90）の間活動のない終了済みセッションが起動時に削除され、アクティブなセッションには決して触れません。セッション履歴は過去の会話にわたる `session_search` の想起を支えるため、選択はあなた次第です。`/autoprune on|off`、または `config.yaml` の `sessions.auto_prune` で設定します。自動整理がオフの場合でも、1回限りのクリーンアップには `hermes sessions prune` を引き続き使用できます（整理をまったく行わない場合に観測された失敗モード: 約1000セッションの384 MBの `state.db` がFTS5の挿入や `/resume` の一覧表示を遅くする）。
:::
