---
sidebar_position: 2
---

# プロファイル: 複数のエージェントを実行する

同じマシン上で複数の独立した Hermes エージェントを実行できます。それぞれが独自の設定、APIキー、メモリ、セッション、スキル、ゲートウェイ状態を持ちます。

## プロファイルとは何ですか？

プロファイルは独立した Hermes ホームディレクトリです。各プロファイルは、独自の `config.yaml`、`.env`、`SOUL.md`、メモリ、セッション、スキル、cronジョブ、状態データベースを含む専用のディレクトリを持ちます。Hermes が `~/.hermes/profiles/` 配下のディレクトリをプロファイルとして認識するのは、それらのアイデンティティファイル（`config.yaml`、`.env`、`SOUL.md`、`profile.yaml`、`auth.json`、`state.db`）のいずれかを含む場合だけです。ログや cron によって残された空のディレクトリは、`profile list`、ゲートウェイ、`-p` で無視されます。プロファイルを使うと、コーディングアシスタント、個人用ボット、リサーチエージェントなど、目的の異なる複数のエージェントを Hermes の状態を混在させることなく実行できます。

:::caution すべてのエージェントに専用のプロファイルを与える
2 つのエージェントプロセスを同じプロファイル（同じ Hermes ホーム）に向けないでください。どちらもメモリを自動的に書き込み、それぞれがセッション開始時に相手の書き込みをシステムプロンプトに読み込みます — そのため、1 つのホームに 2 つの書き手がいると互いの状態が積み重なり、やがてあなたが設定したものとはかけ離れたものになります。プロファイルはまさにこれを防ぐために存在します。メモリを共有する必要があるエージェントは、代わりに[外部メモリプロバイダー](./features/memory-providers.md)を使ってください。
:::

プロファイルを作成すると、自動的にそれ自体が1つのコマンドになります。`coder` というプロファイルを作成すれば、すぐに `coder chat`、`coder setup`、`coder gateway start` などが使えるようになります。

### プロファイル、エージェント、ボット {#profiles-agents-and-bots}

これらの用語は Hermes の異なる部分を表します。

- **プロファイル** は、アシスタントの設定とデータの永続的なホームです。
  会話や再起動をまたいで同じ状態を保持します。

- **エージェント** は、その設定と状態を使って動作している Hermes アシスタントです。
  「Hermes Agent」は製品名でもあります。

- **Bot Mode のボット** は、デスクトップの [Bot Mode](/docs/user-guide/bot-mode) の一覧に、アバターと
  永続的な Bot Chat を持つ名前付きエントリとして表示されるプロファイルです。
  同じプロファイルは CLI からも引き続きアクセスできます。すべての Bot はプロファイルですが、
  すべてのプロファイルが Bot というわけではありません: Bot Mode がプロファイルのメタデータに一覧での表示情報
  （タイトル、アバター、セクション、非表示状態）を保存し、正規の Bot Chat を固定すると、
  プロファイルは Bot になります。CLI、Docker、ゲートウェイからしか操作せず、一覧に一度も追加しない
  プロファイルは、通常のプロファイルのままです。

- **メッセージングボット** は、Telegram、Discord、Slack などのプラットフォーム上のアカウントで、
  ゲートウェイを通じて Hermes に接続されます。その
  [ボットトークン](#different-bot-tokens)がそのプラットフォームのアカウントを識別します。

- **サブエージェント** は、タスクのために [`delegate_task`](./features/delegation.md) によって
  起動される子アシスタントで、新しい会話を持ちます。別の会話であることは、別のプロファイルであることとは
  異なります。

## クイックスタート

```bash
hermes profile create coder       # creates profile + "coder" command alias
coder chat                        # start chatting (keys and model come from the current profile)
```

これだけです。`coder` は、独自の設定、メモリ、状態を持つ Hermes プロファイルになりました。

## プロファイルを作成する

:::tip
最も手早いセットアップ: 新しいプロファイル内で `hermes setup --portal` を実行すると、モデルとツールを一度に設定できます。[Nous Portal](/docs/integrations/nous-portal) を参照してください。
:::

### 空のプロファイル

```bash
hermes profile create mybot
```

バンドルされたスキルをシードした新しいプロファイルを作成します。現在のプロファイルのAPIキー（ボットトークンと許可リストを除く）とモデルを受け取り、デフォルトプロファイルのログインを共有します。これは Desktop のボットページの「Share keys & accounts」のデフォルトと同じです。ゲートウェイトークンについては `mybot setup` を実行してください。

クローンされたプロファイルが独自のモデル選択を持たない場合、作成時にはプロバイダーのモデルカタログを照会せずに、現在のプロファイルのモデルを継承します。選択されたプロバイダーとモデルはソースの選択に固定されたままですが、URL と認証情報の参照は新しいプロファイルの環境を使います。ソースのモデルレベルのキー、キーのポインター、コンテキスト長の設定はコピーされません。明示的なモデル選択は引き続き検証されます。

- `--isolated-auth` は、ログイン（`auth.json`）を共有する代わりに一度だけコピーします。名前付きプロファイルから作成されたプロファイルは、常にそのプロファイルのログインのコピーを受け取ります。そのサブスクリプション（使い捨ての OAuth）ログインはコピーできないため、それらのプロバイダーについては新しいプロファイルがデフォルトプロファイルのものを読み込み、`profile create` がその旨を表示します。
- `--no-credentials` は、キー、ログイン、モデルを一切コピーしません。後で `mybot setup` を実行してください。

現在のプロファイルのゲートウェイが実行中の場合、`profile create` はゲートウェイにプロファイルの作成を依頼するため、CLI、ダッシュボード、Desktop アプリはすべて同じ実行中のプロセスを通じて作成します。`HERMES_TUI_GATEWAY_URL` が設定されている場合、プロファイルはそのゲートウェイのマシン上に作成されます。到達できない場合はコマンドが失敗し、ローカルには何も作成されません。ゲートウェイが実行されていない場合は、従来どおりターミナル自身のプロセスでプロファイルが作成されます。

このプロファイルを kanban ワーカーとして使う予定がある場合（または kanban オーケストレーターに作業を振り分けさせたい場合）は、作成時に `--description "<role>"` を渡して、オーケストレーターがその得意分野を把握できるようにします。

```bash
hermes profile create researcher --description "Reads source code and external docs, writes findings."
```

説明は後から `hermes profile describe` で設定または自動生成することもできます — ルーティングモデルの全体については [Kanban ガイド](./features/kanban#auto-vs-manual-orchestration) を参照してください。

### 設定のみをクローン（`--clone`）

```bash
hermes profile create work --clone
```

現在のプロファイルの `config.yaml`、`.env`、`SOUL.md`、スキル、そしてキュレーションされたメモリファイル `memories/MEMORY.md` と `memories/USER.md` を新しいプロファイルにコピーします — メモリは `SOUL.md` と同様に、エージェントのアイデンティティの一部として扱われます。`config.yaml` が外部メモリプロバイダー（`memory.provider`）を選択している場合は、そのプロバイダー自身の設定も一緒にコピーされます — プロファイルホーム配下の `<provider>/` ディレクトリまたは `<provider>.json`（例: `hindsight/config.json`）です — そのため、クローンのメモリは黙ってオフになるのではなく利用可能になります。クローンされた `local_embedded` の hindsight 設定は、クローンに独自の hindsight の `profile`/`bank_id` を与えるまで、ソースの埋め込みデーモンとバンクを引き続き共有します（[#81815](https://github.com/NousResearch/hermes-agent/issues/81815)）。インストール済みのプラグインもコピーされます: `plugins/`（カタログ、Git、手動でインストールしたプラグインと、そのインストール記録）がコピーされるため、プラグインカタログからインストールしたメモリプロバイダーは、クローンでも同じリビジョンのまま動作し続け、カタログのインストール記録も保持されます。プラグインの Python 依存関係は共有環境にあり、再インストールされません。セッション、`state.db`、cron ジョブ、その他すべては空の状態で始まります。メモリも空にしたい場合は、`--clone` なしでプロファイルを作成するか、後で 2 つのファイルを削除してください。それらがない場合でも、エージェントが別のプロファイルのメモリにフォールバックすることはありません。異なるAPIキーを使うには `~/.hermes/profiles/work/.env` を、異なるパーソナリティにするには `~/.hermes/profiles/work/SOUL.md` を編集してください。

#### クローンのインポート済みエージェント設定を同期し続ける（`--sync-imports`） {#keep-a-clones-imported-agent-setups-synced---sync-imports}

ソースプロファイルで [`hermes import-agent`](/docs/user-guide/import-from-other-agents)（`agent-importers` プラグイン）を実行したことがある場合、その `import-sync.json` には、どの外部の Claude Code / Codex / Cursor のツリーをインポートしたかが記録されています。`--clone` はそのマニフェストを置いていくため、クローンはそれらのスキルとメモリの一度きりのコピーを受け取り、それで終わりです。マニフェストを引き継ぐには `--sync-imports` を追加します。

```bash
hermes profile create work --clone --sync-imports
hermes -p work import-agent --sync        # pulls changes from the same ~/.claude / ~/.codex / ~/.cursor
```

これは明示的なオプトインで一方向であり、クローンを**外部エージェントのツリーにのみ**結び付けます — ソースプロファイルに結び付けることは決してありません。両方のプロファイルは独立した島のままです: 後からソースの `config.yaml`、`SOUL.md`、スキルを編集しても、クローンに届くことはありません。`--clone-all` は完全なコピーの一部としてマニフェストをコピーします。

### すべてをクローン（`--clone-all`） {#clone-everything---clone-all}

```bash
hermes profile create backup --clone-all
```

**すべて** をコピーします。設定、APIキー、パーソナリティ、すべてのメモリ、スキル、プラグイン。完全な作業用スナップショットです。プロファイルごとの履歴は除外されます（セッション履歴、`state.db`、`backups/`、`state-snapshots/`、`checkpoints/`）— これらはソースプロファイルに属し、数十 GB に達することがあります。デフォルトプロファイルからクローンする場合は、ローカルモデルのランタイムツリー（`models/`、`runtimes/`、`node/` — ダウンロードされた重みと管理されたバイナリで、必要に応じて再取得されます）も、`hermes backup` がすでにそうしているようにスキップされます。**cron ジョブもクローンされません**: それらはソースプロファイルとその配信チャネルに結び付いたスケジュールされた作業であり、それを継承したクローンはすべてのジョブを 2 回実行してしまいます（2 つのゲートウェイ、同じジョブ ID）。新しいプロファイルは空の `cron/` で始まります。履歴と cron ジョブを含む完全なバックアップには、代わりに `hermes profile export` または `hermes backup` を使ってください。

:::note OAuth ログインはコピーではなく共有されます
Anthropic（Claude Pro/Max）、OpenAI Codex、xAI の OAuth ログインは **使い捨てのリフレッシュトークン** を使います — そのコピーは 2 つ目の認証情報ではなく、所有者が 2 人いる同じ認証情報であり、最初にリフレッシュしたプロファイルが他のすべてのコピーに対してそれを失効させます。そのため `--clone-all`（および `--isolated-auth` や名前付きプロファイルから作成されたログインのコピー）は、それらの OAuth の行をコピーから除外します。新しいプロファイルはルートの `~/.hermes/auth.json` からログインを読み込み続け、どのプロファイル内で行われたトークンのリフレッシュもルートに書き戻されるため、すべてのプロファイルがサインインしたままになります。静的な API キーは通常どおりコピーされます。プロファイルに独自の別個の OAuth ログインを与えるには、その中で `hermes -p <name> auth add <provider>` を実行します。そのログインは、ルートにそのプロバイダーのログインがまだない場合でも、プロファイル自身の `auth.json` に保存され、以後そのプロファイルはルートのログインの代わりにそれを使います。ルートの `auth.json` は変更されません。ログインを保存できない場合、コマンドは `Added` と表示する代わりにエラーで失敗します。
:::

### 特定のプロファイルからクローンする

```bash
hermes profile create work --clone-from coder
```

`--clone-from <source>` はソースプロファイルを直接選択し、設定/スキル/SOUL のクローンを意味します。そのソースプロファイルの完全なコピーが必要な場合は、`--clone-all` と組み合わせます。

```bash
hermes profile create work-backup --clone-from coder --clone-all
```

### メッセージングチャネルは決してクローンされません（オプトインは `--clone-channels`） {#messaging-channels-are-never-cloned---clone-channels-to-opt-in}

すべてのクローン — `--clone`、`--clone-from`、`--clone-all`、およびダッシュボード / Desktop / TUI の
「clone from profile」オプション — は、ソースを**メッセージングチャネルなしで**コピーします: ボットトークン
と許可リスト（`TELEGRAM_BOT_TOKEN`、`DISCORD_ALLOWED_USERS`、`WHATSAPP_ENABLED`、
`API_SERVER_KEY`、`WEBHOOK_SECRET` など）、`config.yaml` の `platforms:` / `telegram:` / `discord:` セクション、
`gateway.multiplex_profiles` / `profile_routes`、そして（`--clone-all` の場合）
ペアリングストア、WhatsApp セッション、その他のボットごとの状態です。プロバイダーとツールの API キー、モデル
ブロック、メモリ設定、スキル、`SOUL.md` は従来どおりコピーされます。コマンドは、どの
プラットフォームが置いていかれたかを表示します。

その理由は、ボットは 1 つのプロファイルにしか属せないからです: 同じトークンを持つ 2 つのスタンドアロンゲートウェイは
その long-poll を奪い合い、[多重化ゲートウェイ](/docs/user-guide/multi-profile-gateways)は
重複したアダプターを待機させます（そして `hermes gateway migrate --multiplex` はプラットフォームごとに 1 つの
重複認証情報のブロッカーを示して拒否します）。新しいプロファイル自身のボットは、
`hermes -p <name> setup` またはダッシュボードの Messaging ページで設定してください。

```bash
hermes profile create twin --clone --clone-channels   # keep the source's bots anyway
```

`--clone-channels` は、実行中の多重化ゲートウェイがすでにソースを処理している場合は拒否されます
（コピーはすぐに待機状態になってしまうため）— CLI、ダッシュボード、TUI のいずれからでも同様です — それ以外の場合は、
ソースと共有されることになったプラットフォームを示す警告を表示します。クローンのフラグなしで使うと
エラーになります。`hermes profile list` は、ボットの認証情報がデフォルトのものとバイト単位で同一である既存の
プロファイルについても同じ警告を表示するため、古いクローンは問題を起こす前に
表面化します。

**チャネル設定とみなされるもの** — この一覧は所有権に基づいており、*ソース*プロファイルの
プラグインスコープ（そのプライベートな `plugins/` アダプターを含む）で判定されます。

- アダプターが宣言するすべてのキー（トークン、アプリ/クライアント ID、許可リスト、全許可スイッチ、ホーム
  チャネル）と、その `<PLATFORM>_` プレフィックス配下のすべてのキー（歴史的なエイリアスを含む:
  `WECOM_*`、`SMS_*`/`TWILIO_*`、`QQ_*`、`HASS_*`、`EMAIL_*`）。
- ゲートウェイ全体のチャネルポリシー `GATEWAY_ALLOW_ALL_USERS` / `GATEWAY_ALLOWED_USERS` と、
  リレー登録のアイデンティティ `GATEWAY_RELAY_ID` / `GATEWAY_RELAY_SECRET` / `GATEWAY_RELAY_DELIVERY_KEY`。
- `--clone-all` の場合、ボットごとの状態ファイル**とディレクトリ**（`platforms/`、ペアリングの台帳、
  `google_chat_user_tokens/`、`<platform>_*`）。

メッセージングアダプターがチャネル以外の機能と共有する認証情報 — `HASS_TOKEN`/`HASS_URL`
（Home Assistant プラグインのツールでも使用。プラグインのアダプターはこれらを `shared_env_prefixes` として宣言します）、`TWILIO_*`（テレフォニースキルでも使用）、`EMAIL_*`（メール
送信スクリプトでも使用）— が除去されるのは、**ソースのゲートウェイがそのアダプターを実行する場合だけ**です
（プラットフォームが `config.yaml` で有効になっているか、認証情報のセットが揃っていて
明示的に無効化されていない場合）。`platforms.homeassistant.enabled: false` のソースは `HASS_TOKEN`
をツールのキーとして使っているため、クローンはそれを保持します。それらのプレフィックス配下の許可リストとポートは常に
チャネル専用であり、常に除去されます。

クローンは `profiles/` の隣にある隠しステージングディレクトリで構築され、除去の後に 1 回の
リネームで公開されるため、実行中の多重化プロセス（作成時に `profiles/` を再スキャンします）が
コピー途中のツリーでアダプターを起動することはありません。シンボリックリンクされたソースの `.env`/`config.yaml` は、
まずプライベートなコピーとして実体化されます — クローンがソースに書き込むことは決してありません。

:::tip Honcho メモリ + プロファイル
メモリプロバイダーは、それを選択しているプロファイルが作成またはリネームされたときにフックを受け取ります。Honcho（1.2.0 以降）はそれを使って、クローンされたプロファイルに同じユーザーワークスペース内の独自のAIピアを与え、リネーム時にもそのピアを維持します。インポートされたプロファイルの場合は `hermes honcho sync` を実行してください。各プロファイルは独自の観察結果とアイデンティティを構築します。詳細は [Honcho -- マルチエージェント / プロファイル](./features/memory-providers.md#honcho) を参照してください。
:::

## プロファイルを使用する

### コマンドエイリアス

すべてのプロファイルには、`~/.local/bin/<name>` にコマンドエイリアスが自動的に作成されます。

```bash
coder chat                    # chat with the coder agent
coder setup                   # configure coder's settings
coder gateway start           # start coder's gateway
coder doctor                  # check coder's health
coder skills list             # list coder's skills
coder config set model.default anthropic/claude-sonnet-4
```

このエイリアスはすべての hermes サブコマンドで機能します。内部的には単に `hermes -p <name>` です。

### `-p` フラグ

どのコマンドでも、プロファイルを明示的に指定することもできます。

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # works in any position
```

### 既定値を固定（`hermes profile use`）

```bash
hermes profile use coder
hermes chat                   # now targets coder
hermes tools                  # configures coder's tools
hermes profile use default    # switch back
```

既定値を設定し、素の `hermes` コマンドがそのプロファイルを対象とするようにします。`kubectl config use-context` のようなものです。

固定されたプロファイルのディレクトリが手作業で削除された場合、Hermes は代わりにデフォルトプロファイルで通常のコマンドを
実行することを拒否します。それらが誤ったプロファイルの状態を読み書きしてしまうためです。警告付きで実行できるのは
脱出手段だけです: `hermes profile list`、`hermes profile use default`、そして
`--data` や `--full` なしの `hermes uninstall`（この 2 つは引き続き拒否されます。対話型の
アンインストールメニューでは、確認プロンプトの後でデフォルトプロファイルの完全な消去を引き続き提示します）。

### 現在地を把握する

CLIは常にどのプロファイルがアクティブかを表示します。

- **プロンプト**: `❯` の代わりに `coder ❯`
- **バナー**: 起動時に `Profile: coder` を表示
- **`hermes profile`**: 現在のプロファイル名、パス、モデル、ゲートウェイの状態を表示

## プロファイル vs ワークスペース vs サンドボックス

プロファイルはワークスペースやサンドボックスとよく混同されますが、別物です。

- **プロファイル** は Hermes に独自の状態ディレクトリを与えます。`config.yaml`、`.env`、`SOUL.md`、セッション、メモリ、ログ、cronジョブ、ゲートウェイ状態です。
- **ワークスペース** または **作業ディレクトリ** は、ターミナルコマンドが開始される場所です。これは `terminal.cwd` によって別途制御されます。
- **サンドボックス** はファイルシステムへのアクセスを制限するものです。プロファイルはエージェントを **サンドボックス化しません**。

デフォルトの `local` ターミナルバックエンドでは、エージェントはあなたのユーザーアカウントと同じファイルシステムアクセスを持ちます。プロファイルは、プロファイルディレクトリ外のフォルダへのアクセスを止めるものではありません。

プロファイルを特定のプロジェクトフォルダで開始させたい場合は、そのプロファイルの `config.yaml` に明示的な絶対パスの `terminal.cwd` を設定してください。

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

ローカルバックエンドで `cwd: "."` を使用すると、「プロファイルディレクトリ」ではなく「Hermes が起動されたディレクトリ」を意味します。

また、次の点に注意してください。

- `SOUL.md` はモデルを導くことができますが、ワークスペースの境界を強制するものではありません。
- `SOUL.md` への変更は新しいセッションでクリーンに反映されます。既存のセッションは古いプロンプト状態を使用し続けている可能性があります。
- モデルに「どのディレクトリにいますか？」と尋ねることは、信頼できる隔離テストではありません。ツールに予測可能な開始ディレクトリが必要な場合は、`terminal.cwd` を明示的に設定してください。

## ゲートウェイを実行する

各プロファイルは、独自のボットトークンを持つ別個のプロセスとして独自のゲートウェイを実行します。

```bash
coder gateway start           # starts coder's gateway
assistant gateway start       # starts assistant's gateway (separate process)
```

### 異なるボットトークン {#different-bot-tokens}

各プロファイルには独自の `.env` ファイルがあります。それぞれに異なる Telegram/Discord/Slack のボットトークンを設定してください。

```bash
# Edit coder's tokens
nano ~/.hermes/profiles/coder/.env

# Edit assistant's tokens
nano ~/.hermes/profiles/assistant/.env
```

### 安全機構: トークンロック

2つのプロファイルが誤って同じボットトークンを使用した場合、2つ目のゲートウェイは、競合するプロファイル名を明示する明確なエラーでブロックされます。Telegram、Discord、Slack、WhatsApp、Signal でサポートされています。

### 永続サービス

```bash
coder gateway install         # creates hermes-gateway-coder systemd/launchd service
assistant gateway install     # creates hermes-gateway-assistant service
```

各プロファイルは独自のサービス名を持ちます。それらは独立して実行されます。

:::note 公式 Docker イメージ内では
プロファイルごとのゲートウェイは [s6-overlay](https://github.com/just-containers/s6-overlay)（コンテナ内の PID 1）によって監視されるため、`hermes profile create <name>` は自動的に `/run/hermes-service/gateway-<name>/` に s6 のサービススロットを登録します。`hermes -p <name> gateway start/stop/restart` は、素のプロセスを起動する代わりに `s6-svc` にディスパッチします — クラッシュは自動的に再起動され、`docker restart` は以前に実行されていたゲートウェイのセットを維持します。詳細は [プロファイルごとのゲートウェイの監視](./docker.md#per-profile-gateway-supervision) を参照してください。
:::

## プロファイルを設定する

各プロファイルは、それぞれ独自の以下を持ちます。

- **`config.yaml`** — モデル、プロバイダー、ツールセット、すべての設定
- **`.env`** — APIキー、ボットトークン
- **`SOUL.md`** — パーソナリティと指示

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

このプロファイルをデフォルトで特定のプロジェクトで動作させたい場合は、独自の `terminal.cwd` も設定してください。

```bash
coder config set terminal.cwd /absolute/path/to/project
```

### ダッシュボードから

[Web ダッシュボード](features/web-dashboard.md#managing-multiple-profiles)
はマシンレベルのサーフェスで、サイドバーのプロファイルスイッチャーを使って**任意の**プロファイルの設定、API
キー、スキル、MCP、モデルを管理できます — プロファイルごとのダッシュボードは
不要です。`coder dashboard` は、`coder` プロファイルが事前選択された状態でマシンの
ダッシュボードに移動します。ダッシュボードの Chat タブも
スイッチャーに従い、選択したプロファイルのホームで会話を起動します。

注意: ダッシュボードの Profiles ページの「Set as active」は、**今後の CLI/ゲートウェイの実行**の
固定デフォルトです（`hermes profile use` と同じ）— ダッシュボードからプロファイルを
編集するには、代わりにスイッチャーを使ってください。

## アップデート

`hermes update` はコードを一度だけ（共有で）取得し、新しいバンドルスキルを **すべての** プロファイルに自動的に同期します。

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

ユーザーが変更したスキルは決して上書きされません。

依存関係の準備では、共有環境が持つべきプラグインのセットを計算するために、すべてのプロファイルの `config.yaml` を
読み込みます。`config.yaml` を解析できない（または `plugins` / `memory` セクションの形が誤っている）
プロファイルがあると、インストール全体でそのステップが失敗します —
[依存関係の準備と保持](./features/plugins.md#dependency-preparation-and-preservation)を参照してください。

## プロファイルを管理する

```bash
hermes profile list           # show all profiles with status
hermes profile show coder     # detailed info for one profile
hermes profile rename coder dev-bot   # rename (updates alias; removes the old name's gateway service)
hermes profile migrate-identity coder dev-bot   # retry a rename's identity migration
hermes profile purge-identity dev-bot   # retry a delete's identity purge
hermes profile export coder   # pack into coder.tar.gz (shareable; keys stripped)
hermes profile import coder.tar.gz   # install an archive as a new profile
```

チャットでは、同じ 2 つが `/export` と `/import` として使えます — デスクトップアプリでは **⌘K → Export/Import profile…** です。[プロファイルを共有する](#sharing-a-profile)を参照してください。

### デフォルトプロファイルに名前を付ける

デフォルトプロファイルの内部 ID は常に `default` です — `~/.hermes` がインストールのルートであるため、
本当の意味でリネームすることはできません。代わりにリネームすると **表示名** が設定され、
UI のサーフェスでは素の ID の代わりにそれが表示されます。

```bash
hermes profile rename default Harumesu   # Unicode fine: 小助手
```

表示名は、`hermes profile list`/`show`、`/profile`
チャットコマンド、ダッシュボード、デスクトップアプリ（Bot Mode の
一覧を含む）に表示されます。これは表示専用です: `-p default`、サービス名、cron ジョブ、
その他のすべての参照は引き続き正規の `default` ID を使います。表示名は
`~/.hermes/profile.yaml` に `display_name` として保存されます。元に戻すにはその行を削除してください。
名前付きプロファイルも `display_name` を持てます（実際のリネームでも保持されます）が、
それらに対する `rename` は引き続きプロファイル自体をリネームします。

## プロファイルを削除する

```bash
hermes profile delete coder
```

これはゲートウェイを停止し、systemd/launchd サービスを削除し、コマンドエイリアスを削除し、すべてのプロファイルデータを削除します。確認のためにプロファイル名の入力を求められます。

確認をスキップするには `--yes` を使用します: `hermes profile delete coder --yes`

:::note
デフォルトプロファイル（`~/.hermes`）は削除できません。すべてを削除するには `hermes uninstall` を使用してください。
:::

## タブ補完

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

補完を永続化するには、この行を `~/.bashrc` または `~/.zshrc` に追加してください。`-p` の後のプロファイル名、プロファイルのサブコマンド、トップレベルのコマンドを補完します。

## 仕組み

プロファイルは `HERMES_HOME` 環境変数を使用します。`coder chat` を実行すると、ラッパースクリプトが hermes を起動する前に `HERMES_HOME=~/.hermes/profiles/coder` を設定します。コードベース内の119以上のファイルが `get_hermes_home()` を介してパスを解決するため、Hermes の状態は自動的にプロファイルのディレクトリにスコープされます。設定、セッション、メモリ、スキル、状態データベース、ゲートウェイPID、ログ、cronジョブがそうです。

これはターミナルの作業ディレクトリとは別物です。ツールの実行は `terminal.cwd`（ローカルバックエンドで `cwd: "."` の場合は起動ディレクトリ）から始まり、`HERMES_HOME` から自動的に始まるわけではありません。

ホストへのインストールでは、ツールのサブプロセスはデフォルトで実際の OS ユーザーの `HOME` を保持するため、
`~` 配下の既存の CLI 認証情報はプロファイルをまたいで動作し続けます。プロファイルのデータは
`HOME` を変更することではなく、`HERMES_HOME` によって分離されます。コンテナバックエンドでは引き続き
永続的なツールの状態に `{HERMES_HOME}/home` を使います。ホストのユーザーで、プロファイルごとに厳密な
ツール設定が必要な場合は、`terminal.home_mode: profile` でオプトインできます。

これは、混同しやすい 2 つのことを意味します。

- `HERMES_HOME` はプロファイルの境界です。Hermes の設定、`.env`、
  メモリ、セッション、スキル、ログ、cron ジョブ、ゲートウェイの状態、その他の Hermes
  データを制御します。
- `HOME` は、外部の CLI が想定するオペレーティングシステム/ユーザーのホームです。ホストへの
  インストールでは、Hermes はデフォルトでそれを実際のユーザーのホームのままにするため、
  `git`、`ssh`、`gh`、`az`、`npm`、Claude Code、Codex などのツールは、通常のシェルで使うのと同じ
  認証情報を見つけられます。

その代償として、ホストのプロファイルはデフォルトで通常のユーザーレベルの CLI の状態を共有します。
プロファイルごとに別々の CLI のアイデンティティが必要な場合は、そのプロファイルの `config.yaml` で `terminal.home_mode:
profile` を設定してください。このモードでは、Hermes はツールの
サブプロセスを `HOME={HERMES_HOME}/home` で起動します。その場合、プロファイル固有の `~/.ssh`、`~/.gitconfig`、`~/.config/gh`、クラウド CLI の認証、
Claude/Codex の認証、npm の状態などのファイルを、そのプロファイルのホーム内で初期化またはリンクする必要があります。

Hermes はサブプロセスに `HERMES_REAL_HOME` も公開するため、`home_mode: profile` が有効な場合でも、
スクリプトは実際のアカウントのホームを見つけられます。

デフォルトプロファイルは単に `~/.hermes` そのものです。移行は不要で、既存のインストールは同一に動作します。

## プロファイルを共有する {#sharing-a-profile}

あるマシンで構築したプロファイルは、別の場所に持っていけます — 自分のワークステーション、チームメイトのラップトップ、またはコミュニティです。2 つの経路があります。

**ファイルを送る。** `/export` はプロファイルを 1 つの `.tar.gz` にパックします — スキル、メモリ、ペルソナ、cron、プラグイン、設定、そして（デスクトップからの場合）テーマとレイアウトです。API キーは除去されます。受け取った人は `/import` を実行します。

マシン固有の PM の状態は移植できません。エクスポート、インポート、ディストリビューションの
インストールでは、プロファイルのルートにある `installs/`、`tools/`、`cache/` が除外されます。
バックアップと復元でも、デフォルトのホームと名前付きプロファイルに同じルールが適用されます。
古いアーカイブが、移行先マシンの PM の選択やツールを置き換えることはできません。
`plugins/example/facts.json` や `skills/example/tools/helper.py` のようなファイルは
ユーザーデータのままであり、保持されます。移行先では、環境をコピーするのではなく、
[PM](/docs/reference/package-management) を通じて依存関係をインストールしてください。

```bash
# In chat, run /export, hand over the file, and they run /import on it
hermes profile export coder
hermes profile import ./coder.tar.gz --name coder
```

**ディストリビューションを公開する。** プロファイルを **gitリポジトリ** としてパッケージ化すると、受け取った人は 1 つのコマンドでインストールでき、後からバージョン管理された更新を取得できます。SOUL、設定、スキル、cronジョブ、MCP接続が含まれます。認証情報、メモリ、セッションはマシンごとに保持されます。

```bash
# Install a whole agent from a git repo
hermes profile install github.com/you/research-bot --alias

# Update later when the author ships a new version (keeps your memories + .env)
hermes profile update research-bot
```

一度きりの受け渡しや移行にはエクスポートファイルを、継続的にリリースしていくエージェントにはディストリビューションを使ってください。両方について — 比較表、オーサリング、公開、更新セマンティクス、セキュリティモデル — は **[プロファイルディストリビューション: エージェント全体を共有する](./profile-distributions.md)** を参照してください。
