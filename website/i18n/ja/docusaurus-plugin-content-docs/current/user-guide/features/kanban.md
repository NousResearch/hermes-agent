---
sidebar_position: 12
title: "Kanban（マルチエージェントボード）"
description: "複数のHermesプロファイルを協調動作させるための、SQLiteベースの永続的なタスクボード"
---

# Kanban — マルチエージェントプロファイル協調

> **手順を追って学びたいですか？** [Kanbanチュートリアル](./kanban-tutorial)をお読みください。4つのユーザーストーリー（ソロ開発、フリート運用、リトライ付きロールパイプライン、サーキットブレーカー）を、それぞれのダッシュボードのスクリーンショット付きで解説しています。このページはリファレンスであり、チュートリアルはナラティブ（読み物）です。

Hermes Kanbanは、すべてのHermesプロファイル間で共有される永続的なタスクボードであり、壊れやすいインプロセスのサブエージェント群を使わずに、複数の名前付きエージェントで作業を協調できます。すべてのタスクは`~/.hermes/kanban.db`内の1行であり、すべての引き継ぎは誰でも読み書きできる1行であり、すべてのワーカーは独自のアイデンティティを持つ完全なOSプロセスです。

### イテレーション上限の前の完了チェックポイント

ディスパッチャーが所有するワーカーは、有限のイテレーション予算の約90%に達したところで、
ツールを呼び出せる呼び出しがまだ残っている状態の新しいツール結果に添付される形で、チェックポイントの通知を1回受け取ります。
より早いしきい値を選ぶには`agent.budget_warning_ratio`を使います。ごく小さな予算では、遅くとも
最後から2番目のイテレーションで警告します。1イテレーションの実行には、上限前のチェックポイントの余地はありません。
通知は次のリクエストの前にセッションのトランスクリプトに保存されます。ワーカーは、
タスクの契約を検証した後でのみ`kanban_complete`を呼び出すか、進捗の
コメントを永続化して作業を続けるべきです。コミットや差分だけで、タスクが自動的に完了することは決してありません。

ハードキャップ、ツールなしの最終サマリー、連続失敗のサーキットブレーカーは
変わりません: それでも予算を使い切ったワーカーは、引き続き上限付きのリトライの対象になります。
これは報告の機会であり、モデルが通知に従うことを保証するものではありません。
通常の会話と委譲された子は、Kanbanの自動
チェックポイントを継承しません。それらのイテレーション警告は引き続きオプトインです。

### 2つの接点: モデルはツール経由で、あなたはCLI経由で対話する

ボードには2つの入り口があり、どちらも同じ`~/.hermes/kanban.db`を背後に持ちます。

- **エージェントは専用の`kanban_*`ツールセットを通じてボードを操作します** — `kanban_show`、`kanban_list`、`kanban_complete`、`kanban_request_review`、`kanban_request_changes`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_attach`、`kanban_attach_url`、`kanban_attachments`、`kanban_create`、`kanban_link`、`kanban_unblock`。ディスパッチャーは各ワーカーを生成する際、これらのツールをすでにスキーマに含めた状態で起動します。オーケストレーター（orchestrator）プロファイルでも、`kanban`ツールセットを明示的に有効化できます。モデルは`hermes kanban`をシェル実行するのでは*なく*、直接ツールを呼び出してタスクを読み取り、ルーティングします。下記の[ワーカーはボードとどう対話するか](#how-workers-interact-with-the-board)を参照してください。
- **あなた（およびスクリプト、cron）は、CLI上の`hermes kanban …`、スラッシュコマンドとしての`/kanban …`、またはダッシュボードを通じてボードを操作します。** これらは人間と自動化のためのものであり、ツール呼び出しを行うモデルが背後にいない接点です。

どちらの接点も同じ`kanban_db`レイヤーを経由するため、読み取りは一貫したビューを参照でき、書き込みがずれることもありません。このページの残りの部分ではコピー&ペーストしやすいためCLIの例を示しますが、すべてのCLI動詞にはモデルが使うツール呼び出しの等価物があります。

これは、`delegate_task`では扱えないワークロードをカバーする形です。

- **リサーチのトリアージ** — 並列のリサーチャー＋アナリスト＋ライター、ヒューマンインザループ。
- **スケジュール運用** — 数週間にわたってジャーナルを構築する、繰り返しの日次ブリーフ。
- **デジタルツイン** — 時間をかけてメモリを蓄積する、永続的な名前付きアシスタント（`inbox-triage`、`ops-review`）。
- **エンジニアリングパイプライン** — 分解 → 並列worktreeでの実装 → レビュー → 反復 → PR。
- **フリートワーク** — N個の対象を管理する1人のスペシャリスト（50個のソーシャルアカウント、12個の監視対象サービス）。

8つの正準的な協調パターンは、下記の[協調パターン](#collaboration-patterns)にまとめられています。

## PRの完了契約

PRの作業は、作成時に`--completion-contract OWNER/REPO`（既存の作業の場合は正確な
`https://github.com/OWNER/REPO/pull/123`のURL）で宣言します。`kanban_create`は
同じ`completion_contract`を受け付けます。意図的にローカルな作業には`local-only`を使います。
既存のカードや宣言のないカードはそのデフォルトを維持します。散文中のURLはポリシーではありません。

公開した後、完了時に`metadata.published_pr`を渡します。最初に一致した
URLがカードに恒久的にバインドされ、リトライで緑色の兄弟PRに差し替えることはできません。
CLIの`show --json`と`kanban_show`は、永続化された契約を公開します。

共有の`complete_task`の境界は、ワーカーのツール、CLI、レビューの承認、
ダッシュボードからの完了をカバーします。これはクラシックなブランチ保護とアクティブなルールセットの
必須コンテキストを読み、正確なheadのチェック実行とレガシーのステータスをページ送りしながら取得し、その後
PRのhead/baseを再読み込みします。任意の失敗/スキップのテレメトリが、受け入れ済みの
必須チェックを拒否することはありません。欠落、保留中、失敗、キャンセル、タイムアウト、古い、スキップ、
中立の**必須**エビデンスでは、カードを完了できません。実行ゼロの
受け入れ、読めないポリシー、GitHub APIの失敗でも同様です。必須
チェックのないリポジトリには、local-onlyの契約が必要です。`gh`は、リポジトリのチェックとルールへの
読み取りアクセスで認証されている必要があります。このゲートがリモートへの書き込みを行うことはありません。
受け入れの読み取りは、**担当者プロファイルの**`gh`のログインとして実行されます — プロファイル自身の`.env`にある
`GH_TOKEN` / `GH_CONFIG_DIR`であり、カードを完了させるプロセスの周囲のログインでは決してありません。
マルチプロファイルのホスト（組織ごとに1つのGitHubアイデンティティ）では、プロファイルごとに`gh`にサインインします
（そのプロファイルの`.env`に`GH_CONFIG_DIR`）。トークンはプロファイルの`.env`（または設定済みのシークレットソース）に
置く必要があります: シェルやsystemdユニットで単にexportされた`GH_TOKEN`は、
`gh`の子プロセスから除去され、再び追加されることはありません。独自の`gh`のログインを持たない担当者プロファイル
— または、もう存在しないもの — は、`~/.config/gh`にフォールスルーするのではなく`not logged in`で拒否されます。
担当者のいないカードは引き続き周囲の
ログインを使います。リポジトリを見られないログインは、リトライ可能なインフラの失敗ではなく、
プロファイルとリポジトリを示した`classification=auth`で拒否されます。

拒否されてもアクティブなカードとワークスペースは保持されます。永続的な`pr_acceptance`イベントには、
PRのURL、SHA、必須コンテキスト、チェックのID/URL、分類、回復の
手順が保存されます。`last_failure_error`は次の手順を示します。失敗を修正するか、
インフラのチェックを再実行するか待ってから、完了を再試行してください。人間の対応が必要な場合は
`kanban_block`を使います。GitHubの一般的な`failure`では、テストが失敗したのか
アーティファクトのアップロードが失敗したのかを判別できません。保持されたURLを調べてください。明示的なインフラの
結論とAPIの失敗は別々に分類されます。追加のワーカーが起動されることはありません。

受領記録の永続化と終端の書き込みは、1つのSQLiteロックのもとで実行/ステータス/契約の所有権を
再確認します: 再取得されたワーカーが、新しい実行を完了させたり受け入れを付与したりすることはできません。
最後のGitHubの読み取りは完了時のスナップショットであり、分散
トランザクションや完了後の継続的な監視ではありません。これは単一ユーザーのライフサイクルの
ガードであり、任意の直接のデータベース書き込みに対するOSレベルの分離ではありません。GitHub Enterpriseは
対象外です。関連する公開/ライフサイクルの作業: #91230、#84254、#52311。ローカルでの
検証と公開だけでは、リモートでの受け入れにはなりません。

## Kanban と `delegate_task` の比較

両者は似ていますが、同じプリミティブではありません。

| | `delegate_task` | Kanban |
|---|---|---|
| 形状 | RPC呼び出し（fork → join） | 永続的なメッセージキュー＋ステートマシン |
| 親 | 子が返るまでブロック | `create`後はファイア・アンド・フォーゲット |
| 子のアイデンティティ | 匿名のサブエージェント | 永続メモリを持つ名前付きプロファイル |
| 再開可能性 | なし — 失敗＝失敗 | block → unblock → 再実行。クラッシュ → 再取得 |
| ヒューマンインザループ | 非対応 | 任意の時点でコメント/unblock |
| タスクあたりのエージェント数 | 1呼び出し＝1サブエージェント | タスクのライフサイクル全体でN個のエージェント（リトライ、レビュー、フォローアップ） |
| 監査証跡 | コンテキスト圧縮で失われる | SQLite内の永続的な行として永久に残る |
| 協調 | 階層的（呼び出し元 → 呼び出し先） | ピア型 — 任意のプロファイルが任意のタスクを読み書き |

**1文での区別:** `delegate_task`は関数呼び出しであり、Kanbanは作業キューであり、すべての引き継ぎが任意のプロファイル（または人間）が見て編集できる1行です。

:::caution サポートカードを、それがアンブロックするはずのカードにリンクしない
`t_parent`でブロックされ、欠けている部分のためにサポートカードを作成するワーカーは、`kanban_link(t_parent, t_support)`を**してはいけません**: リンクするとサポートカードはブロックされた親の*子*になるため、アンブロックするために存在する親の背後でゲートされ、どちらのカードも実行されなくなります。代わりに、サポートカードの本文で親のidを参照してください。`link`/`kanban_link`は、`ready`の子を降格させたときに`gated: true`を報告して`dependency_wait`イベントを記録します（`parents`付きの`kanban_create`も、新しいカードを駐機させるときに同じことを行います）。そのためデッドロックはボード上で可視化され、`hermes kanban unlink <parent> <child>`で解除できます。
:::

**`delegate_task`を使うべきとき:** 親エージェントが処理を続行する前に短い推論の答えを必要とし、人間が関与せず、結果が親のコンテキストに戻る場合。

**Kanbanを使うべきとき:** 作業がエージェントの境界をまたぐ、再起動を生き延びる必要がある、人間の入力が必要になる可能性がある、別のロールに引き取られる可能性がある、または事後に発見可能である必要がある場合。

両者は1つの実行の中でも共存します。Kanbanのタスクはプロファイルのゲートウェイのもとで分離された管理ワーカーとして実行され、その`delegate_task`の子は、そのゲートウェイによってワーカーの実行の下に予約されます: 各子は独自のセッション（カードのセッションのサブエージェントとして一覧表示）を持ち、`delegation.max_concurrent_children`と`delegation.max_spawn_depth`に数えられ、カードのターンの中で結果を返し、ワーカーとともに終了します。子の途中でゲートウェイがダウンした場合、その子は退役し、再開されることはありません。ゲートウェイより長く存続する必要がある作業や、独自のレビュアーが必要な作業は、引き続き子カード（`parents`付きの`kanban_create`）に属します。

## コアコンセプト

- **ボード（Board）** — 独自のSQLite DB、ワークスペースディレクトリ、ディスパッチャーループを持つ、スタンドアロンのタスクキュー。1つのインストールで複数のボードを持てます
  （例: プロジェクト、リポジトリ、ドメインごとに1つ）。下記の[ボード（マルチプロジェクト）](#boards-multi-project)を
  参照してください。単一プロジェクトのユーザーは`default`ボードのままで、このドキュメントのセクション以外で
  「ボード」という言葉を目にすることはありません。
- **タスク（Task）** — タイトル、任意の本文、1人の担当者（プロファイル名）、ステータス（`triage | todo | ready | running | blocked | review | done | archived`）、任意のテナント名前空間、任意の冪等性キー（リトライされる自動化のための重複排除）を持つ1行。
- **リンク（Link）** — 親 → 子の依存関係を記録する`task_links`の行。ディスパッチャーは、すべての親が`done`になると`todo → ready`に昇格させます。実行中の子へのリンクの追加は、すでにクレームされた作業をゲートできないため拒否されます。唯一の例外は、アクティブなワーカーが、ディスパッチャーから提供された実行の所有権のもとで、依存によるブロックの引き継ぎの直前に自分自身のカードをリンクする場合です。
- **コメント（Comment）** — エージェント間のプロトコル。エージェントと人間がコメントを追記します。ワーカーが（再）生成されると、コンテキストの一部としてコメントスレッド全体を読み取ります。
- **ワークスペース（Workspace）** — ワーカーが動作するディレクトリ。3種類あります。
  - `scratch`（デフォルト） — `~/.hermes/kanban/workspaces/<id>/`配下（非defaultボードでは`~/.hermes/kanban/boards/<slug>/workspaces/<id>/`）の新規一時ディレクトリ。**タスクの完了時に削除されます** — scratchは設計上、一時的です。`kanban_complete(artifacts=[...])`または`kanban_request_review(artifacts=[...])`で明示的に宣言されたファイルは、クリーンアップの前にタスクごとの永続的な添付ストレージにコピーされます（レビューの引き継ぎでは、後のレビュアーによる完了がscratchのワークスペースを削除するため、引き継ぎの時点でそれらを退避します）。レガシーの完了サマリー内にある既存の成果物のパスも同じ扱いを受けます。それ以外のscratchのファイルは削除されます。宣言されたscratchのアーティファクトが見つからない場合、タスクは実行中のまま保たれ、ワーカーはパスを修正して再試行できます。ワークスペース全体を残す必要がある場合は`worktree:`または`dir:<path>`を使ってください。インストールで初めてscratchのワークスペースが作成されると、ディスパッチャーは警告をログに出力し、タスクに`tip_scratch_workspace`イベントを発行します（`hermes kanban show <id>`で確認可能）。
  - `dir:<path>` — 既存の共有ディレクトリ（Obsidian vault、メール運用ディレクトリ、アカウントごとのフォルダ）。**絶対パスでなければなりません。** `dir:../tenants/foo/`のような相対パスはディスパッチ時に拒否されます。なぜなら、ディスパッチャーがたまたま置かれているCWDに対して解決されてしまい、曖昧であり、混乱した代理人（confused-deputy）の脱出ベクトルになるからです。それ以外の点では、パスは信頼されます — それはあなたのマシン、あなたのファイルシステムであり、ワーカーはあなたのuidで実行されます。これは信頼されたローカルユーザーという脅威モデルです。Kanbanは設計上シングルホストです。**完了時に保持されます。**
  - `worktree` — コーディングタスク用の`.worktrees/<id>/`配下のgit worktree。正確な対象パスを固定するには`worktree:<path>`を使います。ワーカー側の`git worktree add`で作成され、指定されていれば`--branch`を使います。**完了時に保持されます。**
- **ディスパッチャー（Dispatcher）** — N秒ごと（デフォルト60秒）に次を行う、長期間動作するループ: 古いクレームの再取得、クラッシュしたワーカーの再取得（PIDは消えているがTTLはまだ切れていない）、自分の実行が終わった後も残っているワーカーの回収（自身の`kanban_complete`/`kanban_block`の後もまだ生きているワーカーは、最後のターンを終える時間を与えるため、実行がクローズされてから2分後に終了させられます — PID*と*起動時のフィンガープリントで照合されるため、再利用されたPIDにシグナルが送られることはありません。`terminal_worker_reaped`イベントとして記録されます）、ready状態のタスクの昇格、アトミックなクレーム、割り当てられたプロファイルの生成。デフォルトでは**ゲートウェイ内で**実行されます（`kanban.dispatch_in_gateway: true`）。1つのディスパッチャーがティックごとにすべてのボードを掃引します。ワーカーは`HERMES_KANBAN_BOARD`を固定した状態で生成されるため、他のボードを見ることはできません。同じタスクで`kanban.failure_limit`回連続でスポーンに失敗すると（デフォルト: 2）、ディスパッチャーは最後のエラーを理由として自動的にブロックします — 存在しないプロファイル、マウントできないワークスペースなどのタスクでスラッシング（無駄な再試行）が起きるのを防ぎます。
- **テナント（Tenant）** — ボード*内*の任意の文字列名前空間。1つのスペシャリストフリートが、ワークスペースパスとメモリキープレフィックスによるデータ分離で、複数のビジネス（`--tenant business-a`）にサービスを提供できます。テナントはソフトなフィルターであり、ボードがハードな分離境界です。

## ボード（マルチプロジェクト） {#boards-multi-project}

ボードを使うと、無関係な作業の流れ（プロジェクト、リポジトリ、
ドメインごとに1つ）を、分離されたキューに分けられます。新規インストールでは
`default`という名前のボードがちょうど1つ存在します（後方互換のためDBは`~/.hermes/kanban.db`）。
作業の流れを1つしか必要としないユーザーは、ボードについて知る必要はまったくありません。この機能は
オプトインです。

ボードごとの分離は絶対的です。

- ボードごとに別々のSQLite DB（`~/.hermes/kanban/boards/<slug>/kanban.db`）。
- 別々の`workspaces/`ディレクトリと`logs/`ディレクトリ。
- タスク用に生成されたワーカーは、**そのボードの**タスク**のみ**を見ます —
  ディスパッチャーは子の環境に`HERMES_KANBAN_BOARD`を設定し、ワーカーがアクセスできる
  すべての`kanban_*`ツールがそれを読み取ります。
- ボードをまたいでタスクをリンクすることは許可されていません（スキーマをシンプルに保つため。
  どうしてもクロスプロジェクトの参照が必要な場合は、フリーテキストの言及を使い、
  idで手動検索してください）。

### CLIからのボード管理

```bash
# See what's on disk. Fresh installs show only "default".
hermes kanban boards list

# Create a new board.
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # optional: make it the active board

# Operate on a specific board without switching.
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# Change which board is "current" for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban boards show             # who's active right now?

# Rename the display name (the slug is immutable — it's the directory name).
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# Archive (default) — moves the board's dir to boards/_archived/<slug>-<ts>/.
# Recoverable by moving the dir back.
hermes kanban boards rm atm10-server

# Hard delete — `rm -rf` the board dir. No recovery.
hermes kanban boards rm atm10-server --delete
```

ボードの解決順（優先度が高い順）:

1. CLI呼び出しでの明示的な`--board <slug>`。
2. `HERMES_KANBAN_BOARD`環境変数（ワーカー生成時にディスパッチャーが設定するため、
   ワーカーは他のボードを見られません）。
3. `~/.hermes/kanban/current` — `hermes kanban boards switch`で永続化されたslug。
4. `default`。

slugは検証されます: 小文字の英数字＋ハイフン＋アンダースコア、1〜64
文字、英数字で始まる必要があります。大文字の入力は自動的に小文字化されます。
それ以外（スラッシュ、スペース、ドット、`..`）はCLIレイヤーで拒否され、
パストラバーサルのトリックでボードを命名できないようになっています。

### ダッシュボードからのボード管理

`hermes dashboard` → Kanbanタブでは、ボードが複数存在する（またはいずれかのボードにタスクがある）
とすぐに、上部にボードスイッチャーが表示されます。単一ボードのユーザーには
小さな`+ New board`ボタンだけが表示され、スイッチャーは必要になるまで非表示です。

- **ボードのドロップダウン** — アクティブなボードを選択します。選択内容は
  ブラウザの`localStorage`に保存されるため、リロードをまたいで保持され、
  開いたままにしているターミナルの足元からCLIの`current`ポインターを
  動かしてしまうことはありません。
- **+ New board** — slug、表示名、説明、アイコン、プロジェクトディレクトリ、そして（Projectが存在する場合は）
  **Project**のセレクターを尋ねるモーダルを開きます。新しいボードに自動切り替えするオプションあり。
- **Settings** — 現在のボードの表示名、説明、**プロジェクトディレクトリ**（`default_workdir`）を
  編集するモーダルを開きます。プロジェクトディレクトリは、新しいタスクすべてが
  継承するボードレベルのワークスペースのデフォルトです（gitリポジトリ → 保持されるworktree、通常のディレクトリ → 保持される
  ディレクトリ）。各タスクは作成時にこれを上書きできます。フィールドをクリアすると、
  新しいタスクは使い捨てのscratchワークスペースに戻ります — ただし
  Projectがバインドされている場合は、そのProjectのプライマリフォルダーが使われます。
  scratchに戻すには、先にプロジェクトのバインドを解除してください。
- **Project**（両方のモーダル） — ボードをHermesのProject
  （ボードの`project_id`）にバインドします。
  バインドされたボードで作成されたタスクはそのProjectを継承します。ディレクトリのフィールドが空のときに
  プロジェクトを選ぶと、プロジェクトディレクトリにもProjectのプライマリフォルダーが
  入ります。バインドされたプロジェクトは、ボードのドロップダウンの横に
  `Project: <name>`バッジとして表示されます。その`×`は、プロジェクトディレクトリには触れずにボードのバインドを解除し
  （`project_id: ""`を送信）、Settingsの`No binding`も保存時に同じことを行います。セレクターは
  少なくとも1つのProjectが存在する場合にのみ表示されます。
- **Archive** — `default`以外のボードでのみ表示されます。確認後、
  ボードのディレクトリを`boards/_archived/`に移動します。

すべてのダッシュボードAPIエンドポイントは、ボードのスコープ指定のために`?board=<slug>`を受け付けます。
イベント用WebSocketは接続時にボードに固定されます。UIで切り替えると、
新しいボードに対して新しいWSが開かれます。

### Desktopアプリでのボードの切り替え

Desktopアプリでは、ボードスイッチャーはKanbanページの上部のヘッダー行、
ページタイトルの横にあります: 現在のボードの名前とタスク数を表示する**Board**コントロールで、
シェブロンが付いています — ホバーすると「Switch board」と表示されます。クリックすると、
別のボードを選んだり、ボードの名前変更、設定、エクスポート、インポート、
作成、アーカイブを行ったりできます。Kanbanが分割タイルで開かれている場合、同じ
**Board**コントロールは、ボード自身のヘッダー行のタスク数の後にあります。
ダッシュボードと同様に、デスクトップは独自の
選択（ローカルに永続化）を保持し、CLIの`current`
ポインターを動かしません。


## ファイルの添付

タスクにはファイルの添付 — PDF、画像、ソースドキュメント — を付けられるため、
本文にパスを貼り付けて見つけてくれることを期待しなくても、ワーカーは必要なソース資料を
手にできます。

- **アップロード** — ダッシュボードのドロワーでタスクを開き、
  **Attachments**セクションの*Upload file*ボタンを使います（複数ファイルを一度に
  アップロードできます）。各アップロードの上限は25 MBです。
- **ストレージ** — ファイルは、デフォルトボードでは
  `<hermes-home>/kanban/attachments/<task_id>/`、名前付きボードでは
  `<hermes-home>/kanban/boards/<slug>/attachments/<task_id>/`に置かれます。
  独自の場所に固定するには`HERMES_KANBAN_ATTACHMENTS_ROOT`を設定します。
- **ワーカーに見えるもの** — ディスパッチャーがタスクをワーカーに渡すと、
  ワーカーのコンテキストには、各ファイルの名前と**絶対パス**を列挙した**Attachments**セクションが
  含まれます。ワーカーはファイル/ターミナルのツールに完全にアクセスできるため、
  添付を直接読みます（`read_file`、または`pdftotext`のようなシェル
  ツール）。
- **ダウンロード / 削除** — ドロワーには各添付が、ダウンロード
  リンクと削除（×）コントロールとともに一覧表示されます。添付を削除するとその
  メタデータの行が削除されます。ディスク上のファイルが削除されるのは、他の添付の
  行がもうそれを参照していない場合だけです（複数のタスクで共有されたファイルは、最後の
  参照が削除されるまで残ります）。CLIからは、`hermes kanban attach-rm
  ATTACHMENT_ID`で同じように添付を削除できます。

:::note リモートのターミナルバックエンド
添付のパスは、Kanbanワーカーのデフォルトである**ローカル**のターミナルバックエンドで
直接解決されます。リモートのバックエンド
（Docker、Modal）でワーカーを実行する場合は、ワーカーのコンテキスト内の絶対パスに到達できるよう、
ボードの`attachments/`ディレクトリをサンドボックスにマウントしてください。
:::


## クイックスタート

以下のコマンドは、**あなた**（人間）がボードをセットアップしてタスクを作成するものです。タスクが割り当てられると、ディスパッチャーは割り当てられたプロファイルをワーカーとして生成し、そこから先は**モデルがCLIコマンドではなく`kanban_*`ツール呼び出しを通じてタスクを進めます** — [ワーカーはボードとどう対話するか](#how-workers-interact-with-the-board)を参照してください。

```bash
# 1. Create the board (you)
hermes kanban init

# 2. Start the gateway (hosts the embedded dispatcher)
hermes gateway start

# 3. Create a task (you — or an orchestrator agent via kanban_create)
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. Watch activity live (you)
hermes kanban watch

# 5. See the board (you)
hermes kanban list
hermes kanban stats
```

ディスパッチャーが`t_abcd`を拾い上げて`researcher`プロファイルを生成すると、そのワーカーのモデルが最初に行うのは、`kanban_show()`を呼び出して自分のタスクを読み取ることです。`hermes kanban show t_abcd`を実行するわけではありません。

### ゲートウェイ組み込みディスパッチャー（デフォルト）

ディスパッチャーはゲートウェイプロセス内で実行されます。インストールするものは何もなく、
管理する別個のサービスもありません — ゲートウェイが起動していれば、ready状態のタスクは次のティックで
拾い上げられます（デフォルトで60秒）。

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # default
  dispatch_interval_seconds: 60    # default
  review_dispatch: true            # default: spawn the assigned profile with
                                   # the bundled sdlc-review skill. Set false
                                   # for human-only review boards.
  # dispatch_profiles: [sage]       # unset (key omitted): this home may claim
                                   # cards for any existing profile. Set to a
                                   # list (or comma-separated string) of profile
                                   # names to restrict which assignees this home
                                   # claims. Fail-closed: an empty list, `null`
                                   # or a bare `dispatch_profiles:` claims
                                   # nothing, and so does an unreadable config.
```

デバッグのために、実行時に`HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`で
設定フラグをオーバーライドできます。標準的なゲートウェイ監視が適用されます: `hermes gateway
start`を直接実行するか、ゲートウェイをsystemdユーザーユニットとして
配線します（ゲートウェイのドキュメントを参照）。ゲートウェイが動作していなければ、
`ready`のタスクは1つ起動するまでそのままにとどまります — `hermes kanban create`は
作成時にこれについて警告します。

#### ワーカーとsystemdのcgroup {#workers-and-systemd-cgroups}

ワーカーはディスパッチャーのティックより長く生き続ける送りっぱなしのプロセスなので、
ディスパッチャーがsystemdユニット内で動作している場合、ワーカーは
独自の一時的なスコープ（`systemd-run --user --scope --unit hermes-worker-kanban-<task>-run-<run>`）で起動され、
そのユニットの終了や再起動を乗り越えます。スコープを作成するには、
ディスパッチするユーザーのセッションバス（`/run/user/<uid>/bus`）が必要です。システムレベルの
インストールでは、`sudo loginctl enable-linger <user>`でユーザーにそれを与えてください。

- **ゲートウェイのディスパッチャー**（`dispatch_in_gateway: true`、systemd下のゲートウェイ）:
  スコープは必須です。バスに到達できない場合、起動は拒否され、
  カードに*インフラ*の失敗として記録されます — `metadata.infrastructure: true`付きの`spawn_failed`、
  `last_failure_error`内のlingerによる対処法、
  ゲートウェイのログの警告 — そしてカードは`ready`のままです。これは
  `consecutive_failures`を進め**ない**ため、ホストの一時的な問題でカードが単なる
  `blocked`として駐機されることはありません。リスポーンガードがバスが戻るまでリトライの間隔を空けます
  （`infrastructure_cooldown`、レート制限のクールダウンと同じウィンドウ）。
- **自分のユニットからの`hermes kanban dispatch`**（`Type=oneshot`のタイマー、
  シーケンサーのサービス）: バスに到達できればワーカーは同じスコープを得ます。
  到達できない場合もワーカーは起動されます — *あなたの*ユニットの
  cgroup内で — そしてディスパッチャーは、ユニットの終了でワーカーが強制終了されることを1回、目立つ形でログに出力します。
  デフォルトの`KillMode=control-group`を持つ`Type=oneshot`では、パスが終わってから1秒以内に
  すべてのワーカーが失われます。スコープを作成できるようlingerを有効にするか、
  そのユニットに`KillMode=process`を設定して、その終了で
  ディスパッチャー自身だけが強制終了されるようにしてください。

`hermes kanban daemon`を別個のプロセスとして実行することは**非推奨**です。
ゲートウェイを使ってください。本当にゲートウェイを実行できない場合（ヘッドレスホストの
ポリシーで長期間動作するサービスが禁止されているなど）、`--force`の脱出ハッチで
古いスタンドアロンデーモンを1リリースサイクルだけ生かしておけますが、同じ
`kanban.db`に対してゲートウェイ組み込みディスパッチャーとスタンドアロンデーモンの
両方を実行すると、クレームの競合を引き起こし、サポートされていません。

### ホームをまたいだ共有ボード {#shared-boards-across-homes}

1つの`kanban.db`を複数のHermesホーム（コンテナ、フリートのホスト）にマウントするとボードは
共有されますが、プロファイル名はホームローカルであり、すべてのホームには`default`という名前の
ルートプロファイルがあります — そのため`default`は構造的に衝突します。ディスパッチャーの起動の
ゲートは、*クレームする*ホームに対して`profile_exists(assignee)`を確認するため、
追加の設定がなければ、すべてのホームのディスパッチャーが`default`に割り当てられたカードを
クレーム可能とみなし、誤ったホームがそれをクレームして実行してしまう可能性があります。各
ホームに一意のプロファイル名を与えて共有ボードでは決して`default`にカードを割り当てないか、
ホームごとに`kanban.dispatch_profiles`を設定して、そのホームがクレームできる担当者を
正確に宣言してください — それ以外のものは起動される代わりにディスパッチャーの
`skipped_nonspawnable`のバケットに入り、ゲートウェイの起床プローブにとっての
起動可能な作業として数えられることもなくなります。

### 冪等な作成（自動化/Webhook向け）

```bash
# First call creates the task. Any subsequent call with the same key
# returns the existing task id instead of duplicating.
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### 一括CLI動詞

すべてのライフサイクル動詞は複数のidを受け付けるため、1つのコマンドで
バッチを片付けられます。

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

:::note アンブロックされたタスクの行き先
`unblock`は安全な元のフェーズに戻します: 親が完了しているレビュアー起点の作業は**`review`**、
親が完了している実装作業は**`ready`**、いずれかの親がまだ開いている間は
**`todo`**です。`todo`のタスクは元のフェーズの来歴を保持し、依存のゲートが解除されると
自動的に`review`または`ready`に戻ります。`unblock`が直接`triage`にルーティングすることは決してありません。

タスクをアンブロックした後に**`triage`**に現れた場合、そこに入れたのはアンブロックでは
ありません。その後の*同じ理由での再ブロック*です: タスクが
同じ原因でブロック → アンブロック → 再ブロックを`BLOCK_RECURRENCE_LIMIT`
回（デフォルト`2`）繰り返すと、アンブロックのループのブレーカーはそれを`blocked`に戻すのをやめ
— そこではcronがアンブロックし続けるだけなので — オーケストレーションの注意を引くために`triage`にルーティングします。
これは決定的なDBのガードであり、LLMの判断ではなく、
タスクの本文テキストでオプトアウトすることもできません: 再発カウンターは意図的に
各アンブロックを越えて保持されます（リセットされるのは`complete`が成功したときだけです）。アンブロックされた
タスクを作業プールに留めておくには、アンブロックする前に*なぜ再ブロックされ続けるのか*（未完了の
親、欠けている入力、満たされない能力）を解決するか、ループが想定内であれば
`BLOCK_RECURRENCE_LIMIT`を引き上げてください。
:::

## チャットプロファイルでツールを有効にする

DesktopのKanbanプラグインはボードを表示しますが、チャットエージェントに
タスクを管理する権限を与えるものではありません。作業をオーケストレーションすべきプロファイルと
プラットフォームで、`kanban`ツールセットを有効にしてください:

```bash
hermes -p planner tools enable kanban                      # CLI / TUI / Desktop chats
hermes -p planner tools enable kanban --platform telegram  # a gateway platform
```

各プラットフォームは、`config.yaml`の`platform_toolsets.<platform>`の下に独自の選択を持ちます。
このツールセットは`hermes tools`とダッシュボードのチェックボックスとしても表示されます。
それを持つゲートウェイのエージェントは、チャットから`kanban_create`でき、同じスレッドでそのタスクの
完了/ブロックの通知に自動的に購読されます。この設定を変更した後は
新しいチャットを開始してください。既存の会話はツールスキーマと
プロンプトキャッシュを保持します。`agent.disabled_toolsets`は引き続き優先されます。レガシーのトップレベルの
`toolsets: [kanban]`は、プラットフォームの選択が保存されていない場合にのみフォールバックとして尊重されます。
`all`だけではKanbanのオプトインにはなりません。

ディスパッチャーが所有するワーカーは、タスクのライフサイクルのツールを自動的に受け取ります。
`delegate_task`の子が、ボードを変更する権限を得ることはありません。

## ワーカーはボードとどう対話するか {#how-workers-interact-with-the-board}

**ワーカーは`hermes kanban`をシェル実行しません。** ディスパッチャーがワーカーを生成する際、子の環境に`HERMES_KANBAN_TASK=t_abcd`を設定し、その環境変数がモデルのスキーマ内の専用**kanbanツールセット**を有効化します。同じツールセットは、ツールセット設定で`kanban`を有効化したオーケストレータープロファイルでも利用できます。これらのツールは、CLIと同じく、Pythonの`kanban_db`レイヤーを通じて直接ボードを読み取り、変更します。動作中のワーカーは、これらを他のツールと同じように呼び出します。`hermes kanban` CLIを目にすることも必要とすることもありません。

| ツール | 目的 | 必須パラメータ |
|---|---|---|
| `kanban_show` | 現在のタスク（タイトル、本文、過去の試行、親からの引き継ぎ、コメント、事前整形済みの完全な`worker_context`）を読み取る。デフォルトでは環境のタスクid。 | — |
| `kanban_list` | `assignee`、`status`、`tenant`、アーカイブの表示可否、件数上限のフィルター付きでタスクのサマリーを一覧する。オーケストレーターがボード上の作業を発見するためのもの。 | — |
| `kanban_complete` | `summary`＋`metadata`の構造化された引き継ぎで完了する。 | `summary` / `result`の少なくとも一方 |
| `kanban_request_review` | 永続的な`summary`、任意の`metadata`、任意のレビュアープロファイルで、同じカードでのレビューを開始する。タスクは`review`に移る。これはブロックではない。 | `summary` |
| `kanban_request_changes` | アクティブなレビュー実行からのレビュアーの判定。その実行をクローズし、親のゲートを再適用し、ブロックのループの計上なしにタスクを元の実装者にルーティングする。 | `reason` |
| `kanban_block` | 作業を止め、理由に応じてルーティングする: `kind=dependency`（`todo`で待機し、未完了の親が終わると自動的に再開。開いている親がない場合は、待機が満たされることはないため代わりに`needs_input`として記録される）、`needs_input`/`capability`/`transient`（人間に提示）。同じ種類の再ブロックが繰り返されると自動的に`triage`にエスカレートする。 | `reason` |
| `kanban_heartbeat` | 長時間の処理中に生存を知らせる。純粋な副作用。 | — |
| `kanban_comment` | タスクスレッドに永続的なメモを追記する。 | `task_id`, `body` |
| `kanban_attach` | ファイルのバイト列をインライン（base64）で渡してタスクにファイルを添付する。タスクの添付ディレクトリに保存される（上限25 MB）。 | ファイルのバイト列 + 名前 |
| `kanban_attach_url` | URLでタスクにファイルを添付する。 | `url` |
| `kanban_attachments` | タスクの添付を一覧する。 | — |
| `kanban_create` | （オーケストレーター）`assignee`と任意の`parents`、`skills`などを指定して子タスクにファンアウトする。開いている親によって新しいカードが`todo`に駐機された場合は`gated: true` + `gated_by`を返す。 | `title`, `assignee` |
| `kanban_link` | （オーケストレーター）事後に`parent_id → child_id`の依存エッジを追加する。子が`ready`で、親が完了していないため`todo`に降格された場合は`gated: true`を返す — 子は親が完了した後にのみ実行される。子がすでにクレームされている場合は`child is already running`で拒否される — クレーム後に追加されたエッジでは実行を直列化できないため（ワーカーは`kind=dependency`のブロックの前に、*自分自身の*実行中のカードをリンクすることはできる）。 | `parent_id`, `child_id` |
| `kanban_unblock` | （オーケストレーター）ブロックされたタスクを元のフェーズ（`review`または`ready`）に、親がまだ開いている間は`todo`に戻す。 | `task_id` |

典型的なワーカーのターンは次のようになります。

```
# Model's tool calls, in order:
kanban_show()                                     # no args — uses HERMES_KANBAN_TASK
# (model reads the returned worker_context, does the work via terminal/file tools)
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# (more work)
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

**オーケストレーター**ワーカーは、代わりにファンアウトします。

```
kanban_show()
kanban_create(
    title="research ICP funding 2024-2026",
    assignee="researcher-a",
    body="focus on seed + series A, North America, AI-adjacent",
)
# → returns {"task_id": "t_r1", ...}
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
# → returns {"task_id": "t_r2", ...}
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],                     # promotes to ready when both complete
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

「（オーケストレーター）」のツール — `kanban_list`、`kanban_create`、`kanban_link`、`kanban_unblock`、および外部タスクに対する`kanban_comment` — は同じツールセットを通じて利用できます。慣例（自動注入されるkanbanのガイダンスにエンコードされている）として、ワーカープロファイルは無関係な作業をファンアウトしたりルーティングしたりせず、オーケストレータープロファイルは実装作業を実行しません。ディスパッチャーが生成したワーカーは、破壊的なライフサイクル操作については依然としてタスクスコープに限定され、無関係なタスクを変更できません。

### なぜ`hermes kanban`をシェル実行するのではなくツールを使うのか

理由は3つあります。

1. **バックエンドのポータビリティ。** terminalツールがリモートバックエンド（Docker / Modal / SSH / プラグインのサンドボックス）を指しているワーカーは、`hermes kanban complete`をコンテナ*内部*で実行してしまい、そこには`hermes`がインストールされておらず`~/.hermes/kanban.db`もマウントされていません。kanbanツールはエージェント自身のPythonプロセス内で実行され、terminalバックエンドに関係なく常に`~/.hermes/kanban.db`に到達します。
2. **シェルクオートの脆さがない。** `--metadata '{"files": [...]}'`をshlex＋argparse経由で渡すのは潜在的な落とし穴です。構造化されたツール引数はこれを完全に回避します。
3. **より良いエラー。** ツールの結果は、パースしなければならないstderr文字列ではなく、モデルが推論できる構造化されたJSONです。

**通常のセッションへのスキーマフットプリントはゼロ。** 通常の`hermes chat`セッションのスキーマには、アクティブなプロファイルがオーケストレーター作業のために`kanban`ツールセットを明示的に有効にしていない限り、`kanban_*`ツールが1つもありません。ディスパッチャーが生成したタスクワーカーは、`HERMES_KANBAN_TASK`が設定されているためタスクスコープのツールを得ます。オーケストレータープロファイルは、設定を通じてより広いルーティングの接点を得ます。Kanbanにまったく触れないユーザーにツールの肥大化はありません。

自動注入されるkanbanのガイダンスが、どのツールをいつ、どの順序で呼び出すかをモデルに教えます。これが注入されるのは、タスクを所有するディスパッチャー生成のワーカーに対してだけです: `kanban`ツールセットを有効にしているだけのインタラクティブなセッションはツールを保持しますが、「ONE taskを割り当てられた」とは伝えられません。またワーカーの内部から起動された`delegate_task`の子やcronの実行 — ワーカーの`HERMES_KANBAN_TASK`を継承する — は、ワーカーのプロトコルを受け取ることも、自身のターン予算を使い切ったときにワーカーのカードに対して終端の結果を記録することもありません。

### 推奨される引き継ぎのエビデンス

`kanban_complete(summary=..., metadata={...})`は意図的に柔軟です。
summaryは人間が読めるクロージング情報であり、`metadata`は
下流のエージェント、レビュアー、ダッシュボードが散文をスクレイピングせずに
再利用できる機械可読な引き継ぎです。

エンジニアリングやレビューのタスクでは、次の任意のmetadata形状を推奨します。

```json
{
  "changed_files": ["path/to/file.py"],
  "verification": ["pytest tests/hermes_cli/test_kanban_db.py -q"],
  "dependencies": ["parent task id or external issue, if any"],
  "blocked_reason": null,
  "retry_notes": "what failed before, if this was a retry",
  "residual_risk": ["what was not tested or still needs human review"]
}
```

これらのキーは慣例であり、スキーマの要件ではありません。有用な性質は、
すべてのワーカーが、次の読み手が4つの質問にすばやく答えられるだけの
エビデンスを残すことです。

1. 何が変わったか？
2. どう検証されたか？
3. 失敗した場合、何がこれをunblockまたはリトライできるか？
4. どんなリスクが意図的に未対応のまま残されているか？

シークレット、生ログ、トークン、OAuth関連の情報、無関係なトランスクリプトは
`metadata`に入れないでください。代わりにポインターやサマリーを保存してください。タスクにファイルや
テストがない場合は、`summary`で明示的にそう述べ、ソースURL、issue id、手動レビュー手順など、
実際に存在するエビデンスに`metadata`を使ってください。

### ワーカーのライフサイクル

Kanbanタスクを処理するすべてのプロファイルは、ワーカーのライフサイクルを自動的に得ます — それは起動時にワーカーのシステムプロンプトに注入される（`KANBAN_GUIDANCE`ブロック）ため、**インストールや設定は何も必要ありません**。これは、CLIコマンドではなく**ツール呼び出し**でライフサイクル全体をワーカーに教えます。

1. 生成時に`kanban_show()`を呼び出し、タイトル＋本文＋親からの引き継ぎ＋過去の試行＋コメントスレッド全体を読み取る。
2. （terminalツール経由で）`cd $HERMES_KANBAN_WORKSPACE`し、そこで作業する。
3. 長時間の処理中は数分ごとに`kanban_heartbeat(note="...")`を呼び出す。**作業が1時間を超える可能性がある場合は、少なくとも1時間に1回`kanban_heartbeat`を呼び出してください** — ディスパッチャーは、`kanban.dispatch_stale_timeout_seconds`（デフォルト4時間）を超えて実行され、直近1時間にハートビートのないタスクを、ワーカーが後始末せずにクラッシュしたとみなして再取得します。再取得は無害です（タスクは失敗カウンターを増やさずに再ディスパッチのため`ready`に戻ります）が、現在の実行の進捗は失われます。
4. `kanban_complete(summary="...", metadata={...})`で完了するか、`kanban_request_review(summary="...")`でコードの変更を同じカードでのレビューに引き継ぐか、行き詰まったら`kanban_block(reason="...")`を呼ぶ。

通常のツールの活動もクレームを自動的に延長します（ワーカーはプロセス内の生存状態を約1分ごとにボードに反映します）。この橋渡しが機能するのは、ディスパッチャー自身が起動したプロセスだけです: `HERMES_KANBAN_TASK`の隣に`HERMES_DELEGATED_CHILD_CONTEXT`を持つプロセス（`delegate_task`の子孫、またはワーカーの環境を手で起動したコピー）はボードから遮断されます — その自動ハートビートは`kanban auto-heartbeat for task … refused`の警告を1回ログに出力し、`kanban_complete` / `kanban_request_review`は拒否されます。マーカーをexportで消すのではなく、起動方法を修正してください（ディスパッチャーにワーカーを起動させる）。

その最後の終端のボード呼び出し（`kanban_complete` / `kanban_request_review` /
`kanban_block`。レビュアーは`kanban_complete`または`kanban_request_changes`で終える）は、ワーカーの
プロトコルの一部です。タスクがまだ`running`の状態でワーカープロセスがステータス0で終了した場合、
ディスパッチャーはそれをプロトコル違反として扱い、
`protocol_violation`イベントを発行します。したがって、ターンが失敗したディスパッチャー起動のワーカーは
非ゼロで終了します: 通常の失敗では`1`、プロバイダーがレート制限、過負荷、
5xx、タイムアウトを返した場合、またはアカウントが課金/クォータの壁に達した場合は`75`
（`EX_TEMPFAIL`） — ディスパッチャーはその実行を`rate_limited`として記録し、
失敗を数えずにタスクを再キューするため、クォータのウィンドウがプロトコル違反として
記録されることはありません — そして、リトライでは直せないものをプロバイダーが拒否した場合は`78`（`EX_CONFIG`）です:
プロファイルの認証情報（401/403、失効または無効なキー）、モデル（404 / model not found）、またはTLSのチェーンです。
その**終端のプロバイダーエラー**は、最初の発生でサーキットブレーカーを作動させます:
ディスパッチャーは実行を`exit_kind: terminal_provider`付きの`crashed`として記録し、
`terminal_provider: true`付きの`gave_up`を発行して、カードを`blocked`に駐機させます（固定 — `recompute_ready`は自動再開しません）。
`last_failure_error`にはプロバイダー自身の言葉が入り、`kanban.failure_limit` / `max_retries`が尽きるまで
同じ壁に向かって再起動し続けることはありません。両方のレーンが同じ記録を受けます: 失効した
キーで死んだレビュアーのワーカーは、実装者とまったく同じようにカードを駐機させます。`hermes kanban show`は
それを*Provider rejected this profile's credential or model — blocked
after one attempt*として表示します。担当者プロファイルのプロバイダーを修正し（`hermes -p <profile>
auth` / `setup`）、その後`hermes kanban unblock <id>`を実行してください。ワーカーはまた、自身の終了コードを
自分のログの最後の行（`[kanban-worker-exit] rc=<code>`）として書き込むため、ティックごとの
`hermes kanban dispatch`プロセス — ワーカーを回収しておらず、その終了ステータスを
読めない — も、ゲートウェイ組み込みのディスパッチャーと同じ方法で同じ死を記録します。
その行に到達する前に強制終了されたワーカーは、単なる
`crashed`（`pid <n> not alive`）です。

**エージェント側の予防:** ワーカーが終了する前に、モデルが終端のボードツール呼び出しなしに
停止しようとしていることを検出すると、Hermesは最大2回の合成的な促しを注入します。これは、モデルが次の
ステップを語って（「レポートを書きましょう」）`finish_reason=stop`で停止するというよくあるケースを捉えます。促しは
モデルに`kanban_complete`、`kanban_request_review`、`kanban_block`を
すぐに呼び出すよう念を押します。すでにカードを引き継いだワーカー（`kanban_request_review`、またはレビュアーからの
`kanban_request_changes`）が促されることは決してありません — 引き継ぎがその終端の
呼び出しであり、促しがレビュー中のカードを`kanban_complete`するよう求めることもありません。この
ガードは、ディスパッチャーが起動したワーカー自身に対してのみ有効です（`HERMES_KANBAN_TASK`が
設定され、その実行がそのタスクを所有している） — ワーカーの内部で実行される`delegate_task`の子やcronジョブは
変数を継承しますが、ボードのツールを持たないため促されることはありません —
また`HERMES_KANBAN_STOP_NUDGE=0`で無効にできます。

**ディスパッチャー側の回復:** 促しが尽きた場合や、ワーカーが促しに到達する前に
クラッシュした場合、ディスパッチャーは違反に**上限付きのリトライ**
（最大`_PROTOCOL_VIOLATION_FAILURE_LIMIT`回の連続した違反、デフォルト3）を与えてから、
同じループに再起動する代わりにタスクを自動ブロックします。この
予算は*連続した*クリーン終了のプロトコル違反だけを数えます — 間に挟まった
レート制限による再キューは中立で、それ以外の種類の失敗は
連続をリセットします — またタスクごとの`max_retries`が上限を上書きします。この予算によってブロックされたカードは
ブロックされたままです（`failure_limit`未満の
ブレーカーによるブロックのように自動昇格はしません）。`hermes kanban unblock <id>`を実行するまで続き、それによって新しい
リトライ予算も与えられます。これは通常、
モデルがプレーンテキストの回答を書いて、Kanbanのツールの接点を使わずに
終了したことを意味します。

ライフサイクルと、重要な参照の詳細（ワークスペースの種類、成果物の`artifacts`、作成したカードのクレーム）は、そのシステムプロンプトのブロックに含まれて出荷されるため、どのプロファイルのもとで実行されるかにかかわらず、すべてのワーカーがそれらを持ちます — プロファイルごとのスキルのセットアップは不要です。

**ワーカーのセッション名。** ワーカーのセッションには、起動時にそのカードにちなんだタイトルが付けられます（`Fix the swap modal`、またはボードの行を読めない場合は`Kanban task <id>`）。そのため`hermes sessions`やセッション検索には、モデルの推測ではなくカードが表示されます。ワーカーが、インタラクティブなセッションに名前を付ける補助の`title_generation`モデル呼び出しを行うことは決してありません。ワーカーのセッションでの手動の`/title`は引き続き優先されます。

### 特定のタスクに追加スキルを固定する

ときには、単一のタスクが、担当者プロファイルがデフォルトで持っていないスペシャリストのコンテキストを必要とすることがあります — `translation`スキルが必要な翻訳ジョブ、`github-code-review`が必要なレビュータスク、`security-pr-audit`が必要なセキュリティ監査などです。毎回担当者のプロファイルを編集する代わりに、スキルを直接タスクに付与します。

**オーケストレーターエージェントから**（通常のケース — あるエージェントが別のエージェントに作業をルーティングする）、`kanban_create`ツールの`skills`配列を使います。

```
kanban_create(
    title="translate README to Japanese",
    assignee="linguist",
    skills=["translation"],
)

kanban_create(
    title="audit auth flow",
    assignee="reviewer",
    skills=["security-pr-audit", "github-code-review"],
)
```

**人間から（CLI / スラッシュコマンド）**は、それぞれに`--skill`を繰り返します。

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**ダッシュボードから**は、作成タスクのダイアログの**skills**フィールドにスキルをカンマ区切りで入力します。

ディスパッチャーは列挙されたスキルごとに`--skills <name>`フラグを1つ発行するため、ワーカーは自動注入されるkanbanのガイダンスに加えて、それらすべてをロードした状態で生成されます。スキル名は、担当者のプロファイルに実際にインストールされているスキルと一致しなければなりません（`hermes skills list`を実行して何が利用可能か確認してください）。実行時のインストールはありません。

### タスクごとのモデルの上書き {#per-task-model-override}

担当者プロファイルのデフォルトとは独立に、タスクのワーカーを特定のモデル（および任意でプロバイダー）に固定します:

```bash
# At creation
hermes kanban create "hard refactor" --assignee coder \
    --model claude-opus-4.6 --provider anthropic

# Or later — takes effect on the next dispatch
hermes kanban set-model t_abcd claude-opus-4.6 --provider anthropic
hermes kanban set-model t_abcd none    # clear the override
```

ディスパッチャーは固定されたモデルでワーカーを起動します（設定されていれば`--provider <name>`が渡されます。`--provider`にはモデルが必要です）。ダッシュボードのタスクごとのモデルのドロップダウンも、同じ`model_override`フィールドを操作します。上書きがない場合、ワーカーはプロファイルに設定されたモデルを使います。

### コスト戦略: フロンティアのオーケストレーター、安価なワーカー

Kanbanのプロファイルごとの設定により、プランナー/ワーカーのコスト分割は自然に行えます。プロジェクトを範囲の明確なカードに分解するにはフロンティアレベルの判断が必要ですが、明確なゴール、コンテキスト、引き継ぎのエビデンスをすでに持つカードを実行するのには通常必要ありません — そしてトークンの大部分はワーカーで消費されるため、コストを左右するのはワーカーのモデルです。オーケストレーター/ディスパッチャーのプロファイルをフロンティアモデルで実行し、ワーカーのプロファイルは安価なモデルに向けてください。各プロファイルは`~/.hermes/profiles/<name>/`の下に独自の`config.yaml`を持ち、ディスパッチャーは`hermes -p <assignee>`を起動する際にプロファイルスコープの`HERMES_HOME`を注入するため、各ワーカーは自分のプロファイルのモデル設定を読みます:

```yaml
# ~/.hermes/config.yaml (orchestrator / dispatcher profile)
model:
  default: "your-frontier-model"

# ~/.hermes/profiles/coder/config.yaml (worker profile)
model:
  default: "your-inexpensive-model"

# ~/.hermes/profiles/researcher/config.yaml (another worker profile)
model:
  default: "your-inexpensive-model"
```

品質が重要なカードがたまにある場合は、[タスクごとのモデルの上書き](#per-task-model-override)（作成時の`--model`/`--provider`、後からの`hermes kanban set-model`、またはダッシュボードのモデルのドロップダウン）で、そのタスクだけをより強力なモデルに固定してください — プロファイルの編集は不要です。

### ライフサイクルのプラグインフック

ボードの遷移は[プラグインフック](./hooks.md#plugin-hooks)を発火させます: `kanban_task_claimed`、`kanban_task_completed`、`kanban_task_blocked`で、それぞれ`task_id`と`profile_name`を持ちます。フックはボードのDBの変更がコミットされた**後に**発火するため、コールバックは常に永続化された状態を見ます。プロセスの分割に注意してください: `kanban_task_claimed`は**ディスパッチャー**のプロセスで発火し、`kanban_task_completed`/`kanban_task_blocked`は**ワーカー**のプロセスで発火します — すべての遷移を一元的に観測するには、ディスパッチャーのプロファイルにフックを登録してください。

```python
def register(ctx):
    def on_blocked(task_id=None, profile_name=None, **kw):
        ctx.dispatch_tool("terminal", {"command": f"notify-send 'kanban blocked: {task_id}'"})
    ctx.register_hook("kanban_task_blocked", on_blocked)
```

### ゴールモードのカード（`--goal`） {#goal-mode-cards---goal}

デフォルトでは、各ワーカーはカードに対して**1回だけ**挑戦します — 作業をして、`kanban_complete`/`kanban_block`を呼び、終了します。代わりに`--goal`（CLI）または`goal_mode=True`（`kanban_create`ツール / ダッシュボード）を渡すと、そのワーカーは**ゴールループ**で実行されます。これは`/goal`スラッシュコマンドの背後にあるのと同じRalph形式のエンジンです: 各ターンの後に補助のジャッジがワーカーの出力をカードのタイトル＋本文（受け入れ基準として扱われる）と照合し、作業が終わっていなければ — そしてターン予算が残っていれば — ワーカーは、ジャッジが同意するか、ワーカー自身がタスクを終了させるか、予算が尽きるまで**同じセッションで**続けます（予算が尽きた場合は、黙って終了するのではなく、人間のレビューのためにカードを**ブロック**します）。ジャッジが書かれたとおりのゴールを**達成不可能**と判断した場合、カードはジャッジの理由とともに即座にブロックされます — 不可能なカードがdoneとマークされることは決してなく、そのようなカードに対する`kanban complete` / `kanban request-review`は、`kanban block`や範囲の見直しへの案内とともに拒否されます。

```bash
hermes kanban create "Translate the docs site to French" \
    --body "Acceptance: every page translated, no English left, links intact." \
    --assignee linguist \
    --goal \
    --goal-max-turns 15      # optional; default 20
```

終わりの決まっていない、複数ステップの、または「Xが真になるまで続ける」カードに使ってください。安価な1回きりの作業には使わないでください — ターンごとのジャッジのオーバーヘッドに見合わず、ディスパッチャーの既存のリトライ/サーキットブレーカーがすでにワーカーの一時的な失敗を処理します。ジャッジの品質はゴールのテキスト次第なので、本文は**明示的な受け入れ基準**として書いてください。

`kanban complete` / `kanban request-review`（および対応する`kanban_complete` / `kanban_request_review`ツール）に対するジャッジのゲートは、**本当の判定**があった場合にのみ拒否します。ジャッジの呼び出し自体が失敗した場合 — リレーのエラー、認証、タイムアウト — 引き継ぎは許可され、`goal judge unreachable … allowing lifecycle handoff`の警告がログに記録されるため、到達できないジャッジが人間の承認をブロックすることはありません。ヘッドレスのCLIのゲートは、`specify`/`decompose`と同じタスクごとのリレーのアフィニティキー（`kanban:<task-id>`）を送るため、セッションキーを必要とするリレーもジャッジのリクエストを受け付けます。

ゴールモードのワーカーの**Worker log**（ダッシュボードのドロワー、`hermes kanban log <id>`）には、他のワーカーと同じライブのツールフィードに加えて、判定されたターンごとに`kanban goal loop: turn N/M verdict=…`の行が1つ含まれるため、カードの実行中にループが何をしているかを追えます。

:::note ゴールモードのカードは`/goal`のエンジンを借りますが、それに接続はしません
`--goal`は、継続のループを*その1つのカードのワーカーセッションの中で*実行します。エンジンは[`/goal`スラッシュコマンド](./goals)と共有しますが、状態は共有しません: チャットセッションで`/goal`を設定しても、Kanbanのカードが作成、クレーム、移動されることは決してなく、ゴールモードのカードのループはどのチャットセッションの`/goal status`からも見えません。この会話で反復を続けたい場合は[`/goal`](./goals)を使い、ボード上で作業させたい場合はカードを作成してください。
:::

### オーケストレーターの振る舞い

**行儀の良いオーケストレーターは、作業そのものを自分では行いません。** ユーザーのゴールをタスクに分解し、それらをリンクし、それぞれをあなたがセットアップしたプロファイルのいずれかに割り当て、一歩引きます。オーケストレーターのガイダンス — 誘惑に負けないためのルール、Step-0のプロファイル発見プロンプト（ディスパッチャーは未知の担当者名に対して静かに失敗するため、オーケストレーターはすべてのカードを、あなたのマシンに実際に存在するプロファイルに根付かせなければなりません）、そして`kanban_create` / `kanban_link` / `kanban_comment`をキーとする分解のプレイブック — は、ワーカーのシステムプロンプトに自動的に注入されます。インストールするものは何もありません。

正準的なオーケストレーターのターン（2人の並列リサーチャーがライターに引き継ぐ）:

```
# Goal from user: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # promoted to 'ready' when both researchers complete
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# Optional: add cross-cutting deps discovered later without re-creating tasks
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

オーケストレーターのガイダンスはワーカーのシステムプロンプトに自動的に含まれて出荷されます — プロファイルごとにインストールしたり同期したりするものは何もありません。

**ファンアウトの前に決める。** 設計の判断はワーカーではなくオーケストレーターに属します。2つの並列カードがそれぞれ同じこと — 命名規則、スキーマ、ファイル形式、APIの形 — を選ばなければならない場合、オーケストレーターがそれを一度決め、その判断を**両方の**カードの本文に書き込みます。ワーカーは兄弟カードを見られないため、すべての子カードの本文は、それが依存するすべての判断を持っていなければなりません。例: 並列のカード「エクスポーターを作る」と「インポーターを作る」では、各ワーカーに独自のファイル形式を作らせてはいけません — 事前に1つ（たとえば`version`フィールドを持つ改行区切りのJSON）を選んで両方の本文に書いてください。そうしないと、2つの半分は決して往復できません。

最良の結果を得るには、ツールセットがボード操作（`kanban`、`gateway`、`memory`）に制限されたプロファイルと組み合わせて、オーケストレーターが試みても文字どおり実装タスクを実行できないようにしてください。

## ダッシュボード（GUI）

`/kanban` CLIとスラッシュコマンドだけでもボードをヘッドレスに運用できますが、ヒューマンインザループにはビジュアルなボードのほうが適したインターフェースであることが多いです: トリアージ、プロファイルをまたいだ監督、コメントスレッドの閲覧、列間でのカードのドラッグなど。Hermesはこれを、[ダッシュボードの拡張](./extending-the-dashboard)で示されたモデルに従って、`plugins/kanban/`の**バンドルされたダッシュボードプラグイン**として提供します — コア機能でも別個のサービスでもありません。

次のコマンドで開きます。

```bash
hermes kanban init      # one-time: create kanban.db if not already present
hermes dashboard        # "Kanban" tab appears in the nav, after "Skills"
```

### プラグインが提供するもの

- ステータスごとに1列を表示する**Kanban**タブ: `triage`、`todo`、`ready`、`running`、`blocked`、`done`（トグルがオンのときは`archived`も）。
  - キューの列は、カードをディスパッチの順（優先度、次に古い順 — 先頭のカードが次に起動される）で一覧表示します。`done`列は履歴であり、完了が新しい順に一覧表示します。`hermes kanban list --status done --sort completed-desc`で、CLIでも同じ順序になります。
  - `triage`はラフなアイデアのためのパーキング列です。デフォルト（`kanban.auto_decompose: true`）では、ディスパッチャーはここに着地したタスクに対して自動的に**デコンポーザー**を実行します。組み込みのデコンポーザーは`auxiliary.kanban_decomposer`のモデル経路を使い、プロファイルの一覧（説明付き）を読み、タスクを最適なスペシャリストにルーティングされた子タスクの小さなグラフにファンアウトします。元のタスクはすべての子の親として生き続けるため、その担当者（`kanban.orchestrator_profile`、なければタスクがすでに持っていた担当者、なければアクティブなデフォルトプロファイル）は、すべてが終わったときに再び起きて完了を判断します。ページ上部の**Orchestration: Auto/Manual**のピル（エメラルド = Auto、くすんだグレー = Manual）で切り替えるか、`config.yaml`を直接編集してください。どちらのモードも`hermes kanban specify`と共存します — ファンアウトが不要なときの単一タスクの仕様の書き換えとして、引き続き利用できます。
- カードには、タスクid、タイトル、優先度バッジ、テナントタグ、割り当てプロファイル、コメント/リンク数、**進捗ピル**（タスクに依存タスクがあるとき、完了した子の`N/M`）、「N前に作成」が表示されます。カードごとのチェックボックスで複数選択ができます。
- **Running内のプロファイルごとのレーン** — ツールバーのチェックボックスで、Running列を担当者ごとにサブグループ化するかを切り替えます。
- **WebSocketによるライブ更新** — プラグインは短いポーリング間隔で追記専用の`task_events`テーブルを追従します。いずれかのプロファイル（CLI、ゲートウェイ、別のダッシュボードタブ）が操作した瞬間に、ボードに変更が反映されます。リロードはデバウンスされるため、イベントのバーストでも単一の再取得が発生するだけです。
- 列間でカードを**ドラッグ&ドロップ**してステータスを変更します。ドロップは`PATCH /api/plugins/kanban/tasks/:id`を送り、CLIが使うのと同じ`kanban_db`コードを経由します — 3つの接点が決してずれることはありません。破壊的なステータス（`done`、`archived`、`blocked`）への移動は確認を求めます。タッチデバイスではポインターベースのフォールバックを使うため、タブレットからもボードを操作できます。
- **作成タスクのダイアログ** — 任意の列ヘッダーの`+`をクリックすると、ラベル付きのフィールドを持つモーダルが開きます: タイトル、担当者、優先度、スキル、ワークスペースの種類/パス（ボードのプロジェクトディレクトリから初期値が入り、タスクごとに上書き可能）、ゴールモード、そして（任意で）既存のすべてのタスクから選べるドロップダウンでの親タスク。Enterでタスクを作成し、Shift+Enterでタイトルのフィールドに改行を挿入し、Escapeでキャンセルします。Triage列から作成すると、新しいタスクは自動的にトリアージに駐車されます。
- **一括アクション付きの複数選択** — カードをshift/ctrlクリックするか、そのチェックボックスをチェックして選択に追加します。上部に一括アクションバーが表示され、ステータスの一括遷移、アーカイブ、（プロファイルのドロップダウンまたは「（割り当て解除）」での）再割り当てができます。破壊的なバッチは先に確認します。idごとの部分的な失敗は、残りを中断せずに報告されます。
- （shift/ctrlなしで）**カードをクリック**すると、サイドドロワーが開きます（Escapeまたは外側クリックで閉じる）。内容は次のとおりです。
  - **編集可能なタイトル** — 見出しをクリックしてリネーム。
  - **編集可能な担当者 / 優先度** — メタ行をクリックして書き換え。
  - **編集可能な説明** — デフォルトではmarkdownレンダリング（見出し、太字、斜体、インラインコード、フェンスドコード、`http(s)` / `mailto:`リンク、箇条書き）で、テキストエリアに切り替える「edit」ボタンがあります。markdownレンダリングは小さくXSS安全なレンダラーです — すべての置換はHTMLエスケープ済みの入力に対して実行され、`http(s)` / `mailto:`リンクだけが通過し、`target="_blank"`＋`rel="noopener noreferrer"`が常に設定されます。
  - **依存関係エディター** — 親と子のチップリスト（それぞれリンク解除用の`×`付き）と、新しい親や子を追加するための他のすべてのタスクのドロップダウン。循環の試みはサーバー側で明確なメッセージとともに拒否されます。
  - **ステータスアクション行**（→ triage / → ready / → running / block / unblock / complete / archive）。破壊的な遷移には確認プロンプトが付きます。**Triage**列のカードでは、この行にLLM駆動の2つのアクションも表示されます: **⚗ Decompose**は、タスクを説明に基づいてスペシャリストのプロファイルにルーティングされた子タスクのグラフにファンアウトし、**✨ Specify**は単一タスクの仕様の書き換えを行います。LLMがタスクはファンアウトの恩恵を受けないと判断した場合、Decomposeはspecify形式の昇格にフォールバックするため、厳密な上位互換です。どちらもCLI（`hermes kanban decompose <id>` / `specify <id>` / `--all`）、任意のゲートウェイプラットフォーム（`/kanban decompose <id>`）、およびプログラム的には`POST /api/plugins/kanban/tasks/:id/decompose`と`…/specify`から到達できます。モデルは`config.yaml`の`auxiliary.kanban_decomposer`と`auxiliary.triage_specifier`で設定します。
  - 結果セクション（こちらもmarkdownレンダリング）、Enterで送信できるコメントスレッド、直近20件のイベント。
- **ツールバーフィルター** — フリーテキスト検索、テナントのドロップダウン（`config.yaml`の`dashboard.kanban.default_tenant`がデフォルト）、担当者のドロップダウン、「show archived」トグル、「lanes by profile」トグル、そして次の60秒のティックを待たなくてよいように**Nudge dispatcher**ボタン。

ビジュアルの目標は、おなじみのLinear / Fusionレイアウトです: ダークテーマ、件数付きの列ヘッダー、色付きのステータスドット、優先度とテナントのピルチップ。プラグインはテーマのCSS変数（`--color-*`、`--radius`、`--font-mono`など）だけを読むため、どのダッシュボードテーマがアクティブでも自動的にリスキンされます。

### AutoとManualのオーケストレーション {#auto-vs-manual-orchestration}

Kanbanボードには、Triage列に置いたタスクを扱う方法が2つあります:

**Auto（デフォルト）** — `kanban.auto_decompose: true`。ゲートウェイ組み込みのディスパッチャーは、各ティックで**デコンポーザー**を実行します。トリアージタスクを大量に読み込んでも補助LLMへの出費が一気に増えないよう、`kanban.auto_decompose_per_tick`（デフォルトでティックあたり3タスク）で上限が設けられます。デコンポーザーは、組み込みの分解プロンプトと`auxiliary.kanban_decomposer`のモデル経路を使い、インストール済みのプロファイルとその説明を読み、LLMにJSONのタスクグラフを生成させます: どのタスクを起動するか、誰に割り当てるか、どれがどれに依存するか。元のトリアージタスクはグラフのすべての葉の親になるため、グラフ全体が完了するまで生き続け、その後`ready`に昇格して戻るため、その担当者（`kanban.orchestrator_profile`、なければタスクがすでに持っていた担当者、なければアクティブなデフォルトプロファイル）が完了を判断し、作業が終わっていなければさらにタスクを追加できます。これが「一行を置いて立ち去る」フローです。

完了した組み込みのファンアウトは、その子のグラフとともにアトミックに記録されます。
そのルートをTriageに戻しても、別のグラフは作成されません。通常の前提条件の
リンクは、タスクの最初の分解を妨げません。完了のマーカーは、タスクが削除されるまで
イベントの保持期間を越えて残ります。これは、独立に作成された手動のグラフの意味的な重複排除でも、
以前に刈り取られた履歴の修復でもありません。

新しいタスクがテナントを省略した場合、作成時には指定された順序で、親のうち最初の空でない
テナントを継承します。明示的なテナント（ツールから渡されるワーカーの
アクティブなテナントを含む）が優先されます。ボードは引き続きハードな分離境界です。

**Manual** — `kanban.auto_decompose: false`。トリアージタスクは、あなたが操作するまでトリアージに留まります。カードの**⚗ Decompose**ボタンをクリックするか、`hermes kanban decompose <id>`（または`--all`）を実行するか、チャットから`/kanban decompose <id>`を使ってください。これはデコンポーザー導入以前のボードの挙動と一致し、何がいつ実行されるかを完全に制御したい場合に便利です。

**重要な境界:** Manualモードが無効にするのは、組み込みのTriageデコンポーザーだけです。プロファイルが`kanban_create`を呼び出すことは妨げず、作成者のセッションの起床も無効にしません。`kanban.auto_subscribe_on_create: true`の場合、タスクの終端イベントは合成的なステータスのターンで元のエージェントを再開させるため、エージェントは引き継ぎを調べ、本当に新しいフォローアップ作業が必要かを判断できます。タスクの完了を受動的なままにしたい場合は`auto_subscribe_on_create: false`を設定してください。来歴として、組み込みのデコンポーザーの子は`created_by=auto-decomposer`を使い、再開されたプロファイルが作成したタスクには代わりにそのプロファイル名が付きます。

2つのモードは、Kanbanページ上部の**Orchestration: Auto/Manual**のピル（エメラルド = Auto、くすんだグレー = Manual）で切り替えるか、`config.yaml`を直接編集して切り替えます。どちらのモードも`hermes kanban specify`と共存します — ファンアウトが不要なときの単一タスクの仕様の書き換えとして、引き続き利用できます。

デコンポーザーのルーティングの判断はプロファイルの説明に依存します。これはプロファイルごとのラベル付けのプリミティブで、`hermes profile create --description "..."`、`hermes profile describe <name> --text "..."`、`hermes profile describe <name> --auto`（プロファイルのインストール済みスキル＋モデルからLLMが生成）、または展開された**Orchestration settings**パネルにあるダッシュボードのプロファイルごとのエディターで設定します。説明のないプロファイルも一覧には表示されます — 名前でルーティングできますが、精度は下がります。デコンポーザーが`assignee=None`の子タスクを置くことは決してありません: LLMが未知のプロファイルを選んだ場合、子は`kanban.default_assignee`に、なければルートタスクの担当者（既存のプロファイルを指している場合）に、なければアクティブなデフォルトプロファイルにルーティングされます。

`kanban.orchestrator_profile`は、そのプロファイルのプロンプト、スキル、カスタムロジックを分解の呼び出しに読み込むものではありません。これが制御するのは、ファンアウトの後に誰がルート/オーケストレーションのタスクを所有するかです。デコンポーザーのモデル/プロバイダーを変更するには、`auxiliary.kanban_decomposer`を設定します。組み込みのデコンポーザーの代わりにプロファイル独自のタスク分割のロジックを使うには、Manualモードに切り替えて、そのプロファイルに明示的にタスクを作成または分解させてください。

設定のつまみ（すべて`~/.hermes/config.yaml`の`kanban:`の下）:

| キー | デフォルト | 目的 |
|---|---|---|
| `auto_decompose` | `true` | ディスパッチャーが毎ティック、Triageのタスクに対して組み込みのデコンポーザーを自動実行する。プロファイル主導の`kanban_create`の呼び出しや作成者の起床のターンをゲートするものではない。 |
| `auto_decompose_per_tick` | `3` | ディスパッチャーのティックあたりの分解数の上限。超過分は次のティックに持ち越される。 |
| `orchestrator_profile` | `""` | 分解後にルート/オーケストレーションのタスクに割り当てられるプロファイル。空 = ルートタスクは自身の担当者を維持し、なければアクティブなデフォルトプロファイル。 |
| `default_assignee` | `""` | LLMが未知のプロファイルを選んだときに子タスクが着地する先。空 = ルートタスクの担当者にフォールバックし、なければアクティブなデフォルト。 |
| `auto_subscribe_on_create` | `true` | 永続的なゲートウェイ/TUIセッションの中で`kanban_create`が実行されると、終端イベントが合成的なステータスのターンでその元のエージェントを再開させる。受動的な完了にするか、明示的な`kanban_notify-subscribe`の呼び出しを必須にするには`false`に設定する。`auto_decompose`とは独立。 |
| `notify_in_gateway` | `true` | このゲートウェイからKanbanの購読をポーリングして配信する。通知の購読を所有しないプロファイルでは、アイドル時の5秒ごとの通知機能のポーリングを止めるために`false`に設定する。`dispatch_in_gateway`とは独立。ディスパッチしないゲートウェイも、プロファイル固有の配信アダプターを所有している場合がある。 |
| `done_sub_retention_days` | `30` | 通知の購読は`done`を越えて残り（再オープンに安全）、`archived`で削除される。通知機能のGCは、タスクが`done`または`blocked`のまま新しいイベントがこの日数ない購読を削除し、アーカイブしないボードでの購読テーブルの増加を抑える。`0`でスイープを無効化。 |

そして2つの補助LLMのスロット:

| キー | 目的 |
|---|---|
| `auxiliary.kanban_decomposer` | タスクグラフを生成するモデル（Decomposeが呼び出す）。メインのチャットモデルを上書きするには`provider`/`model`を設定する。 |
| `auxiliary.profile_describer` | プロファイルの説明を自動生成するモデル（`hermes profile describe --auto`が呼び出す）。 |

### アーキテクチャ

GUIは厳密に、独自のドメインロジックを持たない**DB経由の読み取り＋kanban_db経由の書き込み**のレイヤーです。

<!-- ascii-guard-ignore -->
```
┌────────────────────────┐      WebSocket (tails task_events)
│   React SPA (plugin)   │ ◀──────────────────────────────────┐
│   HTML5 drag-and-drop  │                                    │
└──────────┬─────────────┘                                    │
           │ REST over fetchJSON                              │
           ▼                                                  │
┌────────────────────────┐     writes call kanban_db.*        │
│  FastAPI router        │     directly — same code path      │
│  plugins/kanban/       │     the CLI /kanban verbs use      │
│  dashboard/plugin_api.py                                    │
└──────────┬─────────────┘                                    │
           │                                                  │
           ▼                                                  │
┌────────────────────────┐                                    │
│  ~/.hermes/kanban.db   │ ───── append task_events ──────────┘
│  (WAL, shared)         │
└────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

### REST接点

すべてのルートは`/api/plugins/kanban/`配下にマウントされ、ダッシュボードの一時的なセッショントークンで保護されます。

| メソッド | パス | 目的 |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | ステータス列ごとにグループ化したボード全体、加えてフィルタードロップダウン用のテナント＋担当者 |
| `GET` | `/tasks/:id` | タスク＋コメント＋イベント＋リンク |
| `POST` | `/tasks` | 作成（`kanban_db.create_task`をラップし、`triage: bool`と`parents: [id, …]`を受け付ける） |
| `PATCH` | `/tasks/:id` | ステータス / 担当者 / 優先度 / タイトル / 本文 / 結果 |
| `POST` | `/tasks/bulk` | `ids`内のすべてのidに同じパッチ（ステータス / アーカイブ / 担当者 / 優先度）を適用。idごとの失敗は兄弟を中断せずに報告 |
| `POST` | `/tasks/:id/comments` | コメントを追記 |
| `POST` | `/tasks/:id/specify` | トリアージのspecifierを実行 — 補助LLMがタスク本文を肉付けし、`triage`から`todo`に昇格させる。`{ok, task_id, reason, new_title}`を返す。「トリアージにない」/ 補助クライアントなし / LLMエラーの場合の人間可読な理由付きの`ok=false`は、4xxではなく200 |
| `POST` | `/tasks/:id/decompose` | Kanbanのデコンポーザーを実行 — 補助LLMがタスクグラフを生成し、ヘルパーがアトミックに子を作成し、ルートをリンクして、`triage → todo`に切り替える。`{ok, task_id, reason, fanout, child_ids, new_title}`を返す。LLMエラーでも200を返す慣例は`/specify`と同じ。 |
| `GET` | `/profiles` | インストール済みのプロファイルを説明とともに一覧（ダッシュボードのプロファイル説明エディターとオーケストレーターのピッカーが使用）。 |
| `PATCH` | `/profiles/:name` | プロファイルの説明を設定またはクリア（ユーザーが書いたもの — `description_auto: false`）。`{ok, profile, description}`を返す。 |
| `POST` | `/profiles/:name/describe-auto` | `auxiliary.profile_describer`でプロファイルの説明を生成。ダッシュボードが「review」バッジを表示できるよう、`description_auto: true`で永続化する。 |
| `GET` | `/orchestration` | Kanbanのオーケストレーション設定（`orchestrator_profile`、`default_assignee`、`auto_decompose`）と、フォールバック後の*解決済みの*実効値を読む。 |
| `PUT` | `/orchestration` | `config.yaml`の3つのオーケストレーションのキーの1つ以上を更新。空でないプロファイル名が実際に存在するかを検証する。 |
| `POST` | `/links` | 依存関係（`parent_id` → `child_id`）を追加 |
| `DELETE` | `/links?parent_id=…&child_id=…` | 依存関係を削除 |
| `POST` | `/dispatch?max=…&dry_run=…` | ディスパッチャーをつつく — 60秒の待機をスキップ |
| `GET` | `/config` | `config.yaml`から`dashboard.kanban`の設定を読む — `default_tenant`、`lane_by_profile`、`include_archived_by_default`、`render_markdown` |
| `WS` | `/events?since=<event_id>` | `task_events`の行のライブストリーム。`since`がない場合、ストリームはボードの現在の末尾から始まる（`/board`のスナップショットがすでに過去を持っている）。そこから追いつくには`since=<latest_event_id>`を、履歴をリプレイするには`since=0`を渡す |

各ハンドラーは薄いラッパーです — プラグインは約700行のPython（ルーター＋WebSocketの追従＋一括バッチャー＋設定リーダー）で、新しいビジネスロジックを一切追加しません。小さな`_conn()`ヘルパーが、すべての読み書きで`kanban.db`を自動初期化するため、ユーザーが最初にダッシュボードを開いても、RESTを直接叩いても、`hermes kanban init`を実行しても、新規インストールが動作します。

### ダッシュボードの設定

`~/.hermes/config.yaml`の`dashboard.kanban`配下のこれらのキーは、タブのデフォルトを変更します — プラグインはロード時に`GET /config`経由でそれらを読みます。

```yaml
dashboard:
  kanban:
    default_tenant: acme              # preselects the tenant filter
    lane_by_profile: true             # default for the "lanes by profile" toggle
    include_archived_by_default: false
    render_markdown: true             # set false for plain <pre> rendering
```

各キーは任意で、示されたデフォルトにフォールバックします。

### セキュリティモデル

ダッシュボードのHTTP認証ミドルウェアは[`/api/plugins/`を明示的にスキップします](./extending-the-dashboard#backend-api-routes) — ダッシュボードはデフォルトでlocalhostにバインドするため、プラグインのルートは設計上、認証されません。つまり、kanbanのREST接点はホスト上の任意のプロセスから到達可能です。

WebSocketはもう1段階の手順を踏みます: ダッシュボードの一時的なセッショントークンを`?token=…`クエリパラメータとして要求します（ブラウザはアップグレードリクエストに`Authorization`を設定できないため）。これは、ブラウザ内のPTYブリッジが使うパターンと一致します。

`hermes dashboard --host 0.0.0.0`を実行すると、kanbanを含むすべてのプラグインのルートがネットワークから到達可能になります。**共有ホストでこれを行わないでください。** ボードにはタスク本文、コメント、ワークスペースパスが含まれます。これらのルートに到達した攻撃者は、あなたの協調接点全体への読み取りアクセスを得て、タスクの作成 / 再割り当て / アーカイブも行えます。

`~/.hermes/kanban.db`内のタスクは、意図的にプロファイル非依存です（それが協調プリミティブです）。`hermes -p <profile> dashboard`でダッシュボードを開いても、ボードにはホスト上の他のどのプロファイルが作成したタスクも依然として表示されます。すべてのプロファイルを同じユーザーが所有していますが、複数のペルソナが共存する場合は知っておく価値があります。

### ライブ更新

`task_events`は単調増加の`id`を持つ、追記専用のSQLiteテーブルです。WebSocketエンドポイントは各クライアントの最後に見たイベントidを保持し、新しい行が着地するたびにプッシュします。イベントのバーストが到着すると、フロントエンドは（非常に安価な）ボードエンドポイントをリロードします — すべてのイベント種類からローカル状態をパッチしようとするよりもシンプルで正確です。WALモードのおかげで、読み取りループがディスパッチャーの`BEGIN IMMEDIATE`クレームトランザクションをブロックすることはありません。

### 拡張

このプラグインは標準のHermesダッシュボードプラグインの契約を使います — マニフェストの完全なリファレンス、シェルスロット、ページスコープのスロット、Plugin SDKについては[ダッシュボードの拡張](./extending-the-dashboard)を参照してください。追加の列、テナントでフィルターしたレイアウト、完全な`tab.override`置換は、いずれもこのプラグインをフォークせずに表現できます。ダッシュボードにはカードごとのスロットはありません。デスクトップのボードでは、プラグインは`KANBAN_CARD_AREAS.badges`の貢献エリアを通じてカードごとのラベルを追加します（[Desktop Plugin SDK](/docs/developer-guide/desktop-plugin-sdk#kanban-card-badges)を参照）。

削除せずに無効化するには: `config.yaml`に`dashboard.plugins.kanban.enabled: false`を追加する（または`plugins/kanban/dashboard/manifest.json`を削除する）。

### スコープの境界

GUIは意図的に薄くしています。プラグインが行うことはすべてCLIから到達可能であり、プラグインは人間にとって快適にするだけです。自動割り当て、予算、ガバナンスゲート、組織図ビューは、設計仕様のスコープ外セクションに列挙されているとおり、ユーザー空間（ルータープロファイル、別のプラグイン、`tools/approval.py`の再利用）のままです。

## CLIコマンドリファレンス

これは、**あなた**（またはスクリプト、cron、ダッシュボード）がボードを操作するために使う接点です。ディスパッチャー内で動作するワーカーは、同じ操作に`kanban_*`の[ツール接点](#how-workers-interact-with-the-board)を使います — ここでのCLIとそこでのツールはどちらも`kanban_db`を経由するため、2つの接点は構造的に一致します。

```
hermes kanban init                                     # create kanban.db + print daemon hint
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|worktree:<path>|dir:<path>]
                                [--branch <name>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--max-retries N]
                                [--goal] [--goal-max-turns N]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived]
        [--workflow-template-id <id>] [--current-step-key <key>]
        [--sort completed-desc|created|created-desc|priority|priority-desc|status|assignee|title|updated]
        [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # or 'none' to unassign
hermes kanban reassign <id>... <profile>               # bulk re-assign tasks to a profile
hermes kanban edit <id> [--title ...] [--body ...]     # edit task title / body / priority in place
        [--priority N]
hermes kanban promote <id>...                          # move todo/blocked tasks to ready (recovery)
hermes kanban schedule <id> --at <ISO8601>             # set/clear a task's scheduled_at start time
hermes kanban diagnostics [--json]                     # board health snapshot (alias: diag)
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# Bulk verbs — accept multiple ids:
hermes kanban complete <id>... [--result "..."] [--force]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban request-review <id> [--summary "..."] [--metadata JSON] [--reviewer PROFILE]
hermes kanban request-changes <id> "<required changes>"               # active reviewer -> implementer
hermes kanban reopen-review  <id>... [--reason "..."]                 # changes requested: 'review' -> ready/todo

hermes kanban tail <id>                                # follow a single task's event stream
hermes kanban watch [--assignee P] [--tenant T]        # live stream ALL events to the terminal
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # worker liveness signal for long ops
hermes kanban runs <id> [--json]                       # attempt history (one row per run)
hermes kanban assignees [--json]                       # profiles on disk + per-assignee task counts
hermes kanban dispatch [--dry-run] [--max N]           # one-shot pass
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # DEPRECATED — standalone dispatcher (use `hermes gateway start` instead)
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # per-status + per-assignee counts
hermes kanban log <id> [--tail BYTES]                  # worker log from ~/.hermes/kanban/logs/
hermes kanban notify-subscribe <id>                    # gateway bridge hook (used by /kanban in the gateway)
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
        [--chat-type dm|group|channel|thread] [--delivery-mode notify|notify+wake|wake]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # what a worker sees
hermes kanban specify [<id> | --all] [--tenant T]      # flesh out a triage-column idea
        [--author NAME] [--json]                       #   into a full spec and promote to todo
hermes kanban gc [--event-retention-days N]            # workspaces + old events + old logs
        [--log-retention-days N]                       #   (negative N is rejected; 0 disables that sweep)
```

すべてのコマンドは、インタラクティブCLIおよびメッセージングゲートウェイのスラッシュコマンドとしても利用できます（下記の[`/kanban`スラッシュコマンド](#kanban-slash-command)を参照）。

`--max-retries`は、ディスパッチャーに対するタスクごとのサーキットブレーカーの上書きです。`--max-retries 1`は最初の成功しなかった試行でタスクをブロックし、`--max-retries 3`は2回のリトライを許して3回目の失敗でブロックします。省略すると`config.yaml`の`kanban.failure_limit`、次に組み込みのデフォルトが使われます。

### 並行数、スケジューリング、子の昇格の設定

| 設定キー | デフォルト | 何をするか |
|------------|---------|--------------|
| `kanban.max_in_progress` | 未設定（無制限） | 同時に実行されるタスクの数の上限。ボードにすでにN個の実行中のタスクがあると、ディスパッチャーはそれ以上の起動をスキップします — 遅いワーカー（ローカルLLM、リソースの限られたホスト）が、さらに積み上がってタイムアウトする前に手持ちを終えられるようにするのに便利です。無効な値や1未満の値は警告をログに出力し、無制限として振る舞います。 |
| `kanban.max_in_progress_per_profile` | 未設定（無制限） | `max_in_progress`のプロファイルごとの版 — 1つの担当者プロファイルが同時に実行できるタスクの数の上限。1つのプロファイルが遅いかレート制限されているが、他は流し続けたい場合に便利です。ボード全体の`max_in_progress`と並んで適用され、起動が進むには両方が許可する必要があります。 |
| `kanban.dispatch_profiles` | 未設定（既存の任意のプロファイル） | Hermesホームをまたいで共有されるボード向けの、ホームごとのクレームの許可リスト。キーが存在する場合、このホームのディスパッチャーは担当者が列挙されているカードだけをクレームします — フェイルクローズ: 空のリスト、`null`、値のない`dispatch_profiles:`は何もクレームせず、設定の読み取りに失敗した場合は警告をログに出力して何もクレームしません。他の担当者は`skipped_nonspawnable`に入ります。キーを省略した場合にのみ「既存の任意のプロファイル」を意味します。`hermes kanban diagnostics`は、このホームの解決済みの値（`any`、列挙された名前、または`none (fail-closed: …)`）を表示します。[ホームをまたいだ共有ボード](#shared-boards-across-homes)を参照してください。 |
| `kanban.auto_promote_children` | `true` | `decompose_triage_task()`が親のブロッカーとなる依存のない子を生成した後、それらは自動的に`ready`に昇格され、ディスパッチャーが拾えるようになります。手動のレビューを必須にするには`false`に設定します — 子はあなたが昇格させるまで`todo`に留まります。 |
| `kanban.default_workdir` | 未設定 | `--workspace`もタスク自体も上書きしない場合に新しいタスクに適用される、ボードレベルのデフォルトの作業ディレクトリ。タスクごとの`workspace:`が引き続き優先されます。 |

```yaml
kanban:
  max_in_progress: 2
  auto_promote_children: false
  default_workdir: ~/work/active-project
```

### スケジュールされたタスクの開始（`scheduled_at`）

タスクに`scheduled_at`を設定すると、特定の時刻までディスパッチを遅らせられます。ディスパッチャーは`scheduled_at`が未来のreadyタスクをスキップし、そのタイムスタンプの後の最初のティックで拾います。

```bash
hermes kanban create "nightly backup audit" \
  --assignee ops --scheduled-at "2026-06-01T03:00:00Z"
```

### リスポーンガード

ディスパッチャーは、前回の実行でクォータ/認証/429のエラーに当たった場合（`blocker_auth`）、ガードのウィンドウ内で実行が成功裏に完了した場合（`recent_success`）、または最近のタスクのコメントがGitHubのPRにリンクしている場合（`active_pr`）、readyタスクの再起動を拒否します。2つのクールダウンは、カードに不利に数えることなくカードを保留します: クォータの壁による再キューの後の`rate_limit_cooldown`と、ホストがワーカーの配置を拒否した後の`infrastructure_cooldown`（再起動に強いsystemdのスコープがない — [ワーカーとsystemdのcgroup](#workers-and-systemd-cgroups)を参照）です。どちらも`HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`のウィンドウ（デフォルト300秒）を共有します。これにより、人間が追いつくまでの間、同じバグやタスクに対してワーカーが繰り返し殺到することを防ぎます。[イベントリファレンス](#event-reference)の`respawn_guarded`の行を参照してください。

readyのカードが起動されない理由を確認するには、`hermes kanban dispatch --dry-run`を実行します — 出力には保留されたカードごとに`Guarded (<reason>): <task id>`が列挙されます（`--json`では`respawn_guarded`、`rate_limited`、`skipped_locked`、`memory_pressure`も）。ゲートウェイとスタンドアロンのデーモンの「dispatcher stuck」の警告も、直前のティックが何を保留したかを示します。例: `Last tick held back: active_pr=1`。

`recent_success`と`active_pr`が保留するのは**ready**のレーンだけです — それらはレビューの引き継ぎへの入力であって、引き継ぎに反するシグナルではありません。PRがすでに開かれているカードをレビュアー、クローザー、その他の回復用のプロファイルに拾わせるには、`hermes kanban request-review <id>`でレビューのレーンに移す（`running`と同様に`ready`からも受け付けられ、レビューのレーンの起動はどちらのガードの対象にもなりません）か、`hermes kanban assign <id> <profile>`でreadyのカードをそのプロファイルに渡します: PRのコメントの*後に*記録された引き継ぎ — オペレーターによる再割り当て、レビュアーの変更要求の判定、レビューの再オープン — は、そのPRこそがまさに作業対象であるため、現在カードに指定されているプロファイルについて`active_pr`を解除します。*別の*プロファイルへの変更だけが数えられます: 同じプロファイルへの再割り当て、割り当て解除、ディスパッチャー自身による`kanban.default_assignee`の補完ではガードは解除されないため、PRを開いた担当者が、クラッシュ、再取得、何もしない再割り当ての後にそれに対して再起動されることはなく、引き継ぎの後に投稿された新しいPRのコメントは再びガードします。成功の後の意図的な再キュー（`done→ready`へのドラッグ、`unblock`、再昇格）も`recent_success`を解除するため、手動の再実行がウィンドウの間ずっと黙って保留されることはありません。

### ドラッグでの削除と一括削除（ダッシュボード）

ダッシュボードのKanbanページには**ゴミ箱のドロップゾーン**があります — 任意のカードをそこにドラッグするとタスクが削除されます（`task_events`、子のリンク、購読まで連鎖します）。確認のプロンプトが誤操作を防ぎます。一括削除は、JSONボディ`{"ids": ["t_abc", "t_def", ...]}`を伴う`DELETE /api/plugins/kanban/tasks`でも利用できます。

### ワーカーの可視化エンドポイント

ダッシュボードプラグインのAPIは、外部のモニター向けに次の読み取り専用のエンドポイント（と実行制御の動詞1つ）を公開するようになりました:

| エンドポイント | 返すもの |
|----------|---------|
| `GET /api/plugins/kanban/workers/active` | 現在起動されているワーカー（PID、プロファイル、タスクid、開始時刻、最後のハートビート） |
| `GET /api/plugins/kanban/runs/{id}` | 単一の実行の詳細 — タスクid、ステータス、開始/終了、終了コード、ログのパス |
| `POST /api/plugins/kanban/runs/{run_id}/terminate` | 再取得可能な実行を終了させる — ワーカーを停止し、タスクを再ディスパッチのために解放する |
| `GET /api/plugins/kanban/inspect` | ディスパッチャーの統合スナップショット — バックログ、`max_in_progress`に対する実行中の数、最近のイベント |

これらはすべて、Kanbanプラグインの他のAPIと同じダッシュボードプラグインの認証でゲートされます。

### Kanban Swarmのトポロジーヘルパー

`hermes kanban swarm`は、永続的な**Kanban Swarm v1**のグラフを一度に作成します: 完了済みのルート/ブラックボードのカード、N個の並列ワーカーのカード、すべてのワーカーでゲートされたベリファイアーのカード、ベリファイアーでゲートされたシンセサイザーのカードです。共有のスウォームのコンテキスト（「ブラックボード」）は、ルートカード上の構造化されたJSONのコメントとして保存されるため、どのワーカーもそれを読めます。

```bash
hermes kanban swarm "Design a multi-region failover plan" \
  --workers researcher,architect,sre \
  --verifier reviewer --synthesizer writer
```

生成されたグラフはアトミックにコミットされます: ディスパッチャーとダッシュボードの読み手に見えるのは、新しいスウォームがないか完全なトポロジーのどちらかであり、部分的にリンクされたルート/ワーカー/ベリファイアーのグラフが見えることは決してありません。その後は通常どおりディスパッチされます — ワーカーは並列に実行され、それらがすべて終わるとベリファイアーが起き、ベリファイアーが作業をクリーンとマークするとシンセサイザーが起きます。

## `/kanban`スラッシュコマンド {#kanban-slash-command}

すべての`hermes kanban <action>`動詞は、`/kanban <action>`としても到達できます — インタラクティブな`hermes chat`セッションの内部から**および**任意のゲートウェイプラットフォーム（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、メール、SMS）から。どちらの接点も、`hermes kanban`のargparseツリーを再利用するまったく同じ`hermes_cli.kanban.run_slash()`エントリポイントを呼び出すため、引数の接点、フラグ、出力フォーマットはCLI、`/kanban`、`hermes kanban`をまたいで同一です。ボードを操作するためにチャットを離れる必要はありません。

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # flesh out a triage one-liner into a real spec
/kanban specify --all --tenant engineering  # sweep every triage task in one tenant
```

複数語の引数は、シェルと同じようにクオートしてください — `run_slash`は行の残りを`shlex.split`でパースするため、`"..."`と`'...'`の両方が動作します。

### 実行中の使用: `/kanban`は実行中エージェントのガードをバイパスする

ゲートウェイは通常、エージェントがまだ思考中の間はスラッシュコマンドとユーザーメッセージをキューに入れます — それが、最初のターンが進行中に誤って2つ目のターンを開始してしまうのを止める仕組みです。**`/kanban`はこのガードから明示的に除外されています。** ボードは実行中エージェントの状態ではなく`~/.hermes/kanban.db`に存在するため、読み取り（`list`、`show`、`context`、`tail`、`watch`、`stats`、`runs`）も書き込み（`comment`、`unblock`、`block`、`assign`、`archive`、`create`、`link`など）も、ターンの途中であってもすべて即座に通ります。

これがこの分離の本質です。

- ワーカーがピアを待ってブロックしている → あなたはスマホから`/kanban unblock t_abcd`を送り、ディスパッチャーは次のティックでそのピアを拾い上げます。ブロックされていたワーカーは中断されません — ただブロックされなくなるだけです。
- 人間のコンテキストが必要なカードを見つけた → `/kanban comment t_xyz "use the 2026 schema, not 2025"`がタスクスレッドに着地し、そのタスクの*次の*実行が`kanban_show()`でそれを読みます。
- オーケストレーターを止めずにフリートが何をしているか知りたい → `/kanban list --mine`や`/kanban stats`が、メインの会話に触れずにボードを調べます。

### `/kanban create`での自動購読（ゲートウェイのみ）

ゲートウェイから`/kanban create "…"`でタスクを作成すると、発信元のチャット（プラットフォーム＋チャットid＋スレッドid）が、そのタスクの終端イベント（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）に自動的に購読されます。終端イベントごとに1つのメッセージが返ってきます — `completed`時にはワーカーの結果サマリーの最初の行も含めて — ポーリングしたりタスクidを覚えておいたりする必要はありません。

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

購読は、タスクが`done`に達しても残ります — 完了は取り消せる（レビュアーやコントローラーはdoneのタスクを再オープンできる）ため、発信元のセッションは再オープンのサイクルを通じて通知を受け続けます。`archived`（取り消せない最終状態）で自動的に削除されます。アーカイブしないボードでは、GCのスイープが、`done`または`blocked`のまま新しい活動が`kanban.done_sub_retention_days`日（デフォルト30。0で無効化）ないタスクの購読を削除するため、古い行が永遠に蓄積することはありません。`--json`（機械出力）で作成をスクリプト化すると、自動購読はスキップされます — スクリプト化された呼び出し元は`/kanban notify-subscribe`経由で購読を明示的に管理したい、という前提です。

`kanban_create`または`hermes kanban create`でタスクを作成するディスパッチャーのワーカーは、
`parents`の依存リンクがなくても、所有するタスクの永続的な通知の購読をコピーします。
宛先、ルートのアンカー、配信モードは保持されます。
受動的な購読が、自動購読によって起床に格上げされることはありません。これは
既存の購読を`auto_subscribe_on_create`とは独立にコピーするもので、後者が制御するのは
現在の会話を新しい宛先として追加することです。素のCLIセッションや、所有するタスクに購読がない
ワーカーに対して、宛先がでっち上げられることはありません。

`kanban_create`では、セッションの系譜は次の順に解決されます: 明示的な`session_id`、
所有するワーカーのタスクの永続的なセッション、リクエストスコープのAPIの発信元、そして
現在のセッション。候補のidが刻まれるのは、アクティブな
プロファイルの`state.db`の`sessions`テーブルに行がある場合だけです。どこにも解決されないid（行が
一度も書き込まれなかったセッション、または別のプロセスから継承されたid）の場合、
どのデータベースでも解決できないセッションを指す代わりに、`session_id`はNULLのままになります。
組み込みの分解も、そのルートの永続的なセッションを継承します。セッションの系譜それ自体は通知の宛先ではありません:
`session_id`を変更しても既存の購読は置き換えられません。イベントの配信先を変えるには
`notify-subscribe`と`notify-unsubscribe`を使ってください。

チャットから発生した自動購読は`notify+wake`モードで作成されます: 終端イベントで、宛先のエージェントは受動的なメッセージを受け取る**と同時に**本当のターンを取るため、ボードのコンテキストを読んで自分の言葉で返信できます。下記の[配信モード](#delivery-modes)を参照してください。

### メッセージングでの出力の切り詰め

ゲートウェイプラットフォームには実用的なメッセージ長の上限があります。`/kanban list`、`/kanban show`、`/kanban tail`が約3800文字を超える出力を生成した場合、レスポンスは`… (truncated; use \`hermes kanban …\` in your terminal for full output)`というフッターとともに切り詰められます。CLI接点にはそのような上限はありません。

### オートコンプリート

インタラクティブCLIで`/kanban `と入力してTabを押すと、組み込みのサブコマンドリスト（`list`、`ls`、`show`、`create`、`assign`、`link`、`unlink`、`claim`、`comment`、`complete`、`block`、`unblock`、`archive`、`tail`、`dispatch`、`context`、`init`、`gc`）を循環します。上記のCLIリファレンスに列挙された残りの動詞（`watch`、`stats`、`runs`、`log`、`assignees`、`heartbeat`、`notify-subscribe`、`notify-list`、`notify-unsubscribe`、`daemon`）も動作します — まだオートコンプリートのヒントリストに入っていないだけです。

## 協調パターン {#collaboration-patterns}

ボードは、新しいプリミティブなしでこれら8つのパターンをサポートします。

| パターン | 形状 | 例 |
|---|---|---|
| **P1 ファンアウト** | 同じロールのN個の兄弟 | "5つの角度を並列でリサーチ" |
| **P2 パイプライン** | ロールの連鎖: scout → editor → writer | 日次ブリーフの組み立て |
| **P3 投票 / クォーラム** | N個の兄弟 + 1個の集約者 | 3人のリサーチャー → 1人のレビュアーが選ぶ |
| **P4 長期ジャーナル** | 同じプロファイル + 共有ディレクトリ + cron | Obsidian vault |
| **P5 ヒューマンインザループ** | ワーカーがブロック → ユーザーがコメント → unblock | 曖昧な判断 |
| **P6 `@mention`** | 散文からのインラインルーティング | `@reviewer look at this` |
| **P7 スレッドスコープのワークスペース** | スレッド内の`/kanban here` | プロジェクトごとのゲートウェイスレッド |
| **P8 フリート運用** | 1つのプロファイル、N個の対象 | 50個のソーシャルアカウント |
| **P9 トリアージのspecifier** | ラフなアイデア → `triage` → `hermes kanban specify`が本文を展開 → `todo` | "この一行を仕様化されたタスクに変える" |

## フォローアップのカードへのコンテキストの受け渡し（親リンク）

親リンクは単なるスケジューリングのゲートではありません — **完了した**カードから新しいカードへのコンテキストの引き継ぎのチャネルです。`--parent <done-card-id>`でカードを作成すると、2つのことが起こります:

1. **すぐに対象になります。** `create_task`は親の状態に応じてステータスを設定します: 親がすべて`done`の子は、直接`ready`で作成されます — 待機も手動の昇格もありません。（まだ開いている親の子は、最後の親が終わったときに`recompute_ready`が昇格させるまで`todo`に留まります。）
2. **親の引き継ぎが一緒に運ばれます。** 子のために組み立てられるワーカーのコンテキスト（`build_worker_context`、`kanban_show()`が返すもの）には、各親の完了時の`summary`と`metadata`をそのまま含む`## Parent task results`セクションがあります:

```
## Parent task results
### t_77c26979 (completed just now)
Added exponential backoff with jitter to the retry helper.
_metadata_: `{"changed_files": ["hermes_cli/retry.py", "tests/test_retry.py"], "decisions": ["capped backoff at 60s", "jitter = full"]}`
```

これが、完了したカードに対するフォローアップ作業のパターンが**doneのカードの再オープンではなく、新しい子カード**である理由です。完了したカードは不変の履歴です — そのコンテキストは親リンクを通じて前方に流れます。同じカードでのやり直し（失敗しているカードのリトライのループ）は別の仕組みです: *同じ*カードでの以前の試行は、そのカード自身のコンテキストに「prior attempts」として現れます。

worktreeやブランチだけでは代わりになりません: リポジトリの状態はフォローアップのワーカーにコードが*どう*なっているかを伝えますが、*なぜ*かは伝えません — 判断、実行したテスト、触れたファイルは、gitではなく親の構造化された引き継ぎの中にあります。親が完了した時点では存在しなかったエビデンス（例: 後で失敗したCIのログ）は、新しいカードの**本文**に入れます。

```bash
# Implementation card t_impl is done. CI fails two hours later.
hermes kanban create "Fix CI failure from t_impl: test_retry flakes on 3.11" \
    --assignee coder \
    --parent t_impl \
    --body "$(cat <<'EOF'
CI run #4812 failed after t_impl merged.
Log excerpt: FAILED tests/test_retry.py::test_backoff_jitter - TimeoutError
Acceptance: tests/test_retry.py green on 3.11 and 3.12 in CI.
Use a fresh worktree/branch; do not force-push the original branch.
EOF
)"
```

修正のワーカーは、元のカードのサマリーとmetadata（変更されたファイル、判断）に加えて、本文に入れた新しいエビデンスをすでにコンテキストに持った状態で起動されます。

### 衝突するワーカーのブランチの調停

エンジニアリングのパイプライン（worktreeを使うP1/P2）では、2つのワーカーのブランチが
マージ時に衝突することがあります。どちらのワーカーにも自分で裁定させてはいけません — 衝突している
エージェントは相手のコンテキストを持たず、確実に相手側を上書きするか
自分側を放棄します。代わりに、**第三の、中立なプロファイル**に割り当てた調停カードを作成し、
衝突した**両方の**カードを親としてリンクします: 親リンクが
両方の側の完了サマリーを調停者のコンテキストに運ぶため、
それは両方の差分*と*両方の意図を受け取ります。同梱の
[`agent-merge-conflict-arbiter`オプションスキル](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/autonomous-ai-agents/agent-merge-conflict-arbiter/SKILL.md)
は、そのワーカーに完全な手順を与えます: 衝突した各ハンクを分類し、
公平に解決し、検証し、すべての判断を明記したサマリーを返します。

### 並列のキャンペーンにおける衝突のホットスポット

大規模なキャンペーンでは、一部のファイルが衝突の磁石になります: 多くのワーカーがそれぞれ
同じファイルに少しずつ追加し、それを小さく保つ責任者は誰もおらず、そのファイルは
絶え間ないマージの衝突の場になります。緩和策はコメントの慣例であり、新しい
プリミティブではありません。自分の差分が1つのファイルで兄弟と衝突し続けていること —
あるいは自分が触れるファイルが他のカードの最近の
コメントに繰り返し現れること — に気づいたワーカーは、黙って積み重ねるべきではありません。代わりに、自分の
カードに識別しやすいプレフィックスを付けたコメントを残します:

```
hotspot: hermes_cli/kanban_db.py — third conflicting edit to the dispatch loop this wave
```

そして完了時の`metadata`でもそのフラグを繰り返します。**同じパスを指す`hotspot:`のコメントが2つ以上**あるのを
見たオーケストレーター（またはボードをレビューしている人間）は、
そのファイルに触れる作業をさらにキューに入れる**前に**、そのファイル専用のリファクタリング/分解のカードを作成すべきです
— 磁石となっているファイルを分割する方が、それが将来引き起こすすべての衝突を調停するより
安上がりです。*すでに*起きてしまった衝突には、上記の調停カードのパターンと
`agent-merge-conflict-arbiter`オプションスキルを使ってください。ホットスポットのフラグ付けは、
調停者が常設のレーンになってしまうのを防ぐ上流での対策です。

## マルチテナントの利用

1つのスペシャリストフリートが複数のビジネスにサービスを提供する場合、各タスクにテナントをタグ付けします。

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

ワーカーは`$HERMES_TENANT`を受け取り、メモリの書き込みをプレフィックスで名前空間化します。ボード、ディスパッチャー、プロファイル定義はすべて共有され、スコープされるのはデータだけです。

## デスクトップ通知

DesktopアプリのKanbanプラグインは、同じ終端イベントをネイティブに表示します — ゲートウェイのプラットフォームは不要です。Kanbanボードのライブのイベントソケットが接続されている間、`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`、またはtriageへのルーティング（`block_loop_detected`）の各イベントは、ワーカーの引き継ぎ（サマリー、ブロックの理由、またはエラー）と「Open Kanban」アクションを伴うアプリ内のトーストを表示します。Hermesのウィンドウから離れているときは、同じイベントがネイティブのOS通知も発火させる（**Settings ▸ Notifications ▸ Plugin notifications**でゲート）ため、別のアプリにいる間にタスクがブロッカーに当たっても気づけます。

対象となる期間: デスクトップ通知はライブのイベントストリームに乗っているため、Kanbanプラグインが有効な状態でアプリが実行されている間にのみ発火します。アプリが閉じている間に届いたイベントは、次回の起動時に通知としてリプレイされません — アプリが閉じていても届く必要がある配信には、ゲートウェイの購読（下記）を使ってください。

## ゲートウェイ通知

ゲートウェイ（Telegram、Discordなど）から`/kanban create …`を実行すると、発信元のチャットが新しいタスクに自動的に購読されます。ゲートウェイのバックグラウンド通知機能は数秒ごとに`task_events`をポーリングし、終端イベント（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）ごとに1つのメッセージをそのチャットに配信します。完了したタスクは、ワーカーの`--result`の最初の行も送るため、`/kanban show`しなくても結果が分かります。

CLIから購読を明示的に管理することもできます — スクリプト / cronジョブが、自身が発信元でないチャットに通知したい場合に便利です。

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7 \
    --chat-type group --delivery-mode notify+wake
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

購読は、タスクが`done`または`archived`に達すると自動的に自身を削除します。クリーンアップは不要です。

エージェントがゲートウェイのチャットの中から（ターミナルツールを通じて）`notify-subscribe`を実行し、
対象がそのチャット（とトピック）自身である場合、欠けている`--user-id`はそのターンの送信者で補われます。
`group_sessions_per_user`（デフォルト）のグループでは、これにより起床は2つ目の共有セッションではなく、
参加者自身のセッションで行われます。他のチャットやトピックへの購読が呼び出し元のidを取ることは決してなく、
明示的な`--user-id-alt`を与えられた購読も同様です。

**`profile_routes`のもとでのDiscordのスレッド:** ゲートウェイがプロファイルを多重化し、*チャンネル*を
プロファイルにルーティングしている場合、そのチャンネル内の*スレッド*に対してCLIから作成された購読には、スレッドのルートの
アンカーが必要です。そうでないと、通知機能はそれをどのルートにも照合できず、スキップします（WARNINGで1回ログに出力）。
明示的に渡してください — `--parent-chat-id <channel id>`と、Discordの場合は`--guild-id <guild id>`です:

```bash
hermes kanban notify-subscribe t_abcd \
    --platform discord --chat-id <thread id> --thread-id <thread id> --chat-type thread \
    --parent-chat-id <channel id> --guild-id <guild id> --delivery-mode notify+wake
```

チャットの中から作成された購読（`/kanban create`、`kanban_create`）は、これらのアンカーを自動的に記録します。
Telegramのフォーラムトピックや、SlackのDMのスレッドは、それ自身のチャットの中にある（`chat_id`はスレッドではなく
グループやDM）ため、`--parent-chat-id`は不要です: 通知機能は、そのチャットを指定したルートと照合します。
それには明示された`--chat-type`（`dm`、`group`、`channel`。ゲートウェイがあなたの代わりに記録します）が必要です。それのない行は
スレッド自身のidを`chat_id`として保存している可能性があるため、チャンネルのルートに届くには引き続き`--parent-chat-id`が必要です。

### 配信モード {#delivery-modes}

`--delivery-mode`は、通知機能が終端イベントに**どう**反応するかを制御します。すべての購読は3つのモードのいずれかです（`notify`がデフォルトで、元の挙動です）:

| モード | 受動的なメッセージ | エージェントを起こす | 使うとき |
|------|-----------------|-----------------|-------------|
| `notify` | はい | いいえ | チャットに知らせのメッセージがほしいだけの場合（デフォルト）。 |
| `notify+wake` | はい | はい | 宛先のエージェントにも本当のターンを取らせたい場合 — ボードのコンテキストを読み、自分の言葉で返信させる。チャットから発生した自動購読はこれを使う。 |
| `wake` | いいえ | はい | 別の通知なしに、エージェントにイベントに対応させたいだけの場合。 |

`notify+wake`では、受動的な通知が送信されることに加えて、起床がアダプターのターンのキューに受け付けられて初めて配信が完了します。ハンドラーの欠如、拒否されたルート、満杯のキューは、購読を期限切れにすることなく、後の通知機能のティックでリトライされます。送信済みの通知はSQLiteで別にチェックポイントされるため、拒否された起床が、すでにチェックポイントされた通知を繰り返すことはありません。`notify`は受動的なままで、ターンを開始することは決してありません。受け付けはモデルの実行や返信の成功を保証するものではなく、通常のターンのゲートは引き続き適用されます。これは厳密に1回の配信ではありません: 送信とそのチェックポイントの間でプロセスがクラッシュすると通知が繰り返されることがあり、既存の配信前にクレームするカーソルはクラッシュから回復可能なキューではありません。

「起床」は、宛先のゲートウェイのエージェントへの合成的な受信メッセージを作り出すため、エージェントは1行の受動的な通知を受け取る代わりに、通常のターンを取ります（コメントと結果を読み、推論し、返信する）。これが発火するのは、通知機能がライブのゲートウェイのプロセス内で実行されている場合だけです。そうでない場合、`notify+wake`の購読は引き続き受動的なメッセージを配信し、`wake`のみの購読はそのプロセスでは何もしません。

**どのイベントで起こすか。** タスクの結果を返すもの、またはオーケストレーションの注意を必要とするもの: `completed`、`blocked`、`gave_up`、`crashed`、`timed_out`、`review_requested`（ワーカーが実装を終えて`kanban_request_review`で引き継いだ）、`block_loop_detected`（繰り返しのブロックの後にタスクが`triage`にルーティングされた）です。`status`、`archived`、`unblocked`は配信されますが、起こすことは決してありません — それらは記帳上の遷移であり、注意のシグナルではありません。`completed`や`review_requested`のイベントがサマリーを持つ場合、その引き継ぎは起床のターンに乗るため、起こされたエージェントはワーカーが実際に何をしたかを見られます。

`--chat-type`（`dm` | `group` | `channel` | `thread`）は、発信元のチャットの種類を記録するため、起こされたターンはオペレーターの**本当の**セッションに解決されます: `build_session_key`はグループ、チャンネル、スレッドをDMとは異なるキーにするため、不正確な`chat_type`は起床を別の、コンテキストのないセッションにルーティングしてしまいます。`/kanban`の自動購読とスラッシュコマンドの経路はこれを自動的に取得します — 手で設定するのは、スクリプトやcronからチャットを購読させる場合だけです。省略すると既存の購読は変更されません（新しい購読のデフォルトは`dm`）。

### マルチプロファイルの構成: 配信はプロファイルが所有する

プロファイルごとに1つのゲートウェイを置くデプロイメント（1つのディスパッチャー、`writer`、`admin`などの別々のゲートウェイ
プロセス — [マルチゲートウェイの
ガイド](/docs/user-guide/features/kanban-multi-gateway)を参照）では、
ディスパッチと配信の所有者が別々です:

- **ディスパッチは単一の所有者のまま。** ちょうど1つのゲートウェイが
  `kanban.dispatch_in_gateway: true`を維持してディスパッチャーを実行し、他のすべての
  ゲートウェイはそれを`false`に設定します。
- **通知の配信はプロファイルが所有します。** ディスパッチしないものを含むすべてのゲートウェイが
  通知機能を実行し、自分がホストするプラットフォームアダプターを持つプロファイルが刻まれた購読だけを
  ポーリングします。`writer`プロファイルのTelegramから作成されたタスクの`completed`/`blocked`のメッセージは、
  ディスパッチを行ったのが`default`のゲートウェイであっても、`writer`のゲートウェイによって
  配信されます。
- **`gateway.profile_routes`で固定された多重化プロファイル**は、購読の永続化されたプラットフォーム、チャット、スレッド、スコープ、
  親チャンネルのアンカーが`gateway.profile_routes`を通じてまさにその提供されているプロファイルに解決され、
  かつそのプロファイルが購読のプラットフォーム用の独自のアダプターを持たない場合に、プライマリの
  アダプターを使えます。そのプラットフォーム用に接続されたセカンダリのアダプターは
  引き続き優先されます。プロファイルが*他の*プラットフォームで実行しているアダプターは
  配信を妨げません（共有のボットが、固定されたチャットにサービスを提供する唯一の認証情報であり、
  受信のターンと通知の両方で同様です）。一致しない、再割り当てされた、
  無効化された、またはあいまいなルートは、未配信のままでリトライ可能です。必須のルーティングのアンカーが
  欠けた古い行が、推測でプロファイルに割り当てられることはありません。起床のターンは、
  宛先のプロファイルのランタイムのスコープと、認可されたトランスポートを維持します。
- **ステートレス（`api_server`）の購読**は生のセッションidを持ちますが、これは
  どの`profile_routes`のエントリでもアンカーできません（セッションにはチャット/スレッド/ギルドの識別子が
  ありません）。提供されているプロファイルが認可されるのは、購読の
  セッションがまさにそのプロファイル自身の`state.db`にある場合です — 共有のリスナーは
  `/p/<profile>/`をミラーするため、所有権の証明はプラットフォームではなく、セッションストアです。その後、起床は
  そのプロファイルのランタイムのスコープで実行され、その正確なセッションをプロセス内で再開するため、
  2つ目のリスナーもセカンダリの
  `API_SERVER_KEY`も関与しません。不明なセッション、別の
  プロファイル用に刻まれたセッション、提供されていないプロファイルが所有するセッションは未配信のままリトライ可能であり、
  デフォルトプロファイル自身の`api_server`の購読には影響しません。
- プロファイルの刻印より前に作成された**レガシーの購読**（行に
  `notifier_profile`がないもの）は、実際のディスパッチャーのシングルトンロックを保持している
  ゲートウェイだけが配信するため、2つのゲートウェイがそれらを奪い合うことはありません。

ゲートウェイ間での重複配信は、ボードのDBにおけるイベントごとのアトミックなクレームによって防がれます。
リレー、認証情報の共有、追加のディスパッチャーは不要です
— 各プロファイルのゲートウェイは、単に自分自身のアダプターを通じて配信します。

## 実行（Runs） — 試行ごとに1行

正規のワーカーは、割り当てられたプロファイルのゲートウェイの所有者を通じて実行されます。ローカルの
トランスポートは、カスタムのパスや別のプロファイルに割り当てられた共有ボードを含め、
ディスパッチャーの正確なボードのデータベースのパスを運びます。別の
トランスクリプトの書き手を作ることはありません。所有者はタスク/実行/クレームを検証し、そのポリシーを固定します。

所有者や管理されたワーカーが**不明な**受け付けのまま消え、所有者がその試行にバインドした
インタープリター（その`worker_bound`イベント）がまだ実行中である可能性がある場合、
カードは実行とクレームを保持します: 期限切れのクレーム、古いハートビート、
ランタイムの制限によって、そのあいまいな試行が再実行されることはなく、所有者の台帳が
利用できない場合も再取得は一時停止します。そのバインドされたインタープリター自体が死ぬと、
試行をまだ実行しているものは何もないため、クラッシュのスイープは死んだワーカーと同様にそれを再取得します:
実行はcrashedとしてクローズされ、カードは通常のクラッシュの予算のもとで`ready`（または`review`）に戻ります。
これにより、失われた試行がすでに行っていた効果が再実行される可能性があります。
インタープリターを一度もバインドしなかった実行は、その所有者が受け付けを確定させた後
（失われたカードの試行の受け付けは`interrupted`で確定します）、同じように解放されます:
実行について記録されたすべてのワーカーのpidが死んでいる場合、または記録がない場合は
クレームが期限切れになった時点で、実行はcrashedとしてクローズされ（クレームがすでに
失効していた場合は再取得として）、カードは同じクラッシュの予算のもとで`ready`に
戻ります。不明な受け付けを明示的に解決する
（受け付けIDと実行の世代を伴う`prompt.resolve_unknown`）と、同じように確定するため、
別の試行につながる可能性があります。それを避けるには、効果を調べて、先にカードを完了またはブロックしてください。
ブロックされたカードをアンブロックするのは、新しい試行を意図している場合だけにしてください。確認されたタイムアウトとプロバイダーのレート制限の結果は、通常の
リトライ/失敗の計上の挙動を維持します。

タスクは作業の論理単位であり、**実行（run）**はそれを実行する1回の試行です。ディスパッチャーがready状態のタスクをクレームすると、`task_runs`に1行を作成し、`tasks.current_run_id`をそれに向けます。その試行が終わると — 完了、ブロック、クラッシュ、タイムアウト、スポーン失敗、再取得 — 実行の行は`outcome`とともにクローズされ、タスクのポインターはクリアされます。3回試行されたタスクには3つの`task_runs`の行があります。

タスクを単に変更するのではなく2つのテーブルにする理由: 実世界のポストモーテムには**完全な試行履歴**が必要であり（「2回目のレビュアーの試行はapproveまで到達し、3回目はマージした」）、また、どのファイルが変わったか、どのテストが実行されたか、レビュアーがどんな所見を残したかといった、試行ごとのメタデータをぶら下げるきれいな場所が必要です。それらは実行の事実であり、タスクの事実ではありません。

実行は、**構造化された引き継ぎ**が存在する場所でもあります。ワーカーが（`kanban_complete(...)`経由で）タスクを完了するとき、次を渡せます。

- `summary`（ツールパラメータ） / `--summary`（CLI） — 人間向けの引き継ぎ。実行に乗り、下流の子はそれを`build_worker_context`で見ます。
- `metadata`（ツールパラメータ） / `--metadata`（CLI） — 実行上の自由形式のJSON辞書。子はサマリーとともにシリアライズされたものを見ます。
- `result`（ツールパラメータ） / `--result`（CLI） — タスクの行に乗る短いログ行（レガシーフィールド、後方互換のため保持）。

下流の子は、各親について最新の完了した実行のサマリー＋metadataを読みます。リトライするワーカーは、自分のタスクの過去の試行（outcome、summary、error）を読むため、すでに失敗したパスを繰り返しません。

```
# What a worker actually does — a tool call, from inside the agent loop:
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

同じ引き継ぎは、ワーカーがクローズできないタスクを（あなた＝人間が）クローズする必要があるときに、CLIから到達できます — 例えば、放棄されたタスクや、ダッシュボードから手動でdoneにマークしたタスクなど。

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# Review the attempt history on a retried task:
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

実行はダッシュボード（ドロワー内のRun Historyセクション、試行ごとに1つの色付き行）とREST API（`GET /api/plugins/kanban/tasks/:id`が`runs[]`配列を返す）に公開されます。`{status: "done", summary, metadata}`を伴う`PATCH /api/plugins/kanban/tasks/:id`は両方をカーネルに転送するため、ダッシュボードの「mark done」ボタンはCLIと等価です。`task_events`の行は所属する`run_id`を持つため、UIはそれらを試行ごとにグループ化でき、`completed`イベントはペイロードに最初の行のサマリー（400文字に上限）を埋め込むため、ゲートウェイの通知機能は2回目のSQL往復なしで構造化された引き継ぎをレンダリングできます。

**一括クローズの注意点。** `hermes kanban complete a b c --summary X`は拒否されます — 構造化された引き継ぎは実行ごとであり、同じサマリーをN個のタスクにコピー&ペーストするのはほぼ常に誤りです。`--summary` / `--metadata`*なし*の一括クローズは、よくある「事務的なタスクの山を片付けた」ケースでは引き続き動作します。

**完了時の依存関係による拒否は親を示します。** 直接の親が`done`/`archived`でないカード（実行中に親が再オープンされた、または実行中の子の拒否より前のエッジ）に対する`kanban_complete` / `hermes kanban complete` / ダッシュボードの「mark done」と「request review」のアクション（単一と一括）は、一般的な「unknown id, stale run, or already terminal」のテキストではなく`unsatisfied parent dependencies: t_… (todo)`を報告し、カードは実行中のまま保たれます。`kanban_show`は同じ親を`unsatisfied_parents`の下に列挙し、`hermes kanban show` / `hermes kanban diagnostics` / ダッシュボードは、その状態の実行中のカードに`running_with_open_parents`の警告を出します。親を終わらせるか、`hermes kanban unlink <parent> <child>`を実行してください。依存関係のゲートを通る強制の経路はありません。

**完了時のライブクレームのガード。** ワーカーがライブのクレームを保持している`running`のタスクを完了させられるのは、そのワーカー（実行の内部からの`kanban_complete`）か、明示的なオペレーターによる上書き（`hermes kanban complete <id> --force`とダッシュボードの「mark done」アクション）だけです。クレームのない`hermes kanban complete <id>`やオーケストレーターのセッションの`kanban_complete`は、`--force` / `hermes kanban reclaim`への案内とともに拒否されるため、2つ目のセッションがライブのワーカーの足元でその実行をクローズすることはもうできません。クレームなしで`ready`、`blocked`、`review`のカードを完了させることは変わりません。

**ステータス変更からの再取得された実行。** ダッシュボードで実行中のタスクを`running`からドラッグして外す（`ready`に戻す、または直接`todo`へ）、あるいはまだ実行中のタスクをアーカイブすると、進行中の実行は孤立する代わりに`outcome='reclaimed'`でクローズされます。`tasks.current_run_id`が`NULL`のとき`task_runs`の行は常に終端状態にあり、その逆も成り立ちます — この不変条件はCLI、ダッシュボード、ディスパッチャー、通知機能をまたいで保たれます。

**一度もクレームされなかった完了のための合成実行。** 一度もクレームされなかったタスクを完了またはブロックする（例: 人間がダッシュボードから`ready`タスクをサマリー付きでクローズする、またはCLIユーザーが`hermes kanban complete <ready-task> --summary X`を実行する）と、本来は引き継ぎが失われてしまいます。代わりにカーネルは、試行履歴が完全なまま保たれるように、サマリー / metadata / 理由を持つ持続時間ゼロの実行の行（`started_at == ended_at`）を挿入します。`completed` / `blocked`イベントの`run_id`はその行を指します。

**ドロワーのライブ更新。** ダッシュボードのWebSocketイベントストリームが、ユーザーが現在表示しているタスクの新しいイベントを報告すると、ドロワーは自身をリロードします（タスクごとのイベントカウンターを`useEffect`の依存リストに通すことで）。実行の新しい行や更新されたoutcomeを見るために、閉じて開き直す必要はもうありません。

### 前方互換性

`tasks`上の2つのnullable列は、v2のワークフロールーティング用に予約されています: `workflow_template_id`（このタスクが属するテンプレート）と`current_step_key`（そのテンプレート内のどのステップがアクティブか）。v1カーネルはルーティングではこれらを無視しますが、クライアントが書き込むことは許可するため、v2リリースは別のスキーマ移行なしにルーティングの機構を追加できます。

## イベントリファレンス {#event-reference}

すべての遷移は`task_events`に1行を追記します。各行は、UIがイベントを試行ごとにグループ化できるように、任意の`run_id`を持ちます。種類はフィルタリングしやすいように3つのクラスターにグループ化されます（`hermes kanban watch --kinds completed,gave_up,timed_out`）。

**ライフサイクル**（論理単位としてのタスクについて何が変わったか）:

| 種類 | ペイロード | いつ |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | タスクが挿入された。`run_id`は`NULL`。 |
| `promoted` | — | すべての親が`done`に達したため`todo → ready`。`run_id`は`NULL`。 |
| `claimed` | `{lock, expires, run_id}` | ディスパッチャーがスポーンのために`ready`タスクをアトミックにクレームした。 |
| `completed` | `{result_len, summary?}` | ワーカーが`--result` / `--summary`を書き、タスクが`done`に達した。`summary`は最初の行の引き継ぎ（400文字上限）で、完全版は実行の行に存在する。`complete_task`が引き継ぎフィールド付きで一度もクレームされなかったタスクに対して呼ばれた場合、`run_id`が依然として何かを指すように持続時間ゼロの実行が合成される。 |
| `blocked` | `{reason, kind, recurrences}` | ワーカーまたは人間がタスクを`blocked`に切り替えた。`kind`は型付きのブロックの理由（`needs_input`、`capability`、`transient`、または一般的なブロックでは`null`）。`recurrences`はアンブロックのループのカウンター。未完了の親のない`kind=dependency`のブロックは、`todo`では次のディスパッチのティックで再昇格されて再起動されるだけなので、`needs_input`としてここに着地する（ペイロードに`requested_kind: dependency`、`rekind_reason: no_open_parent`が加わる）。一度もクレームされなかったタスクに対して`--reason`付きで呼ばれると、持続時間ゼロの実行を合成する。 |
| `dependency_wait` | `{reason, kind}`または`{reason: parent_not_done, demoted: true, parent}` | 少なくとも1つの親がまだ開いている間にワーカーが`kind=dependency`でブロックした — タスクは別のタスクを待っているだけなので、`blocked`ではなく`todo`（親でゲートされ、自動昇格）にルーティングされ、再発は数えられない。人間は不要。`link`/`kanban_link`が`ready`の子を`done`でない親の下に置いたときにも発行される: 子は`todo`に戻り、このイベントがその理由を記録する（`ready → running`のクレームは親を再確認するため、親が完了するか`hermes kanban unlink`でリンクが削除されるまで、何もそれを実行できない）。 |
| `block_loop_detected` | `{reason, kind, recurrences, limit}` | タスクが同じ理由でアンブロックと再ブロックを`BLOCK_RECURRENCE_LIMIT`回（デフォルト2）繰り返した。再び`blocked`に着地する — そこではcronがアンブロックし続ける — 代わりに、オーケストレーションの注意を引くために`triage`にルーティングされ、アンブロック↔再ブロックのループを断ち切る。 |
| `unblocked` | — | 手動または`/unblock`経由で`blocked → ready`（親がまだ開いていれば`todo`）。ディスパッチャーの`consecutive_failures`をリセットするが、ループのブレーカーが記憶を保つよう`block_recurrences`は意図的に保持する。`run_id`は`NULL`。 |
| `archived` | — | デフォルトのボードから非表示になった。タスクがまだ実行中だった場合、副作用として再取得された実行の`run_id`を持つ。 |

**編集**（遷移ではない、人間が起こした変更）:

| 種類 | ペイロード | いつ |
|---|---|---|
| `assigned` | `{assignee}` | 担当者が変わった（割り当て解除を含む）。 |
| `edited` | `{fields}` | タイトルまたは本文が更新された。 |
| `reprioritized` | `{priority}` | 優先度が変わった。 |
| `status` | `{status}` | ダッシュボードのドラッグ&ドロップがステータスを直接書いた（例: `todo → ready`）。`running`からドラッグして外したときに再取得された実行の`run_id`を持つ。それ以外では`run_id`はNULL。 |

**ワーカーのテレメトリ**（論理タスクではなく、実行プロセスについて）:

| 種類 | ペイロード | いつ |
|---|---|---|
| `spawned` | `{pid}` | ディスパッチャーがワーカープロセスの起動に成功した。 |
| `worker_registered` | `{pid, started_at}` | ディスパッチャーがワーカーを起動した後、そのpidを記録する前に死んだため、ワーカーが最初のモデル呼び出しの前に自分でそれを記録した。その後の生存確認はそれを認識し、2つ目のワーカーを起動する代わりに期限切れのクレームを延長する。そこまで到達する前に実行が再取得されたワーカーは、カードの作業をせずに終了する。 |
| `heartbeat` | `{note?}` | ワーカーが長時間の処理中に生存を知らせるため`hermes kanban heartbeat $TASK`を呼んだ。 |
| `reclaimed` | `{stale_lock}` | クレームのTTLが完了なしに切れた。タスクは`ready`に戻る。自動の再取得は`gave_up`のブレーカーに向けて成功しなかった試行1回として数えられる（ワーカーを一度も起動しなかったクレームは、そうしないとクレーム → 再取得 → クレームを永遠に繰り返すため）。オペレーターによる`reclaim`は代わりにカウンターをリセットする。 |
| `crashed` | `{pid, claimer, exit_kind?, exit_code?, worker_output?}` | ワーカーのPIDがもう生きていないが、TTLはまだ切れていなかった。`worker_output`はワーカー自身のログの末尾（最終応答またはレンダリングされたプロバイダーのエラー、装飾を除去、400文字以内）で、タスクの`last_failure_error`にも追加されるため、ボードには終了コードだけでなく*理由*が表示される。 |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds`を超過した。ディスパッチャーがSIGTERM（その後5秒の猶予の後にSIGKILL）し、再キューした。 |
| `stale` | `{elapsed_seconds, last_heartbeat_at, heartbeat_age_seconds, timeout_seconds, pid, terminated}` | タスクが`kanban.dispatch_stale_timeout_seconds`（デフォルト4時間）より長く実行され、かつ直近1時間に`kanban_heartbeat`が届かなかった。ディスパッチャーはホストローカルのワーカー（あれば）にSIGTERMを送り、タスクを再ディスパッチのため`ready`にリセットした。失敗カウンターは増やさない（staleはディスパッチャー側の不在の検出であり、ワーカーの過失ではない）。長時間の処理を行うワーカーは、これを避けるため少なくとも1時間に1回`kanban_heartbeat`を呼ぶべき。 |
| `reconciled` | `{reason, claim_lock, claim_expires, worker_pid}` | 孤立したカードの照合: カードが壊れたクレームの記帳（`claim_lock`または`claim_expires`がNULL — クレーム途中のクラッシュ、手動のSQL、DBの復元）のまま`running`で、ライブのワーカーがいなかったため、TTL/クラッシュ/staleのどの経路でも回復できなかった。ディスパッチャーが説明のコメント付きで`ready`に再キューした。config.yamlの`kanban.reconcile_orphans`でゲートされる（デフォルト`true`）。 |
| `respawn_guarded` | `{reason}` | ディスパッチャーがこのティックでこのreadyタスクの再起動を拒否した。理由: `infrastructure_cooldown`（ホストが直前の起動を拒否した — 再起動に強いsystemdのスコープがない — かつクールダウンが経過していない。カードに不利に数えられることはない）、`rate_limit_cooldown`（直前の実行がクォータの壁に当たった。同じクールダウンで、数えられない）、`blocker_auth`（直前の失敗がクォータ/認証/429のエラーだった — レートのウィンドウのリセットを待つ）、`recent_success`（直近1時間に完了した実行があった — 再実行の前にレビューを待つ）、`active_pr`（最近のコメントにGitHubのPRのURLがある — 以前のワーカーがすでにPRを開いた）。タスクは`ready`のままで、次のティックで再び起動の機会がある。根本の状態が続く場合は、通常の`consecutive_failures`のサーキットブレーカーが`failure_limit`回の失敗の後に`gave_up`で自動ブロックする。 |
| `spawn_failed` | `{error, failures}` | 1回のスポーン試行が失敗した（PATHが見つからない、ワークスペースがマウントできないなど）。カウンターが増加し、タスクはリトライのため`ready`に戻る。 |
| `skipped_nonspawnable` | `{assignee}` | 担当者のプロファイルがこのホームに存在しない（または`kanban.dispatch_profiles`にない）ため、ディスパッチャーが起動を拒否した。カードごとに1回書き込まれ — 間に別のイベントが入った場合にのみ繰り返される — そのため`show`/`tail`は、ティックごとの行なしに欠けているプロファイルを示す。カードは`ready`のまま。再割り当てするか、プロファイルをインストールすること。 |
| `protocol_violation` | `{pid, claimer, exit_code, protocol_violation, worker_output?}` | タスクがまだ`running`の間にワーカーが正常終了した。通常は、終端のボード呼び出し（`kanban_complete`、`kanban_request_review`、`kanban_block`）なしに回答したため。違反のたびに発行される（ペイロードの`protocol_violation: true`のマーカーは実行のメタデータにコピーされ、違反専用のリトライ予算に入る）。予算内 — 最大`_PROTOCOL_VIOLATION_FAILURE_LIMIT`（デフォルト3）回の*連続した*違反、タスクごとの`max_retries`が上書き — では、タスクは単に別の試行のため`ready`に戻る。連続が上限に達すると、ディスパッチャーは`gave_up`も発行して自動ブロックする。`worker_output`はワーカー自身が最後に出力したテキスト（通常は停止した理由の説明）を持ち、`last_failure_error`にも取り込まれ、リトライするワーカーに以前の試行のエラーとして示される。 |
| `gave_up` | `{failures, effective_limit, limit_source, error, terminal_provider?}` | N回連続の成功しなかった試行の後にサーキットブレーカーが作動した。タスクは最後のエラーとともに自動ブロックされる。実効の上限は、タスクの`max_retries`、次にディスパッチャーの`failure_limit` / `kanban.failure_limit`、次に組み込みのデフォルトの順に解決される。`terminal_provider: true`は、リトライでは直せないプロバイダーのエラー（認証情報の失効、モデルの消失）でワーカーが`78`で終了したことを意味する — 明示的に再ログイン（`hermes auth` / `setup`）を必要とする起動時の認証情報の失敗も含み、それは最初のターンの前に`78`で終了する。その他の起動時の失敗は`1`で終了する — そしてブレーカーは上限にかかわらず、その最初の試行で作動し、固定される。 |

`hermes kanban tail <id>`は、これらを単一タスクについて表示します。`hermes kanban watch`は、これらをボード全体でストリームします。

## スコープ外

Kanbanは意図的にシングルホストです。`~/.hermes/kanban.db`はローカルのSQLiteファイルであり、ディスパッチャーは同じマシン上でワーカーを生成します。2つのホストにまたがって共有ボードを実行することはサポートされていません — 「ホストA上のワーカーX、ホストB上のワーカーY」のための協調プリミティブはなく、クラッシュ検出のパスはPIDがホストローカルであることを前提としています。マルチホストが必要な場合は、ホストごとに独立したボードを実行し、`delegate_task` / メッセージキューを使ってそれらを橋渡ししてください。
