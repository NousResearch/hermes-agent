---
sidebar_position: 12
title: "Kanban（マルチエージェントボード）"
description: "複数のHermesプロファイルを協調動作させるための、SQLiteベースの永続的なタスクボード"
---

# Kanban — マルチエージェントプロファイル協調

> **手順を追って学びたいですか？** [Kanbanチュートリアル](./kanban-tutorial)をお読みください。4つのユーザーストーリー（ソロ開発、フリート運用、リトライ付きロールパイプライン、サーキットブレーカー）を、それぞれのダッシュボードのスクリーンショット付きで解説しています。このページはリファレンスであり、チュートリアルはナラティブ（読み物）です。

Hermes Kanbanは、すべてのHermesプロファイル間で共有される永続的なタスクボードであり、壊れやすいインプロセスのサブエージェント群を使わずに、複数の名前付きエージェントで作業を協調できます。すべてのタスクは`~/.hermes/kanban.db`内の1行であり、すべての引き継ぎは誰でも読み書きできる1行であり、すべてのワーカーは独自のアイデンティティを持つ完全なOSプロセスです。

### 2つの接点: モデルはツール経由で、あなたはCLI経由で対話する

ボードには2つの入り口があり、どちらも同じ`~/.hermes/kanban.db`を背後に持ちます。

- **エージェントは専用の`kanban_*`ツールセットを通じてボードを操作します** — `kanban_show`、`kanban_list`、`kanban_complete`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_create`、`kanban_link`、`kanban_unblock`。ディスパッチャーは各ワーカーを生成する際、これらのツールをすでにスキーマに含めた状態で起動します。オーケストレーター（orchestrator）プロファイルでも、`kanban`ツールセットを明示的に有効化できます。モデルは`hermes kanban`をシェル実行するのでは*なく*、直接ツールを呼び出してタスクを読み取り、ルーティングします。下記の[ワーカーはボードとどう対話するか](#how-workers-interact-with-the-board)を参照してください。
- **あなた（およびスクリプト、cron）は、CLI上の`hermes kanban …`、スラッシュコマンドとしての`/kanban …`、またはダッシュボードを通じてボードを操作します。** これらは人間と自動化のためのものであり、ツール呼び出しを行うモデルが背後にいない接点です。

どちらの接点も同じ`kanban_db`レイヤーを経由するため、読み取りは一貫したビューを参照でき、書き込みがずれることもありません。このページの残りの部分ではコピー&ペーストしやすいためCLIの例を示しますが、すべてのCLI動詞にはモデルが使うツール呼び出しの等価物があります。

これは、`delegate_task`では扱えないワークロードをカバーする形です。

- **リサーチのトリアージ** — 並列のリサーチャー＋アナリスト＋ライター、ヒューマンインザループ。
- **スケジュール運用** — 数週間にわたってジャーナルを構築する、繰り返しの日次ブリーフ。
- **デジタルツイン** — 時間をかけてメモリを蓄積する、永続的な名前付きアシスタント（`inbox-triage`、`ops-review`）。
- **エンジニアリングパイプライン** — 分解 → 並列worktreeでの実装 → レビュー → 反復 → PR。
- **フリートワーク** — N個の対象を管理する1人のスペシャリスト（50個のソーシャルアカウント、12個の監視対象サービス）。

設計の全体的な根拠、Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterpriseとの比較分析、および8つの正準的な協調パターンについては、リポジトリ内の`docs/hermes-kanban-v1-spec.pdf`を参照してください。

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

**`delegate_task`を使うべきとき:** 親エージェントが処理を続行する前に短い推論の答えを必要とし、人間が関与せず、結果が親のコンテキストに戻る場合。

**Kanbanを使うべきとき:** 作業がエージェントの境界をまたぐ、再起動を生き延びる必要がある、人間の入力が必要になる可能性がある、別のロールに引き取られる可能性がある、または事後に発見可能である必要がある場合。

両者は共存します。Kanbanワーカーは実行中に内部で`delegate_task`を呼び出すことがあります。

## コアコンセプト

- **ボード（Board）** — 独自のSQLite DB、ワークスペースディレクトリ、ディスパッチャーループを持つ、スタンドアロンのタスクキュー。1つのインストールで複数のボードを持てます
  （例: プロジェクト、リポジトリ、ドメインごとに1つ）。下記の[ボード（マルチプロジェクト）](#boards-multi-project)を
  参照してください。単一プロジェクトのユーザーは`default`ボードのままで、このドキュメントのセクション以外で
  「ボード」という言葉を目にすることはありません。
- **タスク（Task）** — タイトル、任意の本文、1人の担当者（プロファイル名）、ステータス（`triage | todo | ready | running | blocked | done | archived`）、任意のテナント名前空間、任意の冪等性キー（リトライされる自動化のための重複排除）を持つ1行。
- **リンク（Link）** — 親 → 子の依存関係を記録する`task_links`の行。ディスパッチャーは、すべての親が`done`になると`todo → ready`に昇格させます。
- **コメント（Comment）** — エージェント間のプロトコル。エージェントと人間がコメントを追記します。ワーカーが（再）生成されると、コンテキストの一部としてコメントスレッド全体を読み取ります。
- **ワークスペース（Workspace）** — ワーカーが動作するディレクトリ。3種類あります。
  - `scratch`（デフォルト） — `~/.hermes/kanban/workspaces/<id>/`配下（非defaultボードでは`~/.hermes/kanban/boards/<slug>/workspaces/<id>/`）の新規一時ディレクトリ。
  - `dir:<path>` — 既存の共有ディレクトリ（Obsidian vault、メール運用ディレクトリ、アカウントごとのフォルダ）。**絶対パスでなければなりません。** `dir:../tenants/foo/`のような相対パスはディスパッチ時に拒否されます。なぜなら、ディスパッチャーがたまたま置かれているCWDに対して解決されてしまい、曖昧であり、混乱した代理人（confused-deputy）の脱出ベクトルになるからです。それ以外の点では、パスは信頼されます — それはあなたのマシン、あなたのファイルシステムであり、ワーカーはあなたのuidで実行されます。これは信頼されたローカルユーザーという脅威モデルです。Kanbanは設計上シングルホストです。
  - `worktree` — コーディングタスク用の`.worktrees/<id>/`配下のgit worktree。ワーカー側の`git worktree add`で作成されます。
- **ディスパッチャー（Dispatcher）** — N秒ごと（デフォルト60秒）に次を行う、長期間動作するループ: 古いクレームの再取得、クラッシュしたワーカーの再取得（PIDは消えているがTTLはまだ切れていない）、ready状態のタスクの昇格、アトミックなクレーム、割り当てられたプロファイルの生成。デフォルトでは**ゲートウェイ内で**実行されます（`kanban.dispatch_in_gateway: true`）。1つのディスパッチャーがティックごとにすべてのボードを掃引します。ワーカーは`HERMES_KANBAN_BOARD`を固定した状態で生成されるため、他のボードを見ることはできません。同じタスクで`kanban.failure_limit`回連続でスポーンに失敗すると（デフォルト: 2）、ディスパッチャーは最後のエラーを理由として自動的にブロックします — 存在しないプロファイル、マウントできないワークスペースなどのタスクでスラッシング（無駄な再試行）が起きるのを防ぎます。
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
# ディスク上に何があるか確認する。新規インストールでは "default" のみ表示。
hermes kanban boards list

# 新しいボードを作成する。
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # 任意: これをアクティブなボードにする

# 切り替えずに特定のボードを操作する。
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# 以降の呼び出しで「現在」のボードを変更する。
hermes kanban boards switch atm10-server
hermes kanban boards show             # 今アクティブなのはどれか？

# 表示名をリネームする（slugは不変 — ディレクトリ名なので）。
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# アーカイブ（デフォルト） — ボードのディレクトリを boards/_archived/<slug>-<ts>/ に移動。
# ディレクトリを戻せば復元可能。
hermes kanban boards rm atm10-server

# ハードデリート — ボードのディレクトリを `rm -rf`。復元不可。
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
- **+ New board** — slug、表示名、説明、アイコンを尋ねるモーダルを開きます。
  新しいボードに自動切り替えするオプションあり。
- **Archive** — `default`以外のボードでのみ表示されます。確認後、
  ボードのディレクトリを`boards/_archived/`に移動します。

すべてのダッシュボードAPIエンドポイントは、ボードのスコープ指定のために`?board=<slug>`を受け付けます。
イベント用WebSocketは接続時にボードに固定されます。UIで切り替えると、
新しいボードに対して新しいWSが開かれます。


## クイックスタート

以下のコマンドは、**あなた**（人間）がボードをセットアップしてタスクを作成するものです。タスクが割り当てられると、ディスパッチャーは割り当てられたプロファイルをワーカーとして生成し、そこから先は**モデルがCLIコマンドではなく`kanban_*`ツール呼び出しを通じてタスクを進めます** — [ワーカーはボードとどう対話するか](#how-workers-interact-with-the-board)を参照してください。

```bash
# 1. ボードを作成する（あなた）
hermes kanban init

# 2. ゲートウェイを起動する（組み込みディスパッチャーをホストする）
hermes gateway start

# 3. タスクを作成する（あなた — または kanban_create 経由のオーケストレーターエージェント）
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. アクティビティをライブで監視する（あなた）
hermes kanban watch

# 5. ボードを表示する（あなた）
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
  dispatch_in_gateway: true        # デフォルト
  dispatch_interval_seconds: 60    # デフォルト
```

デバッグのために、実行時に`HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`で
設定フラグをオーバーライドできます。標準的なゲートウェイ監視が適用されます: `hermes gateway
start`を直接実行するか、ゲートウェイをsystemdユーザーユニットとして
配線します（ゲートウェイのドキュメントを参照）。ゲートウェイが動作していなければ、
`ready`のタスクは1つ起動するまでそのままにとどまります — `hermes kanban create`は
作成時にこれについて警告します。

`hermes kanban daemon`を別個のプロセスとして実行することは**非推奨**です。
ゲートウェイを使ってください。本当にゲートウェイを実行できない場合（ヘッドレスホストの
ポリシーで長期間動作するサービスが禁止されているなど）、`--force`の脱出ハッチで
古いスタンドアロンデーモンを1リリースサイクルだけ生かしておけますが、同じ
`kanban.db`に対してゲートウェイ組み込みディスパッチャーとスタンドアロンデーモンの
両方を実行すると、クレームの競合を引き起こし、サポートされていません。

### 冪等な作成（自動化/Webhook向け）

```bash
# 最初の呼び出しでタスクを作成する。同じキーでの以降の呼び出しは、
# 重複させる代わりに既存のタスクidを返す。
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

## ワーカーはボードとどう対話するか {#how-workers-interact-with-the-board}

**ワーカーは`hermes kanban`をシェル実行しません。** ディスパッチャーがワーカーを生成する際、子の環境に`HERMES_KANBAN_TASK=t_abcd`を設定し、その環境変数がモデルのスキーマ内の専用**kanbanツールセット**を有効化します。同じツールセットは、ツールセット設定で`kanban`を有効化したオーケストレータープロファイルでも利用できます。これらのツールは、CLIと同じく、Pythonの`kanban_db`レイヤーを通じて直接ボードを読み取り、変更します。動作中のワーカーは、これらを他のツールと同じように呼び出します。`hermes kanban` CLIを目にすることも必要とすることもありません。

| ツール | 目的 | 必須パラメータ |
|---|---|---|
| `kanban_show` | 現在のタスク（タイトル、本文、過去の試行、親からの引き継ぎ、コメント、事前整形済みの完全な`worker_context`）を読み取る。デフォルトでは環境のタスクid。 | — |
| `kanban_list` | `assignee`、`status`、`tenant`、アーカイブの表示可否、件数上限のフィルター付きでタスクのサマリーを一覧する。オーケストレーターがボード上の作業を発見するためのもの。 | — |
| `kanban_complete` | `summary`＋`metadata`の構造化された引き継ぎで完了する。 | `summary` / `result`の少なくとも一方 |
| `kanban_block` | `reason`付きで人間の入力を求めてエスカレートする。 | `reason` |
| `kanban_heartbeat` | 長時間の処理中に生存を知らせる。純粋な副作用。 | — |
| `kanban_comment` | タスクスレッドに永続的なメモを追記する。 | `task_id`, `body` |
| `kanban_create` | （オーケストレーター）`assignee`と任意の`parents`、`skills`などを指定して子タスクにファンアウトする。 | `title`, `assignee` |
| `kanban_link` | （オーケストレーター）事後に`parent_id → child_id`の依存エッジを追加する。 | `parent_id`, `child_id` |
| `kanban_unblock` | （オーケストレーター）ブロックされたタスクを`ready`に戻す。 | `task_id` |

典型的なワーカーのターンは次のようになります。

```
# モデルのツール呼び出し、順番に:
kanban_show()                                     # 引数なし — HERMES_KANBAN_TASK を使用
# （モデルは返された worker_context を読み、terminal/file ツールで作業を行う）
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# （さらに作業）
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
# → {"task_id": "t_r1", ...} を返す
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
# → {"task_id": "t_r2", ...} を返す
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],                     # 両方が完了すると ready に昇格する
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

「（オーケストレーター）」のツール — `kanban_list`、`kanban_create`、`kanban_link`、`kanban_unblock`、および外部タスクに対する`kanban_comment` — は同じツールセットを通じて利用できます。慣例（`kanban-orchestrator`スキルによって強制される）として、ワーカープロファイルは無関係な作業をファンアウトしたりルーティングしたりせず、オーケストレータープロファイルは実装作業を実行しません。ディスパッチャーが生成したワーカーは、破壊的なライフサイクル操作については依然としてタスクスコープに限定され、無関係なタスクを変更できません。

### なぜ`hermes kanban`をシェル実行するのではなくツールを使うのか

理由は3つあります。

1. **バックエンドのポータビリティ。** terminalツールがリモートバックエンド（Docker / Modal / Singularity / SSH）を指しているワーカーは、`hermes kanban complete`をコンテナ*内部*で実行してしまい、そこには`hermes`がインストールされておらず`~/.hermes/kanban.db`もマウントされていません。kanbanツールはエージェント自身のPythonプロセス内で実行され、terminalバックエンドに関係なく常に`~/.hermes/kanban.db`に到達します。
2. **シェルクオートの脆さがない。** `--metadata '{"files": [...]}'`をshlex＋argparse経由で渡すのは潜在的な落とし穴です。構造化されたツール引数はこれを完全に回避します。
3. **より良いエラー。** ツールの結果は、パースしなければならないstderr文字列ではなく、モデルが推論できる構造化されたJSONです。

**通常のセッションへのスキーマフットプリントはゼロ。** 通常の`hermes chat`セッションのスキーマには`kanban_*`ツールが1つもありません。各ツールの`check_fn`は`HERMES_KANBAN_TASK`が設定されているときだけTrueを返し、それはディスパッチャーがこのプロセスを生成したときにのみ起こります。Kanbanにまったく触れないユーザーにツールの肥大化はありません。

`kanban-worker`スキルと`kanban-orchestrator`スキルが、どのツールをいつ、どの順序で呼び出すかをモデルに教えます。

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

### ワーカースキル

Kanbanタスクを処理できるべきプロファイルは、`kanban-worker`スキルをロードしなければなりません。これは、CLIコマンドではなく**ツール呼び出し**でライフサイクル全体をワーカーに教えます。

1. 生成時に`kanban_show()`を呼び出し、タイトル＋本文＋親からの引き継ぎ＋過去の試行＋コメントスレッド全体を読み取る。
2. （terminalツール経由で）`cd $HERMES_KANBAN_WORKSPACE`し、そこで作業する。
3. 長時間の処理中は数分ごとに`kanban_heartbeat(note="...")`を呼び出す。
4. `kanban_complete(summary="...", metadata={...})`で完了するか、行き詰まったら`kanban_block(reason="...")`を呼ぶ。

`kanban-worker`はバンドルされたスキルで、インストールおよびアップデート時に
すべてのプロファイルに同期されます — 別途Skills Hubでのインストール手順はありません。Kanbanワーカーに使う
いずれのプロファイル（`researcher`、`writer`、`ops`など）にも存在することを
確認してください。

```bash
hermes -p <your-worker-profile> skills list | grep kanban-worker
```

バンドルされたコピーが見つからない場合は、そのプロファイルに復元してください。

```bash
hermes -p <your-worker-profile> skills reset kanban-worker --restore
```

ディスパッチャーは、すべてのワーカーを生成する際に`--skills kanban-worker`も自動的に渡すため、プロファイルのデフォルトスキル設定に含まれていなくても、ワーカーは常にパターンライブラリを利用できます。

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

**ダッシュボードから**は、インラインの作成フォームの**skills**フィールドにスキルをカンマ区切りで入力します。

これらのスキルは、組み込みの`kanban-worker`に**追加される**ものです — ディスパッチャーはそれぞれに（および組み込みのものに）`--skills <name>`フラグを1つずつ発行するため、ワーカーはそれらすべてをロードした状態で生成されます。スキル名は、担当者のプロファイルに実際にインストールされているスキルと一致しなければなりません（`hermes skills list`を実行して何が利用可能か確認してください）。実行時のインストールはありません。

### オーケストレータースキル

**行儀の良いオーケストレーターは、作業そのものを自分では行いません。** ユーザーのゴールをタスクに分解し、それらをリンクし、それぞれをあなたがセットアップしたプロファイルのいずれかに割り当て、一歩引きます。`kanban-orchestrator`スキルは、これをツール呼び出しのパターンとしてエンコードします: 誘惑に負けないためのルール、Step-0のプロファイル発見プロンプト（ディスパッチャーは未知の担当者名に対して静かに失敗するため、オーケストレーターはすべてのカードを、あなたのマシンに実際に存在するプロファイルに根付かせなければなりません）、そして`kanban_create` / `kanban_link` / `kanban_comment`をキーとする分解のプレイブックです。

正準的なオーケストレーターのターン（2人の並列リサーチャーがライターに引き継ぐ）:

```
# ユーザーからのゴール: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # 両方のリサーチャーが完了すると 'ready' に昇格
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# 任意: タスクを再作成せずに、後で発見した横断的な依存を追加
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

`kanban-orchestrator`はバンドルされたスキルです。インストールおよびアップデート時に
各プロファイルに同期されるため、別途Skills Hubでのインストール手順はありません。あなたの
オーケストレータープロファイルに存在することを確認してください。

```bash
hermes -p orchestrator skills list | grep kanban-orchestrator
```

バンドルされたコピーが見つからない場合は、そのプロファイルに復元してください。

```bash
hermes -p orchestrator skills reset kanban-orchestrator --restore
```

最良の結果を得るには、ツールセットがボード操作（`kanban`、`gateway`、`memory`）に制限されたプロファイルと組み合わせて、オーケストレーターが試みても文字どおり実装タスクを実行できないようにしてください。

## ダッシュボード（GUI）

`/kanban` CLIとスラッシュコマンドだけでもボードをヘッドレスに運用できますが、ヒューマンインザループにはビジュアルなボードのほうが適したインターフェースであることが多いです: トリアージ、プロファイルをまたいだ監督、コメントスレッドの閲覧、列間でのカードのドラッグなど。Hermesはこれを、[ダッシュボードの拡張](./extending-the-dashboard)で示されたモデルに従って、`plugins/kanban/`の**バンドルされたダッシュボードプラグイン**として提供します — コア機能でも別個のサービスでもありません。

次のコマンドで開きます。

```bash
hermes kanban init      # 一度きり: まだなければ kanban.db を作成する
hermes dashboard        # "Skills" の後に "Kanban" タブがナビに表示される
```

### プラグインが提供するもの

- ステータスごとに1列を表示する**Kanban**タブ: `triage`、`todo`、`ready`、`running`、`blocked`、`done`（トグルがオンのときは`archived`も）。
  - `triage`は、specifier（仕様化担当）が肉付けすることを想定したラフなアイデアのためのパーキング列です。`hermes kanban create --triage`で（またはTriage列のインライン作成経由で）作成されたタスクはここに着地し、人間またはspecifierが`todo` / `ready`に昇格させるまで、ディスパッチャーはそれらに手を出しません。`hermes kanban specify <id>`を実行すると、補助LLMがトリアージタスクを具体的な仕様（ゴール・アプローチ・受け入れ基準を含むタイトル＋本文）に展開し、一気に`todo`へ昇格させます。`--all`はすべてのトリアージタスクを一度に掃引します。どのモデルがspecifierを実行するかは、`config.yaml`の`auxiliary.triage_specifier`で設定します。
- カードには、タスクid、タイトル、優先度バッジ、テナントタグ、割り当てプロファイル、コメント/リンク数、**進捗ピル**（タスクに依存タスクがあるとき、完了した子の`N/M`）、「N前に作成」が表示されます。カードごとのチェックボックスで複数選択ができます。
- **Running内のプロファイルごとのレーン** — ツールバーのチェックボックスで、Running列を担当者ごとにサブグループ化するかを切り替えます。
- **WebSocketによるライブ更新** — プラグインは短いポーリング間隔で追記専用の`task_events`テーブルを追従します。いずれかのプロファイル（CLI、ゲートウェイ、別のダッシュボードタブ）が操作した瞬間に、ボードに変更が反映されます。リロードはデバウンスされるため、イベントのバーストでも単一の再取得が発生するだけです。
- 列間でカードを**ドラッグ&ドロップ**してステータスを変更します。ドロップは`PATCH /api/plugins/kanban/tasks/:id`を送り、CLIが使うのと同じ`kanban_db`コードを経由します — 3つの接点が決してずれることはありません。破壊的なステータス（`done`、`archived`、`blocked`）への移動は確認を求めます。タッチデバイスではポインターベースのフォールバックを使うため、タブレットからもボードを操作できます。
- **インライン作成** — 任意の列ヘッダーの`+`をクリックして、タイトル、担当者、優先度、そして（任意で）既存のすべてのタスクから選べるドロップダウンで親タスクを入力します。Triage列から作成すると、新しいタスクは自動的にトリアージに駐車されます。
- **一括アクション付きの複数選択** — カードをshift/ctrlクリックするか、そのチェックボックスをチェックして選択に追加します。上部に一括アクションバーが表示され、ステータスの一括遷移、アーカイブ、（プロファイルのドロップダウンまたは「（割り当て解除）」での）再割り当てができます。破壊的なバッチは先に確認します。idごとの部分的な失敗は、残りを中断せずに報告されます。
- （shift/ctrlなしで）**カードをクリック**すると、サイドドロワーが開きます（Escapeまたは外側クリックで閉じる）。内容は次のとおりです。
  - **編集可能なタイトル** — 見出しをクリックしてリネーム。
  - **編集可能な担当者 / 優先度** — メタ行をクリックして書き換え。
  - **編集可能な説明** — デフォルトではmarkdownレンダリング（見出し、太字、斜体、インラインコード、フェンスドコード、`http(s)` / `mailto:`リンク、箇条書き）で、テキストエリアに切り替える「edit」ボタンがあります。markdownレンダリングは小さくXSS安全なレンダラーです — すべての置換はHTMLエスケープ済みの入力に対して実行され、`http(s)` / `mailto:`リンクだけが通過し、`target="_blank"`＋`rel="noopener noreferrer"`が常に設定されます。
  - **依存関係エディター** — 親と子のチップリスト（それぞれリンク解除用の`×`付き）と、新しい親や子を追加するための他のすべてのタスクのドロップダウン。循環の試みはサーバー側で明確なメッセージとともに拒否されます。
  - **ステータスアクション行**（→ triage / → ready / → running / block / unblock / complete / archive）。破壊的な遷移には確認プロンプトが付きます。**Triage**列のカードでは、この行に**✨ Specify**ボタンも表示され、補助LLM（`config.yaml`の`auxiliary.triage_specifier`）を呼び出して、一行のアイデアを具体的な仕様（ゴール・アプローチ・受け入れ基準を含むタイトル＋本文）に展開し、タスクを`todo`に昇格させます。同じ動作は、CLI（`hermes kanban specify <id>` / `--all`）、任意のゲートウェイプラットフォーム（`/kanban specify <id>`）、およびプログラム的には`POST /api/plugins/kanban/tasks/:id/specify`からも到達できます。
  - 結果セクション（こちらもmarkdownレンダリング）、Enterで送信できるコメントスレッド、直近20件のイベント。
- **ツールバーフィルター** — フリーテキスト検索、テナントのドロップダウン（`config.yaml`の`dashboard.kanban.default_tenant`がデフォルト）、担当者のドロップダウン、「show archived」トグル、「lanes by profile」トグル、そして次の60秒のティックを待たなくてよいように**Nudge dispatcher**ボタン。

ビジュアルの目標は、おなじみのLinear / Fusionレイアウトです: ダークテーマ、件数付きの列ヘッダー、色付きのステータスドット、優先度とテナントのピルチップ。プラグインはテーマのCSS変数（`--color-*`、`--radius`、`--font-mono`など）だけを読むため、どのダッシュボードテーマがアクティブでも自動的にリスキンされます。

### アーキテクチャ

GUIは厳密に、独自のドメインロジックを持たない**DB経由の読み取り＋kanban_db経由の書き込み**のレイヤーです。

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
| `POST` | `/links` | 依存関係（`parent_id` → `child_id`）を追加 |
| `DELETE` | `/links?parent_id=…&child_id=…` | 依存関係を削除 |
| `POST` | `/dispatch?max=…&dry_run=…` | ディスパッチャーをつつく — 60秒の待機をスキップ |
| `GET` | `/config` | `config.yaml`から`dashboard.kanban`の設定を読む — `default_tenant`、`lane_by_profile`、`include_archived_by_default`、`render_markdown` |
| `WS` | `/events?since=<event_id>` | `task_events`の行のライブストリーム |

各ハンドラーは薄いラッパーです — プラグインは約700行のPython（ルーター＋WebSocketの追従＋一括バッチャー＋設定リーダー）で、新しいビジネスロジックを一切追加しません。小さな`_conn()`ヘルパーが、すべての読み書きで`kanban.db`を自動初期化するため、ユーザーが最初にダッシュボードを開いても、RESTを直接叩いても、`hermes kanban init`を実行しても、新規インストールが動作します。

### ダッシュボードの設定

`~/.hermes/config.yaml`の`dashboard.kanban`配下のこれらのキーは、タブのデフォルトを変更します — プラグインはロード時に`GET /config`経由でそれらを読みます。

```yaml
dashboard:
  kanban:
    default_tenant: acme              # テナントフィルターを事前選択する
    lane_by_profile: true             # "lanes by profile" トグルのデフォルト
    include_archived_by_default: false
    render_markdown: true             # プレーンな <pre> レンダリングにするには false
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

このプラグインは標準のHermesダッシュボードプラグインの契約を使います — マニフェストの完全なリファレンス、シェルスロット、ページスコープのスロット、Plugin SDKについては[ダッシュボードの拡張](./extending-the-dashboard)を参照してください。追加の列、カスタムのカードクローム、テナントでフィルターしたレイアウト、完全な`tab.override`置換は、いずれもこのプラグインをフォークせずに表現できます。

削除せずに無効化するには: `config.yaml`に`dashboard.plugins.kanban.enabled: false`を追加する（または`plugins/kanban/dashboard/manifest.json`を削除する）。

### スコープの境界

GUIは意図的に薄くしています。プラグインが行うことはすべてCLIから到達可能であり、プラグインは人間にとって快適にするだけです。自動割り当て、予算、ガバナンスゲート、組織図ビューは、設計仕様のスコープ外セクションに列挙されているとおり、ユーザー空間（ルータープロファイル、別のプラグイン、`tools/approval.py`の再利用）のままです。

## CLIコマンドリファレンス

これは、**あなた**（またはスクリプト、cron、ダッシュボード）がボードを操作するために使う接点です。ディスパッチャー内で動作するワーカーは、同じ操作に`kanban_*`の[ツール接点](#how-workers-interact-with-the-board)を使います — ここでのCLIとそこでのツールはどちらも`kanban_db`を経由するため、2つの接点は構造的に一致します。

```
hermes kanban init                                     # kanban.db を作成 + デーモンのヒントを表示
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|dir:<path>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived] [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # 割り当て解除は 'none'
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# 一括動詞 — 複数のidを受け付ける:
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban tail <id>                                # 単一タスクのイベントストリームを追従
hermes kanban watch [--assignee P] [--tenant T]        # すべてのイベントをターミナルにライブストリーム
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # 長時間処理向けのワーカー生存シグナル
hermes kanban runs <id> [--json]                       # 試行履歴（実行ごとに1行）
hermes kanban assignees [--json]                       # ディスク上のプロファイル + 担当者ごとのタスク数
hermes kanban dispatch [--dry-run] [--max N]           # 一回限りのパス
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # 非推奨 — スタンドアロンディスパッチャー（代わりに `hermes gateway start` を使用）
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # ステータスごと + 担当者ごとの件数
hermes kanban log <id> [--tail BYTES]                  # ~/.hermes/kanban/logs/ からのワーカーログ
hermes kanban notify-subscribe <id>                    # ゲートウェイブリッジのフック（ゲートウェイ内の /kanban が使用）
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # ワーカーが見るもの
hermes kanban specify [<id> | --all] [--tenant T]      # トリアージ列のアイデアを
        [--author NAME] [--json]                       #   完全な仕様に肉付けして todo に昇格
hermes kanban gc [--event-retention-days N]            # ワークスペース + 古いイベント + 古いログ
        [--log-retention-days N]
```

すべてのコマンドは、インタラクティブCLIおよびメッセージングゲートウェイのスラッシュコマンドとしても利用できます（下記の[`/kanban`スラッシュコマンド](#kanban-slash-command)を参照）。

## `/kanban`スラッシュコマンド {#kanban-slash-command}

すべての`hermes kanban <action>`動詞は、`/kanban <action>`としても到達できます — インタラクティブな`hermes chat`セッションの内部から**および**任意のゲートウェイプラットフォーム（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、メール、SMS）から。どちらの接点も、`hermes kanban`のargparseツリーを再利用するまったく同じ`hermes_cli.kanban.run_slash()`エントリポイントを呼び出すため、引数の接点、フラグ、出力フォーマットはCLI、`/kanban`、`hermes kanban`をまたいで同一です。ボードを操作するためにチャットを離れる必要はありません。

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # トリアージの一行を本物の仕様に肉付け
/kanban specify --all --tenant engineering  # 1つのテナント内のすべてのトリアージタスクを掃引
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

… 約8分後 …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

購読は、タスクが`done`または`archived`に達すると自動的に自身を削除します。`--json`（機械出力）で作成をスクリプト化すると、自動購読はスキップされます — スクリプト化された呼び出し元は`/kanban notify-subscribe`経由で購読を明示的に管理したい、という前提です。

### メッセージングでの出力の切り詰め

ゲートウェイプラットフォームには実用的なメッセージ長の上限があります。`/kanban list`、`/kanban show`、`/kanban tail`が約3800文字を超える出力を生成した場合、レスポンスは`… (truncated; use \`hermes kanban …\` in your terminal for full output)`というフッターとともに切り詰められます。CLI接点にはそのような上限はありません。

### オートコンプリート

インタラクティブCLIで`/kanban `と入力してTabを押すと、組み込みのサブコマンドリスト（`list`、`ls`、`show`、`create`、`assign`、`link`、`unlink`、`claim`、`comment`、`complete`、`block`、`unblock`、`archive`、`tail`、`dispatch`、`context`、`init`、`gc`）を循環します。上記のCLIリファレンスに列挙された残りの動詞（`watch`、`stats`、`runs`、`log`、`assignees`、`heartbeat`、`notify-subscribe`、`notify-list`、`notify-unsubscribe`、`daemon`）も動作します — まだオートコンプリートのヒントリストに入っていないだけです。

## 協調パターン

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

それぞれの具体例については、`docs/hermes-kanban-v1-spec.pdf`を参照してください。

## マルチテナントの利用

1つのスペシャリストフリートが複数のビジネスにサービスを提供する場合、各タスクにテナントをタグ付けします。

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

ワーカーは`$HERMES_TENANT`を受け取り、メモリの書き込みをプレフィックスで名前空間化します。ボード、ディスパッチャー、プロファイル定義はすべて共有され、スコープされるのはデータだけです。

## ゲートウェイ通知

ゲートウェイ（Telegram、Discordなど）から`/kanban create …`を実行すると、発信元のチャットが新しいタスクに自動的に購読されます。ゲートウェイのバックグラウンド通知機能は数秒ごとに`task_events`をポーリングし、終端イベント（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）ごとに1つのメッセージをそのチャットに配信します。完了したタスクは、ワーカーの`--result`の最初の行も送るため、`/kanban show`しなくても結果が分かります。

CLIから購読を明示的に管理することもできます — スクリプト / cronジョブが、自身が発信元でないチャットに通知したい場合に便利です。

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

購読は、タスクが`done`または`archived`に達すると自動的に自身を削除します。クリーンアップは不要です。

## 実行（Runs） — 試行ごとに1行

タスクは作業の論理単位であり、**実行（run）**はそれを実行する1回の試行です。ディスパッチャーがready状態のタスクをクレームすると、`task_runs`に1行を作成し、`tasks.current_run_id`をそれに向けます。その試行が終わると — 完了、ブロック、クラッシュ、タイムアウト、スポーン失敗、再取得 — 実行の行は`outcome`とともにクローズされ、タスクのポインターはクリアされます。3回試行されたタスクには3つの`task_runs`の行があります。

タスクを単に変更するのではなく2つのテーブルにする理由: 実世界のポストモーテムには**完全な試行履歴**が必要であり（「2回目のレビュアーの試行はapproveまで到達し、3回目はマージした」）、また、どのファイルが変わったか、どのテストが実行されたか、レビュアーがどんな所見を残したかといった、試行ごとのメタデータをぶら下げるきれいな場所が必要です。それらは実行の事実であり、タスクの事実ではありません。

実行は、**構造化された引き継ぎ**が存在する場所でもあります。ワーカーが（`kanban_complete(...)`経由で）タスクを完了するとき、次を渡せます。

- `summary`（ツールパラメータ） / `--summary`（CLI） — 人間向けの引き継ぎ。実行に乗り、下流の子はそれを`build_worker_context`で見ます。
- `metadata`（ツールパラメータ） / `--metadata`（CLI） — 実行上の自由形式のJSON辞書。子はサマリーとともにシリアライズされたものを見ます。
- `result`（ツールパラメータ） / `--result`（CLI） — タスクの行に乗る短いログ行（レガシーフィールド、後方互換のため保持）。

下流の子は、各親について最新の完了した実行のサマリー＋metadataを読みます。リトライするワーカーは、自分のタスクの過去の試行（outcome、summary、error）を読むため、すでに失敗したパスを繰り返しません。

```
# ワーカーが実際に行うこと — エージェントループ内部からのツール呼び出し:
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

# リトライされたタスクの試行履歴を確認する:
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

実行はダッシュボード（ドロワー内のRun Historyセクション、試行ごとに1つの色付き行）とREST API（`GET /api/plugins/kanban/tasks/:id`が`runs[]`配列を返す）に公開されます。`{status: "done", summary, metadata}`を伴う`PATCH /api/plugins/kanban/tasks/:id`は両方をカーネルに転送するため、ダッシュボードの「mark done」ボタンはCLIと等価です。`task_events`の行は所属する`run_id`を持つため、UIはそれらを試行ごとにグループ化でき、`completed`イベントはペイロードに最初の行のサマリー（400文字に上限）を埋め込むため、ゲートウェイの通知機能は2回目のSQL往復なしで構造化された引き継ぎをレンダリングできます。

**一括クローズの注意点。** `hermes kanban complete a b c --summary X`は拒否されます — 構造化された引き継ぎは実行ごとであり、同じサマリーをN個のタスクにコピー&ペーストするのはほぼ常に誤りです。`--summary` / `--metadata`*なし*の一括クローズは、よくある「事務的なタスクの山を片付けた」ケースでは引き続き動作します。

**ステータス変更からの再取得された実行。** ダッシュボードで実行中のタスクを`running`からドラッグして外す（`ready`に戻す、または直接`todo`へ）、あるいはまだ実行中のタスクをアーカイブすると、進行中の実行は孤立する代わりに`outcome='reclaimed'`でクローズされます。`tasks.current_run_id`が`NULL`のとき`task_runs`の行は常に終端状態にあり、その逆も成り立ちます — この不変条件はCLI、ダッシュボード、ディスパッチャー、通知機能をまたいで保たれます。

**一度もクレームされなかった完了のための合成実行。** 一度もクレームされなかったタスクを完了またはブロックする（例: 人間がダッシュボードから`ready`タスクをサマリー付きでクローズする、またはCLIユーザーが`hermes kanban complete <ready-task> --summary X`を実行する）と、本来は引き継ぎが失われてしまいます。代わりにカーネルは、試行履歴が完全なまま保たれるように、サマリー / metadata / 理由を持つ持続時間ゼロの実行の行（`started_at == ended_at`）を挿入します。`completed` / `blocked`イベントの`run_id`はその行を指します。

**ドロワーのライブ更新。** ダッシュボードのWebSocketイベントストリームが、ユーザーが現在表示しているタスクの新しいイベントを報告すると、ドロワーは自身をリロードします（タスクごとのイベントカウンターを`useEffect`の依存リストに通すことで）。実行の新しい行や更新されたoutcomeを見るために、閉じて開き直す必要はもうありません。

### 前方互換性

`tasks`上の2つのnullable列は、v2のワークフロールーティング用に予約されています: `workflow_template_id`（このタスクが属するテンプレート）と`current_step_key`（そのテンプレート内のどのステップがアクティブか）。v1カーネルはルーティングではこれらを無視しますが、クライアントが書き込むことは許可するため、v2リリースは別のスキーマ移行なしにルーティングの機構を追加できます。

## イベントリファレンス

すべての遷移は`task_events`に1行を追記します。各行は、UIがイベントを試行ごとにグループ化できるように、任意の`run_id`を持ちます。種類はフィルタリングしやすいように3つのクラスターにグループ化されます（`hermes kanban watch --kinds completed,gave_up,timed_out`）。

**ライフサイクル**（論理単位としてのタスクについて何が変わったか）:

| 種類 | ペイロード | いつ |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | タスクが挿入された。`run_id`は`NULL`。 |
| `promoted` | — | すべての親が`done`に達したため`todo → ready`。`run_id`は`NULL`。 |
| `claimed` | `{lock, expires, run_id}` | ディスパッチャーがスポーンのために`ready`タスクをアトミックにクレームした。 |
| `completed` | `{result_len, summary?}` | ワーカーが`--result` / `--summary`を書き、タスクが`done`に達した。`summary`は最初の行の引き継ぎ（400文字上限）で、完全版は実行の行に存在する。`complete_task`が引き継ぎフィールド付きで一度もクレームされなかったタスクに対して呼ばれた場合、`run_id`が依然として何かを指すように持続時間ゼロの実行が合成される。 |
| `blocked` | `{reason}` | ワーカーまたは人間がタスクを`blocked`に切り替えた。一度もクレームされなかったタスクに対して`--reason`付きで呼ばれると、持続時間ゼロの実行を合成する。 |
| `unblocked` | — | 手動または`/unblock`経由で`blocked → ready`。`run_id`は`NULL`。 |
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
| `heartbeat` | `{note?}` | ワーカーが長時間の処理中に生存を知らせるため`hermes kanban heartbeat $TASK`を呼んだ。 |
| `reclaimed` | `{stale_lock}` | クレームのTTLが完了なしに切れた。タスクは`ready`に戻る。 |
| `crashed` | `{pid, claimer}` | ワーカーのPIDがもう生きていないが、TTLはまだ切れていなかった。 |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds`を超過した。ディスパッチャーがSIGTERM（その後5秒の猶予の後にSIGKILL）し、再キューした。 |
| `spawn_failed` | `{error, failures}` | 1回のスポーン試行が失敗した（PATHが見つからない、ワークスペースがマウントできないなど）。カウンターが増加し、タスクはリトライのため`ready`に戻る。 |
| `gave_up` | `{failures, error}` | N回連続の`spawn_failed`の後にサーキットブレーカーが作動した。タスクは最後のエラーとともに自動ブロックされる。デフォルトN = 5。`--failure-limit`でオーバーライド。 |

`hermes kanban tail <id>`は、これらを単一タスクについて表示します。`hermes kanban watch`は、これらをボード全体でストリームします。

## スコープ外

Kanbanは意図的にシングルホストです。`~/.hermes/kanban.db`はローカルのSQLiteファイルであり、ディスパッチャーは同じマシン上でワーカーを生成します。2つのホストにまたがって共有ボードを実行することはサポートされていません — 「ホストA上のワーカーX、ホストB上のワーカーY」のための協調プリミティブはなく、クラッシュ検出のパスはPIDがホストローカルであることを前提としています。マルチホストが必要な場合は、ホストごとに独立したボードを実行し、`delegate_task` / メッセージキューを使ってそれらを橋渡ししてください。

## 設計仕様

完全な設計 — アーキテクチャ、並行性の正しさ、他システムとの比較、実装計画、リスク、未解決の問題 — は、`docs/hermes-kanban-v1-spec.pdf`に存在します。動作を変更するPRを提出する前に、それをお読みください。
