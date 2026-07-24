# Kanbanチュートリアル

Hermes Kanbanシステムが設計対象とした4つのユースケースを、ダッシュボードをブラウザで開いた状態でひと通り見ていきます。[Kanban概要](./kanban)をまだ読んでいない場合は、まずそちらから始めてください — このチュートリアルは、タスク・実行（run）・担当者（assignee）・ディスパッチャーが何かを知っている前提です。

## セットアップ

```bash
hermes kanban init           # オプション。最初の `hermes kanban <なにか>` で自動初期化される
hermes dashboard             # ブラウザで http://127.0.0.1:9119 を開く
# 左ナビの Kanban をクリック
```

ダッシュボードは、**あなた**がシステムを眺めるのに最も快適な場所です。ディスパッチャーが起動するエージェントワーカーは、ダッシュボードもCLIも一切目にしません — 専用の `kanban_*` [ツールセット](./kanban#how-workers-interact-with-the-board)（`kanban_show`、`kanban_list`、`kanban_complete`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_create`、`kanban_link`、`kanban_unblock`）を通じてボードを操作します。3つの面 — ダッシュボード、CLI、ワーカーツール — はすべて、ボードごとの同じSQLite DB（デフォルトボードは `~/.hermes/kanban.db`、後で作成する任意のボードは `~/.hermes/kanban/boards/<slug>/kanban.db`）を経由するため、変更が垣根のどちら側から来たものであっても、各ボードは一貫しています。

このチュートリアルでは全体を通して `default` ボードを使います。複数の隔離されたキュー（プロジェクト/リポジトリ/ドメインごとに1つ）が欲しい場合は、概要の[Boards（マルチプロジェクト）](./kanban#boards-multi-project)を参照してください — 同じCLI / ダッシュボード / ワーカーのフローがボードごとに適用され、ワーカーは物理的に他のボードのタスクを見ることができません。

チュートリアルを通じて、**`bash` とラベル付けされたコードブロックは*あなた*が実行するコマンドです。** `# worker tool calls` とラベル付けされたコードブロックは、起動されたワーカーのモデルがツール呼び出しとして発するものです — ループを端から端まで見られるようにここに示しているだけで、あなた自身がこれらを実行することは決してありません。

## ボードの全体像

![Kanbanボード概要](/img/kanban-tutorial/01-board-overview.png)

左から右に6つのカラム：

- **Triage** — 生のアイデア。誰かが作業に取りかかる前に、specifierが仕様を肉付けします。任意のtriageカードの **✨ Specify** ボタンをクリックする（またはチャットから `hermes kanban specify <id>` / `/kanban specify <id>` を実行する）と、補助LLMが一行のメモを完全な仕様（ゴール、アプローチ、受け入れ基準）に変換し、一気に `todo` へ昇格させます。これを実行するモデルは `config.yaml` の `auxiliary.triage_specifier` で設定します。
- **Todo** — 作成済みだが依存関係を待っている、またはまだ割り当てられていない。
- **Ready** — 割り当て済みで、ディスパッチャーがクレームするのを待っている。
- **In progress** — ワーカーがタスクをアクティブに実行中。「Lanes by profile」がオン（デフォルト）の場合、このカラムは担当者ごとにサブグループ化され、各ワーカーが何をしているか一目で分かります。
- **Blocked** — ワーカーが人間の入力を求めた、またはサーキットブレーカーが作動した。
- **Done** — 完了。

上部バーには検索、テナント、担当者のフィルター、さらに `Lanes by profile` トグルと、デーモンの次の間隔を待つ代わりに今すぐディスパッチを1ティック実行する `Nudge dispatcher` ボタンがあります。任意のカードをクリックすると、右側にそのドロワーが開きます。

### フラットビュー

プロファイルレーンがうるさい場合は「Lanes by profile」をオフに切り替えると、In Progressカラムがクレーム時刻順に並んだ単一のフラットなリストに折りたたまれます：

![Lanes by profile をオフにしたボード](/img/kanban-tutorial/02-board-flat.png)

## ストーリー1 — 機能を出荷する個人開発者

あなたは機能を作っています。典型的なフロー：スキーマを設計し、APIを実装し、テストを書く。親→子の依存関係を持つ3つのタスクです。

```bash
SCHEMA=$(hermes kanban create "Design auth schema" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --body "Design the user/session/token schema for the auth module." \
    --json | jq -r .id)

API=$(hermes kanban create "Implement auth API endpoints" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --parent $SCHEMA \
    --body "POST /register, POST /login, POST /refresh, POST /logout." \
    --json | jq -r .id)

hermes kanban create "Write auth integration tests" \
    --assignee qa-dev --tenant auth-project --priority 2 \
    --parent $API \
    --body "Cover happy path, wrong password, expired token, concurrent refresh."
```

`API` は親に `SCHEMA` を持ち、`tests` は親に `API` を持つため、`ready` で始まるのは `SCHEMA` だけです。他の2つは、親が完了するまで `todo` に留まります。これは依存関係の昇格エンジンが仕事をしている証で — テスト対象のAPIができるまで、他のワーカーがテスト作成を拾うことはありません。

次のディスパッチャーティック（デフォルトで60秒ごと、または **Nudge dispatcher** を押せば即座）で、`backend-dev` プロファイルが環境に `HERMES_KANBAN_TASK=$SCHEMA` を持つワーカーとして起動します。エージェントの内側から見たワーカーのツール呼び出しループは次のようになります：

```python
# worker tool calls — あなたが実行するコマンドではない
kanban_show()
# → title、body、worker_context、parents、prior attempts、comments を返す

# （ワーカーは worker_context を読み、terminal/file ツールを使ってスキーマを設計し、
#  マイグレーションを書き、自前のチェックを実行し、コミットする — 実作業はここで行われる）

kanban_heartbeat(note="schema drafted, writing migrations now")

kanban_complete(
    summary="users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); "
            "refresh tokens stored as sessions with type='refresh'",
    metadata={
        "changed_files": ["migrations/001_users.sql", "migrations/002_sessions.sql"],
        "decisions": ["bcrypt for hashing", "JWT for session tokens",
                      "7-day refresh, 15-min access"],
    },
)
```

`kanban_show` は `task_id` をデフォルトで `$HERMES_KANBAN_TASK` にするため、ワーカーは自分自身のidを知る必要がありません。`kanban_complete` はサマリーとメタデータを現在の `task_runs` 行に書き込み、その実行をクローズし、タスクを `done` に遷移させます — すべて `kanban_db` を通じた1回のアトミックなホップで行われます。

`SCHEMA` が `done` に達すると、依存関係エンジンが自動的に `API` を `ready` に昇格させます。APIワーカーは、拾い上げる際に `kanban_show()` を呼び出し、`SCHEMA` のサマリーとメタデータが親のハンドオフに添付されているのを見ます — そのため、長い設計ドキュメントを読み直すことなくスキーマの決定事項を把握できます。

ボードで完了したスキーマタスクをクリックすると、ドロワーがすべてを表示します：

![個人開発者 — 完了したスキーマタスクのドロワー](/img/kanban-tutorial/03-drawer-schema-task.png)

下部のRun Historyセクションが重要な追加機能です。1回の試行：結果は `completed`、ワーカーは `@backend-dev`、所要時間、タイムスタンプ、そしてハンドオフサマリーの全文。メタデータのblob（`changed_files`、`decisions`）も実行に保存され、この親を読む下流の任意のワーカーに提示されます。

同じデータはいつでもターミナルから確認できます — これらのコマンドは、ワーカーではなく**あなた**がボードを覗くものです：

```bash
hermes kanban show $SCHEMA
hermes kanban runs $SCHEMA
# #  OUTCOME       PROFILE       ELAPSED  STARTED
# 1  completed     backend-dev        0s  2026-04-27 19:34
#     → users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); refresh tokens ...
```

## ストーリー2 — フリートファーミング

あなたには3人のワーカー（翻訳者、書き起こし担当、コピーライター）と、独立したタスクの山があります。3人全員を並行して引っ張らせ、目に見える進捗を出させたい。これは最もシンプルなkanbanのユースケースで、元の設計が最適化対象としたものです。

作業を作成します：

```bash
for lang in Spanish French German; do
    hermes kanban create "Translate homepage to $lang" \
        --assignee translator --tenant content-ops
done
for i in 1 2 3 4 5; do
    hermes kanban create "Transcribe Q3 customer call #$i" \
        --assignee transcriber --tenant content-ops
done
for sku in 1001 1002 1003 1004; do
    hermes kanban create "Generate product description: SKU-$sku" \
        --assignee copywriter --tenant content-ops
done
```

ゲートウェイを起動してその場を離れます — ゲートウェイは、同じ
kanban.db上で3つのスペシャリストプロファイルすべてのタスクを拾う
組み込みディスパッチャーをホストします：

```bash
hermes gateway start
```

ボードを `content-ops` にフィルタリングする（または単に "Transcribe" で検索する）と、次のようになります：

![transcribeタスクにフィルタリングされたフリートビュー](/img/kanban-tutorial/07-fleet-transcribes.png)

書き起こし2件完了、1件実行中、2件は次のディスパッチャーティックを待つready。In Progressカラムはプロファイルごとにグループ化されている（「Lanes by profile」のデフォルト）ため、混在したリストを走査せずに各ワーカーのアクティブなタスクが見えます。現在のタスクが完了次第、ディスパッチャーは次のreadyタスクをrunningに昇格させます。3つのデーモンが3つの担当者プールで並行して作業することで、コンテンツキュー全体がそれ以上の人間の入力なしに消化されます。

**ストーリー1で構造化ハンドオフについて述べたことは、すべてここにも当てはまります。** コールを完了した翻訳ワーカーは `kanban_complete(summary="translated 4 pages, style matched existing marketing voice", metadata={"duration_seconds": 720, "tokens_used": 2100})` を発します — 分析のためにも、これに依存する下流タスクのためにも有用です。

## ストーリー3 — 再試行を伴うロールパイプライン

ここがKanbanがフラットなTODOリストに対して真価を発揮するところです。PMが仕様を書く。エンジニアが実装する。レビュアーが最初の試行を却下する。エンジニアが変更を加えて再挑戦する。レビュアーが承認する。

`auth-project` でフィルタリングしたダッシュボードビュー：

![複数ロールの機能のパイプラインビュー](/img/kanban-tutorial/08-pipeline-auth.png)

3段階のチェーンが一度に見えます：`Spec: password reset flow`（DONE、pm）、`Implement password reset flow`（DONE、backend-dev）、`Review password reset PR`（READY、reviewer）。それぞれ下部に親が緑色で、子が依存関係として表示されます。

興味深いのは実装タスクです。なぜなら、これはブロックされて再試行されたからです。各ワーカーのモデルが行うツール呼び出しとして示した、3エージェントの完全な振り付けがこちらです：

```python
# --- PMワーカーが $SPEC で起動し、受け入れ基準を書く ---
# worker tool calls
kanban_show()
kanban_complete(
    summary="spec approved; POST /forgot-password sends email, "
            "GET /reset/:token renders form, POST /reset applies new password",
    metadata={"acceptance": [
        "expired token returns 410",
        "reused last-3 password returns 400 with message",
        "successful reset invalidates all active sessions",
    ]},
)
# → $SPEC が done になる。$IMPL が todo から ready へ自動昇格

# --- エンジニアワーカーが $IMPL で起動（最初の試行） ---
# worker tool calls
kanban_show()   # worker_context 内で $SPEC のサマリー + 受け入れメタデータを読む
# （エンジニアがコードを書き、テストを実行し、PRを開く）
# レビュアーのフィードバックが届く — エンジニアは懸念が妥当だと判断しブロックする
kanban_block(
    reason="Review: password strength check missing, reset link isn't "
           "single-use (can be replayed within 30min)",
)
# → $IMPL が blocked に遷移。run 1 は outcome='blocked' でクローズ
```

ここであなた（人間、または別のレビュアープロファイル）がブロック理由を読み、修正の方向性が明確だと判断し、ダッシュボードの「Unblock」ボタンから — またはCLI / スラッシュコマンドから — ブロックを解除します：

```bash
hermes kanban unblock $IMPL
# またはチャットから: /kanban unblock $IMPL
```

ディスパッチャーは `$IMPL` を `ready` に戻し、次のティックで `backend-dev` ワーカーを再起動します。この2度目の起動は、同じタスク上の**新しい実行**です：

```python
# --- エンジニアワーカーが $IMPL で起動（2回目の試行） ---
# worker tool calls
kanban_show()
# → worker_context には run 1 のブロック理由が含まれるようになるため、このワーカーは
#   仕様全体を読み直す代わりに、修正すべき2点を把握している
# （エンジニアが zxcvbn チェックを追加し、reset トークンを単回使用にし、テストを再実行）
kanban_complete(
    summary="added zxcvbn strength check, reset tokens are now single-use "
            "(stored + deleted on success)",
    metadata={
        "changed_files": [
            "auth/reset.py",
            "auth/tests/test_reset.py",
            "migrations/003_single_use_reset_tokens.sql",
        ],
        "tests_run": 11,
        "review_iteration": 2,
    },
)
```

実装タスクをクリックします。ドロワーは**2回の試行**を表示します：

![2回の実行を持つ実装タスク — ブロック後に完了](/img/kanban-tutorial/04b-drawer-retry-history-scrolled.png)

- **Run 1** — `@backend-dev` によって `blocked`。レビューのフィードバックが結果のすぐ下にあります：「password strength check missing, reset link isn't single-use (can be replayed within 30min)」。
- **Run 2** — `@backend-dev` によって `completed`。新しいサマリー、新しいメタデータ。

各実行は、独自の結果、サマリー、メタデータを持つ `task_runs` の行です。再試行履歴は「最新状態」のタスクの上に重ねられた概念的な後付けではありません — それが主要な表現です。再試行するワーカーがタスクを開くと、`build_worker_context` が以前の試行を表示するため、2回目のワーカーは最初のパスがなぜブロックされたかを見て、最初からやり直すのではなく、それらの具体的な指摘に対処します。

次にレビュアーが拾い上げます。`Review password reset PR` を開くと、次が見えます：

![パイプラインのレビュアーのドロワービュー](/img/kanban-tutorial/09-drawer-pipeline-review.png)

親リンクは完了した実装です。レビュアーのワーカーが `Review password reset PR` で起動して `kanban_show()` を呼び出すと、返される `worker_context` には親の直近で完了した実行のサマリー + メタデータが含まれます — そのためレビュアーは「added zxcvbn strength check, reset tokens are now single-use」を読み、diffを見る前に変更されたファイルのリストを手にします。

## ストーリー4 — サーキットブレーカーとクラッシュ回復

実際のワーカーは失敗します。認証情報の欠落、OOMキル、一時的なネットワークエラー。ディスパッチャーには2つの防御線があります：ボードが永遠にスラッシングしないようにN回連続失敗後に自動ブロックする**サーキットブレーカー**と、ワーカーのPIDがTTL期限切れ前に消えたタスクを回収する**クラッシュ検出**です。

### サーキットブレーカー — 恒久的に見える失敗

プロファイルの環境に `AWS_ACCESS_KEY_ID` が設定されていないため、ワーカーを起動できないデプロイタスク：

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops
```

ディスパッチャーはワーカーの起動を試みます。起動が失敗します（`RuntimeError: AWS_ACCESS_KEY_ID not set`）。ディスパッチャーはクレームを解放し、失敗カウンターをインクリメントし、次のティックで再試行します。3回連続で失敗する（デフォルトの `failure_limit`）と、サーキットが作動します：タスクは結果 `gave_up` で `blocked` になります。人間がブロックを解除するまで、それ以上の再試行はありません。

ブロックされたタスクをクリックします：

![サーキットブレーカー — 2回の spawn_failed + 1回の gave_up](/img/kanban-tutorial/11-drawer-gave-up.png)

3回の実行、すべて `error` フィールドに同じエラー。最初の2つは `spawn_failed`（再試行可能）、3つ目は `gave_up`（最終的）です。上部のイベントログが全シーケンスを示します：`created → claimed → spawn_failed → claimed → spawn_failed → claimed → gave_up`。

ターミナルでは：

```bash
hermes kanban runs t_ef5d
# #   OUTCOME        PROFILE        ELAPSED  STARTED
# 1   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 2   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 3   gave_up        deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
```

Telegram / Discord / Slackが接続されている場合、`gave_up` イベントでゲートウェイ通知が発火するため、ボードを確認しなくても障害を知ることができます。

### クラッシュ回復 — ワーカーが実行中に死ぬ

起動には成功したが、ワーカープロセスが後で死ぬことがあります — segfault、OOM、`systemctl stop`。ディスパッチャーは `kill(pid, 0)` をポーリングして死んだpidを検出します。クレームが解放され、タスクは `ready` に戻り、次のティックで新しいワーカーに渡されます。

シードデータの例は、メモリ不足になっていたマイグレーションです：

```bash
# ワーカーがクレームし、240万行のスキャンを開始、約230万行でOOMキルされる
# ディスパッチャーが死んだpidを検出し、クレームを解放し、試行カウンターをインクリメント
# チャンク化戦略での再試行が成功する
```

ドロワーは2回の試行履歴の全体を表示します：

![クラッシュと回復 — 1回の crashed + 1回の completed](/img/kanban-tutorial/06-drawer-crash-recovery.png)

Run 1 — `crashed`、エラー `OOM kill at row 2.3M (process 99999 gone)`。Run 2 — `completed`、メタデータに `"strategy": "chunked with LIMIT + WHERE id > last_id"`。再試行したワーカーはコンテキスト内でrun 1のクラッシュを見て、より安全な戦略を選びました。メタデータにより、将来の観察者（やポストモーテムの執筆者）にとって何が変わったかが明白になります。

## 構造化ハンドオフ — なぜ `summary` と `metadata` が重要か

上記のすべてのストーリーで、ワーカーは最後に `kanban_complete(summary=..., metadata=...)` を呼びました。これは飾りではありません — ワークフローの各段階の間の主要なハンドオフチャネルです。

タスクB上のワーカーが起動して `kanban_show()` を呼ぶと、返ってくる `worker_context` には次が含まれます：

- Bの**以前の試行**（過去の実行：結果、サマリー、エラー、メタデータ）。再試行するワーカーが失敗したパスを繰り返さないようにするため。
- **親タスクの結果** — 各親について、直近で完了した実行のサマリーとメタデータ — 下流のワーカーが上流の作業がなぜ、どのように行われたかを見られるようにするため。

これは、フラットなkanbanシステムを悩ませる「コメントと作業出力を掘り返す」やり取りを置き換えます。PMが仕様のメタデータに受け入れ基準を書くと、エンジニアのワーカーは親のハンドオフでそれを構造的に見ます。エンジニアがどのテストを実行し何件パスしたかを記録すると、レビュアーのワーカーはdiffを開く前にそのリストを手にします。

一括クローズのガードが存在するのは、このデータが実行ごとだからです。`hermes kanban complete a b c --summary X`（CLIからのあなた）は拒否されます — 同じサマリーを3つのタスクにコピペするのは、ほぼ常に間違いです。ハンドオフフラグなしの一括クローズは、よくある「事務的なタスクの山を終わらせた」ケースのために依然として機能します。ツール面には一括版がまったく公開されていません。`kanban_complete` は同じ理由で常に1タスクずつです。

## 現在実行中のタスクを確認する

完全を期すため — まだ進行中のタスクのドロワーがこちらです（ストーリー1のAPI実装で、`backend-dev` がクレームしたがまだ未完了）：

![クレーム済み、進行中のタスク](/img/kanban-tutorial/10-drawer-in-flight.png)

ステータスは `Running` です。アクティブな実行はRun Historyセクションに、結果 `active`、`ended_at` なしで表示されます。このワーカーが死ぬかタイムアウトすると、ディスパッチャーは適切な結果でこの実行をクローズし、次のクレームで新しい実行を開きます — 試行行が消えることはありません。

## 次のステップ

- [Kanban概要](./kanban) — 完全なデータモデル、イベント語彙、CLIリファレンス。
- `hermes kanban --help` — すべてのサブコマンド、すべてのフラグ。
- `hermes kanban watch --kinds completed,gave_up,timed_out` — ボード全体の最終的なイベントをターミナルにライブストリーム。
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — 特定のタスクが完了したときにゲートウェイのpingを受け取る。
