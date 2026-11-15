# Kanbanチュートリアル

Hermes Kanbanシステムが設計対象とした4つのユースケースを、ダッシュボードをブラウザで開いた状態でひと通り見ていきます。[Kanban概要](./kanban)をまだ読んでいない場合は、まずそちらから始めてください — このチュートリアルは、タスク・実行（run）・担当者（assignee）・ディスパッチャーが何かを知っている前提です。

## セットアップ

```bash
hermes kanban init           # optional; first `hermes kanban <anything>` auto-inits
hermes dashboard             # opens http://127.0.0.1:9119 in your browser
# click Kanban in the left nav
```

ダッシュボードは、**あなた**がシステムを眺めるのに最も快適な場所です。ディスパッチャーが起動するエージェントワーカーは、ダッシュボードもCLIも一切目にしません — 専用の `kanban_*` [ツールセット](./kanban#how-workers-interact-with-the-board)（`kanban_show`、`kanban_list`、`kanban_complete`、`kanban_request_review`、`kanban_request_changes`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_attach`、`kanban_attach_url`、`kanban_attachments`、`kanban_create`、`kanban_link`、`kanban_unblock`）を通じてボードを操作します。3つの面 — ダッシュボード、CLI、ワーカーツール — はすべて、ボードごとの同じSQLite DB（デフォルトボードは `~/.hermes/kanban.db`、後で作成する任意のボードは `~/.hermes/kanban/boards/<slug>/kanban.db`）を経由するため、変更が垣根のどちら側から来たものであっても、各ボードは一貫しています。

このチュートリアルでは全体を通して `default` ボードを使います。複数の隔離されたキュー（プロジェクト/リポジトリ/ドメインごとに1つ）が欲しい場合は、概要の[Boards（マルチプロジェクト）](./kanban#boards-multi-project)を参照してください — 同じCLI / ダッシュボード / ワーカーのフローがボードごとに適用され、ワーカーは物理的に他のボードのタスクを見ることができません。

チュートリアルを通じて、**`bash` とラベル付けされたコードブロックは*あなた*が実行するコマンドです。** `# worker tool calls` とラベル付けされたコードブロックは、起動されたワーカーのモデルがツール呼び出しとして発するものです — ループを端から端まで見られるようにここに示しているだけで、あなた自身がこれらを実行することは決してありません。

## ボードの全体像

![Kanbanボード概要](/img/kanban-tutorial/01-board-overview.png)

左から右に6つのカラム：

- **Triage** — 生のアイデア。デフォルトでは、ディスパッチャーがここにあるタスクに対して**decomposer**を自動実行します。組み込みのdecomposerは `auxiliary.kanban_decomposer` を使い、プロファイルの一覧と説明を読み取って、最適なスペシャリストに振り分けられた子タスクのグラフを生成します。元のタスクは親として保持されるため、その担当者（`kanban.orchestrator_profile`、未設定の場合はアクティブなデフォルトプロファイル）は、すべてが終わったときに再び起きて完了を判定します。モードを切り替えるには、kanbanページ上部の **Orchestration: Auto/Manual** ピルを切り替えます。Manualモードでは、カードの **⚗ Decompose** をクリックするか、`hermes kanban decompose <id>` / `/kanban decompose <id>` を実行します。ファンアウトが不要な単一タスクには、**✨ Specify** が一発で仕様を書き直し（ゴール、アプローチ、受け入れ基準）、`todo` へ昇格させます。モデルは `config.yaml` の `auxiliary.kanban_decomposer` と `auxiliary.triage_specifier` で設定します。メインのKanbanガイドの[Auto vs Manual orchestration](./kanban#auto-vs-manual-orchestration)を参照してください。
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
# worker tool calls — NOT commands you run
kanban_show()
# → returns title, body, worker_context, parents, prior attempts, comments

# (worker reads worker_context, uses terminal/file tools to design the schema,
#  write migrations, run its own checks, commit — the real work happens here)

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

このスクリーンショットは**事前作成された下流カード**モデルを使っています：実装カードには専用のレビュアー子カードがあります。このモデルでは、レビュアー子カードが `todo` から出られるよう、エンジニアは実装の準備ができたら `kanban_complete` を呼ぶ必要があります。レビューを依頼するためだけに実装の親をブロックしてはいけません。

同じカードが実装とレビューの両方を担うワークフローでは、代わりにファーストクラスのレビューライフサイクルを使います。実装 → レビュー → 変更 → 再レビューの完全な振り付けは次のとおりです：

```python
# --- Engineer: first implementation attempt ---
kanban_show()
# (write code, run tests, prepare the candidate)
kanban_request_review(
    summary="implemented reset flow; candidate is ready for review",
    metadata={"changed_files": ["auth/reset.py"], "tests_run": 8},
    reviewer="reviewer",
)
# → the same card enters review; the implementation run closes as
#   outcome='review_requested'

# --- Reviewer: request concrete changes ---
kanban_show()
# (inspect the handoff and candidate)
kanban_request_changes(
    reason="Add password-strength validation and make reset tokens single-use."
)
# → the review run closes as outcome='changes_requested'; the card returns
#   to backend-dev in ready/todo without touching block-loop accounting

# --- Engineer: second implementation attempt ---
kanban_show()  # prior review evidence is in worker_context
# (apply feedback and re-run tests)
kanban_request_review(
    summary="added zxcvbn validation and single-use reset tokens",
    metadata={
        "changed_files": [
            "auth/reset.py",
            "auth/tests/test_reset.py",
            "migrations/003_single_use_reset_tokens.sql",
        ],
        "tests_run": 11,
        "review_iteration": 2,
    },
    reviewer="reviewer",
)

# --- Reviewer: approve ---
kanban_complete(summary="review passed; acceptance criteria verified")
# → done
```

タスクの実行履歴には `review_requested → changes_requested → review_requested → completed` が記録されます。各試行は独自のアクター、サマリー、メタデータ、結果を持つため、2回目のエンジニアはレビュアーが何を却下したかを正確に把握でき、最終承認も監査可能なまま残ります。`kanban_block` は本当の外部エスカレーション（アクセス権の欠如、プロダクト上の判断、利用できないインフラ）のためのもので、通常のレビューフィードバックには使いません。

スクリーンショットに示した下流カードモデルを意図的に使う場合、レビュアーは実装の親が完了した後に `Review password reset PR` を開きます：

![パイプラインのレビュアーのドロワービュー](/img/kanban-tutorial/09-drawer-pipeline-review.png)

レビュアーカードの `worker_context` には、完了した実装のハンドオフが含まれます。これは別のカードワークフローです。同じカードでの `kanban_request_review` と組み合わせないでください。組み合わせるとレビューレーンが重複します。

## ストーリー4 — サーキットブレーカーとクラッシュ回復

実際のワーカーは失敗します。認証情報の欠落、OOMキル、一時的なネットワークエラー。ディスパッチャーには2つの防御線があります：ボードが永遠にスラッシングしないようにN回連続失敗後に自動ブロックする**サーキットブレーカー**と、ワーカーのPIDがTTL期限切れ前に消えたタスクを回収する**クラッシュ検出**です。

### サーキットブレーカー — 恒久的に見える失敗

プロファイルの環境に `AWS_ACCESS_KEY_ID` が設定されていないため、ワーカーを起動できないデプロイタスク：

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops \
    --max-retries 3
```

ディスパッチャーはワーカーの起動を試みます。起動が失敗します（`RuntimeError: AWS_ACCESS_KEY_ID not set`）。ディスパッチャーはクレームを解放し、失敗カウンターをインクリメントし、次のティックで再試行します。この例では `--max-retries 3` を設定しているため、3回連続で失敗するとサーキットが作動します：タスクは結果 `gave_up` で `blocked` になります。このフラグを省略した場合、Hermesは `kanban.failure_limit`（デフォルト: 2）を使います。人間がブロックを解除するまで、それ以上の再試行はありません。

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
# Worker claims, starts scanning 2.4M rows, OOM kills it at ~2.3M
# Dispatcher detects dead pid, releases claim, increments attempt counter
# Retry with a chunked strategy succeeds
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

## 完了したカードのフォローアップ — 親リンクによるCI修正

ストーリー1の実装カードは `done` です。2時間後、マージされたブランチでCIが失敗します。完了したカードを再オープンしてはいけません — 完了したカードは履歴であり、そのハンドオフは前方へ流れます。完了したカードを**親**として、修正用のカードを作成します：

```bash
hermes kanban create "Fix CI: test_backoff_jitter flakes on 3.11" \
    --assignee backend-dev \
    --parent t_impl \
    --workspace worktree --branch wt/ci-fix-backoff \
    --body "CI run #4812 failed after t_impl completed.
FAILED tests/test_retry.py::test_backoff_jitter - TimeoutError
Acceptance: tests/test_retry.py green on 3.11 and 3.12."
```

これがうまく機能する理由は3つあります：

- **即時ディスパッチ。** 親がすでに `done` なので、子は直接 `ready` で作成されます — ディスパッチャーは次のティックでクレームできます。（まだ完了していない親の子は `todo` で待機します。）
- **引き継がれるコンテキスト。** 修正ワーカーのコンテキストには、`t_impl` の完了サマリーとメタデータ — 元のワーカーが記録した変更ファイルと決定事項 — を含む *Parent task results* セクションが含まれます。そのため、コードを1行も読む前に、なぜそのような形になっているかを把握できます。
- **本文に新しい証拠。** CIログは `t_impl` の完了時には存在しなかったため、親のハンドオフには含まれません — 明示的な受け入れ基準とともに、新しいカードの本文に入れます。

修正カードには新しいworktree/ブランチを使うことを推奨します。元のブランチをチェックアウトすると、ワーカーはリポジトリの*状態*は得られますが、*理由*は何も得られません — それを運ぶのは親のハンドオフです。通常は同じ担当者プロファイルが適切です：コードを書いたプロファイルは、それを修正するスキルを持っています。

## 現在実行中のタスクを確認する

完全を期すため — まだ進行中のタスクのドロワーがこちらです（ストーリー1のAPI実装で、`backend-dev` がクレームしたがまだ未完了）：

![クレーム済み、進行中のタスク](/img/kanban-tutorial/10-drawer-in-flight.png)

ステータスは `Running` です。アクティブな実行はRun Historyセクションに、結果 `active`、`ended_at` なしで表示されます。このワーカーが死ぬかタイムアウトすると、ディスパッチャーは適切な結果でこの実行をクローズし、次のクレームで新しい実行を開きます — 試行行が消えることはありません。

## 次のステップ

- [Kanban概要](./kanban) — 完全なデータモデル、イベント語彙、CLIリファレンス。
- `hermes kanban --help` — すべてのサブコマンド、すべてのフラグ。
- `hermes kanban watch --kinds completed,gave_up,timed_out` — ボード全体の最終的なイベントをターミナルにライブストリーム。
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — 特定のタスクが完了したときにゲートウェイのpingを受け取る。
