---
sidebar_position: 1
title: "CLI インターフェース"
description: "Hermes Agent のターミナルインターフェースを使いこなす — コマンド、キーバインド、パーソナリティなど"
---

# CLI インターフェース

Hermes Agent の CLI は、Web UI ではなく本格的なターミナルユーザーインターフェース（TUI）です。複数行編集、スラッシュコマンドの自動補完、対話履歴、割り込みとリダイレクト、ツール出力のストリーミングを備えています。ターミナルで生活する人のために作られています。

:::tip 初回セットアップ
コマンド 1 つ — `hermes setup --portal` — を実行すれば、すぐに `hermes chat` を始められます。詳しくは [Nous Portal](/docs/integrations/nous-portal) をご覧ください。
:::

:::tip
Hermes は、モーダルオーバーレイ、マウス選択、ノンブロッキング入力を備えたモダンな TUI も提供しています。`hermes --tui` で起動できます。詳しくは [TUI](tui.md) ガイドをご覧ください。
:::

## CLI の実行

```bash
# Start an interactive session (default)
hermes

# Single query mode (non-interactive)
hermes chat -q "Hello"

# Single query from a file or stdin — nothing is shell-interpreted, so
# arbitrary text (quotes, $(...), backticks) arrives verbatim
hermes chat --query-file prompt.txt
hermes chat --query-file - < prompt.txt

# With a specific model
hermes chat --model "anthropic/claude-sonnet-4"

# With a specific provider
hermes chat --provider nous        # Use Nous Portal
hermes chat --provider openrouter  # Force OpenRouter

# With specific toolsets
hermes chat --toolsets "web,terminal,skills"

# Start with one or more skills preloaded (rendered once into the session's prompt)
hermes chat -s hermes-agent-dev,github-auth

# Resume previous sessions
hermes --resume <session_id>  # Resume a specific session by ID (-r)
hermes --resume "my thread"   # Resume by title (latest "#N" continuation in the lineage)
hermes -c "my thread" --create-if-missing  # Resume that titled session, creating it if absent
hermes --continue             # Resume the most recent session (bare -c)
hermes --resume latest        # Same as bare -c
hermes --resume latest --in ./dir  # Resume ./dir's latest session, staying in ./dir

# Verbose mode (debug output)
hermes chat --verbose

# Isolated git worktree (for running multiple agents in parallel)
hermes -w                         # Interactive mode in worktree
hermes -w -z "Fix issue #123"     # Single query in worktree
```

### ワークツリーのクリーンアップ

`hermes -w` セッションは、`<repo>/.worktrees/` 配下に使い捨てのワークツリーを作成します。`hermes --tui -w` では、ターミナルを閉じるとゲートウェイが所有するセッションから切り離されます。クリーンなワークツリーは、まだ何かがそれを必要としていない限り、終了時にそのブランチとともに削除されます。次の場合は保持されます: コミットされていない変更やプッシュされていないコミットがある場合、ゲートウェイがそこでまだターンをキューに入れているか実行中の場合、その中のセッションでアクティブな `/loop`、`/heartbeat`、`/goal` がある場合、有効な cron ジョブがその中で実行されるかそのようなセッションに紐付いている場合、または別のビューアー（たとえば Desktop ウィンドウ）がまだそのセッションを保持している場合です。セッションの再開ではその作業ディレクトリが使われます。終了時にワークツリーが削除されていた場合、`hermes --tui --resume <id>`、`hermes --tui -c`、および従来の `hermes --resume`/`-c` は、アタッチする前に同じパスにワークツリーを再作成します（ブランチがまだ存在すればそのブランチで、なければリポジトリの現在の `HEAD` から同じ名前の新しいブランチで）。再作成されるのは、`hermes --tui -w` 自身が作成したワークツリーのみです。終了時のチェックはセッションストアを読み取ってからツリーを削除します。その 2 つの間の瞬間にアタッチして送信したビューアーは検知されないため、そのターンではディレクトリが消えている可能性があります。スイープは `hermes -w` の起動時と、cron ティッカーから 6 時間ごとに自動的に実行されます。アイドル状態でクリーンかつマージ済みのツリーは削除され、コミットされていない変更やプッシュされていないコミットを持つアイドル状態のツリーは **アーカイブされてから削除** されます（`hermes worktree restore <name>` で復元できます。[古いワークツリーのスイープ](./git-worktrees.md#stale-worktree-sweep-and-restore) をご覧ください）。明示的に回収または確認するには:

```bash
hermes worktree list              # audit: age, size, verdict, reason per tree + archives
hermes worktree list --json       # machine-readable audit (trees, external trees, branches, archives)
hermes worktree prune             # run the sweep now + delete merged branches
hermes worktree prune --dry-run   # print every archive/remove/skip decision and why; changes nothing
hermes worktree restore <name>    # recreate an archived tree (add --path DIR to put it elsewhere)
hermes worktree prune --older-than 7   # only reap trees idle for 7+ days
hermes worktree prune --trees-only     # leave local branches alone
hermes worktree prune --branches-only  # leave worktrees alone
```

`.worktrees/` の **外** に登録されたワークツリー（手動または別のツールで作成されたもの）は、`list` の出力に読み取り専用として表示され、削除されることはありません。唯一の例外はメタデータです。ディレクトリがもう存在しない登録は `git worktree prune` で削除されます（ファイルには一切触れません）。`--older-than DAYS` は回収対象を絞り込む方向にのみ働き、その実行中に古いワークツリーのスイープがアーカイブ・削除するものも対象になります。アイドル期間が DAYS 未満のツリーには決して触れません。

セッション内では、`/worktree prune [--dry-run]` が同じことを行います（セッションが実行されているツリーには決して触れません）。

安全性の保証（すべてのモード、経過期間を問わず）:

- コミットされていない変更と **固有のプッシュされていないコミット** は、復元可能なアーカイブ（`refs/hermes/archive/<tree>`）なしに削除されることはありません。`worktrees.archive_dirty: false` の場合は一切削除されません。アップストリームで rebase/squash マージされたコミットは `git cherry` のパッチ等価性によって検出されマージ済みとして扱われます。これにより、主流だった「PR はマージ済みなのにツリーが永遠に残る」というリークがようやく回収できるようになります。
- **リモートのないリポジトリ** は、ローカルのトランク（`main`/`master`、なければメインワークツリーでチェックアウトされているブランチ）を基準に判定されます。コミットがそのトランクから到達可能な、またはパッチ等価なツリーとブランチのみが回収されます。比較対象のトランクがない場合、すべてのツリーとブランチが保持されます。
- **プッシュ済みのオープン PR レーンは何も失わずにディスクを解放します**: クリーンなツリーのブランチヘッドが `origin` の保持するものと完全に一致する場合（スイープごとに 1 回の `git ls-remote` で確認）、チェックアウトは冗長です。ツリーは削除されますが **ブランチ ref は保持される** ため、`git worktree add .worktrees/<name> <branch>` 1 回でレーンを復元できます。リモートに到達できない場合、ツリーは保持されます。
- **使用中** のツリーには決して触れません: 実行中の hermes セッションのロック、または作業ディレクトリ（Linux では開いているファイル）がツリー内にある稼働中のプロセスがある場合です。各ツリーはアーカイブまたは削除の直前に再度チェックされ、アーカイブの作成後にもう一度チェックされます。その間に変更されたツリーは保持されます。
- **追跡されていないファイルのみのスクラッチ**（PR 本文の下書き、メモ）は、ツリーが削除される前に `~/.hermes/archive/worktree-prune/` にアーカイブされます — 破棄されることはありません。
- **再現不能なものがアーカイブなしに削除されることはありません**: 埋め込みリポジトリ（gitignore されたものであっても）、ワークツリーごとの ref（`refs/worktree/*`、`refs/bisect/*`）からのみ到達できるコミット、アーカイブサイズ上限を超える追跡されていないファイルを含むツリーは保持されます。無視されたファイル（`.env` のコピー、キャッシュ、ビルド出力）はアーカイブされず、ツリーとともに削除されます。無視されたフォルダ内のネストされた git リポジトリはツリーを保持させます。[無視されたファイル](./git-worktrees.md#stale-worktree-sweep-and-restore) をご覧ください。
- ブランチの削除は名前ではなく内容で判定されます: コミットがすべてアップストリームにあるローカルブランチは安全に削除できます。固有の作業を含むブランチ、チェックアウト中のブランチ、`main`/`master`/`develop` は常に保持されます。

同じスイープは cron スケジューラーからも（プロファイルごとに最大 6 時間に 1 回、バックグラウンドで）、そのプロファイルがワークツリーを作成したすべてのリポジトリに対して実行されます。そのため、何日も誰も `hermes -w` を起動しないゲートウェイ専用マシンでも、CLI セッションの合間にスクラッチツリーが溜まることはなくなりました。

`.worktrees/` が 10 ツリーまたは 5 GB を超えると、起動時にこれらのコマンドを案内する 1 行の通知が表示されます。

### プラグインの管理

`hermes plugins` コマンドは、ネイティブの Hermes プラグインとポータブルな Agent Plugins v1 パッケージを、同じオプトイン方式のワークフローで管理します。

```bash
hermes plugins install owner/repository --no-enable
hermes plugins list
hermes plugins enable <plugin-name>
hermes plugins disable <plugin-name>
hermes plugins update <plugin-name>
hermes plugins remove <plugin-name>
```

ポータブルパッケージは、明示的に有効化するまで無効のままです。Hermes は現在、ポータブルな Agent Skills と stdio MCP エントリを読み込みます。正確にサポートされるサブセットと信頼境界については、[プラグイン開発者ガイド](../developer-guide/plugins/index.md#portable-agent-plugins-v1-packages) をご覧ください。

## インターフェースのレイアウト

:::info 従来の CLI はゲートウェイクライアントです
`hermes chat`（および `display.interface` が `cli` のときの引数なしの `hermes` プロンプト）は、もはやターミナルのプロセス内でエージェントを実行しません。プロファイルのゲートウェイを検出または起動し（`hermes gateway ensure`）、そこでセッションを作成または再開して、ゲートウェイのローカル WebSocket 経由でアタッチします — Desktop ウィンドウ、TUI、Telegram のトピックが同時に開けるのと同じセッションです。ターミナルを閉じると切り離されます（`Detached; accepted work continues at the gateway.`）。ターンはキャンセルされません。ターミナルの喪失（タブを閉じる、SSH や VS Code リモート接続の切断、SIGHUP/SIGTERM）は切断として扱われます。セッションのバックグラウンドサブエージェントは実行を続け、その結果は引き続きセッションに届きます。`/exit`、`/quit`、`/new`、Ctrl+D はチャットを閉じ、それらを停止します。作業ディレクトリ、`--model`、`--provider`、`--reasoning`、`--toolsets`、`--max-turns`、`--skills`、`--checkpoints`、`--yolo`、`--accept-hooks`、`--pass-session-id`、`--ignore-rules`、`--base-url`/`--api-key`、`--safe-mode`/`--ignore-user-config` は、作成時にセッションへ固定されます。`--resume` では同じ値を繰り返し指定できますが、変更はできません。`--resume <id-or-title>` と `-c <title>` はセッションを指定します。ゲートウェイは、読み取り可能なセッションの中からタイトルを解決し（まず完全一致する ID、次に最新の `"<title> #N"` 継続を、その最新の先端までたどります）、見つからない場合は `No session found matching '…'`（終了コード 1）を報告します。引数なしの `-c` と `--resume latest` は、ゲートウェイ経由でこのワークスペースの直近の CLI セッションを再開します。`-c <title> --create-if-missing` は、そのタイトルのセッションが存在しなければ作成し、存在すれば再開するため、プログラムからの呼び出し元（Bot Mode の `Bot Chat` ターン）は決定的なスレッドを得られます。プロセス内エージェントでのみ意味を持つオプション — `--image`、`--worktree`/`-w`、`--no-restore-cwd`、`--run-budget`、`--verbose`/`-v`、`--compact`、および `-c <name>` なしの `--create-if-missing` — は最初に拒否されます（`Unsupported gateway CLI options: …`、終了コード 2）。拒否メッセージには、フラグごとにその機能が現在どこにあるか（TUI のフラグ、`config.yaml` のキー、または `hermes` のサブコマンド）が一覧表示され、ローカルエージェントへのフォールバックは一切行われません。`--safe-mode` / `--ignore-user-config` で起動するとプロファイルのデフォルトモデルを読み込まないため、モデルを指定する必要があります。拒否メッセージには `hermes chat --safe-mode --provider openrouter --model anthropic/claude-sonnet-4 -q "hello"` のような完全な例が表示されます。このモードで使えるスラッシュコマンドは `/stop`、`/approve <id> <choice>`、`/answer <id> <text>`、`/discard <admission_id>`、`/branch [title]`、`/model <model> [--provider name]`、`/compress [here [N] | <focus>] [--preview]`、`/yolo [on|off]`、`/help`、`/quit` です。その他のスラッシュコマンドは `Unsupported gateway CLI command; use /help.` を表示します。スラッシュコマンドの完全な一覧を使うには TUI（`hermes --tui`）を使ってください。
:::

有限の CLI クエリがキューに入っている間に実行中のターンがオーナーを失った場合、CLI は終了コード `3` で終了します。受け付けられた入力はキューに残ります。表示されたセッションを対話的に再開し、状態不明のターンを確認してから、`/discard <admission_id>` でそれを承認してから続行してください。キューに入っている入力を再送信しないでください。

<img className="docs-terminal-figure" src="/docs/img/docs/cli-layout.svg" alt="Stylized preview of the Hermes CLI layout showing the banner, conversation area, and fixed input prompt." />
<p className="docs-figure-caption">Hermes CLI のバナー、対話ストリーム、固定された入力プロンプトを、壊れやすいテキストアートではなく安定したドキュメント図として描画したものです。</p>

### レイアウトスタイル

`display.cli_layout` は、従来の CLI の描画方法を選択します。デフォルトの `modern` は、スクロールバック内のすべてを、1 つのグリフで始まりターミナル幅に依存しないプレーンな行として出力するため、ウィンドウのサイズを変更しても以前の出力が崩れることはありません。

```
❯ add a multiply function and run the tests
∴ Thought for 3s · Planning the change
📖 Read calc.py
🔧 Patch  0.2s
│ calc.py
│ @@ -1,2 +1,6 @@
│ +def multiply(a, b):
│ +    return a * b
✗ $ pytest -q  0.4s [exit 1]
☤ Hermes
Added multiply; pytest isn't installed here, so I ran the tests directly: 2 passed.
─ 15.0s · edited 1 file +4 -0 · read 2 files · 8:27 AM
── ⠹ Running tests · 14s ─────────────────────────── Add multiply to calc.py ─
❯ type to redirect this turn · Tab queues it for after
──────────────────────────────────────────────────────────────────────────────
 gpt-5-mini · ctx [░░░░░░░░] 4% · ~/project (main)     enter interrupt · tab queue · ctrl+c stop
```

- 40 桁のバーが、各ユーザーメッセージと回答（アクセント色）および各推論ブロック（暗色）の先頭に付きます。推論は従来どおりライブでストリーミングされ、`∴ Thought for 3s · <title>` の行で終わります。`❯` はあなたのメッセージ、完了したツール呼び出しはその絵文字で始まります（`📖 Read`、`💻 $`。失敗した場合は赤い `✗` で、終了コード付き。拒否された `skill_manage` の書き込みは、エージェント自身が再試行するため、絵文字を保ったまま理由を暗色で表示します）。`│` はその行の下に表示される編集の差分、`☤` は回答、`─` はターンの要約行です。
- 入力欄（コンポーザー）の上の罫線には、左側にライブのアクティビティ（スピナー、実行中の内容、経過時間、ストリーミングされたトークン）、右側にセッションタイトルが表示されます。スクロールバックにはスピナーの行は出力されません。
- ステータス行はコンポーザーの下にあります: 従来のステータスバーと同じフィールド（モデル、コンテキスト、キャッシュヒット、レイテンシー、トークン/秒、バックグラウンド作業、ゴール、経過時間、YOLO…）に加えて、左側に作業ディレクトリ（git ブランチ付き）、右側にその時点で機能するキーが表示されます。
- ウェルカムバナーはどちらのレイアウトでも同じです。

ボックス型の推論パネルと回答パネル、および入力欄の上のステータスバーを維持するには、`display.cli_layout: classic` を設定します。

ウェルカムバナーには、モデル、ターミナルバックエンド、作業ディレクトリ、利用可能なツール、インストール済みスキルが一目で表示されます。

### ステータスバー

`classic` レイアウトでは、入力エリアの上に常時表示されるステータスバーがあり、リアルタイムで更新されます（`modern` レイアウトのコンポーザー下のステータス行も同じデータを読み取ります。明示的な `display.status_bar.fields` のリストは、その左側の内容も選択します）。

```
 ☤ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| 要素 | 説明 |
|---------|-------------|
| モデル名 | 現在のモデル（26 文字を超える場合は切り詰め） |
| トークン数 | 使用中のコンテキストトークン数 / 最大コンテキストウィンドウ。`~` は推定値を示します |
| コンテキストバー | しきい値で色分けされた視覚的な充填インジケーター |
| コスト | 推定セッションコスト（不明または価格ゼロのモデルでは `n/a`）。料金は Hermes 同梱の公式価格表、次にプロバイダーの `/models` 一覧から取得されます。直接のファーストパーティ API（OpenAI、xAI、Anthropic、Google、DeepSeek、Xiaomi）で、どちらにもないモデルは models.dev のベンダー定価で計算されます。同じモデル ID を提供するプロキシ、リレー、カスタムエンドポイントはその価格を引き継がず、`n/a` のままです。 |
| 🗜️ N | **コンテキスト圧縮回数** — 実行中のセッションが自動圧縮された回数。最初の圧縮が発動すると表示されます。 |
| ▶ N | **アクティブなバックグラウンドタスク** — 現在のセッションでまだ実行中の `/bg` プロンプトの数。少なくとも 1 つのタスクが進行中であれば表示されます。 |
| 経過時間 | セッションの経過時間 |
| セッションタイトル | セッションにタイトルが付くと、右端に固定された金色のバッジとして表示されます。長いタイトルは、重要なモデルとコンテキストのフィールドを押し出す前に切り詰められます。 |
| ⚠ YOLO | **YOLO モードの警告** — `HERMES_YOLO_MODE` がオンのとき（起動時の `hermes --yolo`、またはセッション中に `/yolo` で切り替えた場合）に表示されます。バナー行の警告と同じもので、自動承認モードであることを忘れないようにします。 |

コンテキストの数値やパーセンテージの前の `~` は、ローカルの推定値を含むことを意味します。これはゲートウェイの `/status` と `/context`、TUI、Desktop のコンテキストゲージにも当てはまります。変化のないプロバイダー使用量の値には `~` が付きません。プロバイダーの基準値に価格未確定の新しいメッセージが加わった場合は付きます。`/context` は選択されたソースを報告します。カテゴリ、空き容量、スキル、ツールセットの内訳は、全体の占有率がプロバイダーの使用量から得られる場合でも、常にローカルの推定値です。これらの表示ラベルは、圧縮の判断を変えることも、追加のプロバイダーリクエストを発生させることもありません。

バーはターミナル幅に適応します。76 桁以上でフルレイアウト、52〜75 桁でコンパクト、52 桁未満で最小（モデル名＋経過時間、アクティブな場合は YOLO バッジも）になります。

**コンテキストの色分け:**

| 色 | しきい値 | 意味 |
|-------|-----------|---------|
| 緑 | < 50% | 余裕あり |
| 黄 | 50〜80% | 埋まってきている |
| オレンジ | 80〜95% | 上限に近づいている |
| 赤 | ≥ 95% | オーバーフロー間近 — `/compress` を検討 |

入力トークンと出力トークンなどカテゴリ別のコストを含む詳細な内訳は `/usage` で確認できます。

`openai-codex` プロバイダーでは、`/usage` に ChatGPT アカウントで蓄積されている使用量上限のリセットも表示されます（"You have N resets banked - use /usage reset to activate"）。`/usage reset` は蓄積されたリセットを 1 つ使い、5 時間と週間の上限を完全に回復させます。上限を使い切っていない間は、Hermes は使用を拒否します（蓄積されたリセットは全量を回復させるため、早く使うと無駄になります）。それでも使う場合は `/usage reset --force` を指定してください。

### セッション再開の表示

以前のセッションを再開すると（`hermes -c` または `hermes --resume <id>`）、バナーと入力プロンプトの間に「Previous Conversation」パネルが表示され、対話履歴のコンパクトな要約が示されます。詳細と設定については [セッション — 再開時の対話要約](sessions.md#conversation-recap-on-resume) をご覧ください。

## キーバインド {#keybindings}

macOS では、`F6`/`F7` は最上段に表示されているメディア/システムコントロールではなく、物理的なファンクションキーを意味します。ファンクションキーを押す間 **Fn**（新しいキーボードでは **地球儀** キー）を押し続けるか、**システム設定 → キーボード → キーボードショートカット → ファンクションキー** で **F1、F2 などのキーを標準のファンクションキーとして使用** を有効にしてください。確実なターミナル上の代替は、`F6` には **Ctrl+T**、`F7` には **Ctrl+R** です。

| キー | 動作 |
|-----|--------|
| `Enter` | メッセージを送信 |
| `Alt+Enter`、`Ctrl+J`、または `Shift+Enter` | 改行（複数行入力）。`Shift+Enter` には、それを `Enter` と区別できるターミナルが必要です（下記参照）。Windows Terminal では `Alt+Enter` がターミナルに捕捉される（全画面表示の切り替え）ため、代わりに `Ctrl+Enter` または `Ctrl+J` を使ってください。 |
| `Alt+V` | ターミナルが対応している場合、クリップボードから画像を貼り付け |
| `Ctrl+V` | テキストを貼り付け、可能であればクリップボードの画像も添付 |
| `Ctrl+B` | ボイスモードが有効な場合、音声録音の開始/停止（`voice.record_key`、デフォルト: `ctrl+b`） |
| `Ctrl+G` | 現在の入力バッファを `$EDITOR`（vim/nvim/nano/VS Code など）で開く。保存して終了すると、編集したテキストが次のプロンプトとして送信されます — 長い複数段落のプロンプトに最適です。 |
| `Ctrl+X Ctrl+E` | 外部エディタ用の Emacs スタイルの代替バインド（`Ctrl+G` と同じ動作）。 |
| `Ctrl+S` | **プロンプトの退避（スタッシュ）。** 現在の下書きを退避してコンポーザーをクリアし、先に別の内容を送信できるようにします。空のコンポーザーで再度 `Ctrl+S` を押すと下書きが戻ります（カーソルは末尾、添付画像も復元）。繰り返し押すと上書きではなくスタックとして積み上がるため、以前の下書きが黙って失われることはありません — 2 つ以上退避している場合、`Ctrl+S` で一覧パネルが開きます（`↑`/`↓` で移動、`Enter` で復元、`D` で破棄、`Esc` または `Ctrl+S` で閉じる）。ステータスバーの `📌 N` バッジは退避中の下書きの数を示します。複数行の下書きは空行も含めて正確に復元されます。下書きには秘密情報が含まれることが多いため、スタッシュはセッション中のメモリにのみ保持され、ディスクには何も書き込まれません。 |
| `Ctrl+C` | まず入力中の下書きをクリア（エージェントの作業中でも）。プロンプトが空の場合はエージェントを割り込み（2 秒以内に 2 回押すと強制終了） |
| `Esc`, `Esc` | **バックトラック**（`/rewind` と同じ）。アイドル状態の空のプロンプトで、1 回目の `Esc` は `esc again to edit previous message` を表示し（入力を待ちます。他のキーを押すとキャンセルされ通常どおり動作します）、2 回目で **トランスクリプトブラウザー** が開きます: セッションのすべてのプロンプト（`›`）と返信（`•`）が画面に合わせたサイズのパネルに表示され、最新のプロンプトが反転表示で選択されます。`←`/`h` と `→`/`l` で古い/新しいプロンプトを選び、`↑`/`↓`（`k`/`j`）で 1 行、`PgUp`/`PgDn` で 1 ページスクロールします。`Enter` で選択したプロンプトを編集用にコンポーザーへ戻し、編集内容を送信するとセッションはそのプロンプトの直前まで切り戻されます（送信するまで何も削除されません。`Esc Esc`、`Ctrl+C`、または空の送信でキャンセル）。`f` はそのプロンプトの直前から、それを事前入力した新しいセッションをフォークし、元のセッションはそのまま残します。`Esc` または `q` で閉じ、その他のキーはブラウザーを閉じて通常どおり入力されます。下書きがある場合、`Esc Esc` は代わりにそれを破棄します（Up で呼び戻せます）。 |
| `Ctrl+T` / `F6` | コンポーザーの下書きを失わずに、全画面のライブ作業モニター（サブエージェントとバックグラウンドプロセス）を開く。ライブドックはステータスバーの上に自動的に表示されます。矢印キーでワーカーまたはプロセスを選択し、`Enter` で最近のログを表示、`s` でワーカーを誘導、`x` で確認付きの停止を要求します。[サブエージェントの監視](./features/delegation.md#monitoring-running-subagents-agents) をご覧ください。 |
| `Ctrl+R` / `F7` | コンポーザーのフォーカスを移さずに、ライブ作業ドックを複数行のプレビューと 1 行の要約の間で切り替える。macOS がファンクションキーの段を占有している場合は `Ctrl+R` が確実な代替です。ドックには、サブエージェントとバックグラウンドプロセスのほか、最上段に継続中の `/goal`（アクティブ、保留、または一時停止中で、使用済みターン数付き）、最下段に `/queue` で待機中のプロンプトが表示されます。 |
| `Ctrl+D` | 終了（空のプロンプトで。テキストがある場合はカーソル位置の文字を削除）。ゲートウェイがターンを実行中の場合は、ターンを実行したまま終了します: ゲートウェイがターンを完了させてからセッションを終了し、`hermes --resume` で返信を確認できます。その間に質問（承認や確認）を行うターンは、タイムアウトするかあなたが再開するまで待機します。ターンを停止したい場合は代わりに `Ctrl+C` を使ってください。 |
| `Ctrl+Z` | Hermes をバックグラウンドにサスペンド（Unix のみ）。シェルで `fg` を実行すると再開します。 |
| `Tab` | 自動サジェスト（ゴーストテキスト）を確定、またはスラッシュコマンドを自動補完。`modern` レイアウトでエージェントの作業中に入力済みのメッセージで `Tab` を押すと、次のターン用にキューに入ります |
| `!<command>` | **シェルモード** — モデルのターンを消費せずに自分でシェルコマンドを実行します（例: `!git status`、`!pytest -x`）。下記参照。 |

**複数行貼り付けのプレビュー。** 複数行のブロックを貼り付けると、CLI はペイロード全体をスクロールバックに流し込む代わりに、コンパクトな単一行プレビュー（`[pasted: 47 lines, 1,842 chars — press Enter to send]`）を表示します。送信される内容はあくまで全文であり、これは表示上の工夫にすぎません。

### `!` シェルモード

行を `!` で始めると、エージェントに送信する代わりにシェルコマンドとして実行します。

```
> !git status
> !ls -la
> !pytest -x tests/hermes_cli
```

- **コストゼロ。** モデルは一切呼び出されません — API 呼び出しも、トークンも、レイテンシーもありません。
- **対話には何も入りません。** コマンドとその出力は履歴に追加されないため、コンテキストはクリーンなまま保たれ、プロンプトキャッシュにも影響しません。
- **あなたのマシン上、セッションの作業ディレクトリで実行されます。** デフォルトのローカルターミナルバックエンドでは、`!pwd` はエージェントから見えるものと一致します。リモートまたはサンドボックス化された `terminal.backend`（`ssh`、`docker` など）は `!` コマンドには **使われません** — 常に Hermes 自体が動作しているホスト上で実行されるため、`!hostname` はあなたのマシン名を示し、エージェントの `terminal` ツールはバックエンド名を示します。バックエンドの *中で* 何かを実行するには、エージェントに頼む（またはターゲット上でシェルを開く）ようにしてください。一方、コンポーザーでのパス補完は設定されたバックエンドに従い、ターゲットのファイルシステムを一覧表示します。
- **承認は引き続き適用されます。** 危険なコマンド（`rm -rf`、`~/.hermes/config.yaml` への書き込みなど）は、エージェントの `terminal` ツールが使うのと同じ承認プロンプトを通ります。`!` はコスト/レイテンシーのための近道であり、セキュリティの回避手段ではありません。
- **ゼロ以外の終了コードは表示されます。** 失敗したコマンドは、出力の後に `! exited <code>` を表示します。
- `!` 単体では、1 行の使い方のリマインダーを表示します。

シェルモードは CLI 専用です。ゲートウェイプラットフォーム（Discord、Telegram、Slack）や cron の実行では無視されます — それらのユーザーはすでに自分のシェルを持っています。

**最終応答での Markdown 除去。** CLI は、エージェントの*最終*応答から最も冗長な Markdown フェンスや `**bold**` / `*italic*` のラッパーを取り除き、生のソースではなく読みやすいターミナルの文章として描画します。コードブロックとリストは保持されます。これはゲートウェイプラットフォームやツール結果には影響しません — それらはネイティブ描画のために Markdown を保持します。

## スラッシュコマンド

`/` を入力すると自動補完のドロップダウンが表示されます。Hermes は、多数の CLI スラッシュコマンド、動的なスキルコマンド、ユーザー定義のクイックコマンドに対応しています。

よく使う例:

| コマンド | 説明 |
|---------|-------------|
| `/help` | コマンドのヘルプを表示 |
| `/model` | 現在のモデルを表示または変更 |
| `/tools` | 現在利用可能なツールを一覧表示 |
| `/skills browse` | スキルハブと公式のオプションスキルを閲覧 |
| `/bg <prompt>` | プロンプトを別のバックグラウンドセッションで実行 |
| `/btw <question>` | 現在の対話を中断せずに、それについての脇道の質問をする |
| `/skin` | アクティブな CLI スキンを表示または切り替え |
| `/voice on` | CLI ボイスモードを有効化（`Ctrl+B` で録音） |
| `/voice tts` | Hermes の応答の音声再生を切り替え |
| `/reasoning high` | 推論の労力を増やす |
| `/title My Session` | 現在のセッションに名前を付ける |
| `/status` | セッション情報 — モデル/プロファイル/トークン/経過時間 — に続いて、ローカルの **Session recap** ブロック（最近のターン数、よく使われたツール、触れたファイル、最新のユーザープロンプトとアシスタントの返信）を表示。純粋にローカルでの計算で、LLM 呼び出しはありません。 |
| `/context [all]` | コンテキスト使用量の視覚的な内訳 — グリフのブロックグリッドとカテゴリ別のトークン表（システムプロンプト / ツール / スキル / メモリ / 対話 / 空き容量）。`/context all` はスキルごと・ツールセットごとのコストを追加します。 |
| `/sessions` | 従来の CLI 内で対話型のセッションピッカーを開く（TUI と同じ画面）。入力で絞り込み、矢印キーで移動、Enter で再開します。 |

組み込み CLI とメッセージングの完全な一覧については、[スラッシュコマンドリファレンス](../reference/slash-commands.md) をご覧ください。

セットアップ、プロバイダー、無音調整、メッセージング/Discord でのボイス利用については、[ボイスモード](features/voice-mode.md) をご覧ください。

:::tip
コマンドは大文字小文字を区別しません — `/HELP` は `/help` と同じように動作します。インストール済みのスキルも自動的にスラッシュコマンドになります。
:::

## クイックコマンド

LLM を呼び出さずにシェルコマンドを即座に実行するカスタムコマンドを定義できます。これらは CLI とメッセージングプラットフォーム（Telegram、Discord など）の両方で動作します。

```yaml
# ~/.hermes/config.yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

その後、任意のチャットで `/status`、`/gpu`、`/restart` と入力します。さらに多くの例については、[設定ガイド](./configuration.md#quick-commands) をご覧ください。

## 起動時にスキルをプリロードする

セッションで有効にしたいスキルがすでに分かっている場合は、起動時に渡せます。

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

Hermes は、最初のターンの前に、指定された各スキルをセッションのプロンプトに読み込みます。同じフラグは対話型モードと単発クエリモードの両方で機能します。

### 設定による永続的な自動読み込み {#persistent-auto-load-via-config}

**すべての** 新しいセッション — CLI、TUI、ゲートウェイ、cron、API セッションのいずれも — の開始時に同じスキルを有効にするには、`config.yaml` で `skills.auto_load` を設定します。

```yaml
skills:
  auto_load:
    - hermes-agent-dev
    - github-pr-workflow
```

各エントリはスキル名です。リストはセッションのシステムプロンプトが最初に構築されるときに一度だけ解決され、描画されたバイト列は対話の全期間（モデルの切り替え、圧縮を含む）で再利用されるため、プロンプトキャッシュはそのまま維持されます。設定の変更は次のセッションから反映されます。存在しないスキルや無効化されたスキルは警告をログに出してスキップされます。リストと重複する `-s` の名前は一度だけ読み込まれます。

`--ignore-rules`（`HERMES_IGNORE_RULES=1` と同等）は、AGENTS.md、SOUL.md、`.cursorrules`、メモリの注入とともに自動読み込みもスキップします。明示的な `-s` のスキルは引き続き読み込まれます。この設定はプロファイル単位です: 各プロファイルの `config.yaml` がそれぞれのリストを管理します。

## スキルのスラッシュコマンド

`~/.hermes/skills/` にインストールされたすべてのスキルは、自動的にスラッシュコマンドとして登録されます。スキル名がそのままコマンドになります。

```
/excalidraw sketch the request flow for our API
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

## パーソナリティ

定義済みのパーソナリティを設定して、エージェントの口調を変更できます。

```
/personality pirate
/personality kawaii
/personality concise
```

組み込みのパーソナリティには次のものがあります: `helpful`、`concise`、`technical`、`creative`、`teacher`、`kawaii`、`catgirl`、`pirate`、`shakespeare`、`surfer`、`noir`、`uwu`、`philosopher`、`hype`。

デフォルト（オーバーレイなし）に戻すには、`/personality none` を使います — `default` や `neutral` でも機能します。

`~/.hermes/config.yaml` でカスタムパーソナリティを定義することもできます。

```yaml
personalities:
  helpful: "You are a helpful, friendly AI assistant."
  kawaii: "You are a kawaii assistant! Use cute expressions..."
  pirate: "Arrr! Ye be talkin' to Captain Hermes..."
  # Add your own!
```

## 複数行入力

複数行のメッセージを入力する方法は 2 つあります。

1. **`Alt+Enter`、`Ctrl+J`、または `Shift+Enter`** — 改行を挿入します
2. **バックスラッシュ継続** — 行末を `\` で終わらせて続けます:

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

`Ctrl+J` とバックスラッシュ継続はデフォルトで有効で、Claude Code / Codex / OpenCode の複数行ショートカットに合わせています。iTerm2 などの対応ターミナルでは、Hermes は拡張キーレポートも要求するため、`Shift+Enter` が独立した改行キーとして届きます。ターミナルが通常の `Enter` で LF を送信し、従来の `Ctrl+J` を送信として扱うフォールバックが必要な場合は、オプトアウトしてください。

```yaml
# ~/.hermes/config.yaml
display:
  cli_multiline_shortcuts: false
```

:::info
複数行テキストの貼り付けに対応しています — 上記のいずれかの改行キーを使うか、単純に内容を直接貼り付けてください。

Kitty キーボードプロトコルを使うターミナルでは、テンキーの `Alt+Enter` でも改行が挿入されます。折りたたまれた貼り付けの隣でも同様です。修飾キー付きのテンキーのナビゲーションキーは、テンキー以外の対応するキーと同じように動作します。
:::

### Shift+Enter の互換性

ほとんどのターミナルはデフォルトで `Enter` と `Shift+Enter` に同じバイトシーケンスを送信するため、アプリケーションはこれらを区別できません。Hermes が `Shift+Enter` を認識するのは、ターミナルが [Kitty キーボードプロトコル](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) または xterm の `modifyOtherKeys` モードを介して固有のシーケンスを送信する場合のみです。

| ターミナル | 状況 |
|---|---|
| Kitty、foot、WezTerm、Ghostty | 固有の `Shift+Enter` がデフォルトで有効 |
| iTerm2（最近のバージョン）、Alacritty、VS Code ターミナル、Warp | 設定で Kitty プロトコルを有効にすると対応 |
| Windows Terminal Preview 1.25+ | 設定で Kitty プロトコルを有効にすると対応 |
| macOS の Terminal.app、標準の Windows Terminal（安定版） | 非対応 — `Shift+Enter` は `Enter` と区別できません |

ターミナルがこれらを区別できない場合でも、`Alt+Enter` と `Ctrl+J` はデフォルトでどこでも引き続き機能します。**特に Windows Terminal では、`Alt+Enter` がターミナルに捕捉され（全画面表示の切り替え）Hermes には届かないため、改行には `Ctrl+Enter`（`Ctrl+J` として送られます）または `Ctrl+J` を直接使ってください。**

## ターン途中でのエージェントの方向修正

エージェントの作業中に、新しいターンを開始せずに修正を送ることができます。

- **新しいメッセージを入力して Enter** — 修正内容を使って進行中のターンの方向を変えます
- **`Ctrl+C`** — 現在の操作を割り込み（2 秒以内に 2 回押すと強制終了）。プロンプトに下書きがある場合、1 回目の押下でそれがクリアされ、ターンは実行を続けます
- 完了したツール作業とすでに表示された推論はコンテキストに残ります
- 実行中のツールは、修正が適用される前に安全な区切りまで到達します

### ビジー入力モード

`display.busy_input_mode` の設定キーは、エージェントの作業中に Enter を押したときの動作を制御します。

| モード | 動作 |
|------|----------|
| `"interrupt"`（デフォルト） | メッセージが進行中のターンの方向を変えます。表示済みの推論と完了した作業は保持されたまま、モデルの生成が再開されます。実行中のフォアグラウンドのターミナルコマンドはバックグラウンドへ移され（終了はされず、完了通知が届きます）、メッセージがすぐに読まれるようにします。その他の実行中のツールは先に完了します |
| `"queue"` | メッセージはエージェントの完了後の次のターンとしてキューに入れられます。キューはゲートウェイ上にあります（クラッシュしても失われず、アタッチしているすべてのクライアントから見えます）。 |
| `"steer"` | メッセージは `/steer` 経由で現在の実行に注入され、次のツール呼び出しの後にエージェントへ届きます — 割り込みも新しいターンもありません |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # or "queue" or "interrupt" (default)
```

`"queue"` モードは別のフォローアップのターンを準備します。`"steer"` は常に次のツール結果の区切りを待ちます。デフォルトの `"interrupt"` モードは、実行中のツールのキャンセルを避けつつ、モデルの生成中により早く応答します。長時間のフォアグラウンドの `terminal` コマンド（ビルドやポーリングなど）はバックグラウンドへ引き渡されるため、エージェントはコマンドの終了後ではなくすぐにメッセージを確認できます。ターンとそのフォアグラウンドの作業をキャンセルしたい場合は `/stop` を使ってください。不明な値は `"interrupt"` にフォールバックします。

`"steer"` には自動フォールバックが 2 つあります。エージェントがまだ開始していない場合、または画像が添付されている場合、メッセージは `"queue"` の動作にフォールバックし、何も失われないようにします。

どのモードでも、`/queue <prompt>` はフォローアップのターンを明示的にキューに入れます。また、`/queue list`、`/queue rm N`、`/queue edit N …`、`/queue move A B` は、実行中であっても保留中のキューに即座に作用します。ライブ作業ドックには待機中の内容が一覧表示されます。

CLI がゲートウェイのセッション上で動作している場合（デフォルトのローカル構成）、ターンの実行中に `/queue` またはキューモードの Enter でキューに入れたプロンプトは、そのセッション用のゲートウェイ自身のキューに入ります。これはゲートウェイの再起動後も残り、入力した順に一度だけ実行されます。画像付きのフォローアップは引き続き CLI 側で待機します。このキューは共有されます: `/queue list`、`rm`、`edit`、`clear` は、別のウィンドウや自動化がキューに入れたものも含め、そのセッションでキューに入っているすべてのプロンプトを表示・変更します。ゲートウェイはキューを順番どおりに実行し並べ替えはできないため、`/queue move` はゲートウェイが保持するプロンプトの移動を拒否します。自分の `/queue` プロンプトの後に、ビジー状態で入力したスラッシュコマンドは、そのプロンプトのターンが開始した後に実行されます。待機は最大でも数秒なので、一時停止中のキューが入力をブロックすることはありません。

CLI 内で変更することもできます。

```text
/busy queue
/busy steer
/busy interrupt
/busy status
```

:::tip 初回ヒント
Hermes の作業中に初めて Enter を押すと、Hermes は `/busy` の設定項目を説明する 1 行のリマインダーを表示します。これはインストールごとに 1 回だけ表示されます — `config.yaml` の `onboarding.seen.busy_input_prompt` が表示済みであることを記録します。このキーを削除すると、ヒントを再度表示できます。
:::

### バックグラウンドへのサスペンド

Unix システムでは、**`Ctrl+Z`** を押すと Hermes をバックグラウンドにサスペンドできます — 他のターミナルプロセスと同様です。シェルは確認メッセージを表示します。

```
Hermes Agent has been suspended. Run `fg` to bring Hermes Agent back.
```

シェルで `fg` と入力すると、中断したところから正確にセッションを再開します。これは Windows では非対応です。

## ツール進行状況の表示

CLI は、エージェントの作業に合わせてアニメーション付きのフィードバックを表示します。

**思考アニメーション**（API 呼び出し中）:
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

**ツール実行フィード:**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

`/verbose` で表示モードを切り替えます: `off → new → all → verbose`。このコマンドはメッセージングプラットフォーム向けにも有効化できます — [設定](./configuration.md#display-settings) をご覧ください。

### ツールプレビューの長さ

`display.tool_preview_length` の設定キーは、ツール呼び出しのプレビュー行（ファイルパス、ターミナルコマンドなど）に表示される最大文字数を制御します。デフォルトは `0` で、制限なしを意味します — 完全なパスとコマンドが表示されます。

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # Truncate tool previews to 80 chars (0 = no limit)
```

これは、幅の狭いターミナルや、ツールの引数に非常に長いファイルパスが含まれる場合に便利です。

### インライン画像

画像はトランスクリプト内に直接表示されます: 添付した画像（`/paste`、`/image`、ドラッグ＆ドロップ）はメッセージの下に、成功したツール呼び出しが生成または参照したローカル画像（`vision_analyze`、`image_generate`、ブラウザーまたはコンピューター操作のスクリーンショット）はそのツールの行の下に表示されます。`read_file` は画像を読めないため、何も表示しません。同じ設定が [TUI](./tui.md) にも適用されます。

```yaml
# ~/.hermes/config.yaml
display:
  inline_images: auto   # auto | kitty | iterm | unicode | off
```

- **`auto`** は、kitty と Ghostty（kitty グラフィックスプロトコル、Unicode プレースホルダー）、および従来の CLI では iTerm2、WezTerm、Konsole（予約した行の上に描画する iTerm2 インライン画像プロトコル）でフル解像度の画像を描画します。VS Code ターミナルでは、従来の CLI が起動時に画像が有効かどうかをターミナルに問い合わせ（`terminal.integrated.enableImages`。応答には約 10 ms かかります）、有効な場合はフル解像度の画像を描画し、無効な場合はときどき有効化を勧めるヒントを表示します。それ以外の環境（tmux、GNU screen、VS Code への SSH セッション、kitty/Ghostty 以外での TUI を含む）では、トゥルーカラーのハーフブロックで表示されます。
- **`kitty`** は kitty プロトコルを強制します。たとえば `set -g allow-passthrough on` を設定した tmux 内の従来の CLI で使います。
- **`iterm`** は従来の CLI で iTerm2 プロトコルを強制します（TUI ではここで `auto` にフォールバックします）。
- **`unicode`** は常にハーフブロックを使います。
- **`off`** は従来どおりファイルパスのみを表示します。

描画されるのはローカルファイルのみで、リモートの画像 URL は取得されません。Sixel ターミナルではハーフブロックで表示されます。

ターミナルのサイズ変更後、CLI はトランスクリプトを再描画します。kitty とハーフブロックの画像は再描画で復元されます。iTerm2 プロトコルの画像（iTerm2、WezTerm、Konsole）は再描画された画面では空白の行になり、その上のスクロールバック内でのみ表示されたままになります。

## セッション管理

### セッションの再開

CLI セッションを終了すると、再開用のコマンドが表示されます。

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

再開のオプション:

```bash
hermes --continue                          # Resume the most recent CLI session
hermes -c                                  # Short form
hermes -c "my project"                     # Resume a named session (latest in lineage)
hermes -c "my project" --create-if-missing # Same, creating the titled session if none exists
hermes --resume 20260225_143052_a1b2c3     # Resume a specific session by ID
hermes --resume "refactoring auth"         # Resume by title
hermes --resume latest                     # Resume the most recent session (same as -c)
hermes --resume latest --in ./my-project   # Latest session for ./my-project's workspace
hermes -r 20260225_143052_a1b2c3           # Short form
```

再開すると、SQLite から完全な対話履歴が復元されます。エージェントは、以前のすべてのメッセージ、ツール呼び出し、応答を参照できます — まるで一度も離れなかったかのようにです。

チャット内で `/title My Session Name` を使って現在のセッションに名前を付けるか、コマンドラインから `hermes sessions rename <id> <title>` を使います。過去のセッションを閲覧するには `hermes sessions list` を使います。

### セッションの保存

CLI セッションは、Hermes の SQLite 状態データベース `~/.hermes/state.db` に保存されます。このデータベースは次のものを保持します。

- セッションメタデータ（ID、タイトル、タイムスタンプ、トークンカウンター）
- メッセージ履歴
- 圧縮/再開されたセッションをまたぐ系統
- `session_search` が使用する全文検索インデックス

一部のメッセージングアダプターは、データベースとは別にプラットフォームごとのトランスクリプトファイルも保持しますが、CLI 自体は SQLite のセッションストアから再開します。

### コンテキスト圧縮

長い対話は、コンテキスト上限に近づくと自動的に要約されます。

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.50    # Compress at 50% of context limit by default

# Summarization model configured under auxiliary:
auxiliary:
  compression:
    model: ""  # Leave empty to use the main chat model (default). Or pin a cheap fast model, e.g. "google/gemini-3-flash-preview".
```

圧縮が発動すると、中間のターンが要約され、最初の 3 ターンと最後の 20 ターンは常に保持されます。

## バックグラウンドセッション {#background-sessions}

他の作業のために CLI を使い続けながら、別のバックグラウンドセッションでプロンプトを実行できます。

```
/bg Analyze the logs in /var/log and summarize any errors from today
```

Hermes はすぐにタスクを確認し、プロンプトを返します。

```
🔄 Background task #1 started: "Analyze the logs in /var/log and summarize..."
   Task ID: bg_143022_a1b2c3
```

### 仕組み

各 `/bg` プロンプトは、デーモンスレッド内に **完全に独立したエージェントセッション** を生成します。

- **独立した対話** — バックグラウンドエージェントは、現在のセッションの履歴を一切認識しません。提供したプロンプトのみを受け取ります。
- **同じ設定** — バックグラウンドエージェントは、現在のセッションからモデル、プロバイダー、ツールセット、推論設定、フォールバックモデルを引き継ぎます。
- **ノンブロッキング** — フォアグラウンドのセッションは完全に対話可能なままです。チャットしたり、コマンドを実行したり、さらにバックグラウンドタスクを開始したりできます。
- **複数タスク** — 複数のバックグラウンドタスクを同時に実行できます。それぞれに番号付きの ID が割り当てられます。

### 結果

バックグラウンドタスクが完了すると、結果がターミナルにパネルとして表示されます。

```
╭─ ☤ Hermes (background #1) ──────────────────────────────────╮
│ Found 3 errors in syslog from today:                         │
│ 1. OOM killer invoked at 03:22 — killed process nginx        │
│ 2. Disk I/O error on /dev/sda1 at 07:15                      │
│ 3. Failed SSH login attempts from 192.168.1.50 at 14:30      │
╰──────────────────────────────────────────────────────────────╯
```

タスクが失敗した場合は、代わりにエラー通知が表示されます。設定で `display.bell_on_complete` が有効になっている場合、タスクの完了時にターミナルのベルが鳴ります。

### ユースケース

- **長時間のリサーチ** — コードに取り組みながら「/bg research the latest developments in quantum error correction」
- **ファイル処理** — 対話を続けながら「/bg analyze all Python files in this repo and list any security issues」
- **並行調査** — 複数のバックグラウンドタスクを開始し、異なる角度から同時に探求する

:::info
バックグラウンドセッションはメインの対話履歴には表示されません。独自のタスク ID（例: `bg_143022_a1b2c3`）を持つスタンドアロンのセッションです。
:::

## クワイエットモード

デフォルトでは、CLI はクワイエットモードで動作し、次のことを行います。
- ツールからの冗長なログを抑制する
- kawaii スタイルのアニメーションフィードバックを有効にする
- 出力をクリーンでユーザーフレンドリーに保つ

デバッグ出力を得るには:
```bash
hermes chat --verbose
```
