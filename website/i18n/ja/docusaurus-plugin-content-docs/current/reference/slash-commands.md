---
sidebar_position: 2
title: "スラッシュコマンドリファレンス"
description: "インタラクティブ CLI およびメッセージングのスラッシュコマンドの完全リファレンス"
---

# スラッシュコマンドリファレンス

Hermes には 2 つのスラッシュコマンドのインターフェースがあり、どちらも `hermes_cli/commands.py` 内の中央の `COMMAND_REGISTRY` によって駆動されます:

- **インタラクティブ CLI スラッシュコマンド** — `cli.py` によってディスパッチされ、レジストリからの自動補完を備えます
- **メッセージングスラッシュコマンド** — `gateway/run.py` によってディスパッチされ、ヘルプテキストとプラットフォームメニューがレジストリから生成されます

インストール済みスキルも、両方のインターフェースで動的なスラッシュコマンドとして公開されます。（`/plan` は以前はその 1 つでしたが、現在は組み込みコマンドです — 下記のセッションの表を参照してください。）名前が組み込みコマンド（またはそのエイリアスのいずれか）と一致するスキルには、独自の `/<name>` が割り当てられることはありません — 組み込みが優先され、スキルは `/skill <name>` で引き続き読み込めます。`/skills list`、`/help skills`、コマンドパレットでは、そのようなスキルに `slash command /<name> unavailable — name taken by built-in; use /skill <name>` という印が付きます。

## 権限と管理者/ユーザーの区分 {#permissions-and-adminuser-split}

ユーザーごとの許可リストをサポートするすべてのメッセージングプラットフォーム（Telegram、Discord、Slack、Matrix、Mattermost、Signal など）は、2 段階のスラッシュコマンド区分もサポートしています: **管理者**は登録されたすべてのコマンドを使用でき、**一般ユーザー**は `user_allowed_commands` に列挙した名前（と、常に許可される最低限の `/help` と `/whoami`）のみを使用できます。`~/.hermes/config.yaml` 内のプラットフォームの `extra:` ブロックで、`allow_admin_from` と `user_allowed_commands`（およびグループごとの同等設定 `group_allow_admin_from` / `group_user_allowed_commands`）を設定します。

例については各プラットフォームのドキュメントを参照してください — 構造はプラットフォーム間で同一です:

- [Telegram](../user-guide/messaging/telegram.md#slash-command-access-control)
- [Discord](../user-guide/messaging/discord.md)
- [Slack](../user-guide/messaging/slack.md)
- [Matrix](../user-guide/messaging/matrix.md)
- [Mattermost](../user-guide/messaging/mattermost.md)
- [Signal](../user-guide/messaging/signal.md)

あるスコープで `allow_admin_from` が未設定の場合、そのスコープは制限のない後方互換モードのままになります — 許可されたすべてのユーザーがすべてのコマンドを実行できます。

## インタラクティブ CLI スラッシュコマンド {#interactive-cli-slash-commands}

CLI で `/` を入力すると自動補完メニューが開きます。組み込みコマンドは大文字小文字を区別しません。

### セッション {#session}

| コマンド | 説明 |
|---------|------|
| `/new [name]`（エイリアス: `/reset`） | 新しいセッションを開始（新しいセッション ID + 履歴）。オプションの `[name]` は最初のセッションタイトルを設定します — 例えば `/new my-experiment` は、最初から `my-experiment` というタイトルが付いた新しいセッションを開くため、後で `/resume` や `/sessions` で簡単に見つけられます。確認モーダルをスキップするには `now`、`--yes`、`-y` を付けます — 例: `/reset now`、`/new --yes my-experiment`。 |
| `/clear` | 画面をクリアして新しいセッションを開始 |
| `/history` | 会話履歴を表示（`/timestamps` の設定に従う） |
| `/save <json\|md\|html> [filename] [redact]` | 現在のセッションを JSON スナップショット、Markdown トランスクリプト、またはスタンドアロンの HTML ページとしてエクスポート（`markdown` と `snapshot` もエイリアスとして受け付けます）。`redact` は書き込み前に API キー、トークン、認証情報を除去し、常にオプションのファイル名の後、最後に置きます: `/save md notes.md redact`。セッションがリモートゲートウェイ上で実行されている場合でも、ファイルは手元のマシンに書き込まれます（`session.export` を持たない古いゲートウェイでは、デフォルト名の JSON をゲートウェイ自身のホストにしか保存できません）。CLI と TUI: `filename` は現在のディレクトリを基準に解決され、指定しない場合は `~/.hermes/sessions/saved/` 配下に自動で名前が付けられます。Desktop は保存ダイアログを開きます（`filename` を指定した場合はそれが事前入力されます）。引数なしの `/save` は、クラシック CLI では使い方を表示し、TUI と Desktop では JSON を保存します。 |
| `/prompt`（エイリアス: `/compose`） | インライン入力の代わりに `$EDITOR`（Markdown）で次のプロンプトを作成 — 長い、複数行の、または慎重に整形したプロンプトに便利です。 |
| `/retry` | 最後のメッセージを再試行（エージェントに再送信） |
| `/undo` | 最後のユーザー/アシスタントのやり取りを削除 |
| `/rewind [N \| fork N]` | **巻き戻し**（クラシック CLI。アイドル状態の空のプロンプトで `Esc`、`Esc` でも可: 1 回目で `esc again to edit previous message` が表示されます）。プロンプトと応答のトランスクリプトブラウザを開き、最新のプロンプトが反転表示で選択された状態になります: `←`/`→`（`h`/`l`）でプロンプトを選び、`↑`/`↓`（`k`/`j`）と `PgUp`/`PgDn` でスクロールします。`Enter` で選択したプロンプトを編集用にコンポーザーに戻します: 編集内容を送信すると、セッションはそのプロンプトの直前まで切り戻され（以降のメッセージは `/undo` と同様にアーカイブされます）、新しいテキストが送信されます。`Esc Esc`、`Ctrl+C`、または空の送信でキャンセルした場合は何も削除されません。`f` は代わりにフォークします: そのプロンプトより前の履歴だけを持つ新しい分岐セッションが作られ、プロンプトが事前入力されます（送信はされません）。元のセッションはそのまま保持されます。`Esc` または `q` で閉じます。`/rewind N` はプロンプト N を直接編集し、`/rewind fork N` はその直前でフォークします。 |
| `/title` | 現在のセッションにタイトルを設定（使い方: /title My Session Name） |
| `/compress [here [N] \| focus topic]` | 会話コンテキストを手動で圧縮（メモリをフラッシュ + 要約）。`/compress here [N]` は、最新の N 件のやり取り（デフォルト 2）をそのまま残し、それ以外をすべて要約します — 圧縮の境界を自分で選べます。フォーカストピックを指定すると、完全な要約が保持する内容を絞り込めます。 |
| `/rollback` | ファイルシステムのチェックポイントを一覧表示または復元（使い方: /rollback [number]） |
| `/diff [staged\|all\|session] [--stat] [path...]` | 作業ディレクトリの git の変更を表示。デフォルト: ステージされていない変更と未追跡ファイル。`staged` はコミット用にステージされた内容、`all` は HEAD 以降のすべて、`session` は Hermes がここで変更したすべての累積差分を表示します（保持されている最も古いチェックポイントのベースラインから — チェックポイントの有効化が必要で、`/rollback diff <N>` を補完します）。`--stat` は変更ファイルの概要のみを表示し、パス引数で差分を絞り込めます。 |
| `/snapshot [create\|restore <id>\|prune]`（エイリアス: `/snap`） | Hermes の設定/状態のスナップショットを作成または復元。`create [label]` はスナップショットを保存、`restore <id>` はそれに戻す、`prune [N]` は古いスナップショットを削除、引数なしですべてを一覧表示します。クラシックの `hermes chat` では、復元時にローカルのリーダーを閉じてローカルゲートウェイをドレインしてからオフラインで復元し、復元された履歴で再接続します（スナップショットがセッションより古い場合は新しいセッションを作成します）。多重化されたセカンダリプロファイルは、他のプロファイルを再起動せずに停止・再提供されます。起動プロファイルを復元する場合はゲートウェイ全体がドレインされます。データベースの復元では、排他メンテナンスのゲートと SQLite バックアップの安全性チェックが維持されます: 停止できない保持者がいる場合は、引き続き復元が妨げられます。 |
| `/stop` | 実行中のすべてのバックグラウンドプロセスを終了 |
| `/queue <prompt>`（エイリアス: `/q`） | 次のターン用にプロンプトをキューに入れる（現在のエージェント応答を中断しない）。CLI では、引数なしの `/queue` は保留中のキューを一覧表示し、`/queue list`、`/queue edit N <new prompt>`、`/queue rm N`、`/queue move A B`、`/queue clear` で、送信前にキュー内のプロンプトを確認・管理できます。管理用の語で始まるプロンプトは `/queue add <prompt>` で強制的にキューに入れられます。ゲートウェイのセッションでは、実行中にキューに入れたプロンプトはそのセッション用のゲートウェイ共有キューに置かれます: `list`/`rm`/`edit`/`clear` はそこにキューされたすべてのプロンプト（他のウィンドウのものも含む）に作用し、ゲートウェイが保持するプロンプトについては、順番に実行するため `move` は拒否されます。 |
| `/steer <prompt>` | **次のツールコールの後**にエージェントに届く実行中の注記を注入 — 割り込みなし、新しいユーザーターンなし。テキストは、現在のツールが完了すると最後のツール結果のコンテンツに追加され、現在のツールコールループを壊さずにエージェントに新しいコンテキストを与えます。タスクの途中で方向を微調整するのに使います（例: エージェントがテストを実行している間に「認証モジュールに集中して」）。 |
| `/goal <text>` | Hermes がターンをまたいで取り組む常設のゴールを設定 — Ralph ループに対する私たちのアプローチです。各ターンの後、補助的なジャッジモデルがゴールが完了したかどうかを判断します。完了していなければ Hermes が自動で続行します。サブコマンド: `/goal status`、`/goal pause`、`/goal resume`、`/goal clear`。予算はデフォルトで 20 ターン（`goals.max_turns`）。実際のユーザーメッセージは継続ループをプリエンプトし、状態は `/resume` を生き残ります。完全なウォークスルーは [Persistent Goals](../user-guide/features/goals.md) を参照してください。 |
| `/subgoal <text>` | ループ実行中に、ユーザーが指定した基準をアクティブなゴールに追加。継続プロンプトはすべてのサブゴールをそのままエージェントに提示し、ジャッジはそれらを DONE/CONTINUE の判定に反映します — そのため、元のゴール**と**すべてのサブゴールが満たされるまでゴールは完了とみなされません。サブコマンド: `/subgoal`（一覧）、`/subgoal remove <N>`、`/subgoal clear`。アクティブな `/goal` が必要です。 |
| `/heartbeat every <interval> <prompt>`（エイリアス: `/hb`） | アイドル状態で間隔が経過するたびに、通常のユーザーターンとして**このセッション**に再投入される定期プロンプトを設定（最小 60 秒。逃したティックはまとめられます）。サブコマンド: `/heartbeat status`、`/heartbeat pause`、`/heartbeat resume`、`/heartbeat clear`。セッションスコープかつプロセス内で動作します — 永続的で分離されたスケジュールには `hermes cron` を使用してください。[Session Heartbeats](/docs/user-guide/features/heartbeat) を参照してください。 |
| `/loop [interval] <prompt> [--times N] [--until <condition>]`（エイリアス: `/proactive`） | **このセッション**で、プロンプト（または別のスラッシュコマンド）を定期的な間隔で再実行 — 固定間隔（`/loop 5m check the deploy`）、または間隔を指定しない場合は連続して再発火します。`--times N` は実行回数に上限を設け、`--until <condition>` は条件が満たされたときに補助ジャッジがループを停止できるようにします。サブコマンド: `/loop status`、`/loop pause`、`/loop resume`、`/loop stop`。[Loops](/docs/user-guide/features/loops) を参照してください。 |
| `/refine [focus]` | 自動のターン後トリガーを待たずに、バックグラウンドのメモリ/スキル自己改善レビューを**今すぐ**実行。オプションのフォーカステキストでレビューの方向を指定できます（例: `/refine save the deploy workflow as a skill`）。会話のスナップショットに対してバックグラウンドのフォークで実行されるため、ライブセッションとプロンプトキャッシュには影響しません。完了すると結果が報告されます。 |
| `/review [instructions]` | 直前に話し合った作業 — PR、コード、ドキュメント、直近 10 件のチャットメッセージで参照された任意の成果物 — をレビューする、独立したフル権限のレビュアーサブエージェントを起動。バックグラウンドで調査し（PR を開き、差分を読み、コードを実行）、完全なレビューはバックグラウンドサブエージェントの完了としてこのセッションに戻り、プライマリエージェントがそれに基づいて行動できます。config.yaml の `auxiliary.review` で専用のレビューモデルを固定できます（デフォルトはメインモデル）。[Subagent Delegation](../user-guide/features/delegation.md#the-review-command) を参照してください。 |
| `/moa <prompt>` | 1 つのプロンプトをデフォルトの [Mixture of Agents](/docs/user-guide/features/mixture-of-agents) プリセットで実行し、その後現在のモデルに戻します。1 回限り — セッションのモデルは変更しません。 |
| `/resume [name]` | 以前に名前を付けたセッションを再開。引数なしの `/resume` は、このチャットの名前付きセッションを最近のアクティビティ順に一覧表示します（圧縮によって何度もローテーションされた長期間の会話は 1 つのエントリとして数えられ、ライブの最先端で表示されます）。クラシック CLI では、ターンの実行中はこのコマンド（と `/sessions <id>`）は拒否されます — CLI は 1 つのエージェントをセッション間で共有しているため、ターンの途中で切り替えると、実行中のターンの残りが別のセッションに記録されてしまうためです。 |
| `/sessions`（TUI エイリアス: `/switch`） | クラシック CLI: インタラクティブなピッカーで以前のセッションをブラウズして再開。TUI: 現在開いている TUI セッションのライブセッションスイッチャーを開きます。TUI で別のライブセッションをすぐに開始するには `/sessions new` を使います。 |
| `/egress [status]` | Docker エグレスプロキシのステータスを表示 — 有効/設定済み/実行中の状態、認証情報のソース、トークンマッピング、カバーされていないプロバイダー、次の対処手順。CLI、TUI、Desktop チャット、メッセージングゲートウェイで動作します。 |
| `/redraw` | UI 全体を強制的に再描画（tmux のリサイズ後のターミナルのずれ、マウス選択のアーティファクトなどから復旧） |
| `/status` | セッション情報 — モデル、プロバイダー、プロファイル、セッション ID、作業ディレクトリ、タイトル、作成/更新タイムスタンプ、トークン合計、エージェントの実行状態 — を表示し、続いてローカルの **Session recap** ブロック（最近のユーザー/アシスタントのターン数、ツール結果数、よく使われたツール、直近に触れたいくつかのファイル、最新のユーザープロンプト、最新のアシスタント応答）を表示します。recap はメモリ上の会話からローカルで計算されます。LLM 呼び出しはなく、プロンプトキャッシュにも影響しません。 |
| `/context [all]`（エイリアス: `/ctx`） | コンテキストウィンドウの視覚的な内訳。CLI/TUI では、5×20 のグリフブロックグリッド（各セル ≈ モデルウィンドウの 1%）と、カテゴリ別の推定表 — システムプロンプト、ツール定義、ルール、スキルインデックス、MCP、サブエージェント、メモリ、会話 — と空き容量を表示します。メッセージングプラットフォームでは、自動圧縮のしきい値/余裕を含む使用量ゲージ、圧縮の統計、累積スループット、同じカテゴリ表をプレーンテキストで表示します。CLI、メッセージングプラットフォーム、コンピュートホスト上の TUI/Desktop セッションでは、出力の最後にファイルごとの **Context files** 一覧（.hermes.md、AGENTS.md チェーン、CLAUDE.md、.cursorrules + .cursor/rules/*.mdc、SOUL.md）が表示され、各ファイルのトークン推定値と、読み込まれたか、`context_file_max_chars` を超えて切り詰められたか、より優先度の高いコンテキスト種別に隠されたか、インジェクションスキャンでブロックされたか、空/読み取り不能か、インストールツリーのガードで抑止されたかが示されます — 「なぜ CLAUDE.md が無視されるのか？」への答えです。`/context all` は、スキルごと・ツールセットごとのコスト一覧（インデックスのコストと SKILL.md 読み込みのコスト、ツールセットごとのスキーマトークン）を追加します。読み取り専用でローカルに計算されます — LLM 呼び出しはなく、プロンプトキャッシュにも影響しません。 |
| `/agents`（エイリアス: `/tasks`） | 現在のセッションのアクティブなエージェントと実行中のタスクを表示。 |
| `/bg <prompt>` | 別のバックグラウンドセッションでプロンプトを実行。エージェントはプロンプトを独立して処理 — 現在のセッションは他の作業のために空いたままです。結果はタスク完了時にパネルとして表示されます。[CLI Background Sessions](../user-guide/cli.md#background-sessions) を参照してください。 |
| `/btw <question>` | 現在の会話を中断せずに、**現在の会話についての**ちょっとした質問をする。1 回限りの補助 LLM 呼び出しが、トランスクリプトの読み取り専用スナップショットから回答します — ライブセッションの履歴とプロンプトキャッシュには影響せず、現在のターンは実行を続けます。新しいコンテキストでの独立した作業には `/bg` を使用してください。 |
| `/branch [--here] [name]`（エイリアス: `/fork`） | 現在のセッションを独立したコピーにブランチ（別のパスを探る）。Discord、Telegram、Slack、Matrix では、ブランチは**新しい兄弟スレッド**で開かれ、現在のチャットは元のセッションに留まります。`--here` は代わりに現在のチャットをブランチに切り替えます（#66023 以前の動作）。CLI とスレッドのないプラットフォームでは、常にその場でブランチします。クラシック CLI: `/handoff` と同様にターンの途中では拒否されます — 現在の応答が終わるのを待ってから再試行してください。 |
| `/worktree [new [name]\|list]` | **CLI 専用。** セッションの途中で、分離された git worktree を確認または作成します（Copilot CLI の `/worktree new` に着想を得たもの）。引数なしの `/worktree` はアクティブな worktree を表示し、`/worktree list` はリポジトリの worktree を一覧表示し、`/worktree new [name]` は `.worktrees/` 配下に worktree を作成して（`worktree_sync` に従い、新たにフェッチしたリモートの最先端からブランチ）、セッションのターミナルツールとファイルツールの対象をそこに切り替えます。名前付きのツリーには指定した名前（`hermes/<name>` ブランチ）が使われ、名前なしのものにはランダムな `hermes-<id>` が付きます。終了時、未コミットまたは未プッシュのツリーがあると、保持 / アーカイブ / 削除の質問が表示されます — `hermes -w` と同じライフサイクルです。[Git Worktrees](../user-guide/git-worktrees.md) を参照してください。 |
| `/handoff <platform>` | **CLI 専用。** 現在のセッションをメッセージングプラットフォーム（Telegram、Discord、Slack、WhatsApp、Signal、Matrix）に引き継ぎます。ゲートウェイが即座に引き取り、スレッドをサポートするプラットフォーム（Telegram トピック、Discord テキストチャンネルスレッド、Slack と Matrix のメッセージアンカースレッド）では新しいスレッドを作成し、ロール認識のフルトランスクリプトが再生されるように宛先を CLI の session_id に再バインドし、合成ユーザーターンを生成してエージェントが新しい場所で動作していることを確認します。成功時、CLI は `/resume` ヒントとともにクリーンに終了します。`/resume <title>` でいつでもローカルに再開できます。ターンの途中では拒否されます。ゲートウェイが実行中で、ターゲットプラットフォームにホームチャンネルが設定されている（宛先チャットから `/sethome`）必要があります。[Cross-Platform Handoff](../user-guide/sessions.md#cross-platform-handoff) を参照してください。 |
| `/journey [list\|delete <id>\|edit <id>]`（エイリアス: `/learning`、`/memory-graph`） | 学習したスキルとメモリの学習ジャーニーのタイムラインを開く。クラシック CLI、TUI のオーバーレイ、デスクトップアプリ（Star Map パネル）で動作します。メッセージングプラットフォームでは利用できません。[Learning Journey](../user-guide/features/memory.md#learning-journey-journey) を参照してください。 |

### 設定 {#configuration}

| コマンド | 説明 |
|---------|------|
| `/config` | 現在の設定を表示 |
| `/model [model-name]` | 現在のモデルを表示または変更。サポート: `/model claude-sonnet-4`、`/model provider:model`（プロバイダー切り替え）、`/model custom:model`（カスタムエンドポイント）、`/model custom:name:model`（名前付きカスタムプロバイダー）、`/model custom`（エンドポイントから自動検出）、OpenRouter のアカウントプリセット（`/model @preset/<slug>` または `/model <model>@preset/<slug>` — プリセットはアカウントスコープのため、公開モデル一覧のチェックをスキップします）、ユーザー定義エイリアス（`/model fav`、`/model grok` — [カスタムモデルエイリアス](#custom-model-aliases)を参照）。フラグ: `--global` は変更を config.yaml に永続化、`--session` はセッション限定を強制、`--once` は次のターンのみに適用、`--refresh` はプロバイダーのモデル一覧を再取得、`--provider <name>` はバックエンドを切り替え（`--global` がない限りセッション限定）、`--reasoning <level>` は推論エフォート（`none`、`minimal` … `ultra`）を、選択と同じステップ・同じスコープで設定します。`/model reset`（フラグなし）は、このセッションでの選択を破棄し、会話を失わずに `config.yaml` のモデルに戻します。単純な `/model <name>` は、`model.persist_switch_by_default: true` が設定されていない限りセッション限定です — ただし、`model.default`/`model.provider` がまだ設定されていない場合は例外で、プロファイルが実際のデフォルトを持てるように最初の選択が永続化されます。同じルールがデスクトップのコンポーザーのピッカーにも適用されます。**インタラクティブピッカー:** 引数なしで `/model` を実行すると、プロバイダー→モデルのピッカーが開きます。モデル一覧では**入力してあいまい検索で絞り込み**できます（例: `grok` と入力すると一致するモデルに絞り込まれます）。Backspace でフィルターを削り、Esc でクリアします（またはピッカーを閉じます）。選択は常に 1 つの具体的なモデルに解決されます — フィルターは一覧を絞り込むだけで、推測はしません。モデルの後の 3 番目のステップでは、そのモデルの推論エフォート（または **Keep current effort**）を選べます。カタログ上でそのルートに推論の制御がない場合、このステップはスキップされます。**注:** `/model` は既に設定済みのプロバイダー間でのみ切り替えできます。新しいプロバイダーを追加するには、セッションを終了してターミナルから `hermes model` を実行してください。**コストに関する注意:** 会話の途中でモデルを切り替えるとプロンプトキャッシュがリセットされます — キャッシュキーにはモデルが含まれるため、次のターンでは約 75% 割引のキャッシュ料金ではなく、通常の入力料金で会話全体が再読み込みされます。想定どおりで避けられませんが、長いセッションでは知っておく価値があります。 |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | OpenAI/Codex モデルと、`~/.codex/config.toml` にも定義されている名前付きカスタムプロバイダー向けに、オプションの [Codex app-server ランタイム](../user-guide/features/codex-app-server-runtime)を切り替えます。`auto`（デフォルト）は Hermes の標準の chat completions を使用し、`codex_app_server` は対象のターンを `codex app-server` サブプロセスに渡して、ネイティブシェル、apply_patch、ChatGPT サブスクリプション認証、移行済みの Codex プラグインを利用します。次のセッションから有効になります。 |
| `/personality` | 定義済みのパーソナリティを設定。`/personality none`（または `default` / `neutral`）はオーバーレイをクリアし、基本の動作に戻します。 |
| `/verbose` | ツール進捗表示を循環: off → new → all → verbose。設定で[メッセージング向けに有効化](#notes)できます。 |
| `/focus [on\|off\|status]` | **フォーカスビュー**を切り替え — プロンプトと最終応答だけを表示する、表示専用の出力削減モードです。`/verbose` と組み合わせて動作します: オンにするとツール進捗が `off` に切り替わり、以前のモードが記憶され、`/focus off` でそれが復元されます。各ターンの最後には薄い色の復元用の行（`⋯ 7 tool lines hidden · /focus off to show`）が表示され、ステータスバーには常に `◉ focus` バッジが表示されるため、削減ビューにいることが常にわかります。モデルに送られる内容は何も変わりません — 詳細は隠されるだけで、破棄されることはありません。 |
| `/fast [normal\|fast\|auto\|cold\|status]` | 高速モード — OpenAI Priority Processing / Anthropic Fast Mode。`fast` = すべてのリクエスト、`auto` = 各ターンの最初の `agent.fast_auto_seconds`（デフォルト 60 秒）内のリクエストのみ、`cold` = セッションの最初のターンのみ、同じ時間枠で適用。デフォルトは `normal`（オフ）。[Fast mode](../user-guide/configuration.md#fast-mode) を参照してください。 |
| `/reasoning [level\|show\|hide\|full\|clamp] [--global]` | 推論エフォートと表示を管理。レベルには `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` / `ultra` があります。`show` / `hide`（または `on` / `off`）は推論表示を切り替え、`full` と `clamp` は推論の表示方法を調整します。`--global` はエフォートを設定に永続化します。 |
| `/skin` | 表示スキン/テーマを表示または変更 |
| `/export [profile] [-o out.tar.gz]` | **CLI 専用。** プロファイルを共有可能な `.tar.gz` にパック — スキル、メモリ、ペルソナ、cron、プラグイン、設定、（デスクトップからの場合）テーマとレイアウト。認証情報（`auth.json`、`.env`）は除去されます。デフォルトはアクティブなプロファイルと、現在のディレクトリの `<name>.tar.gz` です。`hermes profile export` と同じアーカイブです。バージョン管理された更新可能な共有には、代わりに[プロファイル配布](../user-guide/profile-distributions.md)を使用してください。 |
| `/import <archive.tar.gz> [--name <name>]` | **CLI 専用。** プロファイルアーカイブを新しいプロファイルとしてインストールします。`--name` を指定しない限り、名前はアーカイブから推測されます。既存のプロファイルの上書きは拒否され、`default` としてインポートすることはできません。名前が空いている場合はシェルラッパーを作成します。[Export and import a profile file](../user-guide/profile-distributions.md#export-and-import-a-profile-file) を参照してください。 |
| `/statusbar`（エイリアス: `/sb`） | コンテキスト/モデルのステータスバーをオン/オフ切り替え |
| `/battery [on\|off\|status]` | ステータスバーの最初の要素として、色分けされたバッテリー表示を切り替え（デフォルトはオフ。バッテリーがない場合は何もしません）。 |
| `/voice [on\|off\|tts\|status]` | CLI ボイスモードと音声再生を切り替え。録音には `voice.record_key`（デフォルト: `Ctrl+B`）を使用します。 |
| `/yolo` | YOLO モードを切り替え — 危険なコマンドの承認プロンプトをすべてスキップ。 |
| `/approvals [manual\|smart\|off]` | 危険なコマンドの承認モード（永続設定）を表示または設定。 |
| `/footer [on\|off\|status]` | 最終応答にゲートウェイのランタイムメタデータフッターを切り替え（モデル、コンテキスト %、cwd を表示）。 |
| `/busy [queue\|steer\|interrupt\|status]` | Hermes が作業中にメッセージを送ったときの動作を制御 — 新しいメッセージをキューに入れる、ターン途中で steer する、即座に割り込む。CLI とメッセージングゲートウェイで動作します。 |
| `/autoprune [on\|off\|later\|status]` | 終了した古いセッションの自動削除を決定（デフォルトはオフ）。`state.db` があるディスクの空き容量が少なくなると、Hermes が 1 回だけ尋ねます。`on`/`off` は `sessions.auto_prune` を保存して質問を終了し、`later` は 1 週間後に再通知します。CLI、TUI、Desktop、メッセージングゲートウェイで動作します（Slack では `/hermes autoprune`）。 |
| `/indicator [kaomoji\|emoji\|unicode\|ascii]` | CLI 専用: TUI のビジーインジケータースタイルを選択。 |
| `/timestamps [on\|off\|status]` | CLI 専用: メッセージと `/history` の `[HH:MM]` タイムスタンプを切り替え。 |
| `/wake [on\|off\|status]` | CLI 専用: 「Hey Hermes」ウェイクワードのリスナーを切り替え。 |

### ツールとスキル {#tools--skills}

| コマンド | 説明 |
|---------|------|
| `/tools [list\|disable\|enable] [name...]` | ツールを管理: 利用可能なツールを一覧表示、または現在のセッションで特定のツールを無効化/有効化。ツールを無効化すると、エージェントのツールセットから削除され、セッションリセットがトリガーされます。 |
| `/toolsets` | 利用可能なツールセットを一覧表示 |
| `/browser [connect\|disconnect\|status]` | ローカルの Chromium 系 CDP 接続を管理。`connect` は実行中の Chrome、Brave、Chromium、Edge インスタンス（デフォルト: `http://127.0.0.1:9222`）にブラウザツールをアタッチします。`disconnect` はデタッチします。`status` は現在の接続を表示します。デバッガが検出されない場合は、サポートされている Chromium 系ブラウザを自動起動します。 |
| `/skills` | オンラインレジストリからスキルを検索、インストール、検査、管理。スキル書き込み承認ゲートのレビュー画面も兼ねます: `/skills pending`、`/skills diff <id>`、`/skills approve <id>`、`/skills reject <id>`、`/skills approval on\|off`。[Gating agent skill writes](../user-guide/features/skills.md#gating-agent-skill-writes-skillswrite_approval) を参照してください。 |
| `/memory [pending\|approve\|reject\|approval]` | 書き込み承認ゲート（`memory.write_approval`）によってステージされた保留中のメモリ書き込みをレビューし、ゲートを切り替えます。[Controlling memory writes](../user-guide/features/memory.md#controlling-memory-writes-write_approval) を参照してください。 |
| `/bundles` | 設定済みのスキルバンドル — 複数のスキルを一度に事前読み込みする `/<name>` スラッシュエイリアス — を一覧表示。`~/.hermes/config.yaml` の `bundles:` で設定します。[Skill Bundles](../user-guide/features/skills.md#skill-bundles) を参照してください。 |
| `/learn <what to learn from>` | 説明したあらゆるもの — ディレクトリ、URL、エージェントに手順を示したばかりのワークフロー、貼り付けたメモ — から再利用可能なスキルを抽出します。自由形式: エージェントは自身のツールでソースを収集し、社内の作成基準に従って `SKILL.md` を作成します。CLI、メッセージングゲートウェイ、TUI、ダッシュボードの Skills ページで動作します。 |
| `/plan [task]` | アクティブなワークスペースの `.hermes/plans/` に Markdown の実装プランを書き込みます — プランのみで、実行はしません。引数が空の場合は、会話からタスクを推測します。（以前はバンドル済みの `plan` スキルでしたが、Telegram/Discord のコマンドメニューの上限に影響されないよう、現在は組み込みです。） |
| `/init [notes]` | リポジトリのスキャンから `AGENTS.md` のプロジェクト指示を生成または更新します（Codex の `/init` の移植）。エージェントは読み取り専用のツールでマニフェスト、レイアウト、ツールチェーンの設定を調べ、簡潔な `AGENTS.md` を書き込みます — 既に存在する場合は、内容を保持したままマージ更新します。オプションのメモで重点を指定できます。CLI、メッセージングゲートウェイ、TUI で動作します。 |
| `/cron` | スケジュールタスクを管理（list、add/create、edit、pause、resume、run、remove） |
| `/suggestions [accept\|dismiss N\|catalog\|clear]`（エイリアス: `/suggest`） | 提案された自動化をレビュー。`/suggestions` で保留中の提案を一覧表示、`/suggestions accept <id>` で提案された自動化を作成、`/suggestions dismiss <id>` で却下、`/suggestions catalog` で厳選されたスターター自動化を追加、`/suggestions clear` で解決済みの提案レコードをクリアします。承認されたジョブは、現在の画面を配信元として保持します。 |
| `/blueprint [name] [slot=value ...]`（エイリアス: `/bp`） | ブループリントテンプレートから自動化をセットアップ。引数なしの `/blueprint` はカタログを一覧表示し、`/blueprint <name>` は次のエージェントターンでスロットを埋めるガイド付きフローを開始し、`/blueprint <name> slot=value ...` はジョブを直接作成します。 |
| `/curator` | バックグラウンドのスキルメンテナンス — `status`、`run`、`pin`、`archive`。[Curator](../user-guide/features/curator.md) を参照してください。 |
| `/kanban <action>` | チャットを離れずにマルチプロファイル・マルチプロジェクトのコラボレーションボードを操作。`hermes kanban` のフルインターフェースが利用可能: `/kanban list`、`/kanban show t_abc`、`/kanban create "title" --assignee X`、`/kanban comment t_abc "text"`、`/kanban unblock t_abc`、`/kanban dispatch` など。マルチボードサポートも含まれます: `/kanban boards list`、`/kanban boards create <slug>`、`/kanban boards switch <slug>`、`/kanban --board <slug> <action>`。[Kanban slash command](../user-guide/features/kanban.md#kanban-slash-command) を参照してください。 |
| `/reload-mcp`（エイリアス: `/reload_mcp`） | config.yaml から MCP サーバーを再読み込みし、ツールの利用可否を再確認（セッション途中で現れた認証情報/デーモン） |
| `/connectors` | TUI 専用。コネクターアカウントの一覧を開きます: アカウントごとに 1 行、`app · name · status`。キー: ↑/↓ で選択、`r` で名前変更（インライン入力。Enter で保存、Esc でキャンセル）、`a` で別のアカウントを追加（アプリを選び、名前を入力し、リンクを開く）、`c` で選択したアカウントを再接続（修復が必要な場合 — 失敗、期限切れ、取り消し、非アクティブ — のみ表示）、`x` で削除（確認あり）、Tab で廃止済みアカウントの表示/非表示、Esc または `q` で閉じる。[Multiple connector accounts](../user-guide/features/tool-gateway.md#multiple-connector-accounts) を参照してください。 |
| `/reload-skills`（エイリアス: `/reload_skills`） | 新しくインストールまたは削除されたスキルがないか `~/.hermes/skills/` を再スキャン |
| `/reload` | `.env` 変数を実行中のセッションに再読み込み（再起動なしで新しい API キーを取得） |
| `/plugins [update\|auto-update [on\|off]]` | インストール済みプラグインとそのステータスを一覧表示。`update` は更新のあるすべてのプラグインを更新し、`auto-update` は無人でのプラグイン更新を表示または設定します |
| `/pet [list\|<slug>]` | [petdex](/docs/user-guide/features/pets) のマスコットを切り替えまたは迎え入れます。`/pet` はペインを切り替え、`/pet list` はインストール済みのペットを表示し、`/pet <slug>` は特定のペットを迎え入れます。 |
| `/hatch <description>`（エイリアス: `/generate-pet`） | 設定済みの画像バックエンド（OpenRouter / Nous Portal）を使い、テキストの説明からまったく新しい petdex のペットを生成します。[Pets](/docs/user-guide/features/pets) を参照してください。 |

### 情報 {#info}

| コマンド | 説明 |
|---------|------|
| `/help` | 利用可能なコマンドをカテゴリ別に表示。デフォルトではコアコマンドが表示され、スキルコマンドは 1 行の件数に折りたたまれます。`/help skills` はすべてのスキルコマンドを一覧表示し、`/help <text>` はコマンド（と一致するスキル）を部分文字列で絞り込みます。 |
| `/palette` | あいまい検索のコマンドパレットを開く（**Ctrl+P** でも可） — 入力してすべてのコマンド + スキルを絞り込み、↑/↓ で移動、Enter で選択したコマンドをコンポーザーに挿入（自動実行はしません）、Esc でキャンセル。一致はコマンド名を優先して順位付けされるため、短いクエリでも正確です。 |
| `/version` | Hermes Agent のバージョン、ビルド、環境情報を表示。 |
| `/whoami` | スラッシュコマンドのアクセスレベル（admin / user）を表示。 |
| `/usage` | トークン使用量、コスト内訳、セッション継続時間、そして — アクティブなプロバイダーから利用可能な場合 — プロバイダーの API からライブで取得した残りクォータ / クレジット / プラン使用量を含む **Account limits** セクションを表示。 |
| `/topup` | Nous の残高を表示し、ポータルで請求を管理します（以前の `/credits` と `/billing` コマンドを置き換えます）。 |
| `/subscription`（エイリアス: `/upgrade`） | **CLI 専用。** Nous のプランを表示し、ブラウザで変更します。 |
| `/login` | Nous アカウントでサインイン。ターン外で実行されます: 同意リンクとコードがセッションに届き、ブラウザで承認するとサインインが完了します。 |
| `/insights` | 使用状況のインサイトと分析を表示（過去 30 日間） |
| `/update` | Hermes Agent を最新バージョンに更新。 |
| `/platforms`（エイリアス: `/gateway`） | ゲートウェイ/メッセージングプラットフォームのステータスを表示（CLI 専用の概要ビュー）。 |
| `/paste` | クリップボードの画像を添付 |
| `/copy [number]` | 最後のアシスタント応答をクリップボードにコピー（数値を指定すると後ろから N 番目）。CLI 専用。 |
| `/image <path>` | 次のプロンプト用にローカル画像ファイルを添付。 |
| `/debug [upload\|nous upload\|local]` | デバッグレポート。引数なしの `/debug` はプライバシーに関する通知を表示し、何もアップロードしません。`/debug upload` は公開ペーストにアップロードして共有可能なリンクを返し、`/debug nous upload` は Nous 内部のストレージにアップロードします（クラシック CLI のみ: メッセージング、TUI、デスクトップアプリでは拒否され、`hermes debug share --nous` または Desktop の Send diagnostics が案内されます）。`/debug local` はアップロードせずにレポートを表示します。メッセージングでも利用可能。 |
| `/update` | Hermes Agent を最新バージョンに更新。 |
| `/profile` | アクティブなプロファイル名とホームディレクトリを表示 |

### 終了 {#exit}

| コマンド | 説明 |
|---------|------|
| `/quit` | CLI を終了（`/exit` も可）。 |

### 動的 CLI スラッシュコマンド {#dynamic-cli-slash-commands}

| コマンド | 説明 |
|---------|------|
| `/<skill-name>` | インストール済みのスキルをオンデマンドコマンドとして読み込む。例: `/youtube-content`、`/github-pr-workflow`、`/excalidraw`。 |
| `/skills ...` | レジストリと公式のオプションスキルカタログからスキルを検索、ブラウズ、検査、インストール、監査、公開、設定。 |

### クイックコマンド {#quick-commands}

ユーザー定義のクイックコマンドは、短いスラッシュコマンドをシェルコマンドまたは別のスラッシュコマンドにマッピングします。`~/.hermes/config.yaml` で設定します:

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  deploy:
    type: exec
    command: scripts/deploy.sh
  inbox:
    type: alias
    target: /gmail unread
```

そして、CLI またはメッセージングプラットフォームで `/status`、`/deploy`、`/inbox` と入力します。クイックコマンドはディスパッチ時に解決され、すべての組み込み自動補完/ヘルプテーブルに表示されるとは限りません。

文字列のみのプロンプトショートカットは、クイックコマンドとしてサポートされていません。長くて再利用可能なプロンプトはスキルに入れるか、`type: alias` を使って既存のスラッシュコマンドを指してください。

### カスタムモデルエイリアス {#custom-model-aliases}

よく使うモデルに自分用の短い名前を定義し、実行中のセッションでは `/model <alias>`、起動時には `hermes chat --model <alias>`、または任意のメッセージングプラットフォームで呼び出せます。エイリアスはこれらの経路で同一に動作し、セッション限定（デフォルト）と `--global` の切り替えのどちらでも機能します。

2 つの設定フォーマットがサポートされています:

**フルフォーム** — 正確なモデル、プロバイダー、オプションでベース URL を固定します。これを `~/.hermes/config.yaml` に入れます:

```yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
  ollama-qwen:
    model: qwen3-coder:30b
    provider: custom
    base_url: http://localhost:11434/v1
  theta:
    model: theta-1
    provider: custom
    base_url: https://theta.example.com/v1
    key_env: THETA_API_KEY        # or: api_key: "${THETA_API_KEY}"
```

独自の `base_url` を持つエイリアスは、そのエンドポイントの認証情報を
`api_key`（リテラル、または `"${VAR}"` 参照）または `key_env`（環境変数名）で
持たせることができます。両方が設定されている場合は `api_key` が優先されます。どちらも設定されていない場合、キーはエイリアスの**ホスト**から
解決され、切り替え前にアクティブだったプロバイダーから引き継がれることはありません。

**ショートフォーム** — `provider/model` を 1 つの文字列で。YAML を編集せずにシェルから設定:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

そしてチャット内で:

```
/model fav            # session-only
/model grok --global  # also persists current-model change to config.yaml
```

ユーザーエイリアスは組み込みの短縮名より優先されるため、エイリアスを `sonnet`、`kimi`、`opus` などと名付けると組み込みをシャドウします。エイリアス名は大文字小文字を区別しません。

### エイリアスの解決 {#alias-resolution}

コマンドはプレフィックスマッチングをサポートします: `/h` と入力すると `/help` に、`/mod` は `/model` に解決されます。プレフィックスが曖昧（複数のコマンドにマッチ）な場合、レジストリ順での最初のマッチが優先されます。完全なコマンド名と登録されたエイリアスは、常にプレフィックスマッチより優先されます。

## メッセージングスラッシュコマンド {#messaging-slash-commands}

> **Slack スレッドのコマンド（`!` プレフィックス）:**
> Slack 自体がメッセージスレッド内のネイティブスラッシュコマンドをブロックし（"/queue is not supported in threads. Sorry!"）、Hermes に届けることはありません。Slack スレッド内では、代わりに `!` プレフィックスを使ってください — `!stop`、`!new`、`!status` — ゲートウェイはスラッシュ形式とまったく同じようにディスパッチします。`@Hermes !stop` と `@Hermes /stop` もスレッド内で機能します。既知のコマンドリストと照合されるのは最初のトークンだけなので、`!nice work` のようなメッセージはそのままエージェントに渡されます。詳細は [Using commands inside threads](../user-guide/messaging/slack.md#using-commands-inside-threads-the-cmd-prefix) を参照してください。

メッセージングゲートウェイは、Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant、Teams のチャット内で次の組み込みコマンドをサポートします:

| コマンド | 説明 |
|---------|------|
| `/start` | プラットフォームプロトコルのコマンド。多くのチャットプラットフォーム（Telegram、Discord など）は、ユーザーが初めてボットとの会話を開いたときに自動的に `/start` を送信します。Hermes はこの ping を静かに受け取ります — エージェントの応答もセッションの消費もないため、最初の接触のハンドシェイクでターンが無駄になりません。ゲートウェイに到達できることを確認するために、明示的に送信することもできます。 |
| `/new [name]`（エイリアス: `/reset`） | 新しいセッションを開始（新しいセッション ID + 履歴）。オプションの `[name]` は最初のセッションタイトルを設定します。確認モーダルをスキップするには `now`、`--yes`、`-y` を付けます — 例: `/reset now`、`/new --yes my-experiment`。 |
| `/status` | セッション情報を表示し、続いてローカルの **Session recap** ブロック（最近のターン数、よく使われたツール、触れたファイル、最新のプロンプト + 応答）を表示。 |
| `/stop` | 実行中のすべてのバックグラウンドプロセスを終了し、実行中のエージェントを割り込む。 |
| `/discard` | 再起動で Hermes が失ったターンを破棄し、このチャットのキュー内のメッセージが再び実行されるようにします。失われたターンは再実行されません（再送するかどうかはあなたにしかわかりません）。中断されたものとして記録されます。Hermes が失われたターンで会話を一時停止したときに、これを提案します。Slack では `/hermes discard` を使います。 |
| `/pause [reason \| off]` | グローバルな緊急停止で、[`hermes pause`](./cli-commands.md) のチャット内版です: `/pause off` までの間、新しい cron、kanban、ゲートウェイの作業を保留にします。実行中の作業は通常どおり完了し、一時停止中もスラッシュコマンドは動作し続けるため、メッセージングのみのオペレーターでもいつでも解除できます。 |
| `/model [provider:model]` | モデルを表示または変更。プロバイダー切り替え（`/model zai:glm-5`）、カスタムエンドポイント（`/model custom:model`）、名前付きカスタムプロバイダー（`/model custom:local:qwen`）、自動検出（`/model custom`）、OpenRouter のアカウントプリセット（`/model @preset/<slug>` — アカウントスコープのため、公開モデル一覧のチェックをスキップ）、ユーザー定義エイリアス（`/model fav`、`/model grok` — [カスタムモデルエイリアス](#custom-model-aliases)を参照）をサポート。`/model reset` は、このチャットのセッションのオーバーライド（メモリ、永続化されたコピー、保留中の `--once` の復元）をクリアし、会話は保持したまま、`channel_overrides` / config.yaml が再びモデルを決めるようにします。変更を config.yaml に永続化するには `--global` を使います。`--global` での選択（入力またはピッカー）に成功すると、このチャットのセッション限定のオーバーライドも破棄されるため、ゲートウェイの再起動後は config.yaml だけがモデルを決めます（`channel_overrides` のモデルを持つチャットはオーバーライドを維持します。そうしないとチャンネルの設定が config.yaml より優先されてしまうためです。CLI/TUI は、再開時にそのチャットが使っていたモデルを復元できるよう、意図的にセッションごとの固定を維持します）。**注:** `/model` は既に設定済みのプロバイダー間でのみ切り替えできます。新しいプロバイダーを追加したり API キーをセットアップしたりするには、（チャットセッションの外で）ターミナルから `hermes model` を使用してください。**コストに関する注意:** セッションの途中でモデルを切り替えるとプロンプトキャッシュがリセットされるため（キャッシュキーにはモデルが含まれます）、次のメッセージでは会話全体が通常の入力料金で再読み込みされます。 |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | オプションの [Codex app-server ランタイム](../user-guide/features/codex-app-server-runtime)を切り替え。config.yaml の `model.openai_runtime` に永続化し、キャッシュされたエージェントを破棄するため、次のメッセージで新しいランタイムが使われます。次のセッションから有効になります。 |
| `/personality [name]` | セッションにパーソナリティオーバーレイを設定。`/personality none`（または `default` / `neutral`）でクリアします。 |
| `/fast [normal\|fast\|auto\|cold\|status]` | 高速モード — OpenAI Priority Processing / Anthropic Fast Mode。`auto`/`cold` は、ターンごと / セッションごとに期間限定の高速ウィンドウを開きます。 |
| `/retry` | 最後のメッセージを再試行。 |
| `/undo` | 最後のやり取りを削除。 |
| `/sethome`（エイリアス: `/set-home`） | 現在のチャットを配信用のプラットフォームホームチャンネルとしてマーク。 |
| `/compress [here [N] \| focus topic]` | 会話コンテキストを手動で圧縮。`/compress here [N]` は、最新の N 件のやり取り（デフォルト 2）をそのまま残し、残りを要約します。フォーカストピックを指定すると、完全な要約が保持する内容を絞り込めます。 |
| `/topic [off\|help\|session-id]` | **Telegram DM 専用。** ユーザー管理のマルチセッショントピックモードを管理。`/topic` は有効化またはステータス表示、`/topic off` は無効化してバインディングをクリア、`/topic help` は使い方を表示、トピック内の `/topic <session-id>` は以前のセッションを復元します。[Multi-session DM mode](../user-guide/messaging/telegram.md#multi-session-dm-mode-topic) を参照してください。 |
| `/save <json\|md\|html> [filename] [redact]` | 現在のセッションを JSON、Markdown、または HTML としてエクスポートし、ファイルをドキュメントとしてチャットに送信します。引数なしの `/save` は使い方を表示し、`redact` は先にシークレットを除去します。 |
| `/title [name]` | セッションタイトルを設定または表示。 |
| `/resume [name]` | 以前に名前を付けたセッションを再開。 |
| `/sessions [all] [search <query>]` | このチャットの以前のセッションを一覧表示。アクティブなセッションには `(current)` マーカーが付きます。`/sessions search <query>` はタイトル/ID の一致で絞り込みます（最近アクティブなものから順に）。`/sessions all` はすべての発信元にわたって一覧表示します（管理者のみ — 管理者以外には通知が表示され、チャットスコープの一覧が返されます）。 |
| `/usage` | トークン使用量、推定コスト内訳（入力/出力）、コンテキストウィンドウの状態、セッション継続時間、そして — アクティブなプロバイダーから利用可能な場合 — プロバイダーの API からライブで取得した残りクォータ / クレジットを含む **Account limits** セクションを表示。 |
| `/topup` | Nous の残高を表示し、ポータルで請求を管理します。 |
| `/login` | Nous アカウントでサインイン。**ペアリング済みのダイレクトメッセージのみ** — グループ、チャンネル、ブロードキャスト型のプラットフォームでは Hermes は拒否します。Slack では `/hermes login` を使います。 |
| `/whoami` | スラッシュコマンドのアクセスレベル（admin / user）を表示。 |
| `/insights [days]` | 使用状況の分析を表示。 |
| `/reasoning [level\|show\|hide\|full\|clamp] [--global]` | 推論エフォートを変更（`max` / `ultra` までのレベル）、または推論表示を切り替え（`full` / `clamp` を含む）。`--global` は設定に永続化します。 |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | チャットでの音声応答を制御。`join`/`channel`/`leave` は Discord ボイスチャンネルモードを管理。 |
| `/rollback [number]` | ファイルシステムのチェックポイントを一覧表示または復元。 |
| `/diff [staged\|all\|session] [--stat]` | 作業ディレクトリの git の変更を表示（コードフェンスで囲まれ、プラットフォームのメッセージ上限に合わせて切り詰められます）。`session` は Hermes が変更したすべての累積差分を表示し、`--stat` は概要のみを表示します。 |
| `/bg <prompt>` | 別のバックグラウンドセッションでプロンプトを実行。結果はタスク完了時に同じチャットに配信されます。[Messaging Background Sessions](../user-guide/messaging/index.md#background-sessions) を参照してください。 |
| `/btw <question>` | 現在の会話を中断せずに、会話についてちょっとした質問をする。トランスクリプトのスナップショットから回答され、準備ができると回答がチャットに送信されます。 |
| `/queue <prompt>`（エイリアス: `/q`） | 現在のターンを中断せずに、次のターン用にプロンプトをキューに入れる。 |
| `/steer <prompt>` | 割り込みなしで次のツールコールの後にメッセージを注入 — モデルは新しいターンとしてではなく、次のイテレーションでそれを取り込みます。 |
| `/goal <text>` | Hermes がターンをまたいで取り組む常設のゴールを設定 — Ralph ループに対する私たちのアプローチです。ジャッジモデルが各ターンの後にチェックし、完了していなければ、完了するか、あなたが pause/clear するか、ターン予算（デフォルト 20）に達するまで Hermes が自動で続行します。サブコマンド: `/goal status`、`/goal pause`、`/goal resume`、`/goal clear`。status/pause/clear はエージェント実行中でも安全に実行できます。新しいゴールの設定にはまず `/stop` が必要です。[Persistent Goals](../user-guide/features/goals.md) を参照してください。 |
| `/subgoal <text>` | ループ実行中に、アクティブな `/goal` に基準を追加（`/subgoal`、`/subgoal remove <N>`、`/subgoal clear`）。 |
| `/heartbeat every <interval> <prompt>`（エイリアス: `/hb`） | アイドル時にこのセッションに再投入される定期プロンプトを設定。サブコマンド: `status`、`pause`、`resume`、`clear`。Slack では `/hermes heartbeat …` を使います。 |
| `/loop [interval] <prompt> [--times N] [--until <condition>]`（エイリアス: `/proactive`） | このセッションで、プロンプトを定期的な間隔で再実行。サブコマンド: `status`、`pause`、`resume`、`stop`。[Loops](/docs/user-guide/features/loops) を参照してください。 |
| `/refine [focus]` | メモリ/スキルの自己改善レビューを今すぐ実行（オプションでフォーカスの指示付き）。アイドル状態のチャットでも動作します（先にメッセージを送る必要はありません）: 保存された会話をレビューし、結果を同じチャットに投稿します。Slack では `/hermes refine …` を使います。 |
| `/review [instructions]` | 直前に話し合った作業（PR、コード、ドキュメント）に対して独立したレビュアーサブエージェントを起動します。完了するとレビューがこのチャットに戻ります。Slack では `/hermes review …` を使います。 |
| `/moa <prompt>` | 1 つのプロンプトをデフォルトの [Mixture of Agents](/docs/user-guide/features/mixture-of-agents) プリセットで実行し、その後セッションのモデルに戻します。 |
| `/branch [--here] [name]`（エイリアス: `/fork`） | 現在のセッションをブランチ。スレッド対応のプラットフォーム（Discord、Telegram、Slack、Matrix）では、ブランチを新しい兄弟スレッドで開き、このチャットは元のセッションに留まります。`--here` はこのチャットをブランチに切り替えます。 |
| `/agents`（エイリアス: `/tasks`） | アクティブなエージェントと実行中のタスクを表示。 |
| `/sessions` | 以前のセッションをブラウズして再開。 |
| `/context [all]`（エイリアス: `/ctx`） | コンテキストウィンドウの使用量ゲージとカテゴリ別の内訳（メッセージング向けのテキスト形式）。`/context all` はスキルごと / ツールセットごとのコストの詳細を追加します。 |
| `/egress [status]` | Docker エグレスプロキシのステータスを表示。 |
| `/init [notes]` | リポジトリのスキャンから `AGENTS.md` を生成または更新。 |
| `/learn <what to learn from>` | 説明したあらゆるものから再利用可能なスキルを抽出。 |
| `/plan [task]` | `.hermes/plans/` に Markdown の実装プランを書き込みます。実行はしません。 |
| `/bundles` | 設定済みのスキルバンドル（複数のスキルを事前読み込みする `/<name>` エイリアス）を一覧表示。 |
| `/reload-skills`（エイリアス: `/reload_skills`） | 新しくインストールまたは削除されたスキルがないか `~/.hermes/skills/` を再スキャン。 |
| `/footer [on\|off\|status]` | 最終応答にランタイムメタデータフッターを切り替え（モデル、コンテキスト %、cwd を表示）。 |
| `/curator [status\|run\|pin\|archive]` | バックグラウンドのスキルメンテナンスコントロール。 |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` | 提案された自動化をチャット内で直接レビュー。`/suggestions` は保留中の提案を一覧表示し、`catalog` は厳選されたスターター自動化を追加し、`clear` は解決済みの提案レコードを整理します。承認された提案は、このチャット/スレッドをジョブの配信元として保持します。 |
| `/blueprint [name] [slot=value ...]` | cron ブループリントをブラウズ、ガイド付きのスロット入力の会話を開始、またはブループリントジョブを直接作成します。直接作成されたジョブは現在のチャット/スレッドに配信されます。 |
| `/memory [pending\|approve\|reject\|approval]` | 書き込み承認ゲート（`memory.write_approval`）によってステージされた保留中のメモリ書き込みをレビューし — チャット内で直接承認または拒否 — `/memory approval on\|off` でゲートを切り替えます。[Controlling memory writes](../user-guide/features/memory.md#controlling-memory-writes-write_approval) を参照してください。 |
| `/skills [pending\|approve\|reject\|diff\|approval]` | 書き込み承認ゲート（`skills.write_approval`）によってステージされた保留中の**スキル**書き込みをレビュー。ステージされた書き込みごとに 1 行の要旨を表示します。`/skills diff <id>` はチャット向けに切り詰められます — 完全な差分は CLI か `~/.hermes/pending/skills/<id>.json` で確認してください。ゲートがオンの場合（またはステージされた書き込みが残っている場合）にのみ表示され、検索/インストールは引き続き CLI 専用です。 |
| `/kanban <action>` | チャットからマルチプロファイル・マルチプロジェクトのコラボレーションボードを操作 — CLI と同一の引数インターフェース。実行中エージェントのガードをバイパスするため、`/kanban unblock t_abc`、`/kanban comment t_abc "…"`、`/kanban list --mine`、`/kanban boards switch <slug>` などがターンの途中で機能します。`/kanban create …` は、発信元のチャットを新しいタスクのターミナルイベントに自動サブスクライブします。[Kanban slash command](../user-guide/features/kanban.md#kanban-slash-command) を参照してください。 |
| `/platform <list\|pause\|resume> [name]` | 実行中のゲートウェイプラットフォームをチャットから直接操作。`/platform list` はすべてのアダプターとその状態（running、paused-by-breaker、manually-paused）を表示し、`/platform pause <name>` はアダプターをアンロードせずに新しいメッセージのディスパッチを停止し、`/platform resume <name>` はそれを再度有効化し、上流が正常になった時点で作動したサーキットブレーカーをクリアします。 |
| `/reload-mcp`（エイリアス: `/reload_mcp`） | 設定から MCP サーバーを再読み込みし、ツールの利用可否を再確認。 |
| `/verbose` | ツール進捗表示を循環。**メッセージングではデフォルトでオフ** — `config.yaml` で `display.tool_progress_command: true` を設定して有効化します。 |
| `/yolo` | YOLO モードを切り替え — 危険なコマンドの承認プロンプトをすべてスキップ。 |
| `/commands [page]` | すべてのコマンドとスキルをブラウズ（ページ分け）。 |
| `/approve [session\|always]` | 保留中の危険なコマンドを承認して実行。`session` はこのセッションのみ承認、`always` は永続的な許可リストに追加。 |
| `/deny` | 保留中の危険なコマンドを拒否。 |
| `/update` | Hermes Agent を最新バージョンに更新。 |
| `/restart` | アクティブな実行をドレインした後、ゲートウェイを正常に再起動。ゲートウェイがオンラインに戻ると、リクエスト元のチャット/スレッドに確認を送信します。 |
| `/debug [upload]` | 引数なしの `/debug` はプライバシーに関する通知を表示し、何もアップロードしません。`/debug upload` は許可リストに載った概要（バージョン、設定キー名、エラー件数。ログやメッセージのテキストは含まない）を公開ペーストにアップロードし、共有可能なリンクを返します。完全なログは `hermes debug share --nous` 経由でプライベートストレージにのみ送られます。 |
| `/help` | メッセージングヘルプを表示。 |
| `/<skill-name>` | インストール済みのスキルを名前で呼び出す。 |

## 注記 {#notes}

- `/skin`、`/snapshot`、`/export`、`/import`、`/reload`、`/tools`、`/toolsets`、`/browser`、`/config`、`/cron`、`/platforms`、`/paste`、`/image`、`/statusbar`、`/battery`、`/focus`、`/plugins`、`/indicator`、`/wake`、`/journey`、`/redraw`、`/clear`、`/history`、`/copy`、`/handoff`、`/prompt`、`/pet`、`/hatch`、`/timestamps`、`/subscription`、`/quit` は **CLI 専用**のコマンドです。
- `/skills` は**検索/ブラウズ/インストールについては CLI 専用**です。書き込み承認のレビュー用サブコマンド（`pending`、`approve`、`reject`、`diff`、`approval`）は、`skills.write_approval` がオンのときメッセージングプラットフォームでも動作します。`/memory` は**両方**のインターフェースで動作します。
- `/verbose` は**デフォルトでは CLI 専用**ですが、`config.yaml` で `display.tool_progress_command: true` を設定することでメッセージングプラットフォーム向けに有効化できます。有効化すると、`display.tool_progress` モードを循環させ、設定に保存します。
- `/focus` と `/verbose` は 1 つの抑制経路（`display.tool_progress`）を共有するため、互いに矛盾することはありません: `/focus on` はツール進捗を `off` に固定し、あなたのモードを `display.focus_saved_tool_progress` に退避します。`/focus off` でそれを復元します。フォーカスがオンの間に `/verbose` を循環させると、モードが戻ってフォーカスバッジがクリアされます。フォーカスビューは表示専用です — 会話履歴、システムプロンプト、モデルに送られるものを一切変更しないため、プロンプトキャッシュへの影響はまったくありません。
- `/sethome`、`/restart`、`/approve`、`/deny`、`/topic`、`/platform`、`/pause`、`/discard`、`/commands` は**メッセージング専用**のコマンドです。
- `/status`、`/egress`、`/version`、`/whoami`、`/bg`、`/btw`、`/queue`、`/steer`、`/voice`、`/reload-mcp`、`/reload-skills`、`/rollback`、`/diff`、`/debug`、`/fast`、`/approvals`、`/busy`、`/footer`、`/curator`、`/kanban`、`/topup`、`/login`、`/suggestions`、`/blueprint`、`/learn`、`/init`、`/sessions`、`/loop`、`/yolo` は CLI とメッセージングゲートウェイの**両方**で動作します。
- `/voice join`、`/voice channel`、`/voice leave` は Discord でのみ意味を持ちます。
- `/connectors` は **TUI 専用**です。
- TUI では、`/sessions` は現在の TUI プロセス内のライブセッションを表示します。保存済みまたは閉じたトランスクリプトには、`/resume [name]` または `hermes --tui --resume <id-or-title>` を使用してください。

## 破壊的なコマンドの確認プロンプト {#confirmation-prompts-for-destructive-commands}

CLI は、保存されていないセッション状態を破棄するスラッシュコマンドを実行する前に確認を求めます。現在の破壊的なコマンドは次のとおりです:

| コマンド | 破棄されるもの |
|---------|------------------|
| `/clear` | 画面をクリアして新しいセッションを開始します — 現在のセッション ID とメモリ上の履歴は失われます。 |
| `/new` / `/reset` | 新しいセッションを開始します（新しいセッション ID + 空の履歴）。 |
| `/undo` | 最後のユーザー/アシスタントのやり取りを履歴から削除します。 |
| `/exit --delete` / `/quit --delete` | 終了し、**さらに**現在のセッションの SQLite 履歴とディスク上のトランスクリプトを完全に削除します。 |

これらのそれぞれについて、CLI は 3 択のモーダルを開きます: **Approve Once**（今回は実行する）、**Always Approve**（実行し、`approvals.destructive_slash_confirm: false` を永続化して、以降の破壊的なコマンドを確認なしで実行する）、**Cancel**。

**インラインでのスキップ:** 1 回の呼び出しだけモーダルを回避するには、`now`、`--yes`、`-y` を付けます — 例: `/reset now`、`/new --yes my-session`、`/clear -y`、`/undo -y`。お使いのターミナルでモーダルが正しく表示されない場合（ネイティブの Windows PowerShell については [issue #30768](https://github.com/NousResearch/hermes-agent/issues/30768) を参照）や、CLI に対してスクリプトを実行する場合に便利です。

確認プロンプトをグローバルに無効化するには、`~/.hermes/config.yaml` で `approvals.destructive_slash_confirm: false` を設定します。再度有効化するには `true` に戻します。背景については [Security — Destructive slash command confirmation](../user-guide/security.md#dangerous-command-approval) を参照してください。
