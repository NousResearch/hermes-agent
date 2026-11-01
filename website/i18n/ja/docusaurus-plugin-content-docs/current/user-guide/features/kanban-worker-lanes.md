# カンバンワーカーレーン

**ワーカーレーン** は、カンバンディスパッチャーがタスクをルーティングできるプロセスのクラスです。各レーンは、アイデンティティ（担当者文字列）、スポーン（生成）メカニズム、そしてスポーン後にタスクに対して何をしなければならないかの契約を持ちます。

このページはその契約です。次の 2 つの読者を対象としています:

- ボードに接続するレーンを選ぶ **オペレーター**（どのプロファイルを作成し、どの担当者を使用するか）。
- 新しいレーンの形状を追加したい **プラグイン/連携の作者**（Codex / Claude Code / OpenCode をラップする CLI ワーカー、コンテナ化されたレビューワーカー、API 経由でタスクをプルする非 Hermes サービスなど）。

ワーカーコード自体 — レーンの *内部* で実行されるエージェント — を書いている場合は、[`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) スキルがより深い手続き的な詳細を提供します。

## 階層

```text
Hermes Kanban  =  正規のタスクライフサイクル + 監査証跡
Worker lane    =  割り当てられた 1 枚のカードの実装実行者
Reviewer       =  「完了」をゲートする人間または人間プロキシ
GitHub PR      =  アップストリーム可能な成果物（オプション、コードレーン用）
```

Hermes Kanban はライフサイクルの真実を所有します — `ready` → `running` → `blocked` / `done` / `archived`。ワーカーレーンは作業を実行しますが、その真実を所有することはありません。ワーカーが行うすべては、`kanban_*` ツール（または、非 Hermes の外部ワーカーの場合は API）を介してカンバンカーネルにフローバックします。レビュアーは「コード変更が書かれた」状態から「タスク完了」への遷移をゲートします。

## レーンが提供するもの

カンバンワーカーレーンであるためには、連携は 3 つのものを提供しなければなりません:

### 1. 担当者文字列

ディスパッチャーは `task.assignee` を、Hermes プロファイル名（デフォルトのレーン形状）または登録済みのスポーン不可な識別子（プラグインレーン形状 — 後述の [外部 CLI ワーカーレーンの追加](#adding-an-external-cli-worker-lane) を参照）のいずれかと照合します。担当者が解決できないタスクは、ボードオペレーターが修正できるように `skipped_nonspawnable` イベントとともに `ready` のまま残されます。これらは黙って破棄されたり、任意のフォールバックによって実行されたりすることはありません。

### 2. スポーンメカニズム

Hermes プロファイルレーンの場合、ディスパッチャーの `_default_spawn` は、タスクのピン留めされたワークスペース内で `hermes -p <assignee> chat -q <prompt>`（または `hermes` シムが `$PATH` にない場合は同等のモジュール形式）を実行し、以下の環境変数を設定します:

| 変数 | 保持する内容 |
|---|---|
| `HERMES_KANBAN_TASK` | ワーカーが操作しているタスク ID |
| `HERMES_KANBAN_DB` | ボードごとの SQLite ファイルへの絶対パス |
| `HERMES_KANBAN_BOARD` | ボードのスラッグ |
| `HERMES_KANBAN_WORKSPACES_ROOT` | ボードのワークスペースツリーのルート |
| `HERMES_KANBAN_WORKSPACE` | *この* タスクのワークスペースへの絶対パス |
| `HERMES_KANBAN_RUN_ID` | 現在の実行の ID（ライフサイクルゲート用） |
| `HERMES_KANBAN_CLAIM_LOCK` | クレームロック文字列（`<host>:<pid>:<uuid>`） |
| `HERMES_PROFILE` | ワーカー自身のプロファイル名（`kanban_comment` の作成者帰属用） |
| `HERMES_TENANT` | タスクにテナントがある場合、そのテナント名前空間 |

非 Hermes レーン（プラグイン経由で登録）の場合、プラグインは独自の `spawn_fn` 呼び出し可能オブジェクトを提供します。これは `task`、`workspace`、`board` を受け取り、クラッシュ検出用のオプションの pid を返します。

### 3. ライフサイクルターミネーター

すべてのクレームは、次のいずれか 1 つだけで終了しなければなりません:

- `kanban_complete(summary=..., metadata=...)` — タスクが成功し、ステータスが `done` に切り替わります。
- `kanban_block(reason=...)` — タスクが人間の入力を待ち、ステータスが `blocked` に切り替わります。`kanban_unblock` が実行されるとディスパッチャーが再スポーンします。
- ワーカープロセスがツール呼び出しなしで終了します。カーネルがそれを回収し、`crashed`（PID が死んだ）、`gave_up`（連続失敗ブレーカーが作動した）、または `timed_out`（max_runtime を超過した）を発行します。これは失敗パスです。健全なワーカーはここで終わりません。

カンバンカーネルは、これらのいずれか 1 つだけが各実行を終了させることを強制します。どちらも呼び出さずに正常終了するワーカーはクラッシュとして扱われます。

## 出力とレビュー必須の慣例

ほとんどのコード変更タスクでは、ワーカーが終了した瞬間に作業が本当に *完了* するわけではありません — 人間のレビュアーが必要です。カンバンカーネルはこの区別を強制しません（「コード変更タスク」は曖昧であり、すべてのコードワーカーに complete の代わりに block を強制すると、レビューが不要なフローが壊れてしまうため）。これはその上に重ねられた慣例です:

- **complete の代わりに block** し、`reason` に `review-required: ` というプレフィックスを付けて、ダッシュボード / `hermes kanban show` がその行をレビュー待ちとして表示するようにします。
- **構造化されたメタデータを先に `kanban_comment` に投入** します。`kanban_block` は人間が読める `reason` のみを保持するためです。コメントは耐久性のある注釈チャネルです — 監査関連のすべてのフィールド（changed_files、tests_run、diff_path または PR URL、決定事項）はそこに属します。
- **レビュアーは承認してブロック解除** し（コメントスレッドを引き継ぎ用にしてワーカーを再スポーンします）、または別のコメントで変更を要求します（次のワーカー実行が `kanban_show` のコンテキストの一部として確認します）。

[`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) スキルには、`kanban_complete`（真に終端的なタスク — タイポ修正、ドキュメント変更、リサーチの書き起こし）と `review-required` ブロックパターンの両方の実例があります。

## ログと監査証跡

ディスパッチャーは、タスクごとのワーカーの stdout/stderr を `<board-root>/logs/<task_id>.log` に書き込みます。ログはカンバンメタデータから監査可能です:

- `task_runs` の行には、`log_path`、終了コード（利用可能な場合）、サマリー、メタデータが含まれます。
- `task_events` の行には、すべての状態遷移（`promoted`、`claimed`、`heartbeat`、`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`、`reclaimed`、`claim_extended`）が含まれます。
- `kanban_show` は両方を返すため、タスクを読むレビュアー（またはフォローアップワーカー）は、ダッシュボードへのアクセスを必要とせずに完全な履歴を取得できます。

ダッシュボードは、サマリー、メタデータブロック、終了ステータスバッジとともに実行履歴をレンダリングします。CLI ユーザーは、ライブで追跡するには `hermes kanban tail <task_id>` を、過去の試行リストには `hermes kanban runs <task_id>` を実行できます。

## 既存のレーン形状

### Hermes プロファイルレーン（デフォルト）

現在すべてのカンバンワーカーが取る形状です: 担当者はプロファイル名で、ディスパッチャーが `hermes -p <profile>` をスポーンし、ワーカーが [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) スキルと `KANBAN_GUIDANCE` システムプロンプトブロックを自動ロードし、`kanban_*` ツールを使用して実行を終了します。プロファイルを定義する以外のセットアップは不要です。

フリート用のプロファイルを作成する際は、オーケストレーターにルーティングさせたい *役割* に一致する名前を選んでください。オーケストレーター（存在する場合）は、`hermes profile list` を介してあなたのプロファイル名を発見します — システムが想定する固定のロスターはありません（契約のオーケストレーター側については [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) スキルを参照してください）。

### オーケストレータープロファイルレーン

プロファイルレーンの特殊化です: オーケストレーターは、ツールセットに `kanban` を含むが、実装用の `terminal` / `file` / `code` / `web` を除外した Hermes プロファイルです。その仕事は、`kanban_create` + `kanban_link` を介して高レベルの目標を子タスクに分解し、一歩引くことです。オーケストレータースキルが誘惑への対抗ルールをエンコードしています。

## 外部 CLI ワーカーレーンの追加 {#adding-an-external-cli-worker-lane}

非 Hermes の CLI ツール（Codex CLI、Claude Code CLI、OpenCode CLI、ローカルのコーディングモデルランナーなど）をカンバンワーカーレーンとして配線することは、*まだ整備された道ではありません*。ディスパッチャーのスポーン関数はプラグイン可能で（`spawn_fn` は `dispatch_once` のパラメーターです）、プラグインは非 Hermes の担当者用に独自の `spawn_fn` を登録できますが、周辺の連携作業 — CLI の終了コードを `kanban_complete` / `kanban_block` 呼び出しにラップすること、CLI のワークスペース/サンドボックスの慣例をディスパッチャーの `HERMES_KANBAN_WORKSPACE` 環境にマッピングすること、認証と CLI ごとのポリシーを処理すること — は、依然として連携ごとの設計作業です。

CLI レーンの追加を検討している場合は、具体的な CLI と有効にしようとしているワークフローを記述した issue を開いてください。上記の契約は、そのようなレーンが満たさなければならない制約です。実装の形状（CLI ごとに 1 プラグイン vs 設定でパラメーター化された汎用 CLI ランナープラグイン）は未定です。

これに関する過去の issue は [#19931](https://github.com/NousResearch/hermes-agent/issues/19931) であり、クローズされたがマージされなかった Codex 固有の PR は [#19924](https://github.com/NousResearch/hermes-agent/pull/19924) です — これらは元のアーキテクチャ提案を記述していますが、ランナーは導入されませんでした。

## ディスパッチャーが処理する失敗モード

レーンの作者がこれらを再実装する必要がないように:

- **古いクレームの TTL** — クレームした後にハートビート / complete / block を一切行わないワーカーは、`DEFAULT_CLAIM_TTL_SECONDS`（デフォルト 15 分）後に再クレームされます — ただし、ワーカープロセスが実際に死んでいる場合に限ります。ライブなワーカー（1 回のツールなし LLM 呼び出しに 20 分以上費やす遅いモデル）は、kill される代わりにクレームが *延長* されます。再クレームされるのは死んだ PID だけです。
- **クラッシュしたワーカー** — ホストローカルの PID が消えたワーカーは `detect_crashed_workers` によって検出され回収されます。タスクは `consecutive_failures` をインクリメントし、ブレーカーが作動すると自動ブロックする場合があります。
- **実行レベルのリトライ** — タスクがリトライされる（block 後、crash 後、reclaim 後）とき、ワーカーは終了ツールの `expected_run_id` パラメーターを使用して、自身の実行がすでに置き換えられていた場合に高速に失敗できます。
- **タスクごとの最大ランタイム** — `task.max_runtime_seconds` は、PID の生存状態に関係なく、実行ごとの実時間をハードキャップします。ライブ PID 延長が実行し続けてしまう、真にデッドロックしたワーカーを捕捉します。
- **取り残されたタスクの検出** — 担当者が `kanban.stranded_threshold_seconds`（デフォルト 30 分）以内にクレームを生成しない ready タスクは、`hermes kanban diagnostics` で `stranded_in_ready` 警告として表示されます。重大度はしきい値の 2 倍で error に、6 倍で critical にエスカレートします。タイポした担当者、削除されたプロファイル、ダウンした外部ワーカープールを 1 つのシグナルで捕捉します — アイデンティティに依存せず、ボードごとに管理する許可リストは不要です。

## 関連

- [カンバン概要](./kanban) — ユーザー向けの導入。
- [カンバンチュートリアル](./kanban-tutorial) — ダッシュボードを開いた状態でのウォークスルー。
- [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) — ワーカープロセスがロードするスキル。
- [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) — オーケストレーター側。
