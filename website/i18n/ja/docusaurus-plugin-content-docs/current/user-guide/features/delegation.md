---
sidebar_position: 7
title: "サブエージェント委譲"
description: "delegate_task で並列ワークストリーム用の独立した子エージェントを起動する"
---

# サブエージェント委譲

`delegate_task` ツールは、独立したコンテキスト、制限されたツールセット、独自のターミナルセッションを持つ子AIAgentインスタンスを起動します。各子エージェントは新しい会話を取得し、独立して動作します。最終的な要約のみが親のコンテキストに入ります。

## 単一タスク

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42",
    toolsets=["terminal", "file"]
)
```

## 並列バッチ

デフォルトでは最大3つの同時サブエージェント（設定可能、ハードな上限なし）:

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "toolsets": ["web"]},
    {"goal": "Research topic B", "toolsets": ["web"]},
    {"goal": "Fix the build", "toolsets": ["terminal", "file"]}
])
```

## サブエージェントのコンテキストの仕組み

:::warning 重要: サブエージェントは何も知らない
サブエージェントは**完全に新しい会話**で始まります。親の会話履歴、以前のツール呼び出し、委譲前に議論された内容について一切の知識を持ちません。サブエージェントの唯一のコンテキストは、親エージェントが `delegate_task` を呼び出すときに設定する `goal` と `context` フィールドから来ます。
:::

これは、親エージェントがサブエージェントに必要な**すべて**を呼び出し時に渡さなければならないことを意味します。

```python
# 悪い例 - サブエージェントは「そのエラー」が何か分からない
delegate_task(goal="Fix the error")

# 良い例 - サブエージェントは必要なコンテキストをすべて持つ
delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""The file api/handlers.py has a TypeError on line 47:
    'NoneType' object has no attribute 'get'.
    The function process_request() receives a dict from parse_body(),
    but parse_body() returns None when Content-Type is missing.
    The project is at /home/user/myproject and uses Python 3.11."""
)
```

サブエージェントは、あなたの目標とコンテキストから構築された焦点を絞ったシステムプロンプトを受け取ります。それはタスクを完了し、何をしたか、何を見つけたか、変更したファイル、遭遇した問題について構造化された要約を提供するよう指示します。

## 実践的な例

### 並列リサーチ

複数のトピックを同時に調査し、要約を収集します。

```python
delegate_task(tasks=[
    {
        "goal": "Research the current state of WebAssembly in 2025",
        "context": "Focus on: browser support, non-browser runtimes, language support",
        "toolsets": ["web"]
    },
    {
        "goal": "Research the current state of RISC-V adoption in 2025",
        "context": "Focus on: server chips, embedded systems, software ecosystem",
        "toolsets": ["web"]
    },
    {
        "goal": "Research quantum computing progress in 2025",
        "context": "Focus on: error correction breakthroughs, practical applications, key players",
        "toolsets": ["web"]
    }
])
```

### コードレビュー + 修正

レビューと修正のワークフローを新しいコンテキストに委譲します。

```python
delegate_task(
    goal="Review the authentication module for security issues and fix any found",
    context="""Project at /home/user/webapp.
    Auth module files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py.
    The project uses Flask, PyJWT, and bcrypt.
    Focus on: SQL injection, JWT validation, password handling, session management.
    Fix any issues found and run the test suite (pytest tests/auth/).""",
    toolsets=["terminal", "file"]
)
```

### 複数ファイルのリファクタリング

親のコンテキストを溢れさせるような大規模なリファクタリングタスクを委譲します。

```python
delegate_task(
    goal="Refactor all Python files in src/ to replace print() with proper logging",
    context="""Project at /home/user/myproject.
    Use the 'logging' module with logger = logging.getLogger(__name__).
    Replace print() calls with appropriate log levels:
    - print(f"Error: ...") -> logger.error(...)
    - print(f"Warning: ...") -> logger.warning(...)
    - print(f"Debug: ...") -> logger.debug(...)
    - Other prints -> logger.info(...)
    Don't change print() in test files or CLI output.
    Run pytest after to verify nothing broke.""",
    toolsets=["terminal", "file"]
)
```

## バッチモードの詳細

`tasks` 配列を指定すると、サブエージェントはスレッドプールを使って**並列**で実行されます。

- **最大同時実行数:** デフォルトで3タスク（`delegation.max_concurrent_children` または `DELEGATION_MAX_CONCURRENT_CHILDREN` 環境変数で設定可能。下限は1、ハードな上限なし）。制限を超えるバッチは、静かに切り詰められるのではなく、ツールエラーを返します。
- **スレッドプール:** 設定された同時実行制限を最大ワーカー数として `ThreadPoolExecutor` を使用します
- **進捗表示:** CLIモードでは、ツリービューで各サブエージェントのツール呼び出しをリアルタイムに、タスクごとの完了行とともに表示します。ゲートウェイモードでは、進捗がバッチ化され、親の進捗コールバックに中継されます
- **結果の順序:** 結果は、完了順序に関係なく入力順に合わせてタスクインデックスでソートされます
- **割り込みの伝播:** 親を割り込む（例: 新しいメッセージを送信する）と、アクティブなすべての子が割り込まれます

単一タスクの委譲は、スレッドプールのオーバーヘッドなしに直接実行されます。

## モデルのオーバーライド

`config.yaml` を介して、サブエージェント用に別のモデルを設定できます。単純なタスクをより安価/高速なモデルに委譲するのに便利です。

```yaml
# ~/.hermes/config.yaml 内
delegation:
  model: "google/gemini-flash-2.0"    # サブエージェント用のより安価なモデル
  provider: "openrouter"              # 任意: サブエージェントを別のプロバイダーにルーティングする
```

省略した場合、サブエージェントは親と同じモデルを使用します。

## ツールセット選択のヒント

`toolsets` パラメータは、サブエージェントがアクセスできるツールを制御します。タスクに応じて選択してください。

| ツールセットパターン | ユースケース |
|----------------|----------|
| `["terminal", "file"]` | コード作業、デバッグ、ファイル編集、ビルド |
| `["web"]` | リサーチ、ファクトチェック、ドキュメント検索 |
| `["terminal", "file", "web"]` | フルスタックタスク（デフォルト） |
| `["file"]` | 読み取り専用の分析、実行を伴わないコードレビュー |
| `["terminal"]` | システム管理、プロセス管理 |

指定した内容に関わらず、特定のツールセットはサブエージェントに対してブロックされます。

- `delegation` — リーフサブエージェント（デフォルト）に対してブロックされます。`role="orchestrator"` の子に対しては保持され、`max_spawn_depth` で制限されます。以下の[深さ制限とネストされたオーケストレーション](#depth-limit-and-nested-orchestration)を参照してください。
- `clarify` — サブエージェントはユーザーと対話できません
- `memory` — 共有された永続メモリへの書き込みはできません
- `code_execution` — 子エージェントはステップごとに推論すべきです
- `send_message` — クロスプラットフォームの副作用（例: Telegramメッセージの送信）はできません

## 最大反復回数

各サブエージェントには、何回ツール呼び出しのターンを取れるかを制御する反復制限（デフォルト: 50）があります。

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # 単純なタスクなので、多くのターンは不要
)
```

## 子のタイムアウト

サブエージェントは、`delegation.child_timeout_seconds` の実時間秒を超えて沈黙すると、スタックしたものとして強制終了されます。デフォルトは **600**（10分）です。以前のリリースの300秒から引き上げられました。これは、高度な推論モデルが自明でないリサーチタスクで考えている途中に強制終了されていたためです。インストールごとに調整してください。

```yaml
delegation:
  child_timeout_seconds: 600   # デフォルト
```

高速なローカルモデルでは下げ、難しい問題に取り組む低速な推論モデルでは上げてください。タイマーは子エージェントがAPI呼び出しまたはツール呼び出しを行うたびにリセットされます。本当にアイドル状態のワーカーのみが強制終了をトリガーします。

:::tip ゼロ呼び出しタイムアウト時の診断ダンプ
サブエージェントが**ゼロ**回のAPI呼び出しでタイムアウトした場合（通常は、プロバイダーに到達不能、認証失敗、またはツールスキーマの拒否）、`delegate_task` は、サブエージェントの設定スナップショット、認証情報解決のトレース、初期のエラーメッセージを含む構造化された診断を `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log` に書き込みます。以前の静かなタイムアウト動作よりも根本原因を特定しやすくなっています。
:::

## 実行中のサブエージェントの監視（`/agents`）

TUIには `/agents` オーバーレイ（エイリアス `/tasks`）が付属しており、再帰的な `delegate_task` のファンアウトを一級の監査面に変えます。

- 実行中および最近終了したサブエージェントのライブツリービュー（親ごとにグループ化）
- ブランチごとのコスト、トークン、変更ファイルの集計
- 強制終了と一時停止の制御 — 兄弟を割り込むことなく、特定のサブエージェントを途中でキャンセル
- 事後レビュー: 各サブエージェントがターンごとの履歴を、親に戻った後でもステップごとに確認できます

クラシックなCLIは `/agents` をテキスト要約として出力するだけです。オーバーレイが真価を発揮するのはTUIです。[TUI — スラッシュコマンド](/docs/user-guide/tui#slash-commands)を参照してください。

## 深さ制限とネストされたオーケストレーション {#depth-limit-and-nested-orchestration}

デフォルトでは、委譲は**フラット**です。親（深さ0）が子（深さ1）を起動し、それらの子はそれ以上委譲できません。これにより、暴走する再帰的な委譲を防ぎます。

複数段階のワークフロー（リサーチ → 統合、またはサブ問題に対する並列オーケストレーション）のために、親は独自のワーカーを委譲*できる***オーケストレーター**子を起動できます。

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # この子が独自のワーカーを起動できるようにする
    context="...",
)
```

- `role="leaf"`（デフォルト）: 子はそれ以上委譲できません。フラット委譲の動作と同一です。
- `role="orchestrator"`: 子は `delegation` ツールセットを保持します。`delegation.max_spawn_depth`（デフォルト **1** = フラット。そのため `role="orchestrator"` はデフォルトでは何もしません）で制限されます。`max_spawn_depth` を2に上げると、オーケストレーター子がリーフの孫を起動できるようになります。3にすると3レベル（上限）になります。
- `delegation.orchestrator_enabled: false`: `role` パラメータに関わらず、すべての子を `leaf` に強制するグローバルなキルスイッチです。

**コストの警告:** `max_spawn_depth: 3` と `max_concurrent_children: 3` の場合、ツリーは3×3×3 = 27の同時リーフエージェントに達する可能性があります。レベルが1つ増えるごとに支出が乗算されます。`max_spawn_depth` は意図的に上げてください。

## 寿命と耐久性

:::warning delegate_task は同期的 — 耐久性はない
`delegate_task` は**親の現在のターン内**で実行されます。すべての子が終了する（またはキャンセルされる）まで親をブロックします。これはバックグラウンドジョブキューでは**ありません**。

- 親が割り込まれた場合（ユーザーが新しいメッセージ、`/stop`、`/new` を送信）、アクティブなすべての子がキャンセルされ、`status="interrupted"` を返します。進行中の作業は破棄されます。
- 子は、親のターンが終了した後も実行を**続けません**。
- キャンセルされた子は構造化された結果（`status="interrupted"`、`exit_reason="interrupted"`）を返しますが、親も割り込まれているため、その結果がユーザーに見える返信に到達しないことがよくあります。

割り込みを生き延びる必要がある、または現在のターンよりも長く続く**耐久性のある長時間実行作業**には、次を使用してください。

- `cronjob`（action=`create`） — 別のエージェント実行をスケジュールします。親ターンの割り込みの影響を受けません。
- `terminal(background=True, notify_on_complete=True)` — エージェントが他のことをしている間も実行を続ける長時間実行のシェルコマンド。
:::

## 主要な特性

- 各サブエージェントは**独自のターミナルセッション**を取得します（親とは別）
- **ネストされた委譲はオプトイン** — `role="orchestrator"` の子のみがそれ以上委譲でき、それも `max_spawn_depth` がデフォルトの1（フラット）から上げられた場合のみです。`orchestrator_enabled: false` でグローバルに無効化できます。
- リーフサブエージェントは次を呼び出せ**ません**: `delegate_task`、`clarify`、`memory`、`send_message`、`execute_code`。オーケストレーターサブエージェントは `delegate_task` を保持しますが、他の4つは依然として使用できません。
- **割り込みの伝播** — 親を割り込むと、アクティブなすべての子（オーケストレーター配下の孫を含む）が割り込まれます
- 最終的な要約のみが親のコンテキストに入り、トークン使用量を効率的に保ちます
- サブエージェントは親の **APIキー、プロバイダー設定、認証情報プール**を継承します（レート制限時のキーローテーションを可能にします）

## delegate_task と execute_code の比較

| 要素 | delegate_task | execute_code |
|--------|--------------|-------------|
| **推論** | 完全なLLM推論ループ | 単なるPythonコードの実行 |
| **コンテキスト** | 新しい独立した会話 | 会話なし、スクリプトのみ |
| **ツールアクセス** | 推論を伴うすべての非ブロックツール | RPC経由の7ツール、推論なし |
| **並列性** | デフォルトで3つの同時サブエージェント（設定可能） | 単一スクリプト |
| **適した用途** | 判断を要する複雑なタスク | 機械的な複数ステップのパイプライン |
| **トークンコスト** | 高い（完全なLLMループ） | 低い（stdoutのみ返される） |
| **ユーザー対話** | なし（サブエージェントは確認できない） | なし |

**経験則:** サブタスクが推論、判断、または複数ステップの問題解決を要する場合は `delegate_task` を使用してください。機械的なデータ処理やスクリプト化されたワークフローが必要な場合は `execute_code` を使用してください。

## 設定

```yaml
# ~/.hermes/config.yaml 内
delegation:
  max_iterations: 50                        # 子ごとの最大ターン数（デフォルト: 50）
  # max_concurrent_children: 3              # バッチごとの並列子数（デフォルト: 3）
  # max_spawn_depth: 1                      # ツリーの深さ（1-3、デフォルト1 = フラット）。2に上げるとオーケストレーター子がリーフを起動できる。3で3レベル。
  # orchestrator_enabled: true              # 無効化するとすべての子をリーフロールに強制する。
  model: "google/gemini-3-flash-preview"             # 任意のプロバイダー/モデルのオーバーライド
  provider: "openrouter"                             # 任意の組み込みプロバイダー

# またはプロバイダーの代わりに直接カスタムエンドポイントを使用する:
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
```

:::tip
エージェントは、タスクの複雑さに基づいて委譲を自動的に処理します。明示的に委譲を依頼する必要はありません。理にかなっている場合にエージェントが自動的に行います。
:::
