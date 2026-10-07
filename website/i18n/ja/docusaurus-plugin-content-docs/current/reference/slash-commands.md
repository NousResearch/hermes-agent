---
sidebar_position: 2
title: "スラッシュコマンドリファレンス"
description: "インタラクティブ CLI およびメッセージングのスラッシュコマンドの完全リファレンス"
---

# スラッシュコマンドリファレンス

Hermes には 2 つのスラッシュコマンドのインターフェースがあり、どちらも `hermes_cli/commands.py` 内の中央の `COMMAND_REGISTRY` によって駆動されます:

- **インタラクティブ CLI スラッシュコマンド** — `cli.py` によってディスパッチされ、レジストリからの自動補完を備えます
- **メッセージングスラッシュコマンド** — `gateway/run.py` によってディスパッチされ、ヘルプテキストとプラットフォームメニューがレジストリから生成されます

インストール済みスキルも、両方のインターフェースで動的なスラッシュコマンドとして公開されます。これには `/plan` のようなバンドル済みスキルが含まれます。`/plan` はプランモードを開き、アクティブなワークスペース/バックエンドの作業ディレクトリを基準に `.hermes/plans/` 配下に Markdown のプランを保存します。

## インタラクティブ CLI スラッシュコマンド

CLI で `/` を入力すると自動補完メニューが開きます。組み込みコマンドは大文字小文字を区別しません。

### セッション

| コマンド | 説明 |
|---------|------|
| `/new`（エイリアス: `/reset`） | 新しいセッションを開始（新しいセッション ID + 履歴） |
| `/clear` | 画面をクリアして新しいセッションを開始 |
| `/history` | 会話履歴を表示 |
| `/save` | 現在の会話を保存 |
| `/retry` | 最後のメッセージを再試行（エージェントに再送信） |
| `/undo` | 最後のユーザー/アシスタントのやり取りを削除 |
| `/title` | 現在のセッションにタイトルを設定（使い方: /title My Session Name） |
| `/compress [focus topic]` | 会話コンテキストを手動で圧縮（メモリをフラッシュ + 要約）。オプションのフォーカストピックで、要約が保持する内容を絞り込めます。 |
| `/rollback` | ファイルシステムのチェックポイントを一覧表示または復元（使い方: /rollback [number]） |
| `/snapshot [create\|restore <id>\|prune]`（エイリアス: `/snap`） | Hermes の設定/状態のスナップショットを作成または復元。`create [label]` はスナップショットを保存、`restore <id>` はそれに戻す、`prune [N]` は古いスナップショットを削除、引数なしですべてを一覧表示します。 |
| `/stop` | 実行中のすべてのバックグラウンドプロセスを終了 |
| `/queue <prompt>`（エイリアス: `/q`） | 次のターン用にプロンプトをキューに入れる（現在のエージェント応答を中断しない）。 |
| `/steer <prompt>` | **次のツールコールの後**にエージェントに届く実行中の注記を注入 — 割り込みなし、新しいユーザーターンなし。テキストは、現在のツールが完了すると最後のツール結果のコンテンツに追加され、現在のツールコールループを壊さずにエージェントに新しいコンテキストを与えます。タスクの途中で方向を微調整するのに使います（例: エージェントがテストを実行している間に「認証モジュールに集中して」）。 |
| `/goal <text>` | Hermes がターンをまたいで取り組む常設のゴールを設定 — Ralph ループに対する私たちのアプローチです。各ターンの後、補助的なジャッジモデルがゴールが完了したかどうかを判断します。完了していなければ Hermes が自動で続行します。サブコマンド: `/goal status`、`/goal pause`、`/goal resume`、`/goal clear`。予算はデフォルトで 20 ターン（`goals.max_turns`）。実際のユーザーメッセージは継続ループをプリエンプトし、状態は `/resume` を生き残ります。完全なウォークスルーは [Persistent Goals](/docs/user-guide/features/goals) を参照してください。 |
| `/resume [name]` | 以前に名前を付けたセッションを再開 |
| `/sessions` | インタラクティブなピッカーで以前のセッションをブラウズして再開 |
| `/redraw` | UI 全体を強制的に再描画（tmux のリサイズ後のターミナルのずれ、マウス選択のアーティファクトなどから復旧） |
| `/status` | セッション情報を表示 |
| `/agents`（エイリアス: `/tasks`） | 現在のセッションのアクティブなエージェントと実行中のタスクを表示。 |
| `/background <prompt>`（エイリアス: `/bg`、`/btw`） | 別のバックグラウンドセッションでプロンプトを実行。エージェントはプロンプトを独立して処理 — 現在のセッションは他の作業のために空いたままです。結果はタスク完了時にパネルとして表示されます。[CLI Background Sessions](/docs/user-guide/cli#background-sessions) を参照してください。 |
| `/branch [name]`（エイリアス: `/fork`） | 現在のセッションをブランチ（別のパスを探る） |
| `/handoff <platform>` | **CLI 専用。** 現在のセッションをメッセージングプラットフォーム（Telegram、Discord、Slack、WhatsApp、Signal、Matrix）に引き継ぎます。ゲートウェイが即座に引き取り、スレッドをサポートするプラットフォーム（Telegram トピック、Discord テキストチャンネルスレッド、Slack メッセージアンカースレッド）では新しいスレッドを作成し、ロール認識のフルトランスクリプトが再生されるように宛先を CLI の session_id に再バインドし、合成ユーザーターンを生成してエージェントが新しい場所で動作していることを確認します。成功時、CLI は `/resume` ヒントとともにクリーンに終了します。`/resume <title>` でいつでもローカルに再開できます。ターンの途中では拒否されます。ゲートウェイが実行中で、ターゲットプラットフォームにホームチャンネルが設定されている（宛先チャットから `/sethome`）必要があります。[Cross-Platform Handoff](/docs/user-guide/sessions#cross-platform-handoff) を参照してください。 |

### 設定

| コマンド | 説明 |
|---------|------|
| `/config` | 現在の設定を表示 |
| `/model [model-name]` | 現在のモデルを表示または変更。サポート: `/model claude-sonnet-4`、`/model provider:model`（プロバイダー切り替え）、`/model custom:model`（カスタムエンドポイント）、`/model custom:name:model`（名前付きカスタムプロバイダー）、`/model custom`（エンドポイントから自動検出）、ユーザー定義エイリアス（`/model fav`、`/model grok` — [カスタムモデルエイリアス](#custom-model-aliases)を参照）。`--global` を使うと変更を config.yaml に永続化します。**注:** `/model` は既に設定済みのプロバイダー間でのみ切り替えできます。新しいプロバイダーを追加するには、セッションを終了してターミナルから `hermes model` を実行してください。 |
| `/personality` | 定義済みのパーソナリティを設定 |
| `/verbose` | ツール進捗表示を循環: off → new → all → verbose。設定で[メッセージング向けに有効化](#notes)できます。 |
| `/fast [normal\|fast\|status]` | 高速モードを切り替え — OpenAI Priority Processing / Anthropic Fast Mode。オプション: `normal`、`fast`、`status`。 |
| `/reasoning` | 推論エフォートと表示を管理（使い方: /reasoning [level\|show\|hide]） |
| `/skin` | 表示スキン/テーマを表示または変更 |
| `/statusbar`（エイリアス: `/sb`） | コンテキスト/モデルのステータスバーをオン/オフ切り替え |
| `/voice [on\|off\|tts\|status]` | CLI ボイスモードと音声再生を切り替え。録音には `voice.record_key`（デフォルト: `Ctrl+B`）を使用します。 |
| `/yolo` | YOLO モードを切り替え — 危険なコマンドの承認プロンプトをすべてスキップ。 |
| `/footer [on\|off\|status]` | 最終応答にゲートウェイのランタイムメタデータフッターを切り替え（モデル、ツール回数、タイミングを表示）。 |
| `/busy [queue\|steer\|interrupt\|status]` | CLI 専用: Hermes が作業中に Enter を押したときの動作を制御 — 新しいメッセージをキューに入れる、ターン途中で steer する、即座に割り込む。 |
| `/indicator [kaomoji\|emoji\|unicode\|ascii]` | CLI 専用: TUI のビジーインジケータースタイルを選択。 |

### ツールとスキル

| コマンド | 説明 |
|---------|------|
| `/tools [list\|disable\|enable] [name...]` | ツールを管理: 利用可能なツールを一覧表示、または現在のセッションで特定のツールを無効化/有効化。ツールを無効化すると、エージェントのツールセットから削除され、セッションリセットがトリガーされます。 |
| `/toolsets` | 利用可能なツールセットを一覧表示 |
| `/browser [connect\|disconnect\|status]` | ローカル Chrome CDP 接続を管理。`connect` は実行中の Chrome インスタンス（デフォルト: `ws://localhost:9222`）にブラウザツールをアタッチします。`disconnect` はデタッチします。`status` は現在の接続を表示します。デバッガが検出されない場合は Chrome を自動起動します。 |
| `/skills` | オンラインレジストリからスキルを検索、インストール、検査、管理 |
| `/cron` | スケジュールタスクを管理（list、add/create、edit、pause、resume、run、remove） |
| `/curator` | バックグラウンドのスキルメンテナンス — `status`、`run`、`pin`、`archive`。[Curator](/docs/user-guide/features/curator) を参照してください。 |
| `/kanban <action>` | チャットを離れずにマルチプロファイル・マルチプロジェクトのコラボレーションボードを操作。`hermes kanban` のフルインターフェースが利用可能: `/kanban list`、`/kanban show t_abc`、`/kanban create "title" --assignee X`、`/kanban comment t_abc "text"`、`/kanban unblock t_abc`、`/kanban dispatch` など。マルチボードサポートも含まれます: `/kanban boards list`、`/kanban boards create <slug>`、`/kanban boards switch <slug>`、`/kanban --board <slug> <action>`。[Kanban slash command](/docs/user-guide/features/kanban#kanban-slash-command) を参照してください。 |
| `/reload-mcp`（エイリアス: `/reload_mcp`） | config.yaml から MCP サーバーを再読み込み |
| `/reload-skills`（エイリアス: `/reload_skills`） | 新しくインストールまたは削除されたスキルがないか `~/.hermes/skills/` を再スキャン |
| `/reload` | `.env` 変数を実行中のセッションに再読み込み（再起動なしで新しい API キーを取得） |
| `/plugins` | インストール済みプラグインとそのステータスを一覧表示 |

### 情報

| コマンド | 説明 |
|---------|------|
| `/help` | このヘルプメッセージを表示 |
| `/usage` | トークン使用量、コスト内訳、セッション継続時間、そして — アクティブなプロバイダーから利用可能な場合 — プロバイダーの API からライブで取得した残りクォータ / クレジット / プラン使用量を含む **Account limits** セクションを表示。 |
| `/insights` | 使用状況のインサイトと分析を表示（過去 30 日間） |
| `/platforms`（エイリアス: `/gateway`） | ゲートウェイ/メッセージングプラットフォームのステータスを表示 |
| `/paste` | クリップボードの画像を添付 |
| `/copy [number]` | 最後のアシスタント応答をクリップボードにコピー（数値を指定すると後ろから N 番目）。CLI 専用。 |
| `/image <path>` | 次のプロンプト用にローカル画像ファイルを添付。 |
| `/debug` | デバッグレポート（システム情報 + ログ）をアップロードして共有可能なリンクを取得。メッセージングでも利用可能。 |
| `/profile` | アクティブなプロファイル名とホームディレクトリを表示 |
| `/gquota` | Google Gemini Code Assist のクォータ使用量をプログレスバーで表示（`google-gemini-cli` プロバイダーがアクティブな場合のみ利用可能）。 |

### 終了

| コマンド | 説明 |
|---------|------|
| `/quit` | CLI を終了（`/exit` も可）。 |

### 動的 CLI スラッシュコマンド

| コマンド | 説明 |
|---------|------|
| `/<skill-name>` | インストール済みのスキルをオンデマンドコマンドとして読み込む。例: `/gif-search`、`/github-pr-workflow`、`/excalidraw`。 |
| `/skills ...` | レジストリと公式のオプションスキルカタログからスキルを検索、ブラウズ、検査、インストール、監査、公開、設定。 |

### クイックコマンド

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

よく使うモデルに自分用の短い名前を定義し、CLI または任意のメッセージングプラットフォームで `/model <alias>` で呼び出せます。エイリアスは両方で同一に動作し、セッション限定（デフォルト）と `--global` の切り替えのどちらでも機能します。

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
```

**ショートフォーム** — `provider/model` を 1 つの文字列で。YAML を編集せずにシェルから設定:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

そしてチャット内で:

```
/model fav            # セッション限定
/model grok --global  # 現在のモデル変更を config.yaml にも永続化
```

ユーザーエイリアスは組み込みの短縮名より優先されるため、エイリアスを `sonnet`、`kimi`、`opus` などと名付けると組み込みをシャドウします。エイリアス名は大文字小文字を区別しません。

### エイリアスの解決

コマンドはプレフィックスマッチングをサポートします: `/h` と入力すると `/help` に、`/mod` は `/model` に解決されます。プレフィックスが曖昧（複数のコマンドにマッチ）な場合、レジストリ順での最初のマッチが優先されます。完全なコマンド名と登録されたエイリアスは、常にプレフィックスマッチより優先されます。

## メッセージングスラッシュコマンド

メッセージングゲートウェイは、Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant、Teams のチャット内で次の組み込みコマンドをサポートします:

| コマンド | 説明 |
|---------|------|
| `/new` | 新しい会話を開始。 |
| `/reset` | 会話履歴をリセット。 |
| `/status` | セッション情報を表示。 |
| `/stop` | 実行中のすべてのバックグラウンドプロセスを終了し、実行中のエージェントを割り込む。 |
| `/model [provider:model]` | モデルを表示または変更。プロバイダー切り替え（`/model zai:glm-5`）、カスタムエンドポイント（`/model custom:model`）、名前付きカスタムプロバイダー（`/model custom:local:qwen`）、自動検出（`/model custom`）、ユーザー定義エイリアス（`/model fav`、`/model grok` — [カスタムモデルエイリアス](#custom-model-aliases)を参照）をサポート。`--global` を使うと変更を config.yaml に永続化します。**注:** `/model` は既に設定済みのプロバイダー間でのみ切り替えできます。新しいプロバイダーを追加したり API キーをセットアップしたりするには、（チャットセッションの外で）ターミナルから `hermes model` を使用してください。 |
| `/personality [name]` | セッションにパーソナリティオーバーレイを設定。 |
| `/fast [normal\|fast\|status]` | 高速モードを切り替え — OpenAI Priority Processing / Anthropic Fast Mode。 |
| `/retry` | 最後のメッセージを再試行。 |
| `/undo` | 最後のやり取りを削除。 |
| `/sethome`（エイリアス: `/set-home`） | 現在のチャットを配信用のプラットフォームホームチャンネルとしてマーク。 |
| `/compress [focus topic]` | 会話コンテキストを手動で圧縮。オプションのフォーカストピックで、要約が保持する内容を絞り込めます。 |
| `/topic [off\|help\|session-id]` | **Telegram DM 専用。** ユーザー管理のマルチセッショントピックモードを管理。`/topic` は有効化またはステータス表示、`/topic off` は無効化してバインディングをクリア、`/topic help` は使い方を表示、トピック内の `/topic <session-id>` は以前のセッションを復元します。[Multi-session DM mode](/docs/user-guide/messaging/telegram#multi-session-dm-mode-topic) を参照してください。 |
| `/title [name]` | セッションタイトルを設定または表示。 |
| `/resume [name]` | 以前に名前を付けたセッションを再開。 |
| `/usage` | トークン使用量、推定コスト内訳（入力/出力）、コンテキストウィンドウの状態、セッション継続時間、そして — アクティブなプロバイダーから利用可能な場合 — プロバイダーの API からライブで取得した残りクォータ / クレジットを含む **Account limits** セクションを表示。 |
| `/insights [days]` | 使用状況の分析を表示。 |
| `/reasoning [level\|show\|hide]` | 推論エフォートを変更、または推論表示を切り替え。 |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | チャットでの音声応答を制御。`join`/`channel`/`leave` は Discord ボイスチャンネルモードを管理。 |
| `/rollback [number]` | ファイルシステムのチェックポイントを一覧表示または復元。 |
| `/background <prompt>` | 別のバックグラウンドセッションでプロンプトを実行。結果はタスク完了時に同じチャットに配信されます。[Messaging Background Sessions](/docs/user-guide/messaging/#background-sessions) を参照してください。 |
| `/queue <prompt>`（エイリアス: `/q`） | 現在のターンを中断せずに、次のターン用にプロンプトをキューに入れる。 |
| `/steer <prompt>` | 割り込みなしで次のツールコールの後にメッセージを注入 — モデルは新しいターンとしてではなく、次のイテレーションでそれを取り込みます。 |
| `/goal <text>` | Hermes がターンをまたいで取り組む常設のゴールを設定 — Ralph ループに対する私たちのアプローチです。ジャッジモデルが各ターンの後にチェックし、完了していなければ、完了するか、あなたが pause/clear するか、ターン予算（デフォルト 20）に達するまで Hermes が自動で続行します。サブコマンド: `/goal status`、`/goal pause`、`/goal resume`、`/goal clear`。status/pause/clear はエージェント実行中でも安全に実行できます。新しいゴールの設定にはまず `/stop` が必要です。[Persistent Goals](/docs/user-guide/features/goals) を参照してください。 |
| `/footer [on\|off\|status]` | 最終応答にランタイムメタデータフッターを切り替え（モデル、ツール回数、タイミングを表示）。 |
| `/curator [status\|run\|pin\|archive]` | バックグラウンドのスキルメンテナンスコントロール。 |
| `/kanban <action>` | チャットからマルチプロファイル・マルチプロジェクトのコラボレーションボードを操作 — CLI と同一の引数インターフェース。実行中エージェントのガードをバイパスするため、`/kanban unblock t_abc`、`/kanban comment t_abc "…"`、`/kanban list --mine`、`/kanban boards switch <slug>` などがターンの途中で機能します。`/kanban create …` は、発信元のチャットを新しいタスクのターミナルイベントに自動サブスクライブします。[Kanban slash command](/docs/user-guide/features/kanban#kanban-slash-command) を参照してください。 |
| `/reload-mcp`（エイリアス: `/reload_mcp`） | 設定から MCP サーバーを再読み込み。 |
| `/yolo` | YOLO モードを切り替え — 危険なコマンドの承認プロンプトをすべてスキップ。 |
| `/commands [page]` | すべてのコマンドとスキルをブラウズ（ページ分け）。 |
| `/approve [session\|always]` | 保留中の危険なコマンドを承認して実行。`session` はこのセッションのみ承認、`always` は永続的な許可リストに追加。 |
| `/deny` | 保留中の危険なコマンドを拒否。 |
| `/update` | Hermes Agent を最新バージョンに更新。 |
| `/restart` | アクティブな実行をドレインした後、ゲートウェイを正常に再起動。ゲートウェイがオンラインに戻ると、リクエスト元のチャット/スレッドに確認を送信します。 |
| `/debug` | デバッグレポート（システム情報 + ログ）をアップロードして共有可能なリンクを取得。 |
| `/help` | メッセージングヘルプを表示。 |
| `/<skill-name>` | インストール済みのスキルを名前で呼び出す。 |

## 注記 {#notes}

- `/skin`、`/snapshot`、`/gquota`、`/reload`、`/tools`、`/toolsets`、`/browser`、`/config`、`/cron`、`/skills`、`/platforms`、`/paste`、`/image`、`/statusbar`、`/plugins`、`/busy`、`/indicator`、`/redraw`、`/clear`、`/history`、`/save`、`/copy`、`/handoff`、`/quit` は **CLI 専用**のコマンドです。
- `/verbose` は**デフォルトでは CLI 専用**ですが、`config.yaml` で `display.tool_progress_command: true` を設定することでメッセージングプラットフォーム向けに有効化できます。有効化すると、`display.tool_progress` モードを循環させ、設定に保存します。
- `/sethome`、`/update`、`/restart`、`/approve`、`/deny`、`/topic`、`/commands` は**メッセージング専用**のコマンドです。
- `/status`、`/background`、`/queue`、`/steer`、`/voice`、`/reload-mcp`、`/reload-skills`、`/rollback`、`/debug`、`/fast`、`/footer`、`/curator`、`/kanban`、`/sessions`、`/yolo` は CLI とメッセージングゲートウェイの**両方**で動作します。
- `/voice join`、`/voice channel`、`/voice leave` は Discord でのみ意味を持ちます。
