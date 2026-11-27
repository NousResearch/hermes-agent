---
sidebar_position: 1
title: "CLIコマンドリファレンス"
description: "Hermesのターミナルコマンドおよびコマンドファミリーの公式リファレンス"
---

# CLIコマンドリファレンス

このページのPython依存関係コマンドは、
[PMで準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を前提としています。
依存関係を変更した後は、チェックアウトを再アクティベートしてHermesを再起動してください。

このページでは、シェルから実行する**ターミナルコマンド**を扱います。

チャット内のスラッシュコマンドについては、[スラッシュコマンドリファレンス](./slash-commands.md)を参照してください。

## グローバルエントリーポイント {#global-entrypoint}

```bash
hermes [global-options] <command> [subcommand/options]
```

### グローバルオプション {#global-options}

| オプション | 説明 |
|--------|-------------|
| `--version`, `-V` | バージョンを表示して終了します。 |
| `--profile <name>`, `-p <name>` | この実行で使用するHermesプロファイルを選択します。`hermes profile use` で設定した固定デフォルトを上書きします。 |
| `--resume <session>`, `-r <session>` | IDまたはタイトルで以前のセッションを再開します。キーワード `latest` は最新のセッションを再開します（ワークスペース単位で、`-c` と同じ検索方法）。 |
| `--continue [name]`, `-c [name]` | 最新のセッション、またはタイトルが一致する最新のセッションを再開します。 |
| `--in <dir>` | 開始または再開の前に `<dir>` に移動します。`--resume latest` / `-c` の検索をそのディレクトリのワークスペースに限定し、セッションをそこに留めます（記録されたcwdの復元をスキップします）。 |
| `--worktree`, `-w` | 並列エージェントワークフロー向けに、分離されたgit worktreeで開始します。 |
| `--yolo` | 危険なコマンドの承認プロンプトをバイパスします。 |
| `--pass-session-id` | エージェントのシステムプロンプトにセッションIDを含めます。 |
| `--ignore-user-config` | `~/.hermes/config.yaml` を無視し、組み込みのデフォルトにフォールバックします。`.env` の認証情報は引き続き読み込まれます。 |
| `--ignore-rules` | `AGENTS.md`、`SOUL.md`、`.cursorrules`、メモリ、プリロードされたスキルの自動注入をスキップします。 |
| `--tui` | クラシックCLIの代わりに[TUI](../user-guide/tui.md)を起動します。`HERMES_TUI=1` と同等です。常に `display.interface` より優先されます。 |
| `--cli` | クラシックなprompt_toolkit REPLを強制します。1回の実行だけ `display.interface: tui` を上書きしたい場合に使用します。 |
| `--dev` | `--tui` と併用時: プリビルドのバンドルではなく、`tsx` 経由でTypeScriptソースを直接実行します（TUIコントリビューター向け）。 |

### `hermes-agent`（レガシーな単一クエリランナー） {#hermes-agent-legacy-single-query-runner}

インストールには `hermes-agent` も同梱されています。これは1つのクエリを送信して終了する最小限のランナーです: `hermes-agent --query "summarize README.md"`（または `hermes-agent "summarize README.md"`）。`hermes-agent --help` はそのオプション（`--model`、`--base-url`、`--max-turns`、`--enabled-toolsets`、`--disabled-toolsets`、`--list-tools`、`--save-trajectories`、…）を一覧表示し、`hermes-agent --version` はバージョンを出力します。どちらもエージェントは起動しません。クエリなしで実行すると、同じヘルプを出力して終了します。それ以外の用途には `hermes` を使用してください（スクリプト用のワンショットは `hermes -z <prompt>` です）。

## トップレベルコマンド {#top-level-commands}

| コマンド | 目的 |
|---------|---------|
| `hermes chat` | エージェントとの対話的またはワンショットのチャット。 |
| `hermes model` | デフォルトのプロバイダーとモデルを対話的に選択します。 |
| `hermes moa` | モデルピッカーから選択できる、名前付きのMixture of Agentsプリセットを設定します。 |
| `hermes fallback` | プライマリモデルがエラーになったときに試行されるフォールバックプロバイダーを管理します。 |
| `hermes gateway` | メッセージングゲートウェイサービスを実行または管理します。 |
| `hermes proxy` | OAuthプロバイダーの認証情報を付与する、ローカルのOpenAI互換プロキシ。[サブスクリプションプロキシ](/docs/user-guide/features/subscription-proxy)を参照してください。 |
| `hermes egress` | リモートターミナルサンドボックス向けの、送信時に認証情報を注入するファイアウォール（iron-proxy）。デフォルトでは無効です。[Egressプロキシ](/docs/user-guide/egress/iron-proxy)を参照してください。 |
| `hermes lsp` | Language Server Protocol統合を管理します（write_file/patch向けのセマンティック診断）。 |
| `hermes setup` | 設定の全体または一部を対象とする対話型セットアップウィザード。 |
| `hermes whatsapp` | WhatsAppブリッジを設定・ペアリングします。 |
| `hermes whatsapp-cloud` | Meta公式のWhatsApp Business Cloud APIアダプターを設定します（Businessアカウントと公開webhookが必要）。`hermes whatsapp`（Baileysによる個人アカウントブリッジ）とは別物です。 |
| `hermes slack` | Slackヘルパー（現在: すべてのコマンドをネイティブスラッシュとして含むアプリマニフェストを生成）。 |
| `hermes auth` | 認証情報を管理します — 追加、一覧表示、削除、リセット、ステータス、ログアウト。Codex/Nous/AnthropicのOAuthフローを処理します。 |
| `hermes login` / `logout` | **非推奨** — 代わりに `hermes auth` を使用してください。 |
| `hermes send` | 設定済みのメッセージングプラットフォーム（Telegram、Discord、Slack、Signal、SMS、…）にワンショットのメッセージを送信します。シェルスクリプト、cronジョブ、CIフック、監視デーモンから使うのに便利です — エージェントループもLLMも使いません。 |
| `hermes peer` | 他のマシン上のピアHermesゲートウェイを登録し、そのエージェントの正規のBot ChatにDMを送ります（`hermes peer dm <peer>[/<agent>] "…"`）。マシン間のボット同士のメッセージングを支えるトランスポートです。 |
| `hermes secrets` | `~/.hermes/.env` からではなくプロセス起動時にAPIキーを取得するための、外部シークレットソース（現在はBitwarden Secrets Manager）を管理します。 |
| `hermes migrate` | 別のエージェントから移行する（`migrate grok-bot`、`migrate claude-code`、…、または引数なしで自動検出）、あるいは廃止されたモデルや非推奨の設定に合わせて `config.yaml` を書き換えます（`migrate xai`）。 |
| `hermes codex-runtime` | `/codex-runtime` の非対話版: `migrate [--dry-run] [--json]` は、選択したプロファイルについて `~/.codex/config.toml` 内のHermes管理ブロックを再生成します。[Codex app-serverランタイム](/docs/user-guide/features/codex-app-server-runtime#running-the-migration-from-a-script)を参照してください。 |
| `hermes status` | エージェント、認証、プラットフォームのステータスを表示します。 |
| `hermes usage` | セッションなしで、設定済みアカウントのレート制限ウィンドウ（`/usage` ブロック）を表示します。スクリプト向けには `--json`。 |
| `hermes cron` | cronスケジューラーを検査・実行します。 |
| `hermes pause` / `hermes resume` | グローバルな緊急停止: 再開されるまで、新しいcronの発火（組み込みティッカー、managed-cronのwebhook、取りこぼしのキャッチアップ）、kanbanのディスパッチ、ゲートウェイのターンは開始されません。対話型CLI、TUI、Desktopのターンもゲートウェイのターンなので同様に保留され、各画面には理由とともに「Hermes is paused」と「run `hermes resume`」が表示されます。すでに実行中のターンは完了まで続き、進行中の作業が強制終了されることはありません。 |
| `hermes kanban` | マルチプロファイルのコラボレーションボード（タスク、リンク、ディスパッチャー）。 |
| `hermes project` | 名前付きの複数フォルダーのワークスペース（プロジェクト）を管理します。デスクトップのセッショングループ化の基準となり、プライマリフォルダーを持つプロジェクトにリンクされたkanbanタスクは、デフォルトで決定論的なworktree + ブランチを使用します。状態はプロファイルごとです。 |
| `hermes webhook` | イベント駆動の起動向けに、動的なWebhookサブスクリプションを管理します。 |
| `hermes hooks` | `config.yaml` で宣言されたシェルスクリプトフックを検査、承認、削除します。 |
| `hermes doctor` | 設定や依存関係の問題を診断します。 |
| `hermes security audit` | venv、プラグインの要件、ピン留めされたMCPサーバーを対象とした、オンデマンドのサプライチェーン監査（OSV.dev）。 |
| `hermes approvals` | 承認プロンプト用ツール — 承認履歴を分析して許可リストの提案を作成します。 |
| `hermes dump` | サポート/デバッグ向けの、コピーペースト可能なセットアップ要約。 |
| `hermes prompt-size` | システムプロンプト + ツールスキーマ（スキルインデックス、メモリ、プロファイル）のバイト内訳を表示します。オフラインで動作します。 |
| `hermes debug` | デバッグツール — サポート向けにログとシステム情報をアップロードします。 |
| `hermes backup` | Hermesのホームディレクトリをzipファイルにバックアップします。 |
| `hermes checkpoints` | `~/.hermes/checkpoints/`（`/rollback` が使用するシャドウストア）を検査 / プルーニング / クリアします。引数なしで実行するとステータスの概要を表示します。 |
| `hermes import` | zipファイルからHermesバックアップを復元します。 |
| `hermes logs` | エージェント/ゲートウェイ/エラーのログファイルを表示、追跡、フィルタリングします。 |
| `hermes config` | 設定ファイルを表示、編集、移行、クエリします。 |
| `hermes skin` | 表示スキンの一覧表示、切り替え、調整を行います。 |
| `hermes console` | 安全なHermesコマンドコンソールを開きます。 |
| `hermes pairing` | メッセージングのペアリングコードを承認または取り消します。 |
| `hermes skills` | スキルをブラウズ、インストール、公開、監査、設定します。 |
| `hermes bundles` | 複数のスキルを1つの `/<name>` スラッシュコマンドにまとめます。[スキルバンドル](../user-guide/features/skills.md#skill-bundles)を参照してください。 |
| `hermes curator` | バックグラウンドのスキルメンテナンス — ステータス、実行、一時停止、固定。[Curator](../user-guide/features/curator.md)を参照してください。 |
| `hermes journey`（エイリアス `learning`, `memory-graph`） | 学習したスキルとメモリの時系列タイムライン。 |
| `hermes memory` | 外部メモリプロバイダーを設定します。プラグイン固有のサブコマンド（例: `hermes honcho`）は、そのプロバイダーがアクティブなときに自動的に登録されます。 |
| `hermes acp` | エディター統合向けにHermesをACPサーバーとして実行します。 |
| `hermes mcp` | MCPサーバー設定を管理します。HermesをMCPサーバーとして実行する機能は[mcp-serveプラグイン](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server)が提供します。 |
| `hermes connectors` | ホスト型コネクターのアカウントを一覧表示、接続、リネーム、切断します。 |
| `hermes plugins` | Hermes Agentのプラグインを管理します（インストール、有効化、無効化、削除）。 |
| `hermes portal` | Nous Portalのステータス、サブスクリプションリンク、Tool Gatewayのルーティング。[Tool Gateway](../user-guide/features/tool-gateway.md)を参照してください。 |
| `hermes tools` | プラットフォームごとに有効なツールを設定します。 |
| `hermes computer-use` | Computer Use（cua-driver）バックエンドをインストールまたは確認します（macOS/Windows/Linux）。 |
| `hermes pets` | CLI、TUI、デスクトップアプリに表示される[petdex](/docs/user-guide/features/pets)のアニメーションペットをブラウズ、インストール、選択します。サブコマンド: `list`、`install`、`select`、`show`、`off`、`scale`、`remove`、`doctor`。 |
| `hermes clipboard image` | このマシンのクリップボード画像をファイルに書き出します（スクリプト向けには `--json`）。デバイスローカルで、ゲートウェイもセッションストアも使いません。 |
| `hermes sessions` | セッションをブラウズ、エクスポート、プルーニング、リネーム、削除します。 |
| `hermes insights` | トークン/コスト/アクティビティの分析を表示します。 |
| `hermes claw` | [`agent-importers`](/docs/user-guide/import-from-other-agents) プラグインによるOpenClawの移行（`migrate`、`cleanup`）。`hermes claw migrate` は `hermes import-agent openclaw` と同じです。 |
| `hermes import-agent` | OpenClaw、Claude Code、Codex CLI、Cursorのセットアップをインポートします（[`agent-importers`](/docs/user-guide/import-from-other-agents) プラグインによる）。 |
| `hermes dashboard` | 設定、APIキー、セッションを管理するためのWebダッシュボードを起動します。 |
| `hermes serve` | Hermesバックエンドサーバーを起動します（ヘッドレス。デスクトップアプリとリモートバックエンドを支えます）。 |
| `hermes webapp` | ゲートウェイが配信するDesktopワークスペースをブラウザで開きます。 |
| `hermes desktop`（エイリアス `gui`） | ネイティブのElectronデスクトップアプリをビルドして起動します。 |
| `hermes profile` | プロファイルを管理します — 複数の分離されたHermesインスタンス。 |
| `hermes completion` | シェル補完スクリプトを出力します（bash/zsh/fish）。 |
| `hermes --version` | バージョン情報を表示します。 |
| `hermes update` | 最新の安定版リリース（または選択したチャンネル）に更新し、依存関係を再インストールします。`--check` はインストールせずにプレビューします。`--backup` はpull前の `HERMES_HOME` スナップショットを取得します。 |
| `hermes uninstall` | システムからHermesを削除します。 |

## `hermes chat`

```bash
hermes chat [options]
```

主なオプション:

| オプション | 説明 |
|--------|-------------|
| `-q`, `--query "..."` | プロンプトでセッションを開始します。実際のTTY上では、プロンプトは通常の対話型セッションの最初のターンとして**そのまま**送信され（スラッシュコマンドや `!` シェルエスケープとして解釈されることはありません）、セッションは開いたままになります — OSのランチャーやデスクトップ統合に最適です。`--oneshot`、`-Q`、または非TTYのstdioでは、回答して終了します。 |
| `--query-file PATH` | ファイルからクエリを読み取ります（`-` = stdin）。シェルによる解釈は一切行われないため、引用符、`$(...)`、バッククォートはそのまま届きます — プログラムから渡すメッセージや信頼できないメッセージ本文に使用してください（Bot ModeのチームメイトDMはこれを使用します）。`-q` とは同時に指定できません。 |
| `--oneshot` | `-q`/`--query-file` と併用時: 対話型セッションを開始する代わりに、クエリに回答して終了します（0.21以前の単一クエリの動作）。非TTYのstdioおよび `-Q` では暗黙的に有効になります。 |
| `-m`, `--model <model>` | この実行のモデルを上書きします。 |
| `-t`, `--toolsets <csv>` | カンマ区切りのツールセットを有効にします。 |
| `--provider <provider>` | プロバイダーを強制指定します: `auto`, `openrouter`, `nous`, `openai-codex`（エイリアス `chatgpt`, `chatgpt-codex`）, `anthropic`, `gemini`, `huggingface`, `novita`（エイリアス `novita-ai`, `novitaai`）, `openai-api`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `kilocode`, `xiaomi`, `arcee`, `gmi`, `upstage`（エイリアス `solar`）, `alibaba`, `alibaba-cn`, `alibaba-coding-plan`（エイリアス `alibaba_coding`）, `alibaba-coding-plan-cn`, `alibaba-token-plan`, `alibaba-token-plan-cn`, `deepseek`, `nvidia`, `ollama-cloud`, `xai`（エイリアス `grok`）, `xai-oauth`（エイリアス `grok-oauth`）, `qwen-oauth`, `bedrock`, `opencode-zen`, `opencode-go`, `commandcode`, `commandcode-anthropic`, `ai-gateway`, `azure-foundry`, `lmstudio`, `stepfun`, `tencent-tokenhub`（エイリアス `tencent`, `tokenhub`）, `router`（エイリアス `ramp-router`, `ramp`）, `nebius-token-factory`（エイリアス `nebius`, `nebius-tf`, `tokenfactory`）, `tencent-tokenplan`（エイリアス `tokenplan`, `tencent-lkeap`）。 |
| `-s`, `--skills <name>` | セッション向けに1つ以上のスキルをプリロードします（繰り返し指定またはカンマ区切り可）。 |
| `-v`, `--verbose` | 詳細出力。 |
| `-Q`, `--quiet` | プログラム実行モード: バナー/スピナー/ツールプレビューを抑制します。 |
| `--format stream-json` | `-q` / `--query` の実行で構造化JSONLを出力します。`--quiet` を含意し、`--tui` とは併用できません。 |
| `--image <path>` | 単一のクエリにローカル画像を添付します。 |
| `--resume <session>` / `--continue [name]` | `chat` から直接セッションを再開します。 |
| `--worktree` | この実行向けに分離されたgit worktreeを作成します。 |
| `--checkpoints` | 破壊的なファイル変更の前にファイルシステムのチェックポイントを有効にします。 |
| `--yolo` | 承認プロンプトをスキップします。 |
| `--pass-session-id` | システムプロンプトにセッションIDを渡します。 |
| `--ignore-user-config` | `~/.hermes/config.yaml` を無視し、組み込みのデフォルトを使用します。`.env` の認証情報は引き続き読み込まれます。分離されたCI実行、再現可能なバグレポート、サードパーティ統合に便利です。 |
| `--ignore-rules` | `AGENTS.md`、`SOUL.md`、`.cursorrules`、永続メモリ、プリロードされたスキルの自動注入をスキップします。完全に分離された実行には `--ignore-user-config` と組み合わせてください。 |
| `--safe-mode` | トラブルシューティングモード: ユーザー設定、ルール/メモリの注入、プラグイン、シェルフック、MCPサーバーといった、すべてのカスタマイズを無効にします（`--ignore-user-config` と `--ignore-rules` を含意）。ゲートウェイはコードのデフォルトと明示的に指定したオプションをセッションに固定し、プロファイルを一切読まない分離されたワーカープロセスでターンを実行するため、壊れた `config.yaml` に妨げられることはありません。明示的な `--model` が必要です（継承するプロファイルのデフォルトがないため）。完全に明示的なエンドポイントを使うには `--provider custom --base-url <url> --api-key <key>` と組み合わせてください。 |
| `--base-url <url>` / `--api-key <key>` | `--provider custom` 向けの、明示的なOpenAI互換エンドポイントと起動時のみのキー。キーはこのセッションの間、実行中のゲートウェイのメモリに保持され、ディスクに書き込まれることはありません。 |
| `--source <tag>` | フィルタリング用のセッションソースタグ（デフォルト: `cli`。ワンショット実行のデフォルトは `oneshot` で、ピッカーには表示されません）。ユーザーのセッション一覧に表示すべきでないサードパーティ統合には `tool` を使用します。明示的な `--source` は、TUIやDesktopのセッション内から起動したワンショット実行であっても、常に指定どおりに保存されます。 |
| `--max-turns <N>` | 会話の1ターンあたりのツール呼び出し反復回数の上限（デフォルト: 無制限、または設定の `agent.max_turns`）。 |

例:

```bash
hermes
hermes chat -q "Summarize the latest PRs"          # seeds an interactive session
hermes chat --oneshot -q "Summarize the latest PRs"  # answer and exit
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
hermes chat --quiet -q "Return only JSON"
hermes chat -q "Inspect this repository" --format stream-json
hermes chat --worktree -q "Review this repo and open a PR"
hermes chat --ignore-user-config --ignore-rules -q "Repro without my personal setup"
hermes chat --safe-mode --model gpt-4.1 -q "Is this bug mine or Hermes'?"
hermes chat --safe-mode --provider custom --base-url http://127.0.0.1:8000/v1 --model local -q "Isolate against a local server"
```

### `--format stream-json` — 構造化JSONL出力 {#--format-stream-json--structured-jsonl-output}

プログラムがターミナル出力をスクレイピングせずに進捗を取得する必要がある場合は、
`--format stream-json` を使用してください。`-q` / `--query`（または `--query-file`）が必要で、
静かな非対話型CLIモードを含意し、明示的な `--tui` の指定は拒否されます。stdoutの
各行は1つのJSONオブジェクトです。診断情報と `session_id:` 行はstderrに出力されます。

```bash
hermes chat -q "Summarize this repository" --format stream-json
```

すべてのイベントは `timestamp`（Unixエポックのミリ秒）を持ちます。

| イベントの `type` | フィールド |
|---|---|
| `system` | `subtype: "init"`, `model`, `session_id` |
| `text` | `text` — ストリーミングされるアシスタントのテキスト差分 |
| `tool_use` | `name`。ツールの引数が利用可能な場合は `input` |
| `tool_result` | `name`, `output`（5000文字で切り詰め）, `duration_ms`, `is_error` |
| `result` | `session_id`, `exit_code`, `text`, `tokens`（`input`, `output`, `total`, `cache_read`, `cache_write`）, `duration_ms`。ターンが失敗した場合は `error`。実行がゲートウェイのストリームに遅れて再アタッチしたため `text` イベントが途中で止まった場合は `replay_gap: true`（ここでの `text` は依然として返信全体です） |

会話が始まると、最後のレコードは常に `result` になります — Ctrl-Cで中断された場合の
`exit_code: 130` も含みます。このレコードを完了のシグナルとして扱ってください。プロセスの
終了コードはその `exit_code` と一致します。

#### ワンショット実行の終了コード {#exit-codes-for-one-shot-runs}

chatが回答して終了する場合（`-Q`、`chat --oneshot`、または非TTYのstdioでのクエリ）、
プロセスの終了コードはターンの結果を表します。これは静かなパスでもそうでないパスでも同じです:
`0` ターンが完了した。`1` 失敗した、途中で停止した（`partial`）、反復回数の上限に達した、
または実行されなかった（認証情報 / エージェントの初期化に失敗）。`130` 中断された。
Kanbanディスパッチャーが生成したワーカー（`HERMES_KANBAN_TASK` が設定済み）で、
プロバイダーのレート制限、過負荷、5xxの返却、タイムアウト、またはアカウントが
課金/クォータの上限に達したことだけが原因でターンが失敗した場合は、
`75`（`EX_TEMPFAIL`）で終了します。これによりディスパッチャーは失敗としてカウントせずに
タスクを再キューします。`--format stream-json` では、最後の `result` レコードが
同じ `exit_code` を持ちます。

#### 有限のchat実行での委任 {#delegation-in-finite-chat-runs}

chatが回答して終了する場合、`delegate_task` は子をバックグラウンドで実行する代わりに、子の完了を待って
同じターン内でその結果を返します。詳細と制限:
[委任: 有限のchat実行](../user-guide/features/delegation.md#finite-chat-runs)。

### `hermes -z <prompt>` — スクリプト用ワンショット {#hermes--z-prompt--scripted-one-shot}

プログラムから呼び出す側（シェルスクリプト、CI、cron、プロンプトをパイプで渡す親プロセス）にとって、`hermes -z` は最も純粋なワンショットのエントリーポイントです。**単一のプロンプトを入力し、最終的なレスポンステキストのみを出力し、それ以外はstdoutにもstderrにも何も出しません。** バナーもスピナーもツールプレビューも `Session:` 行もなく、エージェントの最終的な返信をプレーンテキストとして出力するだけです。

```bash
hermes -z "What's the capital of France?"
# → Paris.

# Parent scripts can cleanly capture the response:
answer=$(hermes -z "summarize this" < /path/to/file.txt)
```

実行ごとの上書き（`~/.hermes/config.yaml` は変更しません）:

| フラグ | 同等の環境変数 | 目的 |
|---|---|---|
| `-m` / `--model <model>` | `HERMES_INFERENCE_MODEL` | この実行のモデルを上書きします |
| `--provider <provider>` | _（なし）_ | この実行のプロバイダーを上書きします |
| `--usage-file <path>` | _（なし）_ | 実行後にJSONの使用量レポートを書き出します（下記参照） |

```bash
hermes -z "…" --provider openrouter --model openai/gpt-5.5
# or:
HERMES_INFERENCE_MODEL=anthropic/claude-sonnet-4.6 hermes -z "…"
```

同じエージェント、同じツール、同じスキルで、対話的 / 装飾的なレイヤーをすべて取り除くだけです。トランスクリプトにツール出力も必要な場合は、代わりに `hermes chat --oneshot -q` を使用してください。`-z` は明示的に「最終的な答えだけが欲しい」場合のためのものです。

終了コード: `0` ターンが完了した。`2` 失敗した、または途中で停止した（`partial`、
反復回数の上限、`completed: false`） — 説明が出力された場合でも同様です。
`130` 中断された。`1` 完了したターンがテキストを一切生成しなかった。`2` は実行開始前の
使用法エラー（不正なフラグ）にも使われます。これらのコードは上記の `chat -q`/`-Q`
（失敗/partial/上限の場合は `1`、テキストなしで完了したターンは `0` で終了）とは意図的に
異なります: `-z` は `1` を「何も回答しなかった」ために予約しています。
実行の成否は、stdoutが空でないかどうかではなく、終了コード（または `--usage-file` のフラグ）で
判断してください。

#### `--usage-file` — パイプライン向けJSON使用量レポート {#--usage-file--json-usage-report-for-pipelines}

`hermes -z "…" --usage-file /path/report.json` は、実行後に機械可読の使用量レポートを書き出します: `estimated_cost_usd`、`input_tokens` / `output_tokens` / `cache_read_tokens` / `cache_write_tokens` / `reasoning_tokens` / `total_tokens`、`api_calls`、`model`、`provider`、`session_id`、`service_tier`、`completed` / `failed` / `partial` / `interrupted` フラグ、`turn_exit_reason`（`completed` がfalseである理由。例: `max_iterations_reached(3/3)`）。これらのトップレベルのカウンターは**メインのエージェントループ**のみを対象とします。同じ実行で行われた補助LLM呼び出し（タイトル生成、ビジョン、コンテキスト圧縮、バックグラウンドレビュー、…）は `auxiliary` の下に別途報告され — 同じ合計値に加えてタスクごとの `by_task` マップ — `total_including_auxiliary`（`estimated_cost_usd`、`total_tokens`、`api_calls`）が請求の基準となる総合計です。レポートは**実行が失敗した場合でも**書き出されるため、バッチパイプラインは常に支出を把握できます。`-z`/`--oneshot` 以外では効果がなく、使用量の書き込みが失敗しても実行自体の結果が隠されることはありません。

```bash
hermes -z "summarize this repo" --usage-file ~/.hermes/cache/scratch/usage.json
jq .total_including_auxiliary.estimated_cost_usd ~/.hermes/cache/scratch/usage.json
jq .auxiliary.by_task ~/.hermes/cache/scratch/usage.json      # what did title generation / vision cost?
```

## `hermes model`

対話型のプロバイダー + モデルセレクター。**これは、新しいプロバイダーの追加、APIキーのセットアップ、OAuthフローの実行を行うためのコマンドです。** アクティブなHermesチャットセッションの中からではなく、ターミナルから実行してください。

```bash
hermes model
```

次のことをしたいときに使用します:
- **新しいプロバイダーを追加する**（OpenRouter、Anthropic、DeepSeek、カスタムなど）
- OAuthベースのプロバイダーにログインする（Anthropic、Codex、Nous Portal）
- APIキーを入力または更新する
- プロバイダー固有のモデルリストから選択する
- カスタム/セルフホストのエンドポイントを設定する
- 新しいデフォルトを設定に保存する

:::warning hermes model と /model — 違いを把握する
**`hermes model`**（Hermesセッションの外、ターミナルから実行）は、**完全なプロバイダーセットアップウィザード**です。新しいプロバイダーの追加、OAuthフローの実行、APIキーの入力プロンプト、エンドポイントの設定ができます。

**`/model`**（アクティブなHermesチャットセッション内で入力）は、**すでにセットアップ済みのプロバイダーとモデルの切り替え**のみができます。新しいプロバイダーの追加、OAuthの実行、APIキーの入力プロンプトはできません。

**新しいプロバイダーを追加する必要がある場合:** まずHermesセッションを終了し（`Ctrl+C` または `/quit`）、ターミナルのプロンプトから `hermes model` を実行してください。
:::

### `/model` スラッシュコマンド（セッション中） {#model-slash-command-mid-session}

セッションを離れずに、すでに設定済みのモデルを切り替えます:

```
/model                              # Show current model and available options
/model claude-sonnet-4              # Switch model (auto-detects provider)
/model zai:glm-5                    # Switch provider and model
/model custom:qwen-2.5              # Use model on your custom endpoint
/model custom                       # Auto-detect model from custom endpoint
/model custom:local:qwen-2.5        # Use a named custom provider
/model openrouter:anthropic/claude-sonnet-4  # Switch back to cloud
```

デフォルトでは、`/model` の変更は**現在のセッションにのみ**適用されます。`--global` を追加すると、変更を `config.yaml` に永続化します（または `model.persist_switch_by_default: true` を設定すると、すべての切り替えが永続化されます）:

```
/model claude-sonnet-4 --global     # Switch and save as new default
```

:::info OpenRouterのモデルしか表示されない場合は？
OpenRouterのみを設定している場合、`/model` はOpenRouterのモデルのみを表示します。別のプロバイダー（Anthropic、DeepSeekなど）を追加するには、セッションを終了し、ターミナルから `hermes model` を実行してください。
:::

`--global` での切り替え時、プロバイダーとベースURLの変更はモデルとともに `config.yaml` に永続化されます。カスタムエンドポイントから切り替えると、古いベースURLが他のプロバイダーに漏れないようにクリアされます。

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `run` | ゲートウェイをフォアグラウンドで実行します。WSL、Docker、Termuxに推奨です。 |
| `ensure` | サービスをインストールしたり既存のオーナーを置き換えたりせずに、ローカルのセッションランタイムを検出または要求します。JSONを出力します。下記を参照してください。 |
| `attach` | ローカルクライアントに、このプロファイルのゲートウェイのWebSocket URLと使い捨てのセッションチケットを1つ渡します。JSONを出力します。下記を参照してください。 |
| `start` | インストール済みのsystemd/launchdバックグラウンドサービスを開始します。 |
| `stop` | サービス（またはフォアグラウンドプロセス）を停止します。 |
| `restart` | サービスを再起動します。 |
| `status` | サービスのステータスを表示します。 |
| `list` | **すべてのプロファイル**と、各プロファイルのゲートウェイが現在実行中かどうか（可能な場合はPID付き）を一覧表示します。複数のプロファイルを並行して実行していて、1つの概要で確認したい場合に便利です。 |
| `install` | systemd（Linux）またはlaunchd（macOS）のバックグラウンドサービスとしてインストールします。 |
| `uninstall` | インストール済みのサービスを削除します。 |
| `setup` | 対話型のメッセージングプラットフォームセットアップ。 |
| `migrate` | プロファイルごとのスタンドアロンゲートウェイを、1つのホストゲートウェイに統合します（`--multiplex`、唯一のモード — 実際の境界によって妨げられない限り、`hermes update` が自動的に実行します）。再実行すると、移行が途中で止まったホストを収束させます。ディスク上のマニフェストは再開用の記録であり、ロールバックではありません（`--standalone` はありません）。プリフライト（重複したボットトークン、`/p/<profile>/` のイングレスを持たない二次的なポートバインダー）を実行し、ブロックされた場合は何も変更しません。フラグ: `--dry-run`、`-y`/`--yes`。`--backends [--json]` は代わりに、serve/dashboardのバックグラウンドサービスとその所有関係を報告し、サービス、ユニット、plist、台帳は一切変更しません（通常の起動時の記録処理は引き続き実行されます。[ダッシュボードのバックグラウンドサービスの報告](../user-guide/features/web-dashboard.md#reporting-dashboard-background-services)を参照してください）。[プロファイルごとのゲートウェイからの移行](/docs/user-guide/multi-profile-gateways#migrating-from-per-profile-gateways)を参照してください。 |
| `migrate-legacy` | リネーム前のインストールから残っているレガシーな `hermes.service` ユニットを削除します。プロファイルのユニット（`hermes-gateway-<profile>.service`）や無関係なサービスには決して触れません。フラグ: `--dry-run`、`-y`/`--yes`。 |
| `enroll` | 実験的: このゲートウェイをリレーコネクターに登録し、コネクターを使うプラットフォーム向けのリレー認証情報を保存します。[Hermes Relay](/docs/user-guide/messaging/relay)を参照してください。 |

オプション:

| オプション | 説明 |
|--------|-------------|
| `--all` | `start` / `restart` / `stop` で: アクティブな `HERMES_HOME` だけでなく、**すべてのプロファイルの**ゲートウェイに対して動作します。複数のプロファイルを並行して実行していて、`hermes update` 後にそれらすべてを再起動したい場合に便利です。 |
| `-y`, `--yes` | `stop` / `restart` で: ゲートウェイがチャットのターン、cronジョブ、バックグラウンドの委任を実行中の場合、コマンドはそれらを一覧表示し、中断する前に `y/N` で確認します。`--yes` はこの確認をスキップします。確認できるターミナルがない場合（スクリプト、cron）、`--yes` が渡されない限り拒否します。`--interrupt` はエイリアスです。 |
| `--detach` | `restart` で: ゲートウェイサービスがインストールされていない場合、置き換えのゲートウェイを現在のターミナルではなくバックグラウンドで起動します（`hermes gateway ensure` が起動するのと同じ方法）。デスクトップアプリとアップデーターのリカバリー再起動はこれを渡すため、短命な再起動呼び出しがゲートウェイそのものになることはありません。サービス、スーパーバイザー、Windowsタスクの再起動は変わりません。 |
| `--no-supervise` | `run` で: s6-overlayのDockerイメージ内で自動監視をオプトアウトし、s6以前のフォアグラウンドのセマンティクスを使用します — ゲートウェイはコンテナのメインプロセスとして実行され、自動再起動はありません。s6イメージ以外では何もしません。`HERMES_GATEWAY_NO_SUPERVISE=1` を設定するのと同等です。 |
| `--external-supervisor` | `run` で: ラッパーが提供するプロセスマネージャーがフォアグラウンドのゲートウェイを管理していることを宣言します。`sudo`、`env -i`、その他のラッパーがlaunchd/systemdのネイティブな環境マーカーを取り除いてしまう場合に使用します。チャット内での再起動と更新は、切り離された置き換えを生成する代わりに、そのマネージャーに戻るように終了します。 |

`--external-supervisor` は再起動ポリシーの契約です: チャット内での再起動、
`hermes gateway restart`、またはサービス再起動を伴う更新は、ステータス `75` で終了します
（その後CLIは、自前のフォアグラウンドゲートウェイを実行する代わりに、スーパーバイザーの
新しいPIDを待ちます）。そのため、ラッパーのスーパーバイザーは、その非ゼロ終了の後に
ゲートウェイを再起動する必要があります。systemdでは `Restart=on-failure` または
`Restart=always` を使用し、`RestartPreventExitStatus` に `75` を含めないでください。
launchdでは、失敗した終了の後に再起動するよう `KeepAlive` を設定してください。
そのポリシーがない場合、要求された再起動によってゲートウェイは停止したままになります。

`hermes gateway enroll` は `--token`、`--connector-url`、`--gateway-id`、`--wake-url` を受け付けます。登録トークンをコネクターと交換し、得られた `GATEWAY_RELAY_ID`、`GATEWAY_RELAY_SECRET`、`GATEWAY_RELAY_DELIVERY_KEY`、オプションの `GATEWAY_RELAY_URL`、および（`--wake-url` が指定された場合）`GATEWAY_RELAY_WAKE_URL` の値を、アクティブなプロファイルの `.env` に書き込みます。

:::tip WSLユーザー向け
`hermes gateway start` の代わりに `hermes gateway run` を使用してください — WSLのsystemdサポートは信頼性に欠けます。永続化のためにtmuxでラップしてください: `tmux new -s hermes 'hermes gateway run'`。詳細は[WSL FAQ](./faq.md#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails)を参照してください。
:::

### ローカルランタイムの検出と起動 {#local-runtime-discovery-and-startup}

`hermes gateway ensure [--json] [--timeout 60]` は、アクティブなプロファイルのセッションランタイムを
検出または起動し、1つのJSONオブジェクトを出力します（`state`、`reason_code`、および認証済みの検出で
互換性のある準備完了が報告された場合のみ `endpoint`。チケットや認証情報は決して含みません）。
既存のオーナーや起動中のオーナーを置き換えるのではなく、その準備を待ち、サービスをインストールすることは
ありません。終了コード `0` は準備完了を意味します。それ以外のコードは拒否または期限切れであり、置き換えを
起動してよいという意味では決してありません。サービスの同一性チェック、`unsafe_control_permissions` の
背後にあるホームディレクトリのパーミッションポリシー、およびすべての終了コードについては:
[ゲートウェイ内部: `hermes gateway ensure`](/docs/developer-guide/gateway-internals#hermes-gateway-ensure)。

`hermes gateway attach [--json] [--start] [--recover] [--timeout 60]` は、ゲートウェイに自ら接続する
ローカルクライアント（例えばサードパーティのチャットUI）向けのアタッチ手順です。`ensure` のJSONに加えて、
ゲートウェイがアタッチ可能になると、`url`（`ws://127.0.0.1:<port>/api/ws`）、`protocol`（`hermes-gateway-v1`）、
`protocols`（接続時に使うWebSocketサブプロトコル）、`ticket` を出力します。チケットは使い捨てで、アクティブな
プロファイルに紐付けられ、発行から30秒で失効します。チケットは呼び出し元プロセス向けにのみstdoutに出力され、
ログに記録されることはありません。オプションなしの呼び出しはゲートウェイを検出するだけです。`--start` は
プロファイルを提供するゲートウェイがない場合に起動します。`--recover` はまず検出を行い、準備完了でない場合に
のみ起動します（または古いHermesのゲートウェイを置き換えます）。`attach` が起動した管理外のゲートウェイ
（サービス未インストール）は、しばらくアイドル状態が続くと自ら停止します。
アタッチするゲートウェイがない場合、
JSONには `message`（`gateway <state>: <reason>` の行。対処法がある場合はそれも含む）が含まれ、同じテキストが
stderrに出力され、終了コードはその判定に対する `ensure` のものになります。Ink TUIも同じコードを通じて
アタッチします。詳細と接続ルール:
[ゲートウェイ内部: `hermes gateway attach`](/docs/developer-guide/gateway-internals#hermes-gateway-attach)。

## `hermes lsp`

```bash
hermes lsp <subcommand>
```

Language Server Protocol統合を管理します。LSPは実際の
言語サーバー（pyright、gopls、rust-analyzer、…）をバックグラウンドで
実行し、その診断結果を `write_file` と `patch` が使用する書き込み後の
チェックに渡します。gitワークスペースの検出によって制御されており
— LSPはcwdまたは編集対象のファイルがgit worktree内にある場合にのみ
実行されます。

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `status` | サービスの状態、設定済みのサーバー、インストール状況を表示します。 |
| `list` | サポートされているサーバーのレジストリを出力します。未インストールのものを除外するには `--installed-only` を渡します。 |
| `install <id>` | 1つのサーバーのバイナリを先行してインストールします。 |
| `install-all` | 既知の自動インストール手順があるすべてのサーバーをインストールします。 |
| `restart` | 実行中のクライアントを終了し、次の編集時に再起動されるようにします。 |
| `which <id>` | 1つのサーバーについて、解決されたバイナリのパスを出力します。 |

完全なガイド、サポートされている言語、設定項目については
[LSP — セマンティック診断](/docs/user-guide/features/lsp)を参照してください。

## `hermes setup`

```bash
hermes setup [model|tts|terminal|gateway|tools|agent] [--non-interactive] [--reset] [--quick] [--reconfigure] [--portal]
```

**最も簡単な方法:** `hermes setup --portal` — Nous PortalへのOAuthログインと[Tool Gateway](../user-guide/features/tool-gateway.md)へのオプトインを一度に行います。

**初回実行時:** 初回セットアップウィザードを起動します。

**再訪ユーザー（設定済み）:** 完全な再設定ウィザードに直接入ります — 各プロンプトには現在の値がデフォルトとして表示され、Enterを押すと保持、新しい値を入力すると変更します。メニューはありません。

**別のエージェントから移行する場合:** 初回実行時にOpenClaw、Claude Code、Codex CLI、Cursorのセットアップ（`~/.openclaw`、`~/.claude`、`~/.codex`、`~/.cursor`）が見つかると、まず **「Import your setup from …?」** と尋ねます（デフォルトはyes）。複数見つかった場合は、続いてチェックリストからどれをインポートするかを選びます。セットアップは、まだ導入されていなければカタログから[`agent-importers`](/docs/user-guide/import-from-other-agents)プラグインをインストールし、選択ごとにそのインポートを実行します。インポートはプレビューを表示し、何かを変更する前に確認を求めます（以前の組み込みステップと同様に、OpenClawの許可リストに含まれるAPIキーとボットトークンも含まれます）。モデルとそのキーを取り込んだインポートの後のクイックセットアップでは、**「Keep your imported model (…) or switch to Nous Portal?」** と尋ねます（デフォルトはkeepで、Nous Portalへのログインをスキップします）。何も見つからない場合、noと答えた場合、`--non-interactive` の場合、またはダッシュボードから開始されたセットアップの場合: 質問もインストールも行われません。すでに設定済みのインストールでは、代わりに `hermes import-agent` を実行してください。

完全なウィザードの代わりに、1つのセクションに直接入ります:

| セクション | 説明 |
|---------|-------------|
| `model` | プロバイダーとモデルのセットアップ。 |
| `terminal` | ターミナルバックエンドとサンドボックスのセットアップ。 |
| `gateway` | メッセージングプラットフォームのセットアップ。 |
| `tools` | プラットフォームごとのツールの有効化/無効化。 |
| `agent` | エージェントの動作設定。 |

オプション:

| オプション | 説明 |
|--------|-------------|
| `--quick` | 再訪ユーザーの実行時: 欠落または未設定の項目のみを尋ねます。すでに設定済みの項目はスキップします。 |
| `--non-interactive` | プロンプトなしでデフォルト / 環境値を使用します。 |
| `--reset` | セットアップ前に設定をデフォルトにリセットします。 |
| `--reconfigure` | 後方互換性のエイリアス — 既存のインストールで素の `hermes setup` を実行すると、現在はデフォルトでこの動作になります。 |
| `--portal` | ワンショットのNous Portalセットアップ: OAuthでログインし、Nousを推論プロバイダーに設定し、[Tool Gateway](../user-guide/features/tool-gateway.md)にオプトインします。ウィザードの残りはスキップします。 |

## `hermes portal`

```bash
hermes portal [status|open|tools]
```

Nous Portalの認証とTool Gatewayのルーティングを確認し、サブスクリプションページを開きます。サブコマンドなしで呼び出すと `status` を実行します。

| サブコマンド | 説明 |
|------------|-------------|
| `status`（デフォルト） | Portalの認証状態 + ツールごとのTool Gatewayルーティングの要約。サブコマンドを指定しない場合にも表示されます。 |
| `open` | デフォルトのブラウザで `portal.nousresearch.com/manage-subscription` を開きます。 |
| `tools` | すべてのTool Gatewayパートナー（Firecrawl、FAL、OpenAI TTS、Browser Use、Modal）と、そのうちNous経由でルーティングされているものを一覧表示します。 |

ゲートウェイ自体の設定については[Tool Gateway](../user-guide/features/tool-gateway.md)を参照してください。ワンショットのセットアップ方法については、上記の `hermes setup --portal` を参照してください。

## `hermes whatsapp`

```bash
hermes whatsapp
```

モード選択とQRコードのペアリングを含む、WhatsAppのペアリング/セットアップフローを実行します。

## `hermes slack`

```bash
hermes slack manifest              # print manifest to stdout
hermes slack manifest --write      # write to ~/.hermes/slack-manifest.json
hermes slack manifest --long-description-file AGENTS.md --write
hermes slack manifest --slashes-only  # just the features.slash_commands array
```

`COMMAND_REGISTRY`（`/btw`、`/stop`、`/model`、…）のすべてのゲートウェイコマンドを
ファーストクラスのSlackスラッシュコマンドとして登録するSlackアプリマニフェストを
生成します — DiscordおよびTelegramとの一貫性に合わせています。出力を
[https://api.slack.com/apps](https://api.slack.com/apps) → あなたのアプリ →
**Features → App Manifest → Edit** のSlackアプリ設定に貼り付け、**Save** してください。
スコープまたはスラッシュコマンドが変更された場合、Slackは再インストールを求めます。

| フラグ | デフォルト | 目的 |
|------|---------|---------|
| `--write [PATH]` | stdout | stdoutの代わりにファイルに書き込みます。素の `--write` は `$HERMES_HOME/slack-manifest.json` に書き込みます。 |
| `--name NAME` | `Hermes` | Slackでのボット表示名。 |
| `--description DESC` | デフォルトの説明文 | Slackアプリディレクトリに表示されるボットの説明。 |
| `--long-description TEXT` | 未設定 | `display_information.long_description` をインラインで設定します（175〜4,000文字）。`--slashes-only` とは併用できません。 |
| `--long-description-file PATH` | 未設定 | UTF-8テキストファイルから長い説明を読み取り、その内容をそのまま保持します。`--long-description` とは同時に指定できず、`--slashes-only` とも併用できません。 |
| `--slashes-only` | off | 手動でメンテナンスするマニフェストにマージするため、`features.slash_commands` のみを出力します。 |

`hermes update` の後、新しいコマンドを取り込むには `hermes slack manifest --write` を
再度実行してください。


## `hermes send`

```bash
hermes send --to <target> "message text"
hermes send --to <target> --file <path>
echo "message" | hermes send --to <target>
hermes send --list [platform]
```

エージェントやゲートウェイのループを起動せずに、設定済みのメッセージングプラットフォームにワンショットのメッセージを送信します。ゲートウェイですでに設定済みの認証情報（`~/.hermes/.env` + `~/.hermes/config.yaml`）を再利用するため、運用スクリプト、cronジョブ、CIフック、監視デーモンは、各プラットフォームのRESTクライアントを再実装することなくステータス更新を投稿できます。

ボットトークン方式のプラットフォーム（Telegram、Discord、Slack、Signal、SMS、WhatsApp-CloudAPI）では、実行中のゲートウェイは不要です — `hermes send` はプラットフォームのRESTエンドポイントと直接通信します。永続的なアダプターを必要とするプラグインプラットフォームでは、引き続き稼働中のゲートウェイが必要です。

| オプション | 説明 |
|--------|-------------|
| `-t`, `--to <TARGET>` | 配信ターゲット。形式: `platform`（ホームチャンネルを使用）、`platform:chat_id`、`platform:chat_id:thread_id`、または `platform:#channel-name`。例: `telegram`、`telegram:-1001234567890`、`discord:#ops`、`slack:C0123ABCD`、`signal:+15551234567`。 |
| `-f`, `--file <PATH>` | `PATH` からメッセージ本文を読み取ります（テキストファイルのみ — ログ、レポート、Markdown）。stdinからの読み取りを強制するには `-` を渡します。画像やその他のバイナリファイルを送信するには `MEDIA:<path>` を使用してください（下記参照）。 |
| `-s`, `--subject <LINE>` | メッセージ本文の前に件名/ヘッダー行を付加します。 |
| `-l`, `--list [platform]` | すべてのプラットフォーム（または指定したプラットフォームのみ）の設定済みターゲットを一覧表示します。 |
| `-q`, `--quiet` | 成功時のstdout出力を抑制します — スクリプトで便利です（終了コードのみに依存）。 |
| `--json` | 人間が読める出力の代わりに、生のJSON結果を出力します。 |

位置引数の `message` も `--file` も指定されない場合、stdinがTTYでなければ `hermes send` はstdinから読み取ります。終了コード: 成功時 `0`、配信/バックエンドの失敗時 `1`、使用法エラー時 `2`。

### 画像やその他のメディアの送信 {#sending-images-and-other-media}

`--file` は*テキスト*本文専用です。画像、ドキュメント、動画、音声ファイルをプラットフォームネイティブの添付ファイルとして配信するには、メッセージテキスト内で `MEDIA:<local_path>` ディレクティブを使って参照してください:

```bash
hermes send --to telegram "MEDIA:~/.hermes/cache/scratch/screenshot.png"
hermes send --to telegram "Build chart for today MEDIA:~/.hermes/cache/scratch/chart.png"   # with caption
hermes send --to discord:#ops "MEDIA:~/.hermes/cache/scratch/report.pdf"
```

デフォルトでは、画像ファイルは写真として送信されます（Telegramなどのプラットフォームはこれを再圧縮します）。代わりに非圧縮のファイル添付として配信するには、メッセージに `[[as_document]]` を追加してください:

```bash
hermes send --to telegram "[[as_document]] MEDIA:~/.hermes/cache/scratch/screenshot.png"
```

例:

```bash
hermes send --to telegram "deploy finished"
echo "RAM 92%" | hermes send --to telegram:-1001234567890
hermes send --to discord:#ops --file ~/.hermes/cache/scratch/report.md
hermes send --to slack:#eng --subject "[CI]" --file build.log
hermes send --list                  # all platforms
hermes send --list telegram         # filter by platform
```



## `hermes peer`

```bash
hermes peer add <name> --url http://host:port --key <API_SERVER_KEY>
hermes peer list
hermes peer dm <peer>[/<agent>] "message"
hermes peer run <peer>[/<agent>] --idempotency-key <key> "message"
hermes peer status <peer>[/<agent>] <run_id>
hermes peer stop <peer>[/<agent>] <run_id>
hermes peer remove <name>
```

マシン間のボット同士のDM。別のHermesゲートウェイ（`api_server` プラットフォームを実行している任意のマシン）を
*ピア*として登録し、そのエージェントにメッセージを送ります:
`hermes peer dm` はピアのAPIサーバー経由でリモートエージェントの正規の **Bot Chat** セッションを解決し、
そこでエージェントのターンを1回実行して、返信をstdoutに出力します — ローカルの
`hermes -p <bot> chat --in ~ -c "Bot Chat" …` によるボットメッセージングコマンドの、マシン間版です。

`<peer>` のみの場合はピアゲートウェイのメインエージェントを対象とします。
`<peer>/<agent>` はマルチプレックスされたピア上の名前付きプロファイルを対象とします
（その `/p/<profile>/` ミラー経由でルーティングされます）。

| サブコマンド | 説明 |
|--------|-------------|
| `add <name> --url <URL> [--key <KEY>] [--note TEXT]` | ピアを登録または更新します。URLは `config.yaml`（`bot_peers`）に、キーは `~/.hermes/.env` に `HERMES_PEER_<NAME>_KEY` として保存されます。 |
| `list` | ピアと、それぞれにキーが設定されているかどうかを一覧表示します。 |
| `dm <peer>[/<agent>] [message]` | ピアエージェントの正規のBot Chatにメッセージを送り、返信を出力します（機械可読出力には `--json`。メッセージを省略するとstdinから読み取ります）。 |
| `run <peer>[/<agent>] [message]` | 正規のBot Chatでの長いターンを非同期に開始し、その `run_id`、セッションID、冪等性キーを返します（`--json` 対応）。同じリクエストを再試行する際は `--idempotency-key` を再利用してください。 |
| `status <peer>[/<agent>] <run_id>` | 非同期のピア実行をポーリングし、完了時に最終出力を表示します（`--json` 対応）。 |
| `stop <peer>[/<agent>] <run_id>` | 他のターンを対象にすることなく、その非同期のピア実行だけを停止します（`--json` 対応）。 |
| `remove <name>` | レジストリからピアを削除します（`.env` のキーエントリーはそのまま残ります）。 |

少なくとも1つのピアが登録されている場合、すべての正規のBot Chatに教えられるBot Modeのメッセージングプロトコル
（`agent.bot_mode_protocol`）には、ピアの一覧と `hermes peer dm` のパターンが自動的に含まれるため、
エージェントはSOULを編集しなくてもマシンをまたいだチームメイトを発見できます。
[Bot Mode](/docs/user-guide/bot-mode)を参照してください。

終了コード: 成功時 `0`、配信/ピアの失敗時 `1`、使用法エラー時 `2`。

## `hermes secrets`

```bash
hermes secrets bitwarden <subcommand>
hermes secrets bw <subcommand>          # short alias
```

APIキーを `~/.hermes/.env` に保存する代わりに、プロセス起動時に外部のシークレットマネージャーから取得します。現在は **Bitwarden Secrets Manager** をサポートしています。完全なガイドは[Bitwarden統合](/docs/user-guide/secrets/bitwarden)を参照してください。

`bitwarden`（エイリアス `bw`）のサブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `setup` | 対話型ウィザード: ピン留めされた `bws` バイナリをインストールし、アクセストークンを保存し、プロジェクトを選択します。非対話型で使うために `--project-id`、`--access-token`、`--server-url` を受け付けます。 |
| `status` | 現在の設定、バイナリのパス/バージョン、トークンの検証状況を表示します。 |
| `token` | アクセストークンをローテーションします: 新しいトークンを `.env` に保存する前にBitwardenに対して検証します（拒否されたトークンでは何も変更されません）。非対話型で使うための `--access-token` と、検証をスキップする `--no-verify` を受け付けます。 |
| `sync` | 今すぐシークレットを取得し、変更内容を報告します。実際にシークレットを現在のシェルの環境にエクスポートするには `--apply` を追加します（デフォルトはドライラン）。 |
| `install` | PMでピン留めされた `bws` バイナリをインストールまたは修復します。`--force` は無条件のダウンロードではなく、同じ整合性チェックと修復を要求します。 |
| `disable` | Bitwarden統合を無効にします。 |


## `hermes migrate`

```bash
hermes migrate                 # find another agent's setup here and offer to import it
hermes migrate <agent>         # openclaw, claude-code, codex, cursor, grok-bot or openbot
hermes migrate <type>          # xai, relay
```

素の `hermes migrate` は、このコンピューター上のOpenClaw、Claude Code、Codex CLI、Cursor、Grok Bot、OpenBotを探し、見つかったものをインポートすることを提案します。これは初回の `hermes setup` が実行するのと同じステップです（ただしセットアップ独自の選択は含みません: OpenClawのキーは `hermes migrate openclaw --migrate-secrets` の場合にのみ移行されます）。`hermes migrate <agent>` は、何も検出されなくてもそのエージェントからインポートします: [`agent-importers`](/docs/user-guide/import-from-other-agents)プラグインがなければカタログからインストールし、続いて `hermes import-agent <agent>` を実行します（プレビューの後に確認）。`--source`、`--dry-run`、`--overwrite`、`-y` を受け付けます。`grok-bot` と `openbot` は `--activate-routines` と `--bot <name>` も受け付けます。`openclaw` は `--migrate-secrets` を受け付けます（APIキーとボットトークンはこれを指定した場合にのみ移行されます。これを渡す初回セットアップとは異なります）。

`xai` と `relay` のタイプは、アクティブな `config.yaml` を診断し、（必要に応じて）廃止されたモデルや非推奨の設定への参照を置き換えるよう書き換えます。書き換えの前に、元の `config.yaml` のタイムスタンプ付きバックアップが取得されます（`--no-backup` でスキップ）。

| サブコマンド | 説明 |
|------------|-------------|
| `openclaw`, `claude-code`, `codex`, `cursor`, `grok-bot`, `openbot` | そのエージェントのセットアップをインポートします（上記参照）。 |
| `xai` | 2026年5月15日に廃止予定のxAIモデルへの参照を `config.yaml` から探し、（`--apply` を指定すると）xAIの移行ガイドに従って公式の代替モデルにその場で書き換えます。デフォルトはドライランです。 |

移行サブコマンドの共通フラグ:

| フラグ | 説明 |
|------|-------------|
| `--apply` | `config.yaml` をその場で書き換えます（デフォルト: ドライランで、書き込みなし）。 |
| `--no-backup` | 適用時に `config.yaml` のタイムスタンプ付きバックアップをスキップします。 |



## `hermes codex-runtime`

```bash
hermes codex-runtime migrate [--dry-run] [--json]
```

`/codex-runtime codex_app_server` がトリガーする `~/.codex/config.toml` の移行を、チャットセッションなしで実行します: Hermesの `mcp_servers`（およびインストール済みのcodexプラグインと `default_permissions` のデフォルト）が、選択したプロファイルの管理ブロックに反映されます（`hermes -p <name> codex-runtime migrate`）。ブロック外のユーザーのテキストはそのまま保持されます。Hermesのサーバーと同じ名前のユーザー所有の `[mcp_servers.<name>]` は保持され、その名前に対するHermesの反映はスキップされます（`preserved_user_servers` として報告されます）。結果はアトミックな書き込みの前にTOMLとして検証されます。レポートにエラーが含まれる場合、終了コードは1です。

| フラグ | 説明 |
|------|-------------|
| `--dry-run` | `config.toml` に書き込まずに、移行内容を計算して報告します。 |
| `--json` | 完全な移行レポートをJSONとして出力します（`migrated`、`preserved_user_servers`、`skipped_keys_per_server`、`errors`、`target_path`、`written`）。 |

## `hermes proxy`

```bash
hermes proxy <subcommand>
```

OAuth認証済みの上流プロバイダー（例: Nous Portal、xAI）にリクエストを転送する、ローカルのOpenAI互換HTTPサーバーを実行します。外部アプリは任意のBearerトークンでプロキシを指定でき、プロキシは送信時に実際のOAuth認証情報を付与します。完全なガイドは[サブスクリプションプロキシ](/docs/user-guide/features/subscription-proxy)を参照してください。

| サブコマンド | 説明 |
|------------|-------------|
| `start` | プロキシをフォアグラウンドで実行します。フラグ: `--provider <nous\|xai>`（デフォルト `nous`）、`--host <addr>`（デフォルト `127.0.0.1`。LANに公開するには `0.0.0.0` を使用）、`--port <int>`（デフォルト `8645`）。 |
| `status` | どのプロキシ上流が準備完了か（認証情報があり、OAuthが有効か）を表示します。 |
| `providers` | 利用可能なプロキシ上流プロバイダーを一覧表示します。 |


## `hermes security`

```bash
hermes security <subcommand>
```

[OSV.dev](https://osv.dev) に対するオンデマンドの脆弱性スキャン。Hermesのvenv（インストール済みのPyPIディストリビューション）、`~/.hermes/plugins/` 配下のプラグインが宣言するPython依存関係、`config.yaml` 内のピン留めされた `npx`/`uvx` MCPサーバーを対象とします。グローバルにインストールされたパッケージや、エディター/ブラウザーの拡張機能はスキャンしません。

| サブコマンド | 説明 |
|------------|-------------|
| `audit` | ワンショットのサプライチェーン監査を実行します。 |

`audit` のフラグ:

| フラグ | デフォルト | 説明 |
|------|---------|-------------|
| `--json` | off | 人間が読めるテキストの代わりに、機械可読のJSONを出力します。 |
| `--fail-on <level>` | `critical` | この重大度（`low`、`moderate`、`high`、`critical`）以上の検出結果がある場合に非ゼロで終了します。 |
| `--skip-venv` | off | HermesのPython venvのスキャンをスキップします。 |
| `--skip-plugins` | off | プラグインの要件ファイルのスキャンをスキップします。 |
| `--skip-mcp` | off | `config.yaml` 内のピン留めされたMCPサーバーのスキャンをスキップします。 |


## `hermes login` / `hermes logout` *(非推奨)*

:::caution
`hermes login` は削除されました。OAuth認証情報の管理には `hermes auth` を、プロバイダーの選択には `hermes model` を、完全な対話型セットアップには `hermes setup` を使用してください。
:::

## `hermes auth`

同一プロバイダーのキーローテーション向けに認証情報プールを管理します。完全なドキュメントは[認証情報プール](../user-guide/features/credential-pools.md)を参照してください。

```bash
hermes auth                                              # Interactive wizard
hermes auth list                                         # Show all pools
hermes auth list openrouter                              # Show specific provider
hermes auth add openrouter --api-key sk-or-v1-xxx        # Add API key
hermes auth add openrouter --type oauth                  # Browser login (OpenRouter PKCE) mints a key for you
hermes auth add anthropic --type oauth                   # Add OAuth credential
hermes auth add openai-codex --type oauth --priority 0   # Add an account and try it first
hermes auth add openai-codex --browser                   # Codex: browser auth-code + PKCE on localhost:1455 instead of device code
hermes auth remove openrouter 2                          # Remove by index
hermes auth priority openrouter backup-key 0             # Move a credential to the front of fill_first order
hermes auth reset openrouter                             # Clear cooldowns
hermes auth reset openrouter 2                           # Clear the cooldown on one credential
hermes auth refresh openai-codex work                    # Refresh one OAuth credential and clear its cooldown
hermes auth status anthropic                             # Show auth status for a provider
hermes auth logout anthropic                             # Log out and clear stored auth state
```

サブコマンド: `add`、`list`、`remove`、`reset`、`priority`、`refresh`、`status`、`logout`。Spotifyへのログインは、`spotify` カタログプラグインの `hermes spotify login` に移動しました。サブコマンドなしで呼び出すと、対話型の管理ウィザードを起動します。

## `hermes usage`

`/usage` スラッシュコマンドのアカウント制限ブロック — Codexの5時間 / 週間ウィンドウ、プランと繰り越された
リセット、AnthropicのOAuthウィンドウ、OpenRouterのクレジット — を、セッションを開始せずに表示します。
シェルスクリプトやcronジョブから読み取れます。

```bash
hermes usage                          # configured model provider, human-readable block
hermes usage --provider openai-codex  # a specific provider
hermes usage --json                   # one JSON document on stdout
```

| オプション | 説明 |
|--------|-------------|
| `--provider NAME` | 照会するプロバイダー（デフォルト: 設定済みの `model.provider`）。サポート対象: `openai-codex`、`anthropic`、`openrouter`。 |
| `--json` | 人間が読めるブロックの代わりに、1つのJSONドキュメントを出力します。 |

認証情報は、稼働中のエージェントがないセッションでの `/usage` とまったく同じ方法で解決されます（authストア、次に
認証情報プール）。このコマンドが、チャットで使わない認証情報を追加・更新することはありません。終了コードは
成功時 `0`。プロバイダーに認証情報が設定されていない、プロバイダーに使用量エンドポイントがない、または取得に
失敗した場合は、stderrに1行を出力して `1` になります（stdoutは空のままです）。

`--json` のスキーマ（キーは安定しています。新しいキーが追加される場合があります）:

```json
{
  "provider": "openai-codex",
  "source": "usage_api",
  "title": "Account limits",
  "plan": "Plus",
  "fetched_at": "2026-09-19T07:58:55+00:00",
  "windows": [
    {"label": "Session", "used_percent": 37.0, "resets_at": "2026-09-19T21:00:00+00:00", "detail": null},
    {"label": "Weekly", "used_percent": 12.5, "resets_at": "2026-09-25T09:00:00+00:00", "detail": null}
  ],
  "details": ["You have 1 reset banked - use /usage reset to activate"],
  "unavailable_reason": null
}
```

プロバイダーがウィンドウを報告しなかった場合、`used_percent` は `null` です。`resets_at` はISO-8601のUTCまたは `null` です
（ウィンドウによっては代わりに自由記述の `detail` を持ちます）。`plan` は不明な場合 `null` です。

## `hermes status`

```bash
hermes status [--full] [--deep]
```

デフォルトでは1画面分の要約を出力します: モデル、アクティブなプロバイダー、認証情報を持つすべてのプロバイダー
（`/model` ピッカーが提示するのと同じリスト）、ゲートウェイの状態、ゲートウェイが起動するメッセージング
プラットフォーム、スケジュール済みジョブ。キーの値は出力されません。

| オプション | 説明 |
|--------|-------------|
| `--full` | すべてのセクション（秘匿化されたAPIキー、認証プロバイダー、ターミナルバックエンド、セッション、...）を出力します。`--all` はエイリアスです。 |
| `--deep` | 時間がかかる可能性のある、より深いチェックを実行します。`--full` を含意します。 |

## `hermes cron`

```bash
hermes cron <list|create|edit|pause|resume|run|remove|status|runs|incidents|doctor|tick>
```

| サブコマンド | 説明 |
|------------|-------------|
| `list` | スケジュール済みジョブを表示します。 |
| `create` / `add` | プロンプトからスケジュール済みジョブを作成し、必要に応じて繰り返しの `--skill` で1つ以上のスキルをアタッチします。`--reasoning-effort <none\|minimal\|low\|medium\|high\|xhigh\|max\|ultra>` によるジョブごとの推論レベルの固定をサポートします。 |
| `edit` | ジョブのスケジュール、プロンプト、名前、配信、繰り返し回数、アタッチされたスキルを更新します。`--clear-skills`、`--add-skill`、`--remove-skill`、および `--reasoning-effort`（空文字列で固定を解除）をサポートします。 |
| `pause` | ジョブを削除せずに一時停止します。 |
| `resume` | 一時停止したジョブを再開します。一時停止中に実行時刻を迎えた繰り返しのスロットは実行予定のまま残ります（次のティックで1回のキャッチアップ実行、またはログに記録されたスキップ）。それ以外の場合は、次回の将来の実行が計算されます。 |
| `run` | 次のスケジューラーのティックでジョブをトリガーします。 |
| `remove` | スケジュール済みジョブを削除します。 |
| `status` | cronスケジューラーが実行中かどうかを確認します。 |
| `doctor` | 読み取り専用のフリート健全性チェック: 失敗した実行、失敗した配信、期限切れ/欠落した `next_run_at`、存在しないスクリプトや作業ディレクトリ。問題が見つかった場合は非ゼロで終了します。 |
| `tick` | 実行予定のジョブを1回実行して終了します。 |

cronの**トリガー**は、`cron.provider` 設定キーによって差し替え可能です。空
（デフォルト）の場合は、組み込みのプロセス内ティッカーを使用します。`chronos`（スケール・トゥ・ゼロの
ホスト型ゲートウェイ向けの、NAS管理のプロバイダー）に設定する — `cron.chronos.*` キー（`portal_url`、
`callback_url`、`expected_audience`、`nas_jwks_url`）で設定します — か、`plugins/cron/<name>/` または
`$HERMES_HOME/plugins/<name>/` 配下のカスタムプロバイダーを指定します。不明または利用できないプロバイダーは
組み込みにフォールバックするため、cronがトリガーなしの状態になることはありません。
[cron内部](/docs/developer-guide/cron-internals#gateway-integration)のドキュメントを参照してください。

## `hermes kanban`

```bash
hermes kanban [--board <slug>] <action> [options]
```

マルチプロファイル、マルチプロジェクトのコラボレーションボード。各インストールは多数のボード（プロジェクト、リポジトリ、ドメインごとに1つ）をホストできます。各ボードは独自のSQLite DBとディスパッチャースコープを持つスタンドアロンのキューです。新規インストールは `default` という1つのボードで開始し、そのDBは後方互換性のため `~/.hermes/kanban.db` です。追加のボードは `~/.hermes/kanban/boards/<slug>/kanban.db` に置かれます。ゲートウェイに組み込まれたディスパッチャーは、ティックごとにすべてのボードをスイープします。

**グローバルフラグ（以下のすべてのアクションに適用）:**

| フラグ | 目的 |
|------|---------|
| `--board <slug>` | 特定のボードを操作します。デフォルトは現在のボード（`hermes kanban boards switch`、`HERMES_KANBAN_BOARD` 環境変数、または `default` で設定）です。 |

**これは人間 / スクリプト向けの操作面です。** ディスパッチャーによって生成されたエージェントワーカーは、`hermes kanban` をシェル実行する代わりに、専用の `kanban_*` [ツールセット](../user-guide/features/kanban.md#how-workers-interact-with-the-board)（`kanban_show`、`kanban_complete`、`kanban_request_review`、`kanban_request_changes`、`kanban_block`、`kanban_create`、`kanban_link`、`kanban_comment`、`kanban_heartbeat`。オーケストレーターのプロファイルには `kanban_list` と `kanban_unblock` も提供されます）を通じてボードを操作します。ワーカーは環境に `HERMES_KANBAN_BOARD` が固定されているため、物理的に他のボードを見ることができません。

| アクション | 目的 |
|--------|---------|
| `init` | `kanban.db` が存在しない場合に作成します。冪等です。 |
| `boards list` / `boards ls` | タスク数とともにすべてのボードを一覧表示します。`--json`、`--all`（アーカイブ済みを含む）。 |
| `boards create <slug>` | 新しいボードを作成します。フラグ: `--name`、`--description`、`--icon`、`--color`、`--switch`（アクティブにする）。スラッグはケバブケースで、自動的に小文字化されます。 |
| `boards switch <slug>` / `boards use` | `<slug>` をアクティブなボードとして永続化します（`~/.hermes/kanban/current` に書き込みます）。 |
| `boards show` / `boards current` | 現在アクティブなボードの名前、DBパス、タスク数を出力します。 |
| `boards rename <slug> "<name>"` | ボードの表示名を変更します。スラッグは不変です。 |
| `boards rm <slug>` | ボードをアーカイブ（デフォルト）またはハード削除します。`--delete` はアーカイブステップをスキップします。アーカイブされたボードは `boards/_archived/<slug>-<ts>/` に移動します。`default` に対しては拒否されます。 |
| `create "<title>"` | アクティブなボードに新しいタスクを作成します。フラグ: `--body`、`--assignee`、`--parent`（繰り返し可）、`--workspace scratch\|worktree\|dir:<path>`、`--project <id-or-slug>`（プロジェクトにリンク。プロジェクトにプライマリフォルダーがある場合、デフォルトでその配下のworktree + ブランチを使用）、`--tenant`、`--priority`、`--triage`、`--idempotency-key`、`--max-runtime`、`--max-retries`、`--skill`（繰り返し可）。 |
| `list` / `ls` | アクティブなボードのタスクを一覧表示します。`--mine`、`--assignee`、`--status`、`--tenant`、`--archived`、`--json` でフィルタリングします。 |
| `show <id>` | コメントとイベントを含むタスクを表示します。機械可読出力には `--json`。 |
| `assign <id> <profile>` | 割り当てまたは再割り当てします。割り当て解除には `none` を使用します。タスクの実行中は拒否されます。 |
| `link <parent> <child>` | 依存関係を追加します。循環は検出されます。両方のタスクは同じボードにある必要があります。 |
| `unlink <parent> <child>` | 依存関係を削除します。 |
| `claim <id>` | 準備完了のタスクをアトミックにクレームします。解決されたワークスペースパスを出力します。 |
| `comment <id> "<text>"` | コメントを追加します。次にタスクをクレームしたワーカーが、`kanban_show()` のレスポンスの一部としてそれを読み取ります。 |
| `complete <id>` | タスクを完了としてマークします。フラグ: `--result`、`--summary`、`--metadata`。 |
| `block <id> "<reason>"` | 人間の入力待ちとしてタスクをブロック済みにマークします。理由もコメントとして追加します。 |
| `request-review <id>` | レビュアーへの引き継ぎとともにタスクを `review` に移動します — ブロックではありません。フラグ: `--summary`、`--metadata`、`--reviewer`（レビューのディスパッチ前に再割り当て）。 |
| `request-changes <id> <reason>` | アクティブなレビュー実行に対するレビュアーの判定: レビューの試行を閉じ、タスクを元の実装者に差し戻します。 |
| `reopen-review <id>...` | レビュー中のタスクを修正のために差し戻します（`review` → ready/todo）。フラグ: `--reason`（コメントとして追加）。 |
| `schedule <id> "<reason>"` | 時間待ち/フォローアップの作業を `scheduled` に置き、人間によるブロックとして表示されないようにします。 |
| `unblock <id>` | ブロックされたタスクを元のフェーズ（`review` または `ready`）に戻します。依存関係が未解決の場合は `todo` に戻します。 |
| `archive <id>` | デフォルトのリストから非表示にします。`gc` がscratchワークスペースを削除します。 |
| `tail <id>` | タスクのイベントストリームを追跡します。 |
| `dispatch` | アクティブなボードに対するディスパッチャーの1パス。フラグ: `--dry-run`、`--max N`、`--failure-limit N`、`--json`。 |
| `context <id>` | ワーカーが見る完全なコンテキスト（タイトル + 本文 + 親の結果 + コメント）を出力します。 |
| `specify <id>` / `specify --all` | トリアージ列のタスクを、補助LLMを介して具体的な仕様（タイトル + 目標、アプローチ、受け入れ基準を含む本文）に肉付けし、`todo` に昇格させます。フラグ: `--tenant`（`--all` を1つのテナントにスコープ）、`--author`、`--json`。モデルは `config.yaml` の `auxiliary.triage_specifier` で設定します。 |
| `decompose <id>` / `decompose --all` | トリアージ列のタスクを、説明に基づいて専門プロファイルにルーティングされる子タスクのグラフに展開します。LLMがそのタスクは展開しても利点がないと判断した場合は、specifyと同様の単一タスクの昇格にフォールバックします。フラグは `specify` と同じです。分解モデルは `config.yaml` の `auxiliary.kanban_decomposer` で設定します。`kanban.orchestrator_profile` は、展開後にルート/オーケストレーションのタスクを誰が所有するかだけを制御します。`kanban.auto_decompose: true`（デフォルト）の場合は、ディスパッチャーのティックごとに自動的にも実行されます。[自動と手動のオーケストレーション](../user-guide/features/kanban.md#auto-vs-manual-orchestration)を参照してください。 |
| `gc` | アーカイブされたタスクのscratchワークスペースを削除します。 |

例:

```bash
# Create a second board and put a task on it without switching away.
hermes kanban boards create atm10-server --name "ATM10 Server" --icon 🎮
hermes kanban --board atm10-server create "Restart server" --assignee ops

# Switch the active board for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban list                  # shows atm10-server tasks

# Archive a board (recoverable) or hard-delete it.
hermes kanban boards rm atm10-server
hermes kanban boards rm atm10-server --delete
```

ボードの解決順序（優先度の高い順）: `--board <slug>` フラグ → `HERMES_KANBAN_BOARD` 環境変数 → `~/.hermes/kanban/current` ファイル → `default`。

すべてのアクションは、ゲートウェイのスラッシュコマンド（`/kanban …`）としても利用でき、同じ引数面を持ちます — `boards` サブコマンドと `--board` フラグを含みます。

完全な設計 — Cline Kanban / Paperclip / NanoClaw / Gemini Enterpriseとの比較、8つのコラボレーションパターン、4つのユーザーストーリー、並行性の正しさの証明 — については、[Kanbanユーザーガイド](../user-guide/features/kanban.md)を参照してください。

## `hermes egress`

リモートターミナルサンドボックス向けの、送信時に認証情報を注入するファイアウォール。[iron-proxy](https://github.com/ironsh/iron-proxy) デーモンをラップします — これはTLSを傍受するプロキシで、ネットワークの境界で不透明なプロキシトークンを実際の上流APIの認証情報に差し替えるため、サンドボックスが本物のキーを保持することはありません。デフォルトでは無効です。セットアップとアーキテクチャについては、[Egressプロキシ](/docs/user-guide/egress/iron-proxy)のページ全体を参照してください。

```bash
hermes egress install                  # download the pinned iron-proxy binary
hermes egress install --force          # check and repair the managed copy

hermes egress setup                    # interactive wizard: CA, mappings, config
hermes egress setup --tunnel-port N    # override the tunnel listener port (default 9090)
hermes egress setup --from-bitwarden   # use Bitwarden Secrets Manager as credential source
hermes egress setup --no-bitwarden     # explicitly switch back to env-based credentials
hermes egress setup --rotate-tokens    # mint fresh proxy tokens (default preserves existing)

hermes egress start                    # spawn the managed proxy daemon
hermes egress stop                     # SIGTERM (then SIGKILL after 5s grace)
hermes egress restart                  # stop (if running) then start — needed for secret changes
hermes egress reload                   # hot-reload the ruleset in-place (no restart, no dropped
                                       #   connections) via the loopback management API

hermes egress status                   # binary + config + pid + listening + mappings
hermes egress status --show-tokens     # print proxy tokens in full (default: redacted)

hermes egress disable                  # flip proxy.enabled = false (does not stop a running proxy)
hermes egress config                   # print the path to proxy.yaml for inspection
```

### よくある操作の流れ {#common-flows}

```bash
# First-time setup
export OPENROUTER_API_KEY=…
hermes egress setup && hermes egress start
hermes config set terminal.backend docker   # if not already

# Switching credential source after the fact
hermes egress setup --from-bitwarden       # env → bitwarden
hermes egress setup --no-bitwarden         # bitwarden → env
# (just `setup` without either flag preserves the existing mode)

# Rotating all tokens (e.g. after a suspected token leak)
hermes egress setup --rotate-tokens    # setup offers to restart the running daemon for you
# (running sandboxes still hold old tokens; restart them too)

# Adding a new upstream
# Edit ~/.hermes/config.yaml proxy.extra_allowed_hosts: [api.example.com]
hermes egress setup
hermes egress restart                  # one-command apply (stop + start)
```

### 診断用のショートカット {#diagnostic-shortcuts}

```bash
hermes egress status                     # current state in one view
cat ~/.hermes/proxy/proxy.yaml           # the rendered iron-proxy config
tail -20 ~/.hermes/proxy/iron-proxy.log  # daemon-level diagnostics
tail -f ~/.hermes/proxy/iron-proxy.log | jq  # daemon + per-request log (line-delimited JSON; v0.39 combines both streams)
```

よくある障害のパターンと復旧方法は、[Egressプロキシ → トラブルシューティング](/docs/user-guide/egress/iron-proxy#troubleshooting)で説明しています。

## `hermes project`

```bash
hermes project <create|list|show|add-folder|remove-folder|rename|set-primary|use|archive|restore|bind-board>
```

プロジェクトは、複数のフォルダー / リポジトリにまたがることができる、人間が名前を付けたワークスペースです。デスクトップのセッショングループ化の基準となり、プロジェクトにリンクされたkanbanタスク（`hermes kanban create --project <slug>`、またはkanbanダッシュボードでプロジェクトに紐付けられたボード）は、デフォルトでプロジェクトのプライマリフォルダー配下の決定論的なworktree + ブランチを使用します（プロジェクトにプライマリフォルダーがある場合）。状態はプロファイルごとです。

| サブコマンド | 説明 |
|------------|-------------|
| `create` | 新しいプロジェクトを作成します。 |
| `list`（エイリアス `ls`） | プロジェクトを一覧表示します。 |
| `show` | プロジェクトの詳細を表示します。 |
| `add-folder` | プロジェクトにフォルダー / リポジトリを追加します。 |
| `remove-folder` | プロジェクトからフォルダーを削除します。 |
| `rename` | プロジェクトをリネームします。 |
| `set-primary` | プライマリフォルダーを設定します。 |
| `use` | アクティブなプロジェクトを設定します。 |
| `archive` | プロジェクトをアーカイブします（復元可能）。 |
| `restore` | アーカイブされたプロジェクトを復元します。 |
| `bind-board` | kanbanボードをプロジェクトに記録し、ボードが存在してプロジェクトにプライマリフォルダーがある場合は、ボードのデフォルトの作業ディレクトリをそこに向けます。ボードのタスクをプロジェクトにリンクするものではありません。 |

## `hermes webhook`

```bash
hermes webhook <subscribe|list|remove|test>
```

イベント駆動のエージェント起動向けに、動的なWebhookサブスクリプションを管理します。設定でwebhookプラットフォームが有効になっている必要があります — 設定されていない場合は、セットアップ手順を出力します。

| サブコマンド | 説明 |
|------------|-------------|
| `subscribe` / `add` | Webhookルートを作成します。URLとマスクされたHMACシークレットを出力します。完全なシークレットはHermesホーム配下の `webhook_subscriptions.json` に保存されます。 |
| `list` / `ls` | エージェントが作成したすべてのサブスクリプションを表示します。 |
| `remove` / `rm` | 動的なサブスクリプションを削除します。config.yamlの静的ルートは影響を受けません。 |
| `test` | サブスクリプションが機能しているか確認するため、テストPOSTを送信します。 |

### `hermes webhook subscribe`

```bash
hermes webhook subscribe <name> [options]
```

| オプション | 説明 |
|--------|-------------|
| `--prompt` | `{dot.notation}` のペイロード参照を含むプロンプトテンプレート。 |
| `--events` | 受け入れるイベントタイプのカンマ区切り（例: `issues,pull_request`）。空 = すべて。 |
| `--description` | 人間が読める説明。 |
| `--skills` | エージェント実行向けに読み込むスキル名のカンマ区切り。 |
| `--deliver` | 配信ターゲット: `log`（デフォルト）、`telegram`、`discord`、`slack`、`github_comment`。 |
| `--deliver-chat-id` | クロスプラットフォーム配信のターゲットチャット/チャンネルID。 |
| `--secret` | カスタムHMACシークレット。省略時は自動生成されます。 |
| `--signature` | ルートが受け付ける署名方式: `generic`（新しいルートのデフォルト。`X-Webhook-Signature-V2` + `X-Webhook-Timestamp`、またはStandard Webhooks）、`github`、`gitlab`、`linear`。それ以外のヘッダー系統はすべて拒否されます。更新時に省略した場合は保持されます。[署名方式を固定する](../user-guide/messaging/webhooks.md#pin-the-signature-scheme)を参照してください。 |
| `--deliver-only` | エージェントをスキップ — レンダリングされた `--prompt` をリテラルなメッセージとして配信します。LLMコストゼロ、サブ秒の配信。`--deliver` が実際のターゲット（`log` ではない）である必要があります。 |
| `--mirror-to-session` | 配信された各メッセージをターゲットのチャットのセッションにも書き込み、そのチャットで返信したときにコンテキストがあるようにします。デフォルトはoffです。内容を会話内で信頼できるソースに対してのみ有効にしてください。 |
| `--script` | `~/.hermes/scripts/` 配下のフィルター/変換スクリプト。webhookのペイロードはstdinにJSONとして渡されます。stdoutのJSONがペイロードを置き換え、stdoutが空、`[SILENT]`、または非ゼロの終了コードの場合はwebhookを無視します。[スクリプトによるフィルターと変換](../user-guide/messaging/webhooks.md#script-filters-and-transforms)を参照してください。 |
| `--route-profile` | ルートをマルチプレックスされたプロファイルに紐付けます: その場合ルートには `/p/<profile>/webhooks/<name>` でのみアクセスでき、エージェントはそのプロファイルとして実行されます。既存のプロファイルに対して検証され、更新時に省略した場合は保持されます。グローバルな `-p/--profile` とは異なります。そちらはサブスクリプションファイルを書き込むゲートウェイを選択するものです。[マルチプロファイルゲートウェイ](/docs/user-guide/multi-profile-gateways)を参照してください。 |

サブスクリプションは `~/.hermes/webhook_subscriptions.json` に永続化され、ゲートウェイを再起動せずにwebhookアダプターによってホットリロードされます。既存の名前で `subscribe` を再実行すると、`--secret` / `--signature` / `--route-profile` を渡さない限り、そのシークレット、署名の固定、プロファイルの紐付けは保持されます。`hermes webhook test` は、ルートの固定が期待する方式で署名します（`generic` と未固定のルートではV2）。

## `hermes doctor`

```bash
hermes doctor [--fix]
```

| オプション | 説明 |
|--------|-------------|
| `--fix` | 可能な範囲で自動修復を試みます。 |

終了ステータス: レポートに未解決の問題がない場合は `0`、少なくとも1つ残っている場合（`--fix` で修復できなかった問題を含む）は `1` です。そのため、ヘルスゲートやCIステップで `hermes doctor` をチェックとして信頼できます。

**API Connectivity** セクションには `IPv6 route` チェックが含まれます: 既知のデュアルスタックホストに対して、短い（2秒）IPv6 TCP接続を1回開きます。経路が広告されているのにタイムアウトするだけの場合（ブラックホール化したIPv6プレフィックス）は、対処法 `network.force_ipv4: true` を示す警告として報告されます。IPv6の経路がまったくないことは正常で、OKとして報告されます。`force_ipv4` がすでに設定されている場合、このチェックはスキップされます。

カスタムエンドポイントの設定チェック（どちらも警告のみ。`--fix` はこれらを書き換えません）:

- YAMLのリストでない `custom_providers`（例えば誤った `config set` によって残された文字列）は、キーと受け取った型を示すエラーとして報告されます — リストに戻るまで、ランタイムはすべてのカスタムエンドポイントを無視します。
- 対応する `providers:` エントリー（同じエンドポイントURL）を持たないレガシーな `custom_providers` リストのエントリーは、行うべき移動とともに報告されます: そのようなエントリーは、他のすべての画面が編集する `providers:` マップではなく、廃止されたリストストアから引き続き提供されます（モデルピッカーとCustom Endpointsページは両方を読み取ります）。また、リストを `providers:` に移動したワンショットのv12移行は再実行されません。

**Config Structure** は、1つの引用符付き文字列として保存されたリスト/マッピングの設定（`plugins.enabled: '["a","b"]'`、`model_catalog.excluded_providers: '["openai-api"]'` — 古いバージョンの `config set` が書き込んでいた形式）も検出します: このような文字列はすべての読み取り側で無視されるため、プラグインは黙ってマウントされないままになり、除外も適用されません。検出結果には、キーと、実際のリストを保存する `hermes config set <key> '<literal>'` コマンドが示されます。同じ警告は起動時のバナーにも表示されます。`--fix` はファイルを書き換えません。

## `hermes dump`

```bash
hermes dump [--show-keys]
```

Hermesのセットアップ全体を、コンパクトなプレーンテキストの要約として出力します。サポートを求める際にDiscord、GitHubのissue、Telegramにコピーペーストできるよう設計されています — ANSIカラーも特殊なフォーマットもなく、データだけです。

| オプション | 説明 |
|--------|-------------|
| `--show-keys` | 単なる `set`/`not set` ではなく、秘匿化されたAPIキーのプレフィックス（最初と最後の4文字）を表示します。 |

### 含まれる内容 {#what-it-includes}

| セクション | 詳細 |
|---------|---------|
| **Header** | Hermesのバージョン、リリース日、gitコミットハッシュ |
| **Environment** | OS、Pythonバージョン、OpenAI SDKバージョン |
| **Identity** | アクティブなプロファイル名、HERMES_HOMEパス |
| **Model** | 設定されたデフォルトモデルとプロバイダー |
| **Terminal** | バックエンドの種類（local、docker、sshなど） |
| **API keys** | 全22個のプロバイダー/ツールAPIキーの存在チェック |
| **Features** | 有効なツールセット、MCPサーバー数、メモリプロバイダー |
| **Services** | ゲートウェイのステータス、設定されたメッセージングプラットフォーム |
| **Workload** | cronジョブ数、インストール済みスキル数 |
| **Config overrides** | デフォルトと異なる設定値。そこに含まれる認証情報は秘匿化されます: `fallback_providers` エントリーの `api_key`、およびその `base_url` 内の認証情報（userinfo、`key`/トークンのクエリパラメーター、署名付きURLの署名）。 |

### 出力例 {#example-output}

```
--- hermes dump ---
version:          0.8.0 (2026.4.8) [af4abd2f]
os:               Linux 6.14.0-37-generic x86_64
python:           3.11.14
openai_sdk:       2.24.0
profile:          default
hermes_home:      ~/.hermes
model:            anthropic/claude-opus-4.6
provider:         openrouter
terminal:         local

api_keys:
  openrouter           set
  openai               not set
  anthropic            set
  nous                 not set
  firecrawl            set
  ...

features:
  toolsets:           all
  mcp_servers:        0
  memory_provider:    built-in
  gateway:            running (systemd)
  platforms:          telegram, discord
  cron_jobs:          3 active / 5 total
  skills:             42

config_overrides:
  agent.max_turns: 250
  compression.threshold: 0.85
  display.streaming: True
--- end dump ---
```

### 使用するタイミング {#when-to-use}

- GitHubでバグを報告する — issueにダンプを貼り付けます
- Discordでヘルプを求める — コードブロックで共有します
- 自分のセットアップを他の人のものと比較する
- 何かがうまく動かないときの簡単な健全性チェック

:::tip
`hermes dump` は特に共有向けに設計されています。対話型の診断には `hermes doctor` を、視覚的な概要には `hermes status` を使用してください。
:::

## `hermes debug`

```bash
hermes debug share [options]
```

デバッグの要約を公開のペーストサービスにアップロードして共有可能なURLを取得するか、（`--nous` を指定すると）秘匿化した完全なログを非公開のNousストレージに送信します。

| オプション | 説明 |
|--------|-------------|
| `--lines <N>` | ファイルごとのログ行数: 要約のエラー件数の集計対象としてスキャンされ、`--nous` / `--local` では（秘匿化して）含まれます（デフォルト: 200）。 |
| `--expire <days>` | dpaste.comにフォールバックした場合の保持期間（日数。最小1、デフォルト: 1）。paste.rsのペーストは常に6時間後に削除されます。dpaste.comのペーストは早期に削除できません。 |
| `--nous` | 公開のペーストサービスの代わりに、Nous内部の診断ストレージにアップロードします。Nousのサポートから非公開の診断バンドルを求められた場合に使用してください。 |
| `--local` | アップロードする代わりに、ローカルでレポートを出力します。 |
| `--no-redact` | `--nous` で送信される、または `--local` で出力されるログのシークレット秘匿化を無効にします。ログのテキストを一切含まない公開ペーストには影響しません。 |
| `--yes` | 確認プロンプトをスキップします（stdinがTTYでない場合は必須）。 |

**公開ペースト（デフォルト）。** アップロードされるのは許可リストに基づく要約であり、ログ行やメッセージのテキストは決して含まれません: Hermes、Python、OSのバージョン。インストールの種類。接続済みプラットフォームと設定済みの組み込みツールセットの名前。どの `config.yaml` キーが設定されているか（キー名のみで値は含まず、プロバイダーやMCPサーバーなどユーザーが名前を付けたエントリーは名前ではなく件数のみ）。そして最近のログの末尾から集計した、ロガー、例外クラス、HTTPステータスごとの警告/エラーの件数。ゲートウェイの `/debug upload` コマンドとダッシュボードの **Share debug report** ボタンがアップロードするのも、これと同じ要約です。公開アップロードはpaste.rsに送られ、失敗した場合はdpaste.comにフォールバックします。

**非公開バンドル（`--nous`）。** 完全なデバッグレポート、システムダンプ、エージェント、ゲートウェイ、GUI/ダッシュボード、デスクトップのログ（ファイルあたり512 KB）に加えて、存在する場合はアップデートとDesktopアップデートの引き継ぎログが、非公開のNous診断ストレージに送られます。`--no-redact` が設定されていない限りシークレットは秘匿化されますが、ログ内のその他の個人データ（名前、ID、メッセージのテキスト）は秘匿化されません。返されるビューアーのリンクはNousチーム専用で、14日後に自動的に削除されます。`--local` は、何もアップロードせずに同じバンドルを出力します。

### 例

```bash
hermes debug share              # Upload debug report, print URL
hermes debug share --lines 500  # Scan more log lines for the summary
hermes debug share --expire 30  # Keep a dpaste.com fallback paste for 30 days
hermes debug share --nous       # Upload a private diagnostics bundle for Nous support
hermes debug share --local      # Print report to terminal (no upload)
```

## `hermes backup`

```bash
hermes backup [options]
```

Hermesの設定、スキル、セッション、データのzipアーカイブを作成します。バックアップにはhermes-agentのコードベース自体は含まれず、以前のバックアップ成果物（`backups/`、`state-snapshots/`）も入れ子にはしません — それらはそれぞれ、すでに独自の `state.db` のコピーを含んでいるためです。

| オプション | 説明 |
|--------|-------------|
| `-o`, `--output <path>` | zipファイルの出力パス（デフォルト: `~/hermes-backup-<timestamp>.zip`）。 |
| `-q`, `--quick` | クイックスナップショット: 重要な状態ファイル（config.yaml、state.db、.env、auth、cronジョブ）のみ。完全バックアップよりはるかに高速です。 |
| `-l`, `--label <name>` | スナップショットのラベル（`--quick` でのみ使用）。 |
| `-k`, `--keep <N>` | 完全バックアップの後、出力ディレクトリ内の古い `hermes-backup-*.zip` ファイルを、最新のN個を超える分だけ削除します（デフォルト3。`0` ですべて保持）。独自の名前を付けたzipには決して触れません。 |

バックアップはSQLiteの `backup()` APIを使用して安全にコピーするため、Hermesの実行中でも正しく動作します（WALモードで安全）。

**終了ステータス:** 選択したすべてのファイルがアーカイブに収まった場合にのみ `0` です。一部のファイルを追加できなかった場合（`Backup incomplete: …`）、残りを復元できるようzipは保持されますが、コマンドは `1` で終了します — cronやsystemdタイマーが部分的なアーカイブを成功として報告することはなく、`--keep` によるプルーニングもスキップされるため、古い完全なアーカイブは残ります。`2` は別のバックアップがすでに実行中だったことを意味します。

**zipから除外されるもの:**

- `*.db-wal`、`*.db-shm`、`*.db-journal` — SQLiteのWAL / 共有メモリ / ジャーナルのサイドカー。`*.db` ファイルはすでに `sqlite3.backup()` で一貫したスナップショットを取得済みです。ライブのサイドカーを一緒に同梱すると、復元時に中途半端にコミットされた状態が見えてしまいます。
- `checkpoints/` — セッションごとの軌跡キャッシュ。ハッシュキー付きでセッションごとに再生成されるため、いずれにせよ別のインストールにきれいに移植できません。
- `~/.hermes`（および各 `profiles/<name>/`）の直下にある `models/`、`runtimes/`、`node/` — 再生成可能なランタイムのダウンロードで、数十GBになることもよくあります。より深い階層にある同名のディレクトリ（スキルの `models/` など）は保持されます。
- ブラウザープロファイル: 任意の階層の `browser-profile/`（実プロファイルのスナップショット — コピーされたCookies / Login Data）と `browser-profiles/`（ライブのCDPプロファイル）、および `~/.hermes` と各 `profiles/<name>/` の直下にある `browser_profiles/`（Browser Use CLIバックエンドのChromiumユーザーデータディレクトリで、独自のLogin Data / Cookiesを持つ）。これらはアーカイブに決して入れてはならない認証情報ストアであり、すべて次回の起動時に再生成されます。
- 同じルート直下の `cache/` のうち再生成可能なエントリー — モデル/プラグインのカタログ、スタンプ、ブラウザープロファイル、ツール出力の退避ファイル。永続的な成果物は含まれたままです: `cache/images`、`cache/audio`、`cache/videos`、`cache/documents`、`cache/screenshots`（あなたに配信された、またはあなたから受け取ったメディア）と `cache/citations`（根拠付き引用の台帳）。より深い階層の `cache/`（スキル内）は丸ごと保持されます。
- Unixソケット、デバイス、シンボリックリンク — zipはこれらを保持できません。これらが除外される前は、迷い込んだ `gateway.sock` のせいで、すべての完全バックアップが `Backup incomplete` を報告していました。
- `hermes-agent` のコード自体（これはユーザーデータのバックアップであり、リポジトリのスナップショットではありません）。

### 例

```bash
hermes backup                           # Full backup to ~/hermes-backup-*.zip
hermes backup -o ~/backups/hermes.zip   # Full backup to specific path
hermes backup --quick                   # Quick state-only snapshot
hermes backup --quick --label "pre-upgrade"  # Quick snapshot with label
```

## `hermes checkpoints`

```bash
hermes checkpoints [COMMAND]
```

`~/.hermes/checkpoints/` のシャドウgitストア — セッション内の `/rollback` コマンドの背後にあるストレージレイヤー — を検査・管理します。いつでも安全に実行でき、エージェントの実行は不要です。

| サブコマンド | 説明 |
|------------|-------------|
| `status`（デフォルト） | 合計サイズ、プロジェクト数、プロジェクトごとの内訳を表示します。素の `hermes checkpoints` と同等です。 |
| `list` | `status` のエイリアス。 |
| `prune` | クリーンアップのスイープを強制します — 孤立した古いプロジェクトを削除し、ストアをGCし、サイズ上限を適用します。24時間の冪等性マーカーを無視します。 |
| `clear` | チェックポイントのベース全体を削除します。元に戻せません。`-f` がない限り確認を求めます。 |
| `clear-legacy` | v1→v2の移行で生成された `legacy-<timestamp>/` アーカイブのみを削除します。いずれかのアーカイブを削除できなかった場合（例: Windowsでの読み取り専用のgitオブジェクト）は、`Could not delete N archive(s)` を出力した後に `2` で終了します。 |

### オプション

| オプション | サブコマンド | 説明 |
|--------|------------|-------------|
| `--limit N` | `status`, `list` | 一覧表示するプロジェクトの最大数（デフォルト20）。 |
| `--retention-days N` | `prune` | `last_touch` がN日より古いプロジェクトを削除します（デフォルト7）。 |
| `--max-size-mb N` | `prune` | 孤立/古いもののパス後、合計ストアサイズが N MB 以下になるまでプロジェクトごとに最も古いコミットを削除します（デフォルト500）。 |
| `--keep-orphans` | `prune` | 作業ディレクトリがもはや存在しないプロジェクトの削除をスキップします。 |
| `-f`, `--force` | `clear`, `clear-legacy` | 確認プロンプトをスキップします。 |

### 例

```bash
hermes checkpoints                                  # status overview
hermes checkpoints prune --retention-days 3         # aggressive cleanup
hermes checkpoints prune --max-size-mb 200          # tighten size cap once
hermes checkpoints clear-legacy -f                  # drop v1 archive dirs
hermes checkpoints clear -f                         # wipe everything
```

完全なアーキテクチャとセッション内コマンドについては、[チェックポイントと `/rollback`](../user-guide/checkpoints-and-rollback.md)を参照してください。

## `hermes import`

```bash
hermes import <zipfile> [options]
```

以前作成したHermesバックアップをHermesのホームディレクトリに復元します。アーカイブ内のすべてのファイルがHermesホームの既存ファイルを上書きします。`--force` は、ターゲットにすでにHermesがインストールされている場合に発生する確認プロンプトをスキップするだけです。

| オプション | 説明 |
|--------|-------------|
| `-f`, `--force` | 既存インストールの確認プロンプトをスキップします。 |

:::warning
実行中のプロセスとの競合を避けるため、インポート前にゲートウェイを停止してください。
:::

**終了ステータス:** アーカイブが破損している場合は `1` です — 何かを書き込む前に、すべてのメンバーが一度展開されてCRCがチェックされます。いずれかが失敗した場合、コマンドは問題のあるメンバーとともに `Error: backup archive is damaged (N member(s) …)` を出力し、Hermesホームに手を付けずに停止します。アーカイブ内のいずれかのファイルを復元できなかった場合（`Warnings (N files skipped)` の下に一覧表示され、`Import incomplete: …` として要約されます）も `1` です。復元できたファイルはそのまま残りますが、スクリプトやダッシュボードが部分的な復元を成功として報告することはありません。インポートが意図的にこのマシンのものを保持するランタイムファイル（`gateway.pid`、`gateway_state.json`、…）と、下記の古いバックアップに関するセッションの警告は、終了ステータスに影響しません。

### SQLiteデータベース {#sqlite-databases}

`.db` メンバー（`state.db`、`kanban.db`、`response_store.db`、…）は、通常のファイルのようにリネームによって配置されることはありません。リネームすると、ゲートウェイ、ダッシュボード、WebUIのプロセスが古いinodeを開いたままの状態でファイルのinodeが置き換わってしまいます: そのプロセスはインポート前のページを読み続け、他の誰からも見えないセッションを書き続けます。そしてそれらのセッションは、次に誰もが開くデータベースには単に存在せず — 何もログに残りません。代わりに、インポートされたページは `/snapshot restore` と同じ方法で**既存のデータベースファイルの中に**書き込まれるため、開いているすべての接続がインポートされたデータに収束します。

稼働中のデータベースを安全に置き換えられない場合 — ページのコピーが失敗し、*かつ*別のプロセスがまだファイルを開いている場合 — インポートはそのデータベースに手を付けず、`Warnings (N files skipped)` の下に一覧表示します。ファイルを開いているプロセスを停止して再実行してください。

新しい作業の上に古いバックアップをインポートすることは引き続き可能ですが、もはや黙って行われることはありません。インポートされた `state.db` が置き換え前のものよりメッセージ数が少ない場合、要約でそれが報告されます:

```
  ⚠ Session data replaced by older backup contents:
    state.db: 12 session(s) / 8912 message(s) -> 3 / 24
    Anything recorded after the backup was taken is not in it.
    Recover from a newer backup or snapshot: hermes snapshot list
```

### 例
```bash
hermes import ~/hermes-backup-20260423.zip           # Prompts before overwriting existing config
hermes import ~/hermes-backup-20260423.zip --force   # Overwrite without prompting
```

## `hermes logs`

```bash
hermes logs [log_name] [options]
```

Hermesのログファイルを表示、追跡、フィルタリングします。すべてのログは `~/.hermes/logs/`（非デフォルトプロファイルの場合は `<profile>/logs/`）に保存されます。

### ログファイル {#log-files}

| 名前 | ファイル | 記録される内容 |
|------|------|-----------------|
| `agent`（デフォルト） | `agent.log` | すべてのエージェントアクティビティ — API呼び出し、ツールディスパッチ、セッションのライフサイクル（INFO以上） |
| `errors` | `errors.log` | 警告とエラーのみ — agent.logのフィルタリングされたサブセット |
| `gateway` | `gateway.log` | メッセージングゲートウェイのアクティビティ — プラットフォーム接続、メッセージディスパッチ、webhookイベント |
| `gui` | `gui.log` | ダッシュボード / TUIゲートウェイ / PTYブリッジ / websocketのイベント |
| `desktop` | `desktop.log` | Electronデスクトップアプリ — 起動、バックエンドの起動時の出力、最近のPythonトレースバック |
| `mcp` | `mcp-stderr.log` | すべてのstdio MCPサーバーのstderr。起動ごとに `starting MCP server` のバナーが1つ記録されます |
| `update` | `update.log` | `hermes update` の実行のstdout/stderrの完全なミラー（追記のみ） — 更新/依存関係の失敗の根本原因 |
| `handoff` | `desktop-update-handoff.log` | Desktop主導のアップデートの引き継ぎ段階。Desktopの再ビルドの再試行の出力を含みます |

### オプション

| オプション | 説明 |
|--------|-------------|
| `log_name` | 表示するログ: `agent`（デフォルト）、`errors`、`gateway`、`gui`、`desktop`、`update`、`handoff`、または利用可能なファイルをサイズとともに表示する `list`。 |
| `-n`, `--lines <N>` | 表示する行数（デフォルト: 50）。 |
| `-f`, `--follow` | `tail -f` のように、ログをリアルタイムで追跡します。停止するにはCtrl+Cを押します。 |
| `--level <LEVEL>` | 表示する最小ログレベル: `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。 |
| `--session <ID>` | セッションIDの部分文字列を含む行をフィルタリングします。 |
| `--since <TIME>` | 相対的な時間からの行を表示します: `30m`、`1h`、`2d` など。`s`（秒）、`m`（分）、`h`（時間）、`d`（日）をサポートします。 |
| `--component <NAME>` | コンポーネントでフィルタリングします: `gateway`、`agent`、`tools`、`cli`、`cron`。 |

トレースバックのフレームや複数行メッセージの続きのように、独自のタイムスタンプを持たない行は、その上にあるタイムスタンプ付きの行と一緒に表示または非表示になります。

### 例

```bash
# View the last 50 lines of agent.log (default)
hermes logs

# Follow agent.log in real time
hermes logs -f

# View the last 100 lines of gateway.log
hermes logs gateway -n 100

# Show only warnings and errors from the last hour
hermes logs --level WARNING --since 1h

# Filter by a specific session
hermes logs --session abc123

# Follow errors.log, starting from 30 minutes ago
hermes logs errors --since 30m -f

# List all log files with their sizes
hermes logs list
```

### フィルタリング {#filtering}

フィルターは組み合わせられます。複数のフィルターがアクティブな場合、ログ行は表示されるためにそれらの**すべて**を通過する必要があります:

```bash
# WARNING+ lines from the last 2 hours containing session "tg-12345"
hermes logs --level WARNING --since 2h --session tg-12345
```

解析可能なタイムスタンプを持たない行は、`--since` がアクティブなときに含まれます（複数行のログエントリーの継続行である可能性があります）。検出可能なレベルを持たない行は、`--level` がアクティブなときに含まれます。

### ログローテーション {#log-rotation}

HermesはPythonの `RotatingFileHandler` を使用します。古いログは自動的にローテーションされます — `agent.log.1`、`agent.log.2` などを探してください。`hermes logs list` サブコマンドは、ローテーションされたものを含むすべてのログファイルを表示します。


## `hermes prompt-size`

```bash
hermes prompt-size [--platform <name>] [--json]
```

新しいセッションの固定的なプロンプト予算 — 会話の内容より*前に*、すべてのAPI呼び出しで送信されるもの — を
報告します。下流のアダプターやプロキシのプロンプト予算がモデルのコンテキストウィンドウより厳しい場合や、
どのブロック（スキルインデックス、メモリ、プロファイル）が大半を占めているかを確認したい場合に便利です。

エージェントが構築するのと同じシステムプロンプトを構築し、その内訳を示します:

- **System prompt total** — 組み立てられたプロンプト全体（アイデンティティ、ガイダンス、スキル
  インデックス、コンテキストファイル、メモリ、プロファイル、タイムスタンプ）。
- **Skills index** — `<available_skills>` ブロック。多数のスキルがインストールされている場合、
  単一のブロックとしては最大になることがよくあります。
- **Memory** と **user profile** — `MEMORY.md` / `USER.md` のスナップショット。
- **Prompt tiers** — stable / context / volatile。キャッシュ効率のためにHermesがプロンプトを
  階層化する方法に対応しています。
- **Tool schemas** — 有効なすべてのツールのJSON（呼び出しごとの固定ペイロードの残り半分）。

完全にオフラインで動作します — API呼び出しは行わず、認証情報が設定されていなくても動作します。

```bash
# Human-readable breakdown for the CLI platform (default)
hermes prompt-size

# Simulate a messaging platform's prompt (different platform hint)
hermes prompt-size --platform telegram

# Machine-readable output for scripts
hermes prompt-size --json
```

:::tip
スキルインデックスとツールスキーマは、有効にしているスキルとツールの数に応じて大きくなります。
プロンプトを小さくするには、使わないツールセットを無効にする（`hermes tools`）か、不要なスキルを
アンインストールしてください（`hermes skills`）。現在のディレクトリにあるコンテキストファイル（AGENTS.md、
.cursorrules）も合計に含まれます。
:::

## `hermes config`

```bash
hermes config <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `show` | 現在の設定値を表示します。 |
| `edit` | エディターで `config.yaml` を開きます。 |
| `get <key> [--json] [--raw]` | ドット区切りのキーで単一の設定値を出力します（例: `hermes config get model.default`）。`--json` は機械可読の出力を行います。認証情報らしき値（`api_key`、`*_TOKEN`、`*_SECRET`、`password`、…）はマスクされます（`sk-o...7890`）。これは、トランスクリプトが永続化されるセッションからエージェントがこのコマンドを実行するためです。`--raw` は実際の値を出力します（または `security.redact_secrets: false` を設定してください）。既知のセクション配下にある、スキーマで定義されていないネストされたキー（`compression.compressor.enabled`）でも、ファイル内の値は出力され、加えてHermesがそれを読み取らない可能性がある旨の通知がstderrに出力されます。stdoutと終了コード（0）は変わりません。 |
| `set <key> <value> [--force]` | 設定値を設定します。ドット区切りのパスは `config.yaml` に書き込まれます。すべての `UPPER_SNAKE` 名（`OPENROUTER_API_KEY`、`DISCORD_HOME_CHANNEL`、`TELEGRAM_GROUP_ALLOWED_USERS`、`HERMES_TIMEZONE`、…）は環境変数として `.env` に書き込まれます — プラットフォームのセットアップフローや `/sethome` が書き込むのと同じファイルであり、すべてのランタイムの読み取り側が参照するファイルです。`config set` は、`--force` を指定しても、`UPPER_SNAKE` キーを `config.yaml` に書き込むことはありません。環境変数ライターの拒否リストにある名前（`HERMES_YOLO_MODE`、`PATH`、…）は即座に拒否されます。それ以外の `UPPER_SNAKE` 名はそのまま `.env` に保存されます（プラグイン、スキル、外部ツールはプロセスの環境からそれを読み取ります）。誤ったプレフィックスの下に書かれた既知のキー（`gateway.discord.foo` で、`discord.foo` 自体が既知のキーである場合）は、did-you-meanの提案とともに拒否され、何も書き込まれません。既知のセクション配下のその他の未知のパス（`agent.max_turnz`、またはランタイムが読み取るがデフォルト値が用意されていないキー）はdid-you-meanの通知とともに書き込まれ、未知の小文字の*トップレベル*キーは通知とともに書き込まれます（トップレベルのスカラー値はスキル向けに環境変数へ橋渡しされます）。`--force` は、拒否された誤ったプレフィックスのパスも書き込みます。値はスキーマに対して型チェックされます: リストまたはマッピングを保持しなければならないキー（`custom_providers`、`model.aliases`、`display.platforms`、`plugins.enabled`/`plugins.disabled`、`model_catalog.excluded_providers`、すでにリストやマッピングを保持している任意のキー）は、プレーンな文字列や形の誤ったリテラルを拒否します。また、リスト/マッピングのように見えるが有効なYAML/JSONでない値は、文字列として保存されるのではなく拒否されます — 何も書き込まれず、エラーには期待される型が示されます。YAML/JSONのリテラルを渡してください（`hermes config set custom_providers '[{name: x, base_url: https://...}]'`）。単に `[` や `{` で始まるだけの文字列を保存したい場合は、YAMLで引用符で囲んでください（`"'[text'"`）。`--force` は引き続きマッピングのセクション全体を置き換えます。リストの位置にリスト以外を指定した場合に上書きする手段はありませんが、寛容に読み取られる名前のリスト（`agent.disabled_toolsets`、`skills.disabled`）に対する単独の名前は、1要素のリストとして保存されます。 |
| `unset <key>` | 設定キーを削除し、組み込みのデフォルトに戻します。`UPPER_SNAKE` 名の場合は `.env` のエントリーを削除し、さらに古い `config set` の実行によって残された、`config.yaml` のトップレベルにある古いコピーも削除します（`get` はそのようなコピーを古いものとして報告します）。 |
| `path` | 設定ファイルのパスを出力します。 |
| `env-path` | `.env` ファイルのパスを出力します。 |
| `check` | 欠落または古い設定をチェックします。 |
| `migrate` | 新しく導入されたオプションを対話的に追加します。 |

`config set model.provider <provider>` は、`model:` ブロックを1つのルートに保ちます: 以前のプロバイダーから
残った `model.base_url` / `model.api_mode` が別のプロバイダーのエンドポイントである場合、それは削除され（一覧表示されます）
— そうしないと、新しいプロバイダーのキーが古いエンドポイントに送信され、誤ったプロバイダー名を示す認証情報エラーで
失敗してしまいます。新しいプロバイダー自身のエンドポイント、名前付きの `custom_providers` エントリーのエンドポイント、
または `custom`/ローカルのエイリアス配下の任意のURLはそのまま残ります。認識できないホスト（プロキシ、LAN上のサーバー）は、
引き続き適用される旨の警告とともに残ります。

### キー名に含まれるドット {#dots-inside-key-names}

`hermes config set/get/unset` は `.` をネストの区切り文字として使用しますが、実際の
キー名の多くはリテラルのドットを含みます — モデルID（`grok-4.6`、`glm-5.3-flash`）、
MatrixのルームID（`!room:example.org`）、バージョン付きのプロバイダー名などです。
次の2つのルールによって、これらを指定できるようになっています:

- **既存のキーはそのまま動作します。** 既存のマッピングをたどる際、ドット区切りの残りの部分に
  一致する既存のリテラルキーは、分割よりも優先されます。`hermes config set providers.p.models.grok-4.6.supports_vision true`
  は実際の `grok-4.6` エントリーを更新します（`get`/`unset` も同じ方法で解決されます）。
- **ドットを含む新しいキーを作成するにはエスケープが必要です。** リテラルのドットはバックスラッシュで
  エスケープします: `hermes config set 'providers.p.models.grok-4\.7.context_length' 128000`
  はリテラルの `grok-4.7` キーを作成します。（シェルがバックスラッシュを保持するよう、キーを
  引用符で囲んでください。）

エスケープされていない書き込みが、既存のドットを含む兄弟キーを覆い隠すネストされたマッピングを
作成してしまう場合（例: 既存の `grok-4.6` の隣に `grok-4` を作成する）、コマンドは、ランタイムが
決して読み取らない幻のエントリーを黙って書き込むのではなく、エラーで失敗します。

## `hermes pairing`

```bash
hermes pairing <list|approve|revoke|clear-pending>
```

| サブコマンド | 説明 |
|------------|-------------|
| `list` | 保留中および承認済みのユーザーを表示します。 |
| `approve <platform> <code>` | ペアリングコードを承認します。 |
| `revoke <platform> <user-id>` | ユーザーのアクセスを取り消します。 |
| `clear-pending` | 保留中のペアリングコードをクリアします。 |

## `hermes skills`

```bash
hermes skills <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `browse` | スキルレジストリのページネーション付きブラウザー。 |
| `search` | スキルレジストリを検索します。 |
| `install` | スキルをインストールします。 |
| `inspect` | スキルをインストールせずにプレビューします。 |
| `list` | インストール済みのスキルを一覧表示します。 |
| `check` | インストール済みのハブスキルに上流の更新がないか確認します。 |
| `update` | 上流の変更がある場合に、ハブスキルを再インストールします。 |
| `audit` | インストール済みのハブスキルを再スキャンします。 |
| `uninstall` | ハブからインストールされたスキルを削除します。 |
| `reset` | `user_modified` とフラグ付けされたバンドルスキルを、マニフェストエントリーをクリアして解除します。`--restore` を付けると、ユーザーのコピーをバンドルバージョンに置き換えます。 |
| `opt-out` | バンドルスキルがアクティブなプロファイルに配置されないようにします。`.no-bundled-skills` マーカーを書き込み、インストーラー、`hermes update`、およびすべての同期がバンドルスキルの配置をスキップするようにします。デフォルトでは安全で、ディスク上のものには一切触れません。`--remove` を付けると、すでに存在するバンドルスキルのうち**変更されていない**ものも削除します（ユーザーが編集したスキル、ハブからインストールしたスキル、手書きのスキルは決して削除されません。最初にプレビューして確認を求め、`--yes` でスキップできます）。 |
| `opt-in` | `.no-bundled-skills` マーカーを削除して `opt-out` を取り消し、次回の `hermes update` でバンドルスキルが再び配置されるようにします。`--sync` を付けると、すぐに再配置します。 |
| `publish` | スキルをレジストリに公開します。 |
| `snapshot` | スキル設定をエクスポート/インポートします。 |
| `tap` | カスタムスキルソースを管理します。 |
| `config` | プラットフォームごとにスキルの有効化/無効化を対話的に設定します。 |

よくある例:

```bash
hermes skills browse
hermes skills browse --source official
hermes skills search react --source skills-sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect official/security/1password
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install official/mcp/fastmcp
hermes skills install skills-sh/anthropics/skills/pdf --force
hermes skills install https://sharethis.chat/SKILL.md                     # Direct URL (+ referenced support files)
hermes skills install https://example.com/SKILL.md --name my-skill        # Override name when frontmatter has none
hermes skills check
hermes skills update
hermes skills config
hermes skills reset google-workspace
hermes skills reset google-workspace --restore --yes
hermes skills opt-out                  # stop future bundled-skill seeding (nothing deleted)
hermes skills opt-out --remove --yes   # also delete UNMODIFIED bundled skills
hermes skills opt-in --sync            # undo: remove marker and re-seed now
```

注意:
- `--force` は、サードパーティ/コミュニティスキルに対する危険でないポリシーブロックを上書きできます。
- `--force` は `dangerous` のスキャン判定を上書きしません。
- `--source skills-sh` は公開の `skills.sh` ディレクトリを検索します。
- `--source well-known` を使うと、`/.well-known/skills/index.json` を公開しているサイトにHermesを向けられます。
- `--source browse-sh` は、[browse.sh](https://browse.sh) の200以上のサイト固有のブラウザー自動化スキルのカタログを検索します。識別子は `browse-sh/airbnb.com/search-listings-ddgioa` のような形式です。
- `http(s)://…/*.md` のURLを渡すと、`SKILL.md` に加えて、`references/`、`templates/`、`scripts/`、`assets/`、`examples/` 配下の明示的に参照されたファイルをインストールします。frontmatterに `name:` がなく、URLのスラッグが有効な識別子でない場合、対話型ターミナルで名前の入力を求めます。非対話型の操作面（TUI内の `/skills install`、ゲートウェイプラットフォーム）では、代わりに `--name <x>` が必要です。

## `hermes bundles`

```bash
hermes bundles <subcommand>
```

スキルバンドルは、複数のスキルを1つの `/<bundle-name>` スラッシュコマンドにまとめます。バンドルを呼び出すと、参照されているすべてのスキルが1つの結合されたユーザーメッセージに読み込まれます。保存場所: `~/.hermes/skill-bundles/<slug>.yaml`。YAMLスキーマと動作については[スキルバンドル](../user-guide/features/skills.md#skill-bundles)を参照してください。

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `list` | インストール済みのバンドルを一覧表示します（サブコマンドがない場合のデフォルト） |
| `show <name>` | 1つのバンドルの名前、説明、スキル、ファイルパスを表示します |
| `create <name>` | 新しいバンドルを作成します。`--skill <id>` を（繰り返し）渡すか、省略すると対話的に入力します。`--description`、`--instruction`、`--force` も利用できます。 |
| `delete <name>` | バンドルファイルを削除します |
| `reload` | `~/.hermes/skill-bundles/` を再スキャンし、追加/削除されたバンドルを報告します |

例:

```bash
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work"

hermes bundles list
hermes bundles show backend-dev
hermes bundles delete backend-dev
```

チャットセッションでは、`/bundles` がインストール済みのバンドルを一覧表示し、`/<bundle-name>` がバンドルを読み込みます。

## `hermes curator`

```bash
hermes curator <subcommand>
```

curatorは補助モデルのバックグラウンドタスクで、エージェントが作成したスキルを定期的にレビューし、古いものをプルーニングし、重複を統合し、廃止されたスキルをアーカイブします。バンドルスキルとハブからインストールされたスキルは決して変更されません。アーカイブは復元可能で、自動削除は決して行われません。

| サブコマンド | 説明 |
|------------|-------------|
| `status` | curatorのステータスとスキル統計を表示します |
| `run` | curatorレビューを今すぐトリガーします（LLMパスが終わるまでブロックします） |
| `run --background` | LLMパスをバックグラウンドスレッドで開始し、すぐに返ります |
| `run --dry-run` | プレビューのみ — 変更なしでレビューレポートを生成します |
| `backup` | `~/.hermes/skills/` の手動tar.gzスナップショットを取得します（curatorも実際の実行前に自動的にスナップショットを取得します） |
| `rollback` | スナップショットから `~/.hermes/skills/` を復元します（デフォルトは最新） |
| `rollback --list` | 利用可能なスナップショットを一覧表示します |
| `rollback --id <ts>` | IDで特定のスナップショットを復元します |
| `rollback -y` | 確認プロンプトをスキップします |
| `pause` | 再開されるまでcuratorを一時停止します |
| `resume` | 一時停止したcuratorを再開します |
| `pin <skill>` | curatorが自動遷移しないようにスキルを固定します |
| `unpin <skill>` | スキルの固定を解除します |
| `restore <skill>` | アーカイブされたスキルを復元します |
| `archive <skill>` | スキルを手動でアーカイブします |
| `prune` | curatorが通常クリーンアップするスキルを手動でプルーニングします |
| `list-archived` | アーカイブされたスキルを一覧表示します（`restore` で復元可能） |

新規インストール時、最初のスケジュール済みパスは1回分の `interval_hours`（デフォルトで7日）だけ遅延されます — ゲートウェイは `hermes update` 後の最初のティックで即座にキュレーションを行いません。それが起きる前にプレビューするには `hermes curator run --dry-run` を使用してください。

動作と設定については[Curator](../user-guide/features/curator.md)を参照してください。

## `hermes moa`

名前付きのMixture of Agentsプリセットを設定します。プリセットは、すべてのモデルピッカーで `Mixture of Agents` プロバイダーの下に選択可能なモデルとして表示されます。`/moa <prompt>` は、1つのプロンプトをデフォルトのプリセットで実行します。

```bash
hermes moa list
hermes moa configure [name]
hermes moa delete <name>
```

`hermes moa configure` は、各参照モデルとアグリゲーターについて、Hermesのプロバイダー → モデルのピッカーを再利用します。プリセットは実行モードの設定であり、プライマリモデルやプロバイダーではありません。

## `hermes fallback`

```bash
hermes fallback <subcommand>
```

フォールバックプロバイダーチェーンを管理します。フォールバックプロバイダーは、プライマリモデルがレート制限、過負荷、または接続エラーで失敗したときに順番に試行されます。

| サブコマンド | 説明 |
|------------|-------------|
| `list`（エイリアス: `ls`） | 現在のフォールバックチェーンを表示します（サブコマンドがない場合のデフォルト） |
| `add` | プロバイダー + モデルを選択（`hermes model` と同じピッカー）し、チェーンに追加します |
| `remove`（エイリアス: `rm`） | チェーンから削除するエントリーを選択します |
| `clear` | すべてのフォールバックエントリーを削除します |
| `check [--live] [--json]` | 各エントリーが使用可能な認証情報で解決されるかを確認します（`--live`: 静的に準備完了と判定された固定ルートのエントリーごとに、そのエントリー自身のプロバイダー/モデルに対して最小限のリクエストを1回送信します。他のものには送信しません）。準備完了でないエントリーがある場合は1で終了します。エージェントのセッションは作成されません |

[フォールバックプロバイダー](../user-guide/features/fallback-providers.md)を参照してください。

## `hermes hooks`

```bash
hermes hooks <subcommand>
```

`~/.hermes/config.yaml` で宣言されたシェルスクリプトフックを検査し、合成ペイロードに対してテストし、`~/.hermes/shell-hooks-allowlist.json` の初回使用同意の許可リストを管理します。

| サブコマンド | 説明 |
|------------|-------------|
| `list`（エイリアス: `ls`） | マッチャー、タイムアウト、同意ステータスとともに設定済みのフックを一覧表示します |
| `test <event>` | `<event>` に一致するすべてのフックを合成ペイロードに対して実行します |
| `revoke`（エイリアス: `remove`, `rm`） | コマンドの許可リストエントリーを削除します（次回の再起動時に有効） |
| `doctor` | 各設定済みフックをチェックします: 実行ビット、許可リスト、mtimeのずれ、JSONの有効性、合成実行のタイミング |

イベントシグネチャとペイロードの形状については[フック](../user-guide/features/hooks.md)を参照してください。

## `hermes memory`

```bash
hermes memory <subcommand>
```

外部メモリプロバイダープラグインをセットアップ・管理します。バンドルされているプロバイダー: retaindb、byterover。honcho、hindsight、supermemory、mem0、openviking、holographic（プラグインカタログ）は `hermes plugins install <name>` の後に利用できます（`memory.provider` ですでに指定されているプロバイダーについては、`hermes update` が自動的にこれを行います）。一度にアクティブにできる外部プロバイダーは1つだけです。組み込みメモリ（MEMORY.md/USER.md）は常にアクティブです。

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `setup` | 対話型のプロバイダー選択と設定。 |
| `status` | 現在のメモリプロバイダー設定を表示します。 |
| `off` | 外部プロバイダーを無効にします（組み込みのみ）。 |

:::info プロバイダー固有のサブコマンド
外部メモリプロバイダーがアクティブなとき、プロバイダー固有の管理向けに独自のトップレベル `hermes <provider>` コマンドを登録する場合があります（例: Honchoがアクティブなときの `hermes honcho`）。非アクティブなプロバイダーはサブコマンドを公開しません。現在何が組み込まれているかを確認するには `hermes --help` を実行してください。
:::

## `hermes acp`

```bash
hermes acp
```

エディター統合向けに、HermesをACP（Agent Client Protocol）stdioサーバーとして起動します。

関連するエントリーポイント:

```bash
hermes-acp
python -m acp_adapter
```

最初にサポートをインストールしてください:

```bash
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['acp'], explicit=True)"
```

[ACPエディター統合](../user-guide/features/acp.md)と[ACP内部](/docs/developer-guide/acp-internals)を参照してください。

## `hermes mcp`

```bash
hermes mcp <subcommand>
```

MCP（Model Context Protocol）サーバー設定を管理します。Hermes自体をMCPサーバーとして実行する機能は `mcp-serve` プラグインに移動しました（`hermes plugins install mcp-serve` の後に `hermes mcp-serve`）。`hermes mcp serve` はその新しい書き方を出力します。

| サブコマンド | 説明 |
|------------|-------------|
| *(なし)* または `picker` | 対話型のカタログピッカー — Nousが承認したMCPをブラウズし、インストール/有効化/無効化します。 |
| `catalog` | Nousが承認したMCPを一覧表示します（プレーンテキストで、スクリプトから利用可能）。 |
| `install <name>` | カタログのエントリーをインストールします（例: `hermes mcp install deepwiki`）。 |
| `add <name> [--url URL] [--command CMD] [--auth oauth\|header] [--args ...]` | 自動ツール検出とともにカスタムMCPサーバーを追加します。`--args` は残りのargvをstdioコマンドに渡すため、最後に置いてください。 |
| `remove <name>`（エイリアス: `rm`） | 設定からMCPサーバーを削除します。 |
| `list`（エイリアス: `ls`） | 設定済みのMCPサーバーを一覧表示します。 |
| `test <name>` | MCPサーバーへの接続をテストします。 |
| `configure <name>`（エイリアス: `config`） | サーバーのツール選択を切り替えます。 |
| `login <name>` | OAuthベースのMCPサーバーに対して再認証を強制します。 |

[MCP設定リファレンス](./mcp-config-reference.md)、[HermesでMCPを使う](../guides/use-mcp-with-hermes.md)、[MCPサーバーモード](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server)を参照してください。

## `hermes connectors`

```bash
hermes connectors <subcommand>
```

ホスト型コネクターのアカウント（Nous Tool Gateway経由で利用するGmailなどのアプリ）を管理します。Nous Portalへのサインインが必要です。1つのアプリに複数の名前付きアカウントを持たせることができます。[複数のコネクターアカウント](../user-guide/features/tool-gateway.md#multiple-connector-accounts)を参照してください。

| サブコマンド | 説明 |
|------------|-------------|
| `list` | コネクターと、それぞれが接続済みかどうかを一覧表示します。 |
| `accounts [app] [--all]` | 1つのアプリ、またはすべてのアプリのアカウントを一覧表示します。`--all` は廃止されたアカウントも含みます。 |
| `connect <app> [--alias NAME] [--reconnect [--force]]` | アカウントを接続するためのリンクを出力し、接続されるか期限が過ぎるまで待機します。Ctrl-Cで待機を中止します。`--alias` はアカウントに名前を付けます。新しい名前を指定すると別のアカウントが追加されます。`--reconnect` は指定したアカウントを修復し、何も削除しません。すでに接続済みのアカウントに対しては、`--force` が指定されない限り、その旨を伝えて何もしません。 |
| `rename <app> <name-or-label> --to <new>` | アカウントをリネームします。名前のないアカウントは、アプリのラベルで照合されます。 |
| `disconnect <app> [--alias NAME] [--yes]` | アカウントを削除します。`--yes` を渡さない限り確認を求めます。 |

リネームには、それをサポートするバージョンのNous Portalが必要です。それまでは「This Nous Portal cannot rename connector accounts yet.」と報告されます。

## `hermes plugins`

```bash
hermes plugins [subcommand]
```

統合されたプラグイン管理 — 一般プラグイン、メモリプロバイダー、コンテキストエンジンを1か所で。サブコマンドなしで `hermes plugins` を実行すると、2つのセクションを持つ複合的な対話画面が開きます:

- **General Plugins** — インストール済みプラグインを有効化/無効化する複数選択チェックボックス
- **Provider Plugins** — メモリプロバイダーとコンテキストエンジンの単一選択設定。カテゴリーでENTERを押すとラジオピッカーが開きます。

| サブコマンド | 説明 |
|------------|-------------|
| *(なし)* | 複合的な対話UI — 一般プラグインの切り替え + プロバイダープラグインの設定。 |
| `install <identifier> [--force] [--ref COMMIT_SHA] [--allow-removed]` | Hermesプラグインカタログ（エントリー名のみ）、Git URL、または `owner/repo` の短縮形からプラグインをインストールします。カタログ名はレビュー済みの40桁16進のコミットSHAに解決され、宣言された機能が表示されます。ソース、チェックアウトされたリビジョン、ネストされたカタログ由来情報のブロックは、インストーラーの `plugins/.install-metadata.json` に記録されます。プラグインディレクトリ内の `.hermes-catalog.json` のコピーは利便性のためだけのもので、決して信頼されません。生のURLはカスタム（未レビュー）ソースとしてフラグ付けされます。`--ref` は完全なSHAによるカスタムの固定を選択し、実際にチェックアウトされたSHAを記録します。`--allow-removed`（危険）は、インストール時に削除済みプラグインのブロックリストを回避し、そのインストールを更新、有効化、読み込み時のキルリストのチェックから除外します。 |
| `search [term] [--json]` | Hermesプラグインカタログを検索します（エントリー名、説明、宣言されたツールに一致。`term` を省略するとすべてを一覧表示）。カタログはリポジトリ内（`plugin-catalog/`）でキュレーションされ、6時間のキャッシュでライブのリポジトリから更新され、オフライン時はツリー内のコピーにフォールバックします。カタログに載っていること ≠ 監査済みであること — 掲載時のレビューはエントリーを対象とし、コードは対象としません。 |
| `update <name>` | カタログからのインストールをレビュー済みのカタログSHAに再固定するか、カスタムのGitインストールを記録されたソース/フィードから更新します。PMはコードを公開する前に、アクティブなプラグインの依存関係を検証します。明示的なカスタムの固定は、`install --force --ref <new-commit>` によってのみ移動します。 |
| `update [--yes]` | 名前を指定しない場合: `update <name>` で適用できる更新（カタログの再固定またはGit pull）があるインストール済みプラグインをすべて一覧表示し、スキップされるものとその理由も示した上で `Update all N? [Y/n]` と尋ね、単一プラグインの手順で1つずつ更新します（プラグインごとの同意プロンプトは引き続き適用されます）。1つが失敗しても残りは停止しません。いずれかが失敗した場合は1で終了します。ターミナルがない場合、`--yes` が渡されない限り何も更新されません。`--yes` でも同意は安全側に倒れたままです（新しい機能は許可されないままです）。 |
| `remove <name>`（エイリアス: `rm`, `uninstall`） | インストール済みプラグインを削除します。 |
| `enable <name>` | 無効化されたプラグインを有効化します。 |
| `disable <name>` | プラグインを削除せずに無効化します。 |
| `list`（エイリアス: `ls`） | インストール済みプラグインを有効/無効ステータスとともに一覧表示します。 |
| `idle-tasks [name] [--pause TASK \| --resume TASK]` | このプロファイルでプラグインが `ctx.register_idle_task` によって登録したアイドルタスク（ステータス、実行回数、前回と次回の実行、前回の要約またはエラー）を一覧表示するか、指定したプラグインの1つのタスクを一時停止/再開します。 |
| `doctor [path-or-id] [--ci]` | ネイティブプラグインを、実際のマニフェストパーサー、ローダー、登録処理を通じて検証します。`--ci` はエラー時に1で終了します。 |
| `pack install <path-or-url> [--force]` | プラグインパック（`hermes-pack.yaml`）をインストールします — それぞれが正確な40文字のコミットSHAに固定されたプラグインの宣言的なセットです。必須のレビュー画面（すべてのプラグイン、ソース、固定された参照、宣言された機能）を表示し、パックの内容について1回確認を求めた後、通常の固定インストールを実行します。各プラグインの宣言された機能は、引き続き標準のプラグインごとの同意を経ます — パックが一括で許可することはありません。部分的な失敗はプラグインごとに報告され、いずれかのプラグインが失敗した場合は非ゼロで終了します。対話型のみです（`--yes` はありません）。 |
| `pack export [--enabled-only] [--name NAME]` | 現在のインストールからパックのYAMLをstdoutに出力します: gitでインストールされた各プラグインのリポジトリ + 正確なSHAと、サニタイズされたシークレットでない `plugins.entries` の設定です。ローカルのみのプラグイン（gitの由来情報がないもの）は、インストール可能なエントリーとしてではなく、警告のコメントとして記載されます。シークレット、機能の許可、`allow_*` のゲートは常に取り除かれます。 |
| `pack show <path-or-url>` | ドライラン: 何もインストールせずに、パックを解析、検証、表示します。 |

プロバイダープラグインの選択は `config.yaml` に保存されます:
- `memory.provider` — アクティブなメモリプロバイダー（空 = 組み込みのみ）
- `context.engine` — アクティブなコンテキストエンジン（`"compressor"` = 組み込みのデフォルト）

一般プラグインの無効化リストは `config.yaml` の `plugins.disabled` に保存されます。
Gitインストールでは、正規のソース、インストールされた正確なリビジョン、固定の状態のみが、
プロファイルローカルの `plugins/.install-metadata.json` サイドカーにも記録されます。
プラグインの設定、環境変数の値、シークレット、機能の許可は含まれません。

[プラグイン](../user-guide/features/plugins.md)と[Hermesプラグインを作る](../developer-guide/plugins/index.md)を参照してください。

## `hermes tools`

```bash
hermes tools [--summary]
```

| オプション | 説明 |
|--------|-------------|
| `--summary` | 現在の有効ツールの要約を出力して終了します。 |

`--summary` なしの場合、これはプラットフォームごとのツール設定の対話型UIを起動します。

## `hermes computer-use`

```bash
hermes computer-use <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `install` | 上流のcua-driverインストーラーを実行します（macOS、Windows、Linux）。 |
| `install --upgrade` | cua-driverがすでにPATH上にある場合でもインストーラーを再実行します。上流のスクリプトは常に最新のリリースを取得するため、これはその場でのアップグレードになります。 |
| `status` | `cua-driver` が `$PATH` 上にあるかどうかと、インストールされているバージョンを出力します。 |
| `doctor [--include CHECK] [--skip CHECK] [--json]` | cua-driverのヘルスレポートを実行し、そのプラットフォームチェックを表示します。 |
| `permissions status [--json]` | macOSのアクセシビリティと画面収録の許可状況を報告します。 |
| `permissions grant` | Cua Driverにアクセシビリティと画面収録を許可するようmacOSに要求します。 |

`hermes computer-use install` は、`computer_use` ツールセットが使用する
[cua-driver](https://github.com/trycua/cua) バイナリをインストールするための
安定したエントリーポイントです。Computer Useを初めて有効にしたときに
`hermes tools` が呼び出すのと同じ上流のインストーラーを実行するため、
ツールセットの切り替えがインストールをトリガーしなかった場合（例えば再訪
ユーザーのセットアップ時）に、インストールを再実行するのに安全に使用できます。

cua-driverがすでに存在する場合、Hermesはそのバージョンとランタイムマニフェストを
確認します。互換性のある0.20.0以降のインストールはそのまま残されます。古い、または
不完全な標準インストールは、現在の上流のインストーラーで修復されます。Hermesは、
`HERMES_CUA_DRIVER_CMD` で選択されたカスタムバイナリを決して置き換えません。その
バイナリを直接更新するか、上書き設定を削除してください。
修復が必要な場合は `hermes computer-use status` が報告します。

組み込みの `computer_use` ツールセットが推奨されるHermesの統合方法です。
Cuaの低レベルなツール語彙が必要な場合は、生のCua MCPツールを登録するという代替手段も
あります。`cua-driver skills install` はHermesを検出し、Cuaのスキルパックを
Hermesのスキルディレクトリに自動的にリンクします。

パーミッションモードと機能マニフェストの承認は、
ランタイムの起動時に扱われます。制限付きモードでは、HermesはCuaの正規の
`--capability-manifest` および `--approve-capability-manifest` フラグを渡します。すべてのMCP
トランスポートは、ランタイム内に専用のライフサイクルセッションを持ちます。公開のセッション
名はカーソルとセッションの状態にラベルを付けるものであり、ランタイムを所有したり共有したりはしません。

`hermes update` は、cua-driverがPATH上にある場合、更新の最後に上流のインストーラーを
自動的に再実行するため、ほとんどのユーザーは手動で `--upgrade` を呼び出す必要は
ありません。次のHermesの更新を待たずに、上流が出した修正をすぐに取り込みたい場合に
使用してください。

## `hermes clipboard`

```bash
hermes clipboard image --out PATH [--json]
```

CLIとTUIが画像の貼り付けに使用するのと同じ抽出処理で、クリップボードの画像を `PATH` に書き出します（作成または上書き）:
macOSでは `pngpaste`、次に `osascript`。WindowsではPowerShell（`powershell`、次に `pwsh`）。Linuxでは
WSL上の `powershell.exe`、次にWaylandでの `wl-paste`、次に `xclip`。結果はPNGですが、コピーされた画像*ファイル*
（Windows/WSLのファイルドロップ）はそのままコピーされます。このコマンドはこのマシンのクリップボードのみを
読み取ります: 実行中のゲートウェイは不要で、セッションストアやプロファイルのファイルも読まないため、
ゲートウェイが別のマシンで動作しているクライアントは、ユーザーがいるマシン上でこれを呼び出します。
NixOSのコンテナモードでは、他のすべての `hermes` サブコマンドと同様に、このコマンドは管理対象のコンテナ内で
実行されるため、ホストのクリップボードは見えず、`--out` はコンテナ内のパスになります。

| オプション | 説明 |
|--------|-------------|
| `--out PATH` | 書き込むファイル（必須）。親ディレクトリは作成されます。画像は同じディレクトリ内の一時ファイルに抽出されてから `PATH` にリネームされるため、何らかの失敗があっても既存の `PATH` は元のまま残ります。既存のファイルはパーミッションビットを保持します（0600の一時ファイルは非公開のままです）。`PATH` にあるシンボリックリンクは通常のファイルに置き換えられます。 |
| `--json` | stdoutに1つのJSONオブジェクトを出力します。これがない場合、成功時はパスを出力し、失敗時はstderrに1行を出力します。 |

| 終了コード | `reason` | 意味 |
|-----------|----------|---------|
| `0` | （なし、`ok: true`） | 画像が `path` に書き込まれました。 |
| `1` | `no_image` | クリップボードに画像がありません（またはPNGに変換できない画像です）。 |
| `3` | `no_tool` | このプラットフォームのクリップボードツールが1つもインストールされていません。`tools` には試行した順にそれらが一覧表示されます。 |
| `4` | `write_failed` | `--out` に書き込めません: ディレクトリが書き込み不可または作成できない、既存の `PATH` がディレクトリまたは読み取り専用である、または最後のリネームに失敗しました。 |
| `2` | （JSONなし） | 引数パーサーによる使用法エラー。 |

`--json` の応答（キーは安定しています。新しいキーが追加される場合があります。`path` は `ok` がtrueの場合にのみ読み取ってください）:

```json
{"ok": true, "path": "/home/me/paste.png", "mime": "image/png", "bytes": 48213}
{"ok": false, "reason": "no_image", "message": "the clipboard holds no image"}
{"ok": false, "reason": "no_tool", "tools": ["wl-paste", "xclip"], "message": "no clipboard tool found; install one of: wl-paste, xclip"}
```

## `hermes pets`

```bash
hermes pets <list|install|select|show|off|scale|remove|doctor>
```

[Petdex](https://github.com/crafter-station/petdex) は、コーディングエージェント向けのアニメーションスプライトのペットを集めた公開ギャラリーです。1つインストールすると、HermesはCLI、TUI、デスクトップアプリ全体で、エージェントのアクティビティに反応するペットを表示します。

| サブコマンド | 説明 |
|------------|-------------|
| `list` | petdexのギャラリーをブラウズします。 |
| `install` | ギャラリーからペットをインストールします。 |
| `select` | アクティブなペットを設定します（`display.pet.*` に書き込みます）。 |
| `show` | アクティブなペットをターミナルでアニメーション表示します。 |
| `off` | ペットの表示を無効にします。 |
| `scale` | すべての場所でペットのサイズを変更します（`display.pet.scale`）。 |
| `remove` | インストール済みのペットを削除します。 |
| `doctor` | ペットのセットアップとターミナルのグラフィックスサポートを確認します。 |

`/hatch` スラッシュコマンドを使えば、テキストの説明からまったく新しいペットを生成することもできます。[ペット](/docs/user-guide/features/pets)を参照してください。

## `hermes sessions`

```bash
hermes sessions <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `list` | 最近のセッションを一覧表示します。 |
| `browse` | 検索と再開ができる対話型セッションピッカー。各行には、ライフサイクルのステータスタグ（`done` / `intr` / `err` / `empty`。セッションの最後のメッセージから導出）とメッセージ数が表示されます。ハイライトされた行で（検索フィルターが空の間に）`d` を押すと、y/Nの確認の後にそのセッションを削除します。フィルターがアクティブな間は、`d` は検索欄に入力されます。 |
| `export <output> [--session-id ID]` | セッションをJSONLにエクスポートします。 |
| `delete <session-id>` | 1つのセッションを削除します。 |
| `prune` | フィルターに一致するセッションを削除します: 時間の範囲 `--older-than`/`--newer-than`/`--before`/`--after`（`5h`/`2d` のような期間、日数のみ、またはISOタイムスタンプ）。属性 `--source`、`--title`、`--model`、`--provider`、`--branch`、`--end-reason`、`--user`、`--chat-id`、`--chat-type`、`--cwd`。数値の範囲 `--min/--max-messages`、`--min/--max-tokens`、`--min/--max-cost`、`--min/--max-tool-calls`。さらに `--include-archived`、`--dry-run`、`--yes`。デフォルト: 90日より古いもの。 |
| `archive` | `prune` と同じフィルターに一致するセッションを一括でアーカイブします（ソフトに非表示にするだけで、削除はしません）。少なくとも1つのフィルターが必要です。 |
| `stats` | セッションストアの統計を表示します。 |
| `rename <session-id> <title>` | セッションタイトルを設定または変更します。 |
| `optimize` | ディスク容量を回収します: FTS5インデックスのセグメントをマージ + VACUUM。非破壊的で、セッションのデータは変更されません。 |
| `optimize-storage` | 全文検索インデックスをコンパクトなv23の外部コンテンツ形式に移行します。大きなデータベースでは、`state.db` のかなりの部分を回収できます。 |
| `repair` | 不正な形式の `state.db` スキーマ（例: `table messages_fts already exists`）を修復し、隠れていたセッションが再び表示されるようにします。最初にバックアップが作成されます。 |
| `repair-routing` | ルーティングのIDを失ったセッション行に取り残されたゲートウェイの会話を再接続します（再起動後にチャットが「過去に戻る」現象）。デフォルトはドライランです。`--apply` で引き継ぎを実行します（先にゲートウェイを停止してください）。`--max-gap-seconds N` で連続性の判定ウィンドウを調整します。曖昧さのないケースのみが修復されます。[セッション → 取り残されたゲートウェイセッションの修復](../user-guide/sessions.md#repair-stranded-gateway-sessions)を参照してください。 |
| `repair-prompts` | #122822以前のメンテナンス時のコンパクションのバグによって劣化したことが証明できる、保存済みのシステムプロンプトを報告します。デフォルトは報告のみです。`--apply` は検証済みの行をクリアして次のターンで再構築されるようにし、`--json` は機械可読の出力を行い（`--apply` と組み合わせると非対話型になります）、明示的な `session_id` の指定は正常なプロンプトでさえクリアしうる破壊的な上書きです。読み取り可能なtools[]の固定がない行、またはメモリのみの固定を持つ行は、検証不能として報告され、スキャンでは変更されません（メモリのみのセッションを再開すると完全なツール面が再固定され、その後のスキャンでクリアできるようになります）。修復した行を反映させるには、`--apply` の後に実行中のゲートウェイを再起動してください。[セッション → 劣化した保存済みプロンプトの修復](../user-guide/sessions.md#repair-degraded-stored-prompts)を参照してください。 |
| `refresh-prompts` | 継続中のセッションが、次のターンで現在の `SOUL.md`、スキル、コンテキストファイルから保存済みのシステムプロンプトを再構築するようにします。会話履歴とセッションIDは保持されます。プロファイルのプロンプトのソースを更新した後に実行してください。`--session ID`（IDまたは一意なプレフィックス）は要求を1つのセッションに限定し、`--json` は記録されたエポックとセッション数を出力し、`-p/--profile` は別のプロファイルを対象にします。ゲートウェイの実行中でも動作し（再起動は不要）、再起動後も維持されます。更新された各セッションでは、プロンプトキャッシュのミスが1回発生します。[セッション → プロファイル更新後のプロンプトの再構築](../user-guide/sessions.md#refresh-prompts-after-a-profile-update)を参照してください。 |
| `repair-profiles` | 誤ったプロファイルの下に入ってしまったセッション、ルーティング、Telegramのトピック、音声モードの状態を整理します（別のプロファイルのストアにある行、セッションキーと一致しないラベル、プロファイルをまたぐ親リンク、削除されたプロファイルのインデックス行）。デフォルトはドライランです。`--apply` はすべてのストアのスナップショットを取った後に修復を実行します（先にゲートウェイを停止してください）。`--legacy-main rekey\|move` は、名前付きプロファイルのストア内にある `agent:main` の行をどう扱うかを決めます。自動化には `--json`。[セッション → プロファイル間で混在した状態の修復](../user-guide/sessions.md#repair-state-crossed-between-profiles)を参照してください。 |
| `recover` | 破損した `state.db` を、オフラインかつ非破壊的に、別のクリーンなデータベースへ復旧します。 |
| `retitle-skills` | `/skill` で開始されたセッションのタイトルを、ユーザーが実際に入力した内容を使って再生成します。`--apply` が渡されない限り、変更内容を一覧表示するだけです。 |

## `hermes insights`

```bash
hermes insights [--days N] [--source platform]
```

| オプション | 説明 |
|--------|-------------|
| `--days <n>` | 過去 `n` 日を分析します（デフォルト: 30）。 |
| `--source <platform>` | `cli`、`telegram`、`discord` などのソースでフィルタリングします。 |

## `hermes import-agent`

```bash
hermes plugins install agent-importers   # once; the command comes from this official plugin
hermes import-agent [openclaw|claude-code|codex|cursor] [options]
```

**OpenClaw**（`~/.openclaw`）、**Claude Code**（`~/.claude`）、**OpenAI Codex CLI**（`~/.codex`）、**Cursor**（`~/.cursor`）のセットアップをHermesにインポートします: 指示とルールはメモリのエントリーになり、`Bash(...)` のパーミッションルールは `command_allowlist`/`approvals.deny` になり、MCPサーバーは `mcp_servers` に入り、スキルはコピーされます。常に最初にプレビューを表示します。APIキーと認証情報は決してインポートされません。ただし、`--migrate-secrets` を渡した場合のOpenClawの許可リストにあるキーは例外です。`--sync` は、登録済みのClaude Code、Codex、Cursorのソースのうち変更があったものを再インポートします。新規インストールでは、`hermes setup` がこれらのエージェントのいずれかを見つけると、プラグインをインストールしてインポートを実行することを提案します。

このコマンドは、`hermes claw`（OpenClawの移行）とともに、コアから公式の `agent-importers` プラグインに移動しました。プラグインは `hermes claw` の書き方も維持しています。プラグインがない場合、`hermes import-agent` と `hermes claw` はインストール用のコマンドを出力します。オプションと対応表については、**[インポートガイド](/docs/user-guide/import-from-other-agents)**を参照してください。

### OpenClaw: `hermes import-agent openclaw`

```bash
hermes import-agent openclaw [options]    # same as the older spelling: hermes claw migrate [options]
hermes claw cleanup [--source <path>] [--dry-run] [--yes]
```

OpenClawのセットアップをHermesに移行します。`~/.openclaw`（またはカスタムパス）から読み取り、`~/.hermes` に書き込みます。レガシーなディレクトリ名（`~/.clawdbot`、`~/.moltbot`）と設定ファイル名（`clawdbot.json`、`moltbot.json`）を自動的に検出します。

| オプション | 説明 |
|--------|-------------|
| `--dry-run` | 何も書き込まずに、移行される内容をプレビューします。 |
| `--preset <name>` | 移行プリセット: `full`（すべての互換設定）または `user-data`（インフラ設定を除外）。どちらのプリセットもシークレットはインポートしません — 明示的に `--migrate-secrets` を渡してください。 |
| `--overwrite` | 競合時に既存のHermesファイルを上書きします（デフォルト: プランに競合がある場合は適用を拒否）。 |
| `--migrate-secrets` | 移行にAPIキーを含めます。`--preset full` の下でも必要です。 |
| `--no-backup` | 移行前の `~/.hermes/` のzipスナップショットをスキップします（デフォルトでは、適用前に単一の復元ポイントアーカイブが `~/.hermes/backups/pre-migration-*.zip` に書き込まれます。`hermes import` で復元可能）。 |
| `--source <path>` | カスタムのOpenClawディレクトリ（デフォルト: `~/.openclaw`）。 |
| `--workspace-target <path>` | ワークスペース指示（AGENTS.md）のターゲットディレクトリ。 |
| `--skill-conflict <mode>` | スキル名の衝突を処理します: `skip`（デフォルト）、`overwrite`、または `rename`。 |
| `--yes` | 確認プロンプトをスキップします。 |

`hermes claw cleanup` は、移行が完了した後に残っているOpenClawのディレクトリを `*.pre-migration` にリネームします（これ以降OpenClawは動作しなくなります。元に戻すには名前を戻してください）。

#### 移行される内容 {#what-gets-migrated}

移行は、ペルソナ、メモリ、スキル、モデルプロバイダー、メッセージングプラットフォーム、エージェント動作、セッションポリシー、MCPサーバー、TTSなど、30以上のカテゴリーをカバーします。項目は、Hermesの相当機能に**直接インポートされる**か、手動レビュー用に**アーカイブされる**かのいずれかです。

**直接インポートされるもの:** SOUL.md、MEMORY.md、USER.md、AGENTS.md、スキル（4つのソースディレクトリ）、デフォルトモデル、カスタムプロバイダー、MCPサーバー、メッセージングプラットフォームのトークンと許可リスト（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost）、エージェントのデフォルト（推論努力、圧縮、ヒューマンディレイ、タイムゾーン、サンドボックス）、承認ルール、TTS設定、ブラウザ設定、ツール設定、execタイムアウト、コマンド許可リスト、ゲートウェイ設定、3つのソースからのAPIキー。

**手動レビュー用にアーカイブされるもの:** cronジョブ、プラグイン、フック/webhook、メモリバックエンド（QMD）、スキルレジストリ設定、UI/アイデンティティ、ロギング、マルチエージェントセットアップ、チャンネルバインディング、IDENTITY.md、TOOLS.md、HEARTBEAT.md、BOOTSTRAP.md。

**APIキーの解決**は、優先順位順に3つのソースをチェックします: 設定値 → `~/.openclaw/.env` → `auth-profiles.json`。すべてのトークンフィールドは、プレーンな文字列、環境変数テンプレート（`${VAR}`）、SecretRefオブジェクトを処理します。

完全な設定キーのマッピング、SecretRefの処理の詳細、移行後のチェックリストについては、**[完全な移行ガイド](../guides/migrate-from-openclaw.md)**を参照してください。

#### 例

```bash
# Preview what would be migrated
hermes import-agent openclaw --dry-run

# Full migration (all compatible settings, no secrets)
hermes import-agent openclaw --preset full

# Full migration including API keys
hermes import-agent openclaw --preset full --migrate-secrets

# Migrate user data only (no secrets), overwrite conflicts
hermes import-agent openclaw --preset user-data --overwrite

# Migrate from a custom OpenClaw path
hermes import-agent openclaw --source /home/user/old-openclaw
```

## `hermes serve`

```bash
hermes serve [options]
```

Hermesの**バックエンドサーバー** — [デスクトップアプリ](/docs/user-guide/desktop)やリモートクライアントが接続するJSON-RPC/WebSocketゲートウェイ — を起動します。これは `hermes dashboard` が実行するのと同じサーバーですが、**ヘッドレス**です: ブラウザーのUIを開くことはありません。デスクトップアプリは独自の `hermes serve` バックエンドを起動します。リモートホスト上でヘッドレスのバックエンドが必要な場合に、このコマンドを直接使用してください。下記の `hermes dashboard` と同じ `--host` / `--port` / `--insecure` / `--skip-build` / `--stop` / `--status` オプションを受け付けます（ループバック以外へのバインドでは同じ認証ゲートが有効になります）。`[web]` extraが必要です。組み込みのChatソケットには、POSIXホスト上でさらに `[pty]` が必要です。

**ポートの競合:** 要求されたポート（デフォルト `9119`）がすでに別のプロセス（例: 2つ目の `hermes serve` やゲートウェイ）に使われている場合、コマンドは汎用的なエラーではなく、機械可読のセンチネル行 `BACKEND_PORT_IN_USE port=<port>` をstdoutに、使用している可能性が高いプロセスを示す人間向けのヒントを出力し、コード **75**（`EX_TEMPFAIL`）で終了します — これにより、スクリプトやデスクトップアプリは「ポートが使用中」と「バックエンドが壊れている」を区別できます。空いているエフェメラルポートにバインドするには `--port 0` を渡してください（起動に成功すると、選ばれたポートが `HERMES_BACKEND_READY port=<port>` で通知されます）。

## `hermes dashboard`

```bash
hermes dashboard [options]
```

設定、APIキー、セッションを管理するためのWebダッシュボードを起動します。ヘッドレスのバックエンドには[`hermes serve`](#hermes-serve)を使用してください。FastAPI、Uvicorn、プラットフォームのPTYヘルパーはコアの依存関係です。`web` extraは正確なHTTPスタックの制約を追加するもので、標準のPMセットアップでは `all` を通じて選択されます。依存関係が壊れている場合は `hermes pm repair` を実行してください。組み込みのChatタブには、Linux、macOS、WSL2などのPOSIX PTY環境が必要です。[Webダッシュボード](../user-guide/features/web-dashboard.md)を参照してください。

| オプション | デフォルト | 説明 |
|--------|---------|-------------|
| `--port` | `9119` | Webサーバーを実行するポート |
| `--host` | `127.0.0.1` | バインドアドレス |
| `--no-open` | — | ブラウザを自動的に開かない |
| `--insecure` | off | **非推奨 / 何もしません。** 以前はループバック以外へのバインド時に認証を回避していました。2026年6月のセキュリティ強化以降、公開バインドには*常に*認証プロバイダー（パスワードまたはOAuth）が必要です。ローカルに留めるには `127.0.0.1` にバインドしてトンネルを使用してください。 |
| `--skip-build` | off | Web UIのビルドステップをスキップし、既存の `dist` を直接配信します。pnpmが利用できない非対話型のコンテキスト（Windowsのスケジュールされたタスク、CI）で便利です。`cd web && pnpm run build` で事前にビルドしてください。 |
| `--isolated` | off | 名前付きプロファイル（`worker dashboard`）から起動した場合に、マシンのダッシュボードにルーティングする代わりに、プロファイル専用のサーバーを実行します。そのプロファイルのみが一覧表示されます。ルーティング/運用上の利便性のためのものであり、セキュリティ境界では**ありません**: 同じOSユーザーのプロファイル同士は互いを読み取れます。テナントを分離するには、別々のOSユーザーやコンテナを使用してください。 |
| `--stop` | — | **このHermesホームの**実行中の `hermes dashboard` / `hermes serve` バックエンドを停止して終了します（どれを対象にするかは `-p <profile>` / `HERMES_HOME` で選択します。他のプロファイルのバックエンド、マシン上の他のインストール、コマンドを入力したシェルには決して触れません — オーナーを読み取れないバックエンドはそのままにされます）。SIGTERMを送り、10秒の猶予の後にSIGKILLを送ります。バックエンドより長く生き残ったホスト型のChat TUIも停止されます（そうしないと、削除された `state.db-wal` を開いたままにして次回の起動を妨げてしまうためです）。ダッシュボードから起動したメッセージングゲートウェイのボットには触れません。 |
| `--status` | — | 実行中の `hermes dashboard` プロセスを一覧表示して終了します。 |

### `hermes dashboard register`

このインストールを、Nous Portalアカウントのセルフホスト型ダッシュボードとして登録します。OAuthクライアントを作成し、`HERMES_DASHBOARD_OAUTH_CLIENT_ID` を `~/.hermes/.env` に書き込み、ログインゲートを有効にする方法を出力します。ログインしている必要があります（`hermes setup`）。

| オプション | 説明 |
|--------|-------------|
| `--name` | ダッシュボードの人間が読めるラベル（デフォルト: 自動生成）。 |
| `--redirect-uri` | 公開HTTPSのOAuthリダイレクトURI（例: `https://hermes.example.com/auth/callback`）。localhostのみで使用する場合は省略します。 |
| `--portal-url` | 登録に使うNous PortalのベースURLを上書きします（デフォルト: ログインしたポータル）。`HERMES_DASHBOARD_PORTAL_URL` でも設定できます。 |

```bash
# Default — opens browser to http://127.0.0.1:9119
hermes dashboard

# Custom port, no browser
hermes dashboard --port 8080 --no-open

# From a profile alias — routes to the machine dashboard with the
# profile preselected in the sidebar switcher (attach if running)
worker dashboard
```

## `hermes webapp`

```bash
hermes webapp [options]
```

Hermes Desktopのワークスペースを通常のブラウザーで開きます。設定で `webapp.enabled` が `true` の場合、ゲートウェイは
独自のリスナーでこれを配信します（[設定 → Webapp](../user-guide/configuration.md#webapp)）。
このコマンドは、Webappのバンドルが古い場合はビルドし、ゲートウェイが実行中でなければ起動し、
使い捨てのサインインリンクを開きます。チャットはそのゲートウェイ上で実行されます。これはCLI、Desktop、メッセージングが
使用するのと同じゲートウェイです。[デスクトップアプリ → ブラウザーからDesktopワークスペースを使う](/docs/user-guide/desktop#use-the-desktop-workspace-from-a-browser)を参照してください。

リンクの有効期限は60秒で、1回だけ使用できます。ブラウザーはリンクを、そのタブで保持するサインインと交換します。
リンクを取得できるのは自分のOSユーザーだけです: ゲートウェイは非公開のコントロールソケット経由でこのコマンドに
応答します。Linuxでは、ゲートウェイはブラウザーの接続が自分のOSユーザーに属していることを確認します。
Windowsでは何も確認しませんが、リンクはブラウザーランチャーのコマンドラインでのみ渡され、他の標準ユーザーは
それを読み取れません。macOS、BSD系、illumosではどちらも成り立たないため、コマンドはリンクをブラウザーランチャーに
渡すことはありません: 自分のOSユーザーだけが読めるページ（ゲートウェイホスト内の `webapp/launch-*.html`）に
リンクを書き込み、そのページを開きます。ブラウザーを開けない場合やWSLでは、自分で開けるようリンクを出力します。
macOSでは、見えない場所でブラウザーが開くことがあります（SSH経由で、Mac本体の画面上など）: どこにも開かなかった
場合は、`hermes webapp --no-open` を実行してリンクを取得してください。リンクは使用されるまで、ブラウザーをあなたとして
サインインさせるものなので、他人に知られないようにしてください。

デフォルトのゲートウェイが配信する名前付きプロファイル（`hermes -p work webapp`）の場合、Webappはその
プロファイルで開きます。

ダッシュボードのサインインゲートの背後にある場合（`webapp.host: 0.0.0.0` または `::`、あるいは `webapp.public_url` が設定済み）、
コマンドは使い捨てのリンクではなく、サインインの戻り先となるWebappのアドレスを開きます:
`webapp.public_url` が設定されていればそのURL。LAN上では、Web Access StartがNousのサインインを登録したアドレス
（`webapp/sign-in-registration.json`）で、それがまだこのマシンのLANアドレスとWebappのポートである場合。それ以外の場合は
ループバックアドレスで、パスワードプロバイダーや、そのために手動で設定されたサインインを提供します。そこでダッシュボードの
アカウントでサインインします。ゲートウェイがWebappを配信できない場合、コマンドはその理由を示します。ゲートウェイは起動時に
Webappの設定を読み取るため、原因を修正した後は `hermes gateway restart` を実行してください。まだビルドされていない
Webappのバンドルは例外です: このコマンドがビルドし、実行中のゲートウェイがすぐにそれを配信します。

| オプション | 説明 |
|--------|-------------|
| `--no-open` | ブラウザーを開く代わりに、使い捨てのリンク（サインインゲートの背後ではアドレス）を出力します |
| `--skip-build` | 既存の `apps/desktop/dist-webapp` をそのまま使用します |
| `--force-build` | ビルドレシートが最新であってもバンドルを再ビルドします |
| `--build-only` | バンドルをビルドして停止します |
| `--status` | 実行中のゲートウェイがWebappを配信しているかどうかと、そのアドレスを表示します |
| `--sign-out` | すべてのブラウザーをサインアウトします: リンクでサインインしたタブには新しいリンクが必要になり、サインインゲートの背後ではすべてのログインで再度サインインが必要になります。それらのログインが保持していたアカウントトークンも、Webappでは機能しなくなります。独自にサインインしたネイティブアプリはサインインしたままです。それらの接続は閉じられ、コマンドはその数を報告します。まだ起動中のゲートウェイは待機します。実行中のゲートウェイがない場合は、保存されている認証情報を置き換えます |

```bash
# Turn it on once, then open it
hermes config set webapp.enabled true
hermes gateway restart
hermes webapp
```

## `hermes profile`

```bash
hermes profile <subcommand>
```

プロファイルを管理します — それぞれ独自の設定、セッション、スキル、ホームディレクトリを持つ、複数の分離されたHermesインスタンス。

| サブコマンド | 説明 |
|------------|-------------|
| `list` | すべてのプロファイルを一覧表示します。 |
| `use <name>` | 固定デフォルトのプロファイルを設定します。 |
| `create <name> [--clone] [--clone-all] [--clone-from <source>] [--no-alias] [--isolated-auth] [--no-credentials]` | 新しいプロファイルを作成します。デフォルトでは、アクティブなプロファイルのAPIキー（ボットトークンは除く）とモデルを受け取り、デフォルトプロファイルのログインを共有します（名前付きプロファイルから作成した場合は、代わりにそのプロファイルのログインがコピーされます）。`--isolated-auth` は代わりにログインをコピーします。`--no-credentials` はこれらを一切コピーしません。`--clone` はアクティブなプロファイルから設定、`.env`、`SOUL.md`、スキル、キュレーションされた `MEMORY.md`/`USER.md` メモリファイルをコピーします。`--clone-all` はすべての状態をコピーします。`--clone-from` はソースプロファイルを指定し、`--clone-all` と組み合わせない限り設定のクローンを含意します。 |
| `delete <name> [-y]` | プロファイルを削除します。 |
| `show <name>` | プロファイルの詳細（ホームディレクトリ、設定など）を表示します。 |
| `alias <name> [--remove] [--name NAME]` | 素早いプロファイルアクセス用のラッパースクリプトを管理します。 |
| `rename <old> <new>` | プロファイルをリネームします。 |
| `export <name> [-o FILE]` | プロファイルを `.tar.gz` アーカイブにエクスポートします（ローカルバックアップ）。 |
| `import <archive> [--name NAME]` | `.tar.gz` アーカイブからプロファイルをインポートします（ローカル復元）。 |
| `install <source> [--name N] [--alias] [--force] [-y]` | git URLまたはローカルディレクトリからプロファイルディストリビューションをインストールします。 |
| `update <name> [--force-config] [-y]` | ディストリビューションを再取得します。ユーザーデータ（メモリ、セッション、auth）を保持します。 |
| `info <name>` | プロファイルのディストリビューションマニフェスト（バージョン、要件、ソース）を表示します。 |

例:

```bash
hermes profile list
hermes profile create work --clone
hermes profile create work --clone --sync-imports   # also carry over the import-agent sync manifest
hermes profile use work
hermes profile alias work --name h-work
hermes profile export work -o work-backup.tar.gz
hermes profile import work-backup.tar.gz --name restored
hermes profile install github.com/user/my-distro --alias
hermes profile update work
hermes -p work chat -q "Hello from work profile"
```

## `hermes completion`

```bash
hermes completion [bash|zsh|fish]
```

シェル補完スクリプトをstdoutに出力します。シェルプロファイルで出力をsourceすると、Hermesのコマンド、サブコマンド、プロファイル名のタブ補完が有効になります。

例:

```bash
# Bash
hermes completion bash >> ~/.bashrc

# Zsh
hermes completion zsh >> ~/.zshrc

# Fish
hermes completion fish > ~/.config/fish/completions/hermes.fish
```

## `hermes pm`

ピン留めされたツール、Python依存関係の環境、およびそれらの診断を管理します。
このコマンドはHermesアプリケーション自体を更新するものではありません。

```bash
hermes pm --help
hermes pm doctor
hermes pm status
hermes pm repair
hermes pm install
hermes pm install chromium
```

ソースからの開発では、セットアップスクリプトを一度実行した後、`source ./activate` または
PowerShellの `. .\activate.ps1` でインストールされた環境をアクティベートしてください。
以前のシェル環境に戻すには `deactivate` を使用します。準備、日常的なコマンド、依存関係の更新、
テスト環境については[開発者向けワークフロー](/docs/reference/package-management#developer-workflow)を
参照してください。

すべてのサブコマンド、ソースとバンドルでの動作の違い、遅延インストールのポリシー、メンテナー向けコマンドについては
[パッケージ管理](/docs/reference/package-management)を参照してください。

## `hermes update`

```bash
hermes update [--gateway] [--check] [--plan] [--no-backup] [--backup] [--yes]
```

許可されたソースチェックアウトを更新し、PMを通じて依存関係を準備します。
更新を適用せずに、設定されたソースのターゲットと比較するには `--check` を使用します。
Desktopのバンドル、Docker、Nix、Termuxのパッケージは、引き続き外部の更新管理者が
更新を担います。[更新とアンインストール](../getting-started/updating.md)を参照してください。

`hermes update` は、このインストールの更新チャンネルに従います。公式リポジトリのソースチェックアウトのデフォルトは `stable` で、公開された最新の `vX.Y.Z` GitHubリリースの正確なコミットです。フォークやミラーのデフォルトは `main` です。代わりに `main` のすべてのコミットに追従するには `hermes update --set-channel main` を実行します。`hermes update --set-channel stable` で元に戻せます。`main` チャンネルでは、別のブランチ上のチェックアウトは、pullの前に更新用のブランチに切り替えられることがあります。ブランチでの作業を更新の自動stashの流れの外に置いておきたい場合は、更新前にコミットしてください。

| オプション | 説明 |
|--------|-------------|
| `--install-id` | このインストールのIDとパスを出力して終了します。 |
| `--set-channel CHANNEL` | 更新を適用せずに、このソースインストールの更新チャンネルを永続化します: `stable`（公開リリース）または `main`（すべてのコミット）。バンドルされたアプリケーションはビルドチャンネルが固定されており、チャンネルの変更を拒否します。 |
| `--channel CHANNEL` | この実行に限り、ソースのチャンネル（`stable` または `main`）を選択します。何も保存されません。 |
| `--branch NAME` | この実行のソースブランチを選択します。ソースのチャンネル選択より優先されます。 |
| `--gateway` | メッセージングの `/update` コマンドが使用する内部モード。ターミナルのstdinから読み取る代わりに、プロンプトと進捗のストリーミングにファイルベースのIPCを使用します。ゲートウェイを再起動するフラグではありません。 |
| `--check` | pull、依存関係のインストール、再起動を一切行わずに、更新が利用可能かどうかを確認します。 |
| `--plan` | 何も変更せずに更新プランを出力して終了します: インストールの種類（git/Docker/Nix/apt）、すべてのプロファイルにわたる実行中のすべてのHermesサービスとそのスーパーバイザーおよび実行中のコードのバージョン、そしてそれぞれがどのように再起動されるか。イメージやパッケージで管理されたインストールでは、代わりに正しい外部の更新コマンドを報告します。読み取り専用です。 |
| `--no-backup` | `updates.pre_update_backup` に関係なく、この実行でのすべての更新前バックアップ（クイックの状態スナップショットと完全なzipの両方）をスキップします。 |
| `--backup` | この実行で**完全な**更新前バックアップを強制します: クイックの状態スナップショットに加えて、`HERMES_HOME`（設定、auth、セッション、スキル、ペアリングデータ）の完全なzipです。デフォルトのモードは `quick` — 軽量な状態スナップショットのみです。恒久的なモードは `config.yaml` の `updates.pre_update_backup: quick | full | off` で設定します。 |
| `--yes`, `-y` | 設定の移行やstashの復元などの対話型プロンプトにyesと答えたものとみなします。APIキーの入力はスキップされます。それらについては別途 `hermes config migrate` を実行してください。 |

追加の動作:

- **ゲートウェイの再起動。** 更新が成功すると、Hermesは更新対象のホーム（そのルートと、その配下のすべての `profiles/<name>`）で実行中のすべてのゲートウェイプロファイルを自動的に再起動し、新しいコードを読み込ませようとします。同じマシン上の別の `HERMES_HOME` に属するゲートウェイや `hermes-gateway*` サービス — 別のインストール、または `hermes update` を実行しているスクラッチ用のホーム — は出力に名前が示され、そのままにされます。更新を適用せずにゲートウェイを再起動したい場合は `hermes gateway restart` を使用してください。
- **再起動フェーズの回復。** 新たにpullしたツリーのインポート中にプロセス内の再起動フェーズが中断した場合、監視下のゲートウェイプロファイルはクリーンなPythonプロセスを通じて再試行されます。systemdによって独立に確認された（`systemctl --user is-active`）再起動のみが検証済みとして報告されます。単に0で終了しただけの再起動は `relaunch_attempted` として記録され、再起動は未完了のまま残ります（`gateway_restart` のフォローアップ。コードは配置済みなので、更新自体は引き続き0で終了します）。手動のゲートウェイとserve/dashboardのランタイムは、再起動の権限がない限り決して強制終了されません。それらは理由とともにスキップとして記録され、正確な再起動コマンドとともに未完了の更新レポートに残ります。
- **更新レシートとフリートのバージョンチェック。** 実行のたびに、固定プロファイルからであっても、ルートのHermesホームの `~/.hermes/logs/update_receipts/` に機械可読のレシートが書き込まれます（更新前のフリートのプラン、ステップ、理由付きのスキップ、再起動の結果、フォローアップ。`latest.json` は最新のものを指し、更新の進行中は `running` を示します）。再起動フェーズの後、アップデーターは稼働中の各ゲートウェイの実行コードを更新後のチェックアウトと照合し、プロファイルごとのバージョンの一覧を出力します。新しいコードが配置された後は、ゲートウェイがまだ更新前のコードで動いていても更新は失敗しません: 更新は正確な再起動コマンドを含む `⚠` 行を出力して0で終了し、`gateway_restart` のフォローアップを記録して、再起動を未完了のまま保持します — CLIは起動のたびにそれを警告し、次回の `hermes update` が再試行します。
- **ローカルのソース変更。** gitインストールでは、ブランチのチェックアウトやpullの前に、変更のある追跡対象ファイルが、更新によって追加されるパスにある未追跡ファイルとともに自動的にstashされます。その他の未追跡のファイルやフォルダーは更新を妨げないため、手を付けずにそのまま残されます。対話型ターミナルでの更新では、何かをstashする前に「Keep these changes after updating?」と尋ねます: yesの場合はstashしてpullの後に元に戻し、noの場合はすぐに削除するため、stashのエントリーは残りません。pullするものがない実行では尋ねません。非対話型の更新ではデフォルトで復元されます。pullが成功した後にローカルのソースの編集を破棄すべき管理されたインストールでのみ、`updates.non_interactive_local_changes: discard` を設定してください。stashの復元が競合した場合やpullが失敗した場合、stashは手動で回復できるようそのまま残されます。
- **npmのロックファイルの変更。** stashやブランチの切り替えの前に、Hermesはnpmのインストールがnpmプロジェクト（WhatsAppブリッジとPhotonサイドカー）に残す、追跡対象の `package-lock.json` の差分をベストエフォートでクリーンアップします。`pnpm-lock.yaml` は決して元に戻されません: インストールがそれを書き換えることはないためです。意図的なロックファイルの編集は、`hermes update` を実行する前にコミットするか、手動でstashしてください。
- **ペアリングデータのスナップショット。** `--backup` がoffのときでも、`hermes update` は `git pull` の前に `~/.hermes/pairing/` とFeishuのコメントルールの軽量なスナップショットを取得します。pullが編集中のファイルを書き換えた場合、`hermes backup restore --state pre-update` でロールバックできます。
- **レガシー `hermes.service` の警告。** Hermesがリネーム前の `hermes.service` systemdユニット（現在の `hermes-gateway.service` ではなく）を検出した場合、フラップループの問題を避けられるよう、一度だけ移行のヒントを出力します。
- **終了コード。** 成功時 `0`、pull/インストール/ポストインストールのエラー時 `1`、`git pull` をブロックする予期しないワーキングツリーの変更時 `2`。

## メンテナンスコマンド {#maintenance-commands}

| コマンド | 説明 |
|---------|-------------|
| `hermes --version` | バージョン情報を出力します。 |
| `hermes update` | 最新の変更を取得して依存関係を再インストールします。 |

| `hermes uninstall [--full] [--gui] [--data] [--dry-run] [--yes]` | 所有するソースインストールのファイルを削除します。`--gui` はソースからビルドしたデスクトップの削除を選択します。`--full` はデータも削除します。`--data` はパッケージが所有するコードを削除せずにユーザーデータを削除します。封印されたインストールでは、アプリケーションの削除はそのパッケージ管理者が行います。`--dry-run` は範囲をプレビューします。`--yes` は確認をスキップします。 |

## 関連項目 {#see-also}

- [スラッシュコマンドリファレンス](./slash-commands.md)
- [CLIインターフェース](../user-guide/cli.md)
- [セッション](../user-guide/sessions.md)
- [スキルシステム](../user-guide/features/skills.md)
- [スキンとテーマ](../user-guide/features/skins.md)
