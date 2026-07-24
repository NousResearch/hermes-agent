---
sidebar_position: 1
title: "CLIコマンドリファレンス"
description: "Hermesのターミナルコマンドおよびコマンドファミリーの公式リファレンス"
---

# CLIコマンドリファレンス

このページでは、シェルから実行する**ターミナルコマンド**を扱います。

チャット内のスラッシュコマンドについては、[スラッシュコマンドリファレンス](./slash-commands.md)を参照してください。

## グローバルエントリーポイント

```bash
hermes [global-options] <command> [subcommand/options]
```

### グローバルオプション

| オプション | 説明 |
|--------|-------------|
| `--version`, `-V` | バージョンを表示して終了します。 |
| `--profile <name>`, `-p <name>` | この実行で使用するHermesプロファイルを選択します。`hermes profile use` で設定した固定デフォルトを上書きします。 |
| `--resume <session>`, `-r <session>` | IDまたはタイトルで以前のセッションを再開します。 |
| `--continue [name]`, `-c [name]` | 最新のセッション、またはタイトルが一致する最新のセッションを再開します。 |
| `--worktree`, `-w` | 並列エージェントワークフロー向けに、分離されたgit worktreeで開始します。 |
| `--yolo` | 危険なコマンドの承認プロンプトをバイパスします。 |
| `--pass-session-id` | エージェントのシステムプロンプトにセッションIDを含めます。 |
| `--ignore-user-config` | `~/.hermes/config.yaml` を無視し、組み込みのデフォルトにフォールバックします。`.env` の認証情報は引き続き読み込まれます。 |
| `--ignore-rules` | `AGENTS.md`、`SOUL.md`、`.cursorrules`、メモリ、プリロードされたスキルの自動注入をスキップします。 |
| `--tui` | クラシックCLIの代わりに[TUI](../user-guide/tui.md)を起動します。`HERMES_TUI=1` と同等です。 |
| `--dev` | `--tui` と併用時: プリビルドのバンドルではなく、`tsx` 経由でTypeScriptソースを直接実行します（TUIコントリビューター向け）。 |

## トップレベルコマンド

| コマンド | 目的 |
|---------|---------|
| `hermes chat` | エージェントとの対話的またはワンショットのチャット。 |
| `hermes model` | デフォルトのプロバイダーとモデルを対話的に選択します。 |
| `hermes fallback` | プライマリモデルがエラーになったときに試行されるフォールバックプロバイダーを管理します。 |
| `hermes gateway` | メッセージングゲートウェイサービスを実行または管理します。 |
| `hermes setup` | 設定の全体または一部を対象とする対話型セットアップウィザード。 |
| `hermes whatsapp` | WhatsAppブリッジを設定・ペアリングします。 |
| `hermes slack` | Slackヘルパー（現在: すべてのコマンドをネイティブスラッシュとして含むアプリマニフェストを生成）。 |
| `hermes auth` | 認証情報を管理します — 追加、一覧表示、削除、リセット、戦略の設定。Codex/Nous/AnthropicのOAuthフローを処理します。 |
| `hermes login` / `logout` | **非推奨** — 代わりに `hermes auth` を使用してください。 |
| `hermes status` | エージェント、認証、プラットフォームのステータスを表示します。 |
| `hermes cron` | cronスケジューラーを検査・実行します。 |
| `hermes kanban` | マルチプロファイルのコラボレーションボード（タスク、リンク、ディスパッチャー）。 |
| `hermes webhook` | イベント駆動の起動向けに、動的なWebhookサブスクリプションを管理します。 |
| `hermes hooks` | `config.yaml` で宣言されたシェルスクリプトフックを検査、承認、削除します。 |
| `hermes doctor` | 設定や依存関係の問題を診断します。 |
| `hermes dump` | サポート/デバッグ向けの、コピーペースト可能なセットアップ要約。 |
| `hermes debug` | デバッグツール — サポート向けにログとシステム情報をアップロードします。 |
| `hermes backup` | Hermesのホームディレクトリをzipファイルにバックアップします。 |
| `hermes checkpoints` | `~/.hermes/checkpoints/`（`/rollback` が使用するシャドウストア）を検査 / プルーニング / クリアします。引数なしで実行するとステータスの概要を表示します。 |
| `hermes import` | zipファイルからHermesバックアップを復元します。 |
| `hermes logs` | エージェント/ゲートウェイ/エラーのログファイルを表示、追跡、フィルタリングします。 |
| `hermes config` | 設定ファイルを表示、編集、移行、クエリします。 |
| `hermes pairing` | メッセージングのペアリングコードを承認または取り消します。 |
| `hermes skills` | スキルをブラウズ、インストール、公開、監査、設定します。 |
| `hermes curator` | バックグラウンドのスキルメンテナンス — ステータス、実行、一時停止、固定。[Curator](../user-guide/features/curator.md)を参照してください。 |
| `hermes memory` | 外部メモリプロバイダーを設定します。プロバイダー固有のサブコマンド（例: `hermes honcho`）は、そのプロバイダーがアクティブなときに自動的に登録されます。 |
| `hermes acp` | エディター統合向けにHermesをACPサーバーとして実行します。 |
| `hermes mcp` | MCPサーバー設定を管理し、HermesをMCPサーバーとして実行します。 |
| `hermes plugins` | Hermes Agentのプラグインを管理します（インストール、有効化、無効化、削除）。 |
| `hermes tools` | プラットフォームごとに有効なツールを設定します。 |
| `hermes computer-use` | cua-driverバックエンドをインストールまたは確認します（macOSのComputer Use）。 |
| `hermes sessions` | セッションをブラウズ、エクスポート、プルーニング、リネーム、削除します。 |
| `hermes insights` | トークン/コスト/アクティビティの分析を表示します。 |
| `hermes claw` | OpenClawの移行ヘルパー。 |
| `hermes dashboard` | 設定、APIキー、セッションを管理するためのWebダッシュボードを起動します。 |
| `hermes profile` | プロファイルを管理します — 複数の分離されたHermesインスタンス。 |
| `hermes completion` | シェル補完スクリプトを出力します（bash/zsh/fish）。 |
| `hermes version` | バージョン情報を表示します。 |
| `hermes update` | 最新のコードを取得して依存関係を再インストールします。`--check` はpullせずにコミットの差分を表示します。`--backup` はpull前の `HERMES_HOME` スナップショットを取得します。 |
| `hermes uninstall` | システムからHermesを削除します。 |

## `hermes chat`

```bash
hermes chat [options]
```

主なオプション:

| オプション | 説明 |
|--------|-------------|
| `-q`, `--query "..."` | ワンショットの非対話型プロンプト。 |
| `-m`, `--model <model>` | この実行のモデルを上書きします。 |
| `-t`, `--toolsets <csv>` | カンマ区切りのツールセットを有効にします。 |
| `--provider <provider>` | プロバイダーを強制指定します: `auto`, `openrouter`, `nous`, `openai-codex`, `copilot-acp`, `copilot`, `anthropic`, `gemini`, `google-gemini-cli`, `huggingface`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `kilocode`, `xiaomi`, `arcee`, `gmi`, `alibaba`, `alibaba-coding-plan`（エイリアス `alibaba_coding`）, `deepseek`, `nvidia`, `ollama-cloud`, `xai`（エイリアス `grok`）, `qwen-oauth`, `bedrock`, `opencode-zen`, `opencode-go`, `ai-gateway`, `azure-foundry`, `lmstudio`, `stepfun`, `tencent-tokenhub`（エイリアス `tencent`, `tokenhub`）。 |
| `-s`, `--skills <name>` | セッション向けに1つ以上のスキルをプリロードします（繰り返し指定またはカンマ区切り可）。 |
| `-v`, `--verbose` | 詳細出力。 |
| `-Q`, `--quiet` | プログラム実行モード: バナー/スピナー/ツールプレビューを抑制します。 |
| `--image <path>` | 単一のクエリにローカル画像を添付します。 |
| `--resume <session>` / `--continue [name]` | `chat` から直接セッションを再開します。 |
| `--worktree` | この実行向けに分離されたgit worktreeを作成します。 |
| `--checkpoints` | 破壊的なファイル変更の前にファイルシステムのチェックポイントを有効にします。 |
| `--yolo` | 承認プロンプトをスキップします。 |
| `--pass-session-id` | システムプロンプトにセッションIDを渡します。 |
| `--ignore-user-config` | `~/.hermes/config.yaml` を無視し、組み込みのデフォルトを使用します。`.env` の認証情報は引き続き読み込まれます。分離されたCI実行、再現可能なバグレポート、サードパーティ統合に便利です。 |
| `--ignore-rules` | `AGENTS.md`、`SOUL.md`、`.cursorrules`、永続メモリ、プリロードされたスキルの自動注入をスキップします。完全に分離された実行には `--ignore-user-config` と組み合わせてください。 |
| `--source <tag>` | フィルタリング用のセッションソースタグ（デフォルト: `cli`）。ユーザーのセッション一覧に表示すべきでないサードパーティ統合には `tool` を使用します。 |
| `--max-turns <N>` | 会話の1ターンあたりのツール呼び出し反復回数の上限（デフォルト: 90、または設定の `agent.max_turns`）。 |

例:

```bash
hermes
hermes chat -q "Summarize the latest PRs"
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
hermes chat --quiet -q "Return only JSON"
hermes chat --worktree -q "Review this repo and open a PR"
hermes chat --ignore-user-config --ignore-rules -q "Repro without my personal setup"
```

### `hermes -z <prompt>` — スクリプト用ワンショット

プログラムから呼び出す側（シェルスクリプト、CI、cron、プロンプトをパイプで渡す親プロセス）にとって、`hermes -z` は最も純粋なワンショットのエントリーポイントです。**単一のプロンプトを入力し、最終的なレスポンステキストのみを出力し、それ以外はstdoutにもstderrにも何も出しません。** バナーもスピナーもツールプレビューも `Session:` 行もなく、エージェントの最終的な返信をプレーンテキストとして出力するだけです。

```bash
hermes -z "What's the capital of France?"
# → Paris.

# 親スクリプトはレスポンスをきれいにキャプチャできます:
answer=$(hermes -z "summarize this" < /path/to/file.txt)
```

実行ごとの上書き（`~/.hermes/config.yaml` は変更しません）:

| フラグ | 同等の環境変数 | 目的 |
|---|---|---|
| `-m` / `--model <model>` | `HERMES_INFERENCE_MODEL` | この実行のモデルを上書きします |
| `--provider <provider>` | `HERMES_INFERENCE_PROVIDER` | この実行のプロバイダーを上書きします |

```bash
hermes -z "…" --provider openrouter --model openai/gpt-5.5
# または:
HERMES_INFERENCE_MODEL=anthropic/claude-sonnet-4.6 hermes -z "…"
```

同じエージェント、同じツール、同じスキルで、対話的 / 装飾的なレイヤーをすべて取り除くだけです。トランスクリプトにツール出力も必要な場合は、代わりに `hermes chat -q` を使用してください。`-z` は明示的に「最終的な答えだけが欲しい」場合のためのものです。

## `hermes model`

対話型のプロバイダー + モデルセレクター。**これは、新しいプロバイダーの追加、APIキーのセットアップ、OAuthフローの実行を行うためのコマンドです。** アクティブなHermesチャットセッションの中からではなく、ターミナルから実行してください。

```bash
hermes model
```

次のことをしたいときに使用します:
- **新しいプロバイダーを追加する**（OpenRouter、Anthropic、Copilot、DeepSeek、カスタムなど）
- OAuthベースのプロバイダーにログインする（Anthropic、Copilot、Codex、Nous Portal）
- APIキーを入力または更新する
- プロバイダー固有のモデルリストから選択する
- カスタム/セルフホストのエンドポイントを設定する
- 新しいデフォルトを設定に保存する

:::warning hermes model と /model — 違いを把握する
**`hermes model`**（Hermesセッションの外、ターミナルから実行）は、**完全なプロバイダーセットアップウィザード**です。新しいプロバイダーの追加、OAuthフローの実行、APIキーの入力プロンプト、エンドポイントの設定ができます。

**`/model`**（アクティブなHermesチャットセッション内で入力）は、**すでにセットアップ済みのプロバイダーとモデルの切り替え**のみができます。新しいプロバイダーの追加、OAuthの実行、APIキーの入力プロンプトはできません。

**新しいプロバイダーを追加する必要がある場合:** まずHermesセッションを終了し（`Ctrl+C` または `/quit`）、ターミナルのプロンプトから `hermes model` を実行してください。
:::

### `/model` スラッシュコマンド（セッション中）

セッションを離れずに、すでに設定済みのモデルを切り替えます:

```
/model                              # 現在のモデルと利用可能なオプションを表示
/model claude-sonnet-4              # モデルを切り替え（プロバイダーを自動検出）
/model zai:glm-5                    # プロバイダーとモデルを切り替え
/model custom:qwen-2.5              # カスタムエンドポイントのモデルを使用
/model custom                       # カスタムエンドポイントからモデルを自動検出
/model custom:local:qwen-2.5        # 名前付きカスタムプロバイダーを使用
/model openrouter:anthropic/claude-sonnet-4  # クラウドに戻す
```

デフォルトでは、`/model` の変更は**現在のセッションにのみ**適用されます。`--global` を追加すると、変更を `config.yaml` に永続化します:

```
/model claude-sonnet-4 --global     # 切り替えて新しいデフォルトとして保存
```

:::info OpenRouterのモデルしか表示されない場合は？
OpenRouterのみを設定している場合、`/model` はOpenRouterのモデルのみを表示します。別のプロバイダー（Anthropic、DeepSeek、Copilotなど）を追加するには、セッションを終了し、ターミナルから `hermes model` を実行してください。
:::

プロバイダーとベースURLの変更は、`config.yaml` に自動的に永続化されます。カスタムエンドポイントから切り替えると、古いベースURLが他のプロバイダーに漏れないようにクリアされます。

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `run` | ゲートウェイをフォアグラウンドで実行します。WSL、Docker、Termuxに推奨です。 |
| `start` | インストール済みのsystemd/launchdバックグラウンドサービスを開始します。 |
| `stop` | サービス（またはフォアグラウンドプロセス）を停止します。 |
| `restart` | サービスを再起動します。 |
| `status` | サービスのステータスを表示します。 |
| `install` | systemd（Linux）またはlaunchd（macOS）のバックグラウンドサービスとしてインストールします。 |
| `uninstall` | インストール済みのサービスを削除します。 |
| `setup` | 対話型のメッセージングプラットフォームセットアップ。 |

オプション:

| オプション | 説明 |
|--------|-------------|
| `--all` | `start` / `restart` / `stop` で: アクティブな `HERMES_HOME` だけでなく、**すべてのプロファイルの**ゲートウェイに対して動作します。複数のプロファイルを並行して実行していて、`hermes update` 後にそれらすべてを再起動したい場合に便利です。 |

:::tip WSLユーザー向け
`hermes gateway start` の代わりに `hermes gateway run` を使用してください — WSLのsystemdサポートは信頼性に欠けます。永続化のためにtmuxでラップしてください: `tmux new -s hermes 'hermes gateway run'`。詳細は[WSL FAQ](/docs/reference/faq#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails)を参照してください。
:::

## `hermes setup`

```bash
hermes setup [model|tts|terminal|gateway|tools|agent] [--non-interactive] [--reset] [--quick] [--reconfigure]
```

**初回実行時:** 初回セットアップウィザードを起動します。

**再訪ユーザー（設定済み）:** 完全な再設定ウィザードに直接入ります — 各プロンプトには現在の値がデフォルトとして表示され、Enterを押すと保持、新しい値を入力すると変更します。メニューはありません。

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

## `hermes whatsapp`

```bash
hermes whatsapp
```

モード選択とQRコードのペアリングを含む、WhatsAppのペアリング/セットアップフローを実行します。

## `hermes slack`

```bash
hermes slack manifest              # マニフェストをstdoutに出力
hermes slack manifest --write      # ~/.hermes/slack-manifest.json に書き込み
hermes slack manifest --slashes-only  # features.slash_commands 配列のみ
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
| `--slashes-only` | off | 手動でメンテナンスするマニフェストにマージするため、`features.slash_commands` のみを出力します。 |

`hermes update` の後、新しいコマンドを取り込むには `hermes slack manifest --write` を
再度実行してください。


## `hermes login` / `hermes logout` *(非推奨)*

:::caution
`hermes login` は削除されました。OAuth認証情報の管理には `hermes auth` を、プロバイダーの選択には `hermes model` を、完全な対話型セットアップには `hermes setup` を使用してください。
:::

## `hermes auth`

同一プロバイダーのキーローテーション向けに認証情報プールを管理します。完全なドキュメントは[認証情報プール](/docs/user-guide/features/credential-pools)を参照してください。

```bash
hermes auth                                              # 対話型ウィザード
hermes auth list                                         # すべてのプールを表示
hermes auth list openrouter                              # 特定のプロバイダーを表示
hermes auth add openrouter --api-key sk-or-v1-xxx        # APIキーを追加
hermes auth add anthropic --type oauth                   # OAuth認証情報を追加
hermes auth remove openrouter 2                          # インデックスで削除
hermes auth reset openrouter                             # クールダウンをクリア
hermes auth status anthropic                             # プロバイダーの認証ステータスを表示
hermes auth logout anthropic                             # ログアウトして保存済みの認証状態をクリア
hermes auth spotify                                      # PKCE経由でHermesをSpotifyに認証
```

サブコマンド: `add`、`list`、`remove`、`reset`、`status`、`logout`、`spotify`。サブコマンドなしで呼び出すと、対話型の管理ウィザードを起動します。

## `hermes status`

```bash
hermes status [--all] [--deep]
```

| オプション | 説明 |
|--------|-------------|
| `--all` | 共有可能な秘匿化フォーマットで、すべての詳細を表示します。 |
| `--deep` | 時間がかかる可能性のある、より深いチェックを実行します。 |

## `hermes cron`

```bash
hermes cron <list|create|edit|pause|resume|run|remove|status|tick>
```

| サブコマンド | 説明 |
|------------|-------------|
| `list` | スケジュール済みジョブを表示します。 |
| `create` / `add` | プロンプトからスケジュール済みジョブを作成し、必要に応じて繰り返しの `--skill` で1つ以上のスキルをアタッチします。 |
| `edit` | ジョブのスケジュール、プロンプト、名前、配信、繰り返し回数、アタッチされたスキルを更新します。`--clear-skills`、`--add-skill`、`--remove-skill` をサポートします。 |
| `pause` | ジョブを削除せずに一時停止します。 |
| `resume` | 一時停止したジョブを再開し、次回の将来の実行を計算します。 |
| `run` | 次のスケジューラーのティックでジョブをトリガーします。 |
| `remove` | スケジュール済みジョブを削除します。 |
| `status` | cronスケジューラーが実行中かどうかを確認します。 |
| `tick` | 実行予定のジョブを1回実行して終了します。 |

## `hermes kanban`

```bash
hermes kanban [--board <slug>] <action> [options]
```

マルチプロファイル、マルチプロジェクトのコラボレーションボード。各インストールは多数のボード（プロジェクト、リポジトリ、ドメインごとに1つ）をホストできます。各ボードは独自のSQLite DBとディスパッチャースコープを持つスタンドアロンのキューです。新規インストールは `default` という1つのボードで開始し、そのDBは後方互換性のため `~/.hermes/kanban.db` です。追加のボードは `~/.hermes/kanban/boards/<slug>/kanban.db` に置かれます。ゲートウェイに組み込まれたディスパッチャーは、ティックごとにすべてのボードをスイープします。

**グローバルフラグ（以下のすべてのアクションに適用）:**

| フラグ | 目的 |
|------|---------|
| `--board <slug>` | 特定のボードを操作します。デフォルトは現在のボード（`hermes kanban boards switch`、`HERMES_KANBAN_BOARD` 環境変数、または `default` で設定）です。 |

**これは人間 / スクリプト向けの操作面です。** ディスパッチャーによって生成されたエージェントワーカーは、`hermes kanban` をシェル実行する代わりに、専用の `kanban_*` [ツールセット](/docs/user-guide/features/kanban#how-workers-interact-with-the-board)（`kanban_show`、`kanban_complete`、`kanban_block`、`kanban_create`、`kanban_link`、`kanban_comment`、`kanban_heartbeat`）を通じてボードを操作します。ワーカーは環境に `HERMES_KANBAN_BOARD` が固定されているため、物理的に他のボードを見ることができません。

| アクション | 目的 |
|--------|---------|
| `init` | `kanban.db` が存在しない場合に作成します。冪等です。 |
| `boards list` / `boards ls` | タスク数とともにすべてのボードを一覧表示します。`--json`、`--all`（アーカイブ済みを含む）。 |
| `boards create <slug>` | 新しいボードを作成します。フラグ: `--name`、`--description`、`--icon`、`--color`、`--switch`（アクティブにする）。スラッグはケバブケースで、自動的に小文字化されます。 |
| `boards switch <slug>` / `boards use` | `<slug>` をアクティブなボードとして永続化します（`~/.hermes/kanban/current` に書き込みます）。 |
| `boards show` / `boards current` | 現在アクティブなボードの名前、DBパス、タスク数を出力します。 |
| `boards rename <slug> "<name>"` | ボードの表示名を変更します。スラッグは不変です。 |
| `boards rm <slug>` | ボードをアーカイブ（デフォルト）またはハード削除します。`--delete` はアーカイブステップをスキップします。アーカイブされたボードは `boards/_archived/<slug>-<ts>/` に移動します。`default` に対しては拒否されます。 |
| `create "<title>"` | アクティブなボードに新しいタスクを作成します。フラグ: `--body`、`--assignee`、`--parent`（繰り返し可）、`--workspace scratch\|worktree\|dir:<path>`、`--tenant`、`--priority`、`--triage`、`--idempotency-key`、`--max-runtime`、`--skill`（繰り返し可）。 |
| `list` / `ls` | アクティブなボードのタスクを一覧表示します。`--mine`、`--assignee`、`--status`、`--tenant`、`--archived`、`--json` でフィルタリングします。 |
| `show <id>` | コメントとイベントを含むタスクを表示します。機械可読出力には `--json`。 |
| `assign <id> <profile>` | 割り当てまたは再割り当てします。割り当て解除には `none` を使用します。タスクの実行中は拒否されます。 |
| `link <parent> <child>` | 依存関係を追加します。循環は検出されます。両方のタスクは同じボードにある必要があります。 |
| `unlink <parent> <child>` | 依存関係を削除します。 |
| `claim <id>` | 準備完了のタスクをアトミックにクレームします。解決されたワークスペースパスを出力します。 |
| `comment <id> "<text>"` | コメントを追加します。次にタスクをクレームしたワーカーが、`kanban_show()` のレスポンスの一部としてそれを読み取ります。 |
| `complete <id>` | タスクを完了としてマークします。フラグ: `--result`、`--summary`、`--metadata`。 |
| `block <id> "<reason>"` | タスクをブロック済みとしてマークします。理由もコメントとして追加します。 |
| `unblock <id>` | ブロックされたタスクを準備完了に戻します。 |
| `archive <id>` | デフォルトのリストから非表示にします。`gc` がscratchワークスペースを削除します。 |
| `tail <id>` | タスクのイベントストリームを追跡します。 |
| `dispatch` | アクティブなボードに対するディスパッチャーの1パス。フラグ: `--dry-run`、`--max N`、`--json`。 |
| `context <id>` | ワーカーが見る完全なコンテキスト（タイトル + 本文 + 親の結果 + コメント）を出力します。 |
| `specify <id>` / `specify --all` | トリアージ列のタスクを、補助LLMを介して具体的な仕様（タイトル + 目標、アプローチ、受け入れ基準を含む本文）に肉付けし、`todo` に昇格させます。フラグ: `--tenant`（`--all` を1つのテナントにスコープ）、`--author`、`--json`。モデルは `config.yaml` の `auxiliary.triage_specifier` で設定します。 |
| `gc` | アーカイブされたタスクのscratchワークスペースを削除します。 |

例:

```bash
# 切り替えずに、2つ目のボードを作成してタスクを置く。
hermes kanban boards create atm10-server --name "ATM10 Server" --icon 🎮
hermes kanban --board atm10-server create "Restart server" --assignee ops

# 以降の呼び出しでアクティブなボードを切り替える。
hermes kanban boards switch atm10-server
hermes kanban list                  # atm10-server のタスクを表示

# ボードをアーカイブ（復元可能）またはハード削除する。
hermes kanban boards rm atm10-server
hermes kanban boards rm atm10-server --delete
```

ボードの解決順序（優先度の高い順）: `--board <slug>` フラグ → `HERMES_KANBAN_BOARD` 環境変数 → `~/.hermes/kanban/current` ファイル → `default`。

すべてのアクションは、ゲートウェイのスラッシュコマンド（`/kanban …`）としても利用でき、同じ引数面を持ちます — `boards` サブコマンドと `--board` フラグを含みます。

完全な設計 — Cline Kanban / Paperclip / NanoClaw / Gemini Enterpriseとの比較、8つのコラボレーションパターン、4つのユーザーストーリー、並行性の正しさの証明 — については、リポジトリ内の `docs/hermes-kanban-v1-spec.pdf` または[Kanbanユーザーガイド](/docs/user-guide/features/kanban)を参照してください。

## `hermes webhook`

```bash
hermes webhook <subscribe|list|remove|test>
```

イベント駆動のエージェント起動向けに、動的なWebhookサブスクリプションを管理します。設定でwebhookプラットフォームが有効になっている必要があります — 設定されていない場合は、セットアップ手順を出力します。

| サブコマンド | 説明 |
|------------|-------------|
| `subscribe` / `add` | Webhookルートを作成します。サービスに設定するURLとHMACシークレットを返します。 |
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
| `--deliver-only` | エージェントをスキップ — レンダリングされた `--prompt` をリテラルなメッセージとして配信します。LLMコストゼロ、サブ秒の配信。`--deliver` が実際のターゲット（`log` ではない）である必要があります。 |

サブスクリプションは `~/.hermes/webhook_subscriptions.json` に永続化され、ゲートウェイを再起動せずにwebhookアダプターによってホットリロードされます。

## `hermes doctor`

```bash
hermes doctor [--fix]
```

| オプション | 説明 |
|--------|-------------|
| `--fix` | 可能な範囲で自動修復を試みます。 |

## `hermes dump`

```bash
hermes dump [--show-keys]
```

Hermesのセットアップ全体を、コンパクトなプレーンテキストの要約として出力します。サポートを求める際にDiscord、GitHubのissue、Telegramにコピーペーストできるよう設計されています — ANSIカラーも特殊なフォーマットもなく、データだけです。

| オプション | 説明 |
|--------|-------------|
| `--show-keys` | 単なる `set`/`not set` ではなく、秘匿化されたAPIキーのプレフィックス（最初と最後の4文字）を表示します。 |

### 含まれる内容

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
| **Config overrides** | デフォルトと異なる設定値 |

### 出力例

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

### 使用するタイミング

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

デバッグレポート（システム情報 + 最近のログ）をペーストサービスにアップロードし、共有可能なURLを取得します。手早いサポート依頼に便利です — ヘルパーが問題を診断するのに必要なすべてを含みます。

| オプション | 説明 |
|--------|-------------|
| `--lines <N>` | ログファイルごとに含めるログ行数（デフォルト: 200）。 |
| `--expire <days>` | ペーストの有効期限（日数、デフォルト: 7）。 |
| `--local` | アップロードする代わりに、ローカルでレポートを出力します。 |

レポートには、システム情報（OS、Pythonバージョン、Hermesバージョン）、最近のエージェントおよびゲートウェイのログ（ファイルあたり512 KB制限）、秘匿化されたAPIキーのステータスが含まれます。キーは常に秘匿化され、シークレットはアップロードされません。

試行されるペーストサービス（順番）: paste.rs、dpaste.com。

### 例

```bash
hermes debug share              # デバッグレポートをアップロードし、URLを出力
hermes debug share --lines 500  # より多くのログ行を含める
hermes debug share --expire 30  # ペーストを30日間保持
hermes debug share --local      # レポートをターミナルに出力（アップロードなし）
```

## `hermes backup`

```bash
hermes backup [options]
```

Hermesの設定、スキル、セッション、データのzipアーカイブを作成します。バックアップにはhermes-agentのコードベース自体は含まれません。

| オプション | 説明 |
|--------|-------------|
| `-o`, `--output <path>` | zipファイルの出力パス（デフォルト: `~/hermes-backup-<timestamp>.zip`）。 |
| `-q`, `--quick` | クイックスナップショット: 重要な状態ファイル（config.yaml、state.db、.env、auth、cronジョブ）のみ。完全バックアップよりはるかに高速です。 |
| `-l`, `--label <name>` | スナップショットのラベル（`--quick` でのみ使用）。 |

バックアップはSQLiteの `backup()` APIを使用して安全にコピーするため、Hermesの実行中でも正しく動作します（WALモードで安全）。

**zipから除外されるもの:**

- `*.db-wal`、`*.db-shm`、`*.db-journal` — SQLiteのWAL / 共有メモリ / ジャーナルのサイドカー。`*.db` ファイルはすでに `sqlite3.backup()` で一貫したスナップショットを取得済みです。ライブのサイドカーを一緒に同梱すると、復元時に中途半端にコミットされた状態が見えてしまいます。
- `checkpoints/` — セッションごとの軌跡キャッシュ。ハッシュキー付きでセッションごとに再生成されるため、いずれにせよ別のインストールにきれいに移植できません。
- `hermes-agent` のコード自体（これはユーザーデータのバックアップであり、リポジトリのスナップショットではありません）。

### 例

```bash
hermes backup                           # ~/hermes-backup-*.zip への完全バックアップ
hermes backup -o /tmp/hermes.zip        # 特定パスへの完全バックアップ
hermes backup --quick                   # 状態のみのクイックスナップショット
hermes backup --quick --label "pre-upgrade"  # ラベル付きクイックスナップショット
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
| `clear-legacy` | v1→v2の移行で生成された `legacy-<timestamp>/` アーカイブのみを削除します。 |

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
hermes checkpoints                                  # ステータス概要
hermes checkpoints prune --retention-days 3         # 積極的なクリーンアップ
hermes checkpoints prune --max-size-mb 200          # サイズ上限を一度引き締める
hermes checkpoints clear-legacy -f                  # v1アーカイブディレクトリを削除
hermes checkpoints clear -f                         # すべてを消去
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

### 例
```bash
hermes import ~/hermes-backup-20260423.zip           # 既存設定の上書き前に確認を求める
hermes import ~/hermes-backup-20260423.zip --force   # 確認なしで上書き
```

## `hermes logs`

```bash
hermes logs [log_name] [options]
```

Hermesのログファイルを表示、追跡、フィルタリングします。すべてのログは `~/.hermes/logs/`（非デフォルトプロファイルの場合は `<profile>/logs/`）に保存されます。

### ログファイル

| 名前 | ファイル | 記録される内容 |
|------|------|-----------------|
| `agent`（デフォルト） | `agent.log` | すべてのエージェントアクティビティ — API呼び出し、ツールディスパッチ、セッションのライフサイクル（INFO以上） |
| `errors` | `errors.log` | 警告とエラーのみ — agent.logのフィルタリングされたサブセット |
| `gateway` | `gateway.log` | メッセージングゲートウェイのアクティビティ — プラットフォーム接続、メッセージディスパッチ、webhookイベント |

### オプション

| オプション | 説明 |
|--------|-------------|
| `log_name` | 表示するログ: `agent`（デフォルト）、`errors`、`gateway`、または利用可能なファイルをサイズとともに表示する `list`。 |
| `-n`, `--lines <N>` | 表示する行数（デフォルト: 50）。 |
| `-f`, `--follow` | `tail -f` のように、ログをリアルタイムで追跡します。停止するにはCtrl+Cを押します。 |
| `--level <LEVEL>` | 表示する最小ログレベル: `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。 |
| `--session <ID>` | セッションIDの部分文字列を含む行をフィルタリングします。 |
| `--since <TIME>` | 相対的な時間からの行を表示します: `30m`、`1h`、`2d` など。`s`（秒）、`m`（分）、`h`（時間）、`d`（日）をサポートします。 |
| `--component <NAME>` | コンポーネントでフィルタリングします: `gateway`、`agent`、`tools`、`cli`、`cron`。 |

### 例

```bash
# agent.log の最後の50行を表示（デフォルト）
hermes logs

# agent.log をリアルタイムで追跡
hermes logs -f

# gateway.log の最後の100行を表示
hermes logs gateway -n 100

# 過去1時間の警告とエラーのみを表示
hermes logs --level WARNING --since 1h

# 特定のセッションでフィルタリング
hermes logs --session abc123

# 30分前からの errors.log を追跡
hermes logs errors --since 30m -f

# すべてのログファイルをサイズとともに一覧表示
hermes logs list
```

### フィルタリング

フィルターは組み合わせられます。複数のフィルターがアクティブな場合、ログ行は表示されるためにそれらの**すべて**を通過する必要があります:

```bash
# 過去2時間の、セッション "tg-12345" を含む WARNING+ の行
hermes logs --level WARNING --since 2h --session tg-12345
```

解析可能なタイムスタンプを持たない行は、`--since` がアクティブなときに含まれます（複数行のログエントリーの継続行である可能性があります）。検出可能なレベルを持たない行は、`--level` がアクティブなときに含まれます。

### ログローテーション

HermesはPythonの `RotatingFileHandler` を使用します。古いログは自動的にローテーションされます — `agent.log.1`、`agent.log.2` などを探してください。`hermes logs list` サブコマンドは、ローテーションされたものを含むすべてのログファイルを表示します。

## `hermes config`

```bash
hermes config <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `show` | 現在の設定値を表示します。 |
| `edit` | エディターで `config.yaml` を開きます。 |
| `set <key> <value>` | 設定値を設定します。 |
| `path` | 設定ファイルのパスを出力します。 |
| `env-path` | `.env` ファイルのパスを出力します。 |
| `check` | 欠落または古い設定をチェックします。 |
| `migrate` | 新しく導入されたオプションを対話的に追加します。 |

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
hermes skills install official/migration/openclaw-migration
hermes skills install skills-sh/anthropics/skills/pdf --force
hermes skills install https://sharethis.chat/SKILL.md                     # 直接URL（単一ファイルのSKILL.md）
hermes skills install https://example.com/SKILL.md --name my-skill        # frontmatterに名前がない場合に名前を上書き
hermes skills check
hermes skills update
hermes skills config
hermes skills reset google-workspace
hermes skills reset google-workspace --restore --yes
```

注意:
- `--force` は、サードパーティ/コミュニティスキルに対する危険でないポリシーブロックを上書きできます。
- `--force` は `dangerous` のスキャン判定を上書きしません。
- `--source skills-sh` は公開の `skills.sh` ディレクトリを検索します。
- `--source well-known` を使うと、`/.well-known/skills/index.json` を公開しているサイトにHermesを向けられます。
- `http(s)://…/*.md` のURLを渡すと、単一ファイルのSKILL.mdを直接インストールします。frontmatterに `name:` がなく、URLのスラッグが有効な識別子でない場合、対話型ターミナルで名前の入力を求めます。非対話型の操作面（TUI内の `/skills install`、ゲートウェイプラットフォーム）では、代わりに `--name <x>` が必要です。

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

外部メモリプロバイダープラグインをセットアップ・管理します。利用可能なプロバイダー: honcho、openviking、mem0、hindsight、holographic、retaindb、byterover、supermemory。一度にアクティブにできる外部プロバイダーは1つだけです。組み込みメモリ（MEMORY.md/USER.md）は常にアクティブです。

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
pip install -e '.[acp]'
```

[ACPエディター統合](../user-guide/features/acp.md)と[ACP内部](../developer-guide/acp-internals.md)を参照してください。

## `hermes mcp`

```bash
hermes mcp <subcommand>
```

MCP（Model Context Protocol）サーバー設定を管理し、HermesをMCPサーバーとして実行します。

| サブコマンド | 説明 |
|------------|-------------|
| `serve [-v\|--verbose]` | HermesをMCPサーバーとして実行します — 会話を他のエージェントに公開します。 |
| `add <name> [--url URL] [--command CMD] [--args ...] [--auth oauth\|header]` | 自動ツール検出とともにMCPサーバーを追加します。 |
| `remove <name>`（エイリアス: `rm`） | 設定からMCPサーバーを削除します。 |
| `list`（エイリアス: `ls`） | 設定済みのMCPサーバーを一覧表示します。 |
| `test <name>` | MCPサーバーへの接続をテストします。 |
| `configure <name>`（エイリアス: `config`） | サーバーのツール選択を切り替えます。 |
| `login <name>` | OAuthベースのMCPサーバーに対して再認証を強制します。 |

[MCP設定リファレンス](./mcp-config-reference.md)、[HermesでMCPを使う](../guides/use-mcp-with-hermes.md)、[MCPサーバーモード](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server)を参照してください。

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
| `install <identifier> [--force]` | Git URLまたは `owner/repo` からプラグインをインストールします。 |
| `update <name>` | インストール済みプラグインの最新の変更を取得します。 |
| `remove <name>`（エイリアス: `rm`, `uninstall`） | インストール済みプラグインを削除します。 |
| `enable <name>` | 無効化されたプラグインを有効化します。 |
| `disable <name>` | プラグインを削除せずに無効化します。 |
| `list`（エイリアス: `ls`） | インストール済みプラグインを有効/無効ステータスとともに一覧表示します。 |

プロバイダープラグインの選択は `config.yaml` に保存されます:
- `memory.provider` — アクティブなメモリプロバイダー（空 = 組み込みのみ）
- `context.engine` — アクティブなコンテキストエンジン（`"compressor"` = 組み込みのデフォルト）

一般プラグインの無効化リストは `config.yaml` の `plugins.disabled` に保存されます。

[プラグイン](../user-guide/features/plugins.md)と[Hermesプラグインを作る](../guides/build-a-hermes-plugin.md)を参照してください。

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
| `install` | 上流のcua-driverインストーラーを実行します（macOSのみ）。 |
| `status` | `cua-driver` が `$PATH` 上にあるかどうかを出力します。 |

`hermes computer-use install` は、`computer_use` ツールセットが使用する
[cua-driver](https://github.com/trycua/cua) バイナリをインストールするための
安定したエントリーポイントです。Computer Useを初めて有効にしたときに
`hermes tools` が呼び出すのと同じ上流のインストーラーを実行するため、
ツールセットの切り替えがインストールをトリガーしなかった場合（例えば再訪
ユーザーのセットアップ時）に、インストールを再実行するのに安全に使用できます。

## `hermes sessions`

```bash
hermes sessions <subcommand>
```

サブコマンド:

| サブコマンド | 説明 |
|------------|-------------|
| `list` | 最近のセッションを一覧表示します。 |
| `browse` | 検索と再開ができる対話型セッションピッカー。 |
| `export <output> [--session-id ID]` | セッションをJSONLにエクスポートします。 |
| `delete <session-id>` | 1つのセッションを削除します。 |
| `prune` | 古いセッションを削除します。 |
| `stats` | セッションストアの統計を表示します。 |
| `rename <session-id> <title>` | セッションタイトルを設定または変更します。 |

## `hermes insights`

```bash
hermes insights [--days N] [--source platform]
```

| オプション | 説明 |
|--------|-------------|
| `--days <n>` | 過去 `n` 日を分析します（デフォルト: 30）。 |
| `--source <platform>` | `cli`、`telegram`、`discord` などのソースでフィルタリングします。 |

## `hermes claw`

```bash
hermes claw migrate [options]
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

### 移行される内容

移行は、ペルソナ、メモリ、スキル、モデルプロバイダー、メッセージングプラットフォーム、エージェント動作、セッションポリシー、MCPサーバー、TTSなど、30以上のカテゴリーをカバーします。項目は、Hermesの相当機能に**直接インポートされる**か、手動レビュー用に**アーカイブされる**かのいずれかです。

**直接インポートされるもの:** SOUL.md、MEMORY.md、USER.md、AGENTS.md、スキル（4つのソースディレクトリ）、デフォルトモデル、カスタムプロバイダー、MCPサーバー、メッセージングプラットフォームのトークンと許可リスト（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost）、エージェントのデフォルト（推論努力、圧縮、ヒューマンディレイ、タイムゾーン、サンドボックス）、セッションリセットポリシー、承認ルール、TTS設定、ブラウザ設定、ツール設定、execタイムアウト、コマンド許可リスト、ゲートウェイ設定、3つのソースからのAPIキー。

**手動レビュー用にアーカイブされるもの:** cronジョブ、プラグイン、フック/webhook、メモリバックエンド（QMD）、スキルレジストリ設定、UI/アイデンティティ、ロギング、マルチエージェントセットアップ、チャンネルバインディング、IDENTITY.md、TOOLS.md、HEARTBEAT.md、BOOTSTRAP.md。

**APIキーの解決**は、優先順位順に3つのソースをチェックします: 設定値 → `~/.openclaw/.env` → `auth-profiles.json`。すべてのトークンフィールドは、プレーンな文字列、環境変数テンプレート（`${VAR}`）、SecretRefオブジェクトを処理します。

完全な設定キーのマッピング、SecretRefの処理の詳細、移行後のチェックリストについては、**[完全な移行ガイド](../guides/migrate-from-openclaw.md)**を参照してください。

### 例

```bash
# 移行される内容をプレビュー
hermes claw migrate --dry-run

# 完全移行（すべての互換設定、シークレットなし）
hermes claw migrate --preset full

# APIキーを含む完全移行
hermes claw migrate --preset full --migrate-secrets

# ユーザーデータのみ移行（シークレットなし）、競合を上書き
hermes claw migrate --preset user-data --overwrite

# カスタムのOpenClawパスから移行
hermes claw migrate --source /home/user/old-openclaw
```

## `hermes dashboard`

```bash
hermes dashboard [options]
```

Webダッシュボードを起動します — 設定、APIキーの管理とセッションのモニタリングを行うブラウザベースのUIです。`pip install hermes-agent[web]`（FastAPI + Uvicorn）が必要です。完全なドキュメントは[Webダッシュボード](/docs/user-guide/features/web-dashboard)を参照してください。

| オプション | デフォルト | 説明 |
|--------|---------|-------------|
| `--port` | `9119` | Webサーバーを実行するポート |
| `--host` | `127.0.0.1` | バインドアドレス |
| `--no-open` | — | ブラウザを自動的に開かない |

```bash
# デフォルト — ブラウザを http://127.0.0.1:9119 に開く
hermes dashboard

# カスタムポート、ブラウザなし
hermes dashboard --port 8080 --no-open
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
| `create <name> [--clone] [--clone-all] [--clone-from <source>] [--no-alias]` | 新しいプロファイルを作成します。`--clone` はアクティブなプロファイルから設定、`.env`、`SOUL.md` をコピーします。`--clone-all` はすべての状態をコピーします。`--clone-from` はソースプロファイルを指定します。 |
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

## `hermes update`

```bash
hermes update [--check] [--backup] [--restart-gateway]
```

最新の `hermes-agent` コードを取得し、venvに依存関係を再インストールしてから、ポストインストールフック（MCPサーバー、スキル同期、補完のインストール）を再実行します。稼働中のインストールでも安全に実行できます。

| オプション | 説明 |
|--------|-------------|
| `--check` | 現在のコミットと最新の `origin/main` コミットを並べて出力し、同期していれば0、遅れていれば1で終了します。pull、インストール、再起動は何も行いません。 |
| `--backup` | pull前に、`HERMES_HOME`（設定、auth、セッション、スキル、ペアリングデータ）のラベル付きアップデート前スナップショットを作成します。デフォルトは**off**です — 以前の常時バックアップの動作は、大きなホームでは更新ごとに数分を追加していました。`config.yaml` の `update.backup: true` で恒久的に有効にできます。 |
| `--restart-gateway` | 更新成功後、稼働中のゲートウェイサービスを再起動します。複数のプロファイルがインストールされている場合は `--all` のセマンティクスを含意します。 |

追加の動作:

- **ペアリングデータのスナップショット。** `--backup` がoffのときでも、`hermes update` は `git pull` の前に `~/.hermes/pairing/` とFeishuのコメントルールの軽量なスナップショットを取得します。pullが編集中のファイルを書き換えた場合、`hermes backup restore --state pre-update` でロールバックできます。
- **レガシー `hermes.service` の警告。** Hermesがリネーム前の `hermes.service` systemdユニット（現在の `hermes-gateway.service` ではなく）を検出した場合、フラップループの問題を避けられるよう、一度だけ移行のヒントを出力します。
- **終了コード。** 成功時 `0`、pull/インストール/ポストインストールのエラー時 `1`、`git pull` をブロックする予期しないワーキングツリーの変更時 `2`。

## メンテナンスコマンド

| コマンド | 説明 |
|---------|-------------|
| `hermes version` | バージョン情報を出力します。 |
| `hermes update` | 最新の変更を取得して依存関係を再インストールします。 |
| `hermes uninstall [--full] [--yes]` | Hermesを削除します。必要に応じてすべての設定/データを削除します。 |

## 関連項目

- [スラッシュコマンドリファレンス](./slash-commands.md)
- [CLIインターフェース](../user-guide/cli.md)
- [セッション](../user-guide/sessions.md)
- [スキルシステム](../user-guide/features/skills.md)
- [スキンとテーマ](../user-guide/features/skins.md)
