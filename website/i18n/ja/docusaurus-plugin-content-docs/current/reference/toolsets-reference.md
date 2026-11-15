---
sidebar_position: 4
title: "ツールセットリファレンス"
description: "Hermes のコア、複合、プラットフォーム、動的ツールセットのリファレンス"
---

# ツールセットリファレンス

ツールセットは、エージェントができることを制御するツールの名前付きバンドルです。プラットフォームごと、セッションごと、タスクごとにツールの利用可否を設定するための主要な仕組みです。

## ツールセットの仕組み

すべてのツールはちょうど 1 つのツールセットに属します。ツールセットを有効にすると、そのバンドル内のすべてのツールがエージェントで利用可能になります。ツールセットには 3 種類あります:

- **コア** — 関連するツールの単一の論理グループ（例: `file` は `read_file`、`write_file`、`patch`、`search_files` をバンドル）
- **複合** — 一般的なシナリオのために複数のコアツールセットを組み合わせたもの（例: `debugging` は file、terminal、web ツールをバンドル）
- **プラットフォーム** — 特定のデプロイコンテキストのための完全なツール設定（例: `hermes-cli` は対話的 CLI セッションのデフォルト）

## ツールセットの設定

### セッションごと（CLI）

```bash
hermes chat --toolsets web,file,terminal
hermes chat --toolsets debugging        # composite — expands to file + terminal + web
hermes chat --toolsets all              # everything
```

### プラットフォームごと（config.yaml）

```yaml
toolsets:
  - hermes-cli          # default for CLI
  # - hermes-telegram   # override for Telegram gateway
```

### 対話的な管理

```bash
hermes tools                            # curses UI to enable/disable per platform
```

またはセッション内で:

```
/tools list
/tools disable browser
/tools enable discord
```

## コアツールセット

| ツールセット | ツール | 目的 |
|---------|-------|------|
| `browser` | `browser_cdp`, `browser_dialog`, `browser_exec`, `browser_vault` | コアのブラウザ自動化。`browser_exec` は、ヘルパーが事前インポートされた Python を実行してブラウザを操作する（`terminal` も持つセッションにのみ提供される）。`browser_cdp`、`browser_dialog`、`browser_vault` は同じブラウザセッション上で動作する。`browser_vault` は認証情報フォーム用にブラウザに付随し、デフォルトではツール検索の背後に遅延される。ブラウジングをオフにするには、このツールセットを無効にする。手早い検索は `web`/`search` ツールセットの役割 — `web_search` は意図的にメンバーに含まれていないため、`disabled_toolsets: [browser]` で取り除かれることはない（#64503）。 |
| `clarify` | `clarify` | エージェントが明確化を必要とするときにユーザーへ質問する。 |
| `code_execution` | `execute_code` | Hermes ツールをプログラム的に呼び出す Python スクリプトを実行する。 |
| `connections` | `manage_connections` | ユーザーをアプリに接続する: Nous ゲートウェイ経由のマネージドコネクタアカウントと、カタログからのローカル MCP サーバー。アプリの 2 つ目の名前付きアカウントを追加したり、アカウント名を変更したりできる。デスクトップでは各アクションがユーザーが確定するまでブロックするカードになる。デスクトップ以外では、マネージドアクションはアプリごとの接続リンクを返す。 |
| `coding` | 複合（`file` + `terminal` + `search` + `web` + `skills` + `browser` + `todo` + `memory` + `session_search` + `clarify` + `code_execution` + `delegation` + `vision`） | ソフトウェア作業向けのコーディング特化バンドル: ファイル編集、ターミナル、検索、Web ドキュメント、スキル、ブラウザ、委譲、コード実行。 |
| `cronjob` | `cronjob_manage` | 繰り返しタスクのスケジュールと管理。 |
| `debugging` | 複合（`file` + `terminal` + `web`） | デバッグバンドル — ファイル、プロセス/ターミナル、web 抽出/検索。 |
| `delegation` | `delegate_task` | 並列作業のために分離されたサブエージェントインスタンスを生成する。 |
| `discord` | `discord` | コアの Discord テキスト/埋め込み/DM アクション（ゲートウェイ専用）。`hermes-discord` ツールセットでアクティブ。 |
| `discord_admin` | `discord_admin` | Discord のモデレーション（BAN、ロール変更、チャネル管理）。`hermes-discord` ツールセットでアクティブ。ボットが該当する Discord 権限を保持している必要がある。 |
| `feishu_doc` | `feishu_doc_read` | Feishu/Lark ドキュメントのコンテンツを読む。Feishu のドキュメントコメント向け知的返信ハンドラーで使用。 |
| `feishu_drive` | `feishu_drive_add_comment`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment` | Feishu/Lark drive のコメント操作。コメントエージェントにスコープされており、`hermes-cli` や他のメッセージングツールセットには公開されない。 |
| `file` | `patch`, `read_file`, `search_files`, `write_file` | ファイルの読み取り、書き込み、検索、編集。 |
| `computer_use` | `computer_use` | cua-driver 経由のバックグラウンドデスクトップ制御 — カーソル/フォーカスを奪わない。ツール対応の任意のモデルで動作。macOS、Windows、Linux に対応；`$PATH` 上に `cua-driver` が必要。 |
| `context_engine` | （可変） | アクティブなコンテキストエンジンプラグインが公開するランタイムツール（プラグインが追加するまでは空）。 |
| `image_gen` | `image_generate` | FAL.ai 経由のテキスト→画像生成（オプトインの OpenAI / xAI バックエンド付き）。 |
| `video_gen` | `video_generate`, `xai_video_edit`, `xai_video_extend` | プラグインが登録したバックエンド（xAI Grok-Imagine、FAL.ai Veo 3.1 / Pixverse v6 / Kling 3.0 / Kling O3）によるテキスト→動画および画像→動画生成。画像をアニメーション化するには `image_url` を渡し、テキスト→動画では省略する。`xai_video_edit` / `xai_video_extend` はプロバイダー固有の編集/延長ツールで、xAI Imagine の認証情報でゲートされる。 |
| `kanban` | `kanban_attach`, `kanban_attach_url`, `kanban_attachments`, `kanban_block`, `kanban_comment`, `kanban_complete`, `kanban_create`, `kanban_heartbeat`, `kanban_link`, `kanban_list`, `kanban_request_changes`, `kanban_request_review`, `kanban_show`, `kanban_unblock` | マルチエージェント連携ツール。ディスパッチャーが生成したタスクワーカー（`HERMES_KANBAN_TASK`）と、保存済みの選択に `kanban` を含むプラットフォーム（`hermes tools enable kanban --platform <p>`。`all`/`*` ワイルドカードでは有効に**ならない**）に登録される。ワーカーはタスクを完了とマークし、正式なレビューを依頼し、ブロックし、ハートビートし、コメントし、フォローアップタスクを作成/リンクできる。オーケストレータープロファイルには、さらに list/unblock などのボードルーティングツールが与えられる。`delegate_task` の子は Kanban 実行のオーナーではない: 親の `HERMES_KANBAN_*` 環境変数が存在していても、その子のスキーマではこのツールセットが除去/無効化され、ランタイムガードがボードへの直接の変更を拒否する。 |
| `memory` | `memory` | セッションをまたぐ永続メモリの管理。 |
| `desktop_ui` | `annotate_preview`, `close_terminal`, `desktop_preview`, `drive_preview`, `focus_pane`, `gui_tour`, `react_to_message`, `read_terminal`, `read_window_below`, `set_theme`, `show_tip` | Hermes デスクトップアプリ自体に作用する機能 — 埋め込みターミナルペインの読み取り/クローズ、アプリ内ブラウザのオープン・読み取り・クローズ・操作・注釈付け、アプリの背後にある OS ウィンドウの特定、ペインの表示、メッセージへのリアクション、ガイドツアーの実行（アプリまたはプレビューペインの UI 要素をハイライト＋ナレーション）、アクセントカラー・ライト/ダークモード・レイアウトの設定（またはユーザーが選ぶためのピッカーカードの表示）。接続先のバックエンド（ローカル、SSH、URL、Hermes Cloud）にかかわらず、ソースがデスクトップアプリのセッションで有効。CLI、TUI、メッセージング、cron セッションには決して存在しない。 |
| `project` | `desktop_project` | 1 つの `create`/`switch`/`list` アクション enum で、デスクトップの [Projects](../user-guide/cli.md)（名前付きの複数フォルダーのワークスペース）を作成・切り替える。GUI / デスクトップセッションのみ。 |
| `catalog` | `manage_catalog` | プラグインカタログとスキルハブを検索し、承認カードを通じてこのチャットのプロファイルにアイテムをインストールする。接続先のバックエンド（ローカル、SSH、URL、Hermes Cloud）にかかわらず、ソースがデスクトップアプリのセッションで有効。設定リストで指定されていても、CLI、TUI、メッセージング、cron セッションには決して存在しない。`all` には含まれない。デフォルトでは `tool_search` の背後に遅延される。`agent.disabled_toolsets: [catalog]` で削除できる。 |
| `safe` | `image_generate`, `vision_analyze`, `web_extract`, `web_search`（`includes` 経由） | 読み取り専用のリサーチ ＋ メディア生成。ファイル書き込み、ターミナル、コード実行はなし。 |
| `search` | `web_search` | Web 検索のみ（抽出なし）。 |
| `session_search` | `session_search` | 過去の会話セッションを検索する。 |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | スキルの CRUD と閲覧。 |
| `terminal` | `process_manage`, `terminal` | シェルコマンドの実行とバックグラウンドプロセスの管理。 |
| `todo` | `todo_list` | セッション内のタスクリスト管理。 |
| `tts` | `text_to_speech` | テキスト読み上げ音声生成。 |
| `vision` | `vision_analyze` | vision 対応モデルによる画像分析。 |
| `video` | `video_analyze` | 動画分析・理解ツール（オプトイン、デフォルトツールセットには含まれない — `--toolsets` で明示的に追加）。 |
| `web` | `web_extract`, `web_search` | Web 検索とページコンテンツの抽出。 |
| `x_search` | `x_search` | xAI の組み込み `x_search` Responses ツールによる、読み取り専用の公開 X 探索。認証付きの X API の読み取りやアカウント操作には `xurl` スキルを使う。デフォルトではオフ。`hermes tools` でオプトインする。xAI の認証情報（SuperGrok OAuth または `XAI_API_KEY`）が設定されている場合にのみスキーマが登録される。 |

## プラットフォームツールセット

プラットフォームツールセットは、デプロイ先の完全なツール設定を定義します。ほとんどのメッセージングプラットフォームは `hermes-cli` と同じセットを使います:

| ツールセット | `hermes-cli` との差分 |
|---------|-------------------------------|
| `hermes-cli` | フルツールセット — 対話的 CLI セッションのデフォルト。file、terminal、web、browser、memory、skills、vision、image_gen、todo、tts、delegation、code_execution、cronjob、session_search、clarify、computer_use、および kanban ツールを含む（すべて実行時に check_fn でゲートされる）。 |
| `hermes-acp` | `clarify`、`cronjob`、`image_generate`、`text_to_speech`、`computer_use`、および kanban ツールを除外。IDE コンテキストでのコーディングタスクに特化。 |
| `hermes-api-server` | `clarify`、`text_to_speech`、`computer_use`、および kanban ツールを除外。それ以外はすべて維持 — ユーザー操作が不可能なプログラム的アクセスに適する。 |
| `hermes-cron` | `hermes-cli` と同じ。 |
| `hermes-telegram` | `hermes-cli` と同じ。 |
| `hermes-discord` | `hermes-cli` に加えて `discord` と `discord_admin` を追加。 |
| `hermes-slack` | `hermes-cli` と同じ。 |
| `hermes-whatsapp` | `hermes-cli` と同じ。 |
| `hermes-signal` | `hermes-cli` と同じ。 |
| `hermes-matrix` | `hermes-cli` と同じ。 |
| `hermes-mattermost` | `hermes-cli` と同じ。 |
| `hermes-email` | `hermes-cli` と同じ。 |
| `hermes-sms` | `hermes-cli` と同じ。 |
| `hermes-bluebubbles` | `hermes-cli` と同じ。 |
| `hermes-dingtalk` | `hermes-cli` と同じ。 |
| `hermes-feishu` | 5 つの `feishu_doc_*` / `feishu_drive_*` ツールを追加（通常のチャットアダプターではなく、ドキュメントコメントハンドラーのみが使用）。 |
| `hermes-qqbot` | `hermes-cli` と同じ。 |
| `hermes-wecom` | `hermes-cli` と同じ。 |
| `hermes-wecom-callback` | `hermes-cli` と同じ。 |
| `hermes-weixin` | `hermes-cli` と同じ。 |
| `hermes-webhook` | 制限された安全なサブセット — `web_search`、`web_extract`、`vision_analyze`、`clarify` のみ。Webhook でトリガーされた実行には、ターミナル、ファイル、ブラウザへのアクセスはない。 |
| `hermes-gateway` | 内部のゲートウェイオーケストレーターツールセット — すべての `hermes-<platform>` ツールセットの和集合；ゲートウェイが任意のメッセージソースを受け入れる必要があるときに使用。 |

プラグインが提供するプラットフォームには、`hermes-gateway` に含まれない暗黙の `hermes-<platform>` バンドルが与えられます。例えば、`hermes-homeassistant` と `hermes-yuanbao` は `homeassistant` / `yuanbao` カタログプラグインをインストールすれば引き続き解決されますが、組み込みのツールセットではなくなりました。

## 動的ツールセット

### MCP サーバーツールセット

設定された各 MCP サーバーは、実行時に `mcp-<server>` ツールセットを生成します。例えば、`github` MCP サーバーを設定すると、そのサーバーが公開するすべてのツールを含む `mcp-github` ツールセットが作成されます。

```yaml
# config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

これにより、`--toolsets` やプラットフォーム設定で参照できる `mcp-github` ツールセットが作成されます。素のサーバー名（`github`）もエイリアスとして使えます。サーバーが組み込みツールセットと同じ名前（`browser`、`web`）の場合、その名前は組み込みツール**に加えて**そのサーバーの `mcp__<server>__*` ツールに解決され、どちらも他方を隠しません。

`platform_toolsets.<platform>` では、MCP サーバーを 1 つも指定しないリストは有効なすべてのサーバーを取得し、サーバーを指定したリストはそのサーバーのみを取得し、`no_mcp` はすべてを除外します。明示的な空リスト（`email: []`）は MCP サーバーを 1 つも取得しません。

### プラグインツールセット

プラグインは、プラグインの初期化中に `ctx.register_tool()` を介して独自のツールセットを登録できます。これらは組み込みツールセットと並んで表示され、同じ方法で有効/無効にできます。

### カスタムツールセット

プロジェクト固有のバンドルを作成するには、`config.yaml` でカスタムツールセットを定義します:

```yaml
toolsets:
  - hermes-cli
custom_toolsets:
  data-science:
    - file
    - terminal
    - code_execution
    - web
    - vision
```

### ワイルドカード

- `all` または `*` — 登録されたすべてのツールセット（組み込み ＋ 動的 ＋ プラグイン）に展開されます

いくつかのツールには、ツールセットへの所属に加えて追加の利用可否チェックがあり、`all`/`*` だけでは有効に**なりません**:

- **機能ゲート**のツール（browser、`computer_use`、`code_execution`、Feishu、cronjob）は、そのバックエンド/認証情報の前提条件が設定されている場合にのみ表示されます。
- **ワークフローゲート**のツール — `kanban` ツールセット — は意図的にオプトインです。`all`/`*` では kanban は有効に**なりません**。`kanban` を明示的に指定する（または `HERMES_KANBAN_TASK` が設定されたディスパッチャー生成のワーカーである）必要があります。Kanban ツールは共有ボードの状態を変更するため、`all` の下でもデフォルトではオフのままです。

## `hermes tools` との関係

`hermes tools` コマンドは、プラットフォームごとに個々のツールを有効/無効に切り替えるための curses ベースの UI を提供します。これはツールレベル（ツールセットより細かい）で動作し、`config.yaml` に永続化されます。ツールセットが有効でも、無効化されたツールはフィルタリングされます。

関連項目: 個々のツールとそのパラメータの完全なリストは [ツールリファレンス](./tools-reference.md) を参照してください。
