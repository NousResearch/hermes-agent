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
hermes chat --toolsets debugging        # 複合 — file + terminal + web に展開
hermes chat --toolsets all              # すべて
```

### プラットフォームごと（config.yaml）

```yaml
toolsets:
  - hermes-cli          # CLI のデフォルト
  # - hermes-telegram   # Telegram ゲートウェイ用のオーバーライド
```

### 対話的な管理

```bash
hermes tools                            # プラットフォームごとに有効/無効を切り替える curses UI
```

またはセッション内で:

```
/tools list
/tools disable browser
/tools enable rl
```

## コアツールセット

| ツールセット | ツール | 目的 |
|---------|-------|------|
| `browser` | `browser_back`, `browser_cdp`, `browser_click`, `browser_console`, `browser_dialog`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `web_search` | コアのブラウザ自動化。手早い検索のフォールバックとして `web_search` を含む。`browser_cdp` と `browser_dialog` は実行時にゲートされる — セッション開始時に CDP エンドポイントが到達可能な場合にのみ登録される（`/browser connect`、`browser.cdp_url` 設定、Browserbase、または Camofox 経由）。`browser_dialog` は、CDP スーパーバイザーがアタッチされているときに `browser_snapshot` が追加する `pending_dialogs` と `frame_tree` フィールドと連携して動作する。 |
| `clarify` | `clarify` | エージェントが明確化を必要とするときにユーザーへ質問する。 |
| `code_execution` | `execute_code` | Hermes ツールをプログラム的に呼び出す Python スクリプトを実行する。 |
| `cronjob` | `cronjob` | 繰り返しタスクのスケジュールと管理。 |
| `debugging` | 複合（`file` + `terminal` + `web`） | デバッグバンドル — ファイル、プロセス/ターミナル、web 抽出/検索。 |
| `delegation` | `delegate_task` | 並列作業のために分離されたサブエージェントインスタンスを生成する。 |
| `discord` | `discord` | コアの Discord テキスト/埋め込み/DM アクション（ゲートウェイ専用）。`hermes-discord` ツールセットでアクティブ。 |
| `discord_admin` | `discord_admin` | Discord のモデレーション（BAN、ロール変更、チャネル管理）。`hermes-discord` ツールセットでアクティブ。ボットが該当する Discord 権限を保持している必要がある。 |
| `feishu_doc` | `feishu_doc_read` | Feishu/Lark ドキュメントのコンテンツを読む。Feishu のドキュメントコメント向け知的返信ハンドラーで使用。 |
| `feishu_drive` | `feishu_drive_add_comment`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment` | Feishu/Lark drive のコメント操作。コメントエージェントにスコープされており、`hermes-cli` や他のメッセージングツールセットには公開されない。 |
| `file` | `patch`, `read_file`, `search_files`, `write_file` | ファイルの読み取り、書き込み、検索、編集。 |
| `homeassistant` | `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services` | Home Assistant 経由のスマートホーム制御。`HASS_TOKEN` が設定されている場合のみ利用可能。 |
| `computer_use` | `computer_use` | cua-driver 経由のバックグラウンド macOS デスクトップ制御 — カーソル/フォーカスを奪わない。ツール対応の任意のモデルで動作。macOS のみ；`$PATH` 上に `cua-driver` が必要。 |
| `image_gen` | `image_generate` | FAL.ai 経由のテキスト→画像生成（オプトインの OpenAI / xAI バックエンド付き）。 |
| `kanban` | `kanban_block`, `kanban_comment`, `kanban_complete`, `kanban_create`, `kanban_heartbeat`, `kanban_link`, `kanban_show` | マルチエージェント連携ツール — エージェントが kanban ディスパッチャーによって生成されたとき（`HERMES_KANBAN_TASK` env が設定されているとき）にのみ登録される。ワーカーが構造化された引き継ぎでタスクを完了とマークし、人間の入力のためにブロックし、長時間の操作中にハートビートし、スレッドにコメントし、（オーケストレーター向けに）子タスクにファンアウトできる。 |
| `memory` | `memory` | セッションをまたぐ永続メモリの管理。 |
| `messaging` | `send_message` | セッション内から他のプラットフォーム（Telegram、Discord など）にメッセージを送る。 |
| `moa` | `mixture_of_agents` | Mixture of Agents によるマルチモデル合意。 |
| `rl` | `rl_check_status`, `rl_edit_config`, `rl_get_current_config`, `rl_get_results`, `rl_list_environments`, `rl_list_runs`, `rl_select_environment`, `rl_start_training`, `rl_stop_training`, `rl_test_inference` | RL 学習環境の管理（Atropos）。 |
| `safe` | `image_generate`, `vision_analyze`, `web_extract`, `web_search`（`includes` 経由） | 読み取り専用のリサーチ ＋ メディア生成。ファイル書き込み、ターミナル、コード実行はなし。 |
| `search` | `web_search` | Web 検索のみ（抽出なし）。 |
| `session_search` | `session_search` | 過去の会話セッションを検索する。 |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | スキルの CRUD と閲覧。 |
| `spotify` | `spotify_albums`, `spotify_devices`, `spotify_library`, `spotify_playback`, `spotify_playlists`, `spotify_queue`, `spotify_search` | ネイティブの Spotify 制御（再生、キュー、検索、プレイリスト、アルバム、ライブラリ）。バンドルされた `spotify` プラグインによって登録される。 |
| `terminal` | `process`, `terminal` | シェルコマンドの実行とバックグラウンドプロセスの管理。 |
| `todo` | `todo` | セッション内のタスクリスト管理。 |
| `tts` | `text_to_speech` | テキスト読み上げ音声生成。 |
| `vision` | `vision_analyze` | vision 対応モデルによる画像分析。 |
| `video` | `video_analyze` | 動画分析・理解ツール（オプトイン、デフォルトツールセットには含まれない — `--toolsets` で明示的に追加）。 |
| `web` | `web_extract`, `web_search` | Web 検索とページコンテンツの抽出。 |
| `yuanbao` | `yb_query_group_info`, `yb_query_group_members`, `yb_search_sticker`, `yb_send_dm`, `yb_send_sticker` | Yuanbao の DM/グループアクションとスタンプ検索。`hermes-yuanbao` でのみ登録される。 |

## プラットフォームツールセット

プラットフォームツールセットは、デプロイ先の完全なツール設定を定義します。ほとんどのメッセージングプラットフォームは `hermes-cli` と同じセットを使います:

| ツールセット | `hermes-cli` との差分 |
|---------|-------------------------------|
| `hermes-cli` | フルツールセット — 対話的 CLI セッションのデフォルト。file、terminal、web、browser、memory、skills、vision、image_gen、todo、tts、delegation、code_execution、cronjob、session_search、clarify、`safe`（読み取り専用）の各バンドルに加え、標準的なメッセージングツールを含む。 |
| `hermes-acp` | `clarify`、`cronjob`、`image_generate`、`send_message`、`text_to_speech`、および 4 つの Home Assistant ツールすべてを除外。IDE コンテキストでのコーディングタスクに特化。 |
| `hermes-api-server` | `clarify`、`send_message`、`text_to_speech` を除外。それ以外はすべて維持 — ユーザー操作が不可能なプログラム的アクセスに適する。 |
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
| `hermes-yuanbao` | `hermes-cli` に加えて 5 つの `yb_*` ツール（DM/グループ/スタンプ）を追加。 |
| `hermes-homeassistant` | `hermes-cli` と同じ（Home Assistant ツールはデフォルトで既に存在し、`HASS_TOKEN` が設定されるとアクティブになる）。 |
| `hermes-webhook` | `hermes-cli` と同じ。 |
| `hermes-gateway` | 内部のゲートウェイオーケストレーターツールセット — すべての `hermes-<platform>` ツールセットの和集合；ゲートウェイが任意のメッセージソースを受け入れる必要があるときに使用。 |

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

これにより、`--toolsets` やプラットフォーム設定で参照できる `mcp-github` ツールセットが作成されます。

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

## `hermes tools` との関係

`hermes tools` コマンドは、プラットフォームごとに個々のツールを有効/無効に切り替えるための curses ベースの UI を提供します。これはツールレベル（ツールセットより細かい）で動作し、`config.yaml` に永続化されます。ツールセットが有効でも、無効化されたツールはフィルタリングされます。

関連項目: 個々のツールとそのパラメータの完全なリストは [ツールリファレンス](./tools-reference.md) を参照してください。
