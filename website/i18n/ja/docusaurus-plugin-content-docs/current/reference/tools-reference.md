---
sidebar_position: 3
title: "組み込みツールリファレンス"
description: "ツールセットごとに分類した、Hermesの組み込みツールの正式なリファレンス"
---

# 組み込みツールリファレンス

このページでは、Hermesの組み込みツールをツールセットごとに分類して説明します。利用可否は、プラットフォーム、認証情報、有効になっているツールセットによって異なります。

**概数（現在のレジストリ）:** 約70ツール — 10個のブラウザツール（コア）+ 2個のCDP制限付きブラウザツール、4個のファイルツール、10個のRLツール、4個のHome Assistantツール、2個のターミナルツール、2個のウェブツール、5個のFeishuツール、7個のSpotifyツール（バンドルされた `spotify` プラグインによって登録）、5個のYuanbaoツール、7個のkanbanツール（kanbanディスパッチャーがエージェントを起動するときに登録）、2個のDiscordツール、そしてひと握りのスタンドアロンツール（`memory`、`clarify`、`delegate_task`、`execute_code`、`cronjob`、`session_search`、`skill_view`/`skill_manage`/`skills_list`、`text_to_speech`、`image_generate`、`vision_analyze`、`video_analyze`、`mixture_of_agents`、`send_message`、`todo`、`computer_use`、`process`）。

:::tip MCPツール
組み込みツールに加えて、HermesはMCPサーバーからツールを動的に読み込めます。MCPツールは `mcp_<server>_` というプレフィックス付きで表示されます（例: `github` MCPサーバーの `mcp_github_create_issue`）。設定については [MCP連携](/docs/user-guide/features/mcp) を参照してください。
:::

## `browser` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `browser_back` | ブラウザの履歴で前のページに戻る。先に browser_navigate を呼び出す必要があります。 | — |
| `browser_click` | スナップショットのref ID（例: '@e5'）で識別される要素をクリックする。ref IDはスナップショット出力の角括弧内に表示されます。先に browser_navigate と browser_snapshot を呼び出す必要があります。 | — |
| `browser_console` | 現在のページからブラウザコンソールの出力とJavaScriptエラーを取得する。console.log/warn/error/info のメッセージと未捕捉のJS例外を返します。サイレントなJavaScriptエラー、失敗したAPI呼び出し、アプリケーションの警告を検出するために使用します。… | — |
| `browser_get_images` | 現在のページ上のすべての画像を、そのURLと代替テキストとともに一覧で取得する。visionツールで分析する画像を見つけるのに便利です。先に browser_navigate を呼び出す必要があります。 | — |
| `browser_navigate` | ブラウザでURLに移動する。セッションを初期化し、ページを読み込みます。他のブラウザツールの前に呼び出す必要があります。単純な情報取得には、web_search や web_extract（より速く安価）を優先してください。ブラウザツールは…が必要なときに使用します。 | — |
| `browser_press` | キーボードのキーを押す。フォームの送信（Enter）、ナビゲーション（Tab）、キーボードショートカットに便利です。先に browser_navigate を呼び出す必要があります。 | — |
| `browser_scroll` | ページをある方向にスクロールする。現在のビューポートの下や上にあるかもしれないコンテンツを表示するために使用します。先に browser_navigate を呼び出す必要があります。 | — |
| `browser_snapshot` | 現在のページのアクセシビリティツリーのテキストベースのスナップショットを取得する。browser_click と browser_type 用のref ID（@e1、@e2 など）付きでインタラクティブ要素を返します。full=false（デフォルト）: インタラクティブ要素を含むコンパクトなビュー。full=true: …の完全な… | — |
| `browser_type` | ref IDで識別される入力フィールドにテキストを入力する。まずフィールドをクリアし、それから新しいテキストを入力します。先に browser_navigate と browser_snapshot を呼び出す必要があります。 | — |
| `browser_vision` | 現在のページのスクリーンショットを撮り、それをvision AIで分析する。ページ上にあるものを視覚的に理解する必要があるときに使用します — 特にCAPTCHA、視覚的な検証チャレンジ、複雑なレイアウト、またはテキストのスナップショットが…のときに便利です。 | — |

## `browser` ツールセット（CDP制限付きツール）

これら2つのツールは `browser` ツールセットに属しますが、セッション開始時にChrome DevTools Protocolのエンドポイントに到達可能な場合にのみ登録されます — `/browser connect`、`browser.cdp_url` 設定、Browserbaseセッション、またはCamofox経由です。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `browser_cdp` | 生のChrome DevTools Protocolコマンドを送信する。上位レベルの `browser_*` ツールでカバーされないブラウザ操作のための脱出ハッチ。https://chromedevtools.github.io/devtools-protocol/ を参照。 | CDPエンドポイント |
| `browser_dialog` | ネイティブのJavaScriptダイアログ（alert / confirm / prompt / beforeunload）に応答する。先に `browser_snapshot` を呼び出してください — 保留中のダイアログはその `pending_dialogs` フィールドに表示されます。その後、`browser_dialog(action='accept'\|'dismiss')` を呼び出します。 | CDPエンドポイント |

## `clarify` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `clarify` | 進める前に、明確化、フィードバック、または決定が必要なときにユーザーに質問する。2つのモードに対応: 1. **多肢選択** — 最大4つの選択肢を提供。ユーザーは1つを選ぶか、5番目の「Other」オプションから自分の答えを入力します。2.… | — |

## `code_execution` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `execute_code` | Hermesツールをプログラム的に呼び出せるPythonスクリプトを実行する。3回以上のツール呼び出しの間に処理ロジックが必要なとき、大きなツール出力をコンテキストに入れる前にフィルタ/縮約する必要があるとき、条件分岐が必要なとき…に使用します。 | — |

## `cronjob` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `cronjob` | 統合スケジュールタスクマネージャー。`action="create"`、`"list"`、`"update"`、`"pause"`、`"resume"`、`"run"`、`"remove"` を使ってジョブを管理します。1つ以上のスキルを付属させたスキルベースのジョブに対応し、update時の `skills=[]` は付属スキルをクリアします。cronの実行は、現在のチャットのコンテキストを持たない新しいセッションで行われます。 | — |

## `delegation` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `delegate_task` | 1つ以上のサブエージェントを起動し、独立したコンテキストでタスクに取り組ませる。各サブエージェントは独自の会話、ターミナルセッション、ツールセットを持ちます。返されるのは最終的な要約のみで — 中間のツール結果があなたのコンテキストウィンドウに入ることはありません。2つ…の… | — |

## `feishu_doc` ツールセット

Feishuのドキュメントコメント向けインテリジェント返信ハンドラー（`gateway/platforms/feishu_comment.py`）に限定されています。`hermes-cli` や通常のFeishuチャットアダプターには公開されません。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `feishu_doc_read` | Feishu/Larkのドキュメント（Docx、Doc、Sheet）の全文コンテンツを、その file_type と token を指定して読み取る。 | Feishuアプリの認証情報 |

## `feishu_drive` ツールセット

Feishuのドキュメントコメントハンドラーに限定されています。ドライブファイルのコメント読み書き操作を駆動します。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `feishu_drive_add_comment` | Feishu/Larkのドキュメントまたはファイルにトップレベルのコメントを追加する。 | Feishuアプリの認証情報 |
| `feishu_drive_list_comments` | Feishu/Larkファイルのドキュメント全体のコメントを、新しい順に一覧表示する。 | Feishuアプリの認証情報 |
| `feishu_drive_list_comment_replies` | 特定のFeishuコメントスレッド（ドキュメント全体またはローカル選択）の返信を一覧表示する。 | Feishuアプリの認証情報 |
| `feishu_drive_reply_comment` | Feishuのコメントスレッドに返信を投稿する。任意で `@` メンション付き。 | Feishuアプリの認証情報 |

## `file` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `patch` | ファイル内の的を絞った検索・置換編集。ターミナルでの sed/awk の代わりにこれを使用します。ファジーマッチング（9つの戦略）を使うため、わずかな空白/インデントの差で壊れることはありません。統合diffを返します。編集後に自動で構文チェックを実行… | — |
| `read_file` | 行番号とページ分割付きでテキストファイルを読む。ターミナルでの cat/head/tail の代わりにこれを使用します。出力形式: 'LINE_NUM\|CONTENT'。見つからない場合は似たファイル名を提案します。大きなファイルには offset と limit を使用します。注意: 画像は…読めません… | — |
| `search_files` | ファイルの内容を検索する、または名前でファイルを探す。ターミナルでの grep/rg/find/ls の代わりにこれを使用します。Ripgrepベースで、シェルの相当品より高速です。コンテンツ検索（target='content'）: ファイル内の正規表現検索。出力モード: 行…付きの完全な一致… | — |
| `write_file` | ファイルに内容を書き込み、既存の内容を完全に置き換える。ターミナルでの echo/cat heredoc の代わりにこれを使用します。親ディレクトリを自動的に作成します。ファイル全体を上書きします — 的を絞った編集には 'patch' を使用してください。 | — |

## `homeassistant` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `ha_call_service` | Home Assistantのサービスを呼び出してデバイスを制御する。各ドメインで利用可能なサービスとそのパラメータを見つけるには ha_list_services を使用します。 | — |
| `ha_get_state` | 単一のHome Assistantエンティティの詳細な状態を、すべての属性（明るさ、色、温度の設定値、センサーの読み取り値など）を含めて取得する。 | — |
| `ha_list_entities` | Home Assistantのエンティティを一覧表示する。任意でドメイン（light、switch、climate、sensor、binary_sensor、cover、fan など）またはエリア名（リビング、キッチン、寝室 など）で絞り込みます。 | — |
| `ha_list_services` | デバイス制御のために利用可能なHome Assistantのサービス（アクション）を一覧表示する。各デバイスタイプでどんなアクションが実行でき、どんなパラメータを受け付けるかを示します。ha_list_entities で見つけたデバイスの制御方法を知るために使用します。 | — |

## `computer_use` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `computer_use` | cua-driver によるバックグラウンドのmacOSデスクトップ制御 — スクリーンショット（SOM / vision / AX）、click / drag / scroll / type / key / wait、list_apps、focus_app。ユーザーのカーソルやキーボードフォーカスを奪いません。ツール対応の任意のモデルで動作します。macOSのみ。 | `$PATH` 上の `cua-driver`（`hermes tools` でインストール）。 |


:::note
**Honchoツール**（`honcho_profile`、`honcho_search`、`honcho_context`、`honcho_reasoning`、`honcho_conclude`）はもはや組み込みではありません。これらは `plugins/memory/honcho/` のHonchoメモリプロバイダープラグイン経由で利用できます。インストールと使い方については [メモリプロバイダー](../user-guide/features/memory-providers.md) を参照してください。
:::

## `image_gen` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `image_generate` | FAL.aiを使ってテキストプロンプトから高品質な画像を生成する。基盤となるモデルはユーザー設定（デフォルト: FLUX 2 Klein 9B、1秒未満の生成）であり、エージェントは選択できません。単一の画像URLを返します。…を使って表示します… | FAL_KEY |

## `kanban` ツールセット

エージェントがkanbanディスパッチャーによって起動された場合（環境変数 `HERMES_KANBAN_TASK` が設定されている場合）にのみ登録されます。ワーカーが構造化されたハンドオフでタスクを完了とマークし、人間の入力のためにブロックし、長時間の操作中にハートビートを送り、スレッドにコメントし、（オーケストレーター向けに）子タスクへとファンアウトできるようにします。完全なワークフローについては [Kanbanマルチエージェント](/docs/user-guide/features/kanban) を参照してください。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `kanban_show` | このワーカーに割り当てられたアクティブなkanbanタスク（タイトル、説明、コメント、依存関係）を表示する。 | `HERMES_KANBAN_TASK` |
| `kanban_complete` | 構造化されたハンドオフのペイロード（結果、成果物、フォローアップ）とともに現在のタスクを完了とマークする。 | `HERMES_KANBAN_TASK` |
| `kanban_block` | 現在のタスクをユーザーへの質問でブロックする — ディスパッチャーは一時停止し、質問を表面化し、人間が返信したら再開します。 | `HERMES_KANBAN_TASK` |
| `kanban_heartbeat` | 長時間実行される操作の間に進捗のハートビートを送り、ディスパッチャーにワーカーがまだ生きていることを知らせる。 | `HERMES_KANBAN_TASK` |
| `kanban_comment` | 状態を変えずにタスクスレッドにコメントを追加する — 中間的な発見を表面化するのに便利です。 | `HERMES_KANBAN_TASK` |
| `kanban_create` | （オーケストレーターのみ）現在のタスクから子タスクへファンアウトする。 | `HERMES_KANBAN_TASK` + オーケストレーターロール |
| `kanban_link` | （オーケストレーターのみ）関連するタスク同士をリンクする（blocks/blocked-by/related）。 | `HERMES_KANBAN_TASK` + オーケストレーターロール |

## `memory` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `memory` | セッションをまたいで残る永続メモリに重要な情報を保存する。あなたのメモリはセッション開始時にシステムプロンプトに現れます — それが会話と会話の間で、ユーザーや環境について物事を覚えておく方法です。保存すべき…とき… | — |

## `messaging` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `send_message` | 接続されたメッセージングプラットフォームにメッセージを送る、または利用可能な送信先を一覧表示する。重要: ユーザーが（単なるプラットフォーム名ではなく）特定のチャンネルや人物への送信を求めた場合は、まず send_message(action='list') を呼び出して利用可能な送信先を確認してください… | — |

## `moa` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `mixture_of_agents` | 難しい問題を複数のフロンティアLLMに協調的にルーティングする。最大の推論努力で5回のAPI呼び出し（4つの参照モデル + 1つのアグリゲーター）を行います — 本当に難しい問題に対して控えめに使用してください。最適な用途: 複雑な数学、高度なアルゴリズム… | OPENROUTER_API_KEY |

## `rl` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `rl_check_status` | トレーニング実行のステータスとメトリクスを取得する。レート制限あり: 同じ実行に対して確認の間に最低30分を強制します。WandBのメトリクスを返します: step、state、reward_mean、loss、percent_correct。 | TINKER_API_KEY, WANDB_API_KEY |
| `rl_edit_config` | 設定フィールドを更新する。まず rl_get_current_config() を使って、選択した環境で利用可能なすべてのフィールドを確認してください。各環境には異なる設定可能なオプションがあります。インフラ設定（tokenizer、URL、lora_rank、learning_ra…） | TINKER_API_KEY, WANDB_API_KEY |
| `rl_get_current_config` | 現在の環境設定を取得する。変更可能なフィールドのみを返します: group_size、max_token_length、total_steps、steps_per_eval、use_wandb、wandb_name、max_num_workers。 | TINKER_API_KEY, WANDB_API_KEY |
| `rl_get_results` | 完了したトレーニング実行の最終結果とメトリクスを取得する。最終メトリクスと学習済み重みへのパスを返します。 | TINKER_API_KEY, WANDB_API_KEY |
| `rl_list_environments` | 利用可能なすべてのRL環境を一覧表示する。環境名、パス、説明を返します。ヒント: file_path をファイルツールで読むと、各環境の仕組み（verifier、データ読み込み、報酬）を理解できます。 | TINKER_API_KEY, WANDB_API_KEY |
| `rl_list_runs` | すべてのトレーニング実行（アクティブと完了）をそのステータスとともに一覧表示する。 | TINKER_API_KEY, WANDB_API_KEY |
| `rl_select_environment` | トレーニング用のRL環境を選択する。環境のデフォルト設定を読み込みます。選択後、rl_get_current_config() で設定を確認し、rl_edit_config() で変更します。 | TINKER_API_KEY, WANDB_API_KEY |
| `rl_start_training` | 現在の環境と設定で新しいRLトレーニング実行を開始する。ほとんどのトレーニングパラメータ（lora_rank、learning_rate など）は固定です。開始前に rl_edit_config() で group_size、batch_size、wandb_project を設定してください。警告: トレーニングは… | TINKER_API_KEY, WANDB_API_KEY |
| `rl_stop_training` | 実行中のトレーニングジョブを停止する。メトリクスが芳しくない、トレーニングが停滞している、または異なる設定を試したい場合に使用します。 | TINKER_API_KEY, WANDB_API_KEY |
| `rl_test_inference` | 任意の環境のクイック推論テスト。OpenRouterを使って数ステップの推論 + スコアリングを実行します。デフォルト: 3ステップ × 16補完 = モデルあたり48ロールアウト、3モデルのテストで合計144。環境の読み込み、プロンプト構築、推論…をテストします… | TINKER_API_KEY, WANDB_API_KEY |

## `session_search` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `session_search` | 過去の会話の長期メモリを検索する。これがあなたのリコールです — 過去のすべてのセッションが検索可能で、このツールが何が起きたかを要約します。次のときに積極的に使用してください: - ユーザーが「以前これをやった」「覚えている？」「前回…」と言ったとき… | — |

## `skills` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `skill_manage` | スキルを管理する（作成、更新、削除）。スキルはあなたの手続き的記憶です — 繰り返し発生するタスクタイプのための再利用可能なアプローチです。新しいスキルは ~/.hermes/skills/ に作られ、既存のスキルはそれがある場所で変更できます。アクション: create（完全な SKILL.m…） | — |
| `skill_view` | スキルを使うと、特定のタスクやワークフローに関する情報のほか、スクリプトやテンプレートを読み込めます。スキルの完全な内容を読み込むか、そのリンクされたファイル（リファレンス、テンプレート、スクリプト）にアクセスします。最初の呼び出しは SKILL.md の内容と…を返します… | — |
| `skills_list` | 利用可能なスキル（名前 + 説明）を一覧表示する。完全な内容を読み込むには skill_view(name) を使用します。 | — |

## `terminal` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `process` | terminal(background=true) で開始したバックグラウンドプロセスを管理する。アクション: 'list'（すべて表示）、'poll'（ステータス + 新しい出力を確認）、'log'（ページ分割付きの全出力）、'wait'（完了またはタイムアウトまでブロック）、'kill'（終了）、'write'（送信…） | — |
| `terminal` | Linux環境でシェルコマンドを実行する。ファイルシステムは呼び出し間で保持されます。長時間実行されるサーバーには `background=true` を設定します。`notify_on_complete=true`（`background=true` と併用）を設定すると、プロセス終了時に自動通知を受け取れます — ポーリング不要です。cat/head/tail は使わないでください — read_file を使用します。grep/rg/find は使わないでください — search_files を使用します。 | — |

## `todo` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `todo` | 現在のセッションのタスクリストを管理する。3ステップ以上の複雑なタスク、またはユーザーが複数のタスクを提供したときに使用します。パラメータなしで呼び出すと現在のリストを読みます。書き込み: - 項目を作成/更新するには 'todos' 配列を提供 - merge=… | — |

## `vision` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `vision_analyze` | AIビジョンを使って画像を分析する。包括的な説明を提供し、画像の内容に関する特定の質問に答えます。 | — |

## `video` ツールセット

オプトインのツールセット（デフォルトの `hermes-cli` セットには読み込まれません）。`--toolsets video` で追加するか、`toolsets:` 設定に `video` を含めます。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `video_analyze` | URLまたはファイルパスから動画コンテンツを分析する — キャプション、シーンの分解、重要なタイムスタンプ、視覚的な説明。 | — |

## `web` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `web_search` | ウェブで情報を検索する。デフォルトで最大5件の結果を、タイトル、URL、説明とともに返します。任意の `limit`（1-100、デフォルト5）を受け付けます。クエリは設定されたバックエンドにそのまま渡されるため、`site:domain`、`filetype:pdf`、`intitle:word`、`-term`、`"exact phrase"` などの演算子は、バックエンドが対応していれば機能する場合があります。 | EXA_API_KEY または PARALLEL_API_KEY または FIRECRAWL_API_KEY または TAVILY_API_KEY |
| `web_extract` | ウェブページのURLからコンテンツを抽出する。ページの内容をmarkdown形式で返します。PDFのURLにも対応 — PDFのリンクを直接渡すと、markdownテキストに変換します。5000文字未満のページは完全なmarkdownを返し、より大きなページはLLMで要約されます。 | EXA_API_KEY または PARALLEL_API_KEY または FIRECRAWL_API_KEY または TAVILY_API_KEY |

## `tts` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `text_to_speech` | テキストを音声オーディオに変換する。プラットフォームがボイスメッセージとして配信する MEDIA: パスを返します。Telegramではボイスバブルとして再生され、Discord/WhatsAppではオーディオの添付として再生されます。CLIモードでは ~/voice-memos/ に保存します。ボイスとプロバイダー… | — |

## `discord` ツールセット

`hermes-discord` プラットフォームのツールセット（ゲートウェイのみ）に登録されます。メッセージングアダプターと同じボットトークンを使用します。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `discord` | Discordサーバーを読み、参加する。アクションには `search_members`、`fetch_messages`、`send_message`、`react`、`fetch_channel`、`list_channels` などが含まれます。 | `DISCORD_BOT_TOKEN` |

## `discord_admin` ツールセット

`hermes-discord` プラットフォームのツールセットに登録されます。モデレーションアクションには、ボットが対応するDiscordの権限を保持している必要があります。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `discord_admin` | REST API経由でDiscordサーバーを管理する: ギルド/チャンネル/ロールの一覧表示、チャンネルの作成/編集/削除、ロール付与の管理、タイムアウト、キック、バン。 | `DISCORD_BOT_TOKEN` + ボットの権限 |

## `spotify` ツールセット

バンドルされた `spotify` プラグインによって登録されます。OAuthトークンが必要 — 認可するには一度 `hermes spotify setup` を実行します。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `spotify_playback` | Spotifyの再生を制御し、アクティブな再生状態を確認し、または最近再生したトラックを取得する。 | Spotify OAuth |
| `spotify_devices` | Spotify Connectデバイスを一覧表示する、または再生を別のデバイスに移す。 | Spotify OAuth |
| `spotify_queue` | ユーザーのSpotifyキューを確認する、またはそこに項目を追加する。 | Spotify OAuth |
| `spotify_search` | Spotifyのカタログでトラック、アルバム、アーティスト、プレイリスト、ショー、エピソードを検索する。 | Spotify OAuth |
| `spotify_playlists` | Spotifyのプレイリストを一覧表示、確認、作成、更新、変更する。 | Spotify OAuth |
| `spotify_albums` | Spotifyのアルバムのメタデータまたはアルバムのトラックを取得する。 | Spotify OAuth |
| `spotify_library` | ユーザーが保存したSpotifyのトラックまたはアルバムを一覧表示、保存、削除する。 | Spotify OAuth |

## `hermes-yuanbao` ツールセット

`hermes-yuanbao` プラットフォームのツールセットにのみ登録されます。YuanbaoはTencentのチャットアプリで、これらのツールはそのDM/グループ/スタンプAPIを駆動します。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `yb_query_group_info` | グループ（アプリ内では「派/Pai」と呼ばれる）の基本情報を照会する: 名前、オーナー、メンバー数。 | Yuanbaoの認証情報 |
| `yb_query_group_members` | グループのメンバーを照会する（`@` メンション、名前でのユーザー検索、ボットの一覧表示のため）。 | Yuanbaoの認証情報 |
| `yb_send_dm` | グループ内のユーザーにプライベート/ダイレクトメッセージを送る。任意でメディアファイル付き。 | Yuanbaoの認証情報 |
| `yb_search_sticker` | 組み込みのYuanbaoスタンプ（TIMフェイス）カタログをキーワードで検索する。 | Yuanbaoの認証情報 |
| `yb_send_sticker` | 現在のYuanbaoチャットに組み込みのスタンプを送る。 | Yuanbaoの認証情報 |
