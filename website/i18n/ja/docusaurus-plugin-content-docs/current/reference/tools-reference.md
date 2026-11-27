---
sidebar_position: 3
title: "組み込みツールリファレンス"
description: "ツールセットごとに分類した、Hermesの組み込みツールの正式なリファレンス"
---

# 組み込みツールリファレンス

このページでは、Hermesの組み込みツールをツールセットごとに分類して説明します。利用可否は、プラットフォーム、認証情報、有効になっているツールセットによって異なります。

**概数（現在のレジストリ）:** 約85ツール — 4個のブラウザツール（`browser_exec`、`browser_cdp`、`browser_dialog`、`browser_vault`）、4個のファイルツール、2個のターミナルツール（`terminal`、`process_manage`）、11個のデスクトップGUIツール（`read_terminal`、`close_terminal`、`desktop_preview`、`drive_preview`、`annotate_preview`、`read_window_below`、`focus_pane`、`react_to_message`、`gui_tour`、`show_tip`、`set_theme` — デスクトップアプリのセッションのみ）、2個のウェブツール、5個のFeishuツール、14個のkanbanツール（kanbanディスパッチャーがエージェントを起動するときに登録）、1個のプロジェクトツール（`desktop_project`。デスクトップ/GUIセッション）、2個のDiscordツール、3個の動画ツール（`video_generate`、`xai_video_edit`、`xai_video_extend`）、そしてひと握りのスタンドアロンツール（`memory`、`clarify`、`delegate_task`、`execute_code`、`cronjob_manage`、`session_search`、`skill_view`/`skill_manage`/`skills_list`、`text_to_speech`、`image_generate`、`vision_analyze`、`video_analyze`、`todo_list`、`computer_use`、`x_search`）。

:::tip MCPツール
組み込みツールに加えて、HermesはMCPサーバーからツールを動的に読み込めます。MCPツールは `mcp__<server>__` というプレフィックス付きで表示されます（例: `github` MCPサーバーの `mcp__github__create_issue`）。設定については [MCP連携](../user-guide/features/mcp.md) を参照してください。
:::

## `browser` ツールセット

`browser_exec` と `browser_vault` はブラウザセッション — Hermesが起動するChromium、クラウドブラウザ、または `/browser connect` で接続したブラウザ — を操作します。`browser_cdp` と `browser_dialog` は、CDPエンドポイントが設定されている間（`/browser connect`、`BROWSER_CDP_URL` または `browser.cdp_url`）にのみ提供され、そのブラウザに対して動作します。起動したセッションやクラウドセッションには存在しません。[ブラウザ自動化](../user-guide/features/browser.md) を参照してください。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `browser_exec` | 事前にインポートされたヘルパー（`new_tab`、`goto_url`、`wait_for_load`、`page_info`、`js`、`fill_input`、`click_at_xy`、`capture_screenshot`、`cdp`、`ensure_real_tab`）付きでPythonを実行し、実際のウェブブラウザを操作する。ブラウザの状態は呼び出しをまたいで保持されますが、Pythonの変数は保持されません。`session` で並行作業を分離します。`timeout_s` のデフォルトは300（最大1800）です。 | `terminal` ツールセット |
| `browser_cdp` | 生のChrome DevTools Protocolコマンドを送信する。`browser_exec` のヘルパーでカバーされないブラウザ操作のための脱出ハッチ。https://chromedevtools.github.io/devtools-protocol/ を参照。 | CDPエンドポイント |
| `browser_dialog` | ページをブロックしているネイティブのJavaScriptダイアログ（alert / confirm / prompt / beforeunload）に応答する: `browser_dialog(action='accept'\|'dismiss')`。`prompt()` には `prompt_text` を、複数がキューに入っている場合は `dialog_id` を指定します。 | CDPエンドポイント |
| `browser_vault` | 保存されたログイン、パスワード、カード、住所、ワンタイムコードを、シークレットを会話に入れることなく現在のページで使う。アクション: `list`、`fill`、`save_login`、`enter_code`、`unlock`。 | — |

## `clarify` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `clarify` | 進める前に、明確化、フィードバック、または決定が必要なときにユーザーに1つ以上の質問をする。すべての呼び出しで `questions`（1〜5エントリの配列。質問が1つなら1エントリの配列）を渡します。各質問は3つのモードに対応します: 1. **単一選択の多肢選択** — 最大4つの選択肢。ユーザーは1つを選ぶか、5番目の「Other」オプションから自分の答えを入力します。2. **複数選択の多肢選択** — `multi_select=true` でチェックボックスを表示し、選択された選択肢のリストを返します。3. **自由回答** — 選択肢なし。ユーザーは自由形式で回答を入力します。選択肢は最良のものから順に並べるため、最初の選択肢はすべてのサーフェスで `(Recommended)` とラベル付けされ、デフォルトでハイライトされます。ラベルは表示上のものだけで、エージェントが読む回答からは取り除かれます。クラシックCLIでは複数選択はSpaceで切り替えるチェックボックスを使います。ネイティブのチェックボックスUIがないメッセージングプラットフォームでは、ユーザーはカンマ/スペース区切りの番号（例: "1, 3"）または選択肢のテキストで返信します。 | — |

### 複数の質問を一度に尋ねる

`clarify` ツールは `questions` 配列（1〜5個の独立した質問。それぞれが独自の `choices` と `multi_select` を持つ）を受け取るため、エージェントは複数の確認事項を順番に尋ねる代わりに、1つのプロンプトにまとめられます。結果は `{responses, outcome}` です: `responses` は同じ順序の配列で、各エントリには質問テキスト、`choices_offered`、`status`（`answered`、`skipped`、`unanswered`）、`user_response`（回答済みでなければnull）が含まれます。`outcome` は `submitted`、`cancelled`、`timed_out`、`undelivered` のいずれかです。

サーフェスごとの動作:

- **Desktop** はすべての質問を1枚のカードに表示します。選択や入力した回答はローカルに保持され、1つの **Confirm and continue** ボタン（少なくとも1つの質問に回答があると有効になる）でバッチ全体を送信します。空の質問はスキップされます。保持された回答は確定するまで編集できます。Skipはバッチ全体をキャンセルします。
- **TUIとCLI** はコンパクトなステータスリスト（`✓` 回答済み / `▸` アクティブ / `·` 保留中）を表示し、アクティブな質問の選択肢だけを展開します。Enterでアクティブな回答を確定し、次の未回答の質問にジャンプします。Tabで質問間を移動して任意の順序で回答できます。空のまま送信するとその質問をスキップします。Escでバッチをキャンセルします。
- **メッセージングプラットフォーム**（Telegram、Discord、…）は質問を1つずつ、質問ごとに1枚のカードで尋ねます。1つの質問をスキップするには `skip` と返信します。ユーザーが応答しなくなった場合、残りの質問は送信されません。

プロンプトが途中でタイムアウトした場合、ユーザーがすでに確定した回答は保持されます: ツールの結果は `"outcome": "timed_out"` とともにそれらを運び、残りを `"status": "unanswered"` とマークするため、エージェントは意図的なスキップとユーザーの不在を区別できます。メッセージングプラットフォームでは、結果に待機が終了した理由を示す `"notice"` も含まれます（`[user did not respond within Nm]`、またはプラットフォームがカードを拒否した場合は `[clarify prompt could not be delivered]` — Hermesはまず質問をプレーンな番号付きリストのメッセージとして再試行し、それも失敗した場合にのみこれを報告します。実行にプロンプトを出すチャットがない場合は `[clarify prompt could not be delivered: no chat surface]`）。そのため、配信されなかったプロンプトがユーザーの無応答として報告されることはありません。

## `connections` ツールセット

両方の種類の外部アプリに対応する1つのツールです。ターゲットは、マネージドコネクター（`"gmail"` または
`{"name": "gmail"}`。Nousゲートウェイを通じて認可）か、ローカルのMCPサーバー
（`{"name": "linear", "mcp": true}`。`mcp_servers` のエントリ）です。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `manage_connections` | マネージドのアクション: `display`（デスクトップアプリのみ: ホストされたすべてのコネクターのカードを、よく使うアプリから順に表示し、アプリをクリックするとそのサインインが始まります。`connectors` のスラッグはSuggestedとして先頭に固定されます。ユーザーがContinueを押すか300秒経過するまで待機し、`connected`、`newly_connected`、`suggested` のスラッグを返し、期限で終了した場合は `timed_out: true` を追加します。他のサーフェスでは `status` のリストを返し、デスクトップアプリまたはポータルから接続するようユーザーに伝えます）、`status`（各コネクターはその `accounts`: エイリアス、ラベル、ステータス、アクティブかどうかを一覧表示します）、`connect`、`reconnect`（接続されていないものだけを修復します。`force: true` は動作中のものを再起動します）、`rename`。connectやreconnectのターゲットにはアカウント名を含められます（`{"name": "gmail", "alias": "work"}`）: 新しい名前でのconnectは別のアカウントを追加し、名前付きのreconnectはそのアカウントだけを修復し、何も削除しません。`rename` は1つのターゲット `{"name": "gmail", "alias": "work", "to": "office"}` を取ります。`alias` には名前のないアカウントのアプリラベルを指定できます。MCPのアクション（`mcp: true` のターゲットのみ）: カタログエントリの `install`、無効化された設定済みサーバーの `enable`、`authorize`（OAuth）。デスクトップアプリ、ターミナルUI、クラシックCLIでは、すべてのアクションがカードを表示し、各ターゲットが接続されるか、スキップされるか、300秒の期限が過ぎるまでブロックします。結果はターゲットを `connected`、`skipped`、`not_connected` として一覧表示し、リンクは含みません。MCPの `install` は、エントリのセットアップ値をカードで収集し、エントリにOAuthがある場合はカードから実行し（ユーザーがリンクを開きます。自動的に開くものはありません）、サーバーがトークンを受け入れると設定、トークン、値をまとめて保存します。接続されたMCPターゲットは `tools`（登録された名前）と `tools_listing` を持ち、それらのツールは同じターン内で `tool_describe`/`tool_call` を通じて呼び出せます。`discovery_error` を持つターゲットは認可済みですが、そのツールを一覧表示できませんでした。新たな同意なしにディスカバリーを再試行するには、そのターゲットに対して再度 `install` または `authorize` を呼び出します。カードのないサーフェス（メッセージング、スクリプトによるディスパッチ）では、マネージドターゲットはユーザーが開くためのアプリごとの `connect_url` を返し、MCPターゲットは即座に実行されます: `authorize` とOAuthの `install` は認可URLを返し、その他のinstallと `enable` は結果を報告し、認証情報が欠けている場合は設定すべき変数名を示した `failed` が返されます。アカウントの切断、削除、取り消しはできません。それはユーザーが行います。 | — |

1回の呼び出しの期限は5分で、呼び出しの開始時にバックエンドによって固定されます。
チャットを開き直したりデスクトップを再起動したりしても延長されることはありません。このツールは、
Nous Portalがサインイン中のアカウントに対してコネクターを有効にしている場合（そのトークンの
`managed_tools` クレーム）にのみ存在します。それ以外のセッションには表示されません。

## `code_execution` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `execute_code` | Hermesツールをプログラム的に呼び出せるPythonスクリプトを実行する。3回以上のツール呼び出しの間に処理ロジックが必要なとき、大きなツール出力をコンテキストに入れる前にフィルタ/縮約する必要があるとき、条件分岐が必要なとき（… | — |

## `cronjob` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `cronjob_manage` | 統合スケジュールタスクマネージャー。`action="create"`、`"list"`、`"update"`、`"pause"`、`"resume"`、`"run"`、`"remove"` を使ってジョブを管理します。1つ以上のスキルを付属させたスキルベースのジョブに対応し、update時の `skills=[]` は付属スキルをクリアします。cronの実行は、現在のチャットのコンテキストを持たない新しいセッションで行われます。 | — |

## `delegation` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `delegate_task` | 分離されたコンテキストでサブエージェントを起動する。各サブエージェントは独自の会話、ターミナルセッション、ツールセットを持ち、最終的なサマリーだけがあなたに返されます。単一のタスクには 'goal' を、並列バッチには 'tasks' を指定します（制限とネストのルール… | — |

## `feishu_doc` ツールセット

Feishuのドキュメントコメントのインテリジェント返信ハンドラー（`gateway/platforms/feishu_comment.py`）にスコープされています。`hermes-cli` や通常のFeishuチャットアダプターには公開されません。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `feishu_doc_read` | file_type とトークンを指定して、Feishu/Larkのドキュメント（Docx、Doc、Sheet）の全文テキストを読み取る。 | Feishuアプリの認証情報 |

## `feishu_drive` ツールセット

Feishuのドキュメントコメントハンドラーにスコープされています。ドライブファイルに対するコメントの読み書き操作を行います。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `feishu_drive_add_comment` | Feishu/Larkのドキュメントまたはファイルにトップレベルのコメントを追加する。 | Feishuアプリの認証情報 |
| `feishu_drive_list_comments` | Feishu/Larkファイルのドキュメント全体へのコメントを、新しい順に一覧表示する。 | Feishuアプリの認証情報 |
| `feishu_drive_list_comment_replies` | 特定のFeishuコメントスレッド（ドキュメント全体またはローカル選択）の返信を一覧表示する。 | Feishuアプリの認証情報 |
| `feishu_drive_reply_comment` | Feishuのコメントスレッドに返信を投稿する。任意で `@` メンションを付けられます。 | Feishuアプリの認証情報 |

## `file` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `patch` | ファイル内の的を絞った検索・置換編集。ターミナルでの sed/awk の代わりにこれを使用します。ファジーマッチング（9つの戦略）を使うため、わずかな空白/インデントの差で壊れることはありません。統合diffを返します。編集後に自動で構文チェックを実行… | — |
| `read_file` | 行番号とページ分割付きでテキストファイルを読む。ターミナルでの cat/head/tail の代わりにこれを使用します。出力形式: 'LINE_NUM\|CONTENT'。見つからない場合は似たファイル名を提案します。大きなファイルには offset と limit を使用します。約100K文字を超える読み取りは行の境界で切り詰められ、next_offset を返します。Jupyterノートブック（.ipynb）、Wordドキュメント（.docx）、Excelワークブック（.xlsx）は… | — |
| `search_files` | ファイルの内容を検索する、または名前でファイルを探す。ターミナルでの grep/rg/find/ls の代わりにこれを使用します。Ripgrepベースで、シェルの相当品より高速です。コンテンツ検索（target='content'）: ファイル内の正規表現検索。出力モード: 行…付きの完全な一致… | — |
| `write_file` | ファイルに内容を書き込み、既存の内容を完全に置き換える。ターミナルでの echo/cat heredoc の代わりにこれを使用します。親ディレクトリを自動的に作成します。ファイル全体を上書きします — 的を絞った編集には 'patch' を使用してください。既存のファイルの場合は、先に read_file を呼び出してください: タスクがそのファイルの現在の完全な読み取り/書き込みを持っていない場合、またはその後ファイルがディスク上で変更された場合、write_file は拒否します（ファイルは変更されません） — 拒否されたら read_file し、マージして再試行します。.py/.json/.yaml/.toml やその他のリント対象言語では自動で構文チェックを実行し、書き込みによって新たに発生したエラーだけを表示します。 | — |

ローカルファイルの場合、伏せ字なしの完全な読み取り（同じファイルバージョンのすべてのページを含む）、成功した `write_file`、または完全に読み取ったファイルに対する自分自身の `patch` が、ファイル全体のベースラインを提供します。その後により小さな範囲を読み取っても、バイトが変わらない限りそのベースラインは破棄されません。別の書き込み元によって変更されたファイル、未読のファイル、完全な読み取りなしにパッチされたファイル、または非表示/伏せ字/切り詰められた内容を含むビューは、置き換えの前に現在の完全な読み取りが引き続き必要です。的を絞った編集には `patch` を引き続き利用できます。ターミナルコマンドや `execute_code` を通じた書き込みは、`write_file` のベースラインを確立しません。

## `computer_use` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `computer_use` | cua-driver によるバックグラウンドのデスクトップ制御 — スクリーンショット（SOM / vision / AX）、click / drag / scroll / type / key / wait、list_apps、focus_app。ユーザーのカーソルやキーボードフォーカスを奪いません。ツール対応の任意のモデルで動作します。macOS、Windows、Linux。 | `$PATH` 上の `cua-driver`（`hermes tools` でインストール）。 |


:::note
**Honchoツール**（`honcho_profile`、`honcho_search`、`honcho_context`、`honcho_reasoning`、`honcho_conclude`）はもはや組み込みではありません。これらはプラグインカタログのHonchoメモリプロバイダープラグイン（`hermes plugins install honcho`）経由で利用できます。インストールと使い方については [メモリプロバイダー](../user-guide/features/memory-providers.md) を参照してください。

**Home Assistantツール**（`ha_list_entities`、`ha_get_state`、`ha_list_services`、`ha_call_service`、ツールセット `homeassistant`）はもはや組み込みではありません。これらは `homeassistant` カタログプラグイン（`hermes plugins install homeassistant`）から提供されます。[Home Assistant](../user-guide/messaging/homeassistant.md) を参照してください。

**Spotifyツール**（`spotify_playback`、`spotify_devices`、`spotify_queue`、`spotify_search`、`spotify_playlists`、`spotify_albums`、`spotify_library`、ツールセット `spotify`）はもはや組み込みではありません。これらは `spotify` カタログプラグイン（`hermes plugins install spotify` の後に `hermes spotify login`）から提供されます。[Spotify](../user-guide/features/spotify.md) を参照してください。

**Yuanbaoツール**（`yb_query_group_info`、`yb_query_group_members`、`yb_send_dm`、`yb_search_sticker`、`yb_send_sticker`、ツールセット `yuanbao`）はもはや組み込みではありません。これらは `yuanbao` Hermes Labsプラグイン（`hermes plugins install yuanbao`）から提供されます。[Yuanbao](../user-guide/messaging/yuanbao.md) を参照してください。
:::

## `image_gen` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `image_generate` | ユーザーが設定したバックエンド（FAL.ai、OpenAI、OpenAI Codex認証、xAI、Krea）を介して、テキストプロンプトから画像を生成する（text-to-image）か、既存の画像を編集/変換する（image-to-image）。画像を編集するには `image_url` を、スタイルの参照には `reference_image_urls` を渡します。text-to-imageでは両方とも省略します。モデルはユーザー設定であり、エージェントは選択できません。単一の画像URLまたはローカルパスを返します。 | FAL_KEY / OPENAI_API_KEY / Codex OAuth / xAI OAuth / KREA_API_KEY |

## `kanban` ツールセット

エージェントが (a) kanbanディスパッチャーによって起動された場合（環境変数 `HERMES_KANBAN_TASK` が設定されている）、または (b) `kanban` ツールセットを明示的に有効にしたプロファイルで実行されている場合に登録されます。タスクにスコープされたワーカーは、割り当てられたタスクに対してライフサイクルツールを使います。オーケストレーターのプロファイルは、さらに `kanban_list` や `kanban_unblock` のようなボードのルーティングツールを得ます。完全なワークフローについては [Kanbanマルチエージェント](../user-guide/features/kanban.md) を参照してください。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `kanban_show` | このワーカーに割り当てられたアクティブなkanbanタスク（タイトル、説明、コメント、依存関係）を表示する。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_list` | フィルター付きでボードのタスクを一覧表示する。オーケストレーター専用で、ディスパッチャーが起動したタスクワーカーには表示されません。 | `kanban` ツールセットを持つプロファイル |
| `kanban_complete` | 構造化されたハンドオフのペイロード（結果、成果物、フォローアップ）とともに現在のタスクを完了とマークする。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_block` | 現在のタスクをユーザーへの質問でブロックする — ディスパッチャーは一時停止し、質問を表面化し、人間が返信したら再開します。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_request_review` | `summary`、任意の構造化された `metadata`、任意のレビュアープロファイルとともに、実装をレビュアーに引き渡す。同じタスクを `review` に移動します。これはブロックではなく、ブロックループのカウントにも影響しません。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_request_changes` | アクティブにクレームされたレビュー実行に対するレビュアーの判定。レビュー実行を閉じ、親のゲートを再適用し、ブロックを使わずにタスクを元の実装者に戻します。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_heartbeat` | 長時間実行される操作の間に進捗のハートビートを送り、ディスパッチャーにワーカーがまだ生きていることを知らせる。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_comment` | 状態を変えずにタスクスレッドにコメントを追加する — 中間的な発見を表面化するのに便利です。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_create` | 現在のタスクから子タスクへファンアウトする。オーケストレーターや、フォローアップを生成するワーカーが使います。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_link` | 親 → 子の依存関係エッジでタスク同士をリンクする。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_unblock` | ブロックされたタスクを、すべての親が完了していれば `ready` に、いずれかの親がまだ開いていれば `todo` に移動する。オーケストレーター専用で、ディスパッチャーが起動したタスクワーカーには表示されません。 | `kanban` ツールセットを持つプロファイル |
| `kanban_attach` | バイトをインライン（base64）で渡してタスクにファイルを添付する。タスクの添付ディレクトリ配下に実際の添付ファイルとして保存され、上限は25 MBです。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_attach_url` | URLでタスクにファイルを添付する — Hermesがサーバー側でダウンロードし、実際の添付ファイルとして保存します（上限25 MB）。http/httpsのURLのみ。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |
| `kanban_attachments` | タスクに添付されたファイルを一覧表示する: id、ファイル名、content_type、サイズ、アップロード者、ディスク上の絶対パス。 | `HERMES_KANBAN_TASK` または `kanban` ツールセット |

## `project` ツールセット

デスクトップの [Projects](../user-guide/cli.md) — 名前付きの複数フォルダーのワークスペース — を操作するためのツールです。`project` ツールセットが有効な場合（主にデスクトップアプリ / ダッシュボードのサーフェス）に登録されます。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `desktop_project` | Projectの3つの操作を1つのアクションenumで扱う: `create` はデスクトップのProject（名前付きワークスペース）を作成し、このチャットをそこに切り替えます — リポジトリ/フォルダーに紐づけるには `path` を渡します。`list` はデスクトップのProjectと、どれがアクティブかを表示します。`switch` はこのチャットを既存のProject（名前、スラッグ、またはidで指定）に移動し、セッションのワークスペースをプロジェクトのプライマリフォルダーに移します。 | — |

## `memory` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `memory` | セッションをまたいで残る永続メモリに重要な情報を保存する。メモリはセッション開始時にシステムプロンプトに表示されます -- これが、会話の間でユーザーや環境について覚えておく方法です。保存するタイミング… | — |

## `catalog` ツールセット

ソースがデスクトップアプリであるセッションで、接続先のバックエンドを問わず有効になります。CLI、TUI、`hermes -z`、cron、kanban、メッセージングのセッションは、設定のリストで指定されていても決してこれを得ません。`all` にも含まれません。`agent.disabled_toolsets: [catalog]` で削除できます。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `manage_catalog` | デスクトップセッション専用で、デフォルトでは `tool_search` の背後に遅延されます。`search` は `query`（任意で1つの `kind`）に一致するカタログプラグインとハブのスキルを、`id`、`kind`、`display`、`tier`、`platforms`、`installed`（このチャットのプロファイルに存在するか）とともに一覧表示します。何も変更しません。`install` は `items: [{kind, id}]` を受け取り、アイテムごとに1行（Install、Advanced、Skip）を持つ1枚の承認カードを表示します。カタログが知らないidや、このOSで実行できないプラグインは、理由とともに失敗として表示されます。承認された行は、カタログのレビュー済みコミットで、Pluginsタブと同じキルリスト、セキュリティスキャン、ライブアクティベーションを伴って、このチャットのプロファイル（またはAdvancedで選んだプロファイル）にインストールされるため、プラグインのMCPツールとスキルはそのプロファイルの開いているチャットですぐに使えます。結果は各行を `connected`（`tools` と `skill` 付き）、`skipped`、`failed`（`detail` 付き）、`not_connected` として一覧表示します。失敗した行はインストール済みやスキップ済みの行と同様に待機を終了させ、installはディスク上にあるが有効になっていないカタログプラグインを有効にします。モデルはソース、コミット、プロファイル、設定を渡せません。それ以外の場所では、呼び出しは代わりに実行すべき `hermes plugins install` / `hermes skills install` コマンドを返します。 | — |

## `session_search` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `session_search` | ローカルのセッションDBに保存された過去のセッションを検索する、またはその中をスクロールする。FTS5ベースの取得で、DBから実際のメッセージを返します（LLM呼び出しなし）。4つの形式: ディスカバリー（`query` を渡す）、スクロール（`session_id` + `around_message_id` を渡す）、読み取り（`session_id` のみを渡す）、ブラウズ（引数なし）。ディスカバリーは時間の範囲（`after`/`before` — ISO日付、または `7d`、`24h`、`2w` のような相対期間）と、反復的な再検索のための `exclude_session_ids` に対応します。 | — |

## `skills` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `skill_manage` | スキルを管理する（作成、更新、削除）。スキルはあなたの手続き的記憶です — 繰り返し発生するタスクタイプのための再利用可能なアプローチ。新しいスキルは ~/.hermes/skills/ に保存されます。既存のスキルは、どこにあっても変更できます。アクション: create（完全な SKILL.m… | — |
| `skill_view` | スキルを使うと、特定のタスクやワークフローに関する情報、さらにスクリプトやテンプレートを読み込めます。スキルの完全な内容を読み込むか、リンクされたファイル（references、templates、scripts）にアクセスします。最初の呼び出しは SKILL.md の内容と… | — |
| `skills_list` | 利用可能なスキル（名前 + 説明）を一覧表示する。完全な内容を読み込むには skill_view(name) を使います。 | — |

## `terminal` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `process_manage` | terminal(background=true) で開始したバックグラウンドプロセスを管理する。アクション: 'list'（すべて表示）、'poll'（ステータス + 新しい出力を確認）、'log'（ページ分割付きの完全な出力）、'wait'（完了またはタイムアウトまでブロック）、'kill'（終了）、'write'（送… | — |
| `terminal` | Linux環境でシェルコマンドを実行する。ファイルシステムは呼び出し間で保持されます。長時間実行するサーバーには `background=true` を設定します。（`background=true` とともに）`notify_on_complete=true` を設定すると、プロセスの終了時に自動的に通知を受け取れます — ポーリングは不要です。さらに `heartbeat=N`（秒、最小60）を追加すると、前回以降に生成された出力を含む定期的な通知も受け取れます — マージトレインや完全なテストスイートのような長い有限のジョブ向けで、失敗を終了時ではなくN秒以内に把握できます。新しい出力のないティックはスキップされ、Desktop/TUIではこの起床はトランスクリプトに表示されません（エージェントの返信だけが表示されます）。cat/head/tail は使わないでください — read_file を使います。grep/rg/find は使わないでください — search_files を使います。 | — |

## `desktop_ui` ツールセット

ソースがHermesデスクトップアプリであるセッションで、接続先のバックエンド
（ローカル、SSH、URL、Hermes Cloud）を問わず有効になります。CLI、TUI、
メッセージング、cronのセッションには存在しません。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `read_terminal` | HermesデスクトップGUIのアプリ内ターミナルペイン（このチャットの隣にある組み込みシェル）に現在表示されている内容を読み取る。 | — |
| `close_terminal` | HermesデスクトップGUIで、バックグラウンドプロセスの読み取り専用ターミナルタブを閉じる。プロセスはkillしません — タブ/ビューを閉じるだけです。停止するには process_manage(action='kill') を使います。 | — |
| `desktop_preview` | Hermesデスクトップアプリのチャットの隣にあるプレビューペインを操作する: ウェブURL、localhostの開発サーバーURL、またはファイルパスを `open` します（HTMLはライブでレンダリングされます）。ペイン全体（`url` を省略）またはその中の1つのタブ（URLまたはファイルパスを渡す）を `close` します。ペインに現在表示されている内容を `read` します — アプリ内ブラウザのページテキスト（URL + タイトル + レンダリングされたテキスト。`start`/`count` でページ分割可能）、またはファイル/アーティファクトタブの識別情報です。 | — |
| `drive_preview` | アプリ内ブラウザで開いているページを操作する: `elements` はクリック可能・入力可能なものを一覧化し（それぞれ `btn-sign-in` や `inp-email` のような名前を示すref、ロール、ラベル、値を持つ）、その後 `click`、`hover`、`type`、`scroll`、`press` がrefに対して動作し、`back`/`forward`/`reload` がペインの履歴を操作します。ポインターとキーボードは実際の入力であるため、ホバーメニューも開きます。refはページが遷移するまで有効で、要素を再構築する再レンダリングをまたいでも保持されるため、最初の一覧化の後は、各アクションがページ全体ではなく差分 — 追加、削除、変更、再バインドされたもの — だけで応答します。 | — |
| `annotate_preview` | アプリ内ブラウザで要素を枠で囲み、削除されるまでそのマークを残す — `drive_preview` が作業中に描く一時的な合図に対する、意図的な対応物です。`add` は任意の短いラベル付きでrefをマークし、`remove` は1つを外し、`clear` はすべてを外します。マークは要素に追従し、要素とともに消えるため、ナビゲーションでクリアされます。 | — |
| `read_window_below` | Hermesデスクトップウィンドウの真下にあるOSのウィンドウを識別する — アプリ名、タイトル、境界（メタデータのみで、ピクセルは決して取得しません）。macOSでは、他のアプリのタイトルは画面収録がすでに許可されている場合にのみ表示されます。ツールがその許可を求めることはありません。 | — |
| `focus_pane` | Hermesデスクトップアプリのペイン（chat、files、terminal、review、sessions）を表示してフォーカスする。 | — |
| `react_to_message` | iMessageのTapbackのように、メッセージに1つの絵文字でリアクションする。Settings → Appearance（`display.message_reactions`）でオプトインします。 | — |
| `gui_tour` | ライブのガイド付きツアーを行う: 画面を暗くし、要素をハイライトし、ナレーション付きのポップオーバーを添えます（driver.js）。Hermesアプリ自身のUIと、プレビューペインで開いている任意のページで動作します。ステップなしの `start` はアプリの組み込みツアー（`preset` は `quick` または `full`、デフォルトは `full`）を実行します。カスタムツアーでは、`targets` で画面上にあるものを発見し、`show` でステップごとにナレーションし、`steps` 付きの `start` でユーザーにNext/Prevの操作を渡します。 | — |
| `show_tip` | 小さなアクセントの吹き出しと矢印で1つの要素を指し示す — `gui_tour` の控えめな兄弟で、暗転、スポットライト、Next/Prevはありません。同じ `data-tour` ハンドルと、同じ `tour(action='targets')` のディスカバリー呼び出しを使います。 | — |
| `set_theme` | Hermesデスクトップアプリのアクセントカラー（スウォッチid、`theme`、または `#rrggbb`）、ライト/ダーク/システムモード、レイアウトプリセット（Simple: `sidebar-left`、`sidebar-right`。Advanced: `default`、`basic`、`focus`、`terminal-deck`、`quad`。または保存されたレイアウトのid）を設定する。結果は何が適用されたかを報告し、不明な値の場合は利用可能なidとともに返されます。`picker`（`accent` または `layout`）はチャットにピッカーカードを表示し、選択を最大5分間待ちます。 | — |

### ツアー

アプリ全体をざっと見て回るには、ステップなしの `gui_tour(action='start')` でデスクトップアプリの組み込みツアーを実行します。`preset='quick'` は4つの要点（sessions、composer、new session、model）を表示し、`preset='full'`（デフォルト）は現在のインターフェースモードの停止点を追加します。要素が画面上にない停止点はスキップされます。組み込みツアーは `surface='app'` でのみ実行され、`preset` は `steps` と組み合わせられません。

カスタムツアーでは、`gui_tour` ツールが自分でターゲットを発見します — `action='targets'` を呼び出すと、画面上のアドレス可能なすべての要素が、セレクター、ラベル、`stable` フラグとともに返されます。安定したセレクターは識別情報（`data-tour`、`id`、`data-testid`、`aria-label`）に基づいており、再レンダリング後も残ります。位置ベースの `nth-child` パスは残らないため、安定したものが先にソートされ、優先して使うべきです。

要素に自分で永続的なハンドルを付けるには、マークアップします:

```html
<div data-tour="composer">…</div>
```

ハンドルは呼び出し箇所ではなく **プリミティブ** に適用されるため、1回の編集ですべてのインスタンスに名前が付きます。すでに存在するもの:

| ハンドル | 名前を付ける対象 |
|---|---|
| `overlay-nav` | 任意のルートオーバーレイ（settings、cron、profiles、agents）の左ナビ |
| `nav-<id>` | そのナビの1行 — `nav-models`、`nav-appearance`、… |
| `field-<schemaKey>` | 設定キーによる1つの設定行 — `field-model`、`field-provider`、… |
| `page-tabs` | 任意の `PageSearchShell` ページ（artifacts、skills、…）のフィルタータブ |
| `artifact-card` | グリッド内のアーティファクトカード |

サーフェスを追加するときは、画面を1つずつタグ付けするのではなく、その共有プリミティブに同じ方法でタグを付けてください — そうすることでツアーの語彙が小さく保たれ、セレクターが劣化するのを防げます。

同じエンジンがデスクトップアプリのキュレーションされた（エージェントによらない）ツアーも支えているため、機能が独自のウォークスルーを同梱できます:

```ts
import { startTour, showTourStep, stopTour } from '@/lib/tour'

startTour([
  { selector: '[data-tour="composer"]', title: 'Composer', text: 'Type here.' },
  { selector: '[data-tour="files"]', title: 'Files', text: 'Browse your project.' }
])
```

ステップはアプリをターゲットのある場所に移動させることもでき、ツアーが終わると元に戻します:

```ts
startTour([
  { navigate: '/artifacts', selector: '[data-tour="page-tabs"]', title: 'Filters', text: '…' },
  { pane: 'sessions', selector: '[data-slot="sidebar"]', title: 'Sessions', text: '…' }
])
```

`navigate` はルートパスを、`pane` はデスクトップのペイン名を受け取ります。どちらもステップに入るときに実行され、遅れてマウントされるターゲットは待機され、ツアーを閉じると — Escを含むどの方法でも — 開始した場所に戻ります。

アプリの代わりにプレビューペインのページに対して実行するには、2番目の引数に `'preview'` を渡します。

### ヒント

ヒントは演出を省いたツアーのステップです: 1つの吹き出し、1つの矢印、スクリムなし、
ページ送りするものもなし。「モデル名はボタンです」のように、対象を指で示した方が
分かりやすくなる一文には、アプリ全体を暗くするよりもこちらがちょうどよい重さです。

`show_tip` ツールは `gui_tour(action='targets')` が報告するのと同じセレクターを受け取るため、
ディスカバリーは両方で1回の呼び出しで済み、上記の永続的な `data-tour` ハンドルは
どちらのターゲットにも名前を付けます。画面上に表示されるヒントは一度に1つで、新しいものが前のものを置き換えます。

アプリも独自のヒントを表示でき、アプリの機能の組み込みカタログを順番にたどります。
そのペースは通知ではなくゲームのロード画面のヒントのようなものです: 早くても起動から
数分後、その後は最大で6時間に1回、しかも本当にアイドルな瞬間にだけ表示されます。
Hermesからのヒントはそのクールダウンを共有するため、ユーザーにローテーションからの
6時間の静けさももたらします。ローテーションは1周だけです: 各カタログのヒントは、
タイムアウトしたか ✕ で閉じられたかにかかわらず1回だけ表示され、すべてのヒントが
一巡するとアプリは静かになります。設定行から1周をやり直せます。

ヒントとツアーはどちらもデフォルトでオンで、Settings → Appearance
（`display.in_app_tips`、`display.in_app_tours`）でオフにできます。オフはアプリだけでなく
Hermesにも適用されます: スイッチは接続されたゲートウェイの設定に反映され、ツールは
モデルのスキーマから外れるため、エージェントが使用を許されていないサーフェスについて
知らされることはありません。他のスキーマ変更と同様に、これは次のセッションから反映されます —
実行中の会話は開始時のツールセットを保持し、その間アプリは呼び出しを拒否します。

## `todo` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `todo_list` | 現在のセッションのタスクリストを管理する。3ステップ以上の複雑なタスクや、ユーザーが複数のタスクを提示したときに使います。現在のリストを読むには、パラメータなしで呼び出します。アイテムはネストできます: アイテムの任意の `parent` フィールドが別のアイテムのidを指すとサブタスクになり、サーフェスはツリーをインデントして表示します。 | — |

## `vision` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `vision_analyze` | AIビジョンを使って画像を分析する。ビジョン対応のメインモデルでは、生の画像ピクセルをマルチモーダルなツール結果として返すため、モデルは次のターンでネイティブにそれを見ます。テキストのみのメインモデルでは、画像を説明する補助ビジョンモデルにフォールバックし、その説明をテキストとして返します。ツールのシグネチャはどちらの場合も同じです。 | — |

## `video` ツールセット

オプトインのツールセットです（デフォルトの `hermes-cli` セットには読み込まれません）。`--toolsets video` で追加するか、`toolsets:` 設定に `video` を含めます。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `video_analyze` | URLまたはファイルパスから動画の内容を分析する — キャプション、シーンの内訳、主要なタイムスタンプ、視覚的な説明。 | — |

## `video_gen` ツールセット

オプトインのツールセットです（デフォルトの `hermes-cli` セットには読み込まれません）。`--toolsets video_gen` で追加するか、`hermes tools` → Video Generation で有効にします。後者ではバックエンドの選択も案内されます。

バックエンドは `plugins/video_gen/<name>/` 配下のプラグインとして提供されます:

- **xAI Grok-Imagine** — text-to-videoとimage-to-video（SuperGrok OAuthまたは `XAI_API_KEY`）。
- **FAL.ai** — Veo 3.1、Pixverse v6、Kling 3.0 / O3（`FAL_KEY` が必要）。
- **OpenRouter** — OpenRouterの動画APIにあるすべての生成モデル（Veo 3.1、Sora 2 Pro、Kling 3、Seedance 2、Wan 3、Hailuo 3、Grok Imagine、FLUX 3 Video、…）。text-to-video、image-to-video、reference-to-video。カタログとモデルごとの制限はライブで取得されます（`OPENROUTER_API_KEY` または `hermes auth add openrouter` で追加した認証情報が必要で、OpenRouterのクレジットに課金されます）。
- **DeepInfra** — OpenAI互換のvideosエンドポイント経由のライブな `video-gen` カタログ（`DEEPINFRA_API_KEY` が必要）。

単一の `video_generate` ツールが両方のモダリティをカバーします — 静止画をアニメーション化するには `image_url` を渡し、テキストだけから生成するには省略します。アクティブなバックエンドが適切なエンドポイントに自動的にルーティングします。`image_generate` と同様に、モデルはユーザー設定（`video_gen.model`）であり、エージェントは選択できません — どの動画ツールも `model` 引数を取りません。ツールの説明は、アクティブなバックエンドの実際の能力（モダリティ、アスペクト比、解像度、長さの範囲、最大参照画像数、音声サポート）を反映するよう、セッション開始時に再構築されます。バックエンドの作成については [動画生成プロバイダープラグイン](/docs/developer-guide/video-gen-provider-plugin) を参照してください。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `video_generate` | ユーザーが設定した動画生成バックエンドを使って、テキストプロンプトから動画を生成する（text-to-video）か、静止画をアニメーション化する（image-to-video）。その画像をアニメーション化するには `image_url` を渡し、テキストだけから生成するには省略します。バックエンドが適切なエンドポイントに自動的にルーティングします。`video` フィールドでHTTP URLまたは絶対ファイルパスを返します。 | アクティブな `video_gen` プラグイン + その認証情報（例: `XAI_API_KEY`、`FAL_KEY`） |
| `xai_video_edit` | xAI Imagineで既存の動画を編集する。プロバイダー固有です（`video_generate` とは別）。`video_url` は、以前のImagineの結果の公開HTTPS MP4 URLでなければなりません。 | xAI Imagineの認証情報（SuperGrok OAuthまたは `XAI_API_KEY`） |
| `xai_video_extend` | xAI Imagineで既存の動画を延長する。プロバイダー固有です（`video_generate` とは別）。`video_url` は、以前のImagineの結果の公開HTTPS MP4 URLでなければなりません。 | xAI Imagineの認証情報（SuperGrok OAuthまたは `XAI_API_KEY`） |

## `web` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `web_search` | ウェブで情報を検索する。デフォルトでは、タイトル、URL、説明付きで最大5件の結果を返します。任意の `limit`（1〜100、デフォルト5）を受け付けます。クエリは設定されたバックエンドにそのまま渡されるため、`site:domain`、`filetype:pdf`、`intitle:word`、`-term`、`"exact phrase"` などの演算子は、バックエンドが対応していれば機能する場合があります。 | EXA_API_KEY または PARALLEL_API_KEY または FIRECRAWL_API_KEY または TAVILY_API_KEY または PERPLEXITY_API_KEY または KEENABLE_API_KEY |
| `web_extract` | ウェブページのURLから内容を抽出する。クリーンなページ内容をmarkdown/テキストで返します（LLMによる要約なし — 高速）。PDFのURL（arxivの論文、ドキュメント）でも動作します — PDFのリンクを直接渡します。文字数の予算（デフォルト15000）内のページはそのまま返され、より大きなページは先頭+末尾のウィンドウと、ディスクに保存された全文を指すフッターとともに返されます。1回の呼び出しで最大5つのURL。 | EXA_API_KEY または PARALLEL_API_KEY または FIRECRAWL_API_KEY または TAVILY_API_KEY または PERPLEXITY_API_KEY または KEENABLE_API_KEY |

## `x_search` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `x_search` | xAIの組み込みの `x_search` Responsesツールを使って、X（Twitter）の投稿、プロフィール、スレッドを検索する。公開されたX上の現在の議論、反応、主張を見つけるための読み取り専用の公開Xディスカバリーです（一般的なウェブページ向けではありません）。投稿、返信、いいね、DM、メディアのアップロード、削除、認証済みXアカウントの調査は行いません — それらには別の認証済みX APIサーフェス（例: `xurl` スキル）が必要です。デフォルトではオフ — `hermes tools` → 🐦 X (Twitter) Search でオプトインします。スキーマはxAIの認証情報が設定されている場合にのみ登録されます（check_fnでゲート）。 | XAI_API_KEY **または** xAI Grok OAuth（SuperGrok / Premium+）のログイン |

## `tts` ツールセット

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `text_to_speech` | テキストを音声に変換する。プラットフォームが音声メッセージとして配信する MEDIA: パスを返します。Telegramでは音声バブルとして、Discord/WhatsAppでは音声添付ファイルとして再生されます。CLIモードでは ~/voice-memos/ に保存します。音声とプロバイダー… | — |

## `discord` ツールセット

`hermes-discord` プラットフォームツールセット（ゲートウェイのみ）に登録されます。メッセージングアダプターと同じボットトークンを使います。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `discord` | Discordサーバーを読み、参加する。アクションには `search_members`、`fetch_messages`、`send_message`、`react`、`fetch_channel`、`list_channels` などがあります。 | `DISCORD_BOT_TOKEN` |

## `discord_admin` ツールセット

`hermes-discord` プラットフォームツールセットに登録されます。モデレーションのアクションには、ボットが対応するDiscordの権限を持っている必要があります。

| ツール | 説明 | 必要な環境 |
|------|-------------|----------------------|
| `discord_admin` | REST API経由でDiscordサーバーを管理する: ギルド/チャンネル/ロールの一覧表示、チャンネルの作成/編集/削除、ロールの付与、タイムアウト、キック、BANの管理。 | `DISCORD_BOT_TOKEN` + ボットの権限 |
