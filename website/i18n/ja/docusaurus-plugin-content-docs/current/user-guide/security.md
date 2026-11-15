---
sidebar_position: 8
title: "セキュリティ"
description: "セキュリティモデル、危険なコマンドの承認、ユーザー認可、コンテナ分離、本番デプロイのベストプラクティス"
---

# セキュリティ

Hermes Agentは、多層防御（defense-in-depth）のセキュリティモデルで設計されています。このページでは、コマンド承認からコンテナ分離、メッセージングプラットフォームでのユーザー認可まで、あらゆるセキュリティ境界を解説します。

## 概要

セキュリティモデルは8つのレイヤーで構成されています:

1. **ユーザー認可** — 誰がエージェントと会話できるか（許可リスト、DMペアリング）
2. **危険なコマンドの承認** — 破壊的な操作に対する人間の介在（human-in-the-loop）
3. **ファイル書き込みの安全性** — `write_file`/`patch` に対する拒否リストとオプションの書き込みサンドボックス
4. **コンテナ分離** — 強化された設定によるDocker/Modal（およびSingularityプラグイン）のサンドボックス化
5. **MCP認証情報のフィルタリング** — MCPサブプロセスの環境変数の分離
6. **コンテキストファイルのスキャン** — プロジェクトファイル内のプロンプトインジェクション検出
7. **セッション間の分離** — セッションは互いのデータや状態にアクセスできません。cronジョブのストレージパスはパストラバーサル攻撃に対して強化されています
8. **入力のサニタイズ** — ターミナルツールのバックエンドにおける作業ディレクトリパラメータは、シェルインジェクションを防ぐため許可リストに対して検証されます

## 危険なコマンドの承認 {#dangerous-command-approval}

コマンドを実行する前に、Hermesはそれを厳選された危険なパターンのリストと照合します。一致が見つかった場合、ユーザーは明示的にそれを承認する必要があります。

### 承認モード

承認システムは、`~/.hermes/config.yaml` の `approvals.mode` で設定する3つのモードをサポートします:

```yaml
approvals:
  mode: smart                     # smart | manual | off
  timeout: 300                    # seconds to wait for user response (default: 300)
  cron_mode: deny                 # deny | approve — what cron jobs do when they hit a dangerous command
  single_query_mode: deny         # deny | approve — what single-query (-q) sessions do on a dangerous command
  unattended_mode: deny           # deny | approve — what webhook/API sessions do on a dangerous command
  mcp_reload_confirm: true        # /reload-mcp asks before invalidating the MCP tool cache
  destructive_slash_confirm: true # /clear, /new, /reset, /undo prompt before discarding state
```

すべてのキー:

| Key | Default | What it controls |
|---|---|---|
| `mode` | `smart` | 危険なシェルコマンドに対する承認ポリシー — 下の表を参照してください。 |
| `timeout` | `300` | Hermesが承認の返信を待ってからタイムアウトするまでの秒数。 |
| `discord_prompt_timeout` | `300` | Discordのインタラクティブボタン（承認、clarify、update、スラッシュコマンドのプロンプト）がクリック可能な状態を保つ秒数。最小30。[Discord](./messaging/discord.md) を参照してください。 |
| `cron_mode` | `deny` | [cronジョブ](./features/cron.md) が危険なコマンドのプロンプトをトリガーしたときのヘッドレスでの挙動。`deny` はコマンドをブロックし（エージェントは別の方法を見つける必要があります）、`approve` はcronコンテキストですべてを自動承認します。 |
| `single_query_mode` | `deny` | 単発の [`hermes chat -q`](./cli.md) セッションが危険なコマンドのプロンプトをトリガーしたときの挙動。`-q` セッションは1ターンだけ実行して終了し、プロンプトに答えるユーザーは待機していません。`deny` はコマンドをブロックし（エージェントは別の方法を見つける必要があります）、`approve` は単一クエリのコンテキストですべてを自動承認します。`cron_mode` と同様です。 |
| `unattended_mode` | `deny` | 無人のプログラム的プラットフォーム（webhook、msgraph_webhook、api_server）上のセッションが危険なコマンドのプロンプトをトリガーしたときの挙動。これらのサーフェスには `/approve` に答えられる人間がいないため、承認タイムアウトの間ずっとブロックする代わりに、`deny` は即座にコマンドをブロックし（エージェントは別の方法を見つける必要があります）、`approve` は無人コンテキストですべてを自動承認します。例外: クライアントがカードに応答できるapi_serverセッション（`/v1/runs` とストリーミングのchat completions、`POST /v1/runs/{id}/approval` 経由）は引き続き承認リクエストを受け取ります。`cron_mode` と同様です。 |
| `approvers` | `["*"]` | メッセージングのチャットで保留中の承認に誰が応答できるか。`["*"]`: 参加を認められた任意のメンバー。`[]` またはユーザーIDのリスト: リクエスト者、それらのID、プラットフォームの管理者のみ — [共有チャットで誰が承認に応答できるか](#who-can-answer-an-approval-in-a-shared-chat-approvalsapprovers) を参照してください。 |
| `mcp_reload_confirm` | `true` | trueの場合、`/reload-mcp` はMCPツールセットを再構築する前に確認します。再構築はプロバイダーのプロンプトキャッシュを無効化するため（ツールスキーマはシステムプロンプト内にあります）、次のメッセージでは入力トークン全体が再送信されます。**Always Approve** をクリックしたユーザーは、このキーが `false` に切り替わります。 |
| `destructive_slash_confirm` | `true` | trueの場合、破壊的なセッションのスラッシュコマンド（`/clear`、`/new`、`/reset`、`/undo`）は会話の状態を破棄する前に確認します。3択のダイアログ（Approve Once / Always Approve / Cancel）は、Telegram、Discord、Slackではネイティブのyes/noボタンで、それ以外ではテキストのフォールバックで表示されます。**Always Approve** をクリックしたユーザーは、このキーが `false` に切り替わります。TUIも `/clear`、`/new`、`/reset` のモーダルでこの設定に従います。`HERMES_TUI_NO_CONFIRM=1` は設定値に関係なくそのモーダルを強制的にスキップします。 |

| Mode | Behavior |
|------|----------|
| **smart**（デフォルト） | 補助LLMを使ってリスクを評価します。低リスクのコマンド（例: `python -c "print('hello')"`）はそのコマンドに限り自動承認されます。本当に危険なコマンドは自動拒否されます。不確実なケースは手動プロンプトにエスカレーションされます。 |
| **manual** | 危険なコマンドに対して常にユーザーに承認を求めます。 |
| **off** | すべての承認チェックを無効化 — `--yolo` での実行と同等です。すべてのコマンドがプロンプトなしで実行され、`trust: untrusted` のMCPサーバーに対する書き込み呼び出しも信頼プロンプトなしで実行されます。MCPサーバーが宣言した支出上限は引き続き確認されます（[YOLOモード](#yolo-mode) を参照）。 |

:::warning
`approvals.mode: off` を設定すると、すべての安全プロンプトが無効になります。信頼できる環境（CI/CD、コンテナなど）でのみ使用してください。
:::

### YOLOモード {#yolo-mode}

YOLOモードは、現在のセッションで**すべての**危険なコマンド承認プロンプトをバイパスします。次の3つの方法で有効化できます:

1. **CLIフラグ**: `hermes --yolo` または `hermes chat --yolo` でセッションを開始
2. **スラッシュコマンド**: セッション中に `/yolo` と入力してオン/オフを切り替え
3. **環境変数**: `HERMES_YOLO_MODE=1` を設定

`/yolo` コマンドは**トグル**です — 使用するたびにモードのオン/オフが切り替わります:

```
> /yolo
  ⚡ YOLO mode ON — all commands auto-approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

YOLOモードはCLIとゲートウェイの両方のセッションで利用できます。`/yolo` は入力されたセッションにのみ適用されます。この設定はセッションとともに保存されるため、再起動後も維持されます: メッセージングのチャットではゲートウェイの再起動をまたいで維持され、TUIとDesktopアプリではセッションを再度開いたり再開したりしても維持されます。メッセージングプラットフォームでは、`/new` と `/resume` は承認を有効に戻した状態で次の会話を開始します。

YOLOが有効な間、承認プロンプトがバイパスされていることを忘れにくくするため、Hermesは2つの常時表示の視覚的リマインダーを表示します:

- YOLOが既に有効な場合、セッション開始時に赤いバナー行 `⚠ YOLO mode — all approval prompts bypassed` を表示します。YOLOがオフのときは非表示になり、デフォルトのバナーがすっきりした状態に保たれます。
- ステータスバーのすべての幅の段階で `⚠ YOLO` の断片を表示し、YOLOのオン/オフを切り替えるとライブで更新されます（リッチテキストのレンダラーとプレーンテキストのフォールバックの両方）。

:::danger
YOLOモードは、セッションの**すべての**危険なコマンド安全チェックを無効化します — ただしハードラインのブロックリスト（後述）は**例外**です。生成されるコマンドを完全に信頼できる場合（例: 使い捨て環境での十分にテストされた自動化スクリプト）にのみ使用してください。
:::

YOLOと `approvals.mode: off` は、[`trust: untrusted`](../reference/mcp-config-reference.md) のMCPプロンプトもスキップします: 信頼されていないサーバー上の書き込み可能なツールは、各ソースの通常の到達範囲で、確認なしに実行されます。`/yolo` は入力されたチャットのみを対象とし（別のチャットでは引き続き確認されます）、`--yolo` と `HERMES_YOLO_MODE` は指定されたHermesプロセスのセッションを対象とし、`approvals.mode: off` はそのプロファイルのすべてのチャットとcronジョブを対象とします。次のものはスキップ**されません**:

- **MCPサーバーの支出上限。** チャットを宣言済みの [`spend_limit`](./features/mcp.md#tool-costs-and-spend-limits) を超えさせる呼び出しは引き続き確認され、誰も応答できない場所（cronジョブ、`hermes chat -q`）では拒否されます。この上限はあなたが設定した予算であり、安全プロンプトではありません。
- **MCPサーバーからの質問**（[elicitation](./features/mcp.md#mcp-elicitation-support)）。これらは許可ではなく入力を求めるものです。
- **[認証情報ボールト](/docs/user-guide/features/credential-vault#paying-and-filling-addresses) からの支払いカード入力プロンプト**（ページにカードを渡すもの）と、MCPのイベントサブスクリプション（常設のトリガーをインストールするもの）。

破壊的なセッションのスラッシュコマンド（`/clear`、`/new` / `/reset`、`/undo`、`/quit --delete` — `/exit --delete` はエイリアス）についても、CLIは実行前に確認を求めます。[スラッシュコマンド — 破壊的なコマンドの確認プロンプト](../reference/slash-commands.md#confirmation-prompts-for-destructive-commands) を参照してください。

### 監視下ゲートウェイのライフサイクル制限

ターミナルツールには、ゲートウェイ自身の監視下プロセス内からゲートウェイを停止または再起動することに対する、上書きできない別のガードがあります。自己再起動はツールが完了する前にそれを終了させ、スーパーバイザー/自動再開のループを引き起こす可能性があります。ユーザーの承認、YOLOモード、`force=True` のいずれもこのガードをバイパスしません。

このガードは、ゲートウェイが実行されているインタープリターのイメージを狙ったプロセス終了コマンド — `taskkill /F /IM python.exe`、`taskkill /FI "IMAGENAME eq python.exe"`、`Stop-Process -Name python`、`pkill -9 python3`、`killall python`、`pkill -f python`、および `pgrep python | xargs kill` のような名前から導出したkill — も拒否します。監視下のゲートウェイは文字どおり `python` プロセスであり、そのようなコマンドはゲートウェイ（およびエージェント自身のターン）を停止させてしまうためです。エージェントが所有するプロセスに限定したkillは通過します: バックグラウンドジョブの `proc_*` ID（`process(action="kill", …)`）や明示的なPID（`taskkill /F /PID <pid>`、`kill <pid>`）です。他のイメージ名（`taskkill /F /IM notepad.exe`）は影響を受けません。このガードは、生成されるすべてのランチャー — systemdユニット、launchd plist、s6のrunスクリプト、Windowsのスケジュールされたタスク — の下で、それらがエクスポートする `HERMES_SUPERVISED_CHILD` マーカーを通じて有効になります。

macOSでは、実行された `launchctl submit` と `launchctl bootstrap` コマンドは**ジョブのラベルに関係なく**制限されます。これは中立的なラベルを持つ間接的な再起動ヘルパーを捕捉することを目的とした保守的な登録制限であり、対象のplistを検査するものではありません。また、`RunAtLoad=false` で `KeepAlive` キーを持たない独立したスケジュール済みジョブも拒否します。拒否されたことは、そのジョブがKeepAliveを使用している、あるいはHermesを制御していることを意味する**わけではありません**。

正当なLaunchAgentのメンテナンスには、実行中のゲートウェイの外にある別のシェルを使用してください。一部の独立した `load`/`unload` コマンドは現在ラベルベースのチェックを通過しますが、これは対象を検証した上での例外ではなく、`bootstrap` の拒否を回避するためのサポートされた方法でもありません。読み取り専用の `launchctl print` はライフサイクル操作ではありません。外部でのメンテナンス後は、ディスク上のplistと読み込まれたジョブを区別してください: アクティベーションを報告する前に、plistを検証し、読み込まれたスケジュールを読み返してください。

ツールによる拒否は、そのツール呼び出しを通じてコマンドが実行されなかったことを意味します。アシスタントが呼び出しの発行を断ることは別のモデルの判断であり、モデルを変更してもターミナルガードのポリシーは変わりません。

### ハードラインのブロックリスト（常時有効の最低ライン）

一部のコマンドは、取り返しのつかないファイルシステムの消去、フォーク爆弾、ブロックデバイスへの直接書き込みなど、あまりに壊滅的であるため、Hermesは次の条件に**関係なく**実行を拒否します:

- `--yolo` / `/yolo` がオン
- `approvals.mode: off`
- cronジョブがヘッドレスの `approve` モードで実行中
- ユーザーが明示的に「常に許可」をクリック

ブロックリストは `--yolo` の下にある最低ラインです。承認レイヤーがコマンドを目にする**前に**作動し、上書きフラグはありません。現在カバーされているパターン（網羅的ではなく、`tools/approval.py::UNRECOVERABLE_BLOCKLIST` と同期されています）:

| Pattern | Why it's hardline |
|---|---|
| `rm -rf /` および明白な変種 | ファイルシステムのルートを消去 |
| `rm -rf --no-preserve-root /` | 「はい、ルートのつもりです」という明示的な変種 |
| `:(){ :\|:& };:`（bashフォーク爆弾） | 再起動するまでホストを占有 |
| マウントされたルートデバイスへの `mkfs.*` | 稼働中のシステムをフォーマット |
| `dd if=/dev/zero of=/dev/sd*` | 物理ディスクをゼロ埋め |
| rootfsトップレベルで信頼できないURLを `sh` にパイプ | リモートコード実行の攻撃ベクトルであり、承認するには範囲が広すぎる |

ブロックリストに該当した場合、ツール呼び出しはエージェントに説明的なエラーを返し、何も実行されません。正当なワークフローでこれらのコマンドのいずれかが必要な場合（例: あなたがワイプ＆再インストールのパイプラインのオペレーターである場合）は、エージェントの外で実行してください。

この最低ラインは、シェルのクォートを解析できないコマンド（`grep 'unterminated`）に対してもフェイルクローズし、エラーには `malformed executable payload` と表示されます。クォートは書かれたとおりのコマンドに基づいて判断されるため、クォートされたパターン内のシェルとして有効なエスケープ（`grep -o "[^\"]*" file`）は不正とはみなされず、区切り文字の前のエスケープされたクォート（`echo "a\"b"; reboot`）によって後続のコマンドが隠されることもありません。

### ユーザー定義の拒否ルール（`approvals.deny`） {#user-defined-deny-rules-approvalsdeny}

ハードラインのブロックリストは固定されており、コードとともに出荷されます。`approvals.deny` はそのユーザー編集可能な対応物です: 一致するターミナルコマンドを無条件にブロックするglobパターンのリストで、`--yolo`、`/yolo`、`approvals.mode: off` が参照される**前に**適用されます。「エージェントにすべてを任せるが、これらの特定のことだけは絶対にさせない」という、例外付きのyolo運用に使えます。

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"
```

詳細:

- パターンは [fnmatch](https://docs.python.org/3/library/fnmatch.html) のglob（`*`、`?`、`[...]`）で、コマンドのテキスト全体と個々の実行可能コマンドの候補に対して**大文字小文字を区別せず**に照合されます。`git push --force*` は `git push --force origin main` に一致しますが、`git push origin main` には一致しません。
- 照合は、危険パターン検出器が使用するのと同じ正規化/難読化解除されたコマンドの変種に対して行われるため、単純なクォートのトリック（`git pu""sh --force`）でルールをすり抜けることはできません。
- 実行可能コマンドの候補はリテラルのパスを保持し、そのベース名にも一致します: `sudo *` は `/usr/bin/sudo -n id` と `./sudo -n id` をカバーします。`/usr/bin/sudo *` のようなパス固有のルールは、`sudo` という名前のすべてのバイナリに対するルールにはなり**ません**。
- クォートを考慮した解析により、代入、先頭のリダイレクト、`;`、`&&`、`||`、パイプライン、グループ、コマンド置換、通常の `if`/`then`/`else`/`do` の遷移の後にあるコマンドも取り出されます。サポートされるランチャーには `sudo`、`env`、`command`、`exec`、`nohup`、`setsid`、`time`、`nice`、`timeout`、`stdbuf`、`ionice`、`chrt`、`taskset`、`chroot` が含まれます。既知のオプションのオペランドはスキップされ、`command -v`/`-V` による検索は実行とはみなされません。シェルの `-c` ペイロードは再帰的に検査されます。`env -S` / `--split-string` 内のリテラルの実行ファイルと引数の文字列はGNUのクォートとエスケープ（`\_` の単語境界と `\c` による終了を含む）を使用し、残りのコマンド引数が後ろに追加されます。それらの引数内のシェルの句読点は、実際のシェルの `-c` が消費しない限りデータのままです。`env -a` / `--argv0` の値は引数であり、実行ファイル名ではありません。シェルとGNUのsplit-stringのコメントは、実行可能コマンドの候補を生み出しません。
- 追加の実行可能コマンド候補では、単語**間**の空白は詰められますが、クォートされた引数の内容と引数のパスは保持されます。そのため、`git status` のような完全一致のルールは `env git\tstatus; echo done`（`\t` はタブを表します）にも一致します。`echo 'sudo -n id'` のようなクォート内の言及はコマンドに昇格されません。`*sudo*` のような既存の入力全体に対するglobは、引き続き意図的にどこにある言及にも一致します。
- **YAMLのクォート:** パターンは常にクォートしてください。先頭の裸の `*` はYAMLのエイリアスであり、解析に失敗します。`{`、`!`、`: ` にはそれぞれ独自のYAMLの意味があります。シェル風の内容にはシングルクォートが最も安全です。
- ユーザー定義の拒否ルールは、分離されたコンテナを含むすべてのターミナルバックエンドに、バックエンド固有の承認ショートカットより前に適用されます。
- 拒否されたコマンドは、再試行や言い換えをしないよう指示するBLOCKEDエラーをエージェントに返します。何も実行されません。

承認設定の他の部分と同様に、変更は即座に反映されます（設定キャッシュはmtimeをキーにしています） — セッションの再起動は不要です。

:::note 脅威モデル
拒否ルールはシェルコマンドのポリシーであり、完全なシェルインタープリターでもOSのケイパビリティサンドボックスでもありません。正規化は、任意の変数（GNU `env -S` の `${NAME}` 展開を含む）、エイリアス、関数、名前を変更したバイナリ、スクリプト、インタープリターのプログラム、あらゆるシェル/ランチャーの文法（例えばcaseパターンの構文、まとめて指定されたランチャーのオプション、`env -S` 文字列に埋め込まれたオプション）を解決しません。ベース名による拒否ルールを、ある機能に他の手段で到達できないことの保証として使わないでください。封じ込めには、OSの権限と、マウント、認証情報、ネットワークアクセスを適切に制限した分離バックエンドを使用してください。この照合の挙動は、設定された承認モードや空の拒否リストというデフォルトを変更しません。
:::

### 承認タイムアウト

危険なコマンドのプロンプトが表示されると、ユーザーは設定可能な時間内に応答する必要があります。タイムアウト内に応答がない場合、コマンドはデフォルトで**拒否**されます（フェイルクローズ）。

期限切れのプロンプトを再び開くことはできません: 保留中のエントリは破棄され、エージェントにはそのターン内で自ら再試行しないよう伝えられます。それでも操作を実行したい場合は、それを求める新しいメッセージを送ってください（例えば「では今それを実行して」） — エージェントは新しいツール呼び出しを発行し、それによって新しい承認カードが表示されます。「once」の承認はその呼び出しにのみ適用されます。タイムアウトは拒否としてカウントされないため、再度依頼しても不利になることはありません。

`~/.hermes/config.yaml` でタイムアウトを設定します:

```yaml
approvals:
  timeout: 300  # seconds (default: 300)
```

### 承認をトリガーするもの

次のパターンが承認プロンプトをトリガーします（`tools/approval.py` で定義）:

| Pattern | Description |
|---------|-------------|
| `rm -r` / `rm --recursive` | 再帰的な削除 |
| `rm ... /` | ルートパスでの削除 |
| `chmod 777/666` / `o+w` / `a+w` | 全員/他者が書き込み可能な権限 |
| 安全でない権限での `chmod --recursive` | 再帰的に全員/他者が書き込み可能（ロングフラグ） |
| `chown -R root` / `chown --recursive root` | rootへの再帰的なchown |
| `mkfs` | ファイルシステムのフォーマット |
| `dd if=` | ディスクコピー |
| `> /dev/sd` | ブロックデバイスへの書き込み |
| `DROP TABLE/DATABASE` | SQLのDROP |
| `DELETE FROM`（WHEREなし） | WHEREのないSQLのDELETE |
| `TRUNCATE TABLE` | SQLのTRUNCATE |
| `> /etc/` | システム設定の上書き |
| `systemctl stop/restart/disable/mask` | システムサービスの停止/再起動/無効化 |
| `kill -9 -1` | すべてのプロセスの強制終了 |
| `pkill -9` | プロセスの強制終了 |
| フォーク爆弾のパターン | フォーク爆弾 |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | `-c` フラグによるシェルコマンド実行（`-lc` のような結合フラグを含む） |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | `-e`/`-c` フラグによるスクリプト実行 |
| `curl ... \| sh` / `wget ... \| sh` | リモートコンテンツをシェルにパイプ |
| `bash <(curl ...)` / `sh <(wget ...)` | プロセス置換によるリモートスクリプトの実行 |
| `/etc/`、`~/.ssh/`、`~/.hermes/.env` への `tee` | teeによる機密ファイルの上書き |
| `/etc/`、`~/.ssh/`、`~/.hermes/.env` への `>` / `>>` | リダイレクトによる機密ファイルの上書き |
| `xargs rm` | rmを伴うxargs |
| `find -exec rm` / `find -delete` | 破壊的なアクションを伴うfind |
| `/etc/` への `cp`/`mv`/`install` | システム設定へのファイルのコピー/移動 |
| `/etc/` に対する `sed -i` / `sed --in-place` | システム設定のインプレース編集 |
| hermes/gateway に対する `pkill`/`killall` | 自己終了の防止 |
| `&`/`disown`/`nohup`/`setsid` を伴う `gateway run` | サービスマネージャーの外でのゲートウェイ起動を防止 |
| `docker stop/kill/restart`、`docker compose down/stop/kill/restart` | コンテナのライフサイクル（グローバルフラグや `docker-compose` も捕捉） |
| `docker -H`/`--host`/`--context`、`DOCKER_HOST=`/`DOCKER_CONTEXT=` | Dockerデーモンのリダイレクト — コマンドが別の（多くの場合リモートの）デーモンを対象にする |
| `docker context use` | 以降のすべてのdockerコマンドのデフォルトデーモンを切り替える |
| `podman --remote`/`-r`/`--url`/`--connection`/`--identity`、`CONTAINER_HOST=` | Podmanのリモートデーモンのリダイレクト |

:::info
**コンテナでのバイパス**: `docker`、`modal`、`daytona`、`vercel_sandbox` バックエンド（または `singularity` や `sprites` のようなコンテナプラグインのバックエンド）で実行している場合、コンテナ自体がセキュリティ境界となるため、危険なコマンドのチェックは**スキップ**されます。コンテナ内の破壊的なコマンドはホストに害を与えられません。
:::

### `execute_code` スクリプト {#execute-code-scripts}

[`execute_code`](./features/code-execution.md) で実行されるスクリプトは `subprocess`、`os`、`shutil` を直接呼び出すことができ、それらの呼び出しは上記のパターンを一切通りません。そのため、メッセージング、API、その他のゲートウェイのセッションでは、スクリプトの開始前にスクリプト全体について一度だけ確認します。拒否された、または応答のないスクリプトは決して実行されません。`cron_mode`、`single_query_mode`、`unattended_mode` は、コマンドの場合と同様にスクリプトについても即座に判断し、YOLOと `mode: off` はプロンプトをスキップします。

ローカルセッション（クラシックCLI、TUI、Desktop、ACPエディター）は、代わりに対話型CLIのルールに従います。`manual`（および `smart`）の下では、セッションの閲覧者が入力したターンは**スクリプト全体の承認を受けません**: スクリプト内のPythonレベルのファイル操作やプロセス操作はプロンプトなしで実行され、スクリプトの `terminal()` 呼び出しだけが、one-gatewayランタイム以前のCLIと同様に、それぞれセッション自身のプロンプトで確認されます。この免除には、閲覧者が送信したターンであることと、スクリプト開始時に承認に応答できる閲覧者が接続していることの両方が必要です。次の場合は、ローカルセッションでもスクリプト全体の承認が維持されます:

- そのセッションで誰も入力していないターン: プラグインの `inject_message` ターン、委任の完了、バックグラウンドプロセスやwatchパターンの通知、`/loop` のティック、ハートビート、ホストされたルームのタスク
- `delegate_task` の子と、`/bg` / `/btw` のサイドタスク
- 承認できる閲覧者が接続していない間に実行される閲覧者のターン

### 承認フロー（CLI）

対話型CLIでは、危険なコマンドはインラインの承認プロンプトを表示します:

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf ~/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

4つの選択肢:

- **once** — この1回の実行のみを許可
- **session** — このパターンをセッションの残りの間許可
- **always** — 永続的な許可リストに追加（`config.yaml` に保存）
- **deny**（デフォルト） — コマンドをブロック

### 承認フロー（ゲートウェイ/メッセージング）

メッセージングプラットフォームでは、エージェントが危険なコマンドの詳細をチャットに送信し、ユーザーの返信を待ちます:

- 承認するには **yes**、**y**、**approve**、**ok**、または **go** と返信
- 拒否するには **no**、**n**、**deny**、または **cancel** と返信

メッセージングでの承認が、全体を読めないコマンドの承認を求めることはありません。コマンドが承認カードに収まらないほど長い場合、Hermesはまずコマンド全体を独立したメッセージとして送信し（必要に応じて複数のメッセージに分割）、カードにはプレビューが短縮されたことが表示されます。コマンド全体を表示できない場合（送信が失敗した場合、またはコマンド自体に3連バッククォートのフェンスが含まれる場合）、カードは **Deny** のみを提示します。そのようなコマンドはCLIまたはデスクトップアプリで確認してください。長いメッセージを分割せずに切り詰めるプラットフォーム（ntfy、DingTalk、Photon、LINE）では、プレーンテキストのプロンプトはコマンドの前に回答方法を記載します。それでもコマンドが切り詰められる場合、プロンプトは `/deny` のみを提示し、同じ「プレビュー短縮」の通知を表示します。

ゲートウェイを実行すると、`HERMES_EXEC_ASK=1` 環境変数が自動的に設定されます。

#### 共有チャットで誰が承認に応答できるか（`approvals.approvers`） {#who-can-answer-an-approval-in-a-shared-chat-approvalsapprovers}

デフォルトでは、誰のメッセージによって発生したかにかかわらず、チャットへの参加を認められた任意のメンバーが保留中の承認に応答できます。グループ、チャンネル、共有スレッドでは、代わりに各承認を依頼した本人に紐付けることができます:

```yaml
approvals:
  approvers: []            # default ["*"]: any admitted member answers
  # approvers: ["U024BE7LH", "123456789"]   # requester + these chat user ids + admins
```

`approvers` を `["*"]` から絞り込むと、次の人だけがコマンドを承認または拒否できます:

- 承認を発生させたターンを開始したメッセージのユーザー
- `approvers` に列挙されたユーザーID（各プラットフォーム独自のID形式を使用: Slackの `U…`、Telegramの数値ID、Matrixの `@user:server`）
- 承認が発生したチャットのプラットフォーム管理者: DMでは `allow_admin_from`、グループでは `group_allow_admin_from`（スラッシュコマンドと同じスコープ。DMの管理者はグループの管理者ではありません）。チャットを受信したボットのプロファイルの設定が使われます。

2人の異なるリクエスト者によって同時に発生した同一のコマンドには、それぞれ1つずつプロンプトが表示されるため、一方のリクエスト者の応答がもう一方の承認を決めることはありません。2人のメンバーからのメッセージを含むターン（エージェントが処理中に統合されたフォローアップ）には単一のリクエスト者がいないため、その承認は列挙された承認者と管理者にのみ向けられます。

他の誰かがボタンをクリックしたり、リアクションを追加したり、`/approve`、`/deny`、または単なる「yes」を送信したりすると、*Only &lt;name&gt; can answer this approval.* と表示されます。コマンドは保留のままなので、リクエスト者は引き続き応答できます。プラットフォームが非公開の返信をサポートしている場合、その通知はクリックした人にだけ表示されます: Slackのephemeralメッセージ、Discordのephemeral返信、Telegramのコールバックアラート、Feishuのトースト、Teamsのカード応答です。Matrix、WhatsApp、リレーでは代わりにチャットに投稿されます。

同じルールは、メンバーが他の人に代わって応答できる他のプロンプトにも適用されます: スラッシュコマンドの確認（`/reload-mcp` の Approve Once / Always / Cancel）と、エージェントのclarifyの質問（選択ボタン、「Other」、または入力による回答）です。別のメンバーのタップには同じ通知が表示されてプロンプトは保留のままとなり、別のメンバーが入力したメッセージはその人自身のメッセージとして扱われ、決して回答としては扱われません。

**チャットのスコープにスラッシュの管理者制限（`allow_admin_from` / `group_allow_admin_from`）が設定されている場合**、デフォルトの `approvers: ["*"]` は「参加を認められた任意のメンバー」を意味しなくなります: 承認への応答は `/approve` または `/deny` であり、スラッシュコマンドの確認への応答はそのコマンドを実行することなので、すべての経路（ボタン、リアクション、入力された `/approve`、単なる `yes` / `always`、入力による確認の返信）で、そのチャットでそのコマンドの実行を許可された人だけが応答できます。つまり、スコープの管理者と、`user_allowed_commands` に `approve`（この場合は拒否も可能）、`deny`（拒否のみ）、または確認対象のコマンドが含まれるメンバーです。それ以外の人は、承認を発生させたメッセージのメンバーを含め、どちらの応答もできません: *⛔ /approve is admin-only here* と表示され、プロンプトは管理者のために保留のままとなるか、タイムアウト時に拒否されます。

両方が設定されている場合、絞り込まれた `approvers` が優先されます: これは「誰が承認するか」に対する明示的な答えであるため、`user_allowed_commands` の内容にかかわらず、リクエスト者、列挙されたID、スコープの管理者は、上記のとおりボタン、リアクション、または単なる「yes」で応答できます。`/approve` や `/deny` の入力はそれ自体がスラッシュコマンドなので、引き続きそのコマンドの権限が必要です。

| Config for the chat's scope | Who can answer |
|---|---|
| どちらもなし | 参加を認められた任意のメンバー |
| `allow_admin_from` のみ | 管理者、および `user_allowed_commands` に `approve`（承認または拒否）/ `deny`（拒否のみ）または確認対象のコマンドを持つメンバー |
| `approvers` を絞り込み（`allow_admin_from` の有無を問わない） | リクエスト者、列挙されたID、スコープの管理者 |

ボットの所有者とのDMは、所有者がDMの管理者リストに含まれていれば引き続き機能します。clarifyの質問はコマンドではないため、`approvers` にのみ従います。

リレー経由のDiscordのインタラクションは、より厳しくチェックされます。リレーは、Discordのボタン押下やアプリケーションコマンド（`/approve` や `/deny` など）を送信者を確認せずに転送します。そのため `approvers: ["*"]` の下では、メンバーがそれらで応答できるのは、リレーがそのメンバーのメッセージのいずれかを（このゲートウェイが最後に起動してから）このゲートウェイに配信した後だけです。それ以外の人は、上記のルールに加えて、リクエスト者またはスコープの管理者のいずれかである必要があります。他の送信者には同じ通知が表示され、プロンプトは保留のままとなります。リレーがメッセージを配信したことのない送信者からのその他のDiscordのインタラクション（`/status` や `/new` のようなアプリケーションコマンド、モーダル）は、その人がスコープの管理者でない限り破棄されます。Hermesが再起動すると、メンバーのアプリケーションコマンドは、そのメンバーがメッセージを送信するまで何もしません。通常のメッセージと、リレーの他のプラットフォームでのボタン押下には、上記のルールが引き続き適用されます。

ピッカーカード（`/model`、`/reasoning`、`/fast`、および使用量上限の切り替え提案）と `/update` のYes/Noプロンプトは、`approvers` の内容にかかわらず、常にそれを開いたメンバーに属します: そのメンバーとチャットスコープの管理者だけがそれらをタップでき、`/update` の場合は入力によって応答できます。それ以外の人には同じ非公開の通知が表示され、カードは開いたままとなり、その人が入力したテキストは通常のメッセージとして扱われます。したがって、管理者専用コマンドを拒否されたメンバーが、代わりに管理者が開いているカードを適用することはできません。

この設定は承認が発生した時点で読み込まれるため、マルチプレックスされたプロファイルでは各プロファイル自身の `config.yaml` が判断します。ホストのサーフェス（CLI、TUI、Desktop、APIサーバー）はホストとして応答し、影響を受けません。Matrixは独自のより厳格な `MATRIX_APPROVAL_REQUIRE_SENDER`（デフォルトでオン）を維持します。QQBotは既にセッション自身のユーザーからのクリックのみを受け付けます。

### 永続的な許可リスト

「always」で承認されたコマンドは `~/.hermes/config.yaml` に保存されます:

```yaml
# Permanently allowed dangerous command patterns
command_allowlist:
  - rm
  - systemctl
```

これらのパターンは起動時に読み込まれ、以降のすべてのセッションで暗黙的に承認されます。

エントリには、正確なコマンドテキスト、シェル形式のglob（`podman *`）、または `script execution via heredoc` のような危険パターンのルールキー（承認プロンプトに表示されるキー）を指定できます。ルールキーは無人のものを含むすべてのサーフェスで尊重されます: `cron_mode`/`single_query_mode`/`unattended_mode: deny` の下にあるcronジョブ、`hermes chat -q` の実行、webhookセッションでも、検出されたルールキーが `command_allowlist` に含まれるコマンドは実行されます。

この設定は文字列のリストでなければなりません。リストをクォートされたYAML/JSON文字列として保存していた従来のインストールでは、読み込み時にそのリストが復元され、`hermes config edit` で保存し直すよう警告がログに出力されます。その他の不正な値は警告とともに無視され、1文字ずつの承認になることはありません。読み込みによって設定ファイルが書き換えられることはありません。

:::tip
`hermes config edit` を使って、永続的な許可リストからパターンを確認または削除してください。
:::

:::caution
リストはHermesの起動時に読み込まれます。セッションが既に実行中のときに削除したパターンは、そのセッションが次にファイルを書き込むまで（次にプロンプトに `always` と答えるまで）、またはHermesを再起動するまで、そのセッションでは承認されたままです。安全上の理由で削除した場合は、再起動してください。
:::

### 承認履歴のマイニング（`hermes approvals suggest`）

セッションのたびに同じプロンプトに答える代わりに、過去の承認の判断を許可リストの提案へとマイニングできます:

```bash
hermes approvals suggest            # dry run — prints a numbered proposal
hermes approvals suggest --apply 1,3  # merge picks into command_allowlist
hermes approvals suggest --json     # machine-readable output
```

このコマンドは、セッションデータベース（`~/.hermes/state.db`）から実際に実行された危険と分類されたコマンド — つまりあなたが承認したコマンド — をスキャンし、それらをパターン（`git push *`、または複合コマンドの場合は危険クラスのキー）に集約し、承認の頻度で順位付けします:

```
Proposed command_allowlist additions (from approval history, last 90 days):

  1. git push *    — approved 14x
  2. docker restart/stop/kill (container lifecycle)    — approved 9x (class key)
```

安全ルール:

- **何も自動的には適用されません** — デフォルトの実行は読み取り専用で、明示的な `--apply N[,M...]` だけが `config.yaml` に書き込みます。
- **破壊的なクラスは決して提案されません**。どれだけ頻繁に承認されていても同様です: 再帰的な削除、`sudo`、ディスク/デバイスへの書き込み、認証情報やシステム設定の編集、シェルへのパイプ、SQLのDROP/TRUNCATE、プロセスのkill、およびすべてのハードラインのクラスは完全に除外されます。`rm -rf build/` を100回承認しても、`rm` のエントリが生成されることはありません。
- 既存の `command_allowlist` で既にカバーされている提案はスキップされます。
- **マイニングされたコマンド内の認証情報はマスクされます**（`ghp_…`、`bot<id>:<token>` のURL、`KEY=value` の代入、bearerトークン）。表示される `e.g.` の例と `--json` のペイロードの両方で、ターミナル出力と同じ伏せ字処理が使われます。マスクされたテキストが許可リストのパターンとして使われることはありません: globに認証情報が埋め込まれてしまうコマンド（`TOKEN=… git …`）は、代わりに危険クラスのキーで提案されます。セッションデータベース自体には、実行されたとおりのコマンドが引き続き保持されます。

便利なフラグ: `--days N`（履歴の期間、デフォルト90）、`--min-count N`（対象となる最小承認回数、デフォルト2）、`--limit N`、`--db PATH`。

## ファイル書き込みの安全性 {#file-write-safety}

`write_file` または `patch` がディスクに触れる前に、Hermesは対象パスを拒否リストとオプションのサンドボックスに対してチェックします。ブロックされた書き込みは即座にエージェントにエラーを返します — **承認プロンプトはなく**、チャットUIから上書きする方法もありません。それでもモデルが編集に成功したと主張する場合があります。`display.file_mutation_verifier` がオン（デフォルト）の場合は、アシスタントの締めくくりの要約よりも [ファイル変更検証のフッター](./configuration.md#file-mutation-verifier) を信頼してください。

### 保護されたパス（常にブロック）

これらのカテゴリは、`HERMES_WRITE_SAFE_ROOT` が未設定の場合でも常に拒否されます:

| Category | Examples |
|----------|----------|
| OSの認証情報ストア | `~/.ssh/`（鍵、`authorized_keys`）、`~/.aws/`、`~/.kube/`、`/etc/sudoers`、`~/.netrc` |
| Hermesの機密情報ストア | HERMES_HOME配下（アクティブなプロファイル、`profiles/` 配下のすべてのプロファイル、グローバルルート）の `.env`、`.anthropic_oauth.json`、`auth/google_oauth.json`、Bitwardenと1Passwordのキャッシュ（`bws_cache.json`、`bws_cache.enc.json`、`op_cache.json`）、Photonサイドカーのトークン（`photon-sidecar.json`）、`vault/`、`browser-profile/`、`mcp-tokens/`、`pairing/`。制御ファイル（`auth.json`、`config.yaml`、`webhook_subscriptions.json`）は読み取りが拒否されますが、書き込みは可能なままです。 |
| WindowsのNT/デバイス名前空間のパス | `\??\...`、`\\.\...`、`\\?\UNC\...`、`\\?\GLOBALROOT...` — すべてのプラットフォームで読み取りと書き込みの両方が拒否されます。Windowsでは、そのようなパス（例: `\??\UNC\host\share`）を*解決*するだけで外向きのSMB認証がトリガーされ、ユーザーのNTLMハッシュが漏洩する可能性があります。また、これらのプレフィックスは通常のパス正規化もバイパスします。通常の拡張長のローカルパス（`\\?\C:\...`）と通常のUNC共有（`\\server\share`）は影響を受けません。 |

プロジェクトローカルの `.env`、`.env.local`、`.env.production`、`.envrc` ファイルは、ディスク上のどこにあっても**読み取りが拒否されます**（ファイルツールはそれらの読み取りを拒否します）が、書き込みは可能なままです: エージェントはそれらを作成・編集することはできますが、値を読み返すことはできません。

安全ルート内の機密パスも引き続きブロックされます — `HERMES_WRITE_SAFE_ROOT` を `$HOME` に向けても、`~/.ssh/id_rsa` への書き込みは許可されません。

OSの認証情報の行にある `~` は、プロセスの `HOME` だけでなく、書き込みが到達しうる*すべての*ホームを意味します: OSユーザーの実際のホーム、プロファイルホーム（`TERMINAL_HOME_MODE=profile`、コンテナ、生成されたワーカーの下での `{HERMES_HOME}/home`。ここではプロセスの `HOME` が固定されます）、および名前付きアカウント（`~root/.ssh/authorized_keys`）です。実際のホームの `~/.aws/credentials` への絶対パスは、エージェントのプロセスが `HOME` を別の場所に向けて実行されている場合でも拒否されます。

安全ルートの違反は `Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (…)` を返します。認証情報パスのブロックは `Write denied: '…' is a protected system/credential file.` を使います。

**例外 — `~/.ssh/config` は承認ゲート付きであり、ハードブロックではありません。** SSHの*クライアント設定*には秘密鍵の素材が含まれず、その編集（ホストのエイリアス、`ProxyJump`、VS Code Remote-SSHのターゲット）は日常的な作業であるため、`write_file` / `patch` は、以前適用されていた一律の拒否の代わりに、ターミナルツールが既に `~/.ssh` への書き込みに使用しているのと同じ approve-once/session/always のプロンプトを経由させます。それでも、コマンドを実行する `ProxyCommand` / `Match exec` ディレクティブを含みうるため、書き込みが黙って行われることはありません。非対話型の呼び出し元（ACPファイルブリッジ、人間とのチャネルがないバックグラウンドジョブ）はフェイルクローズします。秘密鍵、`authorized_keys`、およびその他の `~/.ssh/` 配下のすべては、引き続きハードブロックされます。

### HERMES_WRITE_SAFE_ROOT（オプションのサンドボックス）

設定すると、`write_file` と `patch` は列挙されたディレクトリのプレフィックス内のパスのみを対象にできます。外側のものはすべて**ハードブロック**されます — 危険なコマンドの承認を経由することはありません。

- [公式Dockerイメージ](https://github.com/NousResearch/hermes-agent) では自動的に設定されます（`HERMES_WRITE_SAFE_ROOT=/opt/data`）
- Unixでは `:`、Windowsでは `;` で区切って複数のルートをサポートします
- **安易に `~/.hermes/.env` に追加しないでください。** プロジェクトディレクトリに設定すると、エージェントはそのプレフィックスの外にある `~/.hermes/cron/jobs.json`、プロファイルのスキル、その他のHermesの状態に書き込めなくなります

ワークスペースとHermesのホームの両方を許可するには:

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

変数を未設定に戻すと、制限のない書き込みに戻ります（保護されたパスの拒否リストには従います）。完全なリファレンス: [HERMES_WRITE_SAFE_ROOT](../reference/environment-variables.md#hermes_write_safe_root)。

### cronおよびその他のHermesの状態

エージェントに `~/.hermes/cron/jobs.json` を直接 `patch` するよう依頼しないでください。`cronjob_manage` ツール、[`hermes cron`](./features/cron.md)、または `/cron` を使用してください — これらはサポートされたAPIを通じてジョブストアを更新します。書き込みの安全性によって直接の編集がブロックされる場合、他のHermesの制御ファイルにも同じことが当てはまります。

:::note 多層防御であり、厳格な境界ではありません
書き込みガードは `write_file` と `patch` にのみ適用されますが、1つ例外があります: WindowsのNT/デバイス名前空間の行は読み取りにも適用されます — `read_file`、`search_files`、`@file:`/`@folder:` のコンテキスト参照、ACPファイルブリッジはいずれも、何かがそれらを解決する前に、生の文字列の段階でそれらのパスを拒否します。`terminal` ツールは同じOSユーザーとして実行されるため、シェルコマンドを介して拒否されたパスを `cat` したり上書きしたりすることは依然として可能です。拒否リストは偶発的な損害を減らし、モデルに明確な停止シグナルを与えますが、敵対的または侵害されたエージェントをサンドボックス化するものではありません。
:::

## ユーザー認可（ゲートウェイ） {#user-authorization-gateway}

メッセージングゲートウェイを実行する際、Hermesは階層化された認可システムを通じて、誰がボットと対話できるかを制御します。

### 認可チェックの順序

`_is_user_authorized()` メソッドは次の順序でチェックします:

1. **プラットフォームごとのall-allowフラグ**（例: `DISCORD_ALLOW_ALL_USERS=true`）
2. **DMペアリングの承認済みリスト**（ペアリングコードで承認されたユーザー）
3. **プラットフォーム固有の許可リスト**（例: `TELEGRAM_ALLOWED_USERS=12345,67890`）
4. **グローバル許可リスト**（`GATEWAY_ALLOWED_USERS=12345,67890`）
5. **グローバルall-allow**（`GATEWAY_ALLOW_ALL_USERS=true`）
6. **デフォルト: 拒否**

### プラットフォームの許可リスト

許可するユーザーIDを `~/.hermes/.env` にカンマ区切りの値として設定します:

```bash
# Platform-specific allowlists
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# Cross-platform allowlist (checked for all platforms)
GATEWAY_ALLOWED_USERS=123456789

# Per-platform allow-all (use with caution)
DISCORD_ALLOW_ALL_USERS=true

# Global allow-all (use with extreme caution)
GATEWAY_ALLOW_ALL_USERS=true
```

グローバルall-allowは、`config.yaml` に `gateway.allow_all_users: true`（またはトップレベルの `allow_all_users: true`）として記述することもできます。trueの値はゲートウェイ起動時に `GATEWAY_ALLOW_ALL_USERS` へ橋渡しされ（設定の読み込みと再起動のたびに再導出されるため、`false` に戻すとゲートは閉じます）、明示的な環境変数が優先され、ゲートウェイは付与元として `config.yaml` を挙げる警告をログに出力します。マルチプロファイルのゲートウェイでは、セカンダリプロファイルは自身の `.env` で `GATEWAY_ALLOW_ALL_USERS` を設定します（その `config.yaml` がプロセス環境に橋渡しされることはありません）。

:::warning
**許可リストが何も設定されておらず**、`GATEWAY_ALLOW_ALL_USERS` も設定されていない場合、**すべてのユーザーが拒否されます**。ゲートウェイは起動時に警告をログ出力します:

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DMペアリングシステム {#dm-pairing-system}

より柔軟な認可のため、Hermesにはコードベースのペアリングシステムが含まれています。事前にユーザーIDを要求する代わりに、未知のユーザーはワンタイムのペアリングコードを受け取り、ボットの所有者がCLIで承認します。

**仕組み:**

1. 未知のユーザーがボットにDMを送信
2. ボットが8文字のペアリングコードで返信
3. ボットの所有者がCLIで `hermes pairing approve <platform> <code>` を実行
4. そのユーザーはそのプラットフォームで永続的に承認される

`~/.hermes/config.yaml` で、認可されていないダイレクトメッセージの扱い方を制御します:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` はチャット形式のDMプラットフォームでのデフォルトです。認可されていないDMはペアリングコードの返信を受け取ります。
- `ignore` は認可されていないDMを黙って破棄します。
- `decline` はペアリングコードの代わりに短く丁寧な断りの返信（「I can only chat with my owner」）を1回送信し、その後24時間はその送信者からのメッセージを無視します。テキストは `unauthorized_dm_decline_message` でカスタマイズできます。
- メールは、受信箱に無関係な未読メールが含まれている可能性があるため、`platforms.email.unauthorized_dm_behavior: pair` が設定されていない限り `ignore` がデフォルトです。
- プラットフォームセクションはグローバルなデフォルトを上書きするため、Telegramではペアリングを有効にしつつ、WhatsAppでは無音を保つことができます。

**セキュリティ機能**（OWASP + NIST SP 800-63-4 のガイダンスに基づく）:

| Feature | Details |
|---------|---------|
| コード形式 | 32文字の曖昧さのないアルファベット（0/O/1/Iなし）から8文字 |
| ランダム性 | 暗号学的（`secrets.choice()`） |
| コードのTTL | 1時間で失効 |
| レート制限 | ユーザーあたり10分に1リクエスト |
| 保留中の上限 | プラットフォームあたり最大3つの保留中コード |
| ロックアウト | 承認の失敗5回 → 1時間のロックアウト |
| ファイルのセキュリティ | すべてのペアリングデータファイルに `chmod 0600` |
| ロギング | コードがstdoutにログ出力されることはありません |

**ペアリングのCLIコマンド:**

```bash
# List pending and approved users
hermes pairing list

# Approve a pairing code
hermes pairing approve telegram ABC12DEF

# Revoke a user's access
hermes pairing revoke telegram 123456789

# Clear all pending codes
hermes pairing clear-pending
```

:::tip Dockerユーザー: ペアリングコマンドは `hermes` ユーザーとして実行してください
公式Dockerイメージは `gosu` を介して非特権の `hermes` ユーザー（uid 10000）としてゲートウェイを実行しますが、`docker exec` はデフォルトでrootになります。rootによって作成された承認ファイルはモード `0600 root:root` で書き込まれ、ゲートウェイはそれを読み取れません — 承認は黙って無視されます（[#10270][i10270]）。

常に `-u hermes` を指定してください:

```bash
docker exec -u hermes hermes-agent hermes pairing approve telegram ABC12DEF
```

既にrootとしてコマンドを実行してしまい、ユーザーがまだ認可されていない場合は、コンテナを再起動してください — エントリポイントが次回の起動時に所有権を修正します。

[i10270]: https://github.com/NousResearch/hermes-agent/issues/10270
:::

**ストレージ:** ペアリングデータは `~/.hermes/pairing/` にプラットフォームごとのJSONファイルで保存されます:
- `{platform}-pending.json` — 保留中のペアリングリクエスト
- `{platform}-approved.json` — 承認済みユーザー
- `_rate_limits.json` — レート制限とロックアウトの追跡

## コンテナ分離

`docker` ターミナルバックエンドを使用する場合、Hermesはすべてのコンテナに厳格なセキュリティ強化を適用します。

### Dockerのセキュリティフラグ

すべてのコンテナは次のフラグで実行されます（`tools/environments/docker.py` で定義）:

```python
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # Drop ALL Linux capabilities
    "--cap-add", "DAC_OVERRIDE",                  # Root can write to bind-mounted dirs
    "--cap-add", "CHOWN",                         # Package managers need file ownership
    "--cap-add", "FOWNER",                        # Package managers need file ownership
    "--security-opt", "no-new-privileges",         # Block privilege escalation
    "--pids-limit", "256",                         # Limit process count
    # no-tmp: ok — configures the sandbox's own tmpfs
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # Size-limited /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # No-exec /var/tmp
]
```

`SETUID`/`SETGID` はベースのリストには**含まれません** — コンテナがrootとして起動し、init/エントリポイントが権限を降格する必要がある場合（s6の権限降格パス）に条件付きで追加されます。コンテナが既に非rootの `--user` として実行されている場合はスキップされます。`/run` のtmpfsもベースのリストから分離され、イメージごとにマウントされます（デフォルトでは強化された `noexec`、`/run` から実行するs6-overlayイメージの場合のみ `exec`）。

### リソース制限

コンテナのリソースは `~/.hermes/config.yaml` で設定できます:

```yaml
terminal:
  backend: docker
  docker_image: "nousresearch/hermes-sandbox:desktop"
  docker_forward_env: []  # Explicit allowlist only; empty keeps secrets out of the container
  container_cpu: 1        # CPU cores
  container_memory: 5120  # MB (default 5GB)
  container_disk: 51200   # MB (default 50GB, requires overlay2 on XFS)
  container_persistent: true  # Persist filesystem across sessions
```

### ファイルシステムの永続化

- **永続モード**（`container_persistent: true`）: `~/.hermes/sandboxes/docker/<task_id>/` から `/workspace` と `/root` をバインドマウント
- **エフェメラルモード**（`container_persistent: false`）: ワークスペースにtmpfsを使用 — クリーンアップ時にすべて失われる

:::tip
本番のゲートウェイデプロイでは、`docker`、`modal`、`daytona`、`vercel_sandbox` バックエンドを使用して、エージェントのコマンドをホストシステムから分離してください。これにより、危険なコマンドの承認が完全に不要になります。
:::

:::warning
`terminal.docker_forward_env` に名前を追加すると、それらの変数はターミナルコマンドのために意図的にコンテナに注入されます。これは `GITHUB_TOKEN` のようなタスク固有の認証情報に便利ですが、コンテナ内で実行されるコードがそれらを読み取って外部に持ち出せることも意味します。
:::

## ターミナルバックエンドのセキュリティ比較

| Backend | Isolation | Dangerous Cmd Check | Best For |
|---------|-----------|-------------------|----------|
| **local** | なし — ホスト上で実行 | ✅ あり | 開発、信頼できるユーザー |
| **ssh** | リモートマシン | ✅ あり | 別サーバーでの実行 |
| **docker** | コンテナ | ❌ スキップ（コンテナが境界） | 本番ゲートウェイ |
| **singularity**（プラグイン） | コンテナ | ❌ スキップ | HPC環境 |
| **sprites**（プラグイン） | クラウドmicroVM | ❌ スキップ | ステートフルなクラウドサンドボックス |
| **modal** | クラウドサンドボックス | ❌ スキップ | スケーラブルなクラウド分離 |
| **daytona** | クラウドサンドボックス | ❌ スキップ | 永続的なクラウドワークスペース |
| **vercel_sandbox** | クラウドmicroVM | ❌ スキップ | スナップショット永続化を伴うクラウド実行 |

## 環境変数のパススルー {#environment-variable-passthrough}

`execute_code` と `terminal` はどちらも、LLMが生成したコードによる認証情報の持ち出しを防ぐため、子プロセスから機密性の高い環境変数を取り除きます。ただし、`required_environment_variables` を宣言するスキルは、正当にそれらの変数へのアクセスを必要とします。

ファーストパーティのプラットフォーム認証情報 — Buzzメッセージングプラットフォームが使用する `BUZZ_*` 変数 — は、**セッションが実際にBuzzエージェントとして動作している場合に限り**、`terminal` の子プロセス（フォアグラウンドとバックグラウンド/PTYの両方の生成）にパススルーされます: プロセスがBuzz-ACPの管理対象エージェント（Buzz Desktopハーネスによって `BUZZ_MANAGED_AGENT` が設定される）であるか、稼働中のゲートウェイセッションのプラットフォームが `buzz` である場合です。これにより、Buzzプラットフォームのエージェントはターミナルツールからプラットフォームが必須とするCLI（例: `buzz`）を呼び出せる一方、同じホスト上のTelegram/CLI/cronセッションでは変数が取り除かれたままになります。`_sanitize_subprocess_env` は検索ワーカー（例: ddgsのWeb検索サブプロセス）、computer-useのドライバーバイナリ、ユーザースクリプトのランナー（bang `!` コマンド、クイックコマンド、cronスクリプト、webhookフィルタースクリプト）にも使われるため、それらの子プロセスもBuzzセッションから生成された場合は変数を受け取ります。この例外は**terminalのみ**です: `execute_code`、ブラウザ/TUIホストの生成（`hermes_subprocess_env`）、Docker/Modalの子プロセス、`env_passthrough` の登録には適用されず、これらは封じられたままです。

### 仕組み

2つのメカニズムにより、特定の変数がサンドボックスのフィルターを通過できます:

**1. スキルスコープのパススルー（自動）**

スキルが（`skill_view` または `/skill` コマンドで）読み込まれ、`required_environment_variables` を宣言している場合、実際に環境に設定されているそれらの変数はすべて自動的にパススルーとして登録されます。欠落している変数（まだセットアップが必要な状態のもの）は登録され**ません**。

```yaml
# In a skill's SKILL.md frontmatter
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

このスキルを読み込むと、`TENOR_API_KEY` は `execute_code`、`terminal`（ローカル）、**そしてリモートバックエンド（Docker、Modal）**にパススルーされます — 手動の設定は不要です。

:::info Docker と Modal
v0.5.1より前は、Dockerの `forward_env` はスキルのパススルーとは別のシステムでした。現在はこれらが統合されています — スキルが宣言した環境変数は、`docker_forward_env` に手動で追加することなく、自動的にDockerコンテナとModalサンドボックスに転送されます。
:::

**2. 設定ベースのパススルー（手動）**

どのスキルでも宣言されていない環境変数については、`config.yaml` の `terminal.env_passthrough` に追加します:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

どちらのリストも、`terminal`、`execute_code`、`no_agent` のcronスクリプトに同様に適用されます。宣言された変数は、子プロセスが実行されるプロファイルの値で転送されます: 1つのプロセスが複数のプロファイルを提供する場合（マルチプロファイルのゲートウェイ、Desktop/ダッシュボードのバックエンド）、各プロファイルの宣言された値はそのプロファイル自身の `.env` / 機密情報ソースから取得され、起動プロファイルが設定したプロセス環境から取得されることは決してありません。また、起動プロファイルの `.env` の認証情報は、提供される側のプロファイルの子プロセスからは取り除かれます。

### 認証情報ファイルのパススルー（OAuthトークンなど） {#credential-file-passthrough}

一部のスキルは、環境変数だけでなく**ファイル**をサンドボックス内に必要とします — 例えば、Google Workspaceはアクティブなプロファイルの `HERMES_HOME` の下に `google_token.json` としてOAuthトークンを保存します。スキルはこれらをfrontmatterで宣言します:

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

読み込み時、Hermesはこれらのファイルがアクティブなプロファイルの `HERMES_HOME` に存在するかをチェックし、マウント用に登録します:

- **Docker**: 読み取り専用のバインドマウント（`-v host:container:ro`）
- **Modal**: サンドボックス作成時にマウント + 各コマンドの前に同期（セッション途中のOAuthセットアップに対応）
- **Local**: 不要（ファイルは既にアクセス可能）

`config.yaml` で認証情報ファイルを手動で列挙することもできます:

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

パスは `~/.hermes/` からの相対パスです。ファイルはコンテナ内の `/root/.hermes/` にマウントされます。このリストは `tools/credential_files.py`（`terminal.credential_files`）によって読み込まれます — `terminal:` ブロックの下にありますが、コアのターミナルバックエンドではなく認証情報ファイルのモジュールによって読み込まれるため、バンドルされた `DEFAULT_CONFIG` のスナップショットには含まれません。

### 借用したCLIログイン（Codex CLI、Claude Code） {#borrowed-cli-logins}

Hermesが `openai-codex` または `anthropic` について自前の使用可能なログインを持っていない場合、Codex CLIの `~/.codex/auth.json` とClaude Codeの `~/.claude/.credentials.json`（またはキーチェーンのエントリ）を借用し、あなたに代わってそれらをリフレッシュできます。どちらも使い捨てでローテーションするリフレッシュトークンを使用します: 2つのプログラムが1つのトークンファミリーを保持すると、先にリフレッシュした方がもう一方のコピーを無効化してしまい、「ターミナルで一度ログインしたのにHermesが失敗し続ける」（またはその逆）という形で現れます。それらのCLIをHermesと並行して実行する場合は、Hermesに専用のログインを与え、取り込みをオフにしてください:

```yaml
auth:
  adopt_external_logins: false   # default: true
```

このスイッチをオフにすると、Hermesはそれらのファイルを読み込んだりリフレッシュしたりすることはありません: `claude_code` の認証情報プールの行は消え、`hermes auth list` はその旨を1行で表示し、ログにはプロセスごとにINFO行が1行出力されます。影響を受けるのは自動の取り込みだけです — `hermes auth add openai-codex` は、既存のCodex CLIのログインをインポートする前に引き続き確認します。自動復旧も、Hermesが既に保持している認証情報を修復するだけです: 別のChatGPTワークスペースへのCodex CLI/Desktopのログインは警告とともに拒否され（`hermes auth add openai-codex` で再認証してください）、復旧の実行中に完了したログインが上書きされることはありません。自分のログインは `hermes auth add anthropic` / `hermes auth add openai-codex` で追加してください。

### 各サンドボックスがフィルターするもの

| Sandbox | Default Filter | Passthrough Override |
|---------|---------------|---------------------|
| **execute_code** | 名前に `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD`、`AUTH` を含む変数をブロック。安全なプレフィックスの変数のみ通過 | ✅ パススルー変数は両方のチェックをバイパス |
| **terminal**（ローカル） | 明示的なHermesインフラ変数（プロバイダーキー、ゲートウェイトークン、ツールAPIキー）をブロック | ✅ パススルー変数はブロックリストをバイパス |
| **terminal**（Docker） | デフォルトではホストの環境変数なし | ✅ パススルー変数 + `docker_forward_env` が `-e` で転送される |
| **terminal**（SSH） | デフォルトではホストの環境変数なし | ✅ パススルー変数が `SendEnv` で転送される。リモートの `sshd_config` に対応する `AcceptEnv` が必要（[SSHバックエンド](configuration.md#ssh-backend) を参照） |
| **terminal**（Modal） | デフォルトではホストの環境変数/ファイルなし | ✅ 認証情報ファイルがマウントされる。環境変数のパススルーは同期経由 |
| **MCP** | 安全なシステム変数 + 明示的に設定された `env` を除くすべてをブロック | ❌ パススルーの影響を受けない（代わりにMCPの `env` 設定を使用） |

### セキュリティ上の考慮事項

- パススルーは、あなたまたはあなたのスキルが明示的に宣言した変数にのみ影響します — 任意のLLM生成コードに対するデフォルトのセキュリティ姿勢は変わりません
- 認証情報ファイルはDockerコンテナに**読み取り専用**でマウントされます
- Skills Guardは、インストール前にスキルのコンテンツを疑わしい環境変数アクセスのパターンについてスキャンします
- 欠落/未設定の変数は決して登録されません（存在しないものは漏洩しようがありません）
- Hermesのインフラの機密情報（プロバイダーAPIキー、ゲートウェイトークン）は決して `env_passthrough` に追加すべきではありません — これらには専用のメカニズムがあります。そのような名前は宣言時に拒否され、宣言済みの名前を後からプラットフォームアダプターが要求した場合（スキルの読み込み後に登録されるプラグインアダプター）は、その時点から転送されなくなります

## MCP認証情報の取り扱い

MCP（Model Context Protocol）サーバーのサブプロセスは、認証情報の偶発的な漏洩を防ぐため、**フィルタリングされた環境**を受け取ります。

### 安全な環境変数

これらの変数のみがホストからMCPのstdioサブプロセスに渡されます:

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

加えて任意の `XDG_*` 変数も渡されます。それ以外のすべての環境変数（APIキー、トークン、機密情報）は**取り除かれます**。

[外部の機密情報ソース](/docs/user-guide/secrets/)（Bitwarden、1Password、コマンドヘルパー）から供給された変数も渡されますが、Hermesが管理する認証情報は除きます: モデルプロバイダーのキー、Hermes自身が使用するツールのキー、ボット、ゲートウェイ、リレーのトークンです。これには、Hermesがそうした認証情報を読み取るすべての名前が含まれます: 番号付きのローテーションキー（`DEEPSEEK_API_KEY_2`）、[認証情報プール](features/credential-pools.md) のエントリが指す任意の変数（`source: env:MY_KEY`）、プロファイルの `config.yaml` がプロバイダーキーとして指定する任意の変数（`model`、`custom_providers`、`providers`、`auxiliary`、またはフォールバックのエントリにおける `key_env` / `api_key_env`、あるいは `${VAR}` / `${env:VAR}` として書かれた `api_key` のような認証情報フィールド）、およびアクティブなプロファイルが使用するプロファイル接尾辞付きの形式（`TELEGRAM_BOT_TOKEN_<PROFILE>`）です。これらは保存場所にかかわらず取り除かれます。サーバーがそのいずれかを必要とする場合は、そのサーバーの `env` で宣言してください（例: `OPENAI_API_KEY: "${OPENAI_API_KEY}"`）。

MCPサーバーの `env` 設定で明示的に定義された変数は渡されます:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # Only this is passed
```

### 認証情報の伏せ字

MCPツールからのエラーメッセージは、LLMに返される前にサニタイズされます。次のパターンは `[REDACTED]` に置き換えられます:

- GitHub PAT（`ghp_...`）
- OpenAI形式のキー（`sk-...`）
- Bearerトークン
- `token=`、`key=`、`API_KEY=`、`password=`、`secret=` パラメータ

### ウェブサイトアクセスポリシー

エージェントがWebおよびブラウザツールを通じてアクセスできるウェブサイトを制限できます。これは、エージェントが内部サービス、管理パネル、その他の機密URLにアクセスするのを防ぐのに役立ちます。

```yaml
# In ~/.hermes/config.yaml
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

ブロックされたURLがリクエストされると、ツールはそのドメインがポリシーによってブロックされていることを説明するエラーを返します。ブロックリストは `web_search`、`web_extract`、`browser_exec`、およびすべてのURL対応ツールにわたって適用されます。

詳細は設定ガイドの [ウェブサイトブロックリスト](./configuration.md#website-blocklist) を参照してください。

### SSRF保護

すべてのURL対応ツール（Web検索、Web抽出、ビジョン、ブラウザ）は、サーバーサイドリクエストフォージェリ（SSRF）攻撃を防ぐため、フェッチする前にURLを検証します。ブロックされるアドレスには次が含まれます:

- **プライベートネットワーク**（RFC 1918）: `10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`
- **ループバック**: `127.0.0.0/8`、`::1`
- **リンクローカル**: `169.254.0.0/16`（`169.254.169.254` のクラウドメタデータを含む）
- **CGNAT / 共有アドレス空間**（RFC 6598）: `100.64.0.0/10`（Tailscale、WireGuard VPN）
- **クラウドメタデータのホスト名**: `metadata.google.internal`、`metadata.goog`
- **予約済み、マルチキャスト、未指定のアドレス**

SSRF保護はインターネットに面した利用では常に有効で、DNSの失敗はブロックとして扱われます（フェイルクローズ）。リダイレクトチェーンは、リダイレクトベースのバイパスを防ぐため、各ホップで再検証されます。

同じガードは、URLがあなたではなくリモートの相手から来るフェッチもカバーします: 生成プロバイダーが返す画像/動画のURL、モデルが編集用に提供する参照画像のURL、ペットのスプライトシートとpetdexのマニフェスト、skills.shのサイトマップのエントリです。プロバイダーやインデックスがそれらをプライベートまたはメタデータのアドレスに向けた場合、接続が開かれる前に拒否されます。オペレーター自身のプロバイダーの `base_url` は影響を受けません — 設定した `base_url`（OpenRouterの動画コンテンツエンドポイント）から直接フェッチされるダウンロードは、その最初のホップでのみプライベートアドレスのクラスチェックをスキップしますが、クラウドメタデータの最低ラインは引き続き適用され、そこから発行されるリダイレクトは完全に再検証されます — また、LAN上でホストされている画像生成プロバイダーが返す*結果*のURLをローカルにキャッシュするには、`security.allow_private_urls: true`（後述）が必要です。

#### 意図的にプライベートURLを許可する

一部の設定は、正当にプライベート/内部URLアクセスを必要とします — `home.arpa` をRFC 1918空間に解決するホームネットワーク、LANのみのOllama/llama.cppエンドポイント、内部wikiなどです。そうしたケースのために、グローバルなオプトアウトがあります:

```yaml
security:
  allow_private_urls: true   # default: false
```

オンにすると、Webツール、ブラウザ、ビジョンのURLフェッチ、ゲートウェイのメディアダウンロード、cronの `monitor_url` フェッチ、A2Aのプッシュコールバックは、RFC 1918 / ループバック / リンクローカル / CGNAT の宛先を拒否しなくなります。クラウドメタデータのエンドポイント（`169.254.169.254`、`metadata.google.internal` など）は、これをオンにしてもブロックされたままです。**これは意図的な信頼境界です** — ローカルネットワークに対して任意のプロンプトインジェクションされたURLをエージェントが実行することが許容できるリスクであるマシンでのみ有効にしてください。公開向けのゲートウェイではオフのままにすべきです。

ホストの部分文字列ガード（基盤となるIPが公開されている場合でも、紛らわしいUnicodeドメインのトリックをブロックする）は、この設定に関係なくオンのままです。

#### `allow_private_urls` がカバーするもの、しないもの {#what-allow_private_urls-covers-and-what-it-doesnt}

この設定は、Hermesが行うフェッチ、またはリモートブラウザに渡すフェッチを保護するものであり、プライベートホストへのすべての経路を止めるわけではありません。`false` の場合:

| Surface | Private URL (`localhost`, `192.168.x.x`, ...) |
|---|---|
| `web_extract` | バックエンドが呼び出される前に拒否されます。`true` の場合でも、ホストされた抽出バックエンドは自身のサーバーからフェッチするため、あなたのLANには到達できません。 |
| `vision_analyze` のURLフェッチ、画像/動画の結果ダウンロード、ゲートウェイのメディアダウンロード | 拒否 |
| `browser.auto_local_for_private_urls: false` のクラウドプロバイダー上、`/browser connect`（CDP）エンドポイント上、またはDocker/SSH/Modalのターミナルバックエンドでの `browser_exec` | 拒否（`Blocked: URL targets a private or internal address`） |
| 同じバックエンド上での `browser_cdp` の生のCDP呼び出し | プライベートなページが開いている間、および呼び出しがフェッチする絶対 `http(s)://` URL（`url` パラメータ、スクリプト、注入されたHTML）がプライベートな場合に拒否。スキーム相対の `//host` URLはクラウドメタデータについてのみチェックされます |
| `browser.auto_local_for_private_urls: true`（デフォルト）のクラウドプロバイダー上での `browser_exec` | Hermesホスト上のローカルChromiumサイドカーで**開かれます**（[ハイブリッドルーティング](./features/browser.md#hybrid-routing-cloud-for-public-urls-local-for-lanlocalhost)） |
| ローカルのターミナルバックエンドでのローカルChromium上の `browser_exec` | **開かれます**: ブラウザはターミナルが到達できないものには到達しません |
| `terminal`（`curl http://localhost`、...） | **開かれます**: URLガードなし |

クラウドメタデータのエンドポイントは、サイドカーとローカルChromiumを含め、`terminal` 以外のすべての行で拒否されます: `browser_exec` のコード内、生のCDP呼び出しがフェッチするURL内、またはそれが注入するスクリプトやHTML内に書かれたメタデータURLは、ブラウザがそれを目にする前に拒否されます。呼び出しが決してフェッチしない生のCDPパラメータはチェックされません: 入力テキスト（`Input.*`）、ダイアログの応答、Cookieのスコープ URL、リクエストのブロックやインターセプトのパターンです。このチェックはエージェントが書いたURLリテラルを読み取るもので、実行時に組み立てられたURL（文字列の連結、エスケープ）や、ブラウザが除去するタブや改行で分割されたURLは対象外です。また、その後のブラウザを監視するわけではないため、メタデータアドレスにリダイレクトしたり、そこからフェッチしたりする公開ページはこれでは止められません。クラウドVM上では、ネットワークでもメタデータサービスをブロックしてください（ホップ制限1のIMDSv2、ファイアウォールルール）。クラウドプロバイダーが設定されている間もエージェントのブラウザをプライベートホストから遠ざけるには、`browser.auto_local_for_private_urls: false` を設定してください。これは、サイドカーがホスト自身のループバックサービスとそのLANを見ることになるホスト型のインストール（Docker、VPS）で最も重要です。

#### ローカルプロキシのfake-ipレンジ

fake-ipモードのTUNプロキシ（Mihomo/Clashの `fake-ip`、Surgeの拡張モード）は、フィルター外のすべての名前に対して、DNSに自身のブロック — デフォルトでは `198.18.0.0/15`（RFC 2544のベンチマーク用） — のアドレスで応答します。これらの応答はプロキシのセンチネルであり内部ホストではありませんが、それがなければプライベートIPガードはそのようなホスト上のすべての外向きフェッチを拒否してしまいます: `web_extract`、プラットフォームの添付ファイルのダウンロード、ブラウザのリレーはすべて *URL targets a private or internal network address* で失敗し、リクエストはネットワークに到達しません。センチネルを通過させるには、そのブロックを宣言します:

```yaml
security:
  fake_ip_ranges:
    - 198.18.0.0/15
```

デフォルトでは空で、`allow_private_urls` より範囲が狭いものです: 宣言されたブロックだけが例外となり、それらはローカルプロキシが所有するレンジであるべきです（接続は引き続きプロキシに向かい、プロキシ自身が実際の宛先を解決します）。ループバック、RFC 1918、リンクローカル、CGNAT、クラウドメタデータの宛先はブロックされたままです — それらのクラスのいずれかと重なるエントリ（`0.0.0.0/0` や `::/0` を含む）は、ガードを広げるのではなく警告とともに無視されます。クラウドブラウザプロバイダーを使用するホストでは、宣言されたセンチネルは `browser.auto_local_for_private_urls` においてもプライベートとはみなされなくなるため、それらのページは引き続きクラウドブラウザに送られます。

### コンテンツレベルのコマンドチェック {#content-level-command-checks}

危険なコマンドの検出器は、破壊的な動詞のリストでは見逃される2つのコンテンツレベルの形も検出します:

- 機密情報を運ぶ `curl`/`wget` のリクエストボディ: 機密情報の名前を持つ変数（`-d "k=$OPENAI_API_KEY"`）や認証情報ファイル（`-F file=@.env`、`-T ~/.ssh/id_rsa`、`--post-file=/etc/passwd`）、またはアップロードする `curl`/`wget` にパイプされる認証情報ファイル。`Authorization` ヘッダーは通常のAPI利用であり、検出されません。
- 不可視または双方向のUnicode制御文字（ゼロ幅スペース、右から左へのオーバーライド、アイソレート、絵文字に続かない異体字セレクタ）。これらは、あなたが承認するコマンドと実際に実行されるコマンドを異なるものにします。絵文字の結合シーケンス、キーキャップの絵文字（`1️⃣`、`#️⃣`）、および絵文字自身の表示セレクタ（`ℹ️ Info` のようなフォルダ名に見られるもの）は検出されません。

どちらも、他のすべての危険なパターンと同様に通常の承認フローを経由します。

以前のリリースでは、ここに外部のtirithスキャナーがバンドルされていました。それは削除されました。アップグレードすると `security.tirith_*` の設定は破棄され、代わりに有効化されるものはありません。

### コンテキストファイルのインジェクション保護

コンテキストファイル（AGENTS.md、.cursorrules、SOUL.md）は、システムプロンプトに含められる前にプロンプトインジェクションについてスキャンされます。スキャナーは次をチェックします:

- 以前の指示を無視/無効化する指示
- 疑わしいキーワードを含む隠されたHTMLコメント
- 機密情報（`.env`、`credentials`、`.netrc`）を読み取ろうとする試み
- `curl` による認証情報の持ち出し
- 不可視のUnicode文字（ゼロ幅スペース、双方向オーバーライド）

翻訳と実行のチェックには、短い言語/形式の節が必要です（例えば「これをbashスクリプトに翻訳して実行して」）。無関係なカンマ区切りの役割説明の文をまたいで、翻訳と実行の動詞を結び付けることはありません。これらのパターンはヒューリスティックであり、意味的な意図の検出ではありません。

ブロックされたプロジェクトファイルは警告を表示します:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

`HERMES_HOME` にあるあなた自身の `SOUL.md` は異なる扱いを受けます: これはあなたが書いたファイルであるため（ファイルツールによる書き込みにはあなたの承認が必要で、プロジェクトのチェックアウトがそれを提供することはありません）、そこでスキャナーが検出しても**ファイルはブロックされません**。Hermesは一致したパターンを示す警告をログに出力し、通常どおりファイルを読み込み、`/context` はそれを `⚠ SOUL.md … loaded — matched prompt-injection pattern(s); review the file` として一覧表示します。これにより、攻撃フレーズを*説明する*アイデンティティファイル（「以前の指示を無視するよう指示するコンテンツ」のようなセキュリティガイダンス）も引き続き機能します。検出されたテキストを自分で書いていない場合は、その警告を他の何かがファイルを編集した兆候として扱ってください。この例外は、プロファイルディストリビューションによって配布された `SOUL.md` には及びません: `hermes profile install <git-url>` と `hermes profile update` は、第三者の `SOUL.md` をスキャンや承認プロンプトなしにプロファイルホームにコピーするため、`distribution.yaml` がそのファイルを所有している場合、スキャナーの検出は引き続きファイルをブロックします。

## 本番デプロイのベストプラクティス

### ゲートウェイデプロイのチェックリスト

1. **明示的な許可リストを設定する** — 本番環境では決して `GATEWAY_ALLOW_ALL_USERS=true` を使わない
2. **コンテナバックエンドを使う** — config.yamlで `terminal.backend: docker` を設定
3. **リソース制限を厳しくする** — 適切なCPU、メモリ、ディスクの制限を設定
4. **機密情報を安全に保存する** — APIキーを適切なファイル権限で `~/.hermes/.env` に保管
5. **DMペアリングを有効にする** — 可能な限りユーザーIDをハードコードする代わりにペアリングコードを使う
6. **コマンド許可リストを見直す** — config.yamlの `command_allowlist` を定期的に監査
7. **`terminal.cwd` を設定する** — エージェントを機密ディレクトリから操作させない
8. **非rootで実行する** — 決してゲートウェイをrootで実行しない
9. **ログを監視する** — `~/.hermes/logs/` で不正アクセスの試みを確認
10. **最新に保つ** — セキュリティパッチのため `hermes update` を定期的に実行

### APIキーの保護

```bash
# Set proper permissions on the .env file
chmod 600 ~/.hermes/.env

# Keep separate keys for different services
# Never commit .env files to version control
```

### ネットワーク分離 {#network-isolation}

最大限のセキュリティのため、ゲートウェイを別のマシンやVMで実行してください。`config.yaml` で `terminal.backend: ssh` を設定し、`~/.hermes/.env` の環境変数でホストの詳細を指定します:

```yaml
# ~/.hermes/config.yaml
terminal:
  backend: ssh
```

```bash
# ~/.hermes/.env
TERMINAL_SSH_HOST=agent-worker.local
TERMINAL_SSH_USER=hermes
TERMINAL_SSH_KEY=~/.ssh/hermes_agent_key
```

SSH接続の詳細は（`config.yaml` ではなく）`.env` に置かれるため、プロファイルのエクスポートと一緒にチェックインや共有がされません。これにより、ゲートウェイのメッセージング接続をエージェントのコマンド実行から分離して保ちます。

## TLS証明書の信頼

Hermesは `truststore` を通じてプラットフォームの検証器を初期化します。Windowsは証明書ストアを、macOSはシステムの信頼サービスを、LinuxはOpenSSLのシステム信頼パスを使用します。初期化に失敗した場合、Hermesはその失敗をログに出力し、OpenSSLのデフォルトにフォールバックします。

企業のTLSプロキシについては、組織のオペレーティングシステムの信頼手順に従ってそのルート証明書をインストールしてください。Hermesのプロバイダーリゾルバーは、もはや `HERMES_CA_BUNDLE` や旧来のCA環境変数の優先順位によって信頼を選択しません。サンドボックス化されたサブプロセスは、それぞれ独自のCA設定を持つことができます。

これに伴い、以前の起動時の証明書ガードもなくなりました: Hermesは起動時に `HERMES_CA_BUNDLE` / `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` を検証しなくなったため、`SSLConfigurationError` は発生せず、`HERMES_SKIP_SSL_GUARD` という回避手段も効果がありません。`HERMES_CA_BUNDLE` は、Nous Portalのログインフロー（`hermes login`、またはその `--ca-bundle` フラグ）でのみ引き続き尊重されます。標準の `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` 変数は、一部のツールが行う素の `requests`/`urllib` 呼び出し（および `pip`、`uv`、`curl`、Node）によって引き続き読み込まれるため、それらのいずれかに古いパスが設定されていると、起動時ではなくそれを使用する呼び出しの時点で失敗するようになりました。その場合は該当の変数を修正するか、未設定にしてください。

カスタムプロバイダーは、そのエンドポイント用に `ssl_ca_cert` を宣言できます。そのバンドルは、チャット、モデルのメタデータ、モデルカタログのプローブにおいてプラットフォームの信頼を置き換えます。ファイルが存在しない場合は警告が出され、プラットフォームの信頼にフォールバックします。`ssl_verify: false` は証明書の検証を無効にするもので、信頼できないネットワークでは安全ではありません。企業のルート証明書が欠けていることへの恒久的な対処として使わないでください。

プロバイダーのHTTPクライアントは、プロキシの設定を証明書の選択とは分けて保持します。環境に残った古いCAファイルのパスによって、それらのクライアントの起動が妨げられることはありません。PMのインデックスの認証情報は、その正確なHTTPSオリジンにのみ送信され、別のオリジンへのリダイレクトには送信されません。

## 配置によって信頼される拡張ポイント {#trusted-by-placement}

Hermesが実行できるサードパーティのコードの大半は、明示的な許可リストによって制限されています: 一般的なプラグインには `plugins.enabled` が必要で、シェルフックには初回使用時の承認（または `hooks_auto_accept`）が必要で、MCPサーバーは設定に列挙されます。1つのサーフェスだけは意図的に異なります:

| Extension point | Loaded from | Loaded when | Opt-in |
|-----------------|-------------|-------------|--------|
| [ゲートウェイのイベントフック](./features/hooks.md#gateway-event-hooks) | `<profile home>/hooks/<name>/`（`HOOK.yaml` + `handler.py`） | ゲートウェイの起動時（`HookRegistry.discover_and_load()`）、提供されるプロファイルごと | **ディレクトリを配置すること。** `plugins.enabled` のエントリもプロンプトもありません。`HERMES_SAFE_MODE` でもスキップされません。 |

ゲートウェイは、有効なすべてのフックディレクトリをゲートウェイ自身の権限でプロセス内にインポートします。これは見落としではなく、（`3988c3c245f` 以降の）文書化された契約です: プロファイルホームはオペレーターが所有する設定であり、そこに書き込める人は誰でも、`config.yaml` のシェルフックや `plugins.enabled` の編集を通じて既にあなたとしてコードを実行できるため、`hooks/` に別の同意ゲートを設けても信頼境界は動かず、手間が増えるだけです。`~/.hermes/hooks/` の内容は `config.yaml` の内容と同様に扱ってください — `handler.py` は配置する前に確認し、プロファイルホームの他の部分を監査するときは必ず `ls ~/.hermes/hooks/` も含めてください（このディレクトリは [保護されたパスの拒否リスト](#file-write-safety) に含まれていないため、通常の書き込み可能な状態です）。詳細: [ゲートウェイフックの信頼モデル](./features/hooks.md#gateway-hook-trust)。

## サプライチェーンのアドバイザリチェック

Hermesには、アクティブなvenv内のPythonパッケージのうち、既知の侵害されたバージョンの厳選カタログ（2026年5月の `mistralai 2.4.6` の汚染のようなサプライチェーンワーム）に一致するものを検出する、組み込みのアドバイザリスキャナーが同梱されています。実装は `hermes_cli/security_advisories.py` にあります。

実行のされ方:

- **CLI起動時のバナー。** いずれかのアドバイザリに一致した場合、完全な対処方法を確認するための `hermes doctor` への案内とともに、1行の警告が表示されます。
- **`hermes doctor`。** 有効なすべてのアドバイザリを、バージョンの詳細と2〜4ステップの対処手順とともに表示します。
- **ゲートウェイの起動時。** `gateway.log` に記録され、最初の対話メッセージに短いオペレーター向けバナーが付きます。

各アドバイザリには安定したIDがあります。内容を読んで対処したら、恒久的に非表示にできます:

```bash
hermes doctor --ack <advisory-id>
```

確認済みの記録は `config.security.acked_advisories` に永続化され、再起動後も維持されます。古いアドバイザリは意図的にカタログから削除**されません** — 残しておくことで、プライベートミラーにまだキャッシュされている可能性のある、過去に汚染されたバージョンについて新規インストールでも警告されるようにしています。

チェック自体は標準ライブラリのみを使い、アドバイザリごとに1回の `importlib.metadata.version()` の検索で実行されるため、起動のたびに実行しても安全です。

### オプション依存関係の遅延インストール

PMは、オプションのPython機能を `pyproject.toml` のextrasとして管理します。ソースインストーラーは `all` extraを選択します。ネイティブバンドルには、そのターゲットでサポートされるすべてのextrasが含まれます。これらは異なる機能セットです。

バックエンドが利用できないextraを要求すると、`pm.ensure_import("extra-name")` はプラグインの受け入れと同じ依存関係トランザクションを使用します:

1. PMはプラットフォームのサポートと `security.allow_lazy_installs` をチェックします。
2. PMは、既存のextrasと有効なプラグインの要件を含む完全な環境を準備します。
3. プラグインのメンバーがない場合は、コミット済みのロックをそのまま使用します。メンバーがある場合は、凍結されたワークスペースの同期の前に、以前の選択から解決します。
4. 選択を公開する前に候補を検証します。失敗した候補は、以前の環境を選択したままにします。
5. 現在のプロセスが以前の環境を使用している場合、PMはHermesの再起動が必要であることを報告します。インポート済みのライブラリをその場で置き換えることはありません。

出荷されたソース、ロック、署名済みのペイロードは変更されません。追加のツールとPython環境は、ベースのアーティファクトの外にある書き込み可能なストレージを使用します。プラグインの依存関係は完全な環境を共有し、分離されたPythonサンドボックスではありません。互換性のある推移的な依存関係は変わることがありますが、宣言された制約と厳密なピン留めは引き続き拘束力を持ちます。

| Control | Behavior |
|---|---|
| 宣言されたextras | ヘルパーは任意のpipコマンドではなく、プロジェクトのextra名を受け付けます。削除された `LAZY_DEPS` の機能名レジストリは使用されません。 |
| 検証済みのツール | 管理対象のツールアーカイブには、`pm/lock.json` にバージョンとSHA-256ハッシュがあります。 |
| アトミックな選択 | 準備と検証は、新しいランタイムの選択の公開より前に行われます。 |
| 失敗の報告 | 失敗すると `pm.InstallError` が発生します。PMの同期レシートには、失敗したステップとポリシーによる拒否が含まれます。 |
| プラグインの自動削除なし | 依存関係の統合に失敗しても、インストール済みのプラグインが黙って無効化されたり削除されたりすることはありません。 |

オンデマンドのインストールを無効にするには、次を実行します:

```bash
hermes config set security.allow_lazy_installs false
```

既にインストールされた依存関係は引き続き使用できます。明示的なPMのインストールコマンドは、オンデマンドのインストールとは別のものです。バンドルの凍結された機能リストが存在し、遅延インストールが無効になっている場合、要求されるPythonのextra名はそのリストに制限されます。この設定は、明示的なプラグインの受け入れや手動のパッケージマネージャーコマンドを一律に禁止するものではありません。公式Dockerイメージも、内部の環境ポリシーによってオンデマンドのインストールを無効にしています。

依存関係が不足している場合は、`hermes tools` と `hermes doctor` を使って要件を特定してください。署名済みのペイロードやシステムのPythonに対してpipを実行しないでください。インストールの所有権、診断、コマンドの境界については [パッケージ管理](/docs/reference/package-management) を参照してください。
