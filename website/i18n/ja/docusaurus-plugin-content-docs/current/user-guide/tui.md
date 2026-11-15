---
sidebar_position: 2
title: "TUI"
description: "Hermes 向けのモダンなターミナル UI を起動 — マウス操作対応、リッチなオーバーレイ、ノンブロッキング入力。"
---

# TUI

TUI は Hermes のモダンなフロントエンドです。[Classic CLI](cli.md) と同じ Python ランタイムを基盤としたターミナル UI です。同じエージェント、同じセッション、同じスラッシュコマンドで、よりクリーンで応答性の高いインターフェースを通じてそれらを操作できます。

Hermes をインタラクティブに実行する際の推奨方法です。

## 起動

```bash
# Launch the TUI
hermes --tui

# Resume the latest TUI session (falls back to the latest classic session)
hermes --tui -c
hermes --tui --continue
hermes --tui --resume latest

# Resume a specific session by ID or title
hermes --tui -r 20260409_000000_aa11bb
hermes --tui --resume "my t0p session"

# Resume the latest session for a specific project directory
hermes --tui --resume latest --in ./my-project

# Run source directly — skips the prebuild step (for TUI contributors)
hermes --tui --dev
```

環境変数で有効化することもできます:

```bash
export HERMES_TUI=1
hermes          # now uses the TUI
hermes chat     # same
```

または `~/.hermes/config.yaml` で永続的なデフォルトにします:

```yaml
display:
  interface: tui   # "cli" (default) or "tui"
```

`display.interface: tui` にすると、引数なしの `hermes`（および `hermes chat`）で TUI が起動します。明示的なフラグが常に優先されます — 1 回の起動だけクラシック REPL に戻るには `hermes --cli` を、設定のデフォルトが `cli` のときに TUI を強制するには `hermes --tui` / `HERMES_TUI=1` を使います。

クラシック CLI は引き続き出荷時のデフォルトです。[CLI Interface](cli.md) に記載されているもの — スラッシュコマンド、クイックコマンド、スキルのプリロード、パーソナリティ、複数行入力、割り込み — はすべて TUI でも同一に動作します。

## TUI を使う理由

- **即座の初回フレーム** — アプリの読み込みが完了する前にバナーが描画されるため、Hermes の起動中にターミナルがフリーズしたように感じることはありません。
- **ノンブロッキング入力** — セッションの準備が整う前にメッセージを入力してキューに入れられます。最初のプロンプトはエージェントがオンラインになった瞬間に送信されます。
- **リッチなオーバーレイ** — モデルピッカー、セッションピッカー、承認・確認プロンプトはすべてインラインフローではなくモーダルパネルとして描画されます。
- **ライブセッションパネル** — ツールとスキルは初期化されるにつれて漸進的に埋まっていきます。
- **マウス操作対応の選択** — SGR 反転ではなく、均一な背景でドラッグしてハイライトできます。ターミナルの通常のコピー操作でコピーできます。
- **代替スクリーンレンダリング** — 差分更新により、ストリーミング時のちらつきがなく、終了後のスクロールバックの乱れもありません。
- **コンポーザーの利便機能** — 長いスニペットのインラインペースト折りたたみ、`Cmd+V` / `Ctrl+V` のテキストペースト（クリップボード画像へのフォールバック付き）、ブラケットペーストの安全性、画像/ファイルパス添付の正規化。

同じ[skins](features/skins.md)と[personalities](features/personality.md)が適用されます。`/skin ares`、`/personality pirate` でセッション途中に切り替えると、UI がライブで再描画されます。カスタマイズ可能なキーの全リストと、クラシック vs TUI のどちらに適用されるかについては [Skins & Themes](features/skins.md) を参照してください。TUI はバナーパレット、UI カラー、プロンプトのグリフ/カラー、セッション表示、補完メニュー、選択背景、`tool_prefix`、`help_header` を尊重します。

### 折りたたみ可能なバナーセクション

TUI の起動バナーは、ランタイム情報を 4 つの折りたたみ可能なセクションにまとめ、それぞれのセクションタイトルの横に `▸` / `▾` のシェブロンを表示します:

| セクション | デフォルトの状態 |
|---------|---------------|
| Tools | 開く |
| Skills | 折りたたみ |
| System Prompt | 折りたたみ |
| MCP Servers | 折りたたみ |

セクションのヘッダー（またはそのシェブロン）のどこかをクリックすると切り替わります。Tools の一覧はセッション開始時に最もよく確認されるセクションなのでデフォルトで開いており、Skills、System Prompt、MCP Servers はデフォルトで折りたたまれているため、数十個のスキルをインストールしたり多数の MCP サーバーを接続したりしてもバナーはコンパクトなままです。状態はバナーのインスタンスに固有なので、次回の起動ではデフォルトに戻ります。

## 要件

- **Node.js** ≥ 20 — TUI は Python CLI から起動されるサブプロセスとして実行されます。`hermes doctor` でこれを検証できます。
- **TTY** — クラシック CLI と同様に、stdin のパイプや非インタラクティブ環境での実行はシングルクエリモードにフォールバックします。

初回起動時、Hermes は TUI の Node 依存関係を `ui-tui/node_modules` にインストールします（一度限り、数秒）。以降の起動は高速です。新しい Hermes バージョンをプルすると、ソースが dist より新しい場合に TUI バンドルが自動的に再ビルドされます。

:::tip git worktree をまたいで作業していますか？
多数の worktree から `hermes --tui --dev` を実行するコントリビューターは、チェックアウトごとにインストールする代わりに 1 つの `node_modules` を共有できます — [TUI & Desktop from Worktrees](/docs/developer-guide/worktree-ui-dev) を参照してください。
:::

### 外部プリビルド

プリビルド済みバンドルを同梱するディストリビューション（Nix、システムパッケージ）は、Hermes をそれに向けることができます:

```bash
export HERMES_TUI_DIR=/path/to/prebuilt/ui-tui
hermes --tui
```

そのディレクトリには `dist/entry.js` が含まれている必要があります。

## キーバインド

キーバインドは [Classic CLI](cli.md#keybindings) と完全に一致します。動作上の違いは次の点のみです:

macOS では、`F7` は物理的な F7 ファンクションキーを意味します。MacBook のキーボードでは、その列が代わりにシステム制御に使われている場合があります。**F7** を押すときに **Fn**（新しいキーボードでは **地球儀** キー）を押し続けるか、**システム設定 → キーボード → キーボードショートカット → ファンクションキー** で **F1、F2 などのキーを標準のファンクションキーとして使用** を有効にしてください。それでもターミナルがキーを転送しない場合は、下記の同等の **Ctrl+R** バインディングを使ってください。

- **`Ctrl+T`** は、自動のライブ作業ドック（サブエージェントと、`terminal(background=true)` で起動したものの **Processes** ブロック）を全高の `/agents` 一覧に展開します。ワーカーを選んで **Enter**（または **`t`**）でライブのトランスクリプト、**`d`** で詳細、**`e`** で舵取り、**`x`** で停止します。ドックは行数をターミナルの高さに合わせ、コンポーザーの下書きを保持します。[サブエージェントの監視](./features/delegation.md#monitoring-running-subagents-agents)を参照してください。
- 継続中の **`/goal`** は、ライブドックの上に専用の行を持ちます（`⊙ goal · 3/20 turns · 2 subgoals · …`、または理由付きの `⏳ goal parked` / `⏸ goal paused`）。ターン途中のツールを含め、何かがゴールを変更するとすぐに再描画され、ゴールが完了またはクリアされると消えます。キューに入ったフォローアップはその上に一覧表示されます。
- **`Ctrl+R`** は、ライブドックをデフォルトのプレビューと 1 行のサマリーの間で切り替えます。ターミナルがファンクションキーを転送する環境では、**`F7`** も任意のエイリアスとして使えます。これはモニターを開いたりコンポーザーのフォーカスを移したりしません。選択はこの TUI プロセスの間だけ続き、設定は変更しません。
- **マウスドラッグ** で均一な選択背景によってテキストをハイライトします。
- **`Ctrl+P`** は、折りたたまれたペースト（`[[ … [60 lines] … ]]`）を編集可能なテキストに展開し、読んだり変更したりできるようにします。もう一度押す（または送信する）とチップに折りたたまれ、モデルは編集後のテキストを受け取ります。送信したメッセージはチップを保持し、その下に **Pasted text #N** の行が付きます: 行をクリックすると開き、空のコンポーザーで `Ctrl+P` を押すと送信済みのすべてのペーストを開閉します。
- **`Cmd+V` / `Ctrl+V`** はまず通常のテキストペーストを試み、次に OSC52/ネイティブクリップボードの読み取りにフォールバックし、最後にクリップボードまたはペーストされたペイロードが画像として解決された場合に画像添付を行います。
- **`/terminal-setup`** はローカルの VS Code / Cursor / Windsurf のターミナルバインディングをインストールし、macOS で `Cmd+Enter` と undo/redo の互換性を向上させます。
- **スラッシュ自動補完** はインラインドロップダウンではなく、説明付きのフローティングパネルとして開きます。
- **エージェントがビジーな間の入力**（`display.busy_input_mode: queue`、TUI のデフォルト）は、メッセージを即座にゲートウェイに送信します。ゲートウェイの永続的なセッションキューがそれを実行中のターンの後ろに並べるため、ゲートウェイのクラッシュや再起動を乗り越え、セッションにアタッチしているすべてのクライアントの `queued (N)` パネルに `[queued] …` として表示されます — Desktop アプリにも同じ一覧が見えます。メッセージのトランスクリプトのバブルは、そのターンが実際に始まったときに表示されます。`steer` と `interrupt` は従来どおり実行中のターンに作用します。
- **`Ctrl+X`** はライブセッションスイッチャーを開きます。キューに入れられたメッセージ（エージェントの実行中に送信されたもの）がハイライトされているときは、代わりに引き続きそのキュー内のメッセージを削除します — ゲートウェイのキューに入った行の場合はゲートウェイ上でキャンセルされる（`prompt.cancel`）ため、すべての閲覧者から消えます。編集した行を選んで再送信すると、元のメッセージがキャンセルされ、編集内容が新しい入力として送信されます（その時点で元のメッセージがすでに開始されていた場合は停止され、その後に編集内容が実行されます）。**`Esc`** は編集をキャンセルし、削除せずにハイライトを解除します。
- アイドル状態の空のコンポーザーで **`Esc`、`Esc`** を押すと、以前のプロンプトに戻ります（Codex 形式のバックトラック。`/rewind` と同じ）。最初の `Esc` で準備状態になり、ヒント行に `esc again to edit previous message` が表示されます（以前のプロンプトがある場合のみ。ない場合、2 回目の `Esc` で `No previous message to edit.` と表示されます）。時間制限はありませんが、他のキー、クリック、ペースト、新しいターンのいずれかで準備状態は解除され、通常どおりの動作になります。2 回目の `Esc` で **トランスクリプトの閲覧** が開き、最新のプロンプトが選択されて反転表示され、ヒント行 `Browsing transcript · ↑↓/jk scroll · ←→/hl prompts · ↵ edit · f fork · esc back` が表示されます。**←/h** で 1 つ古いプロンプト、**→/l** で 1 つ新しいプロンプトを選びます（端で止まります）。**↑/↓/j/k** はトランスクリプトを 1 行、**PgUp/PgDn** は 1 ページスクロールし、選択は変わりません。**Enter** で閲覧を閉じ、選択したプロンプトを送信せずにコンポーザーに入れ、ヒント `↺ Editing prompt N — Enter rewinds here and sends · Esc Esc cancels` を表示します: 編集して **Enter** を押すと、そのプロンプトとそれ以降のすべてがセッションから削除され、編集内容が新しいターンとして送信されます。送信するまで何も削除されません。空のコンポーザーの送信、スラッシュコマンド、`Esc Esc`、`Ctrl+C`、セッションの切り替えは編集をキャンセルし、何も削除しません。**f** は代わりにフォークします: 選択したプロンプトより*前*のすべてを含む新しいセッションを作り、そのプロンプトをコンポーザーで待機させます（送信はされません）。元のセッションはそのまま残り、最初のプロンプトをフォークすると、それが事前入力された新しいセッションが始まります。**Esc** は以前スクロールしていた位置に戻り、**q** または `Ctrl+C` で閉じ、その他のキーは閲覧を閉じてコンポーザーに入力されます。トランスクリプトがターミナル自身のスクロールバックにある `--tui-native` モードでは、代わりにコンポーザーの上にセッションのプロンプトと返信のトランスクリプトページャーが開き、同じキーが使えます。コンポーザーに下書きがある場合、`Esc Esc` は引き続き下書きを破棄し、ターン途中では引き続き割り込みます。画像を含むプロンプトは番号を保持しますが、選択できません。
- **承認プロンプト** では、**Deny** の後に **No, and tell Hermes what to do instead** が提示されます。それを選び（または番号を押し）、指示を入力して **Enter** を押すと: 指示は実行中のターンに舵取りとして送られ（`/steer` と同様）、コマンドは拒否されるため、Hermes は両方を合わせて受け取り、あなたのやり方で再試行します。ターンがすでに終わっていた場合、指示は次のプロンプトとしてキューに入ります。入力行での **Esc** は選択肢に戻り、選択肢での **Esc** は引き続きそのまま拒否します。
- **`Ctrl+G` / `Ctrl+X Ctrl+E`** — 現在の入力バッファを `$EDITOR` で開き、複数行・長文プロンプトの作成を行います。保存して終了すると、内容がプロンプトとして送り返されます。

## スラッシュコマンド {#slash-commands}

すべてのスラッシュコマンドは変更なく動作します。いくつかは TUI 専用で、インラインパネルではなくよりリッチな出力を生成したりオーバーレイとして描画されたりします:

| コマンド | TUI での動作 |
|---------|-------------|
| `/help` | カテゴリ分けされたコマンドのオーバーレイ、矢印キーで操作可能 |
| `/rewind` | トランスクリプトの閲覧（準備ステップなし）: **←/→** で以前のプロンプトを選び、**Enter** で編集・再送信のためにコンポーザーに入れ（送信するとそれ以降のすべてが削除されます）、**f** でその直前から新しいセッションをフォークし、**Esc** で戻ります。空のコンポーザーでの `Esc`、`Esc` でも同様 |
| `/sessions`（エイリアス `/switch`） | ライブセッションスイッチャー — 開いている TUI セッションを一覧表示し、切り替え、閉じ、または別のセッションを開始 |
| `/model` | プロバイダーごとにグループ化されたモーダルモデルピッカー、コストヒント付き |
| `/skin` | ライブプレビュー — ブラウズしながらテーマ変更が適用される |
| `/details` | 詳細なツールコール表示の切り替え（グローバルまたはセクション単位） |
| `/usage` | リッチなトークン / コスト / コンテキストパネル |
| `/agents`（エイリアス `/tasks`） | 可観測性オーバーレイ — ライブのサブエージェントツリー（kill/pause コントロール付き）、ブランチ単位のコスト / トークン / ファイル集計、ターンごとの履歴 |
| `/connectors` | コネクタアカウントの一覧: アカウントの名前変更、追加、再接続、削除、退役したアカウントの表示。キーについては [Slash Commands](../reference/slash-commands.md#tools--skills) を参照 |
| `/reload` | `~/.hermes/.env` を実行中の TUI プロセスに再読み込みし、新たに追加された API キーを再起動なしで反映 |
| `/mouse [on\|off\|toggle\|wheel\|buttons\|all]` | 実行時にマウストラッキングのプリセットを選択（`config.yaml` の `display.mouse_tracking` にも永続化）。`wheel`（1000+1006）は、tmux がプロンプト行の上で「No image in clipboard」を連発する原因となるホバーイベントなしで、スクロールホイールによるスクロールを維持します。`buttons` はドラッグによる選択を追加します。`all` はホバー駆動の UI を含むデフォルトです。 |

その他すべてのスラッシュコマンド（インストール済みスキル、クイックコマンド、パーソナリティ切り替えを含む）は、クラシック CLI と同一に動作します。[Slash Commands Reference](../reference/slash-commands.md) を参照してください。

## ライブセッションスイッチャー

1 つのターミナルを複数の TUI セッションのディスパッチャーとして使いたいときは、ライブセッションスイッチャーを使います。一覧に表示されるのは、この TUI プロセスで現在ライブなセッションだけです。閉じたセッションは保存済みのトランスクリプトとして残り、`/resume` または `hermes --tui --resume <id-or-title>` で再び開けます。

次のいずれかで開きます:

- TUI から `Ctrl+X`。
- `/sessions` または `/switch`。
- `/sessions new` で新しいライブセッションを即座に作成。
- ステータスラインの `N live sessions` のカウントをクリック。

<img alt="Hermes TUI Session Orchestrator with one live session and a +new row" src="/docs/img/docs/tui-session-orchestrator/session-orchestrator.png" />

<video controls muted loop playsInline src="/docs/img/docs/tui-session-orchestrator/session-orchestrator-demo.mp4" title="Hermes TUI Session Orchestrator demo" style={{maxWidth: '100%'}}></video>

スイッチャー内では:

- `↑` / `↓` で選択を移動します。マウスクリックでも行を選択できます。
- `Enter` で選択したライブセッションに切り替えます。
- `Ctrl+D` で選択したライブセッションを閉じます。
- `Ctrl+N` で空のライブセッションを開始します。
- `Ctrl+R` でライブセッションの一覧を更新します。
- `Esc` でスイッチャーを閉じます。
- `+new` を選び、プロンプトを入力して `Enter` を押すと、新しいライブセッションをディスパッチします。その新しいセッションだけのモデルを選びたい場合は、先に `Tab` を押してください。

## LaTeX 数式レンダリング

TUI の Markdown パイプラインは LaTeX 数式をインラインでレンダリングします: `$E = mc^2$` や `$$\frac{a}{b}$$` は、生の TeX ソースではなく Unicode 整形された数式としてレンダリングされます。インライン数式とブロック数式の両方で動作します。サポートされていない構文は、コピー可能なまま残るように、リテラルの TeX をコードスパンで囲んで表示するフォールバックになります。

これは常時オンで、設定不要です。クラシック CLI は生の TeX を保持します。

## ライトターミナル検出

TUI はライトターミナルを自動検出し、それに応じてライトテーマに切り替えます。検出は 3 つの層で動作します:

1. `HERMES_TUI_THEME` 環境変数 — 最優先。値: `light`、`dark`、または生の 6 文字の背景色 16 進数（例: `ffffff`、`1a1a2e`）。
2. `COLORFGBG` 環境変数 — xterm 派生のターミナルで使われる、おなじみの「背景色は何か?」のヒント。
3. OSC 11 によるターミナル背景プローブ — `COLORFGBG` を設定しないモダンなターミナル（Ghostty、Warp、iTerm2、WezTerm、Kitty）で動作します。

ターミナルに関係なくライトテーマを恒久的に使いたい場合:

```bash
export HERMES_TUI_THEME=light
```

## ビジーインジケーターのスタイル

ステータスバーのビジーインジケーターは差し替え可能です。デフォルトは、エージェント作業中に Hermes のかわいい顔文字パレットを 2.5 秒ごとに回転させます。設定または `/indicator` スラッシュコマンドで別のスタイルを選択できます:

```yaml
display:
  tui_status_indicator: kaomoji   # kaomoji | emoji | unicode | ascii
```

またはセッション中に `/indicator emoji`（など）。各スタイルはグリフ幅が揃って同梱されているため、回転時にステータスバーの残りがガタつきません。

## 自動再開

デフォルトでは、`hermes --tui` は起動ごとに新しいセッションを開始します。最新の TUI セッションに自動的に再アタッチする（ターミナルや SSH 接続が予期せず切断されたときに便利）には、オプトインします:

```bash
export HERMES_TUI_RESUME=1          # most-recent TUI session
# or:
export HERMES_TUI_RESUME=<session-id>   # specific session
```

起動ごとに上書きするには、変数を解除するか `--resume <id>` を明示的に渡してください。

## ステータスライン

TUI のステータスラインはエージェントの状態をリアルタイムで追跡します:

セッションに名前が付くと、そのタイトルがステータスラインの右端にアクセントカラーのバッジとして表示されます。タイトルはワークスペースのラベルの位置を占め、幅の狭いターミナルでは切り詰められます。

| ステータス | 意味 |
|--------|------|
| `starting agent…` | セッション ID は有効。ツールとスキルはまだオンラインになりつつある状態。入力可能 — メッセージはキューに入り、準備ができ次第送信されます。 |
| `ready` | エージェントはアイドルで、入力を受け付けています。 |
| `thinking…` / `running…` | エージェントが推論中またはツール実行中です。 |
| `interrupted` | 現在のターンがキャンセルされました。Enter を押すと再送信します。 |
| `forging session…` / `resuming…` | 初回接続または `--resume` のハンドシェイク。 |

スキンごとのステータスバーのカラーとしきい値はクラシック CLI と共有されます。カスタマイズについては [Skins](features/skins.md) を参照してください。

ステータスラインは次の情報も表示します:

- **作業ディレクトリと git ブランチ** — `~/projects/hermes-agent (docs/two-week-gap-sweep)`。サイドターミナルで `git checkout` するとブランチサフィックスが更新される（mtime キャッシュ）ため、TUI は起動時のブランチではなく実際のアクティブブランチを反映します。
- **プロンプトごとの経過時間** — ターン実行中は `⏱ 12s/3m 45s`（ライブ）、ターン完了後は `⏲ 32s / 3m 45s` に固定されます。最初の数値は最後のユーザーメッセージからの経過時間、2 番目はセッション全体の継続時間です。新しいプロンプトごとにリセットされます。
- **`🗜️ N`** — 実行中のセッションが自動圧縮された回数。最初の圧縮が発生すると表示されます。
- **`▶ N`** — このセッションで現在実行中の `/bg` タスクの数。少なくとも 1 つのタスクが実行中であれば表示されます。
- **`⚠ YOLO`** — YOLO モードがオンのとき（`hermes --yolo`、`/yolo`、または `HERMES_YOLO_MODE=1`）に表示される警告。同じバッジが起動バナーにも表示されるため、自動承認のセッションを気づかずに起動することはありません。

## 設定

TUI はすべての標準 Hermes 設定を尊重します: `~/.hermes/config.yaml`、プロファイル、パーソナリティ、スキン、クイックコマンド、クレデンシャルプール、メモリプロバイダー、ツール/スキルの有効化。TUI 専用の設定ファイルは存在しません。

いくつかのキーは TUI のインターフェースを特に調整します:

```yaml
display:
  skin: default              # any built-in or custom skin
  personality: helpful
  details_mode: collapsed    # hidden | collapsed | expanded — global accordion default
  sections:                  # optional: per-section overrides (any subset)
    thinking: expanded       # always open
    tools: expanded          # always open
    activity: collapsed      # opt back IN to the activity panel (hidden by default)
  mouse_tracking: all        # off | wheel | buttons | all (or true/false for back-compat).
                             #   wheel   — 1000+1006 (scroll + click; no drag, no hover —
                             #             recommended inside tmux to silence the prompt-row
                             #             "No image in clipboard" spam from hover events)
                             #   buttons — adds 1002 for terminal-side drag selection
                             #   all     — adds 1003 for hover (scrollbar paginate-on-hover,
                             #             link mouseenter, etc.)
  inline_images: auto        # auto | kitty | iterm | unicode | off — see the CLI guide's "Inline Images"
```

ランタイム切り替え:

- `/details [hidden|collapsed|expanded|cycle]` — グローバルモードを設定
- `/details <section> [hidden|collapsed|expanded|reset]` — 1 つのセクションを上書き
  （セクション: `thinking`、`tools`、`subagents`、`activity`）

**デフォルトの表示**

TUI は、シェブロンの壁ではなくライブのトランスクリプトとしてターンをストリーミングする、こだわりのセクションごとのデフォルトを同梱しています:

- `thinking` — **expanded**。モデルが推論を出力するにつれてインラインでストリーミングされます。
- `tools` — **expanded**。ツールコールとその結果が開いた状態でレンダリングされます。
- `subagents` — グローバルの `details_mode` にフォールスルー（デフォルトではシェブロン下で collapsed — 委譲が実際に発生するまで静かなまま）。
- `activity` — **hidden**。アンビエントなメタ情報（ゲートウェイのヒント、ターミナル互換のうながし、バックグラウンド通知）は、日常的な使用ではほとんどがノイズです。ツールの失敗は失敗したツール行にインラインでレンダリングされます。アンビエントなエラー/警告は、すべてのパネルが非表示のときにフローティングアラートのバックストップとして表示されます。

セクションごとの上書きは、セクションのデフォルトとグローバルの `details_mode` の両方より優先されます。レイアウトを変更するには:

- `display.sections.thinking: collapsed` — thinking を再びシェブロン下に戻す
- `display.sections.tools: collapsed` — ツールコールを再びシェブロン下に戻す
- `display.sections.activity: collapsed` — アクティビティパネルを再びオプトインする
- 実行時の `/details <section> <mode>`

`display.sections` に明示的に設定したものはデフォルトより優先されるため、既存の設定はそのまま動作し続けます。

## セッション

セッションは TUI とクラシック CLI の間で共有されます — どちらも同じ `~/.hermes/state.db` に書き込みます。一方でセッションを開始し、もう一方で再開できます。セッションピッカーは両方のソースのセッションをソースタグ付きで表示します。

ライフサイクル、検索、圧縮、エクスポートについては [Sessions](sessions.md) を参照してください。

## TUI とゲートウェイの通信方法

デフォルトでは、TUI はプロファイルの共有ゲートウェイにアタッチします。これは `hermes chat`、Desktop アプリ、メッセージングプラットフォームにサービスを提供しているのと同じプロセスです。ゲートウェイが実行されていない場合は TUI が起動します。この方法で起動されたゲートウェイは、しばらくアイドル状態が続くと自ら停止します。TUI は、アタッチ時に得た使い捨てのチケットを使い、`hermes-gateway-v1` プロトコルを話すローカルの WebSocket（`/api/ws`）でゲートウェイに接続します。設定するものはありません。他のローカルプログラムも [`hermes gateway attach --json`](../reference/cli-commands.md#local-runtime-discovery-and-startup) で同じようにアタッチします。

コードベースやログで `HERMES_TUI_GATEWAY_URL` 環境変数を目にすることがあるかもしれません。これは **Web ダッシュボードの内部的な配線の詳細** であり、ユーザー向けのリモートアタッチ用のつまみではありません。ダッシュボードの「Chat」タブ（`hermes dashboard` → `/chat`）を開くと、ダッシュボードの Web サーバーが埋め込みの TUI 子プロセスを起動し、`HERMES_TUI_GATEWAY_URL` を注入して、その子プロセスがループバックの WebSocket（`/api/ws`）経由でダッシュボード自身のプロセス内 `tui_gateway` にアタッチするようにします。このエンドポイントはダッシュボードサーバー（`hermes_cli/web_server.py`）内にあり、そのプロセスのライフタイムと認証に結び付いています。

「任意の TUI を任意のゲートウェイポートに向ける」汎用モードはありません: ゲートウェイ自身の `/api/ws` は、アタッチ時の使い捨てチケットを持つループバックのクライアントしか受け入れません。特に、OpenAI 互換の API サーバー（`hermes gateway` / `api_server` プラットフォーム）は `/api/ws` を提供**しません** — それはモデルバックエンドのサーフェス（`/v1/chat/completions`、`/v1/models` など）であり、TUI の JSON-RPC 制御チャネルを意図的に公開していません。`HERMES_TUI_GATEWAY_URL` をそのポートに設定すると 404 になります。

### 接続が切れた場合

- **WebSocket が切れた場合:** TUI は *Connection to Hermes lost — reconnecting and reopening your chat…* と表示し、バックオフを伸ばしながら再接続して同じセッションに再アタッチします — ゲートウェイ上でまだストリーミング中の返信も含みます。何も再送信されません。
- **ゲートウェイがなくなった場合**（クラッシュ、アイドルによる終了、`hermes gateway stop`）: TUI はゲートウェイを再起動し（1 分あたり最大数回まで）、同じ保存済みセッションを再び開きます。ゲートウェイのクラッシュ時に処理中だった返信は、それとともに失われます。

TUI、`hermes chat`、Desktop アプリ、メッセージングプラットフォームは、すでにプロファイルのゲートウェイとそのセッションを共有しています（[Sessions](sessions.md) を参照）。手動で設定したゲートウェイ URL は何も追加しません。

## クラシック CLI への切り戻し

`hermes`（`--tui` なし）を起動すると、デフォルトではクラシック CLI のままになります。マシンが TUI を優先するようにするには、`~/.hermes/config.yaml` で `display.interface: tui` を設定する（永続的）か、シェルプロファイルで `HERMES_TUI=1` を設定します（シェルごと）。戻すには、`interface: cli` を設定する / 環境変数を解除するか、1 回だけなら `hermes --cli` を渡します。

TUI の起動に失敗した場合（Node なし、バンドル欠落、TTY の問題）、Hermes は診断情報を出力してフォールバックします。スタックしたままにはしません。

## 関連項目

- [CLI Interface](cli.md) — スラッシュコマンドとキーバインドの完全なリファレンス（共有）
- [Sessions](sessions.md) — 再開、ブランチ、履歴
- [Skins & Themes](features/skins.md) — バナー、ステータスバー、オーバーレイのテーマ設定
- [Voice Mode](features/voice-mode.md) — 両方のインターフェースで動作
- [Configuration](configuration.md) — すべての設定キー
