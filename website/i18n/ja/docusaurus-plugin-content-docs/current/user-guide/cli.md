---
sidebar_position: 1
title: "CLI インターフェース"
description: "Hermes Agent のターミナルインターフェースを使いこなす — コマンド、キーバインド、パーソナリティなど"
---

# CLI インターフェース

Hermes Agent の CLI は、Web UI ではなく本格的なターミナルユーザーインターフェース（TUI）です。複数行編集、スラッシュコマンドの自動補完、対話履歴、割り込みとリダイレクト、ツール出力のストリーミングを備えています。ターミナルで生活する人のために作られています。

:::tip
Hermes は、モーダルオーバーレイ、マウス選択、ノンブロッキング入力を備えたモダンな TUI も提供しています。`hermes --tui` で起動できます。詳しくは [TUI](tui.md) ガイドをご覧ください。
:::

## CLI の実行

```bash
# 対話型セッションを開始する（デフォルト）
hermes

# 単発クエリモード（非対話型）
hermes chat -q "Hello"

# 特定のモデルを指定
hermes chat --model "anthropic/claude-sonnet-4"

# 特定のプロバイダーを指定
hermes chat --provider nous        # Nous Portal を使う
hermes chat --provider openrouter  # OpenRouter を強制

# 特定のツールセットを指定
hermes chat --toolsets "web,terminal,skills"

# 1 つ以上のスキルをプリロードして開始
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "open a draft PR"

# 以前のセッションを再開
hermes --continue             # 直近の CLI セッションを再開（-c）
hermes --resume <session_id>  # ID で特定のセッションを再開（-r）

# 詳細モード（デバッグ出力）
hermes chat --verbose

# 独立した git ワークツリー（複数のエージェントを並行実行する場合）
hermes -w                         # ワークツリーで対話型モード
hermes -w -q "Fix issue #123"     # ワークツリーで単発クエリ
```

## インターフェースのレイアウト

<img className="docs-terminal-figure" src="/img/docs/cli-layout.svg" alt="Stylized preview of the Hermes CLI layout showing the banner, conversation area, and fixed input prompt." />
<p className="docs-figure-caption">Hermes CLI のバナー、対話ストリーム、固定された入力プロンプトを、壊れやすいテキストアートではなく安定したドキュメント図として描画したものです。</p>

ウェルカムバナーには、モデル、ターミナルバックエンド、作業ディレクトリ、利用可能なツール、インストール済みスキルが一目で表示されます。

### ステータスバー

入力エリアの上に常時表示されるステータスバーがあり、リアルタイムで更新されます。

```
 ⚕ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| 要素 | 説明 |
|---------|-------------|
| モデル名 | 現在のモデル（26 文字を超える場合は切り詰め） |
| トークン数 | 使用中のコンテキストトークン数 / 最大コンテキストウィンドウ |
| コンテキストバー | しきい値で色分けされた視覚的な充填インジケーター |
| コスト | 推定セッションコスト（不明または価格ゼロのモデルでは `n/a`） |
| 経過時間 | セッションの経過時間 |

バーはターミナル幅に適応します。76 桁以上でフルレイアウト、52〜75 桁でコンパクト、52 桁未満で最小（モデル名＋経過時間のみ）になります。

**コンテキストの色分け:**

| 色 | しきい値 | 意味 |
|-------|-----------|---------|
| 緑 | < 50% | 余裕あり |
| 黄 | 50〜80% | 埋まってきている |
| オレンジ | 80〜95% | 上限に近づいている |
| 赤 | ≥ 95% | オーバーフロー間近 — `/compress` を検討 |

入力トークンと出力トークンなどカテゴリ別のコストを含む詳細な内訳は `/usage` で確認できます。

### セッション再開の表示

以前のセッションを再開すると（`hermes -c` または `hermes --resume <id>`）、バナーと入力プロンプトの間に「Previous Conversation」パネルが表示され、対話履歴のコンパクトな要約が示されます。詳細と設定については [セッション — 再開時の対話要約](sessions.md#conversation-recap-on-resume) をご覧ください。

## キーバインド {#keybindings}

| キー | 動作 |
|-----|--------|
| `Enter` | メッセージを送信 |
| `Alt+Enter`、`Ctrl+J`、または `Shift+Enter` | 改行（複数行入力）。`Shift+Enter` には、それを `Enter` と区別できるターミナルが必要です（下記参照）。Windows Terminal では `Alt+Enter` がターミナルに捕捉される（全画面表示の切り替え）ため、代わりに `Ctrl+Enter` または `Ctrl+J` を使ってください。 |
| `Alt+V` | ターミナルが対応している場合、クリップボードから画像を貼り付け |
| `Ctrl+V` | テキストを貼り付け、可能であればクリップボードの画像も添付 |
| `Ctrl+B` | ボイスモードが有効な場合、音声録音の開始/停止（`voice.record_key`、デフォルト: `ctrl+b`） |
| `Ctrl+G` | 現在の入力バッファを `$EDITOR`（vim/nvim/nano/VS Code など）で開く。保存して終了すると、編集したテキストが次のプロンプトとして送信されます — 長い複数段落のプロンプトに最適です。 |
| `Ctrl+X Ctrl+E` | 外部エディタ用の Emacs スタイルの代替バインド（`Ctrl+G` と同じ動作）。 |
| `Ctrl+C` | エージェントを割り込み（2 秒以内に 2 回押すと強制終了） |
| `Ctrl+D` | 終了 |
| `Ctrl+Z` | Hermes をバックグラウンドにサスペンド（Unix のみ）。シェルで `fg` を実行すると再開します。 |
| `Tab` | 自動サジェスト（ゴーストテキスト）を確定、またはスラッシュコマンドを自動補完 |

**複数行貼り付けのプレビュー。** 複数行のブロックを貼り付けると、CLI はペイロード全体をスクロールバックに流し込む代わりに、コンパクトな単一行プレビュー（`[pasted: 47 lines, 1,842 chars — press Enter to send]`）を表示します。送信される内容はあくまで全文であり、これは表示上の工夫にすぎません。

**最終応答での Markdown 除去。** CLI は、エージェントの*最終*応答から最も冗長な Markdown フェンスや `**bold**` / `*italic*` のラッパーを取り除き、生のソースではなく読みやすいターミナルの文章として描画します。コードブロックとリストは保持されます。これはゲートウェイプラットフォームやツール結果には影響しません — それらはネイティブ描画のために Markdown を保持します。

## スラッシュコマンド

`/` を入力すると自動補完のドロップダウンが表示されます。Hermes は、多数の CLI スラッシュコマンド、動的なスキルコマンド、ユーザー定義のクイックコマンドに対応しています。

よく使う例:

| コマンド | 説明 |
|---------|-------------|
| `/help` | コマンドのヘルプを表示 |
| `/model` | 現在のモデルを表示または変更 |
| `/tools` | 現在利用可能なツールを一覧表示 |
| `/skills browse` | スキルハブと公式のオプションスキルを閲覧 |
| `/background <prompt>` | プロンプトを別のバックグラウンドセッションで実行 |
| `/skin` | アクティブな CLI スキンを表示または切り替え |
| `/voice on` | CLI ボイスモードを有効化（`Ctrl+B` で録音） |
| `/voice tts` | Hermes の応答の音声再生を切り替え |
| `/reasoning high` | 推論の労力を増やす |
| `/title My Session` | 現在のセッションに名前を付ける |

組み込み CLI とメッセージングの完全な一覧については、[スラッシュコマンドリファレンス](../reference/slash-commands.md) をご覧ください。

セットアップ、プロバイダー、無音調整、メッセージング/Discord でのボイス利用については、[ボイスモード](features/voice-mode.md) をご覧ください。

:::tip
コマンドは大文字小文字を区別しません — `/HELP` は `/help` と同じように動作します。インストール済みのスキルも自動的にスラッシュコマンドになります。
:::

## クイックコマンド

LLM を呼び出さずにシェルコマンドを即座に実行するカスタムコマンドを定義できます。これらは CLI とメッセージングプラットフォーム（Telegram、Discord など）の両方で動作します。

```yaml
# ~/.hermes/config.yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

その後、任意のチャットで `/status`、`/gpu`、`/restart` と入力します。さらに多くの例については、[設定ガイド](/docs/user-guide/configuration#quick-commands) をご覧ください。

## 起動時にスキルをプリロードする

セッションで有効にしたいスキルがすでに分かっている場合は、起動時に渡せます。

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

Hermes は、最初のターンの前に、指定された各スキルをセッションのプロンプトに読み込みます。同じフラグは対話型モードと単発クエリモードの両方で機能します。

## スキルのスラッシュコマンド

`~/.hermes/skills/` にインストールされたすべてのスキルは、自動的にスラッシュコマンドとして登録されます。スキル名がそのままコマンドになります。

```
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# スキル名だけを入力すると、それを読み込み、何が必要かエージェントに尋ねさせます:
/excalidraw
```

## パーソナリティ

定義済みのパーソナリティを設定して、エージェントの口調を変更できます。

```
/personality pirate
/personality kawaii
/personality concise
```

組み込みのパーソナリティには次のものがあります: `helpful`、`concise`、`technical`、`creative`、`teacher`、`kawaii`、`catgirl`、`pirate`、`shakespeare`、`surfer`、`noir`、`uwu`、`philosopher`、`hype`。

`~/.hermes/config.yaml` でカスタムパーソナリティを定義することもできます。

```yaml
personalities:
  helpful: "You are a helpful, friendly AI assistant."
  kawaii: "You are a kawaii assistant! Use cute expressions..."
  pirate: "Arrr! Ye be talkin' to Captain Hermes..."
  # Add your own!
```

## 複数行入力

複数行のメッセージを入力する方法は 2 つあります。

1. **`Alt+Enter`、`Ctrl+J`、または `Shift+Enter`** — 改行を挿入します
2. **バックスラッシュ継続** — 行末を `\` で終わらせて続けます:

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

:::info
複数行テキストの貼り付けに対応しています — 上記のいずれかの改行キーを使うか、単純に内容を直接貼り付けてください。
:::

### Shift+Enter の互換性

ほとんどのターミナルはデフォルトで `Enter` と `Shift+Enter` に同じバイトシーケンスを送信するため、アプリケーションはこれらを区別できません。Hermes が `Shift+Enter` を認識するのは、ターミナルが [Kitty キーボードプロトコル](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) または xterm の `modifyOtherKeys` モードを介して固有のシーケンスを送信する場合のみです。

| ターミナル | 状況 |
|---|---|
| Kitty、foot、WezTerm、Ghostty | 固有の `Shift+Enter` がデフォルトで有効 |
| iTerm2（最近のバージョン）、Alacritty、VS Code ターミナル、Warp | 設定で Kitty プロトコルを有効にすると対応 |
| Windows Terminal Preview 1.25+ | 設定で Kitty プロトコルを有効にすると対応 |
| macOS の Terminal.app、標準の Windows Terminal（安定版） | 非対応 — `Shift+Enter` は `Enter` と区別できません |

ターミナルがこれらを区別できない場合でも、`Alt+Enter` と `Ctrl+J` はどこでも引き続き機能します。**特に Windows Terminal では、`Alt+Enter` がターミナルに捕捉され（全画面表示の切り替え）Hermes には届かないため、改行には `Ctrl+Enter`（`Ctrl+J` として送られます）または `Ctrl+J` を直接使ってください。**

## エージェントの割り込み

エージェントはいつでも割り込めます。

- エージェントの作業中に **新しいメッセージを入力して Enter** — 割り込んで新しい指示を処理します
- **`Ctrl+C`** — 現在の操作を割り込み（2 秒以内に 2 回押すと強制終了）
- 進行中のターミナルコマンドは即座に終了されます（SIGTERM、その後 1 秒で SIGKILL）
- 割り込み中に入力された複数のメッセージは 1 つのプロンプトにまとめられます

### ビジー入力モード

`display.busy_input_mode` の設定キーは、エージェントの作業中に Enter を押したときの動作を制御します。

| モード | 動作 |
|------|----------|
| `"interrupt"`（デフォルト） | メッセージが現在の操作を割り込み、即座に処理されます |
| `"queue"` | メッセージは静かにキューに入れられ、エージェントの完了後に次のターンとして送信されます |
| `"steer"` | メッセージは `/steer` 経由で現在の実行に注入され、次のツール呼び出しの後にエージェントへ届きます — 割り込みも新しいターンもありません |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # or "queue" or "interrupt" (default)
```

`"queue"` モードは、実行中の作業を誤ってキャンセルすることなくフォローアップのメッセージを準備したい場合に便利です。`"steer"` モードは、割り込まずにタスクの途中でエージェントの方向を変えたい場合に便利です — 例えばコードの編集中に「やっぱりテストも確認して」と伝えるような場合です。不明な値は `"interrupt"` にフォールバックします。

`"steer"` には自動フォールバックが 2 つあります。エージェントがまだ開始していない場合、または画像が添付されている場合、メッセージは `"queue"` の動作にフォールバックし、何も失われないようにします。

CLI 内で変更することもできます。

```text
/busy queue
/busy steer
/busy interrupt
/busy status
```

:::tip 初回ヒント
Hermes の作業中に初めて Enter を押すと、Hermes は `/busy` の設定項目を説明する 1 行のリマインダー（`"(tip) Your message interrupted the current run…"`）を表示します。これはインストールごとに 1 回だけ表示されます — `config.yaml` の `onboarding.seen.busy_input_prompt` のフラグがこれを記録します。このキーを削除すると、ヒントを再度表示できます。
:::

### バックグラウンドへのサスペンド

Unix システムでは、**`Ctrl+Z`** を押すと Hermes をバックグラウンドにサスペンドできます — 他のターミナルプロセスと同様です。シェルは確認メッセージを表示します。

```
Hermes Agent has been suspended. Run `fg` to bring Hermes Agent back.
```

シェルで `fg` と入力すると、中断したところから正確にセッションを再開します。これは Windows では非対応です。

## ツール進行状況の表示

CLI は、エージェントの作業に合わせてアニメーション付きのフィードバックを表示します。

**思考アニメーション**（API 呼び出し中）:
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

**ツール実行フィード:**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

`/verbose` で表示モードを切り替えます: `off → new → all → verbose`。このコマンドはメッセージングプラットフォーム向けにも有効化できます — [設定](/docs/user-guide/configuration#display-settings) をご覧ください。

### ツールプレビューの長さ

`display.tool_preview_length` の設定キーは、ツール呼び出しのプレビュー行（ファイルパス、ターミナルコマンドなど）に表示される最大文字数を制御します。デフォルトは `0` で、制限なしを意味します — 完全なパスとコマンドが表示されます。

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # ツールプレビューを 80 文字に切り詰め（0 = 制限なし）
```

これは、幅の狭いターミナルや、ツールの引数に非常に長いファイルパスが含まれる場合に便利です。

## セッション管理

### セッションの再開

CLI セッションを終了すると、再開用のコマンドが表示されます。

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

再開のオプション:

```bash
hermes --continue                          # 直近の CLI セッションを再開
hermes -c                                  # 短縮形
hermes -c "my project"                     # 名前付きセッションを再開（系統内の最新）
hermes --resume 20260225_143052_a1b2c3     # ID で特定のセッションを再開
hermes --resume "refactoring auth"         # タイトルで再開
hermes -r 20260225_143052_a1b2c3           # 短縮形
```

再開すると、SQLite から完全な対話履歴が復元されます。エージェントは、以前のすべてのメッセージ、ツール呼び出し、応答を参照できます — まるで一度も離れなかったかのようにです。

チャット内で `/title My Session Name` を使って現在のセッションに名前を付けるか、コマンドラインから `hermes sessions rename <id> <title>` を使います。過去のセッションを閲覧するには `hermes sessions list` を使います。

### セッションの保存

CLI セッションは、Hermes の SQLite 状態データベース `~/.hermes/state.db` に保存されます。このデータベースは次のものを保持します。

- セッションメタデータ（ID、タイトル、タイムスタンプ、トークンカウンター）
- メッセージ履歴
- 圧縮/再開されたセッションをまたぐ系統
- `session_search` が使用する全文検索インデックス

一部のメッセージングアダプターは、データベースとは別にプラットフォームごとのトランスクリプトファイルも保持しますが、CLI 自体は SQLite のセッションストアから再開します。

### コンテキスト圧縮

長い対話は、コンテキスト上限に近づくと自動的に要約されます。

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.50    # デフォルトではコンテキスト上限の 50% で圧縮

# 要約モデルは auxiliary 配下で設定:
auxiliary:
  compression:
    model: ""  # 空のままにするとメインのチャットモデルを使用（デフォルト）。または安価で高速なモデルを固定指定、例: "google/gemini-3-flash-preview"。
```

圧縮が発動すると、中間のターンが要約され、最初の 3 ターンと最後の 20 ターンは常に保持されます。

## バックグラウンドセッション

他の作業のために CLI を使い続けながら、別のバックグラウンドセッションでプロンプトを実行できます。

```
/background Analyze the logs in /var/log and summarize any errors from today
```

Hermes はすぐにタスクを確認し、プロンプトを返します。

```
🔄 Background task #1 started: "Analyze the logs in /var/log and summarize..."
   Task ID: bg_143022_a1b2c3
```

### 仕組み

各 `/background` プロンプトは、デーモンスレッド内に **完全に独立したエージェントセッション** を生成します。

- **独立した対話** — バックグラウンドエージェントは、現在のセッションの履歴を一切認識しません。提供したプロンプトのみを受け取ります。
- **同じ設定** — バックグラウンドエージェントは、現在のセッションからモデル、プロバイダー、ツールセット、推論設定、フォールバックモデルを引き継ぎます。
- **ノンブロッキング** — フォアグラウンドのセッションは完全に対話可能なままです。チャットしたり、コマンドを実行したり、さらにバックグラウンドタスクを開始したりできます。
- **複数タスク** — 複数のバックグラウンドタスクを同時に実行できます。それぞれに番号付きの ID が割り当てられます。

### 結果

バックグラウンドタスクが完了すると、結果がターミナルにパネルとして表示されます。

```
╭─ ⚕ Hermes (background #1) ──────────────────────────────────╮
│ Found 3 errors in syslog from today:                         │
│ 1. OOM killer invoked at 03:22 — killed process nginx        │
│ 2. Disk I/O error on /dev/sda1 at 07:15                      │
│ 3. Failed SSH login attempts from 192.168.1.50 at 14:30      │
╰──────────────────────────────────────────────────────────────╯
```

タスクが失敗した場合は、代わりにエラー通知が表示されます。設定で `display.bell_on_complete` が有効になっている場合、タスクの完了時にターミナルのベルが鳴ります。

### ユースケース

- **長時間のリサーチ** — コードに取り組みながら「/background research the latest developments in quantum error correction」
- **ファイル処理** — 対話を続けながら「/background analyze all Python files in this repo and list any security issues」
- **並行調査** — 複数のバックグラウンドタスクを開始し、異なる角度から同時に探求する

:::info
バックグラウンドセッションはメインの対話履歴には表示されません。独自のタスク ID（例: `bg_143022_a1b2c3`）を持つスタンドアロンのセッションです。
:::

## クワイエットモード

デフォルトでは、CLI はクワイエットモードで動作し、次のことを行います。
- ツールからの冗長なログを抑制する
- kawaii スタイルのアニメーションフィードバックを有効にする
- 出力をクリーンでユーザーフレンドリーに保つ

デバッグ出力を得るには:
```bash
hermes chat --verbose
```
