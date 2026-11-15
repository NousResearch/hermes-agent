---
title: "Windows（ネイティブ）ガイド"
description: "Hermes AgentをWindows 10 / 11でネイティブに実行する — インストール、機能マトリクス、UTF-8コンソール、Git Bash、Scheduled Taskとしてのゲートウェイ、エディタの扱い、PATH、アンインストール、よくある落とし穴"
sidebar_label: "Windows（ネイティブ）"
sidebar_position: 3
---

# Windows（ネイティブ）ガイド

HermesはWindows 10とWindows 11でネイティブに動作します — WSLなし、Cygwinなし、Dockerなし。このページは詳細解説です：何がネイティブで動くか、何がWSL専用か、インストーラーが実際に何をするか、そして触る必要があるかもしれないWindows固有のつまみについてです。

ただインストールしたいだけなら、[ランディングページ](../index.mdx)または[インストールページ](../getting-started/installation#windows-native)のワンライナーだけで十分です。何かに驚いたら、ここに戻ってきてください。

:::tip 代わりにWSLが欲しいですか？
`fork` のセマンティクスやLinux式のファイルウォッチャーのためにPOSIX環境を好む場合は、**[Windows（WSL2）ガイド](./windows-wsl-quickstart.md)**を参照してください。両者はきれいに共存します：ネイティブのデータは `%LOCALAPPDATA%\hermes` 配下に、WSLのデータは `~/.hermes` 配下に置かれます。
:::

## クイックインストール

**PowerShell**（またはWindows Terminal）を開いて実行します：

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

管理者権限は不要です。インストーラーは `%LOCALAPPDATA%\hermes\` に展開し、`hermes` をあなたの**ユーザーPATH**に追加します — 完了後は新しいターミナルを開いてください。

**インストーラーオプション**はscriptblockを使います：

```powershell
& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -NonInteractive -Channel main
```

| パラメータ | 目的 |
|---|---|
| `-Channel stable\|main` | `stable`（デフォルト）：最新の公開リリース。`main`：すべてのコミット。`hermes update --set-channel main` と同様に保存されます。素の `irm \| iex` では `$env:HERMES_INSTALL_CHANNEL` で設定します。 |
| `-Branch NAME` | ソースブランチを選択します。デフォルトは `main`。それ以外のブランチはその先端でインストールされます。 |
| `-Commit SHA` | ブランチのチェックアウト後にコミットを選択します。 |
| `-HermesHome PATH` | データディレクトリを選択します。 |
| `-InstallDir PATH` | ソースチェックアウトのディレクトリを選択します。 |
| `-NonInteractive` | 入力が必要なセットアップとゲートウェイのステージをスキップします。 |
| `-IncludeDesktop` | デスクトップアプリをビルドし、ショートカットを作成します。 |
| `-ShowResolvedPaths` | インストールせずに、解決されたパスをJSONで出力します。 |
| `-Verbose` | ステップごとに1行のステータスではなく、すべての子コマンドの出力をストリーミングします。 |
| `-Manifest` / `-ProtocolVersion` | ブートストラップGUIが使うステージプロトコルを確認します。 |
| `-Stage NAME -Json` | 1つのステージを実行し、その結果フレームを出力します。 |

現在のスクリプトは `-NoVenv` や `-Tag` を受け付けません。`-SkipSetup` は、古いインストールラッパーが引き続きバインドできるよう、`-NonInteractive` の非推奨エイリアスとして今も受け付けられます。
予期しない短いWindowsパスを診断するには、まず `-ShowResolvedPaths` を使ってください。

### MSIX / App InstallerとMicrosoft Store

同梱のデスクトップはソーススクリプトとは別物です。そのMSIXパッケージには**Windows 11 22H2以降**が必要です。ソーススクリプトがWindows 10をサポートしていても、MSIXパッケージがWindows 10をサポートしているわけではありません。

ダウンロードした `.appinstaller` ファイルをWindows App Installerで開きます。署名済みのユニバーサルバンドルがインストールされ、更新元が記録されます。パッケージにはPython、Node、サポートされる依存関係、ビルド済みのインターフェースが含まれます。初回起動時にチェックアウトをクローンしたり、ベースランタイムをビルドしたりすることはありません。

MSIXの実行エイリアスは `hermes`、`hermes-agent`、`hermes-acp` を公開します。別のインストールがエイリアスを隠している場合は、`Get-Command hermes -All` で確認してください。エイリアスはWindowsの設定 → アプリ → アプリの詳細設定 → アプリ実行エイリアスで制御します。

サイドロードの更新には、アプリのUpdateコントロールとWindows App Installerを使います。Hermesは終了処理の前にローカルの記述子をダウンロードし、自動再起動を登録します。`ms-appinstaller:` URLプロトコルは必要ありません。更新チェックの結果が不明であっても、パッケージが最新であることを意味するわけではありません。

Microsoft Store版は、Partner Centerのパッケージ IDとStoreの更新を使用します。サイドロードのフィードは使用しません。どちらの同梱ランタイム内でも、`hermes update` がパッケージファイルに対してGitを実行することはありません。

`Hermes-Setup.exe` は別の、ブートストラップ用インストーラーです。スクリプトを通じてソースチェックアウトを用意します。自己完結型のMSIXパッケージと混同しないでください。[アップデートとアンインストール](../getting-started/updating.md)を参照してください。

### 依存関係のブートストラップ

管理対象のツールはPMが所有します。機能コードはインストーラーを再実行するのではなく、必要なパッケージをPMに要求します（`pm.ensure("<package>")`、例えばComputer Useなら `cua-driver`）。すでにインストールされたツールはPMが記録した状態から再利用されます。欠けているオプションのツールは、[`security.allow_lazy_installs`](/docs/reference/package-management#lazy-install-policy) が許可している場合にのみオンデマンドで取得されます。`install.ps1` には `-Ensure` モードはありません。

```powershell
hermes pm doctor
hermes pm install
```

## ソースインストーラーが行うこと

1. Gitを見つけるか、Gitがない場合は検証済みのGit for Windowsのピン留め版を配置します。
2. 選択したリポジトリのブランチをクローンし、オプションのコミットのピン留めを適用します。
3. uvをブートストラップし、初期のPython環境を作成します。
4. PMを実行して、Python 3.14、必要なツール、`all` Python extraを用意します。
5. データホームの `bin` ディレクトリにCLIランチャーを生成し、それをユーザーPATHに追加します。
6. 設定を準備し、スキップされていなければ対話型のセットアップ/ゲートウェイのステージを呼び出します。
7. 要求された場合は、デスクトップをビルドし、スタートメニュー/デスクトップのショートカットを作成します。
8. ブートストラップ完了マーカーを書き込みます。

ランタイムランチャーはPMのストアのPythonを実行し、インポートの前に依存関係の環境を選択します。PMは、実行中のプロセスがすでに読み込んだライブラリを置き換えることなく、新しい書き込み可能な環境を公開できます。インストールされる機能セットを黙って縮小するような段階的なpipフォールバックはありません。

:::tip Windowsでプロバイダー探しを省く
Windowsでは、ツールごとのAPIキーのセットアップ（Firecrawl、FAL、Browser Use、OpenAI TTS）が、役に立つエージェントを得るうえで最も手間のかかる部分です。[Nous Portal](./features/tool-gateway.md)のサブスクリプションなら、1回のOAuthログインでモデル**と**それらすべてのツールをカバーできます。インストーラーが完了したら、`hermes setup --portal` を実行してすべてを接続してください。
:::

## 機能マトリクス

Windowsのサポートは機能とアーキテクチャによって異なります。基本的なインターフェースはネイティブで動作しますが、一部のオプションのSDKは特定のターゲットから除外されています。

| 機能 | ネイティブWindows | WSL2 |
|---|---|---|
| CLI（`hermes chat`、`hermes setup`、`hermes gateway`、…） | ✓ | ✓ |
| 対話型TUI（`hermes --tui`） | ✓ | ✓ |
| メッセージングゲートウェイ（Telegram、Discord、Slack、WhatsApp、15以上のプラットフォーム） | ✓ | ✓ |
| Cronスケジューラー | ✓ | ✓ |
| ブラウザツール（Node経由のChromium） | ✓ | ✓ |
| MCPサーバー（stdioとHTTP） | ✓ | ✓ |
| ローカルOllama / LM Studio / llama-server | ✓ | ✓（WSLネットワーキング経由） |
| Webダッシュボード（セッション、ジョブ、メトリクス、設定） | ✓ | ✓ |
| ダッシュボードの `/chat` 組み込みターミナルペイン | `pywinpty` 経由のConPTY | POSIX PTY |
| ログイン時の自動起動 | ✓（schtasks） | ✓（systemd） |

ダッシュボードは、Windowsでは `pywinpty`/ConPTYブリッジを、POSIXでは `ptyprocess` を使います。ネイティブの依存関係が欠けていたり壊れていたりすると、ターミナルが利用できなくなることがあります。WSLは代替手段であり、現在の設計で必須というわけではありません。

### オプションの依存関係の制限

- Matrixのネイティブ暗号化アダプターはLinux専用です。Windowsでは、サポートされているプロキシ経路かLinuxバックエンドを使ってください。
- ネイティブWindows ARM64では、`google-chat` SDK extraとopenWakeWordエンジンが除外されます。SherpaはネイティブWindows ARM64をサポートしており、そのターゲットでの自動のウェイクワードのデフォルトです。
- ローカルのFaster-Whisper STTはネイティブWindows ARM64では除外されます。クラウドまたはコマンドベースのSTTプロバイダーを使ってください。Porcupineは引き続きウェイクエンジンの代替手段です。

`pyproject.toml` のプラットフォームマーカーが、パッケージ化される依存関係のセットを定義します。ゲートウェイや音声機能に関する一般的な記述が、それらのマーカーより優先されることはありません。

## HermesがWindowsでシェルコマンドを実行する方法 {#how-hermes-runs-shell-commands-on-windows}

Hermesのターミナルツールは、Claude Codeが使うのと同じ戦略で、**Git Bash** を通じてコマンドを実行します。これは、すべてのツールを書き直すことなく、POSIX対Windowsのギャップを回避します。

`pm.shell()` がBashの解決を担当します。まずPMのファクトに記録されたGitパッケージを確認し、次に用意された `PATH` を確認します。PATHの候補がWindowsAppsパッケージに属している場合、リゾルバーは利用可能であれば通常のGit for Windowsのインストールを優先します。

パッケージ化されたツールは、汎用のホストインストールではありません。外部のPythonプロセスがWindowsAppsペイロードの実行ファイルを起動しようとすると、`WinError 5` で失敗することがあります。パッケージ自身のランチャーを使うか、ソースチェックアウトでは通常のツールを使ってください。この境界を回避するためにWindowsのセキュリティ制御を無効にしないでください。

現在のインストーラーは `HERMES_GIT_BASH_PATH` を設定しません。MinGitは、Bashを含むGit for Windowsの代わりにはなりません。

## WindowsでのUTF-8コンソール

WindowsでのPythonのデフォルトstdioは、コンソールのアクティブなコードページ（通常はcp1252またはcp437）を使います。Hermesのバナー、スラッシュコマンドリスト、ツールフィード、Richパネル、スキル説明はすべてUnicodeを含みます。介入がないと、それらのいずれも `UnicodeEncodeError: 'charmap' codec can't encode character…` でクラッシュします。

修正は `hermes_cli/stdio.py::configure_windows_stdio()` にあり、すべてのエントリーポイント（`cli.py::main`、`hermes_cli/main.py::main`、`gateway/run.py::main`）の早い段階で呼ばれます。これは：

1. `kernel32.SetConsoleCP` / `SetConsoleOutputCP` を介してコンソールコードページをCP_UTF8（65001）に切り替えます。
2. `sys.stdout` / `sys.stderr` / `sys.stdin` を `errors='replace'` でUTF-8に再構成します。
3. `PYTHONIOENCODING=utf-8` と `PYTHONUTF8=1` を設定し（`setdefault` を介するため、明示的なユーザー値が優先される）、子のPythonサブプロセスがUTF-8を継承するようにします。
4. `EDITOR` も `VISUAL` も設定されていなければ `EDITOR=notepad` を設定します（下記のエディタのセクションを参照）。

冪等です。非Windowsでは何もしません。

**オプトアウト:** 環境内の `HERMES_DISABLE_WINDOWS_UTF8=1` は、従来のcp1252 stdioパスにフォールバックします。エンコーディングのバグを二分探索するのに便利ですが、通常運用では正しい設定であることはまずありません。

## エディタ（`Ctrl-X Ctrl-E`、`/edit`）

PR #21561以前は、`Ctrl-X Ctrl-E` を押したり `/edit` と入力したりしても、Windowsでは黙って何も起きませんでした。prompt_toolkitには、Windowsでは決して解決されないPOSIX絶対パスのフォールバックリスト（`/usr/bin/nano`、`/usr/bin/pico`、`/usr/bin/vi`、…）がハードコードされています — フルのGit for Windowsをインストールしていてもです。

HermesのWindows stdioシムは、デフォルトとして `EDITOR=notepad` を設定するようになりました。NotepadはすべてのWindowsインストールに同梱され、ブロッキングエディタとして機能します — `subprocess.call(["notepad", file])` はウィンドウが閉じるまでブロックします。

**ユーザーの上書きが依然として優先されます**（setdefaultの前にチェックされます）：

| エディタ | PowerShellコマンド |
|---|---|
| VS Code | `$env:EDITOR = "code --wait"` |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim | `$env:EDITOR = "nvim"` |
| Helix | `$env:EDITOR = "hx"` |

VS Codeの `--wait` フラグは重要です — これがないとエディタは即座にリターンし、Hermesは空のバッファを受け取ります。

PowerShellプロファイルで永続的に設定します：

```powershell
# In $PROFILE
$env:EDITOR = "code --wait"
```

または、システム設定のユーザー環境変数として設定すると、すべての新しいシェルがそれを拾います。

## CLIで改行するための `Ctrl+Enter`

Windows Terminalは `Ctrl+Enter` を専用のキーシーケンスとしてそのまま渡します。Hermesはこれを「改行を挿入」にバインドしているため、`Esc` から `Enter` にフォールバックすることなく、CLIで複数行のプロンプトを作成できます。Windows Terminal、VS Code統合ターミナル、およびVTエスケープシーケンスを尊重するあらゆるモダンなWindowsコンソールホストで動作します。

レガシーの `cmd.exe` コンソールでは、`Ctrl+Enter` は単なる `Enter` に縮退します — 代わりに `Esc Enter` を使うか、Windows Terminalにアップグレードしてください（無料で、Windows 11にはデフォルトでインストールされています）。

## Windowsログイン時にゲートウェイを実行する

Windowsでの `hermes gateway install` は、Startupフォルダーへのフォールバックを伴う**Scheduled Tasks**を使います — 管理者不要です。

### インストール

```powershell
hermes gateway install
```

裏で何が起きるか：

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN Hermes_Gateway` — ログイン時に標準（非昇格）権限で実行されるタスクを登録します。UACプロンプトはありません。
2. グループポリシーでschtasksがブロックされている場合、小さな `Hermes_Gateway.vbs` ランチャー（`wscript.exe` 経由で非表示で実行）を `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup` に書き込むことにフォールバックします。同じ効果で、少し粗削りです。`cmd.exe` のショートカットではなくVBScriptを使うのは、ログオン時に割り当てられたコンソールがcloseイベントを受け取り、ゲートウェイが起動を終える前に殺されてしまうことがあるためです。
   保持されるのは常にどちらか一方だけです。タスクのインストールに成功するとStartupフォルダーのエントリー（従来の `Hermes_Gateway.cmd` を含む）は削除され、タスクがまだ登録されている間はフォールバックはスキップされます。また、両方を持つ古いインストールは `hermes update` / `hermes doctor --fix` で整理されます。両方があるとログオン時にゲートウェイが2回起動されてしまうためです。
3. ゲートウェイを **`pythonw.exe` 経由でデタッチして**起動します — `python.exe` ではありません。`pythonw.exe` にはコンソールが接続されていないため、兄弟プロセスからの `CTRL_C_EVENT` ブロードキャストに対して免疫があります（同じプロセスグループで何かをCtrl+Cしたときにゲートウェイを殺していた実際の問題でした）。

起動時に使われるフラグ：`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`。

### 管理

```powershell
hermes gateway status      # Merged view: schtasks + Startup folder + running PID
hermes gateway start       # Starts the gateway in the background (asks about login auto-start only on a TTY when nothing is installed)
hermes gateway stop        # Writes the planned-stop marker, waits for the gateway to drain (≤ agent.restart_drain_timeout, capped at 30 s), then force-kills only if it is still alive
hermes gateway restart     # Same drain-first stop, then a fresh start
hermes gateway uninstall   # Removes schtasks entry, Startup shortcut, pid file
```

`hermes gateway status` は冪等です — 1000回連続で呼んでも、誤ってゲートウェイを殺すことは決してありません。（PR #21561以前は、`os.kill(pid, 0)` がCレベルで `CTRL_C_EVENT` と衝突して、黙って殺していました — その経緯を知りたい場合は下記の「プロセス管理の内部」を参照してください。）

ログイン時の自動起動がインストールされるのは、明示的な回答があった場合のみです：`hermes gateway install`、実際のターミナルでの `Y`、または `HERMES_GATEWAY_INSTALL_START_ON_LOGIN=1`。スクリプトやパイプ経由の `hermes gateway start`（TTYなし、または `HERMES_NONINTERACTIVE=1`）は、Scheduled TaskやStartupフォルダーに触れずにゲートウェイを開始します。ターミナルでも質問をスキップするには `HERMES_GATEWAY_INSTALL_START_ON_LOGIN=0` を設定してください。

### なぜWindowsサービスではないのか？

サービスはインストールに管理者権限を必要とし、ゲートウェイのライフサイクルをユーザーログインではなくマシンのブートに結びつけます。典型的なHermesユーザーが望むのは：ログイン → ゲートウェイが利用可能、ログアウト → ゲートウェイが消える、です。Scheduled Tasksは昇格なしでまさにこれを実現します。本当にサービスが欲しい場合は、`nssm` や `sc create` を手動で使ってください — しかしおそらく必要ないでしょう。使う場合は、`Hermes*` という名前にするか、そのバイナリパスをHermesのインストール内（`venv\Scripts\hermes.exe`、チェックアウト、または `gateway-service\`）に向けてください。`hermes update` は、Service Control Managerを通じてHermesが所有していると確実に識別できるサービスのみを停止・再起動し、Scheduled Taskから起動されたゲートウェイはPIDで一時停止します（Task Scheduler自体には一切触れません）。

## データレイアウト

| パス | 内容 |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | スクリプトインストール用のソースチェックアウト。MSIXのみのインストールには存在しません。 |
| `%LOCALAPPDATA%\hermes\tools\` | 書き込み可能な管理ツールストア。MSIXのベースツールはパッケージ内に残ります。 |
| `%LOCALAPPDATA%\hermes\installs\` | インストールごとのランタイム選択、ジャーナル、Pythonの世代。 |
| `%LOCALAPPDATA%\hermes\bin\` | ソースインストールのCLIランチャー。MSIXでは代わりに実行エイリアスが提供されます。 |
| `%LOCALAPPDATA%\hermes\` | ユーザーの設定、認証情報、セッション、プラグイン、スキル、ログ。 |

これらはデフォルトのパスです。`HERMES_HOME` とインストーラーのパス引数で変更できます。`%LOCALAPPDATA%\hermes` を丸ごと削除するとユーザーデータも削除され、それを共有する他のインストールにも影響する可能性があります。アプリを修復するためにそのルートを削除するのではなく、アンインストールコマンドかWindowsのパッケージ削除を使ってください。

## ブラウザツール

ブラウザツール（`browser_exec`）はHermesのPython環境の一部です。ローカルでのブラウジングでは、HermesはPMがインストールしたピン留め版のChromiumを起動します（欠けている場合は `hermes pm install chromium`）。別のChromium系のバイナリを使うには `browser.executable_path` を設定してください。クラウドブラウザは `hermes tools` で選択します。自己完結型のMSIXには、ペイロードにブラウザツールが含まれています。

Windows ARM64では、ピン留め版のChromiumはx64ビルドで、Windowsのx64エミュレーション上で動作します。これはネイティブARM64のPythonランタイムとは異なります。ブラウザのソースについては[ブラウザ自動化](./features/browser.md)を参照してください。

## WindowsでHermesを実行する — 実用上のメモ

### インストール後のPATH

インストーラーは `[Environment]::SetEnvironmentVariable` 経由で `%LOCALAPPDATA%\hermes\bin` をあなたの**ユーザーPATH**に追加します。既存のターミナルはこれを拾いません — インストール後に新しいPowerShellウィンドウ（またはWindows Terminalタブ）を開いてください。何をしているか分かっている場合を除き、手動で `$env:PATH += …` するのではなく、閉じて再度開いてください。

確認します：

```powershell
Get-Command hermes        # should print C:\Users\<you>\AppData\Local\hermes\bin\hermes.exe
hermes --version
```

### 環境変数

Hermesは `$env:X`（プロセススコープ）とユーザー環境変数（永続的、システムのプロパティ → 環境変数で設定）の両方を尊重します。`%LOCALAPPDATA%\hermes\.env`（あなたの `HERMES_HOME`）にAPIキーを設定するのが通常の方法です — Linuxと同じです：

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

すべてのWindowsプロセスにシークレットを見せたいのでない限り（それは望ましくありません）、ユーザー環境変数にシークレットを置かないでください。

### Windows固有の環境変数

これらはネイティブWindowsインストールにのみ影響します：

| 変数 | 効果 |
|---|---|
| `HERMES_DISABLE_WINDOWS_UTF8` | `1` に設定するとUTF-8 stdioシムを無効化し、ロケールのコードページにフォールバックする。エンコーディングのバグの二分探索に便利。 |
| `EDITOR` / `VISUAL` | `/edit` と `Ctrl-X Ctrl-E` 用のエディタ。両方が未設定の場合、Hermesはデフォルトで `notepad` を使う。 |

## アンインストール

PowerShellから：

```powershell
hermes uninstall
```

ソースインストールの場合、アンインストーラーは所有するランチャー、サービスエントリー、アプリケーションファイルを削除します。削除する前に `hermes uninstall --dry-run` を確認してください。`--full` はデータも削除し、`--data` はパッケージ化されたコードを削除せずにデータを削除します。MSIXまたはStoreのインストールの場合は、Windowsの設定 → アプリ → インストールされているアプリからアプリを削除してください。CLIはパッケージが所有するコードの削除を拒否します。

:::caution ユーザーデータの削除
データを削除する前に、選択した `HERMES_HOME` を使用しているすべてのHermesプロセスを停止し、バックアップを作成してください。
データ削除モードを選ぶ前に `hermes uninstall --dry-run` を確認してください。
1つのアプリケーションやプロファイルを修復するために、デフォルトのデータルートを再帰的に削除しないでください。
カスタムの `HERMES_HOME` は別の場所にある場合があり、パッケージを削除してもそのデータは削除されません。
:::

`hermes uninstall` CLIサブコマンドは、schtasksエントリーが別のタスク名で登録されていた場合（古いインストール）も処理します — ハードコードされたタスク名ではなく、インストールパスで検索します。

## プロセス管理の内部

これは背景資料です — 「自分自身を殺している」奇妙な挙動をデバッグしている場合を除き、読み飛ばしてください。

LinuxとmacOSでは、POSIXのイディオム `os.kill(pid, 0)` は何もしない権限チェックです：「このPIDは生きていて、シグナルを送れるか？」。Windowsでは、Pythonの `os.kill` は `sig=0` を `CTRL_C_EVENT` にマップし — それらは整数値0で衝突します — `GenerateConsoleCtrlEvent(0, pid)` を通じてルーティングし、対象PIDを含む**コンソールプロセスグループ全体**にCtrl+Cをブロードキャストします。これが[bpo-14484](https://bugs.python.org/issue14484)で、2012年から未解決です。現在の挙動に依存するスクリプトを壊すことになるため、修正されません。

結果：Windowsで `os.kill(pid, 0)` 経由で「このPIDが生きているか確認する」と言っていたあらゆるコードパスは、黙って対象を殺していました。Hermesはそのような箇所すべて（11ファイルにわたる14箇所）を `gateway.status._pid_exists()` に移行しました。これは `psutil.pid_exists()` を使います（さらにこれはWindowsで `OpenProcess + GetExitCodeProcess` を使う — シグナルなし）。プラグインやパッチを書く場合は、`psutil.pid_exists()` を直接、または `gateway.status._pid_exists()` を使ってください — 決して `os.kill(pid, 0)` を使わないでください。

`scripts/check-windows-footguns.py` がCIでこれを強制します：新しい `os.kill(pid, 0)` 呼び出しは、その行に `# windows-footgun: ok — <reason>` マーカーが付いていない限り、`Windows footguns (blocking)` チェックを失敗させます。

## よくある落とし穴

**インストール直後の `hermes: command not found`。**
新しいPowerShellウィンドウを開いてください。インストーラーは `%LOCALAPPDATA%\hermes\bin` をユーザーPATHに追加しましたが、既存のシェルがそれを拾うには再起動が必要です。それまでの間は `& "$env:LOCALAPPDATA\hermes\bin\hermes.exe"` を実行できます。

**ツール実行時の `WinError 193: %1 is not a valid Win32 application`。**
`.cmd` シムをバイパスしたshebangスクリプトの呼び出しに遭遇しました。Hermesは `shutil.which(cmd, path=local_bin)` 経由でコマンドを解決するためPATHEXTが `.CMD` を拾います — 代わりにハードコードされたパス経由でツールを呼び出している場合は、`.cmd` バリアントに切り替えてください（例 `npx` ではなく `npx.cmd`）。

**`[scriptblock]::Create(...)` が `The assignment expression is not valid` で失敗する。**
ダウンロードした `install.ps1` がUTF-8 BOMを拾いました。`irm | iex` 形式はBOMを自動的に取り除きますが、`[scriptblock]::Create((irm ...))` はそうしません。シンプルな `irm | iex` 形式で再実行するか、スクリプトを手動でダウンロードし、`[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))` 経由でBOMなしで保存してください。

**再起動後にゲートウェイが実行され続けない。**
`hermes gateway status` を確認してください — schtasksエントリー、Startupフォルダーのショートカット（使われている場合）、ライブのPIDをマージします。schtasksが登録されているのに実行されていない場合、グループポリシーが `ONLOGON` トリガーをブロックしている可能性があります。`schtasks /Query /TN Hermes_Gateway /V /FO LIST`（名前付きプロファイルの場合は `Hermes_Gateway_<profile>`）を実行してタスクの失敗理由を確認してください。Startupフォルダーへのフォールバックが自動的に使われるのは、`schtasks` 自体がタスクの登録に失敗した場合のみです。それを強制する環境変数やフラグはありません。

**`$env:EDITOR` を設定した後も `/edit` が何もしない。**
現在のプロセスにのみ設定しました。シェルを閉じて再度開くか、システムのプロパティ → 環境変数でユーザースコープに設定してください。新しいPowerShellウィンドウで `echo $env:EDITOR` で確認します。

**ブラウザツールは起動するがツールがタイムアウトする。**
`hermes doctor` と `hermes pm doctor` を実行してください。選択されたブラウザプロバイダーを確認するには `hermes tools` を使ってください。初回のChromiumの起動が遅い場合は、`browser.command_timeout`（起動タイムアウトの下限、秒単位）で時間を延ばせます。署名済みのアプリペイロードに無関係なPlaywrightのリビジョンをインストールしないでください。

**中国語 / 日本語 / アラビア語の文字がCLIで `?` として表示される。**
UTF-8 stdioシムが有効化されませんでした。`HERMES_DISABLE_WINDOWS_UTF8` が設定されていないことを確認してください（`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`）。空なのにまだ `?` が見える場合、コンソールホスト（非常に古い `cmd.exe`）がUTF-8をまったくサポートしていない可能性があります — Windows Terminalに切り替えてください。

**ゲートウェイがTelegramの写真を送れない — 「`BadRequest: payload contains invalid characters`」。**
これはWindowsとは無関係ですが、ここで最初に表面化することがあります。通常、JSONボディ内のファイルパスにエスケープされていないバックスラッシュが含まれていることを意味します。Telegramは、生のWindowsパスではなく、Hermesが正規化したパスを受け取るはずです — カスタムプラグイン内でこれが見られる場合は、ユーザー入力からの `str(Path(...))` ではなく、Hermesが提供するパスを渡していることを確認してください。

**`git pull` 後の「自分の別のマシンでは動く」エンコーディングの奇妙さ。**
Windowsで非UTF-8エディタ（古いWindowsバージョンのNotepad、一部の中国語IME）を使ってHermesの設定やスキルを編集した場合、ファイルがBOM付きで保存された可能性があります。Hermesはほとんどの設定読み込みで `utf-8-sig` を許容しますが、折りたたまれたYAMLスカラー（`description: >`）内のBOMは黙ってYAMLのパースを壊します。ファイルをBOMなしのプレーンUTF-8として保存し直してください。

## 次に行く場所

- **[インストール](../getting-started/installation.md)** — Linux/macOS/WSL2を含む、完全なインストールページ。
- **[Windows（WSL2）ガイド](./windows-wsl-quickstart.md)** — POSIXセマンティクスやダッシュボードのターミナルペインが欲しい場合。
- **[CLIリファレンス](../reference/cli-commands.md)** — すべての `hermes` サブコマンド。
- **[FAQ](../reference/faq.md)** — Windows固有でない一般的な質問。
- **[メッセージングゲートウェイ](./messaging/index.md)** — WindowsでTelegram/Discord/Slackを実行する。
