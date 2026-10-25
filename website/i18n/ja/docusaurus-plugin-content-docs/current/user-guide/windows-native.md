---
title: "Windows（ネイティブ）ガイド — 早期ベータ"
description: "早期ベータ: Hermes AgentをWindows 10 / 11でネイティブに実行する — インストール、機能マトリクス、UTF-8コンソール、Git Bash、Scheduled Taskとしてのゲートウェイ、エディタの扱い、PATH、アンインストール、よくある落とし穴"
sidebar_label: "Windows（ネイティブ） — ベータ"
sidebar_position: 3
---

# Windows（ネイティブ）ガイド — 早期ベータ

:::warning 早期ベータ
ネイティブのWindowsサポートは**早期ベータ**です。インストールでき、実行でき、私たちのWindows-footgun lintを通過しますが、Linux/macOS/WSL2のパスほどの規模で実地テストされていません。粗削りな部分を覚悟してください — 特にサブプロセスの扱い、パスの癖、非ASCIIのコンソール出力の周辺です。何かに遭遇したら、再現手順とともに[issueを起票](https://github.com/NousResearch/hermes-agent/issues)してください。今日、実戦で鍛えられたセットアップが欲しい場合は、代わりに[WSL2上のLinux/macOSインストーラー](./windows-wsl-quickstart.md)を使ってください。
:::

HermesはWindows 10とWindows 11でネイティブに動作します — WSLなし、Cygwinなし、Dockerなし。このページは詳細解説です：何がネイティブで動くか、何がWSL専用か、インストーラーが実際に何をするか、そして触る必要があるかもしれないWindows固有のつまみについてです。

ただインストールしたいだけなら、[ランディングページ](/)または[インストールページ](../getting-started/installation#windows-native-powershell--early-beta)のワンライナーだけで十分です。何かに驚いたら、ここに戻ってきてください。

:::tip 代わりにWSLが欲しいですか？
本物のPOSIX環境（ダッシュボードの組み込みターミナル、`fork` のセマンティクス、Linux式のファイルウォッチャーなど）を好む場合は、**[Windows（WSL2）ガイド](./windows-wsl-quickstart.md)**を参照してください。両者はきれいに共存します：ネイティブのデータは `%LOCALAPPDATA%\hermes` 配下に、WSLのデータは `~/.hermes` 配下に置かれます。
:::

## クイックインストール

**PowerShell**（またはWindows Terminal）を開いて実行します：

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

管理者権限は不要です。インストーラーは `%LOCALAPPDATA%\hermes\` に展開し、`hermes` をあなたの**ユーザーPATH**に追加します — 完了後は新しいターミナルを開いてください。

**インストーラーオプション**（パラメータを渡すにはscriptblock形式が必要です）：

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1))) -NoVenv -SkipSetup -Branch main
```

| パラメータ | デフォルト | 目的 |
|---|---|---|
| `-Branch` | `main` | 特定のブランチをクローン（PRのテストに便利） |
| `-NoVenv` | off | venvの作成をスキップ（上級者向け — Python管理を自分で行う） |
| `-SkipSetup` | off | インストール後の `hermes setup` ウィザードをスキップ |
| `-HermesHome` | `%LOCALAPPDATA%\hermes` | データディレクトリを上書き |
| `-InstallDir` | `%LOCALAPPDATA%\hermes\hermes-agent` | コードの場所を上書き |

## インストーラーが実際に行うこと

上から下まで、順番に：

1. **`uv` をブートストラップ** — Astralの高速なPythonマネージャー。`%USERPROFILE%\.local\bin` にインストールされます。
2. **`uv` 経由でPython 3.11をインストール**。既存のPythonは不要です。
3. **Node.js 22をインストール**（利用可能ならwinget、なければ `%LOCALAPPDATA%\hermes\node` 配下に展開されるポータブルNode tarball）。ブラウザツールとWhatsAppブリッジに使われます。
4. **ポータブルGitをインストール** — `git` がすでにPATHにあればインストーラーはそれを使い、なければ簡素化された自己完結型の**PortableGit**（約45 MB、公式の `git-for-windows` リリースから）を `%LOCALAPPDATA%\hermes\git` にダウンロードします。管理者不要、Windowsインストーラーのレジストリも汚さず、マシン上の他のものへの干渉もありません。
5. **リポジトリをクローン** — `%LOCALAPPDATA%\hermes\hermes-agent` にクローンし、その中にvirtualenvを作成します。
6. **段階的な `uv pip install`** — まず `.[all]` を試し、`git+https` の依存がレート制限されたGitHubで失敗した場合、徐々に小さいセット（`[messaging,dashboard,ext]` → `[messaging]` → `.`）にフォールバックします。「単一の失敗で素のインストールに落ちる」障害モードを防ぎます。
7. **メッセージングSDKを自動インストール** — `.env` をキーに、`TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` / `WHATSAPP_ENABLED` が存在すれば、`python -m ensurepip --upgrade` と対象を絞った `pip install` を実行し、各プラットフォームのSDKが実際にインポート可能になるようにします。
8. **`HERMES_GIT_BASH_PATH` を設定** — 解決された `bash.exe` に設定し、Hermesが新しいシェルで決定論的にそれを見つけられるようにします。
9. **`%LOCALAPPDATA%\hermes\bin` をユーザーPATHに追加** — 新しいターミナルを開いた後に `hermes` コマンドを公開します。
10. **`hermes setup` を実行** — 通常の初回起動ウィザード（モデル、プロバイダー、ツールセット）。`-SkipSetup` でスキップします。

## 機能マトリクス

ダッシュボードの組み込みターミナルペインを除くすべてが、Windowsでネイティブに動作します。

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
| ダッシュボードの `/chat` 組み込みターミナルペイン | ✗（POSIX PTYが必要） | ✓ |
| ログイン時の自動起動 | ✓（schtasks） | ✓（systemd） |

ダッシュボードの `/chat` タブは、POSIX PTY（`ptyprocess`）を介して本物のターミナルを埋め込みます。ネイティブWindowsには同等のプリミティブがありません。Pythonの `pywinpty` / Windows ConPTYは動作しますが別実装になります — 将来の課題として扱います。**ダッシュボードの残りはネイティブで動作します** — その1つのタブだけが「これにはWSL2を使う」というバナーを表示します。

## HermesがWindowsでシェルコマンドを実行する方法 {#how-hermes-runs-shell-commands-on-windows}

Hermesのターミナルツールは、Claude Codeが使うのと同じ戦略で、**Git Bash** を通じてコマンドを実行します。これは、すべてのツールを書き直すことなく、POSIX対Windowsのギャップを回避します。

`bash.exe` の解決順序：

1. 設定されていれば `HERMES_GIT_BASH_PATH` 環境変数。
2. `%LOCALAPPDATA%\hermes\git\usr\bin\bash.exe`（インストーラー管理のPortableGit）。
3. `%LOCALAPPDATA%\hermes\git\bin\bash.exe`（古いGit-for-Windowsレイアウト）。
4. システムのGit-for-Windowsインストール（`%ProgramFiles%\Git\bin\bash.exe` など）。
5. 最後の手段として、MSYS2、Cygwin、またはPATH上の任意の `bash.exe`。

インストーラーは `HERMES_GIT_BASH_PATH` を明示的に設定するため、新しいPowerShellセッションが再発見する必要はありません。Hermesに特定のbashを使わせたい場合は上書きしてください — 例えば、システムのGit Bashや、シンボリックリンク経由のWSLホストのbashなどです。

**落とし穴:** MinGitのレイアウトは、フルのGit-for-Windowsインストーラーとは異なります — bashは `bin\bash.exe` ではなく `usr\bin\bash.exe` 配下にあります。Hermesは両方をチェックします。MinGit zipを手動で展開する場合は、**非busybox**バリアント（`MinGit-*-64-bit.zip`、`MinGit-*-busybox*.zip` ではない）を選ぶようにしてください — busyboxビルドは `bash` の代わりに `ash` を同梱し、ほとんどのcoreutilsが欠けています。

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

#21561以前は、`Ctrl-X Ctrl-E` を押したり `/edit` と入力したりしても、Windowsでは黙って何も起きませんでした。prompt_toolkitには、Windowsでは決して解決されないPOSIX絶対パスのフォールバックリスト（`/usr/bin/nano`、`/usr/bin/pico`、`/usr/bin/vi`、…）がハードコードされています — フルのGit for Windowsをインストールしていてもです。

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
# $PROFILE 内
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

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN HermesGateway` — ログイン時に標準（非昇格）権限で実行されるタスクを登録します。UACプロンプトはありません。
2. グループポリシーでschtasksがブロックされている場合、`start /min cmd.exe /d /c <wrapper>` ショートカットを `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup` に書き込むことにフォールバックします。同じ効果で、少し粗削りです。
3. ゲートウェイを **`pythonw.exe` 経由でデタッチして**起動します — `python.exe` ではありません。`pythonw.exe` にはコンソールが接続されていないため、兄弟プロセスからの `CTRL_C_EVENT` ブロードキャストに対して免疫があります（同じプロセスグループで何かをCtrl+Cしたときにゲートウェイを殺していた実際の問題でした）。

起動時に使われるフラグ：`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`。

### 管理

```powershell
hermes gateway status      # マージビュー: schtasks + Startupフォルダー + 実行中のPID
hermes gateway start       # スケジュールされたタスクを今すぐ開始
hermes gateway stop        # 優雅なSIGTERM相当（psutil経由のTerminateProcess）
hermes gateway restart
hermes gateway uninstall   # schtasksエントリー、Startupショートカット、pidファイルを削除
```

`hermes gateway status` は冪等です — 1000回連続で呼んでも、誤ってゲートウェイを殺すことは決してありません。（PR #21561以前は、`os.kill(pid, 0)` がCレベルで `CTRL_C_EVENT` と衝突して、黙って殺していました — その経緯を知りたい場合は下記の「プロセス管理の内部」を参照してください。）

### なぜWindowsサービスではないのか？

サービスはインストールに管理者権限を必要とし、ゲートウェイのライフサイクルをユーザーログインではなくマシンのブートに結びつけます。典型的なHermesユーザーが望むのは：ログイン → ゲートウェイが利用可能、ログアウト → ゲートウェイが消える、です。Scheduled Tasksは昇格なしでまさにこれを実現します。本当にサービスが欲しい場合は、`nssm` や `sc create` を手動で使ってください — しかしおそらく必要ないでしょう。

## データレイアウト

| パス | 内容 |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Gitチェックアウト + venv。`Remove-Item -Recurse` して再インストールしても安全。 |
| `%LOCALAPPDATA%\hermes\git\` | PortableGit（インストーラーが用意した場合のみ）。 |
| `%LOCALAPPDATA%\hermes\node\` | ポータブルNode.js（インストーラーが用意した場合のみ）。 |
| `%LOCALAPPDATA%\hermes\bin\` | `hermes.cmd` シム、ユーザーPATHに追加される。 |
| `%USERPROFILE%\.hermes\` | あなたの設定、認証、スキル、セッション、ログ。**再インストールしても残る。** |

この分割は意図的です：`%LOCALAPPDATA%\hermes` は使い捨てのインフラ（吹き飛ばしてもワンライナーが復元します）。`%USERPROFILE%\.hermes` はあなたのデータ — 設定、メモリ、スキル、セッション履歴 — で、Linuxインストールと形が同一です。マシン間でミラーすれば、Hermesがあなたとともに移動します。

**`HERMES_HOME` の上書き:** 環境変数を別のデータディレクトリを指すように設定します。Linuxと同じように動作します。

## ブラウザツール

ブラウザツールは `agent-browser`（Nodeヘルパー）を使ってChromiumを操作します。Windowsでは：

- インストーラーがnpm経由で `agent-browser` をPATHに置きます。
- `shutil.which("agent-browser", path=...)` が `.cmd` シムを自動的に拾います — `CreateProcessW` は拡張子のないshebangを実行できないため、Hermesは常に `.CMD` ラッパーに解決します。shebangスクリプトを手動で呼び出さず、常に `.cmd` を経由してください。
- Playwright Chromiumは初回実行時に自動インストールされます（`npx playwright install chromium`）。インストールが失敗した場合、`hermes doctor` が修正のヒントとともにそれを表面化します。

## WindowsでHermesを実行する — 実用上のメモ

### インストール後のPATH

インストーラーは `[Environment]::SetEnvironmentVariable` 経由で `%LOCALAPPDATA%\hermes\bin` をあなたの**ユーザーPATH**に追加します。既存のターミナルはこれを拾いません — インストール後に新しいPowerShellウィンドウ（またはWindows Terminalタブ）を開いてください。何をしているか分かっている場合を除き、手動で `$env:PATH += …` するのではなく、閉じて再度開いてください。

確認します：

```powershell
Get-Command hermes        # C:\Users\<you>\AppData\Local\hermes\bin\hermes.cmd と表示されるはず
hermes --version
```

### 環境変数

Hermesは `$env:X`（プロセススコープ）とユーザー環境変数（永続的、システムのプロパティ → 環境変数で設定）の両方を尊重します。`%USERPROFILE%\.hermes\.env` にAPIキーを設定するのが通常の方法です — Linuxと同じです：

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

すべてのWindowsプロセスにシークレットを見せたいのでない限り（それは望ましくありません）、ユーザー環境変数にシークレットを置かないでください。

### Windows固有の環境変数

これらはネイティブWindowsインストールにのみ影響します：

| 変数 | 効果 |
|---|---|
| `HERMES_GIT_BASH_PATH` | bash.exeの発見を上書き。任意のbashを指す — フルのGit-for-Windows、シンボリックリンク経由のWSL bash、MSYS2、Cygwin。インストーラーが自動的に設定する。 |
| `HERMES_DISABLE_WINDOWS_UTF8` | `1` に設定するとUTF-8 stdioシムを無効化し、ロケールのコードページにフォールバックする。エンコーディングのバグの二分探索に便利。 |
| `EDITOR` / `VISUAL` | `/edit` と `Ctrl-X Ctrl-E` 用のエディタ。両方が未設定の場合、Hermesはデフォルトで `notepad` を使う。 |

## アンインストール

PowerShellから：

```powershell
hermes uninstall
```

これがクリーンなパスです — schtasksエントリー、Startupフォルダーのショートカット、`hermes.cmd` シムを削除し、`%LOCALAPPDATA%\hermes\hermes-agent\` を削除し、ユーザーPATHを整理します。再インストールに備えて、`%USERPROFILE%\.hermes\`（あなたの設定、認証、スキル、セッション、ログ）はそのままにします。

すべてを消し去るには：

```powershell
hermes uninstall
Remove-Item -Recurse -Force "$env:USERPROFILE\.hermes"
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes"
```

`hermes uninstall` CLIサブコマンドは、schtasksエントリーが別のタスク名で登録されていた場合（古いインストール）も処理します — ハードコードされたタスク名ではなく、インストールパスで検索します。

## プロセス管理の内部

これは背景資料です — 「自分自身を殺している」奇妙な挙動をデバッグしている場合を除き、読み飛ばしてください。

LinuxとmacOSでは、POSIXのイディオム `os.kill(pid, 0)` は何もしない権限チェックです：「このPIDは生きていて、シグナルを送れるか？」。Windowsでは、Pythonの `os.kill` は `sig=0` を `CTRL_C_EVENT` にマップし — それらは整数値0で衝突します — `GenerateConsoleCtrlEvent(0, pid)` を通じてルーティングし、対象PIDを含む**コンソールプロセスグループ全体**にCtrl+Cをブロードキャストします。これが[bpo-14484](https://bugs.python.org/issue14484)で、2012年から未解決です。現在の挙動に依存するスクリプトを壊すことになるため、修正されません。

結果：Windowsで `os.kill(pid, 0)` 経由で「このPIDが生きているか確認する」と言っていたあらゆるコードパスは、黙って対象を殺していました。Hermesはそのような箇所すべて（11ファイルにわたる14箇所）を `gateway.status._pid_exists()` に移行しました。これは `psutil.pid_exists()` を使います（さらにこれはWindowsで `OpenProcess + GetExitCodeProcess` を使う — シグナルなし）。プラグインやパッチを書く場合は、`psutil.pid_exists()` を直接、または `gateway.status._pid_exists()` を使ってください — 決して `os.kill(pid, 0)` を使わないでください。

`scripts/check-windows-footguns.py` がCIでこれを強制します：新しい `os.kill(pid, 0)` 呼び出しは、その行に `# windows-footgun: ok — <reason>` マーカーが付いていない限り、`Windows footguns (blocking)` チェックを失敗させます。

## よくある落とし穴

**インストール直後の `hermes: command not found`。**
新しいPowerShellウィンドウを開いてください。インストーラーは `%LOCALAPPDATA%\hermes\bin` をユーザーPATHに追加しましたが、既存のシェルがそれを拾うには再起動が必要です。それまでの間は `& "$env:LOCALAPPDATA\hermes\bin\hermes.cmd"` を実行できます。

**ツール実行時の `WinError 193: %1 is not a valid Win32 application`。**
`.cmd` シムをバイパスしたshebangスクリプトの呼び出しに遭遇しました。Hermesは `shutil.which(cmd, path=local_bin)` 経由でコマンドを解決するためPATHEXTが `.CMD` を拾います — 代わりにハードコードされたパス経由でツールを呼び出している場合は、`.cmd` バリアントに切り替えてください（例 `npx` ではなく `npx.cmd`）。

**`[scriptblock]::Create(...)` が `The assignment expression is not valid` で失敗する。**
ダウンロードした `install.ps1` がUTF-8 BOMを拾いました。`irm | iex` 形式はBOMを自動的に取り除きますが、`[scriptblock]::Create((irm ...))` はそうしません。シンプルな `irm | iex` 形式で再実行するか、スクリプトを手動でダウンロードし、`[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))` 経由でBOMなしで保存してください。

**再起動後にゲートウェイが実行され続けない。**
`hermes gateway status` を確認してください — schtasksエントリー、Startupフォルダーのショートカット（使われている場合）、ライブのPIDをマージします。schtasksが登録されているのに実行されていない場合、グループポリシーが `ONLOGON` トリガーをブロックしている可能性があります。`schtasks /Query /TN HermesGateway /V /FO LIST` を実行してタスクの失敗理由を確認するか、アンインストールして `HERMES_GATEWAY_FORCE_STARTUP=1` で再インストールしてStartupフォルダーのパスにフォールバックしてください。

**`$env:EDITOR` を設定した後も `/edit` が何もしない。**
現在のプロセスにのみ設定しました。シェルを閉じて再度開くか、システムのプロパティ → 環境変数でユーザースコープに設定してください。新しいPowerShellウィンドウで `echo $env:EDITOR` で確認します。

**ブラウザツールは起動するがツールがタイムアウトする。**
Chromiumは初回実行時に自動インストールされます。インストールが失敗した場合（レート制限されたGitHub、Playwright CDNの不具合）、`hermes doctor` を実行してください — 欠けているChromiumを表面化し、修正するための正確な `npx playwright install chromium` コマンドを表示します。

**`agent-browser` が奇妙なNodeバージョンエラーで失敗する。**
インストーラーは `%LOCALAPPDATA%\hermes\node` にNode 22を用意しますが、あなたのPATHには古いシステムのNode 18が先にあるかもしれません。Hermesのnodeディレクトリをパスの前方に移動するか、Nodeを他で使っていない場合はシステムインストールを削除してください。

**中国語 / 日本語 / アラビア語の文字がCLIで `?` として表示される。**
UTF-8 stdioシムが有効化されませんでした。`HERMES_DISABLE_WINDOWS_UTF8` が設定されていないことを確認してください（`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`）。空なのにまだ `?` が見える場合、コンソールホスト（非常に古い `cmd.exe`）がUTF-8をまったくサポートしていない可能性があります — Windows Terminalに切り替えてください。

**ゲートウェイがTelegramの写真を送れない — 「`BadRequest: payload contains invalid characters`」。**
これはWindowsとは無関係ですが、ここで最初に表面化することがあります。通常、JSONボディ内のファイルパスにエスケープされていないバックスラッシュが含まれていることを意味します。Telegramは、生のWindowsパスではなく、Hermesが正規化したパスを受け取るはずです — カスタムプラグイン内でこれが見られる場合は、ユーザー入力からの `str(Path(...))` ではなく、Hermesが提供するパスを渡していることを確認してください。

**`git pull` 後の「自分の別のマシンでは動く」エンコーディングの奇妙さ。**
Windowsで非UTF-8エディタ（古いWindowsバージョンのNotepad、一部の中国語IME）を使ってHermesの設定やスキルを編集した場合、ファイルがBOM付きで保存された可能性があります。Hermesはほとんどの設定読み込みで `utf-8-sig` を許容しますが、折りたたまれたYAMLスカラー（`description: >`）内のBOMは黙ってYAMLのパースを壊します。ファイルをBOMなしのプレーンUTF-8として保存し直してください。

## 次に行く場所

- **[インストール](../getting-started/installation.md)** — Linux/macOS/WSL2/Termuxを含む、完全なインストールページ。
- **[Windows（WSL2）ガイド](./windows-wsl-quickstart.md)** — POSIXセマンティクスやダッシュボードのターミナルペインが欲しい場合。
- **[CLIリファレンス](../reference/cli-commands.md)** — すべての `hermes` サブコマンド。
- **[FAQ](../reference/faq.md)** — Windows固有でない一般的な質問。
- **[メッセージングゲートウェイ](./messaging/index.md)** — WindowsでTelegram/Discord/Slackを実行する。
