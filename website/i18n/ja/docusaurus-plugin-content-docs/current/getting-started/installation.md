---
sidebar_position: 2
title: "インストール"
description: "Linux、macOS、WSL2、Windows ネイティブ（アーリーベータ）、または Termux 経由の Android に Hermes Agent をインストールする"
---

# インストール

ワンラインインストーラーを使えば、2 分足らずで Hermes Agent を動かせます。

## クイックインストール

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows（ネイティブ、PowerShell）— アーリーベータ {#windows-native-powershell--early-beta}

:::warning アーリーベータ
Windows ネイティブサポートは**アーリーベータ**です。一般的な経路ではインストールでき動作しますが、POSIX 向けインストーラーほど広範にテストされていません。問題に遭遇したら [Issue を報告](https://github.com/NousResearch/hermes-agent/issues)してください。現時点で Windows 上で最も実績のあるセットアップを使いたい場合は、代わりに **WSL2** の中で上記の Linux/macOS ワンライナーを使ってください。
:::

PowerShell を開いて次を実行します:

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

インストーラーは**すべて**を処理します: `uv`、Python 3.11、Node.js 22、`ripgrep`、`ffmpeg`、**そしてポータブルな Git Bash**（PortableGit — `bash.exe` と Hermes がシェルコマンドに使う完全な POSIX ツールチェーンを同梱した、自己完結型の Git-for-Windows ディストリビューション。32 ビット Windows では、インストーラーは bash を持たず terminal-tool / agent-browser 機能を無効化する MinGit にフォールバックします）。リポジトリを `%LOCALAPPDATA%\hermes\hermes-agent` 配下にクローンし、仮想環境を作成し、`hermes` を**ユーザー PATH** に追加します。インストール後、PATH を反映するためにターミナルを再起動してください（または新しい PowerShell ウィンドウを開いてください）。

**Git の扱い方:**
1. `git` がすでに PATH 上にある場合、インストーラーは既存のインストールを使います。
2. そうでなければ、ポータブルな **PortableGit**（約 50MB、公式の `git-for-windows` GitHub リリースから）をダウンロードし、`%LOCALAPPDATA%\hermes\git` に展開します。管理者権限は不要です。完全に分離されており、壊れているかどうかにかかわらず、システムの Git インストールに干渉しません。（32 ビット Windows では、PortableGit が 64 ビットと ARM64 のアセットのみを同梱しているため MinGit にフォールバックします。bash に依存する Hermes 機能は 32 ビットホストでは動作しません。）

**なぜ winget を使わないのか？** 以前の設計では `winget install Git.Git` で Git を自動インストールしていましたが、システムの Git インストールが部分的または壊れた状態にあると winget はひどく失敗します（まさにユーザーがインストーラーに「ただ動いてほしい」と思うときに）。ポータブル Git のアプローチは、winget、Windows インストーラーレジストリ、既存のシステム Git を完全に回避します。Hermes の Git インストール自体が壊れた場合は、`Remove-Item %LOCALAPPDATA%\hermes\git` を実行してインストーラーを再実行してください。システムへの影響もアンインストールの面倒もありません。

インストーラーは、見つかった `bash.exe` を指すように `HERMES_GIT_BASH_PATH` も設定し、Hermes が新しいシェルで確定的に解決できるようにします。

WSL2 を好む場合は、上記の Linux インストーラーがその中で動作します。ネイティブと WSL のインストールは競合せずに共存できます（ネイティブのデータは `%LOCALAPPDATA%\hermes` 配下、WSL のデータは `~/.hermes` 配下に置かれます）。

### Android / Termux

Hermes は Termux を認識するインストーラー経路も提供するようになりました:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

インストーラーは Termux を自動検出し、テスト済みの Android フローに切り替えます:
- システム依存関係（`git`、`python`、`nodejs`、`ripgrep`、`ffmpeg`、ビルドツール）に Termux の `pkg` を使用
- `python -m venv` で仮想環境を作成
- Android ホイールビルド向けに `ANDROID_API_LEVEL` を自動でエクスポート
- 広範な `.[termux-all]` extra を優先し、最初の試行がコンパイルに失敗した場合は、より小さな `.[termux]` extra（最終的にはベースインストール）にフォールバック
- 既定では、未テストの browser / WhatsApp ブートストラップをスキップ

完全に明示的な経路が必要な場合は、専用の [Termux ガイド](./termux.md)に従ってください。

:::note Windows の機能パリティ（アーリーベータ）

Windows ネイティブは**アーリーベータ**です。ブラウザベースのダッシュボードチャットターミナルを除き、すべてが Windows 上でネイティブに動作します:
- **CLI（`hermes chat`、`hermes setup`、`hermes gateway` など）** — ネイティブ、既定のターミナルを使用
- **ゲートウェイ（Telegram、Discord、Slack など）** — ネイティブ、バックグラウンドの PowerShell プロセスとして実行
- **Cron スケジューラ** — ネイティブ
- **ブラウザツール** — ネイティブ（Node.js 経由の Chromium）
- **MCP サーバー** — ネイティブ（stdio と HTTP の両トランスポートに対応）
- **ダッシュボードの `/chat` ターミナルペイン** — **WSL2 のみ**（POSIX PTY を使用。ネイティブ Windows には同等のものがありません）。ダッシュボードの残りの部分（セッション、ジョブ、メトリクス）はネイティブで動作します。ゲートされるのは埋め込み PTY ターミナルタブのみです。

エンコーディング関連のバグに遭遇し、レガシーの cp1252 stdio 経路にフォールバックしたい場合は、環境に `HERMES_DISABLE_WINDOWS_UTF8=1` を設定してください（二分探索による切り分けに便利です）。
:::

### インストーラーの動作内容

インストーラーはすべてを自動的に処理します。あらゆる依存関係（Python、Node.js、ripgrep、ffmpeg）、リポジトリのクローン、仮想環境、グローバルな `hermes` コマンドのセットアップ、LLM プロバイダーの設定です。終わる頃には、チャットを始められる状態になっています。

#### インストールレイアウト

インストーラーが何をどこに置くかは、通常ユーザーとしてインストールするか root としてインストールするかによって異なります:

| インストーラー | コードの場所 | `hermes` バイナリ | データディレクトリ |
|---|---|---|---|
| ユーザー単位（通常） | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes`（シンボリックリンク） | `~/.hermes/` |
| root モード（`sudo curl … \| sudo bash`） | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/`（または `$HERMES_HOME`） |

root モードの **FHS レイアウト**（`/usr/local/lib/…`、`/usr/local/bin/hermes`）は、他のシステム全体の開発者ツールが Linux 上で配置される場所と一致します。1 つのシステムインストールで全ユーザーに対応すべき共有マシンのデプロイに便利です。ユーザー単位の設定（認証、スキル、セッション）は、引き続き各ユーザーの `~/.hermes/` または明示的な `HERMES_HOME` 配下に置かれます。

### インストール後

シェルを再読み込みしてチャットを開始します:

```bash
source ~/.bashrc   # または: source ~/.zshrc
hermes             # チャット開始！
```

後で個々の設定を再構成するには、専用のコマンドを使います:

```bash
hermes model          # LLM プロバイダーとモデルを選択
hermes tools          # 有効にするツールを設定
hermes gateway setup  # メッセージングプラットフォームをセットアップ
hermes config set     # 個々の設定値を設定
hermes setup          # またはフルセットアップウィザードを実行してすべてを一度に設定
```

---

## 前提条件

唯一の前提条件は **Git** です。インストーラーがそれ以外のすべてを自動的に処理します:

- **uv**（高速な Python パッケージマネージャー）
- **Python 3.11**（uv 経由、sudo 不要）
- **Node.js v22**（ブラウザ自動化と WhatsApp ブリッジ用）
- **ripgrep**（高速なファイル検索）
- **ffmpeg**（TTS 用の音声フォーマット変換）

:::info
Python、Node.js、ripgrep、ffmpeg を手動でインストールする**必要はありません**。インストーラーが不足しているものを検出して、あなたの代わりにインストールします。`git` が利用可能であること（`git --version`）だけ確認してください。
:::

:::tip Nix ユーザー
Nix を使っている場合（NixOS、macOS、または Linux 上で）、Nix flake、宣言的な NixOS モジュール、オプションのコンテナモードを備えた専用のセットアップ経路があります。**[Nix と NixOS のセットアップ](./nix-setup.md)**ガイドを参照してください。
:::

---

## 手動 / 開発者向けインストール

リポジトリをクローンしてソースからインストールしたい場合（コントリビュート、特定のブランチからの実行、または仮想環境を完全に制御したい場合）は、Contributing ガイドの [開発セットアップ](../developer-guide/contributing.md#development-setup)セクションを参照してください。

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| `hermes: command not found` | シェルを再読み込み（`source ~/.bashrc`）するか、PATH を確認 |
| `API key not set` | `hermes model` を実行してプロバイダーを設定するか、`hermes config set OPENROUTER_API_KEY your_key` を実行 |
| アップデート後に設定が見つからない | `hermes config check` を実行してから `hermes config migrate` を実行 |

さらなる診断には `hermes doctor` を実行してください。何が不足していてどう修正すればよいかを正確に教えてくれます。
