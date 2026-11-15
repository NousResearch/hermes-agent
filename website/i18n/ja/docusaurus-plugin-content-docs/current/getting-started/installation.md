---
sidebar_position: 2
title: "インストール"
description: "デスクトップバンドル、ソースインストーラー、Docker、Nix、または Termux APT パッケージで Hermes Agent をインストールする"
---

<head>
  <title>Hermes Agent のインストール: 公式ガイド | Nous Research</title>
  <meta property="og:title" content="Hermes Agent のインストール: 公式ガイド | Nous Research" />
</head>

# インストール

2 分足らずで Hermes Agent を動かせます！

:::tip プラットフォームサポート
完全なプラットフォームサポートマトリクス（サポートされる OS、配布方法、
プラットフォームに依存する機能）については、**[プラットフォームサポート](/docs/getting-started/platform-support)** を参照してください。
:::

## クイックインストール
### macOS または Windows のデスクトップパッケージ

お使いのプラットフォーム向けのパッケージを
[Hermes の Web サイト](https://hermes-agent.nousresearch.com/)からダウンロードします。

- **Windows:** ダウンロードした `.appinstaller` を Windows App Installer で開きます。
  署名済みの MSIX バンドルがインストールされ、その更新元が記録されます。
  Microsoft Store のパッケージは Store 側で別途管理されます。
- **macOS:** DMG を開き、`Hermes.app` をアプリケーションフォルダにコピーします。ZIP
  アーティファクトには、自動アップデーターが使用する署名済みアプリが含まれています。

バンドルパッケージには、エージェント、Python、サポートされる依存関係、ビルド済みの
インターフェースが含まれています。初回起動時にそのベースランタイムをビルドすることはありません。プロバイダーへのアクセスや
オプションの連携には、引き続きネットワークアクセスが必要になる場合があります。

`Hermes-Setup` ブートストラップインストーラーは別物です。これはソース
インストールをダウンロードし、デスクトップアプリをビルドします。[Hermes Desktop](/docs/user-guide/desktop) を参照してください。

リモート専用の Desktop ダウンロードは別途用意されていません。別のマシン上の Hermes
バックエンドで Desktop を使うには、上記のパッケージのいずれかをインストールし、
**Settings → Gateways** から接続してください。
[リモートバックエンドへの接続](/docs/user-guide/desktop#connecting-to-a-remote-backend)を参照してください。

:::note
macOS インストーラーは **Apple Silicon 専用**です。x86（Intel）プロセッサーの macOS は[サポート対象外のプラットフォーム](/docs/getting-started/platform-support#unsupported)です。
:::

### Hermes Desktop を使わない場合:
Hermes Desktop を使わずにコマンドラインのみでインストールするには、次を実行します:

#### Linux / macOS / WSL2
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows（ネイティブ） {#windows-native}

PowerShell で実行します:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

コマンドラインのみでインストールした後に Hermes Desktop をインストールして実行したい場合は、次を実行するだけです
```bash
hermes desktop
```

### Android / Termux

aarch64 の Android デバイスでは [Termux APT パッケージ](/docs/getting-started/termux)を使用してください。
`pkg install hermes-agent` を実行する前に、署名済みリポジトリを設定してください。
デスクトップ/サーバー向けのスクリプトは Termux のインストール経路ではありません。

### ソースインストーラーの動作内容

スクリプトはソースをクローンし、uv をブートストラップし、依存関係の準備を
PM に委ねます。PM は、バージョンが固定された Python、Node.js、npm、pnpm、ripgrep、FFmpeg を提供します。ソース
インストールでは `all` Python extra が選択されます。すべてのオプション extra ではありません。
PM は、ブラウザツールとコンピューター操作ツールもデフォルトでインストールします: ローカルブラウジング用に
Hermes が起動する、バージョン固定の Chromium（Chrome for Testing）と、
`cua-driver`（コンピューター操作ドライバー。macOS、Windows、glibc Linux 向け）です。
ダウンロードに失敗してもインストールは完了し、再試行用のコマンドが表示されます。
`browser_exec` の背後にあるブラウザドライバー（Browser Use CLI のエンジンである browser-harness）は
通常の Python 依存関係であるため、Desktop アプリを含むすべてのインストールにすでに
含まれています。PM に Chromium ビルドがない環境（Android/Termux、
musl Linux）では、Hermes は代わりに `PATH` 上のシステムの `chromium` を使用します。
その他のオプションツールは、それぞれの機能固有のインストール経路を使用します。

新規インストールでは最新の安定版リリースが入り、以降 `hermes update` はそのチャネルに
従います（[アップデート](./updating.md)を参照）。代わりに `main` の最新を
インストールするには、POSIX では `--channel main`、Windows では `-Channel main` を渡してください
（ワンライナーの場合は、先に `HERMES_INSTALL_CHANNEL=main` を設定します）。

ブラウザツールを除外するには、POSIX では `--skip-browser`、Windows では `-SkipBrowser`
を渡します。コンピューター操作ドライバーの場合は `--skip-computer-use` /
`-SkipComputerUse` です。Hermes はこれらの選択を記憶します: 以降のインストールや
`hermes update` でそれらが追加し直されることはありません。インストールしてこの選択を取り消すには、
`hermes pm install chromium` または `hermes pm install cua-driver` を実行してください。

スクリプトはランチャーを作成し、データディレクトリを準備します。対話的な実行では、
セットアップとゲートウェイの設定も呼び出されます。POSIX の `--non-interactive`、または
Windows の `-NonInteractive` は、入力が必要なステージをスキップします。オプションの
`--include-desktop` / `-IncludeDesktop` ステージは、ソースからデスクトップをビルドします。

ターミナル上では、スクリプトはステップごとに 1 行のステータスを表示し、git、uv、ビルドの出力を
Hermes データディレクトリ配下の `logs/install.log` に書き込みます。失敗したステップでは、
最後の数行とログのパスが表示されます。CI（`CI` または
`GITHUB_ACTIONS` が設定されている）、出力のリダイレクト、`--verbose` / `-Verbose`、または
`HERMES_INSTALL_VERBOSE=1` の場合は、代わりにすべてがストリーミング出力されます。

#### インストールレイアウト

| 方法 | コード | CLI エントリーポイント | デフォルトのユーザーデータ |
|---|---|---|---|
| POSIX ソーススクリプト | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` ラッパー | `~/.hermes/` |
| Windows ソーススクリプト | `%LOCALAPPDATA%\hermes\hermes-agent\` | `%LOCALAPPDATA%\hermes\bin\` | `%LOCALAPPDATA%\hermes\` |
| デスクトップバンドル | インストールされたアプリパッケージ内 | パッケージ化されたランチャー、Windows 実行エイリアス | プラットフォームのデフォルトの Hermes データディレクトリ |
| Docker | `/opt/hermes/` | イメージのエントリーポイントと `hermes` shim | マウントされた `/opt/data/` |
| Termux APT | `$PREFIX/lib/hermes-agent/` | `$PREFIX/bin/` 内のシンボリックリンク | `~/.hermes/` |

`HERMES_HOME` はユーザーデータの場所を選択します。POSIX スクリプトの `--dir` は、ソース
チェックアウトの場所を独立して選択します。Windows では `-HermesHome` と `-InstallDir` を使用します。
POSIX スクリプトを root として実行しても、FHS レイアウトが自動的に選択されることはありません:
明示的なソースパスを指定しない限り、root のホームが使用されます。

PM のツールストアと、インストールごとの Python 世代は、それぞれ別のライフタイムを持ちます。
それらの場所については[パッケージ管理](/docs/reference/package-management)を参照してください。
アプリケーションのインストールを修復するためにデータルートを削除しないでください。

### インストール後

シェルを再読み込みしてチャットを開始します:

```bash
source ~/.bashrc   # or: source ~/.zshrc
hermes             # Start chatting!
```

後で個々の設定を再構成するには、専用のコマンドを使います:

```bash
hermes model          # Choose your LLM provider and model
hermes tools          # Configure which tools are enabled
hermes gateway setup  # Set up messaging platforms
hermes config set     # Set individual config values
hermes config get     # Inspect individual config values
hermes setup          # Or run the full setup wizard to configure everything at once
```

:::tip 最速の方法: Nous Portal
1 つのサブスクリプションで、300 以上のモデルと [Tool Gateway](../user-guide/features/tool-gateway.md)（Web 検索、画像生成、TTS、クラウドブラウザ）が利用できます。ツールごとのキーのやりくりは不要です:

```bash
hermes setup --portal
```

この 1 つのコマンドで、ログイン、Nous のプロバイダー設定、Tool Gateway の有効化が行われます。
:::

:::tip すでに別のマシンで Hermes を動かしていますか？
セットアップを一から作り直す必要はありません。`hermes import` で完全なバックアップを復元するか（[Hermes を別のマシンにエクスポートする](../reference/faq.md#exporting-hermes-to-another-machine)を参照）、`hermes profile import` で単一のエージェントを移行してください（[単一のプロファイルを別のマシンに移動する](../reference/faq.md#moving-a-single-profile-to-another-machine)を参照）。プロファイルのエクスポートは設計上認証情報を含まないため、エクスポートだけでは完全なバックアップにならない点に注意してください。どちらを使うべきかは [`hermes backup` と `hermes profile export` の違い](../reference/faq.md#hermes-backup-vs-hermes-profile-export)で説明しています。
:::

---

## 前提条件

POSIX ソーススクリプトには、Git、curl、tar、SHA-256 ユーティリティが必要です。
Windows では、Git がない場合、バージョン固定の Git for Windows アーカイブをブートストラップできます。
スクリプトは常に検証済みの固定バージョンの uv をダウンロードします。PATH 上にすでにある uv が使われることはありません。

現在のファーストパーティのインストールは **Python 3.14** で動作します。`pyproject.toml` の
より広い `>=3.11,<3.15` という範囲は、古い Python のインストールが、PM によって 3.14 に
切り替えられる前にアップデーターを実行できるようにするためのものです。3.11〜3.13 での現在の
ランタイムサポートを約束するものではありません。PM は管理対象ツールのバージョンを
`pm/lock.json` から選択します。任意のシステムの Node バージョンをインストール済み
ランタイムとして採用することはありません。

ソースビルドには、ネイティブコンパイラーとプラットフォームの開発ライブラリが必要になる場合があります。
ソースから Electron をビルドする場合は、Node ネイティブモジュールの要件が加わります。これらの
ビルド前提条件は、完全なデスクトップパッケージのインストールには適用されません。
Linux の Chromium には、ディストリビューションが提供するシステムライブラリも必要です。

glibc Linux では、管理対象の Node.js が `libatomic.so.1` にリンクしますが、最小構成の
Debian、Ubuntu、RHEL 系のイメージにはこれが含まれていません。ライブラリが
見つからない場合、インストーラーと `hermes update` はディストリビューションのパッケージ
（apt と zypper では `libatomic1`、dnf/yum と apk では `libatomic`、pacman では `gcc-libs`）
をインストールします。root の場合はパッケージマネージャーを直接実行し、それ以外の場合は
`sudo -n` として実行します。対話的な実行では、依存関係をインストールする前に sudo パスワードを
1 回だけ尋ねます。`--non-interactive` の実行では、入力を求めることはありません。
インストールを実行できない場合、エラーには見つかったパッケージマネージャー向けの正確なコマンドが示されます。

:::tip Nix ユーザー
Nix は**明示的にサポートされるインストール経路ではなくなりました**（ベストエフォートのみ）。すでに Nix を使っている場合（NixOS、macOS、または Linux 上で）、Nix flake、宣言的な NixOS モジュール、オプションのコンテナモードを備えた専用のセットアップ経路があります。**[Nix と NixOS のセットアップ](/docs/getting-started/nix-setup)**ガイドを参照してください。
:::

---

## 手動 / 開発者向けインストール

ソースチェックアウトの場合は、まず
[PM 開発者ワークフロー](/docs/reference/package-management#developer-workflow)から始めてください。
アクティベーション、日常的なコマンド、依存関係の更新、現在のブートストラップの制限について説明しています。
[開発セットアップ](/docs/developer-guide/contributing#development-setup)では、別途のテスト環境とチェックについて説明しています。

---

## sudo なし / システムサービスユーザーでのインストール

ソースインストーラーは、想定するサービスユーザーとして実行してください。そのホーム、ツールストア、
設定、ランチャーは、そのユーザーが所有している必要があります。

1. 管理者として、ソースビルドの前提条件と、Chromium が必要とする Linux
   ライブラリ（サービスがローカルでブラウジングする場合のみ）をインストールします。
2. サービスユーザーとして、通常のインストーラーを実行します:

   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

3. 実際のランチャーディレクトリを、サービスユーザーのシェル環境に追加します:

   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

4. そのアカウントから `hermes doctor` を実行します。ハードコードされた
   `venv/bin/hermes` パスではなく、インストールされたラッパーを使用してください。
5. ログアウト後も存続する必要がある Linux のユーザーサービスの場合は、管理者として lingering を有効にします:

   ```bash
   sudo loginctl enable-linger SERVICE_USER
   ```

現在のソースインストーラーは、Playwright の `--with-deps` ステップを実行しません。
Node.js の `libatomic`（前提条件を参照）を除き、システムパッケージはインストールしません。
PM はツールのバイナリを管理し、システムライブラリは管理者が用意します。sudo を持たない
サービスユーザーには、管理者に実行を依頼するための正確な `libatomic` コマンドが表示されます。
[ブラウザ自動化](../user-guide/features/browser.md)と
[メッセージングゲートウェイ](../user-guide/messaging/index.md)を参照してください。

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| `hermes: command not found` | シェルを再読み込み（`source ~/.bashrc`）するか、PATH を確認 |
| `API key not set` | `hermes model` を実行してプロバイダーを設定するか、`hermes config set OPENROUTER_API_KEY your_key` を実行 |
| アップデート後に設定が見つからない | `hermes config check` を実行してから `hermes config migrate` を実行 |

さらなる診断には `hermes doctor` を実行してください。何が不足していてどう修正すればよいかを正確に教えてくれます。

### シンボリックリンクされたホームディレクトリと外部ストレージ

Hermes は、シンボリックリンクされた `HERMES_HOME` と、`hooks`、`skills`、`sessions`、`logs` を含む
シンボリックリンクされたホームのサブディレクトリをサポートしています。ホームの初期化中、
既存のディレクトリリンクは保持され、リンクされたディレクトリ（および `logs/curator` などの子孫）の
パーミッションは、その所有者に委ねられます。

リンク先が存在しない、アクセスできない、またはディレクトリではない場合、初期化は
パスとリンク先を示すストレージエラーで停止します。Hermes はリンクを置き換えたり、存在しないリンク先を
作成したりは**しません**: そうすると、外部ボリュームや NAS ボリュームがアンマウントされている間に、
データがローカルディスクに書き込まれてしまう可能性があるためです。報告されたリンクを確認し、
マウントを復元するかリンク先を修正し、アクセス権限を確認してから再試行してください。意図的に新しい
dotfiles のリンク先を作る場合は、目的のストレージが利用可能であることを確認してから、自分で作成してください。

`hermes doctor` は、これらの失敗を無効な YAML ではなく、ストレージの問題として報告します。
既存の `config.yaml` はそのままにしてください。利用できないディレクトリの修復方法は `hermes setup` の実行ではありません。
これはディレクトリの可用性チェックであり、マウントの監視ではありません:
既存のディレクトリがあっても、目的のボリュームがマウントされていることは確認できません。

## インストール方法の自動検出

アップデートの担当は、データホームだけでなく、実行中のインストールによって決まります。
ソースチェックアウトは、管理された Git アップデート経路を使用します。デスクトップバンドル、Docker、
Nix、Termux パッケージは、それぞれのパッケージ所有者のアップデート機構を維持します。
`hermes doctor` はインストールの出所を報告します。パッケージが所有するファイルを変更する前に、
[アップデートとアンインストール](./updating.md)を参照してください。
