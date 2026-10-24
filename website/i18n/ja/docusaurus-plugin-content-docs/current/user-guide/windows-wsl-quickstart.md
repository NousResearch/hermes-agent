---
title: "Windows (WSL2) ガイド"
description: "WSL2 経由で Windows 上で Hermes Agent を実行 — セットアップ、Windows と Linux 間のファイルシステムアクセス、ネットワーク、よくある落とし穴"
sidebar_label: "Windows (WSL2)"
sidebar_position: 2
---

# Windows (WSL2) ガイド

Hermes Agent は、Windows ネイティブと WSL2 の**両方**に対応するようになりました。このページでは WSL2 経路を扱います。ネイティブの PowerShell インストールについては、専用の **[Windows（ネイティブ）ガイド](./windows-native.md)** を参照してください。

**ネイティブよりも WSL2 を選ぶべきとき:**
- ダッシュボードの埋め込みターミナル（`/chat` タブ）を使いたい — このペインは POSIX PTY を必要とし、WSL2 のみです。
- POSIX 中心の開発作業を行っており、Hermes セッションが開発ツールと同じファイルシステム／パスを共有してほしい。
- すでに WSL2 環境があり、2 つ目のインストールを保守したくない。

**ネイティブで問題ない（あるいはより良い）とき:**
- 対話的なチャット、ゲートウェイ（Telegram/Discord など）、cron スケジューラ、ブラウザツール、MCP サーバー、その他ほとんどの Hermes 機能は、すべて Windows 上でネイティブに動作します。
- ファイルを参照したり URL を開いたりするたびに、WSL↔Windows の境界をまたぐことを考えたくない。

WSL2 では、実質的に 2 台のコンピュータが動いています: Windows ホストと、WSL が管理する Linux VM です。混乱のほとんどは、いまどちらにいるのか分からないことから生じます。

このガイドでは、その分割のうち特に Hermes に影響する部分を扱います: WSL2 のインストール、Windows と Linux 間でファイルをやり取りすること、双方向のネットワーク、そして実際に人々がはまる落とし穴です。

:::info 他の言語
このページは複数の言語で提供されています。右上の**言語**メニューから切り替えられます。
:::

## なぜ WSL2 なのか（ネイティブ Windows との比較）

ネイティブの Windows インストールは Windows 上で直接動作します: Windows のターミナル（PowerShell、Windows Terminal など）、Windows のファイルシステムパス（`C:\Users\…`）、Windows のプロセスです。Hermes はシェルコマンドの実行に Git Bash を使います。これは Claude Code やその他のエージェントが今日 Windows を扱う方法であり、完全な書き直しなしに POSIX 対 Windows のギャップを回避します。

WSL2 は軽量な VM 内で本物の Linux カーネルを動かすため、その中の Hermes は実質的に Ubuntu 上で動かすのと同一です。本物の POSIX 環境が欲しいときに価値があります: `fork`、`/tmp`、UNIX ソケット、シグナルのセマンティクス、PTY ベースのターミナル、`bash`/`zsh` のようなシェル、そして `rg`、`git`、`ffmpeg` のような Linux 上での挙動どおりに動くツールです。

WSL2 の実際的な帰結:

- Hermes の CLI、ゲートウェイ、セッション、メモリ、スキル、ツールランタイムはすべて Linux VM 内に存在します。
- Windows プログラム（ブラウザ、ネイティブアプリ、ログイン済みプロファイルの Chrome）はその外に存在します。
- 両者を会話させたいたびに（ファイル共有、URL を開く、Chrome の制御、ローカルモデルサーバーへの接続、Hermes ゲートウェイをスマートフォンに公開）境界をまたぎます。それらの境界こそ、このガイドのテーマです。

## WSL2 をインストールする

**管理者 PowerShell** または Windows Terminal から:

```powershell
wsl --install
```

新しい Windows 10 22H2+ または Windows 11 のマシンでは、これにより WSL2 カーネル、Virtual Machine Platform 機能、既定の Ubuntu ディストリビューションがインストールされます。プロンプトが出たら再起動してください。再起動後、Ubuntu が開き、Linux のユーザー名 + パスワードを尋ねます。これは Windows アカウントとは無関係の**新しい Linux ユーザー**です。

実際に WSL2（レガシーの WSL1 ではなく）にいることを確認します:

```powershell
wsl --list --verbose
```

`VERSION  2` が表示されるはずです。ディストリビューションが `VERSION  1` と表示される場合は変換します:

```powershell
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

Hermes は WSL1 上では確実には動作しません。WSL1 は Linux のシステムコールをその場で変換するため、一部の挙動（procfs、シグナル、ネットワーク）が本物の Linux と異なります。

### ディストリビューションの選択

Ubuntu（LTS）が私たちのテスト対象です。Debian も動作します。Arch と NixOS も希望者向けに動作しますが、ワンラインインストーラーは Debian 系の `apt` システムを前提とします。その経路については [Nix セットアップガイド](/docs/getting-started/nix-setup)を参照してください。

### systemd を有効にする（推奨）

hermes ゲートウェイ（およびその他、起動し続けたいもの）は systemd があると管理が容易です。最近の WSL では、ディストリビューション内で一度有効にします:

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true

[interop]
enabled=true
appendWindowsPath=true

[automount]
options = "metadata,umask=22,fmask=11"
EOF
```

そして PowerShell から:

```powershell
wsl --shutdown
```

WSL ターミナルを開き直します。`ps -p 1 -o comm=` が `systemd` を出力するはずです。

上記の `metadata` マウントオプションは重要です。これがないと、`/mnt/c/...` 上のファイルが本物の Linux 権限ビットを保存できず、Windows パス配下のスクリプトに対する `chmod +x` のようなものが壊れます。

### WSL 内に Hermes をインストールする

WSL2 シェルを開いたら:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes
```

インストーラーは WSL2 を素の Linux として扱います。WSL 固有のことは何も必要ありません。完全なレイアウトは [インストール](/docs/getting-started/installation)を参照してください。

## ファイルシステム: Windows ↔ WSL2 の境界をまたぐ

ここが最も多くの人がつまずく部分です。**2 つのファイルシステム**があり、ファイルをどこに置くかが重要です。パフォーマンス、正確性、そしてツールが何を見られるかに関わります。

### 2 つの方向

| 方向 | 内部のパス | 使うパス |
|---|---|---|
| WSL から見た Windows ディスク | `C:\Users\you\Documents` | `/mnt/c/Users/you/Documents` |
| Windows から見た WSL ディスク | `/home/you/code` | `\\wsl$\Ubuntu\home\you\code`（新しいビルドでは `\\wsl.localhost\Ubuntu\...`） |

どちらも実在し、どちらも機能しますが、**同じファイルシステムではありません**。内部では 9P ネットワークプロトコルで橋渡しされています。これには実際のパフォーマンスとセマンティクス上の帰結があります。

### Hermes とプロジェクトをどこに置くか

**目安: Linux らしいものはすべて Linux ファイルシステム内に置く。**

- Hermes のインストール（`~/.hermes/`）— Linux 側。インストーラーがすでにそうします。
- WSL から作業する git リポジトリ — Linux 側（`~/code/...`、`~/projects/...`）。
- モデル、データセット、venv — Linux 側。

このルールに従うことで得られるもの:

- **高速な I/O。** `/mnt/c/...` 上の操作は 9P を経由し、ネイティブの ext4 より 10〜100 倍遅くなります。`~/code` 配下では一瞬に感じる 1 万ファイルのリポジトリの `git status` が、`/mnt/c` 配下では 15 秒以上かかることがあります。
- **正しい権限。** Linux 権限ビットは `/mnt/c` 上ではベストエフォートのエミュレーションです。`ssh` が「bad permissions」でキーを拒否したり、`chmod +x` が静かに失敗したりするのはよくあることです。
- **信頼できるファイルウォッチャー。** 9P をまたぐ inotify は不安定です。ファイルウォッチャー（開発サーバー、テストランナー）は `/mnt/c` 上の変更を頻繁に見逃します。
- **大文字小文字の区別による驚きがない。** Windows パスは既定で大文字小文字を区別しませんが、Linux は区別します。`Readme.md` と `README.md` の両方を持つプロジェクトは、どちら側にいるかで挙動が変わります。

`/mnt/c` には、ファイルが Windows 側に存在する**必要がある**ときだけ置いてください。例えば、Windows GUI アプリから開きたい場合や、Windows Chrome の DevTools MCP が現在のディレクトリを Windows から到達可能なパスにする必要がある場合です。

### ファイルをやり取りする

**Windows → WSL へ:** 最も簡単なのは、エクスプローラーを開いてアドレスバーに `\\wsl.localhost\Ubuntu` と入力することです。そこから `\home\<you>\...` へドラッグ&ドロップできます。または PowerShell から:

```powershell
wsl cp /mnt/c/Users/you/Downloads/file.pdf ~/incoming/
```

**WSL → Windows へ:** `/mnt/c/Users/<you>/...` にコピーすると、すぐに Windows エクスプローラーに表示されます:

```bash
cp ~/reports/output.pdf /mnt/c/Users/you/Desktop/
```

**WSL のファイルを Windows アプリで開く**（GUI エディタ、ブラウザなど）: `explorer.exe` または `wslview` を使います:

```bash
sudo apt install wslu     # 一度だけ — wslview、wslpath、wslopen などが使えるようになります
wslview ~/reports/output.pdf    # Windows の既定のハンドラーで開く
explorer.exe .                  # 現在の WSL ディレクトリを Windows エクスプローラーで開く
```

**2 つの世界の間でパスを変換する:**

```bash
wslpath -w ~/code/project        # → \\wsl.localhost\Ubuntu\home\you\code\project
wslpath -u 'C:\Users\you'        # → /mnt/c/Users/you
```

### 改行コード、BOM、git

Windows 側で Windows エディタを使ってファイルを編集すると、`CRLF` 改行コードになることがあります。Linux 側の `bash` や Python がそれらを読むと、シェルスクリプトは `bad interpreter: /bin/bash^M` で壊れ、Python は BOM 付きの `.env` ファイルで失敗することがあります。

修正方法は、WSL 内（Windows 上ではなく）の健全な git 設定です:

```bash
git config --global core.autocrlf input
git config --global core.eol lf
```

すでに CRLF を持つファイルには:

```bash
sudo apt install dos2unix
dos2unix path/to/script.sh
```

### 「WSL 内でクローンするか、`/mnt/c` 上か？」

WSL 内でクローンしてください。特別な理由がない限り、常にそうです。典型的な Hermes ワークフロー（`hermes chat`、リポジトリを `rg`/`ripgrep` するツール呼び出し、ファイルウォッチャー、バックグラウンドゲートウェイ）は、`/mnt/c/Users/you/myrepo` より `~/code/myrepo` に対する方が劇的に高速で信頼できます。

例外が 1 つ: **Windows バイナリを起動する MCP ブリッジ。** `cmd.exe` を通じて `chrome-devtools-mcp` を使っている場合（[MCP ガイド: WSL → Windows Chrome](/docs/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome) を参照）、Hermes の現在の作業ディレクトリが `~` だと Windows が `UNC` 警告を出すことがあります。その場合は、Windows プロセスがドライブレターの cwd を持つよう、`/mnt/c/` 配下のどこかから Hermes を起動してください。

## ネットワーク: WSL ↔ Windows

WSL2 は独自のネットワークスタックを持つ軽量 VM で動作します。つまり、WSL 内の `localhost` は Windows 上の `localhost` と**同じではありません**。ネットワークの観点からは、2 つの別々のホストです。サービスごとに、トラフィックがどちらの方向に流れるかを決め、正しいブリッジを選ぶ必要があります。

2 つのケースが絶えず出てきます。

### ケース 1 — WSL 内の Hermes が Windows 上のサービスと通信する

最も一般的: **Windows 上で Ollama、LM Studio、または llama-server を実行**していて、（WSL 内の）Hermes がそれに接続する必要がある場合。

この標準的な手順はプロバイダーガイドにあります: **[ローカルモデル向けの WSL2 ネットワーク →](/docs/integrations/providers#wsl2-networking-windows-users)**

要約:

- **Windows 11 22H2+:** ミラーリングネットワークモードを有効にします（`%USERPROFILE%\.wslconfig` で `networkingMode=mirrored`、その後 `wsl --shutdown`）。すると `localhost` が双方向で機能します。
- **Windows 10 または古いビルド:** Windows ホスト IP（WSL の仮想ネットワークの既定ゲートウェイ）を使い、Windows 上のサーバーが `127.0.0.1` だけでなく `0.0.0.0` にバインドするようにします。Windows ファイアウォールにも通常そのポート用のルールが必要です。

完全な表（Ollama / LM Studio / vLLM / SGLang のバインドアドレス、ファイアウォールルールのワンライナー、動的 IP ヘルパー、Hyper-V ファイアウォールの回避策）は、上記のリンクをたどってください。ここでは重複させません。

### ケース 2 — Windows 上（または LAN 上）の何かが WSL 内の Hermes と通信する

これは逆方向で、他の場所ではあまり文書化されていませんが、次の用途で必要になります:

- Windows のブラウザから Hermes の **Web ダッシュボード**を使う。
- Windows 側のツールから **OpenAI 互換 API サーバー**（`API_SERVER_ENABLED=true` のときに `hermes gateway` が公開）を使う。[API サーバーの機能ページ](/docs/user-guide/features/api-server)を参照。
- プラットフォームがローカルの Webhook URL に ping する**メッセージングゲートウェイ**（Telegram、Discord など）をテストする。通常は素のポート転送よりも `cloudflared`/`ngrok` を使います。

#### サブケース 2a: Windows ホスト自身から

**ミラーモードを有効にした Windows 11 22H2+** では、何もする必要はありません。WSL 内で `0.0.0.0:8080`（あるいは `127.0.0.1:8080` でさえ）にバインドするプロセスは、Windows ブラウザから `http://localhost:8080` で到達できます。WSL がバインドを自動的にホストへ公開します。

**NAT モード**（Windows 10 / 古い Windows 11）では、WSL2 の既定の「localhost フォワーディング」が一般に Linux 側の `127.0.0.1` バインドを Windows の `localhost` へ転送するため、`--host 127.0.0.1` で起動した Hermes サービスは通常 Windows から `http://localhost:PORT` で到達できます。到達できない場合:

- WSL 内で明示的に `0.0.0.0` にバインドします。
- `ip -4 addr show eth0 | grep inet` で WSL VM の IP を見つけ、Windows からそれに接続します。

#### サブケース 2b: LAN 上の別のデバイスから（スマートフォン、タブレット、別の PC）

これが本当の苦労どころです。トラフィックは **LAN デバイス → Windows ホスト → WSL VM** と流れ、両方のホップを設定する必要があります:

1. **WSL 内で全インターフェースにバインドする。** `127.0.0.1` でリッスンするプロセスは VM の外から決して到達できません。`0.0.0.0` を使ってください。

2. **Windows → WSL VM をポート転送する。** ミラーモードでは自動です。NAT モードでは、管理者 PowerShell で、ポートごとに自分で行う必要があります:

   ```powershell
   # WSL VM の現在の IP を取得（NAT では WSL 再起動のたびに変わります）
   $wslIp = (wsl hostname -I).Trim().Split(' ')[0]

   # Windows ポート 8080 → WSL:8080 を転送
   netsh interface portproxy add v4tov4 `
     listenaddress=0.0.0.0 listenport=8080 `
     connectaddress=$wslIp connectport=8080

   # Windows ファイアウォールを通過させる
   New-NetFirewallRule -DisplayName "Hermes WSL 8080" `
     -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
   ```

   後で `netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080` で削除します。

3. **LAN デバイスを `http://<windows-lan-ip>:8080` に向ける。**

NAT モードでは WSL VM の IP が再起動のたびに変わるため、1 回限りのルールは次の `wsl --shutdown` までしか持続しません。永続的なものには、ミラーモードを使うか、ポートプロキシ手順を Windows ログイン時に実行されるスクリプトに入れてください。

クラウドメッセージングプロバイダーからの Webhook（Telegram `setWebhook`、Slack イベントなど）には、ポート転送と格闘せず `cloudflared` トンネルを使ってください。[Webhook ガイド](/docs/user-guide/messaging/webhooks)を参照してください。

## Windows 上で Hermes サービスを長期的に実行する

Hermes の [Tool Gateway](/docs/user-guide/features/tool-gateway) と API サーバーは長時間稼働するプロセスです。WSL2 では、これらを起動し続けるためにいくつかの選択肢があります。

### systemd を使って WSL 内で（推奨）

上記のセットアップセクションどおりに systemd を有効にした場合、`hermes gateway` と API サーバーは、任意の Linux マシンと同じように動作します。ゲートウェイのセットアップウィザードを使います:

```bash
hermes gateway setup
```

WSL の起動時にゲートウェイが自動的に立ち上がるよう、systemd ユーザーユニットのインストールを提案してくれます。

### Windows ログイン時に WSL 自体を起動する

WSL の VM は、何かがそれを使っている間だけ生き続けます。ターミナルウィンドウを開かずにゲートウェイを到達可能に保つには、タスクスケジューラ経由で Windows ログイン時に WSL プロセスを起動します:

- **トリガー:** ログオン時（あなたのユーザー）。
- **操作:** プログラムの開始
  - プログラム: `C:\Windows\System32\wsl.exe`
  - 引数: `-d Ubuntu --exec /bin/sh -c "sleep infinity"`

これで VM が生き続け、systemd 管理のゲートウェイが動き続けます。Windows 11 では、新しい `wsl --install --no-launch` + 自動起動フローも機能します。`sleep infinity` のトリックはポータブルな方法です。

## GPU パススルー（ローカルモデル）

WSL2 は WSL カーネル 5.10.43+ 以降、**NVIDIA** GPU をネイティブにサポートします。Windows に標準の NVIDIA ドライバをインストールしてください（WSL 内に Linux 版 NVIDIA ドライバをインストール**しないでください**）。すると WSL 内の `nvidia-smi` が GPU を認識します。そこから、CUDA ツールキット、`torch`、`vllm`、`sglang`、`llama-server` が通常どおり本物の GPU に対してビルドされます。

WSL2 内の AMD ROCm と Intel Arc のサポートはまだ発展途上で、Hermes のテストマトリクスの範囲外です。現在のドライバで動くかもしれませんが、推奨できるレシピはありません。

すでに Windows ドライバ経由で GPU を使う**Windows ネイティブ**のローカルモデルサーバー（Windows 版 Ollama、LM Studio）を実行している場合、WSL GPU パススルーはまったく必要ありません。上記のケース 1 に従い、WSL からネットワーク越しにそれに接続してください。

## よくある落とし穴

**Windows ホストの Ollama / LM Studio への「Connection refused」。**
[WSL2 ネットワーク](/docs/integrations/providers#wsl2-networking-windows-users)を参照してください。9 割方、サーバーが `127.0.0.1` にバインドされていて `0.0.0.0` が必要（Ollama: `OLLAMA_HOST=0.0.0.0`）か、ファイアウォールルールが欠けています。

**リポジトリ内での `git status` / `hermes chat` が極端に遅い。**
おそらく `/mnt/c/...` 配下で作業しています。リポジトリを `~/code/...`（Linux 側）に移動してください。桁違いに高速になります。

**スクリプトでの `bad interpreter: /bin/bash^M`。**
Windows エディタからの CRLF 改行コードです。`dos2unix script.sh` を実行し、WSL の git 設定で `core.autocrlf input` を設定してください。

**MCP 経由で起動した Windows バイナリからの「UNC paths are not supported」警告。**
Hermes の cwd が Linux ファイルシステム内にあり、Windows の `cmd.exe` がそれをどう扱えばよいか分かりません。そのセッションでは `/mnt/c/...` から Hermes を起動するか、Windows 実行ファイルを呼ぶ前に Windows から到達可能なパスへ `cd` するラッパーを使ってください。

**スリープ/休止後のクロックずれ。**
WSL2 のクロックは、ホストがスリープから復帰した後、数分ずれることがあり、証明書ベースのもの（OAuth、HTTPS API）が壊れます。必要に応じて修正します:

```bash
sudo hwclock -s
```

または `ntpdate` をインストールしてログイン時に実行します。

**ミラーモードを有効にした後、または VPN 接続時に DNS が機能しなくなる。**
ミラーモードはホストのネットワーク設定を WSL にプロキシします。Windows の DNS が変な状態（VPN スプリットトンネル、企業のリゾルバ）だと、WSL がそれを継承します。回避策: `resolv.conf` を手動で上書きします（`/etc/wsl.conf` で `generateResolvConf=false` を設定し、`1.1.1.1` または VPN の DNS を記述した独自の `/etc/resolv.conf` を書きます）。

**インストーラー実行後に `hermes` が見つからない。**
インストーラーは `~/.bashrc` 経由でシェルの PATH に `~/.local/bin` を追加します。現在のセッションで反映するには `source ~/.bashrc`（または新しいターミナルを開く）が必要です。

**Windows Defender が WSL ファイルで遅い。**
Defender は Windows からアクセスされたときに 9P ブリッジ経由でファイルをスキャンするため、`/mnt/c` 風の境界をまたぐアクセスの遅さを増幅します。WSL 内からのみ WSL ファイルに触れるなら、これは問題になりません。`\\wsl$\...` に対して Windows ツールを頻繁に使う場合は、WSL ディストリビューションのパスをリアルタイムスキャンから除外することを検討してください。

**ディスクが足りなくなる。**
WSL2 は VM ディスクを `%LOCALAPPDATA%\Packages\...` 配下のスパース VHDX として保存します。これは増えますが、ファイルを削除しても自動的には縮小しません。空き容量を取り戻すには: `wsl --shutdown` の後、管理者 PowerShell から `Optimize-VHD -Path <path-to-ext4.vhdx> -Mode Full` を実行します（Hyper-V ツールが必要）。または WSL ドキュメントに記載のより簡単な `diskpart` 経路を使います。

## 次に読むもの

- **[インストール](/docs/getting-started/installation)** — 実際のインストール手順（Linux/WSL2/Termux はすべて同じインストーラーを使います）。
- **[連携 → プロバイダー → WSL2 ネットワーク](/docs/integrations/providers#wsl2-networking-windows-users)** — ローカルモデルサーバー向けの標準的なネットワーク詳細解説。
- **[MCP ガイド → WSL → Windows Chrome](/docs/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)** — WSL 内の Hermes からログイン済みの Windows Chrome を制御する。
- **[Tool Gateway](/docs/user-guide/features/tool-gateway)** と **[Web ダッシュボード](/docs/user-guide/features/web-dashboard)** — WSL からネットワークの他の部分へ最も頻繁に公開したくなる、長時間稼働サービス。
