---
sidebar_position: 3
title: "FAQ・トラブルシューティング"
description: "Hermes Agent に関するよくある質問と一般的な問題の解決方法"
---

# FAQ・トラブルシューティング

最もよくある質問や問題に対する手早い回答と対処法をまとめています。

---

## よくある質問

### Hermes ではどの LLM プロバイダーが使えますか？

Hermes Agent は OpenAI 互換 API であればどれでも動作します。対応プロバイダーには次のものがあります。

- **[OpenRouter](https://openrouter.ai/)** — 1 つの API キーで何百ものモデルにアクセスできます（柔軟性の高さからおすすめです）
- **Nous Portal** — Nous Research 独自の推論エンドポイント
- **OpenAI** — GPT-5.4、GPT-5-codex、GPT-4.1、GPT-4o など
- **Anthropic** — Claude モデル（直接 API、`hermes login anthropic` による OAuth、OpenRouter、または互換プロキシ経由）
- **Google** — Gemini モデル（`gemini` プロバイダー経由の直接 API、`google-gemini-cli` OAuth プロバイダー、OpenRouter、または互換プロキシ経由）
- **z.ai / ZhipuAI** — GLM モデル
- **Kimi / Moonshot AI** — Kimi モデル
- **MiniMax** — グローバルおよび中国向けエンドポイント
- **ローカルモデル** — [Ollama](https://ollama.com/)、[vLLM](https://docs.vllm.ai/)、[llama.cpp](https://github.com/ggerganov/llama.cpp)、[SGLang](https://github.com/sgl-project/sglang)、その他 OpenAI 互換サーバー経由

プロバイダーは `hermes model` で設定するか、`~/.hermes/.env` を編集して設定します。すべてのプロバイダーキーについては [環境変数](./environment-variables.md) リファレンスを参照してください。

### Windows で動作しますか？

**ネイティブでは動作しません。** Hermes Agent は Unix 系の環境を必要とします。Windows では [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) をインストールし、その内部から Hermes を実行してください。標準のインストールコマンドは WSL2 で問題なく動作します。

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### WSL2 で Hermes を実行しています。通常の Windows 版 Chrome を操作するにはどうするのがベストですか？

`/browser connect` よりも MCP ブリッジを使うことをおすすめします。

推奨パターン:

- Hermes を WSL2 内で実行する
- Windows 上でサインイン済みの通常の Chrome をそのまま使い続ける
- `cmd.exe` または `powershell.exe` を通じて `chrome-devtools-mcp` を MCP サーバーとして追加する
- その結果得られる MCP ブラウザツールを Hermes に使わせる

これは、Hermes のコアブラウザトランスポートを WSL2/Windows の境界をまたいで直接アタッチさせようとするよりも信頼性が高い方法です。

参照:

- [Hermes で MCP を使う](../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)
- [ブラウザ自動化](../user-guide/features/browser.md#wsl2--windows-chrome-prefer-mcp-over-browser-connect)

### Android / Termux で動作しますか？

はい — Hermes には Android スマートフォン向けにテスト済みの Termux インストール手順が用意されています。

クイックインストール:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

完全に明示的な手動手順、対応する追加機能、現在の制限については [Termux ガイド](../getting-started/termux.md) を参照してください。

重要な注意点: `voice` 追加機能が `faster-whisper` → `ctranslate2` に依存しており、`ctranslate2` が Android 向けの wheel を公開していないため、現在 `.[all]` のフル追加機能は Android では利用できません。代わりにテスト済みの `.[termux]` 追加機能を使用してください。

### 自分のデータはどこかに送信されますか？

API 呼び出しは **設定したLLM プロバイダー（例: OpenRouter、ローカルの Ollama インスタンス）にのみ** 送信されます。Hermes Agent はテレメトリ、利用データ、アナリティクスを収集しません。会話、メモリ、スキルはすべて `~/.hermes/` 内にローカル保存されます。

### オフラインで / ローカルモデルで使えますか？

はい。`hermes model` を実行し、**Custom endpoint** を選択して、サーバーの URL を入力します。

```bash
hermes model
# 選択: Custom endpoint (enter URL manually)
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 32768   ← サーバーの実際のコンテキストウィンドウに合わせて設定してください
```

または `config.yaml` で直接設定します。

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

Hermes はエンドポイント、プロバイダー、ベース URL を `config.yaml` に永続化するため、再起動後も保持されます。ローカルサーバーに読み込まれているモデルがちょうど 1 つの場合、`/model custom` がそれを自動検出します。`config.yaml` に `provider: custom` を設定することもできます — これは独立した正式なプロバイダーであり、他の何かのエイリアスではありません。

これは Ollama、vLLM、llama.cpp server、SGLang、LocalAI などで動作します。詳細は [設定ガイド](../user-guide/configuration.md) を参照してください。

:::tip Ollama ユーザーの方へ
Ollama でカスタムの `num_ctx` を設定している場合（例: `ollama run --num_ctx 16384`）、Hermes 側でも一致するコンテキスト長を必ず設定してください — Ollama の `/api/show` はモデルの *最大* コンテキストを報告し、設定した実効的な `num_ctx` は報告しません。
:::

:::tip ローカルモデルでのタイムアウト
Hermes はローカルエンドポイントを自動検出し、ストリーミングのタイムアウトを緩和します（読み取りタイムアウトを 120 秒から 1800 秒に引き上げ、ストリームの停滞検出を無効化）。それでも非常に大きなコンテキストでタイムアウトが発生する場合は、`.env` に `HERMES_STREAM_READ_TIMEOUT=1800` を設定してください。詳細は [ローカル LLM ガイド](../guides/local-llm-on-mac.md#timeouts) を参照してください。
:::

### 費用はどのくらいかかりますか？

Hermes Agent 自体は **無料かつオープンソース**（MIT ライセンス）です。費用がかかるのは選択したプロバイダーの LLM API 利用分だけです。ローカルモデルの実行は完全に無料です。

### 複数人で 1 つのインスタンスを使えますか？

はい。[メッセージングゲートウェイ](../user-guide/messaging/index.md) を使うと、複数のユーザーが Telegram、Discord、Slack、WhatsApp、Home Assistant 経由で同じ Hermes Agent インスタンスとやり取りできます。アクセスは許可リスト（特定のユーザー ID）と DM ペアリング（最初にメッセージを送ったユーザーがアクセス権を取得）で制御されます。

### メモリとスキルの違いは何ですか？

- **メモリ** は **事実** を保存します — エージェントがあなた、あなたのプロジェクト、好みについて知っている事柄です。メモリは関連性に基づいて自動的に取り出されます。
- **スキル** は **手順** を保存します — 物事をどう行うかのステップバイステップの指示です。スキルは、エージェントが類似のタスクに遭遇したときに呼び出されます。

どちらもセッションをまたいで永続化されます。詳細は [メモリ](../user-guide/features/memory.md) と [スキル](../user-guide/features/skills.md) を参照してください。

### 自分の Python プロジェクトの中で使えますか？

はい。`AIAgent` クラスをインポートして、Hermes をプログラムから利用できます。

```python
from run_agent import AIAgent

agent = AIAgent(model="anthropic/claude-opus-4.7")
response = agent.chat("Explain quantum computing briefly")
```

完全な API の使い方については [Python ライブラリガイド](../user-guide/features/code-execution.md) を参照してください。

---

## トラブルシューティング

### インストールに関する問題

#### インストール後に `hermes: command not found` と表示される

**原因:** シェルが更新後の PATH を再読み込みしていません。

**解決策:**
```bash
# シェルプロファイルを再読み込みする
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# または新しいターミナルセッションを開始する
```

それでも動作しない場合は、インストール場所を確認してください。
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
インストーラーは `~/.local/bin` を PATH に追加します。標準的でないシェル設定を使っている場合は、`export PATH="$HOME/.local/bin:$PATH"` を手動で追加してください。
:::

#### Python のバージョンが古すぎる

**原因:** Hermes には Python 3.11 以降が必要です。

**解決策:**
```bash
python3 --version   # 現在のバージョンを確認

# 新しい Python をインストールする
sudo apt install python3.12   # Ubuntu/Debian
brew install python@3.12      # macOS
```

インストーラーはこれを自動的に処理します — 手動インストール中にこのエラーが表示された場合は、先に Python をアップグレードしてください。

#### ターミナルコマンドで `node: command not found`（または `nvm`、`pyenv`、`asdf` …）と表示される

**原因:** Hermes は起動時に一度だけ `bash -l` を実行して、セッションごとの環境スナップショットを構築します。bash のログインシェルは `/etc/profile`、`~/.bash_profile`、`~/.profile` を読み込みますが、**`~/.bashrc` は読み込みません** — そのため、そこに自身をインストールするツール（`nvm`、`asdf`、`pyenv`、`cargo`、カスタムの `PATH` エクスポートなど）はスナップショットから見えないままになります。これは、Hermes が systemd 配下や、対話的シェルプロファイルが事前に読み込まれていない最小限のシェルで実行されている場合に最もよく起こります。

**解決策:** Hermes はデフォルトで `~/.bashrc` を自動的に読み込みます。それでは不十分な場合 — 例えば PATH が `~/.zshrc` にある zsh ユーザーや、`nvm` をスタンドアロンファイルから初期化している場合 — `~/.hermes/config.yaml` に読み込む追加ファイルを列挙します。

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh ユーザー: zsh で管理される PATH を bash スナップショットに取り込む
    - ~/.nvm/nvm.sh                # nvm の直接初期化（シェルに関係なく動作）
    - /etc/profile.d/cargo.sh      # システム全体の rc ファイル
  # このリストを設定すると、デフォルトの ~/.bashrc 自動読み込みは追加されません —
  # 両方が必要な場合は明示的に含めてください:
  #   - ~/.bashrc
  #   - ~/.zshrc
```

存在しないファイルは黙ってスキップされます。読み込みは bash で行われるため、zsh 専用の構文に依存するファイルはエラーになる可能性があります — それが懸念される場合は、rc ファイル全体ではなく、PATH を設定する部分だけ（例: nvm の `nvm.sh` を直接）読み込んでください。

自動読み込みの動作を無効にする（厳密なログインシェルのセマンティクスのみにする）には:

```yaml
terminal:
  auto_source_bashrc: false
```

#### `uv: command not found`

**原因:** `uv` パッケージマネージャーがインストールされていないか、PATH にありません。

**解決策:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### インストール中の Permission denied エラー

**原因:** インストールディレクトリへの書き込み権限が不足しています。

**解決策:**
```bash
# インストーラーで sudo を使わないでください — ~/.local/bin にインストールされます
# 以前 sudo でインストールした場合は、後始末してください:
sudo rm /usr/local/bin/hermes
# その後、標準のインストーラーを再実行します
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

---

### プロバイダー・モデルに関する問題

#### `/model` に 1 つのプロバイダーしか表示されない / プロバイダーを切り替えられない

**原因:** `/model`（チャットセッション内）は **すでに設定済み** のプロバイダー間でしか切り替えられません。OpenRouter しか設定していない場合、`/model` にはそれだけが表示されます。

**解決策:** セッションを抜けて、ターミナルから `hermes model` を使って新しいプロバイダーを追加します。

```bash
# まず Hermes チャットセッションを抜ける（Ctrl+C または /quit）

# プロバイダーの完全なセットアップウィザードを実行する
hermes model

# これにより、プロバイダーの追加、OAuth の実行、API キーの入力、エンドポイントの設定ができます
```

`hermes model` で新しいプロバイダーを追加したら、新しいチャットセッションを開始してください — `/model` に設定済みのすべてのプロバイダーが表示されるようになります。

:::tip クイックリファレンス
| やりたいこと | 使うコマンド |
|-----------|-----|
| 新しいプロバイダーを追加する | `hermes model`（ターミナルから） |
| API キーを入力／変更する | `hermes model`（ターミナルから） |
| セッション途中でモデルを切り替える | `/model <name>`（セッション内） |
| 別の設定済みプロバイダーに切り替える | `/model provider:model`（セッション内） |
:::

#### API キーが機能しない

**原因:** キーが未設定、期限切れ、誤って設定されている、または別のプロバイダー用のものです。

**解決策:**
```bash
# 設定を確認する
hermes config show

# プロバイダーを再設定する
hermes model

# または直接設定する
hermes config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
キーがプロバイダーと一致していることを確認してください。OpenAI のキーは OpenRouter では機能せず、その逆も同様です。`~/.hermes/.env` に競合するエントリがないか確認してください。
:::

#### モデルが利用できない / モデルが見つからない

**原因:** モデル識別子が正しくないか、お使いのプロバイダーで利用できません。

**解決策:**
```bash
# プロバイダーで利用可能なモデルを一覧表示する
hermes model

# 有効なモデルを設定する
hermes config set HERMES_MODEL anthropic/claude-opus-4.7

# またはセッションごとに指定する
hermes chat --model openrouter/meta-llama/llama-3.1-70b-instruct
```

#### レート制限（429 エラー）

**原因:** プロバイダーのレート制限を超えています。

**解決策:** 少し待ってから再試行してください。継続的に利用する場合は、次を検討してください。
- プロバイダーのプランをアップグレードする
- 別のモデルやプロバイダーに切り替える
- `hermes chat --provider <alternative>` を使って別のバックエンドにルーティングする

#### コンテキスト長の超過

**原因:** 会話がモデルのコンテキストウィンドウに対して長くなりすぎたか、Hermes がモデルのコンテキスト長を誤検出しています。

**解決策:**
```bash
# 現在のセッションを圧縮する
/compress

# または新しいセッションを開始する
hermes chat

# より大きなコンテキストウィンドウを持つモデルを使う
hermes chat --model openrouter/google/gemini-3-flash-preview
```

最初の長い会話でこれが起こる場合、Hermes がモデルのコンテキスト長を誤って認識している可能性があります。検出された値を確認してください。

CLI の起動行を見てください — 検出されたコンテキスト長が表示されます（例: `📊 Context limit: 128000 tokens`）。セッション中に `/usage` で確認することもできます。

コンテキスト検出を修正するには、明示的に設定します。

```yaml
# ~/.hermes/config.yaml 内
model:
  default: your-model-name
  context_length: 131072  # モデルの実際のコンテキストウィンドウ
```

またはカスタムエンドポイントの場合は、モデルごとに追加します。

```yaml
custom_providers:
  - name: "My Server"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 32768
```

自動検出の仕組みとすべての上書きオプションについては [コンテキスト長の検出](../integrations/providers.md#context-length-detection) を参照してください。

---

### ターミナルに関する問題

#### コマンドが危険として遮断される

**原因:** Hermes が破壊的になりうるコマンド（例: `rm -rf`、`DROP TABLE`）を検出しました。これは安全機能です。

**解決策:** プロンプトが表示されたら、コマンドを確認し、承認するには `y` を入力します。次のこともできます。
- より安全な代替手段を使うようエージェントに依頼する
- 危険なパターンの完全な一覧を [セキュリティドキュメント](../user-guide/security.md) で確認する

:::tip
これは意図された動作です — Hermes が破壊的なコマンドを黙って実行することはありません。承認プロンプトには、何が実行されるかが正確に表示されます。
:::

#### メッセージングゲートウェイ経由で `sudo` が機能しない

**原因:** メッセージングゲートウェイは対話的ターミナルなしで動作するため、`sudo` がパスワードを要求できません。

**解決策:**
- メッセージングで `sudo` を避ける — 代替手段を見つけるようエージェントに依頼する
- どうしても `sudo` を使う必要がある場合は、`/etc/sudoers` で特定のコマンドに対してパスワード不要の sudo を設定する
- または管理タスクにはターミナルインターフェースに切り替える: `hermes chat`

#### Docker バックエンドが接続しない

**原因:** Docker デーモンが実行されていないか、ユーザーに権限がありません。

**解決策:**
```bash
# Docker が実行中か確認する
docker info

# ユーザーを docker グループに追加する
sudo usermod -aG docker $USER
newgrp docker

# 検証する
docker run hello-world
```

---

### メッセージングに関する問題

#### ボットがメッセージに応答しない

**原因:** ボットが実行されていない、認可されていない、またはユーザーが許可リストに含まれていません。

**解決策:**
```bash
# ゲートウェイが実行中か確認する
hermes gateway status

# ゲートウェイを起動する
hermes gateway start

# エラーがないかログを確認する
cat ~/.hermes/logs/gateway.log | tail -50
```

#### メッセージが配信されない

**原因:** ネットワークの問題、ボットトークンの期限切れ、またはプラットフォームの Webhook の設定ミスです。

**解決策:**
- `hermes gateway setup` でボットトークンが有効か確認する
- ゲートウェイのログを確認する: `cat ~/.hermes/logs/gateway.log | tail -50`
- Webhook ベースのプラットフォーム（Slack、WhatsApp）の場合は、サーバーが公開アクセス可能であることを確認する

#### 許可リストの混乱 — 誰がボットと話せるのか？

**原因:** 認可モードによって誰がアクセスできるかが決まります。

**解決策:**

| モード | 仕組み |
|------|-------------|
| **許可リスト** | 設定に列挙されたユーザー ID のみがやり取りできる |
| **DM ペアリング** | DM で最初にメッセージを送ったユーザーが排他的アクセス権を取得する |
| **オープン** | 誰でもやり取りできる（本番環境では非推奨） |

`~/.hermes/config.yaml` のゲートウェイ設定の下で構成します。[メッセージングドキュメント](../user-guide/messaging/index.md) を参照してください。

#### ゲートウェイが起動しない

**原因:** 依存関係の不足、ポートの競合、またはトークンの設定ミスです。

**解決策:**
```bash
# コアのメッセージングゲートウェイ依存関係をインストールする
pip install "hermes-agent[messaging]"  # Telegram、Discord、Slack、および共有ゲートウェイの依存関係

# ポートの競合を確認する
lsof -i :8080

# 設定を検証する
hermes config show
```

#### WSL: ゲートウェイが切断され続ける、または `hermes gateway start` が失敗する

**原因:** WSL の systemd サポートは信頼性が低いです。多くの WSL2 インストールでは systemd が有効になっておらず、有効にしていても、サービスが WSL の再起動や Windows のアイドルシャットダウンを生き延びられない場合があります。

**解決策:** systemd サービスの代わりにフォアグラウンドモードを使用します。

```bash
# オプション 1: フォアグラウンドで直接実行（最もシンプル）
hermes gateway run

# オプション 2: tmux で永続化（ターミナルを閉じても生き残る）
tmux new -s hermes 'hermes gateway run'
# 後で再アタッチ: tmux attach -t hermes

# オプション 3: nohup でバックグラウンド実行
nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &
```

それでも systemd を試したい場合は、有効になっていることを確認してください。

1. `/etc/wsl.conf` を開く（存在しない場合は作成する）
2. 次を追加する:
   ```ini
   [boot]
   systemd=true
   ```
3. PowerShell から: `wsl --shutdown`
4. WSL ターミナルを再度開く
5. 検証: `systemctl is-system-running` が "running" または "degraded" と表示されるはずです

:::tip Windows 起動時の自動開始
信頼性の高い自動開始のために、Windows タスクスケジューラを使ってログイン時に WSL とゲートウェイを起動します。
1. `wsl -d Ubuntu -- bash -lc 'hermes gateway run'` を実行するタスクを作成する
2. ユーザーログオン時にトリガーするよう設定する
:::

#### macOS: Node.js / ffmpeg / その他のツールがゲートウェイから見つからない

**原因:** launchd サービスは最小限の PATH（`/usr/bin:/bin:/usr/sbin:/sbin`）を継承し、これには Homebrew、nvm、cargo、その他ユーザーがインストールしたツールのディレクトリが含まれていません。これは WhatsApp ブリッジ（`node not found`）や音声文字起こし（`ffmpeg not found`）でよく問題になります。

**解決策:** ゲートウェイは `hermes gateway install` を実行したときにシェルの PATH をキャプチャします。ゲートウェイのセットアップ後にツールをインストールした場合は、インストールを再実行して更新後の PATH をキャプチャしてください。

```bash
hermes gateway install    # 現在の PATH を再スナップショットする
hermes gateway start      # 更新された plist を検出して再読み込みする
```

plist に正しい PATH が設定されているか検証できます。
```bash
/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:PATH" \
  ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

---

### パフォーマンスに関する問題

#### 応答が遅い

**原因:** 大きなモデル、遠い API サーバー、または多数のツールを含む重いシステムプロンプトです。

**解決策:**
- より高速／小型のモデルを試す: `hermes chat --model openrouter/meta-llama/llama-3.1-8b-instruct`
- 有効なツールセットを減らす: `hermes chat -t "terminal"`
- プロバイダーへのネットワークレイテンシを確認する
- ローカルモデルの場合は、十分な GPU VRAM があることを確認する

#### トークン使用量が多い

**原因:** 長い会話、冗長なシステムプロンプト、または多数のツール呼び出しによるコンテキストの蓄積です。

**解決策:**
```bash
# 会話を圧縮してトークンを削減する
/compress

# セッションのトークン使用量を確認する
/usage
```

:::tip
長いセッション中は `/compress` を定期的に使用してください。会話履歴を要約し、コンテキストを保ちながらトークン使用量を大幅に削減します。
:::

#### セッションが長くなりすぎる

**原因:** 長時間にわたる会話はメッセージとツール出力を蓄積し、コンテキスト制限に近づきます。

**解決策:**
```bash
# 現在のセッションを圧縮する（重要なコンテキストを保持）
/compress

# 古いセッションへの参照を持って新しいセッションを開始する
hermes chat

# 必要に応じて後で特定のセッションを再開する
hermes chat --continue
```

---

### MCP に関する問題

#### MCP サーバーが接続しない

**原因:** サーバーバイナリが見つからない、コマンドパスが間違っている、またはランタイムが不足しています。

**解決策:**
```bash
# MCP の依存関係がインストールされていることを確認する（標準インストールには含まれています）
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

# npm ベースのサーバーの場合は、Node.js が利用可能であることを確認する
node --version
npx --version

# サーバーを手動でテストする
npx -y @modelcontextprotocol/server-filesystem /tmp
```

`~/.hermes/config.yaml` の MCP 設定を確認してください。
```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
```

#### MCP サーバーからツールが表示されない

**原因:** サーバーは起動したがツールの検出に失敗した、設定によってツールがフィルタリングされた、またはサーバーが期待した MCP 機能に対応していません。

**解決策:**
- ゲートウェイ／エージェントのログで MCP 接続エラーを確認する
- サーバーが `tools/list` RPC メソッドに応答することを確認する
- そのサーバーの `tools.include`、`tools.exclude`、`tools.resources`、`tools.prompts`、`enabled` の設定を見直す
- リソース／プロンプトのユーティリティツールは、セッションが実際にそれらの機能に対応している場合にのみ登録されることを覚えておく
- 設定を変更したら `/reload-mcp` を使う

```bash
# MCP サーバーが設定されているか検証する
hermes config show | grep -A 12 mcp_servers

# 設定変更後は Hermes を再起動するか MCP を再読み込みする
hermes chat
```

あわせて参照:
- [MCP（Model Context Protocol）](/docs/user-guide/features/mcp)
- [Hermes で MCP を使う](/docs/guides/use-mcp-with-hermes)
- [MCP 設定リファレンス](/docs/reference/mcp-config-reference)

#### MCP タイムアウトエラー

**原因:** MCP サーバーの応答に時間がかかりすぎているか、実行中にクラッシュしました。

**解決策:**
- 対応している場合は、MCP サーバー設定でタイムアウトを増やす
- MCP サーバープロセスがまだ実行中か確認する
- リモートの HTTP MCP サーバーの場合は、ネットワーク接続を確認する

:::warning
MCP サーバーがリクエストの途中でクラッシュすると、Hermes はタイムアウトを報告します。根本原因を診断するには、Hermes のログだけでなくサーバー自身のログを確認してください。
:::

---

## プロファイル {#profiles}

### プロファイルは HERMES_HOME を設定するだけの場合とどう違うのですか？

プロファイルは `HERMES_HOME` の上に構築された管理レイヤーです。毎回コマンドの前に手動で `HERMES_HOME=/some/path` を設定することも *できます* が、プロファイルはすべての面倒な処理を肩代わりします。ディレクトリ構造の作成、シェルエイリアス（`hermes-work`）の生成、`~/.hermes/active_profile` でのアクティブなプロファイルの追跡、すべてのプロファイル間でのスキル更新の自動同期などです。また、タブ補完とも統合されているため、パスを覚えておく必要がありません。

### 2 つのプロファイルで同じボットトークンを共有できますか？

いいえ。各メッセージングプラットフォーム（Telegram、Discord など）は、ボットトークンへの排他的アクセスを必要とします。2 つのプロファイルが同じトークンを同時に使おうとすると、2 番目のゲートウェイは接続に失敗します。プロファイルごとに別々のボットを作成してください — Telegram の場合は [@BotFather](https://t.me/BotFather) に話しかけて追加のボットを作成します。

### プロファイルはメモリやセッションを共有しますか？

いいえ。各プロファイルには独自のメモリストア、セッションデータベース、スキルディレクトリがあります。これらは完全に分離されています。既存のメモリとセッションを持つ新しいプロファイルを開始したい場合は、`hermes profile create newname --clone-all` を使って現在のプロファイルからすべてをコピーします。

### `hermes update` を実行すると何が起こりますか？

`hermes update` は最新のコードを取得し、依存関係を **一度だけ**（プロファイルごとではなく）再インストールします。その後、更新されたスキルをすべてのプロファイルに自動的に同期します。`hermes update` は一度実行するだけでよく — そのマシン上のすべてのプロファイルをカバーします。


### いくつのプロファイルを実行できますか？

明確な上限はありません。各プロファイルは `~/.hermes/profiles/` 配下の単なるディレクトリです。実用上の上限は、ディスク容量と、システムが処理できる同時実行ゲートウェイの数（各ゲートウェイは軽量な Python プロセスです）に依存します。数十のプロファイルを実行しても問題ありません。アイドル状態の各プロファイルはリソースを消費しません。

---

## ワークフローとパターン

### タスクごとに異なるモデルを使う（マルチモデルワークフロー）

**シナリオ:** 普段は GPT-5.4 をメインに使っているが、ソーシャルメディアのコンテンツは Gemini や Grok の方がうまく書ける。毎回手動でモデルを切り替えるのは面倒です。

**解決策: 委任設定。** Hermes はサブエージェントを別のモデルに自動的にルーティングできます。これを `~/.hermes/config.yaml` に設定します。

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # サブエージェントはこのモデルを使う
  provider: "openrouter"                    # サブエージェント用のプロバイダー
```

これで、Hermes に「X についての Twitter スレッドを書いて」と伝えて `delegate_task` サブエージェントを生成すると、そのサブエージェントはメインモデルではなく Gemini で実行されます。メインの会話は GPT-5.4 のままです。

プロンプトの中で明示的に指定することもできます: *「製品ローンチについてのソーシャルメディア投稿を書くタスクを委任して。実際の執筆にはサブエージェントを使ってね。」* エージェントは `delegate_task` を使い、委任設定を自動的に拾い上げます。

委任を使わない一回限りのモデル切り替えには、CLI で `/model` を使います。

```bash
/model google/gemini-3-flash-preview    # このセッション用に切り替える
# ... コンテンツを書く ...
/model openai/gpt-5.4                   # 元に戻す
```

委任の仕組みの詳細については [サブエージェントへの委任](../user-guide/features/delegation.md) を参照してください。

### 1 つの WhatsApp 番号で複数のエージェントを実行する（チャットごとのバインド）

**シナリオ:** OpenClaw では、特定の WhatsApp チャットにバインドされた複数の独立したエージェントがありました — 1 つは家族の買い物リストグループ用、もう 1 つはプライベートチャット用。Hermes でこれはできますか？

**現在の制限:** Hermes のプロファイルはそれぞれ独自の WhatsApp 番号／セッションを必要とします。同じ WhatsApp 番号上の異なるチャットに複数のプロファイルをバインドすることはできません — WhatsApp ブリッジ（Baileys）は番号ごとに 1 つの認証済みセッションを使用します。

**回避策:**

1. **パーソナリティの切り替えで単一のプロファイルを使う。** 異なる `AGENTS.md` コンテキストファイルを作成するか、`/personality` コマンドを使ってチャットごとに振る舞いを変更します。エージェントは自分がどのチャットにいるかを認識し、それに適応できます。

2. **専門的なタスクには cron ジョブを使う。** 買い物リストのトラッカーには、特定のチャットを監視してリストを管理する cron ジョブを設定します — 別のエージェントは不要です。

3. **別々の番号を使う。** 真に独立したエージェントが必要な場合は、各プロファイルを独自の WhatsApp 番号とペアリングします。Google Voice などのサービスの仮想番号がこれに使えます。

4. **代わりに Telegram や Discord を使う。** これらのプラットフォームはチャットごとのバインドをより自然にサポートしています — 各 Telegram グループや Discord チャンネルが独自のセッションを持ち、同じアカウントで複数のボットトークン（プロファイルごとに 1 つ）を実行できます。

詳細は [プロファイル](../user-guide/profiles.md) と [WhatsApp セットアップ](../user-guide/messaging/whatsapp.md) を参照してください。

### Telegram に表示される内容を制御する（ログと推論を隠す）

**シナリオ:** 最終的な出力だけでなく、ゲートウェイの実行ログ、Hermes の推論、ツール呼び出しの詳細が Telegram に表示されてしまいます。

**解決策:** `config.yaml` の `display.tool_progress` 設定で、ツールの活動をどれだけ表示するかを制御します。

```yaml
display:
  tool_progress: "off"   # オプション: off、new、all、verbose
```

- **`off`** — 最終的な応答のみ。ツール呼び出し、推論、ログは表示されません。
- **`new`** — 新しいツール呼び出しが発生したときに表示します（簡潔な一行）。
- **`all`** — 結果を含むすべてのツール活動を表示します。
- **`verbose`** — ツールの引数と出力を含む完全な詳細を表示します。

メッセージングプラットフォームでは、通常 `off` または `new` が望ましいでしょう。`config.yaml` を編集した後は、変更を反映させるためにゲートウェイを再起動してください。

`/verbose` コマンドでセッションごとに切り替えることもできます（有効になっている場合）。

```yaml
display:
  tool_progress_command: true   # ゲートウェイで /verbose を有効にする
```

### Telegram でスキルを管理する（スラッシュコマンドの上限）

**シナリオ:** Telegram にはスラッシュコマンドが 100 個までという上限があり、スキルがそれを超えそうになっています。Telegram で不要なスキルを無効化したいのですが、`hermes skills config` の設定が反映されていないようです。

**解決策:** `hermes skills config` を使って、プラットフォームごとにスキルを無効化します。これは `config.yaml` に書き込まれます。

```yaml
skills:
  disabled: []                    # グローバルに無効化されたスキル
  platform_disabled:
    telegram: [skill-a, skill-b]  # telegram でのみ無効化
```

これを変更した後は、**ゲートウェイを再起動** してください（`hermes gateway restart`、または kill して再起動）。Telegram ボットのコマンドメニューは起動時に再構築されます。

:::tip
説明が非常に長いスキルは、ペイロードサイズの上限内に収めるため、Telegram メニューでは 40 文字に切り詰められます。スキルが表示されない場合、それは 100 コマンドの個数上限ではなく、合計ペイロードサイズの問題かもしれません — 使っていないスキルを無効化することは、どちらの問題にも役立ちます。
:::

### 共有スレッドセッション（複数ユーザー、1 つの会話）

**シナリオ:** 複数の人がボットにメンションする Telegram や Discord のスレッドがあります。そのスレッド内のすべてのメンションを、ユーザーごとに分かれたセッションではなく、1 つの共有された会話の一部にしたいです。

**現在の動作:** Hermes はほとんどのプラットフォームでユーザー ID をキーにしてセッションを作成するため、各人が独自の会話コンテキストを持ちます。これはプライバシーとコンテキストの分離のための意図的な設計です。

**回避策:**

1. **Slack を使う。** Slack のセッションはユーザーではなくスレッドをキーにしています。同じスレッド内の複数のユーザーが 1 つの会話を共有します — まさにあなたが説明している動作です。これが最も自然な選択肢です。

2. **単一のユーザーでグループチャットを使う。** 質問を中継する指定された「オペレーター」が 1 人いれば、セッションは統一されたままになります。他の人は読みながら追えます。

3. **Discord チャンネルを使う。** Discord のセッションはチャンネルをキーにしているため、同じチャンネル内のすべてのユーザーがコンテキストを共有します。共有会話用に専用のチャンネルを使ってください。

### Hermes を別のマシンにエクスポートする

**シナリオ:** 1 つのマシンでスキル、cron ジョブ、メモリを蓄積してきて、それらすべてを新しい専用の Linux マシンに移したいです。

**解決策:**

1. 新しいマシンに Hermes Agent をインストールします:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```

2. **元のマシン** で、完全なバックアップを作成します:
   ```bash
   hermes backup
   ```
   これにより、`~/.hermes/` ディレクトリ全体 — 設定、API キー、メモリ、スキル、セッション、プロファイル — の zip が作成され、ホームディレクトリに `~/hermes-backup-<timestamp>.zip` として保存されます。

3. zip を新しいマシンにコピーしてインポートします:
   ```bash
   # 元のマシンで
   scp ~/hermes-backup-<timestamp>.zip newmachine:~/

   # 新しいマシンで
   hermes import ~/hermes-backup-<timestamp>.zip
   ```

4. 新しいマシンで `hermes setup` を実行し、API キーとプロバイダー設定が機能していることを確認します。

### 単一のプロファイルを別のマシンに移す

**シナリオ:** インストール全体ではなく、特定の 1 つのプロファイルだけを移動または共有したいです。

```bash
# 元のマシンで
hermes profile export work ./work-backup.tar.gz

# ファイルを対象マシンにコピーしてから:
hermes profile import ./work-backup.tar.gz work
```

インポートされたプロファイルには、エクスポート元のすべての設定、メモリ、セッション、スキルが含まれます。新しいマシンのセットアップが異なる場合は、パスを更新したり、プロバイダーで再認証したりする必要があるかもしれません。

### `hermes backup` と `hermes profile export` の違い

| 機能 | `hermes backup` | `hermes profile export` |
| :--- | :--- | :--- |
| **用途** | **マシン全体の移行** | **特定プロファイルの移植／共有** |
| **範囲** | グローバル（`~/.hermes` ディレクトリ全体） | ローカル（単一のプロファイルディレクトリ） |
| **含まれるもの** | すべてのプロファイル、グローバル設定、API キー、セッション | 単一プロファイル: SOUL.md、メモリ、セッション、スキル |
| **認証情報** | **含まれる**（`.env` と `auth.json`） | **除外される**（安全な共有のため取り除かれる） |
| **形式** | `.zip` | `.tar.gz` |

**手動の代替手段（rsync）:** ファイルを直接コピーしたい場合は、コードリポジトリを除外します。
```bash
rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
```

:::tip
`hermes backup` は Hermes が実行中であっても一貫したスナップショットを生成します。復元されたアーカイブからは、`gateway.pid` や `cron.pid` のようなマシンローカルなランタイムファイルは除外されます。
:::

### インストール後にシェルを再読み込みすると Permission denied になる

**シナリオ:** Hermes インストーラーを実行した後、`source ~/.zshrc` で permission denied エラーが出ます。

**原因:** これは通常、`~/.zshrc`（または `~/.bashrc`）のファイル権限が正しくない場合や、インストーラーがそこにきれいに書き込めなかった場合に起こります。Hermes 固有の問題ではなく — シェル設定の権限の問題です。

**解決策:**
```bash
# 権限を確認する
ls -la ~/.zshrc

# 必要なら修正する（-rw-r--r-- または 644 であるべき）
chmod 644 ~/.zshrc

# その後、再読み込みする
source ~/.zshrc

# または新しいターミナルウィンドウを開くだけでも — PATH の変更が自動的に反映されます
```

インストーラーが PATH 行を追加したのに権限が正しくない場合は、手動で追加できます。
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

### 最初のエージェント実行で Error 400 が出る

**シナリオ:** セットアップは問題なく完了するのに、最初のチャットの試行が HTTP 400 で失敗します。

**原因:** 通常はモデル名の不一致です — 設定されたモデルがプロバイダーに存在しないか、API キーにそのモデルへのアクセス権がありません。

**解決策:**
```bash
# どのモデルとプロバイダーが設定されているか確認する
hermes config show | head -20

# モデル選択を再実行する
hermes model

# または既知の動作するモデルでテストする
hermes chat -q "hello" --model anthropic/claude-opus-4.7
```

OpenRouter を使っている場合は、API キーにクレジットがあることを確認してください。OpenRouter からの 400 は、多くの場合、モデルが有料プランを必要としているか、モデル ID にタイプミスがあることを意味します。

---

## それでも解決しませんか？

ここで扱われていない問題の場合:

1. **既存のイシューを検索する:** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **コミュニティに質問する:** [Nous Research Discord](https://discord.gg/nousresearch)
3. **バグレポートを提出する:** OS、Python のバージョン（`python3 --version`）、Hermes のバージョン（`hermes --version`）、および完全なエラーメッセージを記載してください
