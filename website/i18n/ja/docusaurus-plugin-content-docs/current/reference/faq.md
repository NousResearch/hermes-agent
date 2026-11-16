---
sidebar_position: 3
title: "FAQ・トラブルシューティング"
description: "Hermes Agent に関するよくある質問と一般的な問題の解決方法"
---

# FAQ・トラブルシューティング

このページの Python 依存関係のコマンドは、
[PM で準備したソースチェックアウト](/docs/reference/package-management#developer-workflow) を前提としています。
依存関係を変更した後は、チェックアウトを再アクティベートして Hermes を再起動してください。

最もよくある質問や問題に対する手早い回答と対処法をまとめています。

---

## よくある質問

### Hermes ではどの LLM プロバイダーが使えますか？

Hermes Agent は OpenAI 互換 API であればどれでも動作します。対応プロバイダーには次のものがあります。

- **[OpenRouter](https://openrouter.ai/)** — 1 つの API キーで何百ものモデルにアクセスできます（柔軟性の高さからおすすめです）
- **[Nous Portal](/docs/integrations/nous-portal)** — Nous Research のサブスクリプション型ゲートウェイ — 1 回の OAuth ログインで 300 以上のモデルに加えて Web／画像／TTS／ブラウザを利用できます（初めての方におすすめです）
- **OpenAI** — GPT-5.4、GPT-5-codex、GPT-4.1、GPT-4o など
- **Anthropic** — Claude モデル（直接 API、`hermes auth add anthropic` による OAuth、OpenRouter、または互換プロキシ経由）
- **Google** — Gemini モデル（`gemini` プロバイダー経由の直接 API、OpenRouter、または互換プロキシ経由）
- **z.ai / ZhipuAI** — GLM モデル
- **Kimi / Moonshot AI** — Kimi モデル
- **MiniMax** — グローバルおよび中国向けエンドポイント
- **ローカルモデル** — [Ollama](https://ollama.com/)、[vLLM](https://docs.vllm.ai/)、[llama.cpp](https://github.com/ggerganov/llama.cpp)、[SGLang](https://github.com/sgl-project/sglang)、その他 OpenAI 互換サーバー経由

プロバイダーは `hermes model` で設定するか、`~/.hermes/.env` を編集して設定します。すべてのプロバイダーキーについては [環境変数](./environment-variables.md) リファレンスを参照してください。

### Windows／Android／自分のプラットフォームで動作しますか？
プラットフォームごとの対応状況の一覧については **[プラットフォームサポート](/docs/getting-started/platform-support)** を参照してください。

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

### 自分のデータはどこかに送信されますか？ {#is-my-data-sent-anywhere}

API 呼び出しは **設定した LLM プロバイダー（例: OpenRouter、ローカルの Ollama インスタンス）にのみ** 送信されます。Hermes Agent はテレメトリ、利用データ、アナリティクスを収集しません。会話、メモリ、スキルはすべて `~/.hermes/` 内にローカル保存されます。

### オフラインで / ローカルモデルで使えますか？

はい。`hermes model` を実行し、**Custom endpoint** を選択して、サーバーの URL を入力します。

```bash
hermes model
# Select: Custom endpoint (enter URL manually)
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 64000   ← Hermes minimum; set this to match your server's actual context window
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
Ollama でカスタムの `num_ctx` を設定している場合（例: `ollama run --num_ctx 64000`）、Hermes 側でも一致するコンテキスト長を必ず設定してください — Ollama の `/api/show` はモデルの *最大* コンテキストを報告し、設定した実効的な `num_ctx` は報告しません。
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
# Reload your shell profile
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# Or start a new terminal session
```

それでも動作しない場合は、インストール場所を確認してください。
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
インストーラーは `~/.local/bin` を PATH に追加します。標準的でないシェル設定を使っている場合は、`export PATH="$HOME/.local/bin:$PATH"` を手動で追加してください。
:::

#### サポートされていない Python バージョン

現在のファーストパーティのインストールには、任意の新しいバージョンではなく **Python 3.14** が必要です。
`pyproject.toml` の `>=3.11,<3.15` という範囲は、古いインストールが 3.14 に切り替わる前に
アップデーターを実行できるようにするためのものであり、現在のランタイムが 3.11〜3.13 をサポートするという意味ではありません。
インストーラーとパッケージ化されたディストリビューションは、固定されたインタープリターを同梱しています。

手動でソース環境を構築する場合は、
[開発環境のセットアップ](/docs/developer-guide/contributing#development-setup) を使用してください。
インストール済みのアプリやコンテナ内のインタープリターを置き換えないでください。
管理されたインストールでエラーが出た場合は、`hermes doctor` を実行し、そのインストールの
[アップデート方法](../getting-started/updating.md) を使用してください。

#### ターミナルコマンドで `node: command not found`（または `nvm`、`pyenv`、`asdf` …）と表示される

**原因:** Hermes は起動時に一度だけ `bash -l` を実行して、セッションごとの環境スナップショットを構築します。bash のログインシェルは `/etc/profile`、`~/.bash_profile`、`~/.profile` を読み込みますが、**`~/.bashrc` は読み込みません** — そのため、そこに自身をインストールするツール（`nvm`、`asdf`、`pyenv`、`cargo`、カスタムの `PATH` エクスポートなど）はスナップショットから見えないままになります。これは、Hermes が systemd 配下や、対話的シェルプロファイルが事前に読み込まれていない最小限のシェルで実行されている場合に最もよく起こります。

**解決策:** Hermes はデフォルトで `~/.bashrc` を自動的に読み込みます。それでは不十分な場合 — 例えば PATH が `~/.zshrc` にある zsh ユーザーや、`nvm` をスタンドアロンファイルから初期化している場合 — `~/.hermes/config.yaml` に読み込む追加ファイルを列挙します。

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh users: pulls zsh-managed PATH into the bash snapshot
    - ~/.nvm/nvm.sh                # direct nvm init (works regardless of shell)
    - /etc/profile.d/cargo.sh      # system-wide rc files
  # When this list is set, the default ~/.bashrc auto-source is NOT added —
  # include it explicitly if you want both:
  #   - ~/.bashrc
  #   - ~/.zshrc
```

存在しないファイルは黙ってスキップされます。読み込みは bash で行われるため、zsh 専用の構文に依存するファイルはエラーになる可能性があります — それが懸念される場合は、rc ファイル全体ではなく、PATH を設定する部分だけ（例: nvm の `nvm.sh` を直接）読み込んでください。

初期化ファイルとは別に、すべてのターミナルコマンドの `PATH` は、標準のシステムディレクトリ（`/usr/local/bin`、`/opt/homebrew/bin` …）、Hermes が管理するランタイムディレクトリ、および存在する場合は `~/.local/bin`（`pip --user` / `pipx` / `uv tool` のインストール先）で補完されます — これらはあなた自身のエントリの後ろに追加されるため、優先順位は変わりません。これにより、最小限の非対話的な PATH で起動されるバックエンド（systemd、GUI ランチャー、Desktop の SSH リモートバックエンド）も、設定なしでカバーされます。

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
# Don't use sudo with the installer — it installs to ~/.local/bin
# If you previously installed with sudo, clean up:
sudo rm /usr/local/bin/hermes
# Then re-run the standard installer
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

---

### プロバイダー・モデルに関する問題

#### エージェントが「Hermes policy」や「Hermes guardrails」によってリクエストを拒否したと言う

モデルは、自分がなぜリクエストを拒否したのかを確実に特定することはできません。拒否がアシスタントの文章の中にしか現れない場合、隠れた Hermes のランタイムポリシーが原因だという主張は、幻覚による説明であるか、選択したモデルやプロバイダーが適用した制限である可能性があります。

Hermes による制御は明示的です。ブロックされたツール操作は、拒否されたコマンドやパスを示すツールエラーを返し、承認が必要な操作は承認プロンプトを表示します。Hermes がこれらの実行制御を、黙って一般的なコンテンツ拒否レイヤーに変えることはありません。ただし、Amazon Bedrock Guardrails のように、設定されている場合はプロバイダーレベルの制御が適用されることがあります。

原因を切り分けるには:

1. `/status` を実行して、有効なモデルとプロバイダーを確認します。
2. 拒否に実際の Hermes のツールエラーや承認プロンプトが含まれているか確認します。文章だけの場合は、モデルによる原因の説明をランタイムの証拠として扱わないでください。
3. 新しいセッションで、別の設定済みモデルやプロバイダーを使って再試行します。モデルによって変わる拒否は、モデル／プロバイダーの挙動であり、Hermes の実行制御ではありません。
4. 明示的なツールエラーが表示された場合は、問題を報告する際にその正確なテキストを使ってください。

Hermes のドキュメント化された実行制御については [セキュリティ](../user-guide/security.md) を、プロバイダーの設定については [プロバイダー](../integrations/providers.md) を参照してください。

#### 「…refused this request because of a policy on your account」

**意味:** プロバイダーが、再試行しても変わらないアカウントレベルの理由でリクエストを拒否しました — アグリゲーターのデータ／プライバシー設定によってそのモデルのすべてのエンドポイントが除外されたか、モデルの上流プロバイダーがアカウントをブロックしています（例: `this user has been blocked for a previous policy violation`。OpenRouter は、これを本来は成功した HTTP 200 ストリームの中で中継することがあります）。Hermes はリクエストを 1 回だけ送信し、再試行や認証情報のローテーションは行わず、フォールバックチェーンが設定されていればそちらに移ります。

**解決策:** 応答に示されたプロバイダーで、アカウントのステータスとデータ／プライバシー設定を確認するか、`/model` で別のモデルやプロバイダーに切り替えてください。`hermes fallback add` を使うと、今後のブロックを自動的にバックアップへルーティングできます。

#### 「Could not open a stream to `<host>` after N attempts (request X KB)」

**意味:** そのエンドポイントへの接続試行が、ストリームイベントが 1 つも届く前にすべて失敗したため、課金は発生していません。通常の再試行／フォールバックチェーンはその後も実行されます。この行には、実際に接続したホスト、試行回数、シリアライズされたリクエストサイズが示されます — これらは障害とリクエストサイズ制限を区別するための 3 つの手がかりです。

**解決策:** リクエストが大きく（数百 KB — 長いコーディングセッションではコンテキストが増えるとこの大きさに達します）、短い新規チャットは動作する場合、エンドポイントやその前段のプロキシがそのサイズのボディを拒否している可能性が高いです。ボディサイズの上限を引き上げるか、`/compress` を実行してコンテキストを縮小してください。リクエストが小さい場合は、エンドポイントに到達できていません — `base_url` を確認してから `/retry` で再試行してください。`logs/agent.log` には各試行の例外チェーンが記録されます。

#### メッセージングでの応答:「interrupted mid-request」と「not running or is unreachable」と「could not reach」の違い

チャット画面（Telegram、Discord、Slack など）には、生のトランスポート例外は表示されません。ゲートウェイはそれを 3 つの短い応答のいずれかに対応付けており、その違いからどこを確認すべきかがわかります。

| 応答 | 何が起きたか | 対処法 |
|---|---|---|
| 「The connection to the AI model service was **interrupted mid-request** — usually transient.」 | 確立済みの接続が切断されました（`Connection reset by peer`、EOF、`RemoteProtocolError`）。エンドポイントは接続に応答したので、実行中です。 | `/retry`。大きなリクエストで繰り返し発生する場合は、上記の「stream」の項目を参照してください。 |
| 「The AI model service isn't reachable right now — the configured model endpoint is **not running or is unreachable**.」 | 接続を受け付けるものがありませんでした（`Connection refused`、ホストへのルートなし、DNS の失敗）。 | モデルサーバーを起動する／`base_url` を確認してから `/retry`。ホスト上で `hermes doctor` を実行します。 |
| 「Hermes **could not reach** the AI model service (no further detail from the SDK).」 | SDK が汎用的な `APIConnectionError` を報告し、原因を保持していませんでした。上記のどちらとも断定できません。 | `/retry`。解消しない場合は `hermes doctor`。生の例外は `hermes logs` で確認できます。 |

#### `/model` に 1 つのプロバイダーしか表示されない / プロバイダーを切り替えられない

**原因:** `/model`（チャットセッション内）は **すでに設定済み** のプロバイダー間でしか切り替えられません。OpenRouter しか設定していない場合、`/model` にはそれだけが表示されます。

**解決策:** セッションを抜けて、ターミナルから `hermes model` を使って新しいプロバイダーを追加します。

```bash
# Exit the Hermes chat session first (Ctrl+C or /quit)

# Run the full provider setup wizard
hermes model

# This lets you: add providers, run OAuth, enter API keys, configure endpoints
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
# Check your configuration
hermes config show

# Re-configure your provider
hermes model

# Or set directly
hermes config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
キーがプロバイダーと一致していることを確認してください。OpenAI のキーは OpenRouter では機能せず、その逆も同様です。`~/.hermes/.env` に競合するエントリがないか確認してください。
:::

#### モデルが利用できない / モデルが見つからない

**原因:** モデル識別子が正しくないか、お使いのプロバイダーで利用できません。

**解決策:**
```bash
# List available models for your provider
hermes model

# Set a valid model
hermes config set model.default anthropic/claude-opus-4.7

# Or specify per-session
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
# Compress the current session
/compress

# Or start a fresh session
hermes chat

# Use a model with a larger context window
hermes chat --model openrouter/google/gemini-3-flash-preview
```

最初の長い会話でこれが起こる場合、Hermes がモデルのコンテキスト長を誤って認識している可能性があります。検出された値を確認してください。

CLI の起動行を見てください — 検出されたコンテキスト長が表示されます（例: `📊 Context limit: 128000 tokens`）。セッション中に `/usage` で確認することもできます。

**エラーを返さずに沈黙してしまうローカルサーバー（llama.cpp、Ollama）:** プロバイダーがリクエストを大きすぎるとして拒否すると、Hermes は会話を圧縮してリクエストを再構築します。Hermes は再試行の前に、再構築したリクエスト *全体*（システムプロンプト + ツールスキーマ + メッセージ）を測り直し、まだしきい値を超えている場合は上限付きの圧縮パスをさらに実行します。それでもリクエストが収まらない場合、llama.cpp が黙って切り詰めてしまう（サーバーログに `stop processing: n_tokens = 65535, truncated = 1` と出る）ような大きすぎるリクエストを送るのではなく、`Context length exceeded: compression could not reduce the rebuilt request below the safe threshold` でターンを終了します。このメッセージが出た場合、対処法はほぼ必ず上記の `context_length` の設定です。サーバーの実際の `-c` / `--ctx-size` に一致させてください。

**「The model server rejected this request as too large, but this conversation is only about N tokens…」:** ローカルサーバー（localhost、LAN、Tailscale）が測定値を示さずに「context exceeded」と返したものの、Hermes 自身によるリクエストの見積もりが、そのモデルについて把握しているウィンドウを大きく下回っている状態です — そのため Hermes は圧縮を **行わず**、会話のせいにもせず、ターンは再試行可能なままになります。シングルスロットのローカルサーバー（LM Studio、Ollama）では、これはほぼ必ず、その時点で別のリクエストがサーバーのコンテキストを占有していることが原因です — 典型的には、以前のセッションによるバックグラウンドのメモリレビュー（`logs/agent.log` 内の `thread=bg-review`）です。少し待ってから `/retry` してください。他の Hermes プロセスが動いていないのに繰り返し発生する場合、サーバーは Hermes が想定するよりも小さいウィンドウでモデルを読み込んでいます。サーバーのコンテキスト設定を引き上げるか、`model.context_length` をそれに合わせて下げてください。ホスト型プロバイダーでは、Claude サブスクリプションの DirectSDK プラグインのような外部プロセス型プロバイダーも含め、このメッセージは表示されません。それらには待つべき共有スロットがないため、同じ拒否はそのルートの実際のウィンドウが Hermes の想定より小さいことを意味し、Hermes は代わりに圧縮して再試行します。

コンテキスト検出を修正するには、明示的に設定します。

```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  context_length: 131072  # your model's actual context window
```

またはカスタムエンドポイントの場合は、プロバイダーのエントリにモデルごとに追加します。

```yaml
providers:
  my-server:
    api: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
```

（古い設定では従来の `custom_providers:` リストが使われています — 引き続きサポートされており、`providers:` に自動移行されます。）

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
# Check Docker is running
docker info

# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run hello-world
```

---

### メッセージングに関する問題

#### ボットがメッセージに応答しない

**原因:** ボットが実行されていない、認可されていない、またはユーザーが許可リストに含まれていません。

**解決策:**
```bash
# Check if the gateway is running
hermes gateway status

# Start the gateway
hermes gateway start

# Check logs for errors
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
# Install core messaging gateway dependencies
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['messaging'], explicit=True)"  # Telegram, Discord, Slack, and shared gateway deps

# Check for port conflicts
lsof -i :8080

# Verify configuration
hermes config show
```

#### WSL: ゲートウェイが切断され続ける、または `hermes gateway start` が失敗する {#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails}

**原因:** WSL の systemd サポートは信頼性が低いです。多くの WSL2 インストールでは systemd が有効になっておらず、有効にしていても、サービスが WSL の再起動や Windows のアイドルシャットダウンを生き延びられない場合があります。

**解決策:** systemd サービスの代わりにフォアグラウンドモードを使用します。

```bash
# Option 1: Direct foreground (simplest)
hermes gateway run

# Option 2: Persistent via tmux (survives terminal close)
tmux new -s hermes 'hermes gateway run'
# Reattach later: tmux attach -t hermes

# Option 3: Background via nohup
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
hermes gateway install    # Re-snapshots your current PATH
hermes gateway start      # Detects the updated plist and reloads
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
# See exactly what the fixed prompt costs — breakdown by block
# (system prompt, skills index, memory, tool schemas). Runs offline.
hermes prompt-size

# Compress the conversation to reduce tokens
/compress

# Check session token usage
/usage
```

まだ何も入力していないのにベースラインが高く見える場合、それは固定のプロンプト予算 — 毎回の呼び出しで送信されるシステムプロンプトとツールスキーマ — です。[`hermes prompt-size`](./cli-commands.md#hermes-prompt-size) を実行して測定し、削減してください。使わないツールセットを無効化し（`hermes tools`）、不要なスキルをアンインストールまたは無効化します（`hermes skills`）。

:::tip
長いセッション中は `/compress` を定期的に使用してください。会話履歴を要約し、コンテキストを保ちながらトークン使用量を大幅に削減します。
:::

#### セッションが長くなりすぎる

**原因:** 長時間にわたる会話はメッセージとツール出力を蓄積し、コンテキスト制限に近づきます。

**解決策:**
```bash
# Compress current session (preserves key context)
/compress

# Start a new session with a reference to the old one
hermes chat

# Resume a specific session later if needed
hermes chat --continue
```

---

### MCP に関する問題

#### MCP サーバーが接続しない

**原因:** サーバーバイナリが見つからない、コマンドパスが間違っている、またはランタイムが不足しています。

**解決策:**
```bash
# Ensure MCP dependencies are installed (already included in standard install)
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['mcp'], explicit=True)"

# For npm-based servers, ensure Node.js is available
node --version
npx --version

# Test the server manually
npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir
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
# Verify MCP servers are configured
hermes config show | grep -A 12 mcp_servers

# Restart Hermes or reload MCP after config changes
hermes chat
```

あわせて参照:
- [MCP（Model Context Protocol）](../user-guide/features/mcp.md)
- [Hermes で MCP を使う](../guides/use-mcp-with-hermes.md)
- [MCP 設定リファレンス](./mcp-config-reference.md)

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

### スキルに関する問題

#### デスクトップアプリで Skills Hub のページが読み込まれない（403 / ブロック）

**原因:** ドキュメントサイト（`hermes-agent.nousresearch.com`）は Vercel 経由で配信されており、Vercel の WAF は問題があると判定した一部の住宅用 IP レンジを拒否します。お使いのネットワークがそのようなレンジにある場合、このドメインへのすべてのリクエストが 403 のブロックページを返します。

**解決策:** Skills Hub のピッカーはプライマリドメインを確認し、同じカタログを配信している同等の GitHub Pages デプロイ（`nousresearch.github.io/hermes-agent`）へ自動的にフォールバックします。両方のオリジンでページが読み込めない場合は、プロキシ、DNS フィルター、ファイアウォールが両方のホストをブロックしていないか確認してください — また、デプロイ側で確認できるよう、影響を受けるレンジをメンテナーに報告してください。

---

## プロファイル {#profiles}

### プロファイルは HERMES_HOME を設定するだけの場合とどう違うのですか？

プロファイルは `HERMES_HOME` の上に構築された管理レイヤーです。毎回コマンドの前に手動で `HERMES_HOME=/some/path` を設定することも *できます* が、プロファイルはすべての面倒な処理を肩代わりします。ディレクトリ構造の作成、シェルエイリアス（`hermes-work`）の生成、`~/.hermes/active_profile` でのアクティブなプロファイルの追跡、すべてのプロファイル間でのスキル更新の自動同期などです。また、タブ補完とも統合されているため、パスを覚えておく必要がありません。

### 2 つのプロファイルで同じボットトークンを共有できますか？

いいえ。各メッセージングプラットフォーム（Telegram、Discord など）は、ボットトークンへの排他的アクセスを必要とします。2 つのプロファイルが同じトークンを同時に使おうとすると、2 番目のゲートウェイは接続に失敗します。プロファイルごとに別々のボットを作成してください — Telegram の場合は [@BotFather](https://t.me/BotFather) に話しかけて追加のボットを作成します。

### プロファイルはメモリやセッションを共有しますか？

いいえ。各プロファイルには独自のメモリストア、セッションデータベース、スキルディレクトリがあります。これらは完全に分離されています。既存のメモリとセッションを持つ新しいプロファイルを開始したい場合は、`hermes profile create newname --clone-all` を使って現在のプロファイルからすべてをコピーするか、`--clone-from <profile>` を追加して特定のプロファイルからコピーします。

この分離は、*同じ* プロファイルや Hermes ホームに対して 2 つのエージェントを決して実行してはいけない理由でもあります。どちらもメモリを自動的に書き込み、セッション開始時に互いの書き込みを読み込むため、保存された状態がセッションごとに劣化していきます。プロファイル 1 つにつきエージェントは 1 つにしてください。エージェント間で本当にメモリを共有したい場合は、[外部メモリプロバイダー](../user-guide/features/memory-providers.md) を使ってください。

### `hermes update` を実行すると何が起こりますか？

`hermes update` は最新のコードを取得し、依存関係を **一度だけ**（プロファイルごとではなく）再インストールします。その後、更新されたスキルをすべてのプロファイルに自動的に同期します。`hermes update` は一度実行するだけでよく — そのマシン上のすべてのプロファイルをカバーします。


### いくつのプロファイルを実行できますか？

明確な上限はありません。各プロファイルは `~/.hermes/profiles/` 配下のディレクトリで、少なくとも 1 つの識別ファイル（`config.yaml`、`.env`、`SOUL.md`、`profile.yaml`、`auth.json`、`state.db` のいずれか）を持ちます。それを持たない空のディレクトリ（ログローテーションや cron の実行の残骸）はプロファイルではありません — 一覧にも表示されず、提供もされず、`-p <name>` は存在しないと報告し、`hermes profile create <name>` はそのディレクトリを移動または削除するまで上書きを拒否します。実用上の上限は、ディスク容量と、システムが処理できる同時実行ゲートウェイの数（各ゲートウェイは軽量な Python プロセスです）に依存します。数十のプロファイルを実行しても問題ありません。アイドル状態の各プロファイルはリソースを消費しません。

---

## ワークフローとパターン

### タスクごとに異なるモデルを使う（マルチモデルワークフロー）

**シナリオ:** 普段は GPT-5.4 をメインに使っているが、ソーシャルメディアのコンテンツは Gemini や Grok の方がうまく書ける。毎回手動でモデルを切り替えるのは面倒です。

**解決策: 委任設定。** Hermes はサブエージェントを別のモデルに自動的にルーティングできます。これを `~/.hermes/config.yaml` に設定します。

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # subagents use this model
  provider: "openrouter"                    # provider for subagents
```

これで、Hermes に「X についての Twitter スレッドを書いて」と伝えて `delegate_task` サブエージェントを生成すると、そのサブエージェントはメインモデルではなく Gemini で実行されます。メインの会話は GPT-5.4 のままです。

プロンプトの中で明示的に指定することもできます: *「製品ローンチについてのソーシャルメディア投稿を書くタスクを委任して。実際の執筆にはサブエージェントを使ってね。」* エージェントは `delegate_task` を使い、委任設定を自動的に拾い上げます。

委任を使わない一回限りのモデル切り替えには、CLI で `/model` を使います。

```bash
/model google/gemini-3-flash-preview    # switch for this session
# ... write your content ...
/model openai/gpt-5.4                   # switch back
```

:::warning
`/model` で切り替えるたびにプロンプトキャッシュがリセットされます — キャッシュキーにはモデルが含まれるため、切り替え後の最初のメッセージでは会話全体が通常の入力料金で再度読み込まれます。長いセッションでは、何度も行ったり来たり切り替えるよりも、委任（サブエージェントは独自の新しいコンテキストを持ちます）や新しいセッションを使うことをおすすめします。
:::

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
  tool_progress: "off"   # options: off, new, all, verbose
```

- **`off`** — 最終的な応答のみ。ツール呼び出し、推論、ログは表示されません。
- **`new`** — 新しいツール呼び出しが発生したときに表示します（簡潔な一行）。
- **`all`** — 結果を含むすべてのツール活動を表示します。
- **`verbose`** — ツールの引数と出力を含む完全な詳細を表示します。

メッセージングプラットフォームでは、通常 `off` または `new` が望ましいでしょう。`config.yaml` を編集した後は、変更を反映させるためにゲートウェイを再起動してください。

`/verbose` コマンドでセッションごとに切り替えることもできます（有効になっている場合）。

```yaml
display:
  tool_progress_command: true   # enables /verbose in the gateway
```

### Telegram でスキルを管理する（スラッシュコマンドの上限）

**シナリオ:** Telegram にはスラッシュコマンドが 100 個までという上限があり、スキルがそれを超えそうになっています。Telegram で不要なスキルを無効化したいのですが、`hermes skills config` の設定が反映されていないようです。

**解決策:** `hermes skills config` を使って、プラットフォームごとにスキルを無効化します。これは `config.yaml` に書き込まれます。

```yaml
skills:
  disabled: []                    # globally disabled skills
  platform_disabled:
    telegram: [skill-a, skill-b]  # disabled only on telegram
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

### Hermes を別のマシンにエクスポートする {#exporting-hermes-to-another-machine}

**シナリオ:** 1 つのマシンでスキル、cron ジョブ、メモリを蓄積してきて、それらすべてを新しい専用の Linux マシンに移したいです。

**解決策:**

1. 新しいマシンに Hermes Agent をインストールします:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

2. **元のマシン** で、完全なバックアップを作成します:
   ```bash
   hermes backup
   ```
   これにより、zip アーカイブが `~/hermes-backup-<timestamp>.zip` に保存されます。
   完全なバックアップには、Hermes のデータルート配下の設定、認証情報、メモリ、スキル、セッション、
   プロファイルが含まれます。アプリケーションやランタイムのイメージではありません。

3. zip を新しいマシンにコピーしてインポートします:
   ```bash
   # On the source machine
   scp ~/hermes-backup-<timestamp>.zip newmachine:~/

   # On the new machine
   hermes import ~/hermes-backup-<timestamp>.zip
   ```

4. 新しいマシンで `hermes setup` を実行し、API キーとプロバイダー設定が機能していることを確認します。

### 単一のプロファイルを別のマシンに移す {#moving-a-single-profile-to-another-machine}

**シナリオ:** インストール全体ではなく、特定の 1 つのプロファイルだけを移動または共有したいです。

```bash
# On the source machine
hermes profile export work ./work-backup.tar.gz

# Copy the file to the target machine, then:
hermes profile import ./work-backup.tar.gz work
```

インポートされたプロファイルには、エクスポート元のすべての設定、メモリ、セッション、スキルが含まれます。新しいマシンのセットアップが異なる場合は、パスを更新したり、プロバイダーで再認証したりする必要があるかもしれません。

### `hermes backup` と `hermes profile export` の違い {#hermes-backup-vs-hermes-profile-export}

| 機能 | `hermes backup` | `hermes profile export` |
| :--- | :--- | :--- |
| **用途** | **マシン全体の移行** | **特定プロファイルの移植／共有** |
| **範囲** | Hermes のデータルート（下記の除外項目を除く） | 単一のプロファイルディレクトリ |
| **含まれるもの** | すべてのプロファイル、グローバル設定、API キー、セッション | 単一プロファイル: SOUL.md、メモリ、セッション、スキル |
| **認証情報** | **含まれる**（`.env` と `auth.json`） | **除外される**（安全な共有のため取り除かれる） |
| **形式** | `.zip` | `.tar.gz` |

完全なバックアップから除外されるもの:

- ソースチェックアウト、依存関係の環境、ダウンロードしたツール、モデル、ランタイム。
- ビルドキャッシュ、チェックポイント、以前のバックアップ、クイックスナップショット。
- ブラウザプロファイル（実際のブラウザの認証情報のコピーを含む）。
- バイトコード、SQLite のサイドカーファイル、`gateway.pid`、`cron.pid`、`.backup.lock`。

`hermes backup --quick` は、完全なアーカイブの代わりに選択した状態ファイルを保存します。
マシン移行前の完全なバックアップの代わりにはなりません。

完全なバックアップでは、コピーに失敗したファイルが報告されます。そのため、アーカイブが存在していても
データが欠けている場合があります。元のインストールを削除する前に、スキップされたファイルの報告を確認してください。
復元されたパッケージ宣言により、PM は依存関係を再度ダウンロードできます。バイトコードと
SQLite のサイドカーファイルはローカルで再生成されます。これらの除外によって、完全なバックアップから `.env` や
`auth.json` が取り除かれることはありません。

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
# Check permissions
ls -la ~/.zshrc

# Fix if needed (should be -rw-r--r-- or 644)
chmod 644 ~/.zshrc

# Then reload
source ~/.zshrc

# Or just open a new terminal window — it picks up PATH changes automatically
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
# Check what model and provider are configured
hermes config show | head -20

# Re-run model selection
hermes model

# Or test with a known-good model
hermes chat -q "hello" --model anthropic/claude-opus-4.7
```

OpenRouter を使っている場合は、API キーにクレジットがあることを確認してください。OpenRouter からの 400 は、多くの場合、モデルが有料プランを必要としているか、モデル ID にタイプミスがあることを意味します。

---

## それでも解決しませんか？

ここで扱われていない問題の場合:

1. **既存のイシューを検索する:** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **コミュニティに質問する:** [Nous Research Discord](https://discord.gg/nousresearch)
3. **バグレポートを提出する:** OS、Python のバージョン（`python3 --version`）、Hermes のバージョン（`hermes --version`）、および完全なエラーメッセージを記載してください
