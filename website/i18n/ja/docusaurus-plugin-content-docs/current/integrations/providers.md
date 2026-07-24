---
title: "AIプロバイダー"
sidebar_label: "AIプロバイダー"
sidebar_position: 1
---

# AIプロバイダー

このページでは、Hermes Agent向けの推論プロバイダーのセットアップを扱います。OpenRouterやAnthropicのようなクラウドAPI、OllamaやvLLMのようなセルフホスト型エンドポイント、さらに高度なルーティングやフォールバック設定までを網羅します。Hermesを使用するには、少なくとも1つのプロバイダーを設定する必要があります。

## 推論プロバイダー

LLMに接続する手段が少なくとも1つ必要です。`hermes model` を使ってプロバイダーとモデルを対話的に切り替えるか、直接設定してください:

| プロバイダー | セットアップ |
|----------|-------|
| **Nous Portal** | `hermes model`（OAuth、サブスクリプション制） |
| **OpenAI Codex** | `hermes model`（ChatGPT OAuth、Codexモデルを使用） |
| **GitHub Copilot** | `hermes model`（OAuthデバイスコードフロー、`COPILOT_GITHUB_TOKEN`、`GH_TOKEN`、または `gh auth token`） |
| **GitHub Copilot ACP** | `hermes model`（ローカルの `copilot --acp --stdio` を起動） |
| **Anthropic** | `hermes model`（OAuth経由のClaude Max + 追加利用クレジット。Anthropic APIキーや手動のsetup-tokenにも対応 — 下記の注記を参照） |
| **OpenRouter** | `~/.hermes/.env` 内の `OPENROUTER_API_KEY` |
| **AI Gateway** | `~/.hermes/.env` 内の `AI_GATEWAY_API_KEY`（プロバイダー: `ai-gateway`） |
| **z.ai / GLM** | `~/.hermes/.env` 内の `GLM_API_KEY`（プロバイダー: `zai`） |
| **Kimi / Moonshot** | `~/.hermes/.env` 内の `KIMI_API_KEY`（プロバイダー: `kimi-coding`） |
| **Kimi / Moonshot（中国）** | `~/.hermes/.env` 内の `KIMI_CN_API_KEY`（プロバイダー: `kimi-coding-cn`、エイリアス: `kimi-cn`、`moonshot-cn`） |
| **Arcee AI** | `~/.hermes/.env` 内の `ARCEEAI_API_KEY`（プロバイダー: `arcee`、エイリアス: `arcee-ai`、`arceeai`） |
| **GMI Cloud** | `~/.hermes/.env` 内の `GMI_API_KEY`（プロバイダー: `gmi`、エイリアス: `gmi-cloud`、`gmicloud`） |
| **MiniMax** | `~/.hermes/.env` 内の `MINIMAX_API_KEY`（プロバイダー: `minimax`） |
| **MiniMax China** | `~/.hermes/.env` 内の `MINIMAX_CN_API_KEY`（プロバイダー: `minimax-cn`） |
| **Alibaba Cloud** | `~/.hermes/.env` 内の `DASHSCOPE_API_KEY`（プロバイダー: `alibaba`） |
| **Alibaba Coding Plan** | `DASHSCOPE_API_KEY`（プロバイダー: `alibaba-coding-plan`、エイリアス: `alibaba_coding`） — 別個の課金SKUで、エンドポイントも異なります |
| **Kilo Code** | `~/.hermes/.env` 内の `KILOCODE_API_KEY`（プロバイダー: `kilocode`） |
| **Xiaomi MiMo** | `~/.hermes/.env` 内の `XIAOMI_API_KEY`（プロバイダー: `xiaomi`、エイリアス: `mimo`、`xiaomi-mimo`） |
| **Tencent TokenHub** | `~/.hermes/.env` 内の `TOKENHUB_API_KEY`（プロバイダー: `tencent-tokenhub`、エイリアス: `tencent`、`tokenhub`、`tencentmaas`） |
| **OpenCode Zen** | `~/.hermes/.env` 内の `OPENCODE_ZEN_API_KEY`（プロバイダー: `opencode-zen`） |
| **OpenCode Go** | `~/.hermes/.env` 内の `OPENCODE_GO_API_KEY`（プロバイダー: `opencode-go`） |
| **DeepSeek** | `~/.hermes/.env` 内の `DEEPSEEK_API_KEY`（プロバイダー: `deepseek`） |
| **Hugging Face** | `~/.hermes/.env` 内の `HF_TOKEN`（プロバイダー: `huggingface`、エイリアス: `hf`） |
| **Google / Gemini** | `~/.hermes/.env` 内の `GOOGLE_API_KEY`（または `GEMINI_API_KEY`）（プロバイダー: `gemini`） |
| **Google Gemini (OAuth)** | `hermes model` →「Google Gemini (OAuth)」（プロバイダー: `google-gemini-cli`、無料枠対応、ブラウザでのPKCEログイン） |
| **LM Studio** | `hermes model` →「LM Studio」（プロバイダー: `lmstudio`、任意で `LM_API_KEY`） |
| **Custom Endpoint** | `hermes model` →「Custom endpoint」を選択（`config.yaml` に保存） |

公式のAPIキー方式については、専用の [Google Geminiガイド](/docs/guides/google-gemini) を参照してください。

:::tip モデルキーのエイリアス
`model:` 設定セクションでは、モデルIDのキー名として `default:` または `model:` のどちらも使用できます。`model: { default: my-model }` と `model: { model: my-model }` はまったく同じように動作します。
:::


### OAuth経由のGoogle Gemini（`google-gemini-cli`）

`google-gemini-cli` プロバイダーは、GoogleのCloud Code Assistバックエンド（Google自身の `gemini-cli` ツールが使うのと同じAPI）を使用します。これは**無料枠**（個人アカウント向けの寛大な日次クォータ）と**有料枠**（GCPプロジェクト経由のStandard/Enterprise）の両方をサポートします。

**クイックスタート:**

```bash
hermes model
# →「Google Gemini (OAuth)」を選択
# → ポリシー警告を確認し、承認
# → ブラウザが accounts.google.com を開くのでサインイン
# → 完了 — Hermesが最初のリクエスト時に無料枠を自動プロビジョニングします
```

Hermesはデフォルトで、Googleの**公開**された `gemini-cli` デスクトップOAuthクライアント（Googleがオープンソースの `gemini-cli` に含めているのと同じ認証情報）を同梱しています。デスクトップOAuthクライアントは機密扱いではありません（セキュリティはPKCEが提供します）。`gemini-cli` をインストールしたり、独自のGCP OAuthクライアントを登録したりする必要はありません。

**認証の仕組み:**
- `accounts.google.com` に対するPKCE Authorization Codeフロー
- `http://127.0.0.1:8085/oauth2callback` でのブラウザコールバック（ポートが使用中の場合はエフェメラルポートにフォールバック）
- トークンは `~/.hermes/auth/google_oauth.json` に保存（chmod 0600、アトミック書き込み、プロセス間 `fcntl` ロック）
- 有効期限の60秒前に自動更新
- ヘッドレス環境（SSH、`HERMES_HEADLESS=1`）→ ペーストモードにフォールバック
- 進行中のリフレッシュの重複排除 — 2つの同時リクエストが二重にリフレッシュしません
- `invalid_grant`（リフレッシュトークンが失効）→ 認証情報ファイルが消去され、ユーザーに再ログインを促します

**推論の仕組み:**
- トラフィックは `https://cloudcode-pa.googleapis.com/v1internal:generateContent`
  （ストリーミングの場合は `:streamGenerateContent?alt=sse`）に送られ、有料の `v1beta/openai` エンドポイントには送られません
- リクエストボディは `{project, model, user_prompt_id, request}` でラップされます
- OpenAI形式の `messages[]`、`tools[]`、`tool_choice` は、Geminiネイティブの
  `contents[]`、`tools[].functionDeclarations`、`toolConfig` 形式に変換されます
- レスポンスはOpenAI形式に戻され、Hermesの他の部分は変更なく動作します

**ティアとプロジェクトID:**

| あなたの状況 | 対応 |
|---|---|
| 個人のGoogleアカウントで無料枠を使いたい | 何もせず — サインインしてチャットを開始 |
| Workspace / Standard / Enterpriseアカウント | `HERMES_GEMINI_PROJECT_ID` または `GOOGLE_CLOUD_PROJECT` にGCPプロジェクトIDを設定 |
| VPC-SCで保護された組織 | Hermesが `SECURITY_POLICY_VIOLATED` を検出し、自動的に `standard-tier` を強制 |

無料枠は初回使用時にGoogle管理のプロジェクトを自動プロビジョニングします。GCPのセットアップは不要です。

**クォータの監視:**

```
/gquota
```

モデルごとに残りのCode Assistクォータをプログレスバーで表示します:

```
Gemini Code Assist quota  (project: 123-abc)

  gemini-2.5-pro                      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░   85%
  gemini-2.5-flash [input]            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░   92%
```

:::warning ポリシー上のリスク
Googleは、Gemini CLIのOAuthクライアントをサードパーティのソフトウェアで使用することをポリシー違反とみなしています。一部のユーザーからはアカウント制限の報告があります。最もリスクの低い体験を求める場合は、代わりに `gemini` プロバイダー経由で独自のAPIキーを使用してください。HermesはOAuth開始前に事前警告を表示し、明示的な確認を求めます。
:::

**カスタムOAuthクライアント（任意）:**

独自のGoogle OAuthクライアントを登録したい場合 — 例えばクォータと同意を自分のGCPプロジェクトにスコープしたい場合 — 次を設定します:

```bash
HERMES_GEMINI_CLIENT_ID=your-client.apps.googleusercontent.com
HERMES_GEMINI_CLIENT_SECRET=...   # Desktopクライアントでは任意
```

[console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials) で、Generative Language APIを有効化した**Desktop app**のOAuthクライアントを登録してください。

:::info Codexに関する注記
OpenAI Codexプロバイダーはデバイスコードで認証します（URLを開いてコードを入力）。Hermesは得られた認証情報を `~/.hermes/auth.json` 配下の独自の認証ストアに保存し、`~/.codex/auth.json` が存在する場合は既存のCodex CLI認証情報をインポートできます。Codex CLIのインストールは不要です。
:::

:::warning
Nous Portal、Codex、カスタムエンドポイントを使用している場合でも、一部のツール（ビジョン、Web要約、MoA）は別個の「補助（auxiliary）」モデルを使用します。デフォルト（`auxiliary.*.provider: "auto"`）では、Hermesはこれらのタスクをあなたの**メインチャットモデル**（`hermes model` で選んだのと同じモデル）にルーティングします。各タスクを個別に上書きして、より安価／高速なモデル（例: OpenRouter上のGemini Flash）にルーティングできます — [補助モデル](/docs/user-guide/configuration#auxiliary-models) を参照してください。
:::

:::tip Nous Tool Gateway
有料のNous Portalサブスクライバーは、**[Tool Gateway](/docs/user-guide/features/tool-gateway)** へのアクセスも得られます — Web検索、画像生成、TTS、ブラウザ自動化がサブスクリプション経由でルーティングされます。追加のAPIキーは不要です。`hermes model` のセットアップ中に自動的に提案されるほか、後から `hermes tools` で有効化することもできます。
:::

### モデル管理のための2つのコマンド

Hermesには、異なる目的を持つ**2つ**のモデルコマンドがあります:

| コマンド | 実行場所 | 役割 |
|---------|-------------|--------------|
| **`hermes model`** | ターミナル（セッションの外） | 完全なセットアップウィザード — プロバイダーの追加、OAuthの実行、APIキーの入力、エンドポイントの設定 |
| **`/model`** | Hermesのチャットセッション内 | **すでに設定済み**のプロバイダーとモデルの間でのクイック切り替え |

まだセットアップしていないプロバイダーに切り替えようとしている場合（例: OpenRouterのみ設定済みでAnthropicを使いたい場合）は、`/model` ではなく `hermes model` が必要です。まずセッションを終了し（`Ctrl+C` または `/quit`）、`hermes model` を実行してプロバイダーのセットアップを完了し、新しいセッションを開始してください。

### Anthropic（ネイティブ）

OpenRouterのプロキシなしで、Anthropic API経由でClaudeモデルを直接使用します。3つの認証方法をサポートします:

:::caution Claude Maxの「追加利用」クレジットが必要
`hermes model` → Anthropic OAuth（または `hermes auth add anthropic --type oauth`）で認証すると、Hermesはあなたのアカウントに対してClaude Codeとしてルーティングします。**これはClaude Maxプランに加入し、追加利用クレジットを購入している場合にのみ動作します。** ベースのMaxプランの利用枠（Claude Codeにデフォルトで含まれる利用分）はHermesでは消費されません — 上乗せした追加／超過クレジットのみが消費されます。Claude Proのサブスクライバーはこの方式を利用できません。

Max + 追加クレジットがない場合は、代わりに `ANTHROPIC_API_KEY` を使用してください — リクエストはそのキーの組織に対して従量課金されます（標準的なAPI料金で、Claudeサブスクリプションとは独立しています）。
:::

```bash
# APIキーを使用（従量課金）
export ANTHROPIC_API_KEY=***
hermes chat --provider anthropic --model claude-sonnet-4-6

# 推奨: `hermes model` 経由で認証
# 利用可能な場合、HermesはClaude Codeの認証情報ストアを直接使用します
hermes model

# setup-tokenによる手動の上書き（フォールバック／レガシー）
export ANTHROPIC_TOKEN=***  # setup-tokenまたは手動のOAuthトークン
hermes chat --provider anthropic

# Claude Codeの認証情報を自動検出（すでにClaude Codeを使っている場合）
hermes chat --provider anthropic  # Claude Codeの認証情報ファイルを自動的に読み込みます
```

`hermes model` 経由でAnthropic OAuthを選択すると、Hermesはトークンを `~/.hermes/.env` にコピーするよりも、Claude Code自身の認証情報ストアを優先します。これにより、リフレッシュ可能なClaude認証情報がリフレッシュ可能なまま保たれます。

または恒久的に設定します:
```yaml
model:
  provider: "anthropic"
  default: "claude-sonnet-4-6"
```

:::tip エイリアス
`--provider claude` と `--provider claude-code` も `--provider anthropic` の省略形として機能します。
:::

### GitHub Copilot

HermesはGitHub Copilotをファーストクラスのプロバイダーとして、2つのモードでサポートします:

**`copilot` — 直接Copilot API**（推奨）。あなたのGitHub Copilotサブスクリプションを使って、Copilot API経由でGPT-5.x、Claude、Gemini、その他のモデルにアクセスします。

```bash
hermes chat --provider copilot --model gpt-5.4
```

**認証オプション**（この順序でチェックされます）:

1. `COPILOT_GITHUB_TOKEN` 環境変数
2. `GH_TOKEN` 環境変数
3. `GITHUB_TOKEN` 環境変数
4. `gh auth token` CLIフォールバック

トークンが見つからない場合、`hermes model` は**OAuthデバイスコードログイン**を提供します — Copilot CLIやopencodeで使われているのと同じフローです。

:::warning トークンの種類
Copilot APIは従来のPersonal Access Token（`ghp_*`）を**サポートしていません**。サポートされるトークンの種類:

| 種類 | プレフィックス | 取得方法 |
|------|--------|------------|
| OAuthトークン | `gho_` | `hermes model` → GitHub Copilot → Login with GitHub |
| Fine-grained PAT | `github_pat_` | GitHub Settings → Developer settings → Fine-grained tokens（**Copilot Requests** 権限が必要） |
| GitHub Appトークン | `ghu_` | GitHub Appのインストール経由 |

`gh auth token` が `ghp_*` トークンを返す場合は、代わりに `hermes model` でOAuth認証してください。
:::

:::info HermesにおけるCopilot認証の挙動
Hermesはサポートされるトークン（`gho_*`、`github_pat_*`、`ghu_*`）を `api.githubcopilot.com` に直接送信し、Copilot固有のヘッダー（`Editor-Version`、`Copilot-Integration-Id`、`Openai-Intent`、`x-initiator`）を含めます。

HTTP 401の場合、Hermesはフォールバック前にワンショットの認証情報リカバリーを実行します:

1. 通常の優先順位チェーン（`COPILOT_GITHUB_TOKEN` → `GH_TOKEN` → `GITHUB_TOKEN` → `gh auth token`）でトークンを再解決
2. リフレッシュされたヘッダーで共有のOpenAIクライアントを再構築
3. リクエストを1回リトライ

一部の古いコミュニティプロキシは `api.github.com/copilot_internal/v2/token` の交換フローを使用します。このエンドポイントは一部のアカウントタイプでは利用できない場合があります（404を返す）。そのためHermesは直接トークン認証を主要な経路として維持し、堅牢性のためにランタイムでの認証情報リフレッシュ + リトライに依存しています。
:::

**APIルーティング**: GPT-5以上のモデル（`gpt-5-mini` を除く）は自動的にResponses APIを使用します。その他のすべてのモデル（GPT-4o、Claude、Geminiなど）はChat Completionsを使用します。モデルはライブのCopilotカタログから自動検出されます。

**`copilot-acp` — Copilot ACPエージェントバックエンド**。ローカルのCopilot CLIをサブプロセスとして起動します:

```bash
hermes chat --provider copilot-acp --model copilot-acp
# PATH上にGitHub Copilot CLIと、既存の `copilot login` セッションが必要です
```

**恒久的な設定:**
```yaml
model:
  provider: "copilot"
  default: "gpt-5.4"
```

| 環境変数 | 説明 |
|---------------------|-------------|
| `COPILOT_GITHUB_TOKEN` | Copilot API用のGitHubトークン（最優先） |
| `HERMES_COPILOT_ACP_COMMAND` | Copilot CLIバイナリのパスを上書き（デフォルト: `copilot`） |
| `HERMES_COPILOT_ACP_ARGS` | ACP引数を上書き（デフォルト: `--acp --stdio`） |

### ファーストクラスのAPIキープロバイダー

これらのプロバイダーは、専用のプロバイダーIDによる組み込みサポートを備えています。APIキーを設定し、`--provider` で選択します:

```bash
# z.ai / ZhipuAI GLM
hermes chat --provider zai --model glm-5
# 必要: ~/.hermes/.env 内の GLM_API_KEY

# Kimi / Moonshot AI（国際版: api.moonshot.ai）
hermes chat --provider kimi-coding --model kimi-for-coding
# 必要: ~/.hermes/.env 内の KIMI_API_KEY

# Kimi / Moonshot AI（中国版: api.moonshot.cn）
hermes chat --provider kimi-coding-cn --model kimi-k2.5
# 必要: ~/.hermes/.env 内の KIMI_CN_API_KEY

# MiniMax（グローバルエンドポイント）
hermes chat --provider minimax --model MiniMax-M2.7
# 必要: ~/.hermes/.env 内の MINIMAX_API_KEY

# MiniMax（中国エンドポイント）
hermes chat --provider minimax-cn --model MiniMax-M2.7
# 必要: ~/.hermes/.env 内の MINIMAX_CN_API_KEY

# Alibaba Cloud / DashScope（Qwenモデル）
hermes chat --provider alibaba --model qwen3.5-plus
# 必要: ~/.hermes/.env 内の DASHSCOPE_API_KEY

# Xiaomi MiMo
hermes chat --provider xiaomi --model mimo-v2-pro
# 必要: ~/.hermes/.env 内の XIAOMI_API_KEY

# Tencent TokenHub（Hy3 Preview）
hermes chat --provider tencent-tokenhub --model hy3-preview
# 必要: ~/.hermes/.env 内の TOKENHUB_API_KEY

# Arcee AI（Trinityモデル）
hermes chat --provider arcee --model trinity-large-thinking
# 必要: ~/.hermes/.env 内の ARCEEAI_API_KEY

# GMI Cloud
# GMIの /v1/models エンドポイントが返す正確なモデルIDを使用してください。
hermes chat --provider gmi --model zai-org/GLM-5.1-FP8
# 必要: ~/.hermes/.env 内の GMI_API_KEY
```

または `config.yaml` でプロバイダーを恒久的に設定します:
```yaml
model:
  provider: "gmi"
  default: "zai-org/GLM-5.1-FP8"
```

ベースURLは、`GLM_BASE_URL`、`KIMI_BASE_URL`、`MINIMAX_BASE_URL`、`MINIMAX_CN_BASE_URL`、`DASHSCOPE_BASE_URL`、`XIAOMI_BASE_URL`、`GMI_BASE_URL`、`TOKENHUB_BASE_URL` の各環境変数で上書きできます。

:::note Z.AIエンドポイントの自動検出
Z.AI / GLMプロバイダーを使用する場合、Hermesは複数のエンドポイント（グローバル、中国、codingバリアント）を自動的にプローブし、あなたのAPIキーを受け付けるものを見つけます。`GLM_BASE_URL` を手動で設定する必要はありません — 動作するエンドポイントが自動的に検出されキャッシュされます。
:::

### xAI（Grok） — Responses API + プロンプトキャッシュ

xAIはResponses API（`codex_responses` トランスポート）経由で接続され、Grok 4モデルで推論を自動的にサポートします — `reasoning_effort` パラメータは不要で、サーバーがデフォルトで推論します。`~/.hermes/.env` に `XAI_API_KEY` を設定し、`hermes model` でxAIを選択するか、ショートカットとして `grok` を `/model grok-4-1-fast-reasoning` のように指定します。

xAIをプロバイダーとして使用する場合（`x.ai` を含む任意のベースURL）、Hermesはすべてのリクエストに `x-grok-conv-id` ヘッダーを送信することで、自動的にプロンプトキャッシュを有効にします。これにより、会話セッション内のリクエストが同じサーバーにルーティングされ、xAIのインフラがキャッシュされたシステムプロンプトと会話履歴を再利用できるようになります。

設定は不要です — xAIエンドポイントが検出され、セッションIDが利用可能になると、キャッシュが自動的に有効になります。これによりマルチターン会話のレイテンシとコストが削減されます。

xAIは専用のTTSエンドポイント（`/v1/tts`）も提供しています。`hermes tools` → Voice & TTSで **xAI TTS** を選択するか、設定については [Voice & TTS](../user-guide/features/tts.md#text-to-speech) ページを参照してください。

### Ollama Cloud — マネージドOllamaモデル、OAuth + APIキー

[Ollama Cloud](https://ollama.com/cloud) は、ローカルのOllamaと同じオープンウェイトカタログをホストしますが、GPU要件がありません。`hermes model` で **Ollama Cloud** を選択し、[ollama.com/settings/keys](https://ollama.com/settings/keys) のAPIキーを貼り付けると、Hermesが利用可能なモデルを自動検出します。

```bash
hermes model
# →「Ollama Cloud」を選択
# → OLLAMA_API_KEY を貼り付け
# → 検出されたモデルから選択（gpt-oss:120b、glm-4.6:cloud、qwen3-coder:480b-cloud など）
```

または `config.yaml` で直接:
```yaml
model:
  provider: "ollama-cloud"
  default: "gpt-oss:120b"
```

モデルカタログは `ollama.com/v1/models` から動的に取得され、1時間キャッシュされます。`model:tag` 記法（例: `qwen3-coder:480b-cloud`）は正規化を通じて保持されます — ダッシュは使用しないでください。

:::tip Ollama Cloud対ローカルOllama
どちらも同じOpenAI互換APIを話します。Cloudはファーストクラスのプロバイダーです（`--provider ollama-cloud`、`OLLAMA_API_KEY`）。ローカルOllamaはCustom Endpointフロー経由でアクセスします（ベースURL `http://localhost:11434/v1`、キー不要）。ローカルで実行できない大きなモデルにはCloudを、プライバシーやオフライン作業にはローカルを使用してください。
:::

### AWS Bedrock

AWS Bedrock経由でAnthropic Claude、Amazon Nova、DeepSeek v3.2、Meta Llama 4、その他のモデルを利用します。AWS SDK（`boto3`）の認証情報チェーンを使用します — APIキーは不要で、標準的なAWS認証だけです。

```bash
# 最もシンプル — ~/.aws/credentials 内の名前付きプロファイル
hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6

# または明示的な環境変数で
AWS_PROFILE=myprofile AWS_REGION=us-east-1 hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6
```

または `config.yaml` で恒久的に:
```yaml
model:
  provider: "bedrock"
  default: "us.anthropic.claude-sonnet-4-6"
bedrock:
  region: "us-east-1"          # または AWS_REGION を設定
  # profile: "myprofile"       # または AWS_PROFILE を設定
  # discovery: true            # IAMからリージョンを自動検出
  # guardrail:                 # 任意の Bedrock Guardrails
  #   guardrail_identifier: "your-guardrail-id"
  #   guardrail_version: "DRAFT"
```

認証は標準のboto3チェーンを使用します: 明示的な `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`、`~/.aws/credentials` の `AWS_PROFILE`、EC2/ECS/LambdaのIAMロール、IMDS、またはSSO。すでにAWS CLIで認証済みの場合、環境変数は不要です。

Bedrockは内部で**Converse API**を使用します — リクエストはBedrockのモデル非依存の形式に変換されるため、同じ設定がClaude、Nova、DeepSeek、Llamaの各モデルで機能します。デフォルト以外のリージョナルエンドポイントを呼び出す場合のみ `BEDROCK_BASE_URL` を設定してください。

IAMのセットアップ、リージョンの選択、クロスリージョン推論の手順については、[AWS Bedrockガイド](/docs/guides/aws-bedrock) を参照してください。

### Qwen Portal（OAuth）

ブラウザベースのOAuthログインを使うAlibabaのQwen Portalです。`hermes model` で **Qwen OAuth (Portal)** を選択し、ブラウザでサインインすると、Hermesがリフレッシュトークンを永続化します。

```bash
hermes model
# →「Qwen OAuth (Portal)」を選択
# → ブラウザが開く。Alibabaアカウントでサインイン
# → 確認 — 認証情報が ~/.hermes/auth.json に保存されます

hermes chat   # portal.qwen.ai/v1 エンドポイントを使用
```

または `config.yaml` を設定します:
```yaml
model:
  provider: "qwen-oauth"
  default: "qwen3-coder-plus"
```

ポータルエンドポイントが移転した場合のみ `HERMES_QWEN_BASE_URL` を設定してください（デフォルト: `https://portal.qwen.ai/v1`）。

:::tip Qwen OAuth対DashScope（Alibaba）
`qwen-oauth` はOAuthログインを使う消費者向けのQwen Portalを使用します — 個人ユーザーに最適です。`alibaba` プロバイダーはDashScopeのエンタープライズAPIを `DASHSCOPE_API_KEY` で使用します — プログラム的／本番ワークロードに最適です。どちらもQwenファミリーのモデルにルーティングしますが、異なるエンドポイントに存在します。
:::

### Alibaba Coding Plan

Alibabaの**Coding Plan**（標準のDashScope APIアクセスとは別の料金SKU）に加入している場合、Hermesはそれを独自のファーストクラスプロバイダー `alibaba-coding-plan` として公開します。エンドポイント: `https://coding-intl.dashscope.aliyuncs.com/v1`。通常の `alibaba` プロバイダーと同様にOpenAI互換ですが、ベースURLと課金面が異なります。

```yaml
model:
  provider: alibaba_coding     # alibaba-coding-plan のエイリアス
  model: qwen3-coder-plus
```

またはCLIから:

```bash
hermes chat --provider alibaba_coding --model qwen3-coder-plus
```

`alibaba_coding` は、`alibaba` エントリですでに使用しているのと同じ `DASHSCOPE_API_KEY` を使用します — 別個のキーは不要で、ルーティング先が異なるだけです。このプロバイダーが登録される前は、`config.yaml` に `provider: alibaba_coding` を設定したユーザーは黙ってOpenRouterルーティングに流れ落ちていました。

### MiniMax（OAuth）

ブラウザOAuthログインによるMiniMax-M2.7です — APIキーは不要です。`hermes model` で **MiniMax (OAuth)** を選択し、ブラウザでサインインすると、Hermesがアクセストークンとリフレッシュトークンを永続化します。内部ではAnthropic Messages互換のエンドポイント（`/anthropic`）を使用します。

```bash
hermes model
# →「MiniMax (OAuth)」を選択
# → ブラウザが開く。MiniMaxアカウント（グローバルまたはCNリージョン）でサインイン
# → 確認 — 認証情報が ~/.hermes/auth.json に保存されます

hermes chat   # api.minimax.io/anthropic エンドポイントを使用
```

または `config.yaml` を設定します:
```yaml
model:
  provider: "minimax-oauth"
  default: "MiniMax-M2.7"
```

サポートされるモデル: `MiniMax-M2.7`（メイン）と `MiniMax-M2.7-highspeed`（デフォルトの補助モデルとして接続）。OAuth経路は `MINIMAX_API_KEY` / `MINIMAX_BASE_URL` を無視します。

:::tip MiniMax OAuth対APIキー
`minimax-oauth` はOAuthログインを使うMiniMaxの消費者向けポータルを使用します — 課金のセットアップは不要です。`minimax` および `minimax-cn` プロバイダーは `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY` を使用します — プログラム的なアクセス向けです。完全な手順については [MiniMax OAuthガイド](/docs/guides/minimax-oauth) を参照してください。
:::

### NVIDIA NIM

[build.nvidia.com](https://build.nvidia.com)（無料APIキー）またはローカルのNIMエンドポイント経由で、Nemotronその他のオープンソースモデルを利用します。

```bash
# クラウド（build.nvidia.com）
hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
# 必要: ~/.hermes/.env 内の NVIDIA_API_KEY

# ローカルNIMエンドポイント — ベースURLを上書き
NVIDIA_BASE_URL=http://localhost:8000/v1 hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "nvidia"
  default: "nvidia/nemotron-3-super-120b-a12b"
```

:::tip ローカルNIM
オンプレミスのデプロイ（DGX Spark、ローカルGPU）では、`NVIDIA_BASE_URL=http://localhost:8000/v1` を設定します。NIMはbuild.nvidia.comと同じOpenAI互換のchat completions APIを公開するため、クラウドとローカルの切り替えは環境変数1行の変更だけで済みます。
:::

### GMI Cloud

[GMI Cloud](https://www.gmicloud.ai/) 経由でオープンモデルや推論モデルを利用します — OpenAI互換API、APIキー認証。

```bash
# GMI Cloud
hermes chat --provider gmi --model deepseek-ai/DeepSeek-R1
# 必要: ~/.hermes/.env 内の GMI_API_KEY
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "gmi"
  default: "deepseek-ai/DeepSeek-R1"
```

ベースURLは `GMI_BASE_URL` で上書きできます（デフォルト: `https://api.gmi-serving.com/v1`）。

### StepFun

[StepFun](https://platform.stepfun.com) 経由でStepシリーズのモデルを利用します — OpenAI互換API、APIキー認証。

```bash
# StepFun
hermes chat --provider stepfun --model step-3-mini
# 必要: ~/.hermes/.env 内の STEPFUN_API_KEY
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "stepfun"
  default: "step-3-mini"
```

ベースURLは `STEPFUN_BASE_URL` で上書きできます（デフォルト: `https://api.stepfun.com/v1`）。

### Hugging Face Inference Providers

[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) は、統合されたOpenAI互換エンドポイント（`router.huggingface.co/v1`）を通じて20以上のオープンモデルにルーティングします。リクエストは利用可能な最速のバックエンド（Groq、Together、SambaNovaなど）に自動的にルーティングされ、自動フェイルオーバーを備えています。

```bash
# 利用可能な任意のモデルを使用
hermes chat --provider huggingface --model Qwen/Qwen3-235B-A22B-Thinking-2507
# 必要: ~/.hermes/.env 内の HF_TOKEN

# 短いエイリアス
hermes chat --provider hf --model deepseek-ai/DeepSeek-V3.2
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "huggingface"
  default: "Qwen/Qwen3-235B-A22B-Thinking-2507"
```

トークンは [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) で取得してください — 「Make calls to Inference Providers」権限を必ず有効にしてください。無料枠が含まれます（月$0.10のクレジット、プロバイダー料金への上乗せなし）。

モデル名にルーティングサフィックスを付加できます: `:fastest`（デフォルト）、`:cheapest`、または特定のバックエンドを強制する `:provider_name`。

ベースURLは `HF_BASE_URL` で上書きできます。

## カスタム＆セルフホスト型LLMプロバイダー

Hermes Agentは**あらゆるOpenAI互換APIエンドポイント**で動作します。サーバーが `/v1/chat/completions` を実装していれば、Hermesをそこに向けることができます。つまり、ローカルモデル、GPU推論サーバー、マルチプロバイダールーター、または任意のサードパーティAPIを使用できます。

### 一般的なセットアップ

カスタムエンドポイントを設定する3つの方法:

**対話的なセットアップ（推奨）:**
```bash
hermes model
# 「Custom endpoint (self-hosted / VLLM / etc.)」を選択
# 入力: API base URL、API key、Model name
```

**手動設定（`config.yaml`）:**
```yaml
# ~/.hermes/config.yaml 内
model:
  default: your-model-name
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: your-key-or-leave-empty-for-local
```

:::warning レガシー環境変数
`.env` 内の `OPENAI_BASE_URL` と `LLM_MODEL` は**削除されました**。どちらもHermesのどの部分からも読み込まれません — `config.yaml` がモデルとエンドポイントの設定における唯一の信頼できる情報源です。`.env` に古いエントリがある場合は、次回の `hermes setup` または設定の移行時に自動的にクリアされます。`hermes model` を使うか、`config.yaml` を直接編集してください。
:::

どちらの方法も `config.yaml` に永続化され、これがモデル、プロバイダー、ベースURLの信頼できる情報源となります。

### `/model` によるモデルの切り替え

:::warning hermes model 対 /model
**`hermes model`**（ターミナルから、チャットセッションの外で実行）は、**完全なプロバイダーセットアップウィザード**です。新しいプロバイダーの追加、OAuthフローの実行、APIキーの入力、カスタムエンドポイントの設定に使用します。

**`/model`**（アクティブなHermesチャットセッション内で入力）は、**すでにセットアップ済みのプロバイダーとモデルの間での切り替え**のみが可能です。新しいプロバイダーの追加、OAuthの実行、APIキーの入力はできません。1つのプロバイダー（例: OpenRouter）のみを設定している場合、`/model` はそのプロバイダーのモデルのみを表示します。

**新しいプロバイダーを追加するには:** セッションを終了し（`Ctrl+C` または `/quit`）、`hermes model` を実行して新しいプロバイダーをセットアップし、新しいセッションを開始してください。
:::

カスタムエンドポイントを少なくとも1つ設定すると、セッションの途中でモデルを切り替えられます:

```
/model custom:qwen-2.5          # カスタムエンドポイント上のモデルに切り替え
/model custom                    # エンドポイントからモデルを自動検出
/model openrouter:claude-sonnet-4 # クラウドプロバイダーに戻す
```

**名前付きカスタムプロバイダー**（後述）を設定している場合は、トリプル構文を使用します:

```
/model custom:local:qwen-2.5    # 「local」カスタムプロバイダーをモデル qwen-2.5 で使用
/model custom:work:llama3       # 「work」カスタムプロバイダーを llama3 で使用
```

プロバイダーを切り替えると、Hermesはベースアドレスとプロバイダーを設定に永続化するため、変更は再起動後も維持されます。カスタムエンドポイントから組み込みプロバイダーに切り替えると、古いベースURLは自動的にクリアされます。

:::tip
`/model custom`（モデル名なしの素の形）は、エンドポイントの `/models` APIを照会し、ちょうど1つのモデルがロードされている場合はそのモデルを自動選択します。単一のモデルを実行しているローカルサーバーに便利です。
:::

以下のすべては同じパターンに従います — URL、キー、モデル名を変えるだけです。

---

### Ollama — ローカルモデル、設定不要

[Ollama](https://ollama.com/) は、1つのコマンドでオープンウェイトモデルをローカルに実行します。最適な用途: 手軽なローカル実験、プライバシーに配慮した作業、オフライン利用。OpenAI互換API経由のツール呼び出しをサポートします。

```bash
# モデルのインストールと実行
ollama pull qwen2.5-coder:32b
ollama serve   # ポート11434で起動
```

次にHermesを設定します:

```bash
hermes model
# 「Custom endpoint (self-hosted / VLLM / etc.)」を選択
# URLを入力: http://localhost:11434/v1
# APIキーをスキップ（Ollamaは不要）
# モデル名を入力（例: qwen2.5-coder:32b）
```

または `config.yaml` を直接設定します:

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 32768   # 下記の警告を参照
```

:::caution Ollamaはデフォルトで非常に小さいコンテキスト長になります
Ollamaはデフォルトでモデルのフルコンテキストウィンドウを使用しません。VRAMに応じて、デフォルトは次のようになります:

| 利用可能なVRAM | デフォルトのコンテキスト |
|----------------|----------------|
| 24 GB未満 | **4,096トークン** |
| 24〜48 GB | 32,768トークン |
| 48 GB以上 | 256,000トークン |

ツールを使うエージェント用途では、**少なくとも16k〜32kのコンテキストが必要です**。4kでは、システムプロンプト + ツールスキーマだけでウィンドウが埋まってしまい、会話の余地がなくなります。

**増やす方法**（いずれか1つを選択）:

```bash
# オプション1: 環境変数でサーバー全体に設定（推奨）
OLLAMA_CONTEXT_LENGTH=32768 ollama serve

# オプション2: systemd管理のOllamaの場合
sudo systemctl edit ollama.service
# 追加: Environment="OLLAMA_CONTEXT_LENGTH=32768"
# その後: sudo systemctl daemon-reload && sudo systemctl restart ollama

# オプション3: カスタムモデルに組み込む（モデルごとに永続化）
echo -e "FROM qwen2.5-coder:32b\nPARAMETER num_ctx 32768" > Modelfile
ollama create qwen2.5-coder-32k -f Modelfile
```

**OpenAI互換API（`/v1/chat/completions`）からコンテキスト長を設定することはできません。** サーバー側またはModelfile経由で設定する必要があります。これはHermesのようなツールとOllamaを統合する際の混乱の最大の原因です。
:::

**コンテキストが正しく設定されているか確認:**

```bash
ollama ps
# CONTEXT列を確認 — 設定した値が表示されるはずです
```

:::tip
利用可能なモデルは `ollama list` で一覧表示できます。[Ollamaライブラリ](https://ollama.com/library) の任意のモデルを `ollama pull <model>` で取得できます。OllamaはGPUオフロードを自動的に処理します — ほとんどのセットアップで設定は不要です。
:::

---

### vLLM — 高性能なGPU推論

[vLLM](https://docs.vllm.ai/) は本番のLLMサービングの標準です。最適な用途: GPUハードウェアでの最大スループット、大きなモデルのサービング、継続的バッチング。

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --max-model-len 65536 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

次にHermesを設定します:

```bash
hermes model
# 「Custom endpoint (self-hosted / VLLM / etc.)」を選択
# URLを入力: http://localhost:8000/v1
# APIキーをスキップ（または vLLM を --api-key で設定した場合は入力）
# モデル名を入力: meta-llama/Llama-3.1-70B-Instruct
```

**コンテキスト長:** vLLMはデフォルトでモデルの `max_position_embeddings` を読み込みます。それがGPUメモリを超える場合はエラーになり、`--max-model-len` をより低く設定するよう求められます。`--max-model-len auto` を使うと、収まる最大値を自動的に見つけることもできます。`--gpu-memory-utilization 0.95`（デフォルト0.9）を設定すると、より多くのコンテキストをVRAMに詰め込めます。

**ツール呼び出しには明示的なフラグが必要です:**

| フラグ | 目的 |
|------|---------|
| `--enable-auto-tool-choice` | `tool_choice: "auto"`（Hermesのデフォルト）に必要 |
| `--tool-call-parser <name>` | モデルのツール呼び出し形式用のパーサー |

サポートされるパーサー: `hermes`（Qwen 2.5、Hermes 2/3）、`llama3_json`（Llama 3.x）、`mistral`、`deepseek_v3`、`deepseek_v31`、`xlam`、`pythonic`。これらのフラグがないと、ツール呼び出しは機能しません — モデルはツール呼び出しをテキストとして出力します。

:::tip
vLLMは人間が読めるサイズをサポートします: `--max-model-len 64k`（小文字のk = 1000、大文字のK = 1024）。
:::

---

### SGLang — RadixAttentionによる高速サービング

[SGLang](https://github.com/sgl-project/sglang) は、KVキャッシュ再利用のためのRadixAttentionを備えたvLLMの代替です。最適な用途: マルチターン会話（プレフィックスキャッシュ）、制約付きデコード、構造化出力。

```bash
pip install "sglang[all]"
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2 \
  --tool-call-parser qwen
```

次にHermesを設定します:

```bash
hermes model
# 「Custom endpoint (self-hosted / VLLM / etc.)」を選択
# URLを入力: http://localhost:30000/v1
# モデル名を入力: meta-llama/Llama-3.1-70B-Instruct
```

**コンテキスト長:** SGLangはデフォルトでモデルの設定から読み込みます。`--context-length` を使って上書きします。モデルが宣言する最大値を超える必要がある場合は、`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` を設定します。

**ツール呼び出し:** モデルファミリーに適したパーサーを `--tool-call-parser` で指定します: `qwen`（Qwen 2.5）、`llama3`、`llama4`、`deepseekv3`、`mistral`、`glm`。このフラグがないと、ツール呼び出しはプレーンテキストとして返ってきます。

:::caution SGLangはデフォルトで最大出力128トークンになります
レスポンスが途切れているように見える場合は、リクエストに `max_tokens` を追加するか、サーバーに `--default-max-tokens` を設定してください。SGLangのデフォルトは、リクエストで指定しない場合、レスポンスあたりわずか128トークンです。
:::

---

### llama.cpp / llama-server — CPUおよびMetal推論

[llama.cpp](https://github.com/ggml-org/llama.cpp) は、量子化されたモデルをCPU、Apple Silicon（Metal）、コンシューマー向けGPUで実行します。最適な用途: データセンターのGPUなしでモデルを実行、Macユーザー、エッジデプロイ。

```bash
# llama-server のビルドと起動
cmake -B build && cmake --build build --config Release
./build/bin/llama-server \
  --jinja -fa \
  -c 32768 \
  -ngl 99 \
  -m models/qwen2.5-coder-32b-instruct-Q4_K_M.gguf \
  --port 8080 --host 0.0.0.0
```

**コンテキスト長（`-c`）:** 最近のビルドはデフォルトで `0` となり、GGUFメタデータからモデルのトレーニングコンテキストを読み込みます。128k以上のトレーニングコンテキストを持つモデルの場合、フルのKVキャッシュを割り当てようとしてOOM（メモリ不足）になることがあります。`-c` を必要な値（エージェント用途では32k〜64kが適切な範囲）に明示的に設定してください。並列スロット（`-np`）を使用する場合、合計コンテキストはスロット間で分割されます — `-c 32768 -np 4` では、各スロットは8kしか得られません。

次にHermesをそこに向けて設定します:

```bash
hermes model
# 「Custom endpoint (self-hosted / VLLM / etc.)」を選択
# URLを入力: http://localhost:8080/v1
# APIキーをスキップ（ローカルサーバーは不要）
# モデル名を入力 — または1つのモデルだけがロードされている場合は空欄にして自動検出
```

これによりエンドポイントが `config.yaml` に保存され、セッションをまたいで維持されます。

:::caution ツール呼び出しには `--jinja` が必須です
`--jinja` がないと、llama-serverは `tools` パラメータを完全に無視します。モデルはレスポンステキストにJSONを書くことでツールを呼び出そうとしますが、Hermesはそれをツール呼び出しとして認識しません — 実際の検索の代わりに、`{"name": "web_search", ...}` のような生のJSONがメッセージとして表示されます。

ネイティブのツール呼び出しサポート（最高のパフォーマンス）: Llama 3.x、Qwen 2.5（Coderを含む）、Hermes 2/3、Mistral、DeepSeek、Functionary。その他のすべてのモデルは、動作はするが効率が劣る可能性のある汎用ハンドラを使用します。完全なリストについては [llama.cpp function callingドキュメント](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) を参照してください。

ツールサポートが有効かどうかは、`http://localhost:8080/props` を確認することで検証できます — `chat_template` フィールドが存在するはずです。
:::

:::tip
GGUFモデルは [Hugging Face](https://huggingface.co/models?library=gguf) からダウンロードできます。Q4_K_M量子化は品質とメモリ使用量の最良のバランスを提供します。
:::

---

### LM Studio — ローカルモデルを使うデスクトップアプリ

[LM Studio](https://lmstudio.ai/) は、GUIでローカルモデルを実行するデスクトップアプリです。最適な用途: ビジュアルインターフェースを好むユーザー、手軽なモデルテスト、macOS/Windows/Linuxの開発者。

LM Studioアプリからサーバーを起動するか（Developerタブ → Start Server）、CLIを使用します:

```bash
lms server start                        # ポート1234で起動
lms load qwen2.5-coder --context-length 32768
```

次にHermesを設定します:

```bash
hermes model
# 「LM Studio」を選択
# Enterを押して http://localhost:1234/v1 を使用
# 検出されたモデルから1つを選択
# LM Studioサーバーの認証が有効な場合、プロンプトが出たら LM_API_KEY を入力
```

Hermesは自動的に64Kのコンテキスト長でLM Studioモデルをロードします。

LM Studioでコンテキスト長を変更するには:

1. モデルピッカーの隣にある歯車アイコンをクリック
2. 「Context Length」を快適な体験のために少なくとも64000に設定
3. 変更を反映させるためにモデルをリロード
4. マシンが64000に収まらない場合は、より大きなコンテキスト長を持つより小さいモデルの使用を検討してください。

代替として、CLIを使用します: `lms load model-name --context-length 64000`

CLIを使ってモデルが収まるかを見積もることもできます: `lms load model-name --context-length 64000 --estimate-only`

モデルごとの永続的なデフォルトを設定するには: My Modelsタブ → モデルの歯車アイコン → コンテキストサイズを設定。
:::

**ツール呼び出し:** LM Studio 0.3.6以降でサポートされています。ネイティブのツール呼び出しトレーニングを受けたモデル（Qwen 2.5、Llama 3.x、Mistral、Hermes）は自動検出され、ツールバッジ付きで表示されます。その他のモデルは、信頼性が劣る可能性のある汎用フォールバックを使用します。

---

### WSL2ネットワーク（Windowsユーザー） {#wsl2-networking-windows-users}

Hermes AgentはUnix環境を必要とするため、WindowsユーザーはWSL2内で実行します。モデルサーバー（Ollama、LM Studioなど）が**Windowsホスト**上で動作している場合、ネットワークのギャップを橋渡しする必要があります — WSL2は独自のサブネットを持つ仮想ネットワークアダプタを使用するため、WSL2内の `localhost` はLinux VMを指し、**Windowsホストではありません**。

:::tip 両方ともWSL2内なら問題ありません。
モデルサーバーもWSL2内で動作している場合（vLLM、SGLang、llama-serverでは一般的）、`localhost` は期待どおりに機能します — 同じネットワーク名前空間を共有しているためです。このセクションはスキップしてください。
:::

#### オプション1: ミラードネットワークモード（推奨）

**Windows 11 22H2以降**で利用可能なミラードモードは、WindowsとWSL2の間で `localhost` を双方向に機能させます — 最もシンプルな解決策です。

1. `%USERPROFILE%\.wslconfig`（例: `C:\Users\YourName\.wslconfig`）を作成または編集します:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. PowerShellからWSLを再起動します:
   ```powershell
   wsl --shutdown
   ```

3. WSL2ターミナルを再度開きます。`localhost` がWindowsサービスに到達するようになります:
   ```bash
   curl http://localhost:11434/v1/models   # Windows上のOllama — 動作します
   ```

:::note Hyper-Vファイアウォール
一部のWindows 11ビルドでは、Hyper-Vファイアウォールがデフォルトでミラード接続をブロックします。ミラードモードを有効にしても `localhost` がまだ機能しない場合は、**管理者PowerShell**で次を実行してください:
```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```
:::

#### オプション2: WindowsホストIPを使用する（Windows 10 / 古いビルド）

ミラードモードを使用できない場合は、WSL2内からWindowsホストIPを見つけ、`localhost` の代わりにそれを使用します:

```bash
# WindowsホストIPを取得（WSL2の仮想ネットワークのデフォルトゲートウェイ）
ip route show | grep -i default | awk '{ print $3 }'
# 出力例: 172.29.192.1
```

そのIPをHermesの設定で使用します:

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://172.29.192.1:11434/v1   # localhostではなくWindowsホストIP
```

:::tip 動的なヘルパー
ホストIPはWSL2の再起動時に変わることがあります。シェルで動的に取得できます:
```bash
export WSL_HOST=$(ip route show | grep -i default | awk '{ print $3 }')
echo "Windows host at: $WSL_HOST"
curl http://$WSL_HOST:11434/v1/models   # Ollamaをテスト
```

または、マシンのmDNS名を使用します（WSL2に `libnss-mdns` が必要）:
```bash
sudo apt install libnss-mdns
curl http://$(hostname).local:11434/v1/models
```
:::

#### サーバーのバインドアドレス（NATモードで必須）

**オプション2**（ホストIPを使うNATモード）を使用している場合、Windows上のモデルサーバーは `127.0.0.1` の外部からの接続を受け付ける必要があります。デフォルトでは、ほとんどのサーバーはlocalhostでのみリッスンします — NATモードのWSL2接続は別の仮想サブネットから来るため拒否されます。ミラードモードでは `localhost` が直接マッピングされるため、デフォルトの `127.0.0.1` バインドで問題なく動作します。

| サーバー | デフォルトのバインド | 修正方法 |
|--------|-------------|------------|
| **Ollama** | `127.0.0.1` | Ollamaの起動前に `OLLAMA_HOST=0.0.0.0` 環境変数を設定（Windowsのシステム設定 → 環境変数、またはOllamaサービスを編集） |
| **LM Studio** | `127.0.0.1` | Developerタブ → Server settingsで **「Serve on Network」** を有効化 |
| **llama-server** | `127.0.0.1` | 起動コマンドに `--host 0.0.0.0` を追加 |
| **vLLM** | `0.0.0.0` | デフォルトですでにすべてのインターフェースにバインド |
| **SGLang** | `127.0.0.1` | 起動コマンドに `--host 0.0.0.0` を追加 |

**Windows上のOllama（詳細）:** OllamaはWindowsサービスとして動作します。`OLLAMA_HOST` を設定するには:
1. **システムのプロパティ** → **環境変数** を開く
2. 新しい**システム変数**を追加: `OLLAMA_HOST` = `0.0.0.0`
3. Ollamaサービスを再起動（または再起動）

#### Windowsファイアウォール

Windowsファイアウォールは、WSL2を（NATモードとミラードモードの両方で）別個のネットワークとして扱います。上記の手順後も接続が失敗する場合は、モデルサーバーのポートに対するファイアウォールルールを追加してください:

```powershell
# 管理者PowerShellで実行 — PORTをサーバーのポートに置き換えてください
New-NetFirewallRule -DisplayName "Allow WSL2 to Model Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

一般的なポート: Ollama `11434`、vLLM `8000`、SGLang `30000`、llama-server `8080`、LM Studio `1234`。

#### クイック検証

WSL2内から、モデルサーバーに到達できることをテストします:

```bash
# URLをサーバーのアドレスとポートに置き換えてください
curl http://localhost:11434/v1/models          # ミラードモード
curl http://172.29.192.1:11434/v1/models       # NATモード（実際のホストIPを使用）
```

モデルを一覧表示するJSONレスポンスが得られれば成功です。その同じURLをHermes設定の `base_url` として使用してください。

---

### ローカルモデルのトラブルシューティング

これらの問題は、Hermesと共に使用される際の**すべての**ローカル推論サーバーに影響します。

#### WSL2からWindowsホストのモデルサーバーへの「Connection refused」

WSL2内でHermesを実行し、モデルサーバーがWindowsホスト上にある場合、`http://localhost:<port>` はWSL2のデフォルトのNATネットワークモードでは機能しません。修正については上記の [WSL2ネットワーク](#wsl2-networking-windows-users) を参照してください。

#### ツール呼び出しが実行されずテキストとして表示される

モデルが、実際にツールを呼び出す代わりに `{"name": "web_search", "arguments": {...}}` のようなものをメッセージとして出力します。

**原因:** サーバーでツール呼び出しが有効になっていないか、サーバーのツール呼び出し実装を通じてモデルがそれをサポートしていません。

| サーバー | 修正 |
|--------|-----|
| **llama.cpp** | 起動コマンドに `--jinja` を追加 |
| **vLLM** | `--enable-auto-tool-choice --tool-call-parser hermes` を追加 |
| **SGLang** | `--tool-call-parser qwen`（または適切なパーサー）を追加 |
| **Ollama** | ツール呼び出しはデフォルトで有効 — モデルがそれをサポートしているか確認（`ollama show model-name` で確認） |
| **LM Studio** | 0.3.6以降にアップデートし、ネイティブのツールサポートを持つモデルを使用 |

#### モデルがコンテキストを忘れたり一貫性のない応答をする

**原因:** コンテキストウィンドウが小さすぎます。会話がコンテキスト制限を超えると、ほとんどのサーバーは古いメッセージを黙って破棄します。Hermesのシステムプロンプト + ツールスキーマだけで4k〜8kトークンを使用することがあります。

**診断:**

```bash
# Hermesがコンテキストをどう認識しているか確認
# 起動時の行を参照: "Context limit: X tokens"

# サーバーの実際のコンテキストを確認
# Ollama: ollama ps（CONTEXT列）
# llama.cpp: curl http://localhost:8080/props | jq '.default_generation_settings.n_ctx'
# vLLM: 起動引数の --max-model-len を確認
```

**修正:** エージェント用途では、コンテキストを少なくとも **32,768トークン** に設定してください。具体的なフラグについては上記の各サーバーのセクションを参照してください。

#### 起動時に「Context limit: 2048 tokens」と表示される

Hermesはサーバーの `/v1/models` エンドポイントからコンテキスト長を自動検出します。サーバーが低い値を報告する（または全く報告しない）場合、Hermesはモデルが宣言する制限を使用しますが、これは誤っていることがあります。

**修正:** `config.yaml` で明示的に設定します:

```yaml
model:
  default: your-model
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 32768
```

#### レスポンスが文の途中で途切れる

**考えられる原因:**
1. **サーバーの低い出力上限（`max_tokens`）** — SGLangはレスポンスあたり128トークンがデフォルトです。サーバーに `--default-max-tokens` を設定するか、config.yamlの `model.max_tokens` でHermesを設定してください。注意: `max_tokens` はレスポンスの長さのみを制御します — 会話履歴をどれだけ長くできるか（それは `context_length`）とは無関係です。
2. **コンテキストの枯渇** — モデルがコンテキストウィンドウを埋め尽くしました。`model.context_length` を増やすか、Hermesで [コンテキスト圧縮](/docs/user-guide/configuration#context-compression) を有効にしてください。

---

### LiteLLM Proxy — マルチプロバイダーゲートウェイ

[LiteLLM](https://docs.litellm.ai/) は、100以上のLLMプロバイダーを単一のAPIの背後に統合するOpenAI互換プロキシです。最適な用途: 設定変更なしでのプロバイダー間の切り替え、ロードバランシング、フォールバックチェーン、予算管理。

```bash
# インストールと起動
pip install "litellm[proxy]"
litellm --model anthropic/claude-sonnet-4 --port 4000

# または複数のモデル用の設定ファイルで:
litellm --config litellm_config.yaml --port 4000
```

次に `hermes model` → Custom endpoint → `http://localhost:4000/v1` でHermesを設定します。

フォールバック付きの `litellm_config.yaml` の例:
```yaml
model_list:
  - model_name: "best"
    litellm_params:
      model: anthropic/claude-sonnet-4
      api_key: sk-ant-...
  - model_name: "best"
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-...
router_settings:
  routing_strategy: "latency-based-routing"
```

---

### ClawRouter — コスト最適化ルーティング

BlockRunAIによる [ClawRouter](https://github.com/BlockRunAI/ClawRouter) は、クエリの複雑さに基づいてモデルを自動選択するローカルのルーティングプロキシです。リクエストを14の次元で分類し、タスクを処理できる最も安価なモデルにルーティングします。支払いはUSDC暗号通貨経由です（APIキー不要）。

```bash
# インストールと起動
npx @blockrun/clawrouter    # ポート8402で起動
```

次に `hermes model` → Custom endpoint → `http://localhost:8402/v1` → モデル名 `blockrun/auto` でHermesを設定します。

ルーティングプロファイル:
| プロファイル | 戦略 | 節約率 |
|---------|----------|---------|
| `blockrun/auto` | 品質とコストのバランス | 74〜100% |
| `blockrun/eco` | 可能な限り安価 | 95〜100% |
| `blockrun/premium` | 最高品質のモデル | 0% |
| `blockrun/free` | 無料モデルのみ | 100% |
| `blockrun/agentic` | ツール利用に最適化 | 変動 |

:::note
ClawRouterは支払いのため、BaseまたはSolana上のUSDC資金を入れたウォレットが必要です。すべてのリクエストはBlockRunのバックエンドAPIを経由します。ウォレットの状態を確認するには `npx @blockrun/clawrouter doctor` を実行してください。
:::

---

### その他の互換プロバイダー {#other-compatible-providers}

OpenAI互換APIを持つ任意のサービスが機能します。人気のある選択肢をいくつか挙げます:

| プロバイダー | ベースURL | 備考 |
|----------|----------|-------|
| [Together AI](https://together.ai) | `https://api.together.xyz/v1` | クラウドホスト型のオープンモデル |
| [Groq](https://groq.com) | `https://api.groq.com/openai/v1` | 超高速推論 |
| [DeepSeek](https://deepseek.com) | `https://api.deepseek.com/v1` | DeepSeekモデル |
| [Fireworks AI](https://fireworks.ai) | `https://api.fireworks.ai/inference/v1` | 高速なオープンモデルホスティング |
| [GMI Cloud](https://www.gmicloud.ai/) | `https://api.gmi-serving.com/v1` | マネージドのOpenAI互換推論 |
| [Cerebras](https://cerebras.ai) | `https://api.cerebras.ai/v1` | ウェハースケールチップによる推論 |
| [Mistral AI](https://mistral.ai) | `https://api.mistral.ai/v1` | Mistralモデル |
| [OpenAI](https://openai.com) | `https://api.openai.com/v1` | OpenAIへの直接アクセス |
| [Azure OpenAI](https://azure.microsoft.com) | `https://YOUR.openai.azure.com/` | エンタープライズOpenAI |
| [LocalAI](https://localai.io) | `http://localhost:8080/v1` | セルフホスト、マルチモデル |
| [Jan](https://jan.ai) | `http://localhost:1337/v1` | ローカルモデルを使うデスクトップアプリ |

これらのいずれも `hermes model` → Custom endpoint、または `config.yaml` で設定します:

```yaml
model:
  default: meta-llama/Llama-3.1-70B-Instruct-Turbo
  provider: custom
  base_url: https://api.together.xyz/v1
  api_key: your-together-key
```

---

### コンテキスト長の検出 {#context-length-detection}

:::note 2つの設定、混同しやすい
**`context_length`** は**コンテキストウィンドウ全体**です — 入力*と*出力トークンを合わせた予算（例: Claude Opus 4.6では200,000）。Hermesはこれを使って、いつ履歴を圧縮するか、APIリクエストを検証するかを判断します。

**`model.max_tokens`** は**出力の上限**です — モデルが*単一のレスポンス*で生成できるトークンの最大数。会話履歴をどれだけ長くできるかとは無関係です。業界標準の名称である `max_tokens` は混乱のよくある原因です。Anthropicのネイティブ APIはその後、明確化のためにこれを `max_output_tokens` に改名しました。

自動検出がウィンドウサイズを誤る場合に `context_length` を設定してください。
個々のレスポンスの長さを制限する必要がある場合のみ `model.max_tokens` を設定してください。
:::

Hermesは、あなたのモデルとプロバイダーに対する正しいコンテキストウィンドウを検出するために、複数のソースによる解決チェーンを使用します:

1. **設定の上書き** — config.yamlの `model.context_length`（最優先）
2. **カスタムプロバイダーのモデルごと** — `custom_providers[].models.<id>.context_length`
3. **永続キャッシュ** — 以前に検出された値（再起動後も維持）
4. **エンドポイントの `/models`** — サーバーのAPIを照会（ローカル／カスタムエンドポイント）
5. **Anthropicの `/v1/models`** — Anthropic APIに `max_input_tokens` を照会（APIキーユーザーのみ）
6. **OpenRouter API** — OpenRouterからのライブのモデルメタデータ
7. **Nous Portal** — Nousのモデル IDをOpenRouterメタデータにサフィックスマッチング
8. **[models.dev](https://models.dev)** — 100以上のプロバイダーにわたる3800以上のモデルについて、プロバイダー固有のコンテキスト長を持つコミュニティ管理のレジストリ
9. **フォールバックのデフォルト** — 広範なモデルファミリーのパターン（デフォルト128K）

ほとんどのセットアップでは、これがそのまま機能します。このシステムはプロバイダーを認識します — 同じモデルでも、誰が提供するかによって異なるコンテキスト制限を持つことがあります（例: `claude-opus-4.6` はAnthropic直接では1Mですが、GitHub Copilotでは128K）。

コンテキスト長を明示的に設定するには、モデル設定に `context_length` を追加します:

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072  # トークン
```

カスタムエンドポイントの場合、モデルごとにコンテキスト長を設定することもできます:

```yaml
custom_providers:
  - name: "My Local LLM"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 32768
      deepseek-r1:70b:
        context_length: 65536
```

`hermes model` は、カスタムエンドポイントを設定する際にコンテキスト長の入力を求めます。自動検出させる場合は空欄のままにしてください。

:::tip 手動で設定すべき場合
- モデルの最大値よりも低いカスタムの `num_ctx` でOllamaを使用している場合
- モデルの最大値より下にコンテキストを制限したい場合（例: 128kのモデルでVRAMを節約するために8kにする）
- `/v1/models` を公開しないプロキシの背後で実行している場合
:::

---

### 名前付きカスタムプロバイダー

複数のカスタムエンドポイント（例: ローカルの開発サーバーとリモートのGPUサーバー）を扱う場合、`config.yaml` でそれらを名前付きカスタムプロバイダーとして定義できます:

```yaml
custom_providers:
  - name: local
    base_url: http://localhost:8080/v1
    # api_key を省略 — Hermesはキー不要のローカルサーバーには "no-key-required" を使用
  - name: work
    base_url: https://gpu-server.internal.corp/v1
    key_env: CORP_API_KEY
    api_mode: chat_completions   # 任意、URLから自動検出
  - name: anthropic-proxy
    base_url: https://proxy.example.com/anthropic
    key_env: ANTHROPIC_PROXY_KEY
    api_mode: anthropic_messages  # Anthropic互換プロキシ用
```

トリプル構文でセッションの途中で切り替えます:

```
/model custom:local:qwen-2.5       # 「local」エンドポイントを qwen-2.5 で使用
/model custom:work:llama3-70b      # 「work」エンドポイントを llama3-70b で使用
/model custom:anthropic-proxy:claude-sonnet-4  # プロキシを使用
```

対話的な `hermes model` メニューからも、名前付きカスタムプロバイダーを選択できます。

---

### クックブック: Together AI、Groq、Perplexity

[その他の互換プロバイダー](#other-compatible-providers) に挙げたクラウドプロバイダーはすべてOpenAIのREST方言を話すため、`custom_providers:` の下で同じように接続できます。実際に動作する3つのレシピを以下に示します。それぞれ `~/.hermes/config.yaml` に組み込み、対応するAPIキーは `~/.hermes/.env` に入れます。

#### Together AI

オープンウェイトモデル（Llama、MiniMax、Gemma、DeepSeek、Qwen）を、ファーストパーティAPIより大幅に安い価格でホストしています。マルチモデルのフリート向けの良いデフォルトです。

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: together
    base_url: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
    # api_mode: chat_completions  # デフォルト — 設定不要

model:
  default: MiniMaxAI/MiniMax-M2.7   # または together.ai/models の任意のモデル
  provider: custom:together
```

```bash
# ~/.hermes/.env
TOGETHER_API_KEY=your-together-key
```

セッションの途中でモデルを切り替えます:

```
/model custom:together:meta-llama/Llama-3.3-70B-Instruct-Turbo
/model custom:together:google/gemma-4-31b-it
/model custom:together:deepseek-ai/DeepSeek-V3
```

Togetherの `/v1/models` エンドポイントは機能するため、`hermes model` は利用可能なモデルを自動検出できます。

#### Groq

超高速推論（Llama-3.3-70Bで約500 tok/s）。カタログは小さいですが、レイテンシに敏感な対話的用途に強いです。

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: groq
    base_url: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY

model:
  default: llama-3.3-70b-versatile
  provider: custom:groq
```

```bash
# ~/.hermes/.env
GROQ_API_KEY=your-groq-key
```

#### Perplexity

ライブのWeb検索と引用を自動的に行うモデルが欲しいときに便利です。利用可能なモデルについて厳格です — 現在のリストは [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) で確認してください。

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: perplexity
    base_url: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: sonar
  provider: custom:perplexity
```

```bash
# ~/.hermes/.env
PERPLEXITY_API_KEY=your-perplexity-key
```

#### 1つの設定に複数のプロバイダー

3つのレシピは組み合わせられます — すべてを一緒に使い、`/model custom:<name>:<model>` でターンごとに切り替えます:

```yaml
custom_providers:
  - name: together
    base_url: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
  - name: groq
    base_url: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY
  - name: perplexity
    base_url: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: MiniMaxAI/MiniMax-M2.7
  provider: custom:together      # Togetherで起動。後から自由に切り替え
```

:::tip トラブルシューティング
- `hermes doctor` は、#15083のCLIバリデータ修正以降、これらの名前のいずれに対しても `Unknown provider` 警告を出さないはずです。
- プロバイダーの `/v1/models` エンドポイントに到達できない場合（Perplexityがよくある例）、`hermes model` はモデルをハードリジェクトせず、警告付きで永続化します — #15136を参照してください。
- `custom_providers:` を完全にスキップし、`CUSTOM_BASE_URL` 環境変数で素の `provider: custom` を使用するには、#15103を参照してください。
:::

---

### 適切なセットアップの選択

| ユースケース | 推奨 |
|----------|-------------|
| **とにかく動けばよい** | OpenRouter（デフォルト）またはNous Portal |
| **ローカルモデル、簡単なセットアップ** | Ollama |
| **本番のGPUサービング** | vLLMまたはSGLang |
| **Mac / GPUなし** | Ollamaまたはllama.cpp |
| **マルチプロバイダールーティング** | LiteLLM ProxyまたはOpenRouter |
| **コスト最適化** | ClawRouter、または `sort: "price"` を使うOpenRouter |
| **最大限のプライバシー** | Ollama、vLLM、またはllama.cpp（完全ローカル） |
| **エンタープライズ / Azure** | カスタムエンドポイントを使うAzure OpenAI |
| **中国のAIモデル** | z.ai（GLM）、Kimi/Moonshot（`kimi-coding` または `kimi-coding-cn`）、MiniMax、Xiaomi MiMo、またはTencent TokenHub（ファーストクラスプロバイダー） |

:::tip
`hermes model` でいつでもプロバイダーを切り替えられます — 再起動は不要です。どのプロバイダーを使っても、会話履歴、メモリ、スキルは引き継がれます。
:::

## オプションのAPIキー

| 機能 | プロバイダー | 環境変数 |
|---------|----------|--------------|
| Webスクレイピング | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY`、`FIRECRAWL_API_URL` |
| ブラウザ自動化 | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`、`BROWSERBASE_PROJECT_ID` |
| 画像生成 | [FAL](https://fal.ai/) | `FAL_KEY` |
| プレミアムTTSボイス | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| OpenAI TTS + 音声文字起こし | [OpenAI](https://platform.openai.com/api-keys) | `VOICE_TOOLS_OPENAI_KEY` |
| Mistral TTS + 音声文字起こし | [Mistral](https://console.mistral.ai/) | `MISTRAL_API_KEY` |
| RLトレーニング | [Tinker](https://tinker-console.thinkingmachines.ai/) + [WandB](https://wandb.ai/) | `TINKER_API_KEY`、`WANDB_API_KEY` |
| セッションをまたいだユーザーモデリング | [Honcho](https://honcho.dev/) | `HONCHO_API_KEY` |
| セマンティックな長期メモリ | [Supermemory](https://supermemory.ai) | `SUPERMEMORY_API_KEY` |

### Firecrawlのセルフホスティング

デフォルトでは、HermesはWeb検索とスクレイピングに [Firecrawlクラウド API](https://firecrawl.dev/) を使用します。Firecrawlをローカルで実行したい場合は、代わりにHermesをセルフホストインスタンスに向けることができます。完全なセットアップ手順については、Firecrawlの [SELF_HOST.md](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md) を参照してください。

**得られるもの:** APIキー不要、レート制限なし、ページごとのコストなし、完全なデータ主権。

**失うもの:** クラウド版は、高度なアンチボット回避（Cloudflare、CAPTCHA、IPローテーション）のためにFirecrawl独自の「Fire-engine」を使用します。セルフホストは基本的なfetch + Playwrightを使用するため、保護された一部のサイトは失敗することがあります。検索はGoogleの代わりにDuckDuckGoを使用します。

**セットアップ:**

1. Firecrawl Dockerスタックをクローンして起動します（5つのコンテナ: API、Playwright、Redis、RabbitMQ、PostgreSQL — 約4〜8 GBのRAMが必要）:
   ```bash
   git clone https://github.com/firecrawl/firecrawl
   cd firecrawl
   # .env で次を設定: USE_DB_AUTHENTICATION=false, HOST=0.0.0.0, PORT=3002
   docker compose up -d
   ```

2. Hermesをあなたのインスタンスに向けます（APIキー不要）:
   ```bash
   hermes config set FIRECRAWL_API_URL http://localhost:3002
   ```

セルフホストインスタンスで認証が有効になっている場合は、`FIRECRAWL_API_KEY` と `FIRECRAWL_API_URL` の両方を設定することもできます。

## OpenRouterプロバイダールーティング

OpenRouterを使用する場合、リクエストがプロバイダー間でどのようにルーティングされるかを制御できます。`~/.hermes/config.yaml` に `provider_routing` セクションを追加します:

```yaml
provider_routing:
  sort: "throughput"          # "price"（デフォルト）、"throughput"、または "latency"
  # only: ["anthropic"]      # これらのプロバイダーのみを使用
  # ignore: ["deepinfra"]    # これらのプロバイダーをスキップ
  # order: ["anthropic", "google"]  # この順序でプロバイダーを試行
  # require_parameters: true  # すべてのリクエストパラメータをサポートするプロバイダーのみを使用
  # data_collection: "deny"   # データを保存／学習する可能性のあるプロバイダーを除外
```

**ショートカット:** 任意のモデル名に `:nitro` を付加するとスループットでソートされ（例: `anthropic/claude-sonnet-4:nitro`）、`:floor` を付加すると価格でソートされます。

## OpenRouter Pareto Code Router

OpenRouterは、`openrouter/pareto-code` で実験的なコーディングモデルルーターを提供しています。これは、コーディング品質の基準を満たす最も安価なモデル（[Artificial Analysis](https://artificialanalysis.ai/) によるランク付け）にリクエストを自動ルーティングします。このモデルを選択し、`~/.hermes/config.yaml` で `min_coding_score` のつまみを調整します:

```yaml
model:
  provider: openrouter
  model: openrouter/pareto-code

openrouter:
  min_coding_score: 0.65   # 0.0〜1.0。高いほど強力（高価）なコーダー。デフォルト0.65。
```

注意:

- `min_coding_score` は、`model.model` が `openrouter/pareto-code` の場合に**のみ**送信されます。その他のモデルではこの値は無効（no-op）です。
- 空文字列に設定する（または行を削除する）と、OpenRouterが利用可能な最強のコーダーを選択します — pluginsブロックを省略した場合の文書化された挙動です。
- 選択は特定の日における各スコアに対して決定的ですが、実際に選ばれるモデルはPareto最前線が動く（新しいモデル、ベンチマークの更新）につれて変わることがあります。
- 完全なルーター挙動については、OpenRouterの [Pareto Routerドキュメント](https://openrouter.ai/docs/guides/routing/routers/pareto-router) を参照してください。
- メインのエージェントの代わりに特定の**補助タスク**（圧縮、ビジョンなど）にParetoコードルーターを使用するには、そのタスクの下に `extra_body.plugins` を設定します — [補助モデル → 補助タスク向けのOpenRouterルーティング＆Pareto Code](/docs/user-guide/configuration#openrouter-routing--pareto-code-for-auxiliary-tasks) を参照してください。

## フォールバックプロバイダー

プライマリモデルが失敗したとき（レート制限、サーバーエラー、認証失敗）にHermesが順番に試すバックアッププロバイダーのチェーンを設定します。正規の形式はトップレベルの `fallback_providers:` リストです:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
  - provider: anthropic
    model: claude-sonnet-4
    # base_url: http://localhost:8000/v1    # 任意、カスタムエンドポイント用
    # api_mode: chat_completions           # 任意の上書き
```

レガシーの単一ペア `fallback_model:` 辞書も後方互換のために引き続き受け付けられます:

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

起動すると、フォールバックは会話を失うことなくセッションの途中でモデルとプロバイダーを入れ替えます。チェーンはエントリごとに試行され、起動はセッションごとに1回限りです。

サポートされるプロバイダー: `openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`gemini`、`google-gemini-cli`、`qwen-oauth`、`huggingface`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`ollama-cloud`、`bedrock`、`ai-gateway`、`azure-foundry`、`opencode-zen`、`opencode-go`、`kilocode`、`xiaomi`、`arcee`、`gmi`、`stepfun`、`lmstudio`、`alibaba`、`alibaba-coding-plan`、`tencent-tokenhub`、`custom`。

:::tip
フォールバックは `config.yaml` を通じて — または `hermes fallback` で対話的に — 排他的に設定されます。いつ起動するか、チェーンがどのように進むか、補助タスクや委譲とどう相互作用するかの完全な詳細については、[フォールバックプロバイダー](/docs/user-guide/features/fallback-providers) を参照してください。
:::

---

## 関連項目

- [設定](/docs/user-guide/configuration) — 一般的な設定（ディレクトリ構造、設定の優先順位、ターミナルバックエンド、メモリ、圧縮など）
- [環境変数](/docs/reference/environment-variables) — すべての環境変数の完全なリファレンス
