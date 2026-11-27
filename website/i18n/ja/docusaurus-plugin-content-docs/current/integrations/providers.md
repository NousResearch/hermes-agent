---
title: "LLMとモデルのプロバイダー"
sidebar_label: "AIプロバイダー"
sidebar_position: 1
---

# LLMとモデルのプロバイダー

このページでは、Hermes Agent向けの推論プロバイダーのセットアップを扱います。OpenRouterやAnthropicのようなクラウドAPI、OllamaやvLLMのようなセルフホスト型エンドポイント、さらに高度なルーティングやフォールバック設定までを網羅します。Hermesを使用するには、少なくとも1つのプロバイダーを設定する必要があります。

## 推論プロバイダー

LLMに接続する手段が少なくとも1つ必要です。`hermes model` を使ってプロバイダーとモデルを対話的に切り替えるか、直接設定してください:

| プロバイダー | セットアップ |
|----------|-------|
| **Nous Portal** | `hermes model`（OAuth、サブスクリプション制） |
| **OpenAI Codex** | `hermes model` → **ChatGPT or Codex Subscription**（ChatGPT OAuth、Codexモデルを使用） |
| **Anthropic** | `hermes model`（OAuth経由のClaude Max + 追加利用クレジット。Anthropic APIキーや手動のsetup-tokenにも対応 — 下記の注記を参照） |
| **OpenRouter** | `~/.hermes/.env` 内の `OPENROUTER_API_KEY`、または `hermes auth add openrouter --type oauth`（OpenRouterのPKCEフローによるブラウザログイン。認証情報プールにキーを保存） |
| **Ramp Router** | `~/.hermes/.env` 内の `RAMP_ROUTER_API_KEY`（プロバイダー: `router`、エイリアス: `ramp-router`、`ramp`、`router.com`。Responsesネイティブのゲートウェイ、アカウントスコープのライブカタログ） |
| **Fireworks AI** | `~/.hermes/.env` 内の `FIREWORKS_API_KEY`（プロバイダー: `fireworks`、エイリアス: `fireworks-ai`、`fw`） |
| **NovitaAI** | `~/.hermes/.env` 内の `NOVITA_API_KEY`（プロバイダー: `novita`、200以上のモデル、Model API、Agent Sandbox、GPU Cloud） |
| **AI Gateway** | `~/.hermes/.env` 内の `AI_GATEWAY_API_KEY`（プロバイダー: `ai-gateway`） |
| **z.ai / GLM** | `~/.hermes/.env` 内の `GLM_API_KEY`（プロバイダー: `zai`） |
| **Kimi / Moonshot** | `~/.hermes/.env` 内の `KIMI_API_KEY`（プロバイダー: `kimi-coding`） |
| **Kimi / Moonshot（中国）** | `~/.hermes/.env` 内の `KIMI_CN_API_KEY`（プロバイダー: `kimi-coding-cn`、エイリアス: `kimi-cn`、`moonshot-cn`） |
| **Arcee AI** | `~/.hermes/.env` 内の `ARCEEAI_API_KEY`（プロバイダー: `arcee`、エイリアス: `arcee-ai`、`arceeai`） |
| **GMI Cloud** | `~/.hermes/.env` 内の `GMI_API_KEY`（プロバイダー: `gmi`、エイリアス: `gmi-cloud`、`gmicloud`） |
| **Nebius Token Factory** | `~/.hermes/.env` 内の `NEBIUS_API_KEY`（プロバイダー: `nebius-token-factory`、エイリアス: `nebius`、`nebius-tf`、`tokenfactory`） |
| **Actual Computer** | ホスト型リレーの場合は `~/.hermes/.env` 内の `ACTUAL_API_KEY`。ローカルデーモンの場合は `config.yaml` で `model.base_url` を設定（ループバックではキー不要）。プロバイダー: `actual`、エイリアス: `actual-computer`、`actualcomputer`、`aci`。 |
| **MiniMax** | `~/.hermes/.env` 内の `MINIMAX_API_KEY`（プロバイダー: `minimax`） |
| **MiniMax China** | `~/.hermes/.env` 内の `MINIMAX_CN_API_KEY`（プロバイダー: `minimax-cn`） |
| **xAI（Grok） — Responses API** | `~/.hermes/.env` 内の `XAI_API_KEY`（プロバイダー: `xai`） |
| **xAI Grok OAuth（SuperGrok）** | `hermes model` →「xAI Grok OAuth (SuperGrok / Premium+)」 — ブラウザログイン、APIキー不要。[ガイド](/docs/guides/xai-grok-oauth)を参照 |
| **Qwen Cloud（Alibaba DashScope）** | `~/.hermes/.env` 内の `DASHSCOPE_API_KEY`（プロバイダー: `alibaba`。中国本土のエンドポイント: `alibaba-cn`） |
| **Alibaba Cloud（Coding Plan）** | `ALIBABA_CODING_PLAN_API_KEY`（`DASHSCOPE_API_KEY` にフォールバック）（プロバイダー: `alibaba-coding-plan`、エイリアス: `alibaba_coding`。中国本土のエンドポイント: `ALIBABA_CODING_PLAN_CN_API_KEY` を使う `alibaba-coding-plan-cn`、共有キーにフォールバック） — 別個の課金SKUで、エンドポイントも異なります |
| **Alibaba Cloud（Token Plan）** | `~/.hermes/.env` 内の `ALIBABA_TOKEN_PLAN_API_KEY`（プロバイダー: `alibaba-token-plan`。中国本土のエンドポイント: `ALIBABA_TOKEN_PLAN_CN_API_KEY` を使う `alibaba-token-plan-cn`、共有キーにフォールバック） — Model Studioの定額トークン枠 |
| **Kilo Code** | `~/.hermes/.env` 内の `KILOCODE_API_KEY`（プロバイダー: `kilocode`） |
| **Xiaomi MiMo** | `~/.hermes/.env` 内の `XIAOMI_API_KEY`（プロバイダー: `xiaomi`、エイリアス: `mimo`、`xiaomi-mimo`） |
| **Tencent TokenHub** | `~/.hermes/.env` 内の `TOKENHUB_API_KEY`（プロバイダー: `tencent-tokenhub`、エイリアス: `tencent`、`tokenhub`、`tencentmaas`） |
| **Tencent TokenPlan** | `~/.hermes/.env` 内の `TOKENPLAN_API_KEY`（プロバイダー: `tencent-tokenplan`、エイリアス: `tokenplan`、`tencent-lkeap`。Anthropic Messagesエンドポイント） |
| **OpenCode Zen** | `~/.hermes/.env` 内の `OPENCODE_ZEN_API_KEY`（プロバイダー: `opencode-zen`） |
| **CommandCode** | `~/.hermes/.env` 内の `COMMANDCODE_API_KEY`（プロバイダー: `commandcode`、エイリアス: `commandcode-chat`。Claudeモデルは `commandcode-anthropic`、エイリアス: `commandcode-claude`）。GOAT/Pro/Max/Providerプランで動作します（$1のGoプランは不可 — APIアクセスなし）。 |
| **OpenCode Go** | `~/.hermes/.env` 内の `OPENCODE_GO_API_KEY`（プロバイダー: `opencode-go`） |
| **DeepSeek** | `~/.hermes/.env` 内の `DEEPSEEK_API_KEY`（プロバイダー: `deepseek`） |
| **Hugging Face** | `~/.hermes/.env` 内の `HF_TOKEN`（プロバイダー: `huggingface`、エイリアス: `hf`） |
| **Google / Gemini** | `~/.hermes/.env` 内の `GOOGLE_API_KEY`（または `GEMINI_API_KEY`）（プロバイダー: `gemini`） |
| **Google Vertex AI** | `hermes model` →「Google Vertex AI」（プロバイダー: `vertex`。サービスアカウントJSONまたはADCによるOAuth2、GCP課金） |
| **OpenAI API（直接）** | `~/.hermes/.env` 内の `OPENAI_API_KEY`（プロバイダー: `openai-api`、任意で `OPENAI_BASE_URL`） |
| **Azure AI Foundry** | `hermes model` →「Azure AI Foundry」（プロバイダー: `azure-foundry`。Azure OpenAI / Foundryのエンドポイントとキーを使用） |
| **AWS Bedrock** | `hermes model` →「AWS Bedrock」（プロバイダー: `bedrock`。boto3による標準のAWS認証情報チェーン） |
| **NVIDIA Build** | `~/.hermes/.env` 内の `NVIDIA_API_KEY`（プロバイダー: `nvidia`。build.nvidia.com上のNIMホストモデル） |
| **Ollama Cloud** | `hermes model` →「Ollama Cloud」（プロバイダー: `ollama-cloud`。クラウドホストのOllama API） |
| **Qwen OAuth** | `hermes model` →「Qwen OAuth」（プロバイダー: `qwen-oauth`。ブラウザでのPKCEログイン） |
| **MiniMax OAuth** | `hermes model` →「MiniMax (OAuth)」（プロバイダー: `minimax-oauth`。ブラウザでのPKCEログイン） |
| **StepFun** | `~/.hermes/.env` 内の `STEPFUN_API_KEY`（プロバイダー: `stepfun`） |
| **LM Studio** | `hermes model` →「LM Studio」（プロバイダー: `lmstudio`、任意で `LM_API_KEY`） |
| **Custom Endpoint** | `hermes model` →「Custom endpoint」を選択（`config.yaml` に保存） |

組み込みのOpenCodeプロバイダーはどちらも、すべてのリクエストで不透明な会話ごとの `x-opencode-session` ヘッダーを送ります（あらゆるトランスポートでのメインのターンに加え、圧縮、タイトル、承認チェック、`/btw` のサイドクエスチョンなどの補助呼び出し — ターンの終了後にバックグラウンドで実行されるものも含む。ヘッドレスのKanbanの `specify`/`decompose` やダッシュボードの見積もり呼び出しはタスクごとのキーを使い、ライブセッションがまったくない単発の処理、たとえばレビューパネルからのDesktopのコミットメッセージ生成は、新しい一時的なキーを送ります）。OpenCodeはこれを使って会話を1つのバックエンドに固定し、プロンプトキャッシュを温かく保ちます。値はHermesのセッションID（またはKanbanのタスクID）から導出され、個人データは含みません。

2つの組み込みOpenCodeプロバイダーは、それぞれ `opencode.ai` 上の独自のリレーに固定されています（`opencode-zen` → `/zen/v1`、`opencode-go` → `/zen/go/v1`）。もう一方のリレーが残した `model.base_url` は選択したプロバイダーのリレーに修復され、選んだモデル（`-m`、`/model`、フォールバックのエントリ、チャンネルの上書き）がどのリレーを使うかを決めます — そのため、ZenのモデルからGo専用のモデルに切り替えても、リクエストがZenに送られることはありません。同じモデルごとのルーティングは、セッションの再開時（`hermes --resume`、`/resume`、TUIとデスクトップの再開経路）にも再適用されます: セッションが別のOpenCodeモデルを実行していた間に記録されたワイヤー形式やリレーのURLが、セッションを再び開いたモデルに引き継がれることはありません。IDに `-vision` マーカーを持つOpenCodeモデル（たとえば `deepseek-v4-flash-vision-exp`）は、models.devに掲載される前からビジョン対応として扱われるため、`agent.image_input_mode: auto` は `supports_vision` の上書きなしに画像をネイティブに添付します。`providers:` の下で定義した、名前がファミリーのスラッグを拡張するカスタムプロバイダー（たとえば `opencode-go-bridge`）も、ファミリーのモデルごとのAPIモードのルーティングと `/v1` の処理を受けますが、その `base_url` は宣言どおりに扱われます: 実際に指しているリレーにちなんだ名前を付けてください。OpenCodeプロバイダーに向けた補助タスク（`auxiliary.compression`、タイトル、ビジョン、MoA）も同じモデルごとの表に従うため、`gpt-5.6-luna` のようなResponses専用のモデルや `minimax-m2.5` のようなAnthropicワイヤーのモデルも、メインの会話とまったく同じようにそこで動作します。

公式のAPIキー方式については、専用の [Google Geminiガイド](../guides/google-gemini.md) を参照してください。

:::tip モデルキーのエイリアス
`model:` 設定セクションでは、モデルIDのキー名として `default:` または `model:` のどちらも使用できます。`model: { default: my-model }` と `model: { model: my-model }` はまったく同じように動作します。
:::


### Nous Portal {#nous-portal}

[Nous Portal](https://portal.nousresearch.com) は、Nous Researchの統合サブスクリプションゲートウェイであり、**Hermes Agentを実行する推奨の方法**です。1回のOAuthログインで、300以上のフロンティアなエージェント向けモデル（Claude、GPT、Gemini、DeepSeek、Qwen、Kimi、GLM、MiniMax、Grokなど）に加えて [Tool Gateway](../user-guide/features/tool-gateway.md)（Web検索、画像生成、TTS、ブラウザ自動化）をカバーし、プロバイダーごとの別々のアカウントではなく、Nousのサブスクリプションに対して課金されます。

```bash
hermes setup --portal     # fresh install — OAuth + provider + gateway in one command
hermes model              # existing install — pick "Nous Portal" from the list
hermes portal info        # inspect login + routing at any time
```

まだサブスクリプションをお持ちでない場合は、[portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription) で取得してください。

**詳細について:** 専用の [Nous Portal連携ページ](/docs/integrations/nous-portal)（サブスクリプションの内容、モデルカタログ、トラブルシューティング）と、ステップごとの [Nous PortalでHermes Agentを実行するガイド](/docs/guides/run-hermes-with-nous-portal) を参照してください。

**クライアントの識別。** Hermes AgentからのすべてのPortalリクエストには、インストール済みのリリースに自動的に合わせた `client=hermes-client-v<version>` タグ（例: `client=hermes-client-v0.13.0`）が付きます。これはPortalのすべての経路 — メインのチャットループ、補助呼び出し、圧縮の要約器、Web抽出 — で送られ、Portal側のテレメトリがHermesのトラフィックを他のクライアントと区別できるようにします。設定は不要で、`hermes update` するとタグも自動的に更新されます。

**JWT認証（自動）。** HermesはPortalリクエストに対してスコープ付きの `inference:invoke` JWTを優先し、レガシーの不透明なセッションキーの経路をフォールバックとして使います。設定は不要です — 認証情報はOAuthフローによって管理され、透過的にローテーションされます。失効したリフレッシュトークンは、リプレイのループを避けるため隔離されます。


:::info Codexに関する注記
OpenAI Codexプロバイダーは、デフォルトでデバイスコードで認証します（URLを開いてコードを入力）。デバイスコードグラントを無効にしている組織は、代わりにブラウザの認可コード + PKCEフローにオプトインできます: `hermes auth add openai-codex --browser`（1回のログイン）または `config.yaml` の `auth.codex_login_flow: browser`（`hermes model` を含むすべてのCodexログイン）。このフローは `http://localhost:1455/auth/callback` で待ち受けます — Codexクライアントに登録されたリダイレクトURIなので、ポートは固定です。すでに使用中の場合（Codex CLIのサインインが進行中など）、Hermesはその旨を伝えてデバイスコードにフォールバックします。SSH越しではリスナーにトンネルが必要です（`ssh -N -L 1455:127.0.0.1:1455 user@host`、[SSH越しのOAuth](/docs/guides/oauth-over-ssh)を参照）。Hermesは得られた認証情報を `~/.hermes/auth.json` 配下の独自の認証ストアに保存し、`~/.codex/auth.json` が存在する場合は既存のCodex CLI認証情報をインポートできます。Codex CLIのインストールは不要です。（Hermes自身のリフレッシュが失敗したときの）Codex CLIのログインの自動採用は `auth.adopt_external_logins` で制御されます — [借用したCLIログイン](../user-guide/security.md#borrowed-cli-logins)を参照してください。

トークンのリフレッシュが終端のエラー（HTTP 4xx、`invalid_grant`、失効したグラントなど）で失敗した場合、Hermesはリフレッシュトークンを無効とマークしてそれ以上リプレイしないため、同じ認証失敗が大量に表示されることはありません。次のリクエストでは、代わりに型付きの再認証メッセージが表示されます。`hermes auth add openai-codex`（または `hermes model` → **ChatGPT or Codex Subscription**）を実行して新しいログインを開始してください（デバイスコード、またはループバックのPKCEフローには `--browser`）。隔離は次に交換が成功したときに解除されます。

デバイスログインは、中間装置がX25519MLKEM768のような耐量子グループを拒否する場合、Python/OpenSSL 3.5以降で `[SSL: UNEXPECTED_EOF_WHILE_READING]` やTLSハンドシェイクのタイムアウトで失敗することがあります（curlは動作する場合があります）。1回限りの接続の切断は致命的ではありません: ブラウザでの承認を待つ間、Hermesは最大6回連続のトランスポートエラーまでポーリングを続け（デバイスコードのリクエストとトークンの交換は2回リトライします）、それから諦めるため、このエラーが表面化するのはネットワークが継続的に壊れている場合だけです。HermesはデフォルトのTLSポリシーを変更しません。`hermes model` を実行する前に、`Groups` を従来の曲線に制限する設定を `OPENSSL_CONF` で指定するか、TLS 1.2で診断してください:

```ini
openssl_conf = openssl_init

[openssl_init]
ssl_conf = ssl_sect

[ssl_sect]
system_default = system_default_sect

[system_default_sect]
Groups = x25519:secp256r1:secp384r1:x448
```
:::

:::warning
Nous Portal、Codex、カスタムエンドポイントを使用している場合でも、一部の機能（ビジョン、圧縮、MoA）は別個の「補助（auxiliary）」モデルを使用します。デフォルト（`auxiliary.*.provider: "auto"`）では、Hermesはこれらのタスクをあなたの**メインチャットモデル**（`hermes model` で選んだのと同じモデル）にルーティングします。各タスクを個別に上書きして、より安価／高速なモデル（例: OpenRouter上のGemini Flash）にルーティングできます — [補助モデル](../user-guide/configuration.md#auxiliary-models) を参照してください。
:::

:::tip Nous Tool Gateway
有料のNous Portalサブスクライバーは、**[Tool Gateway](../user-guide/features/tool-gateway.md)** へのアクセスも得られます — Web検索、画像生成、TTS、ブラウザ自動化がサブスクリプション経由でルーティングされます。追加のAPIキーは不要です。新規インストールでは、`hermes setup --portal` が1コマンドでログインし、Nousをプロバイダーに設定し、ゲートウェイを有効にします。既存のユーザーは `hermes model` から、またはツールごとに `hermes tools` から有効にできます。ルーティングはいつでも `hermes portal info` で確認できます。
:::

### モデル管理のための2つのコマンド

Hermesには、異なる目的を持つ**2つ**のモデルコマンドがあります:

| コマンド | 実行場所 | 役割 |
|---------|-------------|--------------|
| **`hermes model`** | ターミナル（セッションの外） | 完全なセットアップウィザード — プロバイダーの追加、OAuthの実行、APIキーの入力、エンドポイントの設定 |
| **`/model`** | Hermesのチャットセッション内 | **すでに設定済み**のプロバイダーとモデルの間でのクイック切り替え |

まだセットアップしていないプロバイダーに切り替えようとしている場合（例: OpenRouterのみ設定済みでAnthropicを使いたい場合）は、`/model` ではなく `hermes model` が必要です。まずセッションを終了し（`Ctrl+C` または `/quit`）、`hermes model` を実行してプロバイダーのセットアップを完了し、新しいセッションを開始してください。


### サブスクリプションプラン: プランが何に対して支払われるか

いくつかのプロバイダーでは、APIキーの代わりに**消費者向けサブスクリプション**（Claude Max、ChatGPT、SuperGrok / X Premium+など）でHermesにサインインできます。そのサブスクリプションが実際に何に対して支払われ、何に対しては支払われないかはプロバイダーごとに異なり、課金の驚きの最も一般的な原因です。下の表は短縮版で、詳細は各プロバイダーのセクションにあります。

> *not currently documented* と記されたセルは、まさにその意味です: Hermesのドキュメントはまだその挙動を規定していません。推測せず、プロバイダーの課金ダッシュボードを確認し、未解決の質問として扱ってください。

| プラン / 経路 | Hermesで使えるか？ | 消費されるもの | 消費されないもの | よくある驚き |
|---|---|---|---|---|
| **Anthropic — Claude Max + OAuth** | ✅ はい — `hermes model` → Anthropic OAuth。Max **と**購入した追加利用クレジットが必要 | Maxプランに上乗せした**追加／超過クレジット** | **ベースのMaxプランの利用枠**（Claude Codeにデフォルトで含まれる利用分） | 含まれているMaxの利用枠に手が付いていないのに、Hermesの利用がすべて「追加利用」として課金される |
| **Anthropic — Claude Pro** | ❌ いいえ — ProのサブスクライバーはOAuthの経路を使えない | なし（経路が利用不可） | Proのサブスクリプション | Proでも動きそうに見えるが、動かない。代わりに `ANTHROPIC_API_KEY` を使う（従量課金、Claudeのサブスクリプションとは独立） |
| **OpenAI Codex — ChatGPTプランのOAuth** | ✅ はい — `hermes model` → **ChatGPT or Codex Subscription**（ChatGPTのOAuthデバイスコードログイン、Codexモデルを使用） | *not currently documented* | *not currently documented* | ドキュメントがカバーしているのは認証とトークンのリフレッシュだけで、プランのクォータのセマンティクスはまだ文書化されていない |
| **xAI — SuperGrok / X Premium+のOAuth** | ✅ はい — ブラウザのOAuth、APIキー不要 | あなたの**サブスクリプションのクォータ**（X Searchについては明示的に文書化: OAuthがAPIキーより優先され、「APIの支出の代わりにサブスクリプションのクォータを使う」）。それ以外の推論クォータのセマンティクス: *not currently documented* | OAuthの認証情報が設定され優先されている場合の `XAI_API_KEY` / 従量課金のAPI支出 | ログインに成功した後の `HTTP 403` — アプリ内のサブスクリプションが有効でも、xAIはOAuthのAPIアクセスを特定のSuperGrokの階層に制限している |
| **Google — Geminiの消費者向けプラン（Google AI Pro / Ultra）** | ❌ 文書化された経路なし — `gemini` プロバイダーはAPIキーのみ（`GOOGLE_API_KEY` / `GEMINI_API_KEY`）。Vertex AIはGCPの課金を使う | あなたの**APIキーのクォータ**（無料枠または課金が有効なGoogle Cloudプロジェクト） — *消費者向けプランの消費は not currently documented* | *not currently documented* | Hermesはユーザーの1ターンあたり複数回のモデル呼び出しを行うことがあるため、無料枠のキーは数回のエージェントのターンで使い切られることがある |

**Anthropic。** OAuthの経路はあなたのAnthropicアカウントに対してClaude Codeとしてルーティングし、**購入した追加利用クレジットのあるClaude Maxプランでのみ動作します** — ベースのMaxの利用枠がHermesに消費されることは決してなく、消費されるのは上乗せの追加／超過クレジットだけです。Claude Proのサブスクライバーはこの経路を使えません。サポートされる代替手段は `ANTHROPIC_API_KEY` で、そのキーの組織に対して標準のAPI料金で従量課金されます。下記の [Anthropic（ネイティブ）](#anthropic-native) を参照してください。

**OpenAI Codex。** HermesはChatGPTのデバイスコードOAuthで認証し、認証情報を `~/.hermes/auth.json` に保存し、`~/.codex/auth.json` から既存のCodex CLIの認証情報をインポートできます。どのChatGPTのプラン階層が対象となるか、またHermesの利用がプランのCodexの制限にどう数えられるかは**現在文書化されていません** — [Nous Portal](#nous-portal) の下のCodexに関する注記がカバーしているのは、認証とトークンのリフレッシュの挙動だけです。

**xAI（SuperGrok / X Premium+）。** ブラウザのOAuthは、有効なSuperGrokのサブスクリプション、またはリンクされたXアカウントのX Premium+のサブスクリプションのどちらでも動作し、同じベアラートークンがxAIに直接接続するツール（TTS、画像生成、動画生成、文字起こし、X Search）でも再利用されます。ログインに成功した後で推論が `HTTP 403` を返す場合、それは古いトークンではなくxAI側の階層/権利による制限です — 回避策は `XAI_API_KEY` に切り替えることです。下記の [xAI（Grok）](#xai-grok--responses-api--prompt-caching) と [xAI Grok OAuthガイド](/docs/guides/xai-grok-oauth) を参照してください。

**Google Gemini。** 現在、消費者向けのGeminiサブスクリプションでHermesにサインインする方法はありません — `gemini` プロバイダーはAPIキーを受け取り、[Google Vertex AI](#google-vertex-ai) はGCPプロジェクトに課金します。エージェントでの利用には課金が有効なGoogle Cloudプロジェクトを推奨します。無料枠のクォータは、長時間のエージェントセッションには小さすぎます。[Google Geminiガイド](../guides/google-gemini.md) を参照してください。

:::tip 5つではなく1つのサブスクリプション
プロバイダーごとのプランのセマンティクスを一切追いたくない場合は、[Nous Portal](#nous-portal) が1回のOAuthログインで、1つのサブスクリプションのもとに300以上のモデルをカバーします。
:::

### Anthropic（ネイティブ） {#anthropic-native}

OpenRouterのプロキシなしで、Anthropic API経由でClaudeモデルを直接使用します。3つの認証方法をサポートします:

明示的な環境の認証情報が選択されていない場合、認証情報プール内のHermesが所有するOAuthグラントが、
借用したClaude Codeのログインより優先されます。借用した
ログインは、所有するOAuthグラントが利用できない場合のフォールバックのままです —
ただし `auth.adopt_external_logins: false` が設定されている場合、HermesはClaude Codeの認証情報を
決して読んだりリフレッシュしたりしません（[借用したCLIログイン](../user-guide/security.md#borrowed-cli-logins)を
参照）。
補助の認証の回復は、失敗したリクエストが使った認証情報をリフレッシュし、無関係な
周囲のログインはリフレッシュしません。借用したログインをローテーションすると、その所有者の
リフレッシュトークンが無効になってしまう可能性があるためです。

:::caution Claude Maxの「追加利用」クレジットが必要
`hermes model` → Anthropic OAuth（または `hermes auth add anthropic --type oauth`）で認証すると、Hermesはあなたのアカウントに対してClaude Codeとしてルーティングします。**これはClaude Maxプランに加入し、追加利用クレジットを購入している場合にのみ動作します。** ベースのMaxプランの利用枠（Claude Codeにデフォルトで含まれる利用分）はHermesでは消費されません — 上乗せした追加／超過クレジットのみが消費されます。Claude Proのサブスクライバーはこの方式を利用できません。

Max + 追加クレジットがない場合は、代わりに `ANTHROPIC_API_KEY` を使用してください — リクエストはそのキーの組織に対して従量課金されます（標準的なAPI料金で、Claudeサブスクリプションとは独立しています）。
:::

```bash
# With an API key (pay-per-token)
export ANTHROPIC_API_KEY=***
hermes chat --provider anthropic --model claude-sonnet-4-6

# Preferred: authenticate through `hermes model`
# Hermes will use Claude Code's credential store directly when available
hermes model

# Manual override with a setup-token (fallback / legacy)
export ANTHROPIC_TOKEN=***  # setup-token or manual OAuth token
hermes chat --provider anthropic

# Auto-detect Claude Code credentials (if you already use Claude Code)
hermes chat --provider anthropic  # reads Claude Code credential files automatically
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

### GitHub Copilot（labsプラグイン）

GitHub Copilotのプロバイダー（Copilot API用の `copilot`、ACP経由のCopilot CLI用の `copilot-acp`）は、もはやHermesに組み込まれていません。これらは `github-copilot` **labsプラグイン**として提供されます — Nous Researchが作成していますが、サポート対象外です:

```bash
hermes plugins install github-copilot
```

すでに `copilot` または `copilot-acp` を使っているプロファイル（メインモデル、フォールバック、補助、委譲、MoA、cron）には、`hermes update` によって、または次回のエージェントやゲートウェイの起動時に、プラグインが自動的にインストールされます。

`hermes auth add copilot`（デバイスコードログイン）でサインインするか、`COPILOT_GITHUB_TOKEN`、`GH_TOKEN`、`GITHUB_TOKEN` を設定します。`gh` CLIやCopilot CLIで行ったログインはもう読み込まれません: それに依存していた場合は `hermes auth add copilot` を実行してください。

セットアップ、モデル、環境変数: [github-copilotプラグインのドキュメント](https://github.com/NousResearch/hermes-labs-plugins/tree/main/github-copilot)。

### ファーストクラスのAPIキープロバイダー

これらのプロバイダーは、専用のプロバイダーIDによる組み込みサポートを備えています。APIキーを設定し、`--provider` で選択します:

```bash
# Fireworks AI
hermes chat --provider fireworks --model accounts/fireworks/models/kimi-k2p6
# Requires: FIREWORKS_API_KEY in ~/.hermes/.env

# NovitaAI Model API
hermes chat --provider novita --model moonshotai/kimi-k2.5
# Requires: NOVITA_API_KEY in ~/.hermes/.env

# Ramp Router (model IDs come from your account's live catalog)
hermes chat --provider router --model gpt-5.4-mini
# Requires: RAMP_ROUTER_API_KEY in ~/.hermes/.env

# z.ai / ZhipuAI GLM
hermes chat --provider zai --model glm-5
# Requires: GLM_API_KEY in ~/.hermes/.env

# Kimi / Moonshot AI (international: api.moonshot.ai)
hermes chat --provider kimi-coding --model kimi-for-coding
# Requires: KIMI_API_KEY in ~/.hermes/.env

# Kimi / Moonshot AI (China: api.moonshot.cn)
hermes chat --provider kimi-coding-cn --model kimi-k2.5
# Requires: KIMI_CN_API_KEY in ~/.hermes/.env

# MiniMax (global endpoint)
hermes chat --provider minimax --model MiniMax-M2.7
# Requires: MINIMAX_API_KEY in ~/.hermes/.env

# MiniMax (China endpoint)
hermes chat --provider minimax-cn --model MiniMax-M2.7
# Requires: MINIMAX_CN_API_KEY in ~/.hermes/.env

# Qwen Cloud / DashScope (Qwen models)
hermes chat --provider alibaba --model qwen3.5-plus
# Requires: DASHSCOPE_API_KEY in ~/.hermes/.env

# Xiaomi MiMo
hermes chat --provider xiaomi --model mimo-v2-pro
# Requires: XIAOMI_API_KEY in ~/.hermes/.env

# Tencent TokenHub (Hy4 preview)
hermes chat --provider tencent-tokenhub --model hy4-preview
# Requires: TOKENHUB_API_KEY in ~/.hermes/.env

# Tencent TokenPlan (Hy4 preview via Anthropic Messages endpoint)
hermes chat --provider tencent-tokenplan --model hy4-preview
# Requires: TOKENPLAN_API_KEY in ~/.hermes/.env

# Arcee AI (Trinity models)
hermes chat --provider arcee --model trinity-large-thinking
# Requires: ARCEEAI_API_KEY in ~/.hermes/.env

# Meta Model API (Muse Spark family)
hermes chat --provider meta-ai --model muse-spark-1.2
# Requires: MODEL_API_KEY in ~/.hermes/.env

# GMI Cloud
# Use the exact model ID returned by GMI's /v1/models endpoint.
hermes chat --provider gmi --model zai-org/GLM-5.1-FP8
# Requires: GMI_API_KEY in ~/.hermes/.env

# Nebius Token Factory
hermes chat --provider nebius --model deepseek-ai/DeepSeek-V4-Pro
# Requires: NEBIUS_API_KEY in ~/.hermes/.env
```

Fireworksは、`accounts/fireworks/models/kimi-k2p6` のようなネイティブのスラッシュ形式のカタログIDを使います。`hermes model` を実行して **Fireworks AI** を選び、ライブカタログから選択するか、別のFireworksのモデルIDを入力してください。デフォルトのエンドポイントは `https://api.fireworks.ai/inference/v1` です。別のエンドポイントは `.env` ではなく `config.yaml` の `model.base_url` で設定してください。

または `config.yaml` でプロバイダーを恒久的に設定します:
```yaml
model:
  provider: "gmi"
  default: "zai-org/GLM-5.1-FP8"
```

ベースURLは、`NOVITA_BASE_URL`、`GLM_BASE_URL`、`KIMI_BASE_URL`、`MINIMAX_BASE_URL`、`MINIMAX_CN_BASE_URL`、`DASHSCOPE_BASE_URL`、`XIAOMI_BASE_URL`、`GMI_BASE_URL`、`META_BASE_URL`、`TOKENHUB_BASE_URL` の各環境変数で上書きできます。

:::note Metaのコントリビューター階層
`muse-spark-1.2-contributor` と `muse-spark-1.3-contributor` はMetaのコントリビューター階層です — Metaがあなたのプロンプトと補完を学習に使う可能性があるため、どちらかを使う前に[対話的なモデル選択で確認が求められます](../user-guide/configuring-models.md)。現在の価格とレート制限については、[Meta Model APIの価格とレート制限](https://dev.meta.ai/docs/pricing-rate-limits/)を参照してください。機密性のある作業には、標準の `muse-spark-1.2` / `muse-spark-1.3`（学習なし）を使ってください。
:::

:::note Z.AIエンドポイントの自動検出
Z.AI / GLMプロバイダーを使用する場合、Hermesは複数のエンドポイント（グローバル、中国、codingバリアント）を自動的にプローブし、あなたのAPIキーを受け付けるものを見つけます。`GLM_BASE_URL` を手動で設定する必要はありません — 動作するエンドポイントが自動的に検出されキャッシュされます。
:::

### xAI（Grok） — Responses API + プロンプトキャッシュ {#xai-grok--responses-api--prompt-caching}

xAIはResponses API（`codex_responses` トランスポート）経由で接続され、Grok 4モデルで推論を自動的にサポートします — `reasoning_effort` パラメータは不要で、サーバーがデフォルトで推論します。`~/.hermes/.env` に `XAI_API_KEY` を設定し、`hermes model` でxAIを選択するか、ショートカットとして `grok` を `/model grok-4-fast-reasoning` のように指定します。

SuperGrokとX Premium+のサブスクライバーは、APIキーを使う代わりにブラウザのOAuthでサインインできます — `hermes model` で **xAI Grok OAuth (SuperGrok / Premium+)** を選ぶか、`hermes auth add xai-oauth` を実行します。同じOAuthのベアラートークンは、xAIに直接接続するツール（TTS、画像生成、動画生成、文字起こし）で自動的に再利用されます。全体のフローについては [xAI Grok OAuthガイド](/docs/guides/xai-grok-oauth) を参照してください — Hermesがリモートホストで動作している場合は、必要な `ssh -L` トンネルについて [SSH越しのOAuth / リモートホスト](/docs/guides/oauth-over-ssh) も参照してください。

xAIをプロバイダーとして使用する場合（`x.ai` を含む任意のベースURL）、Hermesはすべてのリクエストに `x-grok-conv-id` ヘッダーを送信することで、自動的にプロンプトキャッシュを有効にします。これにより、会話セッション内のリクエストが同じサーバーにルーティングされ、xAIのインフラがキャッシュされたシステムプロンプトと会話履歴を再利用できるようになります。

設定は不要です — xAIエンドポイントが検出され、セッションIDが利用可能になると、キャッシュが自動的に有効になります。これによりマルチターン会話のレイテンシとコストが削減されます。

xAIは専用のTTSエンドポイント（`/v1/tts`）も提供しています。`hermes tools` → Voice & TTSで **xAI TTS** を選択するか、設定については [Voice & TTS](../user-guide/features/tts.md#text-to-speech) ページを参照してください。

**廃止されるxAIモデルの移行（2026年5月15日）:** xAIは2026-05-15に `grok-4*`、`grok-3`、`grok-code-fast-1`、`grok-imagine-image-pro` を廃止します。`hermes doctor` と `hermes chat` の起動時の両方で、廃止される参照をまだ指している設定が検出され、推奨される置き換えが表示されます。`hermes migrate xai` で設定を一度に書き換えられます — デフォルトはドライランで、変更を書き込むには `--apply` を追加します（以前の設定のタイムスタンプ付きコピーが先に `backups/config/` に置かれます）。

```bash
hermes migrate xai          # preview replacements
hermes migrate xai --apply  # rewrite ~/.hermes/config.yaml in place
```

**xAIのWeb検索バックエンド。** [Web検索](../user-guide/features/web-search.md)ツールセットが有効な場合、`web.backend: xai` は同じ `XAI_API_KEY` / OAuthの認証情報を使って、xAIのホスト型検索エンドポイント経由で検索をルーティングします。xAIがすでにプロバイダーとして設定されていれば、追加のセットアップは不要です。

### NovitaAI

[NovitaAI](https://novita.ai) は、ビルダーとエージェントのためのAIネイティブなクラウドです。3つの製品ラインは、200以上のモデルのためのModel API、AIエージェントを構築・実行するためのAgent Sandbox、スケーラブルなコンピュートのためのGPU Cloudで、すべてが1つのプラットフォームから利用できます。

```bash
# Use any available model
hermes chat --provider novita --model moonshotai/kimi-k2.5
# Requires: NOVITA_API_KEY in ~/.hermes/.env

# Short alias
hermes chat --provider novita-ai --model deepseek/deepseek-v3-0324
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "novita"
  default: "moonshotai/kimi-k2.5"
  base_url: "https://api.novita.ai/openai/v1"
```

APIキーは [novita.ai/settings/key-management](https://novita.ai/settings/key-management) で取得してください。ベースURLは `NOVITA_BASE_URL` で上書きできます。

### Ollama Cloud — マネージドOllamaモデル、OAuth + APIキー

[Ollama Cloud](https://ollama.com/cloud) は、ローカルのOllamaと同じオープンウェイトカタログをホストしますが、GPU要件がありません。`hermes model` で **Ollama Cloud** を選択し、[ollama.com/settings/keys](https://ollama.com/settings/keys) のAPIキーを貼り付けると、Hermesが利用可能なモデルを自動検出します。

```bash
hermes model
# → pick "Ollama Cloud"
# → paste your OLLAMA_API_KEY
# → select from discovered models (gpt-oss:120b, glm-4.6:cloud, qwen3-coder:480b-cloud, etc.)
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

### DeepInfra

DeepInfra（`--provider deepinfra`、`DEEPINFRA_API_KEY`）は、そのカタログからライブで検出されます。推論はDeepInfraのトップレベルの `reasoning_effort` フィールドで制御されるため、`agent.reasoning_effort`、`/reasoning <level>`、`--reasoning`、モデルごとの `agent.reasoning_overrides` は**両方向に**機能します: デフォルトでオフのモデル（DeepSeek-V4.x）では労力の指定で思考がオンになり、デフォルトでオンのモデル（GLM-4.6、Qwen3-Thinking）では `/reasoning none` でオフになります。推論を未設定のままにすると、DeepInfraのモデルごとのデフォルトが維持されます。`xhigh` はネイティブで、`ultra` は `max` として送られます。

### AWS Bedrock

AWS Bedrock経由でAnthropic Claude、Amazon Nova、DeepSeek v3.2、Meta Llama 4、その他のモデルを利用します。AWS SDK（`boto3`）の認証情報チェーンを使用します — APIキーは不要で、標準的なAWS認証だけです。

```bash
# Simplest — named profile in ~/.aws/credentials
hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6

# Or with explicit env vars
AWS_PROFILE=myprofile AWS_REGION=us-east-1 hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6
```

または `config.yaml` で恒久的に:
```yaml
model:
  provider: "bedrock"
  default: "us.anthropic.claude-sonnet-4-6"
bedrock:
  region: "us-east-1"          # or set AWS_REGION
  # profile: "myprofile"       # or set AWS_PROFILE
  # discovery: true            # auto-discover region from IAM
  # guardrail:                 # optional Bedrock Guardrails
  #   guardrail_identifier: "your-guardrail-id"
  #   guardrail_version: "DRAFT"
```

認証は標準のboto3チェーンを使用します: 明示的な `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`、`~/.aws/credentials` の `AWS_PROFILE`、EC2/ECS/LambdaのIAMロール、IMDS、またはSSO。すでにAWS CLIで認証済みの場合、環境変数は不要です。

Bedrockは内部で**Converse API**を使用します — リクエストはBedrockのモデル非依存の形式に変換されるため、同じ設定がClaude、Nova、DeepSeek、Llamaの各モデルで機能します。デフォルト以外のリージョナルエンドポイントを呼び出す場合のみ `BEDROCK_BASE_URL` を設定してください。

IAMのセットアップ、リージョンの選択、クロスリージョン推論の手順については、[AWS Bedrockガイド](../guides/aws-bedrock.md) を参照してください。

### Google Vertex AI {#google-vertex-ai}

VertexのOpenAI互換エンドポイント経由で、Google Cloud Vertex AI上のGeminiモデルを利用します。認証は**OAuth2**です — サービスアカウントのJSONまたはApplication Default Credentials（ADC）から発行される短命なアクセストークン（約1時間）を使います。**静的なAPIキーはありません**。Hermesがトークンを発行して自動的にリフレッシュし、セッション途中の `401` での再発行も行います。

```bash
# Service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env
# or Application Default Credentials
gcloud auth application-default login

hermes model   # → "Google Vertex AI" → project → region → model
```

または `config.yaml` で（プロジェクト/リージョンは秘密ではないのでここに置き、認証情報のパスは `.env` に置きます）:
```yaml
model:
  provider: "vertex"
  default: "google/gemini-3-flash-preview"   # Vertex requires the google/ prefix
vertex:
  project_id: "my-gcp-project"   # blank → use the project embedded in the credentials
  region: "global"               # required for the Gemini 3.x previews
```

`VERTEX_PROJECT_ID` / `VERTEX_REGION` 環境変数は `config.yaml` の値より優先されます。Hermesは初回使用時に `google-auth` を遅延インストールします。管理されたインストールの修復が必要な場合は `hermes setup` を実行してください。全体の手順については [Google Vertex AIガイド](/docs/guides/google-vertex) を、静的なAPIキーを使うAI Studioの経路については [Google Geminiガイド](../guides/google-gemini.md) を参照してください。

### Qwen Portal（OAuth）

ブラウザベースのOAuthログインを使うAlibabaのQwen Portalです。`hermes model` で **Qwen OAuth (Portal)** を選択し、ブラウザでサインインすると、Hermesがリフレッシュトークンを永続化します。

```bash
hermes model
# → pick "Qwen OAuth (Portal)"
# → browser opens; sign in with your Alibaba account
# → confirm — credentials are saved to ~/.hermes/auth.json

hermes chat   # uses portal.qwen.ai/v1 endpoint
```

または `config.yaml` を設定します:
```yaml
model:
  provider: "qwen-oauth"
  default: "qwen3-coder-plus"
```

ポータルエンドポイントが移転した場合のみ `HERMES_QWEN_BASE_URL` を設定してください（デフォルト: `https://portal.qwen.ai/v1`）。

:::tip Qwen OAuth対Qwen Cloud（Alibaba DashScope）
`qwen-oauth` はOAuthログインを使う消費者向けのQwen Portalを使用します — 個人ユーザーに最適です。`alibaba` プロバイダーはQwen Cloud（Alibaba DashScope）を `DASHSCOPE_API_KEY` で使用します — プログラム的／本番ワークロードに最適です。どちらもQwenファミリーのモデルにルーティングしますが、異なるエンドポイントに存在します。
:::

### Alibaba Cloud（Coding Plan）

Alibabaの**Coding Plan**（標準のDashScope APIアクセスとは別の料金SKU）に加入している場合、Hermesはそれを独自のファーストクラスプロバイダー `alibaba-coding-plan` として公開します。エンドポイント: `https://coding-intl.dashscope.aliyuncs.com/v1`。通常の `alibaba` プロバイダーと同様にOpenAI互換ですが、ベースURLと課金面が異なります。

```yaml
model:
  provider: alibaba_coding     # alias for alibaba-coding-plan
  model: qwen3-coder-plus
```

またはCLIから:

```bash
hermes chat --provider alibaba_coding --model qwen3-coder-plus
```

`alibaba_coding` は、`alibaba` エントリですでに使用しているのと同じ `DASHSCOPE_API_KEY` を使用します — 別個のキーは不要で、ルーティング先が異なるだけです。このプロバイダーが登録される前は、`config.yaml` に `provider: alibaba_coding` を設定したユーザーは黙ってOpenRouterルーティングに流れ落ちていました。

中国本土のエンドポイント（`alibaba-coding-plan-cn`、`https://coding.dashscope.aliyuncs.com/v1`）では `ALIBABA_CODING_PLAN_CN_API_KEY` を設定します。CNプロバイダーは引き続き `ALIBABA_CODING_PLAN_API_KEY` / `DASHSCOPE_API_KEY` にフォールバックしますが、共有キーだけが設定されている場合、`/model` ピッカーには国際版の行しか表示されません — CNの行を表示するには、CNのキー（または `config.yaml` の `provider: alibaba-coding-plan-cn`）を設定してください。同じことが `ALIBABA_TOKEN_PLAN_CN_API_KEY` を使う `alibaba-token-plan-cn` にも当てはまります。

### MiniMax（OAuth）

ブラウザOAuthログインによるMiniMax-M2.7です — APIキーは不要です。`hermes model` で **MiniMax (OAuth)** を選択し、ブラウザでサインインすると、Hermesがアクセストークンとリフレッシュトークンを永続化します。内部ではAnthropic Messages互換のエンドポイント（`/anthropic`）を使用します。

```bash
hermes model
# → pick "MiniMax (OAuth)"
# → browser opens; sign in with your MiniMax account (global or CN region)
# → confirm — credentials are saved to ~/.hermes/auth.json

hermes chat   # uses api.minimax.io/anthropic endpoint
```

または `config.yaml` を設定します:
```yaml
model:
  provider: "minimax-oauth"
  default: "MiniMax-M2.7"
```

サポートされるモデル: `MiniMax-M2.7`（メイン）と `MiniMax-M2.7-highspeed`（デフォルトの補助モデルとして接続）。OAuth経路は `MINIMAX_API_KEY` / `MINIMAX_BASE_URL` を無視します。

:::tip MiniMax OAuth対APIキー
`minimax-oauth` はOAuthログインを使うMiniMaxの消費者向けポータルを使用します — 課金のセットアップは不要です。`minimax` および `minimax-cn` プロバイダーは `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY` を使用します — プログラム的なアクセス向けです。完全な手順については [MiniMax OAuthガイド](../guides/minimax-oauth.md) を参照してください。
:::

### NVIDIA NIM

NVIDIA NIMプロバイダーを使うと、NVIDIAのホスト型サービスまたはローカルのNIMエンドポイントを通じて、Nemotronその他のオープンソースモデルにアクセスできます。ホスト型エンドポイントのモデル識別子とローカルデプロイの手順は、[build.nvidia.com/models](https://build.nvidia.com/models) で入手できます。

:::note Nemotron 3.5 Super VL
以下の例では、テキストと画像の入力、推論、ツール呼び出しをサポートするNemotron 3.5 Super VLを使います。
:::

#### クラウド（build.nvidia.com）

[build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) でアカウントと無料のAPIキーを作成し、`~/.hermes/.env` に追加します:

```bash
NVIDIA_API_KEY=nvapi-...
```

その後チャットを開始します:

```bash
hermes chat --provider nvidia --model nvidia/nemotron-3.5-super-vl-120b-a12b
```

:::info BILLING
Hermesは `build.nvidia.com` へのリクエストにNIMの課金元ヘッダーを自動的に含めるため、NVIDIAの課金ダッシュボードで消費が正しい課金元に計上されます。追加の設定は不要です。
:::

#### ローカルNIM

DGX Spark、DGX Station、ローカルGPUなどのローカルまたはオンプレミスのデプロイでは、`~/.hermes/.env` で `NVIDIA_BASE_URL` をNIMエンドポイントに設定します:

```bash
NVIDIA_API_KEY=local-no-auth
NVIDIA_BASE_URL=http://localhost:8000/v1
```

エンドポイントが認証を必要としない場合は、プレースホルダーとして `local-no-auth` を使ってください。そうでなければ、そのAPIキーを指定します。

その後、エンドポイントが提供するモデルを使ってチャットを開始します:

```bash
hermes chat --provider nvidia --model nvidia/nemotron-3.5-super-vl-120b-a12b
```

ローカルのNIMエンドポイントは、NVIDIAのホスト型サービスと同じOpenAI互換のchat completions APIを公開します。エンドポイントを切り替えるには、`NVIDIA_BASE_URL` を更新し、必要に応じてAPIキーとモデル名を調整してください。

:::info VISION
ローカルのNIMモデルがmodels.devのカタログになく、ビジョンをサポートしている場合は、ネイティブの画像サポートのために `model.supports_vision: true` を設定してください — [カスタムプロバイダーのモデルのネイティブビジョン](#named-custom-providers)を参照してください。
:::

#### デフォルトのモデルを設定する

どちらのデプロイでも、毎回の呼び出しで指定しなくて済むよう、プロバイダーとモデルの識別子を `config.yaml` に保存します:

```yaml
model:
  provider: "nvidia"
  default: "nvidia/nemotron-3.5-super-vl-120b-a12b"
```

:::info CONTACT
Nemotronのモデル自体の問題については、[NVIDIA Nemotron開発者フォーラム](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-nemotron/669)を参照してください。
:::

### GMI Cloud

[GMI Cloud](https://www.gmicloud.ai/) 経由でオープンモデルや推論モデルを利用します — OpenAI互換API、APIキー認証。

```bash
# GMI Cloud
hermes chat --provider gmi --model deepseek-ai/DeepSeek-V3.2
# Requires: GMI_API_KEY in ~/.hermes/.env
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "gmi"
  default: "deepseek-ai/DeepSeek-V3.2"
```

ベースURLは `GMI_BASE_URL` で上書きできます（デフォルト: `https://api.gmi-serving.com/v1`）。

### Actual Computer

[Actual Computer](https://actual.inc) を通じて、自分のハードウェアをプライベートな推論クラスタとして利用します。2つの提供モードがあり、どちらもChat Completionsを使うため、推論と最終的な内容が一緒に返されます:

- **ホスト型リレー** — `https://api.actual.inc`、エンドツーエンドで暗号化され、*あなたの*クラスタにルーティングします。[actual.inc/user/keys](https://actual.inc/user/keys) の `ac_` 推論キーで認証します。
- **ローカルデーモン** — デバイス上の `http://127.0.0.1:8080`、完全にオフライン。APIキーは不要です: HermesはループバックのベースURLを検出し、内部のプレースホルダーで自動的に認証します。

```bash
# Hosted relay (ACTUAL_API_KEY in ~/.hermes/.env)
hermes chat --provider actual --model <model-id-from-your-cluster>

# Local daemon (model.base_url in ~/.hermes/config.yaml, no key)
hermes chat --provider actual --model <installed-model-name>
```

プロバイダーの設定は `~/.hermes/config.yaml` に保存します。`.env` に置くのはホスト型のAPIキーだけです:
```yaml
model:
  provider: "actual"
  default: "<model-id>"
  base_url: "http://127.0.0.1:8080/v1" # Omit for the hosted relay.
```

注意:
- モデルIDはクラスタの `GET /v1/models` から取得します — `hermes model` または `curl -s https://api.actual.inc/v1/models -H "Authorization: Bearer $ACTUAL_API_KEY"` で確認してください。
- `model.base_url` のホストのみの値は正規化されます: `http://127.0.0.1:8080` は自動的に `http://127.0.0.1:8080/v1` になります。レガシーの `ACTUAL_BASE_URL` 環境変数は、YAMLでActualのURLが設定されていない場合のフォールバックです。
- Actualは、チャット、圧縮、タイトル生成、その他すべての補助タスクに `/v1/chat/completions` を使います。これは `api.actual.inc` を対象とするカスタムプロバイダー、モデルの切り替え、フォールバックにも当てはまります。メインモデル、カスタムプロバイダー、補助タスクの設定にあるレガシーのResponsesの設定は、自動的に上書きされます。
- 推論の労力はActualがサポートする範囲（`none/low/medium/high/max`）に丸められます — グローバルな `xhigh`/`ultra` の設定でリクエストが400になることはありません。
- 小さなローカルモデル: Hermesのデフォルトの完全なツールセットとシステムプロンプトは32kのコンテキストウィンドウを超えることがあり、llama.cpp系のサーバーで空のストリームのエラーが発生します。ツールセットを制限する（`-t file,web`）か、より大きなコンテキストでモデルを読み込んでください。オプションの `actual-setup` スキル（`hermes skills install official/devops/actual-setup`）は、セットアップとトラブルシューティングを詳しく扱います。
- エイリアス: `actual-computer`、`actualcomputer`、`aci`。

### StepFun

[StepFun](https://platform.stepfun.com) 経由でStepシリーズのモデルを利用します — OpenAI互換API、APIキー認証。

```bash
# StepFun
hermes chat --provider stepfun --model step-3.5-flash
# Requires: STEPFUN_API_KEY in ~/.hermes/.env
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "stepfun"
  default: "step-3.5-flash"
```

ベースURLは `STEPFUN_BASE_URL` で上書きできます（デフォルト: `https://api.stepfun.com/v1`）。

### Hugging Face Inference Providers

[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) は、統合されたOpenAI互換エンドポイント（`router.huggingface.co/v1`）を通じて20以上のオープンモデルにルーティングします。リクエストは利用可能な最速のバックエンド（Groq、Together、SambaNovaなど）に自動的にルーティングされ、自動フェイルオーバーを備えています。

```bash
# Use any available model
hermes chat --provider huggingface --model Qwen/Qwen3.5-397B-A17B
# Requires: HF_TOKEN in ~/.hermes/.env

# Short alias
hermes chat --provider hf --model deepseek-ai/DeepSeek-V3.2
```

または `config.yaml` で恒久的に設定します:
```yaml
model:
  provider: "huggingface"
  default: "Qwen/Qwen3.5-397B-A17B"
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
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter: API base URL, API key, Model name
```

**手動設定（`config.yaml`）:**
```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: your-key-or-leave-empty-for-local
  # key_env: MY_PROVIDER_API_KEY  # env var holding the key (alternative to api_key)
```

:::warning レガシー環境変数
`.env` 内の `LLM_MODEL` は**削除されました** — `config.yaml` がモデルとエンドポイントの設定における唯一の信頼できる情報源です。`OPENAI_BASE_URL` は引き続き尊重されますが、`openai-api` プロバイダーに**限られます**（直接のAPIキーアクセスでOpenAIのエンドポイントを上書きします）。その他のプロバイダーやカスタムエンドポイントでは、`hermes model` を使うか、`config.yaml` で `model.base_url` を直接設定してください。`.env` に古いエントリがある場合は、次回の `hermes setup` または設定の移行時に自動的にクリアされます。
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
/model custom:qwen-2.5          # Switch to a model on your custom endpoint
/model custom                    # Auto-detect the model from the endpoint
/model openrouter:claude-sonnet-4 # Switch back to a cloud provider
```

**名前付きカスタムプロバイダー**（後述）を設定している場合は、トリプル構文を使用します:

```
/model custom:local:qwen-2.5    # Use the "local" custom provider with model qwen-2.5
/model custom:work:llama3       # Use the "work" custom provider with llama3
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
# Install and run a model
ollama pull qwen2.5-coder:32b
ollama serve   # Starts on port 11434
```

次にHermesを設定します:

```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:11434/v1
# Skip API key (Ollama doesn't need one)
# Enter model name (e.g. qwen2.5-coder:32b)
```

または `config.yaml` を直接設定します:

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 64000   # See warning below
```

:::caution Ollamaはデフォルトで非常に小さいコンテキスト長になります
Ollamaはデフォルトでモデルのフルコンテキストウィンドウを使用しません。VRAMに応じて、デフォルトは次のようになります:

| 利用可能なVRAM | デフォルトのコンテキスト |
|----------------|----------------|
| 24 GB未満 | **4,096トークン** |
| 24〜48 GB | 32,768トークン |
| 48 GB以上 | 256,000トークン |

Hermes Agentは、ツールを使うエージェント用途に少なくとも **64,000トークン** のコンテキストを必要とします。システムプロンプト、ツールスキーマ、作業中の会話の状態が信頼できる複数ステップのワークフローに十分な余地を必要とするため、より小さいウィンドウは起動時に拒否されます。

**増やす方法**（いずれか1つを選択）:

```bash
# Option 1: Set server-wide via environment variable (recommended)
OLLAMA_CONTEXT_LENGTH=64000 ollama serve

# Option 2: For systemd-managed Ollama
sudo systemctl edit ollama.service
# Add: Environment="OLLAMA_CONTEXT_LENGTH=64000"
# Then: sudo systemctl daemon-reload && sudo systemctl restart ollama

# Option 3: Bake it into a custom model (persistent per-model)
echo -e "FROM qwen2.5-coder:32b\nPARAMETER num_ctx 64000" > Modelfile
ollama create qwen2.5-coder-64k -f Modelfile
```

**OpenAI互換API（`/v1/chat/completions`）からコンテキスト長を設定することはできません。** サーバー側またはModelfile経由で設定する必要があります。これはHermesのようなツールとOllamaを統合する際の混乱の最大の原因です。
:::

**コンテキストが正しく設定されているか確認:**

```bash
ollama ps
# Look at the CONTEXT column — it should show your configured value
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
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:8000/v1
# Skip API key (or enter one if you configured vLLM with --api-key)
# Enter model name: meta-llama/Llama-3.1-70B-Instruct
```

**コンテキスト長:** vLLMはデフォルトでモデルの `max_position_embeddings` を読み込みます。それがGPUメモリを超える場合はエラーになり、`--max-model-len` をより低く設定するよう求められます。`--max-model-len auto` を使うと、収まる最大値を自動的に見つけることもできます。`--gpu-memory-utilization 0.95`（デフォルト0.9）を設定すると、より多くのコンテキストをVRAMに詰め込めます。

**ツール呼び出しには明示的なフラグが必要です:**

| フラグ | 目的 |
|------|---------|
| `--enable-auto-tool-choice` | `tool_choice: "auto"`（Hermesのデフォルト）に必要 |
| `--tool-call-parser <name>` | モデルのツール呼び出し形式用のパーサー |

サポートされるパーサー: `hermes`（Qwen 2.5、Hermes 2/3）、`llama3_json`（Llama 3.x）、`mistral`、`deepseek_v3`、`deepseek_v31`、`xlam`、`pythonic`。これらのフラグがないと、ツール呼び出しは機能しません — モデルはツール呼び出しをテキストとして出力します。

**Qwenの推論パーサー:** OpenAI互換のサーバーが `reasoning`、`reasoning_content`、ストリーミングされる推論のデルタなどの構造化された推論のメタデータを返す場合、Hermesはそれを保持します。そのメタデータは推論/思考のトレースデータとして扱われ、アシスタントの目に見える回答の代わりにはなりません。vLLMで提供されるQwenの推論モデルでは、ユーザーに見える最終的な応答が引き続き `content` に現れることを確認してください。デプロイで `--reasoning-parser qwen3` によって `content` が空になる場合は、そのパーサーを無効にするか、`extra_body` を通じて `chat_template_kwargs.enable_thinking: false` のようなサーバーがサポートするリクエストオプションを渡してください。

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
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:30000/v1
# Enter model name: meta-llama/Llama-3.1-70B-Instruct
```

**コンテキスト長:** SGLangはデフォルトでモデルの設定から読み込みます。`--context-length` を使って上書きします。モデルが宣言する最大値を超える必要がある場合は、`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` を設定します。

**ツール呼び出し:** モデルファミリーに適したパーサーを `--tool-call-parser` で指定します: `qwen`（Qwen 2.5）、`llama3`、`llama4`、`deepseekv3`、`mistral`、`glm`。このフラグがないと、ツール呼び出しはプレーンテキストとして返ってきます。

:::caution SGLangはデフォルトで最大出力128トークンになります
レスポンスが途切れているように見える場合は、サーバーの生成のデフォルトを確認し、サーバー側で設定してください（例: SGLangの `--default-max-tokens`）。Hermesは出力トークンの上限の設定を公開していません。
:::

---

### llama.cpp / llama-server — CPUおよびMetal推論

[llama.cpp](https://github.com/ggml-org/llama.cpp) は、量子化されたモデルをCPU、Apple Silicon（Metal）、コンシューマー向けGPUで実行します。最適な用途: データセンターのGPUなしでモデルを実行、Macユーザー、エッジデプロイ。

```bash
# Build and start llama-server
cmake -B build && cmake --build build --config Release
./build/bin/llama-server \
  --jinja -fa \
  -c 64000 \
  -ngl 99 \
  -m models/qwen2.5-coder-32b-instruct-Q4_K_M.gguf \
  --port 8080 --host 0.0.0.0
```

**コンテキスト長（`-c`）:** 最近のビルドはデフォルトで `0` となり、GGUFメタデータからモデルのトレーニングコンテキストを読み込みます。128k以上のトレーニングコンテキストを持つモデルの場合、フルのKVキャッシュを割り当てようとしてOOM（メモリ不足）になることがあります。Hermesでは `-c` を少なくとも64,000トークンに明示的に設定してください。並列スロット（`-np`）を使用する場合、合計コンテキストはスロット間で分割されます — `-c 64000 -np 4` では、各スロットは16kしか得られず、これはアクティブなセッションあたりのHermesの最小値を下回ります。

次にHermesをそこに向けて設定します:

```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:8080/v1
# Skip API key (local servers don't need one)
# Enter model name — or leave blank to auto-detect if only one model is loaded
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
lms server start                        # Starts on port 1234
lms load qwen2.5-coder --context-length 64000
```

次にHermesを設定します:

```bash
hermes model
# Select "LM Studio"
# Press Enter to use http://localhost:1234/v1
# Pick one of the discovered models
# If LM Studio server auth is enabled, enter LM_API_KEY when prompted
```

Hermesは、すでに読み込まれているLM Studioのインスタンスのコンテキストを保持します。デフォルトの明示モードで読み込まれていないモデルの場合、Hermesで設定していない限り `context_length` を省略するため、LM Studioは自身のモデル設定を適用できます。その後Hermesは、読み込み後にLM Studioが報告するコンテキスト長だけを使います。

LM Studioでコンテキスト長を変更するには:

1. モデルピッカーの隣にある歯車アイコンをクリック
2. 「Context Length」を快適な体験のために少なくとも64000に設定
3. 変更を反映させるためにモデルをリロード
4. マシンが64000に収まらない場合は、より大きなコンテキスト長を持つより小さいモデルの使用を検討してください。

代替として、CLIを使用します: `lms load model-name --context-length 64000`

CLIを使ってモデルが収まるかを見積もることもできます: `lms load model-name --context-length 64000 --estimate-only`

モデルごとの永続的なデフォルトを設定するには: My Modelsタブ → モデルの歯車アイコン → コンテキストサイズを設定。
:::

LM StudioのJust-In-Timeロード / Auto-Evict機能を使い、通常のチャットリクエストからLM Studioにモデルの読み込みと退避を管理させたい場合は、Hermesの明示的な事前読み込みのステップをスキップします:

```bash
hermes config set model.lmstudio_load_mode jit
```

デフォルトの明示的な事前読み込みの挙動に戻すには:

```bash
hermes config set model.lmstudio_load_mode explicit
```

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
   curl http://localhost:11434/v1/models   # Ollama on Windows — works
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
# Get the Windows host IP (the default gateway of WSL2's virtual network)
ip route show | grep -i default | awk '{ print $3 }'
# Example output: 172.29.192.1
```

そのIPをHermesの設定で使用します:

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://172.29.192.1:11434/v1   # Windows host IP, not localhost
```

:::tip 動的なヘルパー
ホストIPはWSL2の再起動時に変わることがあります。シェルで動的に取得できます:
```bash
export WSL_HOST=$(ip route show | grep -i default | awk '{ print $3 }')
echo "Windows host at: $WSL_HOST"
curl http://$WSL_HOST:11434/v1/models   # Test Ollama
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
# Run in Admin PowerShell — replace PORT with your server's port
New-NetFirewallRule -DisplayName "Allow WSL2 to Model Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

一般的なポート: Ollama `11434`、vLLM `8000`、SGLang `30000`、llama-server `8080`、LM Studio `1234`。

#### クイック検証

WSL2内から、モデルサーバーに到達できることをテストします:

```bash
# Replace URL with your server's address and port
curl http://localhost:11434/v1/models          # Mirrored mode
curl http://172.29.192.1:11434/v1/models       # NAT mode (use your actual host IP)
```

モデルを一覧表示するJSONレスポンスが得られれば成功です。その同じURLをHermes設定の `base_url` として使用してください。

---

### ローカルモデルのトラブルシューティング

これらの問題は、Hermesと共に使用される際の**すべての**ローカル推論サーバーに影響します。

#### WSL2からWindowsホストのモデルサーバーへの「Connection refused」

WSL2内でHermesを実行し、モデルサーバーがWindowsホスト上にある場合、`http://localhost:<port>` はWSL2のデフォルトのNATネットワークモードでは機能しません。修正については上記の [WSL2ネットワーク](#wsl2-networking-windows-users) を参照してください。

#### ツール呼び出しが実行されずテキストとして表示される {#tool-calls-appear-as-text-instead-of-executing}

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
# Check what Hermes thinks the context is
# Look at startup line: "Context limit: X tokens"

# Check your server's actual context
# Ollama: ollama ps (CONTEXT column)
# llama.cpp: curl http://localhost:8080/props | jq '.default_generation_settings.n_ctx'
# vLLM: check --max-model-len in startup args
```

**修正:** エージェント用途では、コンテキストを少なくとも **64,000トークン** に設定してください。具体的なフラグについては上記の各サーバーのセクションを参照してください。

ローカルのエンドポイント（`127.0.0.1`、LAN、Dockerのサービス名）に対する起動時の拒否は、サーバーが提供しているウィンドウを示し、Ollamaだけでなく任意のOpenAI互換サーバー向けの修正方法を示します: サーバーのコンテキストを引き上げる（llama.cppの `-c 64000`、vLLMの `--max-model-len`、Ollamaの `OLLAMA_CONTEXT_LENGTH`/Modelfileの `num_ctx`）か、`config.yaml` の `model.ollama_num_ctx` をサーバーが実際に提供するウィンドウ（少なくとも64K）に設定します。`model.ollama_num_ctx` はすべてのローカルエンドポイントで尊重されます。その背後の自動検出だけがOllamaの `/api/show` を使います。

#### 起動時に「Context limit: 2048 tokens」と表示される

Hermesはサーバーの `/v1/models` エンドポイントからコンテキスト長を自動検出します。サーバーが低い値を報告する（または全く報告しない）場合、Hermesはモデルが宣言する制限を使用しますが、これは誤っていることがあります。

**修正:** `config.yaml` で明示的に設定します:

```yaml
model:
  default: your-model
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 64000
```

#### レスポンスが文の途中で途切れる

**考えられる原因:**
1. **サーバーの低い出力上限** — サーバーの生成のデフォルトを設定してください（例: SGLangの `--default-max-tokens`）。Hermesは出力トークンの上限の設定を公開していません。レスポンスの長さは、会話のコンテキストウィンドウ（`context_length`）とは別物です。
2. **コンテキストの枯渇** — モデルがコンテキストウィンドウを埋め尽くしました。`model.context_length` を増やすか、Hermesで [コンテキスト圧縮](../user-guide/configuration.md#context-compression) を有効にしてください。

---

### LiteLLM Proxy — マルチプロバイダーゲートウェイ

[LiteLLM](https://docs.litellm.ai/) は、100以上のLLMプロバイダーを単一のAPIの背後に統合するOpenAI互換プロキシです。最適な用途: 設定変更なしでのプロバイダー間の切り替え、ロードバランシング、フォールバックチェーン、予算管理。

```bash
# Install and start
pip install "litellm[proxy]"
litellm --model anthropic/claude-sonnet-4 --port 4000

# Or with a config file for multiple models:
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
# Install and start
npx @blockrun/clawrouter    # Starts on port 8402
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
| [Actual Computer](https://actual.inc) | `https://api.actual.inc/v1` | 自分のクラスタへのプライベートなリレー。ローカルデーモンは `http://127.0.0.1:8080/v1` |
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

:::note コンテキストウィンドウと出力の上限は別物
**`context_length`** は**コンテキストウィンドウ全体**です — 入力*と*出力トークンを合わせた予算（例: Claude Opus 4.6では200,000）。Hermesはこれを使って、いつ履歴を圧縮するか、APIリクエストを検証するかを判断します。

出力の上限が制御するのは単一の生成されたレスポンスであり、会話履歴ではありません。
Hermesは `model.max_tokens`、`HERMES_MAX_TOKENS`、プロバイダーの出力上限の
設定、`model_overrides.*.*.max_output_tokens` をもう読み込みません。これらのレガシーの設定は削除してください。
カスタムのOpenAI互換エンドポイントには、カタログのサイズに基づく自動的な出力上限は送られません。
サーバーのデフォルトが適用され、それはモデルの最大値より低いことがあります。
繰り返しのループに陥った返信は引き続き停止されます: ループが始まってから約130,000
文字以内（目に見えるテキストまたは推論のテキスト）で、Hermesはストリームを閉じ、
「Repetition Detected」の通知とともにターンを終えるため、上限のないエンドポイントがループする
モデルを動かし続けることはできません。

ネイティブのAnthropic Messages（ネイティブのAnthropic Bedrockの経路を含む）は
`max_tokens` を必須とするため、Hermesは内部の値を与えます。Bedrock Converseは別の
プロトコルです: その任意の `inferenceConfig.maxTokens` はデフォルトで省略され、これは
[AWSがモデルの最大値として文書化しています](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html)。
内部の上限付きのタスクやプロバイダー固有のプロトコルの要件は、実装の
詳細のままです。省略が普遍的にモデルの最大出力を選ぶわけではありません。

自動検出がウィンドウサイズを誤る場合に `context_length` を設定してください。

:::

Hermesは、あなたのモデルとプロバイダーに対する正しいコンテキストウィンドウを検出するために、複数のソースによる解決チェーンを使用します:

1. **設定の上書き** — config.yamlの `model.context_length`（最優先）。これは明示的な**固定**です: 常にプロバイダーのメタデータより優先されるため、Hermesはウィンドウが表示されるすべての場所（ウェルカムバナー、`/model`、`/usage`、ステータスバー）でそれに `(pinned)` のラベルを付け、固定がプロバイダーが公開していると分かっているウィンドウと食い違う場合は起動時に警告を1回ログに出力します。モデル、プロバイダー、ベースURLを切り替えると、固定は自動的に破棄されます。
2. **カスタムプロバイダーのモデルごと** — `providers.<name>.models.<id>.context_length`
3. **永続キャッシュ** — 以前に検出された値（再起動後も維持）
4. **エンドポイントの `/models`** — サーバーのAPIを照会（ローカル／カスタムエンドポイント）
5. **Anthropicの `/v1/models`** — Anthropic APIに `max_input_tokens` を照会（APIキーユーザーのみ）
6. **OpenRouter API** — OpenRouterからのライブのモデルメタデータ
7. **Nous Portal** — Nousのモデル IDをOpenRouterメタデータにサフィックスマッチング
8. **[models.dev](https://models.dev)** — 100以上のプロバイダーにわたる3800以上のモデルについて、プロバイダー固有のコンテキスト長を持つコミュニティ管理のレジストリ
9. **フォールバックのデフォルト** — 広範なモデルファミリーのパターン（デフォルト128K）

ほとんどのセットアップでは、これがそのまま機能します。このシステムはプロバイダーを認識します — 同じモデルでも、誰が提供するかによって異なるコンテキスト制限を持つことがあります（例: `claude-opus-4.6` はAnthropic直接では1Mですが、一部の再販業者やアグリゲーター経由ではより小さくなります）。

コンテキスト長を明示的に設定するには、モデル設定に `context_length` を追加します:

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072  # tokens
```

カスタムエンドポイントの場合、モデルごとにコンテキスト長を設定することもできます:

```yaml
providers:
  my-local-llm:
    api: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
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

### 名前付きカスタムプロバイダー {#named-custom-providers}

複数のカスタムエンドポイント（例: ローカルの開発サーバーとリモートのGPUサーバー）を扱う場合、`config.yaml` の `providers:` 辞書の下に、プロバイダー名をキーとして名前付きカスタムプロバイダーとして定義できます:

```yaml
providers:
  local:
    api: http://localhost:8080/v1
    # api_key omitted — Hermes uses "no-key-required" for keyless local servers
  work:
    api: https://gpu-server.internal.corp/v1
    key_env: CORP_API_KEY
    transport: chat_completions   # set explicitly by `hermes model` → Custom Endpoint wizard; auto-detection still happens as a fallback
  anthropic-proxy:
    api: https://proxy.example.com/anthropic
    key_env: ANTHROPIC_PROXY_KEY
    transport: anthropic_messages  # for Anthropic-compatible proxies
```

各エントリが受け付けるもの: `api`（エンドポイントのベースURL — `base_url`/`url` も別名として受け付けます）、`name`（任意の表示名。デフォルトは辞書のキー）、`key_env` またはインラインの `api_key` または `key_cmd`（下記参照）、`transport`（`chat_completions` / `anthropic_messages` / `codex_responses`）、`default_model`、`models`、`context_length`、`discover_models`、`extra_body`、`extra_headers`、`session_affinity_header`（セッションを認識するプロキシ向けに、会話idを運ぶヘッダーの名前。設定しない限りオフ）、`ssl_ca_cert` / `ssl_verify`、`catalog_provider`（下記参照）、そして削除せずにエントリを非表示にする `enabled: false` です。

組み込みのプロバイダーと同じキー（大文字小文字を問わず `openrouter:`、`Anthropic:`、`deepseek:`）を持ち、独自の `api`/`base_url` を設定するエントリは、その名前が意味するもの**そのもの**になります: `--provider openrouter`、`/model`、`hermes model`、ダッシュボードはそれを `custom:openrouter` として保存し、すべてのターンがあなたのエンドポイントに送られます。エントリがキー（`api_key`、`key_env`、`key_cmd`）を宣言している場合はそのキーが使われ、組み込みのプロバイダー自身のキーが送られることは決してありません。キーのないエントリが組み込みの環境変数やプールのキーを得るのは、そのアドレスが組み込みのプロバイダー自身のホスト（スキーム、ホスト、ポート）上にある間だけです。それ以外の場所（プロキシ、ミラー、KimiのCNエンドポイントのような別リージョンのホスト）ではキーを送りません。組み込みのキーは、あなたが指定しない限り、自身のホスト以外に送られることは決してないためです。組み込みのキーをあなたのエンドポイントで使うには、それを指定します: `key_env: DEEPSEEK_API_KEY`（組み込みの変数）、またはエントリが設定されている間に `hermes auth add deepseek` でエンドポイントのキーを追加すると、エントリのアドレスが記録されます。エンドポイントのない同名のブロック（たとえば `providers.openrouter: {request_timeout_seconds: 300}`）は、組み込みを調整するだけです。`model.provider` がそのようなエントリに乗っ取られる組み込みを指している場合、`hermes doctor` が警告します。

:::note 移行: 組み込みのキーと同名のエントリ
以前のリリースでは、組み込みの環境変数のキー（たとえば `DEEPSEEK_API_KEY`）が、どのアドレスにあるキーのない `providers.deepseek` エントリにも送られていました。現在、そのキーは組み込み自身のホストに留まります。そのようなエントリが `401` を返すようになった場合は、エントリに `key_env: DEEPSEEK_API_KEY`（あなたのプロバイダーの変数を使ってください）を追加してください。その場合、Hermesはエントリごとに対処法を示す警告を1回ログに出力します。アドレスが記録されるようになる前に組み込みのプールに追加されたプールのキー（`hermes auth add`、ダッシュボード）は、組み込み自身のアドレスに属します。エントリのホストでそれを使うには、エントリが設定されている間に `hermes auth add <name>` で追加し直してください。
:::

#### コマンドで発行される認証情報（`key_cmd`）

ビジョン、思考、ネイティブのローカルモデルの機能のプローブは、認証ヘッダーを組み立てる前に、
チャットが使うのと同じ呼び出し可能な認証情報を実体化します。これらは
チャットクライアントの呼び出し可能オブジェクトを置き換えることなく、コマンドのトークンキャッシュを再利用します。
コマンドが文字列のトークンを発行できない場合、これらのベストエフォートのプローブは、
オブジェクトの表現や優先度の低い設定済みの認証情報ではなく、ベアラーを送りません。
ネイティブのローカルモデルのプローブは、明示的な呼び出し可能オブジェクトが失敗した場合、継承したAuthorizationを削除し、
無関係な設定済みのヘッダーは保持します。チャットは通常の
エラー処理を維持します。

エンタープライズのゲートウェイは、静的なAPIキーではなく、短命なベアラートークン（SSO/OIDCのブローカー、クラウドのIAM、内部の認証プロキシ）を発行することが多いため、`.env` にコピーしたトークンはセッションの途中で古くなり、リクエストが401を返し始めます。`key_cmd` はトークンを*出力する*コマンドを指定します。Hermesはそれを実行し、有効期限の少し前まで結果をキャッシュするため、長いセッションも再起動なしで動作し続けます:

```yaml
providers:
  my-gateway:
    base_url: "https://gateway.internal.example.com/v1"
    api_mode: chat_completions
    key_cmd: "my-auth-cli print-token --profile prod"
```

トークンを出力する任意のヘルパーで動作します — `databricks auth token`、`gcloud auth print-access-token`、`az account get-access-token`、`vault read`、またはClaude Code形式の `apiKeyHelper` スクリプトなど。

コマンドはstdoutにトークン**だけ**を出力する必要があります: そのままか、`access_token` フィールドを持つJSONとして（`expires_in` が尊重され、絶対時刻の `expiry`/`expiresOn` のISOタイムスタンプも同様）。複数行の出力は推測せずに拒否されます。有効期限が示されない場合、トークンは上限付きのウィンドウで再発行されます。

優先順位: 明示的な `--api-key` フラグが引き続き優先され、それ以外では、同じエントリ上の静的な `api_key`/`key_env` より `key_cmd` が優先されます。発行された認証情報は、メインのエージェントのターンにも補助タスク（タイトル生成、圧縮、ビジョン、埋め込み）にも同様に適用されます。

モデルの検出も、`providers:` とレガシーの
`custom_providers:` の両方のエントリについて、`hermes model` のセットアップを含め `key_cmd` を尊重します。ヘルパーが実行されるのは、
認証付きのライブカタログのプローブが必要な場合だけです: 無効化された検出や、温かいカタログの
キャッシュの読み取りではトークンは発行されません。カタログはコマンドのアイデンティティにスコープされるため、
ベアラーをローテーションしてもカタログは無効になりません。プローブのヘルパーは、推論クライアントのトークンキャッシュではなく、
独自の短命なトークンソースを使います。発行されたベアラーが
`config.yaml` に保存されることは決してありません。ヘルパーが失敗した場合、検出はヘルパーの出力を公開せずに、
設定されたモデルにフォールバックします。

`secrets.command` と混同しないでください。それは**起動時に1回**ヘルパーを実行して、プロセス全体の環境変数を設定するものです。多数のシークレットを返すvault/keychainのヘルパーにはそちらを使い、1つのプロバイダーの認証情報をセッションの*途中で*再発行する必要がある場合は `key_cmd` を使ってください。

:::note レガシーの形式
古い設定では、代わりにトップレベルの `custom_providers:` リストを使っていました。それは引き続き動作し — Hermesは両方を読みます — `hermes update` はそれを `providers:` 辞書（config v12）に自動移行します。辞書形式ではフィールド名が少し異なります: レガシーの `model` は `default_model` に、レガシーの `api_mode` は `transport` になります。
:::

**`codex_responses` プロキシでのコンテキストウィンドウ。** `transport: codex_responses` のカスタムエントリ（たとえばローカルのCodexプロキシ）は、Codex OAuthのモデル（`gpt-6-astra`、`gpt-5.6-sol`/`-terra`/`-luna`、`gpt-5.5` など）のコンテキストウィンドウを、1.05Mの直接APIのカタログではなく、Codex OAuthの表（ほとんどのスラッグで272K）から解決します。そのため、圧縮はCodexバックエンドの制限とその272Kの課金階層の前に発動します。判断はホスト名ではなくトランスポートに従います。`HERMES_CODEX_BASE_URL` や `model.base_url` の背後の `openai-codex` も同様です。モデルごとの `models.<id>.context_length`、エントリレベルの `context_length`、`model.context_length` は引き続き優先されます。オプトインの `-900k` のピッカーのバリアントは、検証済みの900Kを維持します。

**カスタムエンドポイントでの推論の労力。** 設定された `reasoning_effort`（`/reasoning max`、`agent.reasoning_effort`）は、`chat_completions` と `codex_responses` の両方のトランスポートで、`max` まで変更されずにカスタムエンドポイントに届きます。Hermes内部の `ultra` だけが `max` に丸められます。2つの例外は、エントリではなくホストに従います: `api.openai.com` を指すカスタムエントリはOpenAIのモデルごとの段階を維持し（そこでは `max` はgpt-5.6専用のレベルです）、プロファイルがモデルごとの語彙を公開しているプロバイダー（Ramp Router）を指すエントリは、そのカタログに丸められます。そのレベルを拒否するエンドポイントは、Hermesが黙って格下げする代わりにHTTP 400で応答します。労力がまったく設定されていない場合、`chat_completions` のリクエストは、エンドポイント自身のデフォルト（kimi-k3では `max`、`medium` の約3倍の推論トークン）に任せるのではなく、`reasoning_effort: medium` を持ちます — Nous PortalとOpenRouterの経路が適用するのと同じデフォルトです。カタログや `model_overrides` で `supports_reasoning: false` とマークされたモデルや、`thinking` の機能を持たないOllamaのモデルでは、このフィールドはオフのままです。

一部のOpenAI互換エンドポイントは、プロバイダー固有のリクエストボディのフィールドを必要とします。対応するカスタムプロバイダーに `extra_body` マップを追加すると、Hermesはそのエンドポイントへの各chat-completionsリクエストにそれをマージします:

```yaml
providers:
  gemma-local:
    api: http://localhost:8080/v1
    default_model: google/gemma-4-31b-it
    extra_body:
      enable_thinking: true
      reasoning_effort: high
```

サーバーが文書化している形を使ってください。たとえば、vLLMのGemmaのデプロイや一部のNVIDIA NIMのエンドポイントは、`enable_thinking` をトップレベルの `extra_body` フィールドとしてではなく、`chat_template_kwargs` の下に置くことを想定しています:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: true
```

vLLMで提供されるQwenの推論モデルでは、推論パーサーが生成されたすべてのテキストを推論のフィールドに分けてアシスタントの `content` を空にしてしまう場合に、同じ形で思考を無効にできます:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

設定された `extra_body` はどこでもプロバイダーに従います: エージェントの構築時にマージされ、**すべてのゲートウェイのターンで維持され**（`/fast` が `service_tier`/`speed` の上書きを重ねるターンも含む — それらは `extra_body` を置き換えるのではなく、その上にマージされます）、**`/model` の切り替え時に再導出されます** — 名前付きカスタムプロバイダーに切り替えるとその `extra_body` が適用され、切り替えて離れるとクリアされるため、別のプロバイダーに漏れることはありません。

`hermes model` → Custom Endpoint のウィザードは、APIモードを明示的に尋ね、回答を `config.yaml` に（プロバイダーのエントリの `transport` として）永続化するようになりました。フィールドが空欄の場合は、URLに基づく自動検出（例: `/anthropic` のパス → `anthropic_messages`）が引き続きフォールバックとして行われます。

**カスタムプロバイダーのモデルのネイティブビジョン。** カスタムエンドポイントがmodels.devにないビジョン対応のモデルを提供している場合は、`model.supports_vision: true` を設定すると、Hermesは添付された画像を `vision_analyze` で前処理する代わりに、ネイティブに（`image_url` パーツとして）ルーティングします。つまみは1つだけです — `agent.image_input_mode: native` も設定する必要はありません。

```yaml
model:
  provider: custom
  base_url: http://localhost:8080/v1
  default: qwen3.6-35b-a3b
  supports_vision: true   # send images natively; otherwise vision_analyze pre-describes them
```

同じキーは名前付きプロバイダーごとのモデル（`providers.<name>.models.<id>.supports_vision`）でも尊重され、標準のYAMLのブール値（`true/false/yes/no/on/off/1/0`）を受け付けます。

カタログが知らないモデルに対してメタデータ（たとえば `context_window`）だけを修正する `model_overrides` のエントリは、ビジョンと推論の機能、および出力トークンの上限を**不明**のままにします — `vision_analyze`、`video_analyze`、推論の労力のピッカーは引き続き利用できます。上書き内の明示的な `supports_vision: false` / `supports_reasoning: false` だけが、モデルをテキスト専用または非推論としてマークします。

**カタログ化されたベンダーのメタデータの継承（`catalog_provider`）。** 名前付きカスタムプロバイダー（ゲートウェイ、プロキシ、再販業者）が、Hermesがすでに組み込みのプロバイダーのもとで知っているモデルを提供している場合は、エントリをそのベンダーに向けると、そのモデルはカタログ化されたコンテキストウィンドウ、出力の上限、ビジョンと推論のフラグを継承します — `model_overrides` は不要です:

```yaml
providers:
  my-gateway:
    api: https://gateway.example.com/v1
    key_env: GATEWAY_API_KEY
    catalog_provider: deepseek   # metadata lookups use DeepSeek's catalog entries
```

`catalog_provider` はHermesのプロバイダーid（`deepseek`、`anthropic`、`openai` など）またはmodels.devのidを受け付けます。影響するのはメタデータの参照だけです — リクエストは引き続きあなたの認証情報で `api` のURLに送られます — また、同じモデルに対する明示的な `model_overrides` のエントリは引き続き優先されます。

トリプル構文でセッションの途中で切り替えます:

```
/model custom:local:qwen-2.5       # Use the "local" endpoint with qwen-2.5
/model custom:work:llama3-70b      # Use the "work" endpoint with llama3-70b
/model custom:anthropic-proxy:claude-sonnet-4  # Use the proxy
```

対話的な `hermes model` メニューからも、名前付きカスタムプロバイダーを選択できます。

---

### クックブック: Together AI、Groq、Perplexity

[その他の互換プロバイダー](#other-compatible-providers) に挙げたクラウドプロバイダーはすべてOpenAIのREST方言を話すため、`providers:` 辞書の下で同じように接続できます。実際に動作する3つのレシピを以下に示します。それぞれ `~/.hermes/config.yaml` に組み込み、対応するAPIキーは `~/.hermes/.env` に入れます。

#### Together AI

オープンウェイトモデル（Llama、MiniMax、Gemma、DeepSeek、Qwen）を、ファーストパーティAPIより大幅に安い価格でホストしています。マルチモデルのフリート向けの良いデフォルトです。

```yaml
# ~/.hermes/config.yaml
providers:
  together:
    api: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
    # transport: chat_completions  # default — no need to set

model:
  default: MiniMaxAI/MiniMax-M2.7   # or any model from together.ai/models
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
providers:
  groq:
    api: https://api.groq.com/openai/v1
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
providers:
  perplexity:
    api: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: sonar
  provider: custom:perplexity
```

```bash
# ~/.hermes/.env
PERPLEXITY_API_KEY=your-perplexity-key
```

PerplexityのAgent API（`api_mode: codex_responses` を伴う `api: https://api.perplexity.ai/v1`）は、関数名 `web_search`、`search_files`、`fetch_url`、`people_search`、`finance_search` を自身の組み込みツール用に予約しています。Hermesは、メインのエージェントループと補助呼び出し（タイトル生成、圧縮、MoAの集約）の両方で、同名のクライアントツールをワイヤー上では `hermes_<name>` にリネームし、ディスパッチの前に元に戻します — OpenCodeの `/v1/responses` エンドポイントが受けるのと同じ扱いです。

#### 1つの設定に複数のプロバイダー

3つのレシピは組み合わせられます — すべてを一緒に使い、`/model custom:<name>:<model>` でターンごとに切り替えます:

```yaml
providers:
  together:
    api: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
  groq:
    api: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY
  perplexity:
    api: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: MiniMaxAI/MiniMax-M2.7
  provider: custom:together      # boot to Together; switch freely after
```

:::tip トラブルシューティング
- `hermes doctor` は、#15083のCLIバリデータ修正以降、これらの名前のいずれに対しても `Unknown provider` 警告を出さないはずです。
- プロバイダーの `/v1/models` エンドポイントに到達できない場合（Perplexityがよくある例）、`hermes model` はモデルをハードリジェクトせず、警告付きで永続化します — #15136を参照してください。
- 名前付きプロバイダーを完全にスキップし、`CUSTOM_BASE_URL` 環境変数で素の `provider: custom` を使用するには、#15103を参照してください。
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
   # In .env, set: USE_DB_AUTHENTICATION=false, HOST=0.0.0.0, PORT=3002
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
  sort: "throughput"          # "price" (default), "throughput", or "latency"
  # only: ["anthropic"]      # Only use these providers
  # ignore: ["deepinfra"]    # Skip these providers
  # order: ["anthropic", "google"]  # Try providers in this order
  # require_parameters: true  # Only use providers that support all request params
  # data_collection: "deny"   # Exclude providers that may store/train on data
  # models:                   # Per-model pins (same keys; unset keys fall through)
  #   "openai/gpt-6-astra": {only: ["openai"]}
  #   "anthropic/claude-fable-5.1": {only: ["anthropic"]}
```

**ショートカット:** 任意のモデル名に `:nitro` を付加するとスループットでソートされ（例: `anthropic/claude-sonnet-4:nitro`）、`:floor` を付加すると価格でソートされます。モデルごとの詳細: [プロバイダールーティング](../user-guide/features/provider-routing.md#per-model-overrides-models)。

## OpenRouter Pareto Code Router

OpenRouterは、`openrouter/pareto-code` で実験的なコーディングモデルルーターを提供しています。これは、コーディング品質の基準を満たす最も安価なモデル（[Artificial Analysis](https://artificialanalysis.ai/) によるランク付け）にリクエストを自動ルーティングします。このモデルを選択し、`~/.hermes/config.yaml` で `min_coding_score` のつまみを調整します:

```yaml
model:
  provider: openrouter
  model: openrouter/pareto-code

openrouter:
  min_coding_score: 0.65   # 0.0–1.0; higher = stronger (more expensive) coders. Default 0.65.
```

注意:

- `min_coding_score` は、`model.model` が `openrouter/pareto-code` の場合に**のみ**送信されます。その他のモデルではこの値は無効（no-op）です。
- 空文字列に設定する（または行を削除する）と、OpenRouterが利用可能な最強のコーダーを選択します — pluginsブロックを省略した場合の文書化された挙動です。
- 選択は特定の日における各スコアに対して決定的ですが、実際に選ばれるモデルはPareto最前線が動く（新しいモデル、ベンチマークの更新）につれて変わることがあります。
- 完全なルーター挙動については、OpenRouterの [Pareto Routerドキュメント](https://openrouter.ai/docs/guides/routing/routers/pareto-router) を参照してください。
- メインのエージェントの代わりに特定の**補助タスク**（圧縮、ビジョンなど）にParetoコードルーターを使用するには、そのタスクの下に `extra_body.plugins` を設定します — [補助モデル → 補助タスク向けのOpenRouterルーティング＆Pareto Code](../user-guide/configuration.md#openrouter-routing--pareto-code-for-auxiliary-tasks) を参照してください。

## フォールバックプロバイダー {#fallback-providers}

プライマリモデルが失敗したとき（レート制限、サーバーエラー、認証失敗）にHermesが順番に試すバックアッププロバイダーのチェーンを設定します。正規の形式はトップレベルの `fallback_providers:` リストです:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
  - provider: anthropic
    model: claude-sonnet-4
    # base_url: http://localhost:8000/v1    # optional, for custom endpoints
    # api_mode: chat_completions           # optional override (`transport:` is an accepted alias)
```

`providers.<name>` ブロックを指定するエントリ（`provider: my-relay` または `provider: custom:my-relay`）は、エントリが何も設定しない場合にそのブロックの `transport` / `api_mode` を継承するため、Responses専用やAnthropic-Messagesのリレーは、フォールバック時にも宣言されたワイヤーを維持します。上書きするにはエントリに `api_mode` を設定してください。

レガシーの単一ペア `fallback_model:` 辞書も後方互換のために引き続き受け付けられます:

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

起動すると、フォールバックは会話を失うことなくセッションの途中でモデルとプロバイダーを入れ替えます。チェーンはエントリごとに試行され、起動はセッションごとに1回限りです。

サポートされるプロバイダー: `openrouter`、`nous`、`novita`、`openai-codex`、`anthropic`、`gemini`、`qwen-oauth`、`huggingface`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`xai-oauth`、`ollama-cloud`、`bedrock`、`ai-gateway`、`azure-foundry`、`opencode-zen`、`opencode-go`、`commandcode`、`commandcode-anthropic`、`kilocode`、`xiaomi`、`arcee`、`gmi`、`actual`、`stepfun`、`lmstudio`、`alibaba`、`alibaba-coding-plan`、`tencent-tokenhub`、`tencent-tokenplan`、`nebius-token-factory`、`router`、`custom`。

:::tip
フォールバックは `config.yaml` を通じて — または `hermes fallback` で対話的に — 排他的に設定されます。いつ起動するか、チェーンがどのように進むか、補助タスクや委譲とどう相互作用するかの完全な詳細については、[フォールバックプロバイダー](../user-guide/features/fallback-providers.md) を参照してください。
:::

---

## 関連項目

- [設定](../user-guide/configuration.md) — 一般的な設定（ディレクトリ構造、設定の優先順位、ターミナルバックエンド、メモリ、圧縮など）
- [環境変数](../reference/environment-variables.md) — すべての環境変数の完全なリファレンス
