---
title: フォールバックプロバイダー
description: プライマリモデルが利用できないときに、バックアップのLLMプロバイダーへ自動的にフェイルオーバーするよう設定します。
sidebar_label: フォールバックプロバイダー
sidebar_position: 8
---

# フォールバックプロバイダー

Hermes Agentには、プロバイダーに問題が発生してもセッションを動かし続ける、3つの回復力（レジリエンス）のレイヤーがあります:

1. **[クレデンシャルプール](./credential-pools.md)** — *同じ* プロバイダーの複数のAPIキーをローテーションする（最初に試される）
2. **プライマリモデルのフォールバック** — メインモデルが失敗したとき、*別の* provider:modelに自動的に切り替える
3. **補助タスクのフォールバック** — ビジョンや圧縮などのサイドタスクに対する独立したプロバイダー解決

クレデンシャルプールは同一プロバイダー内のローテーション（例: 複数のOpenRouterキー）を扱います。このページでは、プロバイダーをまたぐフォールバックを扱います。どちらも任意であり、独立して動作します。

## プライマリモデルのフォールバック

メインのLLMプロバイダーがエラー — レート制限、サーバー過負荷、認証失敗、接続切断 — に遭遇したとき、Hermesは会話を失うことなく、セッションの途中でバックアップのprovider:modelのペアに自動的に切り替えられます。

### 設定

最も簡単な方法は対話型マネージャーです:

```bash
hermes fallback
```

`hermes fallback` は `hermes model` のプロバイダーピッカーを再利用します — 同じプロバイダーリスト、同じ認証情報プロンプト、同じ検証です。チェーンを管理するにはサブコマンド `add`、`list`（エイリアス `ls`）、`remove`（エイリアス `rm`）、`clear` を使い、検証するには `check` を使います（[チェーンの確認](#checking-the-chain) を参照）。変更は `config.yaml` のトップレベルの `fallback_providers:` リストの下に永続化されます。

YAMLを直接編集したい場合は、`~/.hermes/config.yaml` にトップレベルの `fallback_providers` リストを追加します:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

各エントリには `provider` と `model` の両方が必要です。どちらかが欠けているエントリは無視されます。

レート制限の応答がリセット時刻を示している場合、プライマリはちょうどその時刻まで休止させられます（何も示さないプロバイダーには、60秒 → 4時間の指数的バックオフが適用されます）。任意で、プライマリがすぐに再開する場合は切り替えをスキップできます:

```yaml
fallback:
  min_switch_reset_seconds: 120   # 0 (default) = always switch
```

| キー | デフォルト | 効果 |
|-----|---------|--------|
| `fallback.min_switch_reset_seconds` | `0`（オフ） | レート制限されたプライマリが宣言したリセットがこの秒数より早い場合、フォールバックへの入れ替えは行われず、代わりにリトライのバックオフがその期間を待ちます。 |

Geminiのフォールバックエントリでは、`gemini`、`google`、`google-gemini`、
`google-ai-studio` を指定できます。Googleのネイティブ APIエンドポイントでは、いずれもネイティブのGemini
クライアントを使用し、その `generationConfig.thinkingConfig` の変換も含まれます。カスタムの
OpenAI互換のベースURLでは、代わりに引き続き互換クライアントが使用されます。

:::note `fallback_model` vs `fallback_providers`
`fallback_providers`（複数形のリスト）が現在の設定形式で、順番に試される複数のフォールバックをサポートします。`fallback_model`（単数形）はレガシーな単一フォールバックのキーです — Hermesは後方互換性のために引き続き尊重しますが、`hermes fallback` は現在の `fallback_providers` キーに書き込み、書き込み時にレガシーな設定を移行します。両方が設定されている場合は、`fallback_providers` が優先されます。
:::

### チェーンの確認 {#checking-the-chain}

`hermes fallback check` は、エージェントセッションを開始することなく、フォールバックの切り替えが各エントリに到達した場合にそれが機能するかどうかを報告します:

```bash
hermes fallback check            # static: resolve each entry and its credentials, no inference
hermes fallback check --live     # also send one minimal request per entry, in parallel
hermes fallback check --live --json
```

各エントリは、フォールバックの切り替えが解決するのとまったく同じように（その `provider`、`model`、`base_url`、`api_key`/`key_env`）解決され、チェーンを順にたどることはないため、あるエントリが別のエントリを隠すことはありません。`--live` は、静的に準備完了となった各エントリに対して、そのプロバイダー自身のAPI（OpenAI互換、Anthropic Messages、Codex Responses、Bedrock、ACP）で短い補完を1回送信します。リトライはされず（SDKのリトライはオフです。ストリーミングを拒否するAnthropicエンドポイントには、アダプターが常に行う非ストリーミングでの再送信が1回だけ行われます）、別のプロバイダーが応答することもありません。空または不正な形式の応答、`failed`/`cancelled` ステータスの応答、またはchoiceでエラーを報告する応答（OpenRouterの `finish_reason: "error"`）は失敗とみなされます（トークン上限による `incomplete` は応答済みとみなされます）。HTTP 200のボディで中継されたプロバイダーのエラーは、それが記述するリクエストエラーと同様に報告されます。プロバイダーが異なるモデルを報告した場合、エントリは `model_mismatch` として失敗します（要求したモデルの日付付きスナップショット、例えば `gpt-4o` に対する `gpt-4o-2024-08-06` は一致とみなされます）。報告されるモデルが何の証明にもならない場合、モデルのチェックはスキップされます: Anthropic MessagesとCodex Responsesのアダプターは応答したモデルを公開しません。Azureはデプロイメント名ではなくベースモデルを報告します。ルーターのメタモデル（`openrouter/auto`、`@preset/...`、`<model>@preset/...`）は、選択したモデルの名前で応答します。BedrockのARNはエコーバックされません。ローカルやプライベートネットワークのサーバー（llama.cpp、LM Studio、Ollama、LAN上のマシン）はプレースホルダーを報告することがよくあります。ライブリクエストは、設定されている場合はプロバイダーの `request_timeout_seconds` を使用し（切り替え時と同様）、そうでなければ30秒を使用します。ストリーミングされるリクエストでは、タイムアウトは応答全体ではなく各読み取りに適用されます。Bedrock（AWS SDKの認証情報チェーン）や外部プロセスのプロバイダーに対する静的なパスは、ルートが解決されることを意味するだけです。証明するには `--live` を使ってください。静的モードは推論リクエストを送信しませんが、セッションと同じ方法で認証情報を解決するため、期限切れ間近のOAuthトークンが更新され保存されることがあります。`provider: auto`（呼び出しごとにバックエンドを選択）や `provider: moa`（ファンアウトのプリセット）のエントリは1つの固定ルートではないため、ライブリクエストが送信されることはなく、静的な判定のままになります。ライブリクエストは、フォールバックの切り替えが構築するクライアントに送られます。NousとCodexのエントリでは、そのクライアントは切り替え時と同様に保存されたログイン情報を使用します。

いずれかのエントリが準備完了でない場合、コマンドはステータス1で終了するため、cronジョブやCIでゲートとして使えます。`--json` は `profile`、`live`、`ready`、`checked_at` と、エントリごとに1つの結果（`provider`、`model`、`base_url`、`api_mode`、`auth_source`、`configured`、`auth_ready`、`inference_ready`、`latency_ms`、`model_returned`、`error_class`、`error`、`ready`、`status`）を出力します。`status` は `ready`、`ready (static)`、`config`、`auth`、`model_mismatch`、`live_response`（使用可能な補完のない応答）、または失敗したリクエストの場合は `live_<reason>`（`live_auth`、`live_timeout`、`live_rate_limit`、`live_model_not_found`、...）です。認証情報が出力されることはなく、プロバイダーのエラーテキストは秘匿化されます。別のプロファイルのチェーンを確認するには `hermes -p <profile> fallback check` を使います。

### サポート対象プロバイダー

| プロバイダー | 値 | 要件 |
|----------|-------|-------------|
| AI Gateway | `ai-gateway` | `AI_GATEWAY_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes setup --portal`（新規）または `hermes auth add nous`（OAuth） |
| OpenAI Codex | `openai-codex` | `hermes model` → **ChatGPT or Codex Subscription**（ChatGPT OAuth） |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` またはClaude Codeの認証情報 |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax（中国） | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY`（任意: `NVIDIA_BASE_URL`） |
| GMI Cloud | `gmi` | `GMI_API_KEY`（任意: `GMI_BASE_URL`） |
| Upstage Solar | `upstage`（エイリアス `solar`） | `UPSTAGE_API_KEY`（任意: `UPSTAGE_BASE_URL`） |
| StepFun | `stepfun` | `STEPFUN_API_KEY`（任意: `STEPFUN_BASE_URL`） |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY`（エイリアス: `GEMINI_API_KEY`） |
| xAI（Grok） | `xai`（エイリアス `grok`） | `XAI_API_KEY`（任意: `XAI_BASE_URL`） |
| xAI Grok OAuth（SuperGrok） | `xai-oauth`（エイリアス `grok-oauth`） | `hermes model` → xAI Grok OAuth（ブラウザログイン。SuperGrokサブスクリプション） |
| AWS Bedrock | `bedrock` | 標準のboto3認証（`AWS_REGION` + `AWS_PROFILE` または `AWS_ACCESS_KEY_ID`） |
| Qwen Portal（OAuth） | `qwen-oauth` | `hermes model`（Qwen Portal OAuth。任意: `HERMES_QWEN_BASE_URL`） |
| MiniMax（OAuth） | `minimax-oauth` | `hermes model`（MiniMaxポータルOAuth） |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| CommandCode | `commandcode`（エイリアス `commandcode-chat`。Claudeは `commandcode-anthropic` 経由） | `COMMANDCODE_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Ramp Router | `router` | `RAMP_ROUTER_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Nebius Token Factory | `nebius-token-factory` | `NEBIUS_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY`（`DASHSCOPE_API_KEY` にフォールバック） |
| Kimi / Moonshot（中国） | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Tencent TokenPlan | `tencent-tokenplan` | `TOKENPLAN_API_KEY` |
| Microsoft Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio（ローカル） | `lmstudio` | `LM_API_KEY`（ローカルの場合はなしでも可） + `LM_BASE_URL` |
| ローカルllama.cpp（マネージド） | `llamacpp`（エイリアス `llama.cpp`、`llama-cpp`） | Hermesが管理するローカルモデルサーバー（Desktopの **Settings → Providers → Local models**）。`base_url` は不要です。実行されていない場合、そのエントリはスキップされます — Hermesがローカルモデル名を別のプロバイダーに送信することはありません |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| カスタムエンドポイント | `custom` | `base_url` + `key_env`（下記参照） |
| Mixture of Agentsプリセット | `moa`（`model` = プリセット名） | アグリゲーターが認証情報を持つ、設定済みのMoAプリセット — フォールバックはアグリゲーター単体ではなく、プリセット全体（リファレンス + アグリゲーター）を実行します |

### カスタムエンドポイントのフォールバック

カスタムのOpenAI互換エンドポイントの場合は、`base_url` と、任意で `key_env` を追加します:

```yaml
fallback_providers:
  - provider: custom
    model: my-local-model
    base_url: http://localhost:8000/v1
    key_env: MY_LOCAL_KEY            # env var name containing the API key
```

### フォールバックが発動するとき

フォールバックは、プライマリモデルが次のような失敗をしたときに自動的に発動します:

- **レート制限**（HTTP 429） — リトライの試行を使い切った後
- **サーバーエラー**（HTTP 500、502、503） — リトライの試行を使い切った後
- **認証失敗**（HTTP 401、403） — 即座に（リトライしても無意味なため）
- **見つからない**（HTTP 404） — 即座に
- **不正な応答** — APIが不正または空の応答を繰り返し返したとき。アシスタントのテキストがルーターの `Connect timeout, please try again later.` だけで、補完トークンがゼロのHTTP 200のボディも不正とみなされます（ストリーミングか否かを問わず、メインループ、反復上限時のサマリー、補助呼び出しのいずれでも）。そのため、回答として表示されるのではなくリトライされます。ストリーミングされた拒否（モデルが拒否チャネルで説明とともに断るもの）は、空の応答ではなく終端の `content_filter` 結果であるため、リトライされずにそのまま表示されます。ネイティブのAnthropicのワイヤーでは `stop_reason: refusal` が空のボディで届きます。Hermesは応答の `stop_details` から理由（カテゴリと、存在する場合は説明）を、拒否メッセージとログ行（`native_stop_reason=… stop_details=…`）で報告します。

発動すると、Hermesは:

1. フォールバックプロバイダーの認証情報を解決する（`key_cmd` を使う名前付きカスタムプロバイダーを含む）
2. 新しいAPIクライアントを構築する。タイムアウトやリクエストクライアントの再構築をまたいで動的な認証情報ソースを保持する
3. モデル、プロバイダー、クライアントをその場で入れ替える
4. フォールバックモデルの推論エフォートを再解決する（その `agent.reasoning_overrides` エントリ、なければグローバルの `agent.reasoning_effort`）
5. リトライカウンターをリセットし、会話を続行する

切り替えはシームレスです — 会話履歴、ツール呼び出し、コンテキストは保持されます。エージェントは、別のモデルを使うだけで、中断したまさにその場所から続行します。

最初のリクエストの前にプライマリプロバイダーの認証が失敗したためにCLIが **起動時** にフォールバックする場合も、同じ再解決が行われます: フォールバックモデルには、プライマリのものではなく、それ自身に設定されたエフォートが送られます。明示的な `hermes chat --reasoning <level>` は、その起動時の切り替えをまたいで保持されます — それはその実行に対するあなたの意図だからです。

:::warning フォールバックはプロンプトキャッシュをリセットする
プロンプトキャッシュは、リクエストを処理するモデル（およびほとんどのプロバイダーではアカウント）に紐づいています。フォールバックが発動すると、新しいprovider:modelには会話のキャッシュ済みプレフィックスがないため、次のリクエストは約75〜90%割引のキャッシュ料金ではなく、通常の入力トークン料金で履歴全体を読み直します。ターンが終わってプライマリが復元されたときも同様です — プライマリに戻った最初のリクエストも全体の読み直しになります（プライマリのキャッシュTTLが切れていない場合を除く）。これは避けられません — 障害を乗り切るためのコストです — が、プロバイダー間を行き来する長いセッションが、同じ場所にとどまるセッションより目に見えて高くつくことがあるのはこのためです。
:::

:::info Per-Turn, Not Per-Session
フォールバックは **ターン単位** です: 新しいユーザーメッセージごとに、プライマリモデルが復元された状態で始まります。プライマリがターンの途中で失敗した場合、フォールバックはそのターンに限り発動します。次のメッセージでは、Hermesは再びプライマリを試します。単一のターン内では、フォールバックは最大1回しか発動しません — フォールバックも失敗した場合は、通常のエラー処理（リトライ、その後エラーメッセージ）が引き継ぎます。これにより、ターン内でのフェイルオーバーループの連鎖を防ぎつつ、毎ターン、プライマリモデルに新たなチャンスを与えます。

ターンごとのリトライは **リセットを考慮** します: プライマリの認証情報が、まだ経過していないレート制限のリセット時刻を報告している場合（Claude Pro/Maxの5時間ブロックやCodexの週次制限のようなサブスクリプションの期間は、これを数時間や数日として報告します）、Hermesは失敗が確定しているリトライをスキップし、リセットが過ぎるまでフォールバックにとどまります — これにより、ターンごとの無意味な2回のプロバイダー切り替え（と2回のプロンプトキャッシュの無効化）を避けます。期限切れになるとプライマリは後のリトライの対象になりますが、リトライがスケジュールされたり回復が保証されたりするわけではありません。リセット時刻のない一時的な429には、指数的なクールダウンが使われます。

切り替えがそのクールダウンを設定すると、フォールバックの通知にはおおよその残り時間が含まれます。例: `Primary retry eligible in ~60 s; recovery is not guaranteed.` レート制限以外による切り替えや、すでにアクティブなプロバイダー間フォールバックからの切り替えでは、新しいプライマリのクールダウンは通知されません。
:::

### 例

**AnthropicネイティブのフォールバックとしてのOpenRouter:**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

**OpenRouterのフォールバックとしてのNous Portal:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_providers:
  - provider: nous
    model: nous-hermes-3
```

**クラウドのフォールバックとしてのローカルモデル:**
```yaml
fallback_providers:
  - provider: custom
    model: llama-3.1-70b
    base_url: http://localhost:8000/v1
    key_env: LOCAL_API_KEY
```

**フォールバックとしてのCodex OAuth:**
```yaml
fallback_providers:
  - provider: openai-codex
    model: gpt-5.4
```

### フォールバックが機能する場所

| コンテキスト | フォールバック対応 |
|---------|-------------------|
| CLIセッション（対話型と `hermes -z` のワンショット） | ✔（プライマリの認証情報/クォータが失敗した起動時、セッションの途中、およびチャットを開いている間に追加・編集したチェーンは次のターンから適用） |
| メッセージングゲートウェイ（Telegram、Discordなど） | ✔ |
| DesktopアプリとTUIのチャット | ✔（チャットを開いている間に追加・編集したチェーンは次のターンから適用） |
| サブエージェントへの委譲 | ✔（設定されている場合は `delegation.fallback_providers`。それ以外では、ピン留めされていない子のみが親のチェーンを継承します。`[]` で無効化） |
| cronジョブ | ✔（ピン留めされていないジョブは設定済みのチェーンを継承します。独自のprovider/model/base_urlを持つジョブがそれにフォールバックすることはありません） |
| `provider: auto` の補助タスク | ✔（タスクごとのフォールバックを試し、次に組み込みの補助ディスカバリーの前にメインのフォールバックチェーンを試します） |

:::tip
プライマリのフォールバックチェーンに対する環境変数はありません — 設定は `config.yaml` または `hermes fallback` を通じてのみ行います。これは意図的なものです: フォールバックの設定は意図的な選択であり、古いシェルのexportが上書きしてよいものではありません。
:::

---

## 補助タスクのフォールバック

Hermesは、サイドタスクに別の軽量モデルを使います。各タスクには独自のプロバイダー解決チェーンがあり、これが組み込みのフォールバックシステムとして機能します。

### 独立したプロバイダー解決を持つタスク

| タスク | 内容 | 設定キー |
|------|-------------|-----------|
| Vision | 画像分析、ブラウザのスクリーンショット | `auxiliary.vision` |
| Compression | コンテキスト圧縮の要約 | `auxiliary.compression` |
| MCP | MCPヘルパー操作 | `auxiliary.mcp` |
| Approval | スマートなコマンド承認の分類 | `auxiliary.approval` |
| Title Generation | セッションタイトルの要約 | `auxiliary.title_generation` |
| Review | `/review` のレビュアーサブエージェント（単一のLLM呼び出しではなく完全なエージェント） | `auxiliary.review` |
| Triage Specifier | `hermes kanban specify` / ダッシュボードの✨ボタン — 一行のトリアージタスクを実際の仕様に肉付けする | `auxiliary.triage_specifier` |

### 自動検出チェーン

タスクのプロバイダーが `"auto"`（デフォルト）に設定されている場合、Hermesはまずその補助タスクに対してメインプロバイダー + メインモデルを試します。そのルートが利用できない場合、または後で容量系のエラーで失敗した場合、Hermesは設定されたフォールバックポリシーに従い、その後停止します:

```text
Main provider + main model → auxiliary.<task>.fallback_chain →
fallback_providers / fallback_model → skip the task (warn)
```

課金やクォータの失敗によって補助ヘルスのクールダウンの間隔離されるのは、失敗したカスタムエンドポイントだけで、`custom` として登録されたすべてのルートではありません。ベースURLが異なる正常なローカルエンドポイントは、引き続きフォールバックやその後の自動ルーティングの対象になります。同じカスタムエンドポイントのエイリアスは、そのヘルス状態を共有します。組み込みプロバイダーは、共有アカウントのヘルスチェックを維持します。

タスク固有のチェーンが最も精密で、存在する場合はそれが優先されます。トップレベルの `fallback_providers` チェーンはメインエージェントが使うのと同じポリシーであるため、無料のみや同一プロバイダーのフォールバックルールは、`auto` の補助タスクにも適用されます。

**組み込みのテキストディスカバリーチェーン（圧縮、タイトル生成など）:**

```text
OpenRouter → Nous Portal → Custom endpoint → Codex OAuth →
API-key providers (z.ai, Kimi, MiniMax, Xiaomi MiMo, Hugging Face, Anthropic) → give up
```

**組み込みのビジョンディスカバリーチェーン:**

```text
Main provider (if vision-capable) → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → Custom endpoint → give up
```

これらの組み込みチェーンは、**メインプロバイダーが選択されていない場合にのみ** 実行されます（`model.provider: auto` または未設定）。メインプロバイダーを選択した後は、`fallback_chain` / `fallback_providers` がない状態でメインのルートが利用できないと、たまたまログインしている別のプロバイダーを推測する代わりに、警告とともに補助タスクがスキップされます — 期限切れのxAIやCodexのセッションが、知らないうちにNous PortalやOpenRouterの残高に課金することがあってはなりません。フォールバックが必要なら宣言してください。

### 補助プロバイダーの設定

各タスクは `config.yaml` で個別に設定できます:

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # e.g. "openai/gpt-4o"
    base_url: ""                  # direct endpoint (takes precedence over provider)
    api_key: ""                   # API key for base_url

  compression:
    provider: "auto"
    model: ""
    fallback_chain:              # optional, task-specific fallback policy
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free

  mcp:
    provider: "auto"
    model: ""
```

上記のすべてのタスクは、同じ **provider / model / base_url** のパターンに従います。各タスクは独自の `fallback_chain` を宣言することもできます。省略した場合、`provider: auto` はトップレベルの `fallback_providers` チェーンを使います（組み込みのディスカバリーチェーンが適用されるのは、メインプロバイダーが選択されていない場合だけです）。

コンテキスト圧縮は `auxiliary.compression` の下で設定します:

```yaml
auxiliary:
  compression:
    provider: main                                    # Same provider options as other auxiliary tasks
    model: google/gemini-3-flash-preview
    base_url: null                                    # Custom OpenAI-compatible endpoint
```

そしてプライマリのフォールバックチェーンは次を使います:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
    # base_url: http://localhost:8000/v1             # Optional custom endpoint
```

3つすべて — 補助、圧縮、フォールバック — は同じように動作します: `provider` で誰がリクエストを処理するかを選び、`model` でどのモデルかを選び、`base_url` でカスタムエンドポイントを指定します（providerを上書きします）。

### 補助タスクのプロバイダーオプション

これらのオプションは `auxiliary:`、`compression:`、`fallback_providers:` のエントリにのみ適用されます — `"main"` は、トップレベルの `model.provider` の有効な値では **ありません**。カスタムエンドポイントには、`model:` セクションで `provider: custom` を使ってください（[AIプロバイダー](../../integrations/providers.md) を参照）。

| プロバイダー | 説明 | 要件 |
|----------|-------------|-------------|
| `"auto"` | 1つが動作するまで順番にプロバイダーを試す（デフォルト） | 少なくとも1つのプロバイダーが設定されていること |
| `"openrouter"` | OpenRouterを強制 | `OPENROUTER_API_KEY` |
| `"nous"` | Nous Portalを強制 | `hermes auth` |
| `"codex"` | Codex OAuthを強制 | `hermes model` → ChatGPT or Codex Subscription |
| `"main"` | メインエージェントが使うプロバイダーをそのまま使う（補助タスクのみ） | アクティブなメインプロバイダーが設定されていること |
| `"anthropic"` | Anthropicネイティブを強制 | `ANTHROPIC_API_KEY` またはClaude Codeの認証情報 |
| `"llamacpp"` | マネージドなローカルllama.cppサーバーを強制 | ローカルモデルサーバーが実行中であること。オフの場合、ローカルモデル名が別のプロバイダーに送信されることはありません |

### 直接エンドポイントのオーバーライド

任意の補助タスクで、`base_url` を設定するとプロバイダー解決が完全にバイパスされ、リクエストがそのエンドポイントへ直接送られます:

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` は `provider` より優先されます。Hermesは認証に設定された `api_key` を使います。設定されていない場合、`OPENAI_API_KEY` にフォールバックするのは、OpenAIのホスト（`*.openai.com`、`*.openai.azure.com`）か、それと組になっている `OPENAI_BASE_URL` と完全に一致する場合だけです。カスタムエンドポイントに対して `OPENROUTER_API_KEY` を **再利用しません**。

---

## 補助タスクの容量エラー時のフォールバック

明示的な補助プロバイダー（例: `auxiliary.vision.provider: glm`）を設定した場合、Hermesはそれをあなたの優先する選択として扱います — しかし、そのプロバイダーが **容量エラー**（HTTP 402 Payment Required、HTTP 429の日次クォータ枯渇、接続失敗）のためにリクエストをまったく処理できない場合、Hermesは黙って失敗する代わりに、階層化されたチェーンを通じてフォールバックします:

1. **プライマリの補助プロバイダー** — 設定したもの（常に最初に試される）
2. **`auxiliary.<task>.fallback_chain`** — 書いている場合は、タスクごとのオーバーライドリスト
3. **メインエージェントのプロバイダー + モデル** — 最後の手段のセーフティネット（チェーンを書いていなくても常に試される）
4. **警告 + 再送出** — すべてのレイヤーが失敗した場合、HermesはWARNINGレベルで `Auxiliary <task>: ... all fallbacks exhausted` をログに記録し、元のエラーを再送出します

一時的なHTTP 429のレート制限（`Retry-After: ...`）は、容量の問題ではなくリクエストの制約として扱われます — 明示的なプロバイダーの選択を尊重し、フォールバックのはしごを **発動しません**。明示的プロバイダーのゲートをバイパスするのは、日次/月次のクォータ枯渇、支払いエラー、接続失敗だけです。

**明示的なプロバイダーでの認証エラー（HTTP 401）** では、ステップ2だけをたどります: `auxiliary.<task>.fallback_chain` を書いている場合、そのエントリが順番に試されます（リクエスト途中で失敗したチェーンのエントリは次のエントリに引き継ぎます）。メインエージェントのモデルや自動検出チェーンが参照されることはありません。そのタスクをそれらにオプトインしていないためです。チェーンがない場合、タスクは従来どおり認証エラーで失敗します。認証は認証情報全体に及ぶため、同じプロバイダーラベルのチェーンエントリはスキップされます — 予備は別のプロバイダーに向けてください（別の `providers:` エントリでも構いません）。

`provider: auto` のユーザー（明示的な補助プロバイダーなし）の場合、ステップ2〜3の代わりに既存の自動検出チェーンが実行されます。その最初のステップはすでにメインエージェントのモデルであるため、`auto` のユーザーは設定なしで同じ結果を得ます。

### 任意: タスクごとのフォールバックチェーン

「メインエージェントのモデルが最初」とは異なるフォールバックの順序にしたい場合は、`fallback_chain` を明示的に設定します。各エントリには少なくとも `provider` が必要で、`model`、`base_url`、`api_key` は任意です。

```yaml
auxiliary:
  vision:
    provider: glm
    model: glm-4v-flash
    fallback_chain:
      - provider: openrouter
        model: google/gemini-3-flash-preview
      - provider: nous
        model: anthropic/claude-sonnet-4

  compression:
    provider: openrouter
    fallback_chain:
      - provider: openai
        model: gpt-4o-mini
        timeout: 240            # optional — this candidate's own deadline (seconds)
```

フォールバックを得るために `fallback_chain` を設定する必要は **ありません** — メインエージェントのセーフティネットはいずれにせよ実行されます。デフォルトとは異なる順序を特に望む場合にのみ使ってください。

各 `fallback_chain` エントリは、独自の `timeout`（秒）を宣言することもできます。指定しない場合、フォールバック候補はタスクレベルのタイムアウトを継承します — これはプライマリプロバイダー向けに調整されている可能性があります。エントリごとの `timeout` を宣言すると、遅いが信頼できるフォールバック（例: 大きなコンテキストの要約器）が、プライマリの時間制限で打ち切られる代わりに、実際に必要な時間を得られます。

### フォールバックを発動するプロバイダーのクォータエラー

Hermesは、次のものを（一時的なレート制限ではなく）402のクレジット枯渇と同等の容量エラーとして認識します:

- Bedrock / LiteLLM: `Too many tokens per day`、`daily limit`、`tokens per day`
- Vertex AI / GCP: `quota exceeded`、`resource exhausted`、`RESOURCE_EXHAUSTED`
- 汎用: `daily quota`、`quota_exceeded`

プロバイダーが日次クォータの枯渇に対して別の文言を返し、Hermesがフォールバックを発動しない場合、それはバグです — 正確なエラー文字列を添えてissueを開いてください。

---

## コンテキスト圧縮のフォールバック

コンテキスト圧縮は、どのモデルとプロバイダーが要約を処理するかを制御するために `auxiliary.compression` の設定ブロックを使います:

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info Legacy migration
`compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` を持つ古い設定は、初回読み込み時（設定バージョン17）に自動的に `auxiliary.compression.*` へ移行されます。
:::

圧縮に利用できるプロバイダーがない場合、Hermesはセッションを失敗させる代わりに、要約を生成せずに中間の会話ターンを破棄します。

---

## 委譲プロバイダーのオーバーライド

`delegate_task` によって生成されるサブエージェントは、親エージェントのプライマリのフォールバックチェーンを継承します。それでも、コスト最適化のためにサブエージェントを別のプライマリのprovider:modelのペアへルーティングできます:

```yaml
delegation:
  provider: "openrouter"                      # override provider for all subagents
  model: "google/gemini-3-flash-preview"      # override model
  # base_url: "http://localhost:1234/v1"      # or use a direct endpoint
  # api_key: "local-key"
```

完全な設定の詳細は [サブエージェントへの委譲](./delegation.md) を参照してください。

---

## cronジョブのプロバイダー

ピン留めされていないcronジョブは、実行前にプライマリの認証情報の解決に失敗した場合と、実行途中でプロバイダーがエラーになった場合の両方で、設定済みの `fallback_providers` チェーン（またはレガシーな `fallback_model`）を継承します。独自のプロバイダー、モデル、エンドポイントにピン留めされたジョブは継承 **しません**: そのルートが失敗すると、実行は失敗します（同一プロバイダーの [クレデンシャルプール](../configuration.md#credential-pool-strategies) のローテーションは引き続き適用されます）。これは、ピン留めされた [委譲](./delegation.md) の子の動作と一致します。cronジョブをピン留めするには、ジョブ自体に `provider` と `model` のオーバーライドを設定します:

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

ジョブでフォールバックを維持するには、ピン留めせずに、代わりに `cron.model` / `cron.model_provider` でモデルを選んでください。詳細は [スケジュールタスク（Cron）](./cron.md#provider-recovery) を参照してください。

---

## まとめ

| 機能 | フォールバックの仕組み | 設定の場所 |
|---------|-------------------|----------------|
| メインエージェントモデル | config.yamlの `fallback_providers` — エラー時のターン単位フェイルオーバー（毎ターン、プライマリが復元される） | `fallback_providers:`（トップレベルのリスト） |
| 補助タスク（すべて） — autoのユーザー | 容量エラー時に完全な自動検出チェーン（メインエージェントのモデルが最初、次にプロバイダーチェーン） | `auxiliary.<task>.provider: auto` |
| 補助タスク（すべて） — 明示的なプロバイダー | 容量エラー時は `fallback_chain`（設定されている場合） → メインエージェントのモデル → 警告 + 送出。認証エラー（401）は `fallback_chain` のみをたどる | `auxiliary.<task>.fallback_chain` |
| Vision | 階層化（上記参照） + 内部のOpenRouterリトライ | `auxiliary.vision` |
| コンテキスト圧縮 | 階層化（上記参照）。すべてのレイヤーが利用できなければ要約なしに降格 | `auxiliary.compression` |
| MCPヘルパー | 階層化（上記参照） | `auxiliary.mcp` |
| 承認の分類 | 階層化（上記参照） | `auxiliary.approval` |
| タイトル生成 | 階層化（上記参照） | `auxiliary.title_generation` |
| トリアージ仕様化 | 階層化（上記参照） | `auxiliary.triage_specifier` |
| 委譲 | 宣言されている場合は `delegation.fallback_providers` を使用。それ以外では、ピン留めされていない子のみが親のチェーンを継承 | `delegation.provider` / `delegation.model` / `delegation.fallback_providers` |
| cronジョブ | ピン留めされていないジョブは設定済みの `fallback_providers` チェーンを継承。独自の `provider` / `model` / `base_url` を持つジョブがそれにフォールバックすることはない | ジョブごとの `provider` / `model`、または `cron.model` / `cron.model_provider` |
