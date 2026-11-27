---
title: 画像生成
description: FAL.ai 経由で画像を生成 — FLUX 2、GPT Image（1.5 & 2）、Nano Banana Pro、Ideogram、Recraft V4 Pro、Krea 2 などを含む 11 モデルに対応。`hermes tools` で選択可能。
sidebar_label: 画像生成
sidebar_position: 6
---

# 画像生成

Hermes Agent は FAL.ai 経由でテキストプロンプトから画像を生成します。11 のモデルがそのまま使え、それぞれ速度、品質、コストのトレードオフが異なります。アクティブなモデルは `hermes tools` で設定でき、`config.yaml` に保存されます。

## 対応モデル {#supported-models}

| モデル | 速度 | 強み | 価格 |
|---|---|---|---|
| `fal-ai/flux-2/klein/9b` *(既定)* | `<1s` | 高速、くっきりしたテキスト | $0.006/MP |
| `fal-ai/flux-2-pro` | 約 6s | スタジオ品質のフォトリアリズム | $0.03/MP |
| `fal-ai/z-image/turbo` | 約 2s | 英中バイリンガル、6B パラメータ | $0.005/MP |
| `fal-ai/nano-banana-pro` | 約 8s | Gemini 3 Pro、推論の深さ、テキスト描画 | $0.15/画像（1K） |
| `fal-ai/gpt-image-1.5` | 約 15s | プロンプト追従 | $0.034/画像 |
| `fal-ai/gpt-image-2` | 約 20s | SOTA のテキスト描画 + CJK、世界を認識したフォトリアリズム | $0.04〜0.06/画像 |
| `fal-ai/ideogram/v3` | 約 5s | 最高のタイポグラフィ | $0.03〜0.09/画像 |
| `fal-ai/recraft/v4/pro/text-to-image` | 約 8s | デザイン、ブランドシステム、本番対応 | $0.25/画像 |
| `fal-ai/qwen-image` | 約 12s | LLM ベース、複雑なテキスト | $0.02/MP |
| `fal-ai/krea/v2/medium/text-to-image` | 約 15〜25s | イラスト、アニメ、絵画、表現力豊かな/芸術的なスタイル | $0.030〜0.035/画像 |
| `fal-ai/krea/v2/large/text-to-image` | 約 25〜60s | フォトリアリズム、粗い質感のある表現（モーションブラー、粒子、フィルム） | $0.060〜0.065/画像 |

価格は執筆時点での FAL の料金です。最新の数値は [fal.ai](https://fal.ai/) で確認してください。

## セットアップ

:::tip Nous サブスクライバー
有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションをお持ちの場合、FAL API キーなしで **[Tool Gateway](tool-gateway.md)** 経由で画像生成を使えます。モデルの選択はどちらの経路でも引き継がれます。新規インストールでは `hermes setup --portal` を実行すると、ログインしてすべてのゲートウェイツールを一度に有効化できます。既存のインストールでは、`hermes tools` で画像生成のバックエンドとして **Nous Subscription** を選べます。

**Nous Subscription** の行は唯一のマネージドな行です。そのモデルピッカーは、サブスクリプションが運用するすべてのゲートウェイにまたがります — 上記の FAL カタログ、ネイティブの **Krea 3**（`krea-3`）と **Krea 2**（`krea-2-medium`、`krea-2-large`、`krea-2-medium-turbo`）、そして Nous Portal の画像モデルです。各モデルは一度だけ表示され、選んだモデルによってどのゲートウェイがリクエストを処理するかが決まります。無料のツールプールアカウントでは FAL のモデルのみが表示され、Krea と Portal のモデルは有料サブスクリプション向けです。Krea 3 を選択した場合、`image_url` は編集する画像になり（参照画像は合計で最大 4 枚。編集では画像自体の構図が保たれます）、`image_generate` は `thinking`（`fast` または `high`）と `resolution`（`1K` または `2K`）を提供します。Krea 2 のモデルを選択した場合は、代わりに Krea の `creativity` 設定と、`intensity`、`complexity`、`movement` のスライダー（-100〜100）を提供します。

マネージドゲートウェイが特定のモデルに対して `HTTP 4xx` を返す場合、そのモデルはまだ Portal 側でプロキシされていません。エージェントがその旨と対処手順（直接アクセスのために自分の `FAL_KEY` を使って `hermes tools` で FAL.ai に切り替える、または別のモデルを選ぶ）を伝えます。
:::

### FAL API キーを取得する

1. [fal.ai](https://fal.ai/) でサインアップ
2. ダッシュボードから API キーを生成

### 設定してモデルを選ぶ

tools コマンドを実行します:

```bash
hermes tools
```

**🎨 Image Generation** に移動し、バックエンド（Nous Subscription または FAL.ai）を選びます。すると、対応するすべてのモデルが列揃えのテーブルで表示されます。矢印キーで移動し、Enter で選択します:

```
  Model                          Speed    Strengths                    Price
  fal-ai/flux-2/klein/9b         <1s      Fast, crisp text             $0.006/MP   ← currently in use
  fal-ai/flux-2-pro              ~6s      Studio photorealism          $0.03/MP
  fal-ai/z-image/turbo           ~2s      Bilingual EN/CN, 6B          $0.005/MP
  ...
```

選択内容は `config.yaml` に保存されます:

```yaml
image_gen:
  provider: fal                 # `nous` if you picked Nous Subscription
  model: fal-ai/flux-2/klein/9b
  max_parallel_requests: 4      # concurrent images in one tool-call batch
```

`image_gen.provider` は唯一の選択キーです。`nous` はマネージドな Tool Gateway 経由でルーティングし、ベンダー名（`fal`、`openai`、`xai`、`krea` など）は自分のキーで直接接続します。ランタイムは常にこの保存された選択に従います — `provider: nous` の間は `.env` 内の `FAL_KEY` は無視され、`FAL_KEY` なしの `provider: fal` は、黙って別経路にルーティングするのではなく `image_gen is configured to use fal (set via hermes tools), but FAL_KEY is not set. Run 'hermes tools' to change it.` というエラーになります。プロバイダーの変更は、キーの追加/削除ではなく `hermes tools` で行ってください。（古い `use_gateway` ブール値はレガシーです — `true` の場合は引き続き `nous` として読み込まれますが、もう書き込まれることはありません。）

`max_parallel_requests` の既定値は `4` です。Hermes はこの値を最低 1、かつグローバルなツールワーカー数の上限以下に制限するため、画像プロバイダーは上限のある並列リクエストを受け取り、画像バッチがエージェントの並行実行の上限を回避することはありません。

### OpenRouter: Image API の全カタログ

`image_gen.provider: openrouter` の場合、モデルピッカーには OpenRouter のライブ画像カタログ全体が表示されます — 専用の [Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation) モデル（Seedream、FLUX.2、Recraft、Qwen Image、MAI、Krea、Riverflow、Grok Imagine など — 40 以上の ID）に、chat-completions の画像モデルを統合したものです。カタログは `GET /images/models` と `GET /models` からライブで取得されるため、新しいモデルは OpenRouter が提供を始めた時点でピッカーに表示されます。Hermes のアップデートは不要です。生成時には、各モデルがそれを提供するサーフェス（専用の `POST /images/generations` か chat-completions か）へ自動的にルーティングされます。Nous Portal は chat-completions プロトコルのみをプロキシするため、そのピッカーには chat で提供されるモデルが表示されます。

Image API モデル向けのリクエストごとのオプション設定は、スコープされた設定セクション（または `OPENROUTER_IMAGE_API_*` 環境変数）に記述します:

```yaml
image_gen:
  provider: openrouter
  model: bytedance-seed/seedream-4.5
  openrouter:
    resolution: 2K        # model-dependent: 1K / 2K / 4K
    quality: high         # gpt-image models
    output_format: png
```

### GPT-Image の品質

`fal-ai/gpt-image-1.5` と `fal-ai/gpt-image-2` のリクエスト品質は `medium` に固定されています（1024×1024 で約 $0.034〜$0.06/画像）。`low` / `high` のティアをユーザー向けオプションとして公開していないのは、Nous Portal の課金を全ユーザーで予測可能に保つためです。ティア間のコスト差は 3〜22 倍に及びます。より安価な選択肢が欲しい場合は Klein 9B または Z-Image Turbo を、より高品質が欲しい場合は Nano Banana Pro または Recraft V4 Pro を選んでください。

### Meta Model API: Muse Image

`image_gen.provider: meta-ai` の場合、画像は [Meta Model API](https://api.meta.ai)（`https://api.meta.ai/v1`）経由で生成されます。これは Muse Spark のチャットモデルを提供しているのと同じ OpenAI 互換エンドポイントです。同梱の `meta-ai` チャットプロバイダーに対応する画像生成用のプロバイダーです。

| モデル | 速度 | 強み | 価格 |
|---|---|---|---|
| `muse-image-1.0` *(既定)* | 約 10s | Meta Model API による画像生成 | $0.01/画像 |

```yaml
image_gen:
  provider: meta-ai
  model: muse-image-1.0
```

認証には Meta のチャットプロバイダーと同じ環境変数を再利用します — `MODEL_API_KEY`（Meta のドキュメントでの名前）で、`META_API_KEY` / `META_MODEL_API_KEY` もエイリアスとして受け付けます。プロキシや別のホストを指すには `META_BASE_URL` を設定します。現時点ではテキストから画像のみで、レスポンスは `$HERMES_HOME/cache/images/` に保存されます。

## FAL: GPT Image 2.5

`hermes tools` → Image Generation → FAL.ai で **GPT Image 2.5 Flare** または **GPT Image 2.5 Sunburst** を選択します。モデル ID は次のとおりです:

- `openai/gpt-image-2.5/flare/text-to-image`
- `openai/gpt-image-2.5/sunburst/text-to-image`

例:

```bash
hermes config set image_gen.provider fal
hermes config set image_gen.model openai/gpt-image-2.5/flare/text-to-image
```

`image_url` または参照画像を指定すると、対応する `openai/gpt-image-2.5/flare/edit` または `openai/gpt-image-2.5/sunburst/edit` エンドポイントが自動的に選択されます。どちらもソース画像を最大 16 枚受け付けます。Hermes は品質を `medium` に固定しており、FAL の高コストな `high` の既定値ではなく、既存の FAL GPT Image のポリシーに合わせています。横向きと縦向きは最小ピクセル数を満たすために 4:3 のプリセットを使い、正方形は `square_hd` を使います。アップスケールは要求されない限りオフのままです。

FAL は固定の画像単価ではなくトークン単位で課金します: テキスト入力 $5/M、キャッシュされたテキスト入力 $1.25/M、テキスト出力 $10/M、画像入力 $8/M、キャッシュされた画像入力 $2/M、画像出力 $30/M で、リクエストごとに $0.0001 単位で切り上げられます。[Flare](https://fal.ai/models/openai/gpt-image-2.5/flare/text-to-image) と [Sunburst](https://fal.ai/models/openai/gpt-image-2.5/sunburst/text-to-image) のページを参照してください。FAL への直接接続には残高のある `FAL_KEY` が必要です。マネージドゲートウェイでの利用可否はそのゲートウェイのエンドポイント許可リストに依存し、FAL で利用可能であることからは保証されません。既存のプロバイダーとモデルの既定値は変わりません。

## OpenAI API: GPT Image 2.5

**OpenAI** プロバイダーは、`OPENAI_API_KEY` を使って GPT Image 2.5 Flare（高速な日常的な作成）と Sunburst（精密な生成と編集）に対応しています。`hermes tools` → Image Generation → OpenAI で選択するか、次のように設定します:

```bash
hermes config set image_gen.provider openai
hermes config set image_gen.openai.model gpt-image-2.5-flare
```

`gpt-image-2.5-flare` と `gpt-image-2.5-sunburst` は自動品質を使います。固定の品質を選ぶには `-low`、`-medium`、`-high`、`-xhigh`、`-max` を付加します（例: `gpt-image-2.5-sunburst-high`）。どちらも最大 16 枚の参照画像による生成と編集に対応しています。既存の GPT Image 2 の選択と `gpt-image-2-medium` の既定値は変わりません。

これは有料の API 利用であり、ChatGPT/Codex のサブスクリプションとは別です。どちらのモデルも、テキスト入力トークン 100 万あたり $5、画像入力トークン 100 万あたり $8、画像出力トークン 100 万あたり $30 です（キャッシュされた入力の料金はそれぞれ $1.25 と $2）。画像あたりのコストは使用量によって変わります。GPT Image 2 の計算ツールは 2.5 のトークン消費量を見積もりません。公式の [Flare](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare) と [Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst) のドキュメントを参照してください。

**OpenAI (Codex auth)** プロバイダーは 2.5 を提供しません。Codex バックエンドは任意の `model` の値（存在しない ID を含む）を受け付け、サーバー側で管理される独自のエンジンで生成するため、Flare や Sunburst のティアを「選択」しても効果のないラベルにすぎません。2.5 を使うには、OpenAI API への直接接続のプロバイダーか FAL を選んでください。

### カスタムの OpenAI 互換画像エンドポイント

**OpenAI** プロバイダーは、チャットプロバイダーとは独立に、任意の OpenAI 互換の `/v1/images/generations` エンドポイント（ローカルゲートウェイ、タスク単位のプロキシ、サードパーティの API ゲートウェイ）を指すことができ、キーを任意の変数から取得できます:

```yaml
image_gen:
  provider: openai
  openai:
    model: gpt-image-2-medium
    base_url: http://localhost:18081/v1   # → OPENAI_BASE_URL → api.openai.com
    key_env: IMAGE_GATEWAY_TOKEN          # → OPENAI_API_KEY
```

`config.yaml` に保存されるのは変数の *名前* だけで、シークレットは `.env` またはプロセスの環境に残ります。利用可否のチェックと生成は同じ解決方法を使うため、`key_env` を設定すれば十分です — `OPENAI_API_KEY` は不要です。リクエストは Hermes 独自の HTTP クライアントを経由します。これは `HTTP(S)_PROXY`/`NO_PROXY` を尊重しますが、macOS のシステムプロキシ（その例外リストは Python からは見えません）は無視するため、`localhost` のエンドポイントには直接接続します。画像リクエストでは `OpenAI-Project` ヘッダーが空で送信されます: そうしないと、チャット用に設定された `OPENAI_PROJECT_ID` によって、モデル許可リストのあるプロジェクトで画像エンドポイントが 403 `model_not_found` を返してしまうためです。キー自体がすでにプロジェクトを含んでいます。

**ゲートウェイのモデル名。** カタログの ID は OpenAI 向けにマッピングされます: `gpt-image-2-medium` は `model: gpt-image-2` + `quality: medium` として送信されます。`image_gen.openai.model`（または `OPENAI_IMAGE_MODEL`）のそれ以外の値は、`quality` フィールド**なし**でそのまま `model` として送信されます。そのため、独自の画像モデル名（`custom-image-model`、`grok-imagine-image` など）を提供するゲートウェイは正確にその ID を受け取り、拒否される可能性のある quality の列挙値を目にすることはありません。共有のトップレベルの `image_gen.model` が渡されることは決してありません — 以前の選択による別のプロバイダーの ID（たとえば FAL のパス）が入っている可能性があるためです。

**名前付きカスタムエンドポイントの再利用。** ゲートウェイがすでにチャット用に `providers:` の下で宣言されている場合は、URL とキーを繰り返す代わりに、画像プロバイダーを *名前* でそれに向けます:

```yaml
providers:
  my-gateway:
    name: My Gateway
    api: https://gateway.example.com/v1
    key_env: MY_GATEWAY_KEY

image_gen:
  provider: openai
  openai:
    provider: my-gateway        # inherits api + key_env from providers.my-gateway
    model: grok-imagine-image   # sent verbatim, no quality
```

解決順序は、`image_gen.openai.base_url` → 名前付きエンドポイントの URL → `OPENAI_BASE_URL`、および `image_gen.openai.key_env` で指定された変数 → 名前付きエンドポイントの `api_key`/`key_env` → `OPENAI_API_KEY` です。したがって、`provider` と並べて明示的に指定した `base_url` や `key_env` は、エンドポイントのその部分を上書きします。どの `providers:` エントリにも一致しない名前は警告としてログに記録され、無視されます。

## 使い方

エージェント向けのスキーマは意図的に最小限です。モデルは設定した内容をそのまま使います:

```
Generate an image of a serene mountain landscape with cherry blossoms
```

```
Create a square portrait of a wise old owl — use the typography model
```

```
Make me a futuristic cityscape, landscape orientation
```

## 画像から画像 / 編集

アクティブなモデルが対応していれば、同じ `image_generate` ツールで**既存の画像を編集**することもできます — ソース画像を渡すと、バックエンドが自動的に編集用エンドポイントへルーティングします（`video_generate` が画像から動画を扱う方法と同じです）。ソース画像を省略すれば、通常のテキストから画像の生成になります。

```
Take this photo and make it a rainy Tokyo street at night → <image>
```

```
Blend these two product shots into one hero image → <image1> <image2>
```

編集は 2 つの入力によって行われます:

- **`image_url`** — 編集/変換する主要なソース画像（公開 URL またはローカルパス）。
- **`reference_image_urls`** — スタイルや構図の追加の参照画像（上限はモデルごと）。

### 編集に対応しているバックエンド

| バックエンド | 画像から画像 | 参照画像の上限 | 方法 |
|---|---|---|---|
| **FAL.ai**（下記の編集対応モデル） | ✓ | 最大 16（モデルごと） | モデルの `/edit` エンドポイントにルーティング |
| **OpenAI**（GPT Image 2 / 2.5 Flare / Sunburst） | ✓ | 最大 16 | `images.edit()` |
| **xAI**（Grok Imagine） | ✓ | 1 | `/v1/images/edits`（`grok-imagine-image-quality`） |
| **Krea**（`Krea 3`） | ✓ | 最大 4 | 与えられた画像の指示に従った編集（`image_urls`）。ローカルファイルは合計 3 MB までリクエストに埋め込まれます |
| **Krea**（`Krea 2`） | ✓ | 最大 10 | 参照画像に導かれた生成（`image_style_references`）。ローカルファイルは合計 3 MB までリクエストに埋め込まれます |
| **OpenAI (Codex auth)** | ✓ | 最大 16 | インラインの `images[]` データ URL を使った `POST /backend-api/codex/images/edits`（リモート URL はクライアント側で取得されます） |
| **OpenRouter**（Image API モデル） | ✓ | 最大 14〜16（モデルごと） | `POST /images/generations` の `input_references`。chat で提供されるモデルは `image_url` コンテンツパーツを使用（最大 3） |

編集用エンドポイントを持つ FAL モデル: `flux-2/klein/9b`、`flux-2-pro`、`nano-banana-pro`、`gpt-image-1.5`、`gpt-image-2`、`ideogram/v3`、`qwen-image`、および上記の GPT Image 2.5 Flare と Sunburst。テキストから画像専用の FAL モデル（`z-image/turbo`、`recraft`、`krea/*`）は、画像入力を受け取ると、編集対応モデルを案内するわかりやすいエラーで拒否します。

:::note OpenAI (Codex auth): 品質とサイズはバックエンドが決める

Hermes は Codex バックエンドのネイティブな `images/generations` / `images/edits` エンドポイント（公式の Codex クライアントが使うのと同じルート）に直接 POST するため、チャットモデルは関与せず、呼び出しはあなたの ChatGPT プランで現在どのチャットモデルが使えるかに依存しません。ただし、バックエンドは `model`、`quality`、`size` を参考値として扱います: 要求とは異なる品質ティアやサイズを返すことがあります（縦向きのリクエストが正方形で返ってくることもあります）。結果には、要求した内容とともに `reported_quality`、`reported_size`、`pixel_size` が含まれ、OpenAI サポート用の `imagegen_request_id` も付きます。品質とサイズを正確に制御するには、代わりに **OpenAI**（API キー）、**FAL**、または **xAI** バックエンドを設定してください。

:::

アクティブなモデルの編集機能は実行時にツールの説明に表示されるため、エージェントはツールを呼び出す前に `image_url` が尊重されるかどうかを把握できます。

## アスペクト比

すべてのモデルは、エージェントの視点からは同じ 3 つのアスペクト比を受け付けます。内部では、各モデルのネイティブなサイズ指定が自動で埋められます:

| エージェント入力 | image_size (flux/z-image/qwen/recraft/ideogram) | aspect_ratio (nano-banana-pro) | image_size (gpt-image-1.5) | image_size (gpt-image-2) |
|---|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` | `landscape_4_3` (1024×768) |
| `square` | `square_hd` | `1:1` | `1024x1024` | `square_hd` (1024×1024) |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` | `portrait_4_3` (768×1024) |

GPT Image 2 が 16:9 ではなく 4:3 のプリセットにマッピングされるのは、最小ピクセル数が 655,360 だからです。`landscape_16_9` プリセット（1024×576 = 589,824）は拒否されてしまいます。

この変換は `_build_fal_payload()` で行われます。エージェントのコードは、モデルごとのスキーマの違いを知る必要がありません。

## アップスケール

### オプトインのみ

既定でアップスケールするモデルはありません。最新の画像モデルはネイティブで最高の品質を出力し、利用可能なアップスケーラーは内容を微妙に描き直すことのある *創造的な* エンハンサー（拡散処理のパス）であるため、描画されたテキスト、顔、細部を劣化させることがあります。アップスケールは、エージェントが明示的に要求した場合にのみ実行されます。

### `upscale` パラメータ（呼び出しごとのオプトイン）

- `upscale: true` — 生成後に高解像度化のパスをつなげます:

| バックエンド | アップスケーラー |
|---|---|
| **FAL.ai** | Clarity Upscaler（2×、+$0.03/MP） |
| **Krea** | Krea Enhance（2×、上限 8K） |
| その他のバックエンド | アップスケーラーなし。ネイティブ解像度を返す |

- `upscale: false` / 省略 — ネイティブ解像度（既定）

`video_generate` も FAL バックエンドで `upscale: true` を受け付け、生成後に ByteDance の **SeedVR2** 動画アップスケーラー（2×、出力動画 $0.001/MP）をつなげます。

FAL の画像パスが実行される場合、次の設定を使います:

| 設定 | 値 |
|---|---|
| アップスケール倍率 | 2× |
| Creativity | 0.35 |
| Resemblance | 0.6 |
| Guidance scale | 4 |
| Inference steps | 18 |

アップスケールが失敗した場合（ネットワークの問題、レート制限）、元の画像が自動的に返されます。レスポンスは `upscaled: true/false` を報告するため、エージェントはどの解像度の画像を得たかを把握できます。

## 内部の仕組み

1. **モデル解決** — `_resolve_fal_model()` が `config.yaml` の `image_gen.model` を読み、`FAL_IMAGE_MODEL` 環境変数、次に `fal-ai/flux-2/klein/9b` にフォールバックします。
2. **ペイロード構築** — `_build_fal_payload()` が、あなたの `aspect_ratio` をモデルのネイティブ形式（プリセット列挙、アスペクト比列挙、または GPT リテラル）に変換し、モデルの既定パラメータをマージし、呼び出し側の上書きを適用し、モデルの `supports` ホワイトリストでフィルタリングして、未対応のキーが決して送られないようにします。
3. **送信** — `_submit_fal_request()` が、保存された `image_gen.provider` の選択に従って、ダイレクトな FAL クレデンシャルまたはマネージド Nous ゲートウェイ経由でルーティングします。
4. **アップスケール** — エージェントが `upscale: true` を渡した場合にのみ実行されます。すべてのモデルのカタログ既定値はオフです。
5. **配信** — 最終的な画像 URL がエージェントに返され、エージェントが `MEDIA:<url>` タグを出力します。プラットフォームアダプターがこれをネイティブメディアに変換します。
6. **使用量の記録** — トークン課金の画像モデル（`google/gemini-3.1-flash-lite-image` のような OpenRouter の chat-image および Image API モデル、OpenAI の `gpt-image`）は実際のトークン数を返すため、各呼び出しは課金プロバイダーとモデルのもとでタスク `image_generation` として `session_model_usage` に記録され、他のモデル呼び出しと並んで `hermes insights` やダッシュボードの Usage 分析に表示されます。画像単位で課金するバックエンド（FAL、xAI、Krea など）はトークン使用量を返さないため、そこには記録されません。

## デバッグ

デバッグログを有効にします:

```bash
export IMAGE_TOOLS_DEBUG=true
```

デバッグログは `./logs/image_tools_debug_<session_id>.json` に出力され、呼び出しごとの詳細（モデル、パラメータ、タイミング、エラー）が記録されます。

## プラットフォームでの配信

| プラットフォーム | 配信 |
|---|---|
| **CLI** | 画像 URL が Markdown の `![](url)` として出力される — クリックで開く |
| **Telegram** | プロンプトをキャプションにした写真メッセージ |
| **Discord** | メッセージに埋め込み |
| **Slack** | Slack によって URL が展開される |
| **WhatsApp** | メディアメッセージ |
| **その他** | プレーンテキストの URL |

## 制限事項

- アクティブなバックエンドの**クレデンシャルが必要**（FAL の `FAL_KEY` / Nous Subscription、`OPENAI_API_KEY`、xAI OAuth、`KREA_API_KEY`）
- **編集はモデルに依存** — 画像から画像は編集対応モデルでのみ機能します（上の表を参照）。テキストから画像専用のモデルは、画像入力をわかりやすいエラーで拒否します
- **一時的な URL** — バックエンドは数時間〜数日で失効するホスト URL を返します。Hermes はそれらをローカルキャッシュに実体化するため、失効後も配信は機能します
- **モデルごとの制約** — 一部のモデルは `seed`、`num_inference_steps` などに対応しません。`supports` / `edit_supports` フィルタが未対応のパラメータを暗黙的に破棄します。これは想定された動作です
