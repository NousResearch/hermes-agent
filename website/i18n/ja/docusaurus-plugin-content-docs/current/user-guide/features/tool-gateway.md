---
title: "Nous Tool Gateway"
description: "1 つのサブスクリプションで、すべてのツールを。Web 検索、画像生成、TTS、クラウドブラウザ — すべて Nous Portal 経由でルーティングされ、追加の API キーは不要。"
sidebar_label: "Tool Gateway"
sidebar_position: 2
---

# Nous Tool Gateway

**1 つのサブスクリプション。すべてのツールが組み込み。**

Tool Gateway は、有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションすべてに含まれています。Hermes のツール呼び出し（Web 検索、画像生成、テキスト読み上げ、クラウドブラウザ自動化）を、Nous がすでに運用しているインフラ経由でルーティングします。そのため、エージェントを役立たせるためだけに Firecrawl、FAL、OpenAI、Browser Use などへ個別に登録する必要がありません。

<div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', margin: '1.5rem 0'}}>
  <a href="https://portal.nousresearch.com/manage-subscription" style={{background: 'var(--ifm-color-primary)', color: 'white', padding: '0.75rem 1.5rem', borderRadius: '6px', textDecoration: 'none', fontWeight: 'bold'}}>サブスクリプションを開始・管理 →</a>
</div>

## 含まれるもの

| | ツール | できること |
|---|---|---|
| 🔍 | **Web 検索 & 抽出** | Firecrawl 経由のエージェントグレードな Web 検索とフルページ抽出。レート制限を気にする必要はありません。スケーリングはゲートウェイが処理します。 |
| 🎨 | **画像生成** | 1 つのエンドポイントで 9 つのモデル: **FLUX 2 Klein 9B**、**FLUX 2 Pro**、**Z-Image Turbo**、**Nano Banana Pro**（Gemini 3 Pro Image）、**GPT Image 1.5**、**GPT Image 2**、**Ideogram V3**、**Recraft V4 Pro**、**Qwen Image**。生成ごとにフラグで選択するか、Hermes に既定の FLUX 2 Klein を任せられます。 |
| 🔊 | **テキスト読み上げ** | `text_to_speech` ツールに組み込まれた OpenAI TTS の音声。Telegram にボイスノートを送り、パイプライン用の音声を生成し、あらゆるものをナレーションできます。 |
| 🌐 | **クラウドブラウザ自動化** | Browser Use 経由のヘッドレス Chromium セッション。`browser_navigate`、`browser_click`、`browser_type`、`browser_vision` — エージェントを駆動するすべてのプリミティブが、Browserbase アカウント不要で使えます。 |

4 つすべてが、Nous サブスクリプションに対して従量課金されます。任意の組み合わせで使えます。Web と画像にはゲートウェイを使いつつ、TTS には自分の ElevenLabs キーを使い続けることも、すべてを Nous 経由でルーティングすることもできます。

## なぜこれがあるのか

実際に*物事をこなせる*エージェントを作るには、5 つ以上の API サブスクリプションをつなぎ合わせる必要があります。それぞれに独自の登録、レート制限、課金、クセがあります。ゲートウェイはそれを 1 つのアカウントにまとめます:

- **1 つの請求。** Nous に支払えば、残りは私たちが処理します。
- **1 つの登録。** 管理すべき Firecrawl、FAL、Browser Use、OpenAI audio のアカウントはありません。
- **1 つのキー。** あなたの Nous Portal OAuth がすべてのツールをカバーします。
- **同じ品質。** ダイレクトキー経路が使うのと同じバックエンド。私たちが前面に立つだけです。

いつでも自分のキーを持ち込めます。ツール単位で、好きなときに。ゲートウェイはロックインではなく、ショートカットです。

## はじめる

```bash
hermes model          # プロバイダーとして Nous Portal を選択
```

Nous Portal を選択すると、Hermes が Tool Gateway を有効にするか尋ねます。承諾すれば完了です。次回の実行から、対応するすべてのツールが有効になります。

何が有効かはいつでも確認できます:

```bash
hermes status
```

次のようなセクションが表示されます:

```
◆ Nous Tool Gateway
  Nous Portal     ✓ managed tools available
  Web tools       ✓ active via Nous subscription
  Image gen       ✓ active via Nous subscription
  TTS             ✓ active via Nous subscription
  Browser         ○ active via Browser Use key
```

「active via Nous subscription」と表示されたツールはゲートウェイを経由しています。それ以外は自分のキーを使っています。

## 利用資格

Tool Gateway は**有料サブスクリプション**機能です。無料プランの Nous アカウントは推論には Portal を使えますが、マネージドツールは含まれません。ゲートウェイを利用するには [プランをアップグレード](https://portal.nousresearch.com/manage-subscription)してください。

## 組み合わせて使う

ゲートウェイはツール単位です。必要なものだけ有効にできます:

- **すべてのツールを Nous 経由で** — 最も簡単。1 つのサブスクリプションで完了。
- **Web + 画像はゲートウェイ、TTS は自前** — ElevenLabs の音声を維持し、残りは Nous に任せる。
- **キーを持っていないものだけゲートウェイで** — 「Browserbase にはすでに支払っているが、Firecrawl のアカウントは作りたくない」も問題なく機能します。

任意のツールはいつでも次のコマンドで切り替えられます:

```bash
hermes tools          # 各ツールカテゴリの対話的ピッカー
```

ツールを選択し、プロバイダーとして **Nous Subscription**（または好みの任意のダイレクトプロバイダー）を選びます。設定の編集は不要です。

## 個別の画像モデルを使う

画像生成は速度のため既定で FLUX 2 Klein 9B を使います。`image_generate` ツールにモデル ID を渡すことで、呼び出しごとに上書きできます:

| モデル | ID | 適した用途 |
|---|---|---|
| FLUX 2 Klein 9B | `fal-ai/flux-2/klein/9b` | 高速、良い既定 |
| FLUX 2 Pro | `fal-ai/flux-2/pro` | より高忠実度の FLUX |
| Z-Image Turbo | `fal-ai/z-image/turbo` | スタイライズ、高速 |
| Nano Banana Pro | `fal-ai/gemini-3-pro-image` | Google Gemini 3 Pro Image |
| GPT Image 1.5 | `fal-ai/gpt-image-1/5` | OpenAI 画像生成、テキスト+画像 |
| GPT Image 2 | `fal-ai/gpt-image-2` | OpenAI 最新 |
| Ideogram V3 | `fal-ai/ideogram/v3` | 強力なプロンプト追従 + タイポグラフィ |
| Recraft V4 Pro | `fal-ai/recraft/v4/pro` | ベクター調、グラフィックデザイン |
| Qwen Image | `fal-ai/qwen-image` | Alibaba マルチモーダル |

セットは進化します。`hermes tools` → Image Generation で現在の最新リストが表示されます。

---

## 設定リファレンス

ほとんどのユーザーはここに触れる必要はありません。`hermes model` と `hermes tools` があらゆるワークフローを対話的にカバーします。このセクションは、config.yaml を直接書いたり、セットアップをスクリプト化したりする場合向けです。

### ツールごとの `use_gateway` フラグ

各ツールの設定ブロックは `use_gateway` ブール値を取ります:

```yaml
web:
  backend: firecrawl
  use_gateway: true

image_gen:
  use_gateway: true

tts:
  provider: openai
  use_gateway: true

browser:
  cloud_provider: browser-use
  use_gateway: true
```

優先順位: `use_gateway: true` は、`.env` 内のダイレクトキーの有無にかかわらず Nous 経由でルーティングします。`use_gateway: false`（または未指定）は、利用可能ならダイレクトキーを使い、どれも存在しない場合にのみゲートウェイにフォールバックします。

### ゲートウェイを無効にする

```yaml
web:
  use_gateway: false   # Hermes は .env の FIRECRAWL_API_KEY を使うようになります
```

`hermes tools` は、ゲートウェイ以外のプロバイダーを選んだときに自動でフラグをクリアするため、通常はこれが自動で行われます。

### セルフホスト型ゲートウェイ（上級者向け）

独自の Nous 互換ゲートウェイを運用していますか？ `~/.hermes/.env` でエンドポイントを上書きします:

```bash
TOOL_GATEWAY_DOMAIN=your-domain.example.com
TOOL_GATEWAY_SCHEME=https
TOOL_GATEWAY_USER_TOKEN=your-token        # 通常は Portal ログインから自動入力されます
FIRECRAWL_GATEWAY_URL=https://...         # 特定の 1 エンドポイントを上書き
```

これらのつまみは、カスタムインフラ構成（エンタープライズデプロイ、開発環境）向けに存在します。通常のサブスクライバーが設定することはありません。

## FAQ

### Telegram / Discord / その他のメッセージングゲートウェイで動作しますか？

はい。Tool Gateway は CLI ではなく、ツール実行レイヤーで動作します。ツールを呼び出せるすべてのインターフェース（CLI、Telegram、Discord、Slack、IRC、Teams、API サーバーなど）が、透過的にその恩恵を受けます。

### サブスクリプションが失効したらどうなりますか？

ゲートウェイ経由でルーティングされたツールは、更新するか `hermes tools` でダイレクト API キーに差し替えるまで動作を停止します。Hermes は Portal を指す明確なエラーを表示します。

### ツールごとの使用量やコストを確認できますか？

はい。[Nous Portal ダッシュボード](https://portal.nousresearch.com)が使用量をツールごとに分解するので、何が請求を押し上げているか分かります。

### Modal（サーバーレスターミナル）は含まれますか？

Modal は、既定の Tool Gateway バンドルの一部ではなく、Nous サブスクリプションを通じた**オプションのアドオン**として利用できます。シェル実行用のリモートサンドボックスが必要なときは、`hermes setup terminal` 経由、または `config.yaml` で直接設定してください。

### ゲートウェイを有効にするとき、既存の API キーを削除する必要がありますか？

いいえ。`.env` に残しておいてください。`use_gateway: true` のとき、Hermes はダイレクトキーをスキップしてゲートウェイを使います。フラグを `false` に戻せば、あなたのキーが再びソースになります。ゲートウェイはロックインではありません。
